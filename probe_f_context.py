"""
probe_f_context.py — what has the learned prior mean f̂ actually learned?

f̂(φ, z) takes the kernel-shared triplet φ = (g, τ_c, γ) AND a context vector z that
the kernel is BLIND to: z = (v, behind_n, d_junc, conf⁺, conf⁻)  [CONTEXT_COLS].

We can't view a 7-D surface, so:
  • the KERNEL inputs (g, τ_c, γ) are gridded as the DISPLAY surface — a (g × τ_c)
    heatmap per role γ ∈ {0, 1}, exactly like outputs/prescribed_accel;
  • the CONTEXT z is swept on a grid with separation 1 (physical units) and then
    REDUCED OUT at every kernel cell into two scalars:
        MAX  over z   — the strongest response f can produce there ("max response")
        MEAN over z   — the context averaged out (partial-dependence value)

So each (g, τ_c, γ) cell answers: "across every context the controller could be in,
what is the biggest acceleration nudge f asks for here, and what is it on average?"

The context grid is fed through the SAME build_context normalizations (so the head
sees exactly the inputs it saw in training).  Output:
    outputs/f_context_response.png/.pdf   — 2×2 heatmaps (max / mean) × (γ=0 / γ=1)
    outputs/f_context_grid.npz            — full reduced grids + axes + extra reductions

    conda run -n car-following-sumo python probe_f_context.py [n_kernel] [d_junc_max]
       n_kernel    : g and τ_c resolution (default 31 → 31×31 cells per role)
       d_junc_max  : upper bound (m) of the d_junc context sweep (default 30)
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils
import mean_net
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "outputs")

NK        = int(sys.argv[1]) if len(sys.argv) > 1 else 31      # kernel g/τ_c resolution
D_JUNC_HI = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0  # context d_junc upper bound (m)
CKPT_ARG  = sys.argv[3] if len(sys.argv) > 3 else None         # explicit checkpoint path override

# ── context z grid, STEP 1 in physical units (cols 0,5,6,7,8 of build_context's ego_feats)
V_HI, B_HI, C_HI = utils.V0, 6.0, 6.0      # speed cap, max behind-count, max conf± density
V_LO, B_LO, D_JUNC_LO, C_LO = 0.0, 0.0, 0.0, 0.0


def context_grid():
    """[Nz, 5] context vectors, SCALED exactly as mean_net.build_context scales them
    (v/V_SCALE, behind/P_SCALE, clamp(d_junc)/D_SCALE, conf±/P_SCALE).  Step 1 physical."""
    v  = np.arange(V_LO, V_HI + 1e-6, 1.0)
    b  = np.arange(B_LO, B_HI + 1e-6, 1.0)
    d  = np.arange(D_JUNC_LO, D_JUNC_HI + 1e-6, 1.0)
    cp = np.arange(C_LO, C_HI + 1e-6, 1.0)
    cn = np.arange(C_LO, C_HI + 1e-6, 1.0)
    G  = np.meshgrid(v, b, d, cp, cn, indexing="ij")
    phys = np.stack([g.ravel() for g in G], axis=1)             # [Nz, 5] physical units
    z = np.empty_like(phys, dtype=np.float32)
    z[:, 0] = phys[:, 0] / mean_net.V_SCALE                     # v
    z[:, 1] = phys[:, 1] / mean_net.P_SCALE                     # behind_n
    z[:, 2] = np.clip(phys[:, 2], -mean_net.D_SCALE,
                      3 * mean_net.D_SCALE) / mean_net.D_SCALE   # d_junc (same clamp as training)
    z[:, 3] = phys[:, 3] / mean_net.P_SCALE                     # conf⁺
    z[:, 4] = phys[:, 4] / mean_net.P_SCALE                     # conf⁻
    dims = dict(v=v, behind=b, d_junc=d, conf_pos=cp, conf_neg=cn)
    return torch.from_numpy(z), dims


def load_model(ckpt=None):
    if mean_net.ARCH != "mlp_ctx":
        print(f"WARNING: ARCH={mean_net.ARCH!r}, not 'mlp_ctx' — f may have no context input")
    model = mean_net.make_mean_model()
    ckpt  = ckpt or mean_net.ckpt_path("last")        # LATEST epoch (saved every epoch)
    ck = torch.load(ckpt, weights_only=True)
    model.load_state_dict(ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
    model.eval()
    ep = ck.get("epoch", "?") if isinstance(ck, dict) else "?"
    print(f"loaded {ckpt}  (epoch={ep}, ARCH={mean_net.ARCH})")
    return model


def sweep(model, z):
    """For every kernel cell (g, τ_c, γ) evaluate f over the WHOLE context grid z and
    reduce z out.  Returns dict of [2, NK, NK] arrays (axis0: γ=0,1) + the kernel axes."""
    g_vals  = np.linspace(0.0, utils.G_MAX, NK)
    tc_vals = np.linspace(0.0, utils.TAU_C_MAX, NK)
    Nz = z.shape[0]
    shape = (2, NK, NK)                       # [γ, τ_c, g]
    fmax  = np.zeros(shape, np.float32); fmin  = np.zeros(shape, np.float32)
    fmean = np.zeros(shape, np.float32); fabsmax = np.zeros(shape, np.float32)
    fsigned = np.zeros(shape, np.float32)   # SIGNED f at the context where |f| is largest
    print(f"sweeping {2 * NK * NK} kernel cells × {Nz} context points "
          f"(d_junc 0–{D_JUNC_HI:.0f} m, step 1) …")
    with torch.no_grad():
        for ri, gamma in enumerate((0.0, 1.0)):
            for ti, tc in enumerate(tc_vals):
                for gj, g in enumerate(g_vals):
                    phi = torch.tensor([g, tc, gamma], dtype=torch.float32).unsqueeze(0).expand(Nz, 3)
                    f = model.head(phi, z)                       # [Nz]
                    k = int(f.abs().argmax())                    # context with strongest response
                    fmax[ri, ti, gj]  = float(f.max())
                    fmin[ri, ti, gj]  = float(f.min())
                    fmean[ri, ti, gj] = float(f.mean())
                    fabsmax[ri, ti, gj] = float(f.abs().max())
                    fsigned[ri, ti, gj] = float(f[k])            # signed value at argmax|f|
            print(f"  γ={gamma:.0f} done")
    return dict(g=g_vals, tc=tc_vals, fmax=fmax, fmin=fmin, fmean=fmean,
                fabsmax=fabsmax, fsigned=fsigned)


def main():
    plot_style.apply()
    model = load_model(CKPT_ARG)
    z, dims = context_grid()
    R = sweep(model, z)
    g, tc = R["g"], R["tc"]

    # MEAN (partial-dependence) reduction; each panel scaled to its OWN min/max so the
    # full range of f̂ is visible (yielder and passer occupy very different bands).
    cmap = plt.get_cmap("RdBu").copy()                          # blue=+accel, red=−brake
    X, Y = np.meshgrid(g, tc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True,
                             sharex=True, sharey=True)
    roles = [(0, r"$\gamma_i = 0$"), (1, r"$\gamma_i = 1$")]
    for ci, (gi, clabel) in enumerate(roles):
        ax = axes[ci]
        data = R["fmean"][gi]
        vmin, vmax = float(data.min()), float(data.max())
        im = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto")
        ax.axhline(utils.DELTA_SAFE, color=plot_style.GREEN, lw=1.8, ls="--",
                   zorder=4, label=r"$\delta_{safe}$")
        ax.set_title(clabel)
        ax.set_xlabel(r"$g_i$  (gap ratio)")
        if ci == 0:
            ax.set_ylabel(r"$\tau_i$  (conflict gap, s)")
        fig.colorbar(im, ax=ax, label=r"$\hat f$  (m/s²)")
        plot_style.despine(ax)
    axes[0].legend(loc="upper right")
    fig.suptitle("Neural Network learned acceleration")
    tag = os.path.splitext(os.path.basename(CKPT_ARG))[0] if CKPT_ARG else "last"
    out = os.path.join(OUT, f"f_context_response_{tag}")
    plot_style.save(fig, out, dpi=130); print(f"saved {out}.png / .pdf")

    npz = os.path.join(OUT, f"f_context_grid_{tag}.npz")
    np.savez(npz, g=g, tc=tc, gamma=np.array([0.0, 1.0]),
             fmax=R["fmax"], fmin=R["fmin"], fmean=R["fmean"], fabsmax=R["fabsmax"],
             fsigned=R["fsigned"],
             **{f"ctx_{k}": v for k, v in dims.items()})
    print(f"saved {npz}  (keys: fmax, fmin, fmean, fabsmax, fsigned [γ,τ_c,g] + axes)")
    # quick textual summary so the headline numbers are visible without opening the plot
    for gi, name in ((0, "yielder γ=0"), (1, "passer γ=1")):
        print(f"  {name}:  max_z f peak={R['fmax'][gi].max():+.3f}  "
              f"min={R['fmin'][gi].min():+.3f}  signed@argmax|f| range=["
              f"{R['fsigned'][gi].min():+.3f}, {R['fsigned'][gi].max():+.3f}]  m/s²")


if __name__ == "__main__":
    main()
