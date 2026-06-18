"""Two figures from the kernel controller (YIELDER role, r=0):

  outputs/f_contour.png      — the LEARNED mean from sim samples (collect_f.py):
                               f (raw) and Δa (net effect on command), signed
                               most-extreme + std, over the operating envelope.
  outputs/prescribed_accel.png — the PRESCRIBED acceleration (physics/zero-mean
                               command) evaluated on a DENSE synthetic (g, τ_c) grid,
                               so every cell is colored.  Carries the brake/accelerate
                               regime boxes.

Color convention: BLUE = accelerate (a>0), RED = brake (a<0).
Overlays: ○ = anchor points (f canceled there); lime Σw contours (0.25/0.5/0.75 =
f-authority, smaller Σw ⇒ more authority); magenta/blue boxes = brake/accel regimes.

    conda run -n car-following-sumo python plot_f.py            # outputs/f_samples.npz
    conda run -n car-following-sumo python plot_f.py path.npz [vmax]
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import utils
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "outputs")
SRC  = next((a for a in sys.argv[1:] if a.endswith(".npz")), os.path.join(OUT, "f_samples.npz"))
VMAX = next((float(a) for a in sys.argv[1:] if not a.endswith(".npz")), None)

NB_G, NB_T = 50, 64
gbins = np.linspace(0.0, utils.G_MAX,     NB_G + 1)
tbins = np.linspace(0.0, utils.TAU_C_MAX, NB_T + 1)
gc, tc = 0.5 * (gbins[:-1] + gbins[1:]), 0.5 * (tbins[:-1] + tbins[1:])

V_NOM = 8.0       # m/s   nominal ego speed for the prescribed-accel grid
D_NOM = 30.0      # m     nominal ego distance to the conflict point
PRESC_VLIM = 1.0  # m/s²  prescribed-accel color scale ±this (clips beyond → saturated,
                  #       so the sign change / transitions are highlighted)


def binned(g, t, v):
    """Per (g, τ_c) cell: SIGNED value of the largest-|v| sample, and the std."""
    gi = np.clip(np.digitize(g, gbins) - 1, 0, NB_G - 1)
    ti = np.clip(np.digitize(t, tbins) - 1, 0, NB_T - 1)
    flat = gi * NB_T + ti
    order = np.argsort(np.abs(v))
    maxabs = np.full(NB_G * NB_T, np.nan); maxabs[flat[order]] = v[order]
    cnt, _, _ = np.histogram2d(g, t, bins=[gbins, tbins])
    s1, _, _  = np.histogram2d(g, t, bins=[gbins, tbins], weights=v)
    s2, _, _  = np.histogram2d(g, t, bins=[gbins, tbins], weights=v * v)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(cnt > 0, s2 / cnt - (s1 / cnt) ** 2, np.nan)
    return maxabs.reshape(NB_G, NB_T), np.sqrt(np.clip(var, 0, None))


def kernel_authority(r_val=0.0):
    """Σw(φ) = 1ᵀ K⁻¹ k(φ) over the grid at role r_val (≈1 at anchors; f shows through
    as 1−Σw, so Σw<0.5 ⇒ f operates >50%)."""
    anchors = torch.tensor(utils.ANCHOR_FEATS, dtype=torch.float32)
    ls = torch.tensor(utils.LENGTHSCALES, dtype=torch.float32)
    Kinv = utils._anchor_kinv(anchors, ls)
    GG, TT = np.meshgrid(gc, tc, indexing="ij")
    phi = torch.tensor(np.stack([GG, TT, np.full_like(GG, r_val),
                                 np.zeros_like(GG)], -1).reshape(-1, 4),  # p=0 sheet
                       dtype=torch.float32)
    w = utils._kernel_vec(phi, anchors, ls) @ Kinv
    return w.sum(-1).reshape(NB_G, NB_T).numpy()


def prescribed_grid(r_val=0.0):
    """The controller's PRESCRIBED (zero-mean) acceleration on a dense (g, τ_c) grid.
    Each grid point is realized by a synthetic state at the nominal scenario:
    ego at V_NOM with a STOPPED leader (so the g<1 braking regime shows), and a single
    cross predecessor whose timing sets τ_c.  Yielder if r_val<0.5."""
    GG, TT = np.meshgrid(gc, tc, indexing="ij")
    g_t = torch.tensor(GG.ravel(), dtype=torch.float32)
    tau = torch.tensor(TT.ravel(), dtype=torch.float32)
    N = g_t.numel()
    v_ego = torch.full((N,), V_NOM); v_lead = torch.zeros(N)          # stopped leader
    s_des = utils.desired_gap(v_ego, v_lead)
    x_ego = torch.zeros(N); x_lead = g_t * s_des + utils.L_VEH        # → gap_ratio = g_t
    ego_d_pred = torch.full((N,), D_NOM); v_pred = torch.full((N,), V_NOM)
    eta_pred = (ego_d_pred / v_ego.clamp(min=utils.EPS) - tau).clamp(min=0.0)
    has_pred = torch.full((N,), bool(r_val < 0.5))
    pred_override = (tau, eta_pred, ego_d_pred, v_pred, has_pred,
                     torch.zeros(N, 1, dtype=torch.bool))
    dummy = ego_d_pred.unsqueeze(-1)
    a = utils.controller_acceleration(
        x_ego, x_lead, v_ego, v_lead, d_conf=dummy, rival_d=dummy, rival_v=v_pred.unsqueeze(-1),
        rival_valid=torch.ones(N, 1, dtype=torch.bool),
        pred_override=pred_override, predecessor=True, mean_fn=None)
    return a.detach().numpy().reshape(NB_G, NB_T)


def draw_regions(ax):
    ds = utils.DELTA_SAFE
    regions = [
        (0.15, 0.95, 0.3, utils.TAU_C_MAX - 0.2, "BRAKE\nclosing on leader\n($g_i$<1, rear-end)", True),
        (1.10, 4.60, 0.0, 0.28 * utils.TAU_C_MAX, "YIELD\ncross-conflict imminent\n(low $\\tau_i$)", True),
        (1.10, 4.60, ds - 1.8, ds + 0.0, "RESUME\nconflict cleared\n($\\tau_i\\to\\delta_{safe}$)", False),
    ]
    for g0, g1, t0, t1, label, brake in regions:
        c = "magenta" if brake else "royalblue"
        label = f"{label}\n$a_i$ < 0" if brake else f"{label}\n$a_i$ > 0"
        ax.add_patch(Rectangle((g0, t0), g1 - g0, t1 - t0, facecolor=c, alpha=0.15,
                               edgecolor=c, lw=1.4, zorder=4))
        ax.text((g0 + g1) / 2, (t0 + t1) / 2, label, color="white", fontsize=7,
                weight="bold", ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", fc=c, ec="none", alpha=0.9))


def decorate(ax, sw, boxes=False, safety_line=False):
    ax.set_xlabel(r"$g_i$  (gap ratio)"); ax.set_ylabel(r"$\tau_i$  (conflict gap, s)")
    ax.set_xlim(0, utils.G_MAX); ax.set_ylim(0, utils.TAU_C_MAX)
    for x in utils._G_LEVELS:
        ax.axvline(x, color="k", lw=0.3, alpha=0.2)
    for y in utils._TAUC_LEVELS:
        ax.axhline(y, color="k", lw=0.3, alpha=0.2)
    GG, TT = np.meshgrid(utils._G_LEVELS, utils._TAUC_LEVELS)
    ax.scatter(GG.ravel(), TT.ravel(), s=15, color="k", zorder=5, label="anchor points")
    if safety_line:   # δ_safe: at/above this conflict gap the yielder stops braking (HOLD)
        ax.axhline(utils.DELTA_SAFE, color=plot_style.GREEN, lw=2.2, ls="--", zorder=6,
                   label=r"$\delta_{safe}$ (safety threshold)")
    if boxes:
        draw_regions(ax)


def main():
    plot_style.apply()
    cmap_div = plt.get_cmap("RdBu").copy(); cmap_div.set_bad("0.82")   # blue=+ accel, red=− brake
    cmap_seq = plt.get_cmap("viridis").copy(); cmap_seq.set_bad("0.82")
    sw = kernel_authority(r_val=0.0)
    X, Y = np.meshgrid(gbins, tbins)

    # ── FIGURE 2: PRESCRIBED acceleration on a dense grid (fully colored) ────────────
    # two role columns: YIELDER (r=0) and PASSER (r=1); no regime boxes
    a_y, a_p = prescribed_grid(r_val=0.0), prescribed_grid(r_val=1.0)
    sw_p = kernel_authority(r_val=1.0)
    aL = PRESC_VLIM                                  # tight scale → blue/red transitions pop
    figp, axps = plt.subplots(1, 2, figsize=(15, 6.6), constrained_layout=True)
    for ax, a_pre, sw_r, title, sline in [
            (axps[0], a_y, sw,   r"Acceleration at $\gamma_i = 0$", True),   # yielder: δ_safe line
            (axps[1], a_p, sw_p, r"Acceleration at $\gamma_i = 1$", False)]:
        imp = ax.pcolormesh(X, Y, a_pre.T, cmap=cmap_div, vmin=-aL, vmax=aL, shading="flat")
        ax.set_title(title)
        figp.colorbar(imp, ax=ax, label=r"$a_i$ (m/s²)", extend="both")
        decorate(ax, sw_r, boxes=False, safety_line=sline)
    axps[0].legend(loc="upper right")   # yielder panel: anchor scatter + δ_safe threshold
    axps[1].legend(loc="upper right")   # passer panel: anchor scatter
    figp.suptitle("Prescribed acceleration by equality terms")
    outp = os.path.join(OUT, "prescribed_accel")
    plot_style.save(figp, outp, dpi=130); print(f"saved {outp}.png / .pdf")

    # ── FIGURE 1: LEARNED mean f, from the ACTUALLY-COLLECTED rollout samples only ──
    # f was harvested at each vehicle's REAL operating φ with its REAL context z, so we
    # just bin those samples — NO synthetic/invented contexts.  Gray = operating points
    # the collected trajectories never visited (honest lack of coverage).
    # Row 0: most-extreme f (max |f|, signed).  Row 1: std of f over the real contexts.
    if not os.path.exists(SRC):
        print(f"({SRC} not found — run collect_f.py first for the learned-mean figure)")
        return
    npz = np.load(SRC); data = npz["data"]; flows = list(npz["flows"]); nseed = int(npz["nseed"])
    yld = data[:, 2] < 0.5                                   # YIELDER only
    ma, _ = binned(data[yld, 0], data[yld, 1], data[yld, 3])  # most extreme f (max|f|, signed)
    finite = np.abs(ma[np.isfinite(ma)])
    vlim = VMAX if VMAX is not None else float(np.percentile(finite, 95))
    print(f"f (yielder, {int(yld.sum())} collected samples): ±{vlim:.3f} "
          f"(95th; max={finite.max():.3f})")
    fig, ax = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
    im0 = ax.pcolormesh(X, Y, np.ma.masked_invalid(ma).T, cmap=cmap_div,
                        vmin=-vlim, vmax=vlim, shading="flat")
    ax.set_title(r"Yielder ($\gamma_i$=0) — most extreme learned $f_i$ (max $|f_i|$, signed)" + "\n"
                 f"from COLLECTED rollouts only  [{int(yld.sum())} samples, "
                 r"flows " + f"{flows} × {nseed} seeds]  ·  blue=$a_i$>0, red=$a_i$<0, grey=uncovered")
    fig.colorbar(im0, ax=ax, label=r"$f_i$ (m/s²)", extend="both")
    decorate(ax, sw, boxes=False)
    out = os.path.join(OUT, "f_contour")
    plot_style.save(fig, out, dpi=130); print(f"saved {out}.png / .pdf")

    # ── FIGURE 3: PASSER f vs g (1-D) — passers all sit at τ_c≈7.31, so the 2-D panel
    # is a single line; this is the honest view of it (from collected samples).
    pm = data[:, 2] >= 0.5
    if pm.sum() > 0:
        gp, fp = data[pm, 0], data[pm, 3]
        gi = np.clip(np.digitize(gp, gbins) - 1, 0, NB_G - 1)
        mean_f = np.full(NB_G, np.nan); maxabs_f = np.full(NB_G, np.nan)
        for b in range(NB_G):
            sel = gi == b
            if sel.any():
                vals = fp[sel]
                mean_f[b] = vals.mean()
                maxabs_f[b] = vals[np.abs(vals).argmax()]      # max |f|, signed
        figf, axf = plt.subplots(figsize=(9, 5), constrained_layout=True)
        axf.axhline(0, color="k", lw=0.6)
        axf.scatter(gp, fp, s=4, alpha=0.12, color="gray", label="samples (context spread)")
        axf.plot(gc, mean_f, "-o", ms=3, color=plot_style.BLUE, label=r"mean $f_i$")
        axf.plot(gc, maxabs_f, "-s", ms=3, color=plot_style.PINK, label=r"max $|f_i|$ (signed)")
        for x in utils._G_LEVELS:
            axf.axvline(x, color="k", lw=0.3, alpha=0.2)
        axf.axvline(1.0, color=plot_style.GREEN, lw=0.8, ls="--", alpha=0.6)   # g=1 equilibrium
        axf.set_xlabel(r"$g_i$  (gap ratio)"); axf.set_ylabel(r"$f_i$  (m/s²)")
        axf.set_xlim(0, utils.G_MAX); axf.set_title(
            r"Passer ($\gamma_i$=1)  $f_i$ vs $g_i$   (all passers at $\tau_i\approx$7.31)   "
            f"[{int(pm.sum())} samples, flows {flows} × {nseed} seeds]   blue=accel, below 0=brake")
        axf.legend(); plot_style.despine(axf)
        outf = os.path.join(OUT, "passer_f")
        plot_style.save(figf, outf, dpi=130); print(f"saved {outf}.png / .pdf")


if __name__ == "__main__":
    main()
