"""outputs/gate_correction.png — the INEQUALITY correction Δa applied by the gradient-free
PROXY hinge (proxy_hinge_gate), as a function of distance to the junction, split by role.

We run the pytorch-only sim (sim_torch.simulate, nonn kernel + role gate) with the proxy
ON, and record for every yielder/passer in the junction window the command BEFORE and AFTER
the proxy step (before = post role-gate input, after = proxy output — isolating the proxy's
own correction).  We then plot

    Δa = a_after − a_before        (the proxy's leader-accelerate / follower-brake)

vs distance to the junction entry, averaged over the vehicles the proxy ACTUALLY changed
(corrected-only, |Δa| > tol), with a ±std band.  Left = yielders (expect Δa<0, braked as
followers); right = passers (expect Δa>0, accelerated as leaders).

    conda run -n car-following-sumo python plot_gate.py [nseed] [flow]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sim_torch as S
import utils
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "outputs")

NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
FLOW  = int(sys.argv[2]) if len(sys.argv) > 2 else 500
# PROXY τ_safety, DECOUPLED from the kernel δ_safe (kernel stays at utils.DELTA_SAFE=3).
# e.g. 5 ⇒ kernel targets a 3 s gap, the proxy enforces a stricter 5 s gap on top.
PROXY_TS = float(sys.argv[3]) if len(sys.argv) > 3 else None
TOL   = 1e-4        # |Δa| above this ⇒ the proxy corrected this vehicle (corrected-only mean)
MIN_N = 5           # don't draw a distance bin with fewer than this many corrected samples
D_LO, D_HI, NB = -10.0, 60.0, 30     # distance-to-junction bins (m)


def collect():
    """Run nonn kernel + role gate + PROXY hinge over NSEED seeds; gather every probe row."""
    geo, s_cp, path_len, s_junc = S.build_geometry()
    rows = []
    for seed in range(NSEED):
        log = []
        # proxy as the SOLE inequality gate (role_gate=False) so before=kernel, after=proxy
        # isolates the proxy's full leader-accelerate / follower-brake correction.
        # proxy_delta_safe decouples the proxy's τ_safety from the kernel's δ_safe.
        r = S.simulate(FLOW, seed, geo, s_cp, path_len, s_junc,
                       hinge_proxy=True, role_gate=False,
                       proxy_delta_safe=PROXY_TS, gate_log=log)
        rows.extend(log)
        print(f"  seed {seed}: {len(log)} windowed yielder/passer samples, "
              f"{r['collided']} collided, {r['n_cross']} cross pairs")
    d  = np.array([x[0] for x in rows], dtype=float)
    rl = np.array([x[1] for x in rows])
    a0 = np.array([x[2] for x in rows], dtype=float)
    a1 = np.array([x[3] for x in rows], dtype=float)
    return d, rl, a1 - a0          # distance, role, Δa (proxy correction)


def binned_mean_std(d, da):
    edges = np.linspace(D_LO, D_HI, NB + 1)
    ctr   = 0.5 * (edges[:-1] + edges[1:])
    idx   = np.clip(np.digitize(d, edges) - 1, 0, NB - 1)
    mean  = np.full(NB, np.nan); std = np.full(NB, np.nan)
    for b in range(NB):
        sel = idx == b
        if int(sel.sum()) >= MIN_N:
            mean[b] = da[sel].mean(); std[b] = da[sel].std()
    return ctr, mean, std


def main():
    plot_style.apply()
    ts = PROXY_TS if PROXY_TS is not None else utils.DELTA_SAFE
    print(f"running {NSEED} seed(s) @ {FLOW} vph, sim_torch nonn, proxy SOLE gate, "
          f"kernel δ_safe={utils.DELTA_SAFE:.1f}s, proxy τ_safety={ts:.1f}s …")
    d, rl, da = collect()

    corr = np.abs(da) > TOL                      # which samples the proxy actually changed
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True, sharey=True)
    panels = [("yield", axes[0], "#DC2828", r"Change in Acceleration at $\gamma_i = 0$"),
              ("pass",  axes[1], plot_style.GREEN, r"Change in Acceleration at $\gamma_i = 1$")]
    for role, ax, color, title in panels:
        m_all  = (rl == role)                     # all windowed vehicles of this role
        m_corr = m_all & corr                     # the ones the proxy actually corrected
        ax.axhline(0, color=plot_style.INK, lw=0.8, zorder=2)
        ax.axvline(0, color=plot_style.MUTED, lw=1.0, ls=":", zorder=2, label="junction entry")
        # SCATTER every correction: one dot per corrected vehicle-step (distance, Δa)
        ax.scatter(d[m_corr], da[m_corr], s=12, color=color, alpha=0.35,
                   edgecolors="none", zorder=3, label=r"$\Delta a_i$")
        ax.set_title(title)
        ax.set_xlabel(r"distance to junction entry (m)")
        ax.set_xlim(D_LO, D_HI)
        ax.legend(loc="upper right")
        plot_style.despine(ax)
    axes[0].set_ylabel(r"$\Delta a_i = a_{\rm after} - a_{\rm before}$  (m/s²)")
    fig.suptitle("Acceleration correction by inequality terms")
    out = os.path.join(OUT, "gate_correction")
    plot_style.save(fig, out, dpi=130); print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    main()
