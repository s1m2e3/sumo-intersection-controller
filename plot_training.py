"""
Training plots.

Three subplots on a shared time axis (one line / band per snapshot epoch):

  1. TTC*  — minimum across all active vehicles at each timestep
  2. f̂_θ  — max and min across all vehicles (shaded band per epoch)
  3. h_=   — max and min across all vehicles (shaded band per epoch)

Colour goes dark → bright with epoch number.

Run:  python plot_training.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

SNAPSHOTS_PATH = Path("logs") / "epoch_snapshots.pt"
PLOTS_DIR      = Path("logs") / "plots"


def nansmooth(values, window=3):
    out = []
    for i in range(len(values)):
        lo  = max(0, i - window // 2)
        hi  = min(len(values), i + window // 2 + 1)
        seg = [v for v in values[lo:hi] if v == v]   # drop nan
        out.append(sum(seg) / len(seg) if seg else float("nan"))
    return out


def main():
    if not SNAPSHOTS_PATH.exists():
        print(f"No snapshots found at {SNAPSHOTS_PATH}. Run train.py first.")
        return

    snapshots: dict = torch.load(SNAPSHOTS_PATH, weights_only=False)
    epochs = sorted(snapshots.keys())
    if not epochs:
        print("Snapshot file is empty.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(epochs)} snapshots: epochs {epochs[0]}–{epochs[-1]}")

    cmap   = cm.viridis
    colors = {e: cmap(i / max(len(epochs) - 1, 1)) for i, e in enumerate(epochs)}

    # read a_max from the model so axis limits are physics-consistent
    from model import HybridModel
    _m   = HybridModel()
    a_max = _m.physics.a_max.item()

    fig, (ax_ttc, ax_fhat, ax_h) = plt.subplots(
        3, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"hspace": 0.35}
    )
    fig.suptitle(
        "Hybrid Model Training — TTC*, f̂_θ, h_=  per snapshot epoch\n"
        "(colour: dark = early epoch → bright = late epoch)",
        fontsize=12
    )

    for epoch in epochs:
        snap  = snapshots[epoch]
        time  = snap["time"]
        color = colors[epoch]
        label = f"ep {epoch}"

        # ── 1. TTC* minimum ────────────────────────────────────────────────
        ttc_min = nansmooth(snap["ttc_min"])
        ax_ttc.plot(time, ttc_min, color=color, lw=1.4, label=label, alpha=0.85)

        # ── 2. f̂_θ  band (max / min across vehicles) ──────────────────────
        fmax = nansmooth(snap["fhat_max"])
        fmin = nansmooth(snap["fhat_min"])
        t    = np.array(time)
        ax_fhat.fill_between(t, fmin, fmax,
                              color=color, alpha=0.18, linewidth=0)
        ax_fhat.plot(t, fmax, color=color, lw=0.9, alpha=0.7)
        ax_fhat.plot(t, fmin, color=color, lw=0.9, alpha=0.7, label=label)

        hmax = nansmooth(snap["h_max"])
        hmin = nansmooth(snap["h_min"])
        ax_h.fill_between(t, hmin, hmax,
                           color=color, alpha=0.18, linewidth=0)
        ax_h.plot(t, hmax, color=color, lw=0.9, alpha=0.7)
        ax_h.plot(t, hmin, color=color, lw=0.9, alpha=0.7, label=label)

    # ── TTC reference lines ───────────────────────────────────────────────
    t_all = snapshots[epochs[0]]["time"]
    ax_ttc.axhline(3.0, color="red",    lw=1.1, ls="--", alpha=0.8,
                   label="TTC*=3 (dec boundary)")
    ax_ttc.axhline(5.0, color="orange", lw=1.1, ls="--", alpha=0.8,
                   label="TTC*=5 (ff boundary)")
    ax_ttc.fill_between([t_all[0], t_all[-1]], 3, 5,
                         color="yellow", alpha=0.10, label="NN region [3–5s]")
    ax_ttc.set_ylim(0, 35)

    # ── zero reference lines ──────────────────────────────────────────────
    ax_fhat.axhline(0, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax_h.axhline(0,    color="gray", lw=0.8, ls=":", alpha=0.6)

    # ── axis labels ───────────────────────────────────────────────────────
    ax_ttc.set_ylabel("min TTC*  (s)", fontsize=10)
    ax_ttc.set_title("Minimum TTC* across all vehicles", fontsize=10)
    ax_ttc.legend(fontsize=7, loc="upper right", ncol=3)

    ax_fhat.set_ylabel("f̂_θ  (m/s²)", fontsize=10)
    ax_fhat.set_title("f̂_θ  range across vehicles  [band = min…max]", fontsize=10)
    ax_fhat.legend(fontsize=7, loc="upper right", ncol=3)

    ax_h.set_ylabel("h_=  (m/s²)", fontsize=10)
    ax_h.set_title(
        f"h_=  range across vehicles  [band = min…max]  —  bounded ±3·a_max = ±{3*a_max:.1f} m/s²",
        fontsize=10
    )
    ax_h.legend(fontsize=7, loc="upper right", ncol=3)

    ax_h.set_xlabel("Simulation time  (s)", fontsize=10)

    # ── colourbar to read epoch number ────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=epochs[0], vmax=epochs[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_h, orientation="vertical",
                        fraction=0.015, pad=0.02)
    cbar.set_label("Epoch", fontsize=9)

    out = PLOTS_DIR / "training_evolution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    main()
