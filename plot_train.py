"""Render training curves from outputs/train_log.csv (written per epoch by train_mean.py).
Re-run anytime during/after training to see how the metrics evolve.

    conda run -n car-following-sumo python plot_train.py            # outputs/train_log.csv
    conda run -n car-following-sumo python plot_train.py path.csv

Output: outputs/train_curves.png.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "outputs")
SRC  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "train_log.csv")


def main():
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} not found — run train_mean.py (it writes the log per epoch).")
    d = np.genfromtxt(SRC, delimiter=",", names=True)
    if d.size == 0:
        sys.exit(f"{SRC} has no rows yet — let at least one epoch finish.")
    d = np.atleast_1d(d)
    ep = d["epoch"]

    plot_style.apply()
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    # (0,0) throughput / objective
    a = ax[0, 0]
    a.plot(ep, d["mean_J"], "-o", ms=3, color=plot_style.BLUE,   label="mean J̄ (throughput)")
    a.plot(ep, d["score"],  "-s", ms=3, color=plot_style.ORANGE, label="score (J̄ − λ_s·hinge)")
    a.set_title("Objective / throughput"); a.set_xlabel("epoch"); a.legend()

    # (0,1) safety hinge
    a = ax[0, 1]
    a.plot(ep, d["mean_hinge"], "-o", ms=3, color=plot_style.PINK, label="mean hinge (near-miss)")
    a.set_title("Safety hinge"); a.set_xlabel("epoch"); a.legend()

    # (1,0) THE NEW TERMS — want these to DROP as f learns to smooth
    a = ax[1, 0]
    a.plot(ep, d["eff_a2"],   "-o", ms=3, color=plot_style.BLUE,   label="effort  ⟨a²⟩")
    a.plot(ep, d["jerk_da2"], "-s", ms=3, color=plot_style.ORANGE, label="jerk  ⟨(Δa)²⟩")
    a.set_title("Smoothness / energy  (target: ↓)"); a.set_xlabel("epoch")
    a.legend()

    # (1,1) f activity — want these to GROW (f doing real work)
    a = ax[1, 1]
    a.plot(ep, d["f_abs"],      "-o", ms=3, color=plot_style.BLUE,   label="⟨|f|⟩ (probe)")
    a.plot(ep, d["f_conflict"], "-s", ms=3, color=plot_style.ORANGE, label="|f| conflict")
    a.plot(ep, d["f_free"],     "-^", ms=3, color=plot_style.GREEN,  label="|f| free")
    a.plot(ep, d["f_max"],      ":",  lw=2.0, color=plot_style.BASELINE, label="max |f|")
    a.set_title("Learned-mean activity  (target: ↑ then settle)"); a.set_xlabel("epoch")
    a.legend(fontsize=8)

    fig.suptitle(f"Training curves — {len(ep)} epochs  [{os.path.basename(SRC)}]")
    out = os.path.join(OUT, "train_curves")
    plot_style.save(fig, out, dpi=150)
    print(f"saved {out}.png / .pdf  ({len(ep)} epochs)")


if __name__ == "__main__":
    main()
