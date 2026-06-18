"""Line plots comparing throughput and conflict gap vs demand for three controllers:

  Krauss                      — native SUMO         (tc_sweep.json: krauss)
  Created model (untrained)   — plain zero-mean kernel, no NN  (tc_sweep.json: cosim_nonn)
  Created model (trained)     — kernel + trained MLP prior mean (tc_sweep_nn.json: cosim_nn)

  outputs/throughput_vs_demand.png    — steady-state throughput (veh/h) vs demand (vph)
  outputs/conflict_gap_vs_demand.png  — mean conflict gap τ over the run vs demand

The literal MAX τ_c saturates at the soft-clamp ceiling for every controller, so it carries
no signal; the mean τ_c is the informative central tendency and is what we draw.

    conda run -n car-following-sumo python plot_tc_sweep.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import plot_style

HERE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(HERE, "outputs")
SRC    = os.path.join(OUT, "tc_sweep.json")             # cosim_nonn + krauss
SRC_NN = os.path.join(OUT, "tc_sweep_mlp_ctx_best.json")  # cosim_nn = hinge-only trained model

# draw order (front-to-back is reversed by zorder in fancy_line)
CTRLS = ("cosim_nn", "cosim_nonn", "krauss")
LABEL = {"cosim_nn":   "Created model (trained)",
         "cosim_nonn": "Created model (untrained)",
         "krauss":     "Krauss"}
COLOR = {"cosim_nn": plot_style.BLUE, "cosim_nonn": plot_style.PURPLE, "krauss": plot_style.ORANGE}
MARK  = {"cosim_nn": "o", "cosim_nonn": "^", "krauss": "s"}


def load_averages():
    avg = {}
    for path in (SRC, SRC_NN):
        if os.path.exists(path):
            avg.update(json.load(open(path))["averages"])
    return avg


def series(avg, ctrl, key):
    flows = sorted(int(f) for f in avg[ctrl])
    return flows, [avg[ctrl][str(f)][key] for f in flows]


def fancy_line(ax, x, y, color, marker, label):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ax.plot(x, y, "-", color=color, lw=2.6, zorder=3, label=label, solid_capstyle="round")
    ax.plot(x, y, linestyle="none", marker=marker, ms=8.5, color=color,
            markeredgecolor=plot_style.SNOW, markeredgewidth=1.5, zorder=5)


def end_label(ax, x, y, color, fmt):
    """Value label at the rightmost point only — keeps 3 lines uncluttered."""
    ax.annotate(fmt(y[-1]), (x[-1], y[-1]), textcoords="offset points", xytext=(9, 0),
                ha="left", va="center", fontsize=9, color=color, fontweight="bold",
                zorder=6, path_effects=[pe.withStroke(linewidth=2.2, foreground=plot_style.SNOW)])


def make_fig(avg, key, ylabel, title, stem, fmt, pad_frac=0.16, max_flow=None):
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    data = {}
    for c in CTRLS:
        if c in avg and avg[c]:
            xs, ys = series(avg, c, key)
            if max_flow is not None:                       # cap the x-range for this plot
                xs, ys = map(list, zip(*[(x, y) for x, y in zip(xs, ys) if x <= max_flow]))
            data[c] = (xs, ys)
    allx = sorted({xi for x, _ in data.values() for xi in x})
    ally = [yi for _, y in data.values() for yi in y]
    lo, hi = min(ally), max(ally); span = (hi - lo) or 1.0
    floor = lo - pad_frac * span
    ax.set_ylim(floor, hi + pad_frac * span)
    ax.set_xlim(min(allx) - 20, max(allx) + 80)        # right pad for end labels
    for c, (x, y) in data.items():
        fancy_line(ax, x, y, COLOR[c], MARK[c], LABEL[c])
        end_label(ax, x, y, COLOR[c], fmt)
    ax.set_xticks(allx)
    ax.set_xlabel("Demand (vph per approach)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    leg = ax.legend(loc="best", frameon=True, fancybox=True, framealpha=0.95)
    leg.get_frame().set_edgecolor(plot_style.GRID)
    plot_style.despine(ax)
    plot_style.save(fig, os.path.join(OUT, stem))
    plt.close(fig)
    print(f"saved {stem}.png / .pdf  ({', '.join(data)})")


def main():
    avg = load_averages()
    if not avg:
        raise SystemExit("no results found — run run_tc_sweep.py first.")
    plot_style.apply()
    make_fig(avg, "vph", "Throughput (veh/h)", "Throughput vs demand",
             "throughput_vs_demand", fmt=lambda v: f"{v:.0f}", max_flow=700)
    make_fig(avg, "tau_c_mean", r"Mean conflict gap  $\bar{\tau}_i$  (s)",
             "Mean conflict gap vs demand", "conflict_gap_vs_demand",
             fmt=lambda v: f"{v:.2f}")


if __name__ == "__main__":
    main()
