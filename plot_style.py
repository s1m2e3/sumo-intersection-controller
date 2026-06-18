"""Shared plotting aesthetic for this repo's figures.

Mirrors the look of the paper figure `gru_time_mae`: a soft *snow* canvas,
an open (despined) frame, faint grids, a Material-Design accent palette used
semantically (color = series identity), and dual PDF+PNG export.

    import plot_style
    plot_style.apply()                  # set rcParams once, before any plotting
    fig, ax = plt.subplots(...)
    plot_style.despine(ax)              # drop top/right spines (axes the apply()
                                        # rcParams miss, e.g. created before apply)
    plot_style.save(fig, "outputs/foo") # writes foo.png (dpi=150) + foo.pdf
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# ── Canvas / ink ────────────────────────────────────────────────────────────────
SNOW  = "#FFFAFA"   # soft, warm off-white — the "snow" background
INK   = "#222222"   # near-black for spines / text
MUTED = "#555555"   # secondary annotations
GRID  = "#9AA0A6"   # cool grey grid lines

# ── Material-Design accents (color = series identity) ────────────────────────────
BLUE   = "#1976D2"   # primary  / accelerate
PINK   = "#E91E63"   # secondary
GREEN  = "#43A047"
ORANGE = "#FF7043"
PURPLE = "#8E24AA"
TEAL   = "#00897B"
BASELINE = "#7A7A7A"  # neutral reference (use dotted + thick, like the IDM line)

PALETTE = [BLUE, PINK, GREEN, ORANGE, PURPLE, TEAL]


def apply() -> None:
    """Install the snow aesthetic globally via rcParams. Call once up front."""
    plt.rcParams.update({
        # snow canvas everywhere (figure, axes, and exported file)
        "figure.facecolor":  SNOW,
        "axes.facecolor":    SNOW,
        "savefig.facecolor": SNOW,
        # ink
        "axes.edgecolor":  INK,
        "axes.labelcolor": INK,
        "text.color":      INK,
        "xtick.color":     INK,
        "ytick.color":     INK,
        # color = series identity
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        # open frame
        "axes.spines.top":   False,
        "axes.spines.right": False,
        # faint grid (matches the reference's alpha=0.25, lw=0.7)
        "axes.grid":      True,
        "grid.color":     GRID,
        "grid.alpha":     0.25,
        "grid.linewidth": 0.7,
        # Arial italic everywhere (title / axis labels / ticks) — imitates the
        # diffeq `rbf_vs_matern` figure; legend stays upright (set per-axes below).
        "font.family":     "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.style":      "italic",
        # type hierarchy from the reference helper (title 15 / labels 13 / ticks 11)
        "figure.titlesize":  16,
        "axes.titlesize":    15,
        "axes.labelsize":    13,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "legend.fontsize":   9,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  GRID,
        "font.size":         11,
    })


def despine(ax) -> None:
    """Hide the top and right spines (the open 'L' frame)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, stem: str, dpi: int = 150) -> None:
    """Export `stem.png` (raster, dpi) and `stem.pdf` (vector) on a snow canvas."""
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor=SNOW)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor=SNOW)
