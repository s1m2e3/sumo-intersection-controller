"""
demo_cross_conflict.py — multi-agent intersection demo of the 2-D kernel controller.

Every vehicle (ego + rivals) is driven by the SAME utils.controller_acceleration()
on its own approach lane toward a shared conflict point at the origin.  No vehicle
talks to any other: each only sees the others' distance/speed to the conflict point.
Because a rival runs the identical rule and sees δ_rival = −δ, the pair self-assigns
yield/pass — one brakes, one goes — with no communication.

Two scenarios, one PNG each:

  Scenario A  (cross_conflict_demo.png)
      Pure cross-conflict.  Ego heads east, two rivals head north on the crossing
      road (one urgent, one far).  Shows min-over-rivals selection and the
      decentralized yield/pass split.  Longitudinal leaders are far (open road),
      so the cross feature is what shapes the motion.

  Scenario B  (cross_following_demo.png)
      Car-following AND cross-resolution at once.  Ego follows a scripted leader on
      its own lane (gap-ratio control) while a rival crosses from the south.  The
      ego must hold its gap to the leader and yield/pass to the rival simultaneously.

    conda run -n car-following-sumo python demo_cross_conflict.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import utils

DT      = 0.1      # s   integration step
T_END   = 16.0     # s   total sim time (vehicles clear the junction)
V_PHYS  = 20.0     # m/s physical top speed (so v₀ is the binding cap)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-agent intersection simulator
# ─────────────────────────────────────────────────────────────────────────────
#
# Each "crosser" travels along a straight approach lane toward the conflict point
# at the origin.  Its state is a signed along-path coordinate s (s<0 → before the
# point, s>0 → already through) plus speed v and a unit heading `dir`.
#
#   2-D position  = dir · s
#   dist to CP    = |s|         (a conflict only while s<0, i.e. still approaching)
#   ETA to CP  η  = |s| / v
#
# Rivals of vehicle i = every OTHER crosser (they share the conflict point).  A
# same-lane leader is handled separately via each crosser's `leader` callable and
# is NOT a cross rival.

def _far_leader(t, s, v):
    """Open-road longitudinal leader: 300 m ahead at matched speed → g saturates."""
    return s + 300.0, v


def simulate(crossers, t_end=T_END):
    """
    crossers: list of dicts, each with
        name   str
        dir    (dx, dy) unit heading in the plane
        s0     float    initial signed along-path coordinate (negative = approaching)
        v0     float    initial speed
        leader callable(t, s, v) -> (s_lead, v_lead)   (defaults to _far_leader)
    Returns t_arr and a per-vehicle record dict.
    """
    n = int(t_end / DT)
    t_arr = np.arange(n) * DT
    N = len(crossers)

    s = [c["s0"] for c in crossers]
    v = [c["v0"] for c in crossers]
    leaders = [c.get("leader", _far_leader) for c in crossers]

    rec = {c["name"]: {k: np.full(n, np.nan) for k in
                       ("x", "y", "s", "v", "eta", "tau_c", "acc", "gap", "des")}
           for c in crossers}

    for step in range(n):
        t = step * DT
        accs = [0.0] * N
        for i in range(N):
            si, vi = s[i], v[i]
            s_lead, v_lead = leaders[i](t, si, vi)

            xe = torch.tensor(si);       xl = torch.tensor(s_lead)
            ve = torch.tensor(vi);       vl = torch.tensor(v_lead)

            others = [j for j in range(N) if j != i]
            rd  = torch.tensor([max(-s[j], utils.EPS) for j in others])
            rv  = torch.tensor([max(v[j], utils.EPS)   for j in others])
            val = torch.tensor([(s[j] < 0.0) and (si < 0.0) for j in others])
            d_conf = torch.tensor(max(-si, utils.EPS))

            a = utils.controller_acceleration(
                xe, xl, ve, vl,
                d_conf=d_conf, rival_d=rd, rival_v=rv, rival_valid=val)
            accs[i] = float(a)

            # diagnostics (recompute the cross feature for logging)
            tau_c, _, _, any_r, _, _ = utils.conflict_time_gap(d_conf, ve, rd, rv, val)
            r = rec[crossers[i]["name"]]
            dx, dy = crossers[i]["dir"]
            r["x"][step], r["y"][step] = dx * si, dy * si
            r["s"][step], r["v"][step] = si, vi
            r["eta"][step] = (max(-si, utils.EPS) / max(vi, utils.EPS)) if si < 0 else np.nan
            r["tau_c"][step] = float(tau_c) if bool(any_r) else np.nan
            r["acc"][step] = accs[i]
            r["gap"][step] = (s_lead - si - utils.L_VEH)
            r["des"][step] = float(utils.desired_gap(ve, vl))

        # integrate
        for i in range(N):
            v[i] = min(max(v[i] + accs[i] * DT, 0.0), V_PHYS)
            s[i] += v[i] * DT

    return t_arr, rec


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = {"ego": "#d62728", "rival": "#1f77b4", "rival-2": "#17becf",
           "leader": "#9467bd", "rival-S": "#1f77b4"}


def _color(name):
    return _COLORS.get(name, "#2ca02c")


def plot_birdseye(ax, t, rec, leader_path=None):
    """Bird's-eye trajectories in the plane with time-dots and the conflict point."""
    ax.axhline(0, color="#cccccc", lw=8, zorder=0)   # east-west road
    ax.axvline(0, color="#cccccc", lw=8, zorder=0)   # north-south road
    ax.plot(0, 0, marker="x", ms=12, mew=3, color="k", zorder=5, label="conflict point")

    for name, r in rec.items():
        c = _color(name)
        ax.plot(r["x"], r["y"], color=c, lw=2, label=name)
        # time dots every 2 s
        for ti in np.arange(0, t[-1] + 1e-9, 2.0):
            k = int(ti / DT)
            if k < len(t) and np.isfinite(r["x"][k]):
                ax.plot(r["x"][k], r["y"][k], "o", color=c, ms=5, zorder=4)
        # start marker
        ax.plot(r["x"][0], r["y"][0], marker="s", ms=8, color=c, zorder=4)

    if leader_path is not None:
        ax.plot(leader_path[0], leader_path[1], color=_color("leader"),
                lw=2, ls="--", label="leader")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("bird's-eye trajectories  (□ start, ● every 2 s, ✕ conflict pt)")
    ax.legend(loc="upper left", fontsize=8)


def plot_eta(ax, t, rec):
    """ETA-to-conflict-point per vehicle.  Curves crossing zero at the SAME time = a
    collision; the controller keeps them temporally separated (that gap is τ_c)."""
    for name, r in rec.items():
        ax.plot(t, r["eta"], color=_color(name), lw=2, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("η = dist/​speed to CP (s)"); ax.set_xlabel("time (s)")
    ax.set_title("ETA to conflict point (separation = τ_c)")
    ax.legend(loc="upper right", fontsize=8); ax.set_ylim(bottom=-0.2)


def plot_tauc(ax, t, rec):
    for name, r in rec.items():
        if np.any(np.isfinite(r["tau_c"])):
            ax.plot(t, r["tau_c"], color=_color(name), lw=2, label=f"{name} τ_c")
    ax.axhline(utils.DELTA_SAFE, color="green", lw=1.0, ls=":", label="δ_safe")
    ax.set_ylabel("conflict time-gap τ_c (s)"); ax.set_xlabel("time (s)")
    ax.set_title("τ_c — worst-rival conflict time-gap"); ax.legend(loc="lower right", fontsize=8)


def plot_speed(ax, t, rec):
    for name, r in rec.items():
        ax.plot(t, r["v"], color=_color(name), lw=2, label=name)
    ax.axhline(utils.V0, color="green", lw=1.0, ls=":", label="v₀")
    ax.set_ylabel("speed (m/s)"); ax.set_xlabel("time (s)")
    ax.set_title("speeds"); ax.legend(loc="lower right", fontsize=8)


def plot_accel(ax, t, rec):
    for name, r in rec.items():
        ax.plot(t, r["acc"], color=_color(name), lw=2, label=name)
    ax.axhline(utils.A_MAX, color="gray", lw=0.8, ls="--")
    ax.axhline(-utils.B_MAX, color="gray", lw=0.8, ls="--")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("acceleration (m/s²)"); ax.set_xlabel("time (s)")
    ax.set_title("controller output (+ pass / − yield)"); ax.legend(loc="upper right", fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario A — pure cross-conflict, ego + 2 rivals
# ─────────────────────────────────────────────────────────────────────────────

def scenario_A():
    crossers = [
        {"name": "ego",     "dir": (1.0, 0.0), "s0": -50.0, "v0": 10.0},  # east,  η0=5.0
        {"name": "rival",   "dir": (0.0, 1.0), "s0": -40.0, "v0": 10.0},  # north, η0=4.0 (urgent)
        {"name": "rival-2", "dir": (0.0, 1.0), "s0": -85.0, "v0": 10.0},  # north, η0=8.5 (far)
    ]
    t, rec = simulate(crossers)

    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Scenario A — pure cross-conflict (ego yields to urgent rival; far rival ignored)",
                 fontsize=13, fontweight="bold")
    plot_birdseye(ax[0, 0], t, rec)
    plot_eta(ax[0, 1], t, rec)
    plot_tauc(ax[0, 2], t, rec)
    plot_speed(ax[1, 0], t, rec)
    plot_accel(ax[1, 1], t, rec)
    ax[1, 2].axis("off")
    _min_gap_report("Scenario A", rec)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("cross_conflict_demo.png", dpi=130)
    print("saved cross_conflict_demo.png")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario B — car-following + cross-conflict together
# ─────────────────────────────────────────────────────────────────────────────

def _leader_profile(t, s, v):
    """Scripted leader on the ego's own lane: cruises, brakes, re-accelerates.
    Returns the leader's along-path position and speed (independent of ego)."""
    if t < 4.0:
        vl = 9.0
    elif t < 7.0:                       # brake 9 → 3
        vl = 9.0 + (3.0 - 9.0) * (t - 4.0) / 3.0
    elif t < 10.0:                      # crawl
        vl = 3.0
    else:                               # re-accelerate 3 → 11
        vl = min(3.0 + (11.0 - 3.0) * (t - 10.0) / 4.0, 11.0)
    # integrate the leader position analytically-ish via cached state
    s_lead = _leader_profile.s
    _leader_profile.s += vl * DT
    return s_lead, vl


def scenario_B():
    _leader_profile.s = -25.0           # leader starts 25 m ahead of ego's −60 m
    crossers = [
        {"name": "ego",     "dir": (1.0, 0.0), "s0": -60.0, "v0": 9.0,
         "leader": _leader_profile},                                   # follows scripted leader
        {"name": "rival-S", "dir": (0.0, 1.0), "s0": -52.0, "v0": 11.0},  # crosses from south
    ]
    t, rec = simulate(crossers)

    # reconstruct the leader's plotted path (east lane, y=0)
    _leader_profile.s = -25.0
    lead_x = []
    for step in range(len(t)):
        lead_x.append(_leader_profile.s)
        _, vl = _leader_profile(step * DT, 0, 0)
    lead_x = np.array(lead_x); lead_y = np.zeros_like(lead_x)

    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Scenario B — car-following (ego↔leader) AND cross-conflict (ego↔rival) together",
                 fontsize=13, fontweight="bold")
    plot_birdseye(ax[0, 0], t, rec, leader_path=(lead_x, lead_y))
    plot_eta(ax[0, 1], t, rec)
    plot_tauc(ax[0, 2], t, rec)
    plot_speed(ax[1, 0], t, rec)
    plot_accel(ax[1, 1], t, rec)

    # gap-to-leader panel (car-following diagnostic for the ego)
    a = ax[1, 2]
    r = rec["ego"]
    a.plot(t, r["gap"], color="#2ca02c", lw=2, label="net gap to leader")
    a.plot(t, r["des"], color="#9467bd", lw=1.5, ls=":", label="desired gap s_des")
    a.axhline(0, color="k", lw=0.8)
    a.set_ylabel("gap (m)"); a.set_xlabel("time (s)")
    a.set_title("ego car-following: gap vs desired"); a.legend(loc="upper right", fontsize=8)

    _min_gap_report("Scenario B", rec)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("cross_following_demo.png", dpi=130)
    print("saved cross_following_demo.png")


# ─────────────────────────────────────────────────────────────────────────────

def _min_gap_report(label, rec):
    """Closest 2-D approach between every crosser pair (collision check)."""
    names = list(rec)
    worst = np.inf
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ri, rj = rec[names[i]], rec[names[j]]
            d = np.hypot(ri["x"] - rj["x"], ri["y"] - rj["y"])
            m = np.nanmin(d)
            worst = min(worst, m)
            print(f"  [{label}] min separation {names[i]:8s} <-> {names[j]:8s} = {m:5.2f} m")
    print(f"  [{label}] overall closest approach = {worst:.2f} m "
          f"({'OK' if worst > utils.L_VEH else 'COLLISION'})")


if __name__ == "__main__":
    scenario_A()
    scenario_B()
