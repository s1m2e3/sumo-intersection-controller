"""
demo_dynamic_gap.py — leader/follower simulation of the gap-ratio controller.

A single follower is driven by utils.controller_acceleration(); the leader
follows a scripted speed profile that exercises every regime:

    catching up      — follower starts far behind & slow → accelerates (g large)
    gap maintaining  — leader cruises steady → follower holds the desired gap (g→1)
    braking          — leader decelerates hard → follower brakes (g → 0)
    re-acceleration  — leader speeds back up → follower catches up again

Runs twice: Angle 1 (no damping) and Angle 1+2 (first-order-lag damping), and
reports the per-phase acceleration chatter for each.  Saves one PNG per run.

    conda run -n car-following-sumo python demo_dynamic_gap.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import utils

DT          = 0.1     # s   integration step
T_END       = 70.0    # s   total sim time
V_MAX_PHYS  = 20.0    # m/s follower physical top speed (set high so v₀ is the binding cap)


def leader_speed(t: float) -> float:
    if t < 18.0:                       # pull away ABOVE v₀ → follower caps at v₀
        return 15.0
    if t < 30.0:                       # slow to 11 → follower catches up & maintains
        return 15.0 + (11.0 - 15.0) * (t - 18.0) / 12.0
    if t < 40.0:                       # hard decel 11 → 3
        return 11.0 + (3.0 - 11.0) * (t - 30.0) / 10.0
    if t < 52.0:                       # crawl — low-speed gap maintaining
        return 3.0
    if t < 62.0:                       # re-accelerate 3 → 11
        return 3.0 + (11.0 - 3.0) * (t - 52.0) / 10.0
    return 11.0


def simulate(kappa: float = 1.0):
    n = int(T_END / DT)
    t_arr = np.arange(n) * DT

    x_f, v_f = 0.0, 3.0       # follower starts slow and far back → open-road run first
    x_l, v_l = 160.0, leader_speed(0.0)
    a_prev = torch.tensor(0.0)

    rec = {k: np.zeros(n) for k in
           ("xf", "xl", "vf", "vl", "gap", "des", "g", "acc")}

    for i in range(n):
        t = i * DT
        v_l = leader_speed(t)

        xe = torch.tensor(x_f); xl = torch.tensor(x_l)
        ve = torch.tensor(v_f); vl = torch.tensor(v_l)

        a = utils.controller_acceleration(xe, xl, ve, vl, a_prev=a_prev, kappa=kappa)
        a_prev = a

        rec["xf"][i], rec["xl"][i] = x_f, x_l
        rec["vf"][i], rec["vl"][i] = v_f, v_l
        rec["gap"][i] = x_l - x_f - utils.L_VEH
        rec["des"][i] = float(utils.desired_gap(ve, vl))
        rec["g"][i]   = float(utils.gap_ratio(xe, xl, ve, vl))
        rec["acc"][i] = float(a)

        v_f = min(max(v_f + float(a) * DT, 0.0), V_MAX_PHYS)
        x_f += v_f * DT
        x_l += v_l * DT

    return t_arr, rec


def chatter(t, rec, t0, t1):
    m = (t >= t0) & (t < t1)
    a = rec["acc"][m]
    flips = int(np.sum(np.abs(np.diff(np.sign(a))) > 0))
    return a.std(), flips, int(m.sum())


def plot(t, rec, title, out):
    phases = [
        (0.0,  18.0, "#cfe8ff", "open road → cap at v₀"),
        (18.0, 30.0, "#e8e0ff", "catch up → maintain"),
        (30.0, 40.0, "#ffd9d9", "braking"),
        (40.0, 52.0, "#d8f5d8", "low-speed maintain"),
        (52.0, 70.0, "#fff0cc", "re-accelerate"),
    ]
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    def shade(a):
        for t0, t1, c, _ in phases:
            a.axvspan(t0, t1, color=c, alpha=0.5, lw=0)

    a = ax[0, 0]; shade(a)
    a.plot(t, rec["xl"], label="leader",   color="#1f77b4", lw=2)
    a.plot(t, rec["xf"], label="follower", color="#d62728", lw=2, ls="--")
    a.set_ylabel("position (m)"); a.set_title("trajectories"); a.legend(loc="upper left")
    for t0, t1, _, lbl in phases:
        a.text((t0 + t1) / 2, a.get_ylim()[1] * 0.97, lbl, ha="center", va="top",
               fontsize=8, style="italic", color="#444")

    a = ax[0, 1]; shade(a)
    a.plot(t, rec["vl"], label="leader",   color="#1f77b4", lw=2)
    a.plot(t, rec["vf"], label="follower", color="#d62728", lw=2, ls="--")
    a.axhline(utils.V0, color="green", lw=1.0, ls=":", label="v₀ (free speed)")
    a.set_ylabel("speed (m/s)"); a.set_title("speeds"); a.legend(loc="upper right")

    a = ax[1, 0]; shade(a)
    a.plot(t, rec["gap"], label="net gap Δx",      color="#2ca02c", lw=2)
    a.plot(t, rec["des"], label="desired gap s_des", color="#9467bd", lw=1.5, ls=":")
    a.axhline(0.0, color="k", lw=0.8)
    a.set_ylabel("gap (m)"); a.set_xlabel("time (s)")
    a.set_title("gap vs. desired gap"); a.legend(loc="upper right")

    a = ax[1, 1]; shade(a)
    a.plot(t, rec["acc"], color="#d62728", lw=2, label="follower accel")
    a.axhline(utils.A_MAX,  color="gray", lw=0.8, ls="--")
    a.axhline(-utils.B_MAX, color="gray", lw=0.8, ls="--")
    a.set_ylabel("acceleration (m/s²)"); a.set_xlabel("time (s)")
    a.set_title("controller output & gap ratio g")
    a2 = a.twinx()
    a2.plot(t, rec["g"], color="#ff7f0e", lw=1.2, alpha=0.8)
    a2.axhline(1.0, color="#ff7f0e", lw=0.8, ls=":")   # g = 1 equilibrium
    a2.set_ylabel("gap ratio g", color="#ff7f0e")
    a2.set_ylim(0, utils.G_MAX * 1.05)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


def report(label, t, rec):
    print(f"\n[{label}]  min net gap = {rec['gap'].min():.2f} m")
    for lbl, (t0, t1) in {
        "catch-up 5-25":        (5, 25),
        "high-spd maintain 25-30": (25, 30),
        "low-spd maintain 42-52":  (42, 52),
    }.items():
        s, f, n = chatter(t, rec, t0, t1)
        print(f"    {lbl:24s} accel std={s:5.2f}  sign-flips={f:3d}/{n}")


if __name__ == "__main__":
    # Angle 1 — gap-ratio feature, no damping
    t, rec = simulate(kappa=1.0)
    report("Angle 1 (no damping)", t, rec)
    plot(t, rec, "Gap-ratio controller — Angle 1 (no damping)",
         "dynamic_gap_demo_angle1.png")

    # Angle 1 + 2 — first-order-lag damping (brake-exempt)
    t, rec = simulate(kappa=0.3)
    report("Angle 1+2 (kappa=0.3)", t, rec)
    plot(t, rec, "Gap-ratio controller — Angle 1+2 (κ=0.3 damping, brake-exempt)",
         "dynamic_gap_demo_angle2.png")
