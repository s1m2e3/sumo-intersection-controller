"""
investigate_crashes.py — locate and dissect the rear-end crashes.

Runs the intersection sim at a saturating flow, isolates SAME-LANE (rear-end)
collisions, classifies WHERE each one happens (approach / junction / exit), and
dumps the kinematic state of both vehicles through the collision so we can see how
the car-following law fails.

    junction zone : |p| < JUNCTION_R   (paths physically cross here)
    approach      : p <= -JUNCTION_R   (upstream, before the centre)
    exit          : p >=  JUNCTION_R   (downstream, past the centre)

    conda run -n car-following-sumo python investigate_crashes.py
"""
from __future__ import annotations
import numpy as np
import demo_intersection_sim as D
import utils

JUNCTION_R = 10.0     # m, half-width of the "in junction" zone around the centre
FLOW       = 800.0    # veh/h/approach — saturating, lots of rear-ends
SEED       = 0


def zone(p):
    if abs(p) < JUNCTION_R:
        return "JUNCTION"
    return "approach" if p < 0 else "exit"


def state_at(vstate, vid, t):
    """Return (p, v, a, lane) of vid at time t (nearest step), or None."""
    for (tt, p, v, a, ln) in vstate[vid]:
        if abs(tt - t) < D.DT / 2:
            return p, v, a, ln
    return None


def main():
    t, log, crashes, traj, completed, vstate = D.simulate(flow_vph=FLOW, seed=SEED)

    # distinct same-lane (rear-end) pairs = those recorded with sep < 0
    rear_pairs = {}
    for (tc, vi, vj, sep) in crashes:
        if sep < 0:
            key = frozenset((vi, vj))
            if key not in rear_pairs or tc < rear_pairs[key][0]:
                rear_pairs[key] = (tc, vi, vj, sep)

    print(f"=== rear-end crash investigation  (flow={FLOW:.0f} vph, seed={SEED}) ===")
    print(f"distinct rear-end pairs: {len(rear_pairs)}\n")

    # classify all by zone at first contact
    by_zone = {"approach": 0, "JUNCTION": 0, "exit": 0}
    for key, (tc, vi, vj, sep) in rear_pairs.items():
        si, sj = state_at(vstate, vi, tc), state_at(vstate, vj, tc)
        if si is None or sj is None:
            continue
        # leader = larger p (further along path)
        (lead, lp), (foll, fp) = ((vi, si), (vj, sj)) if si[0] > sj[0] else ((vj, sj), (vi, si))
        by_zone[zone(fp[0])] += 1

    print("WHERE rear-ends first occur (by follower's zone):")
    for z, n in by_zone.items():
        print(f"   {z:9s}: {n}")
    print()

    # detailed dump of the first few rear-ends
    print("=== detailed state through the first 4 rear-ends ===")
    for key, (tc, vi, vj, sep) in sorted(rear_pairs.items(), key=lambda kv: kv[1][0])[:4]:
        si, sj = state_at(vstate, vi, tc), state_at(vstate, vj, tc)
        (lead, _), (foll, _) = ((vi, si), (vj, sj)) if si[0] > sj[0] else ((vj, sj), (vi, si))
        ln = state_at(vstate, foll, tc)[3]
        print(f"\n-- pair (lead={lead}, follow={foll}) lane={D._LANE_NAME[ln]} "
              f"first contact t={tc:.1f}s, zone={zone(state_at(vstate, foll, tc)[0])} --")
        print(f"   {'t':>5} | {'lead_p':>7} {'lead_v':>6} {'lead_a':>6} {'zone_L':>8} "
              f"| {'foll_p':>7} {'foll_v':>6} {'foll_a':>6} {'zone_F':>8} "
              f"| {'gap':>6} {'clos.dv':>7}")
        for tt in np.arange(tc - 1.5, tc + 0.3, D.DT):
            sL, sF = state_at(vstate, lead, tt), state_at(vstate, foll, tt)
            if sL is None or sF is None:
                continue
            gap = (sL[0] - sF[0]) - utils.L_VEH
            dv = sF[1] - sL[1]
            flag = "  <-- CONTACT" if gap < 0 else ""
            print(f"   {tt:5.1f} | {sL[0]:7.1f} {sL[1]:6.2f} {sL[2]:6.2f} {zone(sL[0]):>8} "
                  f"| {sF[0]:7.1f} {sF[1]:6.2f} {sF[2]:6.2f} {zone(sF[0]):>8} "
                  f"| {gap:6.2f} {dv:7.2f}{flag}")


if __name__ == "__main__":
    main()
