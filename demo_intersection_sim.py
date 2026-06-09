"""
demo_intersection_sim.py — 120 s PyTorch-only multi-vehicle intersection sim.

A 4-way THROUGH-movement intersection (NS, SN, EW, WE), geometry and vehicle type
borrowed from sumo_files/ (center node at origin, 200 m approaches, vType
accel=2.6 decel=4.5 length=5 maxSpeed=13.89, through flows ~1800 veh/h).  There is
NO traffic signal: every vehicle is driven by utils.controller_acceleration(),

    longitudinal  — gap-ratio car-following to its in-lane leader
    lateral/cross — conflict-time yield/pass vs. perpendicular traffic at the centre

The whole vehicle population is stepped as ONE batched controller call per tick.
Vehicles arrive Poisson per approach and are removed once they clear the junction.

Ground truth is checked in real 2-D with lane offsets (so oncoming/parallel lanes
never false-trigger).  A crash = two cars physically overlapping:
    * same lane  : bumper gap (Δp − L_VEH) < 0
    * crossing   : centre-to-centre Euclidean distance < CRASH_R

    conda run -n car-following-sumo python demo_intersection_sim.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import utils

# ── scenario constants (from sumo_files) ────────────────────────────────────────
DT        = 0.1            # s    integration step
T_END     = 120.0         # s    sim duration
APPROACH  = 200.0         # m    spawn distance upstream of the centre (node radius)
EXIT      = 200.0         # m    removal distance downstream
V0        = utils.V0      # 13.89 m/s desired speed
V_PHYS    = 16.0          # m/s  physical cap (> v0 so v0 is the binding limit)
LANE_OFF  = 1.6           # m    half lane-width lateral offset (right-hand driving)
FLOW_VPH  = 300.0         # veh/h per approach.  Borrowed structure from the SUMO
                          #   1800 vph through flows, scaled to the give-way
                          #   controller's robust crash-free ceiling (~300/approach =
                          #   1200 vph total).  Above ~350 it saturates and crashes:
                          #   rear-end (queue/longitudinal authority) then crossing
                          #   (multi-rival centre-approx coordination) — see the
                          #   flow-sweep notes; both are the documented follow-ups.
SPAWN_HW  = 8.0           # m    min clear distance at the entry to admit a new car
CRASH_R   = 2.5           # m    crossing centre-distance crash threshold
SEED      = 0

# lane id → (travel dir unit vector, lateral lane offset, road axis 0=x/1=y)
#   0 WE  west→east   1 EW  east→west   2 NS  north→south   3 SN  south→north
_LANES = {
    0: (np.array([ 1.0,  0.0]), np.array([ 0.0, -LANE_OFF]), 0),
    1: (np.array([-1.0,  0.0]), np.array([ 0.0,  LANE_OFF]), 0),
    2: (np.array([ 0.0, -1.0]), np.array([-LANE_OFF, 0.0]), 1),
    3: (np.array([ 0.0,  1.0]), np.array([ LANE_OFF, 0.0]), 1),
}
_LANE_NAME = {0: "WE", 1: "EW", 2: "NS", 3: "SN"}
_LANE_COLOR = {0: "#d62728", 1: "#ff7f0e", 2: "#1f77b4", 3: "#2ca02c"}

CONFLICT_CLEAR = 6.0   # m  a rival stays "in conflict" until it is this far past its
                       #    own crossing point (one car length + buffer) — replaces the
                       #    binary p<0 flag so a vehicle can't vanish while in the box

# Per-pair crossing coordinate: SCROSS[L, M] = along-path coord (on lane L) at which
# a vehicle on lane L crosses the path of a vehicle on lane M.  Derived from the line
# intersection of the two offset lanes:  s_cross = dir_L · (off_M − off_L).
_SCROSS = np.zeros((4, 4))
for _L, (_dL, _offL, _) in _LANES.items():
    for _M, (_dM, _offM, _) in _LANES.items():
        _SCROSS[_L, _M] = float(_dL @ (_offM - _offL))


def xy(lane, p):
    """2-D position of a vehicle at along-path coord p on `lane` (p<0 = before CP)."""
    d, off, _ = _LANES[int(lane)]
    return off + d * p


# ─────────────────────────────────────────────────────────────────────────────

def simulate(t_end=T_END, flow_vph=FLOW_VPH, seed=SEED):
    rng = np.random.default_rng(seed)
    n_steps = int(t_end / DT)
    p_spawn = flow_vph / 3600.0 * DT          # Poisson arrival prob per lane per step
    eps = utils.EPS

    # dynamic vehicle state (python lists; rebuilt into tensors each tick)
    lane: list[int] = []
    p:    list[float] = []
    v:    list[float] = []

    completed = 0
    log = {k: np.zeros(n_steps) for k in
           ("n_active", "throughput", "min_cross", "min_gap", "min_acc", "mean_v")}
    crashes = []                                # (t, i, j, dist)
    traj = {}                                   # vid -> list[(t,x,y,lane)] for plotting
    vstate = {}                                 # vid -> list[(t,p,v,a,lane)] for analysis
    vid_seq = 0
    vids: list[int] = []

    for step in range(n_steps):
        t = step * DT

        # ── 1. spawn (Poisson per approach, only if the entry is clear) ──────────
        for ln in _LANES:
            if rng.random() < p_spawn:
                back = min((p[i] for i in range(len(p)) if lane[i] == ln),
                           default=APPROACH)
                gap0 = back - (-APPROACH)               # clearance ahead of the entry
                if gap0 > SPAWN_HW:                     # room behind the queue tail
                    # insert at a speed that can still stop before the tail (SUMO-like
                    # safe insertion) so we never spawn fast onto a stopped queue
                    v_safe = min(V0, float(np.sqrt(2.0 * utils.B_MAX *
                                                   max(gap0 - utils.L_VEH - utils.S0, 0.0))))
                    lane.append(ln); p.append(-APPROACH); v.append(v_safe)
                    vids.append(vid_seq); vid_seq += 1

        N = len(p)
        if N == 0:
            log["n_active"][step] = 0
            log["throughput"][step] = completed
            log["min_cross"][step] = np.nan
            log["min_gap"][step] = np.nan
            continue

        P   = torch.tensor(p, dtype=torch.float32)
        Vv  = torch.tensor(v, dtype=torch.float32)
        Ln  = torch.tensor(lane, dtype=torch.long)
        Ax  = torch.tensor([_LANES[l][2] for l in lane], dtype=torch.long)
        eye = torch.eye(N, dtype=torch.bool)

        # ── 2. in-lane leader (same lane, smallest p strictly ahead) ─────────────
        same_lane = Ln.unsqueeze(1) == Ln.unsqueeze(0)
        ahead     = P.unsqueeze(0) > P.unsqueeze(1)            # j ahead of i
        cand      = same_lane & ahead & ~eye
        gap_pp    = torch.where(cand, P.unsqueeze(0) - P.unsqueeze(1),
                                torch.full((N, N), float("inf")))
        lead_gap, lead_idx = gap_pp.min(dim=1)
        has_lead  = torch.isfinite(lead_gap)
        p_lead    = torch.where(has_lead, P[lead_idx], P + 300.0)
        v_lead    = torch.where(has_lead, Vv[lead_idx], Vv)

        # ── 3. cross rivals — PER-PAIR conflict points + conflict-zone validity ──
        SC      = torch.tensor(_SCROSS, dtype=torch.float32)
        s_ego   = SC[Ln][:, Ln]                                # [i,j] cross coord on ego i's path
        s_riv   = s_ego.t()                                    # [i,j] cross coord on rival j's path
        ego_d   = s_ego - P.unsqueeze(1)                       # ego i's distance to that crossing
        rival_d = s_riv - P.unsqueeze(0)                       # rival j's distance to that crossing
        diff_axis  = Ax.unsqueeze(1) != Ax.unsqueeze(0)
        ego_appr   = P.unsqueeze(1) < s_ego                    # ego still before its crossing
        rival_unclr = P.unsqueeze(0) < (s_riv + CONFLICT_CLEAR)  # rival not yet physically cleared
        rival_valid = diff_axis & ego_appr & rival_unclr & ~eye
        rival_v   = Vv.clamp(min=eps).unsqueeze(0).expand(N, N)
        d_conf    = ego_d                                       # per-pair ego distance [N, N]

        # ── 4. ONE batched controller call for the whole population ──────────────
        a = utils.controller_acceleration(
            P, p_lead, Vv, v_lead,
            d_conf=d_conf, rival_d=rival_d, rival_v=rival_v, rival_valid=rival_valid)

        # ── 5. integrate ─────────────────────────────────────────────────────────
        a_np = a.numpy()
        for i in range(N):
            v[i] = float(min(max(v[i] + a_np[i] * DT, 0.0), V_PHYS))
            p[i] += v[i] * DT

        # ── 6. ground-truth collision check in 2-D ───────────────────────────────
        pos = np.stack([xy(lane[i], p[i]) for i in range(N)])      # [N,2]
        min_cross = np.inf
        min_gap   = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                if lane[i] == lane[j]:
                    g = abs(p[i] - p[j]) - utils.L_VEH
                    min_gap = min(min_gap, g)
                    if g < 0:
                        crashes.append((t, vids[i], vids[j], g))
                else:
                    dist = float(np.hypot(*(pos[i] - pos[j])))
                    if _LANES[lane[i]][2] != _LANES[lane[j]][2]:   # crossing pair
                        min_cross = min(min_cross, dist)
                        if dist < CRASH_R:
                            crashes.append((t, vids[i], vids[j], dist))

        # ── 7. record + remove exited vehicles ───────────────────────────────────
        for i in range(N):
            traj.setdefault(vids[i], []).append((t, pos[i, 0], pos[i, 1], lane[i]))
            vstate.setdefault(vids[i], []).append(
                (t, p[i], v[i], float(a_np[i]), lane[i]))
        log["n_active"][step]   = N
        log["min_cross"][step]  = min_cross
        log["min_gap"][step]    = min_gap
        log["min_acc"][step]    = float(a_np.min())
        log["mean_v"][step]     = float(np.mean(v))

        keep = [i for i in range(N) if p[i] <= EXIT]
        completed += N - len(keep)
        lane = [lane[i] for i in keep]; p = [p[i] for i in keep]
        v = [v[i] for i in keep];       vids = [vids[i] for i in keep]
        log["throughput"][step] = completed

    return np.arange(n_steps) * DT, log, crashes, traj, completed, vstate


# ─────────────────────────────────────────────────────────────────────────────

def plot(t, log, traj, crashes, completed, out="intersection_sim_demo.png"):
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3)
    status = "NO CRASHES" if not crashes else f"{len(crashes)} CRASH EVENTS"
    fig.suptitle(f"120 s PyTorch intersection — 4-way through (NS/SN/EW/WE), no signal "
                 f"— {completed} vehicles cleared — {status}",
                 fontsize=13, fontweight="bold",
                 color="#2ca02c" if not crashes else "#d62728")

    # bird's-eye snapshot at the busiest instant (through-only traffic collapses to
    # two lines over the full run, so a zoomed snapshot is far more informative)
    ax = fig.add_subplot(gs[:, 0])
    ZOOM = 70.0
    ax.axhline(0, color="#f0f0f0", lw=22, zorder=0)
    ax.axvline(0, color="#f0f0f0", lw=22, zorder=0)
    t_peak = float(t[int(np.nanargmax(log["n_active"]))])
    snap = []
    for pts in traj.values():
        for (tt, x, y, ln) in pts:
            if abs(tt - t_peak) < DT / 2:
                snap.append((x, y, ln))
    for ln, col in _LANE_COLOR.items():
        pts = [(x, y) for (x, y, l) in snap if l == ln]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=80, color=col, edgecolor="k", lw=0.5,
                       label=f"{_LANE_NAME[ln]} ({len(pts)})", zorder=3)
    ax.plot(0, 0, "x", ms=14, mew=3, color="k", zorder=4)
    ax.set_xlim(-ZOOM, ZOOM); ax.set_ylim(-ZOOM, ZOOM)
    ax.set_aspect("equal", "box")
    ax.set_title(f"snapshot at busiest instant t={t_peak:.0f}s "
                 f"({len(snap)} vehicles)  ✕ = conflict zone")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", fontsize=8, title="approach (count)")

    # min crossing separation
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, log["min_cross"], color="#1f77b4", lw=1.2)
    ax.axhline(CRASH_R, color="#d62728", lw=1.2, ls="--", label=f"crash radius {CRASH_R} m")
    ax.axhline(utils.L_VEH, color="gray", lw=0.8, ls=":", label="car length")
    ax.set_ylabel("min crossing dist (m)"); ax.set_title("closest perpendicular approach")
    ax.set_ylim(0, 30); ax.legend(loc="upper right", fontsize=8)

    # min same-lane gap
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, log["min_gap"], color="#9467bd", lw=1.2)
    ax.axhline(0, color="#d62728", lw=1.2, ls="--", label="bumper contact")
    ax.set_ylabel("min same-lane gap (m)"); ax.set_title("closest in-lane (rear-end) gap")
    ax.set_ylim(-1, 30); ax.legend(loc="upper right", fontsize=8)

    # active count + throughput
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, log["n_active"], color="#ff7f0e", lw=1.2, label="active vehicles")
    ax.set_ylabel("active", color="#ff7f0e"); ax.set_xlabel("time (s)")
    a2 = ax.twinx()
    a2.plot(t, log["throughput"], color="#2ca02c", lw=1.5, label="cumulative cleared")
    a2.set_ylabel("cleared", color="#2ca02c")
    ax.set_title("population & throughput")

    # min accel (braking authority used) + mean speed
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, log["min_acc"], color="#d62728", lw=1.0, label="hardest brake")
    ax.axhline(-utils.B_MAX, color="gray", lw=0.8, ls="--", label="−b_max")
    ax.set_ylabel("min accel (m/s²)", color="#d62728"); ax.set_xlabel("time (s)")
    a2 = ax.twinx()
    a2.plot(t, log["mean_v"], color="#1f77b4", lw=1.0)
    a2.set_ylabel("mean speed (m/s)", color="#1f77b4")
    a2.axhline(utils.V0, color="#1f77b4", lw=0.6, ls=":")
    ax.set_title("braking authority used & mean speed")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    t, log, crashes, traj, completed, vstate = simulate()
    print(f"\n=== 120 s, 4-way through intersection, flow={FLOW_VPH:.0f} veh/h/approach ===")
    print(f"  vehicles cleared        : {completed}")
    print(f"  peak active             : {int(np.nanmax(log['n_active']))}")
    print(f"  min crossing separation : {np.nanmin(log['min_cross']):.2f} m  "
          f"(crash < {CRASH_R} m)")
    print(f"  min same-lane gap       : {np.nanmin(log['min_gap']):.2f} m  (crash < 0 m)")
    print(f"  hardest brake used      : {np.nanmin(log['min_acc']):.2f} m/s^2  "
          f"(limit -{utils.B_MAX})")
    if crashes:
        print(f"  *** {len(crashes)} CRASH EVENTS ***")
        for c in crashes[:10]:
            print(f"      t={c[0]:.1f}s  veh {c[1]} <-> {c[2]}  sep={c[3]:.2f} m")
    else:
        print("  *** NO CRASHES -- collision-free over the full run ***")
    plot(t, log, traj, crashes, completed)
