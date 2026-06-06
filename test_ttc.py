"""
test_ttc.py — collision demonstration + TTC/distance-to-CP validation

Runs SUMO with speedMode=0 (all SUMO safety checks disabled) and drives
every vehicle with the IDM model via setSpeed().  Because IDM only sees the
vehicle directly ahead in the same lane, vehicles from perpendicular streams
have no awareness of each other and will collide at the junction.

At each step records, for EW_T (westbound through) vehicles:
  - signed distance to conflict point  (ego and each rival)
  - TTC estimate at projected step 0
  - cumulative collision count

Four plot panels per conflict pair:
  1. Signed distance to CP            (decreases → crosses 0 → negative)
  2. |d_ego − d_rival|                (parabola: converge → minimum → diverge)
  3. TTC estimate                     (should approach 0 near simultaneous arrival)
  4. Cumulative collision count       (rises as crashes happen)

Output: logs/ttc_test.png
"""

import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sumo as _sumo_pkg
import torch
import traci

from conflict import build_snapshot, STREAM_NAMES
from model import IDMModel
from safety import run_safety_descent
from simulator import build, load_config
from ttc import build_ttc_surfaces, calibrate_cp_offsets, collect_at_risk, signed_dist_to_cp

SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"


def _bin(name: str) -> str:
    return str(SUMO_BIN / (name + (".exe" if os.name == "nt" else "")))


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run(max_steps: int = 400) -> None:
    cfg     = load_config()
    sumocfg = build()
    dt      = cfg["step_length"]
    model   = IDMModel()
    model.eval()

    os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
    traci.start([
        _bin("sumo"), "-c", str(sumocfg),
        "--step-length",          str(dt),
        "--collision.action",     "warn",
        "--collision.check-junctions",
        "--no-step-log",
    ])

    # calibrate CP offsets from actual SUMO junction geometry
    calibrate_cp_offsets()
    from ttc import CP_CROSS, CP_MERGE
    print(f"[calibrate] CP_CROSS = {CP_CROSS}")
    print(f"[calibrate] CP_MERGE = {CP_MERGE}")

    known: set[str]  = set()   # vehicles already configured
    descent_done     = False   # run descent once at first at-risk step

    # key   : (ego_vid, rival_vid, conflict_stream_name)
    # value : list of (sim_time, d_ego, d_rival, ttc_now)
    records:    dict[tuple, list] = defaultdict(list)
    coll_times: list[float]       = []   # sim_time of each collision event
    total_coll: int               = 0

    try:
        for step in range(max_steps):
            traci.simulationStep()
            sim_t    = step * dt
            all_vids = list(traci.vehicle.getIDList())

            # ── disable SUMO safety for new vehicles ─────────────────────
            for vid in all_vids:
                if vid not in known:
                    traci.vehicle.setSpeedMode(vid, 0)
                    known.add(vid)

            # ── IDM control: read states, compute accel, set speed ────────
            v_d, gap_d, vlead_d = {}, {}, {}
            for vid in all_vids:
                v = traci.vehicle.getSpeed(vid)
                leader = traci.vehicle.getLeader(vid)
                if leader:
                    lid, g       = leader
                    gap          = g
                    vlead        = traci.vehicle.getSpeed(lid)
                else:
                    gap          = 100.0
                    vlead        = v
                v_d[vid]     = v
                gap_d[vid]   = gap
                vlead_d[vid] = vlead

            if all_vids:
                v_t     = torch.tensor([v_d[i]     for i in all_vids], dtype=torch.float32)
                gap_t   = torch.tensor([gap_d[i]   for i in all_vids], dtype=torch.float32)
                vlead_t = torch.tensor([vlead_d[i] for i in all_vids], dtype=torch.float32)

                with torch.no_grad():
                    accel = model(v_t, gap_t, vlead_t)

                v_next = torch.clamp(v_t + accel * dt, min=0.0)
                for i, vid in enumerate(all_vids):
                    traci.vehicle.setSpeed(vid, float(v_next[i]))

            # ── collision count ───────────────────────────────────────────
            n_coll    = traci.simulation.getCollidingVehiclesNumber()
            total_coll += n_coll
            if n_coll > 0:
                coll_times.append(sim_t)

            # ── build conflict snapshot and TTC surfaces ──────────────────
            snap = build_snapshot(all_vids)
            if not snap.vehicle_stream:
                continue

            surfaces, _proj = build_ttc_surfaces(
                snap, model, v_d, gap_d, vlead_d,
                x_seq_dict=None, dt=dt,
            )

            # run descent once at the first step where we have at-risk pairs
            at_risk = collect_at_risk(surfaces, threshold=3.0)
            if at_risk and not descent_done:
                print(f"\n[t={sim_t:.1f}s] First at-risk detection — running 5-step descent:")
                for eta in [0.05, 0.5, 5.0, 50.0]:
                    print(f"\n── η = {eta} ──")
                    run_safety_descent(
                        snap, model, v_d, gap_d, vlead_d,
                        x_seq_dict=None, dt=dt,
                        n_steps=5, eta=eta, sigma=1.0,
                        verbose=True,
                    )
                descent_done = True

            if at_risk:
                print(f"t={sim_t:.1f}s  {len(at_risk)} at-risk pair(s):")
                for p in at_risk[:4]:   # cap output to first 4
                    print(f"  {p.ego_id:20s} vs {p.rival_id:20s}"
                          f"  min_ttc={p.min_ttc:.2f}s"
                          f"  at proj_t={p.t_min * dt:.1f}s"
                          f"  [{STREAM_NAMES[p.ego_stream]} vs"
                          f"  {STREAM_NAMES[p.rival_stream]}]")

            for vid, surf in surfaces.items():
                if STREAM_NAMES.get(surf.stream) != "EW_T":
                    continue

                for j, cs in enumerate(surf.conflict_streams):
                    cs_name = STREAM_NAMES.get(cs, str(cs))
                    rivals  = surf.rival_ids[j]
                    if not rivals:
                        continue

                    d_ego = signed_dist_to_cp(vid, surf.stream, cs)

                    for k, rvid in enumerate(rivals):
                        d_rival = signed_dist_to_cp(rvid, cs, surf.stream)
                        ttc_now = surf.ttc[j][0, k].item()

                        key = (vid, rvid, cs_name)
                        records[key].append((sim_t, d_ego, d_rival, ttc_now,
                                             total_coll))

    finally:
        traci.close()

    print(f"\nTotal collision events : {total_coll}")
    print(f"Steps with collisions  : {len(coll_times)}")
    _plot(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(records: dict) -> None:
    if not records:
        print("[test_ttc] no conflict pairs observed — nothing to plot.")
        return

    keys = list(records.keys())[:6]
    n    = len(keys)

    fig, axes = plt.subplots(4, n, figsize=(5 * n, 13), squeeze=False)
    fig.suptitle(
        "Crash demo (speedMode=0, IDM control)\n"
        "EW_T ego vs conflicting streams", fontsize=12
    )

    for col, key in enumerate(keys):
        ego_vid, rvid, cs_name = key
        data   = records[key]
        times  = [d[0] for d in data]
        d_ego  = [d[1] for d in data]
        d_riv  = [d[2] for d in data]
        sep    = [abs(d[1] - d[2]) for d in data]
        ttc    = [d[3] for d in data]
        colls  = [d[4] for d in data]

        # panel 1: signed dist to CP
        ax = axes[0][col]
        ax.plot(times, d_ego, label="ego",   color="steelblue", lw=1.5)
        ax.plot(times, d_riv, label="rival", color="tomato",    lw=1.5, ls="--")
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_title(f"{ego_vid}\nvs {rvid} ({cs_name})", fontsize=8)
        ax.set_ylabel("signed dist to CP (m)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # panel 2: separation at CP (parabola)
        ax = axes[1][col]
        ax.plot(times, sep, color="purple", lw=1.5)
        ax.set_ylabel("|d_ego − d_rival| (m)")
        ax.set_title("separation at CP")
        ax.grid(True, alpha=0.3)

        # panel 3: TTC estimate
        ax = axes[2][col]
        ax.plot(times, ttc, color="darkorange", lw=1.5)
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_ylabel("TTC (s)  [step 0]")
        ax.set_title("TTC estimate")
        ax.grid(True, alpha=0.3)

        # panel 4: cumulative collisions
        ax = axes[3][col]
        ax.plot(times, colls, color="crimson", lw=1.5)
        ax.set_ylabel("cumulative collisions")
        ax.set_xlabel("sim time (s)")
        ax.set_title("collision count")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path("logs").mkdir(exist_ok=True)
    out = "logs/ttc_test.png"
    plt.savefig(out, dpi=120)
    print(f"[test_ttc] plot saved → {out}")


if __name__ == "__main__":
    run()
