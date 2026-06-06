"""
test_safety_sim.py — 15-second simulation: min TTC with vs without safety correction

Two back-to-back runs (same network, same seed):
  Baseline : HybridModel control only  (no h_leq)
  Safety   : HybridModel + RKHS h_leq  (3 descent iterations per real step, η=50)

At each timestep records the minimum TTC across all inbound conflict pairs.
Plots both traces on the same axes with the TTC=3s safety threshold marked.
"""

import os
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sumo as _sumo_pkg
import torch
import traci

from conflict import build_snapshot, STREAM_NAMES
from model import HybridModel
from safety import run_safety_descent, compute_violation
from simulator import build, load_config
from ttc import build_ttc_surfaces, calibrate_cp_offsets, HORIZON

SUMO_BIN   = Path(_sumo_pkg.__file__).parent / "bin"
MODELS_DIR = Path("models")
SEQ_LEN    = 5

SIM_SECONDS    = 15
DESCENT_STEPS  = 3
ETA            = 50.0
SIGMA          = 1.0
THRESHOLD      = 3.0


def _bin(name: str) -> str:
    return str(SUMO_BIN / (name + (".exe" if os.name == "nt" else "")))


def _load_model() -> HybridModel:
    model = HybridModel(seq_len=SEQ_LEN)
    for ckpt in [MODELS_DIR / "hybrid_model_best.pt",
                 MODELS_DIR / "hybrid_model_latest.pt"]:
        if ckpt.exists():
            model.load_state_dict(
                torch.load(ckpt, map_location="cpu", weights_only=True))
            print(f"  Loaded model: {ckpt}")
            break
    model.eval()
    return model


def _query_states(vids: list[str]) -> tuple[dict, dict, dict]:
    v_d, gap_d, vlead_d = {}, {}, {}
    for vid in vids:
        v = traci.vehicle.getSpeed(vid)
        leader = traci.vehicle.getLeader(vid)
        if leader:
            lid, g = leader
            gap_d[vid]   = g
            vlead_d[vid] = traci.vehicle.getSpeed(lid)
        else:
            gap_d[vid]   = 100.0
            vlead_d[vid] = v
        v_d[vid] = v
    return v_d, gap_d, vlead_d


def _apply_speeds(vids, model, v_d, gap_d, vlead_d,
                  x_seq_dict, delta_a, dt) -> None:
    if not vids:
        return
    v_t     = torch.tensor([v_d[i]     for i in vids], dtype=torch.float32)
    gap_t   = torch.tensor([gap_d[i]   for i in vids], dtype=torch.float32)
    vlead_t = torch.tensor([vlead_d[i] for i in vids], dtype=torch.float32)
    x_seq_t = torch.stack([x_seq_dict[i] for i in vids])   # [N, SEQ_LEN, 3]
    with torch.no_grad():
        accel = model(v_t, gap_t, vlead_t, x_seq_t)
    for idx, vid in enumerate(vids):
        corr = delta_a.get(vid, torch.zeros(HORIZON))[0].item()
        if float(v_t[idx]) < 0.1 and corr < 0:
            corr = 0.0   # don't let negative warm-start hold a stopped vehicle at 0
        v_next = max(0.0, float(v_t[idx]) + (float(accel[idx]) + corr) * dt)
        traci.vehicle.setSpeed(vid, v_next)


def _min_ttc_from_surfaces(surfaces: dict) -> float:
    best = float("inf")
    for surf in surfaces.values():
        for j, ttc_j in enumerate(surf.ttc):
            if surf.rival_ids[j]:
                best = min(best, ttc_j.min().item())
    return best if best < float("inf") else THRESHOLD


def run_simulation(apply_safety: bool, max_steps: int) -> tuple[list, list, int]:
    """Returns (times, min_ttc_per_step, n_collisions)."""
    cfg     = load_config()
    sumocfg = build()
    dt      = cfg["step_length"]
    model   = _load_model()

    os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
    traci.start([
        _bin("sumo"), "-c", str(sumocfg),
        "--step-length",      str(dt),
        "--collision.action", "warn",
        "--collision.check-junctions",
        "--no-step-log",
    ])
    calibrate_cp_offsets()

    known:        set[str] = set()
    obs_buffers:  dict     = {}
    times:        list[float] = []
    min_ttcs:     list[float] = []
    n_collisions: int = 0
    warm_delta_a: dict = {}
    stuck_steps:  dict = {}
    STUCK_LIMIT   = int(5.0 / dt)

    try:
        for step in range(max_steps):
            traci.simulationStep()
            sim_t    = step * dt
            all_vids = list(traci.vehicle.getIDList())

            v_d, gap_d, vlead_d = _query_states(all_vids)

            # initialise buffer for new vehicles (cold-start: repeat first obs)
            for vid in all_vids:
                if vid not in known:
                    traci.vehicle.setSpeedMode(vid, 0)
                    known.add(vid)
                    obs0 = torch.tensor(
                        [v_d[vid], gap_d[vid], vlead_d[vid]],
                        dtype=torch.float32)
                    obs_buffers[vid] = deque([obs0] * SEQ_LEN, maxlen=SEQ_LEN)

            # append current observation to every active vehicle's buffer
            for vid in all_vids:
                obs = torch.tensor(
                    [v_d[vid], gap_d[vid], vlead_d[vid]], dtype=torch.float32)
                obs_buffers[vid].append(obs)

            # evict vehicles stuck at near-zero speed for too long
            for vid in all_vids:
                stuck_steps[vid] = stuck_steps.get(vid, 0) + 1 if v_d[vid] < 0.5 else 0
            for vid in [v for v, n in stuck_steps.items() if n > STUCK_LIMIT]:
                try:
                    traci.vehicle.remove(vid)
                except Exception:
                    pass
                for d in (obs_buffers, stuck_steps, warm_delta_a):
                    d.pop(vid, None)
                known.discard(vid)

            # purge departed/teleported vehicles from both structures
            # refresh all_vids from SUMO after evictions so downstream code is consistent
            all_vids = list(traci.vehicle.getIDList())
            for vid in list(obs_buffers):
                if vid not in set(all_vids):
                    del obs_buffers[vid]
                    known.discard(vid)

            x_seq_dict = {
                vid: torch.stack(list(obs_buffers[vid]))
                for vid in all_vids if vid in obs_buffers
            }

            snap = build_snapshot(all_vids)

            if apply_safety and snap.vehicle_stream:
                init_da = {
                    vid: torch.cat([da[1:], torch.zeros(1)])
                    for vid, da in warm_delta_a.items()
                }
                delta_a, min_ttc = run_safety_descent(
                    snap, model, v_d, gap_d, vlead_d,
                    x_seq_dict=x_seq_dict,
                    dt=dt, n_steps=DESCENT_STEPS,
                    eta=ETA, sigma=SIGMA,
                    threshold=THRESHOLD, beta_0=1.0,
                    verbose=False,
                    init_delta_a=init_da,
                )
                warm_delta_a = delta_a
            else:
                delta_a      = {}
                warm_delta_a = {}
                if snap.vehicle_stream:
                    surfaces, _ = build_ttc_surfaces(
                        snap, model, v_d, gap_d, vlead_d,
                        x_seq_dict=x_seq_dict, dt=dt)
                    min_ttc = _min_ttc_from_surfaces(surfaces)
                else:
                    min_ttc = THRESHOLD

            _apply_speeds(all_vids, model, v_d, gap_d, vlead_d,
                          x_seq_dict, delta_a, dt)

            n_collisions += traci.simulation.getCollidingVehiclesNumber()
            times.append(sim_t)
            min_ttcs.append(min(min_ttc, THRESHOLD))

    finally:
        traci.close()

    return times, min_ttcs, n_collisions


def main() -> None:
    max_steps = int(SIM_SECONDS / 0.1)
    label = {
        False: "Baseline (HybridModel only)",
        True:  f"Safety (HybridModel + h_≤, {DESCENT_STEPS} iters, η={ETA})",
    }

    results = {}
    for apply_safety in [False, True]:
        tag = "safety" if apply_safety else "baseline"
        print(f"\nRunning {label[apply_safety]} ...")
        times, min_ttcs, n_coll = run_simulation(apply_safety, max_steps)
        results[tag] = (times, min_ttcs, n_coll)
        print(f"  collisions: {n_coll}   "
              f"mean min-TTC: {sum(min_ttcs)/len(min_ttcs):.3f}s   "
              f"time below 3s: {sum(1 for v in min_ttcs if v < THRESHOLD) * 0.1:.1f}s")

    fig, ax = plt.subplots(figsize=(11, 5))
    colours = {"baseline": "tomato", "safety": "steelblue"}

    for tag, (times, min_ttcs, n_coll) in results.items():
        ax.plot(times, min_ttcs,
                label=f"{label[tag == 'safety']}  (collisions={n_coll})",
                color=colours[tag], lw=1.5, alpha=0.85)

    ax.axhline(THRESHOLD, color="black", lw=1.0, ls="--",
               label=f"Safety threshold  TTC = {THRESHOLD}s")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Min TTC across all conflict pairs (s)")
    ax.set_title(f"Intersection safety over {SIM_SECONDS}s  —  "
                 f"4-approach, all movements, speedMode=0")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, THRESHOLD + 0.2)
    ax.grid(True, alpha=0.3)

    Path("logs").mkdir(exist_ok=True)
    out = "logs/safety_sim.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    main()
