"""
test_baseline.py

Baseline evaluation: HybridModel (IDM/GRU) + social force only.
No SafetyTransformer correction (δa = 0 everywhere).

Runs N_EPISODES episodes with N_PARALLEL SUMO instances and reports
V_true, wTTC, and teleport counts so you can compare against the
transformer-corrected runs.
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import torch
import sumo as _sumo_pkg
import traci

import traci_cache
from conflict import build_snapshot, CONFLICT_MAP, STREAM_NAMES, clear_route_cache
from model import HybridModel
from simulator import build, load_config
from ttc import calibrate_cp_offsets, _query_d_to_junction, _cp_pair_offsets, _EPS_V

# ── reuse constants from training script ──────────────────────────────────────
from train_safety_transformer_online import (
    DEVICE, SUMO_BIN, MODELS_DIR,
    SEQ_LEN, SIM_SECONDS, DT, PROJ_DT, HORIZON,
    N_PARALLEL, BASE_PORT, WARMUP_SECS,
    THRESHOLD, BETA_0, L_OCC, CONTROL_DIST, STUCK_LIMIT,
    _load_hybrid, _bin,
    compute_social_force, compute_current_violation,
    compute_intersection_min_ttc,
)

N_EPISODES = 5   # how many episodes to average over

# ── apply speeds with zero transformer correction ─────────────────────────────

def _apply_speeds_baseline(vids, hybrid, v_d, gap_d, vlead_d, x_seq_dict, dt,
                            is_turning_d=None, social_a_np=None):
    if not vids:
        return
    d_junc_d = _query_d_to_junction(vids)
    v_t   = torch.tensor([v_d[i]   for i in vids], dtype=torch.float32, device=DEVICE)
    g_t   = torch.tensor([gap_d[i] for i in vids], dtype=torch.float32, device=DEVICE)
    vl_t  = torch.tensor([vlead_d[i] for i in vids], dtype=torch.float32, device=DEVICE)
    xs_t  = torch.stack([
        (x_seq_dict[i] if i in x_seq_dict else torch.zeros(SEQ_LEN, 3)).to(DEVICE)
        for i in vids
    ])
    it_t  = torch.tensor(
        [is_turning_d.get(v, 0.0) for v in vids] if is_turning_d else [0.0] * len(vids),
        dtype=torch.float32, device=DEVICE,
    )
    sa_t  = torch.tensor(
        [social_a_np.get(v, 0.0) for v in vids] if social_a_np else [0.0] * len(vids),
        dtype=torch.float32, device=DEVICE,
    )
    with torch.no_grad():
        accel = hybrid(v_t, g_t, vl_t, xs_t, it_t, social_a=sa_t)
    for idx, vid in enumerate(vids):
        if d_junc_d.get(vid, 999.0) > CONTROL_DIST:
            traci.vehicle.setSpeedMode(vid, 31)
            traci.vehicle.setSpeed(vid, -1)
            continue
        v_next = max(0.0, float(v_t[idx]) + float(accel[idx]) * dt)
        traci.vehicle.setSpeedMode(vid, 0)
        traci.vehicle.setSpeed(vid, v_next)


def run_baseline_episode(hybrid: HybridModel, ep: int) -> dict:
    cfg     = load_config()
    dt      = cfg["step_length"]
    sumocfg = build()

    os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)

    sims_state = []
    for i in range(N_PARALLEL):
        label = f"sim{i}"
        cmd = [
            _bin("sumo"), "-c", str(sumocfg),
            "--step-length",        str(dt),
            "--collision.action",   "teleport",
            "--collision.check-junctions",
            "--no-step-log",
            "--seed",               str(ep * N_PARALLEL + i),
        ]
        traci.start(cmd, port=BASE_PORT + i, label=label)
        if i == 0:
            calibrate_cp_offsets()
        warmup_i = WARMUP_SECS[i] if i < len(WARMUP_SECS) else WARMUP_SECS[-1]
        for _ in range(int(warmup_i / dt)):
            traci.simulationStep()
        sims_state.append({
            "label":       label,
            "warmup_sec":  warmup_i,
            "obs_buffers": {},
            "stuck_steps": {},
            "known":       set(),
            "teleports":   0,
        })

    control_steps = int(SIM_SECONDS / dt)
    v_true_sum = 0.0
    ttc_sum    = 0.0
    ttc_count  = 0
    total_tele = 0
    step_count = 0

    try:
        for ctrl_step in range(control_steps):
            for sim in sims_state:
                traci.switch(sim["label"])
                traci.simulationStep()
                all_vids = list(traci.vehicle.getIDList())
                traci_cache.update(all_vids)

                n_tele = traci.simulation.getStartingTeleportNumber()
                if n_tele > 0:
                    sim["teleports"] += n_tele
                    total_tele += n_tele

                v_d, gap_d, vlead_d = {}, {}, {}
                for vid in all_vids:
                    v = traci_cache.get_speed(vid)
                    try:
                        leader = traci.vehicle.getLeader(vid)
                    except traci.exceptions.TraCIException:
                        leader = None
                    gap_d[vid]   = leader[1] if leader else 100.0
                    vlead_d[vid] = traci_cache.get_speed(leader[0]) if leader else v
                    v_d[vid]     = v

                for vid in all_vids:
                    if vid not in sim["known"]:
                        traci.vehicle.setSpeedMode(vid, 0)
                        sim["known"].add(vid)
                        obs0 = torch.tensor([v_d[vid], gap_d[vid], vlead_d[vid]], dtype=torch.float32)
                        sim["obs_buffers"][vid] = deque([obs0] * SEQ_LEN, maxlen=SEQ_LEN)
                    sim["obs_buffers"][vid].append(
                        torch.tensor([v_d[vid], gap_d[vid], vlead_d[vid]], dtype=torch.float32))

                for vid in all_vids:
                    sim["stuck_steps"][vid] = sim["stuck_steps"].get(vid, 0) + 1 \
                        if v_d[vid] < 0.5 else 0
                for vid in [v for v, n in sim["stuck_steps"].items() if n > STUCK_LIMIT]:
                    try:
                        traci.vehicle.remove(vid)
                    except Exception:
                        pass
                    sim["obs_buffers"].pop(vid, None)
                    sim["stuck_steps"].pop(vid, None)
                    sim["known"].discard(vid)

                all_vids = list(traci.vehicle.getIDList())
                for vid in list(sim["obs_buffers"]):
                    if vid not in set(all_vids):
                        del sim["obs_buffers"][vid]
                        sim["known"].discard(vid)

                x_seq_dict = {
                    vid: torch.stack(list(sim["obs_buffers"][vid]))
                    for vid in all_vids if vid in sim["obs_buffers"]
                }

                snap = build_snapshot(all_vids)
                candidates = [vid for vid, mvmt in snap.vehicle_stream.items()
                              if mvmt is not None and CONFLICT_MAP.get(mvmt)]

                is_turning_all = {
                    vid: (1.0 if STREAM_NAMES.get(snap.vehicle_stream.get(vid), "").endswith(("_R", "_L"))
                          else 0.0)
                    for vid in all_vids
                }

                if not candidates:
                    _apply_speeds_baseline(all_vids, hybrid, v_d, gap_d, vlead_d,
                                           x_seq_dict, dt, is_turning_d=is_turning_all)
                    continue

                with torch.no_grad():
                    d_raw_all  = _query_d_to_junction(candidates)
                    tracked = []
                    ACTIVE_ETA = HORIZON * PROJ_DT
                    for _vid in candidates:
                        _stream = snap.vehicle_stream[_vid]
                        _d_i    = d_raw_all.get(_vid, float("inf"))
                        _v_i    = max(v_d.get(_vid, _EPS_V), _EPS_V)
                        for _cs in CONFLICT_MAP.get(_stream, frozenset()):
                            _off_i, _off_j = _cp_pair_offsets(_stream, _cs)
                            _eta_i = max(_d_i + _off_i, 0.0) / _v_i
                            if _eta_i > ACTIVE_ETA:
                                continue
                            for _rvid in snap.stream_vehicles.get(_cs, []):
                                _d_j  = d_raw_all.get(_rvid, float("inf"))
                                _v_j  = max(v_d.get(_rvid, _EPS_V), _EPS_V)
                                _eta_j = max(_d_j + _off_j, 0.0) / _v_j
                                if _eta_j < ACTIVE_ETA:
                                    tracked.append(_vid)
                                    tracked.append(_rvid)
                    tracked = list(dict.fromkeys(v for v in candidates if v in set(tracked)))

                    if not tracked:
                        _apply_speeds_baseline(all_vids, hybrid, v_d, gap_d, vlead_d,
                                               x_seq_dict, dt, is_turning_d=is_turning_all)
                        continue

                    idx_of     = {vid: k for k, vid in enumerate(tracked)}
                    v0_t       = torch.tensor([v_d[i]   for i in tracked], dtype=torch.float32, device=DEVICE)
                    d_junc_t   = torch.tensor([d_raw_all[i] for i in tracked], dtype=torch.float32, device=DEVICE)

                    social_force = compute_social_force(snap, d_junc_t, v0_t, idx_of, tracked)
                    sa_np = {vid: social_force[i].item() for i, vid in enumerate(tracked)}

                    V_true   = compute_current_violation(snap, d_junc_t, v0_t, idx_of, tracked)
                    min_ttc  = compute_intersection_min_ttc(snap, d_junc_t, v0_t, idx_of, tracked)

                v_true_sum += V_true / N_PARALLEL
                if min_ttc < float("inf"):
                    ttc_sum   += min_ttc / N_PARALLEL
                    ttc_count += 1

                _apply_speeds_baseline(all_vids, hybrid, v_d, gap_d, vlead_d,
                                       x_seq_dict, dt,
                                       is_turning_d=is_turning_all, social_a_np=sa_np)

            step_count += 1
            sim_t = (ctrl_step + 1) * dt + sims_state[0]["warmup_sec"]
            if ctrl_step % 25 == 0:
                ttc_str = f"{ttc_sum/max(ttc_count,1):.3f}s" if ttc_count else "  n/a"
                print(f"    t={sim_t:5.1f}s | V_true={v_true_sum/max(step_count,1):7.3f} | "
                      f"wTTC_avg={ttc_str} | teleports={total_tele}", flush=True)

    finally:
        for sim in sims_state:
            try:
                traci.switch(sim["label"])
                traci.close()
            except Exception:
                pass
        clear_route_cache()
        traci_cache.clear()

    return {
        "mean_V_true":  v_true_sum / max(step_count, 1),
        "mean_wTTC":    ttc_sum    / max(ttc_count,  1),
        "total_tele":   total_tele,
        "steps":        step_count,
    }


def main():
    print("=" * 60)
    print("  BASELINE — HybridModel + social force  (no transformer)")
    print("=" * 60)
    print(f"  Device      : {DEVICE}")
    print(f"  Episodes    : {N_EPISODES}")
    print(f"  Parallel    : {N_PARALLEL}")
    print("-" * 60)

    hybrid = _load_hybrid()
    hybrid.eval()

    all_V, all_ttc, all_tele = [], [], []

    for ep in range(N_EPISODES):
        print(f"\n── Episode {ep + 1}/{N_EPISODES} ──────────────────────────────")
        result = run_baseline_episode(hybrid, ep)
        all_V.append(result["mean_V_true"])
        all_ttc.append(result["mean_wTTC"])
        all_tele.append(result["total_tele"])
        print(f"  → mean V_true={result['mean_V_true']:.4f} | "
              f"mean wTTC={result['mean_wTTC']:.3f}s | "
              f"teleports={result['total_tele']}")

    print("\n" + "=" * 60)
    print("  BASELINE SUMMARY")
    print("=" * 60)
    print(f"  mean V_true  : {sum(all_V)/len(all_V):.4f}  (lower = safer)")
    print(f"  mean wTTC    : {sum(all_ttc)/len(all_ttc):.3f}s  (higher = safer)")
    print(f"  total tele   : {sum(all_tele)}  over {N_EPISODES} episodes")
    print("=" * 60)


if __name__ == "__main__":
    main()
