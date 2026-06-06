"""
demo_gui.py — 20-second SUMO-GUI demo at real speed

  Phase 1 (0–2s)  : SUMO warmup — vehicles inserted, SUMO controls them
  Phase 2 (2–20s) : HybridModel (GRU f̂_θ + h_= KernelCorrection) takes over,
                    + RKHS safety descent with warm-start on top

Vehicle colour coding (visible in the GUI):
  White  : no significant correction  (|δa[0]| < 0.05 m/s²)
  Red    : safety is braking          (δa[0] < -0.05 m/s²)
  Green  : safety is accelerating     (δa[0] >  0.05 m/s²)

The terminal prints a live table each step showing active streams,
min projected TTC, and how many vehicles are being corrected.
"""

import os
from collections import deque
from pathlib import Path

import sumo as _sumo_pkg
import torch
import traci

from conflict import build_snapshot, STREAM_NAMES
from model import HybridModel
from safety import run_safety_descent
from simulator import build, load_config
from ttc import build_ttc_surfaces, calibrate_cp_offsets, HORIZON

SUMO_BIN   = Path(_sumo_pkg.__file__).parent / "bin"
MODELS_DIR = Path("models")
SEQ_LEN    = 5

SIM_SECONDS   = 20
WARMUP_SEC    = 2.0       # SUMO-only warmup before PyTorch takes over
DESCENT_STEPS = 3
ETA           = 50.0
SIGMA         = 1.0
THRESHOLD     = 3.0
DELAY_MS      = "33"      # 33 ms per step ≈ 3× real time (1/3 wall-clock)


def _bin(name: str) -> str:
    return str(SUMO_BIN / (name + (".exe" if os.name == "nt" else "")))


def _load_model() -> HybridModel:
    model = HybridModel(seq_len=SEQ_LEN)
    for ckpt in [MODELS_DIR / "hybrid_model_best.pt",
                 MODELS_DIR / "hybrid_model_latest.pt"]:
        if ckpt.exists():
            model.load_state_dict(
                torch.load(ckpt, map_location="cpu", weights_only=True))
            print(f"  Loaded: {ckpt}")
            break
    model.eval()
    return model


def _query_states(vids):
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


def _color_vehicles(delta_a: dict, all_vids: list) -> None:
    """Paint each vehicle by the sign of its current safety correction."""
    for vid in all_vids:
        corr = delta_a.get(vid, torch.zeros(HORIZON))[0].item()
        if corr < -0.05:
            color = (220, 50, 50, 255)    # red  — braking
        elif corr > 0.05:
            color = (50, 200, 50, 255)    # green — accelerating
        else:
            color = (255, 255, 255, 255)  # white — no correction
        traci.vehicle.setColor(vid, color)


def _apply_speeds(vids, model, v_d, gap_d, vlead_d,
                  x_seq_dict, delta_a, dt) -> None:
    if not vids:
        return
    v_t     = torch.tensor([v_d[i]     for i in vids], dtype=torch.float32)
    gap_t   = torch.tensor([gap_d[i]   for i in vids], dtype=torch.float32)
    vlead_t = torch.tensor([vlead_d[i] for i in vids], dtype=torch.float32)
    x_seq_t = torch.stack([x_seq_dict[i] for i in vids])
    with torch.no_grad():
        accel = model(v_t, gap_t, vlead_t, x_seq_t)
    for idx, vid in enumerate(vids):
        corr = delta_a.get(vid, torch.zeros(HORIZON))[0].item()
        # don't let a negative warm-started correction hold a stopped vehicle at 0
        if float(v_t[idx]) < 0.1 and corr < 0:
            corr = 0.0
        v_next = max(0.0, float(v_t[idx]) + (float(accel[idx]) + corr) * dt)
        traci.vehicle.setSpeed(vid, v_next)


def main():
    cfg     = load_config()
    sumocfg = build()
    dt      = cfg["step_length"]
    model   = _load_model()

    os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
    traci.start([
        _bin("sumo-gui"), "-c", str(sumocfg),
        "--step-length",            str(dt),
        "--collision.action",       "teleport",
        "--collision.check-junctions",
        "--no-step-log",
        "--start",                  # auto-start without clicking Play
        "--delay",      DELAY_MS,   # ms between steps ≈ real time
        "--quit-on-end",
    ])

    # centre the camera on the intersection (x=0, y=0)
    try:
        traci.gui.setOffset("View #0", 0.0, 0.0)
        traci.gui.setZoom("View #0", 350.0)
    except Exception:
        pass   # GUI not ready yet — harmless

    calibrate_cp_offsets()

    warmup_steps  = int(WARMUP_SEC / dt)
    control_steps = int(SIM_SECONDS / dt)

    known:        set[str] = set()
    obs_buffers:  dict     = {}
    warm_delta_a: dict     = {}
    stuck_steps:  dict     = {}          # vid -> consecutive steps at v < 0.5 m/s
    STUCK_LIMIT   = int(5.0 / dt)       # remove after 5 s of being stationary

    print(f"\n{'='*72}")
    print(f"  SUMO-GUI demo  ·  {SIM_SECONDS}s  ·  HybridModel (GRU f̂ + h_=) + RKHS safety")
    print(f"  Phase 1  0 – {WARMUP_SEC:.0f}s   SUMO warmup")
    print(f"  Phase 2  {WARMUP_SEC:.0f} – {SIM_SECONDS}s  PyTorch in control  "
          f"(descent={DESCENT_STEPS} iters, η={ETA}, warm-start)")
    print(f"  Colour: WHITE=no correction  RED=braking  GREEN=accelerating")
    print(f"{'='*72}")
    print(f"  {'t(s)':>5}  {'phase':<8}  {'veh':>4}  {'minTTC':>7}  "
          f"{'corrected':>9}  {'warm':>5}  active streams")
    print(f"  {'-'*70}")

    # ── Phase 1: SUMO warmup ──────────────────────────────────────────────────
    for step in range(warmup_steps):
        traci.simulationStep()
        sim_t    = step * dt
        all_vids = list(traci.vehicle.getIDList())
        print(f"  {sim_t:5.1f}  {'WARMUP':<8}  {len(all_vids):4d}  "
              f"{'—':>7}  {'—':>9}  {'—':>5}  —", flush=True)

    # ── Phase 2: HybridModel + safety descent ────────────────────────────────
    for step in range(control_steps):
        traci.simulationStep()
        sim_t    = WARMUP_SEC + step * dt
        all_vids = list(traci.vehicle.getIDList())

        v_d, gap_d, vlead_d = _query_states(all_vids)

        for vid in all_vids:
            if vid not in known:
                traci.vehicle.setSpeedMode(vid, 0)
                known.add(vid)
                obs0 = torch.tensor(
                    [v_d[vid], gap_d[vid], vlead_d[vid]], dtype=torch.float32)
                obs_buffers[vid] = deque([obs0] * SEQ_LEN, maxlen=SEQ_LEN)

        for vid in all_vids:
            obs = torch.tensor(
                [v_d[vid], gap_d[vid], vlead_d[vid]], dtype=torch.float32)
            obs_buffers[vid].append(obs)

        # track and evict vehicles stuck at near-zero speed
        for vid in all_vids:
            if v_d[vid] < 0.5:
                stuck_steps[vid] = stuck_steps.get(vid, 0) + 1
            else:
                stuck_steps.pop(vid, None)

        for vid in [v for v, n in stuck_steps.items() if n > STUCK_LIMIT]:
            try:
                traci.vehicle.remove(vid)
            except Exception:
                pass
            for d in (obs_buffers, stuck_steps, warm_delta_a):
                d.pop(vid, None)
            known.discard(vid)

        all_vids = list(traci.vehicle.getIDList())   # refresh after evictions
        for vid in list(obs_buffers):
            if vid not in set(all_vids):
                del obs_buffers[vid]
                known.discard(vid)

        x_seq_dict = {
            vid: torch.stack(list(obs_buffers[vid]))
            for vid in all_vids if vid in obs_buffers
        }

        snap = build_snapshot(all_vids)
        warmstarted = bool(warm_delta_a)

        if snap.vehicle_stream:
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
            min_ttc      = THRESHOLD

        _apply_speeds(all_vids, model, v_d, gap_d, vlead_d,
                      x_seq_dict, delta_a, dt)
        _color_vehicles(delta_a, all_vids)

        n_corrected = sum(
            1 for vid, da in delta_a.items() if abs(da[0].item()) > 0.05)
        active_streams = " ".join(sorted(
            STREAM_NAMES.get(s, str(s))
            for s in snap.stream_vehicles)) or "none"
        ws = "WARM" if warmstarted else "COLD"

        print(f"  {sim_t:5.1f}  {'PYTORCH':<8}  {len(all_vids):4d}  "
              f"{min_ttc:7.3f}  {n_corrected:9d}  {ws:>5}  {active_streams}",
              flush=True)

    print(f"\n  Simulation complete. Close the GUI window to exit.")


if __name__ == "__main__":
    main()
