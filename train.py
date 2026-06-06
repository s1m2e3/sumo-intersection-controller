"""
Training loop.

Objective: maximise the sum of velocity integrals over all vehicles
           across a 10-second simulation epoch.

    max  Σ_i ∫₀¹⁰ v_i(t) dt  ≈  Σ_i Σ_t  v_i(t) · Δt

Only the neural network f̂_θ (HybridModel.f_hat) is optimised.
The IDM physics anchors (v_max, a_max, …) are kept fixed.

Warm-start: loads models/hybrid_model_best.pt if it exists and overwrites
            it whenever a better epoch is found.
"""

import os
import torch
import traci
from collections import deque
from pathlib import Path
from torch.optim import Adam

import sumo as _sumo_pkg
from model import HybridModel
from simulator import build, load_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
os.environ["PATH"]      = str(SUMO_BIN) + os.pathsep + os.environ.get("PATH", "")

MODELS_DIR     = Path("models")
BEST_MODEL     = MODELS_DIR / "hybrid_model_best.pt"    # overwritten when improved
LATEST_MODEL   = MODELS_DIR / "hybrid_model_latest.pt"  # overwritten every optimizer step
STATE_PATH     = MODELS_DIR / "training_state.pt"        # epoch counter + best obj
SNAPSHOTS_PATH = Path("logs") / "epoch_snapshots.pt"     # for plotting

EPOCH_DURATION   = 10.0
WARMUP_DURATION  = 2.0   # seconds of SUMO-only stabilisation before PyTorch takes over
NUM_EPOCHS       = 500
LR               = 1e-4
SEQ_LEN          = 5     # GRU history length (5th-order recurrent model)
BATCH_SIZE       = 5       # accumulate gradients over N epochs before 1 optimizer step
COLLECT_EVERY    = 25      # snapshot every N epochs for the plots
CHECKPOINT_EVERY = 10      # save a named checkpoint every N optimizer steps

_INF = float("inf")


def _read_gap_vlead(vids, device):
    gap_list, vl_list = [], []
    for vid in vids:
        leader = traci.vehicle.getLeader(vid)
        gap    = leader[1] if leader else 100.0
        v_lead = traci.vehicle.getSpeed(leader[0]) if leader else traci.vehicle.getSpeed(vid)
        gap_list.append(gap)
        vl_list.append(v_lead)
    return (torch.tensor(gap_list, dtype=torch.float32, device=device),
            torch.tensor(vl_list,  dtype=torch.float32, device=device))


def run_epoch(sumocfg, dt, model, device, collect=False):
    traci.start([
        str(SUMO_BIN / "sumo.exe"), "-c", str(sumocfg),
        "--step-length", str(dt),
        "--collision.action", "warn",
    ])

    known        = set()   # vehicle IDs under PyTorch control
    v_state      = {}      # vid → scalar speed tensor (on device)
    obs_buffers  = {}      # vid → deque of SEQ_LEN obs tensors [3] (on device)
    velocity_sum = torch.tensor(0.0, device=device)
    warmup_steps  = int(WARMUP_DURATION / dt)
    control_steps = int(EPOCH_DURATION / dt)

    snap = {"time": [], "ttc_min": [],
            "fhat_max": [], "fhat_min": [],
            "h_max":    [], "h_min":    [],
            "n_veh": []} if collect else None

    # Per-epoch global stat trackers
    ep_v_min   =  _INF;  ep_v_max   = -_INF
    ep_ttc_min =  _INF;  ep_ttc_max = -_INF
    ep_a_min   =  _INF;  ep_a_max   = -_INF

    # ── Phase 1: SUMO-only warmup ─────────────────────────────────────────
    for _ in range(warmup_steps):
        traci.simulationStep()

    # Hand over post-warmup vehicles; seed obs_buffer with first observation
    # repeated SEQ_LEN times (cold-start padding).
    for vid in traci.vehicle.getIDList():
        spd = traci.vehicle.getSpeed(vid)
        traci.vehicle.setSpeedMode(vid, 0)
        traci.vehicle.setSpeed(vid, spd)
        v_state[vid] = torch.tensor(spd, dtype=torch.float32, device=device)
        leader = traci.vehicle.getLeader(vid)
        g0  = leader[1] if leader else 100.0
        vl0 = traci.vehicle.getSpeed(leader[0]) if leader else spd
        obs0 = torch.tensor([spd, g0, vl0], dtype=torch.float32, device=device)
        obs_buffers[vid] = deque([obs0] * SEQ_LEN, maxlen=SEQ_LEN)
        known.add(vid)

    # ── Phase 2: PyTorch control ──────────────────────────────────────────
    for step in range(control_steps):
        traci.simulationStep()
        current = set(traci.vehicle.getIDList())

        for vid in current - known:
            spd = traci.vehicle.getSpeed(vid)
            traci.vehicle.setSpeedMode(vid, 0)
            traci.vehicle.setSpeed(vid, spd)
            v_state[vid] = torch.tensor(spd, dtype=torch.float32, device=device)
            leader = traci.vehicle.getLeader(vid)
            g0  = leader[1] if leader else 100.0
            vl0 = traci.vehicle.getSpeed(leader[0]) if leader else spd
            obs0 = torch.tensor([spd, g0, vl0], dtype=torch.float32, device=device)
            obs_buffers[vid] = deque([obs0] * SEQ_LEN, maxlen=SEQ_LEN)

        for vid in known - current:
            v_state.pop(vid, None)
            obs_buffers.pop(vid, None)
        known = current

        vids = list(current)
        if not vids:
            if collect:
                snap["time"].append(step * dt)
                for k in ("ttc_min","fhat_max","fhat_min","h_max","h_min"):
                    snap[k].append(float("nan"))
                snap["n_veh"].append(0)
            continue

        v           = torch.stack([v_state[vid] for vid in vids])       # [N]
        gap, v_lead = _read_gap_vlead(vids, device)                      # [N], [N]

        # Append current observation to each vehicle's buffer (detached —
        # truncated BPTT: gradient flows through accel→v_next, not through
        # the stored history, keeping the graph bounded to one GRU unroll).
        for i, vid in enumerate(vids):
            obs = torch.stack([v[i].detach(), gap[i], v_lead[i]])
            obs_buffers[vid].append(obs)

        # Build sequence tensor [N, SEQ_LEN, 3]
        x_seq = torch.stack([
            torch.stack(list(obs_buffers[vid])) for vid in vids
        ])

        accel  = model(v, gap, v_lead, x_seq)
        v_next = torch.clamp(v + accel * dt, min=0.0)

        velocity_sum = velocity_sum + v_next.sum() * dt

        # ── per-step stats ────────────────────────────────────────────────
        with torch.no_grad():
            v_d = v.detach()
            a_d = accel.detach()
            z   = model.correction.ttc_star(v_d, gap, v_lead)

        ep_v_min   = min(ep_v_min,   v_d.min().item())
        ep_v_max   = max(ep_v_max,   v_d.max().item())
        ep_ttc_min = min(ep_ttc_min, z.min().item())
        ep_ttc_max = max(ep_ttc_max, z.clamp(max=200.0).max().item())
        ep_a_min   = min(ep_a_min,   a_d.min().item())
        ep_a_max   = max(ep_a_max,   a_d.max().item())
        # ─────────────────────────────────────────────────────────────────

        if collect:
            with torch.no_grad():
                fhat = model._f_hat(x_seq.detach())
                gate = torch.sigmoid(15.0 * (0.8 - v_d / model.physics.v_max.detach()))
                fhat = fhat - torch.relu(fhat) * (1.0 - gate)
                h_eq = model.correction(v_d, gap, v_lead, fhat)
            snap["time"].append(step * dt)
            snap["ttc_min"].append(z.clamp(max=100.0).min().item())
            snap["fhat_max"].append(fhat.max().item())
            snap["fhat_min"].append(fhat.min().item())
            snap["h_max"].append(h_eq.max().item())
            snap["h_min"].append(h_eq.min().item())
            snap["n_veh"].append(len(vids))

        for i, vid in enumerate(vids):
            v_state[vid] = v_next[i]
            traci.vehicle.setSpeed(vid, float(v_next[i].detach()))

    traci.close()

    ep_stats = {
        "v_min": ep_v_min,   "v_max": ep_v_max,
        "ttc_min": ep_ttc_min, "ttc_max": ep_ttc_max,
        "a_min": ep_a_min,   "a_max": ep_a_max,
    }
    return velocity_sum, snap, ep_stats


def main():
    sumocfg = build()
    cfg     = load_config()
    dt      = cfg["step_length"]

    MODELS_DIR.mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    (MODELS_DIR / "checkpoints").mkdir(exist_ok=True)

    print(f"  Device         : {DEVICE}")
    model = HybridModel(seq_len=SEQ_LEN).to(DEVICE)

    start_epoch = 1
    best_obj    = float("-inf")
    snapshots   = {}
    step_count  = 0

    # ── Warm-start: resume if architecture matches ────────────────────────
    loaded = False
    for ckpt in [LATEST_MODEL, BEST_MODEL]:
        if ckpt.exists():
            try:
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
                print(f"  Loaded         : {ckpt}")
                loaded = True
                break
            except RuntimeError:
                print(f"  Skipped        : {ckpt}  (architecture mismatch — fresh start)")
                break

    if loaded and STATE_PATH.exists():
        state       = torch.load(STATE_PATH, weights_only=False)
        start_epoch = state.get("next_epoch", 1)
        best_obj    = state.get("best_obj",   float("-inf"))
        step_count  = state.get("step_count", 0)
        print(f"  Resuming epoch : {start_epoch}  |  best: {best_obj:.1f} m  |  steps: {step_count}")

    if SNAPSHOTS_PATH.exists() and loaded:
        snapshots = torch.load(SNAPSHOTS_PATH, weights_only=False)
    # ──────────────────────────────────────────────────────────────────────

    for p in model.physics.parameters():
        p.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer  = Adam(trainable, lr=LR)

    print("=" * 65)
    print("  CAR-FOLLOWING HYBRID MODEL — TRAINING  (GRU f̂)")
    print("=" * 65)
    print(f"  Device     : {DEVICE}")
    print(f"  GRU seq    : {SEQ_LEN} steps")
    print(f"  Epochs     : {start_epoch} → {start_epoch + NUM_EPOCHS - 1}")
    print(f"  Warmup     : {WARMUP_DURATION:.0f}s SUMO-only  →  {EPOCH_DURATION:.0f}s PyTorch control")
    print(f"  Batch size : {BATCH_SIZE} epochs / step  (checkpoint every {CHECKPOINT_EVERY} steps)")
    print(f"  LR         : {LR}")
    print("=" * 65)
    print(f"\n{'epoch':>7}  {'Σ∫v dt (m)':>13}  {'batch avg':>11}  "
          f"{'Δ vs best':>12}  {'grad norm':>10}  {'step':>6}  {'saved':>8}")
    print("-" * 76)

    history    = []
    batch_objs = []

    # Global stat accumulators
    gl = {"v_min":  _INF, "v_max": -_INF,
          "ttc_min": _INF, "ttc_max": -_INF,
          "a_min":  _INF, "a_max": -_INF}

    optimizer.zero_grad()

    for i in range(1, NUM_EPOCHS + 1):
        epoch      = start_epoch + i - 1
        do_collect = (epoch % COLLECT_EVERY == 0) or (epoch == start_epoch)

        model.train()

        velocity_sum, snap, ep_stats = run_epoch(sumocfg, dt, model, DEVICE, collect=do_collect)

        # Accumulate global stats
        gl["v_min"]   = min(gl["v_min"],   ep_stats["v_min"])
        gl["v_max"]   = max(gl["v_max"],   ep_stats["v_max"])
        gl["ttc_min"] = min(gl["ttc_min"], ep_stats["ttc_min"])
        gl["ttc_max"] = max(gl["ttc_max"], ep_stats["ttc_max"])
        gl["a_min"]   = min(gl["a_min"],   ep_stats["a_min"])
        gl["a_max"]   = max(gl["a_max"],   ep_stats["a_max"])

        loss = -velocity_sum / BATCH_SIZE
        loss.backward()

        obj = velocity_sum.item()
        batch_objs.append(obj)

        step_taken = (i % BATCH_SIZE == 0) or (i == NUM_EPOCHS)
        grad_norm  = 0.0
        save_tag   = ""

        if step_taken:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, max_norm=0.1)
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            # Always save the latest model after every optimizer step
            torch.save(model.state_dict(), LATEST_MODEL)
            save_tag = "latest"

            # Named checkpoint every CHECKPOINT_EVERY steps
            if step_count % CHECKPOINT_EVERY == 0:
                ckpt = MODELS_DIR / "checkpoints" / f"step_{step_count:04d}.pt"
                torch.save(model.state_dict(), ckpt)
                save_tag = f"ckpt:{step_count}"

        batch_avg = sum(batch_objs[-BATCH_SIZE:]) / min(len(batch_objs), BATCH_SIZE)
        improved  = obj > best_obj
        delta     = obj - best_obj if best_obj > float("-inf") else 0.0

        if improved:
            best_obj = obj
            torch.save(model.state_dict(), BEST_MODEL)
            save_tag += "+best"

        if do_collect and snap:
            snapshots[epoch] = snap
            torch.save(snapshots, SNAPSHOTS_PATH)

        history.append(obj)
        step_marker = f"#{step_count}" if step_taken else ""
        print(f"{epoch:>7}  {obj:>13.3f}  {batch_avg:>11.1f}  "
              f"{delta:>+12.3f}  {grad_norm:>10.4f}  {step_marker:>6}  {save_tag}")

    torch.save({"next_epoch": start_epoch + NUM_EPOCHS,
                "best_obj":   best_obj,
                "step_count": step_count}, STATE_PATH)
    torch.save(snapshots, SNAPSHOTS_PATH)

    n_window  = NUM_EPOCHS // 5
    avg_early = sum(history[:n_window])  / n_window
    avg_late  = sum(history[-n_window:]) / n_window
    trending  = avg_late > avg_early

    print("-" * 76)
    print(f"\n  Best  Σ∫v           : {best_obj:.1f} m")
    print(f"  Start Σ∫v           : {history[0]:.1f} m")
    print(f"  Avg first {n_window:3d} epochs : {avg_early:.1f} m")
    print(f"  Avg last  {n_window:3d} epochs : {avg_late:.1f} m")
    print(f"  Trend               : {'IMPROVING (+' if trending else 'DECLINING ('}"
          f"{abs(avg_late - avg_early):.1f} m)")
    print(f"  Optimizer steps     : {step_count}")
    print(f"  Latest model        → {LATEST_MODEL}")
    print(f"  Best model          → {BEST_MODEL}")

    v_max_cfg = cfg["speed_limit"]
    print()
    print("=" * 65)
    print("  GLOBAL STATS  (all vehicles × all timesteps × all epochs)")
    print("=" * 65)
    print(f"  Speed      (m/s)  :  min = {gl['v_min']:8.3f}   max = {gl['v_max']:8.3f}"
          f"   (limit = {v_max_cfg} m/s)")
    print(f"  TTC        (s)    :  min = {gl['ttc_min']:8.3f}   max = {gl['ttc_max']:8.3f}")
    print(f"  Accel      (m/s²) :  min = {gl['a_min']:8.3f}   max = {gl['a_max']:8.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
