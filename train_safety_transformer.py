"""
train_safety_transformer.py

Imitation learning: train SafetyTransformer to predict the RKHS functional
gradient descent correction δa*_i(0) in one forward pass.

Each training episode:
  1. Run SUMO + HybridModel + TEACHER FGD (N_TEACHER=20 iters, warm-started)
     — vehicles are kept safe by the teacher correction
  2. At every step with tracked vehicles:
       tokens  = build_tokens(proj_at_δa=0)   ← baseline projection, no correction
       targets = δa*_i[0] from teacher FGD    ← well-converged correction
       store (tokens, targets)
  3. After the episode, train transformer with SGD on the stored samples

The tokens are built from the UNCORRECTED projection (δa=0) so the
transformer input is always the "before correction" state. The target is
what the FGD would correct it to.
"""

from __future__ import annotations

import os
import random
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import sumo as _sumo_pkg
import traci

from conflict import build_snapshot
from model import HybridModel
from safety import run_safety_descent
from safety_transformer import (
    SafetyTransformer, build_tokens, collate_fn,
    TARGET_CLIP, HORIZON,
)
from simulator import build, load_config
from ttc import build_ttc_surfaces, calibrate_cp_offsets

# ── devices & paths ───────────────────────────────────────────────────────────

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"

MODELS_DIR  = Path("models")
XFMR_BEST   = MODELS_DIR / "safety_transformer_best.pt"
XFMR_LATEST = MODELS_DIR / "safety_transformer_latest.pt"
XFMR_STATE  = MODELS_DIR / "safety_transformer_state.pt"

# ── hyper-parameters ──────────────────────────────────────────────────────────

SEQ_LEN          = 5      # HybridModel GRU history length
N_TEACHER        = 20     # FGD iterations for teacher (more = better targets)
ETA              = 50.0   # FGD step size (same as inference)
SIGMA            = 1.0    # FGD kernel width (s)
THRESHOLD        = 3.0    # TTC safety threshold (s)

N_EPISODES       = 200    # total training episodes
SIM_SECONDS      = 15.0   # episode length
WARMUP_SEC       = 2.0    # SUMO-only warmup before PyTorch takes over

BATCH_SIZE       = 32     # samples per gradient step
LR               = 3e-4
WEIGHT_DECAY     = 1e-4
TRAIN_STEPS_EP   = 50     # gradient steps per episode


def _bin(name: str) -> str:
    return str(SUMO_BIN / (name + (".exe" if os.name == "nt" else "")))


# ── model loaders ─────────────────────────────────────────────────────────────

def _load_hybrid() -> HybridModel:
    model = HybridModel(seq_len=SEQ_LEN).to(DEVICE)
    for ckpt in [MODELS_DIR / "hybrid_model_best.pt",
                 MODELS_DIR / "hybrid_model_latest.pt"]:
        if ckpt.exists():
            model.load_state_dict(
                torch.load(ckpt, map_location=DEVICE, weights_only=True))
            print(f"  HybridModel loaded: {ckpt}")
            break
    model.eval()
    return model


def _load_transformer() -> SafetyTransformer:
    model = SafetyTransformer().to(DEVICE)
    for ckpt in [XFMR_BEST, XFMR_LATEST]:
        if ckpt.exists():
            model.load_state_dict(
                torch.load(ckpt, map_location=DEVICE, weights_only=True))
            print(f"  Transformer loaded: {ckpt}")
            break
    return model


# ── simulation helpers ────────────────────────────────────────────────────────

def _query_states(vids: list[str]) -> tuple[dict, dict, dict]:
    v_d, gap_d, vlead_d = {}, {}, {}
    for vid in vids:
        v      = traci.vehicle.getSpeed(vid)
        leader = traci.vehicle.getLeader(vid)
        if leader:
            lid, g      = leader
            gap_d[vid]   = g
            vlead_d[vid] = traci.vehicle.getSpeed(lid)
        else:
            gap_d[vid]   = 100.0
            vlead_d[vid] = v
        v_d[vid] = v
    return v_d, gap_d, vlead_d


def _apply_speeds(
    vids, hybrid, v_d, gap_d, vlead_d, x_seq_dict, delta_a, dt
) -> None:
    if not vids:
        return
    v_t     = torch.tensor([v_d[i]     for i in vids], dtype=torch.float32, device=DEVICE)
    gap_t   = torch.tensor([gap_d[i]   for i in vids], dtype=torch.float32, device=DEVICE)
    vlead_t = torch.tensor([vlead_d[i] for i in vids], dtype=torch.float32, device=DEVICE)
    x_seq_t = torch.stack([x_seq_dict[i] for i in vids]).to(DEVICE)
    with torch.no_grad():
        accel = hybrid(v_t, gap_t, vlead_t, x_seq_t)
    for idx, vid in enumerate(vids):
        corr = delta_a.get(vid, torch.zeros(HORIZON))[0].item()
        if float(v_t[idx]) < 0.1 and corr < 0:
            corr = 0.0
        v_next = max(0.0, float(v_t[idx]) + (float(accel[idx]) + corr) * dt)
        traci.vehicle.setSpeed(vid, v_next)


# ── episode data collection ───────────────────────────────────────────────────

def collect_episode(
    hybrid: HybridModel,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Run one SUMO episode; return list of (tokens [N,TOKEN_DIM], targets [N]) samples.

    The teacher FGD (N_TEACHER iterations) controls vehicle safety and provides
    training targets simultaneously. Tokens are built from the UNCORRECTED
    baseline projection (δa=0) so the transformer always sees the pre-correction state.
    """
    cfg     = load_config()
    sumocfg = build()
    dt      = cfg["step_length"]

    os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
    traci.start([
        _bin("sumo"), "-c", str(sumocfg),
        "--step-length",      str(dt),
        "--collision.action", "teleport",
        "--no-step-log",
    ])
    calibrate_cp_offsets()

    warmup_steps  = int(WARMUP_SEC / dt)
    control_steps = int(SIM_SECONDS / dt)

    known:        set   = set()
    obs_buffers:  dict  = {}
    warm_delta_a: dict  = {}
    stuck_steps:  dict  = {}
    STUCK_LIMIT         = int(5.0 / dt)
    samples:      list  = []

    try:
        # ── warmup: SUMO only ──────────────────────────────────────────────
        for _ in range(warmup_steps):
            traci.simulationStep()

        # ── PyTorch + teacher control ──────────────────────────────────────
        for _ in range(control_steps):
            traci.simulationStep()
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

            # evict stuck vehicles
            for vid in all_vids:
                stuck_steps[vid] = stuck_steps.get(vid, 0) + 1 \
                                   if v_d[vid] < 0.5 else 0
            for vid in [v for v, n in stuck_steps.items() if n > STUCK_LIMIT]:
                try:
                    traci.vehicle.remove(vid)
                except Exception:
                    pass
                for d in (obs_buffers, stuck_steps, warm_delta_a):
                    d.pop(vid, None)
                known.discard(vid)

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
            if not snap.vehicle_stream:
                _apply_speeds(all_vids, hybrid, v_d, gap_d, vlead_d,
                              x_seq_dict, {}, dt)
                warm_delta_a = {}
                continue

            # ── build tokens from UNCORRECTED baseline projection ──────────
            surfaces0, proj0 = build_ttc_surfaces(
                snap, hybrid, v_d, gap_d, vlead_d,
                x_seq_dict=x_seq_dict, dt=dt,
                delta_a_dict=None,
            )
            tokens, vids_tracked = build_tokens(proj0, snap)

            # ── teacher FGD: well-converged correction ────────────────────
            init_da = {
                vid: torch.cat([da[1:], torch.zeros(1)])
                for vid, da in warm_delta_a.items()
            }
            delta_a, _ = run_safety_descent(
                snap, hybrid, v_d, gap_d, vlead_d,
                x_seq_dict  = x_seq_dict,
                dt          = dt,
                n_steps     = N_TEACHER,
                eta         = ETA,
                sigma       = SIGMA,
                threshold   = THRESHOLD,
                verbose     = False,
                init_delta_a= init_da,
            )
            warm_delta_a = delta_a

            # targets: δa*_i[0] for each tracked vehicle (clamped)
            targets = torch.tensor([
                delta_a.get(vid, torch.zeros(HORIZON))[0].item()
                for vid in vids_tracked
            ]).clamp(-TARGET_CLIP, TARGET_CLIP)

            if tokens.shape[0] > 0:
                samples.append((tokens.cpu(), targets.cpu()))

            # apply teacher correction to keep vehicles safe
            _apply_speeds(all_vids, hybrid, v_d, gap_d, vlead_d,
                          x_seq_dict, delta_a, dt)

    finally:
        traci.close()

    return samples


# ── transformer training ──────────────────────────────────────────────────────

def train_on_samples(
    model:     SafetyTransformer,
    optimizer: torch.optim.Optimizer,
    samples:   list[tuple[torch.Tensor, torch.Tensor]],
    n_steps:   int,
) -> float:
    """
    Train transformer on collected (tokens, targets) pairs.
    Returns mean loss over the training steps.
    """
    if not samples:
        return float("nan")

    model.train()
    total_loss = 0.0

    for _ in range(n_steps):
        batch = random.choices(samples, k=min(BATCH_SIZE, len(samples)))
        tok, tgt, msk = collate_fn(batch)

        tok = tok.to(DEVICE)   # [B, max_N, TOKEN_DIM]
        tgt = tgt.to(DEVICE)   # [B, max_N]
        msk = msk.to(DEVICE)   # [B, max_N] bool

        pred = model(tok, pad_mask=msk)    # [B, max_N]

        valid = ~msk                       # [B, max_N] True = real vehicle
        loss  = F.mse_loss(pred[valid], tgt[valid])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / n_steps


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"  Device         : {DEVICE}")
    print(f"  Teacher iters  : {N_TEACHER}")
    print(f"  Episodes       : {N_EPISODES}")
    print(f"  Grad steps/ep  : {TRAIN_STEPS_EP}")
    print(f"  Batch size     : {BATCH_SIZE}")
    print(f"  LR             : {LR}")

    hybrid      = _load_hybrid()
    transformer = _load_transformer()
    optimizer   = AdamW(transformer.parameters(), lr=LR,
                        weight_decay=WEIGHT_DECAY)

    start_ep  = 1
    best_loss = float("inf")

    if XFMR_STATE.exists():
        state     = torch.load(XFMR_STATE, weights_only=False)
        start_ep  = state.get("next_episode", 1)
        best_loss = state.get("best_loss", float("inf"))
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        print(f"  Resuming       : episode {start_ep}  best_loss={best_loss:.5f}")

    # rolling buffer across episodes for experience replay
    replay: list[tuple[torch.Tensor, torch.Tensor]] = []
    MAX_REPLAY = 5000   # max stored samples

    print(f"\n{'ep':>5}  {'samples':>8}  {'replay':>8}  {'loss':>10}  {'saved':>8}")
    print("-" * 50)

    for ep in range(start_ep, start_ep + N_EPISODES):
        # ── collect ───────────────────────────────────────────────────────
        new_samples = collect_episode(hybrid)
        replay.extend(new_samples)
        if len(replay) > MAX_REPLAY:
            replay = replay[-MAX_REPLAY:]   # keep most recent

        # ── train ─────────────────────────────────────────────────────────
        loss = train_on_samples(transformer, optimizer, replay, TRAIN_STEPS_EP)

        # ── checkpoint ────────────────────────────────────────────────────
        torch.save(transformer.state_dict(), XFMR_LATEST)
        tag = ""
        if loss < best_loss:
            best_loss = loss
            torch.save(transformer.state_dict(), XFMR_BEST)
            tag = "+best"

        torch.save({
            "next_episode": ep + 1,
            "best_loss":    best_loss,
            "optimizer":    optimizer.state_dict(),
        }, XFMR_STATE)

        print(f"{ep:5d}  {len(new_samples):8d}  {len(replay):8d}  "
              f"{loss:10.5f}  {tag}")

    print(f"\n  Best loss      : {best_loss:.5f}")
    print(f"  Saved          → {XFMR_BEST}")


if __name__ == "__main__":
    main()
