"""
train_safety_transformer_online.py

End-to-end online training of SafetyTransformer via differentiable co-simulation.

At each SUMO real step:
  1. SUMO provides initial conditions  (v0, gap0, vlead0)  — treated as constants
  2. Build tokens from uncorrected projection               — no grad needed
  3. Transformer forward:  δa = T_θ({x_i})  [N]           — RETAIN GRAPH
  4. Differentiable projection: apply δa at t=0, roll out
     v_traj [N,H] and cum_dist [N,H] keeping graph through δa
  5. Compute TTC surfaces from those tensors (still in graph)
  6. L += V = Σ softplus(c − TTC)                          — GRAD FLOWS BACK
  7. Every GRAD_ACCUM steps: L.backward() → optimizer.step()

HybridModel weights are FROZEN (detached); only T_θ is updated.
SUMO advances the world; PyTorch owns the gradient.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import sumo as _sumo_pkg
import traci

import traci_cache
from conflict import (
    build_snapshot, CONFLICT_MAP, STREAM_NAMES,
    ConflictSnapshot, clear_route_cache,
)
from model import HybridModel
import safety_transformer as _st_module
from safety_transformer import (
    SafetyTransformer, build_tokens, TOKEN_DIM, HORIZON,
)
from simulator import build, load_config
from ttc import (
    calibrate_cp_offsets, _query_d_to_junction,
    _cp_pair_offsets, _EPS_V, ProjectionInfo, _project,
)

# ── device & paths ────────────────────────────────────────────────────────────

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
MODELS_DIR = Path("models")

XFMR_BEST        = MODELS_DIR / "safety_transformer_online_best.pt"
XFMR_LATEST      = MODELS_DIR / "safety_transformer_online_latest.pt"
XFMR_STATE       = MODELS_DIR / "safety_transformer_online_state.pt"
XFMR_CKPT_DIR    = MODELS_DIR / "checkpoints_online"
CHECKPOINT_EVERY  = 10    # save a numbered checkpoint every N episodes

# joint-trained GRU checkpoints (separate from standalone train.py outputs)
HYBRID_BEST   = MODELS_DIR / "hybrid_model_joint_best.pt"
HYBRID_LATEST = MODELS_DIR / "hybrid_model_joint_latest.pt"

# ── hyper-parameters ──────────────────────────────────────────────────────────

SEQ_LEN    = 5     # HybridModel GRU history length — also used in _project_diff
LR         = 3e-4
WEIGHT_DECAY = 1e-4
N_EPISODES = 200
SIM_SECONDS = 25.0   # vehicles need ~14s to reach junction; 25s gives ~10s of conflict-zone data
WARMUP_SEC  = 2.0                        # fallback for single-instance train_episode
WARMUP_SECS = [3.0, 7.0, 12.0, 16.0, 20.0, 25.0]  # per-instance warmup — spread traffic phases across 6 instances
GRAD_ACCUM  = 1       # optimize every step
N_RKHS_ITER = 2       # iterative correction steps per sim-step
IDM_CAP     = 1.5     # m/s²  per-iteration correction cap — 2×1.5 = ±3.0 m/s² total budget
THRESHOLD   = 3.0     # TTC safety threshold (s)
BETA_0      = 2.0     # softplus steepness (tighter than training teacher)
L_VEH       = 5.0     # vehicle length (m) — matches vType in routes.xml
W_CROSS     = 2.0     # effective width of the crossing path at the conflict point (m)
L_OCC       = L_VEH + W_CROSS   # distance a vehicle must travel front-to-rear-clear at CP
DT          = 0.2     # SUMO simulation step (s)
PROJ_DT     = 0.2     # projection step (s) — HORIZON×PROJ_DT = 30×0.2 = 6 s coverage
STUCK_LIMIT      = int(5.0 / DT)
SAVE_EVERY_STEPS = int(5.0 / DT)   # save weights every 5 simulation seconds

# ── parallel training ─────────────────────────────────────────────────────────
N_PARALLEL = 6     # simultaneous SUMO instances — more instances = more stable gradients
BASE_PORT  = 8813  # TraCI ports: BASE_PORT … BASE_PORT+N_PARALLEL-1
N_PRINT    = 50   # print batch table every N steps (sim0 only)

# joint GRU training
LAMBDA_THROUGHPUT = 1.0    # weight of throughput reward — compensates for dt²=0.01 scaling
HYBRID_LR         = 1e-4   # matches standalone train.py
HYBRID_GRAD_CLIP  = 0.1    # matches standalone train.py

# RKHS FGD teacher + speed gate
RKHS_ETA    = 0.5    # effective step size for analytical FGD teacher
LAMBDA_TEACH = 0.1   # weight of imitation loss relative to violation loss
LAMBDA_REG   = 0.01   # L2 penalty on δa — balanced: discourages max corrections without collapsing to zero
LAMBDA_DECEL  = 0.02  # penalty on excessive braking only (one-sided); zero cost for positive δa
V_GATE_LOW  = 1.0    # m/s: below this speed correction is fully suppressed
V_GATE_HIGH = 4.0    # m/s: above this speed correction is fully active

TOTAL_CAP   = IDM_CAP * N_RKHS_ITER   # soft-cap target for accumulated correction
_st_module.TOTAL_CAP = TOTAL_CAP       # expose to build_tokens normalisation


@dataclasses.dataclass
class _SimState:
    """Per-instance mutable state for one parallel SUMO connection."""
    label:         str
    warmup_sec:    float = 2.0
    obs_buffers:   dict = dataclasses.field(default_factory=dict)
    stuck_steps:   dict = dataclasses.field(default_factory=dict)
    known:         set  = dataclasses.field(default_factory=set)
    ep_throughput: torch.Tensor = dataclasses.field(default=None)

    def __post_init__(self):
        if self.ep_throughput is None:
            self.ep_throughput = torch.zeros(1, device=DEVICE)


def _bin(name: str) -> str:
    return str(SUMO_BIN / (name + (".exe" if os.name == "nt" else "")))


# ── model loaders ─────────────────────────────────────────────────────────────

def _load_hybrid() -> HybridModel:
    """
    Load best available f̂_θ checkpoint.
    Physics block (IDMPhysics) is frozen; GRU + head are jointly trained.
    Priority: joint best → joint latest → standalone best → standalone latest → random init.
    """
    model = HybridModel(seq_len=SEQ_LEN).to(DEVICE)
    loaded = False
    for ckpt in [HYBRID_BEST, HYBRID_LATEST,
                 MODELS_DIR / "hybrid_model_best.pt",
                 MODELS_DIR / "hybrid_model_latest.pt"]:
        if ckpt.exists():
            try:
                model.load_state_dict(
                    torch.load(ckpt, map_location=DEVICE, weights_only=True))
                print(f"  f̂_θ  (HybridModel) : {ckpt.name}  ✓")
                loaded = True
                break
            except RuntimeError:
                print(f"  f̂_θ  (HybridModel) : {ckpt.name}  skipped "
                      f"(architecture mismatch)")
    if not loaded:
        print("  f̂_θ  (HybridModel) : no checkpoint found — using random weights")
    # Freeze IDM physics anchors; allow GRU + head to train
    for p in model.physics.parameters():
        p.requires_grad_(False)
    model.train()  # cuDNN RNN backward requires training mode to store activations
    return model


def _load_transformer() -> SafetyTransformer:
    """
    Load best available T_θ checkpoint.

    Priority:
      1. safety_transformer_online_best.pt      (best combined loss, online training)
      2. safety_transformer_online_latest.pt    (most recent weights, online training)
      3. checkpoints_online/ep_NNNN.pt          (highest-episode numbered checkpoint)
      4. safety_transformer_best.pt             (imitation-trained warm start)
      5. safety_transformer_latest.pt           (imitation latest)
      6. random init
    """
    model = SafetyTransformer().to(DEVICE)

    candidates = [XFMR_BEST, XFMR_LATEST]

    # highest-episode numbered checkpoint (fallback if best/latest were deleted)
    if XFMR_CKPT_DIR.exists():
        numbered = sorted(
            XFMR_CKPT_DIR.glob("ep_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
            reverse=True,
        )
        if numbered:
            candidates.append(numbered[0])

    candidates += [
        MODELS_DIR / "safety_transformer_best.pt",
        MODELS_DIR / "safety_transformer_latest.pt",
    ]

    for ckpt in candidates:
        if ckpt.exists():
            try:
                model.load_state_dict(
                    torch.load(ckpt, map_location=DEVICE, weights_only=True))
                print(f"  T_θ (Transformer)  : {ckpt.name}  ✓")
                return model
            except RuntimeError:
                print(f"  T_θ (Transformer)  : {ckpt.name}  skipped "
                      f"(architecture mismatch)")

    print("  T_θ (Transformer)  : no checkpoint found — using random init")
    torch.nn.init.uniform_(model.head.weight, -0.01, 0.01)
    torch.nn.init.zeros_(model.head.bias)
    return model


# ── differentiable projection ─────────────────────────────────────────────────

def _project_diff(
    v_traj0:   torch.Tensor,   # [N, H]  baseline speeds    (no grad)
    cum_dist0: torch.Tensor,   # [N, H]  baseline distances (no grad)
    delta_a_n: torch.Tensor,   # [N]     correction — CARRIES GRADIENT from T_θ
    dt:        float,
    horizon:   int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Analytically shift the baseline trajectory by delta_a applied at t=0.

    The hybrid model is called with .detach() at every step in _project, so
    accel_base contributes zero to the gradient regardless.  The only gradient
    path is through the v chain:
        delta_a → v[0] += delta_a*dt → v[1] += delta_a*dt → ...
    giving ∂v[t]/∂delta_a = dt and ∂cum_dist[t]/∂delta_a = t*dt² for all t.
    These are exactly the values this formula produces — the gradient is
    preserved exactly, not approximately.

    v_traj[i,t]   = v_traj0[i,t]   + delta_a[i] * dt
    cum_dist[i,t] = cum_dist0[i,t] + delta_a[i] * t * dt²
    """
    _V_MAX     = 13.89   # m/s — matches vType maxSpeed; prevents projection from overspeeding
    t_idx      = torch.arange(horizon, device=delta_a_n.device, dtype=torch.float32)
    v_traj_d   = (v_traj0   + delta_a_n.unsqueeze(1) * dt).clamp(min=1e-3, max=_V_MAX)
    cum_dist_d = cum_dist0  + delta_a_n.unsqueeze(1) * t_idx * (dt ** 2)
    return v_traj_d, cum_dist_d


# ── differentiable violation ──────────────────────────────────────────────────

def compute_intersection_min_ttc(
    snap:    ConflictSnapshot,
    d_junc:  torch.Tensor,      # [N] current distance to junction entry (m)
    v0:      torch.Tensor,      # [N] current speeds (m/s)
    idx_of:  dict[str, int],
    tracked: list[str],
) -> float:
    """Minimum intersection TTC (s) across all conflicting stream pairs.

    For each conflicting pair (i, j): compute each vehicle's ETA to its
    conflict point using current speed only — no projection, no car-following.
    Returns the smallest |ETA_i − ETA_j|; lower = more dangerous.
    """
    worst = float("inf")
    with torch.no_grad():
        for vid in tracked:
            k      = idx_of.get(vid)
            stream = snap.vehicle_stream.get(vid)
            if k is None or stream is None:
                continue
            for cs in sorted(CONFLICT_MAP.get(stream, frozenset()),
                             key=lambda s: STREAM_NAMES.get(s, "")):
                rivals = [r for r in snap.stream_vehicles.get(cs, []) if r in idx_of]
                if not rivals:
                    continue
                off_i, off_j = _cp_pair_offsets(stream, cs)
                eta_i = (d_junc[k] + off_i).clamp(min=0.0) / v0[k].clamp(min=_EPS_V)
                for rvid in rivals:
                    ri    = idx_of[rvid]
                    eta_j = (d_junc[ri] + off_j).clamp(min=0.0) / v0[ri].clamp(min=_EPS_V)
                    gap   = (eta_i - eta_j).abs().item()
                    if gap < worst:
                        worst = gap
    return worst


def compute_current_violation(
    snap:      ConflictSnapshot,
    d_junc:    torch.Tensor,   # [N] current distance to junction entry (m)
    v0:        torch.Tensor,   # [N] current speeds (m/s)
    idx_of:    dict[str, int],
    tracked:   list[str],
    threshold: float = THRESHOLD,
    beta:      float = BETA_0,
) -> float:
    """
    TRUE instantaneous violation at the current sim state.
    Uses only current v and d_junc — no projection, no car-following, no delta_a.
    eta_i = (d_junc[i] + off_i) / v[i]   (simple constant-speed ETA to conflict point)
    Returns a plain float so it can be displayed without affecting the training graph.
    """
    total = 0.0
    counted: set = set()
    with torch.no_grad():
        for vid in tracked:
            k      = idx_of.get(vid)
            stream = snap.vehicle_stream.get(vid)
            if k is None or stream is None:
                continue
            for cs in sorted(CONFLICT_MAP.get(stream, frozenset()),
                             key=lambda s: STREAM_NAMES.get(s, "")):
                rivals = [r for r in snap.stream_vehicles.get(cs, []) if r in idx_of]
                if not rivals:
                    continue
                off_i, off_j = _cp_pair_offsets(stream, cs)
                v_i       = max(v0[k].item(), _EPS_V)
                t_enter_i = max(d_junc[k].item() + off_i, 0.0) / v_i
                occ_i     = L_OCC / v_i
                for rvid in rivals:
                    ri   = idx_of[rvid]
                    pair = (min(k, ri), max(k, ri))
                    if pair in counted:
                        continue
                    counted.add(pair)
                    v_j       = max(v0[ri].item(), _EPS_V)
                    t_enter_j = max(d_junc[ri].item() + off_j, 0.0) / v_j
                    occ_j     = L_OCC / v_j
                    # occupancy gap: time between one vehicle clearing CP and the other entering
                    # > 0 → safe sequential use;  ≤ 0 → simultaneous occupancy (collision)
                    gap = max(t_enter_i - t_enter_j - occ_j,
                              t_enter_j - t_enter_i - occ_i)
                    total += float(F.softplus(
                        torch.tensor(threshold - gap, dtype=torch.float32), beta=beta))
    return total / max(len(counted), 1)   # per-pair, same scale as compute_violation_diff


def compute_violation_diff(
    snap:      ConflictSnapshot,
    v_traj:    torch.Tensor,       # [N, H]  with grad
    cum_dist:  torch.Tensor,       # [N, H]  with grad
    d_junc:    torch.Tensor,       # [N]     detached
    idx_of:    dict[str, int],
    tracked:   list[str],
    dt:        float,
    threshold: float = THRESHOLD,
    beta:      float = BETA_0,
    normalize: bool  = True,       # divide by pair count so loss is scale-invariant
) -> torch.Tensor:
    """
    Asymmetric violation loss: only the YIELDING vehicle (the one that arrives
    later at the shared conflict point) accumulates a penalty for each pair.

    The passing vehicle does not receive a braking gradient — it should
    maintain or increase speed to clear the zone, which the car-following
    model already does. Penalising it symmetrically would push both vehicles
    to brake, which is not what happens in reality.

    Gradient flows back through v_traj/cum_dist → delta_a_n → T_θ only for
    the yielder in each pair.
    """
    total = torch.zeros(1, device=DEVICE)

    # track which (i, j) pairs have already been counted to avoid double-counting
    counted: set = set()

    for vid in tracked:
        k      = idx_of.get(vid)
        stream = snap.vehicle_stream.get(vid)
        if k is None or stream is None:
            continue

        conf_streams = sorted(CONFLICT_MAP.get(stream, frozenset()),
                              key=lambda s: STREAM_NAMES.get(s, ""))

        for cs in conf_streams:
            rivals = [r for r in snap.stream_vehicles.get(cs, [])
                      if r in idx_of]
            if not rivals:
                continue

            off_i, off_j = _cp_pair_offsets(stream, cs)

            d_cp_i  = d_junc[k] + off_i
            d_rem_i = (d_cp_i - cum_dist[k]).clamp(min=0.0)      # [H]
            eta_i   = (d_rem_i / v_traj[k].clamp(min=_EPS_V)).clamp(max=threshold * 5)  # [H]
            occ_i   = L_OCC   / v_traj[k].clamp(min=_EPS_V)      # [H] time i occupies CP

            for rvid in rivals:
                ri = idx_of[rvid]
                pair = (min(k, ri), max(k, ri))
                if pair in counted:
                    continue
                counted.add(pair)

                d_cp_j  = d_junc[ri] + off_j
                d_rem_j = (d_cp_j - cum_dist[ri]).clamp(min=0.0)
                eta_j   = (d_rem_j / v_traj[ri].clamp(min=_EPS_V)).clamp(max=threshold * 5)  # [H]
                occ_j   = L_OCC   / v_traj[ri].clamp(min=_EPS_V)  # [H]

                # Yielder role: vehicle with the larger arrival ETA at t=0 should yield.
                with torch.no_grad():
                    ego_yields = (eta_i[0] >= eta_j[0])

                if ego_yields:
                    # i yields → gap = eta_i − (eta_j + occ_j); gradient through eta_i only
                    gap = eta_i - (eta_j + occ_j).detach()
                    total = total + F.softplus(threshold - gap, beta=beta).mean()
                else:
                    # j yields → gap = eta_j − (eta_i + occ_i); gradient through eta_j only
                    gap = eta_j - (eta_i + occ_i).detach()
                    total = total + F.softplus(threshold - gap, beta=beta).mean()

    n_pairs = max(len(counted), 1)
    return total / n_pairs if normalize else total


def compute_social_force(
    snap:    ConflictSnapshot,
    d_junc:  torch.Tensor,       # [N] distance to junction stop-line; negative = in junction
    v0:      torch.Tensor,       # [N] current speeds (m/s)
    idx_of:  dict[str, int],
    tracked: list[str],
    A:       float = 2.0,        # max force amplitude (m/s²)
    sigma:   float = THRESHOLD,  # TTC gap at which force ramps to zero (s)
) -> torch.Tensor:               # [N] ≤ 0, clamped to [-b_max, 0]
    """
    TTC-space leader-follower repulsion for conflict-zone vehicles.

    For each vehicle i that is the yielder (higher ETA to shared conflict point),
    apply a deceleration proportional to conflict imminence:

        F_i = -A × Σ_j  max(0, σ − gap_ij) / σ
        gap_ij = eta_i − eta_j − occ_j,   occ_j = L_OCC / v_j

    Active for both inbound and in-junction vehicles — extends car-following
    leader-follower logic into the TTC/conflict domain.  Feed-forward only;
    no gradient flows through this term (called inside torch.no_grad()).
    """
    B_MAX = 3.0   # matches IDMPhysics.b_max
    force = torch.zeros(len(tracked), device=d_junc.device)

    for i, vid in enumerate(tracked):
        stream = snap.vehicle_stream.get(vid)
        if stream is None:
            continue

        k   = idx_of[vid]
        v_i = v0[k].clamp(min=_EPS_V)

        for cs in sorted(CONFLICT_MAP.get(stream, frozenset()),
                         key=lambda s: STREAM_NAMES.get(s, "")):
            rivals = [r for r in snap.stream_vehicles.get(cs, []) if r in idx_of]
            if not rivals:
                continue

            off_i, off_j = _cp_pair_offsets(stream, cs)
            eta_i = (d_junc[k] + off_i).clamp(min=0.0) / v_i

            for rvid in rivals:
                ri    = idx_of[rvid]
                v_j   = v0[ri].clamp(min=_EPS_V)
                eta_j = (d_junc[ri] + off_j).clamp(min=0.0) / v_j
                occ_j = L_OCC / v_j

                if eta_i <= eta_j:
                    continue   # i is the passer — no repulsion

                gap_ij = eta_i - eta_j - occ_j
                if gap_ij >= sigma:
                    continue   # safe gap, no force

                f_mag = A * (sigma - gap_ij.clamp(max=sigma)) / sigma
                force[i] = force[i] - f_mag

    return force.clamp(min=-B_MAX, max=0.0)


# ── simulation helpers ────────────────────────────────────────────────────────

def _query_states(vids: list[str]) -> tuple[dict, dict, dict]:
    v_d, gap_d, vlead_d = {}, {}, {}
    for vid in vids:
        v = traci_cache.get_speed(vid)
        try:
            leader = traci.vehicle.getLeader(vid)
        except traci.exceptions.TraCIException:
            leader = None
        if leader:
            lid, g       = leader
            gap_d[vid]   = g
            vlead_d[vid] = traci_cache.get_speed(lid)
        else:
            gap_d[vid]   = 100.0
            vlead_d[vid] = v
        v_d[vid] = v
    return v_d, gap_d, vlead_d


CONTROL_DIST = 40.0   # m: only override speed within this distance of the junction stop-line

def _apply_speeds(vids, hybrid, v_d, gap_d, vlead_d, x_seq_dict, delta_a_np, dt,
                  is_turning_d: dict | None = None,
                  social_a_np: dict | None = None):
    """Apply corrected speeds to SUMO (detached — SUMO world is not in graph).

    Vehicles farther than CONTROL_DIST from the junction are released back to
    SUMO's own car-following so their rear-end safety is preserved.
    """
    if not vids:
        return
    d_junc_d = _query_d_to_junction(vids)   # cache hit — no extra TraCI round-trips
    v_t  = torch.tensor([v_d[i]   for i in vids], dtype=torch.float32, device=DEVICE)
    g_t  = torch.tensor([gap_d[i] for i in vids], dtype=torch.float32, device=DEVICE)
    vl_t = torch.tensor([vlead_d[i] for i in vids], dtype=torch.float32, device=DEVICE)
    xs_t = torch.stack([
        (x_seq_dict[i] if i in x_seq_dict else torch.zeros(SEQ_LEN, 3)).to(DEVICE)
        for i in vids
    ])
    it_t = torch.tensor(
        [is_turning_d.get(v, 0.0) for v in vids] if is_turning_d else [0.0] * len(vids),
        dtype=torch.float32, device=DEVICE,
    )
    sa_t = torch.tensor(
        [social_a_np.get(v, 0.0) for v in vids] if social_a_np else [0.0] * len(vids),
        dtype=torch.float32, device=DEVICE,
    )
    with torch.no_grad():
        accel = hybrid(v_t, g_t, vl_t, xs_t, it_t, social_a=sa_t)
    for idx, vid in enumerate(vids):
        if d_junc_d.get(vid, 999.0) > CONTROL_DIST:
            traci.vehicle.setSpeedMode(vid, 31)   # restore SUMO safety checks
            traci.vehicle.setSpeed(vid, -1)        # release speed override
            continue
        corr = float(delta_a_np.get(vid, 0.0))
        if float(v_t[idx]) < 0.1 and corr < 0:
            corr = 0.0
        IDM_B = 3.0   # m/s²  comfortable deceleration cap
        total_a = float(accel[idx]) + corr
        if total_a < -IDM_B:
            corr = -IDM_B - float(accel[idx])
        v_next = max(0.0, float(v_t[idx]) + (float(accel[idx]) + corr) * dt)
        traci.vehicle.setSpeedMode(vid, 0)
        traci.vehicle.setSpeed(vid, v_next)


# ── per-step helper (shared by train_episode and train_parallel) ─────────────

def _process_step(
    sim:         _SimState,
    ctrl_step:   int,
    dt:          float,
    transformer: SafetyTransformer,
    hybrid:      HybridModel,
    verbose:     bool = False,
):
    """
    Advance one SUMO step for sim (traci must already be switched to sim.label),
    compute correction and losses.
    Returns (loss, V_val, 0.0, T_gru_val, n_tracked, min_ttc, V_true) or None.
    """
    traci.simulationStep()
    all_vids = list(traci.vehicle.getIDList())
    traci_cache.update(all_vids)

    n_tele = traci.simulation.getStartingTeleportNumber()
    if n_tele > 0:
        _t = (ctrl_step + 1) * dt + sim.warmup_sec
        print(f"    [{sim.label} ⚠ teleport] {n_tele} vehicle(s) at t={_t:.1f}s", flush=True)

    v_d, gap_d, vlead_d = _query_states(all_vids)

    for vid in all_vids:
        if vid not in sim.known:
            traci.vehicle.setSpeedMode(vid, 0)
            sim.known.add(vid)
            obs0 = torch.tensor([v_d[vid], gap_d[vid], vlead_d[vid]], dtype=torch.float32)
            sim.obs_buffers[vid] = deque([obs0] * SEQ_LEN, maxlen=SEQ_LEN)
        sim.obs_buffers[vid].append(
            torch.tensor([v_d[vid], gap_d[vid], vlead_d[vid]], dtype=torch.float32))

    for vid in all_vids:
        sim.stuck_steps[vid] = sim.stuck_steps.get(vid, 0) + 1 if v_d[vid] < 0.5 else 0
    for vid in [v for v, n in sim.stuck_steps.items() if n > STUCK_LIMIT]:
        try:
            traci.vehicle.remove(vid)
        except Exception:
            pass
        sim.obs_buffers.pop(vid, None)
        sim.stuck_steps.pop(vid, None)
        sim.known.discard(vid)

    all_vids = list(traci.vehicle.getIDList())
    for vid in list(sim.obs_buffers):
        if vid not in set(all_vids):
            del sim.obs_buffers[vid]
            sim.known.discard(vid)

    x_seq_dict = {
        vid: torch.stack(list(sim.obs_buffers[vid]))
        for vid in all_vids if vid in sim.obs_buffers
    }

    snap = build_snapshot(all_vids)
    candidates = [vid for vid, mvmt in snap.vehicle_stream.items()
                  if mvmt is not None and CONFLICT_MAP.get(mvmt)]

    # turning flag for every vehicle — used in _apply_speeds and _project
    is_turning_all = {
        vid: (1.0 if STREAM_NAMES.get(snap.vehicle_stream.get(vid), "").endswith(("_R", "_L"))
              else 0.0)
        for vid in all_vids
    }

    if not candidates:
        _apply_speeds(all_vids, hybrid, v_d, gap_d, vlead_d, x_seq_dict, {}, dt,
                      is_turning_d=is_turning_all)
        return None

    with torch.no_grad():
        ACTIVE_ETA = HORIZON * PROJ_DT
        d_raw_all  = _query_d_to_junction(candidates)
        active_vids: set = set()
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
                        active_vids.add(_vid)
                        active_vids.add(_rvid)
        tracked = [v for v in candidates if v in active_vids]

        if not tracked:
            _apply_speeds(all_vids, hybrid, v_d, gap_d, vlead_d, x_seq_dict, {}, dt,
                          is_turning_d=is_turning_all)
            return None

        v0_cpu   = torch.tensor([v_d[i]     for i in tracked], dtype=torch.float32)
        gap0_cpu = torch.tensor([gap_d[i]   for i in tracked], dtype=torch.float32)
        vl0_cpu  = torch.tensor([vlead_d[i] for i in tracked], dtype=torch.float32)
        xs0_cpu  = torch.stack([
            x_seq_dict[i] if i in x_seq_dict else torch.zeros(SEQ_LEN, 3)
            for i in tracked
        ])
        idx_of     = {vid: k for k, vid in enumerate(tracked)}
        d_junc_cpu = torch.tensor([d_raw_all[i] for i in tracked], dtype=torch.float32)

        leader_idx_cpu = torch.full((len(tracked),), -1, dtype=torch.long)
        for i, vid in enumerate(tracked):
            try:
                info = traci.vehicle.getLeader(vid)
            except traci.exceptions.TraCIException:
                info = None
            if info and info[0] in idx_of:
                leader_idx_cpu[i] = idx_of[info[0]]

        d_junc_lead_cpu = torch.full(
            (len(tracked),), _st_module.ARM_LEN, dtype=torch.float32)
        for i in range(len(tracked)):
            li = leader_idx_cpu[i].item()
            if li >= 0:
                d_junc_lead_cpu[i] = d_junc_cpu[li]

        # is_turning for tracked vehicles — used in _project and direct hybrid calls
        is_turning_tracked = torch.tensor([
            1.0 if STREAM_NAMES.get(snap.vehicle_stream.get(vid), "").endswith(("_R", "_L"))
            else 0.0
            for vid in tracked
        ], dtype=torch.float32, device=DEVICE)

        v_traj0, cum_dist0 = _project(
            hybrid,
            v0_cpu.to(DEVICE), gap0_cpu.to(DEVICE), vl0_cpu.to(DEVICE),
            xs0_cpu.to(DEVICE), leader_idx_cpu.to(DEVICE),
            PROJ_DT, HORIZON,
            is_turning=is_turning_tracked)
        dj_gpu = d_junc_cpu.to(DEVICE)

        social_force_gpu = compute_social_force(
            snap, dj_gpu, v0_cpu.to(DEVICE), idx_of, tracked)
        sa_np = {vid: social_force_gpu[i].item() for i, vid in enumerate(tracked)}

    # RKHS iterative correction
    stored_steps = []
    delta_a_n    = torch.zeros(len(tracked), device=DEVICE)
    v_traj_d     = v_traj0
    cum_dist_d   = cum_dist0

    for _rkhs_iter in range(N_RKHS_ITER):
        proj_iter = ProjectionInfo(
            v_traj=v_traj_d.detach(), cum_dist=cum_dist_d.detach(),
            d_junc=dj_gpu, idx_of=idx_of, tracked=tracked)
        tok, _ = build_tokens(
            proj_iter, snap,
            x_seq_dict = x_seq_dict,
            prev_da    = delta_a_n.detach(),
            gap        = gap0_cpu.to(DEVICE),
            v_lead     = vl0_cpu.to(DEVICE),
        )
        raw_out = transformer(tok.unsqueeze(0).to(DEVICE)).squeeze(0)
        # STE: clamp bounds the forward value; gradient flows through as if unclipped
        raw_c = raw_out.clamp(-IDM_CAP, IDM_CAP)
        stored_steps.append(raw_c.detach() + raw_out - raw_out.detach())
        raw_sum   = torch.stack(stored_steps).sum(0)
        raw_sum_c = raw_sum.clamp(-TOTAL_CAP, TOTAL_CAP)
        delta_a_n = raw_sum_c.detach() + raw_sum - raw_sum.detach()
        v_traj_d, cum_dist_d = _project_diff(v_traj0, cum_dist0, delta_a_n, PROJ_DT, HORIZON)

    # h_= gate: suppress transformer correction when vehicle is nearly stopped
    mu_stopped = torch.clamp(
        (V_GATE_HIGH - v0_cpu.to(DEVICE)) / (V_GATE_HIGH - V_GATE_LOW),
        min=0.0, max=1.0)
    delta_a_n  = delta_a_n + mu_stopped * (0.0 - delta_a_n)
    v_traj_d, cum_dist_d = _project_diff(v_traj0, cum_dist0, delta_a_n, PROJ_DT, HORIZON)

    if verbose:
        with torch.no_grad():
            _sim_t = (ctrl_step + 1) * dt + sim.warmup_sec
            print(f"    ┌─ [{sim.label}] t={_sim_t:.1f}s  {len(tracked)} vehicles")
            for _i, _vid in enumerate(tracked):
                _sname = STREAM_NAMES.get(snap.vehicle_stream.get(_vid), "----")
                print(f"    │  {_vid:>20s}  {_sname:<5s}"
                      f"  v={v0_cpu[_i].item():5.2f}"
                      f"  μs={mu_stopped[_i].item():.2f}"
                      f"  δa={delta_a_n[_i].item():+.4f}", flush=True)
            print("    └" + "─" * 55, flush=True)

    V       = compute_violation_diff(snap, v_traj_d, cum_dist_d, dj_gpu, idx_of, tracked, PROJ_DT)
    L_decel = LAMBDA_DECEL * torch.relu(-delta_a_n).pow(2).mean()

    # throughput for GRU only
    accel_base = hybrid(v0_cpu.to(DEVICE), gap0_cpu.to(DEVICE),
                        vl0_cpu.to(DEVICE), xs0_cpu.to(DEVICE),
                        is_turning_tracked)
    v0_d  = v0_cpu.to(DEVICE).detach()
    T_gru = (v0_d + (accel_base + delta_a_n.detach()) * dt).clamp(min=0.0).sum() * dt
    sim.ep_throughput = sim.ep_throughput + T_gru

    step_min_ttc = compute_intersection_min_ttc(snap, dj_gpu, v0_cpu.to(DEVICE), idx_of, tracked)
    V_true = compute_current_violation(snap, dj_gpu, v0_cpu.to(DEVICE), idx_of, tracked)

    da_np = {vid: delta_a_n[i].item() for i, vid in enumerate(tracked)}
    _apply_speeds(all_vids, hybrid, v_d, gap_d, vlead_d, x_seq_dict, da_np, dt,
                  is_turning_d=is_turning_all, social_a_np=sa_np)

    return V + L_decel, V.item(), L_decel.item(), T_gru.item(), len(tracked), step_min_ttc, V_true


# ── parallel training (N simultaneous SUMO instances) ────────────────────────

def train_parallel(
    transformer:      SafetyTransformer,
    hybrid:           HybridModel,
    optimizer_xfmr:   torch.optim.Optimizer,
    optimizer_hybrid: torch.optim.Optimizer,
    n_parallel:       int  = N_PARALLEL,
    gui:              bool = False,
    gui_delay:        int  = 100,
    ep:               int  = 0,
) -> tuple[float, float, int]:
    """
    Run n_parallel SUMO instances simultaneously.
    Each control step, all N instances are advanced and their losses are averaged
    before accumulation — identical to a mini-batch over scenarios.
    Returns (total_safety_loss, avg_throughput_per_instance, n_optimizer_steps).
    """
    cfg = load_config()
    dt  = cfg["step_length"]

    os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)

    # build once — all instances share the same static network/routes files
    sumocfg = build()

    sims: list[_SimState] = []
    for i in range(n_parallel):
        label   = f"sim{i}"
        cmd = [
            _bin("sumo-gui" if (gui and i == 0) else "sumo"),
            "-c", str(sumocfg),
            "--step-length",        str(dt),
            "--collision.action",   "teleport",
            "--collision.check-junctions",          # detect overlaps inside junction area
            "--no-step-log",
            "--seed",               str(ep * n_parallel + i),
        ]
        if gui and i == 0:
            cmd += ["--start", "--quit-on-end", "--delay", str(gui_delay)]
        traci.start(cmd, port=BASE_PORT + i, label=label)
        if i == 0:
            calibrate_cp_offsets()   # network geometry is the same for all
        warmup_i = WARMUP_SECS[i] if i < len(WARMUP_SECS) else WARMUP_SECS[-1]
        for _ in range(int(warmup_i / dt)):
            traci.simulationStep()
        sims.append(_SimState(label=label, warmup_sec=warmup_i))

    control_steps  = int(SIM_SECONDS / dt)
    accum_safety_val = 0.0              # float sum for display; backward called per-step
    pending_grad     = False            # True once any per-step backward has fired
    accum_count    = 0
    total_safety   = 0.0
    n_opt_steps    = 0
    total_T        = torch.zeros(1, device=DEVICE)  # accumulated over episode
    disp_V = disp_Lv = disp_V_true = disp_T = 0.0
    disp_worst_ttc = float("inf")
    V_window_start = None
    V_window_end   = 0.0

    transformer.train()
    optimizer_xfmr.zero_grad()
    optimizer_hybrid.zero_grad()

    try:
        for ctrl_step in range(control_steps):
            step_loss  = torch.zeros(1, device=DEVICE)
            step_V = step_Lv = step_T = 0.0
            step_ttc   = float("inf")
            n_active   = 0

            step_V_true = 0.0
            for idx, sim in enumerate(sims):
                traci.switch(sim.label)
                verbose = (idx == 0 and ctrl_step % N_PRINT == 0)
                result  = _process_step(sim, ctrl_step, dt, transformer, hybrid, verbose=verbose)
                if result is None:
                    continue
                loss, V_val, Lv_val, T_val, _n_trk, ttc, V_true_val = result
                step_loss   = step_loss  + loss      / n_parallel
                step_V     += V_val                  / n_parallel
                step_Lv    += Lv_val                 / n_parallel
                step_T     += T_val                  / n_parallel
                step_V_true += V_true_val            / n_parallel
                if ttc < step_ttc:
                    step_ttc = ttc
                n_active += 1

            if n_active == 0:
                continue

            if V_window_start is None:
                V_window_start = step_V_true
            V_window_end = step_V_true   # track TRUE violation trend across window

            if step_loss.grad_fn is not None:
                (step_loss / GRAD_ACCUM).backward()
                pending_grad = True
            accum_safety_val += step_loss.item()
            disp_V        += step_V
            disp_Lv       += step_Lv
            disp_V_true   += step_V_true
            disp_T        += step_T
            if step_ttc < disp_worst_ttc:
                disp_worst_ttc = step_ttc
            accum_count   += 1

            if ctrl_step > 0 and ctrl_step % SAVE_EVERY_STEPS == 0:
                torch.save(transformer.state_dict(), XFMR_LATEST)
                torch.save(hybrid.state_dict(),      HYBRID_LATEST)
                _t = (ctrl_step + 1) * dt + sims[0].warmup_sec
                print(f"    [save] t={_t:.1f}s → {XFMR_LATEST.name}", flush=True)

            if accum_count >= GRAD_ACCUM:
                if pending_grad:
                    gn_xfmr = torch.nn.utils.clip_grad_norm_(
                        transformer.parameters(), max_norm=1.0)
                    optimizer_xfmr.step()
                else:
                    gn_xfmr = torch.tensor(0.0)
                optimizer_xfmr.zero_grad()

                sim_t   = (ctrl_step + 1) * dt + sims[0].warmup_sec
                ttc_str = f"{disp_worst_ttc:.2f}s" if disp_worst_ttc < float("inf") else "  n/a"
                v_start = V_window_start if V_window_start is not None else 0.0
                v_arrow = "↓" if V_window_end < v_start else "↑"
                print(f"    t={sim_t:5.1f}s | N={n_parallel:2d} | "
                      f"V_proj={disp_V/accum_count:7.3f} | "
                      f"Ld={disp_Lv/accum_count:7.3f} | "
                      f"V_true={disp_V_true/accum_count:7.3f} ({v_start:.1f}{v_arrow}{V_window_end:.1f}) | "
                      f"wTTC={ttc_str} | "
                      f"|gX|={float(gn_xfmr):.3f}"
                      f"{'*' if float(gn_xfmr) > 5.0 else ''}",
                      flush=True)

                total_safety  += accum_safety_val / accum_count
                n_opt_steps   += 1
                accum_safety_val = 0.0
                pending_grad     = False
                accum_count    = 0
                disp_V = disp_Lv = disp_V_true = disp_T = 0.0
                disp_worst_ttc = float("inf")
                V_window_start = None

        # flush remaining steps
        if accum_count > 0:
            if pending_grad:
                gn_xfmr = torch.nn.utils.clip_grad_norm_(
                    transformer.parameters(), max_norm=5.0)
                optimizer_xfmr.step()
                print(f"    [flush] V={accum_safety_val/accum_count:.4f} | "
                      f"|gX|={float(gn_xfmr):.3f}"
                      f"{'*' if float(gn_xfmr) > 5.0 else ''}", flush=True)
                total_safety += accum_safety_val / accum_count
                n_opt_steps  += 1
            optimizer_xfmr.zero_grad()

        # end-of-episode GRU update — average throughput across all N instances
        for sim in sims:
            total_T = total_T + sim.ep_throughput
        if total_T.grad_fn is not None:
            gru_loss = -LAMBDA_THROUGHPUT * total_T / n_parallel
            gru_loss.backward()
            gn_hyb = torch.nn.utils.clip_grad_norm_(
                [p for p in hybrid.parameters() if p.requires_grad],
                max_norm=HYBRID_GRAD_CLIP)
            optimizer_hybrid.step()
            print(f"    [GRU ep-update] T={total_T.item()/n_parallel:.1f}m | "
                  f"|gH|={float(gn_hyb):.4f}"
                  f"{'*' if float(gn_hyb) > HYBRID_GRAD_CLIP else ''}", flush=True)
        optimizer_hybrid.zero_grad()

    finally:
        for sim in sims:
            try:
                traci.switch(sim.label)
                traci.close()
            except Exception:
                pass
        clear_route_cache()
        traci_cache.clear()

    return total_safety, total_T.item() / n_parallel, n_opt_steps


# ── training episode ──────────────────────────────────────────────────────────

def train_episode(
    transformer:      SafetyTransformer,
    hybrid:           HybridModel,
    optimizer_xfmr:   torch.optim.Optimizer,
    optimizer_hybrid: torch.optim.Optimizer,
    gui:              bool = False,
    gui_delay:        int  = 100,
) -> tuple[float, float, int]:
    """
    Run one SUMO episode.
    Returns (total_safety_loss, total_throughput_sum, n_optimizer_steps).
    Both the SafetyTransformer and HybridModel GRU are updated jointly.
    """
    cfg     = load_config()
    sumocfg = build()
    dt      = cfg["step_length"]

    os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
    sumo_cmd = [
        _bin("sumo-gui" if gui else "sumo"), "-c", str(sumocfg),
        "--step-length",        str(dt),
        "--collision.action",   "teleport",
        "--collision.check-junctions",      # detect overlaps inside junction area
        "--no-step-log",
    ]
    if gui:
        sumo_cmd += ["--start", "--quit-on-end", "--delay", str(gui_delay)]
    traci.start(sumo_cmd)
    calibrate_cp_offsets()

    warmup_steps  = int(WARMUP_SEC / dt)
    control_steps = int(SIM_SECONDS / dt)

    known:       set  = set()
    obs_buffers: dict = {}
    stuck_steps: dict = {}

    accum_safety_val = 0.0              # float sum for display; backward called per-step
    pending_grad     = False            # True once any per-step backward has fired
    accum_count   = 0
    ep_throughput = torch.zeros(1, device=DEVICE)   # accumulates entire episode
    total_safety  = 0.0
    n_opt_steps   = 0
    # display-only accumulators (float, no grad) for readable per-window logs
    disp_V         = 0.0
    disp_T         = 0.0
    disp_worst_ttc = float("inf")   # minimum TTC seen in current window
    V_window_start = None           # V at first step of current accum window
    V_window_end   = 0.0            # V at last step of current accum window

    try:
        for _ in range(warmup_steps):
            traci.simulationStep()

        transformer.train()
        optimizer_xfmr.zero_grad()
        optimizer_hybrid.zero_grad()

        for ctrl_step in range(control_steps):
            traci.simulationStep()
            all_vids = list(traci.vehicle.getIDList())

            n_tele = traci.simulation.getStartingTeleportNumber()
            if n_tele > 0:
                _tele_t = (ctrl_step + 1) * dt + WARMUP_SEC
                print(f"    [⚠ teleport] {n_tele} vehicle(s) at t={_tele_t:.1f}s", flush=True)

            v_d, gap_d, vlead_d = _query_states(all_vids)

            # ── init / update observation buffers ────────────────────────
            for vid in all_vids:
                if vid not in known:
                    traci.vehicle.setSpeedMode(vid, 0)
                    known.add(vid)
                    obs0 = torch.tensor([v_d[vid], gap_d[vid], vlead_d[vid]],
                                        dtype=torch.float32)
                    obs_buffers[vid] = deque([obs0] * SEQ_LEN, maxlen=SEQ_LEN)

            for vid in all_vids:
                obs = torch.tensor([v_d[vid], gap_d[vid], vlead_d[vid]],
                                   dtype=torch.float32)
                obs_buffers[vid].append(obs)

            # ── evict stuck vehicles ─────────────────────────────────────
            for vid in all_vids:
                stuck_steps[vid] = stuck_steps.get(vid, 0) + 1 \
                                   if v_d[vid] < 0.5 else 0
            for vid in [v for v, n in stuck_steps.items() if n > STUCK_LIMIT]:
                try:
                    traci.vehicle.remove(vid)
                except Exception:
                    pass
                obs_buffers.pop(vid, None)
                stuck_steps.pop(vid, None)
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
            # All vehicles in a conflicting stream
            candidates = [vid for vid, mvmt in snap.vehicle_stream.items()
                          if mvmt is not None and CONFLICT_MAP.get(mvmt)]

            # turning flag for every vehicle — used in _apply_speeds and _project
            is_turning_all = {
                vid: (1.0 if STREAM_NAMES.get(snap.vehicle_stream.get(vid), "").endswith(("_R", "_L"))
                      else 0.0)
                for vid in all_vids
            }

            if not candidates:
                _apply_speeds(all_vids, hybrid, v_d, gap_d, vlead_d,
                              x_seq_dict, {}, dt, is_turning_d=is_turning_all)
                continue

            # ── uncorrected projection → tokens (CPU, no grad needed) ───
            with torch.no_grad():
                ACTIVE_ETA = HORIZON * PROJ_DT   # 10 s projection window
                d_raw_all  = _query_d_to_junction(candidates)
                active_vids: set = set()
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
                                active_vids.add(_vid)
                                active_vids.add(_rvid)
                tracked = [v for v in candidates if v in active_vids]

                if not tracked:
                    _apply_speeds(all_vids, hybrid, v_d, gap_d, vlead_d,
                                  x_seq_dict, {}, dt, is_turning_d=is_turning_all)
                    continue

                v0_cpu   = torch.tensor([v_d[i]     for i in tracked], dtype=torch.float32)
                gap0_cpu = torch.tensor([gap_d[i]   for i in tracked], dtype=torch.float32)
                vl0_cpu  = torch.tensor([vlead_d[i] for i in tracked], dtype=torch.float32)
                # always provide an x_seq — zero-pad missing vehicles so
                # HybridModel (which always needs x_seq) never gets None
                if all(i in x_seq_dict for i in tracked):
                    xs0_cpu = torch.stack([x_seq_dict[i] for i in tracked])
                else:
                    xs0_cpu = torch.stack([
                        x_seq_dict[i] if i in x_seq_dict
                        else torch.zeros(SEQ_LEN, 3)
                        for i in tracked
                    ])

                idx_of     = {vid: k for k, vid in enumerate(tracked)}
                d_junc_cpu = torch.tensor(
                    [d_raw_all[i] for i in tracked], dtype=torch.float32)

                leader_idx_cpu = torch.full((len(tracked),), -1, dtype=torch.long)
                for i, vid in enumerate(tracked):
                    try:
                        info = traci.vehicle.getLeader(vid)
                    except traci.exceptions.TraCIException:
                        info = None
                    if info and info[0] in idx_of:
                        leader_idx_cpu[i] = idx_of[info[0]]

                # leader's distance to junction — ARM_LEN (far) if no tracked leader
                d_junc_lead_cpu = torch.full(
                    (len(tracked),), _st_module.ARM_LEN, dtype=torch.float32)
                for i in range(len(tracked)):
                    li = leader_idx_cpu[i].item()
                    if li >= 0:
                        d_junc_lead_cpu[i] = d_junc_cpu[li]

                # is_turning for tracked vehicles
                is_turning_tracked = torch.tensor([
                    1.0 if STREAM_NAMES.get(snap.vehicle_stream.get(vid), "").endswith(("_R", "_L"))
                    else 0.0
                    for vid in tracked
                ], dtype=torch.float32, device=DEVICE)

                # baseline projection — all tensors on DEVICE, xs0 always provided
                v_traj0, cum_dist0 = _project(
                    hybrid,
                    v0_cpu.to(DEVICE),
                    gap0_cpu.to(DEVICE),
                    vl0_cpu.to(DEVICE),
                    xs0_cpu.to(DEVICE),
                    leader_idx_cpu.to(DEVICE),
                    PROJ_DT, HORIZON,
                    is_turning=is_turning_tracked)

                dj_gpu = d_junc_cpu.to(DEVICE)

                social_force_gpu = compute_social_force(
                    snap, dj_gpu, v0_cpu.to(DEVICE), idx_of, tracked)
                sa_np = {vid: social_force_gpu[i].item() for i, vid in enumerate(tracked)}

            # ── iterative RKHS-style correction (mirrors safety.py n_steps) ──
            # Each iteration: build tokens from current projected trajectory,
            # run transformer for one step, accumulate, re-project.
            # _project_diff is a closed-form linear shift — no loop, no hybrid calls.
            # ── h_≤ : residual RKHS correction via transformer ───────────
            # Each iteration: transformer sees accumulated correction so far
            # (prev_da) as an explicit token feature and outputs a residual.
            # Running sum is soft-capped via softsign (smooth, nonzero gradient
            # everywhere) before each projection — analogous to the membership
            # function gate, not torch.clamp.
            stored_steps = []
            delta_a_n    = torch.zeros(len(tracked), device=DEVICE)
            v_traj_d     = v_traj0
            cum_dist_d   = cum_dist0

            for _rkhs_iter in range(N_RKHS_ITER):
                proj_iter = ProjectionInfo(
                    v_traj=v_traj_d.detach(), cum_dist=cum_dist_d.detach(),
                    d_junc=dj_gpu, idx_of=idx_of, tracked=tracked)
                # pass accumulated correction as context — transformer learns
                # to output near-zero residual when budget is consumed
                tok, _ = build_tokens(
                    proj_iter, snap,
                    x_seq_dict = x_seq_dict,
                    prev_da    = delta_a_n.detach(),
                    gap        = gap0_cpu.to(DEVICE),
                    v_lead     = vl0_cpu.to(DEVICE),
                )
                raw_out = transformer(tok.unsqueeze(0).to(DEVICE)).squeeze(0)
                # STE: clamp bounds the forward value; gradient flows through as if unclipped
                raw_c = raw_out.clamp(-IDM_CAP, IDM_CAP)
                stored_steps.append(raw_c.detach() + raw_out - raw_out.detach())
                raw_sum   = torch.stack(stored_steps).sum(0)
                raw_sum_c = raw_sum.clamp(-TOTAL_CAP, TOTAL_CAP)
                delta_a_n = raw_sum_c.detach() + raw_sum - raw_sum.detach()
                v_traj_d, cum_dist_d = _project_diff(
                    v_traj0, cum_dist0, delta_a_n, PROJ_DT, HORIZON,
                )

            # ── h_≤ · μ(v) : speed-membership gate ───────────────────────
            # μ_stopped = 1 when v ≤ V_GATE_LOW, 0 when v ≥ V_GATE_HIGH.
            # Anchor = 0.  Final correction:  h_≤ + μ_stopped·(0 − h_≤)
            mu_stopped = torch.clamp(
                (V_GATE_HIGH - v0_cpu.to(DEVICE)) / (V_GATE_HIGH - V_GATE_LOW),
                min=0.0, max=1.0,
            )  # [N]
            delta_a_n  = delta_a_n + mu_stopped * (0.0 - delta_a_n)
            v_traj_d, cum_dist_d = _project_diff(
                v_traj0, cum_dist0, delta_a_n, PROJ_DT, HORIZON,
            )

            # ── [DIAG] batch table — all front-of-lane vehicles at once ──
            with torch.no_grad():
                _sim_t = (ctrl_step + 1) * dt + WARMUP_SEC
                print(f"    ┌─ t={_sim_t:.1f}s  {len(tracked)} front-of-lane vehicles (batch)")
                for _i, _vid in enumerate(tracked):
                    _sname = STREAM_NAMES.get(snap.vehicle_stream.get(_vid), "----")
                    print(
                        f"    │  {_vid:>20s}  {_sname:<5s}"
                        f"  v={v0_cpu[_i].item():5.2f}"
                        f"  μ={mu_stopped[_i].item():.2f}"
                        f"  δa={delta_a_n[_i].item():+.4f}",
                        flush=True,
                    )
                print("    └" + "─" * 55, flush=True)
            # ─────────────────────────────────────────────────────────────

            # ── safety violation + one-sided deceleration penalty ─────────
            V       = compute_violation_diff(
                snap, v_traj_d, cum_dist_d,
                dj_gpu, idx_of, tracked, PROJ_DT,
            )
            L_decel = LAMBDA_DECEL * torch.relu(-delta_a_n).pow(2).mean()

            # ── throughput — GRU only ─────────────────────────────────────
            # accel_base gradient flows to the GRU; delta_a_n is detached so
            # the transformer receives no throughput pull (it would bias all
            # corrections positive, fighting the safety signal).
            accel_base = hybrid(
                v0_cpu.to(DEVICE), gap0_cpu.to(DEVICE),
                vl0_cpu.to(DEVICE), xs0_cpu.to(DEVICE),
                is_turning_tracked,
            )
            v0_d  = v0_cpu.to(DEVICE).detach()
            T_gru = (v0_d + (accel_base + delta_a_n.detach()) * dt).clamp(min=0.0).sum() * dt

            loss_step = V + L_decel

            # ── track worst intersection TTC in this window ───────────────
            # Uses current speeds + distances to junction — no projection.
            step_min_ttc = compute_intersection_min_ttc(
                snap, dj_gpu, v0_cpu.to(DEVICE), idx_of, tracked)
            if step_min_ttc < disp_worst_ttc:
                disp_worst_ttc = step_min_ttc

            if loss_step.grad_fn is not None:
                (loss_step / GRAD_ACCUM).backward()
                pending_grad = True
            accum_safety_val += loss_step.item()
            ep_throughput = ep_throughput + T_gru   # full-episode accumulator for GRU
            disp_V       += V.item()
            disp_T       += T_gru.item()
            if V_window_start is None:
                V_window_start = V.item()   # first step of this window
            V_window_end = V.item()         # always track the last step
            accum_count  += 1

            # ── apply correction to SUMO (detached scalar per vehicle) ───
            da_np = {vid: delta_a_n[i].item() for i, vid in enumerate(tracked)}
            _apply_speeds(all_vids, hybrid, v_d, gap_d, vlead_d,
                          x_seq_dict, da_np, dt,
                          is_turning_d=is_turning_all, social_a_np=sa_np)

            # ── periodic weight save every 5 simulation seconds ──────────
            if ctrl_step > 0 and ctrl_step % SAVE_EVERY_STEPS == 0:
                torch.save(transformer.state_dict(), XFMR_LATEST)
                torch.save(hybrid.state_dict(),      HYBRID_LATEST)
                _t = (ctrl_step + 1) * dt + WARMUP_SEC
                print(f"    [save] t={_t:.1f}s → {XFMR_LATEST.name}", flush=True)

            # ── transformer gradient step every GRAD_ACCUM real steps ───
            if accum_count >= GRAD_ACCUM:
                if pending_grad:
                    gn_xfmr = torch.nn.utils.clip_grad_norm_(
                        transformer.parameters(), max_norm=1.0)
                    optimizer_xfmr.step()
                else:
                    gn_xfmr = torch.tensor(0.0)
                optimizer_xfmr.zero_grad()

                sim_t = (ctrl_step + 1) * dt + WARMUP_SEC
                n_veh = len(tracked)
                ttc_str = f"{disp_worst_ttc:.2f}s" if disp_worst_ttc < float("inf") else "  n/a"
                v_start = V_window_start if V_window_start is not None else 0.0
                v_arrow = "↓" if V_window_end < v_start else "↑"
                print(f"    t={sim_t:5.1f}s | veh={n_veh:3d} | "
                      f"V={disp_V/accum_count:8.4f} | "
                      f"V_window={v_start:.2f}{v_arrow}{V_window_end:.2f} | "
                      f"T_gru={disp_T/accum_count:7.4f} | "
                      f"wTTC={ttc_str} | "
                      f"|gX|={float(gn_xfmr):.3f}"
                      f"{'*' if float(gn_xfmr)>1.0 else ''}",
                      flush=True)

                total_safety   += accum_safety_val / accum_count
                n_opt_steps    += 1
                accum_safety_val = 0.0
                pending_grad     = False
                accum_count     = 0
                disp_V = disp_T = 0.0
                disp_worst_ttc  = float("inf")
                V_window_start  = None

        # ── flush remaining transformer steps ────────────────────────────
        if accum_count > 0:
            if pending_grad:
                gn_xfmr = torch.nn.utils.clip_grad_norm_(
                    transformer.parameters(), max_norm=5.0)
                optimizer_xfmr.step()
                print(f"    [flush] V={accum_safety_val/accum_count:8.4f} | "
                      f"|gX|={float(gn_xfmr):.3f}"
                      f"{'*' if float(gn_xfmr)>1.0 else ''}",
                      flush=True)
                total_safety += accum_safety_val / accum_count
                n_opt_steps  += 1
            optimizer_xfmr.zero_grad()

        # ── end-of-episode GRU update (full-episode throughput signal) ───
        if ep_throughput.grad_fn is not None:
            gru_loss = -LAMBDA_THROUGHPUT * ep_throughput
            gru_loss.backward()
            gn_hyb = torch.nn.utils.clip_grad_norm_(
                [p for p in hybrid.parameters() if p.requires_grad],
                max_norm=HYBRID_GRAD_CLIP)
            optimizer_hybrid.step()
            print(f"    [GRU ep-update] T={ep_throughput.item():.1f}m | "
                  f"|gH|={float(gn_hyb):.4f}"
                  f"{'*' if float(gn_hyb)>HYBRID_GRAD_CLIP else ''}",
                  flush=True)
        optimizer_hybrid.zero_grad()

    finally:
        traci.close()
        clear_route_cache()
        traci_cache.clear()

    return total_safety, ep_throughput.item(), n_opt_steps


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SafetyTransformer online training")
    parser.add_argument("--gui", action="store_true",
                        help="Open SUMO-GUI for visualisation (slower)")
    parser.add_argument("--delay", type=int, default=10,
                        help="GUI step delay in ms (default 10 = 10× real-time, ignored without --gui)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    XFMR_CKPT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  SAFETY TRANSFORMER — online training via SUMO co-sim")
    print("=" * 60)
    print(f"  Device        : {DEVICE}")
    n_sims = 1 if args.gui else N_PARALLEL
    print(f"  Parallel sims : {n_sims}{'  (GUI — single instance)' if args.gui else '  (gradient averaged over all)'}")
    print(f"  Episodes      : {N_EPISODES}")
    print(f"  Grad accum    : {GRAD_ACCUM} steps / optimizer step")
    print(f"  Checkpoint    : every {CHECKPOINT_EVERY} episodes")
    print(f"  LR            : {LR}")
    print(f"  Threshold c   : {THRESHOLD} s")
    print(f"  β (softplus)  : {BETA_0}")
    print(f"  IDM cap/iter  : ±{IDM_CAP} m/s²  ×{N_RKHS_ITER} iters (max ±{IDM_CAP*N_RKHS_ITER:.1f})")
    print(f"  λ_reg         : {LAMBDA_REG}  (L2 penalty on δa)")
    print(f"  λ_decel       : {LAMBDA_DECEL}  (one-sided penalty on negative δa — braking only)")
    print(f"  λ_throughput  : {LAMBDA_THROUGHPUT}  (combined loss = V - λ·T)")
    print(f"  Hybrid LR     : {HYBRID_LR}  (GRU + head only; physics frozen)")
    print(f"  Teacher loss  : MAE (L1)  — targets near zero")
    print("-" * 60)

    # ── load models (best available for each) ───────────────────────────
    hybrid      = _load_hybrid()
    transformer = _load_transformer()

    optimizer_xfmr   = AdamW(transformer.parameters(), lr=LR,
                             weight_decay=WEIGHT_DECAY)
    optimizer_hybrid = torch.optim.Adam(
        [p for p in hybrid.parameters() if p.requires_grad],
        lr=HYBRID_LR,
    )

    start_ep      = 1
    best_combined = float("inf")   # combined = mean_safety - λ * mean_throughput

    if XFMR_STATE.exists():
        state        = torch.load(XFMR_STATE, weights_only=False)
        start_ep     = state.get("next_episode", 1)
        best_combined = state.get("best_combined", float("inf"))
        try:
            if "optimizer_xfmr" in state:
                optimizer_xfmr.load_state_dict(state["optimizer_xfmr"])
            elif "optimizer" in state:
                optimizer_xfmr.load_state_dict(state["optimizer"])
            # Validate shapes — load_state_dict silently copies wrong-shaped tensors
            for p, s in zip(transformer.parameters(), optimizer_xfmr.state.values()):
                if "exp_avg" in s and s["exp_avg"].shape != p.shape:
                    raise ValueError("shape mismatch")
        except (RuntimeError, ValueError):
            optimizer_xfmr.state.clear()
            print("  [warn] optimizer_xfmr state incompatible — reset to fresh")
        try:
            if "optimizer_hybrid" in state:
                optimizer_hybrid.load_state_dict(state["optimizer_hybrid"])
            for p, s in zip(hybrid.parameters(), optimizer_hybrid.state.values()):
                if "exp_avg" in s and s["exp_avg"].shape != p.shape:
                    raise ValueError("shape mismatch")
        except (RuntimeError, ValueError):
            optimizer_hybrid.state.clear()
            print("  [warn] optimizer_hybrid state incompatible — reset to fresh")
        print(f"  Resuming      : episode {start_ep}  "
              f"best_combined={best_combined:.5f}")
    print("=" * 58)

    print(f"\n{'ep':>5}  {'V_safety':>10}  {'T_reward':>10}  "
          f"{'combined':>10}  {'steps':>6}  {'saved':>14}")
    print("-" * 62)

    for ep in range(start_ep, start_ep + N_EPISODES):
        print(f"\n── Episode {ep}/{start_ep + N_EPISODES - 1} "
              f"────────────────────────────────────────")
        ep_safety, ep_throughput, n_steps = train_parallel(
            transformer, hybrid, optimizer_xfmr, optimizer_hybrid,
            n_parallel=1 if args.gui else N_PARALLEL,
            gui=args.gui, gui_delay=args.delay, ep=ep)
        mean_safety     = ep_safety / max(n_steps, 1)
        mean_combined   = mean_safety - LAMBDA_THROUGHPUT * ep_throughput

        # ── always save latest ───────────────────────────────────────────
        torch.save(transformer.state_dict(), XFMR_LATEST)
        torch.save(hybrid.state_dict(),      HYBRID_LATEST)
        tag = "latest"

        # ── save best (by combined objective) ────────────────────────────
        if mean_combined < best_combined:
            best_combined = mean_combined
            torch.save(transformer.state_dict(), XFMR_BEST)
            torch.save(hybrid.state_dict(),      HYBRID_BEST)
            tag = "latest+best"

        # ── periodic numbered checkpoint ─────────────────────────────────
        if ep % CHECKPOINT_EVERY == 0:
            ckpt = XFMR_CKPT_DIR / f"ep_{ep:04d}.pt"
            torch.save(transformer.state_dict(), ckpt)
            tag += f"+ckpt({ep})"

        # ── persist training state for resume ────────────────────────────
        torch.save({
            "next_episode":    ep + 1,
            "best_combined":   best_combined,
            "optimizer_xfmr":  optimizer_xfmr.state_dict(),
            "optimizer_hybrid": optimizer_hybrid.state_dict(),
        }, XFMR_STATE)

        print(f"{ep:5d}  {mean_safety:10.5f}  {ep_throughput:10.1f}  "
              f"{mean_combined:10.5f}  {n_steps:6d}  {tag}")

    print(f"\n  Best combined : {best_combined:.5f}")
    print(f"  Transformer   → {XFMR_BEST}")
    print(f"  HybridModel   → {HYBRID_BEST}")
    print(f"  Checkpoints   → {XFMR_CKPT_DIR}/")


if __name__ == "__main__":
    main()
