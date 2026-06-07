"""
intersection_env.py — Differentiable 4-way intersection environment (pure PyTorch).

Replaces SUMO co-simulation with a 1D arc-length model per vehicle:

  arc ∈ [0, ARM_LENGTH)                  approaching arm  → IDM + social force
  arc ∈ [ARM_LENGTH, ARM_LENGTH+JCT_LEN) in junction      → maintain speed
  arc ∈ [ARM_LENGTH+JCT_LEN, DEPART_ARC] exit arm         → free-flow
  arc > DEPART_ARC                        departed         → alive=False

All physics run on PyTorch tensors so the full 60-second episode is one
computational graph.  Truncated BPTT (detach every W steps) is handled by
the caller (train_hybrid.py).

N_eps episodes run in parallel on a single GPU — batch dimension = episode.
All N_eps share the same vehicle schedule (same stream assignments, same spawn
times) but receive small Gaussian noise on initial speed, giving diverse
trajectories and lower-variance gradient estimates.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F

from social_force import (
    _ALL_MOVEMENTS, SAFE_GAP, A_CROSS, L_GAP, BETA,
    EPSILON_P, YIELD_CONF_SCALE, P_SCALE, N_SCALE, _V_REF,
)
from conflict import CONFLICT_MAP
from model import D_EGO, D_SUM, D_RIVAL, K_MAX, V_TURN_LOW

# ── consistent stream ordering (must match schedule_collector.py) ──────────────
STREAM_LIST = sorted(_ALL_MOVEMENTS)
STREAM_IDX  = {s: i for i, s in enumerate(STREAM_LIST)}
N_STREAMS   = len(STREAM_LIST)

# ── turning-stream mask [N_STREAMS] — streams that need speed capping ──────────
_TURNING_SET = frozenset({
    ("east_in", "north_out"), ("east_in", "south_out"),
    ("west_in", "south_out"), ("west_in", "north_out"),
    ("north_in", "west_out"), ("north_in", "east_out"),
    ("south_in", "east_out"), ("south_in", "west_out"),
})

# ── physics constants — must match demo_hybrid.py exactly ─────────────────────
DT          = 0.2    # s
V_MAX       = 13.89  # m/s
V_APPROACH  = 8.0    # m/s — target speed while approaching junction with rival
B_COMFORT   = 3.0    # m/s²
K_D         = 1.2    # proportional waypoint gain (2*ZETA*OMEGA_N = 2*1.2*0.5)
IDM_ACCEL   = 2.6    # m/s²
IDM_BRAKE   = 4.5    # m/s²
IDM_S0      = 2.0    # m
IDM_T       = 1.5    # s
IDM_DELTA   = 4.0
VEH_LEN     = 5.0    # m
ARM_LENGTH  = 200.0  # m
JCT_LEN     = 20.0   # m — approximate junction traversal distance
DEPART_ARC  = 2 * ARM_LENGTH + JCT_LEN   # 420 m — vehicle removed after this
D_APPROACH  = 30.0   # m — mu_wp approach gate width (reduced from 80 m so NN has authority farther out)
GAP_MAX     = 100.0  # m — normalisation cap for gap ego feature
SEQ_LEN     = 10     # ego history length
ETA_RIVAL_TH = 10.0  # s — has_rival flag threshold
EPS         = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Precomputed conflict structures (built once, then moved to GPU)
# ─────────────────────────────────────────────────────────────────────────────

def _build_conflict_tensors(device: torch.device):
    """
    conflict_mat [N_STREAMS, N_STREAMS] bool  — True when streams conflict.
    rival_idx   [N_STREAMS, K_MAX] long       — conflicting stream indices
                                                 per stream; -1 = empty slot.
    """
    mat = torch.zeros(N_STREAMS, N_STREAMS, dtype=torch.bool)
    for s, rivals in CONFLICT_MAP.items():
        if s not in STREAM_IDX:
            continue
        si = STREAM_IDX[s]
        for r in rivals:
            if r in STREAM_IDX:
                mat[si, STREAM_IDX[r]] = True

    rival_idx = torch.full((N_STREAMS, K_MAX), -1, dtype=torch.long)
    for si in range(N_STREAMS):
        rivals = [sj for sj in range(N_STREAMS) if mat[si, sj]]
        for k, sj in enumerate(rivals[:K_MAX]):
            rival_idx[si, k] = sj

    turning_mask = torch.tensor(
        [1.0 if STREAM_LIST[k] in _TURNING_SET else 0.0
         for k in range(N_STREAMS)]
    )

    return mat.to(device), rival_idx.to(device), turning_mask.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable physics helpers (batched, operate on [E, N] tensors)
# ─────────────────────────────────────────────────────────────────────────────

def _idm(v: torch.Tensor, gap: torch.Tensor, v_lead: torch.Tensor) -> torch.Tensor:
    """IDM acceleration [E, N] — fully differentiable."""
    dv      = (v - v_lead).clamp(min=0.0)
    s_star  = IDM_S0 + v * IDM_T + v * dv / (2.0 * math.sqrt(IDM_ACCEL * IDM_BRAKE))
    s_star  = s_star.clamp(min=IDM_S0)
    a       = IDM_ACCEL * (1.0 - (v / V_MAX).pow(IDM_DELTA) - (s_star / gap.clamp(min=0.1)).pow(2))
    return a.clamp(-IDM_BRAKE, IDM_ACCEL)


def _waypoint(v: torch.Tensor, arc: torch.Tensor, has_rival: torch.Tensor,
              is_turning: torch.Tensor) -> torch.Tensor:
    """
    Waypoint acceleration [E, N]:
      - approaching arm, has rival → target V_APPROACH (kinematic cap)
      - otherwise                  → target V_MAX
      - turning vehicles           → additional braking toward V_TURN_LOW
    Differentiable through v and arc.
    """
    approaching = (arc < ARM_LENGTH)
    d           = (ARM_LENGTH - arc).clamp(min=0.0)
    v_kin       = (V_APPROACH ** 2 + 2.0 * B_COMFORT * d).sqrt()
    v_target    = torch.where(approaching & has_rival, v_kin.clamp(max=V_MAX),
                              torch.full_like(v, V_MAX))
    a_wp        = (K_D * (v_target - v)).clamp(-B_COMFORT, IDM_ACCEL)

    # Turning deceleration bias (blended in above V_TURN_LOW, same as demo_hybrid.py)
    above_turn  = (v > V_TURN_LOW).float()
    mu_t        = ((v - V_TURN_LOW) / (V_TURN_HIGH - V_TURN_LOW)).clamp(0.0, 1.0)
    u_t         = (IDM_ACCEL * (1.0 - (v / max(V_TURN_LOW, 0.1)).pow(IDM_DELTA))).clamp(-B_COMFORT, 0.0)
    turn_bias   = is_turning * mu_t * u_t * above_turn
    return a_wp + turn_bias

V_TURN_HIGH = 11.0   # m/s (mirrors model.py constant)


# ─────────────────────────────────────────────────────────────────────────────
# IntersectionEnv
# ─────────────────────────────────────────────────────────────────────────────

class IntersectionEnv:
    """
    Pure-PyTorch differentiable intersection environment.

    Parameters
    ----------
    schedule   : list[VehicleEntry] from schedule_collector.collect_schedule()
    n_eps      : number of parallel episodes (batch dimension)
    device     : torch device ('cuda' or 'cpu')
    v0_noise   : std of Gaussian speed noise added per vehicle per episode
    """

    def __init__(
        self,
        schedule,           # list[VehicleEntry]
        n_eps:    int   = 8,
        device:   str   = "cuda",
        v0_noise: float = 0.5,
    ):
        self.n_eps    = n_eps
        self.device   = torch.device(device)
        self.v0_noise = v0_noise

        # ── static schedule tensors ────────────────────────────────────────
        self.n_veh       = len(schedule)
        spawn_steps      = [e.spawn_step  for e in schedule]
        stream_ids_list  = [e.stream_idx  for e in schedule]
        v0s_list         = [e.v0          for e in schedule]
        arc0s_list       = [e.arc0        for e in schedule]

        self.spawn_steps = spawn_steps                     # plain list[int]
        self.stream_ids  = torch.tensor(stream_ids_list,
                                        dtype=torch.long,
                                        device=self.device)  # [N]
        self.base_v0s    = torch.tensor(v0s_list,
                                        dtype=torch.float32,
                                        device=self.device)  # [N]
        self.arc0s       = torch.tensor(arc0s_list,
                                        dtype=torch.float32,
                                        device=self.device)  # [N]

        # ── precomputed conflict structures ────────────────────────────────
        self.conflict_mat, self.rival_idx, self.turning_mask = \
            _build_conflict_tensors(self.device)

        # Per-vehicle: is this stream a turning movement?
        self.is_turning = self.turning_mask[self.stream_ids]   # [N]

        # Pairwise conflict mask [N, N] — stream_ids[i] conflicts with stream_ids[j]
        self.c_ij = self.conflict_mat[
            self.stream_ids.unsqueeze(1),   # [N, 1]
            self.stream_ids.unsqueeze(0),   # [1, N]
        ]  # [N, N]

        # Same-stream mask [N, N]
        self.same_stream = (
            self.stream_ids.unsqueeze(1) == self.stream_ids.unsqueeze(0)
        )  # [N, N]

        # Per-vehicle rival slot indices [N, K_MAX]
        self.per_veh_rival_idx = self.rival_idx[self.stream_ids]   # [N, K_MAX]
        self.valid_rival_slot  = (self.per_veh_rival_idx >= 0)     # [N, K_MAX] bool
        self.rival_idx_clamped = self.per_veh_rival_idx.clamp(min=0)

        # ── dynamic state (initialised in reset()) ─────────────────────────
        self.v_t:    torch.Tensor | None = None   # [E, N]
        self.arc_t:  torch.Tensor | None = None   # [E, N]
        self.alive:  torch.Tensor | None = None   # [E, N] bool
        self.hist:   torch.Tensor | None = None   # [E, N, SEQ_LEN, D_EGO]
        self.step_idx = 0

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        """Initialise/reinitialise all dynamic state."""
        E, N = self.n_eps, self.n_veh
        d    = self.device

        # Speed noise sampled fresh each episode (different per episode, fixed per vehicle)
        noise = torch.randn(E, N, device=d) * self.v0_noise

        self.v_t   = torch.zeros(E, N, device=d)
        self.arc_t = torch.zeros(E, N, device=d)
        self.alive = torch.zeros(E, N, dtype=torch.bool, device=d)
        self.hist  = torch.zeros(E, N, SEQ_LEN, D_EGO, device=d)
        self._v0_noisy    = (self.base_v0s.unsqueeze(0) + noise).clamp(min=0.5, max=V_MAX)
        self.step_idx     = 0
        self.n_collisions = 0   # vehicle-step gap overlaps + junction conflicts

        # Pre-activate vehicles with spawn_step=0 (from warmup snapshot)
        for idx, step in enumerate(self.spawn_steps):
            if step == 0:
                self.alive[:, idx] = True
                self.v_t[:, idx]   = self._v0_noisy[:, idx]
                self.arc_t[:, idx] = self.arc0s[idx]

    # ── detach ────────────────────────────────────────────────────────────────

    def detach(self):
        """Sever the computational graph (call after each backward()). """
        self.v_t   = self.v_t.detach()
        self.arc_t = self.arc_t.detach()
        self.hist  = self.hist.detach()

    # ── physics ───────────────────────────────────────────────────────────────

    def _compute_physics(self):
        """
        Compute all physics inputs for the model forward pass.

        Returns (all [E, N] unless noted):
            gap, v_lead, u_wp, mu_wp, mu_cf
            own_summary  [E, N, D_SUM]
            rival_tokens [E, N, K_MAX, D_RIVAL]
        """
        E, N = self.n_eps, self.n_veh
        d    = self.device
        v    = self.v_t      # [E, N]
        arc  = self.arc_t    # [E, N]
        alive = self.alive   # [E, N] bool

        in_jct   = (arc >= ARM_LENGTH) & (arc < ARM_LENGTH + JCT_LEN)  # [E, N]
        on_exit  = (arc >= ARM_LENGTH + JCT_LEN) & alive                # [E, N]
        approaching = alive & ~in_jct & ~on_exit                        # [E, N]

        # ── pairwise arc differences [E, N, N] ────────────────────────────
        arc_i = arc.unsqueeze(2)   # [E, N, 1]  — ego
        arc_j = arc.unsqueeze(1)   # [E, 1, N]  — others
        diff_ji = arc_j - arc_i    # [E, N, N]: positive → j is ahead of i
        diff_ij = arc_i - arc_j    # [E, N, N]: positive → j is behind i

        same = self.same_stream.unsqueeze(0)          # [1, N, N]
        alive_j = alive.float().unsqueeze(1)           # [E, 1, N]
        alive_i = alive.float().unsqueeze(2)           # [E, N, 1]

        # ── leader (nearest same-stream vehicle ahead) ─────────────────────
        valid_lead = same & (diff_ji > 0) & alive_j.bool() & alive_i.bool()
        net_gap_ji = (diff_ji - VEH_LEN)               # bumper-to-bumper gap
        gap_ji_masked = torch.where(valid_lead, net_gap_ji,
                                    torch.full_like(net_gap_ji, 1e6))
        gap_to_lead, lead_idx = gap_ji_masked.min(dim=2)   # [E, N]
        has_lead   = valid_lead.any(dim=2)                  # [E, N]
        gap        = torch.where(has_lead, gap_to_lead.clamp(min=0.0),
                                 torch.full_like(gap_to_lead, 1000.0))
        v_lead     = torch.where(has_lead,
                                 v.gather(1, lead_idx.clamp(0, N-1)),
                                 v)

        # ── ETA to junction centre ─────────────────────────────────────────
        d_jct = (ARM_LENGTH - arc).clamp(min=0.0)     # [E, N]
        eta   = d_jct / (v.clamp(min=0.1))            # [E, N]
        # in_jct and exit vehicles treated as "already at crossing"
        eta   = torch.where(in_jct | on_exit, torch.zeros_like(eta), eta)

        # ── pairwise conflict force ────────────────────────────────────────
        # delta[e,i,j] = eta_i - eta_j: positive → i arrives later → i yields
        eta_i = eta.unsqueeze(2)     # [E, N, 1]
        eta_j = eta.unsqueeze(1)     # [E, 1, N]
        delta  = eta_i - eta_j       # [E, N, N]

        urgency = (1.0 - delta.abs() / SAFE_GAP).clamp(min=0.0)  # [E, N, N]
        yielder = (delta >= 0).float()

        # Platoon softening (matches social_force.py)
        # Compute stream pressure per stream: P_k = sum_j v_j * max(0, 1-gap_behind/L_GAP)
        # gap_behind[e,i] = distance to nearest follower behind on same stream
        valid_foll = same & (diff_ij > 0) & alive_j.bool() & approaching.float().unsqueeze(1).bool()
        gap_foll_masked = torch.where(valid_foll, (diff_ij - VEH_LEN).clamp(min=0.0),
                                      torch.full_like(diff_ij, 1e6))
        gap_behind, foll_idx = gap_foll_masked.min(dim=2)   # [E, N]
        has_foll    = valid_foll.any(dim=2)
        gap_behind  = torch.where(has_foll, gap_behind, torch.full_like(gap_behind, 1e6))
        v_foll      = torch.where(has_foll,
                                  v.gather(1, foll_idx.clamp(0, N-1)),
                                  v)

        pack_w   = (1.0 - gap_behind / L_GAP).clamp(min=0.0)   # [E, N]
        P_veh    = v * pack_w * approaching.float()              # [E, N]

        stream_ids_exp = self.stream_ids.unsqueeze(0).expand(E, N)   # [E, N]
        P_stream  = torch.zeros(E, N_STREAMS, device=d)
        P_stream.scatter_add_(1, stream_ids_exp, P_veh)

        n_stream  = torch.zeros(E, N_STREAMS, device=d)
        n_stream.scatter_add_(1, stream_ids_exp, approaching.float())

        v_sum_stream = torch.zeros(E, N_STREAMS, device=d)
        v_sum_stream.scatter_add_(1, stream_ids_exp, v * approaching.float())
        mv_stream = v_sum_stream / (n_stream + EPS)

        # Platoon ratio for yielder softening
        # p_ego[e,i] = P_stream[e, stream_ids[i]]
        p_ego   = P_stream.gather(1, stream_ids_exp)   # [E, N]
        # p_rival_ij = P_stream[e, stream_ids[j]] — broadcast over i
        p_rival = P_stream[:, self.stream_ids]          # [E, N] indexed by j-stream
        # We need [E, N_ego, N_rival] — broadcast p_ego along rival axis
        p_ego_grid   = p_ego.unsqueeze(2)               # [E, N, 1]
        p_rival_grid = p_rival.unsqueeze(1)              # [E, 1, N]

        yc = (delta / YIELD_CONF_SCALE).clamp(0.0, 1.0)   # yield_confidence [E, N, N]
        ratio = (p_ego_grid / (p_ego_grid + p_rival_grid + EPSILON_P)).clamp(0.0, 1.0)
        mu_softened = urgency * yielder * (1.0 - BETA * ratio * yc)

        # Apply conflict mask, alive mask, ego-not-in-jct mask
        not_in_jct_i = (~in_jct).float().unsqueeze(2)     # [E, N, 1]
        mu_pair = (mu_softened
                   * self.c_ij.float().unsqueeze(0)        # [1, N, N] conflict
                   * alive_j                               # [E, 1, N] rival alive
                   * alive_i                               # [E, N, 1] ego alive
                   * not_in_jct_i)                        # [E, N, 1] ego not in jct

        # Worst conflict gate per ego — capped at 0.7 so NN retains ≥30% authority
        mu_cf, _ = mu_pair.max(dim=2)                     # [E, N]
        mu_cf    = mu_cf.clamp(max=0.7)

        # Per-stream worst mu (for rival token slot 3)
        mu_per_stream = torch.zeros(E, N, N_STREAMS, device=d)
        for k in range(N_STREAMS):
            mask_k = (self.stream_ids == k).float()        # [N]
            mu_k = (mu_pair * mask_k.unsqueeze(0).unsqueeze(0)).max(dim=2).values
            mu_per_stream[:, :, k] = mu_k                 # [E, N]

        # ── has_rival flag [E, N]: any rival ETA < threshold ─────────────
        rival_present = (
            (eta_j < ETA_RIVAL_TH)                        # [E, 1, N]
            & self.c_ij.unsqueeze(0).bool()               # [1, N, N]
            & alive_j.bool()                              # [E, 1, N]
        )
        has_rival = rival_present.any(dim=2)              # [E, N]

        # ── waypoint + IDM → u_wp ─────────────────────────────────────────
        is_turning_exp = self.is_turning.unsqueeze(0).expand(E, N)  # [E, N]
        a_wp  = _waypoint(v, arc, has_rival, is_turning_exp)
        a_idm = _idm(v, gap, v_lead)
        # In junction / exit: just free-flow; IDM and waypoint only on approach arm
        on_approach = approaching.float()
        u_wp = (torch.min(a_idm, a_wp) * on_approach
                + v.new_full((1,), IDM_ACCEL) * (1.0 - (v / V_MAX).pow(IDM_DELTA)).clamp(-2, 2)
                * (1.0 - on_approach))

        # ── mu_wp (matches _mu_wp in demo_hybrid.py) ──────────────────────
        ttc     = gap / (v - v_lead + EPS).abs().clamp(min=EPS)
        mu_dec  = (2.0 * (3.0 - ttc)).clamp(0.0, 1.0)
        # Approach gate: 30 m — physics handles the final ~4 s, NN has authority farther out.
        mu_app  = ((1.0 - d_jct / D_APPROACH) * on_approach).clamp(0.0, 1.0)
        # Cap both gates at 0.7: NN always contributes ≥ (1−0.7)² = 9% even at worst.
        mu_wp   = torch.max(mu_dec, mu_app).clamp(max=0.7)

        # ── own-stream summary token [E, N, D_SUM] ────────────────────────
        P_own  = P_stream.gather(1, stream_ids_exp)
        n_own  = n_stream.gather(1, stream_ids_exp)
        mv_own = mv_stream.gather(1, stream_ids_exp)
        own_summary = torch.stack([
            (P_own  / P_SCALE).clamp(0.0, 1.0),
            (n_own  / N_SCALE).clamp(0.0, 1.0),
            mv_own  / _V_REF,
            v_foll  / _V_REF,
        ], dim=2)                                          # [E, N, D_SUM]

        # ── rival tokens [E, N, K_MAX, D_RIVAL] ───────────────────────────
        # idx_exp [E, N, K_MAX]: conflicting stream index per rival slot
        idx_exp = self.rival_idx_clamped.unsqueeze(0).expand(E, N, K_MAX)

        P_rival  = P_stream.unsqueeze(1).expand(-1, N, -1).gather(2, idx_exp)
        n_rival  = n_stream.unsqueeze(1).expand(-1, N, -1).gather(2, idx_exp)
        mv_rival = mv_stream.unsqueeze(1).expand(-1, N, -1).gather(2, idx_exp)
        mu_rival = mu_per_stream.gather(2, idx_exp)

        valid = self.valid_rival_slot.unsqueeze(0).expand(E, N, K_MAX).float()
        rival_tokens = torch.stack([
            (P_rival  / P_SCALE).clamp(0.0, 1.0) * valid,
            (n_rival  / N_SCALE).clamp(0.0, 1.0) * valid,
            (mv_rival / _V_REF)                   * valid,
            mu_rival                               * valid,
        ], dim=3)                                          # [E, N, K_MAX, D_RIVAL]

        return gap, v_lead, u_wp, mu_wp, mu_cf, own_summary, rival_tokens, d_jct, in_jct

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Advance the simulation by one DT step.

        Gradient flow design
        --------------------
        Physics (IDM, social force, stream pressure) are computed inside
        torch.no_grad() — they are context signals, not differentiated.
        The only gradient path is:

            θ → f_hat → a_out → v_{t+1} → (v/V_MAX ego feature) → hist
              → f_hat_{t+1} → ...  (truncated over BPTT window W)

        This prevents the pairwise [E, N, N] attention matrices from
        accumulating in the graph over W steps and blowing up GPU memory.
        On an RTX 2050 (4 GB) W=150 then costs ~800 MB (model activations
        only), down from >5 GB with fully-differentiable physics.

        Spawning uses torch.where to update v_t without in-place assignment
        (which would corrupt the autograd graph).
        """
        E, N = self.n_eps, self.n_veh

        # ── 1. spawn (no in-place v_t mutation) ──────────────────────────
        spawn_mask   = torch.zeros(N, dtype=torch.bool, device=self.device)
        spawn_v_full = torch.zeros(E, N, device=self.device)
        for idx, sp in enumerate(self.spawn_steps):
            if sp == self.step_idx:
                spawn_mask[idx]      = True
                spawn_v_full[:, idx] = self._v0_noisy[:, idx]

        if spawn_mask.any():
            mask_exp = spawn_mask.unsqueeze(0).expand(E, N)
            # alive and arc are non-differentiable state — update with column mask
            with torch.no_grad():
                self.alive[:, spawn_mask] = True
                self.arc_t[:, spawn_mask] = self.arc0s[spawn_mask].unsqueeze(0).expand(E, -1)
            # v_t: torch.where keeps gradient chain for non-spawn slots
            self.v_t = torch.where(mask_exp, spawn_v_full, self.v_t)

        # ── 2. physics — no gradient needed (pure context) ────────────────
        with torch.no_grad():
            gap, v_lead, u_wp, mu_wp, mu_cf, own_sum, rival_tok, d_jct, in_jct = \
                self._compute_physics()

            # Collision detection (accumulated; read by caller each BPTT window)
            #
            # Type 1: same-stream bumper overlap — IDM gap went negative.
            gap_col = (gap < 0.0) & self.alive                         # [E, N]
            #
            # Type 2: junction conflict — vehicle i is IN the junction box while
            # a conflicting vehicle j has NOT yet cleared the exit arm.
            # We use a wider danger zone [ARM_LENGTH, ARM_LENGTH + JCT_LEN + VEH_LEN]
            # so a vehicle entering while the previous one is still clearing counts.
            danger_zone = (
                (self.arc_t >= ARM_LENGTH) &
                (self.arc_t <  ARM_LENGTH + JCT_LEN + VEH_LEN)  # ≈ 225 m
            ) & self.alive                                              # [E, N]
            dz_i    = danger_zone.unsqueeze(2)                         # [E, N, 1]
            dz_j    = danger_zone.unsqueeze(1)                         # [E, 1, N]
            alive_ij = self.alive.unsqueeze(2) & self.alive.unsqueeze(1)
            jct_col  = dz_i & dz_j & self.c_ij.unsqueeze(0) & alive_ij
            # each vehicle in a live junction conflict counts once
            jct_col_veh = jct_col.any(dim=2)                           # [E, N]
            self.n_collisions += (
                int(gap_col.sum().item()) + int(jct_col_veh.sum().item())
            )

        # ── 3. ego token — v_t IS in graph; physics context is detached ───
        new_ego = torch.stack([
            self.v_t / V_MAX,                         # grad flows through v_t
            (gap    / GAP_MAX ).clamp(0.0, 1.0),      # detached (from no_grad)
            (self.v_t - v_lead) / V_MAX,              # grad flows through v_t
            (d_jct  / ARM_LENGTH).clamp(0.0, 1.0),   # detached
        ], dim=2)                                      # [E, N, D_EGO]

        # Shift history and append (graph-safe cat, no in-place roll)
        self.hist = torch.cat([self.hist[:, :, 1:, :], new_ego.unsqueeze(2)], dim=2)

        # ── 4. model forward ──────────────────────────────────────────────
        EN    = E * N
        a_out = model(
            self.hist.view(EN, SEQ_LEN, D_EGO),
            own_sum.view(EN, D_SUM),
            rival_tok.view(EN, K_MAX, D_RIVAL),
            u_wp.view(EN),
            mu_wp.view(EN),
            mu_cf.view(EN),
        ).view(E, N)                                   # [E, N]

        # ── 5. integrate v_t (in graph) ───────────────────────────────────
        v_new    = (self.v_t + a_out * DT).clamp(0.0, V_MAX)
        self.v_t = v_new * self.alive.float()

        # ── 6. update arc, despawn (no_grad — arc not in gradient path) ───
        with torch.no_grad():
            self.arc_t = self.arc_t + self.v_t.detach() * DT
            self.alive = self.alive & (self.arc_t <= DEPART_ARC)

        # ── reward ────────────────────────────────────────────────────────
        alive_f    = self.alive.float()
        n_alive    = alive_f.sum(dim=1).clamp(min=1.0)
        mean_speed = (self.v_t * alive_f).sum(dim=1) / n_alive   # [E]

        self.step_idx += 1
        return mean_speed
