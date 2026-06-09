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
    _TURN_YIELD_MAP,
)
from conflict import CONFLICT_MAP
from model import D_EGO, D_SUM, D_RIVAL, K_MAX

# ── consistent stream ordering (must match schedule_collector.py) ──────────────
STREAM_LIST = sorted(_ALL_MOVEMENTS)
STREAM_IDX  = {s: i for i, s in enumerate(STREAM_LIST)}
N_STREAMS   = len(STREAM_LIST)

# ── turning-stream masks — split by type to match SUMO junction lane speeds ────
_LEFT_TURN_SET = frozenset({
    ("east_in",  "south_out"),   # E→S
    ("west_in",  "north_out"),   # W→N
    ("north_in", "east_out"),    # N→E
    ("south_in", "west_out"),    # S→W
})
_RIGHT_TURN_SET = frozenset({
    ("east_in",  "north_out"),   # E→N
    ("west_in",  "south_out"),   # W→S
    ("north_in", "west_out"),    # N→W
    ("south_in", "east_out"),    # S→E
})
_TURNING_SET = _LEFT_TURN_SET | _RIGHT_TURN_SET

# ── physics constants — must match demo_hybrid.py exactly ─────────────────────
DT          = 0.2    # s
V_MAX       = 13.89  # m/s — approach arm + through junction  (from intersection.net.xml)
V_LEFT_TURN = 7.33   # m/s — left-turn  junction lane speed   (from intersection.net.xml)
V_RIGHT_TURN= 9.26   # m/s — right-turn junction lane speed   (from intersection.net.xml)
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

# ── waiting-penalty constants ──────────────────────────────────────────────────
V_WAIT      = 1.0   # m/s — below this speed counts as "waiting"
WAIT_PAT    = 25    # steps of patience before penalty ramps in (25 × 0.2 s = 5 s)
WAIT_MAX_W  = 2.0   # weight cap — keeps gradient bounded regardless of wait duration

# ── safe-speed probe constants ────────────────────────────────────────────────
# Iterative Krauss-style probe: at each of len(PROBE_K) iterations we ask
# "is there an imminent conflict at my projected speed?" using a sigmoid gate.
# If safe (mu_sig → 1) we add PROBE_STEP; if crash (mu_sig → 0) we don't.
# The sigmoid sharpens each iteration so gradients stay alive early in training
# but the final gate approaches a hard safety check.
PROBE_STEP = 0.2           # m/s² tentative increment per iteration
PROBE_K    = (0.2, 0.5, 1.0)   # sigmoid steepness schedule (soft → sharp)

# ── waiting-pressure constants (gap-acceptance for stopped streams) ─────────────
# An individual vehicle stopped near the junction for WAIT_RAMP steps (20 s)
# accumulates enough pressure (mu_wait→1) to fully override its conflict gate.
# Each additional stopped vehicle in the same stream shortens the ramp by
# WAIT_BOOST fraction, so a platoon of N reaches full pressure WAIT_BOOST*(N-1)
# times faster than a solo vehicle.
WAIT_RAMP   = 100.0   # steps (20 s at DT=0.2) for single vehicle to reach full pressure
WAIT_BOOST  = 0.5     # each additional waiting stream-mate adds 50% to ramp speed


# ─────────────────────────────────────────────────────────────────────────────
# Precomputed conflict structures (built once, then moved to GPU)
# ─────────────────────────────────────────────────────────────────────────────

def _build_conflict_tensors(device: torch.device):
    """
    conflict_mat    [N_STREAMS, N_STREAMS] bool  — True when streams conflict.
    rival_idx       [N_STREAMS, K_MAX] long      — conflicting stream indices per stream; -1=empty.
    turning_mask    [N_STREAMS] float            — 1 for turning streams.
    turn_yield_mat  [N_STREAMS, N_STREAMS] float — 1 when stream i must always yield to stream j
                                                   (movement priority: turns yield to through).
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

    # Movement priority: turns always yield to designated through movements.
    # Mirrors the _TURN_YIELD_MAP rule in social_force.py.
    turn_yield_mat = torch.zeros(N_STREAMS, N_STREAMS, dtype=torch.float32)
    for ego_stream, rival_streams in _TURN_YIELD_MAP.items():
        if ego_stream not in STREAM_IDX:
            continue
        i = STREAM_IDX[ego_stream]
        for rival_stream in rival_streams:
            if rival_stream in STREAM_IDX:
                turn_yield_mat[i, STREAM_IDX[rival_stream]] = 1.0

    return (mat.to(device), rival_idx.to(device),
            turning_mask.to(device), turn_yield_mat.to(device))


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable physics helpers (batched, operate on [E, N] tensors)
# ─────────────────────────────────────────────────────────────────────────────

def _idm(v: torch.Tensor, gap: torch.Tensor, v_lead: torch.Tensor,
         p=None) -> torch.Tensor:
    """IDM acceleration [E, N] — differentiable through v, v_lead, and physics params p."""
    a_max = p["idm_accel"] if p is not None else IDM_ACCEL
    s0    = p["idm_s0"]    if p is not None else IDM_S0
    t_hw  = p["idm_t"]     if p is not None else IDM_T
    dv     = (v - v_lead).clamp(min=0.0)
    # sqrt(a*b) uses fixed nominals to keep the denominator stable
    s_star = s0 + v * t_hw + v * dv / (2.0 * math.sqrt(IDM_ACCEL * IDM_BRAKE))
    s_star = s_star.clamp(min=IDM_S0)
    a      = a_max * (1.0 - (v / V_MAX).pow(IDM_DELTA) - (s_star / gap.clamp(min=0.1)).pow(2))
    return a.clamp(-IDM_BRAKE, IDM_ACCEL)  # safety bounds stay fixed


def _waypoint(v: torch.Tensor, arc: torch.Tensor, has_rival: torch.Tensor,
              is_turning: torch.Tensor, v_turn_cap: torch.Tensor, p=None) -> torch.Tensor:
    """
    Waypoint acceleration [E, N]:
      - approaching arm, has rival → target v_approach (kinematic cap)
      - otherwise                  → target V_MAX
      - turning vehicles           → additional braking toward SUMO junction speed cap
    v_turn_cap [E, N]: per-vehicle speed ceiling from SUMO net (V_LEFT_TURN / V_RIGHT_TURN / V_MAX).
    Differentiable through v, arc, and physics params p.
    """
    b_comfort  = p["b_comfort"]  if p is not None else B_COMFORT
    v_approach = p["v_approach"] if p is not None else V_APPROACH

    approaching = (arc < ARM_LENGTH)
    d           = (ARM_LENGTH - arc).clamp(min=0.0)
    v_kin       = (v_approach ** 2 + 2.0 * b_comfort * d).sqrt()
    v_target    = torch.where(approaching & has_rival, v_kin.clamp(max=V_MAX),
                              torch.full_like(v, V_MAX))
    a_wp        = (K_D * (v_target - v)).clamp(-B_COMFORT, IDM_ACCEL)  # safety bounds fixed

    # Turning deceleration bias — pushes toward SUMO junction speed limit.
    # Blends in smoothly above v_turn_cap; fully active at V_MAX.
    above_turn = (v > v_turn_cap).float()                               # not differentiated
    denom      = (V_MAX - v_turn_cap).clamp(min=EPS)
    mu_t       = ((v - v_turn_cap) / denom).clamp(0.0, 1.0)
    u_t        = (IDM_ACCEL * (1.0 - (v / v_turn_cap.clamp(min=0.1)).pow(IDM_DELTA))).clamp(-B_COMFORT, 0.0)
    turn_bias  = is_turning * mu_t * u_t * above_turn
    return a_wp + turn_bias


def safe_probe(
    a:           torch.Tensor,   # [E, N]    candidate acceleration (stays in graph)
    v:           torch.Tensor,   # [E, N]    current speed
    d_jct:       torch.Tensor,   # [E, N]    distance to junction (detached)
    eta_j:       torch.Tensor,   # [E, 1, N] rival ETAs  (detached)
    c_ij:        torch.Tensor,   # [N, N]    pairwise conflict mask
    yielder:     torch.Tensor,   # [E, N, N] yield matrix from _compute_physics
    approaching: torch.Tensor,   # [E, N]    bool — only probe approaching vehicles
) -> torch.Tensor:
    """
    Iterative Krauss-style safe-speed probe.

    At each iteration we project one step forward at the candidate speed and
    ask: given that speed, is there an imminent conflict?  If not, we add
    PROBE_STEP (0.2 m/s²).  The sigmoid gate sharpens across PROBE_K so
    gradients stay diffuse early and sharpen toward a near-hard check last.

    Conflict score uses time-domain delta (seconds), not normalised urgency,
    so PROBE_K = (0.2, 0.5, 1.0) gives meaningful discrimination across the
    full [0, SAFE_GAP] range.

    Gradient flows:  a → v_proj → eta_proj_i → delta_proj → conflict → mu_sig → a_new
    v is detached (same stability reason as in _compute_physics ETA computation).
    """
    appr_f  = approaching.float()                          # [E, N]
    c_ij_3d = c_ij.float().unsqueeze(0)                   # [1, N, N]

    for k in PROBE_K:
        # Project speed one step ahead with current candidate a
        v_proj      = (v.detach() + a * DT).clamp(0.0, V_MAX)          # [E, N]
        eta_proj_i  = (d_jct / v_proj.clamp(min=0.1)).unsqueeze(2)     # [E, N, 1]
        delta_proj  = eta_proj_i - eta_j                                 # [E, N, N]

        # Conflict score in seconds: 0 when |delta| >= SAFE_GAP, peaks at delta=0.
        # Using raw time (not normalised) so k values in (0.2, 0.5, 1.0) give
        # sigmoid outputs that sweep from ~0.5 (no conflict) to ~0 (max conflict).
        conflict_3d = (SAFE_GAP - delta_proj.abs()).clamp(min=0.0) * yielder * c_ij_3d
        conflict    = conflict_3d.max(dim=2).values * appr_f            # [E, N]

        # Gate: 1 = safe (add step), 0 = imminent crash (block addition)
        mu_sig = torch.sigmoid(-k * conflict)                           # [E, N]
        a      = a + mu_sig * PROBE_STEP * appr_f

    return a


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
        self.conflict_mat, self.rival_idx, self.turning_mask, self.turn_yield_mat = \
            _build_conflict_tensors(self.device)

        # Per-vehicle: is this stream a turning movement?
        self.is_turning = self.turning_mask[self.stream_ids]   # [N]

        # Per-vehicle SUMO junction speed cap (mirrors intersection.net.xml lane speeds)
        left_mask  = torch.tensor(
            [1.0 if STREAM_LIST[i] in _LEFT_TURN_SET  else 0.0 for i in range(N_STREAMS)],
            device=self.device)
        right_mask = torch.tensor(
            [1.0 if STREAM_LIST[i] in _RIGHT_TURN_SET else 0.0 for i in range(N_STREAMS)],
            device=self.device)
        v_cap_stream = (left_mask  * V_LEFT_TURN
                      + right_mask * V_RIGHT_TURN
                      + (1.0 - left_mask - right_mask) * V_MAX)  # [N_STREAMS]
        self.v_turn_cap = v_cap_stream[self.stream_ids]           # [N]

        # Pairwise conflict mask [N, N] — stream_ids[i] conflicts with stream_ids[j]
        self.c_ij = self.conflict_mat[
            self.stream_ids.unsqueeze(1),   # [N, 1]
            self.stream_ids.unsqueeze(0),   # [1, N]
        ]  # [N, N]

        # Movement priority [N, N]: turn_yield_ij[i,j]=1 → vehicle i always yields to j
        self.turn_yield_ij = self.turn_yield_mat[
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
        self.hist:      torch.Tensor | None = None   # [E, N, SEQ_LEN, D_EGO]
        self.wait_time: torch.Tensor | None = None   # [E, N] int — steps below V_WAIT
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
        self.wait_time    = torch.zeros(E, N, device=d)

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

    def _compute_physics(self, phys=None):
        """
        Compute all physics inputs for the model forward pass.

        phys : dict[str, Tensor] from model.phys_tune.as_dict(), or None for
               module-level constants.  Gradient flows through phys values.

        Returns (all [E, N] unless noted):
            gap, v_lead, u_wp, mu_priority, mu_cf
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
        # Detach v for ETA: d_jct/v creates a positive-feedback Jacobian
        # (faster v → lower ETA → less yield pressure → more acceleration) whose
        # eigenvalue > 1 explodes over BPTT=150 steps.  v_lead and u_wp stay in
        # graph — those paths are stable (IDM Jacobian eigenvalue < 1).
        eta   = d_jct / (v.detach().clamp(min=0.1))   # [E, N]
        # in_jct and exit vehicles treated as "already at crossing"
        eta   = torch.where(in_jct | on_exit, torch.zeros_like(eta), eta)

        # ── pairwise conflict force ────────────────────────────────────────
        # delta[e,i,j] = eta_i - eta_j: positive → i arrives later → i yields
        eta_i = eta.unsqueeze(2)     # [E, N, 1]
        eta_j = eta.unsqueeze(1)     # [E, 1, N]
        delta  = eta_i - eta_j       # [E, N, N]

        urgency = (1.0 - delta.abs() / SAFE_GAP).clamp(min=0.0)  # [E, N, N]

        # Movement-priority override: turns always yield to their designated through
        # movements; through movements never yield back to those turns.
        # tyi[i,j]=1 → ego i must yield to j regardless of delta.
        # tyj[i,j]=1 → rival j must yield to i, so i does not yield to j.
        tyi = self.turn_yield_ij.unsqueeze(0)        # [1, N, N]
        tyj = self.turn_yield_ij.t().unsqueeze(0)    # [1, N, N]
        yielder = ((delta >= 0).float() + tyi - tyj).clamp(0.0, 1.0)

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

        # Apply conflict mask, alive mask, approaching mask.
        # Only approaching vehicles (not in-junction, not past junction) yield or
        # are treated as rivals.  EXIT vehicles have already cleared the box and
        # must not re-enter the conflict computation.
        approaching_i  = approaching.float().unsqueeze(2)             # [E, N, 1]
        alive_j_appr   = (alive & ~on_exit).float().unsqueeze(1)      # [E, 1, N]
        mu_pair = (mu_softened
                   * self.c_ij.float().unsqueeze(0)        # [1, N, N] conflict
                   * alive_j_appr                          # [E, 1, N] rival not in EXIT
                   * alive_i                               # [E, N, 1] ego alive
                   * approaching_i)                        # [E, N, 1] ego approaching only

        # Worst conflict gate per ego — full [0, 1] range (membership function)
        mu_cf, _ = mu_pair.max(dim=2)                     # [E, N]
        mu_cf    = mu_cf.clamp(0.0, 1.0)

        # ── Structural pressure (ETA-independent) ─────────────────────────────
        # Fires whenever a structural rival is approaching, regardless of ETA gap.
        # Unlike mu_cf (which only rises when |eta_i-eta_j| < SAFE_GAP), this is
        # nonzero the moment any vehicle that i must yield to is on the road.
        # Used as a 5th kernel feature so the RKHS can distinguish
        # "no conflict yet but structural rival present" from "truly free".
        struct_pressure = (
            self.turn_yield_ij.unsqueeze(0).float()   # [1, N, N] i yields to j by rule
            * self.c_ij.float().unsqueeze(0)           # [1, N, N] conflicting streams
            * approaching.float().unsqueeze(1)         # [E, 1, N] rival approaching
            * alive_i                                  # [E, N, 1] ego alive
        ).max(dim=2).values                            # [E, N]  in {0, 1}
        struct_pressure = struct_pressure * approaching.float()   # zero for non-approaching

        # ── Waiting pressure: reduce mu_cf for vehicles stuck near the junction ──
        # P_veh = v * pack_w is zero for stopped vehicles, so the platoon-softening
        # path above gives no relief to stopped left-turns.  We build a separate
        # time-based pressure from wait_time (already tracked in step()).
        # A platoon of N waiting vehicles reaches full pressure proportionally faster
        # than a solo vehicle (WAIT_BOOST per additional stream-mate).
        if self.wait_time is not None:
            wait_w   = (self.wait_time / WAIT_RAMP).clamp(0.0, 1.0)   # [E, N]
            # stream-level sum of wait weights and count of waiting vehicles
            Q_str    = torch.zeros(E, N_STREAMS, device=d)
            n_wait_s = torch.zeros(E, N_STREAMS, device=d)
            Q_str.scatter_add_(1, stream_ids_exp, wait_w * approaching.float())
            n_wait_s.scatter_add_(1, stream_ids_exp, (wait_w > 0).float() * approaching.float())
            Q_ego      = Q_str.gather(1, stream_ids_exp)               # [E, N]
            n_wait_ego = n_wait_s.gather(1, stream_ids_exp)            # [E, N]
            # platoon boost: n=1 → ×1.0, n=2 → ×1.5, n=3 → ×2.0 ...
            boost   = 1.0 + (n_wait_ego - 1.0).clamp(min=0.0) * WAIT_BOOST
            mu_wait = (Q_ego * boost).clamp(0.0, 1.0) * approaching.float()
            mu_cf   = mu_cf * (1.0 - mu_wait)

        # Priority push gate: ego has crossing priority (delta < 0) AND has platoon pressure.
        # A turn that must yield (tyi=1) can never be a passer, even if delta < 0.
        passer        = ((delta < 0).float() - tyi).clamp(0.0, 1.0)
        yc_prio       = (-delta / YIELD_CONF_SCALE).clamp(0.0, 1.0)
        prio_pressure = (p_ego_grid / P_SCALE).clamp(0.0, 1.0)
        mu_prio_pair = (urgency * passer * yc_prio * prio_pressure
                        * self.c_ij.float().unsqueeze(0)
                        * alive_j_appr * alive_i * approaching_i)
        mu_priority, _ = mu_prio_pair.max(dim=2)
        mu_priority    = mu_priority.clamp(0.0, 1.0)

        # Junction collision gate: 1 when ego AND a conflicting rival are both inside the
        # junction simultaneously — exactly the jct_col condition used for collision counting.
        # Detached from v_t (arc positions are detached), but increases loss at crash steps
        # so BPTT finds stronger gradient through the speed term and x_seq history.
        in_jct_i_3d = in_jct.float().unsqueeze(2)          # [E, N, 1]
        in_jct_j_3d = in_jct.float().unsqueeze(1)          # [E, 1, N]
        jct_pair = (in_jct_i_3d * in_jct_j_3d
                    * self.c_ij.float().unsqueeze(0)
                    * alive_i * alive_j)
        mu_jct, _ = jct_pair.max(dim=2)                    # [E, N] ∈ {0, 1}

        # Per-stream worst mu (for rival token slot 3)
        mu_per_stream = torch.zeros(E, N, N_STREAMS, device=d)
        for k in range(N_STREAMS):
            mask_k = (self.stream_ids == k).float()        # [N]
            mu_k = (mu_pair * mask_k.unsqueeze(0).unsqueeze(0)).max(dim=2).values
            mu_per_stream[:, :, k] = mu_k                 # [E, N]

        # ── has_rival flag [E, N]: any non-EXIT rival within ETA threshold ──
        rival_present = (
            (eta_j < ETA_RIVAL_TH)                        # [E, 1, N]
            & self.c_ij.unsqueeze(0).bool()               # [1, N, N]
            & alive_j_appr.bool()                         # [E, 1, N] rival not in EXIT
        )
        has_rival = rival_present.any(dim=2)              # [E, N]

        # ── waypoint + IDM → u_wp ─────────────────────────────────────────
        is_turning_exp = self.is_turning.unsqueeze(0).expand(E, N)  # [E, N]
        v_turn_cap_exp = self.v_turn_cap.unsqueeze(0).expand(E, N)   # [E, N]
        a_wp  = _waypoint(v, arc, has_rival, is_turning_exp, v_turn_cap_exp, p=phys)
        a_idm = _idm(v, gap, v_lead, p=phys)
        on_approach = approaching.float()
        a_max = phys["idm_accel"] if phys is not None else IDM_ACCEL
        u_wp = (torch.min(a_idm, a_wp) * on_approach
                + a_max * (1.0 - (v / V_MAX).pow(IDM_DELTA)).clamp(-2, 2)
                * (1.0 - on_approach))

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

        return gap, v_lead, u_wp, mu_priority, mu_cf, mu_jct, own_summary, rival_tokens, d_jct, in_jct, eta_j, yielder, struct_pressure

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

        # ── 2. physics — in graph so BPTT flows through v_lead, eta, mu_cf, u_wp ──
        # gap is naturally detached (arc_t is integrated with v_t.detach()).
        # Collision detection is kept in no_grad — only boolean ops and .item().
        phys = model.phys_tune.as_dict() if hasattr(model, "phys_tune") else None
        gap, v_lead, u_wp, mu_priority, mu_cf, mu_jct, own_sum, rival_tok, d_jct, in_jct, eta_j, yielder, struct_pres = \
            self._compute_physics(phys)

        with torch.no_grad():
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

        # ── 3. ego token ───────────────────────────────────────────────────────
        new_ego = torch.stack([
            self.v_t / V_MAX,                         # grad flows through v_t
            (gap    / GAP_MAX ).clamp(0.0, 1.0),      # detached (arc_t is detached)
            (self.v_t - v_lead) / V_MAX,              # grad flows through v_t and v_lead
            (d_jct  / ARM_LENGTH).clamp(0.0, 1.0),   # detached (arc_t is detached)
        ], dim=2)                                      # [E, N, D_EGO]

        # Shift history and append (graph-safe cat, no in-place roll)
        self.hist = torch.cat([self.hist[:, :, 1:, :], new_ego.unsqueeze(2)], dim=2)

        # ── 4. model forward ──────────────────────────────────────────────
        EN = E * N
        v_over_cap = ((self.v_t - self.v_turn_cap.unsqueeze(0)) / V_MAX
                      ).clamp(0.0, 1.0)                            # [E, N]
        is_turn_f  = self.is_turning.unsqueeze(0).float(
                     ).expand(E, N)                                # [E, N]
        # struct_pressure fades as waiting pressure builds: once a vehicle has
        # waited long enough for mu_wait to override the conflict gate, the
        # structural-hold should also release so the vehicle can move.
        wait_norm_s  = (self.wait_time / WAIT_RAMP).clamp(0.0, 1.0)  # [E, N]
        struct_pres_g = struct_pres * (1.0 - wait_norm_s)             # [E, N]
        a_out = model(
            self.hist.view(EN, SEQ_LEN, D_EGO),
            own_sum.view(EN, D_SUM),
            rival_tok.view(EN, K_MAX, D_RIVAL),
            u_wp.view(EN),
            mu_cf.view(EN),
            mu_priority.view(EN),
            v_over_cap.view(EN),
            is_turn_f.view(EN),
            struct_pres_g.view(EN),
        ).view(E, N)                                               # [E, N]

        # ── 5. safe-speed probe then integrate v_t (in graph) ────────────
        on_exit_s    = (self.arc_t >= ARM_LENGTH + JCT_LEN) & self.alive
        approaching_s = self.alive & ~in_jct & ~on_exit_s
        a_out = safe_probe(a_out, self.v_t, d_jct, eta_j, self.c_ij, yielder, approaching_s)
        v_new    = (self.v_t + a_out * DT).clamp(0.0, V_MAX)
        self.v_t = v_new * self.alive.float()

        # ── 6. update arc, despawn, wait counter (no_grad) ───────────────
        with torch.no_grad():
            self.arc_t = self.arc_t + self.v_t.detach() * DT
            self.alive = self.alive & (self.arc_t <= DEPART_ARC)
            # increment wait counter for alive vehicles below V_WAIT, reset otherwise
            is_slow = (self.v_t.detach() < V_WAIT) & self.alive
            self.wait_time = torch.where(is_slow, self.wait_time + 1,
                                         torch.zeros_like(self.wait_time))

        # ── reward + penalties ────────────────────────────────────────────
        alive_f    = self.alive.float()
        n_alive    = alive_f.sum(dim=1).clamp(min=1.0)
        mean_speed = (self.v_t * alive_f).sum(dim=1) / n_alive   # [E]

        # Wait penalty: soft ramp weight detached, gradient flows through v_t only
        weight        = ((self.wait_time - WAIT_PAT) / WAIT_PAT).clamp(0.0, WAIT_MAX_W)
        speed_deficit = torch.relu(V_WAIT - self.v_t) / V_WAIT
        wait_pen      = (weight * speed_deficit * alive_f).sum(dim=1) / n_alive  # [E]

        # Junction conflict penalty — two terms:
        #   1. mu_cf * approaching_f : ETA-delta gate for vehicles still outside the junction.
        #      SAFE_GAP=8s catches slow-queue vehicles (v~5m/s, d~30m, eta~6s).
        #   2. mu_jct * in_jct_f     : binary gate for vehicles INSIDE the junction that have
        #      a conflicting rival also in the junction simultaneously (= jct_col condition).
        #      mu_jct is detached from v_t, but raises the loss at collision steps so BPTT
        #      finds stronger gradient through the speed term and x_seq history back to the
        #      model's approach decision.
        approaching_f = ((self.arc_t < ARM_LENGTH) & self.alive).float()
        in_jct_f      = (in_jct & self.alive).float()
        n_danger = (approaching_f + in_jct_f).clamp(0, 1).sum(dim=1).clamp(min=1.0)
        ttc_pen  = (mu_cf * approaching_f + mu_jct * in_jct_f).sum(dim=1) / n_danger  # [E]

        self.step_idx += 1
        return mean_speed, wait_pen, ttc_pen
