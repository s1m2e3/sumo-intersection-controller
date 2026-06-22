"""
signal_nn.py — per-phase adaptive signal controller trained in pure PyTorch.

PhaseNet is called once per cycle → all 4 green durations T_0..T_3 at once.
Every T_k stays in the computation graph via the full-cycle soft-role formula:

    c[0] = cycle_start
    c[k] = c[k-1] + T[k-1] + T_AR          (differentiable in T[0..k-1])
    role_m(t) = Σ_{k: m ∈ green_k}
                  σ(β*(t − c[k])) · σ(β*(c[k] + T[k] − t))

All inner loops (leader gap, yield-cap, soft-role, phase-features) are fully
vectorised — no Python for-loops over vehicles.  CUDA is used when available.

Warm-up: fixed default timing for the first WARMUP_T seconds so vehicles
populate the network before the NN sees the feature vector.

CLI:
    conda run -n car-following-sumo python signal_nn.py [epochs=N] [lr=1e-3]
        [hidden=32] [s=200] [l=100] [r=100] [print=10]
"""
import math
import os
import random
import torch
import torch.nn as nn
import sys
import time

import utils
import sim_torch as S
import turns_geom as G

NET_PATH = S.NET_PATH

# ── signal-phase constants ────────────────────────────────────────────────────
_PHASE_GREEN = [
    frozenset({0, 1, 3, 4}),   # Phase 0: EW thru + right
    frozenset({2, 5}),          # Phase 1: EW left
    frozenset({6, 7, 9, 10}),  # Phase 2: NS thru + right
    frozenset({8, 11}),         # Phase 3: NS left
]
N_PHASES = 4
T_AR     = 3.0    # all-red clearance (fixed)
T_MIN    = 5.0    # minimum NN-output green duration
T_MAX    = 60.0   # maximum NN-output green duration
BETA     = 2.0    # sigmoid sharpness
WARMUP_T = 10.0   # seconds of fixed timing before NN takes over

# ETA-based feasibility projection
_ETA_A_MAX = 2.0    # free-flow acceleration used for ETA estimate (m/s²)
_ETA_RANGE = 100.0  # look-ahead distance for approaching vehicles (m)
_ETA_STEP  = 2.0    # additive correction step size per FGD projection (s)

# Opportunistic movement
_OPP_D_CONFLICT = 10.0  # m past stop bar to approximate conflict point
_OPP_GAP_MIN    = 3.0   # s minimum certified gap on each side for opportunistic passage

# Fixed default durations used during warm-up
_DEFAULT_T = torch.tensor([30.0, 15.0, 30.0, 15.0])

# Precomputed lookup: _GREEN_LOOKUP[k, m] = 1 if movement m is in phase k's green set.
# Shape [N_PHASES, 12].  Indexed by (phase, movement) for O(1) in-phase lookup.
_GREEN_LOOKUP = torch.zeros(N_PHASES, 12, dtype=torch.float32)
for _k in range(N_PHASES):
    for _m in range(12):
        _GREEN_LOOKUP[_k, _m] = float(_m in _PHASE_GREEN[_k])
# Alias — avoids repeated .to() calls inside the hot loop when device is CPU
_GL_CPU = _GREEN_LOOKUP  # same object; kept for clarity in soft_role

# Per-phase directional split: two opposing sides share each green phase.
# _PHASE_SIDES[k] = (side_A_movements, side_B_movements)
# Phase 0: EW thru+right {0,1}  vs  WE thru+right {3,4}
# Phase 1: EW left       {2}    vs  WE left        {5}
# Phase 2: NS thru+right {6,7}  vs  SN thru+right  {9,10}
# Phase 3: NS left       {8}    vs  SN left         {11}
_PHASE_SIDES = [
    (frozenset({0, 1}), frozenset({3, 4})),
    (frozenset({2}),    frozenset({5})),
    (frozenset({6, 7}), frozenset({9, 10})),
    (frozenset({8}),    frozenset({11})),
]
# _SIDE_LOOKUP[k, s, m] = 1 if movement m belongs to side s (0=A, 1=B) of phase k.
_SIDE_LOOKUP = torch.zeros(N_PHASES, 2, 12, dtype=torch.float32)
for _k, (_A, _B) in enumerate(_PHASE_SIDES):
    for _m in _A: _SIDE_LOOKUP[_k, 0, _m] = 1.0
    for _m in _B: _SIDE_LOOKUP[_k, 1, _m] = 1.0

DEVICE = torch.device("cpu")   # Na≈20-30 vehicles: CUDA kernel-launch overhead > gain

BEST_CKPT   = "signal_nn_best.pt"
LATEST_CKPT = "signal_nn_latest.pt"


# ── network ───────────────────────────────────────────────────────────────────

class PhaseNet(nn.Module):
    """Cross-phase attention signal controller → 4 green durations per cycle.

    The 4 phases are treated as tokens.  Self-attention lets each phase's
    output duration be conditioned on every other phase's traffic state, so
    the model can learn relationships like "NS is heavy AND EW is light →
    shift time toward NS."

    Input per phase (2 features):
        queue_k / 20        approaching vehicle count (d_junc ∈ (0, 120 m])
        mean_v_k / V0       mean approach speed, normalised

    Architecture:
        MLP embed (4 → hidden) → TransformerEncoder (ReLU, n_layers) → Linear head (hidden → 1)
        → Sigmoid → [T_MIN, T_MAX]

    Output: T [N_PHASES] green durations in seconds
    """
    def __init__(self, hidden: int = 32, n_heads: int = 2, n_layers: int = 1, n_in: int = 4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        enc_layer  = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden * 2,
            dropout=0.0, batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head    = nn.Linear(hidden, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: [N_PHASES, 4] = [queue_A/20, mean_v_A/V0, queue_B/20, mean_v_B/V0]"""
        x   = self.embed(feat).unsqueeze(0)    # [1, N_PHASES, hidden]
        x   = self.encoder(x).squeeze(0)       # [N_PHASES, hidden]
        raw = self.head(x).squeeze(1)           # [N_PHASES]
        return T_MIN + (T_MAX - T_MIN) * torch.sigmoid(raw)


# ── cycle boundaries ──────────────────────────────────────────────────────────

def cycle_boundaries(T_all: torch.Tensor, cycle_start: float):
    """c_stack [N_PHASES], e_stack [N_PHASES] as differentiable tensors.

    c[k] = cycle_start + Σ_{j<k} (T[j] + T_AR)   depends on T[0..k-1]
    e[k] = c[k] + T[k]                              depends on T[0..k]
    """
    c0  = torch.tensor(cycle_start, dtype=T_all.dtype, device=T_all.device)
    cs  = [c0]
    for k in range(1, N_PHASES):
        cs.append(cs[-1] + T_all[k - 1] + T_AR)
    c_stack = torch.stack(cs)           # [N_PHASES]
    e_stack = c_stack + T_all           # [N_PHASES], differentiable in T_all
    return c_stack, e_stack


# ── vectorised soft role ──────────────────────────────────────────────────────

def soft_role(t: float, c_stack: torch.Tensor, e_stack: torch.Tensor,
              mvi: torch.Tensor) -> torch.Tensor:
    """Vectorised soft role for Na vehicles at time t.

    in_green [N_PHASES, Na] = _GREEN_LOOKUP[:, mvi] (device-aware gather)
    window   [N_PHASES]     = σ(β*(t−c)) · σ(β*(e−t))
    role     [Na]           = (in_green * window[:, None]).sum(0)
    """
    gl       = _GL_CPU[:, mvi]                      # [N_PHASES, Na]
    w_in     = torch.sigmoid(BETA * (t - c_stack)) # [N_PHASES]
    w_out    = torch.sigmoid(BETA * (e_stack - t)) # [N_PHASES]
    window   = (w_in * w_out).unsqueeze(1)         # [N_PHASES, 1]
    return (gl * window).sum(0).clamp(0.0, 1.0)    # [Na]


# ── vectorised phase features ─────────────────────────────────────────────────

@torch.no_grad()
def phase_features(s_act: torch.Tensor, v_act: torch.Tensor,
                   mv_act: torch.Tensor, s_junc: torch.Tensor) -> torch.Tensor:
    """Vectorised per-phase, per-direction features.

    Returns feat [N_PHASES, 4]:
        [queue_A/20, mean_v_A/V0, queue_B/20, mean_v_B/V0]
    where A and B are the two opposing directions sharing each phase.
    Features are detached — gradient flows through the role sigmoid, not inputs.
    """
    d      = s_junc[mv_act] - s_act                              # [Na]
    appr   = (d > 0.0) & (d < 120.0)                             # [Na] bool
    sl     = _SIDE_LOOKUP[:, :, mv_act]                          # [N_PHASES, 2, Na]
    mask   = sl * appr.float().unsqueeze(0).unsqueeze(0)         # [N_PHASES, 2, Na]
    queue  = mask.sum(2)                                          # [N_PHASES, 2]
    v_sum  = (mask * v_act.unsqueeze(0).unsqueeze(0)).sum(2)     # [N_PHASES, 2]
    mean_v = v_sum / queue.clamp(min=1.0)
    mean_v = torch.where(queue > 0, mean_v, torch.zeros_like(mean_v))
    return torch.stack([queue[:, 0] / 20.0, mean_v[:, 0] / utils.V0,
                        queue[:, 1] / 20.0, mean_v[:, 1] / utils.V0], dim=1)  # [N_PHASES, 4]


# ── ETA feasibility projection ────────────────────────────────────────────────

def phase_eta(s_act: torch.Tensor, v_act: torch.Tensor,
              mv_act: torch.Tensor, s_junc: torch.Tensor) -> list:
    """Free-flow ETA of the furthest approaching vehicle within _ETA_RANGE m, per phase.

    Uses constant-acceleration kinematic: vehicle accelerates from current speed
    toward V0 at _ETA_A_MAX, then cruises.  Returns T_MAX when no vehicles
    are approaching a phase (no constraint).

    Returns a plain Python list of N_PHASES floats — no autograd involvement.
    """
    etas = []
    mv_list = mv_act.tolist()
    d_all   = s_junc[mv_act] - s_act   # [Na]
    for k in range(N_PHASES):
        green_k = _PHASE_GREEN[k]
        in_phase = torch.tensor([int(m) in green_k for m in mv_list], dtype=torch.bool)
        mask = in_phase & (d_all > 0.0) & (d_all < _ETA_RANGE)
        if not mask.any():
            etas.append(T_MAX)
            continue
        d_m   = d_all[mask]
        v_m   = v_act[mask]
        idx   = int(d_m.argmax())        # furthest vehicle = last to arrive
        d_last = float(d_m[idx])
        v_last = float(v_m[idx])
        v0    = utils.V0
        a     = _ETA_A_MAX
        d_accel = max(0.0, (v0 ** 2 - v_last ** 2) / (2.0 * a))
        if d_last <= d_accel:
            # Junction reached before hitting V0
            t_eta = (-v_last + math.sqrt(max(v_last ** 2 + 2.0 * a * d_last, 0.0))) / a
        else:
            t_accel = (v0 - v_last) / a
            t_eta   = t_accel + (d_last - d_accel) / v0
        etas.append(max(t_eta, 0.0))
    return etas


# ── opportunistic movement gap check ─────────────────────────────────────────

def _active_phase_idx(t: float, c_stack: torch.Tensor,
                      e_stack: torch.Tensor) -> int:
    """Index of the currently active green phase (0-3), or -1 during all-red."""
    for k in range(N_PHASES):
        if c_stack[k].item() <= t < e_stack[k].item():
            return k
    return -1


def compute_opp(t: float, v_a: torch.Tensor, mv_a: torch.Tensor,
                d_junc_a: torch.Tensor,
                c_stack: torch.Tensor, e_stack: torch.Tensor) -> torch.Tensor:
    """Opportunistic flag [Na] — 1.0 for the leader of each non-active phase
    that has a certified ≥_OPP_GAP_MIN s gap in the active-phase conflict stream.

    Active-phase ETAs: constant-speed to conflict point (vehicles already moving).
    Leader ETAs: free-flow kinematic to conflict point (same estimator as phase_eta).
    All plain Python floats — no autograd involvement.
    """
    Na  = len(v_a)
    opp = torch.zeros(Na)
    ak  = _active_phase_idx(t, c_stack, e_stack)
    if ak < 0:
        return opp   # all-red: no opportunistic movement

    mv_list   = mv_a.tolist()
    active_mvs = _PHASE_GREEN[ak]

    # Free-flow kinematic ETAs of active-phase vehicles to the conflict point.
    # Use accelerating estimator so a just-spawned vehicle at v≈0 gets its
    # realistic minimum arrival time, not constant-speed ÷ 0.1 → thousands of s.
    in_active = torch.tensor([int(m) in active_mvs for m in mv_list], dtype=torch.bool)
    if in_active.any():
        eta_active = []
        for _dc, _vc in zip(
                (d_junc_a[in_active] + _OPP_D_CONFLICT).clamp(min=0.0).tolist(),
                v_a[in_active].clamp(min=0.0).tolist()):
            _a  = _ETA_A_MAX
            _da = max(0.0, (utils.V0 ** 2 - _vc ** 2) / (2.0 * _a))
            if _dc <= _da:
                eta_active.append(
                    (-_vc + math.sqrt(max(_vc ** 2 + 2.0 * _a * _dc, 0.0))) / _a)
            else:
                eta_active.append((utils.V0 - _vc) / _a + (_dc - _da) / utils.V0)
    else:
        eta_active = []

    # Pass 1: collect candidates — (veh_idx, eta_at_conflict, queue_size)
    candidates = []
    for k in range(N_PHASES):
        if k == ak:
            continue
        phase_mvs = _PHASE_GREEN[k]
        in_phase  = torch.tensor([int(m) in phase_mvs for m in mv_list], dtype=torch.bool)
        appr      = in_phase & (d_junc_a > 0.0) & (d_junc_a < _ETA_RANGE)
        if not appr.any():
            continue
        appr_idx = appr.nonzero().flatten()
        d_appr   = d_junc_a[appr]
        loc      = int(d_appr.argmin())
        d_lead   = float(d_appr[loc])
        v_lead_  = float(v_a[appr][loc])
        queue    = int(appr.sum())

        d_conf  = d_lead + _OPP_D_CONFLICT
        v0, a   = utils.V0, _ETA_A_MAX
        d_accel = max(0.0, (v0 ** 2 - v_lead_ ** 2) / (2.0 * a))
        if d_conf <= d_accel:
            eta_lead = (-v_lead_ + math.sqrt(max(v_lead_ ** 2 + 2.0 * a * d_conf, 0.0))) / a
        else:
            eta_lead = (v0 - v_lead_) / a + (d_conf - d_accel) / v0

        candidates.append((int(appr_idx[loc]), eta_lead, queue))

    if not candidates:
        return opp

    # Pass 2: filter by active-phase gap only
    feasible = []
    for veh_idx, eta_i, queue_i in candidates:
        before = [eta_i - e for e in eta_active if e < eta_i]
        after  = [e - eta_i for e in eta_active if e > eta_i]
        if (not before or min(before) >= _OPP_GAP_MIN) and \
           (not after  or min(after)  >= _OPP_GAP_MIN):
            feasible.append((veh_idx, eta_i, queue_i))

    # Pass 3: resolve inter-candidate conflicts by queue priority (greedy).
    n        = len(feasible)
    conflict = [[abs(feasible[i][1] - feasible[j][1]) < _OPP_GAP_MIN
                 for j in range(n)] for i in range(n)]
    remaining = set(range(n))
    while remaining:
        best = max(remaining, key=lambda i: feasible[i][2])
        opp[feasible[best][0]] = 1.0
        remaining -= {best} | {j for j in remaining if conflict[best][j]}

    return opp


# ── differentiable training simulation ────────────────────────────────────────

def simulate_differentiable(events: list, net: PhaseNet,
                             s_junc: torch.Tensor, path_len: torch.Tensor,
                             dt: float = S.DT, device: torch.device = DEVICE,
                             t_end: float = 200.0,
                             loss_mode: str = "both"):
    """Vectorised training forward pass. Returns (delay_loss, info_dict).

    Loss is junction-proximity-weighted delay:
        L = Σ_t mean_i [ relu(V0 − v_cmd_i) · 1/(1 + d_junc_i) ] · dt
    Vehicles near the junction (small d_junc) are up-weighted so the
    gradient prioritises clearing the bottleneck, which drives throughput.
    The proximity weight is detached (position is state, not optimised);
    gradient flows through v_cmd → role → cycle boundaries → T_k → net.
    """
    s_junc   = s_junc.to(device)
    path_len = path_len.to(device)

    M    = int(s_junc.shape[0])
    Ntot = len(events)
    depart = torch.tensor([e[0] for e in events], device=device)
    move   = torch.tensor([e[1] for e in events], dtype=torch.long, device=device)
    s      = torch.zeros(Ntot, device=device)
    v      = torch.zeros(Ntot, device=device)
    state  = torch.zeros(Ntot, dtype=torch.long, device=device)
    prev_a = torch.zeros(Ntot, device=device)

    # ── signal timing ─────────────────────────────────────────────────────────
    nn_active = False
    # Warm-up: fixed default timing (no NN, no gradient) for first WARMUP_T s
    def _default_cycle(t0):
        T = _DEFAULT_T.clone().detach().to(device)
        cs, es = cycle_boundaries(T, t0)
        return T, cs, es, float(es[-1].item()) + T_AR

    _net_n_in = net.embed[0].in_features if hasattr(net, "embed") else 4

    def _nn_cycle(t0):
        act = (state == 1).nonzero().flatten()
        if len(act) > 0:
            feat = phase_features(s[act], v[act], move[act], s_junc)
            etas = phase_eta(s[act], v[act], move[act], s_junc)
        else:
            feat = torch.zeros(N_PHASES, 4, device=device)
            etas = [T_MAX] * N_PHASES
        T_raw = net(feat[:, :_net_n_in])
        # ── FGD projection: enforce T_k ≤ ETA_k ──────────────────────────────
        # corr is a plain float tensor — no autograd involvement.
        # Gradient flows through T_raw unchanged; the net is penalised for
        # violations on the next backward pass.
        corr = torch.zeros(N_PHASES)
        for k in range(N_PHASES):
            excess = T_raw[k].item() - etas[k]
            if excess > 0.0:
                corr[k] = math.ceil(excess / _ETA_STEP) * _ETA_STEP
        T_all = (T_raw - corr).clamp(min=T_MIN)
        cs, es = cycle_boundaries(T_all, t0)
        return T_all, cs, es, float(es[-1].item()) + T_AR

    _, c_stack, e_stack, cycle_end = _default_cycle(0.0)

    n_steps      = int(t_end / dt)
    delay_loss   = torch.zeros(1, device=device)
    delay_metric = 0.0   # proximity-weighted speed deficit (no grad, always tracked)
    thru_metric  = 0.0   # proximity-weighted mean speed    (no grad, always tracked)
    arrived      = 0

    for step in range(n_steps):
        t = step * dt

        # Switch from warm-up to NN at WARMUP_T
        if not nn_active and t >= WARMUP_T:
            nn_active = True
            _, c_stack, e_stack, cycle_end = _nn_cycle(t)
        elif nn_active and t >= cycle_end:
            _, c_stack, e_stack, cycle_end = _nn_cycle(cycle_end)

        # ── spawn (O(M) Python loop — M=12, cheap relative to vehicle ops) ────
        for mi in range(M):
            am    = (state == 1) & (move == mi)
            min_s = float(s[am].min()) if am.any() else 1e9
            if min_s >= 2.0 * utils.L_VEH + S.SPAWN_GAP:
                cand = ((state == 0) & (move == mi)
                        & (depart <= t)).nonzero().flatten()
                if len(cand):
                    k_v = int(cand[depart[cand].argmin()])
                    if am.any():
                        v_ld   = float(v[am][s[am].argmin()])
                        gap_in = max(min_s - 2.0 * utils.L_VEH, 0.0)
                        v_safe = (v_ld ** 2 + 2.0 * utils.B_MAX * gap_in) ** 0.5
                    else:
                        v_safe = S.V_PHYS
                    state[k_v] = 1
                    s[k_v]     = utils.L_VEH
                    v[k_v]     = min(S.V_PHYS, v_safe)

        act = (state == 1).nonzero().flatten()
        Na  = len(act)
        if Na == 0:
            continue

        s_a      = s[act]
        v_a      = v[act]
        mv_a     = move[act]
        d_junc_a = s_junc[mv_a] - s_a   # [Na]  +ve = approaching

        # ── soft role (fully vectorised) ──────────────────────────────────────
        role  = soft_role(t, c_stack, e_stack, mv_a)
        role  = (role + (d_junc_a < 0.0).float()).clamp(0.0, 1.0)
        # Opportunistic flag: plain tensor, no autograd — gradient flows through role only
        opp_a = compute_opp(t, v_a, mv_a, d_junc_a, c_stack, e_stack)

        # ── leader gap (vectorised O(Na²) → single batch op) ─────────────────
        appr_lane  = mv_a // 3                  # [Na] approach 0-3
        left_flag  = (mv_a % 3 == 2)            # [Na] bool
        same_lane  = ((appr_lane.unsqueeze(0) == appr_lane.unsqueeze(1)) &
                      (left_flag.unsqueeze(0)  == left_flag.unsqueeze(1)))  # [Na, Na]
        # not_past[i,j] = d_junc_a[j] > -2: potential leader j is not clear past box
        not_past   = d_junc_a.unsqueeze(0) > -2.0   # [1, Na] → broadcasts [Na, Na]
        # dg[i,j] = s[j]-s[i]-L  (positive if j ahead of i by ≥ L; negative for i==j → auto-excluded)
        dg         = s_a.unsqueeze(0) - s_a.unsqueeze(1) - utils.L_VEH
        valid      = same_lane & not_past & (dg >= 0.0)   # diagonal already excluded (dg<0)
        dg_masked  = torch.where(valid, dg, torch.full_like(dg, 1e9))
        gap, li    = dg_masked.min(dim=1)       # [Na] closest valid leader
        has_lead   = gap < 1e8
        v_lead     = torch.where(has_lead, v_a[li], v_a)
        gap        = torch.where(has_lead, gap.clamp(min=0.0),
                                 torch.full_like(gap, 300.0))

        # ── GP kernel (role → gradient path; opp adds 3rd anchor axis) ─────────
        a = utils.controller_acceleration(
            torch.zeros(Na, device=device), gap + utils.L_VEH,
            v_a, v_lead,
            role=role, opp=opp_a, a_prev=prev_a[act],
            kappa=0.5, brake_exempt=True, brake_floor=True)

        # ── yield cap (vectorised) ────────────────────────────────────────────
        eff       = (d_junc_a.clamp(min=0.0) - utils.STOP_OFFSET).clamp(min=0.0)
        yield_cap = (2.0 * utils.B_MAX * eff) ** 0.5
        passer    = (role.detach() > 0.5) | (opp_a > 0.5) | (d_junc_a <= 0.0)
        yield_cap = torch.where(passer, torch.full_like(yield_cap, 1e9), yield_cap)

        # ── velocity command and loss ─────────────────────────────────────────
        v_cmd = (v_a + a * dt).clamp(0.0, S.V_PHYS)
        v_cmd = torch.minimum(v_cmd, yield_cap)
        # w detached (position is state); gradient flows through v_cmd → a → role → T_k
        w = 1.0 / (1.0 + d_junc_a.detach().clamp(min=0.0))   # [Na]
        if loss_mode == "throughput":
            # Maximise proximity-weighted speed.
            # ∂L/∂v_cmd_i = −w_i/Na < 0 for every active vehicle — always non-zero.
            delay_loss = delay_loss - (v_cmd * w).mean() * dt
        elif loss_mode == "delay":
            # Penalise speed below free-flow (relu redundant since v_cmd ≤ V0,
            # but kept for clarity and future flexibility).
            delay_loss = delay_loss + (torch.relu(S.V_PHYS - v_cmd) * w).mean() * dt
        else:
            # "both": delay − throughput  =  relu(V0−v)·w − v·w  =  (V0−2v)·w
            # Gradient = −2w/Na < 0, always non-zero.  No 0.5 rescaling.
            delay_loss = delay_loss + (torch.relu(S.V_PHYS - v_cmd) * w).mean() * dt
            delay_loss = delay_loss - (v_cmd * w).mean() * dt

        # ── metrics + state update (no grad) ─────────────────────────────────
        with torch.no_grad():
            _vc = v_cmd.detach()
            delay_metric += float((torch.relu(S.V_PHYS - _vc) * w).mean() * dt)
            thru_metric  += float((_vc * w).mean() * dt)
            prev_a[act] = a.detach()
            v[act]      = _vc
            s[act]      = s_a + _vc * dt

        done        = act[s[act] >= path_len[mv_a]]
        state[done] = 2
        arrived    += int(len(done))

    return delay_loss.squeeze(), {"arrived": arrived,
                                  "delay_metric": delay_metric,
                                  "thru_metric": thru_metric}


# ── training loop ─────────────────────────────────────────────────────────────

def train(n_epochs: int = 200, lr: float = 1e-3,
          vph: dict = None, n_seeds: int = 5,
          hidden: int = 32, n_heads: int = 2, n_layers: int = 1,
          t_end: float = 200.0, loss_mode: str = "both",
          print_every: int = 10,
          best_ckpt: str = BEST_CKPT, latest_ckpt: str = LATEST_CKPT,
          on_epoch=None):
    """Train PhaseNet on 6 randomly re-sampled Poisson seeds per epoch.

    Loads best_ckpt at startup if it exists.  Saves latest_ckpt after every
    gradient step; saves best_ckpt whenever epoch loss improves.
    """
    vph = vph or {"s": 200, "l": 100, "r": 100}
    print(f"Device: {DEVICE}")

    geo, s_cp, s_junc, _ = G.gate_geometry(NET_PATH)
    mv_list  = G.movements(NET_PATH)
    M        = len(mv_list)
    path_len = torch.tensor([float(geo[i][1][-1]) for i in range(M)])

    best_loss = float("inf")
    if os.path.exists(best_ckpt):
        ckpt     = torch.load(best_ckpt, map_location=DEVICE, weights_only=True)
        arch  = ckpt.get("arch", {})
        if "n_in" not in arch:
            arch["n_in"] = int(ckpt["model"]["embed.0.weight"].shape[1])
        arch.setdefault("hidden",   hidden)
        arch.setdefault("n_heads",  n_heads)
        arch.setdefault("n_layers", n_layers)
        net      = PhaseNet(**arch).to(DEVICE)
        net.load_state_dict(ckpt["model"])
        best_loss = float(ckpt.get("loss", float("inf")))
        ep_saved  = ckpt.get("epoch", "?")
        print(f"Loaded {best_ckpt!r}  (epoch={ep_saved}  loss={best_loss:.1f}  arch={arch})")
    else:
        net = PhaseNet(hidden=hidden, n_heads=n_heads, n_layers=n_layers).to(DEVICE)
        print(f"Starting fresh — no checkpoint at {best_ckpt!r}")

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = random.Random()

    print(f"PhaseNet  params={sum(p.numel() for p in net.parameters())}  "
          f"hidden={hidden}  n_heads={n_heads}  n_layers={n_layers}  n_seeds={n_seeds}")
    print(f"Training: epochs={n_epochs}  lr={lr}  t_end={t_end}s  loss={loss_mode}  "
          f"vph s~U(200,500)  l,r~U(100,300)  (sampled per seed)\n")

    grad_checked = False

    for epoch in range(n_epochs):
        seeds_ep = [rng.randint(0, 999_999) for _ in range(n_seeds)]

        opt.zero_grad()
        total_loss         = torch.zeros(1, device=DEVICE)
        total_arr          = 0
        total_delay_metric = 0.0
        total_thru_metric  = 0.0

        for sd in seeds_ep:
            ep_vph = {
                "s": rng.uniform(200, 500),
                "l": rng.uniform(100, 300),
                "r": rng.uniform(100, 300),
            }
            evts = S.gen_turn_events(ep_vph, sd, t_end=t_end)
            loss, info = simulate_differentiable(evts, net, s_junc, path_len,
                                                 t_end=t_end, loss_mode=loss_mode)
            total_loss         = total_loss + loss
            total_arr         += info["arrived"]
            total_delay_metric += info["delay_metric"]
            total_thru_metric  += info["thru_metric"]

        total_loss = total_loss / n_seeds
        total_loss.backward()

        # ── gradient check (after first backward) ─────────────────────────────
        if not grad_checked:
            grad_checked = True
            print("── Gradient check ──────────────────────────────────────────")
            all_ok = True
            for name, p in net.named_parameters():
                if p.grad is None:
                    print(f"  {name:35s}  NONE"); all_ok = False
                else:
                    gmax = float(p.grad.abs().max())
                    ok   = gmax > 1e-10
                    if not ok:
                        all_ok = False
                    print(f"  {name:35s}  {'YES' if ok else 'ZERO':4s}  max={gmax:.3e}")
            print(f"  → {'PASS' if all_ok else 'FAIL'}")
            print("────────────────────────────────────────────────────────────\n")

        opt.step()

        epoch_loss = float(total_loss)

        _ckpt_payload = {"model": net.state_dict(), "loss": epoch_loss,
                         "epoch": epoch + 1,
                         "arch": {"hidden": hidden, "n_heads": n_heads,
                                  "n_layers": n_layers, "n_in": 4}}

        # Save latest after every gradient step
        torch.save(_ckpt_payload, latest_ckpt)

        # Save best when loss improves
        best_tag = ""
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(_ckpt_payload, best_ckpt)
            best_tag = f"  ★ best={epoch_loss:.1f} → {best_ckpt}"

        if on_epoch is not None:
            on_epoch(epoch + 1, epoch_loss,
                     total_delay_metric / n_seeds,
                     total_thru_metric  / n_seeds)

        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"epoch {epoch+1:4d}  loss={epoch_loss:.1f}  "
                  f"avg_arrived={total_arr / n_seeds:.1f}{best_tag}")

    return net


if __name__ == "__main__":
    # Force line-buffered stdout so prints appear immediately in conda terminals.
    sys.stdout.reconfigure(line_buffering=True)

    toks = sys.argv[1:]

    def _tok(pfx, default):
        t = next((x for x in toks if x.startswith(pfx)), None)
        return t.split("=", 1)[1] if t is not None else default

    n_epochs    = int(_tok("epochs=", 100))
    lr          = float(_tok("lr=", "1e-4"))
    hidden      = int(_tok("hidden=", 32))
    n_heads     = int(_tok("heads=", 2))
    n_layers    = int(_tok("layers=", 1))
    n_seeds     = int(_tok("seeds=", 5))
    t_end       = float(_tok("t_end=", 200.0))
    loss_mode   = _tok("loss=", "both")
    print_every = int(_tok("print=", 10))
    best_ckpt   = _tok("model=", BEST_CKPT)
    vph = {
        "s": float(_tok("s=", 200)),
        "l": float(_tok("l=", 100)),
        "r": float(_tok("r=", 100)),
    }

    # ── live plots: delay metric (left) + throughput metric (right) ───────────
    try:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, (ax_d, ax_t) = plt.subplots(1, 2, figsize=(13, 4))

        ax_d.set_xlabel("Epoch")
        ax_d.set_ylabel("Weighted speed deficit · dt  (lower = better)")
        ax_d.set_title("Delay metric")
        _d_line,      = ax_d.plot([], [], "steelblue", lw=1.2, alpha=0.7, label="per epoch")
        _d_best_line, = ax_d.plot([], [], "navy",      lw=1.5, ls="--",   label="best (min)")
        _d_mean_line, = ax_d.plot([], [], "cornflowerblue", lw=1.5, ls=":", label="mean so far")
        ax_d.legend(fontsize=8)

        ax_t.set_xlabel("Epoch")
        ax_t.set_ylabel("Weighted mean speed · dt  (higher = better)")
        ax_t.set_title("Throughput metric")
        _t_line,      = ax_t.plot([], [], "tomato",     lw=1.2, alpha=0.7, label="per epoch")
        _t_best_line, = ax_t.plot([], [], "darkred",    lw=1.5, ls="--",   label="best (max)")
        _t_mean_line, = ax_t.plot([], [], "lightsalmon", lw=1.5, ls=":",   label="mean so far")
        ax_t.legend(fontsize=8)

        fig.suptitle("PhaseNet — training metrics")
        fig.tight_layout()

        _ep_hist  = []
        _d_hist,  _d_best_hist,  _d_mean_hist  = [], [], []
        _t_hist,  _t_best_hist,  _t_mean_hist  = [], [], []
        _d_best   = float("inf")
        _t_best   = float("-inf")

        def _on_epoch(ep, loss, delay_m, thru_m):
            global _d_best, _t_best
            _d_best = min(_d_best, delay_m)
            _t_best = max(_t_best, thru_m)
            _ep_hist.append(ep)
            _d_hist.append(delay_m);  _d_best_hist.append(_d_best)
            _t_hist.append(thru_m);   _t_best_hist.append(_t_best)
            _d_mean_hist.append(sum(_d_hist) / len(_d_hist))
            _t_mean_hist.append(sum(_t_hist) / len(_t_hist))
            _d_line.set_data(_ep_hist, _d_hist)
            _d_best_line.set_data(_ep_hist, _d_best_hist)
            _d_mean_line.set_data(_ep_hist, _d_mean_hist)
            _t_line.set_data(_ep_hist, _t_hist)
            _t_best_line.set_data(_ep_hist, _t_best_hist)
            _t_mean_line.set_data(_ep_hist, _t_mean_hist)
            ax_d.relim(); ax_d.autoscale_view()
            ax_t.relim(); ax_t.autoscale_view()
            fig.suptitle(
                f"Epoch {ep}  |  "
                f"delay={delay_m:.1f} (best={_d_best:.1f})  "
                f"thru={thru_m:.1f} (best={_t_best:.1f})  "
                f"opt-loss={loss:.1f}")
            fig.canvas.draw_idle()
            plt.pause(0.001)

        _has_plot = True
    except Exception as _e:
        print(f"[live plot unavailable: {_e}]", flush=True)
        _on_epoch  = None
        _has_plot  = False

    t0 = time.time()
    train(n_epochs=n_epochs, lr=lr, vph=vph, n_seeds=n_seeds,
          hidden=hidden, n_heads=n_heads, n_layers=n_layers,
          t_end=t_end, loss_mode=loss_mode,
          print_every=print_every, best_ckpt=best_ckpt,
          on_epoch=_on_epoch)
    print(f"\nTotal wall time: {time.time()-t0:.1f}s")

    if _has_plot:
        plt.ioff()
        plt.show()
