"""
social_force.py — ETA-based cross-traffic conflict resolution.

Works for all 12 movements (through, right turn, left turn) on any conflicting pair.

For each ego vehicle in a tracked movement stream:

  1. Find rivals in geometrically conflicting streams (from CONFLICT_MAP).

  2. Compute signed ETA to the junction centre for ego and rival:
         η = (cp − p) · ê / v        signed: negative = already past crossing
     Junction centre is used as the crossing point for all conflict pairs.
     This is exact for perpendicular through movements and a safe approximation
     for same-axis and turn conflicts in a single-lane network.

  3. If |η_i − η_j| < SAFE_GAP (both arrive within SAFE_GAP seconds of each other):
         urgency  = 1 − |η_i − η_j| / SAFE_GAP  ∈ (0, 1]
         yielder  (η_i > η_j): a_ij = −A_CROSS · urgency · (1 − β · ratio)
         passer   (η_i < η_j): a_ij = 0   (no boost)
     where ratio = P_ego / (P_ego + P_rival + ε) and P is stream packing pressure.

  4. Aggregate: a_social = min(all a_ij)   — most urgent brake wins
               μ_social  = max(all urgency)

  5. Controller uses:  a = min(a_wp, a_social)   (hard override, no blending)

Platoon packing pressure (P_stream):
  P = Σ_i  v_i · max(0, 1 − gap_behind_i / L_GAP)
  for approaching vehicles within D_WINDOW of the junction.
  gap_behind = bumper-to-bumper distance to the next vehicle behind in the same stream.
  A lone vehicle has no follower → gap = ∞ → w = 0 → P = 0.
  A tight, fast platoon has high P → reduces yield force on its own vehicles.
  Two isolated vehicles → both P = 0 → ratio ≈ 0 → original ETA-only behavior.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from conflict import CONFLICT_MAP, ConflictSnapshot

# ── all 12 tracked movement streams ───────────────────────────────────────────
_ALL_MOVEMENTS = frozenset({
    ("east_in",  "west_out"),   # EW_T  through
    ("east_in",  "north_out"),  # EW_R  right
    ("east_in",  "south_out"),  # EW_L  left
    ("west_in",  "east_out"),   # WE_T  through
    ("west_in",  "south_out"),  # WE_R  right
    ("west_in",  "north_out"),  # WE_L  left
    ("north_in", "south_out"),  # NS_T  through
    ("north_in", "west_out"),   # NS_R  right
    ("north_in", "east_out"),   # NS_L  left
    ("south_in", "north_out"),  # SN_T  through
    ("south_in", "east_out"),   # SN_R  right
    ("south_in", "west_out"),   # SN_L  left
})

# Legacy alias — kept so diagnostic/benchmark scripts that import _THROUGH still work
_THROUGH = frozenset({
    ("east_in",  "west_out"),
    ("west_in",  "east_out"),
    ("north_in", "south_out"),
    ("south_in", "north_out"),
})

# ── hyperparameters ────────────────────────────────────────────────────────────
SAFE_GAP          = 5.0  # s   detection window: conflict force activates within this ETA gap
YIELD_CONF_SCALE  = 2.0  # s   ETA advantage at which platoon softening reaches full effect
                          #     Decoupled from SAFE_GAP so wider detection doesn't suppress
                          #     platoon physics at any given absolute delta.
A_CROSS    = 3.0  # m/s²  social braking anchor  (a_k for conflict correction)
A_PUSH     = 3.0  # m/s²  priority push anchor   (a_k for platoon throughput correction)
TURN_GATE  = SAFE_GAP    # s   rival-ETA window for turn-priority yield rule

# ── in-junction emergency: variable anchor for stopped/slow rivals ─────────────
# Once inside the junction, ETA is undefined.  We fall back to proximity-based
# detection and make A_k a function of the rival's speed (variable consequent):
#   A_k = A_CROSS + (A_JCT_MAX - A_CROSS) * (1 − v_rival / V_SLOW_JCT)
# so A_k = A_CROSS for a moving rival and A_JCT_MAX for a fully stopped one.
_D_JCT_EMERG = 20.0  # m    proximity window — detection starts this far out
_V_SLOW_JCT  = 2.0   # m/s  rival below this speed activates variable anchor
_A_JCT_MAX   = 6.0   # m/s² peak variable anchor (v_rival = 0)

# ── turn-priority yield rules ──────────────────────────────────────────────────
# Left turns yield to oncoming through traffic (standard right-of-way).
# Two right turns yield to the crossing through that physically blocks their path
# (from network priority matrix: E→N right/S→N through, W→S right/N→S through).
# N→W right and S→E right have free pass in the network (no conflict with priority
# through movements) and are intentionally omitted.
#
# Membership function:  μ_turn = max(0, 1 − η_rival / TURN_GATE)
# where η_rival is the rival through vehicle's ETA to the junction centre.
# Unlike mu_cf (which uses |delta|), this fires on the rival's absolute ETA so
# the turning vehicle yields regardless of whether it arrives first or second.
_TURN_YIELD_MAP: dict[tuple, frozenset] = {
    # left turns — yield to oncoming through
    ("north_in", "east_out"):  frozenset({("south_in", "north_out")}),
    ("south_in", "west_out"):  frozenset({("north_in", "south_out")}),
    ("east_in",  "south_out"): frozenset({("west_in",  "east_out")}),
    ("west_in",  "north_out"): frozenset({("east_in",  "west_out")}),
    # right turns — yield to crossing through (from network priority matrix)
    ("east_in",  "north_out"): frozenset({("south_in", "north_out")}),
    ("west_in",  "south_out"): frozenset({("north_in", "south_out")}),
}

# Platoon packing force
L_GAP      = 12.0  # m   characteristic inter-vehicle spacing; gap ≥ L_GAP → no packing
D_WINDOW   = 80.0  # m   distance from junction within which vehicles count toward pressure
BETA       = 0.70  # max fraction of yield force a platoon can cancel (0 = off, 1 = full)
EPSILON_P  = 1.0   # regularisation so two isolated vehicles don't ratio to 0.5
VEHICLE_LEN = 5.0  # m   approximate vehicle length for bumper-to-bumper gap

# Stream summary normalisation (used by transformer summary token)
_V_REF  = 13.89          # m/s  reference speed — matches V_MAX in the physics layer
P_SCALE = 2.0 * _V_REF   # saturation at ~3 tight fast vehicles; 6 * → too weak for priority gate
N_SCALE = 6.0             # normalise approaching-vehicle count against 6 slots

# Junction centre — crossing point used for all conflict pairs
_CX = 200.0
_CY = 200.0


# ── SUMO angle → 2-D unit heading vector ──────────────────────────────────────
def _heading(angle_deg: float) -> tuple[float, float]:
    """SUMO convention: 0 = North, clockwise.  Returns (east, north) unit vec."""
    r = np.radians(angle_deg)
    return float(np.sin(r)), float(np.cos(r))


# ── geometric helpers (kept for backward compat / diag scripts) ────────────────
def _crossing_point(
    px_i: float, py_i: float, ex_i: float, ey_i: float,
    px_j: float, py_j: float,
) -> tuple[float, float]:
    """Lane-crossing point for EW × NS perpendicular pairs (legacy)."""
    if abs(ex_i) >= abs(ey_i):
        return px_j, py_i     # EW ego: rival's x, ego's y
    else:
        return px_i, py_j     # NS ego: ego's x, rival's y


def _eta_signed(
    px: float, py: float, ex: float, ey: float, spd: float,
    cx: float, cy: float,
) -> float:
    """
    Signed ETA to crossing point (cx, cy).
    Positive = still approaching; negative = already past (seconds ago).
    """
    dist_along = (cx - px) * ex + (cy - py) * ey
    return dist_along / max(spd, 0.1)


# Legacy aliases used by diag scripts
def _eta_to_cp(
    px: float, py: float, ex: float, ey: float, spd: float,
    cx: float, cy: float,
) -> float:
    return max(0.0, _eta_signed(px, py, ex, ey, spd, cx, cy))


def _eta_weight(eta_i: float, eta_j: float, sigma: float = 0.08) -> float:
    """Sigmoid weight — kept for diag compatibility."""
    return float(1.0 / (1.0 + np.exp(-(eta_i - eta_j) / sigma)))


# ── platoon packing pressure ───────────────────────────────────────────────────
def _stream_pressure(
    stream:   tuple,
    snapshot: "ConflictSnapshot",
    pos:      dict,
    spd_d:    dict,
    in_jct:   dict,
) -> float:
    """
    Packing pressure for one stream.

    Sort approaching vehicles by distance to junction (ascending = closest first).
    For each vehicle k, gap_behind = centre-to-centre distance to vehicle k+1
    minus VEHICLE_LEN.  Last vehicle in line has no follower → gap = inf → w = 0.

    P = Σ_k  v_k · max(0, 1 − gap_behind_k / L_GAP)

    Properties:
      - Lone vehicle:  gap = inf → w = 0  →  P = 0         (no platoon effect)
      - Tight platoon: small gap → w ≈ 1  →  P ≈ N · v     (strong effect)
      - Slow queue:    small gap but low v →  P small        (speed gate)
    """
    svids = snapshot.stream_vehicles.get(stream, [])
    if not svids:
        return 0.0

    # Collect approaching vehicles within D_WINDOW of junction
    approaching: list[tuple[float, str]] = []
    for vid in svids:
        if in_jct.get(vid, False) or vid not in pos:
            continue
        px, py = pos[vid]
        d = math.sqrt((_CX - px) ** 2 + (_CY - py) ** 2)
        if d <= D_WINDOW:
            approaching.append((d, vid))

    if len(approaching) < 2:
        return 0.0  # need at least two vehicles for a gap to exist

    # Closest to junction first
    approaching.sort(key=lambda x: x[0])

    total = 0.0
    for k, (d_k, vid_k) in enumerate(approaching):
        v_k = spd_d.get(vid_k, 0.0)
        if v_k < 0.5:
            continue  # stationary vehicles carry no momentum

        if k + 1 < len(approaching):
            d_next = approaching[k + 1][0]
            gap_behind = max(0.0, (d_next - d_k) - VEHICLE_LEN)
        else:
            gap_behind = float("inf")  # trailing vehicle, no follower

        w = max(0.0, 1.0 - gap_behind / L_GAP)
        total += v_k * w

    return total


# ── shared SUMO query (called once per step, shared by social force + summary) ─
def _query_state(
    all_ids: list[str],
) -> tuple[dict, dict, dict, dict]:
    """Batch-query SUMO for position, speed, heading, and junction status."""
    import traci
    pos: dict[str, tuple[float, float]] = {}
    spd_d: dict[str, float]             = {}
    e_hat: dict[str, tuple[float, float]] = {}
    in_jct: dict[str, bool]             = {}
    for vid in all_ids:
        px, py    = traci.vehicle.getPosition(vid)
        spd       = traci.vehicle.getSpeed(vid)
        ex, ey    = _heading(traci.vehicle.getAngle(vid))
        road      = traci.vehicle.getRoadID(vid)
        pos[vid]  = (px, py)
        spd_d[vid] = spd
        e_hat[vid] = (ex, ey)
        in_jct[vid] = road.startswith(":center")
    return pos, spd_d, e_hat, in_jct


# ── stream summary for transformer summary token ───────────────────────────────
def _stream_summary(
    ego_id:          str,
    ego_stream:      tuple,
    snapshot:        "ConflictSnapshot",
    pos:             dict,
    spd_d:           dict,
    in_jct:          dict,
    stream_pressure: dict,
) -> list[float]:
    """
    4-dim normalised own-queue context for the transformer summary token.

      [0] P_own / P_SCALE   own stream packing pressure     ∈ [0, 1]
      [1] n_own / N_SCALE   approaching vehicles, own stream ∈ [0, 1]
      [2] mean_v / _V_REF   mean speed of own queue
      [3] v_follower/_V_REF immediate follower speed

    Rival context is returned separately as per-stream tokens by
    compute_social_force_2d (see rival_tokens in its return tuple).
    """
    P     = stream_pressure.get(ego_stream, 0.0)
    svids = snapshot.stream_vehicles.get(ego_stream, [])

    approaching: list[tuple[float, str, float]] = []
    for vid_k in svids:
        if in_jct.get(vid_k, False) or vid_k not in pos:
            continue
        px, py = pos[vid_k]
        d = math.sqrt((_CX - px) ** 2 + (_CY - py) ** 2)
        approaching.append((d, vid_k, spd_d.get(vid_k, 0.0)))
    approaching.sort()

    n      = len(approaching)
    mean_v = sum(v for _, _, v in approaching) / max(n, 1)

    ego_idx = next(
        (i for i, (_, id_, _) in enumerate(approaching) if id_ == ego_id), None
    )
    if ego_idx is not None and ego_idx + 1 < n:
        v_follower = approaching[ego_idx + 1][2]
    else:
        v_follower = spd_d.get(ego_id, 0.0)

    return [
        min(P / max(P_SCALE, 1.0), 1.0),
        min(n / N_SCALE, 1.0),
        mean_v     / _V_REF,
        v_follower / _V_REF,
    ]


def _build_rival_tokens(
    c_streams:       frozenset,
    snapshot:        "ConflictSnapshot",
    pos:             dict,
    spd_d:           dict,
    in_jct:          dict,
    stream_pressure: dict,
    mu_per_stream:   dict,
) -> list[list[float]]:
    """
    One 4-dim token per conflicting stream that has vehicles present or urgency > 0.

      [0] P_k / P_SCALE   stream packing pressure        ∈ [0, 1]
      [1] n_k / N_SCALE   approaching vehicles in stream  ∈ [0, 1]
      [2] mean_v_k/_V_REF mean speed of stream
      [3] mu_k            worst conflict gate from this stream ∈ [0, 1]

    Empty streams (no vehicles, no urgency) are omitted — variable length is the
    transformer's strength. Callers pad to K_MAX with zero rows.
    """
    tokens = []
    for rs in c_streams:
        rs_vids = snapshot.stream_vehicles.get(rs, [])
        rs_spds = [
            spd_d.get(v, 0.0)
            for v in rs_vids
            if not in_jct.get(v, False) and v in pos
        ]
        n_k   = len(rs_spds)
        mv_k  = sum(rs_spds) / max(n_k, 1)
        P_k   = stream_pressure.get(rs, 0.0)
        mu_k  = mu_per_stream.get(rs, 0.0)
        if n_k > 0 or mu_k > 0:
            tokens.append([
                min(P_k / max(P_SCALE, 1.0), 1.0),
                min(n_k / N_SCALE, 1.0),
                mv_k / _V_REF,
                mu_k,
            ])
    return tokens


# ── main API ───────────────────────────────────────────────────────────────────
def compute_social_force_2d(
    tracked:  list[str],
    snapshot: ConflictSnapshot,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ETA-based cross-traffic force for each vehicle in `tracked`.

    Handles all 12 movement streams (through, right, left).
    Crossing point = junction centre for all conflict pairs.

    Returns
    -------
    a_social  : Tensor [N]   braking correction (m/s²), ≤ 0
    mu_social : Tensor [N]   urgency ∈ [0,1]
    """
    N = len(tracked)

    # ── query SUMO once for all tracked + rival vehicles ─────────────────────
    all_ids = [v for v, s in snapshot.vehicle_stream.items() if s is not None]
    pos, spd_d, e_hat, in_jct = _query_state(all_ids)

    # Packing pressure per stream (used in yield softening and summary token)
    stream_pressure: dict = {
        s: _stream_pressure(s, snapshot, pos, spd_d, in_jct)
        for s in _ALL_MOVEMENTS
    }

    a_out      = [0.0] * N
    mu_out     = [0.0] * N
    mu_prio_out = [0.0] * N        # priority push gate [N]
    sum_out    = [[0.0] * 4] * N   # own-stream summary token [N, 4]
    rival_out: list[list[list[float]]] = [[] for _ in range(N)]  # variable K per vehicle

    for i, ego_id in enumerate(tracked):
        ego_stream = snapshot.vehicle_stream.get(ego_id)
        if ego_stream not in _ALL_MOVEMENTS or ego_id not in pos:
            continue

        # Own-stream summary: computed regardless of junction state so the model
        # always has queue context even while clearing the junction.
        sum_out[i] = _stream_summary(
            ego_id, ego_stream, snapshot, pos, spd_d, in_jct, stream_pressure
        )

        c_streams = CONFLICT_MAP.get(ego_stream, frozenset()) & _ALL_MOVEMENTS
        if not c_streams:
            continue

        px_i, py_i = pos[ego_id]
        ex_i, ey_i = e_hat[ego_id]
        spd_i      = spd_d[ego_id]

        # Inside the junction ETA is undefined.  Detect slow/stopped rivals by
        # proximity and apply a variable emergency brake: A_k scales with rival
        # slowness so the anchor is fixed for fast traffic and maximal for stopped.
        if in_jct.get(ego_id, False):
            best_a = 0.0; best_mu = 0.0
            for rival_id, rival_stream in snapshot.vehicle_stream.items():
                if rival_id == ego_id or rival_stream not in c_streams:
                    continue
                if rival_id not in pos:
                    continue
                v_rival = spd_d.get(rival_id, 0.0)
                if v_rival >= _V_SLOW_JCT:
                    continue
                dist_ij = math.sqrt(
                    (px_i - pos[rival_id][0]) ** 2 + (py_i - pos[rival_id][1]) ** 2
                )
                if dist_ij >= _D_JCT_EMERG:
                    continue
                urgency     = 1.0 - dist_ij / _D_JCT_EMERG
                slow_factor = 1.0 - v_rival / _V_SLOW_JCT
                A_k  = A_CROSS + (_A_JCT_MAX - A_CROSS) * slow_factor
                a_ij = -A_k * urgency
                mu_ij = slow_factor * urgency        # conflict gate fed to kernel
                if a_ij < best_a:
                    best_a  = a_ij
                    best_mu = mu_ij
            a_out[i]  = best_a
            mu_out[i] = 0.0   # keep mu_cf=0 so turning/struct anchors stay at full strength
            continue

        best_a           = 0.0
        best_mu          = 0.0
        best_mu_priority = 0.0
        mu_per_stream: dict = {}   # rival_stream → worst mu from that stream

        eta_i = _eta_signed(px_i, py_i, ex_i, ey_i, spd_i, _CX, _CY)

        if eta_i < -SAFE_GAP:
            continue

        for rival_id, rival_stream in snapshot.vehicle_stream.items():
            if rival_id == ego_id or rival_stream not in c_streams:
                continue
            if rival_id not in pos:
                continue

            px_j, py_j = pos[rival_id]
            ex_j, ey_j = e_hat[rival_id]
            spd_j      = spd_d[rival_id]

            eta_j = _eta_signed(px_j, py_j, ex_j, ey_j, spd_j, _CX, _CY)
            if in_jct.get(rival_id, False):
                eta_j = max(eta_j, 0.0)

            delta = eta_i - eta_j   # positive: ego arrives later → yield

            if abs(delta) >= SAFE_GAP:
                continue

            urgency = 1.0 - abs(delta) / SAFE_GAP

            if delta >= 0:           # ego is the yielder — apply braking correction
                yield_confidence = min(delta / YIELD_CONF_SCALE, 1.0)

                p_ego   = stream_pressure.get(ego_stream, 0.0)
                p_rival = stream_pressure.get(rival_stream, 0.0)
                ratio   = min(p_ego / (p_ego + p_rival + EPSILON_P), 1.0)
                mu_ij   = urgency * (1.0 - BETA * ratio * yield_confidence)
                a_ij    = -A_CROSS * mu_ij
                if a_ij < best_a:
                    best_a  = a_ij
                    best_mu = mu_ij
                # Track per-stream worst urgency for the rival token
                if mu_ij > mu_per_stream.get(rival_stream, 0.0):
                    mu_per_stream[rival_stream] = mu_ij

            else:                    # delta < 0: ego has priority — compute push gate
                prio_conf = min(-delta / YIELD_CONF_SCALE, 1.0)
                p_ego     = stream_pressure.get(ego_stream, 0.0)
                mu_prio_ij = urgency * prio_conf * min(p_ego / P_SCALE, 1.0)
                if mu_prio_ij > best_mu_priority:
                    best_mu_priority = mu_prio_ij

        # Turn-priority gate — turning vehicles yield to designated through movements
        # regardless of delta sign.  μ_turn = 1 − η_rival/TURN_GATE fires on the
        # rival's absolute ETA, not the difference, so the turn always gives way.
        # Result folds into best_a/best_mu (same A_CROSS anchor → same correction slot).
        priority_rivals = _TURN_YIELD_MAP.get(ego_stream)
        if priority_rivals and not in_jct.get(ego_id, False):
            for rival_id2, rival_stream2 in snapshot.vehicle_stream.items():
                if rival_stream2 not in priority_rivals or rival_id2 not in pos:
                    continue
                px_j2, py_j2 = pos[rival_id2]
                ex_j2, ey_j2 = e_hat[rival_id2]
                eta_j2 = _eta_signed(px_j2, py_j2, ex_j2, ey_j2, spd_d[rival_id2], _CX, _CY)
                if in_jct.get(rival_id2, False):
                    eta_j2 = max(eta_j2, 0.0)
                if eta_j2 >= TURN_GATE:
                    continue
                mu_turn = max(0.0, 1.0 - eta_j2 / TURN_GATE)
                a_turn  = -A_CROSS * mu_turn
                if a_turn < best_a:
                    best_a  = a_turn
                    best_mu = max(best_mu, mu_turn)
                elif mu_turn > best_mu:
                    best_mu = mu_turn
                # propagate to rival token so transformer sees the conflict
                if mu_turn > mu_per_stream.get(rival_stream2, 0.0):
                    mu_per_stream[rival_stream2] = mu_turn

        a_out[i]        = best_a
        mu_out[i]       = best_mu
        mu_prio_out[i]  = best_mu_priority
        rival_out[i] = _build_rival_tokens(
            c_streams, snapshot, pos, spd_d, in_jct, stream_pressure, mu_per_stream
        )

    return (
        torch.tensor(a_out,      dtype=torch.float32),  # [N]    social braking (m/s²)
        torch.tensor(mu_out,     dtype=torch.float32),  # [N]    conflict gate ∈ [0,1]
        torch.tensor(mu_prio_out, dtype=torch.float32), # [N]    priority push gate ∈ [0,1]
        torch.tensor(sum_out,    dtype=torch.float32),  # [N,4]  own-stream summary
        rival_out,                                      # List[List[List[float]]] variable K
    )
