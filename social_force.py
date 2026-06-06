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
import traci

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
SAFE_GAP = 3.5    # s    required time gap between arrivals at crossing
A_CROSS  = 8.0    # m/s² max cross-traffic braking

# Platoon packing force
L_GAP      = 12.0  # m   characteristic inter-vehicle spacing; gap ≥ L_GAP → no packing
D_WINDOW   = 80.0  # m   distance from junction within which vehicles count toward pressure
BETA       = 0.70  # max fraction of yield force a platoon can cancel (0 = off, 1 = full)
EPSILON_P  = 1.0   # regularisation so two isolated vehicles don't ratio to 0.5
VEHICLE_LEN = 5.0  # m   approximate vehicle length for bumper-to-bumper gap

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

    pos:       dict[str, tuple[float, float]] = {}
    spd_d:     dict[str, float]               = {}
    e_hat:     dict[str, tuple[float, float]] = {}
    in_jct:    dict[str, bool]                = {}   # True if on junction internal lane

    for vid in all_ids:
        px, py    = traci.vehicle.getPosition(vid)
        spd       = traci.vehicle.getSpeed(vid)
        ex, ey    = _heading(traci.vehicle.getAngle(vid))
        road      = traci.vehicle.getRoadID(vid)
        pos[vid]  = (px, py)
        spd_d[vid] = spd
        e_hat[vid] = (ex, ey)
        in_jct[vid] = road.startswith(":center")

    # Precompute packing pressure for every stream (used in yield softening)
    stream_pressure: dict = {
        s: _stream_pressure(s, snapshot, pos, spd_d, in_jct)
        for s in _ALL_MOVEMENTS
    }

    a_out  = [0.0] * N
    mu_out = [0.0] * N

    for i, ego_id in enumerate(tracked):
        ego_stream = snapshot.vehicle_stream.get(ego_id)
        if ego_stream not in _ALL_MOVEMENTS or ego_id not in pos:
            continue

        # Once a vehicle is inside the junction it is committed to its path.
        # Social force only gates entry; IDM handles clearance from within.
        if in_jct.get(ego_id, False):
            continue

        # Only conflict with other _ALL_MOVEMENTS streams
        c_streams = CONFLICT_MAP.get(ego_stream, frozenset()) & _ALL_MOVEMENTS
        if not c_streams:
            continue

        px_i, py_i = pos[ego_id]
        ex_i, ey_i = e_hat[ego_id]
        spd_i      = spd_d[ego_id]

        best_a  = 0.0
        best_mu = 0.0

        # ETA to junction centre
        eta_i = _eta_signed(px_i, py_i, ex_i, ey_i, spd_i, _CX, _CY)

        # Skip if ego already well past the junction
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

            # ETA to junction centre for rival.
            # Turning vehicles change heading mid-junction, making heading-based
            # ETA go negative while they're still physically blocking.
            # Cap at 0: a rival IN the junction is treated as "at the crossing now".
            eta_j = _eta_signed(px_j, py_j, ex_j, ey_j, spd_j, _CX, _CY)
            if in_jct.get(rival_id, False):
                eta_j = max(eta_j, 0.0)

            delta = eta_i - eta_j   # positive: ego arrives later → yield

            if abs(delta) >= SAFE_GAP:
                continue             # sufficient gap, no action

            urgency = 1.0 - abs(delta) / SAFE_GAP

            if delta >= 0:           # ego is the yielder
                # Platoon softening: packed, fast ego stream yields less.
                # ratio → 0 for lone vehicles (P_ego=0) → original force.
                # ratio → 1 when ego platoon >> rival  → force reduced by BETA.
                p_ego   = stream_pressure.get(ego_stream, 0.0)
                p_rival = stream_pressure.get(rival_stream, 0.0)
                ratio   = min(p_ego / (p_ego + p_rival + EPSILON_P), 1.0)
                a_ij    = -A_CROSS * urgency * (1.0 - BETA * ratio)
                if a_ij < best_a:
                    best_a  = a_ij
                    best_mu = urgency
            # passer (delta < 0): no force

        a_out[i]  = best_a
        mu_out[i] = best_mu

    return (
        torch.tensor(a_out,  dtype=torch.float32),
        torch.tensor(mu_out, dtype=torch.float32),
    )
