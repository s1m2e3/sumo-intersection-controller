"""
ttc.py — projected Time-to-Collision surface

For every inbound vehicle classified by a ConflictSnapshot, projects dynamics
forward `horizon` steps (default 100 = 10 s at dt=0.1 s) using the model
(HybridModel f+h, or IDMModel) under torch.no_grad(), then computes:

    TTC(t, j)   shape (horizon, num_conflict_streams)

where
  t  — projected timestep index  (0 … horizon−1)
  j  — index into this vehicle's sorted list of conflicting streams

Value semantics:
  + : vehicle i arrives at conflict point AFTER closest rival in stream j
        → stream j leads; i has time to yield
  − : vehicle i arrives BEFORE stream j
        → i is the leader; stream j may collide into i
  ≈0 : simultaneous arrival → collision risk
  INF_TTC : stream j has no inbound vehicles in the current snapshot

Conflict-point geometry:
  Distance from junction stop-line to conflict point is approximated per
  movement type via CP_OFFSETS (tunable). To calibrate, query junction lane
  shapes from the SUMO net or traci.lane.getShape().
"""

from __future__ import annotations
from typing import NamedTuple

import torch
import torch.nn as nn
import traci

from conflict import ConflictSnapshot, CONFLICT_MAP, STREAM_NAMES, Movement, _OUTGOING
import traci_cache

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

HORIZON  = 30     # projection steps  (6 s at PROJ_DT = 0.2 s)
INF_TTC  = 999.0  # sentinel: no rival present or vehicle already past CP
_EPS_V   = 0.1   # speed floor for ETA denominator (m/s)
_ARM_LEN = 200.0  # incoming edge length (m) — must match simulator.py arm_length

# Distance from the junction stop-line to the conflict point (m), per movement type.
#
# Two regimes depending on conflict type:
#
#   CROSSING conflict (paths physically intersect inside the box):
#     T through  : ~7 m  — crossing near junction centre
#     R right    : ~4 m  — short corner path, conflict early
#     L left     : ~9 m  — curves through centre before crossing
#
#   MERGING conflict (both streams share the same exit edge):
#     T through  : ~14 m — full internal through-lane length
#     R right    : ~8 m  — right-turn internal lane
#     L left     : ~22 m — left-turn sweeps the full junction
#
# Override after calibrating against the actual SUMO junction geometry
# (e.g. via traci.lane.getShape on the internal lanes).
CP_CROSS: dict[str, float] = {"T": 7.0, "R": 4.0, "L": 9.0}
CP_MERGE: dict[str, float] = {"T": 14.0, "R": 8.0, "L": 22.0}

# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

class TTCSurface(NamedTuple):
    """
    TTC surface for one vehicle at the current simulation timestep.

    vehicle_id       : SUMO vehicle ID
    stream           : (from_edge, to_edge) for this vehicle
    ttc              : list of Tensors, one per conflict stream j
                         ttc[j] has shape (horizon, K_j)
                         where K_j = number of vehicles currently in stream j
                         ttc[j][t, k] = |eta_i(t) - eta_k(t)|
                                        absolute time gap at the conflict point
                                        between this vehicle and rival k in
                                        stream j, evaluated at projected step t
                         ≈ 0  : simultaneous arrival → collision risk
                         < 3  : dangerously close (see collect_at_risk)
                         large: safe separation
    conflict_streams : movement for each entry in ttc (sorted for determinism)
    rival_ids        : rival_ids[j] lists the vehicle IDs for ttc[j]'s columns
    """
    vehicle_id:       str
    stream:           Movement
    ttc:              list[torch.Tensor]
    conflict_streams: list[Movement]
    rival_ids:        list[list[str]]


class AtRiskPair(NamedTuple):
    """
    A vehicle pair whose minimum projected |TTC| falls below the threshold,
    meaning they are predicted to be dangerously close at the conflict point
    at some point within the projection horizon.

    ego_id       : ego vehicle ID  (i)
    ego_stream   : movement stream of the ego
    rival_id     : rival vehicle ID (k in stream j)
    rival_stream : movement stream of the rival
    min_ttc      : min over t of ttc[j][t, k]  — smallest time gap found
    t_min        : projected step index at which min_ttc occurs
    """
    ego_id:       str
    ego_stream:   Movement
    rival_id:     str
    rival_stream: Movement
    min_ttc:      float
    t_min:        int

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mvt_type(stream: Movement) -> str:
    """Last character of the stream name: 'T', 'R', or 'L'."""
    name = STREAM_NAMES.get(stream, "")
    return name[-1] if name else "T"


def _cp_pair_offsets(a: Movement, b: Movement) -> tuple[float, float]:
    """
    Returns (d_cp_a, d_cp_b): distances from each stream's stop-line to the
    shared conflict point CP(a, b).

    Merging conflict  (a and b share the same exit edge):
      both approach the same exit point — use full internal-lane lengths.
    Crossing conflict (paths intersect inside the junction box):
      both approach the crossing point — use centre-of-box distances.
    """
    if a[1] == b[1]:          # same to_edge → merging
        table = CP_MERGE
    else:                      # paths cross  → crossing
        table = CP_CROSS
    return table[_mvt_type(a)], table[_mvt_type(b)]


def signed_dist_to_cp(vid: str, stream: Movement, rival_stream: Movement) -> float:
    """
    Signed distance of `vid` (travelling in `stream`) to the conflict point
    it shares with `rival_stream`, measured along `vid`'s own path.

      > 0  vehicle is still approaching CP
        0  vehicle is exactly at CP
      < 0  vehicle has passed CP

    Use this at each real simulation step to track the approach/departure
    trajectory: the value decreases from a large positive number, crosses
    zero at the conflict point, and becomes negative afterwards.

    The absolute separation  |signed_dist(ego) − signed_dist(rival)|
    traces a parabola-like shape: large when both are far, minimum near
    simultaneous arrival, large again once one vehicle has cleared.
    """
    off_i, _ = _cp_pair_offsets(stream, rival_stream)
    road = traci.vehicle.getRoadID(vid)
    pos  = traci.vehicle.getLanePosition(vid)

    if road in {"east_in", "west_in", "north_in", "south_in"}:
        # on the approach arm: distance to stop-line + distance inside junction to CP
        return (_ARM_LEN - pos) + off_i
    elif road.startswith(":center"):
        # inside the junction box: off_i - pos_on_internal_lane
        return off_i - pos
    else:
        # on outgoing edge: vehicle has cleared the junction
        return -(_ARM_LEN + pos)


def _query_d_to_junction(vehicle_ids: list[str]) -> dict[str, float]:
    """
    Signed distance (m) from each vehicle's current position to the junction
    stop-line, along the vehicle's path.

      > 0  vehicle is still on the approach arm  (dist to stop-line)
        0  vehicle is exactly at the stop-line
      < 0  vehicle is already inside the junction box (−pos on internal lane)

    Combined with CP_CROSS / CP_MERGE offsets this gives the remaining distance
    to the conflict point:
        d_remaining = max(0,  d_junc + cp_offset  −  cum_dist_projected)
    For internal-lane vehicles: d_junc = −pos, so d_remaining = cp_offset − pos,
    which is 0 once pos ≥ cp_offset (vehicle has passed CP).
    """
    out: dict[str, float] = {}
    for vid in vehicle_ids:
        road = traci_cache.get_road_id(vid)
        pos  = traci_cache.get_lane_pos(vid)
        if road.startswith(":center"):
            out[vid] = -pos
        elif road in _OUTGOING:
            # Vehicle has exited the junction and is on the exit arm.
            # The merge conflict point is at the junction exit (pos=0 on the exit arm).
            # Setting d_junc to a large negative guarantees d_cp = d_junc + any_offset < 0,
            # so d_rem=0 and eta=0 — the vehicle is treated as already at (past) the CP.
            # The occ term (L_OCC/v) then correctly charges approaching vehicles for the
            # time remaining until this vehicle's rear clears the merge zone.
            out[vid] = -(_ARM_LEN + 100.0)
        else:
            out[vid] = max(0.0, _ARM_LEN - pos)
    return out


def calibrate_cp_offsets(junction_id: str = "center") -> None:
    """
    Query actual internal lane lengths from SUMO and update CP_CROSS / CP_MERGE
    in-place.  Call once after traci.start(), before build_ttc_surfaces().

    For each movement stream, finds the SUMO internal lane that connects
    from_edge → to_edge via the junction, reads its length, and sets:
      CP_CROSS[type] = avg_length × 0.5   (crossing: conflict at lane midpoint)
      CP_MERGE[type] = avg_length × 1.0   (merging:  conflict at lane exit)

    If a stream's internal lane cannot be found, the default value is kept.
    """
    from conflict import STREAM_NAMES

    type_lengths: dict[str, list[float]] = {"T": [], "R": [], "L": []}

    for (from_edge, to_edge), name in STREAM_NAMES.items():
        mvt_type = name[-1]                              # "T", "R", or "L"
        lane_idx = 1 if mvt_type == "L" else 0          # left turns use lane 1
        from_lane = f"{from_edge}_{lane_idx}"

        try:
            for link in traci.lane.getLinks(from_lane):
                # link tuple: (successor_lane, ..., via_internal_lane, ...)
                # index 0 = next outgoing lane, index 4 = via (internal) lane
                successor, _, _, _, via_lane, *_ = link
                if successor.startswith(to_edge) and via_lane:
                    length = traci.lane.getLength(via_lane)
                    type_lengths[mvt_type].append(length)
                    break
        except Exception:
            pass   # keep default if TraCI call fails

    for mvt_type, lengths in type_lengths.items():
        if not lengths:
            continue
        avg = sum(lengths) / len(lengths)
        CP_CROSS[mvt_type] = avg * 0.5   # crossing: midpoint of internal lane
        CP_MERGE[mvt_type] = avg * 1.0   # merging:  full lane to exit


class ProjectionInfo(NamedTuple):
    """Auxiliary tensors produced by build_ttc_surfaces; needed by safety.py."""
    v_traj:   torch.Tensor      # [N, horizon]  projected speeds
    cum_dist: torch.Tensor      # [N, horizon]  cumulative distances
    d_junc:   torch.Tensor      # [N]           distance to junction stop-line
    idx_of:   dict[str, int]    # vehicle ID -> batch index
    tracked:  list[str]         # ordered vehicle IDs (aligns with N dimension)


def _project(
    model:      nn.Module,
    v0:         torch.Tensor,         # [N]
    gap0:       torch.Tensor,         # [N]
    vlead0:     torch.Tensor,         # [N]  external-leader speeds (constant fallback)
    x_seq0:     torch.Tensor | None,  # [N, seq_len, 3]  — None for IDMModel
    leader_idx: torch.Tensor,         # [N] long — index into batch, or -1 if external
    dt:         float,
    horizon:    int,
    delta_a:    torch.Tensor | None = None,  # [N, horizon] correction added at each step
    is_turning: torch.Tensor | None = None,  # [N] float 0/1 — passed to HybridModel
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Coupled rollout: every vehicle whose leader is also in the batch uses the
    leader's live projected speed at each step.  Only vehicles whose leader
    is outside the batch (leader_idx == -1) fall back to constant vlead0.

    This makes the gap evolution exact — the same dynamics that drive the
    real simulation are used for the projection, so predicted and actual
    trajectories match within the numerical precision of the Euler step.

    Returns
    -------
    v_traj   : [N, horizon]  speed at each projected step (m/s)
    cum_dist : [N, horizon]  cumulative distance at each projected step (m)
    """
    v     = v0.clone()
    gap   = gap0.clone()
    x_seq = x_seq0.clone() if x_seq0 is not None else None

    in_batch = leader_idx >= 0               # [N] bool
    safe_idx = leader_idx.clamp(min=0)       # avoid -1 index; guarded by in_batch mask

    N   = v.shape[0]
    dev = v.device
    v_traj   = torch.empty(N, horizon, device=dev)
    cum_dist = torch.empty(N, horizon, device=dev)
    total    = torch.zeros(N, device=dev)

    with torch.no_grad():
        for t in range(horizon):
            # effective leader speed: projected for in-batch, constant for external
            vlead_eff = vlead0.clone()
            if in_batch.any():
                vlead_eff[in_batch] = v[safe_idx][in_batch]

            if x_seq is not None:
                accel = (model(v, gap, vlead_eff, x_seq, is_turning)
                         if is_turning is not None
                         else model(v, gap, vlead_eff, x_seq))
            else:
                accel = model(v, gap, vlead_eff)
            if delta_a is not None:
                accel = accel + delta_a[:, t]
            v_next = torch.clamp(v + accel * dt, min=0.0)
            total  = total + v * dt

            v_traj[:, t]   = v_next
            cum_dist[:, t] = total

            gap = torch.clamp(gap + (vlead_eff - v) * dt, min=0.5)

            if x_seq is not None:
                x_seq[:, :-1, :] = x_seq[:, 1:, :].clone()
                x_seq[:, -1, 0]  = v_next
                x_seq[:, -1, 1]  = gap
                x_seq[:, -1, 2]  = vlead_eff

            v = v_next

    return v_traj, cum_dist

# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def build_ttc_surfaces(
    snapshot:      ConflictSnapshot,
    model:         nn.Module,
    v_dict:        dict[str, float],
    gap_dict:      dict[str, float],
    vlead_dict:    dict[str, float],
    x_seq_dict:    dict[str, torch.Tensor] | None = None,
    dt:            float = 0.1,
    horizon:       int   = HORIZON,
    delta_a_dict:  dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, TTCSurface], ProjectionInfo]:
    """
    Build TTC(t, j) surfaces for all classified inbound vehicles.

    Parameters
    ----------
    snapshot     : ConflictSnapshot from conflict.build_snapshot()
    model        : HybridModel (provide x_seq_dict) or IDMModel (x_seq_dict=None)
    v_dict       : vid -> current speed (m/s)
    gap_dict     : vid -> current gap to leader (m)
    vlead_dict   : vid -> leader speed (m/s)
    x_seq_dict   : vid -> Tensor(seq_len, 3) observation history; None for IDMModel
    dt           : simulation timestep (s)
    horizon      : projection steps (default 100 = 10 s)

    Returns
    -------
    dict[vehicle_id -> TTCSurface]
      TTCSurface.ttc has shape (horizon, num_conflict_streams_for_this_vehicle)

    Usage
    -----
    snap     = conflict.build_snapshot(list(traci.vehicle.getIDList()))
    surfaces = build_ttc_surfaces(snap, model, v_d, gap_d, vlead_d, xseq_d)

    for vid, surf in surfaces.items():
        for j, cs in enumerate(surf.conflict_streams):
            # surf.ttc[j] : Tensor (horizon, K_j)
            # surf.ttc[j][t, k] = signed TTC between vid and rival k in stream cs
            #   at projected step t
            #   + → rival k arrives first  (vid can yield)
            #   − → vid arrives first       (vid is the leader)
            #   ≈0 → collision risk
            print(vid, STREAM_NAMES[surf.stream], "vs", STREAM_NAMES[cs],
                  surf.ttc[j].shape, "rivals:", surf.rival_ids[j])
    """
    # only vehicles with a known stream that has at least one active conflict
    tracked = [
        vid for vid, mvmt in snapshot.vehicle_stream.items()
        if mvmt is not None and CONFLICT_MAP.get(mvmt)
    ]
    if not tracked:
        return {}

    # ── batch-project all tracked vehicles together ──────────────────────────
    v0     = torch.tensor([v_dict[i]     for i in tracked], dtype=torch.float32)
    gap0   = torch.tensor([gap_dict[i]   for i in tracked], dtype=torch.float32)
    vlead0 = torch.tensor([vlead_dict[i] for i in tracked], dtype=torch.float32)
    x_seq0 = (torch.stack([x_seq_dict[i] for i in tracked])
               if x_seq_dict else None)   # [N, seq_len, 3] or None

    # ── index lookup (needed by both leader_idx and per-vehicle loop) ────────
    idx_of = {vid: k for k, vid in enumerate(tracked)}

    # ── coupled leader index ─────────────────────────────────────────────────
    # For each tracked vehicle, find whether its SUMO leader is also in the
    # batch.  If so, store its batch index so _project can use the live
    # projected speed instead of a constant fallback.
    leader_idx = torch.full((len(tracked),), -1, dtype=torch.long)
    for i, vid in enumerate(tracked):
        info = traci.vehicle.getLeader(vid)
        if info is not None:
            lid, _ = info
            if lid in idx_of:
                leader_idx[i] = idx_of[lid]

    # ── build delta_a tensor aligned with tracked order ─────────────────────
    if delta_a_dict:
        delta_a = torch.stack([
            delta_a_dict.get(vid, torch.zeros(horizon)) for vid in tracked
        ])  # [N, horizon]
    else:
        delta_a = None

    v_traj, cum_dist = _project(model, v0, gap0, vlead0, x_seq0,
                                 leader_idx, dt, horizon, delta_a=delta_a)
    # both: [N, horizon]

    # ── distance to junction stop-line for each vehicle ──────────────────────
    d_raw  = _query_d_to_junction(tracked)
    d_junc = torch.tensor([d_raw[i] for i in tracked], dtype=torch.float32)  # [N]

    # ── build per-vehicle TTC surface ────────────────────────────────────────
    results: dict[str, TTCSurface] = {}

    for vid in tracked:
        k      = idx_of[vid]
        stream = snapshot.vehicle_stream[vid]

        # deterministic column ordering: sort conflict streams by name
        conf_streams = sorted(
            CONFLICT_MAP[stream], key=lambda s: STREAM_NAMES.get(s, "")
        )

        ttc_cols:  list[torch.Tensor] = []
        rival_ids: list[list[str]]   = []

        for cs in conf_streams:
            rivals = [r for r in snapshot.stream_vehicles.get(cs, [])
                      if r in idx_of]

            if not rivals:
                # no vehicles in this stream right now — placeholder column
                ttc_cols.append(torch.full((horizon, 1), INF_TTC))
                rival_ids.append([])
                continue

            # CP distances are pair-specific: off_i differs per conflict stream j
            off_i, off_j = _cp_pair_offsets(stream, cs)

            d_cp_i  = d_junc[k] + off_i                              # scalar
            d_rem_i = torch.clamp(d_cp_i - cum_dist[k], min=0.0)     # [horizon]
            eta_i   = d_rem_i / v_traj[k].clamp(min=_EPS_V)          # [horizon]

            # project eta for every rival vehicle k in stream cs
            rival_etas: list[torch.Tensor] = []
            for rvid in rivals:
                ri      = idx_of[rvid]
                d_cp_j  = d_junc[ri] + off_j
                d_rem_j = torch.clamp(d_cp_j - cum_dist[ri], min=0.0)
                eta_j   = d_rem_j / v_traj[ri].clamp(min=_EPS_V)
                rival_etas.append(eta_j)

            # signed TTC(i, k, t) = eta_i(t) − eta_k(t)  for every rival k
            rival_stack = torch.stack(rival_etas, dim=0)              # [K_j, horizon]
            signed      = eta_i.unsqueeze(0) - rival_stack            # [K_j, horizon]

            ttc_cols.append(signed.T.abs())  # [horizon, K_j] — absolute time gap
            rival_ids.append(rivals)

        results[vid] = TTCSurface(
            vehicle_id=vid,
            stream=stream,
            ttc=ttc_cols,                # list of (horizon, K_j) tensors
            conflict_streams=list(conf_streams),
            rival_ids=rival_ids,
        )

    proj = ProjectionInfo(
        v_traj=v_traj,
        cum_dist=cum_dist,
        d_junc=d_junc,
        idx_of=idx_of,
        tracked=tracked,
    )
    return results, proj


def collect_at_risk(
    surfaces:  dict[str, TTCSurface],
    threshold: float = 3.0,
) -> list[AtRiskPair]:
    """
    Scan the full (i, j, t, k) TTC cube and return every pair whose minimum
    absolute TTC across the entire projection horizon is below `threshold`.

    A pair appears here if there exists ANY projected timestep t at which
    the two vehicles would be within `threshold` seconds of each other at
    their shared conflict point.

    Parameters
    ----------
    surfaces  : output of build_ttc_surfaces()
    threshold : danger threshold in seconds (default 3.0 s)

    Returns
    -------
    list[AtRiskPair], one entry per (i, j, k) triple that is at risk.
    Sorted by min_ttc ascending so the most dangerous pairs come first.

    Usage
    -----
    at_risk = collect_at_risk(surfaces, threshold=3.0)
    for p in at_risk:
        print(f"{p.ego_id} ({STREAM_NAMES[p.ego_stream]}) "
              f"vs {p.rival_id} ({STREAM_NAMES[p.rival_stream]}) "
              f"min_ttc={p.min_ttc:.2f}s at t={p.t_min * 0.1:.1f}s")
    """
    at_risk: list[AtRiskPair] = []

    for vid, surf in surfaces.items():
        for j, cs in enumerate(surf.conflict_streams):
            rivals = surf.rival_ids[j]
            if not rivals:
                continue
            ttc_j = surf.ttc[j]   # [horizon, K_j]

            for k, rvid in enumerate(rivals):
                col = ttc_j[:, k]                       # [horizon]
                min_val, t_min_idx = col.min(dim=0)
                if min_val.item() < threshold:
                    at_risk.append(AtRiskPair(
                        ego_id=vid,
                        ego_stream=surf.stream,
                        rival_id=rvid,
                        rival_stream=cs,
                        min_ttc=min_val.item(),
                        t_min=int(t_min_idx.item()),
                    ))

    at_risk.sort(key=lambda p: p.min_ttc)
    return at_risk
