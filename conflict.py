"""
conflict.py — intersection conflict stream tracker

Classifies every active SUMO vehicle by its movement stream (from_edge, to_edge)
and provides, for each vehicle, the list of other vehicles whose trajectories
conflict with it inside the junction box.

12 movement streams across 4 approaches:
  EW_T  east_in  -> west_out    westbound through
  EW_R  east_in  -> north_out   westbound right
  EW_L  east_in  -> south_out   westbound left
  WE_T  west_in  -> east_out    eastbound through
  WE_R  west_in  -> south_out   eastbound right
  WE_L  west_in  -> north_out   eastbound left
  NS_T  north_in -> south_out   southbound through
  NS_R  north_in -> west_out    southbound right
  NS_L  north_in -> east_out    southbound left
  SN_T  south_in -> north_out   northbound through
  SN_R  south_in -> east_out    northbound right
  SN_L  south_in -> west_out    northbound left

Conflict types captured:
  - Merging   : two streams share the same exit edge
  - Crossing  : through x through (perpendicular), left x through, left x left
  - Quadrant  : right x left sharing the same junction quadrant
"""

from __future__ import annotations
from collections import defaultdict
from typing import NamedTuple

import traci
import traci_cache

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Movement = tuple[str, str]   # (from_edge, to_edge)

# ---------------------------------------------------------------------------
# Human-readable labels — useful for logging / debugging
# ---------------------------------------------------------------------------

STREAM_NAMES: dict[Movement, str] = {
    ("east_in",  "west_out"):  "EW_T",
    ("east_in",  "north_out"): "EW_R",
    ("east_in",  "south_out"): "EW_L",
    ("west_in",  "east_out"):  "WE_T",
    ("west_in",  "south_out"): "WE_R",
    ("west_in",  "north_out"): "WE_L",
    ("north_in", "south_out"): "NS_T",
    ("north_in", "west_out"):  "NS_R",
    ("north_in", "east_out"):  "NS_L",
    ("south_in", "north_out"): "SN_T",
    ("south_in", "east_out"):  "SN_R",
    ("south_in", "west_out"):  "SN_L",
}

# ---------------------------------------------------------------------------
# Static conflict graph
# ---------------------------------------------------------------------------

# Merging groups: every pair within a group shares an exit edge and conflicts.
_MERGE_GROUPS: list[frozenset[Movement]] = [
    frozenset({("east_in","west_out"),  ("north_in","west_out"),  ("south_in","west_out")}),
    frozenset({("west_in","east_out"),  ("north_in","east_out"),  ("south_in","east_out")}),
    frozenset({("east_in","north_out"), ("west_in","north_out"),  ("south_in","north_out")}),
    frozenset({("east_in","south_out"), ("west_in","south_out"),  ("north_in","south_out")}),
]

# Crossing pairs: paths intersect inside the junction box.
_CROSS_PAIRS: list[tuple[Movement, Movement]] = [
    # through x through (perpendicular approaches)
    (("east_in","west_out"),  ("north_in","south_out")),   # EW_T x NS_T
    (("east_in","west_out"),  ("south_in","north_out")),   # EW_T x SN_T
    (("west_in","east_out"),  ("north_in","south_out")),   # WE_T x NS_T
    (("west_in","east_out"),  ("south_in","north_out")),   # WE_T x SN_T

    # left x opposing through (same axis)
    (("east_in","south_out"), ("west_in","east_out")),     # EW_L x WE_T
    (("west_in","north_out"), ("east_in","west_out")),     # WE_L x EW_T
    (("north_in","east_out"), ("south_in","north_out")),   # NS_L x SN_T
    (("south_in","west_out"), ("north_in","south_out")),   # SN_L x NS_T

    # left x crossing through (non-merging pairs only)
    (("east_in","south_out"), ("south_in","north_out")),   # EW_L x SN_T
    (("west_in","north_out"), ("north_in","south_out")),   # WE_L x NS_T
    (("north_in","east_out"), ("east_in","west_out")),     # NS_L x EW_T
    (("south_in","west_out"), ("west_in","east_out")),     # SN_L x WE_T

    # left x left (perpendicular approaches)
    (("east_in","south_out"), ("north_in","east_out")),    # EW_L x NS_L
    (("east_in","south_out"), ("south_in","west_out")),    # EW_L x SN_L
    (("west_in","north_out"), ("north_in","east_out")),    # WE_L x NS_L
    (("west_in","north_out"), ("south_in","west_out")),    # WE_L x SN_L

    # right x left sharing the same junction quadrant
    (("east_in","north_out"), ("north_in","east_out")),    # EW_R x NS_L  (NE)
    (("west_in","south_out"), ("south_in","west_out")),    # WE_R x SN_L  (SW)
    (("north_in","west_out"), ("west_in","north_out")),    # NS_R x WE_L  (NW)
    (("south_in","east_out"), ("east_in","south_out")),    # SN_R x EW_L  (SE)

    # left x left (same axis, opposite approaches) — paths cross through junction centre
    (("east_in","south_out"), ("west_in","north_out")),    # EW_L x WE_L
    (("north_in","east_out"), ("south_in","west_out")),    # NS_L x SN_L
]


def _build_conflict_map() -> dict[Movement, frozenset[Movement]]:
    cmap: dict[Movement, set[Movement]] = defaultdict(set)

    for group in _MERGE_GROUPS:
        members = list(group)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                cmap[a].add(b)
                cmap[b].add(a)

    for a, b in _CROSS_PAIRS:
        cmap[a].add(b)
        cmap[b].add(a)

    return {k: frozenset(v) for k, v in cmap.items()}


# Precomputed once at import time — never mutated.
CONFLICT_MAP: dict[Movement, frozenset[Movement]] = _build_conflict_map()

_INCOMING  = frozenset({"east_in", "west_in", "north_in", "south_in"})
_OUTGOING  = frozenset({"east_out", "west_out", "north_out", "south_out"})
_JUNCTION_PREFIX = ":center"   # SUMO names internal lanes ":center_*"

# Track vehicles on exit arms until their rear has cleared the merge conflict point.
# L_OCC (≈7m) is the occupancy length; 10m gives a small buffer for timing jitter.
EXIT_TRACK_DIST = 10.0   # m from junction exit


def is_in_conflict_zone(vid: str) -> bool:
    """True while a vehicle is approaching, inside, or just past the junction box."""
    road = traci_cache.get_road_id(vid)
    if road in _INCOMING or road.startswith(_JUNCTION_PREFIX):
        return True
    if road in _OUTGOING:
        return traci_cache.get_lane_pos(vid) < EXIT_TRACK_DIST
    return False

# ---------------------------------------------------------------------------
# Per-timestep snapshot
# ---------------------------------------------------------------------------

class ConflictSnapshot(NamedTuple):
    """
    Immutable snapshot of intersection conflict state for one timestep.

    stream_vehicles : movement -> [vehicle IDs in that stream this step]
    vehicle_stream  : vehicle ID -> its movement (None if not on a tracked route)
    conflicts       : vehicle ID -> [IDs of all vehicles in conflicting streams]
    """
    stream_vehicles: dict[Movement, list[str]]
    vehicle_stream:  dict[str, Movement | None]
    conflicts:       dict[str, list[str]]


_route_cache: dict[str, Movement | None] = {}


def _classify(vid: str) -> Movement | None:
    """Return (from_edge, to_edge) for a vehicle's path through the intersection."""
    if vid in _route_cache:
        return _route_cache[vid]
    route = traci.vehicle.getRoute(vid)
    result = None
    for i, edge in enumerate(route):
        if edge in _INCOMING and i + 1 < len(route):
            result = (edge, route[i + 1])
            break
    _route_cache[vid] = result
    return result


def clear_route_cache() -> None:
    """Call once per episode to evict stale entries from departed vehicles."""
    _route_cache.clear()


def build_snapshot(vehicle_ids: list[str]) -> ConflictSnapshot:
    """
    Build a ConflictSnapshot for the current simulation timestep.
    Call once per step, after traci.simulationStep().

    Example
    -------
    snap = build_snapshot(list(traci.vehicle.getIDList()))
    for vid, rivals in snap.conflicts.items():
        print(vid, STREAM_NAMES.get(snap.vehicle_stream[vid]), "->", rivals)
    """
    sv: dict[Movement, list[str]] = defaultdict(list)
    vs: dict[str, Movement | None] = {}

    for vid in vehicle_ids:
        if not is_in_conflict_zone(vid):
            continue
        mvmt = _classify(vid)
        vs[vid] = mvmt
        if mvmt is not None:
            sv[mvmt].append(vid)

    conf: dict[str, list[str]] = {}
    for vid, mvmt in vs.items():
        if mvmt is None:
            conf[vid] = []
            continue
        rivals: list[str] = []
        for c_stream in CONFLICT_MAP.get(mvmt, frozenset()):
            rivals.extend(sv.get(c_stream, []))
        conf[vid] = rivals

    return ConflictSnapshot(
        stream_vehicles=dict(sv),
        vehicle_stream=vs,
        conflicts=conf,
    )
