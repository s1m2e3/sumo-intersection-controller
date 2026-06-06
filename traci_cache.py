"""
traci_cache.py — per-step subscription cache for SUMO TraCI vehicle attributes.

Call update(vehicle_ids) once after simulationStep() to subscribe any new
vehicles and batch-fetch all data via a single getAllSubscriptionResults()
round-trip.  All subsequent attribute lookups within the same step hit the
in-memory dict instead of the socket.

Subscribed variables (SPEED, ROAD_ID, LANE_POSITION) cover most per-vehicle
queries.  VAR_LEADER is intentionally kept as a direct call because its
subscription requires a version-specific distance parameter — but by reducing
all other queries to cache lookups the total socket round-trips per step drop
from O(N × attrs) to O(1 batch + N_leader).
"""

from __future__ import annotations
import traci
import traci.constants as tc

_VARS = (
    tc.VAR_SPEED,
    tc.VAR_ROAD_ID,
    tc.VAR_LANEPOSITION,
)

_cache:      dict[str, dict] = {}
_subscribed: set[str]        = set()


def update(vehicle_ids: list[str]) -> None:
    """
    Subscribe any newly-appeared vehicles and refresh the cache in one call.
    Must be called immediately after simulationStep() before any queries.
    """
    global _cache
    new_vids = set(vehicle_ids) - _subscribed
    for vid in new_vids:
        traci.vehicle.subscribe(vid, _VARS)
        _subscribed.add(vid)
    _cache = traci.vehicle.getAllSubscriptionResults()
    _subscribed.intersection_update(vehicle_ids)   # drop departed


def get_speed(vid: str) -> float:
    r = _cache.get(vid)
    return r[tc.VAR_SPEED] if r else traci.vehicle.getSpeed(vid)


def get_road_id(vid: str) -> str:
    r = _cache.get(vid)
    return r[tc.VAR_ROAD_ID] if r else traci.vehicle.getRoadID(vid)


def get_lane_pos(vid: str) -> float:
    r = _cache.get(vid)
    return r[tc.VAR_LANEPOSITION] if r else traci.vehicle.getLanePosition(vid)


def clear() -> None:
    """Reset between episodes so departed vehicle IDs don't linger."""
    _cache.clear()
    _subscribed.clear()
