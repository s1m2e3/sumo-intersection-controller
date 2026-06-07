"""
schedule_collector.py — Run SUMO and record the vehicle arrival schedule.

One call = one SUMO episode (warmup + 60-s window).  Run N times with
different seeds to get N diverse schedules for batched training.

Returns a list of VehicleEntry — one per vehicle that appears during the
episode.  Vehicles already in the network at warmup-end get spawn_step=0
and their current arc position; fresh arrivals get spawn_step=their step
within the episode and arc0=0.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import traci

# Reuse route/config writers and path helpers from demo_hybrid
from demo_hybrid import _write_routes, _write_cfg, _bin, DT, FLOW_VPH
from conflict import _INCOMING
from social_force import _ALL_MOVEMENTS

# ── consistent stream ordering shared with intersection_env ────────────────────
STREAM_LIST = sorted(_ALL_MOVEMENTS)         # 12 tuples, sorted for deterministic index
STREAM_IDX  = {s: i for i, s in enumerate(STREAM_LIST)}

WARMUP_SEC  = 30.0   # s — SUMO warmup before the episode window starts
EPISODE_SEC = 60.0   # s — length of one training episode


@dataclass
class VehicleEntry:
    spawn_step: int    # step index (within episode) when the vehicle becomes active
    stream_idx: int    # integer index into STREAM_LIST (0-11)
    v0:         float  # initial speed m/s
    arc0:       float  # initial arc position m (0 for fresh arrivals)


def _get_stream(vid: str) -> tuple | None:
    """Return (from_edge, to_edge) stream for a vehicle, or None if unclassified."""
    route = traci.vehicle.getRoute(vid)
    for i, e in enumerate(route):
        if e in _INCOMING and i + 1 < len(route):
            s = (e, route[i + 1])
            return s if s in STREAM_IDX else None
    return None


def collect_schedule(
    vph:         int         = FLOW_VPH,
    seed:        int         = 0,
    warmup_sec:  float       = WARMUP_SEC,
    episode_sec: float       = EPISODE_SEC,
    ew_vph:      int | None  = None,
    ns_vph:      int | None  = None,
) -> list[VehicleEntry]:
    """
    Run one headless SUMO simulation and return the vehicle schedule.

    Parameters
    ----------
    vph:         symmetric demand shorthand — used when ew_vph/ns_vph are None
    ew_vph:      east-west vehicles-per-hour (overrides vph if given)
    ns_vph:      north-south vehicles-per-hour (overrides vph if given)
    seed:        RNG seed passed to SUMO (controls departure timing)
    warmup_sec:  seconds to run before starting the schedule window
    episode_sec: length of the schedule window in seconds

    Returns
    -------
    List[VehicleEntry] — one entry per vehicle in the episode window,
    ordered by spawn_step.
    """
    if ew_vph is None:
        ew_vph = vph
    if ns_vph is None:
        ns_vph = vph
    routes = _write_routes(ew_vph, ns_vph)
    cfg    = _write_cfg(routes.name)

    cmd = [
        _bin("sumo"), "-c", str(cfg),
        "--step-length",       str(DT),
        "--collision.action",  "warn",
        "--no-step-log",
        "--seed",              str(seed),
    ]
    traci.start(cmd)

    entries:  list[VehicleEntry] = []
    seen:     set[str]           = set()   # vehicles already recorded

    warmup_steps  = round(warmup_sec  / DT)
    episode_steps = round(episode_sec / DT)

    try:
        # ── warmup ────────────────────────────────────────────────────────────
        for _ in range(warmup_steps):
            traci.simulationStep()

        # ── snapshot: vehicles already present on approach arms ───────────────
        for vid in traci.vehicle.getIDList():
            road = traci.vehicle.getRoadID(vid)
            if road not in _INCOMING:
                continue           # skip in-junction and exit-arm vehicles
            stream = _get_stream(vid)
            if stream is None:
                continue
            arc0 = float(traci.vehicle.getLanePosition(vid))
            v0   = float(traci.vehicle.getSpeed(vid))
            entries.append(VehicleEntry(0, STREAM_IDX[stream], v0, arc0))
            seen.add(vid)

        # ── episode: record each vehicle's first appearance ───────────────────
        for step in range(episode_steps):
            traci.simulationStep()
            for vid in traci.vehicle.getIDList():
                if vid in seen:
                    continue
                road = traci.vehicle.getRoadID(vid)
                if road not in _INCOMING:
                    continue
                stream = _get_stream(vid)
                if stream is None:
                    continue
                v0 = float(traci.vehicle.getSpeed(vid))
                entries.append(VehicleEntry(step, STREAM_IDX[stream], v0, 0.0))
                seen.add(vid)

    finally:
        traci.close()

    entries.sort(key=lambda e: e.spawn_step)
    return entries
