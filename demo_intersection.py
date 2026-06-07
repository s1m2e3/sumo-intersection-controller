"""
demo_intersection.py — SUMO co-simulation with all 12 movements (through + turns).

Streams per approach: through (T), right (R), left (L) — 4 approaches = 12 streams.
Flow split: through 60 %, right 20 %, left 20 % of approach volume.

Controller per vehicle per step:
  1. IDM car-following  — same-arm leader
  2. Kinematic approach — smooth decel to V_APPROACH; gated by rival ETA
  3. Social force       — ETA-based yielder/passer for ALL conflict pairs
  Combined:  a = min(a_idm, a_wp);  if social brakes harder → use social
             v_next = clip(v + a*dt, 0, getAllowedSpeed)   ← respects turn radii

Run:
    conda run -n car-following-sumo python demo_intersection.py
    conda run -n car-following-sumo python demo_intersection.py --gui
    conda run -n car-following-sumo python demo_intersection.py --vph 1200
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import math

import numpy as np
import traci

import traci_cache
from conflict import build_snapshot, clear_route_cache, STREAM_NAMES, CONFLICT_MAP
from social_force import compute_social_force_2d, _ALL_MOVEMENTS
from model import V_TURN_LOW, V_TURN_HIGH

# ── paths / constants ────────────────────────────────────────────────────────────
try:
    import sumo as _sumo_pkg
    SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
except ImportError:
    SUMO_BIN = Path(os.environ.get("SUMO_HOME", "")) / "bin"

SUMO_DIR   = Path("sumo_files")

SIM_SECONDS  = 120.0
WARMUP_SEC   = 3.0
DT           = 0.2            # simulation step (s)
FLOW_VPH     = 900            # vehicles per hour per stream
PRINT_EVERY  = 5              # log every N steps ≈ 1 s

V_MAX        = 13.89          # m/s  (~50 km/h)
V_APPROACH   = 8.0            # m/s  target speed at junction entry
ARM_LENGTH   = 200.0          # m    incoming/outgoing arm length (from network)
B_COMFORT    = 3.0            # m/s² comfortable braking for kinematic profile

# Waypoint-controller speed-error gain (from demo_waypoint_tracking.py, zeta=1.2)
OMEGA_N = 0.5
ZETA    = 1.2
K_D     = 2.0 * ZETA * OMEGA_N  # 1.2  — speed-error gain only (no k_p·d)

# IDM car-following parameters (same-stream leader following)
IDM_ACCEL = 2.6   # m/s²
IDM_BRAKE = 4.5   # m/s²
IDM_S0    = 2.0   # m    minimum gap
IDM_T     = 1.5   # s    desired time headway
IDM_DELTA = 4.0

# Edges
_INCOMING = frozenset({"east_in", "west_in", "north_in", "south_in"})
_OUTGOING = frozenset({"east_out", "west_out", "north_out", "south_out"})

# Turning movement streams — right and left turns (not through)
_TURNING = frozenset({
    ("east_in",  "north_out"),  # EW_R
    ("east_in",  "south_out"),  # EW_L
    ("west_in",  "south_out"),  # WE_R
    ("west_in",  "north_out"),  # WE_L
    ("north_in", "west_out"),   # NS_R
    ("north_in", "east_out"),   # NS_L
    ("south_in", "east_out"),   # SN_R
    ("south_in", "west_out"),   # SN_L
})


# ── SUMO file helpers ────────────────────────────────────────────────────────────

def _bin(name: str) -> str:
    return str(SUMO_BIN / (name + (".exe" if os.name == "nt" else "")))


def _write_routes(ew_vph: int, ns_vph: int | None = None,
                  turn_frac: float = 0.2) -> Path:
    """Write routes for all 12 movements.

    ew_vph / ns_vph are the THROUGH flow per stream.
    turn_frac: fraction of through flow for each turn direction (right and left).
    Flow split: through 60 %, right 20 %, left 20 % when turn_frac=0.2.
    Left-turn vehicles depart from lane 1 (the dedicated left-turn lane).
    """
    if ns_vph is None:
        ns_vph = ew_vph
    ew_r = max(1, int(ew_vph * turn_frac))
    ew_l = max(1, int(ew_vph * turn_frac))
    ns_r = max(1, int(ns_vph * turn_frac))
    ns_l = max(1, int(ns_vph * turn_frac))

    SUMO_DIR.mkdir(exist_ok=True)
    path = SUMO_DIR / "routes_all_movements.xml"
    # Flows sorted by begin time (SUMO requirement)
    path.write_text(f"""<?xml version="1.0" ?>
<routes>
  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="{V_MAX}"/>
  <flow id="flow_ew_t" type="car" from="east_in"  to="west_out"  begin="5.0" end="300" vehsPerHour="{ew_vph}" departLane="0" departSpeed="desired"/>
  <flow id="flow_ew_r" type="car" from="east_in"  to="north_out" begin="5.1" end="300" vehsPerHour="{ew_r}"  departLane="0" departSpeed="desired"/>
  <flow id="flow_ew_l" type="car" from="east_in"  to="south_out" begin="5.2" end="300" vehsPerHour="{ew_l}"  departLane="1" departSpeed="desired"/>
  <flow id="flow_we_t" type="car" from="west_in"  to="east_out"  begin="5.3" end="300" vehsPerHour="{ew_vph}" departLane="0" departSpeed="desired"/>
  <flow id="flow_we_r" type="car" from="west_in"  to="south_out" begin="5.4" end="300" vehsPerHour="{ew_r}"  departLane="0" departSpeed="desired"/>
  <flow id="flow_we_l" type="car" from="west_in"  to="north_out" begin="5.5" end="300" vehsPerHour="{ew_l}"  departLane="1" departSpeed="desired"/>
  <flow id="flow_ns_t" type="car" from="north_in" to="south_out" begin="5.6" end="300" vehsPerHour="{ns_vph}" departLane="0" departSpeed="desired"/>
  <flow id="flow_ns_r" type="car" from="north_in" to="west_out"  begin="5.7" end="300" vehsPerHour="{ns_r}"  departLane="0" departSpeed="desired"/>
  <flow id="flow_ns_l" type="car" from="north_in" to="east_out"  begin="5.8" end="300" vehsPerHour="{ns_l}"  departLane="1" departSpeed="desired"/>
  <flow id="flow_sn_t" type="car" from="south_in" to="north_out" begin="5.9" end="300" vehsPerHour="{ns_vph}" departLane="0" departSpeed="desired"/>
  <flow id="flow_sn_r" type="car" from="south_in" to="east_out"  begin="6.0" end="300" vehsPerHour="{ns_r}"  departLane="0" departSpeed="desired"/>
  <flow id="flow_sn_l" type="car" from="south_in" to="west_out"  begin="6.1" end="300" vehsPerHour="{ns_l}"  departLane="1" departSpeed="desired"/>
</routes>
""")
    return path


def _write_sumocfg(routes_name: str) -> Path:
    path = SUMO_DIR / "intersection_through.sumocfg"
    path.write_text(f"""<?xml version="1.0" ?>
<configuration>
  <input>
    <net-file    value="intersection.net.xml"/>
    <route-files value="{routes_name}"/>
  </input>
  <time>
    <begin value="0"/>
    <end   value="300"/>
    <step-length value="{DT}"/>
  </time>
  <report>
    <no-step-log value="true"/>
    <verbose     value="false"/>
  </report>
</configuration>
""")
    return path


# ── controllers ──────────────────────────────────────────────────────────────────

def _idm_accel(v: float, gap: float, v_lead: float) -> float:
    """IDM car-following acceleration (same-stream leader)."""
    dv     = v - v_lead
    s_star = IDM_S0 + v * IDM_T + v * dv / (2.0 * math.sqrt(IDM_ACCEL * IDM_BRAKE))
    s_star = max(s_star, IDM_S0)
    a      = IDM_ACCEL * (1.0 - (v / V_MAX) ** IDM_DELTA - (s_star / max(gap, 0.1)) ** 2)
    return float(np.clip(a, -IDM_BRAKE, IDM_ACCEL))


def _waypoint_accel(vid: str, active: set[str], has_rival: bool = True) -> float:
    """
    Combined IDM + kinematic junction-approach controller.

    1. IDM car-following: handles same-stream leaders.
    2. Kinematic junction approach: v_kin = sqrt(V_APPROACH² + 2·B·d) → smooth
       deceleration to V_APPROACH at the junction entry.
       Skipped (v_target = V_MAX) when no vehicle exists on any conflicting stream.
    3. Return min(a_idm, a_wp) — the more conservative of the two.
    """
    road     = traci_cache.get_road_id(vid)
    lane_pos = traci_cache.get_lane_pos(vid)
    v        = traci_cache.get_speed(vid)

    # ── IDM: same-lane leader ─────────────────────────────────────────────────────
    leader = traci.vehicle.getLeader(vid)
    if leader:
        lid, gap = leader
        v_lead = traci_cache.get_speed(lid) if lid in active else v
        a_idm  = _idm_accel(v, gap, v_lead)
    else:
        a_idm  = _idm_accel(v, 1000.0, v)   # free-flow

    # ── Kinematic waypoint: junction approach ─────────────────────────────────────
    if road in _INCOMING and has_rival:
        d        = max(ARM_LENGTH - lane_pos, 0.0)
        v_kin    = math.sqrt(V_APPROACH ** 2 + 2.0 * B_COMFORT * d)
        v_target = min(v_kin, V_MAX)
    else:
        v_target = V_MAX

    a_wp = float(np.clip(K_D * (v_target - v), -B_COMFORT, IDM_ACCEL))

    # Most conservative: if IDM says brake harder, respect the leader
    return min(a_idm, a_wp)


# ── collision / separation helpers ───────────────────────────────────────────────

def _count_collisions() -> int:
    try:
        return len(traci.simulation.getCollisions())
    except AttributeError:
        return 0


def _min_separation(snap) -> float:
    """Min 2D Euclidean gap between vehicles in conflicting streams."""
    positions: dict[str, tuple[float, float]] = {}
    for s in _ALL_MOVEMENTS:
        for vid in snap.stream_vehicles.get(s, []):
            try:
                positions[vid] = traci.vehicle.getPosition(vid)
            except Exception:
                pass

    min_sep = float("inf")
    for vid, (x1, y1) in positions.items():
        for rival in snap.conflicts.get(vid, []):
            if rival in positions:
                x2, y2 = positions[rival]
                sep = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                min_sep = min(min_sep, sep)
    return min_sep


# ── main simulation loop ─────────────────────────────────────────────────────────

def run(gui: bool = False, flow_vph: int = FLOW_VPH,
        ew_vph: int | None = None, ns_vph: int | None = None) -> int:
    _ew = ew_vph if ew_vph is not None else flow_vph
    _ns = ns_vph if ns_vph is not None else flow_vph
    routes_path = _write_routes(_ew, _ns)
    sumocfg     = _write_sumocfg(routes_path.name)

    os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)

    cmd = [
        _bin("sumo-gui" if gui else "sumo"),
        "-c", str(sumocfg),
        "--step-length",              str(DT),
        "--collision.action",         "warn",
        "--collision.check-junctions",
        "--no-step-log",
    ]
    if gui:
        cmd += ["--start", "--quit-on-end", "--delay", "200"]

    traci.start(cmd)
    if gui:
        try:
            traci.gui.setOffset("View #0", 0.0, 0.0)
            traci.gui.setZoom("View #0", 400.0)
        except Exception:
            pass

    clear_route_cache()
    traci_cache.clear()

    warmup_steps  = int(WARMUP_SEC / DT)
    control_steps = int(SIM_SECONDS / DT)

    known: set[str] = set()
    total_cols      = 0
    step_cols       = 0

    print(f"\n{'='*92}")
    print(f"  4-way intersection | {SIM_SECONDS:.0f}s | all movements | EW={_ew} NS={_ns} vph through/stream (+ 20% R + 20% L)")
    print(f"  Controller: waypoint-tracking (wn={OMEGA_N}, zeta={ZETA}) + 2D social force")
    print(f"              V_MAX={V_MAX} m/s | V_APPROACH={V_APPROACH} m/s | B_COMFORT={B_COMFORT} m/s^2")
    print(f"{'='*92}")
    print(f"  {'t(s)':>5}  {'phase':<8}  {'veh':>4}  {'cols/step':>9}  "
          f"{'total_cols':>10}  {'min_sep(m)':>10}  {'spd(m/s)':>8}  streams")
    print(f"  {'-'*90}")

    # ── Phase 1: warmup — SUMO's own controller drives vehicles ──────────────────
    for step in range(warmup_steps):
        traci.simulationStep()
        all_vids = list(traci.vehicle.getIDList())
        traci_cache.update(all_vids)
        t = (step + 1) * DT
        if (step + 1) % PRINT_EVERY == 0:
            print(f"  {t:5.1f}  {'WARMUP':<8}  {len(all_vids):4d}  "
                  f"{'—':>9}  {'—':>10}  {'—':>10}  {'—':>8}", flush=True)

    # ── Phase 2: our controller owns all vehicles ─────────────────────────────────
    for step in range(control_steps):
        traci.simulationStep()
        t = WARMUP_SEC + (step + 1) * DT

        all_vids = list(traci.vehicle.getIDList())
        traci_cache.update(all_vids)

        # take over speed control for newly-arrived vehicles
        for vid in all_vids:
            if vid not in known:
                traci.vehicle.setSpeedMode(vid, 0)
                known.add(vid)

        # purge departed
        known.intersection_update(all_vids)

        if not all_vids:
            if (step + 1) % PRINT_EVERY == 0:
                print(f"  {t:5.1f}  {'PYTORCH':<8}     0  "
                      f"{'0':>9}  {total_cols:>10}  {'—':>10}  {'—':>8}", flush=True)
            continue

        # ── conflict snapshot ─────────────────────────────────────────────────────
        snap = build_snapshot(all_vids)

        # ── 2D social force — all 12 movement streams ────────────────────────────
        all_tracked = [
            vid for vid, s in snap.vehicle_stream.items()
            if s in _ALL_MOVEMENTS
        ]
        if all_tracked:
            a_soc_t, mu_soc_t, _, _ = compute_social_force_2d(all_tracked, snap)
        else:
            a_soc_t = mu_soc_t = None

        soc_map  = {}   # vid → (a_soc, mu_soc)
        if a_soc_t is not None:
            for j, vid in enumerate(all_tracked):
                soc_map[vid] = (
                    float(a_soc_t[j]),
                    float(mu_soc_t[j]),
                )

        # ── min ETA to junction centre for each movement stream ──────────────────
        # Gate approach braking: only activate if a rival arrives within ETA_RIVAL_THRESHOLD.
        ETA_RIVAL_THRESHOLD = 10.0  # s — widened to cover SAFE_GAP=3.5 + approach time
        _CX, _CY = 200.0, 200.0
        min_rival_eta: dict = {}
        for stream, svids in snap.stream_vehicles.items():
            if stream not in _ALL_MOVEMENTS or not svids:
                continue
            min_eta = float("inf")
            for rvid in svids:
                px, py = traci.vehicle.getPosition(rvid)
                spd = traci_cache.get_speed(rvid)
                dist = math.sqrt((_CX - px) ** 2 + (_CY - py) ** 2)
                min_eta = min(min_eta, dist / max(spd, 0.1))
            min_rival_eta[stream] = min_eta

        # ── apply waypoint + social force for each vehicle ────────────────────────
        active = set(all_vids)
        v_sum = 0.0
        for vid in all_vids:
            v   = traci_cache.get_speed(vid)
            v_sum += v

            # apply approach braking only if a rival will reach the crossing soon
            ego_stream = snap.vehicle_stream.get(vid)
            if ego_stream in _ALL_MOVEMENTS:
                rival_streams = CONFLICT_MAP.get(ego_stream, frozenset()) & _ALL_MOVEMENTS
                has_rival = any(
                    min_rival_eta.get(s, float("inf")) < ETA_RIVAL_THRESHOLD
                    for s in rival_streams
                )
            else:
                has_rival = True  # conservative for unclassified vehicles

            # base: IDM car-following + kinematic junction approach
            a_wp = _waypoint_accel(vid, active, has_rival)

            # social force correction (zero if not in conflict zone)
            a_soc, mu_soc = soc_map.get(vid, (0.0, 0.0))

            # turn speed kernel (from model.py): μ_turn · u_turn
            # ramps 0→1 as v goes V_TURN_LOW→V_TURN_HIGH; u_turn brakes to V_TURN_LOW
            if ego_stream in _TURNING and v > V_TURN_LOW:
                mu_t = min((v - V_TURN_LOW) / (V_TURN_HIGH - V_TURN_LOW), 1.0)
                u_t  = float(np.clip(
                    IDM_ACCEL * (1.0 - (v / V_TURN_LOW) ** IDM_DELTA),
                    -B_COMFORT, 0.0,
                ))
                a_wp = a_wp + mu_t * u_t   # u_t ≤ 0 → brakes turning vehicles above V_TURN_LOW

            # hard override: if social says brake harder than waypoint, use social
            # passer (a_soc=0): pure waypoint; yielder (a_soc<0): min of both
            a = min(a_wp, a_soc) if a_soc < 0.0 else a_wp

            v_next = float(np.clip(v + a * DT, 0.0, V_MAX))
            traci.vehicle.setSpeed(vid, v_next)

        # ── collision tracking ────────────────────────────────────────────────────
        step_cols   = _count_collisions()
        total_cols += step_cols

        # ── periodic log ──────────────────────────────────────────────────────────
        if (step + 1) % PRINT_EVERY == 0:
            n       = len(all_vids)
            mean_v  = v_sum / n if n else 0.0
            min_sep = _min_separation(snap)
            sep_str = f"{min_sep:10.2f}" if min_sep < 1e8 else "         ∞"
            streams = " ".join(sorted(
                STREAM_NAMES.get(s, str(s))
                for s in snap.stream_vehicles)) or "—"
            print(f"  {t:5.1f}  {'CONTROL':<8}  {n:4d}  "
                  f"{step_cols:>9}  {total_cols:>10}  {sep_str}  "
                  f"{mean_v:8.2f}  {streams}", flush=True)

    traci.close()
    print(f"\n{'='*92}")
    print(f"  Simulation complete  |  total collision events: {total_cols}")
    print(f"{'='*92}\n")
    return total_cols


# ── entry point ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="60s through-only intersection: waypoint-tracking + social force")
    ap.add_argument("--gui", action="store_true",  help="open SUMO-GUI")
    ap.add_argument("--vph", type=int, default=FLOW_VPH,
                    help=f"vehicles per hour per stream (default {FLOW_VPH})")
    args = ap.parse_args()

    cols = run(gui=args.gui, flow_vph=args.vph)
    raise SystemExit(0 if cols == 0 else 1)
