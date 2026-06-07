"""
benchmark.py — Systematic baseline sweep across flow conditions and controllers.

Test matrix:
  Symmetric:   300, 600, 900, 1200 vph/stream (all 4 equal)
  Asymmetric:  EW heavy vs NS light — (900,300), (1200,400), (1200,200)

Controllers (all use our IDM car-following — only junction conflict mechanism differs):
  priority       Our IDM + SUMO junction right-of-way (setSpeedMode bit 3)
  tl_fixed       Our IDM + SUMO traffic-light braking (setSpeedMode bit 4)
  ours           Our IDM + 2D social force (setSpeedMode=0, full TraCI control)

Metrics per run:
  avg_speed (m/s), % free-flow, collisions, throughput (arrived vehicles)

Run:
    conda run -n car-following-sumo python benchmark.py
"""
from __future__ import annotations
import os
import math
from pathlib import Path

import numpy as np
import traci

try:
    import sumo as _sumo_pkg
    SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
except ImportError:
    SUMO_BIN = Path(os.environ.get("SUMO_HOME", "")) / "bin"

SUMO_DIR = Path("sumo_files")
os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)

SIM_SECONDS = 60.0
WARMUP_SEC  = 3.0
DT          = 0.2
V_MAX       = 13.89

# ── our controller constants ────────────────────────────────────────────────────
V_APPROACH   = 8.0
V_TURN_LOW   = 8.0    # from model.py — turn kernel lower anchor
V_TURN_HIGH  = 11.0   # from model.py — turn kernel upper anchor
ARM_LENGTH   = 200.0
B_COMFORT    = 3.0
OMEGA_N      = 0.5
ZETA         = 1.2
K_D          = 2.0 * ZETA * OMEGA_N
IDM_ACCEL    = 2.6
IDM_BRAKE    = 4.5
IDM_S0       = 2.0
IDM_T        = 1.5
IDM_DELTA    = 4.0

_INCOMING = frozenset({"east_in", "west_in", "north_in", "south_in"})

_ALL_MOVEMENTS = frozenset({
    ("east_in", "west_out"), ("east_in", "north_out"), ("east_in", "south_out"),
    ("west_in", "east_out"), ("west_in", "south_out"), ("west_in", "north_out"),
    ("north_in", "south_out"), ("north_in", "west_out"), ("north_in", "east_out"),
    ("south_in", "north_out"), ("south_in", "east_out"), ("south_in", "west_out"),
})

_TURNING = frozenset({
    ("east_in", "north_out"), ("east_in", "south_out"),
    ("west_in", "south_out"), ("west_in", "north_out"),
    ("north_in", "west_out"), ("north_in", "east_out"),
    ("south_in", "east_out"), ("south_in", "west_out"),
})

TEST_MATRIX = [
    # label,        ew_vph, ns_vph
    ("sym-300",      300,   300),
    ("sym-600",      600,   600),
    ("sym-900",      900,   900),
    ("sym-1200",    1200,  1200),
    ("asym-900/300", 900,   300),
    ("asym-1200/400",1200,  400),
    ("asym-1200/200",1200,  200),
]

CONTROLLERS = ["priority", "tl_fixed", "ours"]


# ── file helpers ────────────────────────────────────────────────────────────────

def _bin(name: str) -> str:
    return str(SUMO_BIN / (name + (".exe" if os.name == "nt" else "")))


def _write_routes(ew_vph: int, ns_vph: int, tag: str = "bench",
                  turn_frac: float = 0.2) -> Path:
    """All 12 movements. Through=ew/ns_vph, right/left = turn_frac * through."""
    SUMO_DIR.mkdir(exist_ok=True)
    ew_r = max(1, int(ew_vph * turn_frac))
    ew_l = max(1, int(ew_vph * turn_frac))
    ns_r = max(1, int(ns_vph * turn_frac))
    ns_l = max(1, int(ns_vph * turn_frac))
    path = SUMO_DIR / f"routes_{tag}.xml"
    path.write_text(f"""<?xml version="1.0" ?>
<routes>
  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="{V_MAX}"/>
  <flow id="flow_ew_t" type="car" from="east_in"  to="west_out"  begin="5.0" end="300" vehsPerHour="{ew_vph}" departLane="0" departSpeed="desired"/>
  <flow id="flow_ew_r" type="car" from="east_in"  to="north_out" begin="5.1" end="300" vehsPerHour="{ew_r}"   departLane="0" departSpeed="desired"/>
  <flow id="flow_ew_l" type="car" from="east_in"  to="south_out" begin="5.2" end="300" vehsPerHour="{ew_l}"   departLane="1" departSpeed="desired"/>
  <flow id="flow_we_t" type="car" from="west_in"  to="east_out"  begin="5.3" end="300" vehsPerHour="{ew_vph}" departLane="0" departSpeed="desired"/>
  <flow id="flow_we_r" type="car" from="west_in"  to="south_out" begin="5.4" end="300" vehsPerHour="{ew_r}"   departLane="0" departSpeed="desired"/>
  <flow id="flow_we_l" type="car" from="west_in"  to="north_out" begin="5.5" end="300" vehsPerHour="{ew_l}"   departLane="1" departSpeed="desired"/>
  <flow id="flow_ns_t" type="car" from="north_in" to="south_out" begin="5.6" end="300" vehsPerHour="{ns_vph}" departLane="0" departSpeed="desired"/>
  <flow id="flow_ns_r" type="car" from="north_in" to="west_out"  begin="5.7" end="300" vehsPerHour="{ns_r}"   departLane="0" departSpeed="desired"/>
  <flow id="flow_ns_l" type="car" from="north_in" to="east_out"  begin="5.8" end="300" vehsPerHour="{ns_l}"   departLane="1" departSpeed="desired"/>
  <flow id="flow_sn_t" type="car" from="south_in" to="north_out" begin="5.9" end="300" vehsPerHour="{ns_vph}" departLane="0" departSpeed="desired"/>
  <flow id="flow_sn_r" type="car" from="south_in" to="east_out"  begin="6.0" end="300" vehsPerHour="{ns_r}"   departLane="0" departSpeed="desired"/>
  <flow id="flow_sn_l" type="car" from="south_in" to="west_out"  begin="6.1" end="300" vehsPerHour="{ns_l}"   departLane="1" departSpeed="desired"/>
</routes>
""")
    return path


def _write_cfg(net_file: str, routes_name: str, tag: str) -> Path:
    path = SUMO_DIR / f"bench_{tag}.sumocfg"
    path.write_text(f"""<?xml version="1.0" ?>
<configuration>
  <input>
    <net-file    value="{net_file}"/>
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


def _make_tl_adaptive(ew_vph: int, ns_vph: int) -> None:
    """Override the TL program via TraCI to split green proportional to flows."""
    total = ew_vph + ns_vph
    if total == 0:
        return
    ew_ratio = ew_vph / total
    ns_ratio = ns_vph / total
    cycle    = 60
    yellow   = 4

    ew_green = max(int(cycle * ew_ratio) - yellow, 5)
    ns_green = max(int(cycle * ns_ratio) - yellow, 5)

    tl_ids = traci.trafficlight.getIDList()
    if not tl_ids:
        return
    tl_id = tl_ids[0]

    # Build a simple 4-phase program: NS_green → NS_yellow → EW_green → EW_yellow
    # State strings depend on the network; get current to know phase count
    try:
        logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        phases = list(logic.phases)
        if len(phases) >= 4:
            phases[0] = traci.trafficlight.Phase(ns_green, phases[0].state)
            phases[1] = traci.trafficlight.Phase(yellow,   phases[1].state)
            phases[2] = traci.trafficlight.Phase(ew_green, phases[2].state)
            phases[3] = traci.trafficlight.Phase(yellow,   phases[3].state)
            new_logic = traci.trafficlight.Logic(
                "adaptive", 0, 0, phases)
            traci.trafficlight.setProgramLogic(tl_id, new_logic)
    except Exception:
        pass  # fall back to whatever the network has


# ── stats collector ─────────────────────────────────────────────────────────────

def _collect_stats_fair(speedmode_bits: int) -> tuple[float, int, int, int]:
    """
    Our IDM car-following with SUMO handling junction conflicts via speedmode_bits.
      priority  → speedmode_bits=8  (bit 3: yield at junctions / right-of-way)
      tl_fixed  → speedmode_bits=16 (bit 4: brake for traffic lights)

    Same car-following as 'ours' (no Krauss look-ahead); only junction policy differs.
    """
    import traci_cache
    traci_cache.clear()

    total_steps   = int((WARMUP_SEC + SIM_SECONDS) / DT)
    control_start = int(WARMUP_SEC / DT)

    known: set[str] = set()
    total_cols   = 0
    speed_sum    = 0.0
    speed_cnt    = 0
    departed     = 0
    arrived      = 0

    for step in range(total_steps):
        traci.simulationStep()
        departed += traci.simulation.getDepartedNumber()
        arrived  += traci.simulation.getArrivedNumber()

        all_vids = list(traci.vehicle.getIDList())
        traci_cache.update(all_vids)

        for vid in all_vids:
            if vid not in known:
                traci.vehicle.setSpeedMode(vid, speedmode_bits)
                known.add(vid)
        known.intersection_update(all_vids)

        if step < control_start or not all_vids:
            continue

        active = set(all_vids)
        for vid in all_vids:
            v = traci_cache.get_speed(vid)
            speed_sum += v
            speed_cnt += 1
            # No approach braking: junction conflict delegated to SUMO via speedmode_bits
            a_wp = _waypoint_accel(vid, active, has_rival=False)
            v_next = float(np.clip(v + a_wp * DT, 0.0, V_MAX))
            traci.vehicle.setSpeed(vid, v_next)

        try:
            total_cols += len(traci.simulation.getCollisions())
        except AttributeError:
            pass

    avg = speed_sum / max(speed_cnt, 1)
    return avg, total_cols, departed, arrived


# ── IDM + waypoint controller (ours) ────────────────────────────────────────────

def _idm_accel(v: float, gap: float, v_lead: float) -> float:
    dv     = v - v_lead
    s_star = IDM_S0 + v * IDM_T + v * dv / (2.0 * math.sqrt(IDM_ACCEL * IDM_BRAKE))
    s_star = max(s_star, IDM_S0)
    a      = IDM_ACCEL * (1.0 - (v / V_MAX) ** IDM_DELTA - (s_star / max(gap, 0.1)) ** 2)
    return float(np.clip(a, -IDM_BRAKE, IDM_ACCEL))


def _waypoint_accel(vid: str, active: set[str], has_rival: bool = True) -> float:
    import traci_cache
    road     = traci_cache.get_road_id(vid)
    lane_pos = traci_cache.get_lane_pos(vid)
    v        = traci_cache.get_speed(vid)

    leader = traci.vehicle.getLeader(vid)
    if leader:
        lid, gap = leader
        v_lead = traci_cache.get_speed(lid) if lid in active else v
        a_idm  = _idm_accel(v, gap, v_lead)
    else:
        a_idm = _idm_accel(v, 1000.0, v)

    if road in _INCOMING and has_rival:
        d        = max(ARM_LENGTH - lane_pos, 0.0)
        v_kin    = math.sqrt(V_APPROACH ** 2 + 2.0 * B_COMFORT * d)
        v_target = min(v_kin, V_MAX)
    else:
        v_target = V_MAX

    a_wp = float(np.clip(K_D * (v_target - v), -B_COMFORT, IDM_ACCEL))
    return min(a_idm, a_wp)


def _collect_stats_ours() -> tuple[float, int, int, int]:
    """Run sim with our custom controller. Returns (avg_speed, cols, departed, arrived)."""
    import traci_cache
    from conflict import build_snapshot, clear_route_cache, CONFLICT_MAP
    from social_force import compute_social_force_2d

    clear_route_cache()
    traci_cache.clear()

    total_steps   = int((WARMUP_SEC + SIM_SECONDS) / DT)
    control_start = int(WARMUP_SEC / DT)

    known: set[str] = set()
    total_cols   = 0
    speed_sum    = 0.0
    speed_cnt    = 0
    departed     = 0
    arrived      = 0

    for step in range(total_steps):
        traci.simulationStep()
        departed += traci.simulation.getDepartedNumber()
        arrived  += traci.simulation.getArrivedNumber()

        all_vids = list(traci.vehicle.getIDList())
        traci_cache.update(all_vids)

        for vid in all_vids:
            if vid not in known:
                traci.vehicle.setSpeedMode(vid, 0)
                known.add(vid)
        known.intersection_update(all_vids)

        if step < control_start or not all_vids:
            continue

        snap = build_snapshot(all_vids)
        all_tracked = [v for v, s in snap.vehicle_stream.items() if s in _ALL_MOVEMENTS]
        if all_tracked:
            a_soc_t, mu_soc_t, _, _ = compute_social_force_2d(all_tracked, snap)
            soc_map = {vid: (float(a_soc_t[j]), float(mu_soc_t[j]))
                       for j, vid in enumerate(all_tracked)}
        else:
            soc_map = {}

        ETA_RIVAL_THRESHOLD = 10.0
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

        active = set(all_vids)
        for vid in all_vids:
            v = traci_cache.get_speed(vid)
            speed_sum += v
            speed_cnt += 1

            ego_stream = snap.vehicle_stream.get(vid)
            if ego_stream in _ALL_MOVEMENTS:
                rival_streams = CONFLICT_MAP.get(ego_stream, frozenset()) & _ALL_MOVEMENTS
                has_rival = any(
                    min_rival_eta.get(s, float("inf")) < ETA_RIVAL_THRESHOLD
                    for s in rival_streams
                )
            else:
                has_rival = True

            a_wp = _waypoint_accel(vid, active, has_rival)

            # turn speed kernel: μ_turn · u_turn (from model.py)
            if ego_stream in _TURNING and v > V_TURN_LOW:
                mu_t = min((v - V_TURN_LOW) / (V_TURN_HIGH - V_TURN_LOW), 1.0)
                u_t  = float(np.clip(
                    IDM_ACCEL * (1.0 - (v / V_TURN_LOW) ** IDM_DELTA),
                    -B_COMFORT, 0.0,
                ))
                a_wp = a_wp + mu_t * u_t

            a_soc, _ = soc_map.get(vid, (0.0, 0.0))
            a = min(a_wp, a_soc) if a_soc < 0.0 else a_wp
            v_next = float(np.clip(v + a * DT, 0.0, V_MAX))
            traci.vehicle.setSpeed(vid, v_next)

        try:
            total_cols += len(traci.simulation.getCollisions())
        except AttributeError:
            pass

    avg = speed_sum / max(speed_cnt, 1)
    return avg, total_cols, departed, arrived


# ── per-run dispatcher ──────────────────────────────────────────────────────────

def run_scenario(controller: str, ew_vph: int, ns_vph: int) -> dict:
    routes = _write_routes(ew_vph, ns_vph, tag="bench")

    if controller in ("priority", "ours"):
        net = "intersection.net.xml"
    else:  # tl_fixed, tl_adaptive
        net = "intersection_tl.net.xml"

    cfg = _write_cfg(net, routes.name, tag=controller)
    cmd = [
        _bin("sumo"), "-c", str(cfg),
        "--step-length",              str(DT),
        "--collision.action",         "warn",
        "--collision.check-junctions",
        "--no-step-log",
    ]
    traci.start(cmd)

    if controller == "tl_adaptive":
        # let SUMO initialise TL, then override timing
        traci.simulationStep()
        _make_tl_adaptive(ew_vph, ns_vph)

    if controller == "ours":
        avg, cols, dep, arr = _collect_stats_ours()
    else:
        # priority and tl_fixed: our IDM desired speed + SUMO full safety enforcement.
        # setSpeedMode(31) = all SUMO safety bits active: SUMO enforces junction
        # right-of-way, following gap, and TL braking on top of our setSpeed() calls.
        # This removes Krauss's junction look-ahead from the desired speed calculation
        # while keeping SUMO's reactive safety system.
        avg, cols, dep, arr = _collect_stats_fair(31)

    traci.close()
    return {"controller": controller, "ew_vph": ew_vph, "ns_vph": ns_vph,
            "avg_speed": avg, "collisions": cols, "departed": dep, "arrived": arr}


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    results: list[dict] = []
    total_runs = len(TEST_MATRIX) * len(CONTROLLERS)
    run_n = 0

    print(f"\nBenchmark: {len(TEST_MATRIX)} scenarios × {len(CONTROLLERS)} controllers "
          f"= {total_runs} runs  ({SIM_SECONDS:.0f}s each)\n")

    for label, ew_vph, ns_vph in TEST_MATRIX:
        for ctrl in CONTROLLERS:
            run_n += 1
            print(f"  [{run_n:2d}/{total_runs}]  {ctrl:<16}  {label:<16}  "
                  f"EW={ew_vph} NS={ns_vph} ...", end="", flush=True)
            r = run_scenario(ctrl, ew_vph, ns_vph)
            r["label"] = label
            results.append(r)
            pct = r["avg_speed"] / V_MAX * 100
            col_flag = " !" if r["collisions"] > 0 else "  "
            print(f"  {r['avg_speed']:5.2f} m/s ({pct:4.1f}%)  "
                  f"cols={r['collisions']}{col_flag}  arr={r['arrived']}")

    # ── summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*95}")
    header = f"  {'Scenario':<16} " + "".join(
        f"  {c:<20}" for c in CONTROLLERS)
    print(header)
    sub = f"  {'':<16} " + "".join(
        f"  {'spd%  cols  arr':<20}" for _ in CONTROLLERS)
    print(sub)
    print(f"  {'-'*91}")

    for label, ew_vph, ns_vph in TEST_MATRIX:
        row = f"  {label:<16}"
        for ctrl in CONTROLLERS:
            r = next(x for x in results
                     if x["label"] == label and x["controller"] == ctrl)
            pct = r["avg_speed"] / V_MAX * 100
            flag = "!" if r["collisions"] > 0 else " "
            row += f"  {pct:4.1f}%  {r['collisions']:3d}{flag}  {r['arrived']:3d}"
        print(row)

    print(f"{'='*95}")
    print("  Columns: % free-flow speed | collisions | arrived vehicles\n")


if __name__ == "__main__":
    main()
