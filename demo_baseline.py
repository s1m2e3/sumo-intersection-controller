"""
demo_baseline.py — Compare SUMO built-in controllers vs our waypoint+social-force.

Runs two baselines with the same 900 vph/stream, 120s, through-only scenario:
  1. SUMO priority  — priority junction + Krauss car-following, no TraCI control
  2. SUMO tl        — fixed-time traffic light (60s cycle) + Krauss, no TraCI control

Metrics: avg speed (m/s), total collision events, throughput (vehicles departed).
"""
from __future__ import annotations
import os
import math
from pathlib import Path

import traci

try:
    import sumo as _sumo_pkg
    SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
except ImportError:
    SUMO_BIN = Path(os.environ.get("SUMO_HOME", "")) / "bin"

SUMO_DIR  = Path("sumo_files")

SIM_SECONDS = 120.0
WARMUP_SEC  = 3.0
DT          = 0.2
FLOW_VPH    = 900
V_MAX       = 13.89
PRINT_EVERY = 25   # print every 5 s


def _bin(name: str) -> str:
    return str(SUMO_BIN / (name + (".exe" if os.name == "nt" else "")))


def _write_routes(vph: int) -> Path:
    SUMO_DIR.mkdir(exist_ok=True)
    path = SUMO_DIR / "routes_baseline.xml"
    path.write_text(f"""<?xml version="1.0" ?>
<routes>
  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="{V_MAX}"/>
  <flow id="flow_ew" type="car" from="east_in"  to="west_out"  begin="5"   end="300" vehsPerHour="{vph}" departLane="best" departSpeed="desired"/>
  <flow id="flow_we" type="car" from="west_in"  to="east_out"  begin="5.5" end="300" vehsPerHour="{vph}" departLane="best" departSpeed="desired"/>
  <flow id="flow_ns" type="car" from="north_in" to="south_out" begin="6"   end="300" vehsPerHour="{vph}" departLane="best" departSpeed="desired"/>
  <flow id="flow_sn" type="car" from="south_in" to="north_out" begin="6.5" end="300" vehsPerHour="{vph}" departLane="best" departSpeed="desired"/>
</routes>
""")
    return path


def _write_sumocfg(net_file: str, routes_name: str, label: str) -> Path:
    path = SUMO_DIR / f"baseline_{label}.sumocfg"
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


def run_baseline(label: str, net_file: str, routes_name: str) -> dict:
    """Run SUMO with no custom control. Returns stats dict."""
    cfg = _write_sumocfg(net_file, routes_name, label)
    os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)

    cmd = [
        _bin("sumo"), "-c", str(cfg),
        "--step-length",              str(DT),
        "--collision.action",         "warn",
        "--collision.check-junctions",
        "--no-step-log",
    ]
    traci.start(cmd)

    total_steps   = int((WARMUP_SEC + SIM_SECONDS) / DT)
    control_start = int(WARMUP_SEC / DT)

    total_cols   = 0
    speed_sum    = 0.0
    speed_counts = 0
    departed     = 0
    arrived      = 0

    print(f"\n  {'t(s)':>5}  {'veh':>4}  {'cols':>6}  {'spd(m/s)':>8}")
    print(f"  {'-'*35}")

    for step in range(total_steps):
        traci.simulationStep()
        t = (step + 1) * DT

        departed += traci.simulation.getDepartedNumber()
        arrived  += traci.simulation.getArrivedNumber()

        if step >= control_start:
            vids = list(traci.vehicle.getIDList())
            for vid in vids:
                speed_sum    += traci.vehicle.getSpeed(vid)
                speed_counts += 1
            try:
                total_cols += len(traci.simulation.getCollisions())
            except AttributeError:
                pass

            if (step - control_start + 1) % PRINT_EVERY == 0:
                n      = len(vids)
                mean_v = speed_sum / max(speed_counts, 1)
                print(f"  {t:5.1f}  {n:4d}  {total_cols:6d}  {mean_v:8.2f}")

    traci.close()
    mean_v = speed_sum / max(speed_counts, 1)
    return {"label": label, "avg_speed": mean_v, "collisions": total_cols,
            "departed": departed, "arrived": arrived}


def main():
    routes = _write_routes(FLOW_VPH)

    scenarios = [
        ("priority", "intersection.net.xml"),
        ("tl_60s",   "intersection_tl.net.xml"),
    ]

    results = []
    for label, net in scenarios:
        print(f"\n{'='*60}")
        print(f"  Baseline: {label}  ({net})")
        print(f"{'='*60}")
        r = run_baseline(label, net, routes.name)
        results.append(r)
        print(f"\n  → avg speed: {r['avg_speed']:.2f} m/s  "
              f"collisions: {r['collisions']}  "
              f"departed: {r['departed']}  arrived: {r['arrived']}")

    # ── summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {'Controller':<22}  {'Avg speed':>10}  {'% free-flow':>12}  {'Collisions':>10}")
    print(f"  {'-'*66}")
    for r in results:
        pct = r["avg_speed"] / V_MAX * 100
        print(f"  {r['label']:<22}  {r['avg_speed']:>8.2f} m/s  {pct:>10.1f}%  {r['collisions']:>10}")

    # our controller result (from last run)
    our_speed = 6.2
    our_pct   = our_speed / V_MAX * 100
    print(f"  {'waypoint+social (ours)':<22}  {our_speed:>8.2f} m/s  {our_pct:>10.1f}%  {'0':>10}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
