"""
diag_flows.py — headless diagnostic: per-stream vehicle counts and mean speeds.
Runs 20 seconds, prints a table every 1 second.
No PyTorch control — pure SUMO with speedMode default so we can see raw insertion behavior.
"""
import os
from collections import defaultdict
from pathlib import Path

import sumo as _sumo_pkg
import traci

from conflict import STREAM_NAMES, _classify, _INCOMING
from simulator import build, load_config

SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"

def _bin(name):
    return str(SUMO_BIN / (name + (".exe" if os.name == "nt" else "")))

cfg     = load_config()
sumocfg = build()
dt      = cfg["step_length"]

os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
traci.start([
    _bin("sumo"), "-c", str(sumocfg),
    "--step-length", str(dt),
    "--no-step-log",
])

steps    = int(20 / dt)
header   = f"{'t':>5}  {'total':>5}  " + "  ".join(f"{n:>5}" for n in [
    "EW_T","EW_R","EW_L","WE_T","WE_R","WE_L",
    "NS_T","NS_R","NS_L","SN_T","SN_R","SN_L","other"])
print(header)
print("-" * len(header))

for step in range(steps):
    traci.simulationStep()
    if step % 10 != 0:   # print every 1 s (10 steps at dt=0.1)
        continue

    sim_t    = step * dt
    all_vids = list(traci.vehicle.getIDList())

    counts  = defaultdict(int)
    speeds  = defaultdict(list)

    for vid in all_vids:
        road = traci.vehicle.getRoadID(vid)
        spd  = traci.vehicle.getSpeed(vid)
        if road in _INCOMING or road.startswith(":center"):
            mvmt = _classify(vid)
            name = STREAM_NAMES.get(mvmt, "other") if mvmt else "other"
        else:
            name = "other"
        counts[name] += 1
        speeds[name].append(spd)

    def fmt(name):
        n = counts[name]
        v = sum(speeds[name]) / len(speeds[name]) if speeds[name] else 0.0
        return f"{n:2d}({v:4.1f})"

    streams = ["EW_T","EW_R","EW_L","WE_T","WE_R","WE_L",
               "NS_T","NS_R","NS_L","SN_T","SN_R","SN_L","other"]
    row = f"{sim_t:5.1f}  {len(all_vids):5d}  " + "  ".join(f"{fmt(s):>7}" for s in streams)
    print(row)

traci.close()
