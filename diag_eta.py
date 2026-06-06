"""Quick ETA diagnostic — run once to check if yielder/passer is assigned correctly."""
import traci, os
from pathlib import Path
import sumo as _sumo_pkg
import numpy as np

SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)

traci.start([str(SUMO_BIN / "sumo.exe"),
             "-c", "sumo_files/intersection_through.sumocfg",
             "--no-step-log"])

import traci_cache
from conflict import build_snapshot, clear_route_cache
from social_force import _heading, _eta_to_cp, _crossing_point, _eta_weight, _THROUGH

clear_route_cache()
traci_cache.clear()

# advance to t=22s (first collision time from log)
for _ in range(110):
    traci.simulationStep()
    vids = list(traci.vehicle.getIDList())
    traci_cache.update(vids)

snap = build_snapshot(vids)
through = [(vid, s) for vid, s in snap.vehicle_stream.items() if s in _THROUGH]

print(f"\n{'vid':<16} {'stream':<12} {'pos':>20} {'spd':>6} {'rival':>16} {'eta_ego':>8} {'eta_riv':>8} {'cp':>16} {'w':>6} {'role'}")
print("-"*120)

for vid, stream in through:
    px, py = traci.vehicle.getPosition(vid)
    spd    = traci.vehicle.getSpeed(vid)
    ex, ey = _heading(traci.vehicle.getAngle(vid))
    for rvid in snap.conflicts.get(vid, [])[:1]:
        rs = snap.vehicle_stream.get(rvid)
        if not rs or rs not in _THROUGH: continue
        rpx, rpy = traci.vehicle.getPosition(rvid)
        rspd     = traci.vehicle.getSpeed(rvid)
        rex, rey = _heading(traci.vehicle.getAngle(rvid))
        cx, cy   = _crossing_point(px, py, ex, ey, rpx, rpy)
        eta_i    = _eta_to_cp(px, py, ex, ey, spd, cx, cy)
        eta_j    = _eta_to_cp(rpx, rpy, rex, rey, rspd, cx, cy)
        w        = _eta_weight(eta_i, eta_j)
        role     = "YIELD" if w > 0.6 else ("PASS" if w < 0.4 else "TIE")
        sname    = f"{stream[0][:5]}->{stream[1][:5]}"
        rsname   = f"{rs[0][:5]}->{rs[1][:5]}"
        print(f"{vid:<16} {sname:<12} ({px:6.1f},{py:6.1f}) {spd:6.2f}  "
              f"{rvid:<16} {eta_i:8.3f} {eta_j:8.3f}  ({cx:5.1f},{cy:5.1f}) {w:6.3f}  {role}")

traci.close()
