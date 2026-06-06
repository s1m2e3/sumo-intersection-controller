import traci, os, sumo as _sumo_pkg
from pathlib import Path
from simulator import build

SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
sumocfg = build()
traci.start([str(SUMO_BIN / "sumo.exe"), "-c", str(sumocfg), "--no-step-log"])

print("All links out of each incoming lane 0:")
for from_e in ["east_in", "west_in", "north_in", "south_in"]:
    lane = f"{from_e}_0"
    links = traci.lane.getLinks(lane)
    print(f"\n  {lane}:")
    for link in links:
        successor, _, _, _, via, *_ = link
        length = traci.lane.getLength(via) if via else "—"
        print(f"    -> {successor:<20}  via={via:<25}  len={length:.2f}" if via else f"    -> {successor}")

print("\nAll links out of each incoming lane 1:")
for from_e in ["east_in", "west_in", "north_in", "south_in"]:
    lane = f"{from_e}_1"
    links = traci.lane.getLinks(lane)
    print(f"\n  {lane}:")
    for link in links:
        successor, _, _, _, via, *_ = link
        length = traci.lane.getLength(via) if via else "—"
        print(f"    -> {successor:<20}  via={via:<25}  len={length:.2f}" if via else f"    -> {successor}")

traci.close()
