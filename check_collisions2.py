import os, traci
from pathlib import Path
import sumo as _sumo_pkg

SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
def _bin(n): return str(SUMO_BIN / (n + ".exe"))

from simulator import build, load_config
cfg     = load_config()
sumocfg = build()
dt      = cfg["step_length"]

os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
traci.start([
    _bin("sumo"), "-c", str(sumocfg),
    "--step-length", str(dt),
    "--collision.action", "warn",
    "--collision.check-junctions",
    "--no-step-log",
])

collisions = 0
teleports  = 0
for step in range(400):
    traci.simulationStep()
    collisions += traci.simulation.getCollidingVehiclesNumber()
    teleports  += traci.simulation.getStartingTeleportNumber()

traci.close()
print(f"Total collision events : {collisions}")
print(f"Total teleport events  : {teleports}")
