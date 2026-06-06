import os
import time
import subprocess
import traci
from pathlib import Path
import sumo as _sumo_pkg

SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
os.environ["PATH"]      = str(SUMO_BIN) + os.pathsep + os.environ.get("PATH", "")

PORT    = 8813
binary  = str(SUMO_BIN / "sumo-gui.exe")
sumocfg = str(Path("sumo_files/intersection.sumocfg"))

# launch sumo-gui as its own process with TraCI server enabled
print(f"Starting {binary} ...")
proc = subprocess.Popen([
    binary, "-c", sumocfg,
    "--port", str(PORT),
    "--num-clients", "1",
    "--start",          # auto-start simulation (no need to press play)
])

time.sleep(3)           # give the GUI time to open and bind the port

traci.connect(PORT)
print(f"Connected. SUMO version: {traci.getVersion()}")

for step in range(500):
    traci.simulationStep()
    if step % 100 == 0:
        print(f"Step {step}: {len(traci.vehicle.getIDList())} vehicles")

traci.close()
proc.wait()
print("Done.")
