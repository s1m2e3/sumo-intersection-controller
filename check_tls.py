import os, traci
from pathlib import Path
import sumo as _sumo_pkg
from simulator import build, load_config

SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
os.environ["PATH"] = str(SUMO_BIN) + os.pathsep + os.environ.get("PATH", "")

sumocfg = build()
cfg     = load_config()
dt      = cfg["step_length"]

traci.start([str(SUMO_BIN / "sumo.exe"), "-c", str(sumocfg),
             "--step-length", str(dt)])

tls_ids = traci.trafficlight.getIDList()
print(f"Traffic lights in network: {tls_ids}")

for step in range(150):
    traci.simulationStep()

    if step % 10 == 0:
        for tls in tls_ids:
            phase      = traci.trafficlight.getPhase(tls)
            state      = traci.trafficlight.getRedYellowGreenState(tls)
            time_spent = traci.trafficlight.getSpentDuration(tls)
            vids       = list(traci.vehicle.getIDList())
            speeds     = {v: round(traci.vehicle.getSpeed(v), 2) for v in vids}
            print(f"  t={step*dt:>5.1f}s  TLS={tls}  phase={phase}  "
                  f"state={state}  spent={time_spent:.1f}s")
            print(f"           vehicle speeds: {speeds}")

traci.close()
