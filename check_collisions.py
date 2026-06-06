import os, traci
from pathlib import Path
import sumo as _sumo_pkg
from simulator import build, load_config
from model import IDMModel

SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
os.environ["PATH"] = str(SUMO_BIN) + os.pathsep + os.environ.get("PATH", "")

sumocfg = build()
cfg     = load_config()
dt      = cfg["step_length"]
idm     = IDMModel()

traci.start([str(SUMO_BIN / "sumo.exe"), "-c", str(sumocfg),
             "--step-length", str(dt), "--collision.action", "warn"])

known            = set()
total_collisions = 0

for step in range(300):
    traci.simulationStep()
    current = set(traci.vehicle.getIDList())
    for vid in current - known:
        traci.vehicle.setSpeedMode(vid, 0)
    known = current

    from model import IDMModel
    import torch
    vids = list(current)
    if vids:
        v   = torch.tensor([traci.vehicle.getSpeed(v) for v in vids])
        gap = torch.tensor([traci.vehicle.getLeader(v)[1]
                            if traci.vehicle.getLeader(v) else 100.0 for v in vids])
        vl  = torch.tensor([traci.vehicle.getSpeed(traci.vehicle.getLeader(v)[0])
                            if traci.vehicle.getLeader(v) else traci.vehicle.getSpeed(v)
                            for v in vids])
        a   = idm(v, gap, vl)
        vn  = torch.clamp(v + a * dt, min=0.0)
        for i, vid in enumerate(vids):
            traci.vehicle.setSpeed(vid, float(vn[i]))

    cols = traci.simulation.getCollisions()
    if cols:
        total_collisions += len(cols)
        print(f"step {step:>4} (t={step*dt:.1f}s): {len(cols)} collision(s)")
        for c in cols:
            print(f"  {c.collider} hit {c.victim}  type={c.type}")

traci.close()
print(f"\nTotal collision events over 300 steps: {total_collisions}")
print("(0 = vehicles are not colliding, >0 = they are running through each other)")
