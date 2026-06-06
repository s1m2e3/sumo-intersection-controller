"""
Co-simulation synchronization verifier with full vehicle lifecycle logging.
Shows vehicle arrivals, per-step IDM state, and vehicle departures.

Output: logs/sync_verification.txt
"""

import os
import torch
import traci
from pathlib import Path
from datetime import datetime

import sumo as _sumo_pkg
from model import IDMModel
from simulator import build, load_config

SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
os.environ["PATH"]      = str(SUMO_BIN) + os.pathsep + os.environ.get("PATH", "")

LOG_PATH = Path("logs") / "sync_verification.txt"
STEPS    = 300   # 300 steps × 0.1s = 30 simulated seconds

LEGEND = """
COLUMN LEGEND
─────────────────────────────────────────────────────────────────────────────
  step        Timestep index. Each step = {dt}s of simulated time.

  vehicle     Vehicle ID (format: <flow>.<index>)
                flow_ns = North→South,  flow_sn = South→North
                flow_ew = East→West,    flow_we = West→East
              Index increments each time a new vehicle spawns on that flow.

  v_sumo_in   Speed (m/s) SUMO reported at the START of this step.
              This is the current truth we READ from the simulation.

  gap         Net bumper-to-bumper distance (m) to the vehicle ahead.
              100.00 → no leader detected (free road).
              Smaller values → vehicles are following each other closely.

  v_lead      Speed (m/s) of the leading vehicle.
              Equals v_sumo_in when gap=100 (no leader, free-road condition).

  a_idm       Acceleration (m/s²) PyTorch IDM computed.
              Negative → already near/above v0={v0:.2f} m/s, gently braking.
              Positive → below desired speed or recovering from a gap, accelerating.
              Closer to 0 → cruising at steady state.

  v_pytorch   v_sumo_in + a_idm × dt  (Euler step in PyTorch).
              This is the speed we WRITE to SUMO via setSpeed().

  v_sumo_out  Speed SUMO reports AFTER simulationStep().
              Must equal v_pytorch — confirms SUMO accepted our command.

  sync_err    |v_pytorch − v_sumo_out|. Zero = perfect synchronisation.
─────────────────────────────────────────────────────────────────────────────
"""


def read_states(vids):
    rows = {}
    for vid in vids:
        v      = traci.vehicle.getSpeed(vid)
        leader = traci.vehicle.getLeader(vid)
        gap    = leader[1] if leader else 100.0
        v_lead = traci.vehicle.getSpeed(leader[0]) if leader else v
        rows[vid] = (v, gap, v_lead)
    return rows


def main():
    sumocfg = build()
    cfg     = load_config()
    dt      = cfg["step_length"]
    idm     = IDMModel()

    traci.start([
        str(SUMO_BIN / "sumo.exe"), "-c", str(sumocfg),
        "--step-length", str(dt),
        "--collision.action", "warn",
    ])

    known   = set()
    lines   = []
    all_err = []

    COL = dict(step=5, veh=14, vin=11, gap=8, vlead=8,
               aidm=8, vpy=11, vout=11, err=10)

    header = (f"{'step':>{COL['step']}}  {'vehicle':<{COL['veh']}} "
              f"{'v_sumo_in':>{COL['vin']}} {'gap':>{COL['gap']}} "
              f"{'v_lead':>{COL['vlead']}} {'a_idm':>{COL['aidm']}} "
              f"{'v_pytorch':>{COL['vpy']}} {'v_sumo_out':>{COL['vout']}} "
              f"{'sync_err':>{COL['err']}}")
    div = "─" * len(header)

    def row(step, vid, vin, gap, vlead, aidm, vpy, vout, err):
        return (f"{step:>{COL['step']}}  {vid:<{COL['veh']}} "
                f"{vin:>{COL['vin']}.4f} {gap:>{COL['gap']}.2f} "
                f"{vlead:>{COL['vlead']}.4f} {aidm:>{COL['aidm']}.4f} "
                f"{vpy:>{COL['vpy']}.4f} {vout:>{COL['vout']}.4f} "
                f"{err:>{COL['err']}.6f}")

    for step in range(STEPS):
        traci.simulationStep()
        current = set(traci.vehicle.getIDList())

        arrived  = current - known
        departed = known - current
        known    = current

        # configure new arrivals
        for vid in arrived:
            traci.vehicle.setSpeedMode(vid, 0)
            lines.append(f"  >>> ARRIVED   step={step:>4}  vehicle={vid}")

        # log departures
        for vid in sorted(departed):
            lines.append(f"  <<< DEPARTED  step={step:>4}  vehicle={vid}")

        vids = list(current)
        if not vids:
            continue

        states = read_states(vids)
        v_in   = torch.tensor([states[v][0] for v in vids])
        gap_t  = torch.tensor([states[v][1] for v in vids])
        vl_t   = torch.tensor([states[v][2] for v in vids])

        accel  = idm(v_in, gap_t, vl_t)
        v_next = torch.clamp(v_in + accel * dt, min=0.0)

        for i, vid in enumerate(vids):
            traci.vehicle.setSpeed(vid, float(v_next[i]))

        traci.simulationStep()

        # some vehicles may have departed during this second step
        still_active = set(traci.vehicle.getIDList())

        lines.append(div)
        lines.append(f"  t = {step * dt:.1f}s  (step {step})  |  "
                     f"active: {len(vids)}  arrived: {len(arrived)}  "
                     f"departed: {len(departed)}")
        lines.append(header)
        lines.append(div)
        for i, vid in enumerate(sorted(vids)):
            idx = vids.index(vid)
            if vid not in still_active:
                # vehicle left during the confirmation step — log it, skip sync check
                lines.append(f"  {'':>{COL['step']}}  {vid:<{COL['veh']}}  "
                              f"[departed before v_sumo_out could be read]")
                continue
            vout = traci.vehicle.getSpeed(vid)
            e    = abs(v_next[idx].item() - vout)
            all_err.append(e)
            lines.append(row(step, vid,
                             v_in[idx].item(), gap_t[idx].item(), vl_t[idx].item(),
                             accel[idx].item(), v_next[idx].item(), vout, e))

    traci.close()

    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "w") as f:
        f.write("CO-SIMULATION SYNCHRONIZATION VERIFICATION\n")
        f.write(f"Run at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Steps  : {STEPS}  ({STEPS * dt:.1f}s simulated)\n")
        f.write(f"dt     : {dt}s\n")
        f.write(f"IDM    : v0={idm.v0.item():.2f} m/s  T={idm.T.item():.2f}s  "
                f"a={idm.a.item():.2f} m/s²  b={idm.b.item():.2f} m/s²  "
                f"s0={idm.s0.item():.2f}m\n")
        f.write(LEGEND.format(dt=dt, v0=idm.v0.item()))
        f.write("\nSIMULATION LOG\n")
        f.write("\n".join(lines))
        f.write(f"\n\n{'─'*60}\n")
        f.write("SUMMARY\n")
        f.write(f"  Total data rows : {len(all_err)}\n")
        f.write(f"  Max  sync_err   : {max(all_err):.6f}\n")
        f.write(f"  Mean sync_err   : {sum(all_err)/len(all_err):.6f}\n")
        f.write(f"  Perfect sync    : {'YES' if max(all_err) == 0.0 else 'NO'}\n")

    print(f"Log written → {LOG_PATH}")
    print(f"Perfect sync: {'YES' if max(all_err) == 0.0 else 'NO'}  |  "
          f"Max err: {max(all_err):.6f}")


if __name__ == "__main__":
    main()
