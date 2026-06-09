"""
cosim_sumo.py — co-simulate OUR kernel controller inside SUMO via TraCI.

SUMO owns the world (geometry, integration, collision detection); at every step we
read each vehicle's state, compute OUR controller's acceleration, and impose it with
setSpeed (SpeedMode=0 → SUMO applies no safety/right-of-way of its own, so any unsafe
command WILL produce a real SUMO collision).  This validates collision-freeness in an
independent engine and measures throughput on the exact same network as Krauss.

Per-pair conflict points are taken from SUMO's true lane geometry: an x-axis vehicle
at lateral y and a y-axis vehicle at lateral x cross at (x, y); each side's distance
to that point is computed from live positions.

    conda run -n car-following-sumo python cosim_sumo.py [flow] [seed]
"""
import os, sys, math
import numpy as np
import torch
import traci, sumolib
import utils

HERE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_files")
DT     = 0.1
T_END  = 120.0
V_PHYS = 16.0
CLEAR  = utils.L_VEH + 1.0           # rival "still in box" clearance past its crossing

# origin edge -> (axis 0=x/1=y, travel sign along that axis)
ORIGIN = {"east_in": (0, -1.0), "west_in": (0, +1.0),
          "north_in": (1, -1.0), "south_in": (1, +1.0)}


def write_routes(flow, path):
    with open(path, "w") as f:
        f.write('<routes>\n'
                '  <vType id="car" accel="2.6" decel="4.5" sigma="0" length="5" '
                'minGap="2.0" maxSpeed="13.89" carFollowModel="Krauss"/>\n')
        for fid, frm, to in [("ew", "east_in", "west_out"), ("we", "west_in", "east_out"),
                             ("ns", "north_in", "south_out"), ("sn", "south_in", "north_out")]:
            f.write(f'  <flow id="{fid}" type="car" from="{frm}" to="{to}" begin="0" '
                    f'end="120" vehsPerHour="{flow}" departLane="best" departSpeed="desired"/>\n')
        f.write('</routes>\n')


def run(flow=500, seed=0):
    routes = os.path.join(HERE, f"_cosim_routes_{flow}.rou.xml")
    write_routes(flow, routes)
    sumo = sumolib.checkBinary("sumo")
    traci.start([sumo, "-n", os.path.join(HERE, "intersection.net.xml"),
                 "-r", routes, "--begin", "0", "--end", str(T_END),
                 "--step-length", str(DT), "--seed", str(seed),
                 "--no-step-log", "true", "--no-warnings", "true",
                 "--collision.action", "warn", "--collision.check-junctions", "true",
                 "--collision.mingap-factor", "0", "--time-to-teleport", "-1"])

    configured = set()
    collided = set()           # distinct vehicles ever flagged in a collision
    collision_steps = 0
    arrived_series = []
    n_steps = int(T_END / DT)

    for step in range(n_steps):
        vehs = traci.vehicle.getIDList()
        for v in vehs:
            if v not in configured:
                traci.vehicle.setSpeedMode(v, 0)        # full manual control, no SUMO safety
                traci.vehicle.setLaneChangeMode(v, 0)
                configured.add(v)

        # snapshot state
        st = {}
        for v in vehs:
            x, y = traci.vehicle.getPosition(v)
            spd = traci.vehicle.getSpeed(v)
            axis, sgn = ORIGIN[traci.vehicle.getRoute(v)[0]]
            st[v] = (x, y, spd, axis, sgn)

        # compute our acceleration for each vehicle and impose it
        for v in vehs:
            x, y, spd, axis, sgn = st[v]
            ld = traci.vehicle.getLeader(v, 120.0)
            if ld is not None and ld[0] != "":
                gap = max(ld[1], 0.0); v_lead = traci.vehicle.getSpeed(ld[0])
            else:
                gap = 300.0; v_lead = spd

            xe = torch.tensor(0.0); xl = torch.tensor(gap + utils.L_VEH)
            ve = torch.tensor(spd); vl = torch.tensor(v_lead)

            ed, rd, rv, val = [], [], [], []
            for k, (xk, yk, sk, axk, sgk) in st.items():
                if k == v or axk == axis:
                    continue
                if axis == 0:                 # ego travels along x; crossing at x=xk, y=y
                    ego_d = (xk - x) * sgn
                    rival_d = (y - yk) * sgk
                else:                         # ego travels along y; crossing at x=x, y=yk
                    ego_d = (yk - y) * sgn
                    rival_d = (x - xk) * sgk
                ed.append(ego_d); rd.append(rival_d); rv.append(max(sk, utils.EPS))
                val.append((ego_d > 0.0) and (rival_d > -CLEAR))

            if ed:
                a = utils.controller_acceleration(
                    xe, xl, ve, vl,
                    d_conf=torch.tensor(ed), rival_d=torch.tensor(rd),
                    rival_v=torch.tensor(rv), rival_valid=torch.tensor(val))
            else:
                a = utils.controller_acceleration(xe, xl, ve, vl)

            v_new = float(min(max(spd + float(a) * DT, 0.0), V_PHYS))
            traci.vehicle.setSpeed(v, v_new)

        traci.simulationStep()

        col = traci.simulation.getCollidingVehiclesIDList()
        if col:
            collision_steps += 1
            collided.update(col)
        arrived_series.append(traci.simulation.getArrivedNumber())

    # totals
    total_arrived = int(np.sum(arrived_series))
    # steady-state rate over t in [40,120]
    i40 = int(40 / DT)
    ss_rate = np.sum(arrived_series[i40:]) / ((n_steps - i40) * DT) * 3600
    traci.close()
    return dict(flow=flow, seed=seed, arrived=total_arrived,
                ss_rate=ss_rate, collided=len(collided), collision_steps=collision_steps)


if __name__ == "__main__":
    flow = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    r = run(flow, seed)
    print(f"=== OUR controller co-simulated in SUMO (TraCI), {flow} vph/approach, seed {seed} ===")
    print(f"  arrived (cleared, 120s)        : {r['arrived']}")
    print(f"  steady-state throughput        : {r['ss_rate']:.0f} veh/h")
    print(f"  vehicles in a SUMO collision   : {r['collided']}")
    print(f"  steps with a collision         : {r['collision_steps']}")
