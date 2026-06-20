"""SUMO-native Krauss baseline on the SAME turn scenario as run_turns (same net, same
routes/seeds, same priority junction).  SUMO drives entirely (no TraCI speed control, its
own car-following + right-of-way + collision avoidance).  Throughput computed identically
to run_turns (steady-state after 40 s) so the numbers are directly comparable.

    conda run -n car-following-sumo python _krauss_turns.py [seed ...]
"""
import os, sys
from collections import defaultdict
import numpy as np, traci, sumolib
import traci.constants as tc
import cosim_sumo as C, turns_geom as G
import run_turns as R

DT, T_END = C.DT, 120.0
net = os.path.join(C.HERE, "intersection.net.xml")
routes = os.path.join(C.HERE, "_turn_routes.rou.xml")
mv = G.movements(net)
mv_of_route = {(m.frm, m.to): m.idx for m in mv}
DIRNAME = {m.idx: m.dir for m in mv}


def run(seed):
    R.write_turn_routes(routes, {"l": 200.0, "s": 300.0, "r": 200.0})  # match run_turns l=200 s=300 r=200
    traci.start([sumolib.checkBinary("sumo"), "-n", net, "-r", routes, "--begin", "0",
                 "--end", str(T_END), "--step-length", str(DT), "--seed", str(seed),
                 "--no-step-log", "true", "--no-warnings", "true",
                 "--collision.action", "warn", "--collision.check-junctions", "true",
                 "--time-to-teleport", "-1"])
    n_steps = int(T_END / DT)
    arrived = []; collided = set()
    mv_of = {}; dep = defaultdict(int); arr = defaultdict(int)
    for step in range(n_steps):
        for v in traci.simulation.getDepartedIDList():
            r = traci.vehicle.getRoute(v)
            mv_of[v] = mv_of_route.get((r[0], r[-1]), 0); dep[mv_of[v]] += 1
        traci.simulationStep()
        arrived.append(traci.simulation.getArrivedNumber())
        for v in traci.simulation.getArrivedIDList():
            if v in mv_of: arr[mv_of[v]] += 1
        collided.update(traci.simulation.getCollidingVehiclesIDList())
    i40 = int(40 / DT)
    ss = float(np.sum(arrived[i40:]) / ((n_steps - i40) * DT) * 3600)
    traci.close()
    dd = defaultdict(int); ad = defaultdict(int)
    for m in range(len(mv)):
        dd[DIRNAME[m]] += dep[m]; ad[DIRNAME[m]] += arr[m]
    served = {d: (ad[d], dd[d]) for d in ("s", "l", "r")}
    return dict(vph=ss, arrived=int(np.sum(arrived)), collided=len(collided), served=served)


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or list(range(10))
    tot_v = tot_c = 0
    for s in seeds:
        r = run(s)
        sv = r["served"]
        srv = "  ".join(f"{d}:{sv[d][0]}/{sv[d][1]}" for d in ("s", "l", "r"))
        print(f"seed {s}: coll={r['collided']:2d}  vph={r['vph']:4.0f}  served[{srv}]")
        tot_v += r["vph"]; tot_c += r["collided"]
    print(f"KRAUSS TOT:  collisions={tot_c}  mean vph={tot_v/len(seeds):.0f}")
