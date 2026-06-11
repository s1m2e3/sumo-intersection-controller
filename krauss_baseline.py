"""SUMO native Krauss baseline — SUMO fully in control (no override), same
Poisson routes/net. Reports throughput, collisions, teleports.

usage:
    python krauss_baseline.py [flow] [seed] [gui]   # ONE run on a seed (like cosim_sumo); gui→watch
    python krauss_baseline.py <flows,csv> <nseeds>  # sweep: avg over seeds 0..nseeds-1
"""
import os, sys, time
import numpy as np
import traci, sumolib
import cosim_sumo as C


def run_krauss(flow, seed, gui=False):
    routes = os.path.join(C.HERE, f"_krauss_routes_{flow}.rou.xml")   # isolated filename
    # native baseline keeps departSpeed="desired" (inserts at maxSpeed, Krauss semantics)
    C.write_routes(flow, routes, depart_speed="desired")             # same vType (Krauss) + Poisson
    sumo = sumolib.checkBinary("sumo-gui" if gui else "sumo")
    cmd = [sumo,
           "-n", os.path.join(C.HERE, "intersection.net.xml"),
           "-r", routes, "--begin", "0", "--end", str(C.T_END),
           "--step-length", str(C.DT), "--seed", str(seed),
           "--no-step-log", "true", "--no-warnings", "true",
           "--collision.action", "warn", "--collision.check-junctions", "true"]
    if gui:
        cmd += ["--start", "--quit-on-end", "--delay", "0"]
    traci.start(cmd)                                                 # NO setSpeedMode → SUMO drives
    if gui:
        try:
            traci.gui.setSchema("View #0", "real world")
            traci.gui.setBoundary("View #0", 120, 120, 280, 280)
        except traci.TraCIException:
            pass
    n_steps = int(C.T_END / C.DT)
    arrived, collided, teleports = [], set(), 0
    for _ in range(n_steps):
        t0 = time.perf_counter()
        traci.simulationStep()
        arrived.append(traci.simulation.getArrivedNumber())
        collided.update(traci.simulation.getCollidingVehiclesIDList())
        teleports += traci.simulation.getStartingTeleportNumber()
        if gui:                                                     # pace to human (wall-clock) time
            lag = C.DT - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)
    i40 = int(40 / C.DT)
    ss = np.sum(arrived[i40:]) / ((n_steps - i40) * C.DT) * 3600
    traci.close()
    return int(np.sum(arrived)), ss, len(collided), teleports


if __name__ == "__main__":
    tokens = sys.argv[1:]
    gui = "gui" in tokens
    args = [a for a in tokens if a != "gui"]
    arg1 = args[0] if len(args) > 0 else "500"
    if "," in arg1:
        # SWEEP mode: flows csv, nseeds → average over seeds 0..nseeds-1
        flows = [int(x) for x in arg1.split(",")]
        nseed = int(args[1]) if len(args) > 1 else 8
        print("=== SUMO native Krauss baseline (SUMO in full control, same Poisson demand) ===")
        for flow in flows:
            ss, tot, col, tp = [], 0, 0, 0
            for s in range(nseed):
                a, r, c, t = run_krauss(flow, s)
                ss.append(r); tot += a; col += c; tp += t
            print(f"FLOW {flow:4d}: avg_throughput={sum(ss)/len(ss):4.0f} veh/h  "
                  f"collided_total={col}  teleports={tp}", flush=True)
    else:
        # SINGLE run on a specific seed (matches `cosim_sumo.py <flow> <seed>`)
        flow = int(arg1)
        seed = int(args[1]) if len(args) > 1 else 0
        arrived, ss, col, tp = run_krauss(flow, seed, gui=gui)
        print(f"=== SUMO native Krauss baseline, {flow} vph/approach, seed {seed} ===")
        print(f"  arrived (cleared, 120s)        : {arrived}")
        print(f"  steady-state throughput        : {ss:.0f} veh/h")
        print(f"  vehicles in a SUMO collision   : {col}")
        print(f"  teleports                      : {tp}")
