"""run_sweep.py — controller comparison sweep.

3 controllers x 3 flows x 25 seeds, 120 s each (225 SUMO runs):
    cosim_nn    OUR controller in SUMO with the trained transformer prior mean
    cosim_nonn  OUR controller in SUMO, plain zero-mean kernel (no NN)
    krauss      native SUMO Krauss (SUMO fully in control, same net + Poisson demand)

Per run we store: average velocity over the whole simulation (mean speed across all
vehicle-steps), the training hinge evaluated on real SUMO positions (integral and
per-second average), steady-state throughput in veh/h (arrivals from t=40 s on, as
cosim_sumo already computes it), plus arrivals and collision counts.

Results land in outputs/sweep_results.json (rewritten after every run, so a partial
sweep is never lost) with per-(controller, flow) averages over the 25 seeds.

    conda run -n car-following-sumo python run_sweep.py
"""
import os, json, time
import numpy as np
import torch
import traci, sumolib
import traci.constants as tc

import utils
import cosim_sumo as C
import sim_torch as S

T_END  = 120.0                      # 120-s horizon for EVERY run in the sweep
FLOWS  = [300, 400, 500]
SEEDS  = list(range(25))
HERE   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(HERE, "outputs")
OUT_JSON = os.path.join(OUT_DIR, "sweep_results.json")

C.T_END = T_END                     # cosim_sumo.run() reads its module-level horizon

AX = torch.tensor([C.ORIGIN[m][0] for m in C._MOVES])   # axis by movement index
SG = torch.tensor([C.ORIGIN[m][1] for m in C._MOVES])   # sign by movement index


def load_mean_model():
    ckpt = os.path.join(HERE, "mean_net_ckpt.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError("mean_net_ckpt.pt not found — cosim_nn needs the "
                                "trained prior mean")
    import mean_net
    ck = torch.load(ckpt, weights_only=True)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    m = mean_net.MeanTransformer()
    m.load_state_dict(sd)
    m.eval()
    return m


def run_cosim(flow, seed, mean_model):
    r = C.run(flow, seed, mean_model=mean_model, hinge_probe=True)
    return dict(avg_velocity=r["avg_speed"],
                hinge_integral=r["hinge"], avg_hinge=r["hinge"] / T_END,
                vph=r["ss_rate"], arrived=r["arrived"], collided=r["collided"])


def run_krauss(flow, seed):
    """Native Krauss with the SAME measurement: mean speed over all vehicle-steps and
    the SAME hinge probe (relu(D_SAFE_2D - d)^2 over cross-axis pairs near the box,
    SUMO front position shifted L/2 back to the centre) computed from raw SUMO state."""
    routes = os.path.join(C.HERE, f"_krauss_routes_{flow}.rou.xml")
    C.write_routes(flow, routes, depart_speed="desired")   # native Krauss insertion
    sumo = sumolib.checkBinary("sumo")
    cmd = [sumo, "-n", os.path.join(C.HERE, "intersection.net.xml"),
           "-r", routes, "--begin", "0", "--end", str(T_END),
           "--step-length", str(C.DT), "--seed", str(seed),
           "--no-step-log", "true", "--no-warnings", "true",
           "--collision.action", "warn", "--collision.check-junctions", "true"]
    traci.start(cmd)                                       # NO speed override → SUMO drives
    n_steps = int(T_END / C.DT)
    half = utils.L_VEH / 2.0
    mv_id = {m: i for i, m in enumerate(C._MOVES)}
    configured, move_of = set(), {}
    arrived, collided = [], set()
    hinge_total, speed_sum, speed_cnt = 0.0, 0.0, 0
    for _ in range(n_steps):
        ids = traci.vehicle.getIDList()
        for v in ids:
            if v not in configured:
                traci.vehicle.subscribe(v, (tc.VAR_POSITION, tc.VAR_SPEED))
                move_of[v] = mv_id.get(traci.vehicle.getRoute(v)[0], 0)
                configured.add(v)
        sub = traci.vehicle.getAllSubscriptionResults()
        vehs = [v for v in ids if v in sub]
        if vehs:
            xs = torch.tensor([sub[v][tc.VAR_POSITION][0] for v in vehs])
            ys = torch.tensor([sub[v][tc.VAR_POSITION][1] for v in vehs])
            vs = torch.tensor([sub[v][tc.VAR_SPEED]       for v in vehs])
            speed_sum += float(vs.sum()); speed_cnt += len(vehs)
            mv = torch.tensor([move_of[v] for v in vehs])
            axis, sgn = AX[mv], SG[mv]
            cx = xs - torch.where(axis == 0, sgn, torch.zeros_like(sgn)) * half
            cy = ys - torch.where(axis == 1, sgn, torch.zeros_like(sgn)) * half
            coord  = torch.where(axis == 0, xs, ys)
            d_junc = (200.0 - coord) * sgn
            near = (d_junc > -S.JCT_PAST) & (d_junc < 60.0)
            ni = near.nonzero().flatten()
            if len(ni) >= 2:
                dist = ((cx[ni].unsqueeze(0) - cx[ni].unsqueeze(1)) ** 2
                        + (cy[ni].unsqueeze(0) - cy[ni].unsqueeze(1)) ** 2).sqrt()
                cross = axis[ni].unsqueeze(0) != axis[ni].unsqueeze(1)
                cross = cross & torch.triu(torch.ones_like(cross), 1)
                hinge_total += float((torch.relu(S.D_SAFE_2D - dist[cross]) ** 2).sum()) * C.DT
        traci.simulationStep()
        arrived.append(traci.simulation.getArrivedNumber())
        collided.update(traci.simulation.getCollidingVehiclesIDList())
    i40 = int(40 / C.DT)
    ss_rate = float(np.sum(arrived[i40:]) / ((n_steps - i40) * C.DT) * 3600)
    traci.close()
    return dict(avg_velocity=speed_sum / max(speed_cnt, 1),
                hinge_integral=hinge_total, avg_hinge=hinge_total / T_END,
                vph=ss_rate, arrived=int(np.sum(arrived)), collided=len(collided))


def aggregates(records):
    out = {}
    for ctrl in ("cosim_nn", "cosim_nonn", "krauss"):
        out[ctrl] = {}
        for flow in FLOWS:
            cell = [r for r in records if r["controller"] == ctrl and r["flow"] == flow]
            if not cell:
                continue
            out[ctrl][str(flow)] = dict(
                n_seeds=len(cell),
                avg_velocity=float(np.mean([r["avg_velocity"] for r in cell])),
                avg_hinge=float(np.mean([r["avg_hinge"] for r in cell])),
                hinge_integral=float(np.mean([r["hinge_integral"] for r in cell])),
                vph=float(np.mean([r["vph"] for r in cell])),
                arrived=float(np.mean([r["arrived"] for r in cell])),
                collided_total=int(np.sum([r["collided"] for r in cell])))
    return out


def save(records):
    payload = dict(
        meta=dict(t_end_s=T_END, dt=C.DT, flows_vph_per_approach=FLOWS,
                  n_seeds=len(SEEDS), seeds=SEEDS,
                  vph_window="steady-state arrivals from t=40s, scaled to veh/h",
                  avg_hinge="hinge integral relu(D_SAFE_2D - d)^2 * dt / T_END"),
        averages=aggregates(records),
        runs=records)
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, OUT_JSON)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mean_model = load_mean_model()
    runners = [("cosim_nn",   lambda fl, sd: run_cosim(fl, sd, mean_model)),
               ("cosim_nonn", lambda fl, sd: run_cosim(fl, sd, None)),
               ("krauss",     run_krauss)]
    records, total = [], len(runners) * len(FLOWS) * len(SEEDS)
    t_start, k = time.time(), 0
    for ctrl, fn in runners:
        for flow in FLOWS:
            for seed in SEEDS:
                t0 = time.time()
                rec = dict(controller=ctrl, flow=flow, seed=seed, **fn(flow, seed))
                records.append(rec)
                save(records)
                k += 1
                print(f"[{k:3d}/{total}] {ctrl:10s} flow={flow} seed={seed:2d}  "
                      f"throughput={rec['vph']:5.0f} veh/h  "
                      f"v_avg={rec['avg_velocity']:5.2f} m/s  "
                      f"hinge/s={rec['avg_hinge']:7.3f}  "
                      f"col={rec['collided']}  ({time.time() - t0:4.1f}s)", flush=True)
    print(f"\nDONE: {total} runs in {(time.time() - t_start) / 60:.1f} min "
          f"-> {OUT_JSON}")
    for ctrl, cells in aggregates(records).items():
        for flow, a in cells.items():
            print(f"  {ctrl:10s} {flow} vph: v_avg={a['avg_velocity']:5.2f} m/s  "
                  f"hinge/s={a['avg_hinge']:7.3f}  vph={a['vph']:5.0f}  "
                  f"collided_total={a['collided_total']}")


if __name__ == "__main__":
    import sys
    if "smoke" in sys.argv[1:]:        # quick check: 1 seed, 1 flow, throwaway output
        FLOWS, SEEDS = [400], [0]
        OUT_JSON = os.path.join(OUT_DIR, "_smoke.json")
    main()
