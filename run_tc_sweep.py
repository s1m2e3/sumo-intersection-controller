"""Throughput + conflict-gap sweep: OUR model (no NN) vs native Krauss.

Two controllers × flows {300,400,500,600} vph/approach × N seeds × 120 s:
    cosim_nonn  OUR kernel controller in SUMO, plain zero-mean kernel (no NN)
    krauss      native SUMO Krauss (SUMO fully in control, same net + demand)

Per run we record steady-state throughput (veh/h, arrivals from t=40 s) and the
conflict time-gap τ_c (= utils.conflict_time_gap: per ego, min |ETA_ego − ETA_rival|
over valid crossing rivals, soft-clamped at τ_c_max).  τ_c is computed from the
SAME per-pair conflict-point geometry for BOTH controllers, so it is apples-to-apples:
for cosim_nonn via cosim_sumo's built-in conflict_probe, for krauss by replaying the
same geometry on raw SUMO state.  We keep the per-run min / mean / max of τ_c.

Results → outputs/tc_sweep.json (rewritten after every run).

    conda run -n car-following-sumo python run_tc_sweep.py [smoke]
"""
import os, json, time, sys
import numpy as np
import torch
import traci, sumolib
import traci.constants as tc

import utils
import cosim_sumo as C

T_END  = 120.0
FLOWS  = [300, 400, 500, 600]
SEEDS  = list(range(10))
HERE   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(HERE, "outputs")
OUT_JSON = os.path.join(OUT_DIR, "tc_sweep.json")

C.T_END = T_END                       # cosim_sumo.run() reads its module-level horizon

AX = torch.tensor([C.ORIGIN[m][0] for m in C._MOVES])   # axis by movement index
SG = torch.tensor([C.ORIGIN[m][1] for m in C._MOVES])   # sign by movement index


def _extract(r):
    return dict(vph=r["ss_rate"], arrived=r["arrived"], collided=r["collided"],
                tau_c_min=r["tau_c_min"], tau_c_mean=r["tau_c_mean"],
                tau_c_max=r["tau_c_max"])


def run_cosim_nonn(flow, seed):
    return _extract(C.run(flow, seed, mean_model=None, conflict_probe=True))


def run_cosim_nn(flow, seed, model):
    """OUR controller with the TRAINED MLP prior mean attached."""
    return _extract(C.run(flow, seed, mean_model=model, conflict_probe=True))


def load_mlp(kind="best"):
    import mean_net, torch
    p = os.path.join(HERE, mean_net.ckpt_path(kind))
    if not os.path.exists(p):
        raise SystemExit(f"{os.path.basename(p)} not found — train the MLP first.")
    ck = torch.load(p, weights_only=True)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    m = mean_net.make_mean_model(); m.load_state_dict(sd); m.eval()
    print(f"loaded trained prior mean from {os.path.basename(p)} (kind={kind})")
    return m


def run_krauss(flow, seed):
    """Native Krauss; τ_c probe replays cosim_sumo's per-pair conflict-point geometry
    on raw SUMO state (positions + speeds), so it matches cosim_nonn's τ_c exactly."""
    net_path = os.path.join(C.HERE, "intersection.net.xml")
    routes   = os.path.join(C.HERE, f"_krauss_routes_{flow}.rou.xml")
    C.write_routes(flow, routes, depart_speed="desired")     # native Krauss insertion
    cpx, cpy = C._cp_tensors(net_path)                       # [4,4] per-pair CP
    mv_id = {m: i for i, m in enumerate(C._MOVES)}

    sumo = sumolib.checkBinary("sumo")
    cmd = [sumo, "-n", net_path, "-r", routes, "--begin", "0", "--end", str(T_END),
           "--step-length", str(C.DT), "--seed", str(seed),
           "--no-step-log", "true", "--no-warnings", "true",
           "--collision.action", "warn", "--collision.check-junctions", "true"]
    traci.start(cmd)                                         # NO speed override → SUMO drives
    n_steps = int(T_END / C.DT)
    configured, move_of = set(), {}
    arrived, collided = [], set()
    tc_min, tc_max, tc_sum, tc_n = float("inf"), 0.0, 0.0, 0

    for _ in range(n_steps):
        ids = traci.vehicle.getIDList()
        for v in ids:
            if v not in configured:
                traci.vehicle.subscribe(v, (tc.VAR_POSITION, tc.VAR_SPEED))
                move_of[v] = mv_id.get(traci.vehicle.getRoute(v)[0], 0)
                configured.add(v)
        sub  = traci.vehicle.getAllSubscriptionResults()
        vehs = [v for v in ids if v in sub]
        N = len(vehs)
        if N >= 2:
            xs = torch.tensor([sub[v][tc.VAR_POSITION][0] for v in vehs])
            ys = torch.tensor([sub[v][tc.VAR_POSITION][1] for v in vehs])
            vs = torch.tensor([sub[v][tc.VAR_SPEED]       for v in vehs])
            mv = torch.tensor([move_of[v]                 for v in vehs])
            axis, sgn = AX[mv], SG[mv]
            # per-pair conflict points → ego_d, rival_d  (mirrors cosim_sumo lines 276–287)
            CPx = cpx[mv][:, mv]; CPy = cpy[mv][:, mv]
            xi, yi, axi, sgi = xs.unsqueeze(1), ys.unsqueeze(1), axis.unsqueeze(1), sgn.unsqueeze(1)
            xj, yj, axj, sgj = xs.unsqueeze(0), ys.unsqueeze(0), axis.unsqueeze(0), sgn.unsqueeze(0)
            ego_d   = torch.where(axi == 0, (CPx - xi) * sgi, (CPy - yi) * sgi)
            rival_d = torch.where(axj == 0, (CPx - xj) * sgj, (CPy - yj) * sgj)
            rival_v = vs.unsqueeze(0).expand(N, N)
            eye   = torch.eye(N, dtype=torch.bool)
            valid = ((axi != axj) & (~eye) & (ego_d > 0.0) & (rival_d > -C.CLEAR)
                     & ~torch.isnan(CPx))
            ego_d   = torch.nan_to_num(ego_d,   nan=1e3)
            rival_d = torch.nan_to_num(rival_d, nan=-1e3)
            has_conf = valid.any(dim=1)
            if bool(has_conf.any()):
                tau_c_all, *_ = utils.conflict_time_gap(ego_d, vs, rival_d, rival_v, valid)
                tcv = tau_c_all[has_conf]
                tc_min = min(tc_min, float(tcv.min()))
                tc_max = max(tc_max, float(tcv.max()))
                tc_sum += float(tcv.sum()); tc_n += int(tcv.numel())
        traci.simulationStep()
        arrived.append(traci.simulation.getArrivedNumber())
        collided.update(traci.simulation.getCollidingVehiclesIDList())

    i40 = int(40 / C.DT)
    ss_rate = float(np.sum(arrived[i40:]) / ((n_steps - i40) * C.DT) * 3600)
    traci.close()
    return dict(vph=ss_rate, arrived=int(np.sum(arrived)), collided=len(collided),
                tau_c_min=(tc_min if tc_n else float("nan")),
                tau_c_mean=(tc_sum / tc_n if tc_n else float("nan")),
                tau_c_max=(tc_max if tc_n else float("nan")))


def aggregates(records):
    out = {}
    flows = sorted({r["flow"] for r in records})
    for ctrl in sorted({r["controller"] for r in records}):
        out[ctrl] = {}
        for flow in flows:
            cell = [r for r in records if r["controller"] == ctrl and r["flow"] == flow]
            if not cell:
                continue
            def m(k):  # nan-safe mean over seeds
                vals = [r[k] for r in cell if not np.isnan(r[k])]
                return float(np.mean(vals)) if vals else float("nan")
            out[ctrl][str(flow)] = dict(
                n_seeds=len(cell), vph=m("vph"),
                tau_c_min=m("tau_c_min"), tau_c_mean=m("tau_c_mean"),
                tau_c_max=m("tau_c_max"), collided_total=int(np.sum([r["collided"] for r in cell])))
    return out


def save(records):
    payload = dict(
        meta=dict(t_end_s=T_END, dt=C.DT,
                  flows_vph_per_approach=sorted({r["flow"] for r in records}),
                  n_seeds=len(SEEDS), seeds=SEEDS,
                  vph_window="steady-state arrivals from t=40s, scaled to veh/h",
                  tau_c="conflict_time_gap (s): per ego min|ETA_ego-ETA_rival| over valid "
                        "crossing rivals, soft-clamped at τ_c_max; min/mean/max over the run"),
        averages=aggregates(records), runs=records)
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, OUT_JSON)


def main(run_flows=None, runners=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    run_flows = run_flows or FLOWS
    runners = runners or [("cosim_nonn", run_cosim_nonn), ("krauss", run_krauss)]
    # merge: keep existing runs for flows we are NOT re-running, append the new ones
    records = []
    if os.path.exists(OUT_JSON):
        records = [r for r in json.load(open(OUT_JSON)).get("runs", [])
                   if r["flow"] not in run_flows]
        if records:
            print(f"merging into {len(records)} existing runs "
                  f"(flows {sorted({r['flow'] for r in records})})")
    total = len(runners) * len(run_flows) * len(SEEDS)
    t_start, k = time.time(), 0
    for ctrl, fn in runners:
        for flow in run_flows:
            for seed in SEEDS:
                t0 = time.time()
                rec = dict(controller=ctrl, flow=flow, seed=seed, **fn(flow, seed))
                records.append(rec); save(records); k += 1
                print(f"[{k:3d}/{total}] {ctrl:10s} flow={flow} seed={seed:2d}  "
                      f"vph={rec['vph']:5.0f}  τc[min/mean/max]="
                      f"{rec['tau_c_min']:.2f}/{rec['tau_c_mean']:.2f}/{rec['tau_c_max']:.2f}  "
                      f"col={rec['collided']}  ({time.time()-t0:4.1f}s)", flush=True)
    print(f"\nDONE: {total} runs in {(time.time()-t_start)/60:.1f} min -> {OUT_JSON}")
    for ctrl, cells in aggregates(records).items():
        for flow, a in cells.items():
            print(f"  {ctrl:10s} {flow} vph: throughput={a['vph']:5.0f}  "
                  f"τc_max={a['tau_c_max']:.2f}  τc_min={a['tau_c_min']:.2f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    flows = [int(a) for a in args if a.isdigit()] or None
    if "smoke" in args:
        FLOWS, SEEDS = [400], [0]
        OUT_JSON = os.path.join(OUT_DIR, "_tc_smoke.json")
        main()
    elif "nn" in args:
        # OUR controller + TRAINED MLP prior mean.  Pick checkpoint with best|last
        # (default best); each writes its own file so results never clobber.
        kind = "last" if "last" in args else "best"
        import mean_net
        tag = mean_net.ARCH                                   # e.g. mlp / mlp_ctx
        OUT_JSON = os.path.join(OUT_DIR, f"tc_sweep_{tag}_{kind}.json")
        model = load_mlp(kind)
        main(run_flows=flows, runners=[("cosim_nn", lambda fl, sd: run_cosim_nn(fl, sd, model))])
    else:
        # numeric args → run ONLY those flows and merge into the existing JSON
        main(run_flows=flows)
