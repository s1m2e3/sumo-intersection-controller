"""
cosim_sumo.py — single entry point for intersection simulation.

Three controller modes (controller= CLI arg):
  kernel   OUR GP signal-phase kernel (utils.SignalController).  SUMO is a
           black-box dynamics engine (speedMode=0); all yield/stop/role logic
           is ours and differentiable.
  nn       PhaseNet adaptive signal controller — loads signal_nn_best.pt (or
           model=<path>); drives cycle timing with the trained NN, otherwise
           identical to kernel mode (speedMode=0, our gap/yield logic).
  sumo     SUMO's own fixed-time signal controller (intersection_tl.net.xml).
           No TraCI speed override — pure Krauss baseline for comparison.

Per-approach demand:  s=<vph>  l=<vph>  r=<vph>  (veh/h per approach)
Geometry is loaded from turns_geom and passed to the controller — swap the
net file to run on a different intersection without touching utils.py.

    conda run -n car-following-sumo python cosim_sumo.py \\
        [controller=kernel|nn|sumo] [model=<ckpt>] [s=<vph>] [l=<vph>] [r=<vph>] \\
        [seed=<n>] [gui] [speed=<x>] [end=<s>] [gridlock_s=<s>] [log=<path>]
"""
import os, sys, time, json, datetime
import numpy as np
import torch
import traci, sumolib
import traci.constants as tc

import utils

HERE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_files")
DT     = 0.1
T_END  = 120.0
V_PHYS = utils.V0
CLEAR  = utils.L_VEH + 1.0   # clearance past a conflict point (used by sim_torch)

# entry-edge → (axis 0=x/EW 1=y/NS, travel sign ±1)
# kept here because turns_geom and sim_torch import this module
ORIGIN = {
    "east_in":  (0, -1.0),
    "west_in":  (0, +1.0),
    "north_in": (1, -1.0),
    "south_in": (1, +1.0),
}

# 4-movement straight-crossing constants (used by sim_torch for backward compat)
_MOVES   = ["east_in", "west_in", "north_in", "south_in"]
_MOVE_TO = {"east_in": "west_out", "west_in": "east_out",
            "north_in": "south_out", "south_in": "north_out"}

_CP_CACHE: dict = {}

def conflict_points(net_path):
    """Per-pair geometric crossing of the 4 straight movements' paths (used by sim_torch)."""
    if net_path in _CP_CACHE:
        return _CP_CACHE[net_path]
    net = sumolib.net.readNet(net_path, withInternal=True)
    shapes = {}
    for frm, to in _MOVE_TO.items():
        conn   = net.getEdge(frm).getConnections(net.getEdge(to))[0]
        via    = conn.getViaLaneID()
        vshape = list(net.getLane(via).getShape()) if via else []
        shapes[frm] = ([conn.getFromLane().getShape()[-1]] + vshape
                       + [conn.getToLane().getShape()[0]])
    CP = {(a, b): _poly_x(shapes[a], shapes[b])
          for a in _MOVES for b in _MOVES if a != b}
    _CP_CACHE[net_path] = CP
    return CP

ROLE_COLOR = {
    "pass":  ( 40, 200,  40, 255),   # green
    "yield": (220,  40,  40, 255),   # red
}


# ── polyline geometry helpers (used by turns_geom via C._poly_x) ─────────────

def _seg_x(p1, p2, p3, p4):
    """Intersection of segments p1p2 and p3p4, or None."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / d
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def _poly_x(A, B):
    """First intersection of two polylines, or None."""
    for i in range(len(A) - 1):
        for j in range(len(B) - 1):
            pt = _seg_x(A[i], A[i + 1], B[j], B[j + 1])
            if pt is not None:
                return pt
    return None


# ── route writing ─────────────────────────────────────────────────────────────

def write_routes(path, vph, net_path):
    """One SUMO <flow> per movement at the per-direction demand vph
    (dict s/l/r → veh/h per approach).  Zero-demand directions emit no vehicles,
    so l=0 r=0 collapses to the pure straight-cross scenario."""
    import turns_geom as G
    mv = G.movements(net_path)
    with open(path, "w") as f:
        f.write('<routes>\n'
                '  <vType id="car" accel="2.6" decel="4.5" sigma="0" length="5" '
                'minGap="1.0" maxSpeed="13.89" carFollowModel="Krauss"/>\n')
        for m in mv:
            rate = vph.get(m.dir, 0) / 3600.0
            if rate <= 0.0:
                continue
            f.write(f'  <flow id="f{m.idx}" type="car" from="{m.frm}" to="{m.to}" '
                    f'begin="0" end="{T_END:.0f}" period="exp({rate:.5f})" '
                    f'departLane="best" departSpeed="{V_PHYS}"/>\n')
        f.write('</routes>\n')


# ── simulation runner ─────────────────────────────────────────────────────────

def run(vph=None, seed=0, gui=False, gui_speed=3.0, t_end=None,
        controller="kernel", nn_model=None, gridlock_s=40.0, logfile=None):
    """
    Run the intersection simulation.

    Args:
        vph          dict s/l/r → veh/h per approach  (default 200/100/100)
        seed         SUMO random seed
        gui          launch sumo-gui
        gui_speed    GUI playback speed (>1 = faster than real-time; 0 = unthrottled)
        t_end        simulation end time in seconds  (default T_END=120)
        controller   'kernel'  our GP signal-phase controller (differentiable)
                     'nn'      PhaseNet adaptive controller (loads checkpoint)
                     'sumo'    SUMO's own fixed-time TL (Krauss baseline)
        nn_model     path to PhaseNet checkpoint (default: signal_nn_best.pt)
        gridlock_s   stop if no arrivals for this many seconds  (0 = disabled)
        logfile      append one JSON result line to this file

    Returns:
        dict  vph (steady-state throughput), arrived, collided, controller, served
    """
    import turns_geom as G

    vph    = vph or {"s": 200, "l": 100, "r": 100}
    _t_end = t_end or T_END

    # net selection: kernel/nn use the no-TL net (we control everything);
    # sumo mode uses the TL net so SUMO enforces signal phases itself.
    if controller in ("kernel", "nn"):
        net_path = os.path.join(HERE, "intersection.net.xml")
    else:
        net_path = os.path.join(HERE, "intersection_tl.net.xml")

    routes = os.path.join(HERE, "_cosim_routes.rou.xml")
    write_routes(routes, vph, net_path)

    # geometry (needed for kernel controller and for movement → route mapping)
    mv          = G.movements(net_path)
    M           = len(mv)
    mv_of_route = {(m.frm, m.to): m.idx for m in mv}
    DIRNAME     = {m.idx: m.dir for m in mv}

    # kernel/nn-mode geometry: conflict matrix + arc-lengths + merge conflicts
    ctrl     = None
    s_junc_g = None
    if controller in ("kernel", "nn"):
        _geo, _s_cp, s_junc_g, CONF = G.gate_geometry(net_path)
        AX = torch.tensor([m.axis for m in mv])
        SG = torch.tensor([m.sgn  for m in mv])
        CONF = CONF.clone()
        MERGE = torch.zeros((M, M), dtype=torch.bool)
        for _i in range(M):
            for _j in range(M):
                if _i != _j and mv[_i].frm != mv[_j].frm and mv[_i].to == mv[_j].to:
                    CONF[_i, _j]  = True   # add merge conflicts (same exit road)
                    MERGE[_i, _j] = True   # same exit → funnel/follow, not crossing
        ctrl = utils.SignalController(conf=CONF, s_junc=s_junc_g, s_cp=_s_cp, merge=MERGE)

    # nn-mode: load PhaseNet checkpoint
    nn_net = None
    if controller == "nn":
        import signal_nn as SN
        _model_path = nn_model or SN.BEST_CKPT
        if os.path.exists(_model_path):
            _ckpt = torch.load(_model_path, map_location="cpu", weights_only=True)
            _arch = _ckpt.get("arch", {})
            if "n_in" not in _arch:
                _arch["n_in"] = int(_ckpt["model"]["embed.0.weight"].shape[1])
            _arch.setdefault("hidden",  32)
            _arch.setdefault("n_heads", 2)
            _arch.setdefault("n_layers", 1)
            nn_net = SN.PhaseNet(**_arch).to(torch.device("cpu"))
            nn_net.load_state_dict(_ckpt["model"])
            print(f"  [nn] Loaded {_model_path!r}  "
                  f"epoch={_ckpt.get('epoch','?')}  loss={_ckpt.get('loss',float('nan')):.1f}"
                  f"  arch={_arch}")
        else:
            import signal_nn as SN
            print(f"  [nn] WARNING: no checkpoint at {_model_path!r} — random weights")
            _arch  = {"n_in": 4, "hidden": 32, "n_heads": 2, "n_layers": 1}
            nn_net = SN.PhaseNet(**_arch).to(torch.device("cpu"))
        nn_net.eval()

    sumo_bin = sumolib.checkBinary("sumo-gui" if gui else "sumo")
    cmd = [sumo_bin, "-n", net_path, "-r", routes,
           "--begin", "0", "--end", str(_t_end),
           "--step-length", str(DT), "--seed", str(seed),
           "--no-step-log", "true", "--no-warnings", "true",
           "--collision.action", "warn",
           "--collision.check-junctions", "true",
           "--collision.mingap-factor", "0",
           "--time-to-teleport", "-1"]
    if gui:
        cmd += ["--start", "--quit-on-end", "--delay", "0"]

    traci.start(cmd)
    if gui:
        try:
            traci.gui.setSchema("View #0", "real world")
            traci.gui.setBoundary("View #0", 120, 120, 280, 280)
        except traci.TraCIException:
            pass

    n_steps  = int(_t_end / DT)
    configured, mv_idx, exited = set(), {}, set()
    arrived, collided = [], set()

    if controller == "nn":
        rh_signal = SN.RollingHorizonSignal(nn_net)
    dep_dir = {d: 0 for d in "slr"}
    arr_dir = {d: 0 for d in "slr"}
    _no_arrival_steps = 0
    _DBG = {"departed": 0, "inbox_max": 0, "N_max": 0}
    _collision_log = []
    _run_wall_t0 = time.perf_counter()

    for step in range(n_steps):
        t_wall = time.perf_counter()
        ids    = traci.vehicle.getIDList()

        for v in ids:
            if v not in configured:
                if controller in ("kernel", "nn"):
                    traci.vehicle.setSpeedMode(v, 0)
                    traci.vehicle.setLaneChangeMode(v, 0)
                traci.vehicle.subscribe(v, (tc.VAR_SPEED, tc.VAR_DISTANCE,
                                            tc.VAR_ROAD_ID))
                r = traci.vehicle.getRoute(v)
                mv_idx[v] = mv_of_route.get((r[0], r[-1]), 0)
                dep_dir[DIRNAME[mv_idx[v]]] += 1
                configured.add(v)

        sub  = traci.vehicle.getAllSubscriptionResults()
        vehs = [v for v in ids if v in sub]
        N    = len(vehs)

        if N == 0:
            traci.simulationStep()
            _n = traci.simulation.getArrivedNumber()
            arrived.append(_n)
            if gui and gui_speed > 0:
                time.sleep(max(0.0, DT / gui_speed - (time.perf_counter() - t_wall)))
            continue

        vs      = torch.tensor([sub[v][tc.VAR_SPEED]    for v in vehs])
        mvi_t   = torch.tensor([mv_idx[v]               for v in vehs])
        s_front = torch.tensor([sub[v][tc.VAR_DISTANCE] for v in vehs]) + utils.L_VEH
        on_box  = [str(sub[v][tc.VAR_ROAD_ID]).startswith(":") for v in vehs]

        _DBG["N_max"]     = max(_DBG["N_max"],     N)
        _DBG["inbox_max"] = max(_DBG["inbox_max"], sum(on_box))

        if controller in ("kernel", "nn"):
            d_junc = s_junc_g[mvi_t] - s_front

            # release deep-past-box vehicles to SUMO car-following
            for i, v in enumerate(vehs):
                if float(d_junc[i]) < -20.0 and v not in exited:
                    traci.vehicle.setSpeedMode(v, 31)
                    exited.add(v)

            # same-queue leader gap (left lane separate from through+right)
            gap    = torch.full((N,), 300.0)
            v_lead = vs.clone()
            appr      = mvi_t // 3
            left_lane = (mvi_t % 3 == 2)
            for i in range(N):
                for j in range(N):
                    if (i == j
                            or int(appr[j]) != int(appr[i])
                            or bool(left_lane[i]) != bool(left_lane[j])):
                        continue
                    if float(d_junc[j]) <= -2.0:
                        continue
                    dg = float(s_front[j]) - float(s_front[i]) - utils.L_VEH
                    if 0.0 <= dg < float(gap[i]):
                        gap[i] = dg
                        v_lead[i] = float(vs[j])

            t_now = step * DT

            # nn-mode rolling-horizon: delegate entirely to RollingHorizonSignal
            if controller == "nn":
                with torch.no_grad():
                    nn_green_override, _, _ = rh_signal.step(
                        t_now, s_front, vs, mvi_t, s_junc_g)
            else:
                nn_green_override = None

            a, info = ctrl.step(vehs, mvi_t, vs, gap, v_lead, d_junc, on_box, t_now,
                                green_override=nn_green_override)
            is_yield  = info["is_yield"]
            yield_cap = info["yield_cap"]
            box_cap   = info["box_cap"]
            opp_flags = info["opp"]

            # debug: track opp=1 vehicles inside the junction box
            for _i, _v in enumerate(vehs):
                if float(opp_flags[_i]) > 0.5 and on_box[_i]:
                    print(
                        f"  [OPP-IN-BOX t={t_now:.1f}s] {_v}"
                        f"  mv={mv[int(mvi_t[_i])].frm}.{DIRNAME[int(mvi_t[_i])]}"
                        f"  d={float(d_junc[_i]):.1f}"
                        f"  v={float(vs[_i]):.2f}"
                        f"  a={float(a[_i]):.2f}"
                        f"  yield={int(bool(is_yield[_i]))}"
                        f"  ycap={float(yield_cap[_i]):.2f}"
                        f"  bcap={float(box_cap[_i]):.2f}"
                    )

            for i, v in enumerate(vehs):
                if v in exited:
                    if gui:
                        traci.vehicle.setColor(v, (170, 170, 170, 255))
                    continue
                cap_i = min(float(box_cap[i]), float(yield_cap[i]))
                v_cmd = min(max(float(vs[i]) + float(a[i]) * DT, 0.0), V_PHYS, cap_i)
                traci.vehicle.setSpeed(v, v_cmd)
                if gui:
                    traci.vehicle.setColor(
                        v, ROLE_COLOR["yield" if bool(is_yield[i]) else "pass"])

        traci.simulationStep()
        _DBG["departed"] += traci.simulation.getDepartedNumber()
        _n = traci.simulation.getArrivedNumber()
        arrived.append(_n)
        for v in traci.simulation.getArrivedIDList():
            if v in mv_idx:
                arr_dir[DIRNAME[mv_idx[v]]] += 1
        _no_arrival_steps = (_no_arrival_steps + 1) if (_n == 0 and N >= 10) else 0

        new_coll = set(traci.simulation.getCollidingVehiclesIDList()) - collided
        if new_coll:
            idxmap = {v: i for i, v in enumerate(vehs)}
            rec_list = []
            for cv in sorted(new_coll):
                i = idxmap.get(cv)
                if i is None:
                    rec_list.append({"id": cv})
                else:
                    _opp_i = int(opp_flags[i] > 0.5) if controller in ("kernel", "nn") else 0
                    rec_list.append({
                        "id":  cv,
                        "mv":  f"{mv[int(mvi_t[i])].frm}.{DIRNAME[int(mvi_t[i])]}",
                        "v":   round(float(vs[i]), 2),
                        "d":   round(float(d_junc[i]) if controller in ("kernel", "nn") else 0.0, 1),
                        "box": int(on_box[i]),
                        "opp": _opp_i,
                    })
            parts = [f"{r['id']}[{r.get('mv','?')} d={r.get('d','?')} "
                     f"v={r.get('v','?')} box={r.get('box','?')} opp={r.get('opp','?')}]"
                     for r in rec_list]
            print(f"  [COLLISION @ t={step*DT:.1f}s]  " + "   ".join(parts))
            _collision_log.append({"t": round(step * DT, 1), "vehicles": rec_list})
        collided.update(new_coll)

        if gridlock_s > 0 and _no_arrival_steps * DT >= gridlock_s:
            print(f"  [gridlock @ t={step*DT:.1f}s — no arrivals for {gridlock_s:.0f}s]")
            break

        if gui and gui_speed > 0:
            time.sleep(max(0.0, DT / gui_speed - (time.perf_counter() - t_wall)))

    i40       = int(40 / DT)
    n_run     = len(arrived)
    ss_rate   = float(np.sum(arrived[i40:]) / max((n_run - i40) * DT, 1.0) * 3600)
    residual  = _DBG["departed"] - int(np.sum(arrived))
    traci.close()

    print(f"  [dbg] departed={_DBG['departed']}  arrived={int(np.sum(arrived))}  "
          f"STILL_STUCK={residual}  in_network_peak={_DBG['N_max']}  "
          f"in_box_peak={_DBG['inbox_max']}")
    print("  served by movement:  " +
          "  ".join(f"{d}:{arr_dir[d]}/{dep_dir[d]}" for d in ("s", "l", "r")))

    result = dict(vph=ss_rate, arrived=int(np.sum(arrived)),
                  collided=len(collided), controller=controller,
                  served={d: (arr_dir[d], dep_dir[d]) for d in "slr"})

    if logfile:
        rec = {
            "ts":          datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seed":        seed,
            "controller":  controller,
            "vph_demand":  vph,
            "throughput":  round(ss_rate, 1),
            "arrived":     result["arrived"],
            "collisions":  len(collided),
            "still_stuck": residual,
            "wall_s":      round(time.perf_counter() - _run_wall_t0, 2),
            "collision_list": _collision_log,
        }
        os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    return result


if __name__ == "__main__":
    tokens = sys.argv[1:]

    def _tok(pfx, default):
        t = next((x for x in tokens if x.startswith(pfx)), None)
        return t.split("=", 1)[1] if t is not None else default

    controller = _tok("controller=", "kernel")
    nn_model   = _tok("model=", None)
    gui        = "gui" in tokens
    gui_speed  = float(_tok("speed=", 3.0))
    seed       = int(_tok("seed=", next((t for t in tokens if t.isdigit()), "0")))
    vph = {
        "s": float(_tok("s=", 200)),
        "l": float(_tok("l=", 100)),
        "r": float(_tok("r=", 100)),
    }
    t_end      = next((float(t.split("=", 1)[1]) for t in tokens
                       if t.startswith("end=")), None)
    gridlock_s = float(_tok("gridlock_s=", 40.0))
    logfile    = _tok("log=", None)

    r = run(vph=vph, seed=seed, gui=gui, gui_speed=gui_speed, t_end=t_end,
            controller=controller, nn_model=nn_model,
            gridlock_s=gridlock_s, logfile=logfile)

    print(f"\nCOSIM  L/S/R={vph['l']:.0f}/{vph['s']:.0f}/{vph['r']:.0f} vph/approach"
          f"  seed={seed}  controller={controller}:")
    print(f"  throughput  : {r['vph']:.0f} veh/h  (arrived {r['arrived']})")
    print(f"  collisions  : {r['collided']}")
