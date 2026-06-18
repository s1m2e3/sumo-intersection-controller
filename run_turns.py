"""First turn-movement probe: run OUR controller in SUMO with LEFT/THROUGH/RIGHT
movements and measure throughput + conflict gap.

Scope (first cut, deliberately simple):
  • all 12 movements (entry→exit) with TRUE per-pair conflict points (turns_geom);
  • conflict = "paths cross" (CP exists) — catches left-turn vs oncoming-through, etc.;
  • roles from utils.predecessor_gap (timing) PLUS the rule: a TURNING movement yields
    to through traffic unless it has space (a clear predecessor gap) or is promoted;
  • NO 2-D rollout gate / RoleMemory latch (those are still straight-only) — so this is
    the raw controller; collisions are NOT a fair metric here, throughput + τ_c are.

    conda run -n car-following-sumo python run_turns.py [flow] [seed]
"""
import os, sys, time
import numpy as np
import torch
import traci, sumolib
import traci.constants as tc

import utils
import cosim_sumo as C
import turns_geom as G
import sim_torch as S

DT, T_END, V_PHYS = C.DT, 120.0, C.V_PHYS
# per-approach demand by movement direction (vph).  Right turns share the through lane
# (net: r & s both use fromLane 0; l uses fromLane 1) — departLane="best" places them.
VPH = {"l": 300, "s": 500, "r": 300}
ROLE_COLOR = {"yield": (220, 40, 40, 255), "pass": (40, 200, 40, 255),
              "none": (170, 170, 170, 255)}


def write_turn_routes(path):
    mv = G.movements(os.path.join(C.HERE, "intersection.net.xml"))
    with open(path, "w") as f:
        f.write('<routes>\n'
                '  <vType id="car" accel="2.6" decel="4.5" sigma="0" length="5" '
                'minGap="2.0" maxSpeed="13.89" carFollowModel="Krauss"/>\n')
        for m in mv:
            rate = VPH[m.dir] / 3600.0
            f.write(f'  <flow id="f{m.idx}" type="car" from="{m.frm}" to="{m.to}" '
                    f'begin="0" end="120" period="exp({rate:.5f})" '
                    f'departLane="best" departSpeed="{V_PHYS}"/>\n')
        f.write('</routes>\n')


def load_model():
    p = os.path.join(C.HERE if False else os.path.dirname(os.path.abspath(__file__)),
                     __import__("mean_net").ckpt_path("best"))
    import mean_net
    if not os.path.exists(p):
        print("no trained ckpt — running plain kernel (mean_fn=None)"); return None
    ck = torch.load(p, weights_only=True)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    m = mean_net.make_mean_model(); m.load_state_dict(sd); m.eval()
    print(f"loaded trained prior mean: {os.path.basename(p)}")
    return m


def run(seed=0, mean_model=None, gui=False, use_gate=True, box_exclusive=True):
    net = os.path.join(C.HERE, "intersection.net.xml")
    routes = os.path.join(C.HERE, "_turn_routes.rou.xml")
    write_turn_routes(routes)
    mv = G.movements(net)
    M = len(mv)
    geo_g, s_cp_g, s_junc_g, CONF = G.gate_geometry(net)          # 12-movement gate geometry
    GEO_ORDER = list(range(M))                                    # geo_g keyed by movement idx 0..M-1
    COMPAT = ~CONF                                                # movements whose paths DON'T cross
    final_cp = torch.nan_to_num(s_cp_g, nan=-1e9).max(dim=1).values  # [M] LAST conflict point per movement
    AX = torch.tensor([m.axis for m in mv]); SG = torch.tensor([m.sgn for m in mv])
    IS_TURN = torch.tensor([m.dir != "s" for m in mv])
    # Right-of-way PRIORITY bias (seconds of head-start; only shifts who-yields, not spacing).
    # NOTE: a POSITIVE through head-start was tried and REVERTED — making throughs "earliest"
    # floors them to a_max (assert) → they drive into cross traffic the gate can't always catch
    # at saturation (7 collisions vs 2).  Asserting harder is unsafe; only EXTRA YIELDING is.  And
    # box-level yield-forcing (left-metering) was ALSO reverted (see yield block).  So PRIO is 0,
    # and we rebalance the served mix the only safe way: UPSTREAM INFLOW METERING (see meter block).
    PRIO_THROUGH = 0.0
    PRIO = torch.tensor([PRIO_THROUGH if not t else 0.0 for t in IS_TURN])   # [M] per movement
    mv_of_route = {(m.frm, m.to): m.idx for m in mv}
    # ── UPSTREAM INFLOW METERING (closed-loop fairness, decoupled from the box).  Hold the lead
    # approaching vehicle of an OVER-SERVED movement-class at a meter line D_METER m upstream of the
    # box, releasing it once its class falls back to its demand-fair share of throughput.  This
    # reshapes ARRIVAL TIMES so the box's FCFS-by-ETA race naturally serves the starved class — the
    # kernel/gate are untouched (a held vehicle is just a far, stopped, low-priority car, invisible
    # to the conflict race: v=0 → η→∞ → no one yields to it, and it is 80 m from the conflict zone).
    USE_METER = True
    D_METER   = 80.0          # meter line: arc-length from box entry where over-served classes wait
    METER_TOL = 1.0           # release slack (vehicles) around fair share — damps hold/release chatter
    MAX_HOLD  = 8.0           # liveness cap: never hold one vehicle longer than this (no box idling)
    METER_WARM = 6            # don't meter until this many have arrived (fair-share estimate is noisy early)
    FAIR_FRAC = {d: VPH[d] / sum(VPH.values()) for d in VPH}   # demand-proportional fair share l/s/r
    PROTECT   = {"s"}         # NEVER meter these (the structurally-starved through class) — only
                              # over-served TURNS are held; a transient through burst must not self-starve
    # padded geometry [M,Pmax,*] for the batched rollout inside the gate
    Pmax = max(geo_g[i][0].shape[0] for i in range(M))
    PTS = torch.zeros(M, Pmax, 2); CUM = torch.zeros(M, Pmax)
    for i in range(M):
        pts, cum = geo_g[i]; P = pts.shape[0]
        PTS[i, :P] = pts; PTS[i, P:] = pts[-1]; CUM[i, :P] = cum; CUM[i, P:] = cum[-1]

    sumo = sumolib.checkBinary("sumo-gui" if gui else "sumo")
    cmd = [sumo, "-n", net, "-r", routes, "--begin", "0", "--end", str(T_END),
           "--step-length", str(DT), "--seed", str(seed), "--no-step-log", "true",
           "--no-warnings", "true", "--collision.action", "warn",
           "--collision.check-junctions", "true", "--time-to-teleport", "-1"]
    if gui:
        cmd += ["--start", "--quit-on-end", "--delay", "0"]
    traci.start(cmd)
    if gui:
        try:
            traci.gui.setSchema("View #0", "real world")
            traci.gui.setBoundary("View #0", 120, 120, 280, 280)
        except traci.TraCIException:
            pass
        print("car colors:  RED=yield  GREEN=pass  GREY=free  (yield/pass by conflict-gap "
              "ETA — later yields; turns treated like through; promotion clears the stuck)")
    n_steps = int(T_END / DT)
    configured, mv_idx, prev_a, stuck = set(), {}, {}, {}
    released, held_t = set(), {}               # metering: vehicles past the meter line / per-veh hold time
    STUCK_V, STUCK_HOLD = 3.0, 4.0           # a vehicle creeping <3 m/s near the box >4 s is "starved"
    N_PROMOTE = 5                            # vehicles per promoted compatible platoon (window = 5th's clearance)
    phase_mvs, phase_end = [], -1.0          # LATCHED promotion (only active during starvation)
    arrived, collided = [], set()
    from collections import defaultdict
    DIRNAME = {m.idx: m.dir for m in mv}
    dep_dir, arr_dir = defaultdict(int), defaultdict(int)   # served-per-movement (vs Krauss)
    tc_min, tc_max, tc_sum, tc_n = float("inf"), 0.0, 0.0, 0
    _DBG = {"departed": 0, "promoted_steps": 0, "inbox_max": 0, "N_max": 0}

    for step in range(n_steps):
        t_wall = time.perf_counter()
        ids = traci.vehicle.getIDList()
        for v in ids:
            if v not in configured:
                traci.vehicle.setSpeedMode(v, 0); traci.vehicle.setLaneChangeMode(v, 0)
                traci.vehicle.subscribe(v, (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_DISTANCE,
                                            tc.VAR_ROAD_ID))
                r = traci.vehicle.getRoute(v)
                mv_idx[v] = mv_of_route.get((r[0], r[-1]), 0)
                configured.add(v); dep_dir[DIRNAME[mv_idx[v]]] += 1
        sub = traci.vehicle.getAllSubscriptionResults()
        vehs = [v for v in ids if v in sub]
        N = len(vehs)
        if N == 0:
            traci.simulationStep(); arrived.append(traci.simulation.getArrivedNumber())
            if gui:
                time.sleep(max(0.0, DT - (time.perf_counter() - t_wall)))
            continue

        vs = torch.tensor([sub[v][tc.VAR_SPEED] for v in vehs])
        mvi = torch.tensor([mv_idx[v] for v in vehs])
        axis, sgn = AX[mvi], SG[mvi]
        is_turn = IS_TURN[mvi]
        s_front = torch.tensor([sub[v][tc.VAR_DISTANCE] for v in vehs]) + utils.L_VEH
        d_junc = s_junc_g[mvi] - s_front                          # arc-length to box entry (>0 approaching)

        # leader gap / speed (longitudinal) via SUMO
        gap = torch.full((N,), 300.0); v_lead = vs.clone()
        for i, v in enumerate(vehs):
            ld = traci.vehicle.getLeader(v, 100.0)
            if ld and ld[0] in mv_idx:
                gap[i] = max(ld[1], 0.0)
                if ld[0] in sub: v_lead[i] = sub[ld[0]][tc.VAR_SPEED]

        # ARC-LENGTH per-pair conflict geometry from the conflict-point arc-lengths s_cp
        # (correct for curved turn paths — not an entry-axis projection).  ego_d[a,b] =
        # CP(a,b) arc-len along a's path − a's arc-pos; valid iff their paths actually cross.
        scp_e = s_cp_g[mvi][:, mvi]                               # CP arc-len along EGO's path
        scp_r = s_cp_g.t()[mvi][:, mvi]                           # CP arc-len along RIVAL's path
        ego_d = scp_e - s_front.unsqueeze(1)
        rival_d = scp_r - s_front.unsqueeze(0)
        rival_v = vs.unsqueeze(0).expand(N, N)
        eye = torch.eye(N, dtype=torch.bool)
        valid = CONF[mvi][:, mvi] & (~eye) & (ego_d > 0.0) & (rival_d > -C.CLEAR)
        ego_d = torch.nan_to_num(ego_d, nan=1e3); rival_d = torch.nan_to_num(rival_d, nan=-1e3)

        # platoon pressure: same entry-approach vehicles still approaching
        same = (mvi.unsqueeze(0) // 3 == mvi.unsqueeze(1) // 3)   # same approach (3 mv/approach)
        appr = (d_junc > 0.0) & (d_junc < 120.0)
        same_ap = same & appr.unsqueeze(0)
        P = same_ap.float().sum(1)

        has_conf = valid.any(1)
        # CONFLICT GAP τ_c per ego — the arrival-timing the yield/pass decision uses
        tau_c_all, *_ = utils.conflict_time_gap(ego_d, vs, rival_d, rival_v, valid)
        if bool(has_conf.any()):
            tcv = tau_c_all[has_conf]
            tc_min = min(tc_min, float(tcv.min())); tc_max = max(tc_max, float(tcv.max()))
            tc_sum += float(tcv.sum()); tc_n += int(tcv.numel())

        # through-priority head-start (broadcast over the [N,N] pairs): subtract from each
        # side's effective ETA so a higher-priority movement "arrives earlier" for the
        # who-yields comparison (kernel uses the same bias via prio_ego/prio_rival below).
        # NOTE: LEFT METERING (force lone/under-queued lefts to yield until > a few peers queue
        # near the box, à la permissive-left batching) was tried TWO ways and REVERTED — neither
        # recovers Krauss's left/through balance, both vs the clean baseline ~1380 vph / 0-2 coll:
        #   • prio bias into the kernel → 1102 vph, 6 coll (a large prio poisons δ_safe/ETA, the
        #     same failure as the reverted through head-start);
        #   • gate-yield-mask only      → 1316 vph, 3 coll, served ratio basically unchanged.
        # The meter barely binds: at 300 vph/approach the dedicated left lane fills past any small
        # threshold almost immediately, so lefts flood through anyway; and even gate-only forced
        # yields still cost collisions (rear-end / deadlock-resolution).  The left over-serving is
        # STRUCTURAL — lefts ride a dedicated lane (fromLane 1) and reach the box unobstructed, so
        # they win the FCFS-by-ETA race over throughs that share a lane with rights.  Fixing it
        # needs UPSTREAM control (meter left-lane INFLOW / slot reservation), not box-level yield
        # forcing — the same open problem as the promotion note below.
        near = (d_junc > -5.0) & (d_junc < 70.0)               # queued/approaching near the box
        prio = PRIO[mvi]                                       # [N]
        eta_e = ego_d / vs.clamp(min=utils.EPS).unsqueeze(1)
        eta_k = rival_d / rival_v.clamp(min=utils.EPS)
        yield_eta = valid & (eta_k - prio.unsqueeze(0) < eta_e - prio.unsqueeze(1))

        # stuck timer: a vehicle creeping near the box accrues starvation time
        t = step * DT
        for i, v in enumerate(vehs):
            stuck[v] = (stuck.get(v, 0.0) + DT) if (bool(near[i]) and float(vs[i]) < STUCK_V) else 0.0
        stuck_t = torch.tensor([stuck[v] for v in vehs])

        # ── PROMOTION (anti-starvation): latch a compatible group when someone is starved.
        # Window = car-following ETA of the LAST (N_PROMOTE-th) promoted vehicle through its
        # final conflict point.
        if t >= phase_end and bool((stuck_t > STUCK_HOLD).any()):
            seed = int(mvi[int(stuck_t.argmax())])
            phase = [seed]
            for m in range(M):
                if m != seed and all(bool(COMPAT[m, p]) for p in phase):
                    phase.append(m)
            phase_mvs = phase
            sidx = sorted([i for i in range(N) if int(mvi[i]) == seed and bool(near[i])],
                          key=lambda i: float(s_front[i]), reverse=True)
            if sidx:
                iL = sidx[:N_PROMOTE][-1]; v0 = float(vs[iL])
                d = max(float(final_cp[seed]) - float(s_front[iL]), 0.0)
                ta = max((V_PHYS - v0) / utils.A_MAX, 0.0)
                da = v0 * ta + 0.5 * utils.A_MAX * ta ** 2
                T_clear = ((-v0 + (v0 ** 2 + 2 * utils.A_MAX * d) ** 0.5) / utils.A_MAX
                           if d <= da else ta + (d - da) / V_PHYS)
            else:
                T_clear = 3.0
            phase_end = t + max(T_clear, 2.0)

        # per-vehicle PROMOTION FLAG p∈{0,1} for the KERNEL: 1 for the front N_PROMOTE of the
        # latched compatible group near the box.  The kernel's p=1 anchors make them PASS
        # (assert/free, g<1 brake kept); p=0 vehicles negotiate normally and yield to the
        # now-asserting promoted ones by conflict-gap ETA.  No external gate.
        promote = torch.zeros(N)
        promoted = torch.zeros(N, dtype=torch.bool)
        if t < phase_end and phase_mvs:
            in_phase = torch.tensor([int(mvi[i]) in phase_mvs for i in range(N)])
            cand = sorted((in_phase & near).nonzero().flatten().tolist(),
                          key=lambda i: float(s_front[i]), reverse=True)
            for i in cand[:N_PROMOTE]:
                promote[i] = 1.0; promoted[i] = True
        # NOTE: a "promotion grants right-of-way" patch (force cross traffic to yield to the
        # promoted group so its queue drains) was tried and REVERTED — even guarded to only
        # vehicles that can still stop, it reintroduced collisions (seeds 1,2) and did not raise
        # throughput.  Forcing extra yields destabilizes the FCFS safety (a known dead-end here).
        # Through-movement starvation under FCFS remains an open fairness problem; it needs a
        # mechanism that does NOT force yields (e.g. upstream metering / slot reservation).
        is_yield = yield_eta.any(1) & (~promoted)              # for GUI coloring

        mean_fn = None
        if mean_model is not None:
            import mean_net
            roles_i = ["yield" if bool(is_yield[i]) else "none" for i in range(N)]
            behind_n = (same_ap & (d_junc.unsqueeze(0) > d_junc.unsqueeze(1))).float().sum(1)
            ctx = mean_net.build_context(vs, gap, v_lead, P, behind_n, d_junc,
                                         ego_d, rival_d, valid, roles_i, axis, sgn)
            mean_fn = mean_model.make_mean_fn(mean_model.encode(*ctx))

        a_prev_t = torch.tensor([prev_a.get(v, 0.0) for v in vehs])
        # the KERNEL resolves everything: p=0 → ETA cross negotiation + δ_safe brake + g<1
        # floor; p=1 → PASS (the promotion state in the 4-D kernel).  No external gate.
        a = utils.controller_acceleration(
            torch.zeros(N), gap + utils.L_VEH, vs, v_lead,
            d_conf=ego_d, rival_d=rival_d, rival_v=rival_v, rival_valid=valid,
            ego_pressure=P, rival_pressure=P.unsqueeze(0).expand(N, N),
            a_prev=a_prev_t, kappa=0.5, brake_exempt=True,
            brake_floor=True, predecessor=False, promote=promote, mean_fn=mean_fn,
            prio_ego=prio.unsqueeze(1), prio_rival=prio.unsqueeze(0)).detach()

        # ── 2-D rollout gate (now turn-general): the final safety correction the raw
        # kernel lacks.  Box-exclusivity and the committed-crosser test key on the TRUE
        # per-pair crossing (conf=CONF) over all 12 movements, and the rollout sweeps each
        # vehicle's REAL curved path (order=GEO_ORDER).  Promoted vehicles are force-rolled
        # as leaders (they assert), and their yield rows are cleared so the gate treats them
        # as passers — consistent with the kernel's promotion.
        if use_gate:
            gate_ym = yield_eta & (~promoted).unsqueeze(1)        # promoted never yield
            a, _gate_defer = S.rollout_gate(
                a, s_front, vs, mvi, gate_ym, geo_g, s_junc_g,
                return_defer=True, force_roll=promoted, conf=CONF, geo_order=GEO_ORDER,
                box_exclusive=box_exclusive)
            a = a.detach()

        # ── UPSTREAM METER decision (per vehicle).  A vehicle is HELD this step iff it has not yet
        # passed the meter line, has reached it (d_junc ≤ D_METER), its movement-CLASS is OVER its
        # demand-fair share of throughput so far, and it hasn't been held past MAX_HOLD.  Held =
        # brake to a stop at the line (followers queue behind via car-following); otherwise it is
        # RELEASED for good and the kernel/gate drive it normally.  Class fair-share is closed-loop
        # on actual arrivals, so over-served classes wait until the starved class catches up.
        held = torch.zeros(N, dtype=torch.bool)
        if USE_METER:
            total_arr = sum(arr_dir.values())
            # CONTENTION GATE: only meter when an UNDER-served class actually has a vehicle
            # approaching to claim the freed slot.  Without this we hold over-served vehicles into
            # dead air and idle box capacity (the throughput loss).  under_waiting = ∃ approaching
            # vehicle whose class is below its fair share.
            under_cls = {d for d in VPH if arr_dir[d] - FAIR_FRAC[d] * total_arr < -METER_TOL}
            under_waiting = any(DIRNAME[int(mvi[j])] in under_cls and 0.0 < float(d_junc[j]) < 120.0
                                for j in range(N))
            for i, v in enumerate(vehs):
                if v in released:
                    continue
                if float(d_junc[i]) > D_METER:               # not yet at the line — free approach
                    continue
                d = DIRNAME[int(mvi[i])]
                if d in PROTECT:                             # never hold the starved class
                    released.add(v); continue
                over = arr_dir[d] - FAIR_FRAC[d] * total_arr  # >0 ⇒ this class has had more than its share
                if (total_arr >= METER_WARM and over > METER_TOL and under_waiting
                        and held_t.get(v, 0.0) < MAX_HOLD):
                    held[i] = True
                    held_t[v] = held_t.get(v, 0.0) + DT
                else:
                    released.add(v)                          # let it through; never re-trap it

        for i, v in enumerate(vehs):
            prev_a[v] = float(a[i])
            if bool(held[i]):                                # brake to a stop at the meter line
                traci.vehicle.setSpeed(v, float(max(float(vs[i]) - utils.B_MAX * DT, 0.0)))
                if gui:
                    traci.vehicle.setColor(v, (40, 120, 230, 255))   # BLUE = metered (held upstream)
                continue
            traci.vehicle.setSpeed(v, float(min(max(float(vs[i]) + float(a[i]) * DT, 0.0), V_PHYS)))
            if gui:
                # COLOR by what the vehicle is ACTUALLY doing, using PHYSICAL junction
                # occupancy (SUMO ':' internal-lane road id) as ground truth — a vehicle on
                # the junction and moving is CROSSING, so it is a passer (green) whatever its
                # nominal role; the role label evaporates once it's past its conflict points.
                # A vehicle stopped ON the junction shows its role (red if yielding) so a real
                # block is visible; approaching vehicles colour by role as before.
                on_jct = str(sub[v][tc.VAR_ROAD_ID]).startswith(":")
                going = (bool(promoted[i]) or float(a[i]) > 0.0
                         or (on_jct and float(vs[i]) > 0.3)
                         or (float(d_junc[i]) < 0.0 and float(vs[i]) > 0.5))
                role = "pass" if going else ("yield" if bool(is_yield[i]) else "none")
                traci.vehicle.setColor(v, ROLE_COLOR[role])

        _DBG["promoted_steps"] += int(promoted.sum())
        _DBG["inbox_max"] = max(_DBG["inbox_max"], int((d_junc < 0).sum()))
        _DBG["N_max"] = max(_DBG["N_max"], N)
        traci.simulationStep()
        _DBG["departed"] += traci.simulation.getDepartedNumber()
        arrived.append(traci.simulation.getArrivedNumber())
        for v in traci.simulation.getArrivedIDList():
            if v in mv_idx: arr_dir[DIRNAME[mv_idx[v]]] += 1
        collided.update(traci.simulation.getCollidingVehiclesIDList())
        if gui:
            time.sleep(max(0.0, DT - (time.perf_counter() - t_wall)))

    i40 = int(40 / DT)
    ss_rate = float(np.sum(arrived[i40:]) / ((n_steps - i40) * DT) * 3600)
    residual = _DBG['departed'] - int(np.sum(arrived))           # still stuck in the network
    print(f"  [dbg] departed={_DBG['departed']}  arrived={int(np.sum(arrived))}  "
          f"STILL_STUCK={residual}  in_network_peak={_DBG['N_max']}  "
          f"in_box_peak={_DBG['inbox_max']}  promoted_veh-steps={_DBG['promoted_steps']}")
    traci.close()
    served = {d: (arr_dir[d], dep_dir[d]) for d in ("s", "l", "r")}
    print("  served by movement:  " +
          "  ".join(f"{d}:{served[d][0]}/{served[d][1]}" for d in ("s", "l", "r")))
    try:
        traci.close()
    except traci.exceptions.FatalTraCIError:
        pass   # SUMO may already have exited at --end under heavy load; harmless at teardown
    return dict(vph=ss_rate, arrived=int(np.sum(arrived)), collided=len(collided),
                served=served,
                tau_c_min=(tc_min if tc_n else float("nan")),
                tau_c_mean=(tc_sum / tc_n if tc_n else float("nan")),
                tau_c_max=(tc_max if tc_n else float("nan")))


if __name__ == "__main__":
    tokens = sys.argv[1:]
    gui  = "gui" in tokens
    use_gate = "nogate" not in tokens
    box_exclusive = "nobox" not in tokens
    seed = next((int(t) for t in tokens if t.isdigit()), 0)
    model = None if "nonn" in tokens else load_model()
    r = run(seed, mean_model=model, gui=gui, use_gate=use_gate, box_exclusive=box_exclusive)
    print(f"\nTURNS  L/S/R={VPH['l']}/{VPH['s']}/{VPH['r']} vph/approach  seed={seed}"
          f"  ({'nonn' if model is None else 'trained'}):")
    print(f"  throughput       : {r['vph']:.0f} veh/h  (arrived {r['arrived']})")
    print(f"  conflict gap τ_c : min {r['tau_c_min']:.2f} / mean {r['tau_c_mean']:.2f} / "
          f"max {r['tau_c_max']:.2f} s")
    print(f"  collisions       : {r['collided']}  "
          f"({'turn-general 2-D rollout gate ON' if use_gate else 'NO gate (raw kernel)'})")
