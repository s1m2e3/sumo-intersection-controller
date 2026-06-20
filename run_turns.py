"""Unified straight + turn harness: run OUR controller in SUMO over all 12 movements
(LEFT/THROUGH/RIGHT per approach) and measure throughput, conflict gap, and collisions.
The straight-cross case is just the l=0, r=0 subset of this same code — the demand config
(per-direction vph) is the ONLY knob that switches scenarios; geometry, kernel, role gate
and FGD proxy are all movement-general.

Stack (all turn-general):
  • all 12 movements (entry→exit) with TRUE per-pair conflict points (turns_geom);
  • conflict = "paths cross" (CP exists) — catches left-turn vs oncoming-through, etc.;
  • roles from utils.predecessor_gap (timing) PLUS the rule: a TURNING movement yields
    to through traffic unless it has space (a clear predecessor gap) or is promoted;
  • 2-D rollout gate (box-exclusive, true per-pair crossing) — the discrete safety gate;
  • FGD PROXY hinge (use_proxy, default on) AFTER the gate, at a stricter proxy τ_safety,
    sweeping each vehicle's real curved path — the same proxy as the straight cosim path.

    conda run -n car-following-sumo python run_turns.py [seed] [l=..] [s=..] [r=..] \\
        [ds=<kernel τ>] [proxy=<proxy τ>] [nonn] [nogate] [noproxy] [gui]
    e.g.  ... run_turns.py 0 l=100 s=300 r=100 ds=5 proxy=5     # 300 thru / 100 L / 100 R
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


def write_turn_routes(path, vph=None, approaches=None):
    """Emit one SUMO <flow> per movement at the per-direction demand `vph` (defaults to
    the module VPH).  A direction with demand 0 emits no vehicles, so the SAME harness
    collapses to the pure straight-cross case at vph={'l':0,'s':N,'r':0} and to a full
    turning intersection at any nonzero l/r.  `approaches` (set of entry-edge ids, e.g.
    {'east_in','west_in'}) restricts flow to those approaches only — movements from any
    other entry get zero, so we can drive e.g. the EW/WE legs alone with NS/SN empty."""
    vph = VPH if vph is None else vph
    mv = G.movements(os.path.join(C.HERE, "intersection.net.xml"))
    with open(path, "w") as f:
        f.write('<routes>\n'
                '  <vType id="car" accel="2.6" decel="4.5" sigma="0" length="5" '
                'minGap="2.0" maxSpeed="13.89" carFollowModel="Krauss"/>\n')
        for m in mv:
            if approaches is not None and m.frm not in approaches:
                continue                       # approach excluded → no flow
            rate = vph[m.dir] / 3600.0
            if rate <= 0.0:
                continue                       # no demand on this direction → no flow
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


def run(seed=0, mean_model=None, gui=False, use_gate=True, box_exclusive=True,
        use_proxy=True, kernel_delta_safe=None, proxy_delta_safe=5.0, vph=None,
        dump_t=None, approaches=None):
    """Unified straight+turn harness.  `vph` (per-direction demand dict) is the ONLY knob
    that selects the scenario — the geometry, kernel, gate and proxy are movement-general.

      kernel_delta_safe : kernel τ_safety (s).  When given, set globally BEFORE any
                          controller call so the anchor grid / GP inverse rebuild cleanly.
      proxy_delta_safe  : the FGD proxy's stricter τ_safety (s), applied AFTER the role
                          gate (use_proxy).  Decoupled from the kernel δ_safe, exactly like
                          the straight cosim path (kernel targets δ_safe; proxy enforces ≥).
    """
    DUMP_T = dump_t                              # sim time (s) for the one-shot state snapshot
    global VPH
    if vph is not None:
        VPH = dict(vph)                          # demand config drives the whole run
    if kernel_delta_safe is not None:
        utils.set_delta_safe(float(kernel_delta_safe))
    net = os.path.join(C.HERE, "intersection.net.xml")
    routes = os.path.join(C.HERE, "_turn_routes.rou.xml")
    write_turn_routes(routes, VPH, approaches=approaches)
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
    USE_METER = False         # DISABLED: the upstream meter holds over-served classes to a full STOP
                              # D_METER m UPSTREAM of the box — a mid-road stop, which violates the rule
                              # that only a vehicle right before the junction may be fully stopped.  The
                              # only legitimate full stop is the anti-promoted hold AT the stop-line.
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
    N_PROMOTE = 8                            # phase-window length = car-following clearance time of the
                                             # N_PROMOTE-th front seed-lane vehicle (longer window ⇒ more
                                             # of the clique clears per phase; not a cap on who is promoted)
    phase_mvs, phase_end = [], -1.0          # LATCHED promotion (only active during starvation)
    prom_yield = {}                          # veh_id → set of veh_ids it is LATCHED to yield to
                                             # (promotion pass/yield assignment; released per final_cp)
    arrived, collided = [], set()
    from collections import defaultdict
    DIRNAME = {m.idx: m.dir for m in mv}
    dep_dir, arr_dir = defaultdict(int), defaultdict(int)   # served-per-movement (vs Krauss)
    tc_min, tc_max, tc_sum, tc_n = float("inf"), 0.0, 0.0, 0
    _DBG = {"departed": 0, "promoted_steps": 0, "inbox_max": 0, "N_max": 0}
    _DBG["promo_mv"] = defaultdict(int)     # promoted (p=+1) veh-steps per movement idx
    _DBG["anti_mv"]  = defaultdict(int)     # anti-promoted (p=-1) veh-steps per movement idx
    _DBG["seed_mv"]  = defaultdict(int)     # times each movement was the phase SEED (latch)

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
        leader_id = [None] * N                            # physical leader veh-id on the ego's route
        for i, v in enumerate(vehs):
            ld = traci.vehicle.getLeader(v, 100.0)
            if ld and ld[0] in mv_idx:
                gap[i] = max(ld[1], 0.0)
                leader_id[i] = ld[0]
                if ld[0] in sub: v_lead[i] = sub[ld[0]][tc.VAR_SPEED]

        # SUPPLEMENT the route-based getLeader with a same-approach, same-LANE in-lane leader.
        # getLeader follows the EGO's downstream route, so it LOSES a leader that diverges onto a
        # different internal lane at the junction mouth — e.g. a THROUGH stopped at the line is no
        # longer "ahead" for a RIGHT that shares its approach lane (their paths split in the box).
        # The kernel then sees a huge gap and rear-ends it.  Recover it by arc-length: through+right
        # share a lane (idx%3∈{0,1}); left rides its own (idx%3==2).  A same-approach, same-lane-group
        # vehicle that is AHEAD (larger arc-pos) and closer than the route leader sets the gap.
        appr = (mvi // 3)
        left_lane = (mvi % 3 == 2)                                # left has a dedicated lane
        for i in range(N):
            if float(d_junc[i]) <= 0.0:                           # only APPROACHING egos (in-box clears)
                continue
            for j in range(N):
                if i == j or int(appr[j]) != int(appr[i]) or bool(left_lane[i]) != bool(left_lane[j]):
                    continue                                      # not a same-lane neighbour
                if float(d_junc[j]) <= -2.0:                      # leader deep in box on its own diverged
                    continue                                      # internal lane → arc-length gap invalid
                dg = float(s_front[j]) - float(s_front[i]) - utils.L_VEH   # bumper gap (j ahead ⇒ >0)
                if 0.0 <= dg < float(gap[i]):
                    gap[i] = dg; v_lead[i] = float(vs[j])

        # ── gap-belief trace (rear-end diagnostic): for a TRACE_MV near the box, print the vehicle's
        # computed gap, leader speed, gap-ratio g and the raw getLeader — to see whether its OWN belief
        # of the gap is wrong (huge while a leader is close) or correct.  OFF unless enabled via the CLI
        # 'gaptrace=<mv>' flag (optionally 'gtwin=<t0>,<t1>' for the time window; default 74–83 s).
        TRACE_MV = globals().get("_TRACE_MV", -1)
        TRACE_T0, TRACE_T1 = globals().get("_TRACE_T0", 74.0), globals().get("_TRACE_T1", 83.0)
        _tt = step * DT
        if TRACE_MV >= 0 and TRACE_T0 <= _tt <= TRACE_T1:
            for i in range(N):
                if int(mvi[i]) == TRACE_MV and -52.0 < float(d_junc[i]) < 25.0:
                    _ld = traci.vehicle.getLeader(vehs[i], 100.0)
                    _g = float(utils.gap_ratio(torch.zeros(1),
                                               torch.tensor([float(gap[i]) + utils.L_VEH]),
                                               torch.tensor([float(vs[i])]),
                                               torch.tensor([float(v_lead[i])]), utils.L_VEH)[0])
                    print(f"  [gaptrace t={_tt:5.1f}] {vehs[i]:>6} d_junc={float(d_junc[i]):7.1f} "
                          f"v={float(vs[i]):5.2f} gap={float(gap[i]):6.1f} vlead={float(v_lead[i]):5.2f} "
                          f"g={_g:5.2f}  getLeader={_ld}")

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
            _DBG["seed_mv"][seed] += 1
            # Build the compatible-movement clique to PROMOTE.  Movements are grouped 3/approach
            # (idx//3 = approach E,W,N,S; idx%3 = dir r,s,l), and opposing approaches share an
            # axis (appr ^ 1).  The greedy clique is order-sensitive, so we add the OPPOSING
            # same-direction movement FIRST — for a left seed that is the opposing left, the classic
            # PROTECTED-LEFT pair (always mutually compatible, they don't cross).  Adding it before
            # the index-order turns stops a compatible-with-seed-but-conflicts-with-partner movement
            # (e.g. north_in.r, which crosses south_in.l) from greedily locking the partner out and
            # leaving it stranded on the promoted queue's path.  Same-dir movements next, then the rest.
            opp = ((seed // 3) ^ 1) * 3 + (seed % 3)     # opposite-approach, same-direction movement
            phase = [seed]
            for m in sorted((x for x in range(M) if x != seed),
                            key=lambda x: (x != opp, x % 3 != seed % 3, x)):
                if all(bool(COMPAT[m, p]) for p in phase):
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

        # physical junction occupancy (SUMO ':' internal-lane road id) — TRUE ground truth for
        # "this vehicle is inside the box", independent of the arc-length d_junc metric.
        on_box = [str(sub[v][tc.VAR_ROAD_ID]).startswith(":") for v in vehs]

        # ── PROMOTION SET (γ-among-promoted pipeline).  Build the FULL promoted set FIRST, then let
        # γ decide pass/yield within it (rival masking at the kernel call below):
        #   (1) QUEUE  — every vehicle on a compatible movement in the latched phase clique.
        #   (2) MUST-CLEAR — every vehicle physically in the box (':' lane), ANY movement, so it
        #       asserts/serializes OUT instead of freezing mid-junction.
        # Both get promoted=True (kernel p=0 → γ is EVALUATED, not dropped).  γ is then computed only
        # among this set: each promoted vehicle yields iff another PROMOTED vehicle reaches the shared
        # conflict point earlier (ETA) — so the must-clear intruder (earliest) passes and a conflicting
        # phase vehicle yields to it.  Compatible promoted pairs share no conflict point ⇒ both pass.
        # Every INCOMPATIBLE APPROACHER (phase active, not promoted) is ANTI-PROMOTED (p=-1): free-flow
        # to its stop-line, held there, and INVISIBLE to the promoted γ.  No phase ⇒ all p=0 (original).
        promote = torch.zeros(N)
        promoted = torch.zeros(N, dtype=torch.bool)
        if t < phase_end and phase_mvs:
            for i in range(N):
                mv_i = int(mvi[i])
                if (mv_i in phase_mvs) or on_box[i]:              # (1) queue  ∪  (2) must-clear (in-box)
                    promoted[i] = True
                    promote[i] = 1.0                             # p=+1 PROMOTED: pass/yield decided AMONG
                    _DBG["promo_mv"][mv_i] += 1                  #   the promoted set (latched, below)
                else:
                    promote[i] = -1.0                            # p=-1 incompatible → free-flow to line, held
                    _DBG["anti_mv"][mv_i] += 1
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
        # γ-AMONG-PROMOTED: a promoted ego negotiates yield/pass ONLY against OTHER promoted
        # vehicles — the held incompatible traffic is masked out (invisible).  So the kernel's
        # normal ETA cross-resolution, restricted to this subset, decides who passes and who yields
        # WITHIN the promoted group: the must-clear intruder (earliest) passes; the conflicting
        # phase vehicle yields to it.  Non-promoted egos keep the full rival set (unchanged).
        valid_k = valid.clone()
        if bool(promoted.any()):
            valid_k[promoted] = valid[promoted] & promoted.unsqueeze(0)

        # ── PROMOTION ROLE LATCH (pass/yield AMONG the promoted set).  Each promoted vehicle gets a
        # strict PRIORITY RANK: in-box must-clear vehicles outrank the whole approaching queue (huge
        # head-start), ties broken by ETA to the box.  Ego i is latched to YIELD to a CONFLICTING
        # promoted rival j iff j OUTRANKS i — a strict total order, so of any conflicting pair exactly
        # one yields (no mutual-pass crash, no mutual-yield deadlock).  In-box vs in-box therefore
        # serialises by ETA instead of both asserting.  The base is recomputed each step (so a car that
        # enters the box LATER is picked up), and a yield is LATCHED — held until that rival clears its
        # LAST conflict point (s_front > final_cp), then released so the queue serialises through behind
        # it.  earlier_ov[i,j]=True ⇒ i yields to j; fed to the kernel as the who-yields override.
        prom_ids = {vehs[i] for i in range(N) if bool(promoted[i])}
        for vid in list(prom_yield):                              # forget vehicles no longer promoted
            if vid not in prom_ids:
                del prom_yield[vid]
        idx_of = {vehs[i]: i for i in range(N)}
        on_box_t = torch.tensor(on_box)
        eta_box  = d_junc / vs.clamp(min=utils.EPS)               # ETA to the stop-line (in-box ≤ 0)
        rank     = torch.where(on_box_t, eta_box - 1e6, eta_box)  # in-box vehicles get right-of-way
        earlier_ov = torch.zeros(N, N, dtype=torch.bool)
        for i in sorted(range(N), key=lambda k: float(rank[k])):  # highest priority first
            if not bool(promoted[i]):
                continue
            latched = prom_yield.setdefault(vehs[i], set())
            for rv in list(latched):                              # RELEASE: rival cleared its last CP,
                j = idx_of.get(rv)                                # left, OR is now BEHIND us (we lead it —
                if (j is None or float(s_front[j]) > float(final_cp[int(mvi[j])])
                        or leader_id[j] == vehs[i]):              # a stale merge yield would deadlock).
                    latched.discard(rv)
            for j in range(N):                                    # ADD current higher-rank conflictors
                if j == i or not bool(promoted[j]) or not bool(valid_k[i, j]):
                    continue
                # never cross-yield to a vehicle we physically LEAD — that is a car-following
                # relationship (its g<1 brake handles it), and yielding to our own follower at a
                # merge is the f9.7↔f8.8 deadlock.
                if leader_id[j] == vehs[i]:
                    continue
                if float(rank[j]) < float(rank[i]) and vehs[i] not in prom_yield.get(vehs[j], ()):
                    latched.add(vehs[j])
            for rv in latched:
                j = idx_of.get(rv)
                if j is not None:
                    earlier_ov[i, j] = True

        # the KERNEL resolves everything: p=+1 → pass/yield from the LATCHED who-yields override
        # (rivals masked to the promoted subset) + δ_safe brake + g<1 floor; p=-1 → free-flow to
        # the line; p=0 → live ETA cross negotiation against all rivals.
        c_out = utils.controller_acceleration(
            torch.zeros(N), gap + utils.L_VEH, vs, v_lead,
            d_conf=ego_d, rival_d=rival_d, rival_v=rival_v, rival_valid=valid_k,
            ego_pressure=P, rival_pressure=P.unsqueeze(0).expand(N, N),
            a_prev=a_prev_t, kappa=0.5, brake_exempt=True,
            brake_floor=True, predecessor=False, promote=promote, mean_fn=mean_fn,
            prio_ego=prio.unsqueeze(1), prio_rival=prio.unsqueeze(0),
            cross_override=(earlier_ov, promoted),
            return_feat=use_proxy)
        # feat = [N,4] live query features (g, τ_c, r, p) — the proxy builds the
        # controller's own ARD Gram K^φ over them to smooth its hinge step.
        feat = None
        if use_proxy:
            a, feat = c_out
            a = a.detach()
        else:
            a = c_out.detach()
        _a_kernel = a.clone()                  # diag: command straight out of the 4-D kernel
        _a_gate = a.clone()                    # diag: post rollout-gate (set below if use_gate)
        _defer = torch.zeros(N, dtype=torch.bool)

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
            _a_gate = a.clone()                # diag: post rollout-gate, pre FGD proxy
            if _gate_defer is not None:
                _defer = _gate_defer.bool() if torch.is_tensor(_gate_defer) else _defer
            # ── FGD PROXY (turn-general): L2 functional-gradient polish on the 2-D hinge,
            # AFTER the role gate (box-exclusivity intact).  Same machinery as the straight
            # cosim path — but the rollout sweeps each vehicle's REAL curved path
            # (geo_order=GEO_ORDER) and pairs by TRUE per-pair crossing (conf=CONF), so it
            # corrects left-turn-vs-through conflicts the discrete gate may underbrake.
            # delta_safe=proxy_delta_safe enforces a stricter gap than the kernel's δ_safe.
            if use_proxy:
                a = S.hinge_gradient_gate(
                    a, feat.detach(), s_front, vs, mvi, geo_g, s_junc_g,
                    delta_safe=proxy_delta_safe, conf=CONF, geo_order=GEO_ORDER).detach()

            # ── PROMOTED & IN-BOX use the KERNEL command (bypass the gate/proxy cross-corrections).
            # The kernel already resolved γ AMONG the promoted set (rivals masked to promoted above):
            # the must-clear/earliest vehicle passes, a conflicting promoted vehicle YIELDS (brakes to
            # its conflict-line) — so cross safety within the promoted group is in a_kernel.  The gate,
            # by contrast, is ETA-ordered over ALL vehicles and would make a promoted vehicle defer to
            # the held anti-promoted traffic it's invisible to — backwards, and the source of the
            # deadlock — so we discard it here.  a_kernel still carries the g<1 brake_floor (no in-lane
            # rear-end).  Held incompatible approachers stay out of the box via anti_cap / box_cap.
            for i in range(N):
                if bool(promoted[i]) or on_box[i]:
                    a[i] = _a_kernel[i]

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

        # ── ANTI-PROMOTED STOP-AT-LINE (mechanism 2): while a promotion phase owns the box,
        # every INCOMPATIBLE movement is ANTI-PROMOTED (p=-1) — the kernel already drives it as a
        # PASSER, so it is free to ACCELERATE up to the junction.  It must NOT enter the box, but
        # we never freeze it mid-road: instead we cap its speed by the LATEST-BRAKE stop-at-line
        # profile v_cap = √(2·B·d_entry).  Far from the line v_cap exceeds V_PHYS (full free-flow);
        # it tightens to 0 exactly at the stop line, so the vehicle runs at speed until the last
        # moment, then decelerates at ≤B_MAX and stops cleanly at the line, held there until the
        # phase ends.  Guarded by can_hold — a COMMITTED crosser (too close/fast to stop) keeps
        # v_cap=∞ and clears through rather than emergency-braking into the middle of the box.
        anti_cap = torch.full((N,), float("inf"))
        if t < phase_end and phase_mvs:
            for i in range(N):
                if bool(promoted[i]) or on_box[i]:
                    continue                                 # promoted (incl. in-box clearers) or
                                                             # already on the box → never cap/freeze
                d_entry = float(s_junc_g[mvi[i]] - s_front[i]) - utils.STOP_OFFSET
                d_need  = float(vs[i]) ** 2 / (2.0 * utils.B_MAX)
                can_hold = (float(vs[i]) < 0.5) or (d_entry > d_need + S.EPS_ENTRY)
                if float(d_junc[i]) > 0.0 and can_hold:       # approaching & able to stop
                    anti_cap[i] = (2.0 * utils.B_MAX * max(d_entry, 0.0)) ** 0.5

        # ── BOX ENTRY MUTUAL-EXCLUSION (asymmetric, physical occupancy + true per-pair conflict).
        # While a phase owns the box, the PROMOTED clique owns it: its (compatible) members flow and
        # co-occupy freely.  An INCOMPATIBLE (anti-promoted) approacher must NOT enter while a
        # conflicting movement physically occupies the box ('::' lane) — it is capped to STOP at its
        # stop-line, OVERRIDING anti_cap's can_hold escape (a "committed" crosser is forced to stop
        # AT the line rather than nose into the box, since a full stop right at the line is allowed).
        # This is the collision guard: a promoted N-S left and an incompatible E-W left can never both
        # be inside the box (the t≈37 s T-bone).  Promoted vehicles and anything already in/past the
        # box are exempt — they own the box / must clear, never freeze.  Keyed on PHYSICAL ':'
        # occupancy (not arc-length d_junc<0), so it catches what the gate's box-exclusivity misses.
        box_cap = torch.full((N,), float("inf"))
        if t < phase_end and phase_mvs:
            occ_mv = {int(mvi[i]) for i in range(N) if on_box[i]}
            if occ_mv:
                for i in range(N):
                    if on_box[i] or float(d_junc[i]) <= 0.0:
                        continue                                 # already in/past box → clear, not held
                    if any(bool(CONF[occ, int(mvi[i])]) for occ in occ_mv):
                        # a CONFLICTING movement physically occupies the box → hold at the line until
                        # it has fully EXITED (on_box false), not merely passed its conflict point.
                        # Applies to PROMOTED approachers too: the kernel's point-γ releases them as
                        # soon as the clearer passes the point, but its BODY is still in the box — so
                        # this body-aware hold is what prevents the tail T-bone.  Compatible promoted
                        # (no CONF with the occupant) are NOT capped → the clique co-occupies and flows.
                        d_entry = float(s_junc_g[mvi[i]] - s_front[i]) - utils.STOP_OFFSET
                        box_cap[i] = (2.0 * utils.B_MAX * max(d_entry, 0.0)) ** 0.5

        # ── TARGETED TRACE: follow specific vehicle ids over a time window — full state, the
        # raw getLeader, who it is latched to yield to (and that rival's state).  Diagnostic only.
        _TRACK = globals().get("_TRACK_IDS", [])
        if _TRACK and 80.0 <= t <= 100.0:
            for i in range(N):
                if vehs[i] not in _TRACK:
                    continue
                ld = traci.vehicle.getLeader(vehs[i], 120.0)
                yld = prom_yield.get(vehs[i], set())
                ystr = ",".join(f"{rv}@{float(s_front[idx_of[rv]]):.1f}/cp{float(final_cp[int(mvi[idx_of[rv]])]):.1f}"
                                for rv in yld if rv in idx_of) or "-"
                print(f"  [trk t={t:5.1f}] {vehs[i]:>6} {DIRNAME[int(mvi[i])]:>3} d_junc={float(d_junc[i]):6.1f} "
                      f"v={float(vs[i]):5.2f} a_ker={float(_a_kernel[i]):6.2f} a_fin={float(a[i]):6.2f} "
                      f"gap={float(gap[i]):6.1f} tau_c={float(tau_c_all[i]):5.2f} inbox={int(on_box[i])} "
                      f"acap={float(anti_cap[i]):6.1f} bcap={float(box_cap[i]):6.1f} "
                      f"getLeader={ld} yields_to=[{ystr}]")

        for i, v in enumerate(vehs):
            prev_a[v] = float(a[i])
            if bool(held[i]):                                # brake to a stop at the meter line
                traci.vehicle.setSpeed(v, float(max(float(vs[i]) - utils.B_MAX * DT, 0.0)))
                if gui:
                    traci.vehicle.setColor(v, (40, 120, 230, 255))   # BLUE = metered (held upstream)
                continue
            cap_i = min(float(anti_cap[i]), float(box_cap[i]))         # tightest stop-at-line cap
            v_cmd = min(max(float(vs[i]) + float(a[i]) * DT, 0.0), V_PHYS, cap_i)
            if cap_i < float(vs[i]) - 1e-3:                  # held at the line (anti-promoted or box-excluded)
                traci.vehicle.setSpeed(v, v_cmd)
                if gui:
                    traci.vehicle.setColor(v, (200, 120, 0, 255))    # ORANGE = anti-promoted stop-at-line
                continue
            traci.vehicle.setSpeed(v, v_cmd)
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

        # ── STATE SNAPSHOT (diagnostic): at sim time DUMP_T, print the ACTUAL per-vehicle
        # state for every near-box vehicle so a gridlock is fully legible — role label vs
        # command, speed, leader gap, conflict gap τ_c, who it must yield to, promotion and
        # metering.  Shows why a green (passing) car can sit at a≈0 (blocked / capped / held).
        if DUMP_T is not None and abs(t - DUMP_T) < DT / 2 and N:
            ny = yield_eta.sum(1)                                  # #rivals each ego must yield to
            print(f"\n=== STATE @ t={t:.1f}s  (N={N}) — near-box vehicles ===")
            print(f"{'veh':>7} {'dir':>3} {'role':>5} {'d_junc':>7} {'v':>5} "
                  f"{'a_ker':>6} {'a_gate':>6} {'a_fin':>6} {'cap':>6} {'gap':>6} {'tau_c':>6} "
                  f"{'nyld':>4} {'prom':>4} {'pyld':>4} {'dfr':>3} {'held':>4} {'inbox':>5}")
            order = sorted(range(N), key=lambda i: float(d_junc[i]))
            for i in order:
                if not (-10.0 < float(d_junc[i]) < 80.0):
                    continue
                going = (bool(promoted[i]) or float(a[i]) > 0.0
                         or (on_box[i] and float(vs[i]) > 0.3)
                         or (float(d_junc[i]) < 0.0 and float(vs[i]) > 0.5))
                role = "pass" if going else ("yield" if bool(is_yield[i]) else "free")
                cap = float(anti_cap[i]); cap_s = "  inf" if cap == float("inf") else f"{cap:>6.2f}"
                print(f"{vehs[i]:>7} {DIRNAME[int(mvi[i])]:>3} {role:>5} "
                      f"{float(d_junc[i]):>7.1f} {float(vs[i]):>5.2f} "
                      f"{float(_a_kernel[i]):>6.2f} {float(_a_gate[i]):>6.2f} {float(a[i]):>6.2f} "
                      f"{cap_s} {float(gap[i]):>6.1f} {float(tau_c_all[i]):>6.2f} {int(ny[i]):>4} "
                      f"{int(bool(promoted[i])):>4} {len(prom_yield.get(vehs[i], ())):>4} "
                      f"{int(bool(_defer[i])):>3} "
                      f"{int(bool(held[i])):>4} {int(on_box[i]):>5}")

        _DBG["promoted_steps"] += int(promoted.sum())
        _DBG["inbox_max"] = max(_DBG["inbox_max"], int((d_junc < 0).sum()))
        _DBG["N_max"] = max(_DBG["N_max"], N)
        traci.simulationStep()
        _DBG["departed"] += traci.simulation.getDepartedNumber()
        arrived.append(traci.simulation.getArrivedNumber())
        for v in traci.simulation.getArrivedIDList():
            if v in mv_idx: arr_dir[DIRNAME[mv_idx[v]]] += 1
        new_coll = set(traci.simulation.getCollidingVehiclesIDList()) - collided
        if new_coll:
            idxmap = {v: i for i, v in enumerate(vehs)}
            info = []
            for cv in sorted(new_coll):
                i = idxmap.get(cv)
                if i is None:
                    info.append(f"{cv}(gone)")
                else:
                    info.append(f"{cv}[{mv[int(mvi[i])].frm}.{DIRNAME[int(mvi[i])]} "
                                f"prom={int(bool(promoted[i]))} box={int(on_box[i])} "
                                f"v={float(vs[i]):.1f} gap={float(gap[i]):.1f} "
                                f"aker={float(_a_kernel[i]):.2f} afin={float(a[i]):.2f} "
                                f"d={float(d_junc[i]):.1f}]")
            print(f"  [COLLISION @ t={t:.1f}s]  " + "   ".join(info))
        collided.update(new_coll)
        if gui:
            time.sleep(max(0.0, DT - (time.perf_counter() - t_wall)))

    i40 = int(40 / DT)
    ss_rate = float(np.sum(arrived[i40:]) / ((n_steps - i40) * DT) * 3600)
    residual = _DBG['departed'] - int(np.sum(arrived))           # still stuck in the network
    print(f"  [dbg] departed={_DBG['departed']}  arrived={int(np.sum(arrived))}  "
          f"STILL_STUCK={residual}  in_network_peak={_DBG['N_max']}  "
          f"in_box_peak={_DBG['inbox_max']}  promoted_veh-steps={_DBG['promoted_steps']}")
    DIRNAME = {m.idx: m.dir for m in mv}
    seedmv = _DBG["seed_mv"]; promv = _DBG["promo_mv"]
    print(f"  [promo] phase SEEDs by movement: " +
          ("  ".join(f"{i}={mv[i].frm}.{DIRNAME[i]}:{seedmv[i]}" for i in sorted(seedmv))
           or "NONE"))
    print(f"  [promo] promoted veh-steps by movement: " +
          ("  ".join(f"{i}={mv[i].frm}.{DIRNAME[i]}:{promv[i]}" for i in sorted(promv))
           or "NONE"))
    n_turn_promo = sum(v for i, v in promv.items() if DIRNAME[i] != "s")
    print(f"  [promo] TURN promoted veh-steps={n_turn_promo} / total={sum(promv.values())}")
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
    use_proxy = "noproxy" not in tokens
    seed = next((int(t) for t in tokens if t.isdigit()), 0)
    # demand config: l=<vph> s=<vph> r=<vph> (any omitted → module default).  This is the
    # ONLY thing to change between the straight-cross case (l=0 r=0) and a turning one.
    def _tok(pfx, default):
        t = next((x for x in tokens if x.startswith(pfx)), None)
        return float(t.split("=", 1)[1]) if t is not None else default
    vph = {"l": _tok("l=", VPH["l"]), "s": _tok("s=", VPH["s"]), "r": _tok("r=", VPH["r"])}
    # kernel τ_safety (ds=/dsafe=) and proxy τ_safety (proxy=) — default kernel from utils,
    # proxy 5 s, matching the straight cosim convention.
    kds = next((t for t in tokens if t.startswith(("dsafe=", "ds="))), None)
    kernel_ds = float(kds.split("=", 1)[1]) if kds is not None else None
    proxy_ds = _tok("proxy=", 5.0)
    dump_t = next((float(t.split("=", 1)[1]) for t in tokens if t.startswith("dump=")), None)
    _trk = next((t for t in tokens if t.startswith("track=")), None)
    if _trk is not None:
        globals()["_TRACK_IDS"] = _trk.split("=", 1)[1].split(",")
    # gap-belief trace: gaptrace=<movement-idx>  (optional window  gtwin=<t0>,<t1>)
    _gt = next((t for t in tokens if t.startswith("gaptrace=")), None)
    if _gt is not None:
        globals()["_TRACE_MV"] = int(_gt.split("=", 1)[1])
        _gw = next((t for t in tokens if t.startswith("gtwin=")), None)
        if _gw is not None:
            _t0, _t1 = _gw.split("=", 1)[1].split(",")
            globals()["_TRACE_T0"], globals()["_TRACE_T1"] = float(_t0), float(_t1)
    # restrict to specific entry approaches: appr=east_in,west_in  (default: all four)
    at = next((t for t in tokens if t.startswith("appr=")), None)
    approaches = set(at.split("=", 1)[1].split(",")) if at is not None else None
    model = None if "nonn" in tokens else load_model()
    r = run(seed, mean_model=model, gui=gui, use_gate=use_gate, box_exclusive=box_exclusive,
            use_proxy=use_proxy, kernel_delta_safe=kernel_ds, proxy_delta_safe=proxy_ds, vph=vph,
            dump_t=dump_t, approaches=approaches)
    print(f"\nTURNS  L/S/R={VPH['l']:.0f}/{VPH['s']:.0f}/{VPH['r']:.0f} vph/approach  seed={seed}"
          f"  [kernel δ_safe={utils.DELTA_SAFE:.1f}s, proxy τ={proxy_ds:.1f}s, "
          f"{'proxy ON' if use_proxy else 'proxy OFF'}]"
          f"  ({'nonn' if model is None else 'trained'}):")
    print(f"  throughput       : {r['vph']:.0f} veh/h  (arrived {r['arrived']})")
    print(f"  conflict gap τ_c : min {r['tau_c_min']:.2f} / mean {r['tau_c_mean']:.2f} / "
          f"max {r['tau_c_max']:.2f} s")
    print(f"  collisions       : {r['collided']}  "
          f"({'turn-general 2-D rollout gate ON' if use_gate else 'NO gate (raw kernel)'})")
