"""
cosim_sumo.py — co-simulate OUR kernel controller inside SUMO via TraCI.

SUMO owns the world (geometry, integration, collision detection); at every step we
read each vehicle's state, compute OUR controller's acceleration, and impose it with
setSpeed (SpeedMode=0 → SUMO applies no safety/right-of-way of its own, so any unsafe
command WILL produce a real SUMO collision).  This validates collision-freeness in an
independent engine and measures throughput on the exact same network as Krauss.

Per-pair conflict points are the TRUE geometric crossings of the two movements' paths,
extracted once from the junction's internal-lane shapes in the SUMO net (not the centre
approximation).  Each side's distance to that fixed point is computed from live
positions along its travel axis.

Performance: one TraCI subscription round-trip per step (positions+speeds), a single
batched controller call for the whole population, and a cached anchor K⁻¹.

    conda run -n car-following-sumo python cosim_sumo.py [flow] [seed]
"""
import os, sys, math, time
import numpy as np
import torch
import traci, sumolib
import traci.constants as tc

import utils

HERE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_files")
DT     = 0.1
T_END  = 120.0
V_PHYS = utils.V0        # match the controller's free speed (binding physical cap)
CLEAR  = utils.L_VEH + 1.0           # rival "still in box" clearance past its crossing

# ── priority-memory parameters ──────────────────────────────────────────────────
ASSIGN_DIST = 100.0  # m   a vehicle is ASSIGNED a (yield/pass) role within this of the box —
                     #     far enough out to commit to a role early and react before the box
ROLE_HOLD   = 0.2    # s   a once-assigned role is LATCHED (not re-decided) for this long
STUCK_V     = 3.0    # m/s "low speed" threshold for accumulating in-junction stuck time
STUCK_HOLD  = 3.0    # s   a yielder stuck (in-junction, < STUCK_V) longer than this → PASS
V_MIN_PASS  = 0.5    # m/s a PASSER is kept moving (non-zero velocity) — it never freezes

# ── queue arbiter parameters ────────────────────────────────────────────────────
QUEUE_MIN_BEHIND = 3     #     ≥ this many vehicles tightly behind ⇒ a real standing queue
QUEUE_SPAN       = 40.0  # m   window behind the front within which queued cars are counted
QUEUE_WAIT       = 5.0   # s   a queue waiting longer than this earns a clearing window
QUEUE_STARVE     = 10.0  # s   waited this long ⇒ open EVEN IF the cross box is occupied —
                         #     the gate's box exclusivity still sequences physical entry,
                         #     so this only shifts right-of-way, never safety (anti-starvation)
QUEUE_CLEAR      = 5.0   # s   protected window: the promoted passers' role can't be re-estimated
QUEUE_HEADWAY    = 1.5   # s   extra protection per queue position (k-th promoted gets
                         #     QUEUE_CLEAR + k·this), so the whole block has time to cross
QUEUE_N_PASS     = 4     #     how many front-of-queue vehicles are promoted to PASS together
QUEUE_N_MAX      = 8     #     ceiling when the promotion count scales with a LONG queue
ARBITER_GAP      = 2.0   # s   two passers whose bumper-aware occupancy windows come within this
                         #     of each other are INCOMPATIBLE → the later one is demoted to yielder

# GUI role colors (RGBA): visualise each car's resolved priority role in sumo-gui
ROLE_COLOR = {"yield": (220, 40, 40, 255),    # red   — yielding
              "pass":  (40, 200, 40, 255),    # green — passing
              "none":  (170, 170, 170, 255)}  # grey  — unassigned (free-flow approach)

# origin edge -> (axis 0=x/1=y, travel sign along that axis)
ORIGIN = {"east_in": (0, -1.0), "west_in": (0, +1.0),
          "north_in": (1, -1.0), "south_in": (1, +1.0)}


def write_routes(flow, path, depart_speed=None):
    """depart_speed: insertion speed (m/s).  Default V_PHYS so SUMO inserts at the
    controller's speed cap — no first-step 13.89→11 snap, and the torch training sim
    (which spawns at ≤ V_PHYS) sees the same initial speeds.  Pass "desired" for the
    native-Krauss baseline (inserts at maxSpeed)."""
    ds = V_PHYS if depart_speed is None else depart_speed
    with open(path, "w") as f:
        f.write('<routes>\n'
                '  <vType id="car" accel="2.6" decel="4.5" sigma="0" length="5" '
                'minGap="2.0" maxSpeed="13.89" carFollowModel="Krauss"/>\n')
        rate = flow / 3600.0   # veh/s — exp() rate → Poisson (random) inter-arrivals
        for fid, frm, to in [("ew", "east_in", "west_out"), ("we", "west_in", "east_out"),
                             ("ns", "north_in", "south_out"), ("sn", "south_in", "north_out")]:
            f.write(f'  <flow id="{fid}" type="car" from="{frm}" to="{to}" begin="0" '
                    f'end="120" period="exp({rate:.5f})" departLane="best" departSpeed="{ds}"/>\n')
        f.write('</routes>\n')


def write_routes_explicit(events, path):
    """Write an EXACT schedule: events = [(depart_time, move_idx), ...].  Lets the
    torch sim and SUMO run the identical traffic, isolating the collision criterion."""
    rids   = ["rEW", "rWE", "rNS", "rSN"]
    fromto = list(_MOVE_TO.items())            # [(east_in,west_out), (west_in,east_out), ...]
    with open(path, "w") as f:
        f.write('<routes>\n'
                '  <vType id="car" accel="2.6" decel="4.5" sigma="0" length="5" '
                'minGap="2.0" maxSpeed="13.89" carFollowModel="Krauss"/>\n')
        for rid, (frm, to) in zip(rids, fromto):
            f.write(f'  <route id="{rid}" edges="{frm} {to}"/>\n')
        for k, (t, mi) in enumerate(events):
            f.write(f'  <vehicle id="v{k}" type="car" route="{rids[mi]}" '
                    f'depart="{t:.2f}" departLane="best" departSpeed="{V_PHYS}"/>\n')
        f.write('</routes>\n')


# ── true per-pair conflict points from the junction internal-lane geometry ──────
_MOVE_TO = {"east_in": "west_out", "west_in": "east_out",
            "north_in": "south_out", "south_in": "north_out"}
_MOVES   = ["east_in", "west_in", "north_in", "south_in"]   # movement index order
_CP_CACHE = {}


def _seg_x(p1, p2, p3, p4):
    """Intersection point of segments p1p2 and p3p4, or None."""
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
    """First intersection of two polylines (lists of (x,y)), or None."""
    for i in range(len(A) - 1):
        for j in range(len(B) - 1):
            pt = _seg_x(A[i], A[i + 1], B[j], B[j + 1])
            if pt is not None:
                return pt
    return None


def conflict_points(net_path):
    """CP[(move_a, move_b)] = true geometric crossing of the two movements' through
    paths (from-lane end + internal via-lane shape + to-lane start), read from the
    SUMO net.  None where the paths don't cross (parallel/same-axis).  Cached."""
    if net_path in _CP_CACHE:
        return _CP_CACHE[net_path]
    net = sumolib.net.readNet(net_path, withInternal=True)
    shapes = {}
    for frm, to in _MOVE_TO.items():
        conn = net.getEdge(frm).getConnections(net.getEdge(to))[0]
        via  = conn.getViaLaneID()
        vshape = list(net.getLane(via).getShape()) if via else []
        shapes[frm] = ([conn.getFromLane().getShape()[-1]] + vshape
                       + [conn.getToLane().getShape()[0]])
    CP = {(a, b): _poly_x(shapes[a], shapes[b])
          for a in _MOVES for b in _MOVES if a != b}
    _CP_CACHE[net_path] = CP
    return CP


def _cp_tensors(net_path):
    """[4,4] tensors of conflict-point x and y, indexed by movement index (NaN where
    no crossing)."""
    CP = conflict_points(net_path)
    cx = torch.full((4, 4), float("nan"))
    cy = torch.full((4, 4), float("nan"))
    for i, a in enumerate(_MOVES):
        for j, b in enumerate(_MOVES):
            pt = CP.get((a, b))
            if pt is not None:
                cx[i, j], cy[i, j] = float(pt[0]), float(pt[1])
    return cx, cy


def run(flow=500, seed=0, gui=False, realtime=False, track=(), routes_override=None,
        gate2d=True, role_hold=ROLE_HOLD, mean_model=None, hinge_probe=False,
        hinge_gate=False, conflict_probe=False):
    routes   = routes_override or os.path.join(HERE, f"_cosim_routes_{flow}.rou.xml")
    net_path = os.path.join(HERE, "intersection.net.xml")
    if routes_override is None:
        write_routes(flow, routes)
    sumo = sumolib.checkBinary("sumo-gui" if gui else "sumo")
    cmd = [sumo, "-n", net_path, "-r", routes, "--begin", "0", "--end", str(T_END),
           "--step-length", str(DT), "--seed", str(seed),
           "--no-step-log", "true", "--no-warnings", "true",
           "--collision.action", "warn", "--collision.check-junctions", "true",
           "--collision.mingap-factor", "0", "--time-to-teleport", "-1"]
    if gui:
        cmd += ["--start", "--quit-on-end", "--delay", "0"]
    traci.start(cmd)
    if gui:
        try:
            traci.gui.setSchema("View #0", "real world")
            traci.gui.setBoundary("View #0", 120, 120, 280, 280)
        except traci.TraCIException:
            pass
        print("car colors:  RED = yielding   GREEN = passing   GREY = unassigned (free-flow)")

    cpx, cpy = _cp_tensors(net_path)                              # [4,4] per-pair CP
    ax_t  = torch.tensor([ORIGIN[m][0] for m in _MOVES])          # axis by move idx
    sg_t  = torch.tensor([ORIGIN[m][1] for m in _MOVES])          # sign by move idx
    mv_id = {m: i for i, m in enumerate(_MOVES)}

    # 2D rollout gate: path polylines + the gate itself live in sim_torch
    # (lazy import to avoid the circular module load — sim_torch imports this module)
    import sim_torch as S
    from role_memory import RoleMemory   # lazy too (role_memory imports this module)
    geo, _, _, s_junc = S.build_geometry(net_path)

    configured = set()
    move_of    = {}                       # vid -> movement index (cached once)
    prev_a     = {}                       # vid -> last applied accel (Angle-2 lag)
    # priority MEMORY (latch / stuck / queue / arbiter / liveness) — extracted to
    # role_memory.RoleMemory so the TRAINING rollout uses the identical machinery
    role_mem   = RoleMemory(dt=DT)
    collided, col_details, collision_steps = set(), [], 0
    # hinge probe: the TRAINING hinge evaluated on REAL SUMO positions —
    # Σ relu(D_SAFE_2D − d_ij)² over cross-axis pairs near the box (centres).
    hinge_total, hinge_pairs = 0.0, {}      # (vi,vj) -> (min centre dist, t at min)
    track = set(track)
    track_log = {v: [] for v in track}
    arrived_series = []
    speed_sum, speed_cnt = 0.0, 0          # network-mean speed over all vehicle-steps
    accel_sq, accel_n = 0.0, 0             # control ENERGY: Σ a² over vehicle-steps
    jerk_sq,  jerk_n  = 0.0, 0             # SMOOTHNESS: Σ (Δa)² (total change of accel)
    # conflict-gap probe: τ_c (= conflict_time_gap) over egos that actually have a valid
    # crossing rival, tracked over the whole run (min = closest call, max = loosest)
    tc_min, tc_max, tc_sum, tc_n = float("inf"), 0.0, 0.0, 0
    n_steps = int(T_END / DT)

    for step in range(n_steps):
        t_wall = time.perf_counter()
        ids = traci.vehicle.getIDList()
        for v in ids:
            if v not in configured:
                traci.vehicle.setSpeedMode(v, 0)
                traci.vehicle.setLaneChangeMode(v, 0)
                traci.vehicle.subscribe(v, (tc.VAR_POSITION, tc.VAR_SPEED,
                                            tc.VAR_DISTANCE, tc.VAR_ROAD_ID))
                move_of[v] = mv_id.get(traci.vehicle.getRoute(v)[0], 0)
                configured.add(v)
        sub = traci.vehicle.getAllSubscriptionResults()          # ONE round-trip

        vehs = [v for v in ids if v in sub]
        N = len(vehs)
        if N == 0:
            traci.simulationStep()
            arrived_series.append(traci.simulation.getArrivedNumber())
            continue

        xs = torch.tensor([sub[v][tc.VAR_POSITION][0] for v in vehs])
        ys = torch.tensor([sub[v][tc.VAR_POSITION][1] for v in vehs])
        vs = torch.tensor([sub[v][tc.VAR_SPEED]       for v in vehs])
        speed_sum += float(vs.sum()); speed_cnt += N
        # front arc-length along the path (validated: getDistance + L, departPos="base")
        s_front = torch.tensor([sub[v][tc.VAR_DISTANCE] for v in vehs]) + utils.L_VEH
        mv = torch.tensor([move_of[v]                 for v in vehs])
        # SUMO road id: ":" prefix ⇒ vehicle is on an internal junction lane (IN THE BOX).
        # This is the exact box-occupancy test for the stuck timer — distance-to-centre is
        # wrong (a car stopped in the first half of the box hasn't passed centre yet).
        in_box_v = [sub[v][tc.VAR_ROAD_ID].startswith(":") for v in vehs]
        axis, sgn = ax_t[mv], sg_t[mv]

        # platoon pressure per lane: NUMBER of same-lane vehicles still approaching the
        # junction (the platoon length) — so the longer platoon outranks a shorter cross
        # platoon and clears as a unit.  behind_n = how many same-lane vehicles are still
        # farther back, used to HOLD a passer's role until its platoon has gone through.
        coord    = torch.where(axis == 0, xs, ys)
        d_junc   = (200.0 - coord) * sgn                          # >0 approaching, <0 past
        samelane = ((axis.unsqueeze(0) == axis.unsqueeze(1)) &
                    (sgn.unsqueeze(0) == sgn.unsqueeze(1)))       # [N, N]
        appr     = (d_junc > 0.0) & (d_junc < 120.0)              # [N]
        same_ap  = samelane & appr.unsqueeze(0)                   # [N,N] approaching same-lane
        P        = same_ap.float().sum(dim=1)                     # [N] platoon size (count)
        behind_n = (same_ap & (d_junc.unsqueeze(0) > d_junc.unsqueeze(1))).float().sum(dim=1)  # [N]
        a_prev_t = torch.tensor([prev_a.get(v, 0.0) for v in vehs])

        # leader (longitudinal): getLeader for the gap, leader speed from the cache
        gap = torch.full((N,), 300.0)
        v_lead = vs.clone()
        for i, v in enumerate(vehs):
            ld = traci.vehicle.getLeader(v, 120.0)
            if ld is not None and ld[0] != "":
                gap[i] = max(ld[1], 0.0)
                v_lead[i] = sub[ld[0]][tc.VAR_SPEED] if ld[0] in sub else float(vs[i])

        # cross-traffic via TRUE per-pair conflict points → ego_d, rival_d  [N, N]
        CPx = cpx[mv][:, mv]                                       # [N, N]
        CPy = cpy[mv][:, mv]
        xi, yi, axi, sgi = xs.unsqueeze(1), ys.unsqueeze(1), axis.unsqueeze(1), sgn.unsqueeze(1)
        xj, yj, axj, sgj = xs.unsqueeze(0), ys.unsqueeze(0), axis.unsqueeze(0), sgn.unsqueeze(0)
        ego_d   = torch.where(axi == 0, (CPx - xi) * sgi, (CPy - yi) * sgi)
        rival_d = torch.where(axj == 0, (CPx - xj) * sgj, (CPy - yj) * sgj)
        rival_v = vs.unsqueeze(0).expand(N, N)
        eye   = torch.eye(N, dtype=torch.bool)
        valid = ((axi != axj) & (~eye) & (ego_d > 0.0) & (rival_d > -CLEAR)
                 & ~torch.isnan(CPx))
        ego_d   = torch.nan_to_num(ego_d,   nan=1e3)
        rival_d = torch.nan_to_num(rival_d, nan=-1e3)

        # ── conflict-gap probe: same τ_c the controller sees, over conflicted egos ──────
        if conflict_probe:
            has_conf = valid.any(dim=1)
            if bool(has_conf.any()):
                tau_c_all, *_ = utils.conflict_time_gap(ego_d, vs, rival_d, rival_v, valid)
                tcv = tau_c_all[has_conf]
                tc_min = min(tc_min, float(tcv.min()))
                tc_max = max(tc_max, float(tcv.max()))
                tc_sum += float(tcv.sum()); tc_n += int(tcv.numel())

        # ── hinge probe (proxy diagnostics; SUMO position = FRONT centre → shift L/2 back)
        if hinge_probe:
            half = utils.L_VEH / 2.0
            cx = xs - torch.where(axis == 0, sgn, torch.zeros_like(sgn)) * half
            cy = ys - torch.where(axis == 1, sgn, torch.zeros_like(sgn)) * half
            near = (d_junc > -S.JCT_PAST) & (d_junc < 60.0)
            ni = near.nonzero().flatten()
            if len(ni) >= 2:
                dist = ((cx[ni].unsqueeze(0) - cx[ni].unsqueeze(1)) ** 2
                        + (cy[ni].unsqueeze(0) - cy[ni].unsqueeze(1)) ** 2).sqrt()
                cross = axis[ni].unsqueeze(0) != axis[ni].unsqueeze(1)
                cross = cross & torch.triu(torch.ones_like(cross), 1)
                hinge_total += float((torch.relu(S.D_SAFE_2D - dist[cross]) ** 2).sum()) * DT
                for p, q in (cross & (dist < S.D_SAFE_2D)).nonzero().tolist():
                    va, vb = vehs[int(ni[p])], vehs[int(ni[q])]
                    key = (va, vb) if va < vb else (vb, va)
                    d_ = float(dist[p, q])
                    if key not in hinge_pairs or d_ < hinge_pairs[key][0]:
                        hinge_pairs[key] = (d_, round(step * DT, 1))

        # ── PRIORITY ESTIMATION WITH MEMORY (done HERE, in the state step) ──────────────
        # predecessor_gap PROPOSES the instantaneous role; role_memory.RoleMemory (the
        # EXACT machinery that used to live inline here — latch, queue arbiter, passer
        # compatibility arbiter, liveness) resolves the FINAL roles and pred_override.
        # Extracted so the TRAINING rollout consumes the identical role dynamics.
        prop = utils.predecessor_gap(
            ego_d, vs, rival_d, rival_v, valid,
            delta_safe=utils.DELTA_SAFE, ego_P=P, rival_P=P.unsqueeze(0).expand(N, N))
        t_now = step * DT
        pred_override, roles_i = role_mem.step(
            vehs, vs, d_junc, in_box_v, behind_n, mv, ego_d, rival_d, valid,
            prop, t_now, role_hold)

        if gui:
            for i, vco in enumerate(vehs):
                traci.vehicle.setColor(vco, ROLE_COLOR[roles_i[i]])

        # ── transformer prior mean (optional): encode the conflict set + rear pressure
        # ONCE per ego, hand the controller a mean_fn closure.  The GP correction pins
        # the anchor targets exactly regardless of f; the gate below is untouched and
        # consumes the conditional-mean output like any other command.
        mean_fn = None
        if mean_model is not None:
            import mean_net
            ctx = mean_net.build_context(vs, gap, v_lead, P, behind_n, d_junc,
                                         ego_d, rival_d, valid, roles_i, axis, sgn)
            mean_fn = mean_model.make_mean_fn(mean_model.encode(*ctx))

        xe = torch.zeros(N)
        xl = gap + utils.L_VEH
        out = utils.controller_acceleration(
            xe, xl, vs, v_lead,
            d_conf=ego_d, rival_d=rival_d, rival_v=rival_v, rival_valid=valid,
            ego_pressure=P, rival_pressure=P.unsqueeze(0).expand(N, N),
            a_prev=a_prev_t, kappa=0.5, brake_exempt=True,
            pred_override=pred_override, return_roles=gate2d,
            return_feat=(gate2d and hinge_gate), mean_fn=mean_fn)
        if gate2d:
            a, yield_mask, feat = out if hinge_gate else (*out, None)
            # PROTECTED passers (queue- or liveness-promoted) are force-rolled so the
            # gate always commits them and reserves their axis — crossing traffic
            # plans around them instead of being blind to an undesignated passer.
            force_roll = torch.tensor([roles_i[i] == 'pass'
                                       and role_mem.protected(vehs[i], t_now)
                                       for i in range(N)])
            a, defer = S.rollout_gate(a.detach(), s_front, vs, mv, yield_mask, geo, s_junc,
                                      return_defer=True, force_roll=force_roll)
            # FGD polish: L2 functional-gradient descent on the 2-D hinge, AFTER the
            # role gate (box-exclusivity intact).  Can brake past the gate's −3 comfort
            # floor toward −B_MAX when a conflict is imminent — the extra authority the
            # discrete gate lacks.  Runs in the real SUMO loop so it is validated here.
            if hinge_gate:
                a = S.hinge_gradient_gate(a.detach(), feat.detach(),
                                          s_front, vs, mv, geo, s_junc).detach()
            # FEEDBACK: a passer the gate forced to brake (or a vehicle it halted at the
            # stop-line) is deferring → latch it as a yielder for role_hold s, so the gate's
            # tiebreak persists instead of being re-fought (and re-flipped) next step.
            # (protected queue passers are NOT re-labelled — their window holds)
            role_mem.gate_feedback(vehs, defer, t_now, role_hold)
            # POST-GATE liveness: never end a step with zero effective passers — if the
            # gate just neutralized the only one, promote (protected) for the next steps.
            role_mem.ensure_passer(vehs, roles_i, defer, in_box_v, d_junc, t_now, role_hold)
        else:
            a = out.detach()        # raw kernel-interpolation anchors, no 2-D gate

        for i, v in enumerate(vehs):
            ai = float(a[i])
            # YIELDER: no positive acceleration — EXCEPT in the box, where the gate's
            # deadlock breaker is driving it OUT (clamping there freezes it in the box)
            if roles_i[i] == 'yield' and not in_box_v[i]:
                ai = min(ai, 0.0)
            ai_prev = prev_a.get(v)              # previous applied accel (None if new vehicle)
            accel_sq += ai * ai; accel_n += 1
            if ai_prev is not None:              # skip the 0→a₀ jump on a fresh vehicle
                jerk_sq += (ai - ai_prev) ** 2; jerk_n += 1
            prev_a[v] = ai
            v_new = float(min(max(float(vs[i]) + ai * DT, 0.0), V_PHYS))
            # PASSER keeps moving (non-zero v) — but ONLY with room ahead, so the floor never
            # overrides car-following into a rear-end on a stopped leader.
            if (roles_i[i] == 'pass' and not (gate2d and bool(defer[i]))
                    and float(gap[i]) > 2.0 * utils.L_VEH):
                v_new = max(v_new, V_MIN_PASS)
            traci.vehicle.setSpeed(v, v_new)
            if v in track:
                track_log[v].append((round(step * DT, 1), round(float(xs[i]), 2),
                                     round(float(ys[i]), 2), round(float(vs[i]), 3),
                                     round(float(a[i]), 3)))

        traci.simulationStep()
        col = traci.simulation.getCollidingVehiclesIDList()
        if col:
            collision_steps += 1
            new = [c for c in col if c not in collided]
            collided.update(col)
            for c in new[:2]:
                if len(col_details) < 12:
                    try:
                        rd_ = traci.vehicle.getRoadID(c)
                        x_, y_ = traci.vehicle.getPosition(c)
                        sp_ = traci.vehicle.getSpeed(c)
                        ax_ = ORIGIN.get(traci.vehicle.getRoute(c)[0], (-1, 0))[0]
                        zone = ("junction" if rd_.startswith(":") else
                                "approach" if rd_.endswith("_in") else "exit")
                        col_details.append((round(step * DT, 1), c, rd_, zone, ax_,
                                            round(x_, 1), round(y_, 1), round(sp_, 2)))
                    except traci.TraCIException:
                        pass
        arrived_series.append(traci.simulation.getArrivedNumber())

        if realtime:
            lag = DT - (time.perf_counter() - t_wall)
            if lag > 0:
                time.sleep(lag)

    total_arrived = int(np.sum(arrived_series))
    i40 = int(40 / DT)
    ss_rate = np.sum(arrived_series[i40:]) / ((n_steps - i40) * DT) * 3600
    traci.close()
    return dict(flow=flow, seed=seed, arrived=total_arrived, ss_rate=ss_rate,
                avg_speed=speed_sum / max(speed_cnt, 1),
                collided=len(collided), collision_steps=collision_steps,
                col_details=col_details, track_log=track_log,
                hinge=hinge_total, hinge_pairs=hinge_pairs,
                energy=accel_sq / max(accel_n, 1),     # mean a²  (control energy)
                jerk=jerk_sq / max(jerk_n, 1),         # mean (Δa)²  (smoothness)
                tau_c_min=(tc_min if tc_n else float("nan")),
                tau_c_max=(tc_max if tc_n else float("nan")),
                tau_c_mean=(tc_sum / tc_n if tc_n else float("nan")),
                collided_ids=sorted(collided))


if __name__ == "__main__":
    tokens = sys.argv[1:]
    gui    = "gui" in tokens
    # "nogate" → run the raw kernel-interpolation command (anchors only, no 2-D gate),
    # so you can see in the GUI what the GP anchors alone produce.
    gate2d = "nogate" not in tokens
    # "dsafe=<val>" (alias "ds=") → override the kernel's target conflict time-gap δ_safe
    # (default 5 s) so you can sweep how conservatively yielders behave.  Set ONCE here,
    # before any controller call, so the anchor grid + GP inverse rebuild cleanly.
    ds_tok = next((t for t in tokens if t.startswith(("dsafe=", "ds="))), None)
    if ds_tok is not None:
        utils.set_delta_safe(float(ds_tok.split("=", 1)[1]))
    # "hold=<s>" (alias "reest=") → seconds between yield/pass RE-ESTIMATIONS (the role latch
    # period).  hold=0 → re-decide every step (the old stateless behavior).
    h_tok = next((t for t in tokens if t.startswith(("hold=", "reest="))), None)
    role_hold = float(h_tok.split("=", 1)[1]) if h_tok is not None else ROLE_HOLD
    # transformer prior mean: by default load the BEST trained model (train_mean.py
    # checkpoint) if one exists; otherwise fall back to the plain zero-mean kernel.
    #   "nonn" → force the plain kernel even if a checkpoint exists
    #   "nn"   → attach a FRESH zero-init model (f ≡ 0 ⇒ identical to the plain
    #            kernel — kept as the integration check)
    mean_model = None
    if "nn" in tokens:
        import mean_net
        mean_model = mean_net.make_mean_model()
        mean_model.eval()
        print("nn: fresh ZERO-INIT prior mean attached (must match the plain kernel)")
    elif "nonn" not in tokens:
        import mean_net
        ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), mean_net.ckpt_path("best"))
        if os.path.exists(ckpt):
            ck = torch.load(ckpt, weights_only=True)
            sd, bj = ((ck["state_dict"], ck.get("best_j"))
                      if isinstance(ck, dict) and "state_dict" in ck else (ck, None))
            mean_model = mean_net.make_mean_model()
            mean_model.load_state_dict(sd)
            mean_model.eval()
            print(f"loaded best trained prior mean from {os.path.basename(ckpt)}"
                  + (f" (mean J̄={bj:.4f})" if bj is not None else ""))
        else:
            print("no trained model found (mean_net_ckpt.pt) → plain kernel controller")
    # "hinge" → also evaluate the training hinge on the real SUMO positions and report
    # every cross pair that came inside D_SAFE_2D (proxy diagnostics vs real collisions)
    hinge_probe = "hinge" in tokens
    hinge_gate  = "fgd" in tokens          # FGD polish: L2 hinge-gradient correction
    _kv = ("gui", "realtime", "nogate", "nn", "nonn", "hinge", "fgd")
    args = [a for a in tokens
            if a not in _kv and not a.startswith(("dsafe=", "ds=", "hold=", "reest="))]
    flow = int(args[0]) if len(args) > 0 else 500
    seed = int(args[1]) if len(args) > 1 else 0
    r = run(flow, seed, gui=gui, realtime=gui, gate2d=gate2d, role_hold=role_hold,
            mean_model=mean_model, hinge_probe=hinge_probe, hinge_gate=hinge_gate)
    mode = ("kernel+gate" if gate2d else "kernel-only (no gate)") + (" +FGD" if hinge_gate else "")
    if mean_model is not None:
        mode += " +nn-mean (zero-init)" if "nn" in tokens else " +nn-mean (best ckpt)"
    print(f"=== OUR controller co-simulated in SUMO (TraCI), {flow} vph/approach, seed {seed}"
          f"  [{mode}, δ_safe={utils.DELTA_SAFE:.1f}s, re-est={role_hold:.2f}s] ===")
    print(f"  arrived (cleared, 120s)        : {r['arrived']}")
    print(f"  steady-state throughput        : {r['ss_rate']:.0f} veh/h")
    print(f"  vehicles in a SUMO collision   : {r['collided']}")
    print(f"  steps with a collision         : {r['collision_steps']}")
    print(f"  control energy  (mean a²)      : {r['energy']:.4f} (m/s²)²")
    print(f"  smoothness  (mean (Δa)²)       : {r['jerk']:.4f} (m/s²)²")
    if r["col_details"]:
        print("  --- collision contexts (t, veh, road, zone, axis, x, y, v) ---")
        for d in r["col_details"]:
            print(f"    t={d[0]:5.1f} {d[1]:>10} road={d[2]:>10} {d[3]:>8} axis={d[4]} "
                  f"pos=({d[5]:6.1f},{d[6]:6.1f}) v={d[7]:.2f}")
    if hinge_probe:
        hp = sorted(r["hinge_pairs"].items(), key=lambda kv: kv[1][0])
        print(f"  hinge integral relu(6.5-d)^2   : {r['hinge']:.2f}"
              f"   cross pairs inside 6.5 m: {len(hp)}")
        for (va, vb), (dmin, tmin) in hp[:12]:
            mark = "  <- COLLIDED" if va in r["collided_ids"] and vb in r["collided_ids"] else ""
            print(f"    {va:>6} x {vb:<6} min_d={dmin:5.2f} m at t={tmin:6.1f}{mark}")
