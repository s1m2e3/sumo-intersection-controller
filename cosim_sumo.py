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
ROLE_HOLD   = 3.0    # s   a once-assigned role is LATCHED (not re-decided) for this long
STUCK_V     = 3.0    # m/s "low speed" threshold for accumulating in-junction stuck time
STUCK_HOLD  = 3.0    # s   a yielder stuck (in-junction, < STUCK_V) longer than this → PASS
V_MIN_PASS  = 0.5    # m/s a PASSER is kept moving (non-zero velocity) — it never freezes

# ── queue arbiter parameters ────────────────────────────────────────────────────
QUEUE_MIN_BEHIND = 3     #     ≥ this many vehicles tightly behind ⇒ a real standing queue
QUEUE_SPAN       = 40.0  # m   window behind the front within which queued cars are counted
QUEUE_WAIT       = 5.0   # s   a queue waiting longer than this earns a clearing window
QUEUE_CLEAR      = 5.0   # s   protected window: the promoted passers' role can't be re-estimated
QUEUE_N_PASS     = 4     #     how many front-of-queue vehicles are promoted to PASS together
ARBITER_GAP      = 2.0   # s   two passers whose bumper-aware occupancy windows come within this
                         #     of each other are INCOMPATIBLE → the later one is demoted to yielder

# GUI role colors (RGBA): visualise each car's resolved priority role in sumo-gui
ROLE_COLOR = {"yield": (220, 40, 40, 255),    # red   — yielding
              "pass":  (40, 200, 40, 255),    # green — passing
              "none":  (170, 170, 170, 255)}  # grey  — unassigned (free-flow approach)

# origin edge -> (axis 0=x/1=y, travel sign along that axis)
ORIGIN = {"east_in": (0, -1.0), "west_in": (0, +1.0),
          "north_in": (1, -1.0), "south_in": (1, +1.0)}


def write_routes(flow, path):
    with open(path, "w") as f:
        f.write('<routes>\n'
                '  <vType id="car" accel="2.6" decel="4.5" sigma="0" length="5" '
                'minGap="2.0" maxSpeed="13.89" carFollowModel="Krauss"/>\n')
        rate = flow / 3600.0   # veh/s — exp() rate → Poisson (random) inter-arrivals
        for fid, frm, to in [("ew", "east_in", "west_out"), ("we", "west_in", "east_out"),
                             ("ns", "north_in", "south_out"), ("sn", "south_in", "north_out")]:
            f.write(f'  <flow id="{fid}" type="car" from="{frm}" to="{to}" begin="0" '
                    f'end="120" period="exp({rate:.5f})" departLane="best" departSpeed="desired"/>\n')
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
                    f'depart="{t:.2f}" departLane="best" departSpeed="desired"/>\n')
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
        gate2d=True, role_hold=ROLE_HOLD):
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
    geo, _, _, s_junc = S.build_geometry(net_path)

    configured = set()
    move_of    = {}                       # vid -> movement index (cached once)
    prev_a     = {}                       # vid -> last applied accel (Angle-2 lag)
    # ── priority MEMORY (stateful, carried across steps; see priority-estimation block) ──
    role       = {}                       # vid -> 'yield' | 'pass' | 'none'  (latched)
    role_exp   = {}                       # vid -> sim-time the latch holds until
    stuck_time = {}                       # vid -> accumulated s in-junction at v < STUCK_V
    queue_wait = {}                       # movement idx -> s its standing queue has waited
    queue_until = {}                      # vid -> sim-time a queue-promoted PASS is protected to
    collided, col_details, collision_steps = set(), [], 0
    track = set(track)
    track_log = {v: [] for v in track}
    arrived_series = []
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

        # ── PRIORITY ESTIMATION WITH MEMORY (done HERE, in the state step) ──────────────
        # 1. predecessor_gap PROPOSES the instantaneous role (+ live predecessor target).
        # 2. memory resolves the FINAL role per vehicle:
        #      • assign a role only within ASSIGN_DIST of the box; farther out → free-flow
        #      • LATCH it for ROLE_HOLD s (no step-to-step re-deciding)
        #      • a passer keeps its latch while its platoon (behind_n) hasn't cleared
        #      • a yielder stuck in-junction (< STUCK_V) > STUCK_HOLD s is forced to PASS
        # 3. the resolved role overrides has_pred / is_pred and is fed to the kernel as
        #    pred_override, so the kernel consumes the decision rather than re-deriving it.
        prop = utils.predecessor_gap(
            ego_d, vs, rival_d, rival_v, valid,
            delta_safe=utils.DELTA_SAFE, ego_P=P, rival_P=P.unsqueeze(0).expand(N, N))
        tau_c, eta_pred, ego_d_pred, v_pred, has_pred, is_pred = prop
        t_now   = step * DT
        has_res = has_pred.clone()
        is_res  = is_pred.clone()
        tau_res = tau_c.clone()                                  # τ_c fed to the kernel per role

        # ── QUEUE ARBITER: a standing queue (≥ QUEUE_MIN_BEHIND tightly behind a slow front)
        # that has waited > QUEUE_WAIT s gets a clearing window — its front QUEUE_N_PASS
        # vehicles are PROMOTED to PASS and PROTECTED (role frozen) for QUEUE_CLEAR s, which
        # forces the crossing stream to yield until the block clears.  Only the longest-waiting
        # lane is opened at a time (one stream crosses at once).
        for mi in range(4):
            lane = ((mv == mi) & (d_junc > 0.0)).nonzero().flatten()
            if len(lane) == 0:
                queue_wait[mi] = 0.0; continue
            front = lane[torch.argsort(d_junc[lane])[0]]
            fdj   = float(d_junc[front])
            n_behind = int(((mv == mi) & (d_junc > fdj) & (d_junc < fdj + QUEUE_SPAN)).sum())
            if n_behind >= QUEUE_MIN_BEHIND and float(vs[front]) < STUCK_V:
                queue_wait[mi] = queue_wait.get(mi, 0.0) + DT
            else:
                queue_wait[mi] = 0.0
        # don't open a lane whose CROSS axis already has a vehicle in the box — that would
        # promote a passer straight into a committed crosser neither side can yield to.
        occ_ax = {int(ax_t[int(mv[j])]) for j in range(N) if in_box_v[j]}
        ready = [mi for mi in range(4)
                 if queue_wait.get(mi, 0.0) > QUEUE_WAIT
                 and not any(a != int(ax_t[mi]) for a in occ_ax)]
        if ready:
            mi = max(ready, key=lambda m: queue_wait[m])         # longest-waiting lane opens
            lane = ((mv == mi) & (d_junc > 0.0)).nonzero().flatten()
            front4 = lane[torch.argsort(d_junc[lane])[:QUEUE_N_PASS]]
            for li in front4.tolist():
                queue_until[vehs[li]] = t_now + QUEUE_CLEAR      # protected PASS window
            queue_wait[mi] = 0.0                                 # window opened → reset its timer

        roles_i = ['none'] * N                                   # resolved role per vehicle (for clamps)
        for i, vco in enumerate(vehs):
            dj, sp = float(d_junc[i]), float(vs[i])
            # accumulate WAIT time for any slow vehicle near the box (queued at the line OR
            # inside it), so a stuck BLOCK — not only an in-box car — earns a turn: once the
            # front has waited > STUCK_HOLD it is promoted to PASS and the platoon follows.
            if (in_box_v[i] or dj <= ASSIGN_DIST) and sp < STUCK_V:
                stuck_time[vco] = stuck_time.get(vco, 0.0) + DT
            else:
                stuck_time[vco] = 0.0                            # moving or far → reset
            if t_now < queue_until.get(vco, -1.0):
                # PROTECTED queue passer — role frozen for the clearing window, no re-estimation
                r = 'pass'; role[vco] = 'pass'
            else:
                r, exp = role.get(vco, 'none'), role_exp.get(vco, -1.0)
                # passer holds its latch until its platoon behind has cleared
                latched = (t_now < exp) or (r == 'pass' and behind_n[i] > 0)
                if not latched:
                    if dj <= ASSIGN_DIST:
                        r = 'yield' if bool(has_pred[i]) else 'pass'
                        role[vco], role_exp[vco] = r, t_now + role_hold
                    else:
                        r, role[vco] = 'none', 'none'
                # stuck yielder → force PASS (decision-level deadlock break, works in no-gate)
                if r == 'yield' and stuck_time.get(vco, 0.0) > STUCK_HOLD:
                    r, role[vco], role_exp[vco], stuck_time[vco] = 'pass', 'pass', t_now + role_hold, 0.0
            roles_i[i] = r
            if r == 'yield':
                has_res[i] = True                               # keep live predecessor target + τ_c
            else:                                               # 'pass' or 'none' → not yielding
                has_res[i] = False
                is_res[i, :] = False                            # clears its row for the gate's role view
                tau_res[i]  = utils.TAU_C_MAX                    # NO cross conflict → free-flow fires

        # ── PASSER COMPATIBILITY ARBITER ────────────────────────────────────────────────
        # Among tentative PASSERS, confirm earliest-ETA first; demote the later of any crossing
        # pair whose bumper-aware occupancy windows come within ARBITER_GAP s → the confirmed
        # passer set is mutually collision-free.  A demoted passer becomes a strict YIELDER,
        # re-pointed to yield to the passer it lost to.  Queue-promoted passers are protected
        # (confirmed first, never demoted).  Anti-oscillation: a vehicle only STAYS a passer if
        # it's compatible, so it can't flip back into a conflict it just lost.
        eta_pred = eta_pred.clone(); ego_d_pred = ego_d_pred.clone(); v_pred = v_pred.clone()
        CL = utils.CONFLICT_LEN
        committed_i = [bool(in_box_v[i]) or
                       (float(d_junc[i]) <= float(vs[i]) ** 2 / (2 * utils.B_MAX) + utils.STOP_OFFSET)
                       for i in range(N)]

        def _conflict(p, q):
            if not (bool(valid[p, q]) or bool(valid[q, p])):
                return False
            vp, vq = max(float(vs[p]), 0.1), max(float(vs[q]), 0.1)
            dp, dq = float(ego_d[p, q]), float(rival_d[p, q])
            pin, pout = dp / vp, (dp + CL) / vp                 # p occupies [pin, pout]
            qin, qout = dq / vq, (dq + CL) / vq                 # q occupies [qin, qout]
            return max(pin - qout, qin - pout) < ARBITER_GAP    # windows within the margin

        def _demote(i, q):                                      # i yields to passer q
            vq = max(float(vs[q]), 0.1)
            roles_i[i] = 'yield'; role[vehs[i]] = 'yield'; role_exp[vehs[i]] = t_now + role_hold
            has_res[i] = True
            ego_d_pred[i] = ego_d[i, q]
            eta_pred[i]   = (rival_d[i, q] + CL) / vq           # q's rear-out time (bumper-aware)
            v_pred[i]     = vs[q]
            tau_res[i]    = max(float(ego_d[i, q]) / max(float(vs[i]), 0.1)
                                - float(eta_pred[i]), 0.0)
            is_res[i, :] = False; is_res[i, q] = True

        passers   = [i for i in range(N) if roles_i[i] == 'pass']
        protected = [i for i in passers if t_now < queue_until.get(vehs[i], -1.0)]
        rest = sorted((i for i in passers if i not in protected),
                      key=lambda i: float(d_junc[i]) / max(float(vs[i]), 0.5))   # by ETA
        confirmed = list(protected)
        for p in rest:
            q = next((c for c in confirmed if _conflict(p, c)), None)
            if q is None:
                confirmed.append(p)                            # compatible → stays PASSER
            elif committed_i[p] and not committed_i[q] and q not in protected:
                _demote(q, p); confirmed.remove(q); confirmed.append(p)   # p can't stop → q yields
            else:
                _demote(p, q)                                  # later/abletostop → p yields

        # LIVENESS: the junction must always have ≥1 PASSER.  If every contesting vehicle ended
        # up a yielder (a latch/arbiter deadlock — mutual yielding), nobody moves and the box
        # never drains.  Promote the front-most contender (in-box first, else closest to the box)
        # to PASS, overriding its latch, so the intersection always has someone clearing.
        contesting = [i for i in range(N) if in_box_v[i] or 0.0 < float(d_junc[i]) <= ASSIGN_DIST]
        # only force a passer when the junction is TRULY stalled: nobody passing, nobody
        # committed (a committed crosser is the de-facto passer), nobody still moving.
        alive = any(roles_i[i] == 'pass' or committed_i[i] or float(vs[i]) > STUCK_V
                    for i in contesting)
        if contesting and not alive:
            best = min(contesting, key=lambda i: (0 if in_box_v[i] else 1, float(d_junc[i])))
            roles_i[best] = 'pass'; role[vehs[best]] = 'pass'; role_exp[vehs[best]] = t_now + role_hold
            has_res[best] = False; is_res[best, :] = False; tau_res[best] = utils.TAU_C_MAX

        if gui:
            for i, vco in enumerate(vehs):
                traci.vehicle.setColor(vco, ROLE_COLOR[roles_i[i]])
        pred_override = (tau_res, eta_pred, ego_d_pred, v_pred, has_res, is_res)

        xe = torch.zeros(N)
        xl = gap + utils.L_VEH
        out = utils.controller_acceleration(
            xe, xl, vs, v_lead,
            d_conf=ego_d, rival_d=rival_d, rival_v=rival_v, rival_valid=valid,
            ego_pressure=P, rival_pressure=P.unsqueeze(0).expand(N, N),
            a_prev=a_prev_t, kappa=0.5, brake_exempt=True,
            pred_override=pred_override, return_roles=gate2d)
        if gate2d:
            a, yield_mask = out
            a, defer = S.rollout_gate(a.detach(), s_front, vs, mv, yield_mask, geo, s_junc,
                                      return_defer=True)
            # FEEDBACK: a passer the gate forced to brake (or a vehicle it halted at the
            # stop-line) is deferring → latch it as a yielder for role_hold s, so the gate's
            # tiebreak persists instead of being re-fought (and re-flipped) next step.
            for i, vco in enumerate(vehs):
                # protected queue passers are NOT re-labelled by the gate (their window holds)
                if bool(defer[i]) and t_now >= queue_until.get(vco, -1.0):
                    role[vco], role_exp[vco] = 'yield', t_now + role_hold
        else:
            a = out.detach()        # raw kernel-interpolation anchors, no 2-D gate

        for i, v in enumerate(vehs):
            ai = float(a[i])
            if roles_i[i] == 'yield':
                ai = min(ai, 0.0)                                # YIELDER: no positive acceleration
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
                collided=len(collided), collision_steps=collision_steps,
                col_details=col_details, track_log=track_log)


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
    _kv = ("gui", "realtime", "nogate")
    args = [a for a in tokens
            if a not in _kv and not a.startswith(("dsafe=", "ds=", "hold=", "reest="))]
    flow = int(args[0]) if len(args) > 0 else 500
    seed = int(args[1]) if len(args) > 1 else 0
    r = run(flow, seed, gui=gui, realtime=gui, gate2d=gate2d, role_hold=role_hold)
    mode = "kernel+gate" if gate2d else "kernel-only (no gate)"
    print(f"=== OUR controller co-simulated in SUMO (TraCI), {flow} vph/approach, seed {seed}"
          f"  [{mode}, δ_safe={utils.DELTA_SAFE:.1f}s, re-est={role_hold:.1f}s] ===")
    print(f"  arrived (cleared, 120s)        : {r['arrived']}")
    print(f"  steady-state throughput        : {r['ss_rate']:.0f} veh/h")
    print(f"  vehicles in a SUMO collision   : {r['collided']}")
    print(f"  steps with a collision         : {r['collision_steps']}")
    if r["col_details"]:
        print("  --- collision contexts (t, veh, road, zone, axis, x, y, v) ---")
        for d in r["col_details"]:
            print(f"    t={d[0]:5.1f} {d[1]:>10} road={d[2]:>10} {d[3]:>8} axis={d[4]} "
                  f"pos=({d[5]:6.1f},{d[6]:6.1f}) v={d[7]:.2f}")
