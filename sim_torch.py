"""
sim_torch.py — pure-PyTorch multi-vehicle intersection sim with SUMO-faithful
collision capture, for FAST iteration (no TraCI/IPC) and a differentiable path.

Geometry is SUMO's: each movement follows the real path polyline (from-lane +
internal via-lane + to-lane) read from intersection.net.xml, so positions match
SUMO to within the integrator.  Vehicles advance arc-length s by v·DT; the same
batched utils.controller_acceleration drives them; per-pair conflict points come
from the true crossing arc-lengths.

Collision criterion = SUMO's: two vehicles collide when their oriented 5.0×1.8 m
footprints overlap (separating-axis test), broad-phased by centre distance.  This
is the same "boxes overlap" test SUMO uses with collision.mingap-factor=0.

    conda run -n car-following-sumo python sim_torch.py [flow] [seed]
"""
import os, sys, time
import numpy as np
import torch
import sumolib

import utils
import cosim_sumo as C   # reuse _MOVES, _MOVE_TO, ORIGIN, conflict_points, constants

DT     = 0.5                 # s — coarse step for fast iteration (cosim uses 0.1)
T_END, V_PHYS = C.T_END, C.V_PHYS
L_VEH, W_VEH = utils.L_VEH, 1.8
CLEAR     = C.CLEAR
SPAWN_GAP = 8.0
NET_PATH  = os.path.join(C.HERE, "intersection.net.xml")
_AXIS = torch.tensor([C.ORIGIN[m][0] for m in C._MOVES])   # axis (0/1) by move idx


# ── geometry: per-movement path polyline + per-pair conflict-point arc-lengths ──
def _project(pts, cum, pt):
    """Arc-length along polyline (pts, cum) of the point on it nearest to pt."""
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = (((pt - a) * ab).sum(1) / (ab * ab).sum(1).clamp(min=1e-9)).clamp(0, 1)
    proj = a + t.unsqueeze(1) * ab
    i = (proj - pt).norm(dim=1).argmin()
    return float(cum[i] + t[i] * (cum[i + 1] - cum[i]))


def build_geometry(net_path=NET_PATH):
    net = sumolib.net.readNet(net_path, withInternal=True)
    geo = {}
    for m, to in C._MOVE_TO.items():
        conn = net.getEdge(m).getConnections(net.getEdge(to))[0]
        via  = list(net.getLane(conn.getViaLaneID()).getShape()) if conn.getViaLaneID() else []
        poly = list(conn.getFromLane().getShape()) + via + list(conn.getToLane().getShape())
        dedup = [poly[0]]                                   # drop consecutive duplicates
        for p in poly[1:]:
            if abs(p[0] - dedup[-1][0]) > 1e-6 or abs(p[1] - dedup[-1][1]) > 1e-6:
                dedup.append(p)
        pts = torch.tensor(dedup, dtype=torch.float32)
        seg = (pts[1:] - pts[:-1]).norm(dim=1)
        cum = torch.cat([torch.zeros(1), torch.cumsum(seg, 0)])
        geo[m] = (pts, cum)
    CP = C.conflict_points(net_path)
    s_cp = torch.full((4, 4), float("nan"))
    for i, a in enumerate(C._MOVES):
        for j, b in enumerate(C._MOVES):
            pt = CP.get((a, b))
            if pt is not None:
                s_cp[i, j] = _project(geo[a][0], geo[a][1], torch.tensor(pt, dtype=torch.float32))
    path_len = torch.tensor([float(geo[m][1][-1]) for m in C._MOVES])
    s_junc   = torch.nan_to_num(s_cp, nan=1e9).min(dim=1).values   # first conflict ≈ junction entry
    return geo, s_cp, path_len, s_junc


def _interp(pts, cum, s):
    """(x,y) and heading at arc-lengths s [M] along polyline (pts, cum)."""
    s = s.clamp(0.0, float(cum[-1]))
    idx = (torch.searchsorted(cum, s, right=True) - 1).clamp(0, len(cum) - 2)
    s0, s1 = cum[idx], cum[idx + 1]
    t = ((s - s0) / (s1 - s0).clamp(min=1e-6)).unsqueeze(-1)
    p0, p1 = pts[idx], pts[idx + 1]
    xy = p0 + t * (p1 - p0)
    d  = p1 - p0
    return xy, torch.atan2(d[:, 1], d[:, 0])


# ── SUMO-faithful collision: oriented-rectangle (footprint) overlap via SAT ─────
def _corners(xy, heading, hl, hw):
    c, sdir = torch.cos(heading), torch.sin(heading)
    dirv  = torch.stack([c, sdir], -1)          # [M,2] along length
    perp  = torch.stack([-sdir, c], -1)         # [M,2] along width
    signs = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
    cor = (xy.unsqueeze(1)
           + signs[:, 0].view(1, 4, 1) * hl * dirv.unsqueeze(1)
           + signs[:, 1].view(1, 4, 1) * hw * perp.unsqueeze(1))   # [M,4,2]
    return cor, dirv, perp


def _overlap_1d(ci, cj, ax):
    pi, pj = ci @ ax, cj @ ax
    return not (pi.max() < pj.min() or pj.max() < pi.min())


def collisions(xy, heading, length=L_VEH, width=W_VEH):
    """Indices of vehicles whose footprints overlap any other (SUMO-style)."""
    M = xy.shape[0]
    if M < 2:
        return set(), []
    cor, dirv, perp = _corners(xy, heading, length / 2.0, width / 2.0)
    dist = (xy.unsqueeze(0) - xy.unsqueeze(1)).norm(dim=2)
    cand = (dist < (length + width)) & torch.triu(torch.ones(M, M, dtype=torch.bool), 1)
    hit, pairs = set(), []
    for i, j in cand.nonzero().tolist():
        axes = (dirv[i], perp[i], dirv[j], perp[j])
        if all(_overlap_1d(cor[i], cor[j], ax) for ax in axes):
            hit.add(i); hit.add(j); pairs.append((i, j))
    return hit, pairs


# ── 2D constant-velocity rollout gate (replaces the 1D iterative correction) ────
H_2D      = 4.0    # s    rollout horizon
DT_2D     = 0.5    # s    rollout step
K_2D      = int(H_2D / DT_2D)
D_SAFE_2D = 6.5    # m    centre-distance threshold: circumscribed circles ≈ 5.3 m
                   #      + ~1.2 m for the ≈5.5 m/sample sweep between rollout samples
DSAFE_GATE_GAIN = 0.15 # m per s of δ_safe — the gate margin WIDENS with the policy time-gap
                   #      δ_safe, so the dsafe= knob tunes the gate too (only ever more
                   #      conservative; the 6.5 m physical floor is never undercut).
GATE_ITERS = 10    #      brake-and-re-roll passes
GATE_STEP  = 0.45  # m/s² brake increment per pass
GATE_BRAKE_MIN = -3.0  # m/s² the gate only steps DOWN while the current command is
                       #      above this; at/below −3 it stops deepening (comfort bound).
                       #      Kernel commands below −3 (e.g. a_brake=−4.5) pass through.
GATE_BRAKE_HARD = -utils.B_MAX  # m/s² deeper floor used ONLY to avoid an already-committed cross
                       #      vehicle (an imminent crash, not a comfort yield) — the gate may
                       #      brake to full −B_MAX to GUARANTEE separation between passers.
EPS_ENTRY = 0.5        # m    min room before the stop-line for a box-exclusivity halt to apply
                       #      (below this it's effectively at the line / committed → let it go).


def _rollout_xy(geo, mv, s_center):
    """(x,y) of vehicle CENTRES at future arc-lengths.  mv [M], s_center [M,K] → [M,K,2]."""
    M, K = s_center.shape
    xy = torch.zeros(M, K, 2)
    for mi in range(4):
        sel = (mv == mi).nonzero().flatten()
        if len(sel):
            pts, cum = geo[C._MOVES[mi]]
            flat, _ = _interp(pts, cum, s_center[sel].reshape(-1))
            xy[sel] = flat.reshape(len(sel), K, 2)
    return xy


JCT_PAST = 30.0   # m past the junction entry beyond which a vehicle has cleared the box
JCT_AHEAD = 30.0  # m before the junction entry: vehicles within this band are ALWAYS rolled
                  # (floor on the speed-dependent reach) so slow/stopped approachers start
                  # their conflict assessment early instead of getting excluded and stalling


def rollout_gate(a, s, v, mv, yield_mask, geo, s_junc, verbose=False, return_defer=False,
                 delta_safe=None, force_roll=None):
    """
    Final 2-D safety gate on the kernel command `a`.  SYMMETRIC: it both brakes
    yielders away from danger AND lets priority vehicles accelerate to clear, in each
    case only as far as the 2-D rollout stays safe.

    SEQUENTIAL / GREEDY (not a parallel one-shot).  Vehicles are corrected ONE AT A
    TIME in order of ETA to the junction (earliest first), and each vehicle rolls its
    rivals forward under their CURRENT believed command `a_corr` at CONSTANT
    ACCELERATION (v_τ = v + a·τ).  So a later vehicle sees the already-committed plans
    of everyone processed before it — including the leader it must yield to, which is
    always earlier in ETA and therefore already corrected.  This fixes the parallel
    scheme's blind spot, where every rival was frozen at constant velocity and no
    vehicle ever reacted to another's freshly-changed acceleration within the step.

    Roles from the per-pair yield mask (yield_mask[i,j] ⇒ i must yield to j):
      • YIELDER  (yields to someone):  brake −GATE_STEP while its rollout comes within
        D_SAFE_2D of a rival it must yield to — until clear or the −3 comfort floor.
      • LEADER   (someone yields to it, and it yields to no one):  accelerate
        +GATE_STEP, capped at a_max, WHILE its rollout stays clear of the vehicles that
        yield to it — asserting right-of-way clears the box faster and undoes the
        rho-gate's over-damping.  Skipped if the kernel is already braking it (a<0 ⇒ a
        closer in-lane leader / g<1 constraint the cross rollout doesn't see).

    Only vehicles in the JUNCTION WINDOW are processed — a cross conflict can only occur
    near the box.  The window, by arc-length distance to the junction entry
    d = s_junc − s, is −JCT_PAST ≤ d ≤ max(v·H_2D + D_SAFE_2D, JCT_AHEAD): from "already
    cleared the box" up to "reachable within the horizon", floored at JCT_AHEAD so slow
    approachers still count.

    Each vehicle is swept along its REAL path under constant-acceleration kinematics,
    so body length, curved junction paths, and box occupancy are all captured.

    return_defer: also return a bool mask of vehicles the gate FORCED to defer — a passer
    it had to brake against a committed crosser, or a vehicle halted by box-exclusivity.
    The caller (cosim memory) re-labels these as 'yield' and holds their re-estimation
    latch, so the gate's tiebreak persists instead of being re-fought every step.

    force_roll: optional [N] bool — PROTECTED passers (queue- or liveness-promoted)
    that must ALWAYS be processed even if nobody designates them in yield_mask.
    Without this, a promoted passer no one yields to is skipped entirely: it never
    enters the committed list, never reserves its axis, and crossing traffic plans
    around it blind.  Forced vehicles are treated as leaders (rolled, committed,
    axis-reserving) but are NOT accelerated past a braking kernel command (a < 0
    means an in-lane constraint the cross rollout can't see).
    """
    defer = torch.zeros(len(s), dtype=torch.bool)
    # δ_safe widens the gate's safety distance (only ever more conservative than the floor)
    ds      = utils.DELTA_SAFE if delta_safe is None else delta_safe
    d_safe  = D_SAFE_2D + DSAFE_GATE_GAIN * ds
    if yield_mask is None:
        return (a, defer) if return_defer else a

    # junction window: only vehicles that can reach/occupy the box within the horizon.
    # Reach is the farther of the speed-dependent horizon distance and a fixed JCT_AHEAD
    # band, so slow/stopped approachers near the box are still rolled (they'd otherwise be
    # excluded — v·H→0 — and stall waiting for an assessment that never runs).
    d_to_junc = s_junc[mv] - s                                  # [N] >0 approaching
    reach     = (v * H_2D + D_SAFE_2D).clamp(min=JCT_AHEAD)
    in_jct    = (d_to_junc <= reach) & (d_to_junc >= -JCT_PAST)

    is_yield = yield_mask.any(dim=1)                            # ego yields to someone
    is_lead  = yield_mask.any(dim=0) & (~is_yield)              # someone yields to ego (& not itself yielding)
    # DEADLOCK BREAKER: a vehicle whose front is past the first conflict point is INSIDE
    # the box (d_to_junc < 0).  It can no longer resolve a conflict by yielding — stopping
    # there blocks every crossing stream (the gridlock seen in the GUI).  So it is FORCED
    # to clear: removed from the yielder set and put in the accelerate role regardless of
    # its kernel command, so the gate drives it to a_max and out of the box instead of
    # braking it to a standstill.  Crossing rivals already treat a committed in-box vehicle
    # as priority (the kernel's point-of-no-return / forced_yield logic), so this is consistent.
    in_box   = (d_to_junc < 0.0) & in_jct
    yielders = is_yield & in_jct & (~in_box)
    leaders  = ((is_lead & (a >= 0.0)) | in_box) & in_jct
    forced   = (force_roll & in_jct & (~yielders)) if force_roll is not None \
               else torch.zeros(len(s), dtype=torch.bool)
    roll = (yielders | leaders | forced)
    if verbose:
        print(f"    gate: active={len(s)} in_junction={int(in_jct.sum())} "
              f"in_box={int(in_box.sum())} yielders={int(yielders.sum())} "
              f"leaders={int(leaders.sum())} corrected={int(roll.sum())}")
    if not bool(roll.any()):
        return (a, defer) if return_defer else a

    t = torch.arange(1, K_2D + 1) * DT_2D                       # [K]

    def _roll(idxs, a_belief):
        """Constant-ACCELERATION 2-D rollout of vehicles `idxs` under `a_belief`.
        v_τ = clamp(v + a·τ, 0, V_PHYS); arc-length integrates that profile.  [n,K,2]."""
        v_prof = (v[idxs].unsqueeze(1) + a_belief.unsqueeze(1) * t).clamp(0.0, V_PHYS)
        s_t    = (s[idxs] - L_VEH / 2.0).unsqueeze(1) + torch.cumsum(v_prof * DT_2D, dim=1)
        return _rollout_xy(geo, mv[idxs], s_t)                  # [n,K,2]

    # GREEDY ORDER: earliest arrival decides first.  d_to_junc<0 (in-box) → eta<0 → front
    # of the queue (highest priority to clear).  Each vehicle then plans around the
    # already-committed a_corr of everyone ahead of it in ETA.
    eta   = d_to_junc / v.clamp(min=utils.EPS)
    order = [i for i in torch.argsort(eta).tolist() if bool(roll[i])]

    axis_all = _AXIS[mv]                                        # travel axis per active vehicle
    a_corr = a.clone()
    committed: list = []                                        # egos already resolved this step (ETA order)
    # BOX EXCLUSIVITY: an axis is "claimed" while a vehicle occupies the box (or is past the
    # point of no return into it).  Seed with whoever is already inside; the ETA loop adds
    # claimers as they commit.  A crossing-axis vehicle that can STILL stop is halted at the
    # stop-line so two streams never co-occupy the box (the seed-3 low-speed pile-up).
    occupied = {int(axis_all[j]) for j in range(len(s)) if bool(in_box[j])}
    for i in order:
        is_y = bool(yielders[i])
        if bool(in_box[i]):
            # ALREADY IN THE BOX → it must CLEAR; ignore all cross rivals (the deadlock-breaker).
            # Two in-box crossers would otherwise brake against each other and freeze the box.
            # Box-exclusivity (below) prevents NEW dual-entry, so this stays safe.
            des_xy = com_xy = None
        else:
            # DESIGNATED rivals (role-based): a yielder watches who IT yields to (its row);
            # a leader/passer watches who yields to IT (its column).  Window-restricted, self out.
            rmask = (yield_mask[i] if is_y else yield_mask[:, i]) & in_jct
            rmask = rmask.clone(); rmask[i] = False
            rj = rmask.nonzero().flatten()
            des_xy = _roll(rj, a_corr[rj]) if len(rj) else None

            # COMMITTED CROSS rivals: every earlier-ETA ego already resolved this step on the
            # CROSS axis, carrying its just-committed plan.  This is the missing guarantee — two
            # PASSERS (empty designated set) would otherwise both accelerate into each other.  By
            # deferring to whoever committed first, the earlier clears and the later brakes.
            com = [j for j in committed if bool(axis_all[j] != axis_all[i])]
            com_xy = _roll(torch.tensor(com), a_corr[torch.tensor(com)]) if com else None

        ai = float(a_corr[i])
        ei = torch.tensor([i])
        com_braked = False                                      # forced to brake by a committed crosser?
        for _ in range(GATE_ITERS):
            ego_xy = _roll(ei, torch.tensor([ai]))              # [1,K,2]
            d_des = float((ego_xy - des_xy).norm(dim=-1).min()) if des_xy is not None else 1e9
            d_com = float((ego_xy - com_xy).norm(dim=-1).min()) if com_xy is not None else 1e9
            unsafe_des = d_des < d_safe
            unsafe_com = d_com < d_safe                          # conflict with a committed car
            # A committed conflict is an imminent crash → brake HARD (down to −B_MAX) to
            # GUARANTEE separation; a plain role-based yield only steps to the −3 comfort floor.
            floor = GATE_BRAKE_HARD if unsafe_com else GATE_BRAKE_MIN
            if unsafe_des or unsafe_com:
                if ai > floor:                                  # defer: slow this (later) vehicle
                    ai -= GATE_STEP
                    com_braked = com_braked or unsafe_com
                else:
                    break
            elif ((not is_y) and ai < utils.A_MAX
                  and not (bool(forced[i]) and float(a[i]) < 0.0)):
                # clear of all → speed it up to clear; EXCEPT a forced passer whose
                # kernel command is braking (in-lane constraint the rollout can't see)
                ai = min(ai + GATE_STEP, utils.A_MAX)
            else:
                break

        # BOX EXCLUSIVITY enforcement (after the rollout deferral).  An axis holds the box
        # while a vehicle occupies it; a crossing vehicle that can still AVOID entering (it is
        # stopped, or has room to brake before the line) is halted; otherwise the proceeding
        # vehicle reaching a FREE box claims its axis so the next crosser halts.
        ax_i     = int(axis_all[i])
        d_entry  = float(s_junc[mv[i]] - s[i]) - utils.STOP_OFFSET   # front → stop-line
        d_need   = float(v[i]) ** 2 / (2.0 * utils.B_MAX)
        can_hold = (float(v[i]) < 0.5) or (d_entry > d_need + EPS_ENTRY)   # can avoid entering
        cross_blocked = any(ax != ax_i for ax in occupied)
        if bool(in_box[i]):
            occupied.add(ax_i)                                  # in box → holds it while clearing
        elif cross_blocked and can_hold:
            # a crossing stream holds/has-reserved the box and this one can still avoid entering
            # → HALT.  Because the reservation is made EARLY (below), a faster crosser is told to
            #   stop while it still has the room — preventing two full-speed cars converging.
            a_stop = -(float(v[i]) ** 2) / (2.0 * max(d_entry, EPS_ENTRY))
            ai = max(min(ai, a_stop), -utils.B_MAX)
        elif not is_y:
            # proceeding (not yielding) and not blocked → RESERVE its axis now.  The loop is
            # ETA-ordered, so the earliest-arriving stream reserves first and every later
            # crossing vehicle this step sees it and halts in time (FCFS at the box by ETA).
            occupied.add(ax_i)

        # ONLY a passer the gate forced to brake against a committed crosser is "deferring" —
        # latch THAT (the two-passer tiebreak).  A box-exclusivity halt is NOT latched, so the
        # halted car keeps re-evaluating and goes the instant the box frees (no lane starvation).
        defer[i] = (not is_y) and com_braked

        a_corr[i] = ai
        committed.append(i)
    return (a_corr, defer) if return_defer else a_corr


# ── simulation ──────────────────────────────────────────────────────────────────
def gen_events(flow, seed):
    """Poisson arrival schedule: list of (depart_time, move_idx), sorted by time."""
    rng  = np.random.default_rng(seed)
    rate = flow / 3600.0
    events = []
    for mi in range(4):
        t = 0.0
        while True:
            t += rng.exponential(1.0 / rate)
            if t >= T_END:
                break
            events.append((t, mi))
    events.sort()
    return events


def simulate(flow=300, seed=0, geo=None, s_cp=None, path_len=None, s_junc=None,
             events=None, gate2d=True, dt=DT, verbose=False, mean_model=None):
    DT_ = dt
    if geo is None:
        geo, s_cp, path_len, s_junc = build_geometry()
    if events is None:
        events = gen_events(flow, seed)
    Ntot = len(events)
    depart = torch.tensor([e[0] for e in events])
    move   = torch.tensor([e[1] for e in events], dtype=torch.long)
    s      = torch.zeros(Ntot)
    v      = torch.zeros(Ntot)
    state  = torch.zeros(Ntot, dtype=torch.long)   # 0 pending, 1 active, 2 done
    prev_a = torch.zeros(Ntot)

    collided, arrived, coll_steps = set(), 0, 0
    pair_kinds: set = set()             # (i, j, 'rear'|'cross') distinct colliding pairs
    n_steps = int(T_END / DT_)

    for step in range(n_steps):
        t = step * DT_
        # spawn: per move, if entry clear, admit the earliest pending due vehicle.
        # New front spawns at s=L_VEH, so the nearest leader's REAR (min_s − L) must
        # clear the new front by SPAWN_GAP: min_s ≥ 2·L + gap.
        for mi in range(4):
            am = (state == 1) & (move == mi)
            min_s = float(s[am].min()) if am.any() else 1e9
            if min_s >= 2.0 * L_VEH + SPAWN_GAP:
                cand = ((state == 0) & (move == mi) & (depart <= t)).nonzero().flatten()
                if len(cand):
                    k = cand[depart[cand].argmin()]
                    # front arc-length starts at L (SUMO departPos="base": back at 0, front at L)
                    # safe insertion speed: able to brake behind the nearest leader
                    if am.any():
                        li_s   = s[am].argmin()
                        v_ld   = float(v[am][li_s])
                        gap_in = max(min_s - 2.0 * L_VEH, 0.0)
                        v_safe = (v_ld ** 2 + 2.0 * utils.B_MAX * gap_in) ** 0.5
                    else:
                        v_safe = V_PHYS
                    state[k] = 1; s[k] = L_VEH; v[k] = min(V_PHYS, v_safe)

        act = (state == 1).nonzero().flatten()
        Na = len(act)
        if Na == 0:
            continue
        s_a, v_a, mv_a = s[act], v[act], move[act]
        ax_a = _AXIS[mv_a]

        # in-lane leader (same movement, ahead)
        same = mv_a.unsqueeze(0) == mv_a.unsqueeze(1)
        ahead = s_a.unsqueeze(0) > s_a.unsqueeze(1)
        eye = torch.eye(Na, dtype=torch.bool)
        gap_ij = torch.where(same & ahead & ~eye, (s_a.unsqueeze(0) - s_a.unsqueeze(1)) - L_VEH,
                             torch.full((Na, Na), 1e9))
        gap, li = gap_ij.min(dim=1)
        has_lead = gap < 1e8
        v_lead = torch.where(has_lead, v_a[li], v_a)
        gap = torch.where(has_lead, gap.clamp(min=0.0), torch.full((Na,), 300.0))

        # cross rivals via per-pair conflict-point arc-lengths
        scp_e = s_cp[mv_a][:, mv_a]                       # [Na,Na] ego's CP arc-len for pair
        scp_r = s_cp.t()[mv_a][:, mv_a]                   # rival's CP arc-len for pair
        ego_d   = scp_e - s_a.unsqueeze(1)
        rival_d = scp_r - s_a.unsqueeze(0)
        rival_v = v_a.unsqueeze(0).expand(Na, Na)
        valid = ((ax_a.unsqueeze(1) != ax_a.unsqueeze(0)) & ~eye
                 & (ego_d > 0.0) & (rival_d > -CLEAR)
                 & ~torch.isnan(ego_d) & ~torch.isnan(rival_d))
        ego_d   = torch.nan_to_num(ego_d, nan=1e3)
        rival_d = torch.nan_to_num(rival_d, nan=-1e3)

        # platoon pressure: sum of speeds of same-lane vehicles still approaching
        appr = s_a < s_junc[mv_a]
        P = ((same & appr.unsqueeze(0)).float() * v_a.unsqueeze(0)).sum(dim=1)

        # transformer prior mean (optional): resolve roles in the state step (same
        # predecessor_gap call the controller would make — fed back as pred_override,
        # so behavior is IDENTICAL), build the context, hand over a mean_fn closure.
        mean_fn, pred_override = None, None
        if mean_model is not None:
            import mean_net
            prop = utils.predecessor_gap(
                ego_d, v_a, rival_d, rival_v, valid,
                delta_safe=utils.DELTA_SAFE, ego_P=P,
                rival_P=P.unsqueeze(0).expand(Na, Na))
            pred_override = prop
            roles = ["yield" if bool(h) else "pass" for h in prop[4]]
            behind_n = (same & appr.unsqueeze(0)
                        & (s_a.unsqueeze(0) < s_a.unsqueeze(1))).float().sum(dim=1)
            d_junc = s_junc[mv_a] - s_a
            ctx = mean_net.build_context(v_a, gap, v_lead, P, behind_n, d_junc,
                                         ego_d, rival_d, valid, roles)
            mean_fn = mean_model.make_mean_fn(mean_model.encode(*ctx))

        out = utils.controller_acceleration(
            torch.zeros(Na), gap + L_VEH, v_a, v_lead,
            d_conf=ego_d, rival_d=rival_d, rival_v=rival_v, rival_valid=valid,
            ego_pressure=P, rival_pressure=P.unsqueeze(0).expand(Na, Na),
            a_prev=prev_a[act], kappa=0.5, brake_exempt=True,
            return_roles=gate2d, pred_override=pred_override, mean_fn=mean_fn)
        if gate2d:
            a, yield_mask = out
            if verbose and step % 20 == 0:
                print(f"  t={t:5.1f}s  active={Na}")
            a = rollout_gate(a.detach(), s_a, v_a, mv_a, yield_mask, geo, s_junc,
                             verbose=verbose and step % 20 == 0)
        else:
            a = out.detach()

        v_new = (v_a + a * DT_).clamp(0.0, V_PHYS)
        prev_a[act] = a
        v[act] = v_new
        s[act] = s_a + v_new * DT_

        # collision check in real 2-D — footprints centred at the vehicle CENTRE
        # (s − L/2); interp(s) is the FRONT bumper, so we must shift back half a length.
        # Checked at the MID-step and post-step positions: at DT=0.5 a car sweeps up
        # to ~5.5 m (> body length), so a single end-of-step check could tunnel.
        s_new = s_a + v_new * DT_
        any_hit = False
        for s_chk in (s_a + v_new * (DT_ / 2.0), s_new):
            xy = torch.zeros(Na, 2); hd = torch.zeros(Na)
            for mi in range(4):
                m = C._MOVES[mi]
                sel = (mv_a == mi).nonzero().flatten()
                if len(sel):
                    xy[sel], hd[sel] = _interp(geo[m][0], geo[m][1], s_chk[sel] - L_VEH / 2.0)
            hit, pairs = collisions(xy, hd)
            if hit:
                any_hit = True
                collided.update(int(act[i]) for i in hit)
                for i, j in pairs:
                    kind = "rear" if int(mv_a[i]) == int(mv_a[j]) else "cross"
                    pair_kinds.add((int(act[i]), int(act[j]), kind))
        coll_steps += int(any_hit)

        # exit
        done = act[s[act] >= path_len[mv_a]]
        state[done] = 2
        arrived += len(done)

    ss = arrived / (T_END - 40.0) * 3600 if arrived else 0.0   # rough steady-state proxy
    n_rear  = sum(1 for *_, k in pair_kinds if k == "rear")
    n_cross = sum(1 for *_, k in pair_kinds if k == "cross")
    return dict(flow=flow, seed=seed, collided=len(collided), coll_steps=coll_steps,
                arrived=arrived, ss_rate=ss, n_rear=n_rear, n_cross=n_cross)


if __name__ == "__main__":
    flow = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    geo, s_cp, path_len, s_junc = build_geometry()
    print("path lengths:", [round(float(x), 1) for x in path_len])
    print("conflict-point arc-lengths s_cp[ego,rival]:")
    print(torch.round(s_cp * 10) / 10)
    print(f"\nrunning flow={flow} seed={seed} (junction-windowed gate, verbose)\n")
    t0 = time.time()
    r = simulate(flow, seed, geo, s_cp, path_len, s_junc, verbose=True)
    print(f"\n=== flow={flow} seed={seed} ===")
    print(f"  collided vehicles : {r['collided']}")
    print(f"  collision steps   : {r['coll_steps']}")
    print(f"  rear / cross pairs: {r['n_rear']} / {r['n_cross']}")
    print(f"  arrived           : {r['arrived']}")
    print(f"  steady-state rate : {r['ss_rate']:.0f} veh/h")
    print(f"  wall time         : {time.time()-t0:.2f}s")
