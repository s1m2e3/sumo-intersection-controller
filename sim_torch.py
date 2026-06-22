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
from typing import NamedTuple
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


_PAD_CACHE = {}


def pad_geometry(geo, order=None):
    """Pack the movement polylines into padded tensors PTS [M,Pmax,2], CUM [M,Pmax]
    (tail-padded with the last point / arc-length), so _interp_batch can gather a
    vehicle's polyline by movement index and interpolate ALL movements in one shot
    instead of a per-movement Python loop.  Cached by (geo identity, order).

    `order` is the ordered list of geo keys whose row i becomes movement index i — so a
    vehicle's movement index `mv` must index into this same order.  Defaults to the 4
    straight movements (C._MOVES) for backward compatibility; pass list(range(M)) for the
    12-movement turn geometry keyed by integer movement idx."""
    order = C._MOVES if order is None else list(order)
    key = (id(geo), tuple(order))
    if key in _PAD_CACHE:
        return _PAD_CACHE[key]
    polys = [geo[m] for m in order]
    Pmax  = max(p[0].shape[0] for p in polys)
    PTS = torch.zeros(len(polys), Pmax, 2)
    CUM = torch.zeros(len(polys), Pmax)
    for i, (pts, cum) in enumerate(polys):
        P = pts.shape[0]
        PTS[i, :P] = pts; PTS[i, P:] = pts[-1]
        CUM[i, :P] = cum; CUM[i, P:] = cum[-1]
    _PAD_CACHE[key] = (PTS, CUM)
    return PTS, CUM


def _interp_batch(PTS, CUM, mv, s):
    """Batched _interp: per-vehicle movement mv [M] and arc-length s [M] → xy [M,2],
    heading [M].  Gathers each vehicle's padded polyline and interpolates at once;
    numerically identical to looping _interp per movement on xy (tail-pad → degenerate
    final segments that resolve to the last point at s = path length).  Differentiable
    in s through the segment fraction t, exactly like _interp."""
    pts = PTS[mv]                                            # [M, P, 2]
    cum = CUM[mv]                                            # [M, P]
    s   = torch.minimum(s.clamp(min=0.0), cum[:, -1])
    idx = (torch.searchsorted(cum, s.unsqueeze(-1), right=True).squeeze(-1) - 1
           ).clamp(0, cum.shape[1] - 2)
    s0  = torch.gather(cum, 1, idx.unsqueeze(-1)).squeeze(-1)
    s1  = torch.gather(cum, 1, (idx + 1).unsqueeze(-1)).squeeze(-1)
    t   = ((s - s0) / (s1 - s0).clamp(min=1e-6)).unsqueeze(-1)
    gi  = idx.view(-1, 1, 1).expand(-1, 1, 2)
    p0  = torch.gather(pts, 1, gi).squeeze(1)
    p1  = torch.gather(pts, 1, gi + 1).squeeze(1)
    xy  = p0 + t * (p1 - p0)
    d   = p1 - p0
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
H_2D      = 10.0   # s    rollout horizon (extended from 4 s so cross conflicts are seen
                   #      ~2.5× earlier — anticipatory rather than reactive-at-the-box)
DT_2D     = 0.5    # s    rollout step (held → K_2D = 20 steps)
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
BOX_RESERVE_T = 2.0    # s    a proceeding vehicle only RESERVES the box (excludes crossing
                       #      streams) once it is within this TIME-to-entry — so the reservation
                       #      tracks imminent occupancy instead of being claimed many seconds out.
                       #      The earliest-ETA vehicle of a conflict still reserves first and the
                       #      farther crosser (which has the room) is still halted, so safety is
                       #      preserved; bounding the hold-time is what frees throughput for the
                       #      12-movement case (a fast vehicle no longer blocks the box ~4 s early).
                       #      Large value ⇒ original "reserve anywhere in the window" behavior.

# ── RKHS-smoothed hinge-gradient polish (runs AFTER rollout_gate) ─────────────────
# A differentiable refinement that directly descends the SAME 2-D hinge the cosim
# probe reports — Σ relu(d_safe − ‖c_i−c_j‖)² over ALL cross-axis pairs near the box,
# integrated over the constant-accel rollout.  Unlike the discrete GATE_STEP search
# (one rival at a time, ±0.45 m/s²), the gradient couples every conflicting pair at
# once and is smoothed through the controller's own ARD Gram K^φ (the "RKHS gradient":
# Δa = −η · K^φ ∇_a Ĵ), so vehicles with similar features (g, τ_c, r) move together.
HINGE_GATE   = True    #      master toggle for the polish pass
N_HINGE_STEPS = 3      #      projected-gradient steps per call (kept small for 0.1-s budget)
HINGE_ETA    = 2.0     #      step size η (the L2 gradient magnitudes are O(1), so η~O(1))
HINGE_DA_MAX = 3.0     # m/s² per-step clamp on |Δa| (lets the polish brake past the gate's
                       #      −3 comfort floor toward −B_MAX when a conflict is imminent)
HINGE_RKHS   = False   #      L2 gradient (K^φ = I).  The K^φ-smoothed "RKHS" step couples
                       #      vehicles with similar (g,τ_c,r) and so AVERAGES AWAY the
                       #      anti-correlated corrections a conflict needs (one yields, one
                       #      proceeds) — measured to break descent.  For this hinge the L2
                       #      gradient is the same object without that harmful smoothing.
HINGE_RIDGE  = 0.0     #      Tikhonov ridge λ added to K^φ before the step (only if RKHS)
_HINGE_DBG   = None    #      set to a list to collect (J_first, J_last) per polish call (debug)


def _rollout_xy(geo, mv, s_center, order=None):
    """(x,y) of vehicle CENTRES at future arc-lengths.  mv [M], s_center [M,K] → [M,K,2].
    Vectorized: one batched interpolation over all M·K (vehicle, horizon) samples,
    each carrying its vehicle's movement geometry — no per-movement loop.  `order` is the
    geo-key order that mv indexes into (see pad_geometry)."""
    M, K = s_center.shape
    PTS, CUM = pad_geometry(geo, order)
    mv_k = mv.unsqueeze(1).expand(M, K).reshape(-1)         # [M*K] movement per sample
    xy, _ = _interp_batch(PTS, CUM, mv_k, s_center.reshape(-1))
    return xy.reshape(M, K, 2)


JCT_PAST = 30.0   # m past the junction entry beyond which a vehicle has cleared the box
JCT_AHEAD = 30.0  # m before the junction entry: vehicles within this band are ALWAYS rolled
                  # (floor on the speed-dependent reach) so slow/stopped approachers start
                  # their conflict assessment early instead of getting excluded and stalling


def rollout_gate(a, s, v, mv, yield_mask, geo, s_junc, verbose=False, return_defer=False,
                 delta_safe=None, force_roll=None, conf=None, geo_order=None,
                 box_exclusive=True):
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

    # CROSS-CONFLICT predicate between two movements.  The straight gate used the entry
    # AXIS (E/W vs N/S) as a 2-bucket proxy for "paths cross"; with the 12-movement turn
    # geometry that is too coarse (a left turn crosses oncoming through on the SAME axis,
    # and two compatible turns on crossing axes don't actually conflict).  When `conf` (a
    # [M,M] bool, paths-cross) is supplied, box-exclusivity and the committed-crosser test
    # key on the TRUE per-pair crossing instead.  Without it, fall back to the axis test —
    # byte-identical to the original straight behavior (and reduces to the same thing, since
    # for straights "different axis" ⇔ "paths cross").
    if conf is not None:
        def _cross(ma, mb):
            return bool(conf[ma, mb])
    else:
        def _cross(ma, mb):
            return bool(_AXIS[ma] != _AXIS[mb])

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
        return _rollout_xy(geo, mv[idxs], s_t, geo_order)      # [n,K,2]

    # GREEDY ORDER: earliest arrival decides first.  d_to_junc<0 (in-box) → eta<0 → front
    # of the queue (highest priority to clear).  Each vehicle then plans around the
    # already-committed a_corr of everyone ahead of it in ETA.
    eta   = d_to_junc / v.clamp(min=utils.EPS)
    order = [i for i in torch.argsort(eta).tolist() if bool(roll[i])]

    a_corr = a.clone()
    committed: list = []                                        # egos already resolved this step (ETA order)
    # BOX EXCLUSIVITY: an axis is "claimed" while a vehicle occupies the box (or is past the
    # point of no return into it).  Seed with whoever is already inside; the ETA loop adds
    # claimers as they commit.  A crossing-axis vehicle that can STILL stop is halted at the
    # stop-line so two streams never co-occupy the box (the seed-3 low-speed pile-up).
    occupied = {int(mv[j]) for j in range(len(s)) if bool(in_box[j])}
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
            com = [j for j in committed if _cross(int(mv[j]), int(mv[i]))]
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
        mv_i     = int(mv[i])
        d_entry  = float(s_junc[mv[i]] - s[i]) - utils.STOP_OFFSET   # front → stop-line
        d_need   = float(v[i]) ** 2 / (2.0 * utils.B_MAX)
        can_hold = (float(v[i]) < 0.5) or (d_entry > d_need + EPS_ENTRY)   # can avoid entering
        cross_blocked = box_exclusive and any(_cross(occ, mv_i) for occ in occupied)
        if bool(in_box[i]):
            occupied.add(mv_i)                                  # in box → holds it while clearing
        elif cross_blocked and can_hold:
            # a crossing stream holds/has-reserved the box and this one can still avoid entering
            # → HALT.  Because the reservation is made EARLY (below), a faster crosser is told to
            #   stop while it still has the room — preventing two full-speed cars converging.
            a_stop = -(float(v[i]) ** 2) / (2.0 * max(d_entry, EPS_ENTRY))
            ai = max(min(ai, a_stop), -utils.B_MAX)
        elif (not is_y) and (d_entry / max(float(v[i]), EPS_ENTRY) < BOX_RESERVE_T):
            # proceeding (not yielding), not blocked, and within BOX_RESERVE_T of entry →
            # RESERVE its movement now.  The loop is ETA-ordered, so the earliest-arriving
            # stream reserves first and every later conflicting vehicle this step sees it and
            # halts in time (FCFS at the box by ETA).  Gating on time-to-entry keeps the
            # reservation tight to actual occupancy instead of claimed seconds out.
            occupied.add(mv_i)

        # ONLY a passer the gate forced to brake against a committed crosser is "deferring" —
        # latch THAT (the two-passer tiebreak).  A box-exclusivity halt is NOT latched, so the
        # halted car keeps re-evaluating and goes the instant the box frees (no lane starvation).
        defer[i] = (not is_y) and com_braked

        a_corr[i] = ai
        committed.append(i)
    return (a_corr, defer) if return_defer else a_corr


def hinge_gradient_gate(a, feat, s, v, mv, geo, s_junc, delta_safe=None,
                        n_steps=None, eta=None, da_max=None, conf=None, geo_order=None):
    """RKHS-smoothed projected-gradient polish that DIRECTLY minimizes the predicted
    2-D hinge, run as a refinement AFTER rollout_gate (role logic / box-exclusivity
    already applied to `a`).

    Predicted hinge over the constant-accel rollout (the same kinematics the gate uses):

        v_i(τ) = clamp(v_i + a_i τ, 0, V_PHYS),  c_i(τ) = γ_i(s_i − L/2 + ∫₀^τ v_i)
        Ĵ(a)   = Σ_k Δτ  Σ_{(i,j)∈C} [ ( d_safe − ‖c_i(τ_k) − c_j(τ_k)‖ )₊ ]²

    C = ALL cross-axis pairs in the junction window (the relu zeroes the rest, so every
    pair that WOULD come within d_safe contributes — the multi-rival coupling).  The
    gradient ∇_a Ĵ is obtained by autograd through the rollout; the step is smoothed by
    the controller's ARD Gram K^φ over the live features (the functional gradient in the
    kernel's RKHS):

        a ← Π_[−B_MAX, A_MAX] ( a − η · K^φ ∇_a Ĵ ),   K^φ_{ij} = k(φ_i, φ_j).

    Returns a NEW command vector (same shape as `a`); vehicles outside the window are
    untouched.  Differentiable end-to-end (the per-step grad is taken on a detached
    leaf, but the op set is autograd-friendly, so the whole pass can also be embedded in
    a training loss later)."""
    # resolve hyperparameters from module globals AT CALL TIME (so they stay tunable,
    # e.g. from a sweep that sets sim_torch.HINGE_ETA) — not frozen as default args.
    n_steps = N_HINGE_STEPS if n_steps is None else n_steps
    eta     = HINGE_ETA     if eta     is None else eta
    da_max  = HINGE_DA_MAX  if da_max  is None else da_max
    ds      = utils.DELTA_SAFE if delta_safe is None else delta_safe
    d_safe  = D_SAFE_2D + DSAFE_GATE_GAIN * ds
    d_to_junc = s_junc[mv] - s
    reach   = (v * H_2D + D_SAFE_2D).clamp(min=JCT_AHEAD)
    near    = (d_to_junc <= reach) & (d_to_junc >= -JCT_PAST)
    idx     = near.nonzero().flatten()
    n = len(idx)
    if n < 2:
        return a

    # conflicting pairs in the window: true per-pair crossing when `conf` is supplied (turn
    # geometry), else the straight axis proxy (different entry axis ⇔ paths cross).
    mvw = mv[idx]
    if conf is not None:
        cross = conf[mvw][:, mvw]
    else:
        ax = _AXIS[mvw]
        cross = ax.unsqueeze(0) != ax.unsqueeze(1)
    pairmask = cross & torch.triu(torch.ones(n, n, dtype=torch.bool), 1)  # each pair once
    if not bool(pairmask.any()):
        return a

    s_i, v_i, mv_i = s[idx], v[idx], mv[idx]
    t   = torch.arange(1, K_2D + 1, dtype=a.dtype) * DT_2D      # [K] rollout times
    if HINGE_RKHS:
        ls   = torch.tensor(utils.LENGTHSCALES, dtype=feat.dtype)
        Kphi = utils._kernel_gram(feat[idx], ls)               # [n,n] controller's ARD Gram
        if HINGE_RIDGE:
            Kphi = Kphi + HINGE_RIDGE * torch.eye(n, dtype=Kphi.dtype)
    else:
        Kphi = torch.eye(n, dtype=a.dtype)                     # plain per-command step (I)
    pm   = pairmask.unsqueeze(-1)                              # [n,n,1]

    a_set = a[idx].clone()
    a_in  = a[idx].clone()
    J_first = J_last = None
    _dbg_first: dict = {}                                      # per-call mechanism probe
    for _ in range(n_steps):
        a_leaf = a_set.detach().requires_grad_(True)
        v_prof = (v_i.unsqueeze(1) + a_leaf.unsqueeze(1) * t).clamp(0.0, V_PHYS)   # [n,K]
        s_t    = (s_i - L_VEH / 2.0).unsqueeze(1) + torch.cumsum(v_prof * DT_2D, dim=1)  # [n,K]
        xy     = _rollout_xy(geo, mv_i, s_t, geo_order)        # [n,K,2]
        d      = (xy.unsqueeze(1) - xy.unsqueeze(0)).norm(dim=-1)   # [n,n,K]
        J      = (torch.relu(d_safe - d).pow(2) * pm).sum() * DT_2D
        if J_first is None:
            J_first = float(J)
        J_last = float(J)
        if float(J) == 0.0:
            break
        grad, = torch.autograd.grad(J, a_leaf)                 # [n] Euclidean per-command grad
        if _HINGE_DBG is not None and _dbg_first.get("done") is None:
            # mechanism probe on the FIRST acting iteration: who violates, are they
            # moving (β≠0), and is the gradient actually nonzero on them?
            viol = ((d < d_safe) & pm)                         # [n,n,K]
            inv  = viol.any(-1).any(-1) | viol.any(-1).any(0)  # [n] vehicles in a violation
            at_lim = (a_in <= -utils.B_MAX + 1e-3) | (a_in >= utils.A_MAX - 1e-3)
            _dbg_first.update(done=True, gmax=float(grad.abs().max()),
                              n_grad=int((grad.abs() > 1e-6).sum()),
                              n_viol=int(inv.sum()),
                              viol_vmin=float(v_i[inv].min()) if bool(inv.any()) else -1.0,
                              viol_vmax=float(v_i[inv].max()) if bool(inv.any()) else -1.0,
                              viol_gmax=float(grad[inv].abs().max()) if bool(inv.any()) else 0.0,
                              viol_atlim=int((inv & at_lim).sum()),
                              viol_ainmin=float(a_in[inv].min()) if bool(inv.any()) else 0.0,
                              viol_ainmax=float(a_in[inv].max()) if bool(inv.any()) else 0.0)
        step  = (-eta * (Kphi @ grad)).clamp(-da_max, da_max)  # RKHS-smoothed, magnitude-capped
        a_set = (a_set + step).clamp(-utils.B_MAX, utils.A_MAX)
    if _HINGE_DBG is not None and J_first:
        da = (a_set - a_in)                                    # net change this call
        rec = dict(J_first=J_first, J_last=J_last, n=n,
                   n_accel=int((da > 1e-4).sum()), n_brake=int((da < -1e-4).sum()),
                   max_accel=float(da.clamp(min=0).max()),
                   max_brake=float((-da).clamp(min=0).max()))
        rec.update({k: v for k, v in _dbg_first.items() if k != "done"})
        _HINGE_DBG.append(rec)
        _dbg_first.clear()

    out = a.clone()
    out[idx] = a_set
    return out


def _arrival_time(d, v, a, vmax=V_PHYS, eps=1e-4):
    """Accel-aware time to traverse arc-length d ≥ 0 under v(τ)=clamp(v+aτ, 0, vmax) —
    the SAME constant-accel kinematics the 2-D rollout integrates, solved in closed form.

    Returns (t, reaches).  reaches=False ⇒ a<0 brakes the vehicle to a halt before
    covering d (it never arrives → not a conflict; caller masks it out).  Branches:
      • cruise  (v≥vmax, a≥0):            t = d / vmax
      • caps out (reaches vmax mid-way):  t = (vmax−v)/a + (d − d_acc)/vmax
      • otherwise:                        t = 2d / (v + √(v²+2ad))   (= d/v as a→0)
    The rationalized last form is stable at a→0 (no 0/0), matching the user's d/v exactly
    when acceleration is zero, and is exact for any constant a in the unclamped regime."""
    d = d.clamp(min=0.0); v = v.clamp(min=0.0)
    disc    = v * v + 2.0 * a * d
    reaches = disc > eps                                   # speed stays > 0 over [0, d]
    sq      = torch.sqrt(disc.clamp(min=eps))              # speed at the crossing (unclamped)
    t_uncl  = 2.0 * d / (v + sq).clamp(min=eps)
    a_pos   = a.clamp(min=eps)
    d_acc   = ((vmax * vmax - v * v) / (2.0 * a_pos)).clamp(min=0.0)   # dist to hit vmax
    t_two   = (vmax - v).clamp(min=0.0) / a_pos + (d - d_acc).clamp(min=0.0) / vmax
    over    = (sq > vmax) & (a > eps)
    cruise  = (v >= vmax) & (a >= 0.0)
    t = torch.where(cruise, d / vmax, torch.where(over, t_two, t_uncl))
    return t, reaches


def time_hinge_gradient_gate(a, feat, s, v, mv, s_cp, s_junc, delta_safe=None,
                             n_steps=None, eta=None, da_max=None):
    """TIME-domain analogue of hinge_gradient_gate.  Rather than penalizing 2-D spatial
    proximity over a rollout, it penalizes the per-pair CONFLICT-TIME-GAP falling below
    δ_safe — directly, in seconds, in the same units as the policy's δ_safe.

        η_i  = accel-aware time for i to reach ITS specific (i,j) crossing point
               (arc-length s_cp[mv_i, mv_j] − s_i, curvature-accurate — the via/internal
               lane shape is baked into s_cp, NO straight-line approximation)
        τ_c(i,j) = |η_i − η_j|
        Ĵ(a) = Σ_{(i,j)∈C} [ ( δ_safe − τ_c(i,j) )₊ ]²

    C = pairs whose paths genuinely cross (s_cp finite both ways), both still APPROACHING
    their crossing (d > 0), and both actually REACHING it (neither brakes to a stop first
    — a vehicle that stops short has yielded and is no conflict).  Projected RKHS-smoothed
    gradient descent, identical machinery / hyperparameters to hinge_gradient_gate:

        a ← Π_[−B_MAX, A_MAX]( a − η · K^φ ∇_a Ĵ ).

    s_cp is the [M,M] per-pair conflict-point arc-length from build_geometry (s_cp[i,j] =
    arc-length along movement i's path to its true geometric crossing with movement j)."""
    n_steps = N_HINGE_STEPS if n_steps is None else n_steps
    eta     = HINGE_ETA     if eta     is None else eta
    da_max  = HINGE_DA_MAX  if da_max  is None else da_max
    ds      = utils.DELTA_SAFE if delta_safe is None else delta_safe
    d_to_junc = s_junc[mv] - s
    reach   = (v * H_2D + D_SAFE_2D).clamp(min=JCT_AHEAD)
    near    = (d_to_junc <= reach) & (d_to_junc >= -JCT_PAST)
    idx     = near.nonzero().flatten()
    n = len(idx)
    if n < 2:
        return a

    mvw, s_w, v_w = mv[idx], s[idx], v[idx]
    scp_e = s_cp[mvw][:, mvw]                       # [n,n] ego i's arc-len to (i,j) crossing
    scp_r = s_cp.t()[mvw][:, mvw]                   # [n,n] rival j's arc-len to same crossing
    finite = ~torch.isnan(scp_e) & ~torch.isnan(scp_r)         # paths actually cross
    d_i = scp_e - s_w.unsqueeze(1)                  # ego remaining distance, per pair
    d_j = scp_r - s_w.unsqueeze(0)                  # rival remaining distance, per pair
    base = (finite & (d_i > 0.0) & (d_j > 0.0)
            & torch.triu(torch.ones(n, n, dtype=torch.bool), 1))   # each crossing pair once
    if not bool(base.any()):
        return a
    di = torch.nan_to_num(d_i, nan=0.0); dj = torch.nan_to_num(d_j, nan=0.0)

    if HINGE_RKHS:
        ls   = torch.tensor(utils.LENGTHSCALES, dtype=feat.dtype)
        Kphi = utils._kernel_gram(feat[idx], ls)
        if HINGE_RIDGE:
            Kphi = Kphi + HINGE_RIDGE * torch.eye(n, dtype=Kphi.dtype)
    else:
        Kphi = torch.eye(n, dtype=a.dtype)

    a_set = a[idx].clone()
    for _ in range(n_steps):
        a_leaf = a_set.detach().requires_grad_(True)
        ti, ri = _arrival_time(di, v_w.unsqueeze(1), a_leaf.unsqueeze(1))   # ego (rows)
        tj, rj = _arrival_time(dj, v_w.unsqueeze(0), a_leaf.unsqueeze(0))   # rival (cols)
        tau_c  = (ti - tj).abs()                                   # [n,n] per-pair time gap
        cm     = (base & ri & rj).to(a.dtype)                      # detached conflict gate
        J      = (torch.relu(ds - tau_c).pow(2) * cm).sum()
        if float(J) == 0.0:
            break
        grad, = torch.autograd.grad(J, a_leaf)
        step  = (-eta * (Kphi @ grad)).clamp(-da_max, da_max)
        a_set = (a_set + step).clamp(-utils.B_MAX, utils.A_MAX)

    out = a.clone()
    out[idx] = a_set
    return out


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


def build_turn_geometry(net_path=NET_PATH):
    """12-movement TURN geometry for the PyTorch sim — the SAME geometry run_turns feeds
    to SUMO (turns_geom.gate_geometry).  Returns geo, s_cp, path_len, s_junc, conf,
    geo_order; pass conf/geo_order into simulate() to switch it into turn mode."""
    import turns_geom as G
    geo, s_cp, s_junc, CONF = G.gate_geometry(net_path)
    M = s_cp.shape[0]
    path_len = torch.tensor([float(geo[i][1][-1]) for i in range(M)])
    return geo, s_cp, path_len, s_junc, CONF, list(range(M))


def gen_turn_events(vph, seed, t_end=None):
    """Poisson schedule over the 12 turn movements at per-direction demand
    vph={'l','s','r'} (vph/approach) — the PyTorch analogue of run_turns.write_turn_routes.
    A direction with demand 0 emits no vehicles (so vph={'l':0,'s':N,'r':0} is straight)."""
    import turns_geom as G
    vph    = vph or {"l": 100.0, "s": 300.0, "r": 100.0}
    t_end  = t_end or T_END
    rng    = np.random.default_rng(seed)
    events = []
    for m in G.movements(NET_PATH):
        rate = float(vph[m.dir]) / 3600.0
        if rate <= 0.0:
            continue
        t = 0.0
        while True:
            t += rng.exponential(1.0 / rate)
            if t >= t_end:
                break
            events.append((t, m.idx))
    events.sort()
    return events


def proxy_hinge_gate(a, s, v, mv, s_cp, s_junc, delta_safe=None, max_iters=200, step=0.1):
    """GRADIENT-FREE proxy for the time hinge.  The exact FGD direction is unnecessary:
    for any violating pair, accelerating the earlier vehicle and braking the later one is
    ALWAYS a descent direction (η_lead↓, η_foll↑ ⇒ τ_c↑), so we apply a fixed ±`step`
    in that direction and use the hinge only as a feasibility / continue-stop oracle.

    GREEDY, LEADER-CENTRED, ITERATE-TO-FEASIBILITY (mirrors rollout_gate):
      each iteration —
        1. recompute every crossing pair's accel-aware τ_c (curvature-accurate s_cp);
           if no pair violates τ_c < δ_safe → STOP (feasible)
        2. LEADER = the earliest-ARRIVING vehicle among those in a violating pair
        3. for each of the leader's violating conflicts, BRAKE whichever vehicle arrives
           LATER at that crossing (−step).  That brakes the follower when the leader leads,
           and brakes the LEADER ITSELF on a conflict where it is actually the later arriver
           (a vehicle has two crossings — EW & WE — so the earliest-overall can still follow
           at one of them; braking it there fixes that violation).
        4. ACCELERATE the leader (+step) iff it is the earlier arriver in ALL its conflicts
           and is neither LOCKED nor already braking (a<0 ⇒ an in-lane constraint the cross
           view can't see).
        5. then RE-SORT (next iteration recomputes τ_c and the leader)
      LOCK: any vehicle once braked may only ever decelerate further — never be accelerated.
      Braking is monotone-good (grows every gap) and terminal, acceleration is bounded by
      A_MAX, so the process cannot livelock: it terminates at feasibility, or when every
      vehicle involved is clamped (A_MAX / −B_MAX) and nothing can move.

    No autograd, no kernel — pure functional."""
    ds = utils.DELTA_SAFE if delta_safe is None else delta_safe
    d_to_junc = s_junc[mv] - s
    reach = (v * H_2D + D_SAFE_2D).clamp(min=JCT_AHEAD)
    near  = (d_to_junc <= reach) & (d_to_junc >= -JCT_PAST)
    idx   = near.nonzero().flatten()
    n = len(idx)
    if n < 2:
        return a
    mvw, s_w, v_w = mv[idx], s[idx], v[idx]
    scp_e  = s_cp[mvw][:, mvw]                       # [n,n] ego i's arc-len to (i,j) crossing
    scp_r  = s_cp.t()[mvw][:, mvw]                   # [n,n] rival j's arc-len to same crossing
    finite = ~torch.isnan(scp_e) & ~torch.isnan(scp_r)
    di = torch.nan_to_num(scp_e - s_w.unsqueeze(1), nan=-1.0)   # ego remaining, per pair
    dj = torch.nan_to_num(scp_r - s_w.unsqueeze(0), nan=-1.0)   # rival remaining, per pair

    a_set  = a[idx].clone()
    braked = torch.zeros(n, dtype=torch.bool)        # LOCK: True ⇒ decel-only henceforth
    BIG = 1e9
    for _ in range(max_iters):
        # per-pair accel-aware arrival times with the CURRENT command
        ti, _ri = _arrival_time(di, v_w.unsqueeze(1), a_set.unsqueeze(1))   # [n,n] vehicle i → (i,j)
        tj, _rj = _arrival_time(dj, v_w.unsqueeze(0), a_set.unsqueeze(0))   # [n,n] vehicle j → (i,j)
        reach_i = di > 0.0; reach_j = dj > 0.0       # still approaching its crossing
        valid = finite & reach_i & reach_j & _ri & _rj
        valid.fill_diagonal_(False)
        tau  = (ti - tj).abs()
        viol = valid & (tau < ds)                    # [n,n] symmetric violating pairs
        if not bool(viol.any()):
            break                                    # FEASIBLE → done
        # leader ranking key: each vehicle's soonest arrival among its violating conflicts
        veh_arr = torch.where(viol, ti, torch.full_like(ti, BIG)).min(dim=1).values   # [n]
        order = torch.argsort(veh_arr).tolist()      # earliest-arriving violator first

        acted = False
        for L in order:
            js = viol[L].nonzero().flatten().tolist()
            if not js:
                continue
            # per conflict: brake whichever vehicle arrives LATER (could be L itself)
            laters = {(L if float(ti[L, j]) > float(tj[L, j]) else j) for j in js}
            leads_all = L not in laters
            can_accel = (leads_all and not bool(braked[L])
                         and float(a_set[L]) >= 0.0 and float(a_set[L]) < utils.A_MAX - 1e-9)
            can_brake = any(float(a_set[w]) > -utils.B_MAX + 1e-9 for w in laters)
            if not (can_accel or can_brake):
                continue                             # this leader fully clamped → try next
            if can_accel:
                a_set[L] = min(float(a_set[L]) + step, utils.A_MAX)
            for w in laters:                         # brake the later arriver(s); lock them
                a_set[w] = max(float(a_set[w]) - step, -utils.B_MAX)
                braked[w] = True
            acted = True
            break                                    # one leader, then RE-SORT
        if not acted:                                # nobody can move → infeasible/clamped
            break

    out = a.clone()
    out[idx] = a_set
    return out


# ── UNIFIED PER-STEP CONTROL ────────────────────────────────────────────────────
# The single control step shared by sim_torch.simulate, run_turns and cosim_sumo so the
# three harnesses can never drift: kernel command → 2-D role gate → hinge proxy polish.
# Movement-GENERAL — pass `conf` (true per-pair crossing) and `geo_order` for the
# 12-movement turn geometry; omit them for the straight axis-proxy (4 movements).
class ControlOut(NamedTuple):
    a:          torch.Tensor          # final command (post gate + proxy), detached
    yield_mask: object                # [N,N] per-pair yield mask, or None (no gate)
    feat:       object                # [N,4] controller query features, or None
    defer:      object                # [N] gate-defer mask, or None
    a_raw:      torch.Tensor          # raw kernel command BEFORE the gate (for f-harvest)
    a_gate:     torch.Tensor          # command AFTER gate, BEFORE proxy (for gate_log)


def control_step(xe, xl, v, v_lead, ego_d, rival_d, rival_v, valid, P, a_prev,
                 s, mv, geo, s_junc, *, s_cp=None, conf=None, geo_order=None,
                 mean_fn=None, pred_override=None, force_roll=None,
                 use_gate=True, role_gate=True, box_exclusive=True,
                 proxy="fgd", delta_safe=None, proxy_delta_safe=None,
                 controller_kwargs=None, verbose=False):
    """One per-step control pass.  Returns a ControlOut.

      proxy ∈ {None, 'fgd', 'time', 'grad-free'} selects the hinge polish AFTER the gate
        ('fgd' = RKHS functional-gradient on the spatial hinge, the default; 'time' = the
        time-domain hinge; 'grad-free' = the iterative proxy).  None ⇒ no polish.
      controller_kwargs : extra kwargs forwarded to utils.controller_acceleration
        (e.g. kappa, brake_exempt, brake_floor, promote, prio_ego/prio_rival) — this is
        where the harnesses' small differences live, NOT in separate control code paths.
    """
    ck = dict(controller_kwargs or {})
    N = valid.shape[0]
    Pmat = P.unsqueeze(0).expand(N, N) if P.dim() == 1 else P
    out = utils.controller_acceleration(
        xe, xl, v, v_lead, d_conf=ego_d, rival_d=rival_d, rival_v=rival_v,
        rival_valid=valid, ego_pressure=P, rival_pressure=Pmat, a_prev=a_prev,
        pred_override=pred_override, mean_fn=mean_fn,
        return_roles=use_gate, return_feat=use_gate, **ck)
    if not use_gate:
        a = out.detach()
        return ControlOut(a, None, None, None, a, a)

    a, yield_mask, feat = out
    a_raw = a.detach().clone()
    defer = None
    if role_gate:
        a, defer = rollout_gate(a.detach(), s, v, mv, yield_mask, geo, s_junc,
                                verbose=verbose, return_defer=True, delta_safe=delta_safe,
                                force_roll=force_roll, conf=conf, geo_order=geo_order,
                                box_exclusive=box_exclusive)
        a = a.detach()
    a_gate = a.detach().clone()                      # post-gate, pre-proxy (gate_log probe)
    if proxy == "fgd":
        a = hinge_gradient_gate(a.detach(), feat.detach(), s, v, mv, geo, s_junc,
                                delta_safe=proxy_delta_safe, conf=conf,
                                geo_order=geo_order).detach()
    elif proxy == "time":
        a = time_hinge_gradient_gate(a.detach(), feat.detach(), s, v, mv, s_cp, s_junc,
                                     delta_safe=proxy_delta_safe).detach()
    elif proxy == "grad-free":
        a = proxy_hinge_gate(a.detach(), s, v, mv, s_cp, s_junc,
                             delta_safe=proxy_delta_safe).detach()
    return ControlOut(a, yield_mask, feat, defer, a_raw, a_gate)


def simulate(flow=300, seed=0, geo=None, s_cp=None, path_len=None, s_junc=None,
             events=None, gate2d=True, dt=DT, verbose=False, mean_model=None,
             hinge_gate=HINGE_GATE, hinge_time=False, hinge_proxy=False,
             role_gate=True, proxy_delta_safe=None, f_log=None, gate_log=None,
             conf=None, geo_order=None, vph=None):
    DT_ = dt
    if geo is None:
        geo, s_cp, path_len, s_junc = build_geometry()
    # turn-mode iff a CONF mask is supplied (12-movement geometry); else straight 4-mv.
    M = int(s_junc.shape[0])
    is_turn = conf is not None
    force_roll = None                  # sim_torch has no promotion latch (cosim/run_turns do)
    # per-movement travel axis (0=EW,1=NS) and sign (±1) — needed for the cross test
    # (straight) and always for build_context.  Turn mode reads the 12-movement table.
    if is_turn:
        import turns_geom as G
        _mv = G.movements(NET_PATH)
        AX_FULL = torch.tensor([m.axis for m in _mv])
        SG_FULL = torch.tensor([float(m.sgn) for m in _mv])
    else:
        AX_FULL = _AXIS
        SG_FULL = torch.tensor([float(C.ORIGIN[m][1]) for m in C._MOVES])
    # padded polylines for batched footprint interpolation (movement-general; straight
    # geo is name-keyed via C._MOVES, turn geo is index-keyed via geo_order=range(M))
    PTS, CUM = pad_geometry(geo, geo_order)
    if events is None:
        events = (gen_turn_events(vph, seed) if is_turn else gen_events(flow, seed))
    Ntot = len(events)
    depart = torch.tensor([e[0] for e in events])
    move   = torch.tensor([e[1] for e in events], dtype=torch.long)
    s      = torch.zeros(Ntot)
    v      = torch.zeros(Ntot)
    state  = torch.zeros(Ntot, dtype=torch.long)   # 0 pending, 1 active, 2 done
    prev_a = torch.zeros(Ntot)

    collided, arrived, coll_steps = set(), 0, 0
    pair_kinds: set = set()             # (i, j, 'rear'|'cross') distinct colliding pairs
    hinge_total = 0.0                   # realized 2-D hinge (same metric as the cosim probe)
    n_steps = int(T_END / DT_)

    for step in range(n_steps):
        t = step * DT_
        # spawn: per move, if entry clear, admit the earliest pending due vehicle.
        # New front spawns at s=L_VEH, so the nearest leader's REAR (min_s − L) must
        # clear the new front by SPAWN_GAP: min_s ≥ 2·L + gap.
        for mi in range(M):
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
        ax_a, sgn_a = AX_FULL[mv_a], SG_FULL[mv_a]

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
        # cross test: TRUE per-pair crossing from CONF in turn mode; straight axis-proxy else
        cross_ij = conf[mv_a][:, mv_a] if is_turn else (ax_a.unsqueeze(1) != ax_a.unsqueeze(0))
        valid = (cross_ij & ~eye & (ego_d > 0.0) & (rival_d > -CLEAR)
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
                                         ego_d, rival_d, valid, roles, ax_a, sgn_a)
            mean_fn = mean_model.make_mean_fn(mean_model.encode(*ctx))

        # select the post-gate hinge polish from the diagnostic flags (default 'fgd')
        proxy_sel = (("grad-free" if hinge_proxy else "time" if hinge_time else "fgd")
                     if hinge_gate else None)
        if verbose and step % 20 == 0 and gate2d:
            print(f"  t={t:5.1f}s  active={Na}")
        co = control_step(
            torch.zeros(Na), gap + L_VEH, v_a, v_lead, ego_d, rival_d, rival_v, valid,
            P, prev_a[act], s_a, mv_a, geo, s_junc, s_cp=s_cp, conf=conf, geo_order=geo_order,
            mean_fn=mean_fn, pred_override=pred_override, force_roll=force_roll,
            use_gate=gate2d, role_gate=role_gate, proxy=proxy_sel,
            proxy_delta_safe=proxy_delta_safe,
            controller_kwargs=dict(kappa=0.5, brake_exempt=True),
            verbose=verbose and step % 20 == 0)
        a, yield_mask, feat = co.a, co.yield_mask, co.feat
        # harvest (g, τ_c, r, f, Δa) at the ACTUAL operating φ* for the f-contour. f = raw
        # learned prior mean; Δa = its NET effect on the PRE-GATE command (with−without mean).
        if gate2d and f_log is not None and mean_fn is not None:
            with torch.no_grad():
                fvals = mean_fn(feat)                            # [Na] learned mean at φ*
                a_zero = utils.controller_acceleration(
                    torch.zeros(Na), gap + L_VEH, v_a, v_lead,
                    d_conf=ego_d, rival_d=rival_d, rival_v=rival_v, rival_valid=valid,
                    ego_pressure=P, rival_pressure=P.unsqueeze(0).expand(Na, Na),
                    a_prev=prev_a[act], kappa=0.5, brake_exempt=True,
                    pred_override=pred_override, mean_fn=None)    # zero-mean command
                da = co.a_raw - a_zero                           # f's net effect on accel
            f_log.append(torch.cat([feat.detach(), fvals.detach().unsqueeze(-1),
                                    da.unsqueeze(-1), a_zero.unsqueeze(-1)], dim=-1))
        # PROXY probe (plot_gate): the polish's own Δa by role/distance
        if gate2d and proxy_sel == "grad-free" and gate_log is not None:
            is_y = yield_mask.any(dim=1)
            d_tj = s_junc[mv_a] - s_a
            for i in range(Na):
                if -30.0 < float(d_tj[i]) < 60.0:
                    gate_log.append((float(d_tj[i]), "yield" if bool(is_y[i]) else "pass",
                                     float(co.a_gate[i]), float(a[i])))

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
            xy, hd = _interp_batch(PTS, CUM, mv_a, s_chk - L_VEH / 2.0)
            hit, pairs = collisions(xy, hd)
            if hit:
                any_hit = True
                collided.update(int(act[i]) for i in hit)
                for i, j in pairs:
                    kind = "rear" if int(mv_a[i]) == int(mv_a[j]) else "cross"
                    pair_kinds.add((int(act[i]), int(act[j]), kind))
        coll_steps += int(any_hit)

        # realized 2-D hinge probe (identical metric to cosim_sumo / run_sweep): cross-axis
        # CENTRE pairs near the box, Σ relu(D_SAFE_2D − d)² · dt.  `xy` here holds the centres
        # at s_new (last s_chk), so it is reused directly.  Plain 6.5 m floor (NOT widened).
        d_to_j = s_junc[mv_a] - s_new
        nh = ((d_to_j > -JCT_PAST) & (d_to_j < 60.0)).nonzero().flatten()
        if len(nh) >= 2:
            dd = (xy[nh].unsqueeze(0) - xy[nh].unsqueeze(1)).norm(dim=-1)
            crossh_ij = (conf[mv_a[nh]][:, mv_a[nh]] if is_turn
                         else (ax_a[nh].unsqueeze(0) != ax_a[nh].unsqueeze(1)))
            crossh = crossh_ij & torch.triu(
                torch.ones(len(nh), len(nh), dtype=torch.bool), 1)
            hinge_total += float((torch.relu(D_SAFE_2D - dd[crossh]).pow(2)).sum()) * DT_

        # exit
        done = act[s[act] >= path_len[mv_a]]
        state[done] = 2
        arrived += len(done)

    ss = arrived / (T_END - 40.0) * 3600 if arrived else 0.0   # rough steady-state proxy
    n_rear  = sum(1 for *_, k in pair_kinds if k == "rear")
    n_cross = sum(1 for *_, k in pair_kinds if k == "cross")
    return dict(flow=flow, seed=seed, collided=len(collided), coll_steps=coll_steps,
                arrived=arrived, ss_rate=ss, n_rear=n_rear, n_cross=n_cross,
                hinge=hinge_total)


if __name__ == "__main__":
    toks = sys.argv[1:]
    hinge_time  = "time" in toks                      # TIME-domain hinge (exact FGD)
    hinge_proxy = "proxy" in toks                     # gradient-free proxy of the time hinge
    sweep       = "sweep" in toks                      # compare spatial / time / proxy over seeds
    nums = [t for t in toks if t.lstrip("-").isdigit()]
    flow = int(nums[0]) if len(nums) > 0 else 300
    seed = int(nums[1]) if len(nums) > 1 else 0
    geo, s_cp, path_len, s_junc = build_geometry()
    print("path lengths:", [round(float(x), 1) for x in path_len])
    print("conflict-point arc-lengths s_cp[ego,rival]:")
    print(torch.round(s_cp * 10) / 10)
    if sweep:
        print(f"\nspatial / TIME / PROXY hinge, flow={flow}, seeds 0-4, H_2D={H_2D}s, δ_safe={utils.DELTA_SAFE}s")
        print("  (coll = colliding vehicles, X = cross-collision pairs, arr = arrived)\n")
        print(f"  {'seed':>4} | {'spatial  coll/X/arr':>20} | {'time  coll/X/arr':>18} | {'proxy  coll/X/arr':>18}")
        for sd in range(5):
            rs = simulate(flow, sd, geo, s_cp, path_len, s_junc, hinge_time=False)
            rt = simulate(flow, sd, geo, s_cp, path_len, s_junc, hinge_time=True)
            rp = simulate(flow, sd, geo, s_cp, path_len, s_junc, hinge_proxy=True)
            f = lambda r: f"{r['collided']}/{r['n_cross']}/{r['arrived']}"
            print(f"  {sd:>4} | {f(rs):>20} | {f(rt):>18} | {f(rp):>18}")
    else:
        mode = "PROXY" if hinge_proxy else ("TIME-domain" if hinge_time else "spatial")
        print(f"\nrunning flow={flow} seed={seed} ({mode} hinge, H_2D={H_2D}s, verbose)\n")
        t0 = time.time()
        r = simulate(flow, seed, geo, s_cp, path_len, s_junc,
                     hinge_time=hinge_time, hinge_proxy=hinge_proxy, verbose=True)
        print(f"\n=== flow={flow} seed={seed}  [{mode} hinge] ===")
        print(f"  collided vehicles : {r['collided']}")
        print(f"  collision steps   : {r['coll_steps']}")
        print(f"  rear / cross pairs: {r['n_rear']} / {r['n_cross']}")
        print(f"  arrived           : {r['arrived']}")
        print(f"  steady-state rate : {r['ss_rate']:.0f} veh/h")
        print(f"  wall time         : {time.time()-t0:.2f}s")
