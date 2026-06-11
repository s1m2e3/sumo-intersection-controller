"""
role_memory.py — the stateful yield/pass role layer, shared by the SUMO co-sim and
the PyTorch training rollout.

This is the EXACT machinery cosim_sumo.run() used before the NN was added, extracted
verbatim so training and co-simulation resolve roles identically:

  1. PRIORITY MEMORY — predecessor_gap PROPOSES an instantaneous role; memory
     resolves the FINAL role: assigned only within ASSIGN_DIST of the box, LATCHED
     for role_hold s, a passer holds its latch while its platoon hasn't cleared,
     a stuck yielder (< STUCK_V for > STUCK_HOLD s) is forced to PASS.
  2. QUEUE ARBITER — a standing queue that has waited > QUEUE_WAIT s gets a
     protected clearing window: its front QUEUE_N_PASS vehicles are promoted to
     PASS for QUEUE_CLEAR s (only the longest-waiting lane opens, never into an
     occupied cross axis).
  3. PASSER COMPATIBILITY ARBITER — tentative passers are confirmed earliest-ETA
     first; the later of any crossing pair whose bumper-aware occupancy windows
     come within ARBITER_GAP s is demoted to a strict yielder, re-pointed at the
     passer it lost to.  Committed vehicles displace stoppable ones.
  4. LIVENESS — if every contesting vehicle ended up a yielder, the front-most
     contender is promoted to PASS so the box always drains.
  5. GATE FEEDBACK (gate_feedback) — a vehicle the 2-D rollout gate forced to
     defer is re-labelled 'yield' and its latch held, so the gate's tiebreak
     persists instead of being re-fought every step.

All state is keyed on caller-supplied stable vehicle ids (SUMO id strings in the
co-sim, global vehicle indices in the torch rollout).  Everything here is
ENVIRONMENT logic — discrete, gradient-free; the kernel/transformer consume the
resolved roles via pred_override.
"""
from __future__ import annotations

import torch

import utils
import cosim_sumo as C

_AX_OF_MOVE = [C.ORIGIN[m][0] for m in C._MOVES]      # movement idx -> travel axis


class RoleMemory:
    def __init__(self, dt):
        self.dt          = dt        # caller's sim step (timers count in seconds)
        self.role        = {}        # id -> 'yield' | 'pass' | 'none'  (latched)
        self.role_exp    = {}        # id -> sim-time the latch holds until
        self.stuck_time  = {}        # id -> accumulated s near/in box at v < STUCK_V
        self.queue_wait  = {}        # movement idx -> s its standing queue has waited
        self.queue_until = {}        # id -> sim-time a queue-promoted PASS is protected to

    def step(self, ids, vs, d_junc, in_box, behind_n, mv, ego_d, rival_d, valid,
             prop, t_now, role_hold=C.ROLE_HOLD):
        """Resolve the final per-vehicle roles for this step.

        ids      list[N]   stable vehicle keys
        vs       [N]       speeds
        d_junc   [N]       distance to the junction CENTRE (>0 approaching, <0 past)
        in_box   list[N]   True while the vehicle occupies the junction internal lane
        behind_n [N]       same-lane vehicles still farther back (platoon hold)
        mv       [N]       movement index per vehicle
        ego_d/rival_d [N,N], valid [N,N]   per-pair conflict tensors
        prop     the 6-tuple from utils.predecessor_gap
        Returns (pred_override, roles) — pred_override is the controller's 6-tuple,
        roles the resolved 'yield'/'pass'/'none' list (clamps, GUI, context).
        """
        N = len(ids)
        tau_c, eta_pred, ego_d_pred, v_pred, has_pred, is_pred = prop
        has_res = has_pred.clone()
        is_res  = is_pred.clone()
        tau_res = tau_c.clone()                                  # τ_c fed to the kernel per role

        # ── QUEUE ARBITER ────────────────────────────────────────────────────────
        queue_len = {}                                           # mi -> standing-queue size
        for mi in range(4):
            lane = ((mv == mi) & (d_junc > 0.0)).nonzero().flatten()
            if len(lane) == 0:
                self.queue_wait[mi] = 0.0; queue_len[mi] = 0; continue
            front = lane[torch.argsort(d_junc[lane])[0]]
            fdj   = float(d_junc[front])
            n_q = int(((mv == mi) & (d_junc > fdj) & (d_junc < fdj + C.QUEUE_SPAN)).sum())
            queue_len[mi] = n_q + 1                              # block incl. the front
            if n_q >= C.QUEUE_MIN_BEHIND and float(vs[front]) < C.STUCK_V:
                self.queue_wait[mi] = self.queue_wait.get(mi, 0.0) + self.dt
            else:
                # DECAY, don't hard-reset: a crawling front that blips over STUCK_V for a
                # step must not zero the accumulated wait of a standing queue (jitter
                # starvation); a genuinely dissolved queue still drains in a few seconds.
                self.queue_wait[mi] = max(0.0, self.queue_wait.get(mi, 0.0) - 2.0 * self.dt)
        # don't open a lane whose CROSS axis already has a vehicle in the box — UNLESS
        # the queue is STARVING (waited > QUEUE_STARVE): at sustained cross flow the box
        # is almost never empty at the sampled instant, so the empty-box condition alone
        # starves the queue forever.  Opening while occupied is SAFE: the rollout gate's
        # box exclusivity still halts the promoted fronts until the box physically frees —
        # the override only transfers right-of-way, not physical entry.
        occ_ax = {_AX_OF_MOVE[int(mv[j])] for j in range(N) if in_box[j]}
        ready = [mi for mi in range(4)
                 if (self.queue_wait.get(mi, 0.0) > C.QUEUE_WAIT
                     and not any(a != _AX_OF_MOVE[mi] for a in occ_ax))
                 or self.queue_wait.get(mi, 0.0) > C.QUEUE_STARVE]
        if ready:
            mi = max(ready, key=lambda m: self.queue_wait[m])    # longest-waiting opens
            lane = ((mv == mi) & (d_junc > 0.0)).nonzero().flatten()
            # liberation scales with the backlog: a massive queue opens a bigger window
            n_pass = min(max(C.QUEUE_N_PASS, queue_len.get(mi, 0) // 2), C.QUEUE_N_MAX)
            frontk = lane[torch.argsort(d_junc[lane])[:n_pass]]
            for rank, li in enumerate(frontk.tolist()):
                # staggered protection: the k-th vehicle needs k headways longer to cross
                self.queue_until[ids[li]] = t_now + C.QUEUE_CLEAR + rank * C.QUEUE_HEADWAY
            self.queue_wait[mi] = 0.0

        # ── PRIORITY MEMORY (latch / stuck promotion / assignment window) ────────
        roles = ['none'] * N
        for i, vco in enumerate(ids):
            dj, sp = float(d_junc[i]), float(vs[i])
            if (in_box[i] or dj <= C.ASSIGN_DIST) and sp < C.STUCK_V:
                self.stuck_time[vco] = self.stuck_time.get(vco, 0.0) + self.dt
            else:
                self.stuck_time[vco] = 0.0
            if t_now < self.queue_until.get(vco, -1.0):
                r = 'pass'; self.role[vco] = 'pass'             # protected queue passer
            else:
                r, exp = self.role.get(vco, 'none'), self.role_exp.get(vco, -1.0)
                latched = (t_now < exp) or (r == 'pass' and behind_n[i] > 0)
                if not latched:
                    if dj <= C.ASSIGN_DIST:
                        r = 'yield' if bool(has_pred[i]) else 'pass'
                        self.role[vco], self.role_exp[vco] = r, t_now + role_hold
                    else:
                        r, self.role[vco] = 'none', 'none'
                if r == 'yield' and self.stuck_time.get(vco, 0.0) > C.STUCK_HOLD:
                    r, self.role[vco], self.role_exp[vco] = 'pass', 'pass', t_now + role_hold
                    self.stuck_time[vco] = 0.0
            roles[i] = r
            if r == 'yield':
                has_res[i] = True
            else:
                has_res[i] = False
                is_res[i, :] = False
                tau_res[i]  = utils.TAU_C_MAX

        # ── PASSER COMPATIBILITY ARBITER ────────────────────────────────────────
        eta_pred = eta_pred.clone(); ego_d_pred = ego_d_pred.clone(); v_pred = v_pred.clone()
        CL = utils.CONFLICT_LEN
        axis_all = [_AX_OF_MOVE[int(mv[j])] for j in range(N)]
        committed_i = [bool(in_box[i]) or
                       (float(d_junc[i]) <= float(vs[i]) ** 2 / (2 * utils.B_MAX)
                        + utils.STOP_OFFSET)
                       for i in range(N)]

        def _conflict(p, q):
            if not (bool(valid[p, q]) or bool(valid[q, p])):
                return False
            vp, vq = max(float(vs[p]), 0.1), max(float(vs[q]), 0.1)
            dp, dq = float(ego_d[p, q]), float(rival_d[p, q])
            pin, pout = dp / vp, (dp + CL) / vp
            qin, qout = dq / vq, (dq + CL) / vq
            return max(pin - qout, qin - pout) < C.ARBITER_GAP

        def _demote(i, q):                                       # i yields to passer q
            vq = max(float(vs[q]), 0.1)
            roles[i] = 'yield'; self.role[ids[i]] = 'yield'
            self.role_exp[ids[i]] = t_now + role_hold
            has_res[i] = True
            ego_d_pred[i] = ego_d[i, q]
            eta_pred[i]   = (rival_d[i, q] + CL) / vq
            v_pred[i]     = vs[q]
            tau_res[i]    = max(float(ego_d[i, q]) / max(float(vs[i]), 0.1)
                                - float(eta_pred[i]), 0.0)
            is_res[i, :] = False; is_res[i, q] = True

        passers   = [i for i in range(N) if roles[i] == 'pass']
        protected = [i for i in passers if t_now < self.queue_until.get(ids[i], -1.0)]
        rest = sorted((i for i in passers if i not in protected),
                      key=lambda i: float(d_junc[i]) / max(float(vs[i]), 0.5))
        confirmed = list(protected)
        for p in rest:
            if in_box[p]:
                # a vehicle PHYSICALLY IN THE BOX is never demoted — not even by a
                # protected queue window.  Demoting it turns the yield clamp into a
                # freeze-in-the-box (the liberated queue then crosses into it).  It is
                # de-facto committed: confirm it and let the gate sequence the others.
                confirmed.append(p)
                continue
            q = next((c for c in confirmed if _conflict(p, c)), None)
            if q is None:
                confirmed.append(p)
            elif committed_i[p] and not committed_i[q] and q not in protected:
                _demote(q, p); confirmed.remove(q); confirmed.append(p)
            else:
                _demote(p, q)

        # ── LIVENESS ────────────────────────────────────────────────────────────
        contesting = [i for i in range(N)
                      if in_box[i] or 0.0 < float(d_junc[i]) <= C.ASSIGN_DIST]
        alive = any(roles[i] == 'pass' or committed_i[i] or float(vs[i]) > C.STUCK_V
                    for i in contesting)
        if contesting and not alive:
            best = min(contesting, key=lambda i: (0 if in_box[i] else 1, float(d_junc[i])))
            roles[best] = 'pass'; self.role[ids[best]] = 'pass'
            self.role_exp[ids[best]] = t_now + role_hold
            # PROTECTED like a queue promotion: the gate's defer feedback must not
            # re-latch the junction's only passer back to yielder the same step
            # (promote -> defer -> re-yield -> promote is the livelock cycle).
            self.queue_until[ids[best]] = t_now + C.QUEUE_CLEAR
            has_res[best] = False; is_res[best, :] = False
            tau_res[best] = utils.TAU_C_MAX

        pred_override = (tau_res, eta_pred, ego_d_pred, v_pred, has_res, is_res)
        return pred_override, roles

    def gate_feedback(self, ids, defer, t_now, role_hold=C.ROLE_HOLD):
        """A vehicle the rollout gate forced to defer is latched as a yielder
        (unless inside a protected queue window)."""
        for i, vco in enumerate(ids):
            if bool(defer[i]) and t_now >= self.queue_until.get(vco, -1.0):
                self.role[vco], self.role_exp[vco] = 'yield', t_now + role_hold

    def protected(self, vid, t_now):
        """True while vid is inside a protected queue-promotion window."""
        return t_now < self.queue_until.get(vid, -1.0)

    def ensure_passer(self, ids, roles, defer, in_box, d_junc, t_now,
                      role_hold=C.ROLE_HOLD):
        """POST-GATE liveness: never end a step with zero effective passers.

        The in-step liveness check runs BEFORE the gate, so the gate can still
        neutralize the junction's only passer (defer it against a committed
        crosser) and end the step passer-less — the observed gridlock.  Effective
        passer = role 'pass' and not gate-deferred, or any vehicle in the box
        (the gate forces those to clear).  If none remain among the contesting
        set, promote the front-most contender to a PROTECTED pass (queue window:
        defer feedback and re-estimation can't undo it) for the NEXT steps.
        Mutates `roles` in place; returns the promoted index or None."""
        N = len(ids)
        contesting = [i for i in range(N)
                      if in_box[i] or 0.0 < float(d_junc[i]) <= C.ASSIGN_DIST]
        if not contesting:
            return None
        if any((roles[i] == 'pass' and not bool(defer[i])) or in_box[i]
               for i in contesting):
            return None
        best = min(contesting, key=lambda i: (0 if in_box[i] else 1, float(d_junc[i])))
        vid = ids[best]
        roles[best] = 'pass'
        self.role[vid] = 'pass'
        self.role_exp[vid] = t_now + role_hold
        self.queue_until[vid] = t_now + C.QUEUE_CLEAR
        return best
