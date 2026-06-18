"""
train_mean.py — train the transformer prior mean to maximize the velocity integral.

Objective:

    J(θ) = ∫₀ᵀ Σᵢ vᵢ(t) dt                  total vehicle-meters (throughput surrogate)

    L = −J̄  +  λ_s · Σ relu(d_hinge − d_ij)²  +  λ_c · mean(a_GP²)

  J̄ is J normalized by (N_total · mean path length) — the average fraction of its
  route the population completes.  The safety hinge runs over CROSS-axis pairs only;
  the effort term keeps accelerations small unless velocity is actually bought.

Training scheme (one AVERAGED optimizer step per WINDOW_S seconds, batched):
  1. INITIAL CONDITIONS come from SUMO: for each (flow, seed) we run headless SUMO
     once and record every vehicle's ACTUAL realized departure time and movement
     (insertion queueing included) — cached to IC_CACHE so SUMO runs only once.
  2. Each epoch samples a batch of scenarios and rolls them in LOCKSTEP windows of
     WINDOW_S sim-seconds: every window, each episode advances WINDOW_S, its window
     loss backwards (gradients ACCUMULATE across the batch), then the gradients are
     divided by the batch size, clipped, and ONE optimizer step is taken — ~11
     averaged updates per epoch instead of 1, for the same simulation compute.
  3. Later windows of an episode run under the just-updated parameters (standard
     truncated-BPTT-with-updates); every step is still averaged over the WHOLE
     batch, so the network never fits a single seed.

Gradient path:
  • differentiable rollout with clone-before-write state updates;
  • the FULL cosim role machinery runs per step (role_memory.RoleMemory: latch,
    queue arbiter, passer-compatibility arbiter, liveness, gate-defer feedback)
    plus cosim's clamps (yielder a≤0 unless in-box, passer V_MIN_PASS floor) — so
    training and co-simulation make the SAME inference; all of it is environment
    (no gradient);
  • rollout gate applied STRAIGHT-THROUGH (trajectory feels it, gradient skips it);
  • the update window IS the truncated-BPTT window (state detached at each step);
  • the first GRAD_WARMUP seconds simulate identically but build no graph/loss.

Only θ (the MeanTransformer) trains — anchors, K⁻¹, lengthscales, gate untouched.

usage:
    python train_mean.py [epochs] [batch] [lr]      # checkpoints → mean_net_ckpt.pt
    python train_mean.py [epochs] [batch] [lr] fresh   # ignore the checkpoint, start from zero-init

If mean_net_ckpt.pt exists it is LOADED and training RESUMES from it (runs build on
each other); pass `fresh` to start from the zero-initialized model instead.
"""
from __future__ import annotations
import os, sys, time, random
from contextlib import nullcontext

import torch
import torch.nn.functional as F_t

import utils
import mean_net
import sim_torch as S
import cosim_sumo as C

DT       = C.DT                 # 0.1 s — MATCHES the SUMO co-sim step, so role timers
                                #  (hold=0.2 s latch, from C.ROLE_HOLD), the Angle-2 lag
                                #  constant, and the gate cadence are identical train/eval
T_END    = S.T_END
V_PHYS   = S.V_PHYS
L_VEH    = S.L_VEH

WINDOW_S = 10.0                 # s — one batched, AVERAGED optimizer step per window;
                                #  this is also the truncated-BPTT span (10 s comfortably
                                #  covers the gate horizon 4 s and δ_safe 5 s)
WINDOW_K = int(WINDOW_S / DT)   # steps per window
GRAD_WARMUP = 10.0              # s — simulate the first seconds WITHOUT building the
                                #  autograd graph (no loss, no backward): the network
                                #  starts empty and vehicles need ~18 s to reach the
                                #  junction, so the early gradient is ~0 (anchors
                                #  dominate on the approach) — pure compute savings;
                                #  the trajectory itself is identical either way
LAM_S    = 3.0                  # safety-hinge weight.  The hinge is the SMOOTH
                                #  saturating h²/(1+h) (h = relu(d_hinge − d)): quadratic
                                #  for shallow intrusions, ~LINEAR for deep ones, so its
                                #  gradient is BOUNDED (→1 per pair as d→0) by shape —
                                #  no spike, nothing for the clip to cut.  At h=6.5 the
                                #  penalty is ~7.5× smaller than the old h², so λ_s is
                                #  scaled up to keep a crash-level episode costing about
                                #  its entire velocity integral.
LAM_V    = 0.0                  # throughput REMOVED from the objective — f is trained on safety
                                #  (hinge) + acceleration (effort/jerk) ONLY; the base kernel
                                #  still carries throughput, so f focuses purely on smooth safety
LAM_C    = 0.0                  # effort (a²) weight — DISABLED: train f on the hinge ALONE so
LAM_J    = 0.0                  # jerk weight — DISABLED.  No acceleration penalty competing with
                                #  safety, so f is free to reduce conflicts without an effort cost
                                #  suppressing it (still logged for monitoring; f stays tanh-bounded
                                #  to ±F_LIM and the gate shields safety regardless)
D_HINGE  = S.D_SAFE_2D          # hinge fires below the gate's own safety distance
GRAD_CLIP = 2.0                 # clip the batch-averaged gradient to norm 2: with the
                                #  saturating hinge the tail is already bounded by shape,
                                #  so this binds rarely — but when a window does produce
                                #  an outsized gradient, it enters Adam at norm ≤ 2
CKPT      = mean_net.ckpt_path("best")   # BEST model by combined score (arch-tagged)
LAST_CKPT = mean_net.ckpt_path("last")   # LATEST model — saved EVERY epoch (arch-tagged)
LOG_CSV   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "train_log.csv")
LOG_COLS  = ["epoch", "mean_J", "mean_hinge", "score", "arrived", "eff_a2", "jerk_da2",
             "f_abs", "f_conflict", "f_free", "f_max", "dtheta", "updates"]

# mid-cell probe points (g, τ_c, r) — far from anchors, where f has authority; the
# mean |f| over these is printed per epoch to show whether the function is moving
F_PROBE  = torch.tensor([[0.6, 2.5, 0.0], [0.6, 2.5, 1.0],
                         [1.6, 1.2, 0.0], [1.6, 1.2, 1.0],
                         [2.7, 6.2, 1.0]])

# movement index → travel axis (0=x/EW, 1=y/NS) and direction sign (±1), for the
# conflicting-approach densities build_context needs
AX_MV = torch.tensor([C.ORIGIN[m][0] for m in C._MOVES])
SG_MV = torch.tensor([C.ORIGIN[m][1] for m in C._MOVES])

# ── initial-condition pool: realized SUMO departures, collected once ────────────
IC_FLOWS = (500, 600, 700)      # train across the saturation range (was 500 only) so f
                                # generalizes to the high-demand densities it under-served
IC_SEEDS = tuple(range(16))     # 16 seeds → 16 scenarios in the pool
IC_CACHE = f"_ic_pool_{'_'.join(map(str, IC_FLOWS))}.pt"   # flow-tagged ⇒ no stale reuse


def collect_sumo_initial_conditions(flows=IC_FLOWS, seeds=IC_SEEDS):
    """Run headless SUMO per (flow, seed) and record each vehicle's ACTUAL
    departure (t, move_idx) — the realized insertion schedule, including SUMO's
    entry queueing.  Cached to IC_CACHE; delete the file to re-collect."""
    if os.path.exists(IC_CACHE):
        pool = torch.load(IC_CACHE, weights_only=True)
        print(f"loaded {len(pool)} cached SUMO initial conditions from {IC_CACHE}")
        return pool
    import traci, sumolib
    mv_id = {m: i for i, m in enumerate(C._MOVES)}
    sumo = sumolib.checkBinary("sumo")
    pool = []
    print(f"collecting initial conditions from SUMO: {len(flows)} flows × {len(seeds)} seeds")
    for flow in flows:
        routes = os.path.join(C.HERE, f"_ic_routes_{flow}.rou.xml")
        C.write_routes(flow, routes)
        for seed in seeds:
            traci.start([sumo, "-n", os.path.join(C.HERE, "intersection.net.xml"),
                         "-r", routes, "--begin", "0", "--end", str(C.T_END),
                         "--step-length", str(C.DT), "--seed", str(seed),
                         "--no-step-log", "true", "--no-warnings", "true"])
            events = []
            for step in range(int(C.T_END / C.DT)):
                traci.simulationStep()
                for vid in traci.simulation.getDepartedIDList():
                    mi = mv_id.get(traci.vehicle.getRoute(vid)[0], 0)
                    events.append((step * C.DT, mi))
            traci.close()
            pool.append(dict(flow=flow, seed=seed, events=events))
            print(f"  flow={flow} seed={seed:2d}: {len(events)} departures")
    torch.save(pool, IC_CACHE)
    print(f"cached {len(pool)} scenarios → {IC_CACHE}")
    return pool


def junction_geometry(geo, net_path=S.NET_PATH):
    """Cosim-faithful per-movement landmarks for the torch rollout:
      s_center [4]  arc-length of the junction CENTRE (cosim's d_junc reference)
      s_box_in/out [4]  internal-lane span — front inside ⇔ SUMO road id ':' (in_box)
    """
    import sumolib
    net = sumolib.net.readNet(net_path, withInternal=True)
    s_center = torch.tensor([S._project(geo[m][0], geo[m][1],
                                        torch.tensor([200.0, 200.0]))
                             for m in C._MOVES])
    s_in, s_out = [], []
    for m, to in C._MOVE_TO.items():
        conn = net.getEdge(m).getConnections(net.getEdge(to))[0]
        def _plen(shape):
            pts = torch.tensor(shape, dtype=torch.float32)
            return float((pts[1:] - pts[:-1]).norm(dim=1).sum()) if len(shape) > 1 else 0.0
        L_from = _plen(conn.getFromLane().getShape())
        L_via  = (_plen(net.getLane(conn.getViaLaneID()).getShape())
                  if conn.getViaLaneID() else 0.0)
        s_in.append(L_from); s_out.append(L_from + L_via)
    return s_center, torch.tensor(s_in), torch.tensor(s_out)


class EpisodeStepper:
    """One episode's differentiable rollout, advanced one WINDOW at a time so the
    caller can batch optimizer updates across episodes every WINDOW_S sim-seconds.
    Physics, role machinery, gate, and clamps are exactly the per-step pipeline
    cosim runs — only the stepping is externalized."""

    def __init__(self, events, geo, s_cp, path_len, s_junc, jgeo):
        from role_memory import RoleMemory   # lazy (role_memory imports cosim_sumo)
        self.geo, self.s_cp = geo, s_cp
        self.path_len, self.s_junc = path_len, s_junc
        self.s_center, self.s_box_in, self.s_box_out = jgeo
        self.Ntot   = len(events)
        self.depart = torch.tensor([e[0] for e in events])
        self.move   = torch.tensor([e[1] for e in events], dtype=torch.long)
        self.s      = torch.zeros(self.Ntot)
        self.v      = torch.zeros(self.Ntot)
        self.state  = torch.zeros(self.Ntot, dtype=torch.long)
        self.prev_a = torch.zeros(self.Ntot)
        self.plm    = float(path_len.float().mean())
        self.role_mem = RoleMemory(dt=DT)    # per-episode stateful role layer (env)
        self.step_i  = 0
        self.n_steps = int(T_END / DT)
        # episode accumulators (floats, reporting only)
        self.loss_f = self.j_sum_f = self.hinge_f = self.eff_f = 0.0
        self.eff_n = self.arrived = 0
        self.jerk_f = 0.0; self.jerk_n = 0          # mean (Δa)² over vehicles active 2 steps running
        self.was_active = torch.zeros(self.Ntot, dtype=torch.bool)   # active last step? (skip fresh spawns)
        self.f_abs = 0.0; self.f_n = 0      # mean |f| at the mid-cell probe points
        # |f| at the ACTUAL operating φ*, split by whether the ego has a cross rival —
        # answers "is f ~0 everywhere, or ~0 in free-flow but ACTIVE when a conflict exists?"
        self.fc_sum = 0.0; self.fc_n = 0    # Σ|f| over conflict vehicle-steps (valid rival)
        self.ff_sum = 0.0; self.ff_n = 0    # Σ|f| over free-flow vehicle-steps
        self.f_max  = 0.0                   # running max |f| at an actual operating point

    def done(self):
        return self.step_i >= self.n_steps

    def j_norm(self):
        return self.j_sum_f / (self.Ntot * self.plm)

    def detach(self):
        """Truncate BPTT at the window boundary."""
        self.s, self.v = self.s.detach(), self.v.detach()
        self.prev_a = self.prev_a.detach()

    def advance(self, model, k=WINDOW_K, train=True):
        """Advance up to k steps.  Returns (window loss TENSOR — graph attached
        unless the window was all warm-up/empty — window J contribution, window
        hinge integral).  The caller owns backward / averaging / detach."""
        loss_win = torch.zeros(())
        w_j = w_hinge = 0.0
        end = min(self.step_i + k, self.n_steps)
        while self.step_i < end:
            step = self.step_i
            self.step_i += 1
            t = step * DT

            # ── spawn (constant writes — no gradient) ──────────────────────────
            self.s = self.s.clone(); self.v = self.v.clone()
            for mi in range(4):
                am = (self.state == 1) & (self.move == mi)
                min_s = float(self.s[am].min()) if am.any() else 1e9
                if min_s >= 2.0 * L_VEH + S.SPAWN_GAP:
                    cand = ((self.state == 0) & (self.move == mi)
                            & (self.depart <= t)).nonzero().flatten()
                    if len(cand):
                        kk = cand[self.depart[cand].argmin()]
                        if am.any():
                            v_ld   = float(self.v[am][self.s[am].argmin()])
                            gap_in = max(min_s - 2.0 * L_VEH, 0.0)
                            v_safe = (v_ld ** 2 + 2.0 * utils.B_MAX * gap_in) ** 0.5
                        else:
                            v_safe = V_PHYS
                        self.state[kk] = 1
                        self.s[kk] = L_VEH; self.v[kk] = min(V_PHYS, v_safe)

            act = (self.state == 1).nonzero().flatten()
            Na = len(act)
            if Na == 0:
                continue
            s_a, v_a, mv_a = self.s[act], self.v[act], self.move[act]
            ax_a = S._AXIS[mv_a]

            # warm-up: identical simulation, but NO autograd graph and NO loss
            warm = train and (t < GRAD_WARMUP)
            with (torch.no_grad() if warm else nullcontext()):
                # ── state estimation (mirrors sim_torch.simulate) ──────────────
                same  = mv_a.unsqueeze(0) == mv_a.unsqueeze(1)
                ahead = s_a.unsqueeze(0) > s_a.unsqueeze(1)
                eye   = torch.eye(Na, dtype=torch.bool)
                gap_ij = torch.where(same & ahead & ~eye,
                                     (s_a.unsqueeze(0) - s_a.unsqueeze(1)) - L_VEH,
                                     torch.full((Na, Na), 1e9))
                gap, li  = gap_ij.min(dim=1)
                has_lead = gap < 1e8
                v_lead = torch.where(has_lead, v_a[li], v_a)
                gap    = torch.where(has_lead, gap.clamp(min=0.0),
                                     torch.full((Na,), 300.0))

                scp_e = self.s_cp[mv_a][:, mv_a]
                scp_r = self.s_cp.t()[mv_a][:, mv_a]
                ego_d   = scp_e - s_a.unsqueeze(1)
                rival_d = scp_r - s_a.unsqueeze(0)
                rival_v = v_a.unsqueeze(0).expand(Na, Na)
                valid = ((ax_a.unsqueeze(1) != ax_a.unsqueeze(0)) & ~eye
                         & (ego_d > 0.0) & (rival_d > -S.CLEAR)
                         & ~torch.isnan(ego_d) & ~torch.isnan(rival_d))
                ego_d   = torch.nan_to_num(ego_d, nan=1e3)
                rival_d = torch.nan_to_num(rival_d, nan=-1e3)

                # cosim-faithful state quantities (centre d_junc, ':'-lane in_box,
                # COUNT pressure in the 0-120 m window, behind_n by d_junc order)
                d_junc = self.s_center[mv_a] - s_a               # >0 approaching
                in_box = ((s_a > self.s_box_in[mv_a])
                          & (s_a <= self.s_box_out[mv_a])).tolist()
                appr    = (d_junc > 0.0) & (d_junc < 120.0)
                same_ap = same & appr.unsqueeze(0)
                P        = same_ap.float().sum(dim=1)
                behind_n = (same_ap & (d_junc.unsqueeze(0)
                                       > d_junc.unsqueeze(1))).float().sum(dim=1)

                # PRIORITY ESTIMATION WITH MEMORY — identical to cosim_sumo
                prop = utils.predecessor_gap(
                    ego_d, v_a, rival_d, rival_v, valid,
                    delta_safe=utils.DELTA_SAFE, ego_P=P,
                    rival_P=P.unsqueeze(0).expand(Na, Na))
                ids = [int(x) for x in act.tolist()]             # stable episode keys
                pred_override, roles = self.role_mem.step(
                    ids, v_a, d_junc, in_box, behind_n, mv_a, ego_d, rival_d, valid,
                    prop, t)

                ctx = mean_net.build_context(v_a, gap, v_lead, P, behind_n, d_junc,
                                             ego_d, rival_d, valid, roles,
                                             AX_MV[mv_a], SG_MV[mv_a])
                mean_fn = model.make_mean_fn(model.encode(*ctx))
                with torch.no_grad():       # diagnostic only: |f| at mid-cell probes
                    fp = mean_fn(F_PROBE.unsqueeze(0).expand(Na, -1, -1))
                    self.f_abs += float(fp.abs().mean()); self.f_n += 1

                # steps 1–3: conditional mean (gradient lives here)
                a_gp, yield_mask, feat = utils.controller_acceleration(
                    torch.zeros(Na), gap + L_VEH, v_a, v_lead,
                    d_conf=ego_d, rival_d=rival_d, rival_v=rival_v,
                    rival_valid=valid, ego_pressure=P,
                    rival_pressure=P.unsqueeze(0).expand(Na, Na),
                    a_prev=self.prev_a[act], kappa=0.5, brake_exempt=True,
                    return_roles=True, return_feat=True,
                    pred_override=pred_override, mean_fn=mean_fn)

                with torch.no_grad():       # |f| at the ACTUAL operating φ*, split conflict/free
                    f_act = mean_fn(feat).abs()                  # [Na]
                    conflict = valid.any(dim=1)                  # ego has a valid cross rival
                    if bool(conflict.any()):
                        self.fc_sum += float(f_act[conflict].sum()); self.fc_n += int(conflict.sum())
                    if bool((~conflict).any()):
                        self.ff_sum += float(f_act[~conflict].sum()); self.ff_n += int((~conflict).sum())
                    self.f_max = max(self.f_max, float(f_act.max()))

                # step 4: gate, straight-through, with defer feedback, force-rolled
                # protected passers, and POST-GATE liveness — identical to cosim
                force_roll = torch.tensor([roles[i] == 'pass'
                                           and self.role_mem.protected(ids[i], t)
                                           for i in range(Na)])
                a_gate, defer = S.rollout_gate(a_gp.detach(), s_a, v_a, mv_a,
                                               yield_mask, self.geo, self.s_junc,
                                               return_defer=True,
                                               force_roll=force_roll)
                self.role_mem.gate_feedback(ids, defer, t)
                self.role_mem.ensure_passer(ids, roles, defer, in_box, d_junc, t)
                a = a_gp + (a_gate - a_gp.detach()).detach()

                # cosim's clamps (both differentiable): yielder a≤0 unless in-box;
                # passer keeps moving (V_MIN_PASS) with room ahead
                yield_t = torch.tensor([roles[i] == 'yield' and not in_box[i]
                                        for i in range(Na)])
                a = torch.where(yield_t, a.clamp(max=0.0), a)

                v_new = (v_a + a * DT).clamp(0.0, V_PHYS)
                floor_t = torch.tensor([roles[i] == 'pass' and not bool(defer[i])
                                        and float(gap[i]) > 2.0 * L_VEH
                                        for i in range(Na)])
                v_new = torch.where(floor_t, v_new.clamp(min=C.V_MIN_PASS), v_new)
                s_new = s_a + v_new * DT

                # ── loss terms ──────────────────────────────────────────────────
                j_step = v_new.sum() * DT                       # ∫Σv dt, this step
                hinge = torch.zeros(())
                near = (d_junc.detach() > -S.JCT_PAST) & (d_junc.detach() < 60.0)
                if int(near.sum()) >= 2:
                    ni = near.nonzero().flatten()
                    # batched interpolation of all near-junction centres at once
                    xy, _ = S._interp_batch(*S.pad_geometry(self.geo),
                                            mv_a[ni], s_new[ni] - L_VEH / 2.0)
                    cross = (ax_a[ni].unsqueeze(0) != ax_a[ni].unsqueeze(1))
                    cross = cross & torch.triu(torch.ones_like(cross), 1)
                    if bool(cross.any()):
                        dist = (xy.unsqueeze(0) - xy.unsqueeze(1)).norm(dim=-1)
                        # SMOOTH SATURATING penalty h²/(1+h): quadratic for shallow
                        # intrusions, ~linear for deep — gradient bounded by SHAPE
                        # (no clamp, no cut gradients, no spikes into Adam)
                        h = F_t.relu(D_HINGE - dist[cross])
                        hinge = (h ** 2 / (1.0 + h)).sum()
                eff = (a_gp ** 2).mean()
                # JERK / smoothness: (Δa)² vs the previous step's applied accel, ONLY for
                # vehicles active in both steps (self.prev_a still holds last step's a here;
                # fresh spawns excluded so the 0→a₀ insertion jump isn't penalized).
                seen = self.was_active[act]
                if bool(seen.any()):
                    jerk = ((a - self.prev_a[act])[seen] ** 2).mean()
                else:
                    jerk = torch.zeros(())

                # hinge enters as a TIME INTEGRAL (×DT) so its scale — and λ_s's
                # sizing against the SUMO probe sweep — is independent of the step
                loss_step = (-LAM_V * (j_step / (self.Ntot * self.plm))
                             + LAM_S * hinge * DT / self.Ntot
                             + LAM_C * eff / self.n_steps
                             + LAM_J * jerk / self.n_steps)
                if not warm:
                    loss_win = loss_win + loss_step
                self.loss_f  += float(loss_step)
                self.j_sum_f += float(j_step)
                self.hinge_f += float(hinge) * DT
                self.eff_f   += float(eff); self.eff_n += 1
                self.jerk_f  += float(jerk); self.jerk_n += 1
                w_j     += float(j_step)
                w_hinge += float(hinge) * DT

                # ── functional write-back ───────────────────────────────────────
                self.s = self.s.clone(); self.s[act] = s_new
                self.v = self.v.clone(); self.v[act] = v_new
                self.prev_a = self.prev_a.clone(); self.prev_a[act] = a
                self.was_active = torch.zeros(self.Ntot, dtype=torch.bool)
                self.was_active[act] = True          # for next step's jerk mask

            done_v = act[self.s[act] >= self.path_len[mv_a]]
            self.state[done_v] = 2
            self.arrived += len(done_v)
        return loss_win, w_j, w_hinge


def rollout_episode(model, events, geo, s_cp, path_len, s_junc, jgeo, train=True):
    """Single-episode convenience wrapper over EpisodeStepper (used by the scratch
    checks): advances window by window; if train, backward per window — gradients
    ACCUMULATE, the caller owns averaging and the optimizer step.
    Returns (loss_total, J_norm, hinge, effort, arrived) as floats."""
    ep = EpisodeStepper(events, geo, s_cp, path_len, s_junc, jgeo)
    while not ep.done():
        loss_win, _, _ = ep.advance(model, train=train)
        if train and loss_win.requires_grad:
            loss_win.backward()
        ep.detach()
    return (ep.loss_f, ep.j_norm(), ep.hinge_f,
            ep.eff_f / max(ep.eff_n, 1), ep.arrived)


def main():
    args   = [a for a in sys.argv[1:] if a != "fresh"]
    fresh  = "fresh" in sys.argv[1:]
    epochs = int(args[0]) if len(args) > 0 else 30
    batch  = int(args[1]) if len(args) > 1 else 8
    lr     = float(args[2]) if len(args) > 2 else 3e-3
    torch.manual_seed(0)
    random.seed(0)

    pool = collect_sumo_initial_conditions()
    geo, s_cp, path_len, s_junc = S.build_geometry()
    jgeo = junction_geometry(geo)
    model = mean_net.make_mean_model()
    best_score = -1e9
    if os.path.exists(CKPT) and not fresh:
        ck = torch.load(CKPT, weights_only=True)
        if isinstance(ck, dict) and "state_dict" in ck:        # current format
            model.load_state_dict(ck["state_dict"])
            # legacy best_j (J-only criterion) is NOT comparable to the combined
            # score — start the bar fresh, weights kept
            best_score = float(ck.get("best_score", -1e9))
        else:                                                  # legacy: bare state_dict
            model.load_state_dict(ck)
        print(f"RESUMING from {CKPT} — best model so far (score={best_score:.4f}); "
              f"pass 'fresh' to start from zero-init")
    else:
        print("starting from the zero-initialized model"
              + (" (fresh requested)" if fresh else f" ({CKPT} not found)"))
    # β₂=0.95: short second-moment memory (~20 updates) so one outlier gradient can't
    # throttle the effective step size for hundreds of updates (we take only ~11/epoch).
    # The hot lr is SAFE here: f is tanh-bounded, anchor-pinned, and double-shielded —
    # a bad f gets corrected, it cannot crash anything.
    opt   = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))
    # GENTLE cosine anneal — decays slowly to 0.3·lr over the whole run (not a fast drop),
    # so late epochs fine-tune without the hot-lr overshoot that saturated the last attempt
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1),
                                                       eta_min=lr * 0.3)

    # per-epoch metric log → outputs/train_log.csv (fresh run truncates; resume appends)
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    if fresh or not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w") as fcsv:
            fcsv.write(",".join(LOG_COLS) + "\n")

    n_windows = int(T_END / WINDOW_S)
    print(f"\ntraining: {epochs} epochs × batch {batch} (pool {len(pool)} SUMO scenarios), "
          f"lr={lr}, window={WINDOW_S:.0f}s, λ_s={LAM_S}, λ_c={LAM_C}")
    print(f"one AVERAGED optimizer step per {WINDOW_S:.0f}s window, batched over the "
          f"episodes — up to {n_windows} updates/epoch (warm-up windows skipped)\n")

    for ep_i in range(epochs):
        t0 = time.time()
        theta0 = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
        scenarios = random.sample(pool, min(batch, len(pool)))
        steppers = [EpisodeStepper(sc["events"], geo, s_cp, path_len, s_junc, jgeo)
                    for sc in scenarios]
        B = len(steppers)
        n_upd = 0

        for w in range(n_windows):
            tw = time.time()
            opt.zero_grad()
            w_loss, w_j, w_hinge, has_grad = 0.0, 0.0, 0.0, False
            for st in steppers:
                loss_win, wj, wh = st.advance(model)
                if loss_win.requires_grad:
                    loss_win.backward()
                    has_grad = True
                st.detach()
                w_loss += float(loss_win); w_j += wj; w_hinge += wh
            if not has_grad:
                continue                       # all-warm-up window: nothing to step on
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= B
            # SKIP the step on a non-finite (inf/nan) gradient — clipping by an inf/nan
            # norm would poison the weights; just drop this window and move on
            finite = all(torch.isfinite(p.grad).all() for p in model.parameters()
                         if p.grad is not None)
            if not finite:
                opt.zero_grad(set_to_none=True)
                print(f"    [epoch {ep_i:3d}] window {w}: non-finite gradient — step SKIPPED")
                continue
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            opt.step(); n_upd += 1
            print(f"    [epoch {ep_i:3d}] update {n_upd:2d}  t={w*WINDOW_S:3.0f}-"
                  f"{(w+1)*WINDOW_S:3.0f}s  mean_win_loss={w_loss/B:+.4f}  "
                  f"hinge={w_hinge/B:6.2f}  |grad|={gnorm:.3f}  ({time.time()-tw:.1f}s)")

        ep_j, ep_hinge, ep_arr, ep_f, ep_score = 0.0, 0.0, 0, 0.0, 0.0
        fc_sum = fc_n = ff_sum = ff_n = 0; f_max = 0.0; ep_eff = ep_jerk = 0.0
        for sc, st in zip(scenarios, steppers):
            j = st.j_norm()
            # throughput is OUT of the objective: select on safety + smoothness only.
            # score = −(λ_s·hinge/N + λ_c·⟨a²⟩ + λ_j·⟨Δa²⟩); higher (less negative) = better
            score = -(LAM_S * st.hinge_f / st.Ntot
                      + LAM_C * st.eff_f / max(st.eff_n, 1)
                      + LAM_J * st.jerk_f / max(st.jerk_n, 1))
            print(f"    [epoch {ep_i:3d}] episode  flow={sc['flow']} "
                  f"seed={sc['seed']:2d}  J̄={j:.4f}  hinge={st.hinge_f:7.2f}  "
                  f"score={score:+.4f}  arrived={st.arrived}")
            ep_j += j; ep_hinge += st.hinge_f; ep_arr += st.arrived
            ep_f += st.f_abs / max(st.f_n, 1); ep_score += score
            fc_sum += st.fc_sum; fc_n += st.fc_n; ff_sum += st.ff_sum; ff_n += st.ff_n
            f_max = max(f_max, st.f_max)
            ep_eff += st.eff_f / max(st.eff_n, 1); ep_jerk += st.jerk_f / max(st.jerk_n, 1)
        ep_j /= B; ep_score /= B
        theta1 = torch.cat([p.detach().flatten() for p in model.parameters()])
        dtheta = float((theta1 - theta0).norm())
        print(f"  epoch {ep_i:3d}  mean_J̄={ep_j:.4f}  mean_hinge={ep_hinge/B:7.2f}  "
              f"score={ep_score:+.4f}  arrived={ep_arr}  updates={n_upd}  "
              f"⟨|f|⟩={ep_f/B:.4f} m/s²  ‖Δθ‖={dtheta:.4f}  ({time.time()-t0:.1f}s)")
        print(f"          |f| at ACTUAL φ*:  conflict={fc_sum/max(fc_n,1):.4f}  "
              f"free={ff_sum/max(ff_n,1):.4f}  max={f_max:.4f} m/s²  "
              f"(conflict steps {fc_n}/{fc_n+ff_n})")
        print(f"          smoothness:  mean a²={ep_eff/B:.4f}  mean (Δa)²={ep_jerk/B:.4f}"
              f"  (λ_c={LAM_C}, λ_j={LAM_J} — want these to DROP as f learns to smooth)\n")
        with open(LOG_CSV, "a") as fcsv:    # one row per epoch → plot_train.py renders curves
            row = [ep_i, ep_j, ep_hinge / B, ep_score, ep_arr, ep_eff / B, ep_jerk / B,
                   ep_f / B, fc_sum / max(fc_n, 1), ff_sum / max(ff_n, 1), f_max, dtheta, n_upd]
            fcsv.write(",".join(f"{x:.6g}" for x in row) + "\n")

        # LATEST: save every epoch (where training currently is, regardless of score)
        torch.save({"state_dict": model.state_dict(), "score": ep_score,
                    "best_j": ep_j, "epoch": ep_i}, LAST_CKPT)
        # BEST: only when the COMBINED score improves (J̄ − λ_s·hinge/N, carried across runs)
        if ep_score > best_score:
            best_score = ep_score
            torch.save({"state_dict": model.state_dict(), "best_score": best_score,
                        "best_j": ep_j}, CKPT)
            print(f"  ↳ new best score={best_score:+.4f} (J̄={ep_j:.4f}) — "
                  f"checkpoint saved to {CKPT}\n")
        sched.step()                      # gentle per-epoch lr anneal

    print(f"done. best score={best_score:+.4f}  checkpoint (best): {CKPT}")


if __name__ == "__main__":
    main()
