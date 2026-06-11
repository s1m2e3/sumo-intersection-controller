"""
train_mean.py — train the transformer prior mean to maximize the velocity integral.

Objective:

    J(θ) = ∫₀ᵀ Σᵢ vᵢ(t) dt                  total vehicle-meters (throughput surrogate)

    L = −J̄  +  λ_s · Σ relu(d_hinge − d_ij)²  +  λ_c · mean(a_GP²)

  J̄ is J normalized by (N_total · mean path length) — the average fraction of its
  route the population completes.  The safety hinge runs over CROSS-axis pairs only;
  the effort term keeps accelerations small unless velocity is actually bought.

Training scheme (epoch = one AVERAGED gradient step):
  1. INITIAL CONDITIONS come from SUMO: for each (flow, seed) we run headless SUMO
     once and record every vehicle's ACTUAL realized departure time and movement
     (insertion queueing included) — cached to IC_CACHE so SUMO runs only once.
  2. Each epoch samples a batch of scenarios from that pool, rolls each through the
     differentiable PyTorch sim, evaluates the loss (printed per scenario), and
     calls backward — gradients ACCUMULATE across the whole batch.
  3. After the batch, gradients are divided by the batch size (averaged), clipped,
     and ONE optimizer step is taken — so the network fits the population of
     scenarios, never a single seed.

Gradient path:
  • differentiable rollout with clone-before-write state updates;
  • the FULL cosim role machinery runs per step (role_memory.RoleMemory: latch,
    queue arbiter, passer-compatibility arbiter, liveness, gate-defer feedback)
    plus cosim's clamps (yielder a≤0, passer V_MIN_PASS floor) — so training and
    co-simulation make the SAME inference; all of it is environment (no gradient);
  • rollout gate applied STRAIGHT-THROUGH (trajectory feels it, gradient skips it);
  • truncated BPTT: backward every BPTT_K steps (accumulates grads, no opt step),
    then the state is detached.

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
                                #  (hold=0.25 s latch!), the Angle-2 lag constant, and
                                #  the gate cadence are identical between train and eval
T_END    = S.T_END
V_PHYS   = S.V_PHYS
L_VEH    = S.L_VEH

BPTT_K   = 150                  # steps per truncated-BPTT window (15 s, as before)
GRAD_WARMUP = 10.0              # s — simulate the first seconds WITHOUT building the
                                #  autograd graph (no loss, no backward): the network
                                #  starts empty and vehicles need ~18 s to reach the
                                #  junction, so the early gradient is ~0 (anchors
                                #  dominate on the approach) — pure compute savings;
                                #  the trajectory itself is identical either way
LAM_S    = 0.5                  # safety-hinge weight — sized so a crash-level hinge
                                #  (~16-47 in the SUMO proxy sweep) costs as much loss
                                #  as the ENTIRE velocity integral of a clean episode
                                #  (J_bar ~0.5): near-misses are never worth the speed
LAM_C    = 1e-3                 # effort (a²) weight
D_HINGE  = S.D_SAFE_2D          # hinge fires below the gate's own safety distance
GRAD_CLIP = 1.0
CKPT     = "mean_net_ckpt.pt"

# ── initial-condition pool: realized SUMO departures, collected once ────────────
IC_FLOWS = (300, 400, 500)
IC_SEEDS = tuple(range(16))     # 16 seeds per flow → 48 scenarios in the pool
IC_CACHE = "_ic_pool.pt"


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


def rollout_episode(model, events, geo, s_cp, path_len, s_junc, jgeo, train=True):
    """One differentiable episode on a fixed initial-condition schedule, with the
    FULL cosim role machinery (role_memory.RoleMemory: latch, queue arbiter, passer
    compatibility arbiter, liveness, gate feedback) and cosim's per-vehicle clamps
    (yielder no-positive-accel, passer keep-moving floor).
    If train, calls backward per BPTT window (gradients ACCUMULATE — caller
    averages and steps).  Returns (loss_total, J_norm, hinge, effort, arrived)."""
    from role_memory import RoleMemory   # lazy (role_memory imports cosim_sumo)
    s_center, s_box_in, s_box_out = jgeo
    Ntot   = len(events)
    depart = torch.tensor([e[0] for e in events])
    move   = torch.tensor([e[1] for e in events], dtype=torch.long)
    s      = torch.zeros(Ntot)
    v      = torch.zeros(Ntot)
    state  = torch.zeros(Ntot, dtype=torch.long)
    prev_a = torch.zeros(Ntot)
    plm    = float(path_len.float().mean())
    role_mem = RoleMemory(dt=DT)         # per-episode stateful role layer (env, no grad)

    n_steps = int(T_END / DT)
    loss_f, j_sum_f, hinge_f, eff_f, eff_n, arrived = 0.0, 0.0, 0.0, 0.0, 0, 0
    loss_win = torch.zeros(())

    for step in range(n_steps):
        t = step * DT
        # ── spawn (constant writes — no gradient) ──────────────────────────────
        s = s.clone(); v = v.clone()
        for mi in range(4):
            am = (state == 1) & (move == mi)
            min_s = float(s[am].min()) if am.any() else 1e9
            if min_s >= 2.0 * L_VEH + S.SPAWN_GAP:
                cand = ((state == 0) & (move == mi) & (depart <= t)).nonzero().flatten()
                if len(cand):
                    k = cand[depart[cand].argmin()]
                    if am.any():
                        v_ld   = float(v[am][s[am].argmin()])
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
        ax_a = S._AXIS[mv_a]

        # warm-up: identical simulation, but NO autograd graph and NO loss until
        # GRAD_WARMUP — the early near-anchor phase contributes ~zero gradient
        warm = train and (t < GRAD_WARMUP)
        with (torch.no_grad() if warm else nullcontext()):
            # ── state estimation (mirrors sim_torch.simulate) ──────────────────
            same  = mv_a.unsqueeze(0) == mv_a.unsqueeze(1)
            ahead = s_a.unsqueeze(0) > s_a.unsqueeze(1)
            eye   = torch.eye(Na, dtype=torch.bool)
            gap_ij = torch.where(same & ahead & ~eye,
                                 (s_a.unsqueeze(0) - s_a.unsqueeze(1)) - L_VEH,
                                 torch.full((Na, Na), 1e9))
            gap, li  = gap_ij.min(dim=1)
            has_lead = gap < 1e8
            v_lead = torch.where(has_lead, v_a[li], v_a)
            gap    = torch.where(has_lead, gap.clamp(min=0.0), torch.full((Na,), 300.0))

            scp_e = s_cp[mv_a][:, mv_a]
            scp_r = s_cp.t()[mv_a][:, mv_a]
            ego_d   = scp_e - s_a.unsqueeze(1)
            rival_d = scp_r - s_a.unsqueeze(0)
            rival_v = v_a.unsqueeze(0).expand(Na, Na)
            valid = ((ax_a.unsqueeze(1) != ax_a.unsqueeze(0)) & ~eye
                     & (ego_d > 0.0) & (rival_d > -S.CLEAR)
                     & ~torch.isnan(ego_d) & ~torch.isnan(rival_d))
            ego_d   = torch.nan_to_num(ego_d, nan=1e3)
            rival_d = torch.nan_to_num(rival_d, nan=-1e3)

            # ── cosim-faithful state quantities (centre-based d_junc, ':'-lane in_box,
            # COUNT platoon pressure in the 0-120 m window, behind_n by d_junc order) ──
            d_junc = s_center[mv_a] - s_a                        # >0 approaching, <0 past
            in_box = ((s_a > s_box_in[mv_a]) & (s_a <= s_box_out[mv_a])).tolist()
            appr    = (d_junc > 0.0) & (d_junc < 120.0)
            same_ap = same & appr.unsqueeze(0)
            P        = same_ap.float().sum(dim=1)
            behind_n = (same_ap & (d_junc.unsqueeze(0) > d_junc.unsqueeze(1))).float().sum(dim=1)

            # ── PRIORITY ESTIMATION WITH MEMORY — identical to cosim_sumo: predecessor_gap
            # proposes, RoleMemory (latch / queue arbiter / passer-compat arbiter / liveness)
            # resolves the final roles and the kernel's pred_override ────────────
            prop = utils.predecessor_gap(
                ego_d, v_a, rival_d, rival_v, valid,
                delta_safe=utils.DELTA_SAFE, ego_P=P,
                rival_P=P.unsqueeze(0).expand(Na, Na))
            ids = [int(k) for k in act.tolist()]                 # stable per-episode keys
            pred_override, roles = role_mem.step(
                ids, v_a, d_junc, in_box, behind_n, mv_a, ego_d, rival_d, valid,
                prop, t)

            ctx = mean_net.build_context(v_a, gap, v_lead, P, behind_n, d_junc,
                                         ego_d, rival_d, valid, roles)
            mean_fn = model.make_mean_fn(model.encode(*ctx))

            # ── steps 1–3: conditional mean (gradient lives here) ──────────────
            a_gp, yield_mask = utils.controller_acceleration(
                torch.zeros(Na), gap + L_VEH, v_a, v_lead,
                d_conf=ego_d, rival_d=rival_d, rival_v=rival_v, rival_valid=valid,
                ego_pressure=P, rival_pressure=P.unsqueeze(0).expand(Na, Na),
                a_prev=prev_a[act], kappa=0.5, brake_exempt=True,
                return_roles=True, pred_override=pred_override, mean_fn=mean_fn)

            # ── step 4: gate, straight-through (trajectory feels it, gradient skips it),
            # with cosim's defer FEEDBACK (gate-deferred passer latched as yielder),
            # force-rolled protected passers, and POST-GATE liveness — identical to cosim
            force_roll = torch.tensor([roles[i] == 'pass'
                                       and role_mem.protected(ids[i], t)
                                       for i in range(Na)])
            a_gate, defer = S.rollout_gate(a_gp.detach(), s_a, v_a, mv_a, yield_mask,
                                           geo, s_junc, return_defer=True,
                                           force_roll=force_roll)
            role_mem.gate_feedback(ids, defer, t)
            role_mem.ensure_passer(ids, roles, defer, in_box, d_junc, t)
            a = a_gp + (a_gate - a_gp.detach()).detach()

            # ── cosim's per-vehicle clamps (both differentiable):
            # YIELDER never accelerates (except in the box — the gate is driving it out);
            # PASSER keeps moving (V_MIN_PASS) with room ahead
            yield_t = torch.tensor([roles[i] == 'yield' and not in_box[i]
                                    for i in range(Na)])
            a = torch.where(yield_t, a.clamp(max=0.0), a)

            v_new = (v_a + a * DT).clamp(0.0, V_PHYS)
            floor_t = torch.tensor([roles[i] == 'pass' and not bool(defer[i])
                                    and float(gap[i]) > 2.0 * L_VEH for i in range(Na)])
            v_new = torch.where(floor_t, v_new.clamp(min=C.V_MIN_PASS), v_new)
            s_new = s_a + v_new * DT

            # ── loss terms ──────────────────────────────────────────────────────
            j_step = v_new.sum() * DT                           # ∫Σv dt, this step
            hinge = torch.zeros(())
            near = (d_junc.detach() > -S.JCT_PAST) & (d_junc.detach() < 60.0)
            if int(near.sum()) >= 2:
                ni = near.nonzero().flatten()
                xy = torch.zeros(len(ni), 2)
                for mi in range(4):
                    selm = (mv_a[ni] == mi).nonzero().flatten()
                    if len(selm):
                        pts, cum = geo[C._MOVES[mi]]
                        xy[selm], _ = S._interp(pts, cum, s_new[ni][selm] - L_VEH / 2.0)
                cross = (ax_a[ni].unsqueeze(0) != ax_a[ni].unsqueeze(1))
                cross = cross & torch.triu(torch.ones_like(cross), 1)
                if bool(cross.any()):
                    dist = (xy.unsqueeze(0) - xy.unsqueeze(1)).norm(dim=-1)
                    hinge = (F_t.relu(D_HINGE - dist[cross]) ** 2).sum()
            eff = (a_gp ** 2).mean()

            # hinge enters as a TIME INTEGRAL (×DT) so its scale — and λ_s's sizing against
            # the SUMO probe sweep, which also integrates — is independent of the sim step
            loss_step = (-(j_step / (Ntot * plm))
                         + LAM_S * hinge * DT / Ntot
                         + LAM_C * eff / n_steps)
            if not warm:
                loss_win = loss_win + loss_step
            loss_f  += float(loss_step)
            j_sum_f += float(j_step); hinge_f += float(hinge) * DT
            eff_f   += float(eff); eff_n += 1

            # ── functional write-back ────────────────────────────────────────────
            s = s.clone(); s[act] = s_new
            v = v.clone(); v[act] = v_new
            prev_a = prev_a.clone(); prev_a[act] = a

        done = act[s[act] >= path_len[mv_a]]
        state[done] = 2
        arrived += len(done)

        # ── truncated BPTT window: backward ACCUMULATES, no optimizer step here ─
        if train and not warm and ((step + 1) % BPTT_K == 0 or step == n_steps - 1):
            if loss_win.requires_grad:
                loss_win.backward()
            loss_win = torch.zeros(())
            s, v, prev_a = s.detach(), v.detach(), prev_a.detach()

    j_norm = j_sum_f / (Ntot * plm)
    return loss_f, j_norm, hinge_f, eff_f / max(eff_n, 1), arrived


def main():
    args   = [a for a in sys.argv[1:] if a != "fresh"]
    fresh  = "fresh" in sys.argv[1:]
    epochs = int(args[0]) if len(args) > 0 else 30
    batch  = int(args[1]) if len(args) > 1 else 8
    lr     = float(args[2]) if len(args) > 2 else 3e-4
    torch.manual_seed(0)
    random.seed(0)

    pool = collect_sumo_initial_conditions()
    geo, s_cp, path_len, s_junc = S.build_geometry()
    jgeo = junction_geometry(geo)
    model = mean_net.MeanTransformer()
    best_j = -1.0
    if os.path.exists(CKPT) and not fresh:
        ck = torch.load(CKPT, weights_only=True)
        if isinstance(ck, dict) and "state_dict" in ck:        # current format
            model.load_state_dict(ck["state_dict"])
            best_j = float(ck.get("best_j", -1.0))
        else:                                                  # legacy: bare state_dict
            model.load_state_dict(ck)
        print(f"RESUMING from {CKPT} — best model so far (mean J̄={best_j:.4f}); "
              f"pass 'fresh' to start from zero-init")
    else:
        print("starting from the zero-initialized model"
              + (" (fresh requested)" if fresh else f" ({CKPT} not found)"))
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\ntraining: {epochs} epochs × batch {batch} (pool {len(pool)} SUMO scenarios), "
          f"lr={lr}, BPTT_K={BPTT_K}, λ_s={LAM_S}, λ_c={LAM_C}")
    print("gradients are AVERAGED over each batch — one optimizer step per epoch\n")

    for ep in range(epochs):
        t0 = time.time()
        scenarios = random.sample(pool, min(batch, len(pool)))
        opt.zero_grad()
        ep_loss, ep_j, ep_hinge, ep_eff, ep_arr = 0.0, 0.0, 0.0, 0.0, 0

        for sc in scenarios:
            t1 = time.time()
            loss, j, hinge, eff, arr = rollout_episode(
                model, sc["events"], geo, s_cp, path_len, s_junc, jgeo, train=True)
            print(f"    [epoch {ep:3d}] loss evaluated  flow={sc['flow']} "
                  f"seed={sc['seed']:2d}  loss={loss:+.4f}  J̄={j:.4f}  "
                  f"hinge={hinge:7.2f}  ⟨a²⟩={eff:.3f}  arrived={arr}  "
                  f"({time.time()-t1:.1f}s)")
            ep_loss += loss; ep_j += j; ep_hinge += hinge; ep_eff += eff; ep_arr += arr

        # ── AVERAGE the accumulated gradients over the batch, then ONE step ─────
        B = len(scenarios)
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= B
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
        opt.step()

        ep_j /= B
        print(f"  epoch {ep:3d}  mean_loss={ep_loss/B:+.4f}  mean_J̄={ep_j:.4f}  "
              f"mean_hinge={ep_hinge/B:7.2f}  mean_⟨a²⟩={ep_eff/B:.3f}  "
              f"arrived={ep_arr}  |grad|={gnorm:.3f}  ({time.time()-t0:.1f}s)\n")

        # the checkpoint is always the BEST model (by epoch mean J̄, carried across runs)
        if ep_j > best_j:
            best_j = ep_j
            torch.save({"state_dict": model.state_dict(), "best_j": best_j}, CKPT)
            print(f"  ↳ new best mean J̄={best_j:.4f} — checkpoint saved to {CKPT}\n")

    print(f"done. best mean J̄={best_j:.4f}  checkpoint (best): {CKPT}")


if __name__ == "__main__":
    main()
