"""
utils.py — feature extraction + GP kernel interpolation for the Hybrid model.

The controller prescribes a longitudinal acceleration by kernel interpolation
over physics anchor points.  Each vehicle is reduced to a feature vector
φ(state); the acceleration is the zero-mean Gaussian-process posterior

        â(φ) = k(φ)ᵀ K⁻¹ y

with a Matérn-1/2 (exponential) ARD kernel, anchor Gram matrix K, and
(possibly state-dependent) anchor targets y.

Feature vector φ = (g, τ_c)  — 2-D
-----------------------------------
  g    = Δx / s_des            dynamic gap RATIO to the leader (longitudinal)
  τ_c  = min_k |η_e − η_k|      conflict time-gap to the worst cross-traffic rival

  s_des(v,Δv) = s0 + v·T + [v·Δv/(2√(a_max b_max))]₊      IDM desired gap
  η_e = d_e / v_e,  η_k = d_k / v_k                        ETA to the conflict point

g=1 is the car-following equilibrium; τ_c≥δ_safe means cross traffic is safely
separated in time.  Both feature axes are continuous and differentiable.

Anchor grid (g × τ_c), targets resolved per-state in controller_acceleration:
    g=0              → 'brake'  (a_brake; leader-following dominates at any τ_c)
    g=1,  τ_c>0      → 0         (HOLD: at desired gap, no cross conflict)
    g≥2,  τ_c=τ_c_max→ 'free'   (a_free(v); fully clear → resume free-flow, cap v₀)
    g≥2,  τ_c=δ_safe → 0         (HOLD pin: conflict resolved → no cross correction,
                                  the τ_c-axis analogue of the g=1 longitudinal HOLD)
    g≥1,  τ_c=0      → 'cross'  (a_cross; yield/pass to bump τ_c to δ_safe)

The 'free' consequent is rival-gated: ã_free = ρ·a_free(v) + (1−ρ)·a_cross with a
smooth proximity gate ρ=ρ(τ_c)∈[0,1] (ρ→1 clear, ρ→0 rival imminent), so the open-
road accel collapses into the yield/pass correction exactly when a rival is near —
a state-dependent anchor target, like a_free's existing dependence on v.

Conflict point
--------------
η uses the distance-to-conflict-point d, passed in by the caller.  Today the
caller may use the junction CENTRE for every pair (approximation).  Per-phase
conflict points are a required refinement — they only change the d values fed
in here, not this code.

All functions are differentiable and broadcast over arbitrary leading dims.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# ── physical constants ─────────────────────────────────────────────────────────
L_VEH    = 5.0     # m     vehicle length (bumper-to-bumper correction)
A_MAX    = 2.6     # m/s²  free-flow / accelerate target
B_MAX    = 4.5     # m/s²  max comfortable deceleration (brake clamp magnitude)
S0       = 2.0     # m     jam / standstill minimum gap
T_HW     = 1.5     # s     desired time headway
V0       = 13.89   # m/s   free / desired speed (open-road cap)
DELTA    = 4.0     # —     IDM free-flow exponent
G_MAX    = 5.0     # —     soft clamp on the gap ratio g (open-road saturation)
DELTA_SAFE = 4.0   # s     target conflict time-gap (bump τ_c up to here)
TAU_C_MAX  = 8.0   # s     soft clamp on τ_c (no-conflict saturation)
EPS      = 1e-3    # numerical floor
_SQRT_AB = math.sqrt(A_MAX * B_MAX)   # for the IDM closing term

# ── kernel + anchor configuration ───────────────────────────────────────────────
# ARD Matérn-1/2 length-scales, one per feature axis (g, τ_c).
LENGTHSCALES = (1.0, 2.0)   # (ℓ_g in units of g, ℓ_τc in seconds)
GP_JITTER    = 1e-6

# 2-D anchor grid in (g, τ_c) space.  Targets are sentinels resolved per-state:
#   'brake' → a_brake(state),  'free' → a_free(v),  'cross' → a_cross(state),  float → constant
# Dense brake ladder at low g (0.25, 0.5) so the saturated a_brake anchor is not
# diluted toward HOLD across g∈(0,1) — the regime where rear-end avoidance lives.
_G_LEVELS    = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
_TAUC_LEVELS = (0.0, DELTA_SAFE, TAU_C_MAX)

def _anchor_target(g: float, tc: float):
    if g < 1.0:
        return "brake"           # gap below desired → brake dominates at any conflict
    if tc == 0.0:
        return "cross"           # imminent cross conflict → yield/pass
    if g == 1.0:
        return 0.0               # longitudinal HOLD at desired gap
    # g ≥ 2: free-flow ONLY in the fully-clear column; the δ_safe column is a HOLD
    # pin so the yield equilibrium settles AT δ_safe instead of diluting below it.
    return "free" if tc >= TAU_C_MAX else 0.0

ANCHOR_FEATS   = tuple((g, tc) for g in _G_LEVELS for tc in _TAUC_LEVELS)   # [M, 2]
ANCHOR_TARGETS = tuple(_anchor_target(g, tc) for g, tc in ANCHOR_FEATS)     # [M]


# ─────────────────────────────────────────────────────────────────────────────
# Longitudinal feature + targets (leader following)
# ─────────────────────────────────────────────────────────────────────────────

def desired_gap(v_ego, v_lead, s0=S0, t_hw=T_HW):
    """IDM desired gap  s_des = s0 + v·T + [v·Δv/(2√(a·b))]₊.  Always ≥ s0."""
    dv      = v_ego - v_lead
    closing = F.relu(v_ego * dv / (2.0 * _SQRT_AB))
    return s0 + v_ego * t_hw + closing


def gap_ratio(x_ego, x_lead, v_ego, v_lead, length=L_VEH, g_max=G_MAX):
    """Dynamic gap ratio g = Δx / s_des, soft-clamped at g_max.  Finite at Δv=0."""
    dx    = x_lead - x_ego - length
    s_des = desired_gap(v_ego, v_lead)
    g_raw = dx / s_des.clamp(min=EPS)
    return g_max - F.softplus(g_max - g_raw)


def brake_to_recover(x_ego, x_lead, v_ego, v_lead, length=L_VEH, b_max=B_MAX):
    """BRAKE anchor target: −Δv²/(2(Δx−s_des)), clamped [−b_max, 0]."""
    dx    = x_lead - x_ego - length
    dv    = v_ego - v_lead
    s_des = desired_gap(v_ego, v_lead)
    d     = (dx - s_des).clamp(min=EPS)
    return (-dv.pow(2) / (2.0 * d)).clamp(min=-b_max, max=0.0)


def free_flow_accel(v_ego, v0=V0, delta=DELTA, a_max=A_MAX, b_max=B_MAX):
    """FREE anchor target: a_max·(1 − (v/v₀)^δ), clamped [−b_max, a_max]."""
    a = a_max * (1.0 - (v_ego.clamp(min=0.0) / v0).pow(delta))
    return a.clamp(min=-b_max, max=a_max)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-traffic feature + target (conflict-point timing)
# ─────────────────────────────────────────────────────────────────────────────

def conflict_time_gap(ego_d, v_ego, rival_d, rival_v, rival_valid, tau_c_max=TAU_C_MAX):
    """
    Conflict time-gap to the most-urgent rival, with PER-PAIR conflict points.

    ego_d   [..., K] or [...]   ego distance to EACH rival's crossing point.  Pass
                                [..., K] for per-pair geometry; a [...] scalar
                                broadcasts (the centre approximation).
    rival_* [..., K]            per-rival distance / speed / validity (bool) to that
                                same crossing point.

    Returns (all [...] unless noted):
        tau_c        = min_k |δ_k|, soft-clamped at tau_c_max   (the feature)
        delta_worst  = signed δ of the selected rival           (sign → yield/pass)
        eta_rival    = η of the selected rival
        any_rival    = bool, whether any valid rival exists
        ego_d_sel    = ego distance to the SELECTED rival's crossing point
    δ_k = η_e,k − η_k with η_e,k = ego_d_k / v_e (per pair); invalid rivals are pushed
    to +∞ so the min ignores them.  The argmin is gathered → gradients flow through
    the chosen rival (and its per-pair ego distance).
    """
    v_e     = v_ego.unsqueeze(-1).clamp(min=EPS)               # [..., 1]
    ego_d_k = ego_d if ego_d.shape == rival_d.shape else ego_d.unsqueeze(-1)
    ego_d_k = ego_d_k.expand_as(rival_d)                       # [..., K]
    eta_e   = ego_d_k / v_e                                    # [..., K]
    eta_k   = rival_d / rival_v.clamp(min=EPS)                 # [..., K]
    delta   = eta_e - eta_k                                    # [..., K]

    big   = torch.full_like(delta, 1e6)
    absd  = torch.where(rival_valid, delta.abs(), big)         # [..., K]
    tau_raw, idx = absd.min(dim=-1)                            # [...], [...]

    idx_u       = idx.unsqueeze(-1)                            # [..., 1]
    delta_worst = torch.gather(delta, -1, idx_u).squeeze(-1)   # [...]
    eta_rival   = torch.gather(eta_k, -1, idx_u).squeeze(-1)   # [...]
    ego_d_sel   = torch.gather(ego_d_k, -1, idx_u).squeeze(-1) # [...]
    v_rival_sel = torch.gather(rival_v, -1, idx_u).squeeze(-1) # [...] selected rival speed

    any_rival = rival_valid.any(dim=-1)                        # [...]
    tau_raw   = torch.where(any_rival, tau_raw, torch.full_like(tau_raw, tau_c_max))
    tau_c     = tau_c_max - F.softplus(tau_c_max - tau_raw)    # soft clamp
    return tau_c, delta_worst, eta_rival, any_rival, ego_d_sel, v_rival_sel


def free_flow_gate(tau_c, delta_safe=DELTA_SAFE):
    """
    Rival-proximity gate ρ ∈ [0, 1] for the free-flow consequent.

        ρ → 1   when τ_c ≥ δ_safe   (no close rival → full open-road accel)
        ρ → 0   when τ_c → 0        (rival imminent → free anchor collapses to a_cross)

    Smoothstep on τ_c/δ_safe (C¹, differentiable).  ρ is a state estimate computed
    from the same rival measurement that yields τ_c — a sibling of a_free's existing
    dependence on v, NOT a function of the query coordinate.
    """
    x = (tau_c / delta_safe).clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def cross_resolve_accel(d_ego, v_ego, eta_rival, delta, delta_safe=DELTA_SAFE,
                        a_max=A_MAX, b_max=B_MAX):
    """
    CROSS anchor target: kinematic acceleration that drives the conflict time-gap
    to ±δ_safe, by reaching a target speed at the conflict point.

        δ ≥ 0  → ego arrives later  → YIELD: η_e* = η_rival + δ_safe  (slow down)
        δ < 0  → ego arrives first  → PASS : η_e* = η_rival − δ_safe  (speed up)
                 …pass only if feasible (η_rival > δ_safe); else fall back to YIELD
        v_e*    = d_ego / η_e*
        a_cross = (v_e*² − v_e²) / (2 d_ego),   clamped [−b_max, a_max]

    The δ=0 sign flip is an accepted kink (the symmetric danger apex).  Because a
    rival runs the same rule with δ_rival = −δ, the pair self-assigns yield/pass.
    """
    d_eff    = d_ego.clamp(min=EPS)
    do_yield = (delta >= 0) | (eta_rival <= delta_safe)        # feasibility fallback
    eta_tgt  = torch.where(do_yield, eta_rival + delta_safe, eta_rival - delta_safe)
    v_tgt    = d_eff / eta_tgt.clamp(min=EPS)
    a        = (v_tgt.pow(2) - v_ego.pow(2)) / (2.0 * d_eff)
    return a.clamp(min=-b_max, max=a_max)


def resolve_cross_accel(ego_d, v_ego, rival_d, rival_v, rival_valid,
                        delta_safe=DELTA_SAFE, a_max=A_MAX, b_max=B_MAX):
    """
    η-ORDERING / FCFS multi-rival resolver (Flaw-1 fix).  Priority is arrival time at
    each pair's conflict point: the EARLIER vehicle has right-of-way.

        η_e,k = ego_d_k / v_e   (ego ETA to rival k's crossing point, per pair)
        η_k   = rival_d_k / v_k (rival ETA)
        ego must YIELD to every rival that arrives earlier (η_k < η_e,k).

      • Any earlier rival  → YIELD: slow to arrive δ_safe BEHIND the binding (latest-
        constraining) earlier rival → v ≤ min_k v_yield_k, v_yield_k = ego_d_k/(η_k+δ_safe).
        As ego nears the point this drives v→0 (a stop-line), and gradients flow to the
        binding rival via the min/gather.
      • Ego earliest of all → PROCEED at a_max (assert right-of-way and clear the box).

    This is a CONSISTENT total order (η): of any conflicting pair exactly one yields,
    so two crossing streams never both accelerate in — fixing the all-yield deadlock
    of a symmetric 'pass only if you clear everyone' rule.  The exact η-tie is the
    accepted kink.  The τ_c gate down-weights a_cross when no rival is imminent, so a
    priority vehicle only floors it to clear when a conflict is actually close.
    """
    BIG    = 1e9
    ego_dk = (ego_d if ego_d.shape == rival_d.shape else ego_d.unsqueeze(-1))
    ego_dk = ego_dk.expand_as(rival_d)                         # [..., K]
    v_e    = v_ego.clamp(min=EPS)
    eta_e  = ego_dk / v_e.unsqueeze(-1)                        # [..., K] per-pair ego ETA
    eta_k  = rival_d / rival_v.clamp(min=EPS)                  # [..., K]

    earlier    = rival_valid & (eta_k < eta_e)                 # rivals with right-of-way
    must_yield = earlier.any(dim=-1)                           # [...]
    any_rival  = rival_valid.any(dim=-1)                       # [...]

    # YIELD: arrive δ_safe behind the binding earlier rival (slowest required speed)
    v_yield_k = ego_dk / (eta_k + delta_safe).clamp(min=EPS)
    v_yield_m = torch.where(earlier, v_yield_k, torch.full_like(v_yield_k, BIG))
    v_yield, iy = v_yield_m.min(dim=-1)                        # [...]
    d_bind    = torch.gather(ego_dk, -1, iy.unsqueeze(-1)).squeeze(-1).clamp(min=EPS)
    a_yield   = ((v_yield.pow(2) - v_e.pow(2)) / (2.0 * d_bind)).clamp(-b_max, a_max)

    a_proceed = torch.full_like(a_yield, a_max)                # ego earliest → clear the box
    a_cross   = torch.where(must_yield, a_yield, a_proceed)
    a_cross   = torch.where(any_rival, a_cross, torch.zeros_like(a_cross))
    return a_cross, must_yield


# ── iterative safety correction ─────────────────────────────────────────────────
SAFE_ITERS  = 5      # number of correction passes (±0.2 each → up to ±1.0 m/s²)
SAFE_STEP   = 0.2    # m/s²  accel adjustment per pass
SAFE_BUFFER = 0.5    # s     margin added to the footprint collision threshold


def _predicted_eta(d, v, a):
    """
    Predicted ego ETA to its conflict point if it holds acceleration a.
    Constant-accel kinematics: v_pt² = v² + 2·a·d → T = 2d/(v + v_pt).  If the ego
    would STOP before reaching d (v_pt² ≤ 0) it never arrives → return a large ETA
    (safe).  Differentiable (sqrt with clamp).
    """
    d  = d.clamp(min=0.0)
    v  = v.clamp(min=EPS)
    v_pt_sq = v.pow(2) + 2.0 * a * d
    v_pt    = torch.sqrt(v_pt_sq.clamp(min=0.0))
    eta     = 2.0 * d / (v + v_pt).clamp(min=EPS)
    return torch.where(v_pt_sq > EPS, eta, torch.full_like(eta, 1e3))


def iterative_safety_correction(a, v_ego, ego_d, eta_rival, v_rival, delta_worst,
                                must_yield, any_rival, g, step=SAFE_STEP,
                                n_iter=SAFE_ITERS, buffer=SAFE_BUFFER, a_max=A_MAX):
    """
    f_{i+1} = f_i + h(f_i).  At each pass a vehicle nudges its acceleration by ±`step`
    in the SAFE direction for its role, re-reading the corrected accel each time:

      YIELDER (δ_worst > 0, arrives later):  brake −step IF the predicted footprint gap
        to the worst rival is still unsafe.  Braking → arrives later → gap grows.

      PASSER  (must_yield == False, earliest of ALL rivals):  accelerate +step IF it
        stays safe — predicted gap after +step still ≥ threshold, accel ≤ a_max, and
        longitudinal room (g ≥ 1, else it could rear-end its leader).  Accelerating →
        arrives earlier → gap grows AND it clears the box sooner.

    Footprint threshold = (L/2)/v_e + (L/2)/v_k + buffer (speed-dependent crash
    predictor).  The passer gate is must_yield (earliest of EVERYONE), NOT δ_worst<0:
    a vehicle earlier than its closest rival can still be later than a farther one, and
    accelerating would then cut in front of the rival it owed a yield to.
    """
    half = L_VEH / 2.0
    thr  = half / v_ego.clamp(min=EPS) + half / v_rival.clamp(min=EPS) + buffer
    is_yield = any_rival & (delta_worst > 0.0)                 # brake candidates
    is_pass  = any_rival & (~must_yield) & (g >= 1.0)          # accel candidates (earliest + room)

    a_corr = a
    for _ in range(n_iter):
        # YIELDER: brake more while still predicted-unsafe
        eta_e    = _predicted_eta(ego_d, v_ego, a_corr)
        unsafe   = is_yield & ((eta_e - eta_rival).abs() < thr)
        a_brake  = torch.where(unsafe, a_corr - step, a_corr)

        # PASSER: accelerate if +step keeps it safe and within limits
        a_try    = a_corr + step
        eta_e_up = _predicted_eta(ego_d, v_ego, a_try)
        accel_ok = is_pass & ((eta_e_up - eta_rival).abs() >= thr) & (a_try <= a_max)
        a_accel  = torch.where(accel_ok, a_try, a_corr)

        a_corr   = torch.where(is_yield, a_brake, a_accel)     # roles are mutually exclusive
    return a_corr


# ── differentiable safety floor ─────────────────────────────────────────────────
BRAKE_FLOOR_BETA  = 4.0    # soft-min temperature (→∞ recovers a hard min)
BRAKE_FLOOR_SHARP = 8.0    # g-gate steepness
BRAKE_FLOOR_KNEE  = 0.9    # g below which the floor is active


def brake_safety_floor(a, a_brake, g, beta=BRAKE_FLOOR_BETA,
                       sharp=BRAKE_FLOOR_SHARP, knee=BRAKE_FLOOR_KNEE):
    """
    Differentiable 'most-restrictive-wins' floor.  In the braking regime (g<knee)
    the command is softly clamped so it is never weaker (less negative) than the
    saturated physics brake a_brake:

        soft_min(a, a_brake) = Σ softmax(−β·[a, a_brake]) · [a, a_brake]

    The softmax weights the more-negative value, so the more aggressive deceleration
    dominates (→ true min as β→∞).  A sigmoid g-gate switches the floor OFF for g≥1
    so open-road acceleration is never capped (a_brake = 0 there would otherwise
    pull the command to zero).
    """
    stack   = torch.stack([a, a_brake], dim=-1)                # [..., 2]
    weights = torch.softmax(-beta * stack, dim=-1)             # weight the smaller more
    soft_min = (weights * stack).sum(dim=-1)                   # [...]
    gate     = torch.sigmoid((knee - g) * sharp)               # 1 if g<knee, 0 if g≫knee
    return (1.0 - gate) * a + gate * soft_min


# ─────────────────────────────────────────────────────────────────────────────
# ARD Matérn-1/2 GP interpolation (zero prior mean)
# ─────────────────────────────────────────────────────────────────────────────

def _kernel_gram(anchors, ls):
    """K [M, M] = exp(−Σ_d |a_d − b_d| / ℓ_d) over anchors [M, D]."""
    diff = anchors.unsqueeze(1) - anchors.unsqueeze(0)         # [M, M, D]
    return torch.exp(-(diff.abs() / ls).sum(-1))


def _kernel_vec(feat, anchors, ls):
    """k(φ) [..., M] = exp(−Σ_d |φ_d − a_d| / ℓ_d), feat [..., D]."""
    diff = feat.unsqueeze(-2) - anchors                        # [..., M, D]
    return torch.exp(-(diff.abs() / ls).sum(-1))


def gp_posterior(feat, anchors, targets, ls, jitter=GP_JITTER):
    """
    Zero-mean ARD GP posterior mean:  â = k(φ)ᵀ K⁻¹ y.

    feat    [..., D]   query feature vectors
    anchors [M, D]     anchor positions
    targets [..., M]   anchor targets (may be state-dependent)
    ls      [D]        per-axis length-scales
    """
    M = anchors.shape[0]
    K = _kernel_gram(anchors, ls) + jitter * torch.eye(M, dtype=anchors.dtype,
                                                        device=anchors.device)
    K_inv = torch.linalg.inv(K)
    k_vec = _kernel_vec(feat, anchors, ls)                     # [..., M]
    weights = k_vec @ K_inv                                    # [..., M]
    return (weights * targets).sum(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level controller (what the model forward() calls)
# ─────────────────────────────────────────────────────────────────────────────

def controller_acceleration(
    x_ego, x_lead, v_ego, v_lead,
    # cross-traffic (optional — omit for pure car-following)
    d_conf=None, rival_d=None, rival_v=None, rival_valid=None,
    length=L_VEH,
    lengthscales=LENGTHSCALES,
    a_prev=None, kappa=1.0, brake_exempt=True,
    brake_floor=True, safe_iter=True,
):
    """
    2-D gap-ratio + conflict-time kernel controller.

        φ = (g, τ_c)
        anchor targets: a_brake (g=0), 0 (HOLD), a_free(v) (FREE), a_cross (cross)
        â = gp_posterior(φ, anchors, targets)

    Cross-traffic args are [..., K] (per conflicting rival): rival_d/rival_v/
    rival_valid plus d_conf = ego distance to each rival's crossing point ([..., K]
    for per-pair conflict points, or a [...] scalar for the centre approximation).
    When omitted, τ_c = TAU_C_MAX and the controller reduces to pure car-following.

    Angle-2 damping (optional): first-order lag a_cmd = a_prev + κ(â−a_prev),
    bypassed when braking harder than a_prev if brake_exempt.
    """
    g       = gap_ratio(x_ego, x_lead, v_ego, v_lead, length)
    a_brake = brake_to_recover(x_ego, x_lead, v_ego, v_lead, length)
    a_free  = free_flow_accel(v_ego)

    if rival_d is None:
        tau_c   = torch.full_like(g, TAU_C_MAX)
        a_cross = torch.zeros_like(g)
    else:
        tau_c, delta_w, eta_w, any_rival, ego_d_sel, v_rival_sel = conflict_time_gap(
            d_conf, v_ego, rival_d, rival_v, rival_valid)
        # η-ordering: satisfy ALL active conflict points with one arrival slot
        a_cross, must_yield = resolve_cross_accel(
            d_conf, v_ego, rival_d, rival_v, rival_valid)

    # rival-gated free-flow consequent: collapses a_free → a_cross as a rival nears
    rho        = free_flow_gate(tau_c)
    a_free_eff = rho * a_free + (1.0 - rho) * a_cross

    feat = torch.stack([g, tau_c], dim=-1)                     # [..., 2]
    anchors = torch.tensor(ANCHOR_FEATS, dtype=feat.dtype, device=feat.device)
    ls = torch.tensor(lengthscales, dtype=feat.dtype, device=feat.device)

    _resolve = {"brake": a_brake, "free": a_free_eff, "cross": a_cross}
    target_cols = [
        _resolve[s] if isinstance(s, str) else torch.full_like(a_brake, float(s))
        for s in ANCHOR_TARGETS
    ]
    targets = torch.stack(target_cols, dim=-1)                 # [..., M]

    a_raw = gp_posterior(feat, anchors, targets, ls)

    # differentiable safety floor: never weaker than the saturated brake when g<1
    if brake_floor:
        a_raw = brake_safety_floor(a_raw, a_brake, g)

    # iterative safety correction: brake harder while the predicted conflict is unsafe
    if safe_iter and rival_d is not None:
        a_raw = iterative_safety_correction(
            a_raw, v_ego, ego_d_sel, eta_w, v_rival_sel, delta_w,
            must_yield, any_rival, g)

    if a_prev is None or kappa >= 1.0:
        return a_raw
    a_damped = a_prev + kappa * (a_raw - a_prev)
    if brake_exempt:
        return torch.where(a_raw < a_prev, a_raw, a_damped)
    return a_damped
