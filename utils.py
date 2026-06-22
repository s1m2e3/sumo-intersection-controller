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
W_VEH    = 1.8     # m     vehicle width
CONFLICT_LEN = L_VEH + W_VEH   # m  longitudinal span a vehicle OCCUPIES the conflict point:
                   #          its own length + the crossing vehicle's width.  Used to turn the
                   #          point-ETA into an OCCUPANCY interval (front-in → rear-out) so the
                   #          conflict gap reflects the 2-D footprints, not a dimensionless point.
A_MAX    = 2.6     # m/s²  free-flow / accelerate target
B_MAX    = 4.5     # m/s²  max comfortable deceleration (brake clamp magnitude)
S0       = 2.0     # m     jam / standstill minimum gap
T_HW     = 1.5     # s     desired time headway
V0       = 11.0    # m/s   free / desired speed (open-road cap)
DELTA    = 4.0     # —     IDM free-flow exponent
G_MAX    = 5.0     # —     soft clamp on the gap ratio g (open-road saturation)
DELTA_SAFE = 3.0   # s     target conflict time-gap (bump τ_c up to here)
STOP_OFFSET = 6.0  # m     yield stop-line: halt this far BEFORE the conflict point (keep box clear)
P_TIE      = 1.0   # m/s   platoon-pressure tie margin (|ΔP| within this = tie → ETA breaks it)
TIE_EPS    = 1.5   # s     ETA band for the near-tie tiebreaker (slower vehicle yields)
TAU_C_MAX  = 8.0   # s     soft clamp on τ_c (no-conflict saturation)
EPS      = 1e-3    # numerical floor
_SQRT_AB = math.sqrt(A_MAX * B_MAX)   # for the IDM closing term

# ── kernel + anchor configuration ───────────────────────────────────────────────
# ARD Matérn-1/2 length-scales, one per feature axis (g, τ_c, r).  ℓ_r is small so the
# two ROLE columns (yield/pass) are nearly independent — role acts as a near-hard switch,
# not a quantity to interpolate across.
LENGTHSCALES = (0.5, 0.3)              # (ℓ_g, ℓ_r) — 2-D kernel, no τ_c axis
GP_JITTER    = 1e-6

# 2-D anchor grid in (g, r) space.  r ∈ {0=YIELDER, 1=PASSER}.
# τ_c is removed: signal phases guarantee that all same-phase vehicles are geometrically
# compatible, so there is no cross-conflict timing to reason about.  The kernel is purely
# car-following (g-axis) gated by the signal-phase role (r-axis).
#   'brake' → a_brake (≤0, saturating)   'free' → a_free(v) (positive)
#   float   → constant accel (0.0 = HOLD)
_G_LEVELS    = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
_ROLE_LEVELS = (0.0, 1.0)        # 0 = YIELDER, 1 = PASSER


def _anchor_target(g: float, r: float):
    if g < 1.0:
        return "brake"   # car-following safety — both roles
    if g == 1.0:
        return 0.0       # HOLD at desired gap — both roles
    # g ≥ 2 (room ahead):
    if r >= 0.5:
        return "free"    # PASSER: free-flow
    return 0.0           # YIELDER: hold speed (yield_cap enforces stop at junction)


ANCHOR_FEATS   = tuple((g, r) for g in _G_LEVELS for r in _ROLE_LEVELS)
ANCHOR_TARGETS = tuple(_anchor_target(g, r) for g, r in ANCHOR_FEATS)   # [M=24]


def set_delta_safe(value):
    """Override the target conflict time-gap δ_safe at RUNTIME (e.g. from a CLI) and
    rebuild every structure derived from it: the middle τ_c anchor level, the anchor
    feature grid (its position moves with δ_safe), and the cached GP inverse (whose Gram
    matrix depends on the moved anchor — the _KINV_CACHE is keyed on shape, not position,
    so it MUST be cleared).  controller_acceleration reads DELTA_SAFE at call time and
    passes it on to free_flow_gate / cross_resolve_accel / resolve_cross_accel, so this
    one call retargets the whole conflict-resolution stack.  Call it ONCE at startup
    before the first controller_acceleration; δ_safe is expected in (0, TAU_C_MAX)."""
    global DELTA_SAFE
    DELTA_SAFE = float(value)
    _KINV_CACHE.clear()
    return DELTA_SAFE


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
                        a_max=A_MAX, b_max=B_MAX, stop_offset=STOP_OFFSET):
    """
    CROSS anchor target: kinematic acceleration that drives the conflict time-gap
    to ±δ_safe, by reaching a target speed at the conflict point.

        δ ≥ 0  → ego arrives later  → YIELD: η_e* = η_rival + δ_safe  (slow down)
        δ < 0  → ego arrives first  → PASS : η_e* = η_rival − δ_safe  (speed up)
                 …pass only if feasible (η_rival > δ_safe); else fall back to YIELD
        v_e*    = d_ego / η_e*
        a_time  = (v_e*² − v_e²) / (2 d_ego)

    STOP-LINE GUARD (yield only): the timing target alone drives v→0 *at* the conflict
    point, so a yielder that can't make δ_safe creeps in and stalls ON the crossing line
    (observed failure mode — slow yielders parked in the box get hit by the crossing
    stream).  We instead require the yielder to be able to HALT before the box: brake to
    v=0 at d_ego − stop_offset.  The yield accel is the MORE restrictive of the two
    (min), so it stops at the line and waits; once the predecessor clears it is no longer
    a rival → free anchor releases it.  The guard is dropped once the ego is already
    inside (d_ego ≤ stop_offset) so anything in the box clears rather than freezing.
    Differentiable throughout.

        a_cross = clamp(min(a_time, a_stop), −b_max, a_max)
    """
    d_eff    = d_ego.clamp(min=EPS)
    do_yield = (delta >= 0) | (eta_rival <= delta_safe)        # feasibility fallback
    eta_tgt  = torch.where(do_yield, eta_rival + delta_safe, eta_rival - delta_safe)
    v_tgt    = d_eff / eta_tgt.clamp(min=EPS)
    a_time   = (v_tgt.pow(2) - v_ego.pow(2)) / (2.0 * d_eff)

    # stop-line guard with a HOLD zone, so the yielder neither creeps onto the point
    # nor freezes once already inside the box:
    #   approach (d > stop_offset)   : brake toward the stop line
    #   hold     (0 < d ≤ stop_offset): do not advance (a ≤ 0) — wait at the line
    #   in box   (d ≤ 0)             : release → clear via the timing target
    d_stop   = (d_ego - stop_offset).clamp(min=EPS)
    a_stop   = -v_ego.pow(2) / (2.0 * d_stop)
    approach = d_ego > stop_offset
    hold     = (~approach) & (d_ego > 0.0)
    a_yield  = torch.where(approach, torch.minimum(a_time, a_stop),
               torch.where(hold,     torch.minimum(a_time, torch.zeros_like(a_time)),
                                     a_time))
    a = torch.where(do_yield, a_yield, a_time)
    return a.clamp(min=-b_max, max=a_max)


def resolve_cross_accel(ego_d, v_ego, rival_d, rival_v, rival_valid,
                        delta_safe=DELTA_SAFE, a_max=A_MAX, b_max=B_MAX,
                        prio_ego=None, prio_rival=None,
                        earlier_override=None, override_mask=None):
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

    PRIORITY BIAS (optional, prio_ego/prio_rival in SECONDS, broadcastable to η): a
    head-start added to a movement class's effective arrival time so it wins the
    right-of-way comparison — e.g. give THROUGH movements priority over turns so the
    higher-conflict-count throughs don't starve under pure FCFS (a turn conflicts with
    ~2 movements, a through with ~6, so unbiased FCFS lets turns repeatedly pre-empt
    throughs → through gridlock at saturation; this is the major-road priority a real
    junction encodes).  CRUCIAL: the bias ONLY shifts the *who-yields* comparison; the
    kinematic yield TARGET still uses the REAL η_k (arrive δ_safe behind the actual
    rival), so safety margins are unchanged — priority changes order, not spacing.
    """
    BIG    = 1e9
    ego_dk = (ego_d if ego_d.shape == rival_d.shape else ego_d.unsqueeze(-1))
    ego_dk = ego_dk.expand_as(rival_d)                         # [..., K]
    v_e    = v_ego.clamp(min=EPS)
    eta_e  = ego_dk / v_e.unsqueeze(-1)                        # [..., K] per-pair ego ETA
    eta_k  = rival_d / rival_v.clamp(min=EPS)                  # [..., K]

    # right-of-way comparison on PRIORITY-ADJUSTED ETAs (default: no bias → pure FCFS)
    eta_e_cmp = eta_e if prio_ego   is None else eta_e - prio_ego
    eta_k_cmp = eta_k if prio_rival is None else eta_k - prio_rival
    earlier    = rival_valid & (eta_k_cmp < eta_e_cmp)         # rivals with right-of-way
    if earlier_override is not None:
        # LATCHED right-of-way (promotion): for egos flagged in override_mask, REPLACE the live
        # ETA ordering with the latched who-yields-to-whom decision.  rival_valid still gates it,
        # so a rival that has physically cleared the pair drops out automatically.  This freezes
        # the promoted set's pass/yield assignment so it can't flap step-to-step.
        om = override_mask.unsqueeze(-1)                       # [..., 1]
        earlier = torch.where(om, rival_valid & earlier_override, earlier)
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


# ── arrival-order (virtual single-file queue) cross feature ─────────────────────

def predecessor_gap(ego_d, v_ego, rival_d, rival_v, rival_valid,
                    delta_safe=DELTA_SAFE, tie_eps=TIE_EPS, tau_c_max=TAU_C_MAX,
                    ego_P=None, rival_P=None):
    """
    Merge all approaches into ONE virtual queue ordered by ETA to the junction, and
    return ego's arrival-time gap to its immediate PREDECESSOR — the cross feature.

    δ_k = η_{e,k} − η_k  (per-pair).  k is a predecessor of ego (ego must follow it)
    iff it clearly arrives earlier (δ_k > tie_eps) OR a near-tie ego loses on the
    slower-yields rule.  The immediate predecessor p* = argmin_{predecessors} δ_k
    (smallest positive gap, gathered → differentiable).

    Returns:
        r_gap      = Δη to p*, soft-clamped at tau_c_max   (= tau_c_max ⇒ ego is FIRST)
        eta_pred, ego_d_pred, v_pred = p*'s ETA / ego-distance-to-its-point / speed
        has_pred   = bool, whether ego has a predecessor (else it leads → free-flow)

    Maintaining Δη ≥ δ_safe behind p* cascades: consecutive δ_safe gaps ⇒ every pair is
    ≥ δ_safe apart ⇒ no two at the junction together.  Ego FIRST ⇒ no predecessor ⇒
    r_gap = tau_c_max ⇒ the free anchor fires ⇒ it accelerates to clear.
    """
    v_e    = v_ego.clamp(min=EPS)
    ego_dk = (ego_d if ego_d.shape == rival_d.shape else ego_d.unsqueeze(-1)).expand_as(rival_d)
    eta_e  = ego_dk / v_e.unsqueeze(-1)                        # [..., K] per-pair ego ETA
    eta_k  = rival_d / rival_v.clamp(min=EPS)                  # [..., K]
    delta  = eta_e - eta_k                                     # >0 ⇒ k arrives before ego

    near = delta.abs() <= tie_eps
    if ego_P is not None and rival_P is not None:
        # platoon-pressure priority: the busier/faster lane has right-of-way; ties broken
        # by ETA (later arrival yields).  Pressure is available far out and changes slowly,
        # so the sparse lane commits to yield EARLY (reaching its stop-line with room) and
        # the decision doesn't flicker step-to-step (less chatter).  The third clause yields
        # to a much-busier lane even when ego is marginally earlier — the early commitment.
        dP      = rival_P - ego_P.unsqueeze(-1)
        busier  = dP > P_TIE
        eqP     = dP.abs() <= P_TIE
        base    = ((delta > tie_eps)
                   | (near & (busier | (eqP & (delta > 0.0))))
                   | (busier & (delta > -tie_eps)))
    else:
        base    = (delta > tie_eps) | (near & (v_e.unsqueeze(-1) < rival_v))

    # POINT OF NO RETURN: a vehicle within its braking distance (+ stop-line margin) of the
    # conflict point can no longer stop → it is COMMITTED to proceed, and priority must NOT
    # flip onto it.  So ego YIELDS to any committed rival (forced_yield), and ego itself never
    # yields once committed (forced_proceed) — overriding the pressure/ETA base rule.  This
    # removes the late-flip-then-emergency-brake failure: by the time a car can't stop, the
    # other side gives way instead.
    d_need_e = v_e.unsqueeze(-1).pow(2) / (2.0 * B_MAX)
    d_need_r = rival_v.pow(2) / (2.0 * B_MAX)
    ego_commit     = ego_dk  <= d_need_e + STOP_OFFSET
    rival_commit   = rival_d <= d_need_r + STOP_OFFSET
    forced_yield   = rival_commit & (~ego_commit)
    forced_proceed = ego_commit & (~rival_commit)
    is_pred = rival_valid & (forced_yield | (base & ~forced_proceed))
    BIG     = 1e9
    delta_m = torch.where(is_pred, delta, torch.full_like(delta, BIG))
    r_raw, idx = delta_m.min(dim=-1)                          # immediate predecessor
    has_pred   = is_pred.any(dim=-1)

    idx_u      = idx.unsqueeze(-1)
    eta_pred   = torch.gather(eta_k,   -1, idx_u).squeeze(-1)   # predecessor FRONT-in time
    ego_d_pred = torch.gather(ego_dk,  -1, idx_u).squeeze(-1)
    v_pred     = torch.gather(rival_v, -1, idx_u).squeeze(-1)

    # 2-D / EXTREME-aware gap.  The predecessor doesn't clear the conflict point instantly —
    # it OCCUPIES it for t_block = CONFLICT_LEN / v_pred (its body + the crosser's width).  The
    # MOST DANGEROUS extreme for the yielder is its own FRONT entering vs the predecessor's
    # REAR leaving, so the true safe gap is r_raw − t_block (front-in minus rear-out), and the
    # yield target becomes "arrive δ_safe after the predecessor's rear clears" — eta_pred is
    # shifted to that rear-out time so cross_resolve_accel (η_tgt = η_pred + δ_safe) stays
    # consistent.  r_raw < t_block ⇒ footprints would overlap ⇒ r_gap clamps to 0 (hard yield).
    t_block    = CONFLICT_LEN / v_pred.clamp(min=EPS)
    eta_pred   = eta_pred + t_block                            # predecessor REAR-out time
    r_raw      = r_raw - t_block                               # ego front-in − predecessor rear-out

    r_gap = torch.where(has_pred, r_raw.clamp(min=0.0),
                        torch.full_like(r_raw, tau_c_max))
    r_gap = tau_c_max - F.softplus(tau_c_max - r_gap)         # soft clamp (same as τ_c)
    return r_gap, eta_pred, ego_d_pred, v_pred, has_pred, is_pred


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


_KINV_CACHE: dict = {}   # (M, ls, dtype, device) → K⁻¹; anchors are fixed constants


def _anchor_kinv(anchors, ls, jitter=GP_JITTER):
    """K⁻¹ for the (fixed) anchor Gram, memoized so the inverse is computed once,
    not per call.  Keyed on (M, ls, dtype, device) — valid because anchor positions
    never change at runtime (they're module constants; not learned).  If anchors are
    ever made learnable, drop this cache (it severs the gradient w.r.t. anchors)."""
    key = (anchors.shape[0], tuple(ls.tolist()), anchors.dtype, str(anchors.device))
    K_inv = _KINV_CACHE.get(key)
    if K_inv is None:
        M = anchors.shape[0]
        K = _kernel_gram(anchors, ls) + jitter * torch.eye(
            M, dtype=anchors.dtype, device=anchors.device)
        K_inv = torch.linalg.inv(K)
        _KINV_CACHE[key] = K_inv
    return K_inv


def gp_posterior(feat, anchors, targets, ls, jitter=GP_JITTER,
                 mean_q=None, mean_X=None):
    """
    ARD GP posterior mean.  Zero prior mean (default):

        â = k(φ)ᵀ K⁻¹ y

    With a prior mean function m(·) (the conditional Gaussian mean formula),
    pass its evaluations at the query (mean_q) and at the anchors (mean_X):

        â = m(φ) + k(φ)ᵀ K⁻¹ (y − m(X))

    Exact at anchors REGARDLESS of m (k(xᵢ)ᵀK⁻¹ is the i-th unit row, so the
    correction cancels m(xᵢ) and returns yᵢ); far from all anchors k → 0 and
    the posterior reverts to the prior mean m(φ) (0 in the default case).

    feat    [..., D]   query feature vectors
    anchors [M, D]     anchor positions
    targets [..., M]   anchor targets (may be state-dependent)
    ls      [D]        per-axis length-scales
    mean_q  [...]      prior mean at the query   (optional, with mean_X)
    mean_X  [..., M]   prior mean at the anchors (optional, with mean_q)
    """
    K_inv = _anchor_kinv(anchors, ls, jitter)                  # cached
    k_vec = _kernel_vec(feat, anchors, ls)                     # [..., M]
    weights = k_vec @ K_inv                                    # [..., M]
    if mean_q is None:
        return (weights * targets).sum(dim=-1)
    return mean_q + (weights * (targets - mean_X)).sum(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level controller (what the model forward() calls)
# ─────────────────────────────────────────────────────────────────────────────

def controller_acceleration(
    x_ego, x_lead, v_ego, v_lead,
    role=None,               # [N] float: 1=passer (green), 0=yielder (red). None → all pass.
    length=L_VEH,
    lengthscales=LENGTHSCALES,
    a_prev=None, kappa=1.0, brake_exempt=True,
    brake_floor=True,
    return_feat=False,
):
    """
    2-D signal-phase kernel controller.

        φ = (g, r)   g = gap-ratio to same-queue leader,  r = signal-phase role
        anchor targets: 'brake' → a_brake(g),  0.0 → HOLD,  'free' → a_free(v)
        â = gp_posterior(φ, anchors, targets)

    role: per-vehicle float tensor — 1.0 = green (passer, free-flow), 0.0 = red (yielder,
    hold speed; yield_cap in the caller enforces the stop at the junction stop-line).
    Angle-2 damping (optional): first-order lag a_cmd = a_prev + κ(â−a_prev),
    bypassed when braking harder than a_prev if brake_exempt.

    mean_fn (optional): a prior MEAN function f(φ) → accel (e.g. the closure from
    mean_net.MeanTransformer.make_mean_fn, with the traffic context baked in).
    The posterior becomes the conditional Gaussian mean

        â = f(φ*) + k(φ*)ᵀ K⁻¹ (y − f(X))

    i.e. anchors pin the physics targets exactly (per-context, regardless of f)
    and the learned mean takes over away from them.  mean_fn must accept φ of
    shape [..., M+1, 3] (query stacked with the M anchors) and return [..., M+1].
    None ⇒ zero prior mean, byte-identical to the original controller.
    """
    g       = gap_ratio(x_ego, x_lead, v_ego, v_lead, length)
    a_brake = brake_to_recover(x_ego, x_lead, v_ego, v_lead, length)
    a_free  = free_flow_accel(v_ego)

    # Role r ∈ {0=YIELDER, 1=PASSER} comes directly from the signal phase assignment.
    # 1 = green phase (pass, free-flow);  0 = red phase (yield, stop at junction).
    r_feat = torch.ones_like(g) if role is None else role.to(g.dtype)

    feat    = torch.stack([g, r_feat], dim=-1)                   # [..., 2]
    anchors = torch.tensor(ANCHOR_FEATS, dtype=feat.dtype, device=feat.device)
    ls      = torch.tensor(lengthscales, dtype=feat.dtype, device=feat.device)

    _resolve = {"brake": a_brake, "free": a_free}
    target_cols = [
        _resolve[s] if isinstance(s, str) else torch.full_like(a_brake, float(s))
        for s in ANCHOR_TARGETS
    ]
    targets = torch.stack(target_cols, dim=-1)                   # [..., M]

    a_raw = gp_posterior(feat, anchors, targets, ls)

    if brake_floor:
        a_raw = brake_safety_floor(a_raw, a_brake, g)

    if a_prev is None or kappa >= 1.0:
        a_out = a_raw
    else:
        a_damped = a_prev + kappa * (a_raw - a_prev)
        a_out = torch.where(a_raw < a_prev, a_raw, a_damped) if brake_exempt else a_damped

    if return_feat:
        return a_out, feat
    return a_out


# ─────────────────────────────────────────────────────────────────────────────
# Signal phase definitions
# ─────────────────────────────────────────────────────────────────────────────
# Movement index = approach*3 + dir
#   approach: 0=E  1=W  2=N  3=S
#   dir:      0=r  1=s  2=l
# E.r=0 E.s=1 E.l=2  W.r=3 W.s=4 W.l=5
# N.r=6 N.s=7 N.l=8  S.r=9 S.s=10 S.l=11

_AR = frozenset()
SIGNAL_PHASES = [
    frozenset({0, 1, 3, 4}),    # Phase 0: EW through + right  (30 s)
    _AR,                          # all-red                       ( 3 s)
    frozenset({2, 5}),            # Phase 1: EW protected left    (15 s)
    _AR,                          # all-red                       ( 3 s)
    frozenset({6, 7, 9, 10}),   # Phase 2: NS through + right   (30 s)
    _AR,                          # all-red                       ( 3 s)
    frozenset({8, 11}),           # Phase 3: NS protected left    (15 s)
    _AR,                          # all-red                       ( 3 s)
]
SIGNAL_PHASE_DURS = [30.0, 3.0, 15.0, 3.0, 30.0, 3.0, 15.0, 3.0]
SIGNAL_CYCLE = sum(SIGNAL_PHASE_DURS)  # 102 s

_SIGNAL_CUM: list = []
_c = 0.0
for _d in SIGNAL_PHASE_DURS:
    _c += _d
    _SIGNAL_CUM.append(_c)


# ─────────────────────────────────────────────────────────────────────────────
# Signal-phase kernel controller
# ─────────────────────────────────────────────────────────────────────────────

class SignalController:
    """Signal-phase GP kernel controller for a signalised intersection.

    Receives geometry (conf, s_junc) from the harness at construction — swap the
    geometry object to run on a different net without touching this class.

    State kept across steps:
        committed  — vehicle ids that entered the box on a green phase; they keep
                     passer role even after the phase switches (premature-on_box can
                     place them on an internal lane up to ~4 m before the arc-length
                     junction entry, so d_junc > 0 alone is not a safe commit test).
        prev_a     — last applied acceleration per vehicle (Angle-2 damping lag).
    """

    def __init__(self, conf: torch.Tensor, s_junc: torch.Tensor):
        """
        conf    [M, M] bool  — True where movements i and j conflict
        s_junc  [M]    float — arc-length from route start to box entry per movement
        """
        self.conf   = conf
        self.s_junc = s_junc
        self.committed: set = set()
        self.prev_a: dict   = {}

    def reset(self):
        self.committed.clear()
        self.prev_a.clear()

    def step(self, vehs: list, mvi: torch.Tensor, vs: torch.Tensor,
             gap: torch.Tensor, v_lead: torch.Tensor,
             d_junc: torch.Tensor, on_box: list, t: float):
        """
        One simulation step — returns accelerations and diagnostic info.

        Args:
            vehs    list[str]   vehicle ids, length N
            mvi     [N] int     movement index per vehicle
            vs      [N] float   current speed (m/s)
            gap     [N] float   bumper-to-bumper gap to same-queue leader (m)
            v_lead  [N] float   leader speed (m/s)
            d_junc  [N] float   arc-length to junction entry (>0 approaching, <0 past)
            on_box  list[bool]  True if SUMO placed vehicle on an internal lane
            t       float       simulation time (s)

        Returns:
            a     [N]  final acceleration commands (m/s²)
            info  dict is_yield, role, yield_cap, box_cap  (all [N])
        """
        N = len(vehs)

        # ── Signal phase → is_yield ──────────────────────────────────────────
        _t_cyc = t % SIGNAL_CYCLE
        _pidx  = next(k for k, cum in enumerate(_SIGNAL_CUM) if _t_cyc < cum)
        _green = SIGNAL_PHASES[_pidx]
        is_yield = torch.tensor(
            [int(mvi[i]) not in _green for i in range(N)], dtype=torch.bool)

        # ── Committed-crosser tracking ───────────────────────────────────────
        # A vehicle is committed once SUMO places it on an internal lane (on_box)
        # while its movement is green.  It keeps passer role to clear the box
        # even if the phase switches before its arc-length d_junc crosses zero.
        for i, v in enumerate(vehs):
            if on_box[i] and not bool(is_yield[i]):
                self.committed.add(v)
        in_box = torch.tensor(
            [(vehs[i] in self.committed) or float(d_junc[i]) < 0.0
             for i in range(N)], dtype=torch.bool)
        role = ((~is_yield) | in_box).float()  # 1 = passer/committed, 0 = yielder

        # ── GP kernel ────────────────────────────────────────────────────────
        a_prev_t = torch.tensor([self.prev_a.get(v, 0.0) for v in vehs])
        a = controller_acceleration(
            torch.zeros(N), gap + L_VEH, vs, v_lead,
            role=role, a_prev=a_prev_t, kappa=0.5,
            brake_exempt=True, brake_floor=True)
        a = a.detach().clone()

        # ── Box-entry mutual exclusion (box_cap) ─────────────────────────────
        # An approaching vehicle brakes to a stop before the box if a conflicting
        # movement physically occupies it.  In-box vehicles are always exempt.
        box_cap = torch.full((N,), float("inf"))
        occ_mv  = {int(mvi[i]) for i in range(N) if on_box[i]}
        if occ_mv:
            for i in range(N):
                if on_box[i]:
                    continue
                _dj = float(d_junc[i])
                if _dj <= 0.0:
                    continue
                if any(bool(self.conf[occ, int(mvi[i])]) for occ in occ_mv):
                    box_cap[i] = (2.0 * B_MAX * _dj) ** 0.5

        # ── Yield cap + decel floor ──────────────────────────────────────────
        # Red-phase vehicles stop STOP_OFFSET m before the box (keeps them clear
        # of SUMO's premature-on_box zone, ~4 m for right-turn arcs).
        # Committed crossers and vehicles already past the arc-length entry are exempt.
        yield_cap = torch.full((N,), float("inf"))
        for i in range(N):
            if not bool(is_yield[i]) or vehs[i] in self.committed:
                continue
            _dj = float(d_junc[i])
            if _dj <= 0.0:
                continue
            yield_cap[i] = (2.0 * B_MAX * max(_dj - STOP_OFFSET, 0.0)) ** 0.5

        for i in range(N):
            if not bool(is_yield[i]) or vehs[i] in self.committed:
                continue
            _dj = float(d_junc[i])
            if _dj <= 0.0:
                continue
            _v_i = float(vs[i])
            if _v_i <= 0.01:
                continue
            _a_req = max(-B_MAX, -(_v_i ** 2) / (2.0 * max(_dj - STOP_OFFSET, 0.001)))
            a[i] = min(float(a[i]), _a_req)

        for i, v in enumerate(vehs):
            self.prev_a[v] = float(a[i])

        return a, dict(is_yield=is_yield, role=role,
                       yield_cap=yield_cap, box_cap=box_cap)
