"""
safety.py — RKHS functional gradient descent for the TTC safety constraint

Constraint : TTC > 3 s  for all (i, j, k) conflict pairs
Violation  : V[f] = Σ_{i,j,k} ∫_0^T softplus(3 − TTC_{ijk}(t; f)) dt

Gradient   : δV/δa_i(t') = dt³ · Σ_{j,k} Σ_{t > t'}
                              σ(3 − TTC(t)) · sign(η_i(t) − η_k(t)) · (t − t') / v_i(t)
             where σ = sigmoid = softplus'

RKHS step  : g_i(s) = Σ_{t'} k(s, t') · δV/δa_i(t')
             k(s,t') = exp(−(s−t')² / 2σ²)   [RBF in step units]

Descent    : f_{n+1}(t) = f_n(t) − η · g_i(t)
             implemented as an additive correction δa accumulated across iterations.

run_safety_descent() prints the total violation after each of n_steps iterations
so you can see how quickly the constraint is being satisfied.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from conflict import ConflictSnapshot, STREAM_NAMES
from ttc import (
    HORIZON, _EPS_V, _cp_pair_offsets, build_ttc_surfaces,
    ProjectionInfo, TTCSurface,
)


# ---------------------------------------------------------------------------
# Violation scalar
# ---------------------------------------------------------------------------

def compute_violation(
    surfaces:  dict[str, TTCSurface],
    dt:        float = 0.1,
    threshold: float = 3.0,
    beta:      float = 1.0,
) -> float:
    """
    Total smooth constraint violation with steepness β:
        V = Σ_{i,j,k,t} dt · softplus_β(threshold − TTC_{ijk}(t))
        softplus_β(x) = (1/β) log(1 + e^{βx})   →  ReLU as β → ∞

    β is increased as the violation shrinks so the approximation tightens
    toward the exact constraint boundary.
    """
    total = 0.0
    for surf in surfaces.values():
        for ttc_j in surf.ttc:
            # F.softplus(x, beta=b) = (1/b)*log(1+exp(b*x))  — numerically stable
            total += F.softplus(threshold - ttc_j, beta=beta).sum().item() * dt
    return total


# ---------------------------------------------------------------------------
# Analytical pointwise gradient  δV / δa_i(t')
# ---------------------------------------------------------------------------

def _pointwise_grad(
    surf:      TTCSurface,
    proj:      ProjectionInfo,
    vid:       str,
    dt:        float,
    threshold: float = 3.0,
    beta:      float = 1.0,
) -> torch.Tensor:
    """
    Returns δV/δa_i(t') for vehicle `vid`, shape [horizon].

    Derivation (ignoring clamp non-linearity, first-order in dt):
      ∂cum_dist_i(t)/∂a_i(t') ≈ (t − t') · dt²   for t > t'
      ∂η_i(t)/∂a_i(t')         ≈ −(t−t')·dt² / v_i(t)
      ∂|TTC|(t)/∂a_i(t')        = sign(η_i−η_k) · ∂η_i/∂a_i(t')
      δV/δa_i(t') = dt · Σ_{j,k,t>t'} σ(c−TTC) · sign(η_i−η_k) · (t−t')·dt² / v_i(t)
                  = dt³ · Σ_{j,k} Σ_{t>t'} W_{jk}(t) · (t−t')
    """
    k     = proj.idx_of[vid]
    v_i   = proj.v_traj[k]        # [H]
    d_i   = proj.cum_dist[k]      # [H]
    H     = v_i.shape[0]

    # W[t] = Σ_{j,kr} σ(c − TTC[j][t,kr]) · sign(η_i_j(t) − η_k(t)) / v_i(t)
    W = torch.zeros(H)

    for j, cs in enumerate(surf.conflict_streams):
        rivals = surf.rival_ids[j]
        if not rivals:
            continue

        off_i, off_j = _cp_pair_offsets(surf.stream, cs)

        # η_i for this conflict point
        d_cp_i  = proj.d_junc[k] + off_i
        d_rem_i = (d_cp_i - d_i).clamp(min=0.0)          # [H]
        eta_i   = d_rem_i / v_i.clamp(min=_EPS_V)         # [H]

        for kr, rvid in enumerate(rivals):
            ri = proj.idx_of.get(rvid)
            if ri is None:
                continue

            d_cp_j  = proj.d_junc[ri] + off_j
            d_rem_j = (d_cp_j - proj.cum_dist[ri]).clamp(min=0.0)
            eta_k   = d_rem_j / proj.v_traj[ri].clamp(min=_EPS_V)  # [H]

            # σ(β·(c − TTC)):  sigmoid of the steepened argument
            sig = torch.sigmoid(beta * (threshold - surf.ttc[j][:, kr]))  # [H]
            sgn = torch.sign(eta_i - eta_k)                       # [H]

            W += sig * sgn / v_i.clamp(min=_EPS_V)

    # δV/δa_i[t'] = dt³ · Σ_{t > t'} W(t) · (t − t')
    #             = dt³ · [Σ_{t>t'} t·W(t)  −  t' · Σ_{t>t'} W(t)]
    t_idx = torch.arange(H, dtype=torch.float32)
    tW    = t_idx * W

    # suffix sums shifted by 1: sum over t > t'
    suffix_W  = torch.cat([W.flip(0).cumsum(0).flip(0)[1:],  torch.zeros(1)])
    suffix_tW = torch.cat([tW.flip(0).cumsum(0).flip(0)[1:], torch.zeros(1)])

    pw_grad = (dt ** 3) * (suffix_tW - t_idx * suffix_W)
    return pw_grad


# ---------------------------------------------------------------------------
# RKHS gradient  (RBF kernel smoothing)
# ---------------------------------------------------------------------------

def _rkhs_grad(
    pw_grad:     torch.Tensor,   # [horizon]
    sigma_steps: float,
) -> torch.Tensor:
    """
    g(s) = Σ_{t'} k(s, t') · δV/δa(t')
    k(s,t') = exp(−(s−t')² / 2σ²)   (RBF in step units)

    Implemented as a 1-D convolution with a discrete Gaussian kernel.
    """
    half_w = int(4 * sigma_steps) + 1
    t_r    = torch.arange(-half_w, half_w + 1, dtype=torch.float32)
    kernel = torch.exp(-t_r ** 2 / (2 * sigma_steps ** 2))
    kernel = kernel / kernel.sum()

    x = pw_grad.unsqueeze(0).unsqueeze(0)          # [1, 1, H]
    k = kernel.unsqueeze(0).unsqueeze(0)            # [1, 1, K]
    g = F.conv1d(x, k, padding=half_w).squeeze()   # [H]
    return g


# ---------------------------------------------------------------------------
# Descent loop
# ---------------------------------------------------------------------------

def run_safety_descent(
    snapshot:     ConflictSnapshot,
    model:        nn.Module,
    v_dict:       dict[str, float],
    gap_dict:     dict[str, float],
    vlead_dict:   dict[str, float],
    x_seq_dict:   dict[str, torch.Tensor] | None = None,
    dt:           float = 0.1,
    n_steps:      int   = 5,
    eta:          float = 0.05,
    sigma:        float = 1.0,
    threshold:    float = 3.0,
    beta_0:       float = 1.0,
    verbose:      bool  = True,
    init_delta_a: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], float]:
    """
    Functional gradient descent on the safety violation.

    Starting from f_0 = f_hat + h_≤ (IDM base), runs n_steps iterations of:
        f_{n+1}(t) = f_n(t) − η · g_i(t)

    init_delta_a: warm-start correction (e.g. shifted result from prior step).
                  Vehicles absent from the current snapshot are silently dropped.

    Returns (delta_a_dict, final_min_ttc).
    """
    sigma_steps   = sigma / dt
    # warm-start: use provided init, drop vehicles no longer tracked
    if init_delta_a:
        delta_a_dict: dict[str, torch.Tensor] = {
            vid: v for vid, v in init_delta_a.items()
            if vid in snapshot.vehicle_stream
        }
    else:
        delta_a_dict = {}
    beta          = beta_0
    V_0           = None
    last_min_ttc  = threshold   # default: no conflicts → safe

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  RKHS safety descent   η={eta}  σ={sigma}s  c={threshold}s  β₀={beta_0}")
        print(f"  {'iter':>4}  {'V':>12}  {'β':>8}  {'min TTC (s)':>12}  at-risk")
        print(f"{'─'*70}")

    for step in range(n_steps + 1):
        surfaces, proj = build_ttc_surfaces(
            snapshot, model, v_dict, gap_dict, vlead_dict,
            x_seq_dict=x_seq_dict, dt=dt,
            delta_a_dict=delta_a_dict if delta_a_dict else None,
        )

        V = compute_violation(surfaces, dt=dt, threshold=threshold, beta=beta)

        if V_0 is None:
            V_0 = V

        # actual minimum TTC across all pairs and projected steps
        min_ttc = float("inf")
        n_pairs = 0
        for surf in surfaces.values():
            for j, ttc_j in enumerate(surf.ttc):
                if not surf.rival_ids[j]:
                    continue
                val = ttc_j.min().item()
                if val < threshold:
                    n_pairs += 1
                min_ttc = min(min_ttc, val)
        if min_ttc == float("inf"):
            min_ttc = threshold

        last_min_ttc = min_ttc
        if verbose:
            print(f"  {step:4d}  {V:12.5f}  {beta:8.3f}  {min_ttc:12.4f}  {n_pairs}")

        if step == n_steps:
            break

        # ── adapt β: tighten only as TTC approaches the boundary ─────────────
        # β = β_0 · (c / gap)  where gap = c − min_ttc
        # Far below threshold (gap ≈ c): β ≈ β_0  → smooth, big steps allowed
        # Near threshold      (gap → 0): β → ∞    → ReLU-like, precise boundary
        gap  = max(threshold - last_min_ttc, 1e-3)   # how far below c we still are
        beta = beta_0 * (threshold / gap)

        # ── compute RKHS gradient and update δa for ALL tracked vehicles ────
        new_delta: dict[str, torch.Tensor] = {}
        accel_count = brake_count = 0
        show_diag   = (step == 0 and verbose)

        if show_diag:
            print(f"\n  {'vehicle':<22} {'stream':<8} {'δa[0] m/s²':>12}  action")
            print(f"  {'─'*22} {'─'*8} {'─'*12}  ──────")

        for vid in proj.tracked:
            surf = surfaces.get(vid)
            if surf is None:
                continue

            pw   = _pointwise_grad(surf, proj, vid, dt, threshold, beta)
            g    = _rkhs_grad(pw, sigma_steps)

            prev            = delta_a_dict.get(vid, torch.zeros(HORIZON))
            correction      = -eta * g
            new_delta[vid]  = prev + correction

            if show_diag:
                da0    = correction[0].item()
                action = "ACCEL ↑" if da0 > 1e-6 else ("BRAKE ↓" if da0 < -1e-6 else "none")
                sname  = STREAM_NAMES.get(surf.stream, str(surf.stream))
                print(f"  {vid:<22} {sname:<8} {da0:>12.6f}  {action}")
                if da0 > 1e-6:
                    accel_count += 1
                elif da0 < -1e-6:
                    brake_count += 1

        if show_diag:
            print(f"\n  → {accel_count} vehicles accelerating, {brake_count} braking\n")

        delta_a_dict = new_delta

    if verbose:
        print(f"{'─'*70}")
        print(f"  V = integral of softplus_β(c−TTC) over all pairs×horizon")
        print(f"  min TTC = actual worst-case time gap in seconds\n")
    return delta_a_dict, last_min_ttc
