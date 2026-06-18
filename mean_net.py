"""
mean_net.py — transformer prior mean f(φ, c) for the kernel-interpolation controller.

The controller's zero-mean GP posterior becomes a conditional Gaussian mean with a
NON-ZERO prior mean supplied by a small transformer:

    a(φ*, c) = f(φ*, c) − k(φ*)ᵀ K⁻¹ ( f(X, c) − y )

f factorizes as  f(φ, c) = head(φ, z),  z = encode(c):  the context embedding z is
computed ONCE per ego per step; only the cheap head sweeps the M anchor points X
(the "correspondence" — anchor evaluations are counterfactual queries "same traffic
context, anchor-level features", well-defined because φ is an explicit head input).

Context c per ego vehicle (what the transformer reasons over):
  • CONFLICT-SET tokens — one per rival with a valid crossing: both parties'
    distances and ETAs to their shared conflict point, the signed ETA margin, the
    rival's resolved role, and the rival's own platoon pressure (its backlog shapes
    how it will behave → future conflict);
  • EGO token — own speed, leader gap & speed, resolved role, distance to the box;
  • REAR PRESSURE — platoon size and count behind within the approach window (the
    pressure building up behind the ego).

Properties guaranteed by construction (see utils.gp_posterior + controller):
  • exactness at anchors holds PER-CONTEXT: at φ* = xᵢ the correction cancels
    head(xᵢ, z) exactly and returns yᵢ for ANY context — no traffic configuration
    can talk the controller out of its anchor prescriptions;
  • far from all anchors k(φ*) → 0 and the output reverts to f(φ*, c), the learned
    policy (not 0 as in the zero-mean model);
  • the head's last layer is ZERO-INITIALIZED ⇒ f ≡ 0 at init ⇒ the controller is
    EXACTLY the validated zero-mean kernel before any training;
  • the head output is tanh-bounded to ±F_LIM so an untrained/adversarial f can
    never command something physically wild before the GP correction even acts.

The downstream gated recurrent correction (sim_torch.rollout_gate) is NOT part of
this module: it consumes the conditional-mean output unchanged, as the outer shield.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

import utils

# ── feature normalization scales (raw SUMO units → O(1) net inputs) ─────────────
D_SCALE   = 100.0          # m    distances (approach is 200 m, assign window 100 m)
V_SCALE   = utils.V0       # m/s  speeds
ETA_SCALE = 10.0           # s    ETAs / ETA margins
P_SCALE   = 10.0           # veh  platoon pressure / counts behind
F_LIM     = 3.0            # m/s² smooth bound on the prior-mean output |f| (tanh squash,
                          #      differentiable everywhere; saturates at ±F_LIM, never exceeds)

EGO_DIM   = 9              # (v, gap, v_lead, role, P, behind_n, d_junc, conf+, conf-)
RIVAL_DIM = 7              # (ego_d, rival_d, v_j, eta_j, eta_margin, role_j, P_j)

# Columns of ego_feats the context-MLP head consumes (kernel-independent extras):
#   v (0), behind-density (5), d_junc (6), conflicting-approach densities conf+ (7), conf- (8)
CONTEXT_COLS = (0, 5, 6, 7, 8)

_ROLE_CODE = {"yield": 0.0, "none": 0.5, "pass": 1.0}


def build_context(vs, gap, v_lead, P, behind_n, d_junc,
                  ego_d, rival_d, valid, roles, axis, sgn):
    """Assemble (ego_feats [N,E], rival_feats [N,N,R], rival_mask [N,N]) from the
    co-sim's per-step tensors.  `roles` is the resolved per-vehicle role list
    ('yield'/'pass'/'none'); `axis`/`sgn` are the per-vehicle travel axis (0=x/EW,
    1=y/NS) and direction sign (±1).  All inputs are the [N]/[N,N] tensors cosim
    already builds; rival rows follow the same convention (entry [i,j] = rival j as
    seen by ego i, masked by `valid`)."""
    role_t = torch.tensor([_ROLE_CODE[r] for r in roles],
                          dtype=vs.dtype, device=vs.device)            # [N]

    # ── conflicting-approach densities ──────────────────────────────────────────
    # count vehicles APPROACHING the box on each (axis, sign) lane; an ego's two
    # conflicting approaches are the perpendicular axis' + and − directions (EW ego
    # → NS and SN).  conf+ / conf- = those two counts, by conflicting-lane sign.
    dt   = vs.dtype
    appr = ((d_junc > 0.0) & (d_junc < 120.0)).to(dt)                  # [N] approaching
    onax0 = (axis == 0).to(dt); onax1 = (axis == 1).to(dt)
    pos   = (sgn > 0).to(dt);   neg = 1.0 - pos
    n0p = (appr * onax0 * pos).sum(); n0n = (appr * onax0 * neg).sum()
    n1p = (appr * onax1 * pos).sum(); n1n = (appr * onax1 * neg).sum()
    # ego on axis 0 conflicts with axis 1 (and vice versa)
    conf_pos = onax0 * n1p + onax1 * n0p                               # [N]
    conf_neg = onax0 * n1n + onax1 * n0n                               # [N]

    ego_feats = torch.stack([
        vs / V_SCALE,
        gap.clamp(max=300.0) / D_SCALE,
        v_lead / V_SCALE,
        role_t,
        P / P_SCALE,
        behind_n / P_SCALE,
        d_junc.clamp(-D_SCALE, 3 * D_SCALE) / D_SCALE,
        conf_pos / P_SCALE,
        conf_neg / P_SCALE,
    ], dim=-1)                                                         # [N, EGO_DIM]

    v_j   = vs.unsqueeze(0).expand_as(rival_d)                         # [N, N]
    eta_e = ego_d / vs.clamp(min=0.1).unsqueeze(1)
    eta_j = rival_d / v_j.clamp(min=0.1)
    rival_feats = torch.stack([
        ego_d.clamp(-D_SCALE, 1e3) / D_SCALE,
        rival_d.clamp(-D_SCALE, 1e3) / D_SCALE,
        v_j / V_SCALE,
        eta_j.clamp(-ETA_SCALE, 10 * ETA_SCALE) / ETA_SCALE,
        (eta_e - eta_j).clamp(-10 * ETA_SCALE, 10 * ETA_SCALE) / ETA_SCALE,
        role_t.unsqueeze(0).expand_as(rival_d),
        P.unsqueeze(0).expand_as(rival_d) / P_SCALE,
    ], dim=-1)                                                         # [N, N, RIVAL_DIM]
    rival_feats = torch.where(valid.unsqueeze(-1), rival_feats,
                              torch.zeros_like(rival_feats))           # zero padded slots
    return ego_feats, rival_feats, valid


class MeanTransformer(nn.Module):
    """enc(c) + head(φ, z).  encode() once per ego per step; head() at the query φ*
    AND the M anchors (batched as [..., M+1, 3] by the controller)."""

    def __init__(self, d_model=32, n_heads=2, d_z=64, d_head=64):
        super().__init__()
        self.rival_embed = nn.Sequential(
            nn.Linear(RIVAL_DIM, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.ego_embed = nn.Linear(EGO_DIM, d_model)
        # learned null token: always-valid key so egos with an empty conflict set
        # attend to SOMETHING (an all-masked key row would NaN the softmax)
        self.null_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.z_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_z), nn.ReLU(), nn.Linear(d_z, d_z))
        self.head_mlp = nn.Sequential(
            nn.Linear(d_z + 3, d_head), nn.ReLU(), nn.Linear(d_head, d_head), nn.ReLU())
        self.head_out = nn.Linear(d_head, 1)
        # ZERO-INIT the output layer ⇒ f ≡ 0 ⇒ controller == validated zero-mean kernel
        nn.init.zeros_(self.head_out.weight)
        nn.init.zeros_(self.head_out.bias)

    def encode(self, ego_feats, rival_feats, rival_mask):
        """ego [N,E], rivals [N,K,R], mask [N,K] (True = real rival) → z [N, d_z]."""
        N = ego_feats.shape[0]
        q  = self.ego_embed(ego_feats).unsqueeze(1)                    # [N, 1, d]
        kv = self.rival_embed(rival_feats)                             # [N, K, d]
        kv = torch.cat([self.null_token.expand(N, 1, -1), kv], dim=1)  # [N, K+1, d]
        pad = torch.cat([torch.zeros(N, 1, dtype=torch.bool, device=rival_mask.device),
                         ~rival_mask], dim=1)                          # True = ignore
        ctx, _ = self.attn(q, kv, kv, key_padding_mask=pad)            # [N, 1, d]
        return self.z_mlp(torch.cat([q.squeeze(1), ctx.squeeze(1)], dim=-1))

    def head(self, phi, z):
        """phi [N, ..., 3] (raw φ = (g, τ_c, r)), z [N, d_z] → f [N, ...]."""
        # normalize φ to O(1) using the kernel's own domain scales
        scale = torch.tensor([utils.G_MAX, utils.TAU_C_MAX, 1.0],
                             dtype=phi.dtype, device=phi.device)
        phi_n = phi / scale
        z_e = z.view(z.shape[0], *([1] * (phi.dim() - 2)), z.shape[-1])
        z_e = z_e.expand(*phi.shape[:-1], z.shape[-1])
        h = self.head_mlp(torch.cat([z_e, phi_n], dim=-1))
        return F_LIM * torch.tanh(self.head_out(h)).squeeze(-1)

    def make_mean_fn(self, z):
        """Closure with the per-step context baked in — the `mean_fn` argument of
        utils.controller_acceleration (called there at [..., M+1, 3])."""
        return lambda phi: self.head(phi, z)


class MeanMLP(nn.Module):
    """Prior mean  f(φ, c) = head([c, φ])  — an MLP with NO conflict-set encoder.

    The kernel input φ = (g_i, τ_i, γ_i) is ALWAYS a head input; `ctx_cols` selects
    extra per-ego scalars from build_context's ego_feats (kernel-independent: the
    kernel/Gram/anchors never see them).  These extras are held fixed per ego and
    broadcast across the M anchor counterfactuals, so anchor-exactness is retained.

      ctx_cols = ()            → context-free f(φ)        (ARCH "mlp")
      ctx_cols = CONTEXT_COLS  → f(φ, v, d_junc, densities) (ARCH "mlp_ctx")

    Same contract either way: output ZERO-INITIALIZED (f ≡ 0 at init ⇒ controller ==
    validated zero-mean kernel) and tanh-bounded to ±F_LIM."""

    def __init__(self, ctx_cols=(), d_head=64):
        super().__init__()
        self.ctx_cols = tuple(ctx_cols)
        self.head_mlp = nn.Sequential(
            nn.Linear(3 + len(self.ctx_cols), d_head), nn.ReLU(),
            nn.Linear(d_head, d_head), nn.ReLU())
        self.head_out = nn.Linear(d_head, 1)
        # ZERO-INIT ⇒ f ≡ 0 ⇒ controller == validated zero-mean kernel before training
        nn.init.zeros_(self.head_out.weight)
        nn.init.zeros_(self.head_out.bias)

    def encode(self, ego_feats, *rest):
        """Return the per-ego context vector z = ego_feats[:, ctx_cols] ([N, C]).
        C=0 (empty) for the context-free variant."""
        return ego_feats[:, list(self.ctx_cols)]

    def head(self, phi, z=None):
        """phi [N, ..., 3] (raw φ = (g, τ_c, r)) → f [N, ...]; z [N, C] broadcast."""
        scale = torch.tensor([utils.G_MAX, utils.TAU_C_MAX, 1.0],
                             dtype=phi.dtype, device=phi.device)
        h_in = phi / scale
        if self.ctx_cols:
            z_e = z.view(z.shape[0], *([1] * (phi.dim() - 2)), z.shape[-1])
            z_e = z_e.expand(*phi.shape[:-1], z.shape[-1])             # broadcast over anchors
            h_in = torch.cat([z_e, h_in], dim=-1)
        return F_LIM * torch.tanh(self.head_out(self.head_mlp(h_in))).squeeze(-1)

    def make_mean_fn(self, z=None):
        return lambda phi: self.head(phi, z)


# ── architecture selector + checkpoint naming ───────────────────────────────────
# Switch the active prior-mean architecture here:
#   "mlp_ctx"     = MLP over φ + (v, d_junc, behind/conflicting densities)  [default]
#   "mlp"         = context-free MLP over φ only
#   "transformer" = conflict-set attention encoder + head
ARCH = "mlp_ctx"


def make_mean_model():
    """Instantiate the active prior-mean architecture (see ARCH)."""
    if ARCH == "transformer":
        return MeanTransformer()
    return MeanMLP(ctx_cols=CONTEXT_COLS if ARCH == "mlp_ctx" else ())


def ckpt_path(kind="best", here=None):
    """Architecture-tagged checkpoint filename so the variants never clobber each
    other.  The transformer keeps the legacy names (mean_net_ckpt.pt / _last.pt)."""
    base = {"best": "ckpt", "last": "last"}[kind]
    tag  = "" if ARCH == "transformer" else f"_{ARCH}"
    name = f"mean_net{tag}_{base}.pt"
    return os.path.join(here, name) if here else name
