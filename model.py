import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

V_TURN_LOW  = 8.0    # m/s — turning gate opens below this speed (no braking needed)
V_TURN_HIGH = 11.0   # m/s — turning gate fully active above this speed

# ── ego-token feature layout (D_EGO = 4, all values normalized ∈ [0,1] or signed) ──
# slot 0: v / V_MAX                    normalized speed
# slot 1: min(gap, 100) / 100          normalized gap to leader
# slot 2: (v − v_lead) / V_MAX         signed closing speed
# slot 3: max(0, d_jct) / ARM_LENGTH   normalized distance to junction
D_EGO      = 4
ARM_LENGTH = 200.0   # m — road arm length used for d_jct normalisation

# ── own-stream summary token layout (D_SUM = 4) ──────────────────────────────
# slot 0: P_own / P_SCALE              own stream packing pressure ∈ [0, 1]
# slot 1: n_own / N_SCALE              approaching vehicles in own stream ∈ [0, 1]
# slot 2: mean_v_own / V_REF           mean speed of own queue
# slot 3: v_follower / V_REF           immediate follower speed
D_SUM      = 4       # see social_force._stream_summary

# ── rival-stream token layout (D_RIVAL = 4, one token per conflicting stream) ─
# slot 0: P_k / P_SCALE                rival stream packing pressure ∈ [0, 1]
# slot 1: n_k / N_SCALE                approaching vehicles in rival stream ∈ [0, 1]
# slot 2: mean_v_k / V_REF             mean speed of rival stream
# slot 3: mu_k                         worst conflict gate from this stream ∈ [0, 1]
# Streams with no vehicles and no urgency are omitted (variable K per vehicle).
# Callers pad to K_MAX with zero rows; zero tokens carry no information and
# the transformer naturally learns to ignore them (zero input → zero after linear).
D_RIVAL    = 4
K_MAX      = 8       # upper bound on conflicting streams in a 4-way intersection

# ──────────────────────────────────────────────────────────────────────────────
# IDM physics terms (learnable parameters)
# ──────────────────────────────────────────────────────────────────────────────

class IDMPhysics(nn.Module):
    """
    Computes the two physics anchors used by KernelCorrection:

      u_ff  — free-flow acceleration (no leader effect)
      u_dec — braking term (interaction with leader)

    Both come from the IDM equations. Parameters are learnable.

    Inputs (all shape [N]):
      v      — ego speed (m/s)
      gap    — net gap to leader (m)          [gross = gap + l]
      v_lead — leader speed (m/s)
    """

    def __init__(
        self,
        v_max:  float = 13.89,  # desired / max speed (m/s)
        a_max:  float = 1.0,    # max acceleration (m/s²)
        b_max:  float = 3.0,    # comfortable deceleration (m/s²) — matches u_dec clamp floor
        delta:  float = 4.0,    # velocity exponent
        s0:     float = 2.0,    # minimum gap (m)
        T:      float = 1.5,    # safe time headway (s)
        l:      float = 5.0,    # vehicle length — not learned
    ):
        super().__init__()
        self.v_max = nn.Parameter(torch.tensor(v_max))
        self.a_max = nn.Parameter(torch.tensor(a_max))
        self.delta = nn.Parameter(torch.tensor(delta))
        self.s0    = nn.Parameter(torch.tensor(s0))
        self.T     = nn.Parameter(torch.tensor(T))
        self.l     = l      # geometric constant, not learnable
        self.b_max = b_max  # deceleration constant, not learnable

    def u_ff(self, v: torch.Tensor) -> torch.Tensor:
        """Free-flow acceleration. Self-regulates at v_max (positive below, negative above).
        Clamped to [-2, +2] m/s² so the full model output stays within the same bound."""
        raw = self.a_max * (1.0 - (v / self.v_max) ** self.delta)
        return torch.clamp(raw, min=-2.0, max=2.0)

    def u_turn(self, v: torch.Tensor, v_turn: float = V_TURN_LOW) -> torch.Tensor:
        """
        Braking anchor for turning vehicles: same IDM free-flow formula but
        targeting v_turn instead of v_max.  Clamped to [-b_max, 0] — never
        accelerates; releases to zero when v ≤ v_turn.
        """
        raw = self.a_max * (1.0 - (v / max(v_turn, 0.1)) ** self.delta)
        return torch.clamp(raw, min=-self.b_max, max=0.0)

    def u_dec(self, v: torch.Tensor, gap: torch.Tensor,
              v_lead: torch.Tensor) -> torch.Tensor:
        """
        Braking term using the standard IDM desired-gap formula:
            s* = s0 + v·T + v·Δv / (2·√(a·b))
        where Δv = max(v - v_lead, 0) is the closing speed.

        Δv is relu'd so s* only grows when approaching — it can never go negative,
        so u_dec is always ≤ 0 (never spuriously accelerates).
        Hard-clamped to [-b_max, 0] m/s².
        """
        gross_gap  = torch.clamp(gap + self.l, min=0.1)
        dv         = torch.relu(v - v_lead)                          # closing speed ≥ 0
        two_sqrt_ab = 2.0 * torch.sqrt(self.a_max * self.b_max)
        s_star     = self.s0 + v * self.T + v * dv / two_sqrt_ab    # always ≥ 0
        raw        = -self.a_max * (s_star / gross_gap)
        return torch.clamp(raw, min=-self.b_max, max=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Kernel correction  h_=
# ──────────────────────────────────────────────────────────────────────────────

# Must match A_CROSS in social_force.py
U_SOCIAL_ANCHOR = -8.0   # m/s²  — full cross-traffic braking anchor

# Approach gate: mu_wp ramps to 1 within this distance of the junction (m)
D_APPROACH = 30.0


class KernelCorrection(nn.Module):
    """
    Physics-grounded correction.

    u_wp is the unconditional baseline — always present in the output.
    f_hat is a residual correction the NN adds on top of u_wp.
    Two gates control how much of that residual is allowed:

      μ_wp       = max(μ_dec(TTC*), μ_approach(d_junction))
                   Suppresses NN residual when following closely or deep in
                   the approach zone.  At μ_wp = 1 → no NN correction.

      μ_conflict = urgency · (1 − β · platoon_ratio)
                   Blends output toward u_social when conflict is urgent.
                   At μ_conflict = 1 → output = u_social regardless of NN.

    Structure:
        a_long  = u_wp + (1 − μ_wp) · f̂_θ        # NN corrects around physics
        output  = (1 − μ_conflict) · a_long
                +      μ_conflict  · u_social       # conflict overrides toward braking

    Gradient d(output)/d(f̂_θ) = (1 − μ_wp) · (1 − μ_conflict).
    Non-zero whenever neither gate is fully active.
    u_wp is always present: even at f̂_θ = 0, output ≥ u_wp · (1 − μ_conflict).
    """

    def __init__(self, physics: IDMPhysics, eps: float = 1e-3):
        super().__init__()
        self.physics = physics
        self.eps     = eps

    # ── gate helpers (callable externally) ────────────────────────────────────

    def ttc_star(self, v: torch.Tensor, gap: torch.Tensor,
                 v_lead: torch.Tensor) -> torch.Tensor:
        dv = torch.clamp(v - v_lead, min=self.eps)
        return gap.abs() / dv

    @staticmethod
    def mu_dec(z: torch.Tensor) -> torch.Tensor:
        """Full at TTC* ≤ 2.5 s, ramps to 0 at TTC* = 3 s."""
        return torch.clamp(2.0 * (3.0 - z), 0.0, 1.0)

    @staticmethod
    def mu_approach(d_junction: torch.Tensor) -> torch.Tensor:
        """Ramps from 0 at d = D_APPROACH to 1 at the junction entrance."""
        return torch.clamp(1.0 - d_junction / D_APPROACH, 0.0, 1.0)

    def forward(
        self,
        f_hat:       torch.Tensor,   # [N]  NN residual ∈ (−2, +2)
        u_wp:        torch.Tensor,   # [N]  waypoint+IDM baseline (pre-computed)
        mu_wp:       torch.Tensor,   # [N]  ∈ [0,1]  longitudinal residual gate
        mu_conflict: torch.Tensor,   # [N]  ∈ [0,1]  conflict blend gate
    ) -> torch.Tensor:
        u_social = torch.full_like(f_hat, U_SOCIAL_ANCHOR)

        # Suppress positive baseline during conflict: if the physics says accelerate
        # but there is an active conflict, that acceleration is unsafe. We suppress
        # the positive part of u_wp proportional to mu_conflict — a smooth, differentiable
        # product that fades to zero at mu_conflict=1 and leaves u_wp unchanged at 0.
        # This is not a hard clip; it is equivalent to saying the conflict gate also
        # gates the "accelerate" component of the physics anchor.
        u_wp_safe = u_wp - torch.relu(u_wp) * mu_conflict

        # Longitudinal: physics always present, NN adds gated residual
        a_long = u_wp_safe + (1.0 - mu_wp) * f_hat

        # Conflict: blend a_long toward u_social as urgency rises
        return (1.0 - mu_conflict) * a_long + mu_conflict * u_social


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid model:  f = f̂_θ + h_=
# ──────────────────────────────────────────────────────────────────────────────

class HybridModel(nn.Module):
    """
    Physics-grounded hybrid model with transformer over ego history + stream summary.

    Token sequence fed to the transformer (no masking, no padding):
        [ ego_{t-T+1}, ..., ego_{t-1}, ego_t,  stream_summary ]
          oldest ──────────────────────── newest  spatial context
          positions 0 ──────────────────── T-1        T

    The most-recent ego token (position T-1) aggregates the full temporal
    history and the spatial queue context via self-attention, then is read
    out through a zero-init linear head → f̂_θ ≈ 0 at init.

    Output formula (from KernelCorrection):
        u_wp_safe = u_wp − relu(u_wp) · μ_conflict   (no acceleration into conflict)
        a_long    = u_wp_safe + (1 − μ_wp) · f̂_θ
        output    = (1 − μ_conflict) · a_long + μ_conflict · u_social

    At f̂_θ = 0 → output = physics baseline → untrained model is safe by construction.
    """

    def __init__(
        self,
        d_model:    int   = 64,
        nhead:      int   = 4,
        num_layers: int   = 2,
        dim_ff:     int   = 128,
        seq_len:    int   = 10,
        f_scale:    float = 1.5,   # NN residual bound (m/s²)
        **physics_kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.f_scale = f_scale
        self.physics    = IDMPhysics(**physics_kwargs)
        self.correction = KernelCorrection(self.physics)

        # Three separate input projections — all project to d_model so tokens
        # can attend to each other freely in the shared transformer.
        self.ego_proj   = nn.Linear(D_EGO,   d_model)  # temporal ego history
        self.sum_proj   = nn.Linear(D_SUM,   d_model)  # own-stream summary
        self.rival_proj = nn.Linear(D_RIVAL, d_model)  # per rival-stream tokens

        # Learned positional embeddings: T ego + 1 own-summary + K_MAX rival slots.
        # Initialised near zero so early training is stable.
        self.pos_embed = nn.Parameter(torch.zeros(seq_len + 1 + K_MAX, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Pre-norm transformer (norm_first=True) — more stable for small models.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Zero-init head → f̂_θ = 0 at init → pure physics baseline from day 0
        self.head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _f_hat(
        self,
        x_seq:         torch.Tensor,   # [N, T, D_EGO]      oldest → newest
        own_summary:   torch.Tensor,   # [N, D_SUM]          own-stream context
        rival_tokens:  torch.Tensor,   # [N, K, D_RIVAL]     K rival-stream tokens (padded)
    ) -> torch.Tensor:
        """Returns f̂_θ [N] ∈ (−f_scale, +f_scale) m/s².

        Sequence layout (left → right = earlier → later in position):
          [ ego×T  |  own_summary×1  |  rival_k×K ]

        Read-out: position T-1 (most-recent ego token) — has attended to the full
        history, own platoon context, and all rival queues via self-attention.
        Zero-padded rival slots carry all-zero inputs → zero after the linear proj,
        no information injected (the model learns to ignore them naturally).
        """
        T = x_seq.shape[1]
        K = rival_tokens.shape[1]

        ego_tokens = self.ego_proj(x_seq)                         # [N, T, d]
        sum_token  = self.sum_proj(own_summary.unsqueeze(1))      # [N, 1, d]
        riv_tokens = self.rival_proj(rival_tokens)                # [N, K, d]

        ego_tokens = ego_tokens + self.pos_embed[:T]              # [N, T, d]
        sum_token  = sum_token  + self.pos_embed[T : T + 1]       # [N, 1, d]
        riv_tokens = riv_tokens + self.pos_embed[T + 1 : T + 1 + K]  # [N, K, d]

        tokens = torch.cat([ego_tokens, sum_token, riv_tokens], dim=1)  # [N, T+1+K, d]

        # Gradient checkpointing: recomputes attention activations during backward
        # instead of storing them. Saves ~30× memory per step — critical for BPTT
        # over long windows on the RTX 2050 (4 GB VRAM).
        if self.training and tokens.requires_grad:
            out = grad_checkpoint(self.transformer, tokens, use_reentrant=False)
        else:
            out = self.transformer(tokens)                               # [N, T+1+K, d]

        h     = out[:, T - 1, :]    # read from last ego token           [N, d]
        f_raw = self.head(h).squeeze(-1)                                 # [N]
        return torch.tanh(f_raw) * self.f_scale

    def forward(
        self,
        x_seq:         torch.Tensor,   # [N, T, D_EGO]   ego history (normalised)
        own_summary:   torch.Tensor,   # [N, D_SUM]       own-stream context
        rival_tokens:  torch.Tensor,   # [N, K, D_RIVAL]  rival-stream tokens (padded to K_MAX)
        u_wp:          torch.Tensor,   # [N]               physics baseline
        mu_wp:         torch.Tensor,   # [N]               longitudinal gate
        mu_conflict:   torch.Tensor,   # [N]               conflict gate
    ) -> torch.Tensor:
        """
        Gradient d(output)/d(θ) = (1−μ_wp)·(1−μ_conflict).
        Non-zero whenever neither gate is fully clamped.
        f̂_θ = 0  →  physics baseline exactly (safe at init, safe if NN diverges).
        """
        f_hat = self._f_hat(x_seq, own_summary, rival_tokens)

        # x_seq[:, -1, 0] = v_t / V_MAX (most recent normalised speed)
        v_norm = x_seq[:, -1, 0]
        v_ms   = v_norm * self.physics.v_max.detach()

        # Suppress upward correction as speed approaches 0.8·v_max
        gate_up   = torch.sigmoid(15.0 * (0.8 - v_norm))
        # Suppress downward correction near zero speed (don't hold vehicles stopped)
        gate_down = torch.sigmoid(15.0 * (v_ms - 1.0))

        f_hat = f_hat - torch.relu( f_hat) * (1.0 - gate_up)
        f_hat = f_hat - torch.relu(-f_hat) * (1.0 - gate_down)

        return self.correction(f_hat, u_wp, mu_wp, mu_conflict)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy models (kept for reference)
# ──────────────────────────────────────────────────────────────────────────────

class IDMModel(nn.Module):
    """Full IDM with learnable parameters (used for baseline comparison)."""

    def __init__(self, v0=13.89, T=1.5, a=1.0, b=1.5, delta=4.0, s0=2.0):
        super().__init__()
        self.v0    = nn.Parameter(torch.tensor(v0))
        self.T     = nn.Parameter(torch.tensor(T))
        self.a     = nn.Parameter(torch.tensor(a))
        self.b     = nn.Parameter(torch.tensor(b))
        self.delta = nn.Parameter(torch.tensor(delta))
        self.s0    = nn.Parameter(torch.tensor(s0))

    def forward(self, v, gap, v_lead):
        dv     = v - v_lead
        s_star = self.s0 + v * self.T + v * dv / (2.0 * torch.sqrt(self.a * self.b))
        s_star = torch.clamp(s_star, min=self.s0.detach())
        gap    = torch.clamp(gap, min=0.1)
        return self.a * (1.0 - (v / self.v0) ** self.delta - (s_star / gap) ** 2)


class VehicleDynamicsNet(nn.Module):
    """Residual MLP: learns f(s_t, a_t) -> s_{t+1}."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),             nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, action):
        return state + self.net(torch.cat([state, action], dim=-1))
