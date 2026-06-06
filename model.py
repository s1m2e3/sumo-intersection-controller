import torch
import torch.nn as nn

V_TURN_LOW  = 8.0   # m/s — turning gate opens below this speed (no braking needed)
V_TURN_HIGH = 11.0  # m/s — turning gate fully active above this speed

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

class KernelCorrection(nn.Module):
    """
    Physics-anchored correction via kernel interpolation.

    TTC domain partition:
        [0, 3)   → μ_dec active  →  output anchored to u_dec
        [3, 5]   → both μ = 0   →  NN f̂_θ works alone (learning region)
        (5, 6]   → μ_ff ramps   →  output transitioning to u_ff
        (6, ∞)   → μ_ff = 1     →  output anchored to u_ff

    The correction:
        h_= = μ_dec(z)·(u_dec − f̂_θ) + μ_ff(z)·(u_ff − f̂_θ)

    Full model:
        f = f̂_θ + h_=
          = (1 − μ_dec − μ_ff)·f̂_θ  +  μ_dec·u_dec  +  μ_ff·u_ff

    K_{zz} = I by construction (inducing points chosen so the kernel matrix
    is the identity), so K_{zz}^{-1} disappears from the formula.
    """

    def __init__(self, physics: IDMPhysics, eps: float = 1e-3):
        super().__init__()
        self.physics = physics
        self.eps     = eps

    def ttc_star(self, v: torch.Tensor, gap: torch.Tensor,
                 v_lead: torch.Tensor) -> torch.Tensor:
        """
        Symmetric TTC*:
            TTC* = |gap| / max(v - v_lead, ε)
        Using absolute value in the numerator makes the kernel symmetric.
        When v ≤ v_lead (no approach), dv = ε → TTC* → large → free-flow regime.
        """
        dv = torch.clamp(v - v_lead, min=self.eps)
        return gap.abs() / dv

    @staticmethod
    def mu_ff(z: torch.Tensor) -> torch.Tensor:
        """Active in (5, ∞). Zero on [0, 5], ramps to 1 at z = 6."""
        return torch.clamp(z - 5.0, 0.0, 1.0)

    @staticmethod
    def mu_dec(z: torch.Tensor) -> torch.Tensor:
        """Active in [0, 3). Equals 1 on [0, 2.5], ramps to 0 at z = 3.
        Steeper ramp (2x) means full braking held longer before releasing."""
        return torch.clamp(2.0 * (3.0 - z), 0.0, 1.0)

    @staticmethod
    def mu_turn(v: torch.Tensor, is_turning: torch.Tensor) -> torch.Tensor:
        """
        Active only for turning vehicles (_L / _R streams).
        Ramps 0 → 1 as speed goes from V_TURN_LOW to V_TURN_HIGH.
        """
        ramp = torch.clamp(
            (v - V_TURN_LOW) / (V_TURN_HIGH - V_TURN_LOW), 0.0, 1.0)
        return is_turning * ramp

    # ── waypoint anchor ───────────────────────────────────────────────────────

    @staticmethod
    def u_waypoint(
        v:       torch.Tensor,
        d_wp:    torch.Tensor,   # distance to waypoint along road (m), > 0
        v_des:   torch.Tensor,   # desired speed at that waypoint (m/s)
        omega_n: float = 0.5,    # natural frequency  (rad/s)
        zeta:    float = 1.2,    # damping ratio (slightly overdamped)
    ) -> torch.Tensor:
        """
        Second-order mass-damped tracking acceleration:
            a = ω_n² · d_wp + 2ζω_n · (v_des − v)

        Position term pulls vehicle toward the waypoint; velocity term regulates
        speed to v_des.  Clamped to [-3, +2] m/s² to stay within physical limits.
        """
        k_p = omega_n ** 2
        k_d = 2.0 * zeta * omega_n
        return torch.clamp(k_p * d_wp + k_d * (v_des - v), min=-3.0, max=2.0)

    def forward(self, v: torch.Tensor, gap: torch.Tensor,
                v_lead: torch.Tensor, f_hat: torch.Tensor,
                is_turning: torch.Tensor | None = None,
                social_a:   torch.Tensor | None = None,
                waypoints:  "list | None"        = None,
                social_2d:  torch.Tensor | None  = None,
                mu_social:  torch.Tensor | None  = None) -> torch.Tensor:
        """
        waypoints : list of 2 × (d_wp [N], v_des [N]) — closest road waypoints.
                    When provided, replaces u_ff so waypoint tracking governs
                    the free-road (TTC* > 5) regime.

        social_2d : [N] ≤ 0 — 2-D vector social braking acceleration from
                    social_force.compute_social_force_2d().
        mu_social : [N] ∈ [0,1] — kernel membership for social_2d anchor.
                    Applied as:  h_eq += μ_social · (u_social − f̂_θ)
                    This is additive alongside μ_dec / μ_ff (no budget gate)
                    because cross-traffic threats are orthogonal to the
                    longitudinal IDM regime and must not be suppressed.
        """
        z     = self.ttc_star(v, gap, v_lead)
        μ_ff  = self.mu_ff(z)
        μ_dec = self.mu_dec(z)

        u_dec = self.physics.u_dec(v, gap, v_lead)

        if waypoints is not None:
            # Two closest waypoints: compute per-waypoint tracking accelerations,
            # take the maximum (most favourable for forward progress / speed match).
            d1, v_des1 = waypoints[0]
            d2, v_des2 = waypoints[1]
            u1   = self.u_waypoint(v, d1, v_des1)
            u2   = self.u_waypoint(v, d2, v_des2)
            u_ff = torch.max(u1, u2)
        else:
            u_ff = self.physics.u_ff(v)

        # in [3,5]: both μ = 0 → h_= = 0 → f = f̂_θ alone
        h_eq = μ_dec * (u_dec - f_hat) + μ_ff * (u_ff - f_hat)

        if is_turning is not None:
            # Budget gate: μ_turn only fills the NN-only region so it cannot
            # stack additively on top of μ_dec or μ_ff and invert f̂_θ.
            budget   = torch.clamp(1.0 - μ_dec - μ_ff, 0.0, 1.0)
            μ_t      = self.mu_turn(v, is_turning) * budget
            u_t      = self.physics.u_turn(v)
            h_eq     = h_eq + μ_t * (u_t - f_hat)

        if social_a is not None:
            # Legacy RKHS-based social force — added directly (no membership gate).
            h_eq = h_eq + social_a

        if social_2d is not None and mu_social is not None:
            # 2-D vector social force (kernel interpolation anchor).
            # Pulls output from f̂_θ toward the social braking target u_social,
            # weighted by how urgently cross-traffic threatens (μ_social).
            h_eq = h_eq + mu_social * (social_2d - f_hat)

        return h_eq


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid model:  f = f̂_θ + h_=
# ──────────────────────────────────────────────────────────────────────────────

class HybridModel(nn.Module):
    """
    Full model combining a GRU-based f̂_θ and the physics-anchored h_=.

    f̂_θ is a GRU over the last seq_len observations (v, gap, v_lead),
    giving the model temporal context: gap trend, leader deceleration,
    ego momentum — things an instantaneous MLP cannot detect.

    Input per vehicle: sequence of seq_len × (v, gap, v_lead)
    Output: scalar acceleration.

    Behaviour by TTC* region:
        [0, 3)  → physics braking   (u_dec)
        [3, 5]  → learned dynamics  (f̂_θ)
        (5, ∞)  → physics free-flow (u_ff)
    """

    def __init__(self, hidden_dim: int = 64, seq_len: int = 5, **physics_kwargs):
        super().__init__()

        self.seq_len    = seq_len
        self.physics    = IDMPhysics(**physics_kwargs)
        self.correction = KernelCorrection(self.physics)

        # GRU f̂_θ: processes last seq_len observations → scalar acceleration
        self.gru  = nn.GRU(3, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def _f_hat(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq : [N, seq_len, 3]  —  sequence of (v, gap, v_lead)
        returns: [N]              —  f̂ ∈ (-2, +2) m/s²
        """
        out, _ = self.gru(x_seq)                       # [N, seq_len, hidden_dim]
        f_raw  = self.head(out[:, -1, :]).squeeze(-1)  # [N], unbounded
        return torch.tanh(f_raw) * 2.0

    def forward(self, v: torch.Tensor, gap: torch.Tensor,
                v_lead: torch.Tensor, x_seq: torch.Tensor,
                is_turning: torch.Tensor | None = None,
                social_a:   torch.Tensor | None = None,
                waypoints:  "list | None"        = None,
                social_2d:  torch.Tensor | None  = None,
                mu_social:  torch.Tensor | None  = None) -> torch.Tensor:
        """
        v, gap, v_lead : [N]             — current timestep scalars
        x_seq          : [N, seq_len, 3] — last seq_len observations
        is_turning     : [N] float 0/1   — 1 for _L / _R stream vehicles
        social_a       : [N] float ≤ 0   — legacy RKHS repulsion (m/s²)
        waypoints      : list of 2 (d_wp [N], v_des [N]) for closest wps
        social_2d      : [N] float ≤ 0   — 2-D vector social braking
        mu_social      : [N] ∈ [0,1]     — kernel membership for social_2d
        returns        : [N]             — acceleration (m/s²)
        """
        f_hat = self._f_hat(x_seq)                      # [N], ∈ (-2, +2)

        # Smooth speed cap: suppress positive f̂ as v → v_max
        v_max = self.physics.v_max.detach()
        gate  = torch.sigmoid(15.0 * (0.8 - v / v_max))
        f_hat = f_hat - torch.relu(f_hat) * (1.0 - gate)

        h_eq  = self.correction(v, gap, v_lead, f_hat, is_turning,
                                social_a=social_a, waypoints=waypoints,
                                social_2d=social_2d, mu_social=mu_social)
        return f_hat + h_eq


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
