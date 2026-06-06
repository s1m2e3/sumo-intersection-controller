"""
demo_social_dense.py

Multi-vehicle dense-traffic simulation of the 2-D social force.
12 vehicles spread across all 4 through-movement streams.

Lane layout (right-hand traffic):
  EW_T  y = +2 m   heading East   (+1,  0)
  WE_T  y = -2 m   heading West   (-1,  0)
  NS_T  x = +2 m   heading South  ( 0, -1)
  SN_T  x = -2 m   heading North  ( 0, +1)

Crossing points (lane-lane intersections, not all at origin):
  EW_T × NS_T  →  (+2, +2)
  EW_T × SN_T  →  (-2, +2)
  WE_T × NS_T  →  (+2, -2)
  WE_T × SN_T  →  (-2, -2)

Each vehicle:
  • Gets same-stream IDM car-following (base acceleration).
  • Gets cross-stream 2-D social force (h_= kernel anchor, f̂=0).
  • Total  h_= = IDM_follow  +  μ_social·(u_social − 0)

Runs for A_SOCIAL ∈ {10, 20} and compares:
  • 2-D trajectories (side-by-side)
  • Min TTC_2D over time (across all conflicting pairs)
  • Per-pair collision table
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass, field

# ── constants ───────────────────────────────────────────────────────────────────
RADIUS      = 50.0    # m    social-force radius
T_GATE      = 8.0     # s    membership ramp-off (earlier activation)
_EPS_TTC    = 0.05    # s
VEH_LEN     = 5.0     # m    vehicle length (for collision / clearance checks)
BOX_HALF    = 10.0    # m    half-width of intersection box (cleared = past this)
B_MAX       = 3.0     # m/s² max social-force braking per rival
A_ACC_MAX   = 1.0     # m/s² max social-force acceleration (passer boost)

DT   = 0.05
T    = 20.0

A_VALUES = [20.0, 35.0]

# ── stream geometry ─────────────────────────────────────────────────────────────
HEADINGS = {
    'EW_T': np.array([ 1.0,  0.0]),
    'WE_T': np.array([-1.0,  0.0]),
    'NS_T': np.array([ 0.0, -1.0]),
    'SN_T': np.array([ 0.0,  1.0]),
}

STREAM_COLORS = {
    'EW_T': '#1f77b4',   # blue
    'WE_T': '#d62728',   # red
    'NS_T': '#2ca02c',   # green
    'SN_T': '#ff7f0e',   # orange
}

# Through-movement conflict graph (from conflict.py logic)
CROSS_CONFLICTS = {
    'EW_T': {'NS_T', 'SN_T'},
    'WE_T': {'NS_T', 'SN_T'},
    'NS_T': {'EW_T', 'WE_T'},
    'SN_T': {'EW_T', 'WE_T'},
}

# Fixed lane-intersection points for each (ego_stream, rival_stream) pair
CROSSING_LOOKUP: dict[tuple[str, str], tuple[float, float]] = {
    ('EW_T', 'NS_T'): ( 2.0,  2.0),
    ('EW_T', 'SN_T'): (-2.0,  2.0),
    ('WE_T', 'NS_T'): ( 2.0, -2.0),
    ('WE_T', 'SN_T'): (-2.0, -2.0),
    ('NS_T', 'EW_T'): ( 2.0,  2.0),
    ('NS_T', 'WE_T'): ( 2.0, -2.0),
    ('SN_T', 'EW_T'): (-2.0,  2.0),
    ('SN_T', 'WE_T'): (-2.0, -2.0),
}

# ── initial vehicle specs ───────────────────────────────────────────────────────
# (stream, x0, y0, v0, label)
# EW_T: y=+2, heading East  — x increases toward intersection (x~+2 is conflict zone)
# WE_T: y=-2, heading West  — x decreases toward intersection (x~-2 is conflict zone)
# NS_T: x=+2, heading South — y decreases toward intersection (y~+2 is conflict zone)
# SN_T: x=-2, heading North — y increases toward intersection (y~-2 is conflict zone)
VEHICLE_SPECS = [
    # EW_T  — 3 vehicles approaching from the west
    ('EW_T', -30.0,  2.0, 13.0, 'EW1'),
    ('EW_T', -56.0,  2.0, 11.0, 'EW2'),
    ('EW_T', -82.0,  2.0, 12.5, 'EW3'),

    # WE_T  — 3 vehicles approaching from the east
    ('WE_T',  35.0, -2.0, 12.0, 'WE1'),
    ('WE_T',  60.0, -2.0, 13.0, 'WE2'),
    ('WE_T',  85.0, -2.0, 11.5, 'WE3'),

    # NS_T  — 3 vehicles approaching from the north
    ('NS_T',  2.0,  28.0, 11.0, 'NS1'),
    ('NS_T',  2.0,  55.0, 12.5, 'NS2'),
    ('NS_T',  2.0,  80.0, 13.0, 'NS3'),

    # SN_T  — 3 vehicles approaching from the south
    ('SN_T', -2.0, -33.0, 12.0, 'SN1'),
    ('SN_T', -2.0, -58.0, 11.0, 'SN2'),
    ('SN_T', -2.0, -83.0, 12.5, 'SN3'),
]


# ── vehicle dataclass ───────────────────────────────────────────────────────────

@dataclass
class Vehicle:
    stream: str
    pos:    np.ndarray
    v:      float
    label:  str
    e:      np.ndarray = field(init=False)
    cleared: bool = False

    def __post_init__(self):
        self.e = HEADINGS[self.stream].copy()

    def vel_vec(self) -> np.ndarray:
        return self.v * self.e

    def mark_cleared(self):
        """Flag once the vehicle is past the intersection box."""
        x, y = self.pos
        if self.stream == 'EW_T' and x >  BOX_HALF: self.cleared = True
        if self.stream == 'WE_T' and x < -BOX_HALF: self.cleared = True
        if self.stream == 'NS_T' and y < -BOX_HALF: self.cleared = True
        if self.stream == 'SN_T' and y >  BOX_HALF: self.cleared = True

    def long_pos(self) -> float:
        """Signed position along stream axis (increases toward intersection)."""
        x, y = self.pos
        if self.stream == 'EW_T': return  x
        if self.stream == 'WE_T': return -x
        if self.stream == 'NS_T': return -y
        if self.stream == 'SN_T': return  y
        return 0.0


def make_vehicles() -> list[Vehicle]:
    return [
        Vehicle(stream=s, pos=np.array([x, y], float), v=v, label=lbl)
        for s, x, y, v, lbl in VEHICLE_SPECS
    ]


# ── physics ─────────────────────────────────────────────────────────────────────

def idm_accel(v: float, gap: float, v_lead: float,
              v_max: float = 13.89, a: float = 1.5,
              b: float = 3.0, s0: float = 3.0, T: float = 1.5) -> float:
    """Standard IDM, clamped to [-b, a]."""
    dv     = max(0.0, v - v_lead)
    s_star = s0 + v * T + v * dv / (2.0 * np.sqrt(a * b))
    s_star = max(s_star, s0)
    gap    = max(gap, 0.5)
    raw    = a * (1.0 - (v / max(v_max, 0.1)) ** 4 - (s_star / gap) ** 2)
    return float(np.clip(raw, -b, a))


def _eta_to_cp(v: Vehicle, cx: float, cy: float) -> float:
    """Remaining time (s) for vehicle v to reach crossing point (cx, cy)."""
    dp = np.array([cx, cy]) - v.pos
    dist_along = float(np.dot(dp, v.e))
    return max(0.0, dist_along) / max(v.v, 0.1)


def _eta_weight(eta_i: float, eta_j: float, sigma: float = 0.15) -> float:
    """
    sigmoid((eta_i − eta_j) / σ):
      → 1 when ego arrives later (yielder),  → 0 when ego arrives earlier (passer).
    sigma = 0.15 s: 0.1 s ETA edge → w ≈ 0.74 (meaningful asymmetry on small differences).
    """
    return 1.0 / (1.0 + np.exp(-(eta_i - eta_j) / sigma))


def pair_social(vi: Vehicle, vj: Vehicle, A: float) -> tuple[float, float]:
    """
    ETA-weighted bidirectional social force on vi from vj.
    Yielder (arrives later): braking signal ≤ 0.
    Passer (arrives first):  gentle acceleration ≥ 0.
    Returns (a_ij, ttc_2d).  a_ij clamped to [-B_MAX, A_ACC_MAX].
    """
    rx, ry = vj.pos - vi.pos
    dist   = float(np.hypot(rx, ry))
    if dist > RADIUS or dist < 0.5:
        return 0.0, np.inf

    dvx, dvy = vj.vel_vec() - vi.vel_vec()
    rdv      = rx * dvx + ry * dvy
    if rdv >= 0.0:
        return 0.0, np.inf

    ttc  = dist ** 2 / max(-rdv, _EPS_TTC * dist)
    fmag = A / max(ttc ** 2, _EPS_TTC ** 2)
    dot  = abs((rx / dist) * vi.e[0] + (ry / dist) * vi.e[1])

    key = (vi.stream, vj.stream)
    w = 0.5
    if key in CROSSING_LOOKUP:
        cx, cy  = CROSSING_LOOKUP[key]
        eta_i   = _eta_to_cp(vi, cx, cy)
        eta_j   = _eta_to_cp(vj, cx, cy)
        w       = _eta_weight(eta_i, eta_j)

    # (1-2w): -1 → brake (yielder, w=1), +1 → accelerate (passer, w=0)
    a_ij = (1.0 - 2.0 * w) * fmag * dot
    return float(np.clip(a_ij, -B_MAX, A_ACC_MAX)), ttc


def mu_soc(ttc: float) -> float:
    return float(np.clip(1.0 - ttc / T_GATE, 0.0, 1.0))


# ── simulation loop ─────────────────────────────────────────────────────────────

def simulate(A: float):
    """Run one scenario; return logs for all vehicles."""
    vehicles = make_vehicles()
    N        = len(vehicles)
    steps    = int(T / DT)

    t_log    = np.zeros(steps + 1)
    pos_log  = np.zeros((steps + 1, N, 2))
    v_log    = np.zeros((steps + 1, N))
    ttc_min_log = np.zeros(steps + 1)   # min TTC_2D across all conflict pairs
    sep_min_log = np.zeros(steps + 1)   # min separation across conflict pairs

    def snapshot():
        return np.array([vh.pos.copy() for vh in vehicles]), \
               np.array([vh.v          for vh in vehicles])

    pos_log[0], v_log[0] = snapshot()
    ttc_min_log[0], sep_min_log[0] = _min_metrics(vehicles, A)

    for k in range(steps):
        accels = np.zeros(N)

        for i, vi in enumerate(vehicles):
            if vi.cleared:
                continue

            # ── same-stream IDM ──────────────────────────────────────────────
            # Find the nearest vehicle ahead in the same stream
            a_base  = idm_accel(vi.v, 1000.0, 13.89)  # free-flow when no leader
            min_gap = np.inf
            for j, vj in enumerate(vehicles):
                if i == j or vj.stream != vi.stream:
                    continue
                # gap along stream axis (positive = vj is ahead)
                gap_long = vj.long_pos() - vi.long_pos() - VEH_LEN
                if 0.0 < gap_long < min_gap:
                    min_gap   = gap_long
                    a_base    = idm_accel(vi.v, gap_long, vj.v)

            # ── cross-stream social force ────────────────────────────────────
            rival_pairs = []
            for j, vj in enumerate(vehicles):
                if i == j or vj.cleared:
                    continue
                if vj.stream not in CROSS_CONFLICTS[vi.stream]:
                    continue
                a_ij, ttc_ij = pair_social(vi, vj, A)
                if ttc_ij < np.inf:
                    rival_pairs.append((a_ij, ttc_ij))

            if rival_pairs:
                forces   = [a for a, _ in rival_pairs]
                a_brake  = min((a for a in forces if a < 0), default=0.0)
                a_boost  = max((a for a in forces if a > 0), default=0.0)
                u_social = a_brake + a_boost
                ttc_near = min(t for _, t in rival_pairs)
                mu       = mu_soc(ttc_near)
                h_social = mu * u_social
            else:
                h_social = 0.0

            # speed gate: don't add extra brake if nearly stopped (allows re-acceleration)
            if h_social < 0:
                speed_gate = min(1.0, vi.v / 2.0)
                h_social  *= speed_gate
            accels[i] = a_base + h_social

        # Euler step
        for i, vi in enumerate(vehicles):
            vi.v   = max(0.0, vi.v + accels[i] * DT)
            vi.pos = vi.pos + vi.v * vi.e * DT
            vi.mark_cleared()

        t_log[k + 1]           = (k + 1) * DT
        pos_log[k + 1], v_log[k + 1] = snapshot()
        ttc_min_log[k + 1], sep_min_log[k + 1] = _min_metrics(vehicles, A)

    return t_log, pos_log, v_log, ttc_min_log, sep_min_log, vehicles


def _min_metrics(vehicles: list[Vehicle], A: float) -> tuple[float, float]:
    """Min TTC_2D and min separation across all conflicting pairs."""
    min_ttc = 10.0
    min_sep = 1e9
    for i, vi in enumerate(vehicles):
        if vi.cleared:
            continue
        for j, vj in enumerate(vehicles):
            if j <= i or vj.cleared:
                continue
            if vj.stream not in CROSS_CONFLICTS.get(vi.stream, set()):
                continue
            _, ttc = pair_social(vi, vj, A)
            sep    = float(np.linalg.norm(vj.pos - vi.pos))
            if ttc  < min_ttc: min_ttc = ttc
            if sep  < min_sep: min_sep = sep
    return min_ttc, min_sep


# ── plotting ─────────────────────────────────────────────────────────────────────

def plot_results(results: dict):
    n = len(A_VALUES)
    fig = plt.figure(figsize=(6*n, 11))
    traj_axes = [fig.add_subplot(2, n, k+1) for k in range(n)]
    ax_ttc    = fig.add_subplot(2, n, n+1)
    ax_sep    = fig.add_subplot(2, n, n+2)
    for ax in fig.axes[n+2:]:
        ax.set_visible(False)

    for A, ax_traj in zip(A_VALUES, traj_axes):
        t, pos, v, ttc_min, sep_min, vehicles_end = results[A]
        ax_traj.axhline(0, color='gray', lw=0.4, ls=':')
        ax_traj.axvline(0, color='gray', lw=0.4, ls=':')

        # Intersection box
        box = plt.Rectangle((-BOX_HALF, -BOX_HALF), 2*BOX_HALF, 2*BOX_HALF,
                             fill=True, facecolor='lightyellow', edgecolor='gray',
                             linewidth=1.0, zorder=0)
        ax_traj.add_patch(box)

        for i, (_, x0, y0, v0, lbl) in enumerate(VEHICLE_SPECS):
            stream = VEHICLE_SPECS[i][0]
            c      = STREAM_COLORS[stream]
            ax_traj.plot(pos[:, i, 0], pos[:, i, 1], color=c, lw=1.3, alpha=0.8)
            ax_traj.plot(x0, y0, marker='^', ms=7, color=c, zorder=4)
            ax_traj.annotate(lbl, (x0, y0), fontsize=6,
                             textcoords='offset points', xytext=(4, 3))

        # Crossing markers
        for cx, cy in [(2,2),(-2,2),(2,-2),(-2,-2)]:
            ax_traj.plot(cx, cy, marker='x', ms=7, color='red', zorder=5, lw=1.5)

        ax_traj.set_aspect('equal')
        ax_traj.set_xlim(-100, 100); ax_traj.set_ylim(-100, 100)
        ax_traj.set_xlabel('x (m)'); ax_traj.set_ylabel('y (m)')
        ax_traj.set_title(f'Trajectories  A_SOCIAL = {A:.0f}  '
                          f'(× = crossing points)',
                          fontsize=10)

        # Legend patches
        from matplotlib.patches import Patch
        ax_traj.legend(handles=[
            Patch(color=STREAM_COLORS[s], label=s) for s in STREAM_COLORS
        ], fontsize=8, loc='lower right')
        ax_traj.grid(True, alpha=0.2)

    # ── TTC_2D panel ────────────────────────────────────────────────────────────
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(A_VALUES)))
    color_of = {A: palette[k] for k, A in enumerate(A_VALUES)}
    for A in A_VALUES:
        t, _, _, ttc_min, sep_min, _ = results[A]
        ax_ttc.plot(t, ttc_min, lw=2, color=color_of[A], label=f'A = {A:.0f}')

    ax_ttc.axhline(3.0, color='red',  lw=1.2, ls='--', label='Danger threshold 3 s')
    ax_ttc.axhline(0.0, color='black', lw=0.4)
    ax_ttc.fill_between(results[A_VALUES[0]][0], 0, 3.0, alpha=0.07, color='red')
    ax_ttc.set_xlabel('Time (s)'); ax_ttc.set_ylabel('Min TTC_2D (s, capped at 10)')
    ax_ttc.set_title('Minimum TTC_2D  across all conflicting pairs\n'
                     '(falls = vehicles converging, rises = diverging after crossing)',
                     fontsize=9)
    ax_ttc.legend(fontsize=9); ax_ttc.grid(True, alpha=0.25)
    ax_ttc.set_ylim(-0.2, 10.5)

    # ── Min separation panel ─────────────────────────────────────────────────────
    for A in A_VALUES:
        t, _, _, _, sep_min, _ = results[A]
        ax_sep.plot(t, sep_min, lw=2, color=color_of[A], label=f'A = {A:.0f}')

    ax_sep.axhline(VEH_LEN, color='red', lw=1.2, ls='--',
                   label=f'Vehicle length ({VEH_LEN:.0f} m)')
    ax_sep.set_xlabel('Time (s)'); ax_sep.set_ylabel('Min separation (m)')
    ax_sep.set_title('Minimum 2-D separation across all conflicting pairs',
                     fontsize=9)
    ax_sep.legend(fontsize=9); ax_sep.grid(True, alpha=0.25)
    ax_sep.set_ylim(bottom=0)

    plt.suptitle(
        f'Dense Intersection: 12 vehicles × 4 through streams\n'
        r'$h_{=} = a_{IDM} + \mu_{social}(u_{social} - \hat{f}),\quad \hat{f}=0$'
        r'$\quad u_{social} = w_{ETA} \cdot (-A/\mathrm{TTC}_{2D}^2)(\hat{r}\cdot\hat{e})$'
        f'  |  agg=min  RADIUS={RADIUS}m  T_GATE={T_GATE}s',
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('social_force_dense.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── collision / clearance table ─────────────────────────────────────────────────

def print_table(results: dict):
    print(f"\n{'─'*72}")
    print(f"  {'A':>6}  {'min TTC_2D':>12}  {'min sep (m)':>12}  "
          f"{'unsafe steps':>14}  {'vehicles cleared':>17}")
    print(f"{'─'*72}")

    steps = int(T / DT)
    for A in A_VALUES:
        t, pos, v, ttc_min, sep_min, veh_end = results[A]
        n_unsafe  = int((ttc_min < 3.0).sum())
        n_cleared = sum(1 for vh in veh_end if vh.cleared)
        print(f"  {A:>6.0f}  {ttc_min.min():>12.3f}  {sep_min.min():>12.3f}  "
              f"{n_unsafe:>14d}  {n_cleared:>17d}/{len(VEHICLE_SPECS)}")
    print(f"{'─'*72}")


# ── main ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = {}
    for A in A_VALUES:
        print(f"Simulating A = {A:.0f} …", end='  ', flush=True)
        results[A] = simulate(A)
        print("done")

    print_table(results)
    plot_results(results)
    print('Saved → social_force_dense.png')
