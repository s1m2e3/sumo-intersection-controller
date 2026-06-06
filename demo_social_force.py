"""
demo_social_force.py

Standalone 2-D social force simulation — no SUMO required.

Scenario (geometrically asymmetric so the yielder is correctly identified):
  V1  EW_T  (-38, 0)  heading East   v = 13 m/s  →  ETA to centre = 2.92 s
  V2  NS_T  ( 0, 28)  heading South  v = 11 m/s  →  ETA to centre = 2.55 s

V2 arrives FIRST.  V1 is the yielder.

Why the geometry correctly identifies V1 as the yielder:
  V2 is at (0,28) relative to V1 heading East  →  r̂·ê₁ = 38/47.2 = 0.805
  V1 is at (-38,0) relative to V2 heading South →  r̂·ê₂ = 28/47.2 = 0.593
  Larger dot on V1  →  V1 receives stronger braking  →  V1 waits for V2.

Without social force: V2 clears around t=3.9 s, but V1 entered at t=2.92 s
  → overlap in the intersection → collision.

With social force: V1 is delayed, TTC_2D grows, collision avoided.

Four panels:
  [0] 2-D trajectories  (all A values + baseline)
  [1] TTC_2D over time  ← requested key diagnostic
  [2] Speed vs time
  [3] h_= social acceleration vs time (V1 only — the yielder)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── parameters ─────────────────────────────────────────────────────────────────
RADIUS   = 50.0
T_GATE   = 5.0
_EPS_TTC = 0.05

DT = 0.02
T  = 9.0

# V1: EW_T heading East, V2: NS_T heading South
X1_0, Y1_0, V1_0 = -38.0,  0.0, 13.0
X2_0, Y2_0, V2_0 =   0.0, 28.0, 11.0
E1 = np.array([1.0,  0.0])   # east
E2 = np.array([0.0, -1.0])   # south

VEH_LEN = 5.0    # m  (used in clearance annotation only)
A_SWEEP = [0.0, 10.0, 20.0, 40.0]


# ── core social-force math (mirrors social_force.py) ───────────────────────────

def pair_social(px_i, py_i, vx_i, vy_i, ex_i, ey_i,
                px_j, py_j, vx_j, vy_j, A) -> tuple[float, float]:
    """Returns (a_ij ≤ 0, ttc_2d).  (0, +∞) if diverging / out of range."""
    rx, ry = px_j - px_i, py_j - py_i
    dist   = np.hypot(rx, ry)
    if dist > RADIUS or dist < 0.5:
        return 0.0, np.inf
    dvx, dvy = vx_j - vx_i, vy_j - vy_i
    rdv = rx * dvx + ry * dvy
    if rdv >= 0.0:
        return 0.0, np.inf
    ttc  = dist ** 2 / max(-rdv, _EPS_TTC * dist)
    fmag = A / max(ttc ** 2, _EPS_TTC ** 2)
    dot  = (rx / dist) * ex_i + (ry / dist) * ey_i
    return min(0.0, -fmag * dot), ttc


def mu_social(ttc: float) -> float:
    return float(np.clip(1.0 - ttc / T_GATE, 0.0, 1.0))


# ── simulation ──────────────────────────────────────────────────────────────────

def simulate(A: float):
    steps = int(T / DT)
    x1, y1, spd1 = X1_0, Y1_0, V1_0   # V1 moves along x-axis
    x2, y2, spd2 = X2_0, Y2_0, V2_0   # V2 moves along y-axis (negative)

    t_log   = np.zeros(steps + 1)
    p1_log  = np.zeros((steps + 1, 2))
    p2_log  = np.zeros((steps + 1, 2))
    v1_log  = np.zeros(steps + 1)
    v2_log  = np.zeros(steps + 1)
    a1_log  = np.zeros(steps)   # h_= social contribution on V1
    ttc_log = np.zeros(steps + 1)  # TTC_2D between V1 and V2

    p1_log[0] = [x1, y1];  p2_log[0] = [x2, y2]
    v1_log[0] = spd1;       v2_log[0] = spd2

    # initial TTC
    _, ttc0 = pair_social(x1,y1, spd1*E1[0],spd1*E1[1], E1[0],E1[1],
                           x2,y2, spd2*E2[0],spd2*E2[1], 1.0)
    ttc_log[0] = min(ttc0, 10.0)

    for k in range(steps):
        vx1, vy1 = spd1 * E1[0], spd1 * E1[1]
        vx2, vy2 = spd2 * E2[0], spd2 * E2[1]

        # V1 sees V2
        a12, ttc12 = pair_social(x1,y1, vx1,vy1, E1[0],E1[1],
                                  x2,y2, vx2,vy2, A)
        h1 = mu_social(ttc12) * a12   # kernel interpolation: μ·(u−f̂), f̂=0

        # V2 sees V1
        a21, ttc21 = pair_social(x2,y2, vx2,vy2, E2[0],E2[1],
                                  x1,y1, vx1,vy1, A)
        h2 = mu_social(ttc21) * a21

        a1_log[k] = h1
        spd1 = max(0.0, spd1 + h1 * DT)
        spd2 = max(0.0, spd2 + h2 * DT)

        x1 += spd1 * DT
        y2 -= spd2 * DT      # V2 moves south (y decreasing)

        t_log[k+1]   = (k+1) * DT
        p1_log[k+1]  = [x1, y1]
        p2_log[k+1]  = [x2, y2]
        v1_log[k+1]  = spd1
        v2_log[k+1]  = spd2

        # TTC_2D: recompute at new positions
        _, ttc_new = pair_social(x1,y1, spd1*E1[0],spd1*E1[1], E1[0],E1[1],
                                  x2,y2, spd2*E2[0],spd2*E2[1], 1.0)  # A=1 to get ttc only
        ttc_log[k+1] = min(ttc_new, 10.0)

    return t_log, p1_log, p2_log, v1_log, v2_log, a1_log, ttc_log


# ── plot ────────────────────────────────────────────────────────────────────────

def plot_all():
    cmap   = plt.cm.plasma
    colors = [cmap(i / max(len(A_SWEEP)-1, 1)) for i in range(len(A_SWEEP))]
    labels = [f"A = {a:.0f}" if a > 0 else "No force (collision)" for a in A_SWEEP]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_traj, ax_ttc, ax_spd, ax_acc = axes.flat

    print(f"\n{'A_SOCIAL':>10}  {'min TTC_2D (s)':>15}  {'min sep (m)':>12}  {'V1 entry - V2 exit (s)':>22}")
    print('-' * 70)

    for A, color, lbl in zip(A_SWEEP, colors, labels):
        t, p1, p2, v1, v2, a1, ttc = simulate(A)
        lw  = 2.0 if A > 0 else 1.0
        ls  = '-'  if A > 0 else '--'
        alp = 1.0  if A > 0 else 0.6

        # ── trajectories ──────────────────────────────────────────────────────
        ax_traj.plot(p1[:,0], p1[:,1], color=color, lw=lw, ls=ls, alpha=alp,
                     label=f'V1 {lbl}')
        ax_traj.plot(p2[:,0], p2[:,1], color=color, lw=lw, alpha=alp,
                     linestyle=':', label=f'V2 {lbl}')

        # ── TTC_2D ────────────────────────────────────────────────────────────
        ax_ttc.plot(t, ttc, color=color, lw=lw, alpha=alp, label=lbl)

        # ── speed ─────────────────────────────────────────────────────────────
        ax_spd.plot(t, v1, color=color, lw=lw, ls='-',  alpha=alp, label=f'V1 {lbl}')
        ax_spd.plot(t, v2, color=color, lw=lw, ls='--', alpha=alp, label=f'V2 {lbl}')

        # ── h_= accel on V1 ───────────────────────────────────────────────────
        if A > 0:
            ax_acc.plot(t[:-1], a1, color=color, lw=lw, label=lbl)

        # ── diagnostics ───────────────────────────────────────────────────────
        sep = np.linalg.norm(p2 - p1, axis=1)
        min_sep = sep.min()
        min_ttc = ttc[ttc > 0].min() if (ttc > 0).any() else 0.0

        # V1 reaches x=0 (intersection entry) → V2 exits y=0 (y < -VEH_LEN/2)
        v1_entry = t[np.argmax(p1[:,0] >= 0)] if (p1[:,0] >= 0).any() else np.inf
        v2_exit  = t[np.argmax(p2[:,1] <= -VEH_LEN)] if (p2[:,1] <= -VEH_LEN).any() else np.inf
        gap_s    = v1_entry - v2_exit

        col = "COLLISION ⚠" if min_sep < VEH_LEN else "clear ✓"
        print(f"{A:>10.1f}  {min_ttc:>15.3f}  {min_sep:>12.3f}  {gap_s:>+22.3f}  {col}")

    # ── annotations ───────────────────────────────────────────────────────────
    ax_traj.axhline(0, color='gray', lw=0.5, ls=':')
    ax_traj.axvline(0, color='gray', lw=0.5, ls=':')
    ax_traj.scatter([0], [0], c='red', s=120, zorder=5, marker='x',
                    label='Intersection centre')
    ax_traj.scatter([X1_0],[Y1_0], c='k', s=80, marker='^', zorder=5, label='V1 start')
    ax_traj.scatter([X2_0],[Y2_0], c='k', s=80, marker='s', zorder=5, label='V2 start')
    ax_traj.set_aspect('equal'); ax_traj.grid(True, alpha=0.25)
    ax_traj.set_xlabel('x (m)'); ax_traj.set_ylabel('y (m)')
    ax_traj.set_title('2-D Trajectories  (V1 → East, V2 ↓ South)\nV1 is the yielder (larger dot product)')
    ax_traj.legend(fontsize=6, ncol=2)

    ax_ttc.axhline(3.0, color='red',    lw=1.0, ls='--', label='TTC threshold = 3 s')
    ax_ttc.axhline(0.0, color='black',  lw=0.5)
    ax_ttc.set_xlabel('Time (s)'); ax_ttc.set_ylabel('TTC_2D (s, capped at 10)')
    ax_ttc.set_title('TTC_2D over Time\n(rises → vehicles separating, falls → approaching)')
    ax_ttc.legend(fontsize=8); ax_ttc.grid(True, alpha=0.25)
    ax_ttc.set_ylim(-0.2, 10.5)

    ax_spd.set_xlabel('Time (s)'); ax_spd.set_ylabel('Speed (m/s)')
    ax_spd.set_title('Speed vs Time\n(solid=V1 yielder, dashed=V2 passer)')
    ax_spd.legend(fontsize=6, ncol=2); ax_spd.grid(True, alpha=0.25)
    ax_spd.set_ylim(bottom=0)

    ax_acc.axhline(0, color='k', lw=0.6, ls='--')
    ax_acc.fill_between(
        np.linspace(0, T, int(T/DT)), 0, 0,
        alpha=0.0,  # invisible, just for axis shape
    )
    ax_acc.set_xlabel('Time (s)'); ax_acc.set_ylabel('h_= contribution (m/s²)')
    ax_acc.set_title('Social-Force Term on V1 (yielder)\n'
                     r'$h_{=}$ += $\mu_{social}(u_{social} - \hat{f})$, $\hat{f}=0$')
    ax_acc.legend(fontsize=8); ax_acc.grid(True, alpha=0.25)

    plt.suptitle(
        r'2-D Social Force   EW_T × NS_T   —   $u_{social} = -A/\mathrm{TTC}_{2D}^2 \cdot (\hat{r}\cdot\hat{e})$'
        f'\nRADIUS={RADIUS}m  T_GATE={T_GATE}s  V1=({X1_0},{Y1_0}) v={V1_0} m/s  '
        f'V2=({X2_0},{Y2_0}) v={V2_0} m/s',
        fontsize=10,
    )
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.savefig('social_force_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_all()
    print('\nSaved → social_force_demo.png')
