"""
demo_waypoint_tracking.py

Standalone simulation of the h_= waypoint-tracking anchor.
No GRU (f_hat = 0), no leader (TTC* → ∞ → mu_ff = 1 → full waypoint authority).

Road: 60m straight → 90° right curve (R=25m) → 60m straight.
Waypoints spaced ~8m apart; desired speed 10 m/s on straights, 6 m/s in the curve.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection

# ── Road geometry ──────────────────────────────────────────────────────────────

def make_road(n_straight=9, n_curve=8, R=25.0):
    """
    Returns (N, 2) array of (x, y) waypoints:
      Segment A: straight East,  0 → 60 m
      Segment B: 90° right arc,  centre at (60, -R)
      Segment C: straight South, 60 m
    """
    pts = []

    # A — straight East
    for x in np.linspace(0, 60, n_straight):
        pts.append((x, 0.0))

    # B — right arc (angle π/2 → 0, i.e. top → right of circle)
    cx, cy = 60.0, -R
    for theta in np.linspace(np.pi / 2, 0.0, n_curve + 1)[1:]:
        pts.append((cx + R * np.cos(theta),
                    cy + R * np.sin(theta)))

    # C — straight South
    x0, y0 = pts[-1]
    for y in np.linspace(y0, y0 - 60, n_straight)[1:]:
        pts.append((x0, y))

    return np.array(pts)


def arc_lengths(waypoints):
    """Cumulative arc-length for each waypoint."""
    dx = np.diff(waypoints[:, 0])
    dy = np.diff(waypoints[:, 1])
    return np.concatenate([[0.0], np.cumsum(np.hypot(dx, dy))])


def s_to_xy(s, arc_s, waypoints):
    """Interpolate (x, y) from scalar arc-length s."""
    idx = np.clip(np.searchsorted(arc_s, s, side='right') - 1,
                  0, len(arc_s) - 2)
    span = arc_s[idx + 1] - arc_s[idx]
    t = (s - arc_s[idx]) / (span if span > 1e-9 else 1.0)
    return (waypoints[idx] + t * (waypoints[idx + 1] - waypoints[idx]))


# ── Desired speed profile ──────────────────────────────────────────────────────

def desired_speed_at(s, arc_s):
    """
    v_des(s):
      ≤ 40m           → 10 m/s (open straight)
      40 – 60m        → ramp 10 → 6 m/s (approach curve)
      60m – curve end → 6 m/s  (arc)
      curve end – +20m → ramp 6 → 10 m/s (exit)
      beyond          → 10 m/s
    """
    v_fast, v_slow = 10.0, 6.0
    curve_start = arc_s[8]          # end of first straight (index 8)
    curve_end   = arc_s[8 + 8]      # end of arc  (8 straight + 8 arc pts)
    ramp = 20.0

    if s < curve_start - ramp:
        return v_fast
    elif s < curve_start:
        t = (s - (curve_start - ramp)) / ramp
        return v_fast + t * (v_slow - v_fast)
    elif s <= curve_end:
        return v_slow
    elif s < curve_end + ramp:
        t = (s - curve_end) / ramp
        return v_slow + t * (v_fast - v_slow)
    else:
        return v_fast


# ── Second-order mass-damped waypoint tracking (NumPy, mirrors model.py) ──────

def u_waypoint(v, d_wp, v_des, omega_n=0.5, zeta=1.2):
    """
    a = ω_n² · d_wp + 2ζω_n · (v_des − v),  clamped to [−3, +2] m/s²
    """
    k_p = omega_n ** 2
    k_d = 2.0 * zeta * omega_n
    return np.clip(k_p * d_wp + k_d * (v_des - v), -3.0, 2.0)


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(dt=0.05, T=30.0):
    road   = make_road()
    arc_s  = arc_lengths(road)
    S_max  = arc_s[-1]

    # desired speed at every waypoint (for colouring the road later)
    vdes_wp = np.array([desired_speed_at(s, arc_s) for s in arc_s])

    n_steps = int(T / dt)
    s_log   = np.zeros(n_steps + 1)
    v_log   = np.zeros(n_steps + 1)
    a_log   = np.zeros(n_steps)

    s, v = 0.0, 0.0

    for k in range(n_steps):
        # distances from current position s to all waypoints (positive = ahead)
        d_all = arc_s - s
        ahead = np.where(d_all > 0.05)[0]

        if len(ahead) == 0:
            # past last waypoint: coast to a stop
            a = np.clip(-v / dt, -3.0, 0.0)
        else:
            # two closest waypoints ahead
            i1 = ahead[0]
            d1 = d_all[i1]
            vd1 = desired_speed_at(arc_s[i1], arc_s)
            a1 = u_waypoint(v, d1, vd1)

            if len(ahead) >= 2:
                i2  = ahead[1]
                d2  = d_all[i2]
                vd2 = desired_speed_at(arc_s[i2], arc_s)
                a2  = u_waypoint(v, d2, vd2)
                a   = max(a1, a2)          # take the more permissive/aggressive
            else:
                a = a1

        a_log[k] = a
        v = max(0.0, v + a * dt)
        s = min(s + v * dt, S_max)

        s_log[k + 1] = s
        v_log[k + 1] = v

    t    = np.linspace(0.0, T, n_steps + 1)
    xy   = np.array([s_to_xy(s, arc_s, road) for s in s_log])
    vdes = np.array([desired_speed_at(s, arc_s) for s in s_log])

    return road, arc_s, vdes_wp, t, s_log, v_log, a_log, xy, vdes


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(road, arc_s, vdes_wp, t, s_log, v_log, a_log, xy, vdes):
    fig = plt.figure(figsize=(17, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── Panel 1 : road geometry + vehicle trajectory ───────────────────────────
    # Road centreline
    ax1.plot(road[:, 0], road[:, 1], color='#888', lw=3,
             solid_capstyle='round', label='Road centreline', zorder=1)

    # Waypoints coloured by v_des
    sc = ax1.scatter(road[:, 0], road[:, 1],
                     c=vdes_wp, cmap='RdYlGn', s=70, zorder=4,
                     vmin=5.5, vmax=10.5, edgecolors='k', linewidths=0.4)
    plt.colorbar(sc, ax=ax1, label='$v_{des}$ (m/s)', fraction=0.046, pad=0.04)

    # Vehicle trajectory coloured by speed
    points = xy.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    norm   = plt.Normalize(0, 11)
    lc     = LineCollection(segs, cmap='viridis', norm=norm, lw=2, zorder=3)
    lc.set_array(v_log[:-1])
    ax1.add_collection(lc)

    # Start / end markers
    ax1.plot(*xy[0],  'g^', ms=9, zorder=5, label='Start')
    ax1.plot(*xy[-1], 'rs', ms=9, zorder=5, label='End')

    ax1.set_aspect('equal')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Road geometry & trajectory\n(trajectory colour = vehicle speed)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.25)

    # ── Panel 2 : speed ────────────────────────────────────────────────────────
    ax2.plot(t, v_log,  color='steelblue', lw=2,   label='$v$ (vehicle)')
    ax2.plot(t, vdes,   color='tomato',    lw=1.5,
             ls='--', alpha=0.8,           label='$v_{des}$ (road profile)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Speed (m/s)')
    ax2.set_title('Speed vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(bottom=0)

    # ── Panel 3 : acceleration ─────────────────────────────────────────────────
    t_a = t[:-1]
    ax3.plot(t_a, a_log, color='mediumpurple', lw=2)
    ax3.axhline(0, color='k', lw=0.8, ls='--')
    ax3.fill_between(t_a, a_log, 0,
                     where=(a_log >= 0), alpha=0.15, color='green',  label='accel')
    ax3.fill_between(t_a, a_log, 0,
                     where=(a_log <  0), alpha=0.15, color='red',    label='brake')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.set_title('$h_{=}$ waypoint command\n($\\hat{f}=0$, no leader)')
    ax3.legend()
    ax3.grid(True, alpha=0.25)

    out = 'waypoint_tracking_demo.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {out}")


if __name__ == '__main__':
    results = simulate()
    plot(*results)
