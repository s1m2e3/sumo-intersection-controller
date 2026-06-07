"""
analyze_gradients.py — Measure the NN-active domain of the new hybrid model.

Runs the FULL controller (waypoint+IDM + social force) so vehicle trajectories
are safe (0 collisions), then at each step records the gate values:

    mu_wp       = max(mu_dec(TTC*), mu_approach(d_junction))
    mu_conflict = urgency * (1 - beta * ratio)   [from social_force, platoon-adjusted]
    nn_weight   = (1 - mu_wp) * (1 - mu_conflict)

Reports: what fraction of the state-space has nn_weight > 0, and what the
vehicle state looks like in that region (speed, TTC*, d_junction, road type).

Run:
    conda run -n car-following-sumo python analyze_gradients.py
"""
from __future__ import annotations
import os, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import traci

try:
    import sumo as _sumo_pkg
    SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
except ImportError:
    SUMO_BIN = Path(os.environ.get("SUMO_HOME", "")) / "bin"

os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
SUMO_DIR = Path("sumo_files")

# ── controller constants (must match demo_intersection.py) ─────────────────────
SIM_SECONDS  = 120.0
WARMUP_SEC   = 5.0
DT           = 0.2
FLOW_VPH     = 900

V_MAX        = 13.89
V_APPROACH   = 8.0
ARM_LENGTH   = 200.0
B_COMFORT    = 3.0
OMEGA_N      = 0.5
ZETA         = 1.2
K_D          = 2.0 * ZETA * OMEGA_N
IDM_ACCEL    = 2.6
IDM_BRAKE    = 4.5
IDM_S0       = 2.0
IDM_T        = 1.5
IDM_DELTA    = 4.0
ETA_RIVAL_THRESHOLD = 10.0

# ── model gate constants ───────────────────────────────────────────────────────
D_APPROACH   = 80.0    # m — approach gate distance (matches model.py)
_CX, _CY     = 200.0, 200.0
EPS_TTC      = 1e-3

_INCOMING = frozenset({"east_in", "west_in", "north_in", "south_in"})

from model import V_TURN_LOW, V_TURN_HIGH


def _bin(name):
    return str(SUMO_BIN / (name + (".exe" if os.name == "nt" else "")))


# ── IDM + waypoint (same as demo_intersection.py) ─────────────────────────────

def _idm_accel(v, gap, v_lead):
    dv     = v - v_lead
    s_star = IDM_S0 + v * IDM_T + v * dv / (2.0 * math.sqrt(IDM_ACCEL * IDM_BRAKE))
    s_star = max(s_star, IDM_S0)
    a      = IDM_ACCEL * (1.0 - (v / V_MAX)**IDM_DELTA - (s_star / max(gap, 0.1))**2)
    return float(np.clip(a, -IDM_BRAKE, IDM_ACCEL))


def _waypoint_accel(vid, active, has_rival):
    import traci_cache
    road     = traci_cache.get_road_id(vid)
    lane_pos = traci_cache.get_lane_pos(vid)
    v        = traci_cache.get_speed(vid)

    leader = traci.vehicle.getLeader(vid)
    if leader:
        lid, gap = leader
        v_lead   = traci_cache.get_speed(lid) if lid in active else v
        a_idm    = _idm_accel(v, gap, v_lead)
    else:
        a_idm = _idm_accel(v, 1000.0, v)

    if road in _INCOMING and has_rival:
        d        = max(ARM_LENGTH - lane_pos, 0.0)
        v_kin    = math.sqrt(V_APPROACH**2 + 2.0 * B_COMFORT * d)
        v_target = min(v_kin, V_MAX)
    else:
        v_target = V_MAX

    a_wp = float(np.clip(K_D * (v_target - v), -B_COMFORT, IDM_ACCEL))
    return min(a_idm, a_wp)


# ── gate helpers ───────────────────────────────────────────────────────────────

def _mu_dec(ttc):
    return float(np.clip(2.0 * (3.0 - ttc), 0.0, 1.0))


def _mu_approach(d, road):
    if road not in _INCOMING:
        return 0.0
    return float(np.clip(1.0 - d / D_APPROACH, 0.0, 1.0))


def main():
    from conflict import build_snapshot, clear_route_cache, CONFLICT_MAP
    from social_force import compute_social_force_2d, _ALL_MOVEMENTS
    _TURNING = frozenset({
        ("east_in", "north_out"), ("east_in", "south_out"),
        ("west_in", "south_out"), ("west_in", "north_out"),
        ("north_in", "west_out"), ("north_in", "east_out"),
        ("south_in", "east_out"), ("south_in", "west_out"),
    })
    import traci_cache

    # Write routes (sym-900, all 12 movements)
    vph  = FLOW_VPH
    turn = max(1, int(vph * 0.2))
    routes = SUMO_DIR / "routes_grad_analysis.xml"
    routes.write_text(f"""<?xml version="1.0" ?>
<routes>
  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="{V_MAX}"/>
  <flow id="flow_ew_t" type="car" from="east_in"  to="west_out"  begin="5.0" end="300" vehsPerHour="{vph}"  departLane="0" departSpeed="desired"/>
  <flow id="flow_ew_r" type="car" from="east_in"  to="north_out" begin="5.1" end="300" vehsPerHour="{turn}" departLane="0" departSpeed="desired"/>
  <flow id="flow_ew_l" type="car" from="east_in"  to="south_out" begin="5.2" end="300" vehsPerHour="{turn}" departLane="1" departSpeed="desired"/>
  <flow id="flow_we_t" type="car" from="west_in"  to="east_out"  begin="5.3" end="300" vehsPerHour="{vph}"  departLane="0" departSpeed="desired"/>
  <flow id="flow_we_r" type="car" from="west_in"  to="south_out" begin="5.4" end="300" vehsPerHour="{turn}" departLane="0" departSpeed="desired"/>
  <flow id="flow_we_l" type="car" from="west_in"  to="north_out" begin="5.5" end="300" vehsPerHour="{turn}" departLane="1" departSpeed="desired"/>
  <flow id="flow_ns_t" type="car" from="north_in" to="south_out" begin="5.6" end="300" vehsPerHour="{vph}"  departLane="0" departSpeed="desired"/>
  <flow id="flow_ns_r" type="car" from="north_in" to="west_out"  begin="5.7" end="300" vehsPerHour="{turn}" departLane="0" departSpeed="desired"/>
  <flow id="flow_ns_l" type="car" from="north_in" to="east_out"  begin="5.8" end="300" vehsPerHour="{turn}" departLane="1" departSpeed="desired"/>
  <flow id="flow_sn_t" type="car" from="south_in" to="north_out" begin="5.9" end="300" vehsPerHour="{vph}"  departLane="0" departSpeed="desired"/>
  <flow id="flow_sn_r" type="car" from="south_in" to="east_out"  begin="6.0" end="300" vehsPerHour="{turn}" departLane="0" departSpeed="desired"/>
  <flow id="flow_sn_l" type="car" from="south_in" to="west_out"  begin="6.1" end="300" vehsPerHour="{turn}" departLane="1" departSpeed="desired"/>
</routes>
""")
    cfg = SUMO_DIR / "grad_analysis.sumocfg"
    cfg.write_text(f"""<?xml version="1.0" ?>
<configuration>
  <input>
    <net-file    value="intersection.net.xml"/>
    <route-files value="{routes.name}"/>
  </input>
  <time><begin value="0"/><end value="300"/><step-length value="{DT}"/></time>
  <report><no-step-log value="true"/><verbose value="false"/></report>
</configuration>
""")

    traci.start([_bin("sumo"), "-c", str(cfg),
                 "--step-length", str(DT),
                 "--collision.action", "warn",
                 "--collision.check-junctions",
                 "--no-step-log"])

    clear_route_cache()
    traci_cache.clear()

    total_steps   = int((WARMUP_SEC + SIM_SECONDS) / DT)
    control_start = int(WARMUP_SEC / DT)
    known: set[str] = set()
    total_cols = 0
    records: list[dict] = []

    for step in range(total_steps):
        traci.simulationStep()
        all_vids = list(traci.vehicle.getIDList())
        traci_cache.update(all_vids)

        for vid in all_vids:
            if vid not in known:
                traci.vehicle.setSpeedMode(vid, 0)
                known.add(vid)
        known.intersection_update(all_vids)

        if step < control_start or not all_vids:
            continue

        snap = build_snapshot(all_vids)
        all_tracked = [v for v, s in snap.vehicle_stream.items() if s in _ALL_MOVEMENTS]

        # ── social force (same as demo_intersection.py) ────────────────────────
        if all_tracked:
            a_soc_t, mu_cf_t, _, _ = compute_social_force_2d(all_tracked, snap)
            soc_map = {vid: (float(a_soc_t[j]), float(mu_cf_t[j]))
                       for j, vid in enumerate(all_tracked)}
        else:
            soc_map = {}

        # ── ETA rival gate (same as demo_intersection.py) ──────────────────────
        min_rival_eta: dict = {}
        for stream, svids in snap.stream_vehicles.items():
            if stream not in _ALL_MOVEMENTS or not svids:
                continue
            min_eta = float("inf")
            for rvid in svids:
                px, py = traci.vehicle.getPosition(rvid)
                spd    = traci_cache.get_speed(rvid)
                dist   = math.sqrt((_CX - px)**2 + (_CY - py)**2)
                min_eta = min(min_eta, dist / max(spd, 0.1))
            min_rival_eta[stream] = min_eta

        active = set(all_vids)
        for vid in all_vids:
            v          = traci_cache.get_speed(vid)
            road       = traci_cache.get_road_id(vid)
            px, py     = traci.vehicle.getPosition(vid)
            d_jct      = math.sqrt((_CX - px)**2 + (_CY - py)**2)
            in_jct     = road.startswith(":center")
            ego_stream = snap.vehicle_stream.get(vid)

            # ── apply controller (identical to demo_intersection.py) ───────────
            if ego_stream in _ALL_MOVEMENTS:
                rival_streams = CONFLICT_MAP.get(ego_stream, frozenset()) & _ALL_MOVEMENTS
                has_rival = any(
                    min_rival_eta.get(s, float("inf")) < ETA_RIVAL_THRESHOLD
                    for s in rival_streams
                )
            else:
                has_rival = True

            a_wp = _waypoint_accel(vid, active, has_rival)

            if ego_stream in _TURNING and v > V_TURN_LOW:
                mu_t = min((v - V_TURN_LOW) / (V_TURN_HIGH - V_TURN_LOW), 1.0)
                u_t  = float(np.clip(
                    IDM_ACCEL * (1.0 - (v / V_TURN_LOW)**IDM_DELTA),
                    -B_COMFORT, 0.0))
                a_wp = a_wp + mu_t * u_t

            a_soc, mu_cf = soc_map.get(vid, (0.0, 0.0))
            a      = min(a_wp, a_soc) if a_soc < 0.0 else a_wp
            v_next = float(np.clip(v + a * DT, 0.0, V_MAX))
            traci.vehicle.setSpeed(vid, v_next)

            # ── compute gate values for analysis ───────────────────────────────
            leader = traci.vehicle.getLeader(vid)
            if leader:
                lid, gap  = leader
                v_lead    = traci_cache.get_speed(lid)
                dv        = max(v - v_lead, EPS_TTC)
                ttc       = max(gap, 0.0) / dv
            else:
                ttc = 999.0

            mu_dec_v   = _mu_dec(ttc)
            mu_app_v   = _mu_approach(d_jct, road)
            mu_wp_v    = max(mu_dec_v, mu_app_v)
            nn_weight  = (1.0 - mu_wp_v) * (1.0 - mu_cf)

            records.append({
                "v":           v,
                "ttc":         min(ttc, 999.0),
                "d_jct":       d_jct,
                "mu_wp":       mu_wp_v,
                "mu_dec":      mu_dec_v,
                "mu_approach": mu_app_v,
                "mu_conflict": mu_cf,
                "nn_weight":   nn_weight,
                "road":        ("junction" if in_jct
                                else ("approach" if road in _INCOMING else "other")),
            })

        try:
            total_cols += len(traci.simulation.getCollisions())
        except AttributeError:
            pass

    traci.close()

    print(f"\n  Simulation complete — collisions: {total_cols}")

    if not records:
        print("No records collected.")
        return

    # ── aggregate analysis ─────────────────────────────────────────────────────
    N          = len(records)
    nn_weights = np.array([r["nn_weight"]   for r in records])
    mu_wps     = np.array([r["mu_wp"]       for r in records])
    mu_apps    = np.array([r["mu_approach"] for r in records])
    mu_cfs     = np.array([r["mu_conflict"] for r in records])
    speeds     = np.array([r["v"]           for r in records])
    ttcs       = np.array([r["ttc"]         for r in records])
    d_jcts     = np.array([r["d_jct"]       for r in records])

    active_mask    = nn_weights > 0.05
    dominated_mask = nn_weights < 0.05

    print(f"\n{'='*70}")
    print(f"  Gradient domain analysis  --  {N} vehicle-steps  ({SIM_SECONDS:.0f}s sim)")
    print(f"{'='*70}")

    print(f"\n  NN weight distribution  (d_output/d_f_hat = nn_weight):")
    for lo, hi in [(0.0, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
        mask = (nn_weights >= lo) & (nn_weights < hi)
        pct  = 100.0 * mask.sum() / N
        bar  = "#" * int(pct / 2)
        print(f"    [{lo:.2f}, {hi:.2f})  {pct:5.1f}%  {bar}")

    print(f"\n  Gradient-ACTIVE   (nn_weight > 0.05):  {100*active_mask.mean():.1f}% of steps")
    if active_mask.sum() > 0:
        print(f"    avg speed         : {speeds[active_mask].mean():.2f} m/s")
        ttc_clipped = np.clip(ttcs[active_mask], 0, 30)
        print(f"    avg TTC*          : {ttc_clipped.mean():.2f} s  (clipped at 30)")
        print(f"    avg d_junction    : {d_jcts[active_mask].mean():.1f} m")
        print(f"    avg mu_wp         : {mu_wps[active_mask].mean():.3f}")
        print(f"    avg mu_conflict   : {mu_cfs[active_mask].mean():.3f}")
        road_counts: dict = defaultdict(int)
        for r in records:
            if r["nn_weight"] > 0.05:
                road_counts[r["road"]] += 1
        tot = sum(road_counts.values())
        print(f"    road breakdown    : " +
              "  ".join(f"{k}={100*v/tot:.1f}%" for k, v in sorted(road_counts.items())))

    print(f"\n  Physics-DOMINATED (nn_weight < 0.05): {100*dominated_mask.mean():.1f}% of steps")
    if dominated_mask.sum() > 0:
        print(f"    mu_dec active (>0.5)    : {100*(mu_wps[dominated_mask*(mu_apps<0.3)]>0.5).mean():.1f}%")
        print(f"    mu_approach active(>0.3): {100*(mu_apps[dominated_mask]>0.3).mean():.1f}%")
        print(f"    mu_conflict active(>0.1): {100*(mu_cfs[dominated_mask]>0.1).mean():.1f}%")

    # ── domain map: mean nn_weight by (speed, d_junction) ────────────────────
    print(f"\n  NN weight by zone  (rows=speed, cols=distance to junction):")
    spd_bins = [(0, 4), (4, 7), (7, 10), (10, 14)]
    d_bins   = [(0, 20), (20, 50), (50, 80), (80, 200)]
    cols_hdr = " | ".join(f"d={lo}-{hi}m" for lo, hi in d_bins)
    hdr      = f"  {'speed':>12} | {cols_hdr}"
    print(f"  {'-'*len(hdr)}")
    print(hdr)
    print(f"  {'-'*len(hdr)}")
    for s_lo, s_hi in spd_bins:
        row = f"  {s_lo}-{s_hi} m/s    | "
        cells = []
        for d_lo, d_hi in d_bins:
            mask = ((speeds >= s_lo) & (speeds < s_hi) &
                    (d_jcts >= d_lo) & (d_jcts < d_hi))
            cells.append(f" {nn_weights[mask].mean():.2f} " if mask.sum() >= 5 else "  --- ")
        row += " | ".join(cells)
        print(row)
    print(f"  {'-'*len(hdr)}")
    print("  Values = mean nn_weight  (0.0=physics dominant, 1.0=NN fully free)")

    # ── gradient magnitude perspective ─────────────────────────────────────────
    print(f"\n  Expected gradient scale by zone:")
    print(f"    Mean nn_weight overall          : {nn_weights.mean():.3f}")
    print(f"    Mean nn_weight (approach roads) : {nn_weights[np.array([r['road']=='approach' for r in records])].mean():.3f}")
    print(f"    Mean nn_weight (in junction)    : {nn_weights[np.array([r['road']=='junction' for r in records])].mean():.3f}")
    print(f"    Mean nn_weight (exit/other)     : {nn_weights[np.array([r['road']=='other'    for r in records])].mean():.3f}")
    print(f"\n  Fraction where NN is COMPLETELY free (nn_weight > 0.9) : {100*(nn_weights>0.9).mean():.1f}%")
    print(f"  Fraction where NN is HALF free      (nn_weight > 0.5) : {100*(nn_weights>0.5).mean():.1f}%\n")


if __name__ == "__main__":
    main()
