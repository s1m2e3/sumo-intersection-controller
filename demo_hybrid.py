"""
demo_hybrid.py — HybridModel in the 4-way intersection (all 12 movements).

Architecture: transformer over 10-step ego history + stream summary token.
  Ego token features  (D_EGO=4, normalised):  v, gap, closing_speed, d_jct
  Summary token (D_SUM=7, normalised):
    own queue:   P_stream, n_approaching, mean_v, v_follower
    rival queue: n_rival, P_rival_max
    conflict:    mu_conflict

u_wp and mu_wp are computed identically to demo_intersection.py.
mu_conflict + stream_summary come from compute_social_force_2d.
Zero-init head → f_hat ≈ 0 → pure physics baseline at init.

Run:
    conda run -n car-following-sumo python demo_hybrid.py [--vph N] [--gui]
"""
from __future__ import annotations
import os, math, argparse
from pathlib import Path
from collections import deque

import numpy as np
import torch
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
PRINT_EVERY  = 25

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
_CX, _CY     = 200.0, 200.0
D_APPROACH   = 80.0       # must match model.py

_INCOMING = frozenset({"east_in", "west_in", "north_in", "south_in"})

from model import HybridModel, V_TURN_HIGH, D_APPROACH, D_EGO, D_SUM, D_RIVAL, K_MAX, ARM_LENGTH as _ARM_LENGTH
from intersection_env import V_LEFT_TURN, V_RIGHT_TURN, _LEFT_TURN_SET, _RIGHT_TURN_SET

GAP_MAX = 100.0   # m — normalisation cap for gap feature in ego tokens

_TURNING = _LEFT_TURN_SET | _RIGHT_TURN_SET

def _v_turn_cap(stream):
    if stream in _LEFT_TURN_SET:  return V_LEFT_TURN
    if stream in _RIGHT_TURN_SET: return V_RIGHT_TURN
    return float("inf")


def _bin(name):
    return str(SUMO_BIN / (name + (".exe" if os.name == "nt" else "")))


# ── IDM + waypoint (identical to demo_intersection.py) ────────────────────────

def _idm_accel(v, gap, v_lead):
    dv     = v - v_lead
    s_star = IDM_S0 + v*IDM_T + v*dv / (2.0*math.sqrt(IDM_ACCEL*IDM_BRAKE))
    s_star = max(s_star, IDM_S0)
    a      = IDM_ACCEL * (1.0 - (v/V_MAX)**IDM_DELTA - (s_star/max(gap, 0.1))**2)
    return float(np.clip(a, -IDM_BRAKE, IDM_ACCEL))


def _waypoint_accel(vid, active, has_rival, traci_cache):
    road     = traci_cache.get_road_id(vid)
    lane_pos = traci_cache.get_lane_pos(vid)
    v        = traci_cache.get_speed(vid)

    leader = traci.vehicle.getLeader(vid)
    if leader:
        lid, gap = leader
        v_lead   = traci_cache.get_speed(lid) if lid in active else v
        a_idm    = _idm_accel(v, gap, v_lead)
    else:
        gap, v_lead = 1000.0, v
        a_idm = _idm_accel(v, 1000.0, v)

    if road in _INCOMING and has_rival:
        d        = max(ARM_LENGTH - lane_pos, 0.0)
        v_kin    = math.sqrt(V_APPROACH**2 + 2.0*B_COMFORT*d)
        v_target = min(v_kin, V_MAX)
    else:
        v_target = V_MAX

    a_wp = float(np.clip(K_D*(v_target - v), -B_COMFORT, IDM_ACCEL))
    return min(a_idm, a_wp), gap, v_lead


def _mu_wp(ttc_star, d_jct, road):
    mu_dec = float(np.clip(2.0*(3.0 - ttc_star), 0.0, 1.0))
    mu_app = float(np.clip(1.0 - d_jct/D_APPROACH, 0.0, 1.0)) if road in _INCOMING else 0.0
    return max(mu_dec, mu_app)


# ── route / config writers ─────────────────────────────────────────────────────

def _write_routes(ew_vph, ns_vph, turn_frac=0.2, suffix=""):
    SUMO_DIR.mkdir(exist_ok=True)
    ew_r = max(1, int(ew_vph*turn_frac)); ew_l = ew_r
    ns_r = max(1, int(ns_vph*turn_frac)); ns_l = ns_r
    p = SUMO_DIR / f"routes_hybrid{suffix}.xml"
    p.write_text(f"""<?xml version="1.0" ?>
<routes>
  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="{V_MAX}"/>
  <flow id="flow_ew_t" type="car" from="east_in"  to="west_out"  begin="5.0" end="300" vehsPerHour="{ew_vph}" departLane="0" departSpeed="desired"/>
  <flow id="flow_ew_r" type="car" from="east_in"  to="north_out" begin="5.1" end="300" vehsPerHour="{ew_r}"   departLane="0" departSpeed="desired"/>
  <flow id="flow_ew_l" type="car" from="east_in"  to="south_out" begin="5.2" end="300" vehsPerHour="{ew_l}"   departLane="1" departSpeed="desired"/>
  <flow id="flow_we_t" type="car" from="west_in"  to="east_out"  begin="5.3" end="300" vehsPerHour="{ew_vph}" departLane="0" departSpeed="desired"/>
  <flow id="flow_we_r" type="car" from="west_in"  to="south_out" begin="5.4" end="300" vehsPerHour="{ew_r}"   departLane="0" departSpeed="desired"/>
  <flow id="flow_we_l" type="car" from="west_in"  to="north_out" begin="5.5" end="300" vehsPerHour="{ew_l}"   departLane="1" departSpeed="desired"/>
  <flow id="flow_ns_t" type="car" from="north_in" to="south_out" begin="5.6" end="300" vehsPerHour="{ns_vph}" departLane="0" departSpeed="desired"/>
  <flow id="flow_ns_r" type="car" from="north_in" to="west_out"  begin="5.7" end="300" vehsPerHour="{ns_r}"   departLane="0" departSpeed="desired"/>
  <flow id="flow_ns_l" type="car" from="north_in" to="east_out"  begin="5.8" end="300" vehsPerHour="{ns_l}"   departLane="1" departSpeed="desired"/>
  <flow id="flow_sn_t" type="car" from="south_in" to="north_out" begin="5.9" end="300" vehsPerHour="{ns_vph}" departLane="0" departSpeed="desired"/>
  <flow id="flow_sn_r" type="car" from="south_in" to="east_out"  begin="6.0" end="300" vehsPerHour="{ns_r}"   departLane="0" departSpeed="desired"/>
  <flow id="flow_sn_l" type="car" from="south_in" to="west_out"  begin="6.1" end="300" vehsPerHour="{ns_l}"   departLane="1" departSpeed="desired"/>
</routes>
""")
    return p


def _write_cfg(routes_name, suffix=""):
    p = SUMO_DIR / f"hybrid{suffix}.sumocfg"
    p.write_text(f"""<?xml version="1.0" ?>
<configuration>
  <input>
    <net-file    value="intersection.net.xml"/>
    <route-files value="{routes_name}"/>
  </input>
  <time><begin value="0"/><end value="300"/><step-length value="{DT}"/></time>
  <report><no-step-log value="true"/><verbose value="false"/></report>
</configuration>
""")
    return p


# ── main run ───────────────────────────────────────────────────────────────────

def run(gui=False, ew_vph=FLOW_VPH, ns_vph=FLOW_VPH, model_path=None, sim_seconds=None, seed=None):
    from conflict import build_snapshot, clear_route_cache, CONFLICT_MAP
    from social_force import compute_social_force_2d, _ALL_MOVEMENTS
    import traci_cache

    # ── model ──────────────────────────────────────────────────────────────────
    model = HybridModel(seq_len=10)
    if model_path and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
        print(f"  Loaded model weights from {model_path}")
    else:
        print("  No weights loaded — running zero-init model (= physics baseline)")
    model.eval()

    SEQ_LEN = model.seq_len
    # Each history entry: (v/V_MAX, gap/GAP_MAX, dv/V_MAX, d_jct/ARM_LENGTH) — normalised
    histories: dict[str, deque] = {}   # vid → deque of 4-tuples

    _sfx   = f"_{os.getpid()}"
    routes = _write_routes(ew_vph, ns_vph, suffix=_sfx)
    cfg    = _write_cfg(routes.name, suffix=_sfx)

    cmd = [_bin("sumo-gui" if gui else "sumo"), "-c", str(cfg),
           "--step-length", str(DT),
           "--collision.action", "warn",
           "--collision.check-junctions",
           "--no-step-log"]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if gui:
        cmd += ["--start", "--quit-on-end", "--delay", "200"]

    traci.start(cmd)
    clear_route_cache()
    traci_cache.clear()

    from train_hybrid import LAMBDA_TTC
    from collections import deque as _deque

    _sim_sec      = sim_seconds if sim_seconds is not None else SIM_SECONDS
    warmup_steps  = int(WARMUP_SEC / DT)
    control_steps = int(_sim_sec / DT)
    known: set[str] = set()
    total_cols = 0
    speed_sum = 0.0; speed_cnt = 0; arrived = 0
    mu_cf_history: dict[str, _deque] = {}   # vid → deque of (step, mu_cf)

    print(f"\n{'='*88}")
    _seed_str = f" seed={seed}" if seed is not None else ""
    print(f"  HybridModel | 4-way | {_sim_sec:.0f}s | EW={ew_vph} NS={ns_vph} vph{_seed_str} (+20% turns)")
    print(f"{'='*88}")
    print(f"  {'t(s)':>5}  {'veh':>4}  {'cols':>6}  {'arrived':>7}  {'spd(m/s)':>9}  {'nn_weight':>10}")
    print(f"  {'-'*60}")

    # warmup: SUMO drives
    for _ in range(warmup_steps):
        traci.simulationStep()
        arrived += traci.simulation.getArrivedNumber()

    # control loop
    for step in range(control_steps):
        traci.simulationStep()
        arrived += traci.simulation.getArrivedNumber()

        all_vids = list(traci.vehicle.getIDList())
        traci_cache.update(all_vids)

        for vid in all_vids:
            if vid not in known:
                traci.vehicle.setSpeedMode(vid, 0)
                known.add(vid)
        known.intersection_update(all_vids)

        if not all_vids:
            continue

        snap = build_snapshot(all_vids)
        all_tracked = [v for v, s in snap.vehicle_stream.items() if s in _ALL_MOVEMENTS]

        # social force → mu_conflict + own-stream summary + rival tokens per tracked vehicle
        _ZERO_SUM   = [0.0] * D_SUM
        _ZERO_RIVAL = [[0.0] * D_RIVAL] * K_MAX  # full zero pad for untracked vehicles
        if all_tracked:
            a_sf_t, mu_cf_t, mu_prio_t, sum_t, rival_list = compute_social_force_2d(all_tracked, snap)
            mu_cf_map   = {vid: float(mu_cf_t[j])   for j, vid in enumerate(all_tracked)}
            mu_prio_map = {vid: float(mu_prio_t[j]) for j, vid in enumerate(all_tracked)}
            sum_map     = {vid: sum_t[j].tolist()    for j, vid in enumerate(all_tracked)}
            rival_map   = {vid: rival_list[j]        for j, vid in enumerate(all_tracked)}
            a_sf_map    = {vid: float(a_sf_t[j])    for j, vid in enumerate(all_tracked)}
            for vid, mu in mu_cf_map.items():
                if vid not in mu_cf_history:
                    mu_cf_history[vid] = _deque(maxlen=25)  # 5 s lookback at DT=0.2
                mu_cf_history[vid].append((step, mu))
        else:
            mu_cf_map   = {}
            mu_prio_map = {}
            sum_map     = {}
            rival_map   = {}
            a_sf_map    = {}

        # rival ETA gate (same as demo_intersection.py)
        min_rival_eta: dict = {}
        for stream, svids in snap.stream_vehicles.items():
            if stream not in _ALL_MOVEMENTS or not svids:
                continue
            min_eta = float("inf")
            for rvid in svids:
                px, py = traci.vehicle.getPosition(rvid)
                spd    = traci_cache.get_speed(rvid)
                dist   = math.sqrt((_CX-px)**2 + (_CY-py)**2)
                min_eta = min(min_eta, dist / max(spd, 0.1))
            min_rival_eta[stream] = min_eta

        active = set(all_vids)

        # ── collect per-vehicle inputs ─────────────────────────────────────────
        vids_batch  = []
        u_wp_list   = []; mu_cf_list = []; mu_priority_list = []
        x_seq_list  = []; sum_list   = []; rival_list_batch = []
        nn_weights_step = []

        for vid in all_vids:
            v      = traci_cache.get_speed(vid)
            road   = traci_cache.get_road_id(vid)
            px, py = traci.vehicle.getPosition(vid)
            d_jct  = math.sqrt((_CX - px) ** 2 + (_CY - py) ** 2)

            ego_stream = snap.vehicle_stream.get(vid)

            # has_rival gate
            if ego_stream in _ALL_MOVEMENTS:
                rival_streams = CONFLICT_MAP.get(ego_stream, frozenset()) & _ALL_MOVEMENTS
                has_rival = any(
                    min_rival_eta.get(s, float("inf")) < ETA_RIVAL_THRESHOLD
                    for s in rival_streams
                )
            else:
                has_rival = True

            # u_wp: waypoint+IDM + turn kernel
            a_wp, gap, v_lead = _waypoint_accel(vid, active, has_rival, traci_cache)
            v_cap = _v_turn_cap(ego_stream)
            if ego_stream in _TURNING and v > v_cap:
                mu_t = min((v - v_cap) / max(V_TURN_HIGH - v_cap, 1e-3), 1.0)
                u_t  = float(np.clip(
                    IDM_ACCEL * (1.0 - (v / max(v_cap, 0.1)) ** IDM_DELTA),
                    -B_COMFORT, 0.0))
                a_wp = a_wp + mu_t * u_t

            mu_cf_v   = mu_cf_map.get(vid, 0.0)
            mu_prio_v = mu_prio_map.get(vid, 0.0)

            # Ego token (4-dim, all normalised) — oldest entry added first, newest last
            obs = (
                v / V_MAX,
                min(gap, GAP_MAX) / GAP_MAX,
                (v - v_lead) / V_MAX,                            # signed closing speed
                max(0.0, min(d_jct, _ARM_LENGTH)) / _ARM_LENGTH, # distance to junction
            )
            if vid not in histories:
                histories[vid] = deque([obs] * SEQ_LEN, maxlen=SEQ_LEN)
            else:
                histories[vid].append(obs)

            vids_batch.append(vid)
            u_wp_list.append(a_wp)
            mu_cf_list.append(mu_cf_v)
            mu_priority_list.append(mu_prio_v)
            x_seq_list.append(list(histories[vid]))
            sum_list.append(sum_map.get(vid, _ZERO_SUM))
            # Pad rival tokens to K_MAX rows (zero rows = no information)
            raw_rivals = rival_map.get(vid, [])
            padded = raw_rivals[:K_MAX] + [[0.0] * D_RIVAL] * max(0, K_MAX - len(raw_rivals))
            rival_list_batch.append(padded)
            nn_weights_step.append(1.0 - mu_cf_v)

        if not vids_batch:
            continue

        # ── model forward (batched) ────────────────────────────────────────────
        x_seq_t   = torch.tensor(x_seq_list,        dtype=torch.float32)  # [N, T, D_EGO]
        sum_t     = torch.tensor(sum_list,           dtype=torch.float32)  # [N, D_SUM]
        rival_t   = torch.tensor(rival_list_batch,   dtype=torch.float32)  # [N, K_MAX, D_RIVAL]
        u_wp_t    = torch.tensor(u_wp_list,          dtype=torch.float32)  # [N]
        mu_cf_t   = torch.tensor(mu_cf_list,         dtype=torch.float32)  # [N]
        mu_prio_t = torch.tensor(mu_priority_list,   dtype=torch.float32)  # [N]

        with torch.no_grad():
            a_out = model(x_seq_t, sum_t, rival_t, u_wp_t, mu_cf_t, mu_prio_t)  # [N]

        # ── apply speeds ───────────────────────────────────────────────────────
        for j, vid in enumerate(vids_batch):
            v = traci_cache.get_speed(vid)
            a = float(a_out[j])
            a_sf = a_sf_map.get(vid, 0.0)
            if a_sf < a and traci_cache.get_road_id(vid).startswith(':center'):
                a = a_sf
            v_next = float(np.clip(v + a*DT, 0.0, V_MAX))
            traci.vehicle.setSpeed(vid, v_next)
            speed_sum += v
            speed_cnt += 1

        try:
            cols = traci.simulation.getCollisions()
            if cols:
                t_s = WARMUP_SEC + step * DT
                for col in cols:
                    for role, vid in (("collider", col.collider), ("victim", col.victim)):
                        mu_now  = mu_cf_map.get(vid, float("nan"))
                        hist    = list(mu_cf_history.get(vid, []))
                        mu_vals = [m for _, m in hist]
                        mu_max  = max(mu_vals) if mu_vals else float("nan")
                        mu_mean = sum(mu_vals) / len(mu_vals) if mu_vals else float("nan")
                        print(
                            f"  [COLLISION] t={t_s:.1f}s  {role}={vid}"
                            f"  mu_cf_now={mu_now:.3f}"
                            f"  mu_cf_max5s={mu_max:.3f}"
                            f"  mu_cf_mean5s={mu_mean:.3f}"
                            f"  λ*mu_now={LAMBDA_TTC*mu_now:.4f}"
                            f"  λ*mu_max={LAMBDA_TTC*mu_max:.4f}"
                        )
                total_cols += len(cols)
        except AttributeError:
            pass

        if (step+1) % PRINT_EVERY == 0:
            t       = WARMUP_SEC + (step+1)*DT
            n       = len(all_vids)
            mean_v  = speed_sum / max(speed_cnt, 1)
            mean_nn = float(np.mean(nn_weights_step)) if nn_weights_step else 0.0
            print(f"  {t:5.1f}  {n:4d}  {total_cols:6d}  {arrived:7d}  {mean_v:9.2f}  {mean_nn:10.3f}")

    traci.close()
    # clean up departed vehicles from history
    for vid in list(histories):
        if vid not in known:
            del histories[vid]

    mean_v = speed_sum / max(speed_cnt, 1)
    print(f"\n{'='*88}")
    print(f"  RESULT  collisions={total_cols}  arrived={arrived}  avg_speed={mean_v:.2f} m/s ({100*mean_v/V_MAX:.1f}% free-flow)")
    print(f"{'='*88}\n")
    return total_cols


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vph",    type=int, default=FLOW_VPH, help="vph per stream (EW+NS)")
    p.add_argument("--ew",     type=int, default=None,     help="EW vph override")
    p.add_argument("--ns",     type=int, default=None,     help="NS vph override")
    p.add_argument("--gui",    action="store_true")
    p.add_argument("--model",    type=str,   default=None,       help="path to saved model weights")
    p.add_argument("--duration", type=float, default=SIM_SECONDS, help="simulation seconds (default 120)")
    p.add_argument("--seed",     type=int,   default=None,       help="SUMO random seed")
    args = p.parse_args()
    ew = args.ew if args.ew is not None else args.vph
    ns = args.ns if args.ns is not None else args.vph
    run(gui=args.gui, ew_vph=ew, ns_vph=ns, model_path=args.model,
        sim_seconds=args.duration, seed=args.seed)
