"""
eval_model.py — Evaluate trained HybridModel checkpoints.

Scans the checkpoint directory for iter_NNNN.pt periodic checkpoints (saved
every 5 training iterations), plus latest.pt and a zero-init baseline.
Runs each through N 60-second PyTorch simulations on the same held-out
schedule set, then prints a sorted comparison table.

Metrics per model
-----------------
  Avg speed      : mean vehicle speed across all alive vehicles and all steps
  W1 (0–30 s)   : same, first BPTT window only (vehicles still arriving)
  W2 (30–60 s)  : same, second window (queue clearing)
  Throughput     : mean vehicles that crossed the junction per episode
  Collisions     : total collision events across all episodes (gap<0 or jct conflict)

Usage
-----
    conda run -n car-following-sumo python eval_model.py [options]

Options
-------
    --out          PATH   checkpoint directory (default: checkpoints/)
    --n-seeds      N      fresh SUMO schedule seeds for evaluation (default: 4)
    --n-eps        N      parallel episodes per seed (default: 8)
    --vph          N      traffic demand for eval schedules (default: 900)
    --no-sumo             use a synthetic schedule instead of SUMO (quick smoke test)
    --sumo-native         also run SUMO with its built-in traffic light + Krauss
                          car-following (no external intervention) as an extra row
    --device       STR    cuda / cpu (default: auto)
    --max-ckpts    N      max periodic checkpoints to evaluate; evenly sampled
                          (default: 8; use 0 for all)
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from model import HybridModel
from intersection_env import IntersectionEnv, DT, DEPART_ARC
from schedule_collector import VehicleEntry

EPISODE_STEPS = 300   # 60 s at DT=0.2
WINDOW        = 150   # 30 s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out",        default="checkpoints")
    p.add_argument("--n-seeds",    type=int,  default=4)
    p.add_argument("--n-eps",      type=int,  default=8)
    p.add_argument("--vph",        type=int,  default=900)
    p.add_argument("--no-sumo",      action="store_true")
    p.add_argument("--sumo-native",  action="store_true",
                   help="add SUMO built-in controller as a baseline row")
    p.add_argument("--device",       default="")
    p.add_argument("--max-ckpts",    type=int,  default=8,
                   help="max periodic checkpoints to evaluate (0=all)")
    return p.parse_args()


# ── schedule helpers ──────────────────────────────────────────────────────────

def _synthetic_schedule(seed: int, n_veh: int = 30) -> list[VehicleEntry]:
    rng = random.Random(seed)
    sched = [
        VehicleEntry(
            spawn_step = rng.randint(0, 120),
            stream_idx = rng.randint(0, 11),
            v0         = rng.uniform(5.0, 10.0),
            arc0       = rng.uniform(0.0, 150.0) if rng.random() < 0.3 else 0.0,
        )
        for _ in range(n_veh)
    ]
    sched.sort(key=lambda e: e.spawn_step)
    return sched


def _sumo_schedule(vph: int, seed: int) -> list[VehicleEntry]:
    from schedule_collector import collect_schedule
    return collect_schedule(vph=vph, seed=seed)


def run_sumo_native(ew_vph: int, ns_vph: int, seed: int) -> dict:
    """Run one episode under SUMO's built-in traffic light + Krauss controller.

    No external intervention — SUMO drives every vehicle itself.  Measures the
    same metrics as run_eval() so the results are directly comparable.

    Speed is averaged only over vehicles on the approach arms (incoming edges),
    matching the PyTorch env's `mean_speed` computation.  Throughput counts
    vehicles that completed their route (exited the network) during the window.
    """
    import traci
    from demo_hybrid import _write_routes, _write_cfg, _bin, DT as _DT
    from conflict import _INCOMING
    from schedule_collector import WARMUP_SEC

    routes = _write_routes(ew_vph, ns_vph)
    cfg    = _write_cfg(routes.name)
    cmd = [
        _bin("sumo"), "-c", str(cfg),
        "--step-length",      str(_DT),
        "--collision.action", "warn",
        "--no-step-log",
        "--seed",             str(seed),
    ]
    traci.start(cmd)

    warmup_steps = round(WARMUP_SEC / _DT)
    speeds_w1: list[float] = []
    speeds_w2: list[float] = []
    n_departed   = 0
    n_collisions = 0

    try:
        for _ in range(warmup_steps):
            traci.simulationStep()

        for step in range(EPISODE_STEPS):
            traci.simulationStep()

            step_speeds = [
                traci.vehicle.getSpeed(vid)
                for vid in traci.vehicle.getIDList()
                if traci.vehicle.getRoadID(vid) in _INCOMING
            ]
            mean_spd = sum(step_speeds) / len(step_speeds) if step_speeds else 0.0
            (speeds_w1 if step < WINDOW else speeds_w2).append(mean_spd)

            n_departed   += traci.simulation.getArrivedNumber()
            n_collisions += traci.simulation.getCollidingVehiclesNumber()

    finally:
        traci.close()

    w1  = sum(speeds_w1) / len(speeds_w1) if speeds_w1 else 0.0
    w2  = sum(speeds_w2) / len(speeds_w2) if speeds_w2 else 0.0
    avg = (w1 * WINDOW + w2 * (EPISODE_STEPS - WINDOW)) / EPISODE_STEPS

    return {
        "avg_speed":  avg,
        "w1_speed":   w1,
        "w2_speed":   w2,
        "throughput": float(n_departed),
        "collisions": float(n_collisions),
    }


# ── single-episode evaluation ─────────────────────────────────────────────────

@torch.no_grad()
def run_eval(model: HybridModel, env: IntersectionEnv) -> dict:
    """Run one 60-second episode. Returns metric dict."""
    model.eval()
    env.reset()

    speeds_w1, speeds_w2 = [], []
    prev_alive = env.alive.clone()
    total_departed = torch.zeros(env.n_eps, device=env.device)

    for step in range(EPISODE_STEPS):
        speed = env.step(model)          # [E]

        if step < WINDOW:
            speeds_w1.append(speed)
        else:
            speeds_w2.append(speed)

        # count vehicles that departed this step (alive → not alive)
        just_left = prev_alive & ~env.alive   # [E, N]
        total_departed += just_left.float().sum(dim=1)
        prev_alive = env.alive.clone()

    w1 = torch.stack(speeds_w1).mean().item() if speeds_w1 else 0.0
    w2 = torch.stack(speeds_w2).mean().item() if speeds_w2 else 0.0

    return {
        "avg_speed":   (w1 * len(speeds_w1) + w2 * len(speeds_w2)) / EPISODE_STEPS,
        "w1_speed":    w1,
        "w2_speed":    w2,
        "throughput":  total_departed.mean().item(),
        "collisions":  env.n_collisions,
    }


# ── aggregate over multiple seeds ─────────────────────────────────────────────

def evaluate(
    models:    dict[str, HybridModel],
    schedules: list,
    n_eps:     int,
    device:    str,
) -> dict[str, dict]:
    """
    For each (model_name, model) pair, run every schedule and average results.
    Returns {name: {metric: float}}.
    """
    results: dict[str, dict] = {name: {} for name in models}
    accum:   dict[str, dict] = {name: {k: [] for k in
                                        ("avg_speed","w1_speed","w2_speed",
                                         "throughput","collisions")}
                                for name in models}

    for s_idx, sched in enumerate(schedules):
        env = IntersectionEnv(sched, n_eps=n_eps, device=device, v0_noise=0.3)
        print(f"  Seed {s_idx+1}/{len(schedules)} — {len(sched)} vehicles", end="")

        for name, model in models.items():
            model.to(device)
            m = run_eval(model, env)
            for k, v in m.items():
                accum[name][k].append(v)
            print(f"  [{name}: {m['avg_speed']:.3f} m/s]", end="", flush=True)

        print()   # newline after all models on this seed

    # average over seeds
    for name in models:
        results[name] = {k: sum(vs)/len(vs) for k, vs in accum[name].items()}

    return results


# ── pretty table ──────────────────────────────────────────────────────────────

def print_table(results: dict[str, dict], baseline_name: str = "Baseline"):
    cols = ["avg_speed", "w1_speed", "w2_speed", "throughput", "collisions"]
    hdrs = ["Avg m/s", "W1 0-30s", "W2 30-60s", "Throughput", "Collisions"]
    col_w = 11

    # sort: Baseline, SUMO-native, iter checkpoints by number, anything else last
    def _sort_key(name):
        if name == baseline_name:
            return (-2, name)
        if name == "SUMO-native":
            return (-1, name)
        if name.startswith("iter_"):
            try:
                return (int(name.split("_")[1]), name)
            except ValueError:
                pass
        return (10**9, name)

    ordered = sorted(results.items(), key=lambda kv: _sort_key(kv[0]))

    name_w = max(len(n) for n in results) + 2
    header = f"{'Model':<{name_w}}" + "".join(f"{h:>{col_w}}" for h in hdrs)
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))

    base_speed = results.get(baseline_name, {}).get("avg_speed", 0.0)

    for name, m in ordered:
        delta = ""
        if name != baseline_name and base_speed > 0:
            pct = (m["avg_speed"] - base_speed) / base_speed * 100
            delta = f"  ({pct:+.1f}% vs baseline)"

        col_str = f"{m['collisions']:.0f}"
        if m["collisions"] > 0:
            col_str += " !"

        row = (
            f"{name:<{name_w}}"
            f"{m['avg_speed']:>{col_w}.4f}"
            f"{m['w1_speed']:>{col_w}.4f}"
            f"{m['w2_speed']:>{col_w}.4f}"
            f"{m['throughput']:>{col_w}.1f}"
            f"{col_str:>{col_w}}"
        )
        print(row + delta)

    print("─" * len(header))
    print("  W1 = first 30 s (vehicles arriving)  |  "
          "W2 = last 30 s (queue clearing)")
    print("  Throughput = mean vehicles completed per episode\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    out_dir = Path(args.out)

    # ── discover periodic checkpoints ────────────────────────────────────────
    all_periodic = sorted(out_dir.glob("iter_*.pt"),
                          key=lambda p: int(p.stem.split("_")[1]))

    max_ck = args.max_ckpts
    if max_ck > 0 and len(all_periodic) > max_ck:
        # evenly sample max_ck checkpoints, always including the last one
        step = len(all_periodic) / max_ck
        indices = sorted({min(round(i * step), len(all_periodic) - 1)
                          for i in range(max_ck)})
        sampled = [all_periodic[i] for i in indices]
    else:
        sampled = all_periodic

    print("=" * 60)
    print("  HybridModel Evaluation")
    print("=" * 60)
    print(f"  Device        : {device}")
    print(f"  Checkpoints   : {out_dir}/")
    print(f"  Periodic found: {len(all_periodic)}  "
          f"(evaluating {len(sampled)})")
    print(f"  Seeds         : {args.n_seeds}  |  Episodes/seed: {args.n_eps}")
    print(f"  Schedule      : {'synthetic (--no-sumo)' if args.no_sumo else f'{args.vph} vph from SUMO'}")
    if args.sumo_native:
        print(f"  SUMO-native   : enabled  (traffic light + Krauss, same seeds)")
    print("=" * 60)

    # ── load models ───────────────────────────────────────────────────────────
    models: dict[str, HybridModel] = {}

    # zero-init baseline (pure physics — no NN correction)
    models["Baseline"] = HybridModel(seq_len=10)
    print("  Baseline     : zero-init (physics only)")

    # periodic checkpoints
    for path in sampled:
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        m = HybridModel(seq_len=10)
        m.load_state_dict(state)
        iter_n  = ckpt.get("iter", "?")
        ew, ns  = ckpt.get("ew_vph", "?"), ckpt.get("ns_vph", "?")
        label   = f"iter_{iter_n:04d}" if isinstance(iter_n, int) else path.stem
        models[label] = m
        print(f"  {label:<14}: {path.name}  (vph={ew}ew/{ns}ns)")

    # latest.pt — only add if it's not already the last periodic checkpoint
    latest_path = out_dir / "latest.pt"
    if latest_path.exists():
        ckpt   = torch.load(latest_path, map_location="cpu", weights_only=False)
        iter_n = ckpt.get("iter", 0)
        label  = f"iter_{iter_n:04d}" if isinstance(iter_n, int) else "latest"
        if label not in models:
            state = ckpt.get("model", ckpt)
            m = HybridModel(seq_len=10)
            m.load_state_dict(state)
            models[label] = m
            ew, ns = ckpt.get("ew_vph", "?"), ckpt.get("ns_vph", "?")
            print(f"  {label:<14}: latest.pt  (vph={ew}ew/{ns}ns)")

    print()

    # ── collect schedules ─────────────────────────────────────────────────────
    schedules = []
    if args.no_sumo:
        print(f"Generating {args.n_seeds} synthetic schedules...")
        for seed in range(args.n_seeds):
            schedules.append(_synthetic_schedule(seed + 100))
    else:
        print(f"Collecting {args.n_seeds} schedules from SUMO ({args.vph} vph)...")
        for seed in range(args.n_seeds):
            print(f"  Seed {seed+1}/{args.n_seeds} ...", end=" ", flush=True)
            t0 = time.time()
            sched = _sumo_schedule(args.vph, seed=seed + 200)  # offset from training seeds
            schedules.append(sched)
            print(f"{len(sched)} vehicles  ({time.time()-t0:.1f}s)")

    print()

    # ── SUMO-native baseline ──────────────────────────────────────────────────
    sumo_native_results: dict | None = None
    if args.sumo_native and not args.no_sumo:
        print(f"Running SUMO-native baseline ({args.n_seeds} seeds)...")
        accum = {k: [] for k in ("avg_speed","w1_speed","w2_speed","throughput","collisions")}
        for seed_i in range(args.n_seeds):
            seed_val = seed_i + 200   # same offset as schedule collection
            print(f"  Seed {seed_i+1}/{args.n_seeds} (seed={seed_val}) ...",
                  end=" ", flush=True)
            t0 = time.time()
            m = run_sumo_native(args.vph, args.vph, seed_val)
            for k, v in m.items():
                accum[k].append(v)
            print(f"avg={m['avg_speed']:.3f} m/s  "
                  f"throughput={m['throughput']:.0f}  ({time.time()-t0:.1f}s)")
        sumo_native_results = {k: sum(vs) / len(vs) for k, vs in accum.items()}
        print()

    # ── run PyTorch model evaluation ──────────────────────────────────────────
    print("Running evaluation...")
    t0 = time.time()
    results = evaluate(models, schedules, args.n_eps, device)
    elapsed = time.time() - t0

    if sumo_native_results is not None:
        results["SUMO-native"] = sumo_native_results

    print(f"\nDone in {elapsed:.1f}s")

    # ── print results ─────────────────────────────────────────────────────────
    print_table(results, baseline_name="Baseline")


if __name__ == "__main__":
    main()
