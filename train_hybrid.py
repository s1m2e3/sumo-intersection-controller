"""
train_hybrid.py — Differentiable training loop for HybridModel.

Domain randomization
--------------------
At startup a pool of N_POOL schedules is collected from SUMO, each with a
randomly drawn total demand (300–1200 vph) and directional split (east-west
vs north-south, ±30% asymmetry).  At the start of every training iteration
one schedule is sampled from the pool at random.  Over 200 iterations the
model sees the full range of traffic conditions without any manual scenario
specification.

Training pipeline
-----------------
1.  Collect N_POOL diverse SUMO schedules (SUMO closed after this step).
2.  Training loop — each iteration:
      a. Sample a schedule from the pool.
      b. Build IntersectionEnv with that schedule + per-episode speed noise.
      c. Run EPISODE_STEPS steps, computing loss every BPTT_WINDOW steps
         (truncated BPTT), then detach state.
3.  Checkpoints: latest.pt always, best.pt when a new best avg speed is seen.
    On resume, model weights come from best.pt, iter count from latest.pt.

Usage
-----
    conda run -n car-following-sumo python train_hybrid.py [options]

Key options
-----------
    --n-pool   N   schedules in diversity pool (default 40)
    --n-eps    N   parallel episodes per iteration (default 8)
    --iters    N   total training iterations (default 200)
    --bptt     N   BPTT window in steps (default 150 = 30 s)
    --lr       F   initial learning rate (default 5e-5, cosine-decays to 5%)
    --out      P   checkpoint directory (default checkpoints/)
    --device   S   cuda / cpu (default: auto)
    --validate     run demo_hybrid.py in SUMO after every 10 iterations
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from model import HybridModel
from schedule_collector import collect_schedule, EPISODE_SEC
from intersection_env import IntersectionEnv, DT

# ── defaults ──────────────────────────────────────────────────────────────────
N_POOL        = 40      # diverse schedules collected once at startup
N_EPS         = 8       # parallel episodes per gradient step
BPTT_WINDOW   = 150     # steps per BPTT window  (150 × 0.2 s = 30 s)
EPISODE_STEPS = 300     # steps per iteration     (300 × 0.2 s = 60 s)
LR            = 1e-4
ITERS         = 200
CHECKPOINT_DIR = Path("checkpoints")
VALIDATE_EVERY = 10
LAMBDA_WAIT   = 0.5   # weight on the waiting-time regularizer
LAMBDA_TTC    = 0.3   # weight on the smooth TTC violation penalty

# ── traffic randomization ranges ──────────────────────────────────────────────
VPH_MIN    = 300    # minimum total vph
VPH_MAX    = 1200   # maximum total vph
VPH_STEP   = 50     # round to nearest N vph for cleaner SUMO configs
EW_FRAC_LO = 0.35   # min east-west fraction of total demand
EW_FRAC_HI = 0.65   # max east-west fraction  (so neither direction > 1.86×)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HybridModel — domain randomized")
    p.add_argument("--n-pool",   type=int,   default=N_POOL)
    p.add_argument("--n-eps",    type=int,   default=N_EPS)
    p.add_argument("--iters",    type=int,   default=ITERS)
    p.add_argument("--bptt",     type=int,   default=BPTT_WINDOW)
    p.add_argument("--lr",             type=float, default=LR)
    p.add_argument("--phys-lr-scale", type=float, default=0.50,
                   help="LR multiplier for physics-tuning params (default 0.50 = 2× smaller)")
    p.add_argument("--lambda-wait",   type=float, default=LAMBDA_WAIT,
                   help="weight on the waiting-time regularizer (default 0.5)")
    p.add_argument("--lambda-ttc",    type=float, default=LAMBDA_TTC,
                   help="weight on the smooth TTC violation penalty (default 0.3)")
    p.add_argument("--out",          type=str,   default=str(CHECKPOINT_DIR))
    p.add_argument("--device",       type=str,   default="")
    p.add_argument("--validate",     action="store_true")
    return p.parse_args()


# ── traffic sampling ──────────────────────────────────────────────────────────

def _random_traffic(rng: random.Random) -> tuple[int, int]:
    """
    Sample (ew_vph, ns_vph) uniformly over demand and directional split.

    Total demand ∈ [VPH_MIN, VPH_MAX] (rounded to VPH_STEP).
    EW fraction ∈ [EW_FRAC_LO, EW_FRAC_HI] → max asymmetry ≈ 1.86×.
    """
    n_steps = (VPH_MAX - VPH_MIN) // VPH_STEP
    total   = VPH_MIN + rng.randint(0, n_steps) * VPH_STEP
    ew_frac = rng.uniform(EW_FRAC_LO, EW_FRAC_HI)
    ew_vph  = max(VPH_STEP, round(total * ew_frac / VPH_STEP) * VPH_STEP)
    ns_vph  = max(VPH_STEP, total - ew_vph)
    return ew_vph, ns_vph


# ── pool collection ───────────────────────────────────────────────────────────

def collect_pool(n: int, rng: random.Random) -> list[tuple[int, int, list]]:
    """
    Collect n diverse schedules.  Returns list of (ew_vph, ns_vph, schedule).
    """
    pool = []
    for i in range(n):
        ew_vph, ns_vph = _random_traffic(rng)
        total = ew_vph + ns_vph
        print(f"  [{i+1:>3}/{n}] total={total:>5} vph  "
              f"ew={ew_vph} ns={ns_vph}  seed={i}", end="  ", flush=True)
        t0    = time.time()
        sched = collect_schedule(ew_vph=ew_vph, ns_vph=ns_vph, seed=i)
        print(f"{len(sched):>4} vehicles  ({time.time()-t0:.1f}s)")
        pool.append((ew_vph, ns_vph, sched))
    return pool


# ── SUMO validation ───────────────────────────────────────────────────────────

def validate_in_sumo(model: HybridModel, model_path: Path):
    import subprocess, sys
    torch.save(model.state_dict(), model_path)
    result = subprocess.run(
        [sys.executable, "demo_hybrid.py", "--model", str(model_path)],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "RESULT" in line:
            print(f"    SUMO: {line.strip()}")
            return
    print("    SUMO validation: (no RESULT line)")
    if result.returncode != 0:
        print(f"    stderr: {result.stderr[-400:]}")


# ── training ──────────────────────────────────────────────────────────────────

def train():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(0)   # reproducible pool + sampling

    # ── device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    n_windows  = math.ceil(EPISODE_STEPS / args.bptt)
    iter_sec   = EPISODE_STEPS * DT
    window_sec = args.bptt * DT

    print("=" * 70)
    print("  HybridModel Training  —  domain randomized")
    print("=" * 70)
    print(f"  Device       : {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Demand range : {VPH_MIN}–{VPH_MAX} vph total  "
          f"(EW/NS split {EW_FRAC_LO:.0%}–{EW_FRAC_HI:.0%})")
    print(f"  Pool size    : {args.n_pool} schedules")
    print(f"  Episodes     : {args.n_eps} parallel")
    print(f"  Episode len  : {iter_sec:.0f}s  ({EPISODE_STEPS} steps)")
    print(f"  BPTT window  : {window_sec:.0f}s  ({args.bptt} steps, "
          f"{n_windows} windows/iter)")
    print(f"  Iterations   : {args.iters}")
    print(f"  LR           : {args.lr} → cosine → {args.lr*0.05:.2e}")
    print(f"  Checkpoints  : {out_dir}/  (every 5 iters + latest)")
    print("=" * 70)

    # ── collect diverse schedule pool ─────────────────────────────────────────
    print(f"\nCollecting {args.n_pool} diverse schedules from SUMO...")
    pool = collect_pool(args.n_pool, rng)
    print(f"Pool ready — {sum(len(s) for _, _, s in pool)} vehicles total\n")

    # ── model ─────────────────────────────────────────────────────────────────
    model = HybridModel(seq_len=10).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # ── load weights (always train for args.iters fresh iterations) ──────────
    latest_ckpt = out_dir / "latest.pt"

    if latest_ckpt.exists():
        state = torch.load(latest_ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if missing:
            print(f"  New params (start at nominal): {missing}")
        print(f"  Loaded weights from {latest_ckpt.name}  (prior iter {state.get('iter', '?')})")
    else:
        print("  No checkpoint found — training from scratch")

    # Two param groups: transformer at full LR, physics tuning at much smaller LR
    phys_params = list(model.phys_tune.parameters())
    phys_ids    = {id(p) for p in phys_params}
    main_params = [p for p in model.parameters() if id(p) not in phys_ids]
    optimizer = optim.AdamW([
        {"params": main_params, "lr": args.lr,                        "weight_decay": 1e-4},
        {"params": phys_params, "lr": args.lr * args.phys_lr_scale,   "weight_decay": 1e-2},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iters, eta_min=args.lr * 0.05
    )

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\n{'Iter':>6}  {'Win':>5}  {'Step':>6}  "
          f"{'Speed':>10}  {'WaitPen':>7}  {'TTCPen':>7}  "
          f"{'Loss':>8}  {'GNorm':>7}  {'PhysGN':>10}  {'Col':>6}  {'t':>7}")
    print("-" * 95)

    for iteration in range(args.iters):
        # ── sample schedule for this iteration ────────────────────────────────
        ew_vph, ns_vph, sched = rng.choice(pool)
        total_vph = ew_vph + ns_vph

        t_iter = time.time()
        model.train()

        env = IntersectionEnv(
            schedule = sched,
            n_eps    = args.n_eps,
            device   = str(device),
            v0_noise = 0.5,
        )
        env.reset()

        iter_speeds: list[float] = []
        iter_losses: list[float] = []
        window_speeds:    list[torch.Tensor] = []
        window_wait_pens: list[torch.Tensor] = []
        window_ttc_pens:  list[torch.Tensor] = []
        win_idx = 0
        optimizer.zero_grad()

        for step in range(EPISODE_STEPS):
            mean_speed, wait_pen, ttc_pen = env.step(model)
            window_speeds.append(mean_speed)
            window_wait_pens.append(wait_pen)
            window_ttc_pens.append(ttc_pen)

            if (step + 1) % args.bptt == 0 or step == EPISODE_STEPS - 1:
                speed_window    = torch.stack(window_speeds,    dim=0)
                wait_pen_window = torch.stack(window_wait_pens, dim=0)
                ttc_pen_window  = torch.stack(window_ttc_pens,  dim=0)
                loss = (-speed_window.mean()
                        + args.lambda_wait * wait_pen_window.mean()
                        + args.lambda_ttc  * ttc_pen_window.mean())

                loss.backward()

                # Main grad norm (transformer + physics together)
                gn = sum(p.grad.norm().item() ** 2
                         for p in model.parameters()
                         if p.grad is not None) ** 0.5
                # Physics-param grad norm tracked separately
                phys_gn = sum(p.grad.norm().item() ** 2
                              for p in model.phys_tune.parameters()
                              if p.grad is not None) ** 0.5

                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                w_speed   = speed_window.mean().item()
                w_wait    = wait_pen_window.mean().item()
                w_ttc     = ttc_pen_window.mean().item()
                w_loss    = loss.item()
                n_col     = env.n_collisions
                iter_losses.append(w_loss)
                iter_speeds.append(w_speed)

                env.n_collisions = 0
                env.detach()
                window_speeds    = []
                window_wait_pens = []
                window_ttc_pens  = []
                win_idx += 1

                t_so_far = time.time() - t_iter
                col_str  = f"{n_col:>6}" if n_col == 0 else f"{n_col:>5}!"

                print(f"{iteration+1:>6}  {win_idx:>3}/{n_windows:<2}  "
                      f"{step+1:>6}  "
                      f"{w_speed:>10.4f}  {w_wait:>7.4f}  {w_ttc:>7.4f}  "
                      f"{w_loss:>8.4f}  {gn:>7.3f}  pgn={phys_gn:.2e}  "
                      f"{col_str}  {t_so_far:>6.1f}s")

        avg_speed = sum(iter_speeds) / max(len(iter_speeds), 1)
        elapsed   = time.time() - t_iter

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        iter_num = iteration + 1
        print(f"{'─'*95}")
        print(f"  Iter {iter_num}/{args.iters}  "
              f"vph={total_vph} (ew={ew_vph}/ns={ns_vph})  "
              f"avg={avg_speed:.4f} m/s  "
              f"lr={current_lr:.2e}  ({elapsed:.1f}s)")
        if iter_num % 10 == 0:
            print(f"  Physics: {model.phys_tune.summary()}")
        print(f"{'─'*95}")

        # ── checkpoints ───────────────────────────────────────────────────────
        ckpt = {
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter":      iter_num,
            "avg_speed": avg_speed,
            "ew_vph":    ew_vph,
            "ns_vph":    ns_vph,
        }
        torch.save(ckpt, latest_ckpt)
        if iter_num % 5 == 0:
            periodic = out_dir / f"iter_{iter_num:04d}.pt"
            torch.save(ckpt, periodic)
            print(f"  Saved {periodic.name}")

        # ── SUMO validation ───────────────────────────────────────────────────
        if args.validate and (iteration + 1) % VALIDATE_EVERY == 0:
            model.eval()
            validate_in_sumo(model, out_dir / "tmp_validate.pt")

    n_periodic = len(list(out_dir.glob("iter_*.pt")))
    print(f"\nTraining complete.  {n_periodic} periodic checkpoints in {out_dir}/")


if __name__ == "__main__":
    train()
