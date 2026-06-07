"""Dry-run: 3 iterations with fake schedule to verify print layout."""
import math, time, random, warnings
warnings.filterwarnings("ignore")

import torch
from torch.nn.utils import clip_grad_norm_
from model import HybridModel
from intersection_env import IntersectionEnv, DT
from schedule_collector import VehicleEntry

random.seed(0)
sched = [VehicleEntry(random.randint(0, 80), random.randint(0, 11),
                      random.uniform(6, 10),
                      random.uniform(0, 150) if random.random() < 0.3 else 0.0)
         for _ in range(30)]
sched.sort(key=lambda e: e.spawn_step)

BPTT = 150; EPISODE_STEPS = 300; N_EPS = 8; ITERS = 3
n_windows = math.ceil(EPISODE_STEPS / BPTT)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = HybridModel(seq_len=10).to(device)
env   = IntersectionEnv(sched, n_eps=N_EPS, device=device)
opt   = torch.optim.AdamW(model.parameters(), lr=3e-4)
best_speed = 0.0

print(f"\n{'Iter':>6}  {'Win':>5}  {'Step':>6}  "
      f"{'Speed m/s':>10}  {'Loss':>8}  {'GNorm':>7}  {'Col':>6}  {'t_iter':>7}")
print("-" * 72)

for iteration in range(ITERS):
    t_iter = time.time()
    model.train(); env.reset()
    iter_speeds, iter_losses = [], []
    window_speeds = []; win_idx = 0; opt.zero_grad()

    for step in range(EPISODE_STEPS):
        ms = env.step(model)
        window_speeds.append(ms)

        if (step + 1) % BPTT == 0 or step == EPISODE_STEPS - 1:
            sw   = torch.stack(window_speeds)
            loss = -sw.mean()
            loss.backward()
            gn = sum(p.grad.norm().item() ** 2
                     for p in model.parameters() if p.grad is not None) ** 0.5
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
            w_speed = sw.mean().item(); w_loss = loss.item()
            iter_speeds.append(w_speed); iter_losses.append(w_loss)
            n_col = env.n_collisions
            env.n_collisions = 0
            env.detach(); window_speeds = []; win_idx += 1
            t_so_far = time.time() - t_iter
            col_str = f"{n_col:>6}" if n_col == 0 else f"{n_col:>5}!"
            print(f"{iteration+1:>6}  {win_idx:>3}/{n_windows:<2}  {step+1:>6}  "
                  f"{w_speed:>10.4f}  {w_loss:>8.4f}  {gn:>7.3f}  {col_str}  {t_so_far:>5.1f}s")

    avg_speed = sum(iter_speeds) / max(len(iter_speeds), 1)
    elapsed   = time.time() - t_iter
    marker    = " *" if avg_speed > best_speed else ""
    print("─" * 72)
    print(f"  Iter {iteration+1}/{ITERS}  avg_speed={avg_speed:.4f} m/s  "
          f"best={max(best_speed, avg_speed):.4f}  ({elapsed:.1f}s){marker}")
    print("─" * 72)
    if avg_speed > best_speed:
        best_speed = avg_speed

print(f"\nDone. Best: {best_speed:.4f} m/s")
