"""Test BPTT window + full 60-second episode timing on GPU."""
import time
import torch
from model import HybridModel
from intersection_env import IntersectionEnv, DT, SEQ_LEN
from schedule_collector import VehicleEntry

EPISODE_STEPS = 300   # 60 s at DT=0.2
BPTT          = 150   # 30 s

# Synthetic schedule: ~30 vehicles across several streams
import random
random.seed(42)
ALL_STREAMS = list(range(12))
sched = []
for i in range(30):
    sched.append(VehicleEntry(
        spawn_step = random.randint(0, 80),
        stream_idx = random.choice(ALL_STREAMS),
        v0         = random.uniform(6.0, 10.0),
        arc0       = random.uniform(0.0, 150.0) if random.random() < 0.3 else 0.0,
    ))
sched.sort(key=lambda e: e.spawn_step)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}  |  EPISODE_STEPS={EPISODE_STEPS}  BPTT={BPTT}  N_EPS=8\n")

model = HybridModel(seq_len=10).to(device)
env   = IntersectionEnv(sched, n_eps=8, device=device)
env.reset()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
window_speeds = []
optimizer.zero_grad()

t0 = time.time()
for step in range(EPISODE_STEPS):
    speed = env.step(model)       # [8]
    window_speeds.append(speed)

    if (step + 1) % BPTT == 0:
        loss = -torch.stack(window_speeds).mean()
        loss.backward()
        gn = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        env.detach()
        window_speeds = []
        print(f"  step {step+1:4d}  loss={loss.item():.4f}  grad_norm={gn:.4f}  "
              f"speed={speed.mean().item():.3f} m/s")

elapsed = time.time() - t0
print(f"\n60-s episode wall time: {elapsed:.2f}s  ({elapsed*1000/EPISODE_STEPS:.1f}ms/step)")
if device == "cuda":
    print(f"GPU memory: {torch.cuda.max_memory_allocated()/1e6:.1f} MB peak")
print("BPTT test passed.")
