"""Quick smoke test for intersection_env.py — no SUMO needed."""
import torch
from model import HybridModel
from intersection_env import IntersectionEnv, STREAM_LIST, N_STREAMS
from schedule_collector import VehicleEntry

sched = [
    VehicleEntry(spawn_step=0,  stream_idx=0, v0=8.0, arc0=50.0),
    VehicleEntry(spawn_step=0,  stream_idx=3, v0=9.0, arc0=30.0),
    VehicleEntry(spawn_step=0,  stream_idx=6, v0=7.5, arc0=10.0),
    VehicleEntry(spawn_step=10, stream_idx=9, v0=8.5, arc0=0.0),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = HybridModel(seq_len=10).to(device)
env   = IntersectionEnv(sched, n_eps=4, device=device, v0_noise=0.3)
env.reset()

speeds = []
for step in range(5):
    s = env.step(model)
    speeds.append(s)
    print(f"  step {step}: mean_speed = {s.mean().item():.4f} m/s  (episodes: {s.tolist()})")

loss = -torch.stack(speeds).mean()
loss.backward()

grad_norm = sum(
    p.grad.norm().item() ** 2
    for p in model.parameters()
    if p.grad is not None
) ** 0.5

print(f"\nLoss:      {loss.item():.6f}")
print(f"Grad norm: {grad_norm:.6f}  (nonzero = gradients flow correctly)")
print(f"N_STREAMS: {N_STREAMS}")
print(f"STREAM_LIST[:3]: {STREAM_LIST[:3]}")
print("\nOK — env forward + backward pass succeeded.")
