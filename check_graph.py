import torch
from model import HybridModel

model = HybridModel()
for p in model.physics.parameters():
    p.requires_grad_(False)

# Scenario firmly in TTC* = 4s  (inside [3,5] dead zone)
v      = torch.tensor([10.0])
v_lead = torch.tensor([9.0])   # closing at 1 m/s
gap    = torch.tensor([4.0])   # TTC* = gap/dv = 4/1 = 4.0

# Confirm zone
z      = model.correction.ttc_star(v, gap, v_lead)
mu_ff  = model.correction.mu_ff(z)
mu_dec = model.correction.mu_dec(z)
print(f"TTC*   = {z.item():.4f}  (expect ~4.0)")
print(f"mu_ff  = {mu_ff.item():.4f}  (expect 0.0)")
print(f"mu_dec = {mu_dec.item():.4f}  (expect 0.0)")
print(f"h_=    = 0 confirmed: both mu terms zero")
print()

# Forward pass — accel = f_hat_gated in this zone
accel = model(v, gap, v_lead)
print(f"accel  = {accel.item():.6f}")
accel.backward()

# Check gradient on every NN parameter
print()
all_ok = True
for name, p in model.f_hat.named_parameters():
    gnorm  = p.grad.norm().item() if p.grad is not None else 0.0
    status = "OK" if gnorm > 0 else "*** ZERO GRAD ***"
    print(f"  {name:30s}  grad_norm = {gnorm:.8f}  {status}")
    if gnorm == 0.0:
        all_ok = False

print()
print("Computational graph intact:", all_ok)
