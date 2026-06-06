import torch
from model import KernelCorrection, IDMPhysics

p  = IDMPhysics()
kc = KernelCorrection(p)

ttcs = torch.tensor([0.0, 1.0, 2.0, 2.5, 2.75, 3.0, 4.0, 5.0, 5.5, 6.0])
mff  = kc.mu_ff(ttcs)
mdec = kc.mu_dec(ttcs)

print(f"{'TTC*':>6}  {'mu_dec':>8}  {'mu_ff':>8}  {'who drives'}")
print("-" * 55)
for z, mf, md in zip(ttcs, mff, mdec):
    if   md == 1 and mf == 0:    driver = "u_dec  full braking"
    elif md >  0 and mf == 0:    driver = f"u_dec  partial ({md.item():.2f})"
    elif md == 0 and mf == 0:    driver = "f_hat  NN alone"
    elif mf >  0 and md == 0:    driver = f"u_ff   partial ({mf.item():.2f})"
    else:                        driver = "u_ff   full free-flow"
    print(f"{z.item():>6.2f}  {md.item():>8.4f}  {mf.item():>8.4f}  {driver}")
