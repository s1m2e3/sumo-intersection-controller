"""
Verifies:
  1. TTC* distribution actually hits the [3,5] learning region
  2. f_hat parameters receive non-zero gradients
"""
import os, torch
from pathlib import Path
import sumo as _sumo_pkg
from model import HybridModel, KernelCorrection
from simulator import build, load_config
from train import run_epoch

SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
os.environ["PATH"]      = str(SUMO_BIN) + os.pathsep + os.environ.get("PATH", "")

sumocfg = build()
cfg     = load_config()
dt      = cfg["step_length"]

model = HybridModel()
for p in model.physics.parameters():
    p.requires_grad_(False)

# --- patch correction.forward to log TTC* values ---
ttc_log = []
_orig_correction = model.correction.forward

def _instrumented_correction(v, gap, v_lead, f_hat):
    with torch.no_grad():
        ttc = model.correction.ttc_star(v, gap, v_lead)
        ttc_log.extend(ttc.tolist())
    return _orig_correction(v, gap, v_lead, f_hat)

model.correction.forward = _instrumented_correction

# --- run one epoch with random spawn speeds ---
velocity_sum = run_epoch(sumocfg, dt, model)
(-velocity_sum).backward()

# --- TTC* distribution ---
ttc_t  = torch.tensor(ttc_log)
mu_ff  = model.correction.mu_ff(ttc_t)
mu_dec = model.correction.mu_dec(ttc_t)
in_nn  = ((mu_ff == 0) & (mu_dec == 0))

print(f"TTC* distribution over epoch:")
print(f"  total samples      : {len(ttc_log)}")
print(f"  min                : {ttc_t.min():.2f}s")
print(f"  mean               : {ttc_t.mean():.2f}s")
print(f"  % in [0,3)  dec    : {100*(mu_dec>0).float().mean():.1f}%")
print(f"  % in [3,5]  NN     : {100*in_nn.float().mean():.1f}%")
print(f"  % in (5,∞)  ff     : {100*(mu_ff>0).float().mean():.1f}%")

# --- gradient check ---
print(f"\nf_hat gradients after backward:")
any_nonzero = False
for name, p in model.f_hat.named_parameters():
    if p.grad is not None:
        gmax = p.grad.abs().max().item()
        if gmax > 0:
            any_nonzero = True
            print(f"  {name:<30}  grad_max={gmax:.2e}")
        else:
            print(f"  {name:<30}  grad=0 (in graph but zero)")
    else:
        print(f"  {name:<30}  grad=None (not in graph)")

print(f"\nphysics gradients (should be None — frozen):")
for name, p in model.physics.named_parameters():
    print(f"  {name:<20}  grad={p.grad}")

print(f"\nGradient flow OK: {any_nonzero}")
