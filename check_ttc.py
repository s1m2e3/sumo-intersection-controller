import torch
from pathlib import Path

snaps = torch.load(Path("logs/epoch_snapshots.pt"), weights_only=False)

lowest_ttc  = float("inf")
worst_epoch = None
worst_step  = None

below_01 = []

for epoch, s in snaps.items():
    for step, v in enumerate(s["ttc_min"]):
        if v != v:          # nan
            continue
        if v < lowest_ttc:
            lowest_ttc  = v
            worst_epoch = epoch
            worst_step  = step
        if v < 0.1:
            below_01.append((epoch, step, v))

print(f"Lowest TTC* : {lowest_ttc:.6f} s")
print(f"  at epoch {worst_epoch}, step {worst_step} (t = {worst_step * 0.1:.1f}s)")
print()
if below_01:
    print(f"Steps with TTC* < 0.1s  ({len(below_01)} total):")
    for ep, st, v in sorted(below_01, key=lambda x: x[2]):
        print(f"  epoch={ep:>4}  step={st:>3} (t={st*0.1:.1f}s)  TTC*={v:.6f}s")
else:
    print("No steps with TTC* < 0.1s — all safe.")

# distribution summary
all_vals = [v for s in snaps.values() for v in s["ttc_min"] if v == v]
import statistics
finite = [v for v in all_vals if v < 1000]
print(f"\nTTC* summary across {len(all_vals)} total timesteps:")
print(f"  min    = {min(all_vals):.4f} s")
print(f"  median = {statistics.median(finite):.4f} s  (finite values only)")
print(f"  mean   = {sum(finite)/len(finite):.4f} s  (finite values only)")
print(f"  < 0.5s : {sum(1 for v in finite if v < 0.5)} steps")
print(f"  < 1.0s : {sum(1 for v in finite if v < 1.0)} steps")
print(f"  < 3.0s : {sum(1 for v in finite if v < 3.0)} steps  (dec region)")
print(f"  3-5s   : {sum(1 for v in finite if 3.0 <= v <= 5.0)} steps  (NN region)")
print(f"  > 5s   : {sum(1 for v in finite if v > 5.0)} steps  (ff region)")
