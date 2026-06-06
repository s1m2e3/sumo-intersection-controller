import torch
from pathlib import Path

snaps = torch.load(Path("logs/epoch_snapshots.pt"), weights_only=False)

print(f"{'epoch':>7}  {'TTC_min':>10}  {'TTC_zeros':>10}  "
      f"{'h_min':>8}  {'h_max':>8}  {'fhat_min':>9}  {'fhat_max':>9}  {'ISSUES':}")
print("-" * 90)

total_zero_ttc = 0
total_steps    = 0

for epoch in sorted(snaps.keys()):
    s = snaps[epoch]

    ttc  = [v for v in s["ttc_min"]   if v == v]   # drop nan
    hmax = [v for v in s["h_max"]     if v == v]
    hmin = [v for v in s["h_min"]     if v == v]
    fmax = [v for v in s["fhat_max"]  if v == v]
    fmin = [v for v in s["fhat_min"]  if v == v]

    if not ttc:
        continue

    n_zero = sum(1 for v in ttc if v < 0.01)   # essentially 0
    total_zero_ttc += n_zero
    total_steps    += len(ttc)

    issues = []
    if n_zero > 0:
        issues.append(f"TTC≈0 x{n_zero}")
    if hmax and max(hmax) > 3.05:
        issues.append(f"h_=>3 ({max(hmax):.2f})")
    if hmin and min(hmin) < -3.05:
        issues.append(f"h_=<-3 ({min(hmin):.2f})")
    if fmax and max(fmax) > 1.01:
        issues.append(f"fhat>{1:.2f} ({max(fmax):.3f})")
    if fmin and min(fmin) < -1.01:
        issues.append(f"fhat<-1 ({min(fmin):.3f})")

    print(f"{epoch:>7}  {min(ttc):>10.4f}  {n_zero:>10}  "
          f"{min(hmin):>8.4f}  {max(hmax):>8.4f}  "
          f"{min(fmin):>9.4f}  {max(fmax):>9.4f}  "
          f"{'  '.join(issues) if issues else 'OK'}")

print("-" * 90)
print(f"\nTotal steps checked : {total_steps}")
print(f"Steps with TTC≈0    : {total_zero_ttc}  "
      f"({'SAFE' if total_zero_ttc == 0 else 'WARNING — vehicles touching!'})")
print(f"\nExpected bounds:")
print(f"  TTC_min  > 0     always")
print(f"  h_=      in [-3, +3] m/s²  (±3·a_max clamp)")
print(f"  f_hat    in (-1, +1) m/s²  (tanh·a_max)")
