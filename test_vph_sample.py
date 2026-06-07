import random
VPH_MIN, VPH_MAX, VPH_STEP, EW_FRAC_LO, EW_FRAC_HI = 300, 1200, 50, 0.35, 0.65

def _random_traffic(rng):
    n_steps = (VPH_MAX - VPH_MIN) // VPH_STEP
    total   = VPH_MIN + rng.randint(0, n_steps) * VPH_STEP
    ew_frac = rng.uniform(EW_FRAC_LO, EW_FRAC_HI)
    ew_vph  = max(VPH_STEP, round(total * ew_frac / VPH_STEP) * VPH_STEP)
    ns_vph  = max(VPH_STEP, total - ew_vph)
    return ew_vph, ns_vph

rng     = random.Random(0)
samples = [_random_traffic(rng) for _ in range(40)]
totals  = [ew + ns for ew, ns in samples]
ratios  = [ew / ns for ew, ns in samples]

print(f"Total vph  min={min(totals)}  max={max(totals)}  mean={sum(totals)/len(totals):.0f}")
print(f"EW/NS ratio  min={min(ratios):.2f}  max={max(ratios):.2f}  mean={sum(ratios)/len(ratios):.2f}")
print("\nFirst 8 samples:")
for ew, ns in samples[:8]:
    bar_ew = "█" * (ew // 50)
    bar_ns = "█" * (ns // 50)
    print(f"  EW={ew:>5} {bar_ew}")
    print(f"  NS={ns:>5} {bar_ns}  total={ew+ns}  ratio={ew/ns:.2f}")
    print()
