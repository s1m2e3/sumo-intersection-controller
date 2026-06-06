"""
Health check for the trained HybridModel.

Runs one full epoch with the best saved model and reports per-step:
  - Collisions detected by SUMO
  - TTC* distribution across all vehicles
  - Which regime each vehicle is in (dec / NN / ff)
  - Whether h_= is correctly anchoring to physics at the boundaries
  - Any vehicles exceeding v_max (sign of uncontrolled acceleration)

Output: logs/health_check.txt
"""

import os
import torch
import traci
from pathlib import Path
from datetime import datetime

import sumo as _sumo_pkg
from model import HybridModel
from simulator import build, load_config
from train import _read_gap_vlead

SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
os.environ["PATH"]      = str(SUMO_BIN) + os.pathsep + os.environ.get("PATH", "")

LOG_PATH      = Path("logs") / "health_check.txt"
MODEL_PATH    = Path("models") / "hybrid_model_best.pt"
EPOCH_DURATION = 10.0


def run_health_check():
    sumocfg = build()
    cfg     = load_config()
    dt      = cfg["step_length"]
    v_max   = cfg["speed_limit"]
    n_steps = int(EPOCH_DURATION / dt)

    # ── load trained model ──────────────────────────────────────────────────
    model = HybridModel()
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model_label = f"trained ({MODEL_PATH})"
    else:
        model_label = "untrained (random init) — model file not found"
    model.eval()

    traci.start([
        str(SUMO_BIN / "sumo.exe"), "-c", str(sumocfg),
        "--step-length", str(dt),
        "--collision.action", "warn",
        "--collision.mingap-factor", "0",
    ])

    known   = set()
    v_state = {}
    lines   = []

    # ── per-step counters ───────────────────────────────────────────────────
    total_collisions   = 0
    total_overspeed    = 0
    ttc_all            = []
    regime_counts      = {"dec": 0, "nn": 0, "ff": 0}
    step_summaries     = []

    for step in range(n_steps):
        traci.simulationStep()
        current = set(traci.vehicle.getIDList())

        # ── collision check ────────────────────────────────────────────────
        collisions = traci.simulation.getCollisions()
        if collisions:
            total_collisions += len(collisions)
            for c in collisions:
                lines.append(f"  [COLLISION] step={step:>4} t={step*dt:.1f}s  "
                              f"{c.collider} → {c.victim}  type={c.type}")

        # ── new vehicles ───────────────────────────────────────────────────
        for vid in current - known:
            traci.vehicle.setSpeedMode(vid, 0)
            init_v = float(torch.zeros(1).uniform_(2.0, v_max))
            v_state[vid] = torch.tensor(init_v, dtype=torch.float32)
            traci.vehicle.setSpeed(vid, init_v)
        for vid in known - current:
            v_state.pop(vid, None)
        known = current

        vids = list(current)
        if not vids:
            continue

        with torch.no_grad():
            v       = torch.stack([v_state[vid] for vid in vids]).detach()
            gap, vl = _read_gap_vlead(vids)

            # ── regime classification ──────────────────────────────────────
            z     = model.correction.ttc_star(v, gap, vl)
            mu_ff  = model.correction.mu_ff(z)
            mu_dec = model.correction.mu_dec(z)

            in_dec = (mu_dec > 0)
            in_ff  = (mu_ff  > 0)
            in_nn  = (~in_dec) & (~in_ff)

            regime_counts["dec"] += int(in_dec.sum())
            regime_counts["nn"]  += int(in_nn.sum())
            regime_counts["ff"]  += int(in_ff.sum())
            ttc_all.extend(z.tolist())

            # ── model forward ─────────────────────────────────────────────
            accel  = model(v, gap, vl)
            v_next = torch.clamp(v + accel * dt, min=0.0)

            # ── h_= verification: check anchoring at boundaries ───────────
            x     = torch.stack([v, gap, vl], dim=-1)
            f_hat = torch.tanh(model.f_hat(x).squeeze(-1)) * model.physics.a_max
            h_eq  = model.correction(v, gap, vl, f_hat)
            f_total = f_hat + h_eq

            # in dec region: f_total should ≈ u_dec
            # in ff  region: f_total should ≈ u_ff
            u_ff_vals  = model.physics.u_ff(v)
            u_dec_vals = model.physics.u_dec(v, gap, vl)

            # ── overspeed check ────────────────────────────────────────────
            overspeed = (v_next > v_max * 1.5)    # >50% above limit = flag
            if overspeed.any():
                total_overspeed += int(overspeed.sum())
                for i in overspeed.nonzero(as_tuple=True)[0]:
                    lines.append(f"  [OVERSPEED] step={step:>4} t={step*dt:.1f}s  "
                                  f"{vids[i]} v={v_next[i].item():.2f} m/s "
                                  f"(limit={v_max:.2f})")

        # update state and SUMO
        for i, vid in enumerate(vids):
            v_state[vid] = v_next[i].detach()
            traci.vehicle.setSpeed(vid, float(v_next[i].detach()))

        # ── per-step summary (every 10 steps) ─────────────────────────────
        if step % 10 == 0:
            ttc_step = z.tolist()
            finite   = [t for t in ttc_step if t < 1000]
            step_summaries.append(
                f"  t={step*dt:>5.1f}s  veh={len(vids):>3}  "
                f"dec={int(in_dec.sum()):>3}  nn={int(in_nn.sum()):>3}  "
                f"ff={int(in_ff.sum()):>3}  "
                f"TTC_min={min(ttc_step):.2f}s  "
                f"TTC_finite={'none' if not finite else f'{min(finite):.2f}-{max(finite):.2f}s'}  "
                f"col={len(collisions):>2}  "
                f"mean_v={v.mean().item():.2f}m/s"
            )

    traci.close()

    # ── h_= anchor accuracy ─────────────────────────────────────────────────
    # run one more quick pass with fixed test vectors to verify physics anchoring
    test_ttcs     = torch.tensor([1.0, 2.0, 4.0, 5.5, 7.0])
    test_v        = torch.full((5,), 10.0)
    test_gap      = test_ttcs * 2.0           # dv=2 → gap = TTC×dv
    test_vl       = torch.full((5,), 8.0)

    with torch.no_grad():
        x_test    = torch.stack([test_v, test_gap, test_vl], dim=-1)
        fhat_test = torch.tanh(model.f_hat(x_test).squeeze(-1)) * model.physics.a_max
        heq_test  = model.correction(test_v, test_gap, test_vl, fhat_test)
        ftot_test = fhat_test + heq_test
        uFF_test  = model.physics.u_ff(test_v)
        uDec_test = model.physics.u_dec(test_v, test_gap, test_vl)
        mu_ff_t   = model.correction.mu_ff(test_ttcs)
        mu_dec_t  = model.correction.mu_dec(test_ttcs)

    anchor_lines = []
    anchor_lines.append(f"\n{'TTC*':>6}  {'regime':>10}  {'u_dec':>8}  {'f_hat':>8}  "
                        f"{'h_=':>8}  {'f_total':>8}  {'u_ff':>8}  {'anchor_ok?':>12}")
    anchor_lines.append("─" * 80)
    for i, ttc in enumerate(test_ttcs.tolist()):
        if mu_dec_t[i] > 0 and mu_ff_t[i] == 0:
            regime  = "dec"
            anchor  = abs(ftot_test[i].item() - uDec_test[i].item()) < 0.01
            anchor_str = f"≈u_dec {'OK' if anchor else 'FAIL'}"
        elif mu_ff_t[i] > 0 and mu_dec_t[i] == 0:
            regime  = "ff"
            anchor  = abs(ftot_test[i].item() - uFF_test[i].item()) < 0.01
            anchor_str = f"≈u_ff {'OK' if anchor else 'FAIL'}"
        else:
            regime  = "NN [3,5]"
            anchor  = abs(heq_test[i].item()) < 1e-5
            anchor_str = f"h_==0 {'OK' if anchor else 'FAIL'}"

        anchor_lines.append(
            f"{ttc:>6.1f}  {regime:>10}  {uDec_test[i].item():>8.4f}  "
            f"{fhat_test[i].item():>8.4f}  {heq_test[i].item():>8.4f}  "
            f"{ftot_test[i].item():>8.4f}  {uFF_test[i].item():>8.4f}  "
            f"{anchor_str:>12}"
        )

    # ── TTC* global stats ───────────────────────────────────────────────────
    ttc_t     = torch.tensor(ttc_all)
    mu_ff_g   = model.correction.mu_ff(ttc_t)
    mu_dec_g  = model.correction.mu_dec(ttc_t)
    in_nn_g   = ((mu_ff_g == 0) & (mu_dec_g == 0))
    finite_ttc = ttc_t[ttc_t < 1000]

    total_vehicle_steps = sum(regime_counts.values())

    # ── write log ───────────────────────────────────────────────────────────
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "w") as f:
        f.write("HYBRID MODEL HEALTH CHECK\n")
        f.write(f"Run at    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model     : {model_label}\n")
        f.write(f"Duration  : {EPOCH_DURATION:.0f}s  ({n_steps} steps, dt={dt}s)\n")
        f.write(f"v_max     : {v_max:.2f} m/s\n\n")

        f.write("=" * 70 + "\n")
        f.write("1. COLLISION REPORT\n")
        f.write("=" * 70 + "\n")
        if total_collisions == 0:
            f.write("  NO COLLISIONS detected across all 100 steps.\n")
        else:
            f.write(f"  WARNING: {total_collisions} collision event(s) detected!\n")
            for l in lines:
                if "COLLISION" in l:
                    f.write(l + "\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("2. OVERSPEED REPORT  (threshold: 1.5 × v_max)\n")
        f.write("=" * 70 + "\n")
        if total_overspeed == 0:
            f.write(f"  No vehicles exceeded 1.5 × v_max ({v_max*1.5:.2f} m/s).\n")
        else:
            f.write(f"  WARNING: {total_overspeed} overspeed event(s)!\n")
            for l in lines:
                if "OVERSPEED" in l:
                    f.write(l + "\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("3. TTC* DISTRIBUTION  (confirms h_= regime activation)\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Total vehicle-steps     : {total_vehicle_steps}\n")
        f.write(f"  In dec region  [0,3)    : {regime_counts['dec']:>6}  "
                f"({100*regime_counts['dec']/max(total_vehicle_steps,1):.1f}%)  "
                f"→ h_= anchors to u_dec\n")
        f.write(f"  In NN  region  [3,5]    : {regime_counts['nn']:>6}  "
                f"({100*regime_counts['nn']/max(total_vehicle_steps,1):.1f}%)  "
                f"→ f_hat works alone (learned dynamics)\n")
        f.write(f"  In ff  region  (5,∞)    : {regime_counts['ff']:>6}  "
                f"({100*regime_counts['ff']/max(total_vehicle_steps,1):.1f}%)  "
                f"→ h_= anchors to u_ff\n")
        if len(finite_ttc) > 0:
            f.write(f"\n  Finite TTC* stats (TTC* < 1000s, {len(finite_ttc)} samples):\n")
            f.write(f"    min  = {finite_ttc.min().item():.3f}s\n")
            f.write(f"    mean = {finite_ttc.mean().item():.3f}s\n")
            f.write(f"    max  = {finite_ttc.max().item():.3f}s\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("4. h_= ANCHOR VERIFICATION  (fixed test vectors)\n")
        f.write("=" * 70 + "\n")
        for l in anchor_lines:
            f.write(l + "\n")
        f.write("\n  In dec region: f_total should equal u_dec  (h_= = u_dec - f_hat)\n")
        f.write("  In NN  region: h_= should be exactly 0     (f_total = f_hat)\n")
        f.write("  In ff  region: f_total should equal u_ff   (h_= = u_ff - f_hat)\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("5. STEP-BY-STEP SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"  {'time':>7}  {'veh':>4}  {'dec':>4}  {'nn':>4}  {'ff':>4}  "
                f"{'TTC_min':>8}  {'TTC_range(finite)':>22}  {'col':>4}  {'mean_v':>8}\n")
        f.write("  " + "─" * 78 + "\n")
        for s in step_summaries:
            f.write(s + "\n")

    print(f"Health check written → {LOG_PATH}")
    print(f"Collisions   : {total_collisions}")
    print(f"Overspeed    : {total_overspeed}")
    print(f"Regime split : dec={regime_counts['dec']}  "
          f"nn={regime_counts['nn']}  ff={regime_counts['ff']}")


if __name__ == "__main__":
    run_health_check()
