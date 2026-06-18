"""Collect the learned prior-mean f over the kernel's operating points, sweeping
seeds × flows.  Harvests (g, τ_c, r, f) at the ACTUAL operating φ* every vehicle-step
(sim_torch f_log hook) and saves the raw samples.  Plotting is a SEPARATE step
(plot_f.py) so the figure can be re-rendered without re-simulating.

    conda run -n car-following-sumo python collect_f.py            # 25 seeds × 300/400/500, LATEST model
    conda run -n car-following-sumo python collect_f.py 10 300,400 # custom seeds/flows
    conda run -n car-following-sumo python collect_f.py best        # force the BEST checkpoint

Output: outputs/f_samples.npz  (data [N,4] = g, τ_c, r, f; plus flows, nseed).
"""
import os, sys
import numpy as np
import torch

import sim_torch as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)

USE_BEST = "best" in sys.argv[1:]                       # default = LATEST model
args  = [a for a in sys.argv[1:] if a != "best"]
NSEED = int(args[0]) if len(args) > 0 else 25
FLOWS = [int(x) for x in args[1].split(",")] if len(args) > 1 else [500, 600, 700]


def load_model():
    last = os.path.join(HERE, "mean_net_last.pt")       # saved every epoch by train_mean
    best = os.path.join(HERE, "mean_net_ckpt.pt")       # best-by-score
    # default LATEST (where training ended up); else BEST.  Skip any checkpoint with
    # non-finite weights (a diverged run leaves NaN/Inf → would harvest NaN f).
    import mean_net
    order = [best, last] if USE_BEST else [last, best]
    for path in order:
        if not os.path.exists(path):
            continue
        ck = torch.load(path, weights_only=True)
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        m = mean_net.MeanTransformer(); m.load_state_dict(sd); m.eval()
        if all(torch.isfinite(p).all() for p in m.parameters()):
            print(f"loaded model: {os.path.basename(path)}")
            return m
        print(f"  ({os.path.basename(path)} has non-finite weights (diverged) — skipping)")
    sys.exit("no usable checkpoint (mean_net_last.pt / mean_net_ckpt.pt) — train first.")


def main():
    geo = S.build_geometry()
    m = load_model()
    rows = []
    for flow in FLOWS:
        for seed in range(NSEED):
            log = []
            S.simulate(flow, seed, *geo, mean_model=m, f_log=log)
            n = sum(int(t.shape[0]) for t in log)
            if log:
                rows.append(torch.cat(log, dim=0))      # [steps*Na, 4]
            print(f"  flow={flow} seed={seed:2d}  samples={n}", flush=True)
    data = torch.cat(rows, dim=0).numpy()               # [N, 4] = g, τ_c, r, f
    path = os.path.join(OUT, "f_samples.npz")
    np.savez(path, data=data, flows=FLOWS, nseed=NSEED)
    f = data[:, 3]
    print(f"\ntotal samples: {len(f)}  |f| mean={np.abs(f).mean():.4f}  max={np.abs(f).max():.4f}")
    print(f"saved {path}\nnow render with:  python plot_f.py")


if __name__ == "__main__":
    main()
