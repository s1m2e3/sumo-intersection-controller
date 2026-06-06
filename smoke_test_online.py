import torch
from safety_transformer import SafetyTransformer, TOKEN_DIM
from train_safety_transformer_online import (
    _load_hybrid, _load_transformer,
    _project_diff, compute_violation_diff, DEVICE, HORIZON
)

print(f"device={DEVICE}")
hybrid = _load_hybrid()
xfmr   = _load_transformer()
print(f"HybridModel params : {sum(p.numel() for p in hybrid.parameters()):,}")
print(f"Transformer params : {sum(p.numel() for p in xfmr.parameters()):,}")

# synthetic forward pass
N = 6
v0     = torch.rand(N, device=DEVICE) * 13.0
gap0   = torch.rand(N, device=DEVICE) * 50 + 5
vl0    = torch.rand(N, device=DEVICE) * 13.0
lidx   = torch.full((N,), -1, dtype=torch.long)
tokens = torch.randn(1, N, TOKEN_DIM, device=DEVICE)

delta_a_n = xfmr(tokens).squeeze(0)   # [N], has grad
print(f"delta_a_n requires_grad={delta_a_n.requires_grad}  shape={tuple(delta_a_n.shape)}")

xs0 = torch.zeros(N, 5, 3, device=DEVICE)   # SEQ_LEN=5, obs_dim=3
v_traj, cum_dist = _project_diff(v0, gap0, vl0, hybrid, xs0, lidx, delta_a_n, 0.1, HORIZON)
print(f"v_traj   requires_grad={v_traj.requires_grad}   shape={tuple(v_traj.shape)}")
print(f"cum_dist requires_grad={cum_dist.requires_grad}  shape={tuple(cum_dist.shape)}")

# fake violation loss
V = (v_traj + cum_dist).sum() * 1e-6
V.backward()
print(f"grad on delta_a_n after backward: {delta_a_n.grad is not None}")
print(f"grad norm on transformer head   : "
      f"{xfmr.head.weight.grad.norm().item():.6f}")
print("OK — graph flows correctly")
