import torch
from safety_transformer import SafetyTransformer, collate_fn, TOKEN_DIM, N_STREAMS, HORIZON

print(f"TOKEN_DIM={TOKEN_DIM}  N_STREAMS={N_STREAMS}  HORIZON={HORIZON}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

model = SafetyTransformer().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"params={n_params:,}")

B, maxN = 4, 8
tok = torch.randn(B, maxN, TOKEN_DIM, device=device)
msk = torch.zeros(B, maxN, dtype=torch.bool, device=device)
msk[:, 7:] = True
out = model(tok, pad_mask=msk)
print(f"output shape={tuple(out.shape)}  (expected [{B}, {maxN}])")

samples = [
    (torch.randn(5, TOKEN_DIM), torch.randn(5)),
    (torch.randn(3, TOKEN_DIM), torch.randn(3)),
]
t, tgt, msk2 = collate_fn(samples)
print(f"collate: tokens={tuple(t.shape)} targets={tuple(tgt.shape)} mask={tuple(msk2.shape)}")
print("OK")
