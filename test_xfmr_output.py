import torch
import torch.nn.functional as F
from safety_transformer import SafetyTransformer, TOKEN_DIM, SEQ_LEN

IDM_CAP = 1.5

def test_ckpt(path):
    m = SafetyTransformer()
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    m.eval()
    v_vals = [4., 7., 2., 9., 3.5, 6., 1.5, 8.]
    tokens = []
    for v in v_vals:
        tok = ([v, 15., 3., 5., v+1.]
               + [v]*SEQ_LEN + [5.]*SEQ_LEN + [v+1.]*SEQ_LEN
               + [0.] + [1.,0.,0.,0.] + [1.,0.,0.] + [1.])
        tokens.append(tok)
    x = torch.tensor(tokens).unsqueeze(0)
    with torch.no_grad():
        raw = m(x).squeeze()
        da = F.softsign(raw) * IDM_CAP
    print(f"{path}:")
    print(f"  raw[0]={raw[0].item():.3f}  da[0]={da[0].item():.4f}")
    print(f"  max_da={da.max().item():.4f}  min_da={da.min().item():.4f}")
    print(f"  any > 0: {(da > 0).any().item()}")

test_ckpt('models/safety_transformer_online_best.pt')
test_ckpt('models/safety_transformer_online_latest.pt')
test_ckpt('models/checkpoints_online/ep_0010.pt')
