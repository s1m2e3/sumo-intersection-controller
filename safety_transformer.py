"""
safety_transformer.py

Permutation-equivariant transformer that predicts the RKHS safety correction
delta_a_i for every tracked vehicle in one forward pass.

Token for vehicle i  (29 dims total):
  v_n          [1]   current speed / V_MAX
  d_junc_n     [1]   distance to junction stop-line / ARM_LEN
  eta_n        [1]   d_junc / (v * ETA_NORM)  — arrival-time proxy, clamp[0,1]
  gap_n        [1]   gap to leader / GAP_NORM, 1.0 if free-flow
  v_lead_n     [1]   leader speed / V_MAX, ego speed if no leader
  v_hist       [5]   last SEQ_LEN speeds / V_MAX            (oldest → newest)
  gap_hist     [5]   last SEQ_LEN gaps / GAP_NORM
  vlead_hist   [5]   last SEQ_LEN leader speeds / V_MAX
  prev_da      [1]   accumulated correction so far / TOTAL_CAP
  entry_dir_oh [4]   one-hot: approach arm  E / W / N / S
  movement_oh  [3]   one-hot: movement type Through / Right / Left
  is_ew        [1]   1 if East-West corridor, 0 if North-South corridor

The history gives the transformer temporal context (decelerating? gap closing?
approaching junction fast?) without the expensive per-rival ETA loop.
Spatial identity (entry_dir_oh + movement_oh + is_ew) lets attention heads
reason about which vehicles are in conflict and which axis yields.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from conflict import STREAM_NAMES, ConflictSnapshot
from ttc import HORIZON, ProjectionInfo, _EPS_V

# ── normalisation constants ───────────────────────────────────────────────────

V_MAX    = 13.89   # m/s
ARM_LEN  = 200.0   # m
ETA_NORM = 20.0    # s
GAP_NORM = 100.0   # m

# kept for external imports
DT        = 0.2
DIST_NORM = HORIZON * V_MAX * DT

SEQ_LEN = 5   # observation history length — must match train script SEQ_LEN

# set at runtime by train_safety_transformer_online.py
TOTAL_CAP = None

# ── spatial identity maps ─────────────────────────────────────────────────────

_ENTRY_DIR_IDX: dict[str, int] = {"E": 0, "W": 1, "N": 2, "S": 3}
_MOVEMENT_IDX:  dict[str, int] = {"T": 0, "R": 1, "L": 2}
_AXIS_OF:       dict[str, int] = {"EW": 0, "WE": 0, "NS": 1, "SN": 1}

TOKEN_DIM = (
    1          # v_n
    + 1        # d_junc_n
    + 1        # eta_n
    + 1        # gap_n
    + 1        # v_lead_n
    + SEQ_LEN  # v_hist
    + SEQ_LEN  # gap_hist
    + SEQ_LEN  # vlead_hist
    + 1        # prev_da
    + 4        # entry_dir_oh
    + 3        # movement_oh
    + 1        # is_ew
)  # = 29

TARGET_CLIP = 5.0   # m/s²


# ── token builder ─────────────────────────────────────────────────────────────

def build_tokens(
    proj:       ProjectionInfo,
    snapshot:   ConflictSnapshot,
    x_seq_dict: dict | None   = None,   # vid -> Tensor[SEQ_LEN, 3]  (v, gap, vlead)
    prev_da:    "torch.Tensor | None" = None,  # [N] accumulated correction
    gap:        "torch.Tensor | None" = None,  # [N] current gap to leader (m)
    v_lead:     "torch.Tensor | None" = None,  # [N] leader speed (m/s)
    # legacy kwarg — accepted but ignored (kept for call-site compatibility)
    d_junc_lead: "torch.Tensor | None" = None,
) -> tuple[torch.Tensor, list[str]]:
    """
    Build token matrix for all tracked vehicles.  O(N) — no per-rival loops.

    Uses current state (v, d_junc, gap, v_lead) plus SEQ_LEN-step history from
    x_seq_dict, and static spatial identity (stream name parsed into one-hots).
    """
    vids = proj.tracked
    if not vids:
        return torch.zeros(0, TOKEN_DIM), []

    dev = proj.v_traj.device
    rows = []

    for i, vid in enumerate(vids):
        stream = snapshot.vehicle_stream.get(vid)

        v0 = proj.v_traj[i, 0].clamp(min=_EPS_V)
        dj = proj.d_junc[i].clamp(min=0.0)

        v0_n  = (v0  / V_MAX).clamp(0.0, 1.0).unsqueeze(0)
        dj_n  = (dj  / ARM_LEN).clamp(0.0, 1.0).unsqueeze(0)
        eta_n = (dj  / (v0 * ETA_NORM)).clamp(0.0, 1.0).unsqueeze(0)

        gap_n_val  = ((gap[i:i+1]   / GAP_NORM).clamp(0.0, 1.0)
                      if gap   is not None else torch.ones(1,  device=dev))
        vlead_n    = ((v_lead[i:i+1] / V_MAX).clamp(0.0, 1.0)
                      if v_lead is not None else v0_n.clone())

        # ── SEQ_LEN-step history ──────────────────────────────────────────────
        if x_seq_dict is not None and vid in x_seq_dict:
            hist       = x_seq_dict[vid].to(dev)          # [SEQ_LEN, 3]
            v_hist     = (hist[:, 0] / V_MAX).clamp(0.0, 1.0)
            gap_hist   = (hist[:, 1] / GAP_NORM).clamp(0.0, 1.0)
            vlead_hist = (hist[:, 2] / V_MAX).clamp(0.0, 1.0)
        else:
            v_hist     = v0_n.expand(SEQ_LEN)
            gap_hist   = gap_n_val.expand(SEQ_LEN)
            vlead_hist = vlead_n.expand(SEQ_LEN)

        # ── accumulated correction ────────────────────────────────────────────
        if prev_da is not None and TOTAL_CAP:
            da_n = (prev_da[i:i+1] / TOTAL_CAP).clamp(-1.0, 1.0)
        else:
            da_n = torch.zeros(1, device=dev)

        # ── spatial identity ──────────────────────────────────────────────────
        entry_dir_oh = torch.zeros(4, device=dev)
        movement_oh  = torch.zeros(3, device=dev)
        is_ew        = torch.zeros(1, device=dev)
        sname = STREAM_NAMES.get(stream, "") if stream is not None else ""
        if sname:
            approach, mvmt = sname.split("_")
            d_idx = _ENTRY_DIR_IDX.get(approach[0], -1)
            m_idx = _MOVEMENT_IDX.get(mvmt, -1)
            if d_idx >= 0:
                entry_dir_oh[d_idx] = 1.0
            if m_idx >= 0:
                movement_oh[m_idx] = 1.0
            is_ew[0] = float(_AXIS_OF.get(approach, 1) == 0)

        rows.append(torch.cat([
            v0_n,          # [1]
            dj_n,          # [1]
            eta_n,         # [1]
            gap_n_val,     # [1]
            vlead_n,       # [1]
            v_hist,        # [SEQ_LEN]
            gap_hist,      # [SEQ_LEN]
            vlead_hist,    # [SEQ_LEN]
            da_n,          # [1]
            entry_dir_oh,  # [4]
            movement_oh,   # [3]
            is_ew,         # [1]
        ]))

    return torch.stack(rows), vids


# ── model ─────────────────────────────────────────────────────────────────────

class SafetyTransformer(nn.Module):
    """
    Input : tokens [B, N, TOKEN_DIM]
    Output: corrections [B, N]  (raw logits — caller applies softsign * IDM_CAP)

    Pre-LN transformer encoder, permutation-equivariant over vehicles.
    No positional encoding — vehicle order is arbitrary.
    """

    def __init__(
        self,
        d_model:    int   = 128,
        nhead:      int   = 4,
        num_layers: int   = 4,
        dim_ff:     int   = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(TOKEN_DIM, d_model),
            nn.LayerNorm(d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_ff,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head    = nn.Linear(d_model, 1)

    def forward(
        self,
        tokens:   torch.Tensor,               # [B, N, TOKEN_DIM]
        pad_mask: torch.Tensor | None = None,  # [B, N] bool, True = padded
    ) -> torch.Tensor:                         # [B, N]
        x = self.input_proj(tokens)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(x).squeeze(-1)


# ── collation for variable-N batches ─────────────────────────────────────────

def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_n = max(t.shape[0] for t, _ in batch)
    B     = len(batch)
    tok = torch.zeros(B, max_n, TOKEN_DIM)
    tgt = torch.zeros(B, max_n)
    msk = torch.ones(B, max_n, dtype=torch.bool)
    for i, (tokens, targets) in enumerate(batch):
        n = tokens.shape[0]
        tok[i, :n] = tokens
        tgt[i, :n] = targets
        msk[i, :n] = False
    return tok, tgt, msk
