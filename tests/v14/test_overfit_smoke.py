"""Tiny-subset overfit smoke gate (Task D4).

A trainable model with 4 trials and 200 AdamW steps must drive slot CE below
0.05 and exhaustive-decode trial-level exact match above 0.75. This is the
last structural check before DCC submission — a failure here points to grid
reshape in `GridMixer`, `parcel_emb` broadcast across `T`, decoder BOS, or
`token_active_mask` zeroing legitimate cells.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.v14.dataset import N_TIER1_PARCELS, T_SAMPLES
from speech_decoding.v14.model import NeuralFieldPerceiver
from speech_decoding.v14.train import LR, WD, compute_slot_ce, train_step


def _fixed_overfit_batch() -> dict:
    torch.manual_seed(0)
    B, N_e, H_p, W_p = 4, 16, 8, 16
    signal = torch.randn(B, N_e, T_SAMPLES)
    layout = torch.tensor(
        [[[i // W_p, i % W_p] for i in range(N_e)] for _ in range(B)],
        dtype=torch.long,
    )
    active = torch.ones(B, N_e, dtype=torch.bool)
    support = torch.rand(B, N_e, N_TIER1_PARCELS)
    labels = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [1, 3, 5]], dtype=torch.long
    )
    prev_tokens = torch.full((B, 3), -1, dtype=torch.long)
    prev_tokens[:, 1:] = labels[:, :-1]
    return {
        "signal": signal,
        "electrode_grid_layout": layout,
        "electrode_grid_shape": (H_p, W_p),
        "electrode_active_mask": active,
        "support": support,
        "labels": labels,
        "prev_tokens": prev_tokens,
        "patient_id": "SXX",
    }


@pytest.mark.slow
def test_overfit_smoke_gate() -> None:
    torch.manual_seed(0)
    model = NeuralFieldPerceiver()
    batch = _fixed_overfit_batch()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    final_loss = 1.0
    for _ in range(200):
        final_loss = train_step(model, batch, opt)

    assert final_loss < 0.05, f"final CE {final_loss:.4f} >= 0.05"

    model.eval()
    with torch.no_grad():
        # Direct teacher-forced check first (cheap)
        logits = model(batch, mode="train")
        ce = compute_slot_ce(logits, batch["labels"]).item()
        assert ce < 0.05, f"teacher-forced CE {ce:.4f} >= 0.05"
        # Then the actual exhaustive decode (the contract for eval)
        pred = model(batch, mode="eval")

    trial_match = (pred == batch["labels"]).all(dim=1).float().mean().item()
    assert trial_match > 0.75, f"trial-level exact match {trial_match:.2f} <= 0.75"
