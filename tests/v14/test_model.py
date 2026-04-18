"""Unit tests for the v14-core top-level model assembly (Task C5)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.v14.dataset import N_TIER1_PARCELS, T_SAMPLES
from speech_decoding.v14.model import NeuralFieldPerceiver


def _fake_batch(B: int = 2, N_e: int = 16, H_p: int = 8, W_p: int = 16) -> dict:
    """Build a batch matching the collator's output contract."""

    signal = torch.randn(B, N_e, T_SAMPLES)
    # Simple row-major placement across the grid — within bounds for (8, 16)
    layout = torch.tensor(
        [[[i // W_p, i % W_p] for i in range(N_e)] for _ in range(B)],
        dtype=torch.long,
    )
    active = torch.ones(B, N_e, dtype=torch.bool)
    active[0, -1] = False  # one inactive electrode for coverage
    support = torch.rand(B, N_e, N_TIER1_PARCELS)
    labels = torch.zeros(B, 3, dtype=torch.long)
    prev_tokens = torch.full((B, 3), -1, dtype=torch.long)
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


class TestNeuralFieldPerceiver:
    def test_train_forward_returns_B_3_9(self) -> None:
        model = NeuralFieldPerceiver().eval()
        batch = _fake_batch()
        logits = model(batch, mode="train")
        assert logits.shape == (2, 3, 9)
        assert torch.isfinite(logits).all()

    def test_eval_forward_returns_B_3_long_in_range(self) -> None:
        model = NeuralFieldPerceiver().eval()
        batch = _fake_batch()
        with torch.no_grad():
            pred = model(batch, mode="eval")
        assert pred.shape == (2, 3)
        assert pred.dtype == torch.long
        assert int(pred.min()) >= 0 and int(pred.max()) < 9

    def test_N_e_100_on_128_cell_grid(self) -> None:
        """Plan test case: N_e = 100 on an (8, 16) grid (28 cells padded)."""

        model = NeuralFieldPerceiver().eval()
        batch = _fake_batch(B=1, N_e=100, H_p=8, W_p=16)
        logits = model(batch, mode="train")
        assert logits.shape == (1, 3, 9)

    def test_backward_finite_gradients(self) -> None:
        model = NeuralFieldPerceiver()
        model.train()
        batch = _fake_batch(B=1, N_e=16)
        logits = model(batch, mode="train")
        logits.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"

    def test_rejects_invalid_mode(self) -> None:
        model = NeuralFieldPerceiver().eval()
        with pytest.raises(ValueError, match="mode"):
            model(_fake_batch(), mode="predict")  # type: ignore[arg-type]
