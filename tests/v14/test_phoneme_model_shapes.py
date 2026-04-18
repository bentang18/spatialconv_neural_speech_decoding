"""Shape + overfit tests for NeuralFieldPerceiverPerPhoneme (plan P4)."""

from __future__ import annotations

import torch

from speech_decoding.v14.phoneme_dataset import (
    BOS_TOKEN,
    N_TIER1_PARCELS,
    T_RAW_SAMPLES,
)
from speech_decoding.v14.phoneme_model import NeuralFieldPerceiverPerPhoneme


def _fake_batch(
    *, B: int = 4, N_e: int = 16, H_p: int = 8, W_p: int = 16, seed: int = 0
) -> dict:
    """Fake per-phoneme batch shaped like the collator output."""
    torch.manual_seed(seed)
    signal = torch.randn(B, N_e, T_RAW_SAMPLES)
    rows = torch.arange(H_p).repeat_interleave(W_p)[:N_e]
    cols = torch.arange(W_p).repeat(H_p)[:N_e]
    layout_per = torch.stack([rows, cols], dim=1)
    layout = layout_per.unsqueeze(0).expand(B, -1, -1).contiguous()
    active = torch.ones(B, N_e, dtype=torch.bool)
    support = torch.rand(B, N_e, N_TIER1_PARCELS)
    labels = torch.randint(0, 9, (B,), dtype=torch.long)
    prev_tokens = torch.tensor(
        [BOS_TOKEN, 0, 1, 2][:B], dtype=torch.long
    )[:B]
    # Pad if B > 4
    if B > 4:
        extra = torch.randint(0, 9, (B - 4,), dtype=torch.long)
        prev_tokens = torch.cat([prev_tokens, extra], dim=0)
    return {
        "signal": signal,
        "electrode_grid_layout": layout,
        "electrode_grid_shape": (H_p, W_p),
        "electrode_active_mask": active,
        "support": support,
        "labels": labels,
        "prev_tokens": prev_tokens,
        "phoneme_pos": torch.zeros(B, dtype=torch.long),
        "trial_id": torch.arange(B, dtype=torch.long),
        "patient_id": "SXX",
    }


class TestForwardShape:
    def test_full_128_strip_shapes(self) -> None:
        model = NeuralFieldPerceiverPerPhoneme()
        batch = _fake_batch(B=4, N_e=128, H_p=8, W_p=16)
        logits = model(batch)
        assert logits.shape == (4, 9)

    def test_256_grid_shapes(self) -> None:
        model = NeuralFieldPerceiverPerPhoneme()
        batch = _fake_batch(B=2, N_e=256, H_p=16, W_p=16)
        logits = model(batch)
        assert logits.shape == (2, 9)

    def test_encode_memory_shape(self) -> None:
        """Memory is (B, n_cells * T_tokens, d)."""
        model = NeuralFieldPerceiverPerPhoneme()
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        memory = model.encode_memory(batch)
        # (4,8) pool = 32 cells; Conv1d 130→11 tokens; d=32.
        assert memory.shape == (2, 32 * 11, 32)


class TestParamBudget:
    def test_total_params_near_plan_target(self) -> None:
        """Plan target is ~45k total params."""
        model = NeuralFieldPerceiverPerPhoneme()
        total = sum(p.numel() for p in model.parameters())
        assert 30_000 <= total <= 60_000, f"got {total}, expected ~45k"


class TestOverfit:
    def test_ten_steps_drop_loss_by_at_least_0_1(self) -> None:
        """10 AdamW steps on a fixed batch should drive CE down by ≥0.1."""
        torch.manual_seed(0)
        model = NeuralFieldPerceiverPerPhoneme()
        batch = _fake_batch(B=4, N_e=128, H_p=8, W_p=16)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()

        losses: list[float] = []
        for _ in range(10):
            opt.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        assert losses[0] - losses[-1] >= 0.1, f"losses={losses}"
