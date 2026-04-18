"""Unit tests for `collate_v14_batch` (plan Task B2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.v14.dataset import (
    N_TIER1_PARCELS,
    T_SAMPLES,
    build_sample_from_epoch,
    collate_v14_batch,
)


def _make_sample(token: str, *, patient_id: str = "SXX", n_e: int = 16):
    layout = np.array([(i // 4, (i % 4)) for i in range(n_e)], dtype=np.int64)
    support = np.zeros((n_e, N_TIER1_PARCELS), dtype=np.float32)
    active = np.ones(n_e, dtype=bool)
    signal = np.zeros((n_e, T_SAMPLES), dtype=np.float32)
    return build_sample_from_epoch(
        signal=signal,
        token=token,
        patient_id=patient_id,
        electrode_grid_layout=layout,
        electrode_grid_shape=(8, 16),
        electrode_active_mask=active,
        support=support,
    )


class TestCollate:
    def test_stacks_two_samples_and_builds_teacher_forcing(self) -> None:
        samples = [_make_sample("bak"), _make_sample("gub")]
        batch = collate_v14_batch(samples)
        assert batch["patient_id"] == "SXX"
        assert batch["signal"].shape == (2, 16, T_SAMPLES)
        assert batch["signal"].dtype == torch.float32
        assert batch["electrode_grid_layout"].shape == (2, 16, 2)
        assert batch["electrode_grid_shape"] == (8, 16)
        assert batch["electrode_active_mask"].shape == (2, 16)
        assert batch["electrode_active_mask"].dtype == torch.bool
        assert batch["support"].shape == (2, 16, N_TIER1_PARCELS)
        assert batch["labels"].shape == (2, 3)
        assert torch.equal(
            batch["labels"],
            torch.tensor([[2, 0, 5], [3, 7, 2]], dtype=torch.long),
        )
        # Teacher forcing: prev_tokens = [[-1, label[0], label[1]] per row]
        assert torch.equal(
            batch["prev_tokens"],
            torch.tensor([[-1, 2, 0], [-1, 3, 7]], dtype=torch.long),
        )

    def test_rejects_mixed_patient_id(self) -> None:
        s0 = _make_sample("bak", patient_id="S14")
        s1 = _make_sample("gub", patient_id="S26")
        with pytest.raises(ValueError, match="patient_id"):
            collate_v14_batch([s0, s1])

    def test_rejects_mixed_N_e(self) -> None:
        s0 = _make_sample("bak", n_e=16)
        s1 = _make_sample("gub", n_e=12)
        # N_e=12 layout needs grid_shape that contains it; synth fits in 8x16
        with pytest.raises(ValueError, match="N_e"):
            collate_v14_batch([s0, s1])

    def test_rejects_empty_batch(self) -> None:
        with pytest.raises(ValueError, match="empty batch"):
            collate_v14_batch([])

    def test_prev_tokens_dtype_matches_labels(self) -> None:
        batch = collate_v14_batch([_make_sample("bak")])
        assert batch["prev_tokens"].dtype == batch["labels"].dtype == torch.long
