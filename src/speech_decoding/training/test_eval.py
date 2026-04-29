"""Unit tests for slot-averaged PER (Task D2)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.training.eval import slot_averaged_per


class TestSlotAveragedPER:
    def test_exact_match_is_zero(self) -> None:
        pred = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        assert slot_averaged_per(pred, pred.clone()) == 0.0

    def test_all_wrong_is_one(self) -> None:
        pred = torch.zeros(4, 3, dtype=torch.long)
        true = torch.ones(4, 3, dtype=torch.long)
        assert slot_averaged_per(pred, true) == 1.0

    def test_one_wrong_out_of_six_is_one_sixth(self) -> None:
        pred = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        true = pred.clone()
        true[0, 0] = 9
        assert slot_averaged_per(pred, true) == pytest.approx(1.0 / 6.0)

    def test_rejects_shape_mismatch(self) -> None:
        pred = torch.zeros(2, 3, dtype=torch.long)
        true = torch.zeros(2, 4, dtype=torch.long)
        with pytest.raises(ValueError, match=r"\(B, L\)"):
            slot_averaged_per(pred, true)

    def test_rejects_wrong_rank(self) -> None:
        with pytest.raises(ValueError, match=r"\(B, L\)"):
            slot_averaged_per(torch.zeros(6, dtype=torch.long), torch.zeros(6, dtype=torch.long))

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            slot_averaged_per(torch.zeros(0, 3, dtype=torch.long), torch.zeros(0, 3, dtype=torch.long))
