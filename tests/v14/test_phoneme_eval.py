"""Unit tests for per-phoneme eval helpers (plan P5)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.v14.eval import (
    evaluate_per_phoneme,
    exhaustive_ar_per,
)
from speech_decoding.v14.phoneme_decoder import D1Decoder


class TestExhaustiveARPER:
    def test_perfect_decoder_recovers_labels(self) -> None:
        """When the decoder's logits are a one-hot matching the label, the
        argmax hypothesis must equal the ground-truth triplet."""
        torch.manual_seed(0)

        # Build 2 trials with known labels.
        labels = torch.tensor([[2, 0, 5], [3, 7, 2]], dtype=torch.long)

        # Fake decoder: return a signal proportional to the label, via a
        # fixed memory.mean() map. We mock by hooking a decoder subclass.
        class LabelDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vocab_size = 9
            def forward(self, memory, prev, cell_query=None):  # noqa: ARG002
                # memory[:, :, 0] encodes the label index; argmax on one-hot
                # at that index.
                idx = memory[:, 0, 0].long().clamp(0, 8)
                out = torch.full((memory.shape[0], 9), -100.0)
                out[torch.arange(memory.shape[0]), idx] = 10.0
                return out

        dec = LabelDecoder()
        # Pack labels into memory[:, 0, 0].
        T, d = 2, 4
        mem0 = torch.zeros(T, 5, d)
        mem1 = torch.zeros(T, 5, d)
        mem2 = torch.zeros(T, 5, d)
        mem0[:, 0, 0] = labels[:, 0].float()
        mem1[:, 0, 0] = labels[:, 1].float()
        mem2[:, 0, 0] = labels[:, 2].float()

        per = exhaustive_ar_per([mem0, mem1, mem2], labels, dec)
        assert per == pytest.approx(0.0)

    def test_rejects_bad_shapes(self) -> None:
        dec = D1Decoder()
        mem = torch.randn(3, 5, 32)
        bad_labels = torch.zeros(2, 3, dtype=torch.long)
        with pytest.raises(ValueError, match="labels shape"):
            exhaustive_ar_per([mem, mem, mem], bad_labels, dec)

        with pytest.raises(ValueError, match="expected memories"):
            exhaustive_ar_per([mem, mem], torch.zeros(3, 3, dtype=torch.long), dec)


class TestEvaluatePerPhoneme:
    def test_chance_on_random_model(self) -> None:
        """A random-weight model should return ~chance error rate."""

        class RandomModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(1, 9)
            def forward(self, batch):
                B = batch["labels"].shape[0]
                return torch.randn(B, 9)

        torch.manual_seed(0)
        model = RandomModel()
        # 81 samples — enough for chance to show up cleanly.
        batches = []
        for _ in range(9):
            batches.append({
                "labels": torch.randint(0, 9, (9,), dtype=torch.long),
                "signal": torch.zeros(9, 1),
            })
        per = evaluate_per_phoneme(model, batches)
        # Chance = 8/9 ≈ 0.889. Allow ±0.15 on 81 samples.
        assert 0.7 < per < 1.0
