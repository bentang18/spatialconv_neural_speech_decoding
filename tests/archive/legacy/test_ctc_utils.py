"""Tests for CTC loss wrapper, greedy decoder, and label encoding."""
import torch
import pytest

from speech_decoding.training.ctc_utils import (
    ctc_loss,
    greedy_decode,
    encode_labels_for_ctc,
    compute_per,
    blank_ratio,
)


class TestCTCLoss:
    def test_basic_loss(self):
        """Loss should be a finite positive scalar."""
        # (B, T, C) log probs
        log_probs = torch.randn(4, 60, 10).log_softmax(dim=-1)
        targets = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 3, 5]]
        loss = ctc_loss(log_probs, targets)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_loss_decreases_with_correct_alignment(self):
        """Loss should be lower when model output matches targets."""
        B, T, C = 2, 20, 10
        # Random baseline
        log_probs_rand = torch.randn(B, T, C).log_softmax(dim=-1)
        # "Correct" output: target phonemes have high probability
        targets = [[1, 2, 3], [4, 5, 6]]
        log_probs_good = torch.full((B, T, C), -10.0)
        # Make target phonemes likely at spread-out positions
        for b in range(B):
            for i, p in enumerate(targets[b]):
                pos = (i + 1) * T // 4
                log_probs_good[b, pos, p] = 10.0
        log_probs_good = log_probs_good.log_softmax(dim=-1)

        loss_rand = ctc_loss(log_probs_rand, targets)
        loss_good = ctc_loss(log_probs_good, targets)
        assert loss_good < loss_rand

    def test_gradient_flows(self):
        log_probs = torch.randn(2, 20, 10, requires_grad=True).log_softmax(dim=-1)
        # Need to retain grad through log_softmax
        log_probs.retain_grad()
        targets = [[1, 2, 3], [4, 5, 6]]
        loss = ctc_loss(log_probs, targets)
        loss.backward()
        assert log_probs.grad is not None


class TestGreedyDecode:
    def test_basic_decode(self):
        """Should collapse repeats and remove blanks."""
        # (B, T, C) with blank=0
        log_probs = torch.full((1, 10, 10), -10.0).log_softmax(dim=-1)
        # Set up: blank blank 1 1 blank 2 2 blank 3 blank
        for t, c in [(0, 0), (1, 0), (2, 1), (3, 1), (4, 0),
                      (5, 2), (6, 2), (7, 0), (8, 3), (9, 0)]:
            log_probs[0, t, c] = 10.0
        log_probs = log_probs.log_softmax(dim=-1)

        decoded = greedy_decode(log_probs)
        assert decoded == [[1, 2, 3]]

    def test_batch_decode(self):
        """Multiple sequences in batch."""
        log_probs = torch.full((2, 5, 10), -10.0)
        # Batch 0: 1 1 blank 2 3
        log_probs[0, 0, 1] = 10; log_probs[0, 1, 1] = 10
        log_probs[0, 2, 0] = 10; log_probs[0, 3, 2] = 10
        log_probs[0, 4, 3] = 10
        # Batch 1: blank 4 blank 5 blank
        log_probs[1, 0, 0] = 10; log_probs[1, 1, 4] = 10
        log_probs[1, 2, 0] = 10; log_probs[1, 3, 5] = 10
        log_probs[1, 4, 0] = 10
        log_probs = log_probs.log_softmax(dim=-1)

        decoded = greedy_decode(log_probs)
        assert decoded == [[1, 2, 3], [4, 5]]

    def test_all_blank(self):
        log_probs = torch.full((1, 5, 10), -10.0)
        log_probs[0, :, 0] = 10.0
        log_probs = log_probs.log_softmax(dim=-1)
        decoded = greedy_decode(log_probs)
        assert decoded == [[]]


class TestEncodeLabelsCTC:
    def test_encode(self):
        labels = [[1, 2, 3], [4, 5, 6]]
        targets, target_lengths = encode_labels_for_ctc(labels)
        assert targets.shape == (6,)  # concatenated
        assert target_lengths.tolist() == [3, 3]


class TestComputePER:
    def test_perfect(self):
        pred = [[1, 2, 3]]
        target = [[1, 2, 3]]
        assert compute_per(pred, target) == 0.0

    def test_one_substitution(self):
        pred = [[1, 2, 4]]
        target = [[1, 2, 3]]
        # Edit distance 1, target length 3 → PER = 1/3
        assert abs(compute_per(pred, target) - 1 / 3) < 1e-6

    def test_empty_pred(self):
        pred = [[]]
        target = [[1, 2, 3]]
        # Edit distance 3, target length 3 → PER = 1.0
        assert compute_per(pred, target) == 1.0

    def test_batch_average(self):
        pred = [[1, 2, 3], [1, 2, 4]]
        target = [[1, 2, 3], [1, 2, 3]]
        # PER = (0 + 1/3) / 2 = 1/6
        assert abs(compute_per(pred, target) - 1 / 6) < 1e-6


class TestBlankRatio:
    def test_all_blank(self):
        log_probs = torch.full((1, 10, 10), -10.0)
        log_probs[0, :, 0] = 10.0
        log_probs = log_probs.log_softmax(dim=-1)
        assert blank_ratio(log_probs) == pytest.approx(1.0, abs=0.01)

    def test_no_blank(self):
        log_probs = torch.full((1, 10, 10), -10.0)
        log_probs[0, :, 1] = 10.0  # always phoneme 1
        log_probs = log_probs.log_softmax(dim=-1)
        assert blank_ratio(log_probs) == pytest.approx(0.0, abs=0.01)
