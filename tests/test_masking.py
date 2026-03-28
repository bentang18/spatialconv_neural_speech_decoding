"""Tests for temporal span masking."""
import pytest
import torch
import numpy as np

from speech_decoding.pretraining.masking import (
    generate_span_mask,
    SpanMasker,
)


class TestGenerateSpanMask:
    def test_output_shape(self):
        mask = generate_span_mask(T=30, mask_ratio=(0.4, 0.6), n_spans=(3, 6), rng=np.random.RandomState(42))
        assert mask.shape == (30,)
        assert mask.dtype == bool

    def test_mask_ratio_in_range(self):
        for seed in range(50):
            mask = generate_span_mask(T=30, mask_ratio=(0.4, 0.6), n_spans=(3, 6),
                                       rng=np.random.RandomState(seed))
            frac = mask.sum() / len(mask)
            assert 0.25 <= frac <= 0.75, f"Seed {seed}: mask fraction {frac:.2f}"

    def test_spans_are_contiguous(self):
        mask = generate_span_mask(T=30, mask_ratio=(0.4, 0.6), n_spans=(3, 6),
                                   rng=np.random.RandomState(42))
        transitions = np.diff(mask.astype(int))
        n_spans = (transitions == 1).sum()
        assert 1 <= n_spans <= 8

    def test_deterministic_with_seed(self):
        m1 = generate_span_mask(T=30, mask_ratio=(0.4, 0.6), n_spans=(3, 6),
                                 rng=np.random.RandomState(42))
        m2 = generate_span_mask(T=30, mask_ratio=(0.4, 0.6), n_spans=(3, 6),
                                 rng=np.random.RandomState(42))
        np.testing.assert_array_equal(m1, m2)


class TestSpanMasker:
    def test_mask_token_learnable(self):
        masker = SpanMasker(d=64)
        assert masker.mask_token.requires_grad
        assert masker.mask_token.shape == (64,)

    def test_apply_mask_shape(self):
        masker = SpanMasker(d=64)
        x = torch.randn(4, 30, 64)
        mask = torch.zeros(30, dtype=torch.bool)
        mask[5:10] = True
        x_masked, applied_mask = masker(x, mask)
        assert x_masked.shape == x.shape
        assert applied_mask.shape == (30,)

    def test_masked_frames_are_mask_token(self):
        masker = SpanMasker(d=64)
        x = torch.randn(4, 30, 64)
        mask = torch.zeros(30, dtype=torch.bool)
        mask[5:10] = True
        x_masked, _ = masker(x, mask)
        for t in range(5, 10):
            assert torch.allclose(x_masked[0, t], masker.mask_token.data, atol=1e-6)

    def test_unmasked_frames_unchanged(self):
        masker = SpanMasker(d=64)
        x = torch.randn(4, 30, 64)
        mask = torch.zeros(30, dtype=torch.bool)
        mask[5:10] = True
        x_masked, _ = masker(x, mask)
        assert torch.allclose(x_masked[:, :5], x[:, :5])
        assert torch.allclose(x_masked[:, 10:], x[:, 10:])

    def test_same_mask_all_positions_in_batch(self):
        masker = SpanMasker(d=64)
        x = torch.randn(4, 30, 64)
        mask = torch.zeros(30, dtype=torch.bool)
        mask[5:10] = True
        x_masked, _ = masker(x, mask)
        for b in range(4):
            for t in range(5, 10):
                assert torch.allclose(x_masked[b, t], masker.mask_token.data, atol=1e-6)
