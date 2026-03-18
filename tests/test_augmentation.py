"""Tests for data augmentation pipeline."""
import numpy as np
import pytest
import torch

from speech_decoding.data.augmentation import (
    time_shift,
    amplitude_scale,
    channel_dropout,
    gaussian_noise,
    augment_batch,
)


@pytest.fixture
def sample_batch():
    """(B, H, W, T) grid-shaped HGA batch."""
    torch.manual_seed(42)
    return torch.randn(4, 8, 16, 300)


class TestTimeShift:
    def test_shape_preserved(self, sample_batch):
        out = time_shift(sample_batch, max_frames=20)
        assert out.shape == sample_batch.shape

    def test_zero_shift_is_identity(self, sample_batch):
        out = time_shift(sample_batch, max_frames=0)
        assert torch.allclose(out, sample_batch)

    def test_shifts_are_per_trial(self, sample_batch):
        """Different trials should get different shifts."""
        torch.manual_seed(0)
        out = time_shift(sample_batch, max_frames=20)
        # At least some trials should differ from original
        diffs = [(out[i] - sample_batch[i]).abs().sum().item() for i in range(4)]
        assert any(d > 0 for d in diffs)


class TestAmplitudeScale:
    def test_shape_preserved(self, sample_batch):
        out = amplitude_scale(sample_batch, std=0.15)
        assert out.shape == sample_batch.shape

    def test_zero_std_is_identity(self, sample_batch):
        out = amplitude_scale(sample_batch, std=0.0)
        assert torch.allclose(out, sample_batch)

    def test_per_trial_per_channel(self, sample_batch):
        """Scale factors should vary per trial and be constant within a trial."""
        torch.manual_seed(42)
        out = amplitude_scale(sample_batch, std=0.15)
        # Each (trial, row, col) electrode should be uniformly scaled across time
        for b in range(2):
            for r in range(2):
                for c in range(2):
                    orig = sample_batch[b, r, c, :]
                    aug = out[b, r, c, :]
                    # Skip near-zero channels
                    if orig.abs().max() > 0.1:
                        ratios = aug / (orig + 1e-10)
                        # Ratios should be roughly constant (same scale factor)
                        assert ratios.std() < 0.1 * ratios.abs().mean() + 1e-5


class TestChannelDropout:
    def test_shape_preserved(self, sample_batch):
        out = channel_dropout(sample_batch, max_p=0.2)
        assert out.shape == sample_batch.shape

    def test_zero_p_is_identity(self, sample_batch):
        out = channel_dropout(sample_batch, max_p=0.0)
        assert torch.allclose(out, sample_batch)

    def test_some_channels_zeroed(self, sample_batch):
        torch.manual_seed(42)
        out = channel_dropout(sample_batch, max_p=0.5)
        # At least some electrodes should be zeroed
        n_zeroed = (out.abs().sum(dim=-1) == 0).sum().item()
        assert n_zeroed > 0


class TestGaussianNoise:
    def test_shape_preserved(self, sample_batch):
        out = gaussian_noise(sample_batch, frac=0.02)
        assert out.shape == sample_batch.shape

    def test_zero_frac_is_identity(self, sample_batch):
        out = gaussian_noise(sample_batch, frac=0.0)
        assert torch.allclose(out, sample_batch)

    def test_noise_magnitude(self, sample_batch):
        torch.manual_seed(42)
        out = gaussian_noise(sample_batch, frac=0.02)
        diff = (out - sample_batch).abs()
        # Noise should be small relative to signal
        assert diff.mean() < 0.1 * sample_batch.abs().mean()


class TestAugmentBatch:
    def test_returns_correct_shape(self, sample_batch):
        out = augment_batch(sample_batch)
        assert out.shape == sample_batch.shape

    def test_eval_mode_is_identity(self, sample_batch):
        out = augment_batch(sample_batch, training=False)
        assert torch.allclose(out, sample_batch)
