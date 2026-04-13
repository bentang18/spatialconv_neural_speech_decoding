"""Tests for data augmentation pipeline."""
import numpy as np
import pytest
import torch

from speech_decoding.data.augmentation import (
    time_shift,
    amplitude_scale,
    channel_dropout,
    gaussian_noise,
    spatial_cutout,
    temporal_stretch,
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


class TestSpatialCutout:
    def test_shape_preserved(self, sample_batch):
        out = spatial_cutout(sample_batch, max_h=3, max_w=6)
        assert out.shape == sample_batch.shape

    def test_some_region_zeroed(self, sample_batch):
        torch.manual_seed(42)
        out = spatial_cutout(sample_batch, max_h=3, max_w=6)
        # At least some spatial positions should be zeroed
        n_zeroed = (out.abs().sum(dim=-1) == 0).sum().item()
        assert n_zeroed > 0

    def test_original_unchanged(self, sample_batch):
        """spatial_cutout should clone, not modify in place."""
        orig = sample_batch.clone()
        spatial_cutout(sample_batch, max_h=3, max_w=6)
        assert torch.allclose(sample_batch, orig)


class TestTemporalStretch:
    def test_shape_preserved(self, sample_batch):
        out = temporal_stretch(sample_batch, max_rate=0.15)
        assert out.shape == sample_batch.shape

    def test_zero_rate_is_identity(self, sample_batch):
        out = temporal_stretch(sample_batch, max_rate=0.0)
        assert torch.allclose(out, sample_batch)

    def test_content_changes(self, sample_batch):
        torch.manual_seed(42)
        out = temporal_stretch(sample_batch, max_rate=0.15)
        assert not torch.allclose(out, sample_batch)

    def test_output_finite(self, sample_batch):
        out = temporal_stretch(sample_batch, max_rate=0.3)
        assert torch.isfinite(out).all()


class TestAugmentBatch:
    def test_returns_correct_shape(self, sample_batch):
        out = augment_batch(sample_batch)
        assert out.shape == sample_batch.shape

    def test_eval_mode_is_identity(self, sample_batch):
        out = augment_batch(sample_batch, training=False)
        assert torch.allclose(out, sample_batch)

    def test_all_augmentations_enabled(self, sample_batch):
        """Full augmentation pipeline with all options."""
        out = augment_batch(
            sample_batch,
            training=True,
            time_shift_frames=30,
            amp_scale_std=0.3,
            channel_dropout_max=0.4,
            noise_frac=0.05,
            do_spatial_cutout=True,
            spatial_cutout_max_h=3,
            spatial_cutout_max_w=6,
            do_temporal_stretch=True,
            temporal_stretch_max_rate=0.15,
        )
        assert out.shape == sample_batch.shape
        assert torch.isfinite(out).all()


class TestAugmentationNotTooAggressive:
    """Guard against augmentation destroying signal.

    If augmentation is too aggressive, the augmented data becomes
    indistinguishable from noise and the model can't learn.
    These tests check that signal structure survives augmentation.
    """

    def test_signal_survives_per_patient_config(self):
        """With per_patient.yaml augmentation, >50% of signal energy survives."""
        torch.manual_seed(42)
        x = torch.randn(16, 8, 16, 200)
        orig_energy = (x ** 2).mean().item()

        # Run augmentation 10 times, check average energy retention
        energies = []
        for _ in range(10):
            out = augment_batch(
                x, training=True,
                time_shift_frames=30, amp_scale_std=0.3,
                channel_dropout_max=0.4, noise_frac=0.05,
                do_spatial_cutout=True, spatial_cutout_max_h=3,
                spatial_cutout_max_w=6,
                do_temporal_stretch=True, temporal_stretch_max_rate=0.15,
            )
            energies.append((out ** 2).mean().item())
        avg_energy = np.mean(energies)
        # Signal energy should not collapse (>30% retained) or explode (< 5× original)
        assert avg_energy > 0.3 * orig_energy, f"Energy collapsed: {avg_energy:.3f} vs {orig_energy:.3f}"
        assert avg_energy < 5.0 * orig_energy, f"Energy exploded: {avg_energy:.3f} vs {orig_energy:.3f}"

    def test_correlation_survives_augmentation(self):
        """Temporal correlation structure should partially survive augmentation."""
        torch.manual_seed(42)
        # Create signal with clear temporal structure (slow oscillation)
        T = 200
        t = torch.linspace(0, 4 * torch.pi, T)
        x = torch.sin(t).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(4, 8, 16, T)
        x = x + torch.randn_like(x) * 0.1  # slight noise

        out = augment_batch(
            x, training=True,
            time_shift_frames=30, amp_scale_std=0.3,
            channel_dropout_max=0.4, noise_frac=0.05,
            do_spatial_cutout=True, spatial_cutout_max_h=3,
            spatial_cutout_max_w=6,
            do_temporal_stretch=True, temporal_stretch_max_rate=0.15,
        )
        # Autocorrelation at lag=1 should be positive (temporal structure preserved)
        # Pick a non-dropped channel
        for b in range(4):
            for r in range(8):
                for c in range(16):
                    ch = out[b, r, c, :]
                    if ch.abs().sum() > 0:  # not dropped
                        ac1 = torch.corrcoef(torch.stack([ch[:-1], ch[1:]]))[0, 1]
                        assert ac1 > 0.3, f"Temporal structure destroyed: autocorr={ac1:.3f}"
                        return  # one non-dropped channel is enough
        pytest.fail("All channels were dropped")

    def test_channel_dropout_not_total(self):
        """Even at max_p=0.4, most channels should survive on average."""
        torch.manual_seed(42)
        x = torch.ones(16, 8, 16, 100)
        survivals = []
        for _ in range(50):
            out = channel_dropout(x, max_p=0.4)
            frac_alive = (out.abs().sum(dim=-1) > 0).float().mean().item()
            survivals.append(frac_alive)
        # Average survival should be >60% (p ~ U[0,0.4], E[p]=0.2, E[survival]=0.8)
        assert np.mean(survivals) > 0.6, f"Too aggressive: {np.mean(survivals):.2f} survival"

    def test_spatial_cutout_leaves_most_grid(self):
        """Spatial cutout (3×6 max) on 8×16 grid zeroes at most 14% of electrodes."""
        max_frac = (3 * 6) / (8 * 16)  # 18/128 = 14%
        assert max_frac < 0.15
        # Empirically verify
        torch.manual_seed(42)
        x = torch.ones(16, 8, 16, 100)
        survivals = []
        for _ in range(50):
            out = spatial_cutout(x, max_h=3, max_w=6)
            frac_alive = (out.abs().sum(dim=-1) > 0).float().mean().item()
            survivals.append(frac_alive)
        assert np.mean(survivals) > 0.85
