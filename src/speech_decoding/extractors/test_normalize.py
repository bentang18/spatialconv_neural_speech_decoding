"""Tests for the Nv14 robust-z normalizer (T1.6)."""

from __future__ import annotations

import torch

from speech_decoding.extractors.normalize import (
    SCALE_TO_SIGMA,
    Nv14RobustZTransform,
    robust_z,
)


def test_robust_z_matches_known_median_mad() -> None:
    """Hand-computed: for x = [1, 2, 3, 4, 100] the median is 3 and MAD is 1
    (absolute deviations [2, 1, 0, 1, 97] → median 1). Robust σ = 1.4826;
    z for value 3 should be 0, z for value 100 ≈ (100−3)/1.4826 ≈ 65.43."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 100.0]])
    z = robust_z(x)
    assert abs(z[0, 2].item() - 0.0) < 1e-5, f"median position z must be 0, got {z[0, 2]}"
    expected_outlier = (100.0 - 3.0) / SCALE_TO_SIGMA
    assert abs(z[0, 4].item() - expected_outlier) < 1e-3, (
        f"outlier z mismatch: got {z[0, 4]}, expected {expected_outlier}"
    )


def test_robust_z_is_per_axis_independent() -> None:
    """Two channels with very different distributions must be z-scored
    independently. Channel 0: tight around 10. Channel 1: spread around -50."""
    x = torch.tensor([
        [10.0, 9.0, 11.0, 10.0, 10.0],
        [-50.0, -100.0, 0.0, 50.0, -50.0],
    ])
    z = robust_z(x)
    # Channel 0 median is 10, MAD is 0 → σ < floor → all zero by the floor rule.
    assert torch.allclose(z[0], torch.zeros(5), atol=1e-5), (
        "near-constant channel must be zeroed by the sigma floor"
    )
    # Channel 1 spans a much wider range; its z-scored output must vary,
    # not collapse to a single value (which would mean the floor zeroed it).
    assert z[1].std() > 0.1, "channel-1 z must vary across its time axis"
    # And it must NOT be all zeros (independent of channel 0's behavior).
    assert z[1].abs().max() > 0.1, "channel-1 z must register some signal"


def test_robust_z_respects_valid_bin_mask() -> None:
    """A bin marked invalid in the per-corpus mask must return zero, regardless
    of input values. Mask shape must be broadcastable to x.shape — for a
    per-freq mask on (C, F, T), reshape to (1, F, 1)."""
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    x[:, 2:] = 999.0  # invalid bins have garbage
    mask = torch.tensor([True, True, False, False]).reshape(1, 4, 1)
    z = robust_z(x, valid_bin_mask=mask)
    assert torch.allclose(z[:, 2:], torch.zeros(1, 2, 8), atol=1e-5), (
        "invalid bins must return zero"
    )
    # Valid bins should still produce sensible (finite) z-scores.
    assert torch.isfinite(z[:, :2]).all()


def test_robust_z_handles_constant_input_without_inf() -> None:
    """Pure-constant input has σ = 0; the floor must kick in (z = 0) rather
    than producing ±inf or NaN."""
    x = torch.full((3, 7, 11), 4.2)
    z = robust_z(x)
    assert torch.isfinite(z).all()
    assert torch.allclose(z, torch.zeros_like(z), atol=1e-6)


def test_robust_z_on_multistft_shape_preserves_dims() -> None:
    """Shape preservation on the canonical Multi-STFT output: (C, F=30, T)."""
    torch.manual_seed(0)
    x = torch.randn(8, 30, 17)
    z = robust_z(x)
    assert z.shape == (8, 30, 17)
    assert torch.isfinite(z).all()


def test_nv14_robust_z_transform_callable_matches_function() -> None:
    """The Pydantic transform delegates to the same function — same output."""
    torch.manual_seed(0)
    x = torch.randn(2, 30, 17)
    t = Nv14RobustZTransform()
    a = robust_z(x)
    b = t(x)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
