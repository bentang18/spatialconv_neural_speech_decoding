"""Tests for the Nv14 robust-z normalizer (T1.6)."""

from __future__ import annotations

import torch

from speech_decoding.extractors.normalize import (
    SCALE_TO_SIGMA,
    Nv14RobustZTransform,
    SessionRobustZNormalizer,
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


# ---------------------------------------------------------------------------
# WS-C / C3: session-level robust-z (fit-on-train / apply-frozen)
# ---------------------------------------------------------------------------


def test_session_robust_z_fits_per_electrode_freq_over_time() -> None:
    """``fit`` stores per-(electrode, freq) median + MAD-σ over the session time
    axis → shape (C, F, 1). ``transform`` applies the frozen stats and preserves
    the clip's (C, F, T) shape."""
    torch.manual_seed(0)
    session = torch.randn(4, 30, 200)  # (C, F, T_session)
    norm = SessionRobustZNormalizer().fit(session)
    assert norm.median.shape == (4, 30, 1)
    assert norm.sigma.shape == (4, 30, 1)
    clip = torch.randn(4, 30, 20)
    z = norm.transform(clip)
    assert z.shape == (4, 30, 20)
    assert torch.isfinite(z).all()


def test_session_robust_z_is_gain_invariant() -> None:
    """C3 gain-invariance (ρ=1.0): a pure ×k gain on one channel — present in
    BOTH the fit data and the applied clip — leaves the normalized output
    unchanged, because median and σ both scale by k and cancel."""
    torch.manual_seed(1)
    session = torch.randn(4, 30, 200)
    clip = torch.randn(4, 30, 20)

    norm = SessionRobustZNormalizer().fit(session)
    z = norm.transform(clip)

    k = 7.3
    session_g = session.clone()
    session_g[1] *= k
    clip_g = clip.clone()
    clip_g[1] *= k
    norm_g = SessionRobustZNormalizer().fit(session_g)
    z_g = norm_g.transform(clip_g)

    torch.testing.assert_close(z, z_g, atol=1e-4, rtol=1e-4)


def test_session_robust_z_uses_train_stats_not_clip_stats() -> None:
    """C3 'stats fit on train only': a clip held constant at
    ``train_median + train_σ`` normalizes to +1 everywhere under the FROZEN
    train stats. A per-clip normalizer would see a constant (σ→0, floored to 0),
    so this distinguishes train-fit from clip-fit."""
    torch.manual_seed(2)
    session = torch.randn(4, 30, 500)
    norm = SessionRobustZNormalizer().fit(session)
    # Constant-in-time clip = train_median + train_σ.
    offset_clip = (norm.median + norm.sigma).expand(4, 30, 5).clone()
    z = norm.transform(offset_clip)
    torch.testing.assert_close(z, torch.ones_like(z), atol=1e-4, rtol=1e-4)


def test_session_robust_z_zeros_constant_nonzero_bin_under_jitter() -> None:
    """σ-floor zeroing (mutant guard): a bin CONSTANT but NONZERO across the
    training session (median≠0, σ=0) must transform to exactly 0 even when a clip
    jitters it slightly. Without ``transform``'s ``where(sigma >= floor, z, 0)``
    branch, ``(7.001 − 7.0) / max(0, 1e-6)`` blows the jitter up to ~1e3. This is
    distinct from the all-zero pad-channel case (median is nonzero here), and from
    the per-call ``robust_z`` floor tests (this pins the FROZEN-stat transform)."""
    torch.manual_seed(5)
    session = torch.randn(2, 4, 300)
    session[:, 1, :] = 7.0  # bin (·,1,·): constant nonzero → median 7, MAD 0, σ 0
    norm = SessionRobustZNormalizer(sigma_floor=1e-6).fit(session)
    assert torch.allclose(norm.sigma[:, 1], torch.zeros_like(norm.sigma[:, 1]))
    assert torch.allclose(norm.median[:, 1], torch.full_like(norm.median[:, 1], 7.0))

    clip = torch.randn(2, 4, 10)
    clip[:, 1, :] = 7.0 + 1e-3 * torch.randn(2, 10)  # small jitter around the constant
    z = norm.transform(clip)
    torch.testing.assert_close(z[:, 1], torch.zeros_like(z[:, 1]), atol=0.0, rtol=0.0)
    assert torch.isfinite(z).all()


def test_session_robust_z_transform_before_fit_raises() -> None:
    import pytest

    norm = SessionRobustZNormalizer()
    with pytest.raises(RuntimeError, match="fit"):
        norm.transform(torch.randn(2, 30, 8))


def test_session_robust_z_honors_valid_bin_mask() -> None:
    """C3/C5: invalid freq bins (per the per-corpus mask) return z=0 on
    transform regardless of input value."""
    torch.manual_seed(3)
    session = torch.randn(2, 30, 200)
    mask = torch.ones(30, dtype=torch.bool)
    mask[22:] = False  # SWEC k22-29 invalid
    norm = SessionRobustZNormalizer().fit(
        session, valid_bin_mask=mask.reshape(1, 30, 1),
    )
    clip = torch.randn(2, 30, 20)
    clip[:, 22:] = 999.0
    z = norm.transform(clip)
    assert torch.allclose(z[:, 22:], torch.zeros_like(z[:, 22:]), atol=1e-5)
    assert torch.isfinite(z[:, :22]).all()
