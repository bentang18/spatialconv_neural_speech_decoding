"""P3-03 tests: teacher-side triangular pool 50 → 8 Hz.

Pools the frozen Whisper-L8 hidden state from ``(250, 1280)`` (50 Hz
× 5 s clip) to ``(40, 1280)`` (8 Hz × 5 s clip). Triangular FWHM =
250 ms = 12.5 Whisper frames per bucket; sum-to-1 normalized per
bucket. Zero-padded at the start / end.

P3-02 ships alongside: the teacher rate constant is locked at 8 Hz
per the B05 + B06 lock (2026-05-25 PM).
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.extractors.whisper_teacher_pool import (
    P3_TEACHER_RATE_HZ,
    P3_WHISPER_NATIVE_RATE_HZ,
    triangular_pool_50_to_8_hz,
    triangular_pool_weight_matrix,
)


def test_p3_02_teacher_rate_pinned_at_8_hz() -> None:
    """B05 + B06 lock 2026-05-25 PM: P3 teacher rate = 8 Hz."""
    assert P3_TEACHER_RATE_HZ == 8.0


def test_p3_02_whisper_native_rate_pinned_at_50_hz() -> None:
    """Whisper-large native frame rate = 50 Hz (10 ms hop)."""
    assert P3_WHISPER_NATIVE_RATE_HZ == 50.0


# ---------- weight-matrix contract -----------------------------------


def test_weight_matrix_shape_50_to_8_hz_for_5s_clip() -> None:
    """5 s × 50 Hz = 250 input frames; 5 s × 8 Hz = 40 output frames."""
    W = triangular_pool_weight_matrix(n_in=250, n_out=40)
    assert W.shape == (40, 250)


def test_weight_matrix_rows_sum_to_one() -> None:
    """Sum-to-1 per output bucket — required for instance-norm
    invariance downstream."""
    W = triangular_pool_weight_matrix(n_in=250, n_out=40)
    assert torch.allclose(W.sum(dim=-1), torch.ones(40), atol=1e-6)


def test_weight_matrix_is_finite_and_non_negative() -> None:
    """Triangular weights are non-negative; no NaN / -inf from edge
    padding."""
    W = triangular_pool_weight_matrix(n_in=250, n_out=40)
    assert torch.isfinite(W).all()
    assert (W >= 0).all()


def test_weight_matrix_centers_track_uniform_8hz_grid() -> None:
    """Each output bucket's weight peak (argmax along input axis) sits
    at the bucket's centre on the input grid — i.e. ``round(i * 50/8)``
    for output index ``i``. Pin the centring contract so a future
    refactor that off-by-ones the bucket centres is caught."""
    W = triangular_pool_weight_matrix(n_in=250, n_out=40)
    expected_centres = [round(i * 50.0 / 8.0) for i in range(40)]
    got_centres = W.argmax(dim=-1).tolist()
    # Centre may differ by ≤ 1 because the FWHM rounds; allow that.
    for i, (got, exp) in enumerate(zip(got_centres, expected_centres)):
        assert abs(got - exp) <= 1, (
            f"bucket {i}: centre {got} drifts > 1 from expected {exp}"
        )


def test_weight_matrix_fwhm_around_12p5_input_frames() -> None:
    """250 ms FWHM @ 50 Hz = 12.5 Whisper frames. Each row's support
    (number of positions with weight > 1% of the row max) should land
    around that band — allow [10, 20] frames as the structural sanity
    range (exact FWHM depends on the triangular's discrete profile)."""
    W = triangular_pool_weight_matrix(n_in=250, n_out=40)
    # Avoid the two edges where zero-pad truncates the triangle.
    centre_rows = W[5:35]
    row_max = centre_rows.max(dim=-1).values
    above_threshold = (centre_rows > 0.01 * row_max.unsqueeze(-1)).sum(dim=-1)
    fwhm_in_frames = above_threshold.float().mean().item()
    assert 10 <= fwhm_in_frames <= 20, (
        f"expected FWHM ~12.5 frames; got mean support {fwhm_in_frames:.1f}"
    )


def test_weight_matrix_zero_pads_edges() -> None:
    """The first and last output buckets have zero contribution from
    out-of-range input frames (no wrap-around, no negative indexing)."""
    W = triangular_pool_weight_matrix(n_in=250, n_out=40)
    # Bucket 0 must not borrow weight from input frame ≥ 50 (10 buckets out).
    assert (W[0, 50:] == 0).all()
    # Bucket 39 must not borrow weight from input frame < 200.
    assert (W[39, :200] == 0).all()


# ---------- pool function ---------------------------------------------


def test_triangular_pool_function_shape_and_finite() -> None:
    """``(B, 250, 1280) → (B, 40, 1280)``."""
    feat = torch.randn(2, 250, 1280)
    out = triangular_pool_50_to_8_hz(feat)
    assert out.shape == (2, 40, 1280)
    assert torch.isfinite(out).all()


def test_triangular_pool_constant_feature_passes_through_unchanged() -> None:
    """If the input is constant along the time axis, every output row
    is the same constant (sum-to-1 weights × constant = constant)."""
    constant_row = torch.tensor([1.0, 2.0, 3.0, 4.0])
    feat = constant_row.unsqueeze(0).unsqueeze(0).repeat(1, 250, 1)  # (1, 250, 4)
    out = triangular_pool_50_to_8_hz(feat)
    assert out.shape == (1, 40, 4)
    expected = constant_row.unsqueeze(0).unsqueeze(0).repeat(1, 40, 1)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_triangular_pool_rejects_wrong_input_length() -> None:
    """A non-250 input length implies a mis-sized clip — reject loudly."""
    feat = torch.randn(2, 200, 1280)
    with pytest.raises(ValueError, match="expected 250"):
        triangular_pool_50_to_8_hz(feat)
