"""P3-03 tests: teacher-side triangular pool 50 → 8 Hz.

Pools the frozen Whisper-L8 hidden state from ``(250, 1280)`` (50 Hz
× 5 s clip) to ``(40, 1280)`` (8 Hz × 5 s clip). Triangle base = 250 ms
= 12.5 frames (true half-max FWHM = 125 ms); per-bucket support ~12-13
nonzero frames; sum-to-1 normalized per bucket. Zero-padded at the
start / end.

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


def test_weight_matrix_rows_sum_to_one_at_fp32_epsilon() -> None:
    """WS-F F3 regression pin: the row-normalization lands every bucket sum on
    1.0 to within fp32 epsilon (2⁻²³ ≈ 1.19e-7). A looser-than-this deviation
    means the row-sum clamp / normalization drifted — fail CI. (Tighter than
    the ``1e-6`` structural check above; this one pins the exact numeric
    floor.)"""
    W = triangular_pool_weight_matrix(n_in=250, n_out=40)
    assert (W.sum(dim=-1) - 1.0).abs().max().item() < 1.2e-7


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


def test_weight_matrix_interior_centroid_lands_on_8hz_grid() -> None:
    """WS-F F3 regression pin: 50→8 Hz grid alignment, exact. For interior
    buckets (away from the zero-padded edges) the triangle is symmetric, so its
    weighted centroid ``Σ wᵢ·i`` equals the uniform-grid centre ``i·(50/8)`` to
    <0.05 frames. Pins the bucket placement tighter than the ±1 argmax check —
    an off-by-one in the centre formula (e.g. integer ``50//8`` instead of the
    6.25 stride) would shift the centroid by ≈0.5+ frames and fail here."""
    W = triangular_pool_weight_matrix(n_in=250, n_out=40)
    in_positions = torch.arange(250, dtype=torch.float32)
    centroid = (W * in_positions).sum(dim=-1)  # rows sum to 1 ⇒ weighted mean
    expected = torch.arange(40, dtype=torch.float32) * (250.0 / 40.0)
    interior = slice(3, 37)
    assert (centroid[interior] - expected[interior]).abs().max().item() < 0.05


def test_weight_matrix_base_support_around_12p5_input_frames() -> None:
    """Triangle base = 250 ms @ 50 Hz = 12.5 frames (true half-max FWHM
    = 125 ms); per-bucket support ~12-13 nonzero frames. Each row's
    support (number of positions with weight > 1% of the row max) should
    land around that band — allow [10, 20] frames as the structural
    sanity range (exact support depends on the triangle's discrete
    profile)."""
    W = triangular_pool_weight_matrix(n_in=250, n_out=40)
    # Avoid the two edges where zero-pad truncates the triangle.
    centre_rows = W[5:35]
    row_max = centre_rows.max(dim=-1).values
    above_threshold = (centre_rows > 0.01 * row_max.unsqueeze(-1)).sum(dim=-1)
    base_support_frames = above_threshold.float().mean().item()
    assert 10 <= base_support_frames <= 20, (
        f"expected base support ~12-13 frames; got mean support "
        f"{base_support_frames:.1f}"
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


def test_triangular_pool_preserves_dtype_and_accumulates_in_fp32() -> None:
    """The cache is fp16; the pool upcasts to fp32 for the einsum (a fp16
    accumulation over ~12-13 frames per bucket loses precision) then casts back
    to the caller's dtype. Output dtype matches input, and the fp16 result is
    within fp16 quantization of the fp32 reference."""
    torch.manual_seed(0)
    feat32 = torch.randn(1, 250, 1280)
    ref = triangular_pool_50_to_8_hz(feat32)
    assert ref.dtype == torch.float32
    out16 = triangular_pool_50_to_8_hz(feat32.to(torch.float16))
    assert out16.dtype == torch.float16  # dtype contract preserved
    assert (out16.float() - ref).abs().max().item() < 1e-2


def test_triangular_pool_fp32_accumulation_holds_under_bf16_autocast() -> None:
    """WS-F bug-hunt pin: ``einsum`` is autocast-eligible, so under an outer
    bf16 autocast (Lightning's bf16-mixed in the P3 distill step) it would
    silently downcast the explicit fp32 cast and defeat the documented fp32
    accumulation. The pool disables autocast internally, so the fp32-input
    result is identical inside vs outside an autocast region. Before the fix
    this deviated ~0.4%; after, it is bit-exact."""
    torch.manual_seed(0)
    feat = torch.randn(1, 250, 1280)  # fp32 input
    ref = triangular_pool_50_to_8_hz(feat)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        got = triangular_pool_50_to_8_hz(feat)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, ref, atol=0.0, rtol=0.0)


def test_triangular_pool_rejects_wrong_input_length() -> None:
    """A non-250 input length implies a mis-sized clip — reject loudly."""
    feat = torch.randn(2, 200, 1280)
    with pytest.raises(ValueError, match="expected 250"):
        triangular_pool_50_to_8_hz(feat)
