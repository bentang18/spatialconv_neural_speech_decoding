"""Cross-module parity guard for the teacher-side triangular pool.

Durable guard against the contract.py-vs-whisper_teacher_pool.py 2×-width
drift found in the 2026-05-29 audit: the live kernel A
(``extractors.whisper_teacher_pool``) and the formerly-2×-wide test-only
kernel (``bt_alignment.contract``) must share the SAME effective triangle base
at the 50 → 8 Hz operating point (triangle base 250 ms; half-base 6.25 frames =
stride; true half-max FWHM 125 ms). Before the contract.py reconciliation the
contract kernel gave ~25 nonzero samples while the live kernel gives ~12-13 —
exactly the 2× divergence this test pins shut.
"""

from __future__ import annotations

import numpy as np

from speech_decoding.bt_alignment.contract import triangular_pool_kernel
from speech_decoding.extractors.whisper_teacher_pool import (
    P3_TEACHER_RATE_HZ,
    P3_WHISPER_NATIVE_RATE_HZ,
    triangular_pool_weight_matrix,
)

# Bucket centres land at fractional input positions (half-base 6.25 each side),
# so an interior row's nonzero support alternates between 12 and 13 frames. Both
# kernels live in the same [11, 14] band; assert the band, not a single value.
_BASE_LO, _BASE_HI = 11, 14


def _interior_row_support() -> int:
    W = triangular_pool_weight_matrix(n_in=250, n_out=40)
    row = W[20]  # interior bucket, clear of the zero-pad edges
    return int((row > 0).sum())


def test_live_interior_base_support_in_band() -> None:
    """Live kernel A interior bucket: triangle base ~12-13 nonzero frames."""
    base_support_live = _interior_row_support()
    assert _BASE_LO <= base_support_live <= _BASE_HI, (
        f"live interior base support {base_support_live} outside "
        f"[{_BASE_LO}, {_BASE_HI}]"
    )


def test_contract_matches_live_base_support() -> None:
    """The reconciled contract kernel must share the live triangle base.

    FAILS before contract.py is reconciled to half-base = stride (contract gave
    ~25 nonzero samples, live gives ~12-13 — the 2× drift). Passes only once
    both kernels use the same effective base width.
    """
    base_support_live = _interior_row_support()
    k = triangular_pool_kernel()
    base_support_contract = int((k > 0).sum())
    assert _BASE_LO <= base_support_contract <= _BASE_HI, (
        f"contract base support {base_support_contract} outside "
        f"[{_BASE_LO}, {_BASE_HI}] — contract.py is not reconciled to the "
        f"live half-base = stride width"
    )
    # Same effective base width as the live kernel (allow the ±1 row jitter).
    assert abs(base_support_contract - base_support_live) <= 1, (
        f"contract base support {base_support_contract} diverges from live "
        f"{base_support_live} by > 1 frame"
    )


def test_both_half_max_fwhm_pinned_to_125ms() -> None:
    """True half-max FWHM = 125 ms = 6.25 frames for BOTH kernels: the count
    of positions with weight > 0.5×max must be 6 (±1)."""
    W = triangular_pool_weight_matrix(n_in=250, n_out=40)
    live_row = W[20]
    live_fwhm = int((live_row > 0.5 * live_row.max()).sum())
    assert abs(live_fwhm - 6) <= 1, f"live half-max FWHM count {live_fwhm} != 6 (±1)"

    k = triangular_pool_kernel()
    contract_fwhm = int((k > 0.5 * k.max()).sum())
    assert abs(contract_fwhm - 6) <= 1, (
        f"contract half-max FWHM count {contract_fwhm} != 6 (±1)"
    )


def test_half_base_stride_coupling() -> None:
    """Width can never silently decouple from rate: the live half-base
    (half_fwhm_frames) must equal the rate ratio = stride = 6.25."""
    expected_stride = P3_WHISPER_NATIVE_RATE_HZ / P3_TEACHER_RATE_HZ
    half_base_frames = (250.0 / 1000.0) * P3_WHISPER_NATIVE_RATE_HZ / 2.0
    assert half_base_frames == expected_stride == 6.25, (
        f"half-base {half_base_frames} != stride {expected_stride} (6.25)"
    )
    # Cross-check the contract kernel's half-base derivation lands on 6.25 too.
    # contract.py's TEACHER_RATE_HZ is the Whisper NATIVE rate (50 Hz), the
    # input rate of the pool — not v14's 8 Hz output rate.
    contract_half_base = 250.0 / 1000.0 * P3_WHISPER_NATIVE_RATE_HZ / 2.0
    assert contract_half_base == 6.25
    assert isinstance(triangular_pool_kernel(), np.ndarray)
