"""The clip starts are SECONDS. This is the whole test file's reason to exist.

Passing frame indices instead cost a GPU job (2754310): ``_window_bands`` multiplies by FPS
internally, so frames-as-seconds overshoots by 32x and surfaces as "start=30763.0000s out of
cache bounds" -- an error that reads like a corrupt cache, not a caller unit bug. The units
are not visible at the call site, so they are pinned here.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.neuroprobe.v3_mae_recon import HZ, clip_starts_seconds


def test_starts_are_seconds_not_frames() -> None:
    """A 2-hour session at 32 Hz must yield starts of hours, not of hundreds of thousands."""
    n_frames = 215408                       # the real S4 t0 length, from the failed job's log
    starts = clip_starts_seconds(n_frames, clip_frames=64, n_clips=8)
    assert starts[0] == 0.0
    assert starts[-1] == pytest.approx((n_frames - 64) / HZ)
    assert starts[-1] < n_frames            # the bug: frame indices would fail this


def test_the_last_clip_ends_exactly_at_the_session_end_never_past_it() -> None:
    """``_window_bands`` rejects ``end > n_native``. If the final start rounded up by a single
    frame the whole dump would fail on its last window, after paying for the forward."""
    for n_frames in (215408, 100_000, 1025, 129):
        starts = clip_starts_seconds(n_frames, clip_frames=64, n_clips=8)
        end = np.rint(starts * HZ).astype(np.int64) + 64
        assert end.max() <= n_frames, (n_frames, end.max())


def test_starts_land_on_exact_integer_frames() -> None:
    """The seconds are divided down from frames, so the round-trip is exact rather than
    merely close -- no clip is silently offset by a frame from where it was placed."""
    starts = clip_starts_seconds(215408, clip_frames=64, n_clips=8)
    round_trip = starts * HZ
    np.testing.assert_allclose(round_trip, np.rint(round_trip), atol=0)


def test_the_clips_are_spread_across_the_session_not_bunched_at_the_front() -> None:
    """Consecutive windows overlap; a strip built from the first N in a row would look more
    self-consistent than the model is."""
    n_frames, clip_frames = 215408, 64
    starts = clip_starts_seconds(n_frames, clip_frames, n_clips=8)
    gaps = np.diff(starts)
    assert (gaps > clip_frames / HZ).all(), "clips overlap"
    assert np.allclose(gaps, gaps[0], rtol=1e-3), "uneven spread"


def test_a_session_shorter_than_one_clip_raises_instead_of_returning_junk() -> None:
    with pytest.raises(ValueError, match="shorter than"):
        clip_starts_seconds(32, clip_frames=64, n_clips=8)
