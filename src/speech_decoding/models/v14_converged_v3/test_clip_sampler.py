"""v14_converged_v3 Phase D2a — guard-2 clip-t0 sampler (TDD).

Memo project-v3-pipeline-build-contract-2026-07-10: clips are random-CONTINUOUS
windows on the per-session continuous spec cache, and guard-2 (bad TIME spans) is a
STRICT any-overlap drop realised REJECT-FREE — instead of sample-and-retry, sample
t0 directly from the complement of the forbidden set.

A clip ``[t0, t0+L)`` overlaps a bad span ``[a, b)`` iff ``t0 < b`` and
``t0+L > a`` ⇔ ``t0 ∈ (a-L, b)``. So the forbidden t0-set is ``⋃[a-L, b)`` over
all spans, and the allowed set is its complement within ``[0, T-L]`` (the last
legal start). We sample t0 uniformly over total allowed LENGTH (each allowed
sub-interval weighted by its width) so the marginal over valid clips is uniform —
no bias toward clips adjacent to bad spans, no retry loop.

Everything is in the 32 Hz spec-frame clock (clip_len 3.0 s = 96 frames): the bad
spans arrive in hop-invariant SECONDS and are converted once (``round(s * fps)``).
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v3.clip_sampler import (
    allowed_start_intervals,
    sample_clip_start,
)

FPS = 32.0


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_no_bad_spans_allows_the_whole_range() -> None:
    # T=320 frames, L=96 → legal starts [0, 224]. Empty spans ⇒ one interval.
    ivals = allowed_start_intervals(n_frames=320, clip_frames=96, bad_spans_s=[], fps=FPS)
    assert ivals == [(0, 224)]


def test_forbidden_set_is_span_minus_L_to_end() -> None:
    # One bad span [3.0s, 4.0s) = [96,128) frames, L=96 → forbidden t0 in (96-96,128)=(0,128).
    # Allowed = [0,0] (just t0=0, since (0,128) excludes 0) ∪ [128, 224].
    ivals = allowed_start_intervals(
        n_frames=320, clip_frames=96, bad_spans_s=[(3.0, 4.0)], fps=FPS
    )
    # forbidden (0,128): t0=0 allowed (strict >), t0=1..127 forbidden, t0>=128 allowed
    assert ivals == [(0, 0), (128, 224)]


def test_span_at_start_removes_low_starts() -> None:
    # Bad [0,1s)=[0,32) frames, L=96 → forbidden (0-96,32)=(-96,32) → clip to [0,32).
    # Allowed t0 in [32, 224].
    ivals = allowed_start_intervals(
        n_frames=320, clip_frames=96, bad_spans_s=[(0.0, 1.0)], fps=FPS
    )
    assert ivals == [(32, 224)]


def test_overlapping_spans_are_merged() -> None:
    # Two spans whose forbidden intervals overlap must merge into one gap.
    # [3s,4s)=[96,128)→forbid(0,128); [3.5s,5s)=[112,160)→forbid(16,160). Union (0,160).
    ivals = allowed_start_intervals(
        n_frames=400, clip_frames=96, bad_spans_s=[(3.0, 4.0), (3.5, 5.0)], fps=FPS
    )
    # allowed: t0=0, then [160, 304]
    assert ivals == [(0, 0), (160, 304)]


def test_fully_blocked_session_returns_no_intervals() -> None:
    # A bad span covering everything leaves no legal start.
    ivals = allowed_start_intervals(
        n_frames=200, clip_frames=96, bad_spans_s=[(0.0, 10.0)], fps=FPS
    )
    assert ivals == []


def test_sample_returns_valid_start_avoiding_spans() -> None:
    # Draw many starts; every clip must avoid the bad span entirely.
    spans = [(3.0, 4.0)]  # [96,128)
    g = _gen(0)
    for _ in range(200):
        t0 = sample_clip_start(
            n_frames=320, clip_frames=96, bad_spans_s=spans, fps=FPS, generator=g
        )
        assert 0 <= t0 <= 224
        # clip [t0, t0+96) must not overlap [96,128)
        assert not (t0 < 128 and t0 + 96 > 96)


def test_sample_length_weights_intervals() -> None:
    # Two allowed gaps of very different width: a big gap should be drawn far more
    # often than a width-1 gap (length-proportional sampling).
    # Construct spans so allowed = {t0=0} ∪ [128,224] (width 1 vs 97).
    spans = [(3.0, 4.0)]
    g = _gen(1)
    counts = {"zero": 0, "big": 0}
    for _ in range(2000):
        t0 = sample_clip_start(
            n_frames=320, clip_frames=96, bad_spans_s=spans, fps=FPS, generator=g
        )
        counts["zero" if t0 == 0 else "big"] += 1
    # the width-1 point should be rare; the width-97 gap dominant
    assert counts["big"] > counts["zero"] * 20


def test_sample_raises_when_no_valid_window() -> None:
    g = _gen(0)
    with pytest.raises(ValueError, match="no valid clip"):
        sample_clip_start(
            n_frames=200, clip_frames=96, bad_spans_s=[(0.0, 10.0)], fps=FPS, generator=g
        )


def test_clip_longer_than_session_has_no_start() -> None:
    ivals = allowed_start_intervals(n_frames=64, clip_frames=96, bad_spans_s=[], fps=FPS)
    assert ivals == []
