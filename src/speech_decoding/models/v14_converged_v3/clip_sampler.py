"""v14_converged_v3 — guard-2 reject-free clip-t0 sampler (Phase D2a).

Random-continuous clip windows on the per-session continuous spec cache, with the
guard-2 bad-TIME-span drop realised WITHOUT a retry loop (memo project-v3-pipeline-
build-contract-2026-07-10). A clip ``[t0, t0+L)`` overlaps a bad span ``[a, b)``
iff ``t0 < b and t0+L > a`` ⇔ ``t0 ∈ (a-L, b)``; so the forbidden t0-set is
``⋃(a-L, b)`` and we sample t0 directly from its complement inside the legal start
range ``[0, T-L]``, weighting each allowed sub-interval by its width (uniform over
valid clips, no bias toward span-adjacent windows).

Frame clock = 32 Hz spec frames (clip_len 3.0 s = 96 frames). Bad spans arrive in
hop-invariant SECONDS and convert once via ``round(s * fps)``. All intervals are
INCLUSIVE integer ``(lo, hi)`` frame ranges: a span's forbidden starts are
``[a-L+1, b-1]`` (the open real interval, in integers).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort + merge overlapping/adjacent inclusive integer intervals."""
    if not intervals:
        return []
    intervals.sort()
    out = [intervals[0]]
    for lo, hi in intervals[1:]:
        plo, phi = out[-1]
        if lo <= phi + 1:  # overlap or touch
            out[-1] = (plo, max(phi, hi))
        else:
            out.append((lo, hi))
    return out


def allowed_start_intervals(
    *,
    n_frames: int,
    clip_frames: int,
    bad_spans_s: Sequence[tuple[float, float]],
    fps: float,
) -> list[tuple[int, int]]:
    """Legal clip-start frames as inclusive ``(lo, hi)`` intervals, guard-2 applied.

    Returns ``[]`` if the clip is longer than the session or every start is blocked.
    """
    hi_start = n_frames - clip_frames
    if hi_start < 0:
        return []

    forbidden: list[tuple[int, int]] = []
    for a_s, b_s in bad_spans_s:
        a = round(a_s * fps)
        b = round(b_s * fps)
        lo = max(0, a - clip_frames + 1)
        hi = min(hi_start, b - 1)
        if lo <= hi:
            forbidden.append((lo, hi))
    forbidden = _merge(forbidden)

    allowed: list[tuple[int, int]] = []
    cursor = 0
    for lo, hi in forbidden:
        if lo > cursor:
            allowed.append((cursor, lo - 1))
        cursor = hi + 1
    if cursor <= hi_start:
        allowed.append((cursor, hi_start))
    return allowed


def sample_clip_start(
    *,
    n_frames: int,
    clip_frames: int,
    bad_spans_s: Sequence[tuple[float, float]],
    fps: float,
    generator: torch.Generator,
) -> int:
    """Draw one guard-2-valid clip start (frames), length-weighted over allowed gaps."""
    allowed = allowed_start_intervals(
        n_frames=n_frames, clip_frames=clip_frames, bad_spans_s=bad_spans_s, fps=fps
    )
    if not allowed:
        raise ValueError(
            f"no valid clip start: session {n_frames} frames, clip {clip_frames}, "
            f"{len(bad_spans_s)} bad spans block every window"
        )
    widths = torch.tensor([hi - lo + 1 for lo, hi in allowed], dtype=torch.float64)
    which = int(torch.multinomial(widths / widths.sum(), 1, generator=generator).item())
    lo, hi = allowed[which]
    offset = int(torch.randint(0, hi - lo + 1, (1,), generator=generator).item())
    return lo + offset
