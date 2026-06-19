"""SSL clip-level bad-time-window filter — Layer 2 of the v14 bad-electrode defense.

The static exclusion (loader/anatomy) drops whole flaky CONTACTS; the winsor clamp
(normalize) bounds every cell. This layer drops whole pretrain CLIPS that overlap a
brief, time-localized glitch span where the post-static, post-CAR ``|STFT|`` robust-z
blows up across many electrodes at once (a common-mode transient the per-cell winsor
cannot fix because it is a JOINT pattern, not a single cell). The electrodes are fine;
the 5 s window is garbage.

Contract
--------
* A per-session **sidecar** (one JSON per ``(subject_id, trial_id)``) lists bad
  ``[start_s, end_s]`` spans in NEURAL seconds from recording start — the same clock as
  a Word event's ``start`` (``est_idx / sample_rate``; see ``word_events.py``). The
  spans are produced offline by the per-window scan run on the rebuilt static cache
  (``scripts/neuroprobe/precompute_bad_windows.py``).
* :func:`filter_events_by_bad_windows` drops every TRIGGER (clip-anchor) event whose
  neural clip window ``[start + clip_start_s, start + clip_start_s + clip_dur_s]``
  overlaps any bad span for that event's session. Only rows matching the segmenter's
  ``trigger_query`` (the Word anchors) are eligible — the continuous ``Ieeg`` data row
  (``start == 0``) is always kept, else an early bad span would orphan the whole
  session. It removes ROWS of the events DataFrame only — never an electrode — so the
  DP4 electrode-row alignment (support / valid_mask vs the front-end voltage) is
  untouched.
* **Pretrain-only.** The filter runs only when the datamodule is given a
  ``bad_window_dir``; the Phase-4 eval datamodule leaves it unset so the Neuroprobe
  parity-locked clip sets are never altered.
"""

from __future__ import annotations

import json
import typing as tp
from pathlib import Path

import pandas as pd


def bad_window_session_key(subject_id: tp.Any, trial_id: tp.Any) -> str:
    """Canonical per-session key, e.g. ``btbank2_t2`` — the single source of truth for
    both the sidecar filename (precompute) and the lookup (filter). Mirrors the
    ``btbank{subject}_{trial}`` tag the per-window scan already emits. ``subject_id`` /
    ``trial_id`` may arrive as the str columns the events DataFrame carries."""
    return f"btbank{int(subject_id)}_t{int(trial_id)}"


def load_bad_windows(
    bad_window_dir: str | Path,
) -> dict[str, list[tuple[float, float]]]:
    """Read every ``*.json`` sidecar in ``bad_window_dir`` into
    ``{session_key: [(start_s, end_s), ...]}``.

    A missing directory fails loud (a typo'd path must not silently disable the
    filter); an existing-but-empty directory returns ``{}`` (e.g. before the precompute
    has run, or a cohort with no glitches) and the filter then drops nothing. A session
    absent from the map is treated as having no bad windows (keep all its clips)."""
    root = Path(bad_window_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"bad_window_dir does not exist: {root}. Set it to the directory of "
            "per-session bad-window sidecars produced by precompute_bad_windows.py, "
            "or leave it unset to disable the clip filter."
        )
    out: dict[str, list[tuple[float, float]]] = {}
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text())
        spans = payload.get("bad_windows_s", [])
        key = payload.get("session") or path.stem
        windows: list[tuple[float, float]] = []
        for span in spans:
            lo, hi = float(span[0]), float(span[1])
            if not (hi > lo):
                raise ValueError(
                    f"{path.name}: bad window {span!r} is not a positive interval"
                )
            windows.append((lo, hi))
        out[key] = windows
    return out


def _overlaps_any(
    lo: float, hi: float, spans: list[tuple[float, float]]
) -> bool:
    """Half-open interval overlap: ``[lo, hi)`` meets ``[b0, b1)`` iff
    ``lo < b1 and b0 < hi``. A clip merely abutting a bad span (touching at an
    endpoint) is kept."""
    return any(lo < b1 and b0 < hi for b0, b1 in spans)


def filter_events_by_bad_windows(
    events: pd.DataFrame,
    bad_windows: dict[str, list[tuple[float, float]]],
    *,
    clip_start_s: float,
    clip_dur_s: float,
    trigger_query: str | None = None,
) -> pd.DataFrame:
    """Return ``events`` with every TRIGGER (clip-anchor) row dropped whose neural clip
    window overlaps a bad span for its ``(subject_id, trial_id)`` session.

    ``clip_start_s`` / ``clip_dur_s`` are the segmenter's ``start`` (neural lag) and
    ``duration`` (clip length) — the window actually sliced from the neural stream is
    ``[event.start + clip_start_s, event.start + clip_start_s + clip_dur_s]``.

    ``trigger_query`` is the segmenter's ``trigger_query`` (e.g. ``"type == 'Word'"``).
    ONLY rows matching it are clip anchors eligible for dropping; every other row is kept
    unconditionally. This guard is load-bearing: ``study.run()`` events also carry the
    continuous **Ieeg** recording row (one per session, ``start == 0.0``) that the
    segmenter's extractors source from. Without the guard, that ``start==0`` row's
    ``[0, clip_dur]`` window is tested against the session's bad spans, so a single early
    bad window (a glitch in the first ``clip_dur`` seconds) DROPS the whole session's
    Ieeg event — orphaning every Word in it ("N segments are missing events of type
    Ieeg"). ``None`` (legacy) treats every row as an anchor; production callers pass the
    segmenter's trigger query.

    Rows for sessions with no sidecar (or no bad spans) are kept unchanged. The returned
    frame keeps the input column set and dtypes; only rows are removed (index reset)."""
    if not bad_windows:
        return events.reset_index(drop=True)
    for col in ("start", "subject_id", "trial_id"):
        if col not in events.columns:
            raise KeyError(
                f"filter_events_by_bad_windows needs a {col!r} column; got "
                f"{list(events.columns)}"
            )

    if trigger_query is None:
        is_anchor: list[bool] = [True] * len(events)
    else:
        is_anchor = events.eval(trigger_query).to_numpy(dtype=bool).tolist()

    keep = []
    for anchor, start, subject_id, trial_id in zip(
        is_anchor, events["start"], events["subject_id"], events["trial_id"]
    ):
        if not anchor:
            keep.append(True)  # non-anchor (continuous data) row — never dropped
            continue
        spans = bad_windows.get(bad_window_session_key(subject_id, trial_id))
        if not spans:
            keep.append(True)
            continue
        lo = float(start) + clip_start_s
        hi = lo + clip_dur_s
        keep.append(not _overlaps_any(lo, hi, spans))
    return events.loc[keep].reset_index(drop=True)
