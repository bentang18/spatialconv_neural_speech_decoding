"""Tests for the SSL clip-level bad-time-window filter (Layer 2 of the bad-electrode
defense). The filter drops pretrain CLIPS overlapping a glitch span; it must drop the
right rows (lag- and duration-aware overlap), keep everything else, and never touch a
column — DP4 row-alignment depends on it removing rows only."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from speech_decoding.experiments.bad_windows import (
    bad_window_session_key,
    filter_events_by_bad_windows,
    load_bad_windows,
)


def _events(rows: list[dict]) -> pd.DataFrame:
    """Minimal events frame carrying the columns the filter reads plus a payload
    column the filter must preserve untouched."""
    return pd.DataFrame(rows)


def test_session_key_casts_string_columns() -> None:
    # events carry subject_id / trial_id as strings; the key must canonicalize them
    assert bad_window_session_key("2", "2") == "btbank2_t2"
    assert bad_window_session_key(2, 0) == "btbank2_t0"


def test_load_bad_windows_reads_sidecars(tmp_path: Path) -> None:
    (tmp_path / "btbank2_t2.json").write_text(
        json.dumps({"session": "btbank2_t2", "bad_windows_s": [[330.0, 335.0]]})
    )
    (tmp_path / "btbank4_t0.json").write_text(
        json.dumps({"session": "btbank4_t0", "bad_windows_s": [[10.0, 15.0], [20.0, 25.0]]})
    )
    bw = load_bad_windows(tmp_path)
    assert bw == {
        "btbank2_t2": [(330.0, 335.0)],
        "btbank4_t0": [(10.0, 15.0), (20.0, 25.0)],
    }


def test_load_bad_windows_missing_dir_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="bad_window_dir does not exist"):
        load_bad_windows(tmp_path / "nope")


def test_load_bad_windows_empty_dir_is_empty(tmp_path: Path) -> None:
    assert load_bad_windows(tmp_path) == {}


def test_load_bad_windows_rejects_degenerate_span(tmp_path: Path) -> None:
    (tmp_path / "btbank1_t0.json").write_text(
        json.dumps({"session": "btbank1_t0", "bad_windows_s": [[5.0, 5.0]]})
    )
    with pytest.raises(ValueError, match="not a positive interval"):
        load_bad_windows(tmp_path)


def test_filter_drops_only_overlapping_clips() -> None:
    # clip window = [start + lag, start + lag + dur]; lag=0.5, dur=5.0
    events = _events([
        {"subject_id": "2", "trial_id": "2", "start": 100.0, "text": "a"},  # [100.5,105.5] miss
        {"subject_id": "2", "trial_id": "2", "start": 328.0, "text": "b"},  # [328.5,333.5] HITS [330,335]
        {"subject_id": "2", "trial_id": "2", "start": 500.0, "text": "c"},  # miss
    ])
    bw = {"btbank2_t2": [(330.0, 335.0)]}
    out = filter_events_by_bad_windows(events, bw, clip_start_s=0.5, clip_dur_s=5.0)
    assert out["text"].tolist() == ["a", "c"]
    # columns + dtypes preserved, index reset (DP4: rows removed, nothing else touched)
    assert list(out.columns) == list(events.columns)
    assert out.index.tolist() == [0, 1]


def test_filter_lag_offset_changes_membership() -> None:
    """The lag must enter the overlap test: an event whose RAW start misses the glitch
    but whose lagged clip window hits it is dropped (and vice-versa)."""
    events = _events([{"subject_id": "2", "trial_id": "2", "start": 329.0, "text": "x"}])
    bw = {"btbank2_t2": [(330.0, 335.0)]}
    # lag 0.0 dur 0.5 -> [329.0, 329.5] misses -> kept
    kept = filter_events_by_bad_windows(events, bw, clip_start_s=0.0, clip_dur_s=0.5)
    assert kept["text"].tolist() == ["x"]
    # lag 1.5 dur 0.5 -> [330.5, 331.0] hits -> dropped
    dropped = filter_events_by_bad_windows(events, bw, clip_start_s=1.5, clip_dur_s=0.5)
    assert dropped.empty


def test_filter_abutting_clip_is_kept() -> None:
    """Half-open overlap: a clip ending exactly at a bad-span start (or starting at its
    end) does not overlap and is kept."""
    events = _events([
        {"subject_id": "2", "trial_id": "2", "start": 325.0, "text": "end-at"},   # [325,330) | [330,335)
        {"subject_id": "2", "trial_id": "2", "start": 335.0, "text": "start-at"},  # [335,340) | [330,335)
    ])
    bw = {"btbank2_t2": [(330.0, 335.0)]}
    out = filter_events_by_bad_windows(events, bw, clip_start_s=0.0, clip_dur_s=5.0)
    assert out["text"].tolist() == ["end-at", "start-at"]


def test_filter_keeps_sessions_without_sidecar() -> None:
    events = _events([
        {"subject_id": "2", "trial_id": "2", "start": 331.0, "text": "hit"},     # dropped
        {"subject_id": "7", "trial_id": "0", "start": 331.0, "text": "no-side"},  # no sidecar -> kept
    ])
    bw = {"btbank2_t2": [(330.0, 335.0)]}
    out = filter_events_by_bad_windows(events, bw, clip_start_s=0.0, clip_dur_s=5.0)
    assert out["text"].tolist() == ["no-side"]


def test_filter_empty_bad_windows_is_passthrough() -> None:
    events = _events([{"subject_id": "2", "trial_id": "2", "start": 331.0, "text": "x"}])
    out = filter_events_by_bad_windows(events, {}, clip_start_s=0.0, clip_dur_s=5.0)
    assert out["text"].tolist() == ["x"]


def test_filter_keeps_continuous_ieeg_row_with_trigger_query() -> None:
    """Reproduces the 9964-missing-Ieeg orphaning bug: ``study.run()`` events carry the
    continuous Ieeg recording row (``start==0``) alongside Word anchors. With a bad span
    in the first clip_dur seconds, only the Word anchors that overlap must drop — the
    Ieeg data row must SURVIVE (dropping it leaves the session with no neural source, so
    the strict segmenter raises 'missing events of type Ieeg' for every Word)."""
    events = _events([
        {"type": "Ieeg", "subject_id": "2", "trial_id": "2", "start": 0.0, "text": "rec"},
        {"type": "Word", "subject_id": "2", "trial_id": "2", "start": 2.0, "text": "hit"},   # [2,7) HITS [1,3)
        {"type": "Word", "subject_id": "2", "trial_id": "2", "start": 100.0, "text": "ok"},   # miss
    ])
    bw = {"btbank2_t2": [(1.0, 3.0)]}  # early span: overlaps the start==0 clip window
    out = filter_events_by_bad_windows(
        events, bw, clip_start_s=0.0, clip_dur_s=5.0, trigger_query="type == 'Word'"
    )
    # Ieeg recording row kept (non-anchor); only the overlapping Word dropped.
    assert out["text"].tolist() == ["rec", "ok"]
    assert "Ieeg" in out["type"].tolist()


def test_filter_without_trigger_query_drops_ieeg_row() -> None:
    """The legacy ``trigger_query=None`` path treats EVERY row as an anchor, so the
    start==0 Ieeg row is wrongly dropped — this is the pre-fix behavior the production
    caller avoids by passing the segmenter's trigger query. Pinned so the guard can't
    silently regress to dropping data rows."""
    events = _events([
        {"type": "Ieeg", "subject_id": "2", "trial_id": "2", "start": 0.0, "text": "rec"},
        {"type": "Word", "subject_id": "2", "trial_id": "2", "start": 100.0, "text": "ok"},
    ])
    bw = {"btbank2_t2": [(1.0, 3.0)]}
    out = filter_events_by_bad_windows(events, bw, clip_start_s=0.0, clip_dur_s=5.0)
    assert out["text"].tolist() == ["ok"]  # Ieeg row dropped (the bug, when no anchor guard)


def test_filter_missing_column_fails_loud() -> None:
    events = pd.DataFrame([{"subject_id": "2", "trial_id": "2"}])  # no 'start'
    with pytest.raises(KeyError, match="start"):
        filter_events_by_bad_windows(
            events, {"btbank2_t2": [(0.0, 1.0)]}, clip_start_s=0.0, clip_dur_s=5.0
        )
