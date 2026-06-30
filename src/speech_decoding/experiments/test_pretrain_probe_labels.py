"""Tests for the leaderboard-faithful label/clip source + split adapter (#7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from speech_decoding.experiments.pretrain_probe_labels import build_session_targets


def _events(subject_id: int, trial_id: int) -> pd.DataFrame:
    """Two tasks over 8 neural windows in interleaved (alternating-class) item order,
    mimicking ``_word_event_rows(balance=True)``. 'onset' uses windows 0-7; 'volume'
    uses a 4-window subset (so some clips are outside its balanced set → NaN labels)."""
    starts = [round(0.5 * i, 3) for i in range(8)]
    rows = []
    # onset: all 8 windows, interleaved labels 1,0,1,0,...
    for i, s in enumerate(starts):
        rows.append(_row(s, "onset", i % 2 == 0, subject_id, trial_id))
    # volume: windows {0,2,4,6}, interleaved labels 1,0,1,0
    for j, i in enumerate([0, 2, 4, 6]):
        rows.append(_row(starts[i], "volume", j % 2 == 0, subject_id, trial_id))
    return pd.DataFrame(rows)


def _row(start, task, positive, subject_id, trial_id):
    return {
        "type": "Word", "start": start, "duration": 1.0, "text": "<word>",
        "task": task, "label": 1 if positive else 0,
        "subject_id": str(subject_id), "trial_id": str(trial_id),
        "timeline": "tl", "movie_onset_s": start + 100.0,
    }


def test_union_axis_is_unique_sorted_windows():
    t = build_session_targets(_events(2, 1), subject_id=2, trial_id=1)
    assert t.clip_starts.tolist() == [round(0.5 * i, 3) for i in range(8)]
    assert t.clip_durations.tolist() == [1.0] * 8
    np.testing.assert_allclose(t.clip_movie_onsets, t.clip_starts + 100.0)


def test_labels_nan_outside_task_balanced_set():
    t = build_session_targets(_events(2, 1), subject_id=2, trial_id=1)
    assert np.isfinite(t.labels["onset"]).all()                 # onset covers all 8
    vol = t.labels["volume"]
    assert np.isfinite(vol[[0, 2, 4, 6]]).all()                 # volume covers the evens
    assert np.isnan(vol[[1, 3, 5, 7]]).all()                    # odds outside → NaN


def test_ws_split_partitions_each_task_into_union_indices():
    t = build_session_targets(_events(2, 1), subject_id=2, trial_id=1, n_folds=2)
    for fold in (0, 1):
        sp = t.ws_split["onset"][fold]
        allrows = np.concatenate([sp["train"], sp["val"], sp["test"]])
        assert set(allrows.tolist()) <= set(range(8))
        assert len(set(allrows.tolist())) == len(allrows)       # disjoint
        # KFold2: train is one half, val+test the other → all 8 onset windows covered.
        assert sorted(allrows.tolist()) == list(range(8))


def test_cs_split_is_chronological_halves_per_task():
    t = build_session_targets(_events(2, 1), subject_id=2, trial_id=1)
    sp = t.cs_split["volume"]
    # volume's 4 ordered windows {0,2,4,6} → val = first 2, test = last 2.
    assert sp["val"].tolist() == [0, 2]
    assert sp["test"].tolist() == [4, 6]


def test_split_indices_reference_only_finite_labels():
    t = build_session_targets(_events(2, 1), subject_id=2, trial_id=1)
    for task in ("onset", "volume"):
        y = t.labels[task]
        for fold in (0, 1):
            for name in ("train", "val", "test"):
                idx = t.ws_split[task][fold][name]
                assert np.isfinite(y[idx]).all()
        for name in ("val", "test"):
            assert np.isfinite(y[t.cs_split[task][name]]).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
