"""Unit tests for :class:`BTWordEvents`."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from speech_decoding.studies.braintreebank import word_events as we
from speech_decoding.studies.braintreebank.labels import (
    derive_label_indices,
    ordered_dataset_labels,
    ordered_dataset_source_indices,
)
from speech_decoding.studies.braintreebank.word_events import (
    BTWordEvents,
    _assign_cross_session_split,
    _word_event_rows,
)


def _synthetic_ieeg_events() -> pd.DataFrame:
    """Two timelines (one Lite subject, two trials): mimic what
    :class:`Wang2024Treebank` emits."""
    return pd.DataFrame(
        [
            {
                "type": "Ieeg",
                "start": 0.0,
                "duration": 100.0,
                "frequency": 2048.0,
                "subject": "Wang2024Treebank/btbank2",
                "subject_id": "2",
                "trial_id": "0",
                "timeline": "Wang2024Treebank:subject=btbank2,subject_id=2,trial_id=0",
            },
            {
                "type": "Ieeg",
                "start": 0.0,
                "duration": 100.0,
                "frequency": 2048.0,
                "subject": "Wang2024Treebank/btbank2",
                "subject_id": "2",
                "trial_id": "4",
                "timeline": "Wang2024Treebank:subject=btbank2,subject_id=2,trial_id=4",
            },
        ]
    )


def _stub_words_df() -> pd.DataFrame:
    """8 word rows with monotonic ``start``, no enrichment columns."""
    return pd.DataFrame(
        {
            "start": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "end": [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0, 81.0],
            "original_index": list(range(8)),
            "full_word": list("ABCDEFGH"),
        }
    )


def _stub_nonverbal_df() -> pd.DataFrame:
    """8 nonverbal rows interleaved with the word starts."""
    return pd.DataFrame(
        {
            "start": [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0],
            "end": [16.0, 26.0, 36.0, 46.0, 56.0, 66.0, 76.0, 86.0],
        }
    )


def test_word_event_rows_matches_upstream_interleaved_order() -> None:
    """Row order must mirror upstream Dataset.__getitem__ verbatim:
    items strictly interleave classes via ``(idx + 1) % n_classes``.
    Sorting by overall ``start`` instead would flip val/test majority class
    whenever one class temporally clusters differently."""
    words = _stub_words_df()
    nonverbal = _stub_nonverbal_df()
    rows = _word_event_rows(
        subject_id=2,
        trial_id=4,
        timeline="tl",
        words_df=words,
        nonverbal_df=nonverbal,
        tasks=("speech",),
        binary_tasks=True,
        lite=False,
        nano=False,
        random_seed=42,
        duration=1.0,
    )
    label_indices = derive_label_indices(
        words_df=words,
        nonverbal_df=nonverbal,
        task="speech",
        binary_tasks=True,
        lite=False,
        nano=False,
        random_seed=42,
    )
    expected_labels = ordered_dataset_labels(label_indices)
    expected_src_indices = ordered_dataset_source_indices(label_indices)

    assert (rows["type"] == "Word").all()
    assert len(rows) == len(expected_labels)
    np.testing.assert_array_equal(rows["label"].to_numpy(), expected_labels)
    # First-half / second-half cut on this order is class-balanced by construction
    cut = len(rows) // 2
    val_labels = rows["label"].iloc[:cut].to_numpy()
    test_labels = rows["label"].iloc[cut:].to_numpy()
    assert (val_labels == 0).sum() == (val_labels == 1).sum()
    assert (test_labels == 0).sum() == (test_labels == 1).sum()
    # And the source `start` matches what the upstream interleaved order would emit
    for row_i, (lbl, src) in enumerate(zip(expected_labels, expected_src_indices)):
        src_df = nonverbal if int(lbl) == 0 else words
        assert rows.iloc[row_i]["start"] == float(src_df.iloc[int(src)]["start"])


def test_assign_cross_session_split_halves_test_trial_chronologically() -> None:
    df = pd.DataFrame(
        {
            "type": ["Word"] * 8,
            "start": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "duration": [1.0] * 8,
            "task": ["speech"] * 8,
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
            "subject_id": ["2"] * 4 + ["2"] * 4,
            "trial_id": ["0"] * 4 + ["4"] * 4,
            "timeline": ["tl0"] * 4 + ["tl4"] * 4,
        }
    )
    out = _assign_cross_session_split(df, test_subject_id=2, test_trial_id=4)
    train = out.loc[out["split"] == "train"]
    val = out.loc[out["split"] == "val"]
    test = out.loc[out["split"] == "test"]

    assert len(train) == 4
    assert (train["trial_id"] == "0").all()
    assert len(val) == 2 and len(test) == 2
    assert (val["trial_id"] == "4").all() and (test["trial_id"] == "4").all()
    assert val["start"].max() < test["start"].min(), "val precedes test chronologically"


def test_assign_cross_session_split_drops_other_subjects() -> None:
    df = pd.DataFrame(
        {
            "type": ["Word"] * 4,
            "start": [10.0, 20.0, 30.0, 40.0],
            "duration": [1.0] * 4,
            "task": ["speech"] * 4,
            "label": [0, 1, 0, 1],
            "subject_id": ["1", "1", "2", "2"],
            "trial_id": ["1", "1", "4", "4"],
            "timeline": ["s1"] * 2 + ["s2"] * 2,
        }
    )
    out = _assign_cross_session_split(df, test_subject_id=2, test_trial_id=4)
    assert len(out) == 2
    assert (out["subject_id"] == "2").all()


def test_btwordevents_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="unknown tasks"):
        BTWordEvents(tasks=("does_not_exist",))


def test_btwordevents_enrich_only_needed_for_continuous_tasks() -> None:
    assert BTWordEvents(tasks=("speech",))._enrich_needed() is False
    assert BTWordEvents(tasks=("face_num",))._enrich_needed() is False
    assert BTWordEvents(tasks=("delta_volume",))._enrich_needed() is True
    assert BTWordEvents(tasks=("onset",))._enrich_needed() is True


def test_btwordevents_run_appends_word_rows_with_split(monkeypatch) -> None:
    """End-to-end on the EventsTransform: ``speech`` task on a 2-timeline
    synthetic Ieeg DataFrame yields a chained ``Word``-rows table."""
    monkeypatch.setattr(
        we,
        "_load_words_and_nonverbal",
        lambda subject_id, trial_id, *, bt_root, enrich: (
            _stub_words_df(), _stub_nonverbal_df(),
        ),
    )
    step = BTWordEvents(
        tasks=("speech",),
        binary_tasks=True,
        eval_mode="CrossSession",
        test_subject_id=2,
        test_trial_id=4,
        bt_root="/dev/null",
    )
    out = step(_synthetic_ieeg_events())

    ieeg = out.loc[out["type"] == "Ieeg"]
    words = out.loc[out["type"] == "Word"]
    assert len(ieeg) == 2
    assert len(words) > 0
    assert set(words["split"]) == {"train", "val", "test"}
    assert set(words["task"]) == {"speech"}
    assert (words["label"].isin([0, 1])).all()


@pytest.mark.skipif(
    os.environ.get("ROOT_DIR_BRAINTREEBANK") is None,
    reason=(
        "neuroprobe.config imports os.environ['ROOT_DIR_BRAINTREEBANK'] at "
        "module-import time; skip on environments without BT data mounted "
        "(local). Runs cleanly on DCC where ROOT_DIR_BRAINTREEBANK is set."
    ),
)
def test_btwordevents_real_vendored_csvs_balanced_val_test_halves(monkeypatch) -> None:
    """Regression for the chronological-sort bug: on real (2, 0) + (2, 4) speech
    data, ``val`` and ``test`` halves must each be exactly 50/50 balanced.
    Previously sort-by-``start`` produced 61/39 vs 39/61 splits for trial 4,
    silently making ``test_acc ≈ 1 − val_acc`` regardless of model quality."""
    from pathlib import Path
    csv_dir = Path(we._vendored_csv_dir())
    if not (csv_dir / "subject2_trial4_words_df.csv").exists():
        pytest.skip("vendored BT CSVs not available")

    def _load(subject_id, trial_id, *, bt_root, enrich):
        words = pd.read_csv(csv_dir / f"subject{subject_id}_trial{trial_id}_words_df.csv")
        nonverbal = pd.read_csv(csv_dir / f"subject{subject_id}_trial{trial_id}_nonverbal_df.csv")
        return words, nonverbal

    monkeypatch.setattr(we, "_load_words_and_nonverbal", _load)

    ieeg = pd.DataFrame(
        [
            {
                "type": "Ieeg",
                "start": 0.0, "duration": 10000.0, "frequency": 2048.0,
                "subject": "Wang2024Treebank/btbank2",
                "subject_id": "2", "trial_id": trial,
                "timeline": f"tl{trial}",
            }
            for trial in ("0", "4")
        ]
    )
    step = BTWordEvents(
        tasks=("speech",), binary_tasks=True, lite=True,
        eval_mode="CrossSession",
        test_subject_id=2, test_trial_id=4,
        bt_root="/dev/null",
    )
    out = step(ieeg)
    words = out.loc[out["type"] == "Word"]
    for split in ("val", "test"):
        labels = words.loc[words["split"] == split, "label"].to_numpy()
        n_zero = int((labels == 0).sum())
        n_one = int((labels == 1).sum())
        assert abs(n_zero - n_one) <= 1, (
            f"{split} split must be ≤1 off from 50/50 balanced; got "
            f"n_zero={n_zero} n_one={n_one} (total {len(labels)})"
        )
    # And train trial (2, 0) — full, also balanced
    train_labels = words.loc[words["split"] == "train", "label"].to_numpy()
    assert abs(int((train_labels == 0).sum()) - int((train_labels == 1).sum())) <= 1


def test_btwordevents_run_raises_on_empty_match(monkeypatch) -> None:
    monkeypatch.setattr(
        we,
        "_load_words_and_nonverbal",
        lambda *args, **kw: (_stub_words_df(), _stub_nonverbal_df()),
    )
    step = BTWordEvents(
        tasks=("speech",),
        eval_mode="CrossSession",
        test_subject_id=99,  # no Ieeg with this subject
        test_trial_id=0,
        bt_root="/dev/null",
    )
    with pytest.raises(RuntimeError, match="matched zero usable timelines"):
        step(_synthetic_ieeg_events())
