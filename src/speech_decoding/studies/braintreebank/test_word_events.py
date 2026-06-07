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
from speech_decoding.studies.braintreebank.manifest import (
    BT_LITE_SESSIONS,
    V14_PRETRAIN_SESSIONS,
)
from speech_decoding.studies.braintreebank.word_events import (
    BTWordEvents,
    _assign_cross_session_split,
    _assign_pretrain_split,
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
    """8 word rows, no enrichment columns. ``est_idx`` (neural-clock samples) is
    deliberately offset +200 s from transcript ``start`` so tests prove events
    slice at ``est_idx``, not ``start`` (C3)."""
    starts = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    return pd.DataFrame(
        {
            "start": starts,
            "end": [s + 1.0 for s in starts],
            "est_idx": [round((s + 200.0) * 2048.0) for s in starts],
            "original_index": list(range(8)),
            "full_word": list("ABCDEFGH"),
        }
    )


def _stub_nonverbal_df() -> pd.DataFrame:
    """8 nonverbal rows interleaved with the word starts (+ offset ``est_idx``)."""
    starts = [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]
    return pd.DataFrame(
        {
            "start": starts,
            "end": [s + 1.0 for s in starts],
            "est_idx": [round((s + 200.0) * 2048.0) for s in starts],
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
    # And the emitted `start` is the NEURAL-clock onset `est_idx / SR` (C3), not
    # the transcript `start` — the fixtures offset est_idx +200 s, so asserting
    # against transcript `start` would now fail.
    sr = we._neural_sample_rate()
    for row_i, (lbl, src) in enumerate(zip(expected_labels, expected_src_indices)):
        src_df = nonverbal if int(lbl) == 0 else words
        assert rows.iloc[row_i]["start"] == float(src_df.iloc[int(src)]["est_idx"]) / sr


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
    # face_num is a transcript-derived column -> needs enrichment (H7)
    assert BTWordEvents(tasks=("face_num",))._enrich_needed() is True
    assert BTWordEvents(tasks=("delta_volume",))._enrich_needed() is True
    assert BTWordEvents(tasks=("onset",))._enrich_needed() is True
    # mixed speech + face_num still needs enrichment (face_num forces it)
    assert BTWordEvents(tasks=("speech", "face_num"))._enrich_needed() is True


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


# --- Pretrain (leakage-decouple #82) split ---------------------------------

def _pretrain_df(sessions_rows: list[tuple[tuple[int, int], int]]) -> pd.DataFrame:
    """Rows for ``_assign_pretrain_split``: subject_id/trial_id as strings
    (the production dtype), one ``task`` so grouping is per-session."""
    rows: list[dict] = []
    for (s, t), n in sessions_rows:
        for i in range(n):
            rows.append(
                {"subject_id": str(s), "trial_id": str(t), "task": "speech",
                 "label": i % 2}
            )
    return pd.DataFrame(rows)


def test_assign_pretrain_split_holds_out_positional_tail_per_session() -> None:
    df = _pretrain_df([((1, 0), 10), ((2, 1), 10)])
    out = _assign_pretrain_split(df, holdout_fraction=0.2)
    # n_hold = min(round(10*0.2)=2, (10-1)//2=4) = 2 → 6 train / 2 val / 2 test
    # PER session, so 12 / 4 / 4 globally.
    assert (out["split"] == "train").sum() == 12
    assert (out["split"] == "val").sum() == 4
    assert (out["split"] == "test").sum() == 4
    # Tail ordering within each session: last 2 → test, preceding 2 → val.
    for s, t in [("1", "0"), ("2", "1")]:
        grp = out[(out["subject_id"] == s) & (out["trial_id"] == t)]
        splits = list(grp["split"])
        assert splits == ["train"] * 6 + ["val"] * 2 + ["test"] * 2


def test_assign_pretrain_split_keeps_at_least_one_train_per_session() -> None:
    # n=3, frac=0.49 would round to 1 hold each; (n-1)//2 = 1 caps it → 1/1/1.
    out = _assign_pretrain_split(_pretrain_df([((1, 0), 3)]), holdout_fraction=0.49)
    assert sorted(out["split"]) == ["test", "train", "val"]


def test_assign_pretrain_split_small_session_stays_train() -> None:
    # A < 3-row session can't yield all three splits → it stays entirely in
    # train, while the rest of the corpus still supplies global val/test.
    out = _assign_pretrain_split(
        _pretrain_df([((1, 0), 2), ((2, 1), 20)]), holdout_fraction=0.2
    )
    small = out[(out["subject_id"] == "1") & (out["trial_id"] == "0")]
    assert set(small["split"]) == {"train"}
    # corpus-global val/test come from the large session.
    assert {"val", "test"} <= set(out["split"])


def test_assign_pretrain_split_rejects_out_of_range_fraction() -> None:
    for bad in (0.0, 0.5, 0.6, -0.1):
        with pytest.raises(ValueError, match="holdout_fraction"):
            _assign_pretrain_split(_pretrain_df([((1, 0), 10)]), holdout_fraction=bad)
    # Validation precedes the empty short-circuit: same contract for empty input.
    empty = _pretrain_df([((1, 0), 1)]).iloc[0:0]
    with pytest.raises(ValueError, match="holdout_fraction"):
        _assign_pretrain_split(empty, holdout_fraction=0.9)


def test_assign_pretrain_split_raises_clear_error_when_corpus_too_small() -> None:
    # Every session < 3 rows → global val/test empty → Data.build would crash
    # with a misdirected message. The splitter must fail here with the real cause.
    tiny = _pretrain_df([((1, 0), 2), ((2, 1), 2)])
    with pytest.raises(ValueError, match="whole corpus"):
        _assign_pretrain_split(tiny, holdout_fraction=0.2)


def test_assign_pretrain_split_robust_to_nonunique_index() -> None:
    # A non-unique index must not cross-assign between sessions (label-based
    # .loc hazard). reset_index at entry makes the positional tail correct.
    df = _pretrain_df([((1, 0), 10), ((2, 1), 4)])
    df.index = [0] * len(df)  # pathological: all-identical index labels
    out = _assign_pretrain_split(df, holdout_fraction=0.25)
    for s, t, n in [("1", "0", 10), ("2", "1", 4)]:
        grp = out[(out["subject_id"] == s) & (out["trial_id"] == t)]
        assert (grp["split"] == "train").sum() >= 1
        assert (grp["split"] == "val").sum() >= 1
        assert (grp["split"] == "test").sum() >= 1
        assert len(grp) == n


def test_assign_pretrain_split_never_emits_eval_session() -> None:
    # Even if (defensively) an eval pair reached the splitter, the corpus it
    # operates on is legal-only; this asserts the OUTPUT carries no eval session.
    df = _pretrain_df([(sess, 8) for sess in V14_PRETRAIN_SESSIONS[:3]])
    out = _assign_pretrain_split(df, holdout_fraction=0.25)
    seen = {(int(s), int(t)) for s, t in zip(out["subject_id"], out["trial_id"])}
    assert seen.isdisjoint(set(BT_LITE_SESSIONS))


def test_pretrain_split_train_spans_every_subject_of_corpus() -> None:
    """LC3b: the pretrain SPLITTER routes every legal session into train (only a
    per-session positional tail goes to val/test), so the train split
    structurally spans every subject AND every session of the corpus. This is the
    splitter-side anti-starvation property — the structural reason the realized
    train loader isn't starved — complementing the realized-loader guard (LC3).
    5 rows/session keeps the 0.2 holdout tail from consuming a whole session."""
    df = _pretrain_df([(sess, 5) for sess in V14_PRETRAIN_SESSIONS])
    out = _assign_pretrain_split(df, holdout_fraction=0.2)
    train = out[out["split"] == "train"]
    train_sessions = {
        (int(s), int(t)) for s, t in zip(train["subject_id"], train["trial_id"])
    }
    # Every corpus session appears in train; train spans all 7 cohort subjects.
    assert train_sessions == set(V14_PRETRAIN_SESSIONS)
    assert sorted({s for s, _ in train_sessions}) == [1, 2, 3, 4, 6, 8, 9]


def _pretrain_ieeg_events() -> pd.DataFrame:
    """Two legal pretrain sessions + one eval session — the eval session must be
    dropped by ``_timeline_is_used`` under eval_mode='Pretrain'."""
    legal = list(V14_PRETRAIN_SESSIONS[:2])  # e.g. (1,0), (2,1)
    eval_sess = BT_LITE_SESSIONS[2]  # (2, 0) — off-limits
    rows = []
    for s, t in [*legal, eval_sess]:
        rows.append(
            {
                "type": "Ieeg", "start": 0.0, "duration": 100.0,
                "frequency": 2048.0, "subject": f"Wang2024Treebank/btbank{s}",
                "subject_id": str(s), "trial_id": str(t),
                "timeline": f"tl{s}_{t}",
            }
        )
    return pd.DataFrame(rows)


def test_btwordevents_pretrain_mode_drops_eval_sessions(monkeypatch) -> None:
    """End-to-end: eval_mode='Pretrain' keeps only V14_PRETRAIN_SESSIONS, splits
    them train/val/test, and never emits an off-limits eval session."""
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
        eval_mode="Pretrain",
        pretrain_holdout_fraction=0.25,
        bt_root="/dev/null",
    )
    out = step(_pretrain_ieeg_events())
    words = out.loc[out["type"] == "Word"]
    assert len(words) > 0
    seen = {(int(s), int(t)) for s, t in zip(words["subject_id"], words["trial_id"])}
    # Only the two legal sessions; the eval session (2,0) is gone.
    assert seen == set(V14_PRETRAIN_SESSIONS[:2])
    assert seen.isdisjoint(set(BT_LITE_SESSIONS))
    assert set(words["split"]) == {"train", "val", "test"}


def test_btwordevents_emits_string_subject_trial_id_schema(monkeypatch) -> None:
    """LC8: realized BT Word-row triggers carry subject_id/trial_id as STRING.
    Both leakage guards depend on this schema — the column-presence fail-closed
    branch and the int()-cast used to compare against the (int,int) eval/cohort
    sets. A silent switch to int columns would make `int(v)` still work but a
    switch to a different label (e.g. dropping the column) must fail loudly; this
    pins the dtype the guards were written against."""
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
        eval_mode="Pretrain",
        pretrain_holdout_fraction=0.25,
        bt_root="/dev/null",
    )
    out = step(_pretrain_ieeg_events())
    words = out.loc[out["type"] == "Word"]
    assert len(words) > 0
    for col in ("subject_id", "trial_id"):
        assert all(isinstance(v, str) for v in words[col]), f"{col} must be str-typed"
        # Round-trips through int() — the exact cast both guards perform.
        assert all(int(v) >= 0 for v in words[col])
