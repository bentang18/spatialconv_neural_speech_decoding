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
from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS
from speech_decoding.studies.braintreebank.word_events import (
    BTWordEvents,
    _assign_all_cells_split,
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
    """8 nonverbal rows mirroring REAL ``nonverbal_df``: ``start`` IS the neural
    clock (``== est_idx/SR``) and there is NO movie-clock column (unlike
    ``words_df``). Neural times sit inside the words' neural span [210, 280] s so
    the movie-onset re-key interpolates rather than clamps. (Fixed 2026-06-08: the
    old fixture gave nonverbal a fake +200 s movie clock, masking the P3 bug.)"""
    neural_s = [215.0, 225.0, 235.0, 245.0, 255.0, 265.0, 270.0, 275.0]
    return pd.DataFrame(
        {
            "start": neural_s,  # neural clock, == est_idx/SR (no movie clock)
            "end": [s + 1.0 for s in neural_s],
            "est_idx": [round(s * 2048.0) for s in neural_s],
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
    sr = we._neural_sample_rate(2)
    for row_i, (lbl, src) in enumerate(zip(expected_labels, expected_src_indices)):
        src_df = nonverbal if int(lbl) == 0 else words
        assert rows.iloc[row_i]["start"] == float(src_df.iloc[int(src)]["est_idx"]) / sr


def test_neural_sample_rate_is_subject_aware_s9_is_half() -> None:
    """LG1: the neural-clock rate is per-subject (single source =
    ``BT_SUBJECT_NATIVE_RATE_HZ``) — 2048 for a default subject, 1024 for S9. So
    the SAME ``est_idx`` (native grid for both ``words_df`` and ``nonverbal_df``,
    verified by est_idx span ≤ native h5 length) yields an onset that is exactly
    2× larger in seconds for S9 than a 2048 subject. The voltage loader resamples
    S9 1024→2048, preserving wall-clock time, so this doubled onset is correct."""
    assert we._neural_sample_rate(2) == 2048.0
    assert we._neural_sample_rate(9) == 1024.0

    words = _stub_words_df()
    nonverbal = _stub_nonverbal_df()
    rows_2048 = _word_event_rows(
        subject_id=2, trial_id=0, timeline="tl", words_df=words,
        nonverbal_df=nonverbal, tasks=("speech",), binary_tasks=True,
        lite=False, nano=False, random_seed=42, duration=1.0, balance=False,
    )
    rows_1024 = _word_event_rows(
        subject_id=9, trial_id=0, timeline="tl", words_df=words,
        nonverbal_df=nonverbal, tasks=("speech",), binary_tasks=True,
        lite=False, nano=False, random_seed=42, duration=1.0, balance=False,
    )

    # Match rows by est_idx (start * native_rate) and assert S9 onset == 2× the
    # 2048-subject onset == est_idx / 1024 — never est_idx / 2048.
    by_text_2048 = {r["text"]: r for _, r in rows_2048.iterrows()}
    n_verbal = 0
    for _, r9 in rows_1024.iterrows():
        if r9["text"] == "<nonverbal>":
            continue
        r2 = by_text_2048[r9["text"]]
        est_idx = round(r2["start"] * 2048.0)  # recover the native est_idx
        assert r9["start"] == pytest.approx(est_idx / 1024.0)
        assert r9["start"] == pytest.approx(2.0 * r2["start"])
        n_verbal += 1
    assert n_verbal > 0


def test_nonverbal_movie_onset_is_rekeyed_from_words_df() -> None:
    """REGRESSION (P3 nonverbal clock bug, 2026-06-08).

    ``nonverbal_df`` carries only the neural clock (``start == est_idx/SR``, no
    movie column). The P3 Whisper teacher is keyed by ``movie_onset_s``, so
    nonverbal anchors MUST be re-keyed onto the movie clock via the same-session
    ``words_df`` (est_idx → start) map — never emitted with their neural ``start``
    (which would key the teacher hundreds of seconds off-movie)."""
    words = _stub_words_df()
    nonverbal = _stub_nonverbal_df()
    sr = we._neural_sample_rate(2)
    # Precondition: the fixture mirrors real data — nonverbal start IS the neural
    # clock, and nonverbal has no movie-clock column distinct from est_idx/SR.
    assert np.allclose(
        nonverbal["start"].to_numpy(), nonverbal["est_idx"].to_numpy() / sr
    )
    rows = _word_event_rows(
        subject_id=2, trial_id=4, timeline="tl", words_df=words,
        nonverbal_df=nonverbal, tasks=("speech",), binary_tasks=True,
        lite=False, nano=False, random_seed=42, duration=1.0, balance=False,
    )
    xp = words.sort_values("est_idx")["est_idx"].to_numpy(dtype=float)
    fp = words.sort_values("est_idx")["start"].to_numpy(dtype=float)
    nv_rows = rows[rows["text"] == "<nonverbal>"]
    assert len(nv_rows) == len(nonverbal)
    for _, r in nv_rows.iterrows():
        est_idx = r["start"] * sr  # neural-clock onset back to samples
        expected_movie = float(np.interp(est_idx, xp, fp))
        # Re-keyed onto the movie clock (here neural − 200 s by construction)…
        assert r["movie_onset_s"] == pytest.approx(expected_movie)
        # …and decidedly NOT the neural-clock value it would have been pre-fix.
        assert abs(r["movie_onset_s"] - r["start"]) > 100.0


def test_movie_onset_unified_via_trigger_track() -> None:
    """Unified clock map: when the BT trigger track is supplied, BOTH verbal and
    nonverbal anchors key the Whisper teacher via the SAME np.interp(est_idx) over
    the trigger track — not words_df.start (verbal) nor the sparse words map
    (nonverbal). The trigger track is BT's authoritative neural↔movie alignment."""
    words = _stub_words_df()
    nonverbal = _stub_nonverbal_df()
    sr = we._neural_sample_rate(2)
    # Linear trigger map over the whole neural span: movie = neural − 50 s. The
    # −50 (vs the stubs' +200 s est_idx offset) makes the map's verbal answer
    # differ from words_df.start, so a pass proves the trigger track was used.
    trig_index = np.array([0.0, 600.0]) * sr
    trig_movie = np.array([0.0, 600.0]) - 50.0
    rows = _word_event_rows(
        subject_id=2, trial_id=4, timeline="tl", words_df=words,
        nonverbal_df=nonverbal, tasks=("speech",), binary_tasks=True,
        lite=False, nano=False, random_seed=42, duration=1.0, balance=False,
        neural_to_movie=(trig_index, trig_movie),
    )
    for _, r in rows.iterrows():
        est_idx = r["start"] * sr
        assert r["movie_onset_s"] == pytest.approx(est_idx / sr - 50.0)
    # Verbal onsets come from the trigger map (neural−50), NOT words_df.start.
    vb = rows[rows["text"] != "<nonverbal>"]
    for _, r in vb.iterrows():
        src = words[words["est_idx"] == round(r["start"] * sr)].iloc[0]
        assert abs(r["movie_onset_s"] - float(src["start"])) == pytest.approx(150.0)
    # Nonverbal onsets are the movie clock (neural−50), NOT the neural start.
    nv = rows[rows["text"] == "<nonverbal>"]
    for _, r in nv.iterrows():
        assert r["movie_onset_s"] == pytest.approx(r["start"] - 50.0)


def test_movie_onset_freezes_across_pause() -> None:
    """THE pause-bridging regression (2026-06-08, found by the bytes→gradient
    audit). A nonverbal anchor INSIDE a recording pause must key the teacher at the
    FROZEN movie time the trigger track records — not the value a sparse words map
    bridges linearly across the word-free pause gap (was up to 89 s off-movie)."""
    sr = we._neural_sample_rate(2)
    # Trigger track with a pause: movie_time freezes at 100 s while the neural index
    # advances from 100 s to 400 s of neural, then resumes at movie 200 s.
    trig_index = np.array([0.0, 100.0, 400.0, 500.0]) * sr
    trig_movie = np.array([0.0, 100.0, 100.0, 200.0])  # frozen across the gap
    # One nonverbal anchor at neural 250 s — squarely inside the pause.
    nonverbal = pd.DataFrame(
        {"start": [250.0], "end": [251.0], "est_idx": [round(250.0 * sr)]}
    )
    # Words bracket the pause but have NO samples inside it (no speech while paused)
    # — exactly the sparse-map blind spot. Word "b" sits at neural 500 s / movie
    # 200 s (the pause added 300 s of neural time to the same movie content).
    words = pd.DataFrame(
        {
            "start": [100.0, 200.0],
            "end": [101.0, 201.0],
            "est_idx": [round(100.0 * sr), round(500.0 * sr)],
            "original_index": [0, 1],
            "full_word": ["a", "b"],
        }
    )
    rows = _word_event_rows(
        subject_id=2, trial_id=4, timeline="tl", words_df=words,
        nonverbal_df=nonverbal, tasks=("speech",), binary_tasks=True,
        lite=False, nano=False, random_seed=42, duration=1.0, balance=False,
        neural_to_movie=(trig_index, trig_movie),
    )
    nv = rows[rows["text"] == "<nonverbal>"]
    assert len(nv) == 1
    onset = float(nv.iloc[0]["movie_onset_s"])
    # Trigger track freezes at 100 s across the pause — the correct teacher slice.
    assert onset == pytest.approx(100.0, abs=1e-6)
    # The sparse words map would have BRIDGED: interp(neural 250 s) over the two
    # words (neural 100→movie 100, neural 500→movie 200) = 137.5 s — wrong scene.
    bridged = float(
        np.interp(
            250.0 * sr,
            words["est_idx"].to_numpy(float),
            words["start"].to_numpy(float),
        )
    )
    assert bridged == pytest.approx(137.5, abs=0.5)
    assert abs(onset - bridged) > 30.0  # the residual the trigger track removes


def test_load_neural_to_movie_map_strict_and_dedup(tmp_path) -> None:
    """The trigger-track loader: None when no bt_root (laptop tests); raises when
    bt_root is set but the file is missing (loud on a misconfigured BT root); and
    returns a sorted, index-deduplicated map (np.interp needs strictly-increasing
    xp, and pause boundaries duplicate an index)."""
    assert we._load_neural_to_movie_map(2, 4, None) is None
    with pytest.raises(FileNotFoundError):
        we._load_neural_to_movie_map(2, 4, tmp_path)
    d = tmp_path / "subject_timings"
    d.mkdir()
    # A realistic track (>=100 rows, >30 s span) so the LG14 sanity guard passes;
    # we still exercise out-of-order rows + a duplicate-index pause boundary.
    n = 200
    lines = ["type,movie_time,index"]
    for i in range(n):  # index step 1000 @ ~2048 Hz -> ~97 s movie span
        lines.append(f"trigger,{i * 1000 / 2048.0:.6f},{i * 1000}")
    lines[1], lines[3] = lines[3], lines[1]  # scramble two rows: loader must sort
    dup_index = 5000  # duplicate index 5000 -> keep last (the pause row)
    lines.append(f"pause,{(5000 / 2048.0) + 0.05:.6f},{dup_index}")
    (d / "sub_2_trial004_timings.csv").write_text("\n".join(lines) + "\n")
    result = we._load_neural_to_movie_map(2, 4, tmp_path)
    assert result is not None
    idx, mt = result
    assert np.all(np.diff(idx) > 0)  # strictly increasing xp (the real contract)
    assert len(idx) == n  # 201 rows, one duplicate index -> 200 unique (dedup happened)
    # The code keeps one of the two twins for the duplicate index; per its own
    # comment EITHER is valid (they differ by <= one trigger spacing, sub-frame at
    # 8 Hz), and pandas' tie order is not stable — so assert only that bound.
    pos = int(np.where(idx == dup_index)[0][0])
    assert abs(mt[pos] - 5000 / 2048.0) <= 0.06  # within one ~85 ms trigger spacing


def test_word_event_rows_balance_false_keeps_all_anchors_chronological() -> None:
    """SSL (balance=False): every word + every nonverbal anchor is emitted (no
    minority-class bottleneck) in chronological est_idx order, so the positional
    pretrain split is a clean temporal holdout. balance=True (P4) stays the
    minority-class-balanced, interleaved set."""
    n_words, n_nonverbal = 10, 3
    words = pd.DataFrame(
        {
            "start": [float(i) for i in range(n_words)],
            "end": [float(i) + 1.0 for i in range(n_words)],
            "est_idx": [round((i + 200.0) * 2048.0) for i in range(n_words)],
            "original_index": list(range(n_words)),
            "full_word": list("ABCDEFGHIJ"),
        }
    )
    nv_starts = [0.5, 4.5, 8.5]
    nonverbal = pd.DataFrame(
        {
            "start": nv_starts,
            "end": [s + 1.0 for s in nv_starts],
            "est_idx": [round((s + 200.0) * 2048.0) for s in nv_starts],
        }
    )
    balanced = _word_event_rows(
        subject_id=2, trial_id=4, timeline="tl", words_df=words,
        nonverbal_df=nonverbal, tasks=("speech",), binary_tasks=True,
        lite=False, nano=False, random_seed=42, duration=1.0, balance=True,
    )
    assert len(balanced) == 2 * n_nonverbal  # min(10, 3) per class

    unbalanced = _word_event_rows(
        subject_id=2, trial_id=4, timeline="tl", words_df=words,
        nonverbal_df=nonverbal, tasks=("speech",), binary_tasks=True,
        lite=False, nano=False, random_seed=42, duration=1.0, balance=False,
    )
    assert len(unbalanced) == n_words + n_nonverbal  # every anchor kept
    assert (unbalanced["label"] == 1).sum() == n_words
    assert (unbalanced["label"] == 0).sum() == n_nonverbal
    starts = unbalanced["start"].to_numpy()
    assert np.all(starts[:-1] <= starts[1:])  # chronological


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


def test_assign_all_cells_split_keeps_every_row_as_train() -> None:
    # AllCells (materialization-only) keeps every labeled word — no leaderboard
    # split, no dropped subjects/trials — and stamps split="train" (the downstream
    # baseline forms its own splits; all-train fails loud if misused for training).
    df = pd.DataFrame(
        {
            "type": ["Word"] * 4,
            "start": [10.0, 20.0, 30.0, 40.0],
            "duration": [1.0] * 4,
            "task": ["speech"] * 4,
            "label": [0, 1, 0, 1],
            "subject_id": ["1", "2", "3", "4"],
            "trial_id": ["1", "4", "0", "1"],
            "timeline": ["a", "b", "c", "d"],
        }
    )
    out = _assign_all_cells_split(df)
    assert len(out) == 4                                  # nothing dropped
    assert (out["split"] == "train").all()                # every row train
    assert set(zip(out["subject_id"], out["trial_id"])) == {
        ("1", "1"), ("2", "4"), ("3", "0"), ("4", "1")
    }


def test_timeline_is_used_all_cells_gates_on_lite_set() -> None:
    # _timeline_is_used must accept every BT_LITE eval cell and reject anything
    # outside it, so a wider study universe can't leak a non-lite timeline into the
    # (firewalled) lite-eval baseline.
    step = BTWordEvents(tasks=("delta_volume",), eval_mode="AllCells")
    for s, t in BT_LITE_SESSIONS:
        assert step._timeline_is_used(int(s), int(t))
    lite = {tuple(c) for c in BT_LITE_SESSIONS}
    non_lite = next(
        (s, t) for s in range(1, 11) for t in range(6) if (s, t) not in lite
    )
    assert not step._timeline_is_used(*non_lite)


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
    # No real trigger track for the synthetic /dev/null root: exercise the sparse
    # words_df movie-onset fallback (these tests cover splits/schema, not onset).
    monkeypatch.setattr(
        we, "_load_neural_to_movie_map",
        lambda subject_id, trial_id, bt_root: None,
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
    # Balanced eval doesn't consume movie_onset_s; skip the trigger track (the
    # /dev/null root has none) and use the sparse words_df fallback.
    monkeypatch.setattr(
        we, "_load_neural_to_movie_map",
        lambda subject_id, trial_id, bt_root: None,
    )

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
    # No real trigger track for the synthetic /dev/null root: exercise the sparse
    # words_df movie-onset fallback (these tests cover splits/schema, not onset).
    monkeypatch.setattr(
        we, "_load_neural_to_movie_map",
        lambda subject_id, trial_id, bt_root: None,
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
    # No real trigger track for the synthetic /dev/null root: exercise the sparse
    # words_df movie-onset fallback (these tests cover splits/schema, not onset).
    monkeypatch.setattr(
        we, "_load_neural_to_movie_map",
        lambda subject_id, trial_id, bt_root: None,
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
