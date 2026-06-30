"""TDD for the online-probe dataset core (labels + selection + assembly).

The load-bearing check is **label parity**: the dense per-window ±1 labels
:func:`pm1_labels` produces must match the upstream ``derive_label_indices`` class
assignment exactly (class 1↔+1, class 0↔−1) — a wrong binarization would make the
whole probe number meaningless. Verified against the REAL upstream function, plus
hand-computed quartile / first-vs-second-word cases, deterministic selection, and
the §6 firewall assert on the in-memory dataset.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from speech_decoding.experiments import online_probe as op
from speech_decoding.experiments import online_probe_dataset as opd
from speech_decoding.studies.braintreebank.labels import derive_label_indices
from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS


# ------------------------------------------------------- label parity (the crux)
def test_pm1_continuous_parity_with_upstream_derive() -> None:
    """``pm1_labels`` ±1 set must equal upstream ``derive_label_indices`` classes
    (balance=False, no cap → every class index, sorted). delta_volume → delta_rms."""
    rng = np.random.default_rng(0)
    wdf = pd.DataFrame({"delta_rms": rng.standard_normal(200)})
    mine = opd.pm1_labels(wdf, "delta_volume")
    ref = derive_label_indices(
        words_df=wdf, nonverbal_df=pd.DataFrame(), task="delta_volume",
        binary_tasks=True, balance=False, lite=False,
    )
    assert np.array_equal(np.where(mine == 1.0)[0], np.sort(ref[1]))
    assert np.array_equal(np.where(mine == -1.0)[0], np.sort(ref[0]))
    # the middle 50% is NaN (dropped from the fit), and +/-1 are ~the quartiles
    assert np.isnan(mine).sum() > 0
    assert abs((mine == 1.0).sum() - 50) <= 3 and abs((mine == -1.0).sum() - 50) <= 3


def test_pm1_word_length_parity_with_upstream() -> None:
    rng = np.random.default_rng(1)
    wdf = pd.DataFrame({"word_length": rng.integers(1, 12, 300).astype(float)})
    mine = opd.pm1_labels(wdf, "word_length")
    ref = derive_label_indices(
        words_df=wdf, nonverbal_df=pd.DataFrame(), task="word_length",
        binary_tasks=True, balance=False, lite=False,
    )
    assert np.array_equal(np.where(mine == 1.0)[0], np.sort(ref[1]))
    assert np.array_equal(np.where(mine == -1.0)[0], np.sort(ref[0]))


def test_pm1_word_position_maps_to_word_index_parity() -> None:
    """``word_position`` is the spec name for upstream ``word_index`` (first vs
    second word in sentence): idx 0 → +1, idx 1 → −1, else NaN."""
    idx = np.array([0, 1, 2, 0, 1, 3, 0, 1], dtype=float)
    wdf = pd.DataFrame({"idx_in_sentence": idx})
    mine = opd.pm1_labels(wdf, "word_position")
    assert np.where(mine == 1.0)[0].tolist() == [0, 3, 6]
    assert np.where(mine == -1.0)[0].tolist() == [1, 4, 7]
    assert np.isnan(mine[[2, 5]]).all()
    ref = derive_label_indices(
        words_df=wdf, nonverbal_df=pd.DataFrame(), task="word_index",
        binary_tasks=True, balance=False, lite=False,
    )
    assert np.array_equal(np.where(mine == 1.0)[0], np.sort(ref[1]))
    assert np.array_equal(np.where(mine == -1.0)[0], np.sort(ref[0]))


def test_pm1_continuous_hand_computed_quartiles() -> None:
    """Explicit values: 0..99 uniform → pct = i/100; +1 above .75 (i>75), −1 below
    .25 (i<25); the rest NaN."""
    wdf = pd.DataFrame({"delta_rms": np.arange(100, dtype=float)})
    mine = opd.pm1_labels(wdf, "delta_volume")
    assert (mine[76:] == 1.0).all() and np.isnan(mine[75])
    assert (mine[:25] == -1.0).all() and np.isnan(mine[25])


def test_pm1_raises_on_nan_feature() -> None:
    wdf = pd.DataFrame({"delta_rms": [1.0, np.nan, 3.0]})
    with pytest.raises(ValueError, match="non-finite"):
        opd.pm1_labels(wdf, "delta_volume")


def test_pm1_word_part_speech_parity_with_upstream() -> None:
    """High-level depth-test categorical: +1 VERB / −1 NOUN, other POS → NaN."""
    pos = np.array(["VERB", "NOUN", "ADJ", "VERB", "DET", "NOUN", "PRON", "VERB"])
    wdf = pd.DataFrame({"pos": pos})
    mine = opd.pm1_labels(wdf, "word_part_speech")
    assert np.where(mine == 1.0)[0].tolist() == [0, 3, 7]
    assert np.where(mine == -1.0)[0].tolist() == [1, 5]
    assert np.isnan(mine[[2, 4, 6]]).all()
    ref = derive_label_indices(
        words_df=wdf, nonverbal_df=pd.DataFrame(), task="word_part_speech",
        binary_tasks=True, balance=False, lite=False,
    )
    assert np.array_equal(np.where(mine == 1.0)[0], np.sort(ref[1]))
    assert np.array_equal(np.where(mine == -1.0)[0], np.sort(ref[0]))


def test_pm1_word_head_pos_parity_with_upstream() -> None:
    """+1 bin_head==0 / −1 bin_head==1, other values → NaN."""
    wdf = pd.DataFrame({"bin_head": np.array([0, 1, 2, 0, 1, 0], dtype=float)})
    mine = opd.pm1_labels(wdf, "word_head_pos")
    assert np.where(mine == 1.0)[0].tolist() == [0, 3, 5]
    assert np.where(mine == -1.0)[0].tolist() == [1, 4]
    assert np.isnan(mine[2])
    ref = derive_label_indices(
        words_df=wdf, nonverbal_df=pd.DataFrame(), task="word_head_pos",
        binary_tasks=True, balance=False, lite=False,
    )
    assert np.array_equal(np.where(mine == 1.0)[0], np.sort(ref[1]))
    assert np.array_equal(np.where(mine == -1.0)[0], np.sort(ref[0]))


def test_pm1_face_num_parity_with_upstream() -> None:
    """Visual control: +1 any face (>0) / −1 none (==0)."""
    wdf = pd.DataFrame({"face_num": np.array([0, 1, 3, 0, 2, 0], dtype=float)})
    mine = opd.pm1_labels(wdf, "face_num")
    assert np.where(mine == 1.0)[0].tolist() == [1, 2, 4]
    assert np.where(mine == -1.0)[0].tolist() == [0, 3, 5]
    assert not np.isnan(mine).any()                       # binary face_num labels every row
    ref = derive_label_indices(
        words_df=wdf, nonverbal_df=pd.DataFrame(), task="face_num",
        binary_tasks=True, balance=False, lite=False,
    )
    assert np.array_equal(np.where(mine == 1.0)[0], np.sort(ref[1]))
    assert np.array_equal(np.where(mine == -1.0)[0], np.sort(ref[0]))


def test_pm1_categorical_int_raises_on_nan() -> None:
    """The integer-categorical NaN-guard (LG14) fires on a join-miss for face_num."""
    wdf = pd.DataFrame({"face_num": [0.0, np.nan, 2.0]})
    with pytest.raises(ValueError, match="non-finite"):
        opd.pm1_labels(wdf, "face_num")


# ------------------------------------------------------------------- selection
def test_select_window_indices_all_when_under_cap() -> None:
    assert np.array_equal(opd.select_window_indices(100, 3500, seed=0), np.arange(100))


def test_select_window_indices_seeded_sorted_subset() -> None:
    a = opd.select_window_indices(10_000, 3500, seed=7)
    b = opd.select_window_indices(10_000, 3500, seed=7)
    c = opd.select_window_indices(10_000, 3500, seed=8)
    assert a.shape == (3500,) and np.array_equal(a, np.sort(a))       # sorted (time order)
    assert np.array_equal(a, b)                                       # deterministic
    assert not np.array_equal(a, c)                                   # seed-sensitive
    assert len(np.unique(a)) == 3500                                  # no repeats


# ----------------------------------------------- bad-window (CLIP) respect at 1 s width
def test_probe_clip_contract_is_zero_lag_one_second() -> None:
    """Ben's Phase-2b answer: the probe forwards 1 s word-onset clips with NO
    neural lag — so the bad-window overlap test uses start=0.0 / dur=1.0."""
    assert opd.PROBE_CLIP_START_S == 0.0
    assert opd.PROBE_CLIP_DUR_S == 1.0


def _events(starts, *, subject=2, trial=1) -> pd.DataFrame:
    return pd.DataFrame({
        "type": ["Word"] * len(starts),
        "start": list(starts),
        "subject_id": [subject] * len(starts),
        "trial_id": [trial] * len(starts),
    })


def test_filter_probe_events_uses_one_second_clip_width(tmp_path) -> None:
    """The probe must respect the CLIP sidecars at its OWN 1 s clip width (not the
    5 s SSL width). A word at start=9.2 s has a 1 s clip [9.2,10.2) that clears a
    bad span [10.5,10.8) (kept), but its 5 s clip [9.2,14.2) would overlap — so this
    row is the 1 s-vs-5 s discriminator. start=10.0 ([10.0,11.0)) overlaps → dropped;
    start=8.0 ([8.0,9.0)) clears → kept."""
    (tmp_path / "btbank2_t1.json").write_text(json.dumps(
        {"session": "btbank2_t1", "bad_windows_s": [[10.5, 10.8]]}
    ))
    ev = _events([8.0, 9.2, 10.0])
    out = opd.filter_probe_events(ev, str(tmp_path))
    assert out["start"].tolist() == [8.0, 9.2]          # 10.0 dropped, 9.2 kept (1 s width)


def test_filter_probe_events_noop_when_no_dir() -> None:
    """No CLIP dir (run without the bad-window layer) → keep every window."""
    ev = _events([1.0, 2.0, 3.0])
    out = opd.filter_probe_events(ev, None)
    assert out["start"].tolist() == [1.0, 2.0, 3.0]


def test_filter_probe_events_other_session_untouched(tmp_path) -> None:
    """A sidecar for one session never drops another session's clips."""
    (tmp_path / "btbank2_t1.json").write_text(json.dumps(
        {"session": "btbank2_t1", "bad_windows_s": [[10.5, 10.8]]}
    ))
    ev = _events([10.0], subject=3, trial=2)            # different session, same time
    out = opd.filter_probe_events(ev, str(tmp_path))
    assert out["start"].tolist() == [10.0]              # kept (no sidecar for btbank3_t2)


# ------------------------------------------- feature re-attach (est_idx neural-clock join)
def test_probe_feature_columns_derived_from_tasks() -> None:
    """The columns pm1_labels needs, deduped: delta_volume→delta_rms, word_length,
    word_position→idx_in_sentence."""
    assert opd.probe_feature_columns() == ("delta_rms", "word_length", "idx_in_sentence")


def _enriched(est_idx, *, dr=None, wl=None, iis=None) -> pd.DataFrame:
    n = len(est_idx)
    rng = np.random.default_rng(int(est_idx[0]))
    return pd.DataFrame({
        "est_idx": np.asarray(est_idx, dtype=np.int64),
        "delta_rms": rng.standard_normal(n) if dr is None else dr,
        "word_length": (rng.integers(1, 10, n).astype(float) if wl is None else wl),
        "idx_in_sentence": (rng.integers(0, 3, n).astype(float) if iis is None else iis),
        "other_col": rng.standard_normal(n),   # extra features.csv column, ignored
    })


def test_attach_probe_features_joins_on_recovered_est_idx() -> None:
    """start = est_idx/native_rate (2048 Hz) round-trips to est_idx; the join pulls
    the right feature row per window, row order preserved."""
    est = np.array([100, 5000, 73, 99999], dtype=np.int64)
    ew = _enriched(est, dr=np.array([1., 2., 3., 4.]),
                   wl=np.array([5., 6., 7., 8.]), iis=np.array([0., 1., 2., 0.]))
    # windows in a DIFFERENT order than the enriched df, to prove the per-row join
    order = [2, 0, 3, 1]
    windows = pd.DataFrame({
        "type": "Word",
        "start": est[order] / 2048.0,
        "subject_id": ["2"] * 4,        # string ids (real study dtype)
        "trial_id": [1] * 4,
    })
    out = opd.attach_probe_features(windows, {(2, 1): ew})
    assert out["delta_rms"].tolist() == [3., 1., 4., 2.]
    assert out["word_length"].tolist() == [7., 5., 8., 6.]
    assert out["idx_in_sentence"].tolist() == [2., 0., 0., 1.]
    # the attached words_df labels correctly through pm1_labels
    assert "other_col" not in out.columns


def test_attach_probe_features_s9_half_rate() -> None:
    """S9 est_idx is on the 1024 Hz native clock; recovery must use 1024, not 2048."""
    est = np.array([1024, 2048, 3072], dtype=np.int64)
    ew = _enriched(est, dr=np.array([10., 20., 30.]))
    windows = pd.DataFrame({
        "start": est / 1024.0, "subject_id": [9, 9, 9], "trial_id": [0, 0, 0],
    })
    out = opd.attach_probe_features(windows, {(9, 0): ew})
    assert out["delta_rms"].tolist() == [10., 20., 30.]


def test_attach_probe_features_multi_session() -> None:
    """A subject spanning two trials joins each window against its own session's df."""
    ew1 = _enriched(np.array([100, 200]), dr=np.array([1., 2.]))
    ew2 = _enriched(np.array([100, 300]), dr=np.array([7., 9.]))   # est 100 reused across trials
    windows = pd.DataFrame({
        "start": np.array([100, 100, 300]) / 2048.0,
        "subject_id": [2, 2, 2], "trial_id": [1, 2, 2],
    })
    out = opd.attach_probe_features(windows, {(2, 1): ew1, (2, 2): ew2})
    assert out["delta_rms"].tolist() == [1., 7., 9.]               # (2,1):100 vs (2,2):100 differ


def test_attach_probe_features_fails_loud_on_missing_est_idx() -> None:
    """A window whose est_idx isn't in the enriched df = broken join → raise, never
    a silent NaN label."""
    ew = _enriched(np.array([100, 200]))
    windows = pd.DataFrame({
        "start": np.array([100, 999]) / 2048.0,
        "subject_id": [2, 2], "trial_id": [1, 1],
    })
    with pytest.raises(KeyError, match="no matching"):
        opd.attach_probe_features(windows, {(2, 1): ew})


def test_attach_probe_features_fails_loud_on_clock_mismatch() -> None:
    """A start that doesn't round-trip to an integer est_idx (wrong clock) → raise."""
    ew = _enriched(np.array([100, 200]))
    windows = pd.DataFrame({
        "start": [100.4 / 2048.0, 200 / 2048.0],   # 100.4 not integral
        "subject_id": [2, 2], "trial_id": [1, 1],
    })
    with pytest.raises(ValueError, match="neural-clock mismatch"):
        opd.attach_probe_features(windows, {(2, 1): ew})


def test_attach_probe_features_fails_loud_on_missing_session() -> None:
    windows = pd.DataFrame({
        "start": [100 / 2048.0], "subject_id": [2], "trial_id": [1],
    })
    with pytest.raises(KeyError, match="no enriched words_df"):
        opd.attach_probe_features(windows, {})


def test_attach_probe_features_fails_loud_on_missing_feature_column() -> None:
    ew = pd.DataFrame({"est_idx": [100], "delta_rms": [1.0]})   # no word_length/idx
    windows = pd.DataFrame({
        "start": [100 / 2048.0], "subject_id": [2], "trial_id": [1],
    })
    with pytest.raises(KeyError, match="missing probe feature columns"):
        opd.attach_probe_features(windows, {(2, 1): ew})


def test_attach_probe_features_then_pm1_labels_end_to_end() -> None:
    """The re-attached words_df feeds pm1_labels with no KeyError — the whole point
    of #241: forwarded windows become labellable."""
    est = np.arange(100, 100 + 200, dtype=np.int64)
    rng = np.random.default_rng(3)
    ew = _enriched(est, dr=rng.standard_normal(200))
    windows = pd.DataFrame({
        "start": est / 2048.0, "subject_id": ["2"] * 200, "trial_id": [1] * 200,
    })
    wdf = opd.attach_probe_features(windows, {(2, 1): ew})
    for task in opd.PROBE_TASKS:
        lab = opd.pm1_labels(wdf, task)
        assert lab.shape == (200,)
        assert set(np.unique(lab[np.isfinite(lab)])) <= {-1.0, 1.0}


# ------------------------------------------------- per-subject window selection (n_cap)
def test_select_subject_window_positions_filters_and_caps() -> None:
    """Positions are that subject's rows only, deterministically capped, in time
    order (sorted positional indices preserved for the contiguous WS fold)."""
    triggers = pd.DataFrame({"subject_id": [2, 3, 2, 2, 3, 2]})
    pos = opd.select_subject_window_positions(triggers, 2, n_cap=3500, seed=0)
    assert pos.tolist() == [0, 2, 3, 5]                 # exactly subject-2 rows, in order


def test_select_subject_window_positions_caps_deterministically() -> None:
    triggers = pd.DataFrame({"subject_id": [2] * 1000})
    a = opd.select_subject_window_positions(triggers, 2, n_cap=100, seed=5)
    b = opd.select_subject_window_positions(triggers, 2, n_cap=100, seed=5)
    assert a.shape == (100,) and np.array_equal(a, np.sort(a))
    assert np.array_equal(a, b)                          # deterministic in seed
    assert set(a.tolist()) <= set(range(1000))


def test_select_subject_window_positions_absent_subject_empty() -> None:
    triggers = pd.DataFrame({"subject_id": [2, 2, 3]})
    assert opd.select_subject_window_positions(triggers, 7, n_cap=10, seed=0).tolist() == []


def test_select_subject_window_positions_string_subject_ids() -> None:
    """The real BT study emits ``subject_id`` as strings ('2'); the int cohort
    constants must still match. Regression for the DCC #241 anchor-missing crash:
    int(2) vs str('2') silently selected zero windows."""
    triggers = pd.DataFrame({"subject_id": ["2", "3", "2", "2", "3", "2"]})
    pos = opd.select_subject_window_positions(triggers, 2, n_cap=3500, seed=0)
    assert pos.tolist() == [0, 2, 3, 5]                 # same as the int fixture


def test_present_subjects_dtype_robust_to_string_ids() -> None:
    """``present_subject_ids`` coerces to int so the int cohort set (CS_ANCHOR=2,
    …) intersects a string-typed ``subject_id`` column instead of yielding ∅."""
    triggers = pd.DataFrame({"subject_id": ["1", "2", "2", "9"]})
    present = opd.present_subject_ids(triggers)
    assert present == {1, 2, 9}
    assert {opd.CS_ANCHOR} & present == {2}             # the failing intersection, now non-empty


# ------------------------------------------------------- in-memory dataset + firewall
def _subject_record(sid: int, sessions, nwin=40, c=3, n_parcels=4):
    rng = np.random.default_rng(sid)
    wdf = pd.DataFrame({
        "delta_rms": rng.standard_normal(nwin),
        "word_length": rng.integers(1, 10, nwin).astype(float),
        "idx_in_sentence": rng.integers(0, 3, nwin).astype(float),
    })
    return {
        "slow": torch.zeros(nwin, c, 2, 6, 5),
        "beta": torch.zeros(nwin, c, 6, 17),
        "hg": torch.zeros(nwin, c, 9, 33),
        "parcel_per_electrode": torch.tensor([0, 1, 2])[:c],
        "electrode_mask": torch.ones(c, dtype=torch.bool),
        "words_df": wdf,
        "sessions": sessions,
    }


def test_in_memory_dataset_assembles_and_labels() -> None:
    per = {2: _subject_record(2, [(2, 1), (2, 2)]), 3: _subject_record(3, [(3, 2)])}
    ds = opd.InMemoryProbeDataset(per, n_parcels=4, ws_subjects=[2, 3],
                                  cs_anchor=2, cs_test_subjects=[3])
    sd = ds.subject_data(2)
    assert set(sd.labels) == set(opd.PROBE_TASKS)
    assert sd.labels["delta_volume"].shape == (40,)
    assert sd.slow.shape[0] == 40
    assert ds.n_parcels == 4 and ds.cs_anchor == 2


def test_in_memory_dataset_firewall_blocks_lite_cell() -> None:
    """A probe window from a lite eval cell ((2,0) ∈ BT_LITE_SESSIONS) must hard-fail
    at build (spec §6)."""
    assert (2, 0) in {tuple(s) for s in BT_LITE_SESSIONS}
    per = {2: _subject_record(2, [(2, 1), (2, 0)])}
    with pytest.raises(AssertionError, match="firewall"):
        opd.InMemoryProbeDataset(per, n_parcels=4, ws_subjects=[2],
                                 cs_anchor=2, cs_test_subjects=[])


def test_pretrain_sessions_clean_firewall() -> None:
    """The real S2 pretrain sessions are firewall-clean against the lite set."""
    per = {2: _subject_record(2, [(2, 1), (2, 2), (2, 3), (2, 5), (2, 6)])}
    opd.InMemoryProbeDataset(per, n_parcels=4, ws_subjects=[2],
                             cs_anchor=2, cs_test_subjects=[])  # no raise


# ----------------------------------------------------- end-to-end with run_probe
class _FakeTok:
    def __init__(self):
        from speech_decoding.models.v14_converged import token_metadata
        self.band_id, self.freq_global_id, self.time_slot = token_metadata()


class _FakeFront:
    def __init__(self):
        self.tokenizer = _FakeTok()


class _FakeModel:
    def __init__(self):
        self.student_frontend = _FakeFront()
        self.training = True
        self.n_tokens = self.student_frontend.tokenizer.time_slot.numel()

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def encode_frontend(self, slow, beta, hg):
        b, c = slow.shape[0], slow.shape[1]
        return 0.01 * torch.randn(b, c, self.n_tokens, 4)

    def encode_latent(self, slow, beta, hg, ppe, *, electrode_mask=None):
        b, c = slow.shape[0], slow.shape[1]
        sig = slow[:, :, 0, 0, 0]
        feats = 0.01 * torch.randn(b, c, self.n_tokens, 4)
        feats[..., 0] += sig[:, :, None]
        return feats


def test_run_probe_with_in_memory_dataset_masks_nan_labels() -> None:
    """Smuggle each window's delta_volume ±1 label into slow[:,:,0,0,0]; the latent
    tap decodes it. Confirms run_probe handles NaN-labelled (middle-50%) windows and
    still emits finite WS/CS/gap for the covered windows."""
    # firewall-clean pretrain sessions per subject ((3,1) is a lite eval cell).
    clean_session = {2: (2, 1), 3: (3, 2)}
    per = {}
    for sid in (2, 3):
        rec = _subject_record(sid, [clean_session[sid]], nwin=80)
        # make delta_volume cleanly separable: slow signal = the ±1 label itself
        lab = opd.pm1_labels(rec["words_df"], "delta_volume")
        sig = np.nan_to_num(lab, nan=0.0)
        rec["slow"][:, :, 0, 0, 0] = torch.from_numpy(sig).float()[:, None]
        per[sid] = rec
    ds = opd.InMemoryProbeDataset(per, n_parcels=4, ws_subjects=[2, 3],
                                  cs_anchor=2, cs_test_subjects=[3])
    m = _FakeModel().eval()
    metrics = op.run_probe(m, ds, k_list=(2, 1))
    dv = metrics["val_probe/latent/ws/delta_volume"]
    assert np.isfinite(dv) and dv > 0.9                       # covered windows decode
    assert np.isfinite(metrics["val_probe/latent/gap/delta_volume"])


# ----------------------------------------- #241: nonverbal-anchor exclusion (segmenter)


def _fake_run_data_with_segmenter_keys():
    import types

    extractors = {k: object() for k in opd._PROBE_SEGMENTER_KEYS}
    return types.SimpleNamespace(
        segmenter=types.SimpleNamespace(extractors=extractors)
    )


def _captured_trigger_query(monkeypatch) -> str:
    """The ``trigger_query`` ``_probe_segmenter`` passes to ``ns.dataloader.Segmenter``.
    Patching the Segmenter avoids constructing real (pydantic-validated) extractors —
    the fix under test is the query string, not the segmenter object."""
    import neuralset as ns

    captured: dict[str, object] = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(ns.dataloader, "Segmenter", _capture)
    opd._probe_segmenter(_fake_run_data_with_segmenter_keys())
    return captured["trigger_query"]  # type: ignore[return-value]


def test_probe_segmenter_trigger_query_excludes_nonverbal(monkeypatch) -> None:
    """#241: the probe segmenter must drop nonverbal (speech class-0) anchors. The SSL
    word_events transform emits those as ``type=='Word'`` / ``text=='<nonverbal>'`` with
    an est_idx from nonverbal_df; forwarding them tripped the attach_probe_features
    est_idx join and self-disabled the probe on every ``--task speech`` run."""
    q = _captured_trigger_query(monkeypatch)
    assert "type == 'Word'" in q
    assert "text != '<nonverbal>'" in q


def test_probe_segmenter_trigger_query_semantics_drop_nonverbal(monkeypatch) -> None:
    """The query string actually filters the rows (not merely contains the clause):
    apply the real built query to a mixed events frame and confirm only verbal Word
    rows survive — nonverbal-as-Word dropped, non-Word types dropped."""
    q = _captured_trigger_query(monkeypatch)
    events = pd.DataFrame(
        {
            "type": ["Word", "Word", "Word", "Sentence"],
            "text": ["cat", "<nonverbal>", "dog", "<nonverbal>"],
        }
    )
    kept = events.query(q)
    assert kept["text"].tolist() == ["cat", "dog"]


def test_probe_segmenter_missing_keys_raises() -> None:
    """Loud-fail when the run's data is not the 3STFT segmenter (the probe only
    supports ``--frontend 3stft``)."""
    import types

    bad = types.SimpleNamespace(
        segmenter=types.SimpleNamespace(extractors={"support": object()})
    )
    with pytest.raises(KeyError):
        opd._probe_segmenter(bad)


def test_cartesian_slow_to_5d_row_major_split() -> None:
    # The cache delivers slow as (B, C, 2F, T) = [Re(F) ++ Im(F)] on the freq axis;
    # the materializer must split it to (B, C, 2, F, T) with channel 0 = Re, 1 = Im
    # (row-major), byte-identical to the live v14_converged_module._ingest reshape.
    B, C, F, T = 2, 3, 6, 5
    re = torch.arange(B * C * F * T, dtype=torch.float32).reshape(B, C, F, T)
    im = -re - 1.0
    cat = torch.cat([re, im], dim=2)                  # (B, C, 12, T)
    out = opd.cartesian_slow_to_5d(cat)
    assert tuple(out.shape) == (B, C, 2, F, T)
    assert torch.equal(out[:, :, 0], re)              # channel 0 = Re(F)
    assert torch.equal(out[:, :, 1], im)              # channel 1 = Im(F)


def test_cartesian_slow_to_5d_fails_loud_on_bad_shape() -> None:
    with pytest.raises(ValueError, match="even freq axis"):
        opd.cartesian_slow_to_5d(torch.zeros(2, 3, 7, 5))   # odd freq axis
    with pytest.raises(ValueError, match="2F"):
        opd.cartesian_slow_to_5d(torch.zeros(2, 3, 2, 6, 5))  # already 5-D
