"""TDD for the online-probe dataset core (labels + selection + assembly).

The load-bearing check is **label parity**: the dense per-window ±1 labels
:func:`pm1_labels` produces must match the upstream ``derive_label_indices`` class
assignment exactly (class 1↔+1, class 0↔−1) — a wrong binarization would make the
whole probe number meaningless. Verified against the REAL upstream function, plus
hand-computed quartile / first-vs-second-word cases, deterministic selection, and
the §6 firewall assert on the in-memory dataset.
"""

from __future__ import annotations

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
