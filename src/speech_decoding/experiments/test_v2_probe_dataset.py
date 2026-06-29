"""Laptop TDD for the v2 probe-dataset assembly (:mod:`v2_probe_dataset`).

Pure pandas/torch — no BT voltage, no neuralset, no DCC. Pins the band-agnostic
assembly the segmenter/materialize path feeds: label derivation from ``words_df``
(reusing the already-TDD'd :func:`pm1_labels`), the ``.bands`` contract
:mod:`v2_raw_probe` consumes, the §6 leakage firewall, and that the built dataset
plugs straight into :func:`v2_raw_probe.run_v2_raw_baseline`.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
import torch

from speech_decoding.experiments.v2_probe_dataset import (
    InMemoryV2ProbeDataset,
    V2SubjectProbeData,
)
from speech_decoding.experiments.v2_raw_probe import run_v2_raw_baseline
from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS


def _words_df(n: int, seed: int) -> pd.DataFrame:
    """A minimal enriched words_df carrying exactly the 3 task feature columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "delta_rms": rng.standard_normal(n),
            "word_length": rng.integers(1, 12, size=n).astype(float),
            "idx_in_sentence": rng.integers(0, 4, size=n).astype(float),
        }
    )


def _rec(sid: int, n: int, c: int, *, session: tuple[int, int], seed: int):
    rng = np.random.default_rng(seed)
    lfs = torch.from_numpy(rng.standard_normal((n, c, 3, 2)).astype(np.float32))
    hga = torch.from_numpy(rng.standard_normal((n, c, 2, 4)).astype(np.float32))
    return {
        "bands": [lfs, hga],
        "parcel_per_electrode": torch.tensor([0, 0, 1, 2][:c]),
        "electrode_mask": torch.ones(c, dtype=torch.bool),
        "words_df": _words_df(n, seed),
        "sessions": [session],
    }


# A non-lite (subject, trial): every subject's trial 99 is firewall-safe.
def _safe(sid: int) -> tuple[int, int]:
    s = (sid, 99)
    assert s not in {tuple(x) for x in BT_LITE_SESSIONS}
    return s


def test_subject_data_exposes_bands_and_derived_labels():
    per = {0: _rec(0, 40, 4, session=_safe(0), seed=0)}
    ds = InMemoryV2ProbeDataset(per, n_parcels=3, ws_subjects=[0], cs_anchor=0, cs_test_subjects=[])
    sd = ds.subject_data(0)
    assert isinstance(sd, V2SubjectProbeData)
    assert len(sd.bands) == 2
    assert sd.bands[0].shape == (40, 4, 3, 2)
    assert sd.bands[1].shape == (40, 4, 2, 4)
    # labels: one ±1/NaN vector per task, length N.
    assert set(sd.labels) == {"delta_volume", "word_length", "word_position"}
    for y in sd.labels.values():
        assert y.shape == (40,)
        assert set(np.unique(y[np.isfinite(y)])).issubset({-1.0, 1.0})
    # delta_volume binarization matches the empirical-CDF quartile split.
    dv = sd.labels["delta_volume"]
    assert np.isfinite(dv).sum() == pytest.approx(40 * 0.5, abs=4)


def test_firewall_blocks_lite_session():
    lite_cell = tuple(BT_LITE_SESSIONS[0])
    per = {int(lite_cell[0]): _rec(int(lite_cell[0]), 10, 3, session=lite_cell, seed=1)}
    with pytest.raises(AssertionError, match="firewall"):
        InMemoryV2ProbeDataset(per, n_parcels=3, ws_subjects=[int(lite_cell[0])],
                               cs_anchor=int(lite_cell[0]), cs_test_subjects=[])


def test_built_dataset_runs_raw_baseline():
    # 3 subjects, all firewall-safe; shared parcels {0,1,2}; plugs into the raw floor.
    per = {s: _rec(s, 50, 4, session=_safe(s), seed=10 + s) for s in (1, 2, 3)}
    ds = InMemoryV2ProbeDataset(
        per, n_parcels=3, ws_subjects=[1, 2, 3], cs_anchor=2, cs_test_subjects=[1, 3]
    )
    # synthetic (3,2)/(2,4) bands don't match the real BANDS_V2 ladder → inject specs
    # for the pooled-token floor (real runs use the BANDS_V2 default).
    ds.band_specs = [
        types.SimpleNamespace(freq_patch_bins=(1, 2), kernel_time=2),
        types.SimpleNamespace(freq_patch_bins=(2,), kernel_time=2),
    ]
    out = run_v2_raw_baseline(ds, max_iter=500)
    for tap in ("raw", "raw_tok"):
        for task in ("delta_volume", "word_length", "word_position"):
            assert f"val_probe/{tap}/ws/{task}" in out
        assert f"val_probe/raw/cs/{task}" in out
    # random features → AUROC near chance (sanity, not a perf claim).
    assert 0.2 < out["val_probe/raw/ws/delta_volume"] < 0.8
