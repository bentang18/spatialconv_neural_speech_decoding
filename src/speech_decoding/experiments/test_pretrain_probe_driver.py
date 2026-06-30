"""Tests for the Stage-2 driver (cell → cache grids → readout dispatch)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.experiments.pretrain_probe_driver import (
    SessionTapCache,
    load_cache,
    run_attentive_cell,
    run_linear_cell,
    save_cache,
)
from speech_decoding.experiments.pretrain_probe_suite import ProbeCell
from speech_decoding.experiments.v2_attentive_train import HeadTrainConfig

N = 120
C = 6
P = 4
K = 2
S = 2
D = 12
PE = torch.tensor([0, 0, 1, 1, 2, 2])
EM = torch.ones(C, dtype=torch.bool)


def _split_rows(n: int) -> dict[str, np.ndarray]:
    cut = n // 2
    held = np.arange(cut, n)
    vcut = len(held) // 2
    return {"train": np.arange(cut), "val": held[:vcut], "test": held[vcut:]}


def _make_cache(seed: int, subject_id: int, trial_id: int) -> SessionTapCache:
    rng = np.random.default_rng(seed)
    y = rng.choice([0.0, 1.0], size=N)
    sgn = (y * 2 - 1).astype(np.float32)
    m2 = rng.normal(size=(N, C, S, D)).astype(np.float32)
    m2[:, 0, 0, 0] = sgn * 4 + 0.2 * rng.normal(size=N)          # signal in electrode 0
    m3 = rng.normal(size=(N, P, K, S, D)).astype(np.float32)
    m3[:, 1, 0, 0, 0] = sgn * 4 + 0.2 * rng.normal(size=N)       # signal in parcel 1
    m4 = rng.normal(size=(N, P, K, S, D)).astype(np.float32)
    m4[:, 1, 0, 0, 0] = sgn * 4 + 0.2 * rng.normal(size=N)
    rows = _split_rows(N)
    ws = {0: rows, 1: _split_rows(N)}
    cs = {"val": rows["val"], "test": rows["test"]}
    return SessionTapCache(
        subject_id=subject_id, trial_id=trial_id,
        grids={"M2": torch.from_numpy(m2), "M3": torch.from_numpy(m3),
               "M4": torch.from_numpy(m4)},
        labels={"onset": y, "speech": y},
        parcel_per_electrode=PE, electrode_mask=EM, n_parcels=P,
        ws_split={"onset": ws, "speech": ws},
        cs_split={"onset": cs, "speech": cs},
    )


def _cfg() -> HeadTrainConfig:
    return HeadTrainConfig(
        d_model=D, n_heads=6, parcel_dropout=0.2, lr=3e-3, batch_size=64,
        max_steps=400, eval_every=25, patience=8, swad_warmup=50, seed=0,
    )


@pytest.mark.parametrize("tap", ["M2", "M3", "M4"])
def test_linear_ws_dispatch_recovers_signal(tap):
    cache = _make_cache(0, subject_id=1, trial_id=0)
    cell = ProbeCell("WithinSession", "onset", 1, 0, fold_index=0)
    assert run_linear_cell(cell, tap, test_cache=cache) > 0.85


@pytest.mark.parametrize("tap", ["M2", "M3", "M4"])
def test_linear_cs_dispatch_recovers_signal(tap):
    anchor = _make_cache(1, subject_id=2, trial_id=1)
    test = _make_cache(2, subject_id=3, trial_id=2)
    cell = ProbeCell("CrossSubject", "onset", 3, 2, train_subject_id=2, train_trial_id=1)
    a = run_linear_cell(cell, tap, test_cache=test, anchor_cache=anchor)
    assert a > 0.8


def test_attentive_ws_and_cs_dispatch():
    cache = _make_cache(0, 1, 0)
    anchor = _make_cache(1, 2, 1)
    test = _make_cache(2, 3, 2)
    ws = ProbeCell("WithinSession", "onset", 1, 0, fold_index=0)
    cs = ProbeCell("CrossSubject", "onset", 3, 2, train_subject_id=2, train_trial_id=1)
    assert run_attentive_cell(ws, "M3", _cfg(), test_cache=cache) > 0.75
    assert run_attentive_cell(cs, "M2", _cfg(), test_cache=test, anchor_cache=anchor) > 0.7


def test_cs_without_anchor_raises():
    test = _make_cache(2, 3, 2)
    cs = ProbeCell("CrossSubject", "onset", 3, 2, train_subject_id=2, train_trial_id=1)
    with pytest.raises(ValueError, match="anchor"):
        run_linear_cell(cs, "M3", test_cache=test)


def test_save_load_roundtrip(tmp_path):
    cache = _make_cache(0, 1, 0)
    path = str(tmp_path / "cache.pt")
    save_cache(cache, path)
    loaded = load_cache(path)
    cell = ProbeCell("WithinSession", "onset", 1, 0, fold_index=0)
    a_orig = run_linear_cell(cell, "M3", test_cache=cache)
    a_load = run_linear_cell(cell, "M3", test_cache=loaded)
    assert a_orig == pytest.approx(a_load)
    assert loaded.subject_id == 1 and loaded.n_parcels == P


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
