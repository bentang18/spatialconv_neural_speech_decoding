"""Tests for the runner's Stage-2 `--score` wiring (cache load → sweep/eval → CSV).

The heavy orchestration (`select_best_checkpoint`) is covered in test_pretrain_probe_run;
here we lock the script-level glue: `_load_tap_caches` round-trips the `--encode` cache
filenames, and `run_score` writes a ledger with the cross-subject MEAN rows. Kept fast with
a tiny config + 1-combo grid + CS/onset/M3 only."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from speech_decoding.experiments.pretrain_probe_csv import read_results
from speech_decoding.experiments.pretrain_probe_driver import SessionTapCache, save_cache
from speech_decoding.experiments.pretrain_probe_suite import (
    DEFAULT_CS_TRAIN_ANCHOR,
    PRETRAIN_UNIVERSE,
)
from speech_decoding.experiments.pretrain_probe_sweep import HPCombo
from speech_decoding.experiments.v2_attentive_train import HeadTrainConfig

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts" / "neuroprobe" / "run_pretrain_probe_suite.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("_probe_runner", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


N, C, P, K, S, D = 48, 6, 4, 2, 2, 12
PE = torch.tensor([0, 0, 1, 1, 2, 2])
EM = torch.ones(C, dtype=torch.bool)


def _split_rows(n: int) -> dict[str, np.ndarray]:
    cut = n // 2
    held = np.arange(cut, n)
    vcut = len(held) // 2
    return {"train": np.arange(cut), "val": held[:vcut], "test": held[vcut:]}


def _cache(seed: int, s: int, t: int, *, signal: bool) -> SessionTapCache:
    rng = np.random.default_rng(seed)
    y = rng.choice([0.0, 1.0], size=N)
    sgn = (y * 2 - 1).astype(np.float32)
    m2 = rng.normal(size=(N, C, S, D)).astype(np.float32)
    m3 = rng.normal(size=(N, P, K, S, D)).astype(np.float32)
    m4 = rng.normal(size=(N, P, K, S, D)).astype(np.float32)
    if signal:
        m3[:, 1, 0, 0, 0] = sgn * 5
        m4[:, 1, 0, 0, 0] = sgn * 5
    rows = _split_rows(N)
    return SessionTapCache(
        subject_id=s, trial_id=t,
        grids={"M2": torch.from_numpy(m2), "M3": torch.from_numpy(m3),
               "M4": torch.from_numpy(m4)},
        labels={"onset": y},
        parcel_per_electrode=PE, electrode_mask=EM, n_parcels=P,
        parcel_labels=torch.arange(P),
        ws_split={"onset": {0: rows, 1: _split_rows(N)}},
        cs_split={"onset": {"val": rows["val"], "test": rows["test"]}},
    )


def _write_caches(tmp: Path) -> None:
    for (s, t) in PRETRAIN_UNIVERSE:
        save_cache(_cache(100 * s + t, s, t, signal=True), str(tmp / f"taps_s{s}_t{t}.pt"))


def test_load_tap_caches_round_trips_keys(tmp_path):
    _write_caches(tmp_path)
    (tmp_path / "not_a_cache.pt").write_bytes(b"x")     # ignored: wrong name
    caches = _runner()._load_tap_caches(str(tmp_path))
    assert set(caches) == set(PRETRAIN_UNIVERSE)
    assert caches[DEFAULT_CS_TRAIN_ANCHOR].grids["M3"].shape[1] == P


def test_load_tap_caches_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no taps"):
        _runner()._load_tap_caches(str(tmp_path))


def test_run_score_writes_cross_subject_mean_rows(tmp_path):
    _write_caches(tmp_path)
    out = tmp_path / "scores.csv"
    cfg = HeadTrainConfig(d_model=D, n_heads=6, lr=3e-3, batch_size=64, max_steps=60,
                          eval_every=20, patience=4, swad_warmup=20, seed=0)
    _runner().run_score(
        str(tmp_path), ckpt_tag="60k", anchor=DEFAULT_CS_TRAIN_ANCHOR, out_path=str(out),
        base_cfg=cfg, hp_grid=[HPCombo(0.1, 0.0, 0.1)],
        modes=("CrossSubject",), tasks=("onset",), taps=("M3",),
    )
    rows = read_results(str(out))
    mean_cs = [r for r in rows if r["task"] == "MEAN" and r["eval_mode"] == "CrossSubject"]
    assert mean_cs, "expected a CrossSubject MEAN aggregate row"
    assert {r["readout"] for r in mean_cs} == {"ridge", "attentive"}
    assert all(0.0 <= float(r["auroc"]) <= 1.0 for r in mean_cs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
