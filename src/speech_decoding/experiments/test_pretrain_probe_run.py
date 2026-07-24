"""Tests for the checkpoint-selection orchestrator (#11 CPU core)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.experiments.pretrain_probe_driver import SessionTapCache
from speech_decoding.experiments.pretrain_probe_run import select_best_checkpoint
from speech_decoding.experiments.pretrain_probe_suite import (
    DEFAULT_CS_TRAIN_ANCHOR,
    PRETRAIN_UNIVERSE,
)
from speech_decoding.experiments.v2_attentive_train import HeadTrainConfig

N = 60
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
    ws = {0: rows, 1: _split_rows(N)}
    cs = {"val": rows["val"], "test": rows["test"]}
    return SessionTapCache(
        subject_id=s, trial_id=t,
        grids={"M2": torch.from_numpy(m2), "M3": torch.from_numpy(m3),
               "M4": torch.from_numpy(m4)},
        labels={"onset": y},
        parcel_per_electrode=PE, electrode_mask=EM, n_parcels=P,
        parcel_labels=torch.arange(P),
        ws_split={"onset": ws}, cs_split={"onset": cs},
    )


def _caches_for_universe(seed0: int, *, signal: bool) -> dict[tuple[int, int], SessionTapCache]:
    return {
        (s, t): _cache(seed0 + 100 * s + t, s, t, signal=signal)
        for (s, t) in PRETRAIN_UNIVERSE
    }


def _base_cfg() -> HeadTrainConfig:
    return HeadTrainConfig(
        d_model=D, n_heads=6, lr=3e-3, batch_size=64, max_steps=200,
        eval_every=25, patience=8, swad_warmup=50, seed=0,
    )


def test_anchor_is_in_universe():
    assert DEFAULT_CS_TRAIN_ANCHOR in PRETRAIN_UNIVERSE


@pytest.mark.slow
def test_selects_the_signal_checkpoint_by_cross_subject():
    caches_by_ckpt = {
        "strong": _caches_for_universe(0, signal=True),
        "weak": _caches_for_universe(9, signal=False),
    }
    sel = select_best_checkpoint(
        caches_by_ckpt, _base_cfg(),
        representative_ckpt="strong",
        modes=("CrossSubject",), tasks=("onset",),
        hp_grid=None, sweep_cells=None, taps=("M3",),
    )
    # Every CS cell tests a non-anchor subject; the signal ckpt wins on cross-subject.
    assert sel.best_ckpt == "strong"
    assert sel.summary_by_ckpt["strong"].best_cross_subject[2] > \
        sel.summary_by_ckpt["weak"].best_cross_subject[2]
    assert {c.eval_mode for c in sel.cells} == {"CrossSubject"}


def test_unknown_representative_raises():
    caches_by_ckpt = {"a": _caches_for_universe(0, signal=True)}
    with pytest.raises(ValueError, match="representative_ckpt"):
        select_best_checkpoint(
            caches_by_ckpt, _base_cfg(), representative_ckpt="missing",
            modes=("CrossSubject",), tasks=("onset",), taps=("M3",),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
