"""Tests for the sweep-once HP harness + fixed-HP eval + collector + firewall self-test."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.experiments.pretrain_probe_collect import (
    aggregate_scores,
    firewall_self_test,
)
from speech_decoding.experiments.pretrain_probe_driver import SessionTapCache
from speech_decoding.experiments.pretrain_probe_suite import ProbeCell, enumerate_cells
from speech_decoding.experiments.pretrain_probe_sweep import (
    CellScore,
    cfg_for_hp,
    default_hp_grid,
    fixed_hp_eval,
    sweep_hp,
)
from speech_decoding.experiments.v2_attentive_train import HeadTrainConfig

N = 100
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


def _cache(seed: int, subject_id: int, trial_id: int, *, signal: bool) -> SessionTapCache:
    rng = np.random.default_rng(seed)
    y = rng.choice([0.0, 1.0], size=N)
    sgn = (y * 2 - 1).astype(np.float32)
    m2 = rng.normal(size=(N, C, S, D)).astype(np.float32)
    m3 = rng.normal(size=(N, P, K, S, D)).astype(np.float32)
    m4 = rng.normal(size=(N, P, K, S, D)).astype(np.float32)
    if signal:
        m2[:, 0, 0, 0] = sgn * 4 + 0.2 * rng.normal(size=N)
        m3[:, 1, 0, 0, 0] = sgn * 4 + 0.2 * rng.normal(size=N)
        m4[:, 1, 0, 0, 0] = sgn * 4 + 0.2 * rng.normal(size=N)
    rows = _split_rows(N)
    ws = {0: rows, 1: _split_rows(N)}
    cs = {"val": rows["val"], "test": rows["test"]}
    return SessionTapCache(
        subject_id=subject_id, trial_id=trial_id,
        grids={"M2": torch.from_numpy(m2), "M3": torch.from_numpy(m3),
               "M4": torch.from_numpy(m4)},
        labels={"onset": y, "volume": y},
        parcel_per_electrode=PE, electrode_mask=EM, n_parcels=P,
        ws_split={"onset": ws, "volume": ws},
        cs_split={"onset": cs, "volume": cs},
    )


def _base_cfg() -> HeadTrainConfig:
    return HeadTrainConfig(
        d_model=D, n_heads=6, lr=3e-3, batch_size=64, max_steps=250,
        eval_every=25, patience=8, swad_warmup=50, seed=0,
    )


def test_default_hp_grid_is_ben_approved():
    grid = default_hp_grid()
    assert len(grid) == 3 * 2 * 2
    assert {h.weight_decay for h in grid} == {0.1, 1.0, 3.0}
    assert {h.parcel_dropout for h in grid} == {0.0, 0.2}
    assert {h.dropout for h in grid} == {0.1, 0.5}


def test_cfg_for_hp_stamps_shared_dropout():
    hp = default_hp_grid()[-1]  # wd 3.0, pd 0.2, dropout 0.5
    cfg = cfg_for_hp(_base_cfg(), hp)
    assert cfg.weight_decay == 3.0 and cfg.parcel_dropout == 0.2
    assert cfg.attn_dropout == cfg.mlp_dropout == cfg.residual_dropout == 0.5


def test_sweep_picks_a_finite_hp_on_val_only():
    cache = _cache(0, 1, 0, signal=True)
    caches = {(1, 0): cache}
    cells = [ProbeCell("WithinSession", "onset", 1, 0, fold_index=0)]
    # A small 2-point grid keeps it fast; both produce finite val on signal data.
    grid = default_hp_grid()[:2]
    res = sweep_hp(cells, caches, _base_cfg(), grid, taps=("M3",))
    assert res.best_hp in grid
    assert np.isfinite(res.mean_val_by_hp[res.best_hp])
    assert res.n_cells_scored >= 1


def test_fixed_hp_eval_emits_both_readouts_per_cell_tap():
    caches = {(1, 0): _cache(0, 1, 0, signal=True)}
    cells = [ProbeCell("WithinSession", "onset", 1, 0, fold_index=0)]
    hp = default_hp_grid()[0]
    scores = fixed_hp_eval(cells, caches, _base_cfg(), hp, taps=("M2", "M3"))
    assert len(scores) == 2  # one cell × two taps
    for s in scores:
        assert s.linear > 0.8 and s.attentive > 0.7  # signal recovered by both


def test_aggregate_scores_rolls_up_and_finds_best_cross_subject():
    scores = [
        CellScore("c1", "CrossSubject", "volume", "M4", linear=0.70, attentive=0.72),
        CellScore("c2", "CrossSubject", "volume", "M3", linear=0.66, attentive=0.60),
        CellScore("c3", "WithinSession", "onset", "M4", linear=0.80, attentive=0.81),
    ]
    summ = aggregate_scores(scores)
    assert summ.mean_by_mode_tap[("CrossSubject", "M4", "attentive")] == pytest.approx(0.72)
    # best CS is M4 attentive (0.72), beating M3 readouts and M4 linear.
    assert summ.best_cross_subject[:2] == ("M4", "attentive")
    assert summ.best_cross_subject[2] == pytest.approx(0.72)
    assert summ.n_cells == 3


def test_firewall_self_test_passes_legal_cells_and_catches_leak():
    legal = enumerate_cells(("WithinSession", "CrossSubject"), ("onset",))
    firewall_self_test(legal)  # no raise
    import dataclasses
    leaked = dataclasses.replace(legal[0], test_subject_id=1, test_trial_id=1)  # (1,1) is lite
    with pytest.raises(AssertionError, match="WALL OF FIRE"):
        firewall_self_test([leaked])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
