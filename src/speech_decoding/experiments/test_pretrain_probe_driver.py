"""Tests for the Stage-2 driver (cell → cache grids → readout dispatch)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.experiments.pretrain_probe_driver import (
    LazyCacheMap,
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
        parcel_labels=torch.arange(P),
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


def test_save_cache_is_atomic_no_tmp_left(tmp_path):
    """Atomic write: the final path exists, the .tmp sibling is gone (so the encode's
    resume-skip can trust any present taps_*.pt as complete)."""
    import os

    path = str(tmp_path / "cache.pt")
    save_cache(_make_cache(0, 1, 0), path)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


def _write_caches(tmp_path, keys):
    paths = {}
    for i, (s, t) in enumerate(keys):
        p = str(tmp_path / f"taps_s{s}_t{t}.pt")
        save_cache(_make_cache(i, s, t), p)
        paths[(s, t)] = p
    return paths


def test_lazy_cache_map_dispatches_like_eager_dict(tmp_path):
    """A CS cell driven off a LazyCacheMap yields the same AUROC as off an eager dict —
    the lazy map is a transparent drop-in for the readout dispatch."""
    keys = [(2, 1), (3, 2)]
    paths = _write_caches(tmp_path, keys)
    lazy = LazyCacheMap(paths, maxsize=2)
    eager = {k: load_cache(p) for k, p in paths.items()}
    cell = ProbeCell("CrossSubject", "onset", 3, 2, train_subject_id=2, train_trial_id=1)
    a_lazy = run_linear_cell(cell, "M3", test_cache=lazy[(3, 2)], anchor_cache=lazy[(2, 1)])
    a_eager = run_linear_cell(
        cell, "M3", test_cache=eager[(3, 2)], anchor_cache=eager[(2, 1)]
    )
    assert a_lazy == pytest.approx(a_eager)


def test_lazy_cache_map_evicts_to_maxsize_and_supports_get(tmp_path):
    """LRU caps residency at maxsize (so the readout never holds the whole cohort), while
    keeping the anchor warm under the test→anchor access pattern; .get/contains/iter work."""
    keys = [(2, 1), (1, 0), (3, 2), (4, 2)]
    paths = _write_caches(tmp_path, keys)
    lazy = LazyCacheMap(paths, maxsize=2)
    assert set(lazy) == set(keys) and len(lazy) == 4         # iter/keys/len: no loading
    assert (2, 1) in lazy and (9, 9) not in lazy
    assert lazy.get((9, 9)) is None
    # CS-like pattern: each cell touches a test then the constant anchor (2,1).
    for test_key in [(1, 0), (3, 2), (4, 2)]:
        _ = lazy[test_key]
        _ = lazy[(2, 1)]
        assert len(lazy._lru) <= 2                           # never more than maxsize resident
        assert (2, 1) in lazy._lru                            # anchor stays warm across cells


def test_lazy_cache_map_rejects_maxsize_below_two(tmp_path):
    paths = _write_caches(tmp_path, [(2, 1)])
    with pytest.raises(ValueError, match="maxsize"):
        LazyCacheMap(paths, maxsize=1)


def test_lazy_cache_map_keep_grids_prunes_on_load(tmp_path):
    """keep_grids drops the grids a shard's taps don't read right after load (the cache
    bundles all taps; a CS pair would otherwise hold ~200 GB). A parcel-tap shard keeps only
    M3/M4; the readout it feeds still recovers signal, and d_model reads off a kept grid."""
    paths = _write_caches(tmp_path, [(2, 1), (3, 2)])
    lazy = LazyCacheMap(paths, maxsize=2, keep_grids={"M3", "M4"})
    cache = lazy[(3, 2)]
    assert set(cache.grids) == {"M3", "M4"}                  # M2 pruned
    assert int(next(iter(cache.grids.values())).shape[-1]) == D   # d_model still resolvable
    cell = ProbeCell("CrossSubject", "onset", 3, 2, train_subject_id=2, train_trial_id=1)
    a = run_linear_cell(cell, "M3", test_cache=lazy[(3, 2)], anchor_cache=lazy[(2, 1)])
    assert a > 0.8                                            # kept tap still scores


def test_lazy_cache_map_keep_grids_none_keeps_all(tmp_path):
    paths = _write_caches(tmp_path, [(2, 1)])
    cache = LazyCacheMap(paths, maxsize=2)[(2, 1)]
    assert set(cache.grids) == {"M2", "M3", "M4"}            # back-compat: no pruning


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
