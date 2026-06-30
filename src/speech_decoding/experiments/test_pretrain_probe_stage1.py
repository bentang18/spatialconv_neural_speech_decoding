"""Tests for Stage-1 cache assembly + orchestration (#3), with a stubbed GPU forward."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from speech_decoding.experiments.pretrain_probe_driver import run_linear_cell
from speech_decoding.experiments.pretrain_probe_labels import build_session_targets
from speech_decoding.experiments.pretrain_probe_stage1 import (
    SessionMeta,
    build_caches,
    cache_from_targets,
)
from speech_decoding.experiments.pretrain_probe_suite import ProbeCell

C = 6
P = 4
K = 2
S = 2
D = 12
PE = torch.tensor([0, 0, 1, 1, 2, 2])
EM = torch.ones(C, dtype=torch.bool)
META = SessionMeta(parcel_per_electrode=PE, electrode_mask=EM, n_parcels=P)


def _events(subject_id, trial_id, n_clips=12):
    starts = [round(0.5 * i, 3) for i in range(n_clips)]
    rows = []
    for i, s in enumerate(starts):
        for task in ("onset", "volume"):
            rows.append({
                "type": "Word", "start": s, "duration": 1.0, "text": "<w>",
                "task": task, "label": i % 2, "subject_id": str(subject_id),
                "trial_id": str(trial_id), "timeline": "tl", "movie_onset_s": s + 100.0,
            })
    return pd.DataFrame(rows)


def _stub_forward(targets):
    """Deterministic per-clip grids; onset label (clip parity) rides parcel 1 so the
    readout recovers it through the whole targets→cache→driver path."""
    n = len(targets.clip_starts)
    rng = np.random.default_rng(len(targets.clip_starts))
    sgn = ((np.arange(n) % 2) * 2 - 1).astype(np.float32)
    m2 = rng.normal(size=(n, C, S, D)).astype(np.float32)
    m3 = rng.normal(size=(n, P, K, S, D)).astype(np.float32)
    m4 = rng.normal(size=(n, P, K, S, D)).astype(np.float32)
    m3[:, 1, 0, 0, 0] = sgn * 4
    return {"M2": torch.from_numpy(m2), "M3": torch.from_numpy(m3), "M4": torch.from_numpy(m4)}


def test_cache_from_targets_aligns_and_carries_splits():
    targets = build_session_targets(_events(2, 1), subject_id=2, trial_id=1)
    grids = _stub_forward(targets)
    cache = cache_from_targets(targets, grids, META)
    n = len(targets.clip_starts)
    assert cache.grids["M3"].shape[0] == n
    assert set(cache.labels) == {"onset", "volume"}
    assert cache.ws_split["onset"][0]["train"].dtype == np.int64
    assert cache.n_parcels == P


def test_cache_from_targets_rejects_misaligned_grid():
    targets = build_session_targets(_events(2, 1), subject_id=2, trial_id=1)
    bad = {"M3": torch.zeros(3, P, K, S, D)}  # wrong leading axis
    with pytest.raises(ValueError, match="union clip axis"):
        cache_from_targets(targets, bad, META)


def test_build_caches_loops_and_saves(tmp_path):
    keys = [(2, 1), (3, 2)]
    targets_by = {k: build_session_targets(_events(*k), subject_id=k[0], trial_id=k[1])
                  for k in keys}
    meta_by = {k: META for k in keys}
    paths = {}

    def save_path_fn(key):
        p = str(tmp_path / f"cache_{key[0]}_{key[1]}.pt")
        paths[key] = p
        return p

    caches = build_caches(targets_by, meta_by, _stub_forward, save_path_fn=save_path_fn)
    assert set(caches) == set(keys)
    for k in keys:
        assert (tmp_path / f"cache_{k[0]}_{k[1]}.pt").exists()


def test_end_to_end_targets_to_readout_recovers_signal():
    """Full pure path: events → targets → stub forward → cache → driver readout."""
    targets = build_session_targets(_events(1, 0, n_clips=16), subject_id=1, trial_id=0)
    cache = cache_from_targets(targets, _stub_forward(targets), META)
    cell = ProbeCell("WithinSession", "onset", 1, 0, fold_index=0)
    assert run_linear_cell(cell, "M3", test_cache=cache) > 0.85


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
