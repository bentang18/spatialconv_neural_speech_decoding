"""Tests for Stage-1 cache assembly + orchestration (#3), with a stubbed GPU forward."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from speech_decoding.experiments.pretrain_probe_driver import run_linear_cell
from speech_decoding.experiments.pretrain_probe_labels import build_session_targets
from speech_decoding.experiments.online_probe import (
    feature_matrix,
    parcel_intersection,
)
from speech_decoding.experiments.pretrain_probe_readout import parcel_support
from speech_decoding.experiments.pretrain_probe_stage1 import (
    SessionMeta,
    build_caches,
    cache_from_targets,
    scatter_parcels_to_atlas,
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


def test_scatter_parcels_to_atlas_places_by_dkt_id():
    """Compacted (N,P,k,S,d) -> (N,n_parcels,k,S,d) at the labels' DKT slots, zeros else."""
    grid = torch.randn(3, 2, 2, 2, 4)
    labels = torch.tensor([1, 3])
    out = scatter_parcels_to_atlas(grid, labels, n_parcels=5)
    assert out.shape == (3, 5, 2, 2, 4)
    assert torch.allclose(out[:, 1], grid[:, 0])
    assert torch.allclose(out[:, 3], grid[:, 1])
    for empty in (0, 2, 4):                       # unsupported parcels stay zero
        assert out[:, empty].abs().sum() == 0


def test_scatter_parcels_to_atlas_rejects_bad_input():
    grid = torch.randn(3, 2, 2, 2, 4)
    with pytest.raises(ValueError):               # label id >= atlas
        scatter_parcels_to_atlas(grid, torch.tensor([1, 5]), n_parcels=5)
    with pytest.raises(ValueError):               # labels len != P
        scatter_parcels_to_atlas(grid, torch.tensor([1]), n_parcels=5)
    with pytest.raises(ValueError):               # not 5-D
        scatter_parcels_to_atlas(torch.randn(3, 2, 4), torch.tensor([1, 3]), n_parcels=5)


def test_scattered_grids_align_same_parcel_across_subjects():
    """The CS failure mode the scatter fixes: two subjects with DIFFERENT compacted parcel
    sets must, after scatter, have their SHARED DKT parcel land on the same atlas slot so
    ``parcel_intersection`` + ``feature_matrix`` select the same anatomy from both."""
    n_parcels = 3
    # Subject A: electrodes in parcels {0, 2}; B: {1, 2}. Shared anatomical parcel = 2.
    pe_a, pe_b = torch.tensor([0, 2]), torch.tensor([1, 2])
    em = torch.ones(2, dtype=torch.bool)
    labels_a, labels_b = torch.tensor([0, 2]), torch.tensor([1, 2])
    grid_a = torch.randn(4, 2, 1, 1, 2)           # compacted P=2 (pos0=parcel0, pos1=parcel2)
    grid_b = torch.randn(4, 2, 1, 1, 2)           # compacted P=2 (pos0=parcel1, pos1=parcel2)
    glob_a = scatter_parcels_to_atlas(grid_a, labels_a, n_parcels)
    glob_b = scatter_parcels_to_atlas(grid_b, labels_b, n_parcels)

    present_a = parcel_support(pe_a, em, n_parcels)
    present_b = parcel_support(pe_b, em, n_parcels)
    inter = parcel_intersection(present_a, present_b)
    assert inter.tolist() == [2]                  # only the shared parcel survives

    # feature_matrix on the shared slot recovers each subject's parcel-2 block — A's at
    # compacted pos 1, B's at compacted pos 1, both now at atlas slot 2.
    za = feature_matrix(glob_a, inter)
    zb = feature_matrix(glob_b, inter)
    assert torch.allclose(za, grid_a[:, 1].reshape(4, -1))
    assert torch.allclose(zb, grid_b[:, 1].reshape(4, -1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
