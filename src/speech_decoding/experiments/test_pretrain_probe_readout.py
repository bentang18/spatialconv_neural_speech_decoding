"""Tests for the linear ridge per-cell readouts (Stage 2a)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.experiments.pretrain_probe_readout import (
    linear_cs_cell_auroc,
    linear_ws_cell_auroc,
    parcel_support,
)

N_PARCELS = 4
F = 3


def _rows(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train = first half; val/test = halves of the second half (upstream split)."""
    cut = n // 2
    train = np.arange(cut)
    held = np.arange(cut, n)
    vcut = len(held) // 2
    return train, held[:vcut], held[vcut:]


def _labels(rng, n: int) -> np.ndarray:
    return rng.choice([-1.0, 1.0], size=n)


def _grid_with_signal(rng, y, n_units, signal_unit, *, snr=6.0):
    """(N, n_units, F, 1) grid whose ``signal_unit`` bin-0 encodes the ±1 label."""
    n = len(y)
    g = rng.normal(size=(n, n_units, F, 1)).astype(np.float32)
    g[:, signal_unit, 0, 0] = (y * snr + 0.1 * rng.normal(size=n)).astype(np.float32)
    return torch.from_numpy(g)


def test_parcel_support_marks_only_covered_parcels():
    pe = torch.tensor([0, 0, 1, 1])
    em = torch.tensor([True, True, True, False])  # only one valid electrode in parcel 1
    sup = parcel_support(pe, em, N_PARCELS)
    assert sup.tolist() == [True, True, False, False]


def test_ws_electrode_recovers_signal():
    rng = np.random.default_rng(0)
    n, c = 80, 6
    y = _labels(rng, n)
    grid = _grid_with_signal(rng, y, c, signal_unit=2)
    tr, va, te = _rows(n)
    a = linear_ws_cell_auroc(
        grid, y, train_rows=tr, val_rows=va, test_rows=te, tap_space="electrode",
        parcel_per_electrode=torch.zeros(c, dtype=torch.long),
        electrode_mask=torch.ones(c, dtype=torch.bool), n_parcels=N_PARCELS,
    )
    assert a > 0.9


def test_ws_parcel_recovers_signal():
    rng = np.random.default_rng(1)
    n = 80
    y = _labels(rng, n)
    grid = _grid_with_signal(rng, y, N_PARCELS, signal_unit=1)  # parcel-native grid
    pe = torch.tensor([0, 0, 1, 1, 2, 2])  # parcels 0,1,2 supported; 3 not
    em = torch.ones(6, dtype=torch.bool)
    tr, va, te = _rows(n)
    a = linear_ws_cell_auroc(
        grid, y, train_rows=tr, val_rows=va, test_rows=te, tap_space="parcel",
        parcel_per_electrode=pe, electrode_mask=em, n_parcels=N_PARCELS,
    )
    assert a > 0.9


def test_ws_random_features_are_chance():
    rng = np.random.default_rng(2)
    n, c = 120, 6
    y = _labels(rng, n)
    grid = torch.from_numpy(rng.normal(size=(n, c, F, 1)).astype(np.float32))  # no signal
    tr, va, te = _rows(n)
    a = linear_ws_cell_auroc(
        grid, y, train_rows=tr, val_rows=va, test_rows=te, tap_space="electrode",
        parcel_per_electrode=torch.zeros(c, dtype=torch.long),
        electrode_mask=torch.ones(c, dtype=torch.bool), n_parcels=N_PARCELS,
    )
    assert 0.3 < a < 0.7


def test_cs_electrode_pool_intersect_recovers_shared_parcel_signal():
    rng = np.random.default_rng(3)
    na, nt, c = 60, 60, 6
    pe = torch.tensor([0, 0, 1, 1, 2, 2])  # shared montage → shared parcels
    em = torch.ones(c, dtype=torch.bool)
    ya, yt = _labels(rng, na), _labels(rng, nt)
    # Signal in an electrode of parcel 0 for both subjects (label-aligned).
    ga = _grid_with_signal(rng, ya, c, signal_unit=0)
    gt = _grid_with_signal(rng, yt, c, signal_unit=0)
    _, va, te = _rows(nt)
    a = linear_cs_cell_auroc(
        ga, ya, gt, yt, val_rows=va, test_rows=te, tap_space="electrode",
        pe_anchor=pe, em_anchor=em, pe_test=pe, em_test=em, n_parcels=N_PARCELS,
    )
    assert a > 0.85


def test_cs_parcel_supported_intersection_recovers_signal():
    rng = np.random.default_rng(4)
    na, nt = 60, 60
    pe = torch.tensor([0, 0, 1, 1, 2, 2])
    em = torch.ones(6, dtype=torch.bool)
    ya, yt = _labels(rng, na), _labels(rng, nt)
    ga = _grid_with_signal(rng, ya, N_PARCELS, signal_unit=1)
    gt = _grid_with_signal(rng, yt, N_PARCELS, signal_unit=1)
    _, va, te = _rows(nt)
    a = linear_cs_cell_auroc(
        ga, ya, gt, yt, val_rows=va, test_rows=te, tap_space="parcel",
        pe_anchor=pe, em_anchor=em, pe_test=pe, em_test=em, n_parcels=N_PARCELS,
    )
    assert a > 0.85


def test_cs_empty_intersection_returns_nan():
    rng = np.random.default_rng(5)
    na, nt = 40, 40
    ya, yt = _labels(rng, na), _labels(rng, nt)
    ga = _grid_with_signal(rng, ya, N_PARCELS, signal_unit=0)
    gt = _grid_with_signal(rng, yt, N_PARCELS, signal_unit=2)
    pe_a = torch.tensor([0, 0, 1, 1])   # anchor supports parcels {0,1}
    pe_t = torch.tensor([2, 2, 3, 3])   # test supports parcels {2,3} → disjoint
    em = torch.ones(4, dtype=torch.bool)
    _, va, te = _rows(nt)
    a = linear_cs_cell_auroc(
        ga, ya, gt, yt, val_rows=va, test_rows=te, tap_space="parcel",
        pe_anchor=pe_a, em_anchor=em, pe_test=pe_t, em_test=em, n_parcels=N_PARCELS,
    )
    assert np.isnan(a)


def test_nan_labels_dropped():
    rng = np.random.default_rng(6)
    n, c = 80, 6
    y = _labels(rng, n)
    y[::4] = np.nan  # a quarter of windows uncovered by this task
    grid = _grid_with_signal(rng, np.nan_to_num(y, nan=1.0), c, signal_unit=2)
    tr, va, te = _rows(n)
    a = linear_ws_cell_auroc(
        grid, y, train_rows=tr, val_rows=va, test_rows=te, tap_space="electrode",
        parcel_per_electrode=torch.zeros(c, dtype=torch.long),
        electrode_mask=torch.ones(c, dtype=torch.bool), n_parcels=N_PARCELS,
    )
    assert np.isfinite(a)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
