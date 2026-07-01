"""Tests for the linear ridge per-cell readouts (Stage 2a)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.experiments.pretrain_probe_readout import (
    compacted_positions,
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
    """WS-M2 now pools electrode→parcel (mean). Signal in an electrode of parcel 1
    survives the 2-electrode average and is recovered at parcel resolution."""
    rng = np.random.default_rng(0)
    n, c = 80, 6
    y = _labels(rng, n)
    grid = _grid_with_signal(rng, y, c, signal_unit=2)      # electrode 2 → parcel 1
    pe = torch.tensor([0, 0, 1, 1, 2, 2])
    tr, va, te = _rows(n)
    a = linear_ws_cell_auroc(
        grid, y, train_rows=tr, val_rows=va, test_rows=te, tap_space="electrode",
        parcel_per_electrode=pe,
        electrode_mask=torch.ones(c, dtype=torch.bool), n_parcels=N_PARCELS,
    )
    assert a > 0.9


def test_ws_electrode_pools_to_parcel_width_not_electrode_width():
    """The reduction is parcel-mean: feature width is P_present·F (here 3 parcels × F),
    NOT C·F — this is what keeps WS-M2 off the 46 GB all-electrode flatten. Two
    electrodes sharing a parcel are averaged into one block."""
    import speech_decoding.experiments.pretrain_probe_readout as ro

    rng = np.random.default_rng(10)
    n, c = 40, 6
    grid = torch.from_numpy(rng.normal(size=(n, c, F, 1)).astype(np.float32))
    pe = torch.tensor([0, 0, 1, 1, 2, 2])                   # 3 parcels, 2 electrodes each
    z = ro._pooled_parcel_features_ws(
        grid, pe, torch.ones(c, dtype=torch.bool), N_PARCELS
    )
    assert z.shape == (n, 3 * F)                            # parcel width, not 6·F
    # parcel-0 block is the mean of electrodes 0 and 1.
    expect = ((grid[:, 0] + grid[:, 1]) / 2).reshape(n, F).numpy()
    np.testing.assert_allclose(z[:, :F], expect, rtol=1e-5, atol=1e-5)


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
        parcel_labels=torch.arange(N_PARCELS),
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
        parcel_labels_anchor=torch.arange(N_PARCELS),
        parcel_labels_test=torch.arange(N_PARCELS),
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
        parcel_labels_anchor=torch.arange(N_PARCELS),
        parcel_labels_test=torch.arange(N_PARCELS),
    )
    assert np.isnan(a)


def test_compacted_positions_maps_atlas_to_grid_axis():
    """``searchsorted`` over the sorted compacted ids; absent id is fail-loud."""
    lab = torch.tensor([0, 2, 5, 7])               # compacted parcel grid axis
    pos = compacted_positions(lab, torch.tensor([2, 7]))
    assert pos.tolist() == [1, 3]
    with pytest.raises(ValueError, match="absent"):
        compacted_positions(lab, torch.tensor([3]))  # 3 not in the compacted set


def test_cs_parcel_compacted_grids_align_by_dkt_id():
    """The CS lazy-map the refactor relies on: two subjects with DIFFERENT compacted
    parcel sets share DKT parcel 2; its block must align even though it sits at a
    different compacted position in each grid (here both pos 1, but selected by id)."""
    rng = np.random.default_rng(7)
    n_parcels = 5
    na, nt = 60, 60
    ya, yt = _labels(rng, na), _labels(rng, nt)
    # Subject A: parcels {0,2}; B: {1,2}. Signal rides the SHARED parcel 2.
    ga = _grid_with_signal(rng, ya, 2, signal_unit=1)   # compacted pos1 = parcel 2
    gt = _grid_with_signal(rng, yt, 2, signal_unit=1)
    lab_a, lab_t = torch.tensor([0, 2]), torch.tensor([1, 2])
    pe_a, pe_t = torch.tensor([0, 2]), torch.tensor([1, 2])
    em = torch.ones(2, dtype=torch.bool)
    _, va, te = _rows(nt)
    a = linear_cs_cell_auroc(
        ga, ya, gt, yt, val_rows=va, test_rows=te, tap_space="parcel",
        pe_anchor=pe_a, em_anchor=em, pe_test=pe_t, em_test=em, n_parcels=n_parcels,
        parcel_labels_anchor=lab_a, parcel_labels_test=lab_t,
    )
    assert a > 0.85


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
