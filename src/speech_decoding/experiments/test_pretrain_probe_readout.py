"""Tests for the linear ridge per-cell readouts (Stage 2a)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.experiments.online_probe_raw_baseline import (
    feature_matrix_per_electrode,
)
from speech_decoding.experiments.pretrain_probe_readout import (
    DEFAULT_LAM,
    DEFAULT_LAM_GRID,
    SWEEP_LAM_GRID,
    _finite,
    _lam_scores,
    _lam_scores_from_kernels,
    _ws_all_electrode_kernels,
    compacted_positions,
    linear_cs_cell_auroc,
    linear_ws_cell_auroc,
    linear_ws_cell_scores_all_electrode,
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


def test_anchor_pool_is_memoized_across_cells():
    """The subj-2 anchor grid pools to parcels once and is reused; test grids pool per
    cell. Same anchor + two different test grids → anchor pooled 1×, test 2×, and the
    memoized AUROCs equal a cold (memo-cleared) recompute."""
    import speech_decoding.experiments.pretrain_probe_readout as ro

    rng = np.random.default_rng(11)
    na, nt, c = 60, 60, 6
    pe = torch.tensor([0, 0, 1, 1, 2, 2])
    em = torch.ones(c, dtype=torch.bool)
    ya = _labels(rng, na)
    ga = _grid_with_signal(rng, ya, c, signal_unit=0)
    cells = []
    for _ in range(2):
        yt = _labels(rng, nt)
        cells.append((yt, _grid_with_signal(rng, yt, c, signal_unit=0)))

    calls = {"anchor": 0, "test": 0}
    real = ro.pool_electrodes_to_parcels

    def counting(grid, *a, **k):
        calls["anchor" if grid is ga else "test"] += 1
        return real(grid, *a, **k)

    def score(gt, yt):
        _, va, te = _rows(nt)
        return linear_cs_cell_auroc(
            ga, ya, gt, yt, val_rows=va, test_rows=te, tap_space="electrode",
            pe_anchor=pe, em_anchor=em, pe_test=pe, em_test=em, n_parcels=N_PARCELS,
        )

    ro.pool_electrodes_to_parcels = counting
    try:
        ro._ANCHOR_POOL.clear()
        memoed = [score(gt, yt) for yt, gt in cells]
        assert calls["anchor"] == 1  # pooled once, memo hit on the 2nd cell
        assert calls["test"] == 2    # test grid pooled every cell
        cold = []
        for yt, gt in cells:
            ro._ANCHOR_POOL.clear()  # force a cold recompute each call
            cold.append(score(gt, yt))
        assert np.allclose(memoed, cold, equal_nan=True)
    finally:
        ro.pool_electrodes_to_parcels = real
        ro._ANCHOR_POOL.clear()


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


def _alt(n: int) -> np.ndarray:
    """Alternating ±1 so every contiguous split holds both classes (AUROC defined)."""
    return np.tile([1.0, -1.0], n // 2)


def test_lam_scores_from_kernels_matches_lam_scores():
    """The kernel-form λ sweep reproduces the materialized ``_lam_scores`` exactly."""
    rng = np.random.default_rng(11)
    d, ntr, nv, nte = 12, 24, 10, 10
    ztr, ytr = rng.normal(size=(ntr, d)), _alt(ntr)
    zv, yv = rng.normal(size=(nv, d)), _alt(nv)
    zte, yte = rng.normal(size=(nte, d)), _alt(nte)
    mat = _lam_scores(ztr, ytr, zv, yv, zte, yte, SWEEP_LAM_GRID)
    ker = _lam_scores_from_kernels(
        ztr @ ztr.T, ytr, zv @ ztr.T, yv, zte @ ztr.T, yte, SWEEP_LAM_GRID
    )
    assert [s[0] for s in mat] == [s[0] for s in ker]
    assert np.allclose([s[1:] for s in mat], [s[1:] for s in ker], atol=1e-9, equal_nan=True)


def test_default_lam_is_single_fixed_no_sweep():
    """Scoring defaults to ONE fixed λ (=1.0), no per-cell sweep. The multi-λ grid lives in
    SWEEP_LAM_GRID and is opt-in (run_keepS_ridge diagnostic, equivalence tests above).
    Guards the 2026-07-02 decision that per-cell λ selection is a ~0.004 test-peek optimism
    on a flat-in-λ landscape — so the default readout is a single ridge solve per cell."""
    assert DEFAULT_LAM == 1.0
    assert DEFAULT_LAM_GRID == (DEFAULT_LAM,) and len(DEFAULT_LAM_GRID) == 1
    assert 1.0 in SWEEP_LAM_GRID and len(SWEEP_LAM_GRID) > 1


def test_ws_all_electrode_streamed_matches_materialized():
    """Streamed-Gram all-electrode read == materializing the full (N, C·F) flatten and
    running the normal ridge — including when a masked electrode is dropped and when the
    stream is forced to many single-electrode blocks (target_bytes=1)."""
    rng = np.random.default_rng(12)
    n, c = 64, 5
    y = _alt(n)
    grid = _grid_with_signal(rng, y, c, signal_unit=2)
    em = torch.ones(c, dtype=torch.bool)
    em[4] = False  # dropped electrode must not enter either path
    tr, va, te = _rows(n)

    z = feature_matrix_per_electrode(grid, em).cpu().numpy()
    ztr, ytr = _finite(z[tr], y[tr])
    zv, yv = _finite(z[va], y[va])
    zte, yte = _finite(z[te], y[te])
    mat = _lam_scores(ztr, ytr, zv, yv, zte, yte, SWEEP_LAM_GRID)

    st = linear_ws_cell_scores_all_electrode(
        grid, y, train_rows=tr, val_rows=va, test_rows=te, electrode_mask=em,
        lam_grid=SWEEP_LAM_GRID,
    )
    assert np.allclose([s[1:] for s in mat], [s[1:] for s in st], atol=1e-9, equal_nan=True)

    ker = _ws_all_electrode_kernels(
        grid, y, train_rows=tr, val_rows=va, test_rows=te, electrode_mask=em,
        target_bytes=1,  # force one electrode per block
    )
    assert ker is not None
    blocked = _lam_scores_from_kernels(*ker, SWEEP_LAM_GRID)
    assert np.allclose([s[1:] for s in st], [s[1:] for s in blocked], atol=1e-9, equal_nan=True)


def test_ws_all_electrode_recovers_signal():
    """A signal planted in one electrode is decoded at full electrode resolution."""
    rng = np.random.default_rng(13)
    n, c = 96, 6
    y = _alt(n)
    grid = _grid_with_signal(rng, y, c, signal_unit=3)
    tr, va, te = _rows(n)
    scores = linear_ws_cell_scores_all_electrode(
        grid, y, train_rows=tr, val_rows=va, test_rows=te,
        electrode_mask=torch.ones(c, dtype=torch.bool),
    )
    assert max(s[2] for s in scores) > 0.85


def test_ws_all_electrode_degenerate_returns_empty():
    """<2 finite test labels → no scores (mirrors the pooled degeneracy contract)."""
    rng = np.random.default_rng(14)
    n, c = 40, 4
    y = _alt(n)
    grid = _grid_with_signal(rng, y, c, signal_unit=1)
    tr, va, te = _rows(n)
    y[te] = np.nan  # wipe the test half → degenerate
    assert linear_ws_cell_scores_all_electrode(
        grid, y, train_rows=tr, val_rows=va, test_rows=te,
        electrode_mask=torch.ones(c, dtype=torch.bool),
    ) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
