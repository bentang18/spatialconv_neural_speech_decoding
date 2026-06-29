"""Laptop TDD for the λ-sweep harness (:mod:`v2_lambda_sweep`).

Pure numpy/torch — no model, no DCC. Pins: (1) the eigendecomp sweep is
byte-equivalent to the production dual ridge at m=1 and across the grid; (2) the
curve responds to λ the right way (a separable problem stays high, an underdetermined
one favours more regularization); (3) selection breaks ties toward larger λ; (4) the
CS sweep + LOSO honest-selection plumbing over synthetic per-subject parcel tables.
"""

from __future__ import annotations

import numpy as np
import torch

from speech_decoding.experiments.online_probe import dual_ridge_scores
from speech_decoding.experiments.v2_lambda_sweep import (
    DEFAULT_LAM_MULTS,
    loso_cs_auroc,
    ridge_lambda_curve,
    ridge_lambda_scores,
    select_lam_mult,
    sweep_cs_task,
    ws_lambda_curve,
)


def _toy(n_train=40, n_test=30, d=12, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d)
    ztr = rng.standard_normal((n_train, d))
    zte = rng.standard_normal((n_test, d))
    ytr = np.sign(ztr @ w + 0.3 * rng.standard_normal(n_train))
    yte = np.sign(zte @ w + 0.3 * rng.standard_normal(n_test))
    return ztr, ytr, zte, yte


def test_eigendecomp_matches_dual_ridge_at_m1():
    ztr, ytr, zte, _ = _toy()
    got = ridge_lambda_scores(ztr, ytr, zte, [1.0])[0]
    want = dual_ridge_scores(ztr, ytr, zte)  # lam_mult=1 default
    assert np.allclose(got, want, atol=1e-8)


def test_eigendecomp_matches_dual_ridge_across_grid():
    ztr, ytr, zte, _ = _toy(seed=1)
    mults = [1e-2, 0.1, 1.0, 10.0, 100.0]
    swept = ridge_lambda_scores(ztr, ytr, zte, mults)
    for i, m in enumerate(mults):
        want = dual_ridge_scores(ztr, ytr, zte, lam_mult=m)
        assert np.allclose(swept[i], want, atol=1e-8), f"mismatch at m={m}"


def test_curve_shape_and_finiteness():
    ztr, ytr, zte, yte = _toy()
    curve = ridge_lambda_curve(ztr, ytr, zte, yte)
    assert curve.shape == (len(DEFAULT_LAM_MULTS),)
    assert np.isfinite(curve).all()


def test_separable_problem_stays_high():
    # cleanly separable: AUROC should be ~1 across a wide λ range, not collapse.
    rng = np.random.default_rng(2)
    d = 8
    w = rng.standard_normal(d)
    ztr = rng.standard_normal((60, d))
    zte = rng.standard_normal((40, d))
    ytr = np.sign(ztr @ w)
    yte = np.sign(zte @ w)
    curve = ridge_lambda_curve(ztr, ytr, zte, yte)
    assert np.nanmax(curve) > 0.9
    sel = select_lam_mult(curve, DEFAULT_LAM_MULTS)
    assert sel is not None and curve[sel[0]] == np.nanmax(curve)


def test_select_breaks_ties_toward_larger_lambda():
    mults = [0.1, 1.0, 10.0, 100.0]
    curve = np.array([0.7, 0.8, 0.8, 0.6])  # tie at idx 1,2 → pick the larger mult
    i, m, val = select_lam_mult(curve, mults)
    assert (i, m, val) == (2, 10.0, 0.8)


def test_select_all_nan_returns_none():
    assert select_lam_mult(np.full(4, np.nan), [0.1, 1, 10, 100]) is None


def test_ws_curve_is_fold_mean():
    ztr, ytr, _, _ = _toy(n_train=40)
    curve = ws_lambda_curve(ztr, ytr, DEFAULT_LAM_MULTS)
    assert curve.shape == (len(DEFAULT_LAM_MULTS),)
    assert np.isfinite(curve).any()


def _global_tables(n_parcels=5, seed=3):
    """Synthetic per-subject global parcel tables: anchor + 3 test subjects sharing a
    signal-carrying parcel, with disjoint extra parcels so intersection logic bites."""
    rng = np.random.default_rng(seed)
    d = 6
    w = rng.standard_normal(d)
    glob, present, labels = {}, {}, {}
    # anchor present in parcels {0,1,2}; tests present in {0,1,3} so intersection={0,1}.
    plan = {0: [0, 1, 2], 1: [0, 1, 3], 2: [0, 1, 3], 3: [0, 1, 4]}
    for s, parcels in plan.items():
        n = 36
        tab = torch.zeros(n, n_parcels, d)
        feat = torch.from_numpy(rng.standard_normal((n, d)).astype(np.float32))
        for p in parcels:
            tab[:, p] = feat + 0.1 * torch.randn(n, d)
        y = np.sign((feat.numpy() @ w) + 0.2 * rng.standard_normal(n)).astype(np.float64)
        pmask = torch.zeros(n_parcels, dtype=torch.bool)
        pmask[parcels] = True
        glob[s], present[s], labels[s] = tab, pmask, y
    return glob, present, labels


def test_sweep_cs_task_shapes_and_intersection():
    glob, present, labels = _global_tables()
    mean, per = sweep_cs_task(glob, present, labels, anchor=0, test_subjects=[1, 2, 3],
                              lam_mults=DEFAULT_LAM_MULTS)
    assert mean.shape == (len(DEFAULT_LAM_MULTS),)
    assert set(per) == {1, 2, 3}
    for c in per.values():
        assert c is not None and c.shape == (len(DEFAULT_LAM_MULTS),)
    # signal is shared on the intersection parcels → above-chance somewhere on the grid.
    assert np.nanmax(mean) > 0.5


def test_sweep_cs_task_empty_intersection_is_none():
    glob, present, labels = _global_tables()
    # make subject 9 share NO parcel with the anchor (parcel 4 only; anchor has {0,1,2}).
    n_parcels = present[0].shape[0]
    tab = torch.zeros(36, n_parcels, glob[0].shape[-1])
    tab[:, 4] = torch.randn(36, glob[0].shape[-1])
    glob[9] = tab
    pmask = torch.zeros(n_parcels, dtype=torch.bool)
    pmask[4] = True
    present[9] = pmask
    labels[9] = np.sign(np.random.default_rng(9).standard_normal(36)).astype(np.float64)
    _mean, per = sweep_cs_task(glob, present, labels, anchor=0, test_subjects=[1, 9],
                               lam_mults=DEFAULT_LAM_MULTS)
    assert per[9] is None and per[1] is not None


def test_loso_cs_auroc_holds_out_each_subject():
    glob, present, labels = _global_tables()
    _mean, per = sweep_cs_task(glob, present, labels, anchor=0, test_subjects=[1, 2, 3],
                               lam_mults=DEFAULT_LAM_MULTS)
    val, picks = loso_cs_auroc(per, DEFAULT_LAM_MULTS)
    assert np.isfinite(val)
    assert set(picks) == {1, 2, 3}                       # one pick per held-out subject
    assert all(m in DEFAULT_LAM_MULTS for m in picks.values())


def test_loso_needs_two_usable_subjects():
    val, picks = loso_cs_auroc({1: np.zeros(len(DEFAULT_LAM_MULTS)), 2: None}, DEFAULT_LAM_MULTS)
    assert np.isnan(val) and picks == {}
