"""Decoder-parity arm — the contrast is only meaningful if NOTHING else differs.

A decoder swap that also (quietly) changes which rows are fit, or leaks the val half into the
fit, produces a number that looks like a decoder effect and is not one. These tests pin the
"nothing else differs" half of the claim: same train rows, same test rows, val untouched.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.neuroprobe import fe_abl_decoder as D
from scripts.neuroprobe import v3_board_readout as R
from scripts.neuroprobe.test_v3_board_readout import _rec

TASK = R.BOARD_TASKS[0]


def _trace(mod, rec, cell_fn):
    """Run a cell function with `_feat` patched to record (rows) per call."""
    seen = []
    orig = mod._feat

    def spy(r, enc, rows, col_idx=None):
        seen.append(np.asarray(rows).copy())
        return orig(r, enc, rows, col_idx)

    mod._feat = spy
    try:
        cell_fn()
    finally:
        mod._feat = orig
    return seen


def test_ws_fits_and_scores_the_same_rows_as_the_locked_readout() -> None:
    rec = _rec(n=64, n_parcels=3, feat=32)
    ours = _trace(D, rec, lambda: D._ws_cell(rec, TASK, ("enc0",)))
    theirs = _trace(R, rec, lambda: R._ws_cell(rec, TASK, ("enc0",)))
    # readout draws (train, val, test) per fold; ours draws (train, test)
    assert len(theirs) == 3 * len(rec["ws_split"][TASK])
    assert len(ours) == 2 * len(rec["ws_split"][TASK])
    for f in range(len(rec["ws_split"][TASK])):
        assert np.array_equal(ours[2 * f], theirs[3 * f]), "train rows differ"
        assert np.array_equal(ours[2 * f + 1], theirs[3 * f + 2]), "test rows differ"


def test_ws_never_touches_the_val_half() -> None:
    """The val half is the ridge's λ-selection set. If it reaches the logistic fit at all, the
    arm is no longer upstream's protocol and the comparison is void."""
    rec = _rec(n=64, n_parcels=3, feat=32)
    val = {tuple(sp["val"]) for sp in rec["ws_split"][TASK].values()}
    rows = _trace(D, rec, lambda: D._ws_cell(rec, TASK, ("enc0",)))
    for r in rows:
        assert tuple(r) not in val


def test_cs_fits_the_anchor_and_scores_the_test_half_only() -> None:
    anchor, test = _rec(n=64, n_parcels=3, feat=32), _rec(n=64, n_parcels=3, feat=32)
    rows = _trace(D, anchor, lambda: D._cs_cell(anchor, test, TASK, ("enc0",)))
    assert len(rows) == 2
    assert np.array_equal(rows[0], np.arange(64))                       # full anchor
    assert np.array_equal(rows[1], np.asarray(test["cs_split"][TASK]["test"]))


def test_the_classifier_is_upstreams_untuned_default() -> None:
    """C=1.0 UNTUNED is the whole point — a tuned C would answer a different question.

    Asserted BEHAVIOURALLY, not by attribute: sklearn 1.8 deprecated `penalty` (local is 1.8.0,
    Delta's pytorch-conda/2.8 is 1.7.2), so `clf.penalty == "l2"` is a version check dressed up
    as a contract. What must hold on both is that the default fit is genuinely L2-REGULARIZED at
    C=1.0 — i.e. shrunk relative to an effectively unpenalized fit.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(60), np.ones(60)]
    z = rng.normal(size=(120, 8)) + 0.8 * y[:, None]
    clf = LogisticRegression(random_state=D.SEED, max_iter=10000, tol=1e-3)
    assert clf.C == 1.0 and clf.solver == "lbfgs"
    free = LogisticRegression(random_state=D.SEED, max_iter=10000, tol=1e-3, C=1e6)
    assert np.linalg.norm(clf.fit(z, y).coef_) < np.linalg.norm(free.fit(z, y).coef_)


def test_separable_features_score_near_one() -> None:
    """Sanity: the fit/score path is wired up (a transposed matrix would sit at chance)."""
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(40), np.ones(40)]
    z = rng.normal(size=(80, 6)) + y[:, None]
    assert D._score(z, y, z, y)["test"] > 0.9


def test_a_single_class_train_half_returns_nan_not_a_crash() -> None:
    z = np.random.default_rng(0).normal(size=(20, 4))
    assert np.isnan(D._score(z, np.zeros(20), z, np.r_[np.zeros(10), np.ones(10)])["test"])


# ── the tuned arm ──────────────────────────────────────────────────────────────────

def test_tuned_and_untuned_arms_get_DIFFERENT_column_names() -> None:
    """If both arms wrote `enc0|logreg`, a merge would silently average tuned with untuned."""
    assert D._NAME(None) == "logreg" and D._NAME(D.C_GRID) == "logregcv"


def test_tuned_arm_selects_C_on_the_val_half_and_reports_test() -> None:
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(60), np.ones(60)]
    z = rng.normal(size=(120, 10)) + 0.5 * y[:, None]
    out = D._score(z, y, z, y, z, y, D.C_GRID)
    assert out["C"] in D.C_GRID
    assert 0.0 <= out["test"] <= 1.0


def test_tuned_arm_never_selects_on_test() -> None:
    """C must be chosen by val AUROC. Feeding a val half that is pure noise while test is
    separable must NOT let the arm find the test-optimal C."""
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(50), np.ones(50)]
    z_tr = rng.normal(size=(100, 8)) + y[:, None]
    z_va = rng.normal(size=(100, 8))                       # no signal at all
    z_te = rng.normal(size=(100, 8)) + 3.0 * y[:, None]    # very separable
    a, (v, b) = D._standardize(z_tr, [z_va, z_te])
    per_c = {c: D._fit_score(a, y, [(v, y), (b, y)], c) for c in D.C_GRID}
    best_val = min(c for c in D.C_GRID if per_c[c][0] == max(s[0] for s in per_c.values()))
    best_test = min(c for c in D.C_GRID if per_c[c][1] == max(s[1] for s in per_c.values()))
    assert best_val != best_test, "setup failed: val and test agree, so the test proves nothing"
    got = D._score(z_tr, y, z_te, y, z_va, y, D.C_GRID)
    assert got["C"] == best_val, "C was not selected on the val half"
    assert got["test"] == pytest.approx(per_c[best_val][1]), "reported test is not at the chosen C"


def test_ws_tuned_arm_uses_the_val_half_that_the_untuned_arm_ignores() -> None:
    rec = _rec(n=64, n_parcels=3, feat=32)
    val = {tuple(sp["val"]) for sp in rec["ws_split"][TASK].values()}
    untuned = _trace(D, rec, lambda: D._ws_cell(rec, TASK, ("enc0",)))
    tuned = _trace(D, rec, lambda: D._ws_cell(rec, TASK, ("enc0",), D.C_GRID[:2]))
    assert not any(tuple(r) in val for r in untuned)
    assert any(tuple(r) in val for r in tuned)


def test_fold_mean_keeps_per_fold_C() -> None:
    got = D._fold_mean([{"test": 0.6, "C": 1e-4, "C_pinned": True},
                        {"test": 0.8, "C": 1e4, "C_pinned": True}])
    assert got["test"] == pytest.approx(0.7) and got["C"] == [1e-4, 1e4] and got["C_pinned"]
