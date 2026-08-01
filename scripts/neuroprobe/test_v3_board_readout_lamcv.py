"""λ selected inside the TRAIN half (--lam-cv).

The claim this column makes is narrow and has to stay narrow: it is the SAME fit as the published
`std` column, read at a different λ. So the tests here are mostly about what must NOT move --- the
control, the test curve, and the strict exclusion of held-out data from the selection.

d > n throughout (_D > _N): the CV curve lives on the dual branch, which is the branch every
headline tap takes (enc12_elec is d=1.58e6 against n<=1750). The primal branch is exercised
separately, where the column is expected to be ABSENT rather than wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

import scripts.neuroprobe.v3_board_readout as mod

_N, _D = 120, 200


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(3 * _N, _D)).astype(np.float32)
    beta = rng.normal(size=_D)
    y = ((z @ beta + rng.normal(size=3 * _N) * 8.0) > 0).astype(np.float64)
    return z[:_N], y[:_N], z[_N:2 * _N], y[_N:2 * _N], z[2 * _N:], y[2 * _N:]


def _evals(zv, yv, zt, yt):
    return {"val": (zv, yv), "test": (zt, yt)}


# ── the folds themselves ──────────────────────────────────────────────────────────────────────

def test_cv_folds_are_contiguous_disjoint_and_cover_everything():
    """Contiguous is the point: a shuffled fold would put temporally adjacent windows on both
    sides of the split and report an optimistic curve — upstream avoids exactly this with
    KFold(shuffle=False)."""
    folds = mod._cv_folds(100, 5)
    assert len(folds) == 5
    seen = np.concatenate([te for _, te in folds])
    assert sorted(seen.tolist()) == list(range(100)), "held-out blocks must partition 0..n-1"
    for tr, te in folds:
        assert np.all(np.diff(te) == 1), "held-out block must be CONTIGUOUS, not scattered"
        assert not set(tr.tolist()) & set(te.tolist()), "a row cannot be in both halves"
        assert len(tr) + len(te) == 100


# ── the load-bearing exclusion ────────────────────────────────────────────────────────────────

def test_the_cv_curve_CANNOT_see_val_or_test(monkeypatch, data):
    """THE LOAD-BEARING TEST. The whole justification for this column is that it selects λ without
    touching held-out data. Flip every val and test label; the selection curve must not move by
    one bit. If it does, the column is selecting on held-out data and is not submittable."""
    monkeypatch.setattr(mod, "LAM_CV", True)
    ztr, ytr, zv, yv, zt, yt = data
    a = mod._lam_grid(ztr.copy(), ytr, _evals(zv, yv, zt, yt))["_cv"]
    b = mod._lam_grid(ztr.copy(), ytr, _evals(zv, 1 - yv, zt, 1 - yt))["_cv"]
    assert a == b, "the train-CV curve moved when held-out labels changed — it is leaking"


def test_the_cv_curve_DOES_move_with_train_labels(monkeypatch, data):
    """The negative control for the test above: a curve that never moves would pass that test
    trivially by being a constant."""
    monkeypatch.setattr(mod, "LAM_CV", True)
    ztr, ytr, zv, yv, zt, yt = data
    a = mod._lam_grid(ztr.copy(), ytr, _evals(zv, yv, zt, yt))["_cv"]
    b = mod._lam_grid(ztr.copy(), 1 - ytr, _evals(zv, yv, zt, yt))["_cv"]
    assert a != b, "the curve ignores the train labels — it is not measuring anything"


# ── the control cannot move ───────────────────────────────────────────────────────────────────

def test_lam_cv_leaves_the_published_column_BIT_IDENTICAL(monkeypatch, data):
    """--lam-cv only APPENDS. If the published std column moved, every paired delta is void."""
    ztr, ytr, zv, yv, zt, yt = data
    monkeypatch.setattr(mod, "LAM_CV", False)
    off = mod._lam_grid(ztr.copy(), ytr, _evals(zv, yv, zt, yt))
    monkeypatch.setattr(mod, "LAM_CV", True)
    on = mod._lam_grid(ztr.copy(), ytr, _evals(zv, yv, zt, yt))
    assert "_cv" not in off and "_cv" in on
    assert off["val"] == on["val"] and off["test"] == on["test"]


def test_the_cv_column_reports_the_SAME_test_curve_only_at_a_different_lambda(monkeypatch, data):
    """Nothing is refit. The two columns are paired on one fit by construction, so a delta between
    them can only be the λ choice — not features, splits or arithmetic."""
    monkeypatch.setattr(mod, "LAM_CV", True)
    ztr, ytr, zv, yv, zt, yt = data
    d = mod._lam_grid(ztr.copy(), ytr, _evals(zv, yv, zt, yt))
    cv = mod._cv_selected(d)
    assert cv is not None, "d > n takes the dual branch, so the cv curve must exist"
    assert cv["test"] == d["test"], "the test curve must be shared, not recomputed"
    assert cv["val"] == d["_cv"], "the cv column must SELECT on the train-CV curve"
    sel_std, sel_cv = mod._select_lam(d), mod._select_lam(cv)
    assert sel_cv["test"] == d["test"][sel_cv["lam_mult"]]
    print(f"[check] std λ={sel_std['lam_mult']:.3g} test={sel_std['test']:.4f} | "
          f"cv λ={sel_cv['lam_mult']:.3g} test={sel_cv['test']:.4f}")


def test_no_flag_means_no_column(monkeypatch, data):
    monkeypatch.setattr(mod, "LAM_CV", False)
    ztr, ytr, zv, yv, zt, yt = data
    assert mod._cv_selected(mod._lam_grid(ztr.copy(), ytr, _evals(zv, yv, zt, yt))) is None


def test_primal_branch_yields_NO_column_rather_than_a_wrong_one(monkeypatch):
    """d < n takes _lam_grid_primal, which computes no CV curve. The column must be ABSENT (the
    reader then reports it 'not compared') instead of silently appearing for some taps only."""
    monkeypatch.setattr(mod, "LAM_CV", True)
    rng = np.random.default_rng(1)
    z = rng.normal(size=(3 * _N, 20)).astype(np.float32)      # d=20 < n=120
    y = (rng.normal(size=3 * _N) > 0).astype(np.float64)
    d = mod._lam_grid(z[:_N], y[:_N], _evals(z[_N:2 * _N], y[_N:2 * _N], z[2 * _N:], y[2 * _N:]))
    assert mod._cv_selected(d) is None


# ── degenerate folds ──────────────────────────────────────────────────────────────────────────

def test_a_single_class_holdout_block_is_SKIPPED_not_scored_half(monkeypatch):
    """A single-class block carries no ranking information. Averaging a fabricated 0.5 into the
    curve would drag every λ toward indifference and flatten the selection."""
    monkeypatch.setattr(mod, "LAM_CV", True)
    n = 100
    g = np.eye(n) * 2.0 + 0.1
    # ONLY fold 0 (rows 0..19) is degenerate; folds 1..4 must stay two-class, or the test would be
    # asserting the all-degenerate case instead of the skip.
    y = np.zeros(n)
    y[20:] = (np.arange(20, n) % 2).astype(np.float64)
    assert (y[:20] > 0).min() == (y[:20] > 0).max(), "fixture: fold 0 must be single-class"
    curve = mod._cv_curve(g, y, base=1.0, k=5)
    assert all(np.isfinite(v) for v in curve.values()), "skipping must not NaN the whole curve"
    assert len(mod._cv_folds(n, 5)) == 5


def test_all_folds_degenerate_gives_NaN_not_a_number(monkeypatch):
    """If nothing can be scored, the curve says so rather than inventing a λ."""
    monkeypatch.setattr(mod, "LAM_CV", True)
    n = 50
    curve = mod._cv_curve(np.eye(n) + 0.1, np.ones(n), base=1.0, k=5)
    assert all(np.isnan(v) for v in curve.values())
    assert np.isnan(mod._select_lam({"val": curve, "test": curve})["lam_mult"])


# ── the mechanism can work at all ─────────────────────────────────────────────────────────────

def test_train_cv_beats_a_TINY_val_half_at_picking_lambda(monkeypatch):
    """The premise, demonstrated where it must hold: when the val half is small, selecting on it
    is noisy, and a selector with far more rows should land closer to the test-optimal λ.

    This is a SUFFICIENCY check on synthetic data, NOT evidence about the board — the board's own
    prior (written at LAM_CV) is that this helps CS and hurts WS.
    """
    monkeypatch.setattr(mod, "LAM_CV", True)
    rng = np.random.default_rng(7)
    n_tr, n_va, n_te, d = 400, 12, 600, 800      # val deliberately starved
    z = rng.normal(size=(n_tr + n_va + n_te, d)).astype(np.float32)
    beta = rng.normal(size=d)
    y = ((z @ beta + rng.normal(size=len(z)) * 25.0) > 0).astype(np.float64)
    ztr, zv, zt = z[:n_tr], z[n_tr:n_tr + n_va], z[n_tr + n_va:]
    ytr, yv, yt = y[:n_tr], y[n_tr:n_tr + n_va], y[n_tr + n_va:]
    d_ = mod._lam_grid(ztr.copy(), ytr, _evals(zv, yv, zt, yt))
    by_val = mod._select_lam(d_)["test"]
    by_cv = mod._select_lam(mod._cv_selected(d_))["test"]
    best = max(d_["test"].values())
    print(f"[check] starved-val: val-selected {by_val:.4f}  train-CV {by_cv:.4f}  oracle {best:.4f}")
    assert by_cv > by_val, "train-CV did not beat a 12-row val half — premise not demonstrated"
