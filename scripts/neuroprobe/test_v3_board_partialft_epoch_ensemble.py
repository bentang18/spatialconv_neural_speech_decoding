"""Epoch ENSEMBLING must be a val-only rule, and its k=1 case must reproduce the published entry.

WHY THIS EXISTS. The epoch-curve arm measured a +.0105 oracle gap on WS (112/0/8) that no val-based
STOPPING rule recovers -- every restriction we tried (le4/le6/le8/le12, smooth3) lost to plain
argmax. That says the val trace does not track test well enough to PICK an epoch, which is a
different claim from "the epochs are bad": averaging over them can beat any one of them even when
val cannot say which one is best. So the intervention is to stop selecting.

TWO THINGS CAN GO WRONG AND BOTH ARE SILENT:

  1. TEST LEAKING INTO THE RULE. An ensemble is only submittable if the SET of epochs it averages
     is a function of the val trace alone. `test_rules_average_epochs_chosen_from_val_alone` holds
     val fixed, varies the test scores arbitrarily, and requires the chosen index set not to move.

  2. SCALE DOMINATION. AUROC depends only on the ORDER of scores, but a mean of RAW scores is not
     order-only: one epoch whose ridge lands at a smaller lambda emits larger-magnitude scores and
     silently outvotes the rest. Rank-averaging is the fix, and `test_rank01_is_invariant_to_a
     _positive_affine_rescale` is what pins it.

The k=1 case is the load-bearing self-check: averaging the single highest-val epoch IS the
published selection rule, so `ens_top1` must equal `test_c`. If it does not, the ensemble is
reading a different curve than the one the run selected on, and every other rule is untrustworthy.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


BFT = _mod("v3_board_partialft")


def _auroc(s, y):
    """Rank-based AUROC, independent of the module under test."""
    s, y = np.asarray(s, float), np.asarray(y, float)
    order = np.argsort(s, kind="stable")
    r = np.empty(len(s), float)
    r[order] = np.arange(1, len(s) + 1)
    npos, nneg = y.sum(), len(y) - y.sum()
    return (r[y > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg)


# ── rank transform ──────────────────────────────────────────────────────────────────────────

def test_rank01_is_invariant_to_a_positive_affine_rescale():
    """THE REASON RANKS ARE USED AT ALL. Two epochs that order the test set identically must
    contribute identically, no matter that one's ridge emits scores 1000x larger."""
    rng = np.random.default_rng(0)
    s = rng.normal(size=64)
    assert np.array_equal(BFT._rank01(s), BFT._rank01(3.7 * s + 12.0))
    assert np.array_equal(BFT._rank01(s), BFT._rank01(1e-6 * s))


def test_rank01_preserves_auroc_for_a_single_epoch():
    """The transform must not cost anything when there is nothing to average."""
    rng = np.random.default_rng(1)
    s = rng.normal(size=200)
    y = (rng.random(200) > 0.5).astype(float)
    assert _auroc(BFT._rank01(s), y) == pytest.approx(_auroc(s, y))


def test_rank01_spans_zero_to_one():
    s = np.array([5.0, -1.0, 3.0, 0.0])
    r = BFT._rank01(s)
    assert r.min() == 0.0 and r.max() == 1.0


def test_averaging_identical_epochs_is_a_no_op():
    rng = np.random.default_rng(2)
    s = rng.normal(size=50)
    y = (rng.random(50) > 0.5).astype(float)
    ens = BFT._epoch_ensembles([0.6, 0.6, 0.6], [s, s.copy(), s.copy()], y, _auroc)
    assert ens["ens_all"] == pytest.approx(_auroc(s, y))


def test_a_raw_score_mean_would_be_dominated_but_the_rank_mean_is_not():
    """Constructive: epoch A orders the test set perfectly, epoch B is pure noise scaled 1e6.
    A raw mean is B's ordering; the rank mean keeps A's vote at full weight."""
    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(40), np.ones(40)]
    good = y + rng.normal(scale=0.05, size=80)          # near-perfect ordering, O(1) scale
    junk = 1e6 * rng.normal(size=80)                    # no signal, huge scale
    assert _auroc(good.tolist(), y) > 0.95
    raw = _auroc((np.asarray(good) + np.asarray(junk)) / 2, y)
    ens = BFT._epoch_ensembles([0.9, 0.9], [good, junk], y, _auroc)["ens_all"]
    assert raw < 0.75, "fixture broken: the raw mean was supposed to be swamped"
    assert ens > raw


# ── the rules must read val only ────────────────────────────────────────────────────────────

def test_rules_average_epochs_chosen_from_val_alone():
    """THE SUBMITTABILITY INVARIANT, enforced STRUCTURALLY rather than by sampling. The rules are
    factored into `_ensemble_index_sets`, whose only argument is the val trace -- it cannot consult
    test data because it is never handed any. Asserting the signature is a stronger guarantee than
    checking that a few random test draws happen not to move the answer."""
    import inspect
    sig = inspect.signature(BFT._ensemble_index_sets)
    assert list(sig.parameters) == ["vals"], (
        f"_ensemble_index_sets must take the val trace and NOTHING else, got {list(sig.parameters)}")


def test_the_ensemble_auroc_changes_when_test_scores_change_but_the_index_sets_do_not():
    """The complement of the structural check: the numbers are of course a function of the test
    scores (they are test AUROCs), but WHICH epochs got averaged is not."""
    vals = [0.60, 0.71, 0.65, 0.71, 0.58]
    rng = np.random.default_rng(4)
    n = 64
    y = (rng.random(n) > 0.5).astype(float)
    a = BFT._epoch_ensembles(vals, [rng.normal(size=n) for _ in vals], y, _auroc)
    b = BFT._epoch_ensembles(vals, [rng.normal(size=n) for _ in vals], y, _auroc)
    assert set(a) == set(b)
    assert a["ens_all"] != b["ens_all"]


def test_valge0_keeps_epoch_zero_and_only_epochs_at_least_as_good_on_val():
    vals = [0.60, 0.55, 0.60, 0.72, 0.59]
    sets = BFT._ensemble_index_sets(vals)
    assert sets["ens_valge0"] == [0, 2, 3]


def test_top3_takes_the_three_highest_val_epochs():
    vals = [0.60, 0.71, 0.65, 0.70, 0.58]
    assert sorted(BFT._ensemble_index_sets(vals)["ens_top3"]) == [1, 2, 3]


def test_top3_ties_break_toward_the_earlier_epoch():
    """Matches the run loop's strict `>`: the FIRST epoch to reach a val value keeps it."""
    vals = [0.70, 0.70, 0.70, 0.70]
    assert sorted(BFT._ensemble_index_sets(vals)["ens_top3"]) == [0, 1, 2]


def test_all_includes_epoch_zero_the_frozen_entry():
    vals = [0.60, 0.71, 0.65]
    assert BFT._ensemble_index_sets(vals)["ens_all"] == [0, 1, 2]


# ── k=1 must reproduce the published selection ──────────────────────────────────────────────

def test_top1_is_exactly_the_loops_strict_argmax():
    """`ens_top1` is the SELF-CHECK the run prints. It must pick the same epoch the training loop
    picked -- first index attaining the max val -- or the ensemble is reading a different curve."""
    for vals in ([0.60, 0.71, 0.65], [0.71, 0.71, 0.65], [0.60, 0.65, 0.71], [0.5]):
        best = 0
        for i, v in enumerate(vals):
            if v > vals[best]:
                best = i
        assert BFT._ensemble_index_sets(vals)["ens_top1"] == [best]


def test_top1_ensemble_auroc_equals_that_epochs_own_auroc():
    """One epoch averaged with itself zero times: the rank transform must leave AUROC alone, so
    ens_top1 is comparable to `test_c` at full precision, not approximately."""
    rng = np.random.default_rng(6)
    n = 120
    y = (rng.random(n) > 0.5).astype(float)
    scores = [rng.normal(size=n) for _ in range(4)]
    vals = [0.60, 0.71, 0.65, 0.58]
    ens = BFT._epoch_ensembles(vals, scores, y, _auroc)
    assert ens["ens_top1"] == pytest.approx(_auroc(scores[1], y))


def test_nan_val_epochs_are_dropped_not_ranked_as_worst():
    """A non-finite val is a failed fit, not a bad epoch: it must not be averaged in, and must not
    be able to win top1 by comparing false against everything."""
    vals = [0.60, float("nan"), 0.72]
    sets = BFT._ensemble_index_sets(vals)
    assert 1 not in sets["ens_all"]
    assert sets["ens_top1"] == [2]


def test_a_single_epoch_trace_still_produces_every_rule():
    """Guards the degenerate cell where FT never ran (patience 0 / epochs 0): every rule must
    still return a number so the reader does not see a ragged key set."""
    rng = np.random.default_rng(7)
    y = (rng.random(20) > 0.5).astype(float)
    ens = BFT._epoch_ensembles([0.6], [rng.normal(size=20)], y, _auroc)
    assert set(ens) == {"ens_all", "ens_valge0", "ens_top3", "ens_top1",
                        "ens_last3", "ens_last5"}
    assert all(np.isfinite(v) for v in ens.values())


# ── the CANONICAL rule: last-N, no val selection ────────────────────────────────────────────

def test_lastn_is_the_tail_of_a_GIVEN_trace_and_does_not_reorder_it_by_val():
    """THE TEXTBOOK RULE. Vaswani et al. 2017 averaged the LAST N checkpoints of one run; SWA
    averages the tail of the trajectory.

    ⚠️ SCOPE, AND THE REASON THIS TEST WAS RENAMED. It is tempting to call last-N "val-free". As a
    FUNCTION OF A GIVEN TRACE it is -- that is exactly what this test pins. But the end-to-end
    procedure is NOT val-free, because patience-15 early stopping decides where the trace ENDS,
    and val decides the stopping point. Vaswani trained to a fixed budget with no early stopping,
    so their last-N genuinely never touched val; ours does, through the trace length. Do not quote
    last-N as the val-free rule -- the honest statement is that it does not RE-RANK by val.

    That mismatch is also why last-N is a poor fit here and is reported as a reference point
    rather than the headline: our tail sits ~15 epochs past the val optimum at ~80% of peak LR,
    where Vaswani's tail was the converged end of the run. The headline rule is greedy soup."""
    good = BFT._ensemble_index_sets([0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    bad = BFT._ensemble_index_sets([0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    assert good["ens_last3"] == [3, 4, 5] == bad["ens_last3"]
    assert good["ens_last5"] == [1, 2, 3, 4, 5] == bad["ens_last5"]


def test_lastn_is_shorter_than_n_on_a_short_trace_rather_than_erroring():
    sets = BFT._ensemble_index_sets([0.6, 0.7])
    assert sets["ens_last3"] == [0, 1]
    assert sets["ens_last5"] == [0, 1]


def test_lastn_skips_failed_epochs_rather_than_averaging_a_nan_in():
    """Same contract as every other rule: a non-finite val is a failed fit. 'Last 3' means the
    last 3 epochs that actually produced a model, not indices 3,4,5 whatever happened in them."""
    sets = BFT._ensemble_index_sets([0.6, 0.7, float("nan"), 0.65, 0.66])
    assert sets["ens_last3"] == [1, 3, 4]
