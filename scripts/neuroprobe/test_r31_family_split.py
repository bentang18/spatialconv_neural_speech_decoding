"""Tests for the canonical-arm recomputation of the event/level split.

The point of the file under test is to replace a memo number with a computed one, so the tests
that matter are (a) the estimator is the SAME estimator the ledger used -- a silent drift would
publish a second, different `k` under the same name -- and (b) the split test can come back
negative, since a test that only ever fires is not evidence.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.neuroprobe.r31_family_split import (
    _k, anchor_cells, gap_ci, split_stats,
)
from scripts.neuroprobe.r31_two_axes import EVENT, LEVEL, VISUAL

NONVIS = tuple(EVENT) + tuple(t for t in LEVEL if t not in VISUAL)


def _cells(k_of, tasks=NONVIS, n_subj=5, headrooms=(0.05, 0.12, 0.20)):
    """task -> cell -> (enc0, enc12) with a PLANTED per-task multiplier `k_of(task)`."""
    return {t: {f"S{s}T1_{i}": (0.5 + h, 0.5 + h * k_of(t))
                for s in range(n_subj) for i, h in enumerate(headrooms)}
            for t in tasks}


def test_estimator_is_bit_identical_to_the_ledgers_own_k() -> None:
    """PARITY. If this drifts from `paper_figs_r6._k`, the recomputation is a different number
    wearing the same name, which is exactly the failure the file exists to prevent."""
    from scripts.neuroprobe.paper_figs_r6 import _k as ledger_k
    rng = np.random.default_rng(0)
    names = [f"c{i}" for i in range(9)]
    x = {c: float(v) for c, v in zip(names, 0.5 + rng.uniform(0.001, 0.3, len(names)))}
    y = {c: float(v) for c, v in zip(names, 0.5 + rng.uniform(0.001, 0.4, len(names)))}
    assert _k([(x[c], y[c]) for c in names]) == pytest.approx(
        ledger_k({"enc0": x, "enc12": y}, "enc12"), abs=1e-12)


def test_planted_clean_split_is_detected_with_its_exact_null() -> None:
    cells = _cells(lambda t: 1.30 if t in EVENT else 0.90)
    s = split_stats(cells, list(NONVIS), nperm=2000)
    assert s["clean_split"] is True
    assert s["gap"] == pytest.approx(0.40, abs=1e-9)
    assert s["p_rank"] == pytest.approx(1 / math.comb(11, 2), abs=1e-12)
    assert s["p_gap"] < 0.05


def test_no_planted_split_comes_back_negative() -> None:
    """The other direction. A test that fires on family-blind data is decoration."""
    cells = _cells(lambda t: 1.15)
    s = split_stats(cells, list(NONVIS), nperm=2000)
    assert s["clean_split"] is False
    assert s["gap"] == pytest.approx(0.0, abs=1e-9)
    assert s["p_gap"] > 0.05


def test_reversed_split_is_not_reported_as_clean() -> None:
    """Level ABOVE event must not read as the finding -- `clean_split` is directional."""
    cells = _cells(lambda t: 0.90 if t in EVENT else 1.30)
    s = split_stats(cells, list(NONVIS), nperm=500)
    assert s["clean_split"] is False
    assert s["gap"] < 0


def test_anchor_cells_drops_a_cell_missing_either_tap() -> None:
    """k is a slope of one tap on the other, so a half-present cell is not an observation."""
    pts = [{"task": "onset", "cell": "S1T1", "tap": "enc0", "col": "trainonly",
            "n_is_full": True, "test": 0.6},
           {"task": "onset", "cell": "S1T1", "tap": "enc12", "col": "trainonly",
            "n_is_full": True, "test": 0.7},
           {"task": "onset", "cell": "S2T1", "tap": "enc0", "col": "trainonly",
            "n_is_full": True, "test": 0.6}]                      # enc12 missing -> dropped
    got = anchor_cells(pts, "enc0", "enc12")
    assert set(got["onset"]) == {"S1T1"}


def test_anchor_cells_ignores_non_anchor_points_and_other_columns() -> None:
    base = {"task": "onset", "cell": "S1T1", "col": "trainonly", "n_is_full": True}
    pts = [{**base, "tap": "enc0", "test": 0.6}, {**base, "tap": "enc12", "test": 0.7},
           {**base, "tap": "enc0", "test": 0.9, "n_is_full": False},      # mid-sweep
           {**base, "tap": "enc12", "test": 0.9, "col": "both"}]          # other column
    got = anchor_cells(pts, "enc0", "enc12")
    assert got["onset"]["S1T1"] == (0.6, 0.7)


def test_m2b_parity_reports_drift_rather_than_hiding_it() -> None:
    """The parity check is the reason this is a recomputation of a KNOWN quantity and not a new
    number. If it silently tolerated drift it would be worse than absent, so plant a drifted k and
    check the reported difference actually exceeds the tolerance."""
    from scripts.neuroprobe.r31_family_split import M2B_CS, M2B_TOL, m2b_parity
    clean = m2b_parity(dict(M2B_CS))
    assert len(clean) == len(M2B_CS)
    assert max(d for *_, d in clean) == pytest.approx(0.0, abs=1e-12)

    drifted = m2b_parity({**M2B_CS, "pitch": M2B_CS["pitch"] + 0.25})
    assert max(d for *_, d in drifted) > M2B_TOL


def test_gap_ci_resamples_subjects_not_cells() -> None:
    """Two sessions of one patient are one draw. With every subject identical, resampling
    subjects can only ever return the same gap, so the CI must collapse to a point -- a
    cell-level bootstrap would show spread here."""
    cells = _cells(lambda t: 1.30 if t in EVENT else 0.90, n_subj=4)
    lo, hi, nsub = gap_ci(cells, list(NONVIS), nboot=200)
    assert nsub == 4
    assert lo == pytest.approx(hi, abs=1e-9)
