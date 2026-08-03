"""Tests for the across-task decomposition of the across-N constant.

The whole file exists to distinguish two shapes -- a line through the origin from a flat line -- so
the tests plant each shape and check the verdict comes back right. A test suite that only checked it
runs would not have caught the thing this analysis is for.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.neuroprobe.r31_two_axes import (
    EVENT, LEDGER_K, LEVEL, VISUAL, ledger_check, selectivity_check, two_axes, verdict,
)

NONVIS = tuple(EVENT) + tuple(t for t in LEVEL if t not in VISUAL)   # the 11-task argument set


def _pts(gap_of, n_subj=6, tasks=("a", "b", "c", "d", "e"), ns=(16, 64, 256, 1024), noise=0.0,
         seed=0):
    """A synthetic point list in the shape `_panel` consumes, with a PLANTED per-task gap.

    Headroom is spread across tasks so a slope is identifiable at all; `gap_of(headroom, task)` is
    the law being planted -- it takes the task name so a family-DEPENDENT law can be planted too,
    which is what `selectivity_check` has to be able to see. One cell per subject keeps the
    cell-mean layer a no-op so the test targets the fit, not the aggregation, which `_panel` is
    already tested for.
    """
    rng = np.random.default_rng(seed)
    heads = np.linspace(0.02, 0.30, len(tasks))
    out = []
    for si in range(n_subj):
        for n in ns:
            for t, h in zip(tasks, heads):
                base = 0.5 + h + (rng.normal(0, noise) if noise else 0.0)
                for tap, v in (("enc0", base), ("enc12", base + gap_of(h, t))):
                    out.append({"cell": f"S{si}_T1", "task": t, "tap": tap, "col": "trainonly",
                                "n_bucket": n, "n_is_full": False, "test": v})
    return out


def test_planted_pure_multiplier_reads_THROUGH_THE_ORIGIN() -> None:
    r = two_axes(_pts(lambda h, t: 0.20 * h), "enc0", "enc12", nboot=200)
    assert verdict(r).startswith("THROUGH THE ORIGIN")
    assert r["K"] == pytest.approx(0.20, abs=1e-6)
    assert r["A"] == pytest.approx(0.0, abs=1e-6)


def test_planted_flat_constant_reads_FLAT() -> None:
    """The refutation case: if every task gained the same amount, the across-task gain law would be
    an intercept and not a multiplier, and this must say so."""
    r = two_axes(_pts(lambda h, t: 0.03), "enc0", "enc12", nboot=200)
    assert verdict(r).startswith("FLAT")
    assert r["K"] == pytest.approx(0.0, abs=1e-6)
    assert r["A"] == pytest.approx(0.03, abs=1e-6)


def test_planted_offset_plus_slope_reads_BOTH() -> None:
    r = two_axes(_pts(lambda h, t: 0.02 + 0.20 * h, noise=0.004), "enc0", "enc12", nboot=400)
    assert verdict(r).startswith("BOTH")


def test_topN_headroom_matches_meanN_when_enc0_does_not_move_with_N() -> None:
    """The two headroom definitions differ ONLY because enc0 grows with N. On a curve where it does
    not, they must agree exactly -- otherwise the robustness knob is measuring its own bug."""
    pts = _pts(lambda h, t: 0.20 * h)
    a = two_axes(pts, "enc0", "enc12", xmode="meanN", nboot=100)
    b = two_axes(pts, "enc0", "enc12", xmode="topN", nboot=100)
    assert a["K"] == pytest.approx(b["K"], abs=1e-9)
    assert a["A"] == pytest.approx(b["A"], abs=1e-9)


def test_dropping_tasks_actually_drops_them() -> None:
    tasks = ("onset", "speech", "pitch") + VISUAL
    r = two_axes(_pts(lambda h, t: 0.20 * h, tasks=tasks), "enc0", "enc12",
                 tasks_drop=VISUAL, nboot=100)
    assert set(r["tasks"]) == {"onset", "speech", "pitch"}
    assert len(r["headroom"]) == 3


def test_ledger_check_finds_a_planted_order_and_a_permuted_one_does_not() -> None:
    """Both directions: the rank test must fire on the ledger's own ordering and must NOT fire on a
    scrambled one, or its p-value is decoration."""
    tasks = [t for t in LEDGER_K if t not in VISUAL]
    heads = np.linspace(0.02, 0.30, len(tasks))
    real = {"tasks": tasks, "headroom": heads,
            "gap": np.array([(LEDGER_K[t] - 1) * h for t, h in zip(tasks, heads)])}
    assert ledger_check(real, nperm=2000)["p"] < 0.01

    scrambled = dict(real, gap=np.random.default_rng(7).permutation(real["gap"]))
    assert ledger_check(scrambled, nperm=2000)["p"] > 0.05


def test_task_axis_mislabelling_is_caught() -> None:
    """`per_task_panel` reconstructs the task order `_panel` used rather than being handed it. If
    that reconstruction ever drifts, every per-task number would be silently mislabelled, so the
    assert has to be real -- feed it a point list whose filter excludes a task and check it trips."""
    from scripts.neuroprobe.r31_two_axes import per_task_panel
    pts = _pts(lambda h, t: 0.20 * h)
    # A task present ONLY under a filtered-out column: `_panel` drops it, the reconstruction must too
    pts.append({"cell": "S0_T1", "task": "zzz_other_col", "tap": "enc0", "col": "both",
                "n_bucket": 16, "n_is_full": False, "test": 0.6})
    _, _, _, _, tasks = per_task_panel(pts, "enc0", "enc12")
    assert "zzz_other_col" not in tasks


def test_selectivity_fires_on_a_planted_event_over_level_split() -> None:
    """Plant the ledger's own shape -- level tasks gain LESS than the common slope -- and the check
    must put both level tasks on the most-negative residuals, with the exact null it will be quoted
    against."""
    r = two_axes(_pts(lambda h, t: 0.20 * h - (0.02 if t in LEVEL else 0.0), tasks=NONVIS),
                 "enc0", "enc12", nboot=100)
    s = selectivity_check(r)
    assert s["testable"] and s["all_bottom"]
    assert set(s["level_ranks"]) == {"volume", "pitch"}
    assert s["p_exact"] == pytest.approx(1 / 55, abs=1e-9)          # C(11,2)


def test_selectivity_does_NOT_fire_when_the_law_is_family_blind() -> None:
    """The other direction. If every task follows one slope, the residuals are numerical dust and
    the level tasks have no reason to sit at the bottom -- if this still fired, the CS result would
    be an artefact of the test rather than a finding."""
    r = two_axes(_pts(lambda h, t: 0.20 * h, tasks=NONVIS, noise=0.004), "enc0", "enc12", nboot=100)
    s = selectivity_check(r)
    assert s["testable"] and not s["all_bottom"]


def test_selectivity_refuses_when_a_task_is_in_neither_family() -> None:
    """The event/level cut comes from the label definitions. A task the cut does not name means the
    grouping is incomplete, and reporting 1/C(n, nl) then would understate the null."""
    r = two_axes(_pts(lambda h, t: 0.20 * h, tasks=NONVIS + ("mystery_task",)),
                 "enc0", "enc12", nboot=100)
    assert selectivity_check(r)["testable"] is False
