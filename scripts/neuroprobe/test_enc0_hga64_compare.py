"""Pin the aggregation, because both easy mistakes here are silent.

A wrong grid key yields an EMPTY grid (reported, survivable). A wrong macro order yields a
populated, plausible number that is not the leaderboard's — which is the one that gets quoted.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from enc0_hga64_compare import BOARD_TASKS, cell_macros, load, macro, paired


def _shard(tmp, mode, name, gk_to_task_auroc):
    """gk_to_task_auroc: {grid_key: {task: auroc}} -> one shard file, readout's own schema."""
    cells = {}
    for gk, per_task in gk_to_task_auroc.items():
        for task, v in per_task.items():
            cells.setdefault(task, {"cells": {}})["cells"][gk] = {"test": v}
    p = tmp / f"{mode}_{name}.json"
    p.write_text(json.dumps({"kind": mode, "name": name, "cells": cells}))
    return p


def test_load_selects_the_requested_grid_key_only(tmp_path):
    _shard(tmp_path, "ws", "s1_t1", {"enc0_elec|std": {"onset": 0.7},
                                     "fm:hga64t32:enc0_elec|std": {"onset": 0.6}})
    assert load(str(tmp_path), "ws", "enc0_elec|std") == {"s1_t1": {"onset": 0.7}}
    assert load(str(tmp_path), "ws", "fm:hga64t32:enc0_elec|std") == {"s1_t1": {"onset": 0.6}}


def test_missing_key_yields_empty_not_a_wrong_number(tmp_path):
    _shard(tmp_path, "ws", "s1_t1", {"enc0_elec|std": {"onset": 0.7}})
    assert load(str(tmp_path), "ws", "nope|std") == {}


def test_mode_prefix_isolates_regimes(tmp_path):
    _shard(tmp_path, "ws", "s1_t1", {"enc0_elec|std": {"onset": 0.7}})
    _shard(tmp_path, "cs", "s3_t0", {"enc0_elec|std": {"onset": 0.5}})
    assert set(load(str(tmp_path), "ws", "enc0_elec|std")) == {"s1_t1"}
    assert set(load(str(tmp_path), "cs", "enc0_elec|std")) == {"s3_t0"}


def test_macro_is_cohort_mean_then_task_mean_not_the_reverse():
    """With a hole the two orders DIFFER, and only one is the leaderboard's."""
    per_cell = {"a": {"onset": 1.0, "speech": 0.0}, "b": {"onset": 0.0}}  # b missing 'speech'
    # cohort-mean first: onset (1.0+0.0)/2 = 0.5, speech = 0.0 -> mean = 0.25
    assert macro(per_cell, ("onset", "speech")) == pytest.approx(0.25)
    # per-cell macro first would be: a=0.5, b=0.0 -> 0.25 here by coincidence, so use a case
    # where they genuinely diverge:
    per_cell2 = {"a": {"onset": 1.0, "speech": 1.0}, "b": {"onset": 0.0}}
    assert macro(per_cell2, ("onset", "speech")) == pytest.approx(0.75)   # (0.5 + 1.0)/2
    mean_of_cell_macros = np.mean(list(cell_macros(per_cell2, ("onset", "speech")).values()))
    assert mean_of_cell_macros == pytest.approx(0.5)                      # (1.0 + 0.0)/2
    assert macro(per_cell2, ("onset", "speech")) != pytest.approx(mean_of_cell_macros)


def test_macro_ignores_tasks_absent_everywhere():
    per_cell = {"a": {"onset": 0.8}}
    assert macro(per_cell, ("onset", "speech")) == pytest.approx(0.8)


def test_paired_uses_shared_cells_only_and_counts_wins():
    a = {"c1": 0.70, "c2": 0.60, "c3": 0.50}
    b = {"c1": 0.68, "c2": 0.62, "c4": 0.10}      # c3/c4 unshared -> dropped
    m, (lo, hi), n, wins, shared = paired(a, b)
    assert shared == ["c1", "c2"] and n == 2
    assert m == pytest.approx((0.02 + -0.02) / 2)
    assert wins == 1


def test_paired_ci_excludes_zero_for_a_consistent_effect():
    a = {f"c{i}": 0.60 + 0.01 for i in range(12)}
    b = {f"c{i}": 0.60 for i in range(12)}
    m, (lo, hi), n, wins, _ = paired(a, b)
    assert m == pytest.approx(0.01) and n == 12 and wins == 12
    assert lo > 0        # zero variance -> CI collapses onto the mean


def test_paired_ci_spans_zero_when_the_sign_flips():
    a = {"c1": 0.7, "c2": 0.5}
    b = {"c1": 0.6, "c2": 0.6}
    m, (lo, hi), n, wins, _ = paired(a, b)
    assert lo < 0 < hi and wins == 1


def test_board_tasks_is_exactly_15_unique():
    assert len(BOARD_TASKS) == 15 and len(set(BOARD_TASKS)) == 15
