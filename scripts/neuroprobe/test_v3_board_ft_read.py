"""The gates must REFUSE. A gate that only prints a warning is not a gate.

Every test here builds a synthetic arm and asserts the reader dies on it. The failure these guard
is not a crash -- it is a plausible-looking number printed from an arm that should never have been
read, which is precisely how a 6-cell subset gets quoted as a board macro.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "v3_board_ft_read.py")


def _mod():
    spec = importlib.util.spec_from_file_location("v3_board_ft_read", _SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["v3_board_ft_read"] = m
    spec.loader.exec_module(m)
    return m


RD = _mod()


def _row(cell, task, fold, a=0.70, c=0.71, top1=None, top3=0.715):
    return {"cell": cell, "task": task, "fold": fold,
            "test_frozen_vallam": a, "test_c": c,
            "soup_top1": c if top1 is None else top1, "ens_top3": top3}


def _arm(tmp_path, n_cells, n_tasks=15, folds=2, **kw):
    d = tmp_path / "arm"
    d.mkdir(exist_ok=True)
    for ci in range(n_cells):
        rows = [_row(f"S{ci}T0", f"task{t}", f, **kw)
                for t in range(n_tasks) for f in range(folds)]
        (d / f"ft_ws_{ci}.json").write_text(json.dumps(rows))
    return d


def _run(d, *extra):
    return subprocess.run([sys.executable, _SCRIPT, "--dir", str(d), "--regime", "ws", *extra],
                          capture_output=True, text=True)


def test_a_complete_12_cell_arm_is_read(tmp_path):
    r = _run(_arm(tmp_path, 12))
    assert r.returncode == 0, r.stderr
    assert "[GATE 1 PASS]" in r.stdout and "[GATE 2 PASS]" in r.stdout
    assert "n macro units = 180" in r.stdout


def test_partial_arm_is_REFUSED_not_quietly_averaged(tmp_path):
    """THE LOAD-BEARING TEST. Six cells is the shape every screening arm has, and it must not
    produce a printable macro."""
    r = _run(_arm(tmp_path, 6))
    assert r.returncode != 0, "a 6-cell arm MUST NOT read as a board macro"
    assert "GATE 1 FAIL" in (r.stdout + r.stderr)


def test_partial_arm_is_readable_only_when_explicitly_labelled_unquotable(tmp_path):
    r = _run(_arm(tmp_path, 6), "--allow-partial")
    assert r.returncode == 0
    assert "NOT QUOTABLE" in r.stdout


def test_missing_one_task_is_caught_even_though_all_cells_are_present(tmp_path):
    """Cell count alone is not completeness -- a cell can land with 14 of 15 tasks."""
    d = _arm(tmp_path, 12)
    rows = json.loads((d / "ft_ws_11.json").read_text())
    (d / "ft_ws_11.json").write_text(json.dumps([r for r in rows if r["task"] != "task14"]))
    r = _run(d)
    assert r.returncode != 0 and "GATE 1 FAIL" in (r.stdout + r.stderr)


def test_soup_top1_mismatch_is_fatal(tmp_path):
    """Averaging ONE state is the identity. Any delta means the snapshots are not the states the
    run selected on, so every averaging rule is reading the wrong thing."""
    r = _run(_arm(tmp_path, 12, top1=0.7101))
    assert r.returncode != 0
    assert "GATE 2 FAIL" in r.stdout or "gate 2 failed" in (r.stdout + r.stderr)


def test_A_moving_voids_the_arm(tmp_path):
    """A is epoch 0 with zero training steps. If it moved, something touched the frozen path."""
    r = _run(_arm(tmp_path, 12, a=0.60), "--expect-a", "0.6953")
    assert r.returncode != 0
    assert "GATE 3 FAIL" in r.stdout


def test_A_reproducing_passes_the_gate(tmp_path):
    r = _run(_arm(tmp_path, 12, a=0.6953), "--expect-a", "0.6953")
    assert r.returncode == 0 and "[GATE 3 PASS]" in r.stdout


def test_folds_are_AVERAGED_into_the_unit_not_counted_as_independent():
    """360 fold rows must yield 180 units. Treating folds as independent doubles n and inflates
    every p-value -- the two folds of one (cell, task) share a session."""
    units, M = RD.to_macro(
        [_row("S0T0", "t", 0, c=0.70), _row("S0T0", "t", 1, c=0.80)], ["test_c"])
    assert len(units) == 1
    assert M["test_c"][0] == pytest.approx(0.75)


def test_truncated_shard_is_surfaced_not_skipped(tmp_path):
    """A half-written shard must not silently shrink the arm -- that would let GATE 1 pass on an
    arm that is actually incomplete."""
    d = _arm(tmp_path, 12)
    (d / "ft_ws_3.json").write_text('[{"cell": "S3T0",')
    r = _run(d)
    assert r.returncode != 0
    assert "not valid JSON" in (r.stdout + r.stderr)
