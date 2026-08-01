"""The RBF reader's gates must REFUSE, not warn.

A gate that prints a warning and carries on is not a gate -- the failure being guarded is a
plausible-looking macro printed from an arm that should never have been read.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "v3_board_rbf_read.py")

TAP = "enc12_elec"
TASKS = [f"task{i}" for i in range(15)]


def _shard(cell, *, lin=0.70, rbf=0.71, tasks=TASKS):
    return {"kind": "ws", "name": cell,
            "cells": {t: {"cells": {f"{TAP}|std": {"test": lin},
                                    f"{TAP}|std_rbf": {"test": rbf},
                                    f"{TAP}|std_rbfpc32": {"test": rbf - 0.005}}}
                      for t in tasks}}


def _arm(tmp_path, n_cells=12, **kw):
    d = tmp_path / "arm"
    d.mkdir(exist_ok=True)
    for i in range(n_cells):
        (d / f"ws_{i}.json").write_text(json.dumps(_shard(f"S{i}T0", **kw)))
    return d


def _run(d, *extra):
    return subprocess.run([sys.executable, _SCRIPT, "--dir", str(d), "--regime", "ws", *extra],
                          capture_output=True, text=True)


def test_a_complete_arm_reads(tmp_path):
    r = _run(_arm(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "[GATE 1 PASS]" in r.stdout
    assert "n macro units = 180" in r.stdout


def test_partial_arm_is_REFUSED(tmp_path):
    """THE LOAD-BEARING TEST. Six cells is the shape a mid-flight array has, and it must not
    produce a printable macro."""
    r = _run(_arm(tmp_path, n_cells=6))
    assert r.returncode != 0
    assert "GATE 1 FAIL" in (r.stdout + r.stderr)


def test_missing_one_task_is_caught_though_every_cell_is_present(tmp_path):
    d = _arm(tmp_path)
    (d / "ws_11.json").write_text(json.dumps(_shard("S11T0", tasks=TASKS[:-1])))
    r = _run(d)
    assert r.returncode != 0 and "GATE 1 FAIL" in (r.stdout + r.stderr)


def test_a_moved_linear_control_VOIDS_the_arm(tmp_path):
    """--rbf only appends columns, so the linear path cannot move. If it did, the shared code was
    touched and every delta is meaningless."""
    r = _run(_arm(tmp_path, lin=0.60), "--expect-linear", "0.6953")
    assert r.returncode != 0
    assert "GATE 2 FAIL" in r.stdout and "VOID" in (r.stdout + r.stderr)


def test_a_reproducing_control_passes(tmp_path):
    r = _run(_arm(tmp_path, lin=0.6953), "--expect-linear", "0.6953")
    assert r.returncode == 0 and "[GATE 2 PASS]" in r.stdout


def test_truncated_shard_is_surfaced_not_skipped(tmp_path):
    """A skipped shard would shrink the arm silently and let the completeness gate pass."""
    d = _arm(tmp_path)
    (d / "ws_3.json").write_text('{"kind": "ws",')
    r = _run(d)
    assert r.returncode != 0 and "not valid JSON" in (r.stdout + r.stderr)


def test_missing_linear_column_is_fatal(tmp_path):
    """Without the control there is nothing to compare against, so there is no result."""
    d = tmp_path / "nolin"
    d.mkdir()
    for i in range(12):
        sh = _shard(f"S{i}T0")
        for t in sh["cells"]:
            del sh["cells"][t]["cells"][f"{TAP}|std"]
        (d / f"ws_{i}.json").write_text(json.dumps(sh))
    r = _run(d)
    assert r.returncode != 0 and "no control" in (r.stdout + r.stderr)


def test_the_best_column_is_labelled_an_oracle(tmp_path):
    """Max-over-columns on test is a ceiling. If that is printed without the caveat it will be
    quoted as a result."""
    r = _run(_arm(tmp_path))
    assert "ORACLE" in r.stdout


def test_a_column_missing_units_is_not_silently_compared(tmp_path):
    """A nonlinear column short some units would otherwise be averaged over a different, easier
    set than the control it is paired against."""
    d = _arm(tmp_path)
    sh = _shard("S5T0")
    for t in TASKS[:3]:
        del sh["cells"][t]["cells"][f"{TAP}|std_rbf"]
    (d / "ws_5.json").write_text(json.dumps(sh))
    r = _run(d)
    assert r.returncode == 0
    assert "not compared" in r.stdout
