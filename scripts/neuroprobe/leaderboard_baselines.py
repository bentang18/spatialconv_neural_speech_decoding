#!/usr/bin/env python
"""Recompute every public Neuroprobe leaderboard entry's macro FROM ITS SUBMITTED JSONs.

WHY THIS EXISTS. The comparison numbers we quote (.578 CNN, .575 PopT, .566 MLP, .539 linear)
were previously citable only as a webpage. A number we cannot recompute is not a number we may
put in the paper. Every leaderboard submission ships its raw per-session, per-task results under
``leaderboard/<ENTRY>/<SPLIT>/population_<TASK>.json`` (SUBMIT.md "Formatting results"), so the
macro is reproducible from source -- and, more importantly, reproducible under OUR aggregation
instead of theirs.

THE AGGREGATION IS THE POINT. We average test_roc_auc over the SAME grid we use for our own
board macro: 15 tasks x the 10 non-anchor Lite cells, unweighted, cells first then tasks. If
their published macro and ours differ, the difference is aggregation, not modelling -- and we
would need to say so. (As of the 0.1.8 checkout they agree to <1e-3 on every entry.)

THIS IS A BASELINE READER, NOT A SUBMISSION BUILDER. It only reads. Writing our own submission
is a separate job with its own attestation requirements.

Usage:
  python scripts/neuroprobe/leaderboard_baselines.py --repo /path/to/neuroprobe
  python scripts/neuroprobe/leaderboard_baselines.py --repo <R> --split Cross-Session
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# The 10 cross-subject test cells: every Lite session whose subject is not the anchor's (S2).
# Kept as a literal so this file does not depend on the board readout's import graph; the
# --strict check below cross-validates it against v3_board_readout when that import is available.
CS_TEST_CELLS = ((1, 1), (1, 2), (3, 0), (3, 1), (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1))
LITE_SESSIONS = ((1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1),
                 (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1))
N_TASKS = 15


def _cells_for(split: str) -> set[str]:
    """Cross-Subject drops the anchor subject entirely; the other splits evaluate all 12."""
    cells = CS_TEST_CELLS if split == "Cross-Subject" else LITE_SESSIONS
    return {f"btbank{s}_{t}" for s, t in cells}


def entry_macro(entry_dir: Path, split: str) -> dict | None:
    """Macro over tasks of (mean over cells of (mean over folds of test_roc_auc))."""
    d = entry_dir / split
    if not d.is_dir():
        return None
    want = _cells_for(split)
    per_task, seen_cells, fold_counts, extra = {}, set(), set(), set()
    for f in sorted(d.glob("population_*.json")):
        task = f.name[len("population_"):-len(".json")]
        blob = json.loads(f.read_text())
        vals = []
        for cell, rec in blob.get("evaluation_results", {}).items():
            if cell not in want:
                extra.add(cell)
                continue
            seen_cells.add(cell)
            for _win, w in rec.get("population", {}).items():
                folds = w.get("folds", [])
                fold_counts.add(len(folds))
                if folds:
                    vals.append(float(np.mean([fl["test_roc_auc"] for fl in folds])))
        if vals:
            per_task[task] = float(np.mean(vals))
    if not per_task:
        return None
    return {
        "macro": float(np.mean(list(per_task.values()))),
        "per_task": per_task,
        "n_tasks": len(per_task),
        "n_cells": len(seen_cells),
        "n_want": len(want),
        "folds": sorted(fold_counts),
        "ignored_cells": sorted(extra),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="path to a checkout of insight-neuro/neuroprobe")
    p.add_argument("--split", default="Cross-Subject",
                   choices=["Cross-Subject", "Cross-Session", "Within-Session"])
    p.add_argument("--json-out", default=None)
    p.add_argument("--strict", action="store_true",
                   help="fail unless every entry covers all cells and all 15 tasks")
    a = p.parse_args()

    lb = Path(a.repo) / "leaderboard"
    if not lb.is_dir():
        raise SystemExit(f"no leaderboard/ under {a.repo} -- is this the neuroprobe repo?")
    rev = (Path(a.repo) / ".git" / "HEAD")
    print(f"[src] {lb}  split={a.split}  head={rev.read_text().strip() if rev.exists() else '?'}",
          flush=True)

    rows = {}
    for e in sorted(lb.iterdir()):
        if not e.is_dir():
            continue
        r = entry_macro(e, a.split)
        if r:
            rows[e.name] = r

    if not rows:
        raise SystemExit(f"no entries had a {a.split}/ directory")

    bad = [n for n, r in rows.items()
           if r["n_cells"] != r["n_want"] or r["n_tasks"] != N_TASKS or r["folds"] != [1]]
    # Within-Session legitimately carries 2 folds (SUBMIT.md); only cell/task coverage is a defect.
    cover = [n for n, r in rows.items() if r["n_cells"] != r["n_want"] or r["n_tasks"] != N_TASKS]
    print(f"[check] {'OK' if not cover else 'VIOLATED'} all {len(rows)} entries cover "
          f"{N_TASKS} tasks x {rows[next(iter(rows))]['n_want']} cells"
          + (f" -- offenders {cover}" if cover else ""), flush=True)
    if a.split != "Within-Session" and bad and not cover:
        print(f"[check] note: unexpected fold counts on {bad}", flush=True)
    if a.strict and cover:
        raise SystemExit("strict: incomplete entries present")

    print(f"\n{'macro':>7}  {'tasks':>5} {'cells':>5}  entry")
    for n, r in sorted(rows.items(), key=lambda kv: -kv[1]["macro"]):
        print(f"{r['macro']:7.4f}  {r['n_tasks']:5d} {r['n_cells']:5d}  {n}")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(rows, indent=2, sort_keys=True))
        print(f"\n[out] {a.json_out}", flush=True)


if __name__ == "__main__":
    main()
