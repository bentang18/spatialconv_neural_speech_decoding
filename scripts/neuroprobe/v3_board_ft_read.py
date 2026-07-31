"""Read a partial-FT arm as a BOARD MACRO, or refuse.

WHY THIS EXISTS. Every FT number so far has been read with an ad-hoc snippet over whatever JSON
shards happened to be on disk, and that is exactly how a 6-cell subset gets quoted as a board
number. The levels move enormously with the cell set -- the SAME frozen entry reads .7444 on 111
units and .6953 on 180 -- so a partial read is not "approximately right", it is a different
quantity. This script REFUSES to print a macro unless the arm is complete.

THE MACRO UNIT IS (cell, task) WITH FOLDS AVERAGED. ws/csession are the 12 Lite sessions x 15 board
tasks = 180 units; cs is 10 cells x 15 = 150. Folds are averaged INTO the unit, never treated as
independent rows -- the two folds of one (cell, task) share a session and are not exchangeable with
it.

GATES, ALL THREE ENFORCED BEFORE ANY NUMBER IS PRINTED:
  1. COMPLETENESS -- every (cell, task) present, no partial cells.
  2. soup_top1 == test_c BIT-FOR-BIT. Averaging a single state is the identity, so a non-zero delta
     means the per-epoch snapshots are not the states the run actually selected on.
  3. A (frozen, val-selected ridge) reproduces the published entry when --expect-a is given. A is
     computed at epoch 0 with zero training steps, so NO optimizer or schedule change can move it.
     If A moved, the arm is void and the deltas are meaningless.

The gates are the point. A number that clears them is quotable; one that does not is a bug report.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

# v3_board_readout.py:98-118 -- ws/csession are the 12 Lite sessions, cs the 10 CS test cells.
N_CELLS = {"ws": 12, "csession": 12, "cs": 10}
N_TASKS = 15

# Every rule the FT driver dumps, ranked by the 07-31 measurement. `soup_top1` is the identity gate,
# not a candidate.
RULES = ["ens_top3", "soup_top3", "ens_last5", "ens_last10", "ens_last15",
         "soup_last5", "soup_last10", "soup_last15", "soup_ema80", "soup_ema90", "soup_ema95"]


def load_rows(d: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        with open(f) as fh:
            try:
                payload = json.load(fh)
            except json.JSONDecodeError:
                # A shard still being written is INCOMPLETE, not empty -- surface it, never skip
                # silently, or the completeness gate would pass on a truncated arm.
                raise SystemExit(f"[FATAL] {f} is not valid JSON (still being written?)")
        rows.extend(payload)
    return rows


def to_macro(rows: list[dict], keys: list[str]) -> tuple[list[tuple[str, str]], dict]:
    """Average folds into the (cell, task) macro unit."""
    acc: dict[str, dict] = {k: defaultdict(list) for k in keys}
    for r in rows:
        for k in keys:
            acc[k][(r["cell"], r["task"])].append(r[k])
    units = sorted(acc[keys[0]])
    return units, {k: np.array([np.mean(acc[k][u]) for u in units]) for k in keys}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="directory of ft_<regime>_<cell>.json shards")
    p.add_argument("--regime", required=True, choices=sorted(N_CELLS))
    p.add_argument("--expect-a", type=float, default=None,
                   help="published frozen macro. A is epoch 0 with zero training steps, so it MUST "
                        "reproduce exactly; a mismatch voids the arm.")
    p.add_argument("--tol", type=float, default=5e-4, help="tolerance for the --expect-a gate")
    p.add_argument("--allow-partial", action="store_true",
                   help="print a SUBSET read and label it unquotable. Never use for a board number.")
    args = p.parse_args()

    rows = load_rows(args.dir)
    if not rows:
        raise SystemExit(f"[FATAL] no rows in {args.dir}")
    cells = sorted({r["cell"] for r in rows})
    want_cells, want_units = N_CELLS[args.regime], N_CELLS[args.regime] * N_TASKS
    units_present = len({(r["cell"], r["task"]) for r in rows})

    print(f"=== {args.regime} FT board read :: {args.dir}")
    print(f"    cells {len(cells)}/{want_cells}   units {units_present}/{want_units}   "
          f"fold rows {len(rows)}")

    # ---- GATE 1: completeness ------------------------------------------------------------------
    complete = len(cells) == want_cells and units_present == want_units
    if not complete:
        missing = want_units - units_present
        msg = (f"[GATE 1 FAIL] arm is INCOMPLETE -- {missing} macro units missing. "
               f"A partial read is a DIFFERENT QUANTITY, not an approximation "
               f"(the same frozen entry reads .7444 on 111 units and .6953 on 180).")
        if not args.allow_partial:
            raise SystemExit(msg + "\n  Pass --allow-partial to see a subset read, clearly labelled.")
        print(msg + "\n  --allow-partial given: numbers below are a SUBSET, 🚫 NOT QUOTABLE.")
    else:
        print("[GATE 1 PASS] all cells and units present")

    keys = ["test_frozen_vallam", "test_c", "soup_top1"] + [r for r in RULES if r in rows[0]]
    units, M = to_macro(rows, keys)
    a, c = M["test_frozen_vallam"], M["test_c"]

    # ---- GATE 2: soup_top1 is the identity ------------------------------------------------------
    dmax = float(np.abs(np.array([r["soup_top1"] - r["test_c"] for r in rows])).max())
    print(f"[GATE 2 {'PASS' if dmax == 0.0 else 'FAIL'}] soup_top1 == test_c, max|d| = {dmax:.3e}"
          + ("" if dmax == 0.0 else "  <-- per-epoch snapshots are NOT the selected states"))
    if dmax != 0.0:
        raise SystemExit("[FATAL] gate 2 failed; the averaging rules are reading the wrong states.")

    # ---- GATE 3: A reproduces the published frozen entry ----------------------------------------
    if args.expect_a is not None:
        off = abs(float(a.mean()) - args.expect_a)
        ok = off <= args.tol
        print(f"[GATE 3 {'PASS' if ok else 'FAIL'}] A = {a.mean():.4f} vs published "
              f"{args.expect_a:.4f} (off {off:.4f}, tol {args.tol})")
        if not ok and complete:
            raise SystemExit("[FATAL] A moved. A is epoch 0 with ZERO training steps -- no "
                             "optimizer or schedule change can touch it. The arm is VOID.")
    else:
        print(f"[GATE 3 SKIP] no --expect-a given; A = {a.mean():.4f}")

    # ---- the read ------------------------------------------------------------------------------
    print(f"\n{'quantity':16s} {'macro':>8s} {'vs A':>9s} {'+/-':>10s} {'p':>9s}")
    print(f"{'A (frozen)':16s} {a.mean():8.4f} {'--':>9s} {'--':>10s} {'--':>9s}")
    for k in ["test_c"] + [r for r in RULES if r in M]:
        x = M[k]
        d = x - a
        p_ = stats.wilcoxon(x, a).pvalue if np.any(d != 0) else 1.0
        name = "C (FT, argmax)" if k == "test_c" else k
        print(f"{name:16s} {x.mean():8.4f} {d.mean():+9.4f} "
              f"{f'{int((d > 0).sum())}+/{int((d < 0).sum())}-':>10s} {p_:9.2g}")

    best = max((r for r in RULES if r in M), key=lambda r: M[r].mean(), default=None)
    if best is not None:
        d = M[best] - c
        p_ = stats.wilcoxon(M[best], c).pvalue if np.any(d != 0) else 1.0
        print(f"\nbest rule = {best}: {M[best].mean():.4f}, vs C {d.mean():+.4f} (p={p_:.2g})")
    print(f"\nn macro units = {len(units)}"
          + ("" if complete else "   🚫 SUBSET -- NOT A BOARD NUMBER"))


if __name__ == "__main__":
    main()
