"""Read the RBF kernel columns against the linear control as a BOARD MACRO, or refuse.

WHY THIS EXISTS: the same lesson the FT reader was written for. Levels move enormously with the
cell set, so a partial read is a DIFFERENT QUANTITY rather than an approximation, and the only way
a nonlinear column earns a claim is paired against the linear one over the COMPLETE board.

THE CONTROL IS THE GATE. `<tap>|std` is the untouched published path -- --rbf only appends
columns. So the linear column MUST reproduce the published entry exactly; if it moved, something
touched the shared path and every delta below is meaningless.

MACRO UNIT = (cell, task). The shard already averages the board's two folds inside _ws_cell, so a
shard entry IS the unit. ws/csession = 12 cells x 15 tasks = 180; cs = 10 x 15 = 150.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

# v3_board_readout.py:100-104 -- the Lite sessions and the CS test cells.
N_CELLS = {"ws": 12, "csession": 12, "cs": 10}
N_TASKS = 15
# v3_board_readout.py:115-117 -- WS/CSession report the electrode taps, CS the parcel ones.
DEFAULT_TAP = {"ws": "enc12_elec", "csession": "enc12_elec", "cs": "enc12"}


def load_units(shard_dir: str, regime: str, tap: str):
    """{norm: {(cell, task): test AUROC}} for one tap, straight off the shard files."""
    out: dict[str, dict] = defaultdict(dict)
    files = sorted(glob.glob(os.path.join(shard_dir, f"{regime}_*.json")))
    if not files:
        raise SystemExit(f"[FATAL] no {regime}_*.json shards in {shard_dir}")
    for path in files:
        with open(path) as fh:
            try:
                sh = json.load(fh)
            except json.JSONDecodeError:
                # A half-written shard is INCOMPLETE, not empty. Skipping it silently would let
                # the completeness gate pass on an arm that is short a cell.
                raise SystemExit(f"[FATAL] {path} is not valid JSON (still being written?)")
        cell = sh["name"]
        for task, val in (sh.get("cells") or {}).items():
            for gk, s in (val.get("cells") or {}).items():
                t, _, norm = gk.partition("|")
                if t == tap:
                    out[norm][(cell, task)] = float(s["test"])
    if not out:
        raise SystemExit(f"[FATAL] tap {tap!r} not present in any {regime} shard")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="shard dir written by --shard-dir")
    p.add_argument("--regime", required=True, choices=sorted(N_CELLS))
    p.add_argument("--tap", default=None, help="default: the regime's reported unit")
    p.add_argument("--expect-linear", type=float, default=None,
                   help="published macro for '<tap>|std'. --rbf only APPENDS columns, so the "
                        "linear control cannot move; a mismatch voids the arm.")
    p.add_argument("--tol", type=float, default=5e-4)
    p.add_argument("--allow-partial", action="store_true",
                   help="print a SUBSET read, labelled unquotable. Never for a board number.")
    args = p.parse_args()

    tap = args.tap or DEFAULT_TAP[args.regime]
    by_norm = load_units(args.dir, args.regime, tap)
    if "std" not in by_norm:
        raise SystemExit(f"[FATAL] no linear '{tap}|std' column -- there is no control to compare to")

    want = N_CELLS[args.regime] * N_TASKS
    lin = by_norm["std"]
    cells = sorted({c for c, _ in lin})
    print(f"=== {args.regime} RBF board read :: {args.dir}")
    print(f"    tap {tap}   cells {len(cells)}/{N_CELLS[args.regime]}   units {len(lin)}/{want}")

    # ---- GATE 1: completeness -------------------------------------------------------------------
    complete = len(cells) == N_CELLS[args.regime] and len(lin) == want
    if not complete:
        msg = (f"[GATE 1 FAIL] arm is INCOMPLETE -- {want - len(lin)} macro units missing. "
               f"A partial read is a DIFFERENT QUANTITY, not an approximation.")
        if not args.allow_partial:
            raise SystemExit(msg + "\n  Pass --allow-partial for a labelled subset read.")
        print(msg + "\n  --allow-partial given: numbers below are a SUBSET, NOT QUOTABLE.")
    else:
        print("[GATE 1 PASS] all cells and tasks present")

    # ---- GATE 2: the linear control reproduces --------------------------------------------------
    lin_mean = float(np.mean(list(lin.values())))
    if args.expect_linear is not None:
        off = abs(lin_mean - args.expect_linear)
        ok = off <= args.tol
        print(f"[GATE 2 {'PASS' if ok else 'FAIL'}] linear = {lin_mean:.4f} vs published "
              f"{args.expect_linear:.4f} (off {off:.4f}, tol {args.tol})")
        if not ok and complete:
            raise SystemExit("[FATAL] the CONTROL moved. --rbf only appends columns, so the linear "
                             "path cannot change. The arm is VOID.")
    else:
        print(f"[GATE 2 SKIP] no --expect-linear given; linear = {lin_mean:.4f}")

    # ---- the read -------------------------------------------------------------------------------
    units = sorted(lin)
    a = np.array([lin[u] for u in units])
    print(f"\n{'column':16s} {'macro':>8s} {'vs linear':>10s} {'+/-':>10s} {'p':>9s}")
    print(f"{'std (linear)':16s} {a.mean():8.4f} {'--':>10s} {'--':>10s} {'--':>9s}")
    rows = []
    for norm in sorted(k for k in by_norm if k != "std"):
        col = by_norm[norm]
        if not set(units) <= set(col):
            print(f"{norm:16s} {'--':>8s}  MISSING {want - len(col)} units, not compared")
            continue
        x = np.array([col[u] for u in units])
        d = x - a
        p_ = stats.wilcoxon(x, a).pvalue if np.any(d != 0) else 1.0
        print(f"{norm:16s} {x.mean():8.4f} {d.mean():+10.4f} "
              f"{f'{int((d > 0).sum())}+/{int((d < 0).sum())}-':>10s} {p_:9.2g}")
        rows.append((norm, x.mean(), d.mean(), p_))

    if rows:
        best = max(rows, key=lambda r: r[1])
        print(f"\nbest nonlinear column = {best[0]}: {best[1]:.4f} "
              f"({best[2]:+.4f} vs linear, p={best[3]:.2g})")
        # The honest bar, stated with the number rather than left to the reader. Selecting the best
        # of several columns on TEST is an oracle; a column only earns a claim if it was going to
        # be reported anyway and clears the pairing, not because it won a scan.
        print("  ⚠️ this is the MAX OVER COLUMNS on test — an ORACLE ceiling, not a submittable "
              "number. A column earns an entry only if it is named in advance.")
    print(f"\nn macro units = {len(units)}" + ("" if complete else "   SUBSET -- NOT A BOARD NUMBER"))


if __name__ == "__main__":
    main()
