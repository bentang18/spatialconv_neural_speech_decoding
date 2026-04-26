#!/usr/bin/env python3
"""Slice the v14 ablation CSV without reading the whole thing.

Usage:
    scripts/ablation/query.py --patient S14 --d 64
    scripts/ablation/query.py --patient S14 --depth 5
    scripts/ablation/query.py --pool 4x8 --status completed
    scripts/ablation/query.py --recent 10
    scripts/ablation/query.py --job 45707149
    scripts/ablation/query.py --baseline           # always shown when --patient set
    scripts/ablation/query.py --csv path/to/other.csv

Output: a compact table of matching rows (PER mean ± std, vs_baseline,
n_completed/N_total, status, date). Sorted by date_started desc.

Filters compose with AND. The S14 baseline cell (d=32, depth=3, k=3, pool=4x8)
is shown by default whenever any --patient filter resolves to S14, so you
always see what your ablation is being compared against.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

LOCAL_REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = LOCAL_REPO / "docs" / "experiments" / "v14_ablation_log.csv"

BASELINE_KEY = ("S14", 32, 3, 3, 4, 8)
BASELINE_PER = 0.734


def _row_key(r: dict) -> tuple:
    return (
        r["patient"],
        int(r["d_model"]),
        int(r["backbone_depth"]),
        int(r["conv2d_kernel"]),
        int(r["pool_h"]),
        int(r["pool_w"]),
    )


def _matches(r: dict, args: argparse.Namespace) -> bool:
    if args.patient and args.patient not in r["patient"]:
        return False
    if args.d is not None and int(r["d_model"]) != args.d:
        return False
    if args.depth is not None and int(r["backbone_depth"]) != args.depth:
        return False
    if args.k is not None and int(r["conv2d_kernel"]) != args.k:
        return False
    if args.pool:
        try:
            ph, pw = (int(x) for x in args.pool.lower().split("x"))
        except ValueError:
            sys.exit(f"--pool wants HxW, got {args.pool!r}")
        if int(r["pool_h"]) != ph or int(r["pool_w"]) != pw:
            return False
    if args.status and args.status not in r["status"]:
        return False
    if args.job and args.job not in r.get("job_id", ""):
        return False
    return True


def _format_row(r: dict, mark: str = " ") -> str:
    mean = float(r["per_phoneme_mean"]) if r["per_phoneme_mean"] else float("nan")
    std = float(r["per_phoneme_std"]) if r["per_phoneme_std"] else float("nan")
    n_done = int(r["n_completed"]) if r["n_completed"] else 0
    n_total = int(r["n_folds"]) * int(r["n_seeds"])
    vs = r.get("vs_baseline_0.734", "").strip() or "—"
    return (
        f"{mark} {r['experiment_id']:<32}  "
        f"PER {mean:.3f} ± {std:.3f}  "
        f"vs base {vs:>6}  "
        f"{n_done:>2}/{n_total:>2}  "
        f"{r['status']:<10} {r['date_started']}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patient", help="Substring match (S14, pooled, S26, ...)")
    p.add_argument("--d", type=int, help="d_model")
    p.add_argument("--depth", type=int, help="backbone_depth")
    p.add_argument("--k", type=int, help="conv2d_kernel")
    p.add_argument("--pool", help="HxW (e.g. 4x8)")
    p.add_argument("--status", help="Substring (completed, partial, failed, ...)")
    p.add_argument("--job", help="Substring of job_id")
    p.add_argument("--recent", type=int, help="After all filters, keep most recent N")
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Force-include S14 baseline cell at top of output even if filters exclude it.",
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = p.parse_args()

    if not args.csv.exists():
        sys.exit(f"csv not found: {args.csv}")

    with args.csv.open() as f:
        rows = list(csv.DictReader(f))

    matches = [r for r in rows if _matches(r, args)]
    matches.sort(key=lambda r: r.get("date_started", ""), reverse=True)
    if args.recent:
        matches = matches[: args.recent]

    show_baseline = args.baseline or (
        args.patient and "S14" in args.patient
    )
    baseline_row = None
    if show_baseline:
        for r in rows:
            if _row_key(r) == BASELINE_KEY:
                baseline_row = r
                break

    if not matches and not baseline_row:
        print("(no matches)")
        return 0

    if baseline_row and baseline_row not in matches:
        print(_format_row(baseline_row, mark="*"))
    elif baseline_row:
        print(_format_row(baseline_row, mark="*"))
        matches = [r for r in matches if r is not baseline_row]

    for r in matches:
        print(_format_row(r))

    if baseline_row:
        print(f"  (* = S14 d=32 depth=3 baseline; PER target {BASELINE_PER})")

    print(f"  -- {len(matches) + (1 if baseline_row else 0)} rows from {args.csv.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
