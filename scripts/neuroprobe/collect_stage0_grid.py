"""Aggregate a Neuroprobe Stage-0 sweep report directory.

Generic collector for any `<report_dir>/<cell>/<session>/metrics.json` tree
produced by `run_stage0_linear_baseline.py`. Each metrics.json holds one
session's `mean_test_roc_auc` (already averaged over its 15 tasks); this
script aggregates per cell across sessions and writes a ranked summary.

The 5/13-5/18 CrossSubject + band campaign left ~13 such dirs with metrics
and zero analysis files — this is the missing collector.

Usage:
    collect_stage0_grid.py <report_dir> [<report_dir> ...]

Writes `aggregate_summary.csv` into each report dir and prints a ranked
markdown table per dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def collect_dir(report_dir: Path) -> list[dict[str, object]]:
    """Aggregate per cell.

    A "cell" is any directory whose immediate child directories hold a
    metrics.json. This handles both `cell/session/` and protocol-split
    `cell/{CrossSession,CrossSubject}/session/` layouts uniformly — the cell
    name is the directory path relative to report_dir.
    """
    cells: dict[Path, list[tuple[str, float, float]]] = {}
    for mpath in sorted(report_dir.rglob("metrics.json")):
        sess_dir = mpath.parent
        cell_dir = sess_dir.parent
        try:
            m = json.loads(mpath.read_text())
        except json.JSONDecodeError:
            continue
        auroc = m.get("mean_test_roc_auc")
        acc = m.get("mean_test_accuracy")
        if auroc is None:
            continue
        cells.setdefault(cell_dir, []).append(
            (sess_dir.name, float(auroc), float(acc) if acc is not None else float("nan"))
        )

    rows: list[dict[str, object]] = []
    for cell_dir in sorted(cells):
        sessions = sorted(cells[cell_dir])
        cell_name = cell_dir.relative_to(report_dir).as_posix()
        aurocs = [a for _, a, _ in sessions]
        accs = [c for _, _, c in sessions]
        rows.append({
            "cell": cell_name,
            "n_sessions": len(sessions),
            "mean_auroc": statistics.fmean(aurocs),
            "std_auroc": statistics.stdev(aurocs) if len(aurocs) > 1 else 0.0,
            "mean_acc": statistics.fmean(accs),
            "min_session_auroc": min(aurocs),
            "max_session_auroc": max(aurocs),
            "sessions": ";".join(f"{n}:{a:.4f}" for n, a, _ in sessions),
        })
    rows.sort(key=lambda r: r["mean_auroc"], reverse=True)  # type: ignore[arg-type,return-value]
    return rows


def write_csv(report_dir: Path, rows: list[dict[str, object]]) -> Path:
    path = report_dir / "aggregate_summary.csv"
    fields = ["cell", "n_sessions", "mean_auroc", "std_auroc", "mean_acc",
              "min_session_auroc", "max_session_auroc", "sessions"]
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def print_table(report_dir: Path, rows: list[dict[str, object]]) -> None:
    print(f"\n## {report_dir.name}  ({len(rows)} cells)\n")
    print("| rank | cell | n | mean AUROC | std | mean acc |")
    print("|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        print(f"| {i} | {r['cell']} | {r['n_sessions']} | "
              f"{r['mean_auroc']:.4f} | {r['std_auroc']:.4f} | {r['mean_acc']:.4f} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report_dirs", type=Path, nargs="+")
    args = ap.parse_args()
    for rd in args.report_dirs:
        if not rd.is_dir():
            print(f"[skip] not a directory: {rd}", file=sys.stderr)
            continue
        rows = collect_dir(rd)
        if not rows:
            print(f"[skip] no metrics under {rd}", file=sys.stderr)
            continue
        path = write_csv(rd, rows)
        print_table(rd, rows)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
