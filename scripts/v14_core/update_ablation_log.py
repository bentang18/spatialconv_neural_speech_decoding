#!/usr/bin/env python3
"""Aggregate v14 per-phoneme result JSONs into `docs/experiments/v14_ablation_log.csv`.

Walks a results root (typically `/work/ht203/results/v14_per_phoneme/` on DCC,
or a local rsync mirror), groups `*.result.json` files by config tuple
(patient, d_model, depth, conv2d_kernel, pool_h, pool_w), and writes per-cell
stats (mean / std / min / max of `test_per_phoneme`, slot-averaged mean,
best-run tag, n_completed, status) back into the experiment log CSV.

Usage:
    python scripts/v14_core/update_ablation_log.py \
        --results-root /tmp/v14_results \
        --csv docs/experiments/v14_ablation_log.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


BASELINE_PER = 0.734


def _config_key(row: dict) -> tuple:
    pool_shape = row.get("pool_shape", [4, 8])
    return (
        row["patient"],
        int(row.get("d_model", 32)),
        int(row["backbone_depth"]),
        int(row.get("conv2d_kernel", 3)),
        int(pool_shape[0]),
        int(pool_shape[1]),
    )


def _csv_row_key(row: dict) -> tuple:
    return (
        row["patient"],
        int(row["d_model"]),
        int(row["backbone_depth"]),
        int(row["conv2d_kernel"]),
        int(row["pool_h"]),
        int(row["pool_w"]),
    )


def aggregate_results(results_root: Path) -> dict[tuple, dict]:
    """Walk result JSONs and group by config tuple."""

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in results_root.rglob("*.result.json"):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "test_per_phoneme" not in r:
            continue
        r["_path"] = p
        groups[_config_key(r)].append(r)

    out: dict[tuple, dict] = {}
    for key, runs in groups.items():
        pers = [r["test_per_phoneme"] for r in runs]
        slot = [r.get("test_slot_averaged_per") for r in runs]
        slot = [s for s in slot if s is not None]
        best = min(runs, key=lambda r: r["test_per_phoneme"])
        best_tag = Path(best["_path"]).stem.replace(".result", "")
        out[key] = {
            "n_completed": len(runs),
            "per_phoneme_mean": statistics.mean(pers),
            "per_phoneme_std": statistics.stdev(pers) if len(pers) > 1 else 0.0,
            "per_phoneme_min": min(pers),
            "per_phoneme_max": max(pers),
            "slot_avg_mean": statistics.mean(slot) if slot else None,
            "best_run_tag": best_tag,
        }
    return out


def update_csv(csv_path: Path, agg: dict[tuple, dict]) -> tuple[int, int]:
    """Update matching rows in-place; append new rows for new cells."""

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    updated = 0
    seen: set[tuple] = set()
    for row in rows:
        try:
            key = _csv_row_key(row)
        except (KeyError, ValueError):
            continue
        if key not in agg:
            continue
        seen.add(key)
        stats = agg[key]
        row["n_completed"] = str(stats["n_completed"])
        row["per_phoneme_mean"] = f"{stats['per_phoneme_mean']:.3f}"
        row["per_phoneme_std"] = f"{stats['per_phoneme_std']:.3f}"
        row["per_phoneme_min"] = f"{stats['per_phoneme_min']:.3f}"
        row["per_phoneme_max"] = f"{stats['per_phoneme_max']:.3f}"
        if stats["slot_avg_mean"] is not None:
            row["slot_avg_mean"] = f"{stats['slot_avg_mean']:.3f}"
        row["best_run_tag"] = stats["best_run_tag"]
        row["vs_baseline_0.734"] = f"{stats['per_phoneme_mean'] - BASELINE_PER:+.3f}"
        expected = int(row["n_folds"]) * int(row["n_seeds"])
        if stats["n_completed"] >= expected:
            row["status"] = "completed"
        updated += 1

    appended = 0
    for key, stats in agg.items():
        if key in seen:
            continue
        patient, d, depth, k, ph, pw = key
        new_row = {fn: "" for fn in fieldnames}
        new_row.update({
            "experiment_id": f"auto_{patient}_d{d}_depth{depth}_k{k}_pool{ph}x{pw}",
            "patient": patient,
            "d_model": str(d),
            "backbone_depth": str(depth),
            "conv2d_kernel": str(k),
            "pool_h": str(ph),
            "pool_w": str(pw),
            "n_folds": "5",
            "n_seeds": "3",
            "n_completed": str(stats["n_completed"]),
            "per_phoneme_mean": f"{stats['per_phoneme_mean']:.3f}",
            "per_phoneme_std": f"{stats['per_phoneme_std']:.3f}",
            "per_phoneme_min": f"{stats['per_phoneme_min']:.3f}",
            "per_phoneme_max": f"{stats['per_phoneme_max']:.3f}",
            "slot_avg_mean": (
                f"{stats['slot_avg_mean']:.3f}" if stats["slot_avg_mean"] is not None else ""
            ),
            "best_run_tag": stats["best_run_tag"],
            "vs_baseline_0.734": f"{stats['per_phoneme_mean'] - BASELINE_PER:+.3f}",
            "status": "completed" if stats["n_completed"] >= 15 else "running",
            "notes": "auto-appended by update_ablation_log",
        })
        rows.append(new_row)
        appended += 1

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return updated, appended


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True,
                        help="Root dir containing <patient>/<cfg>/*.result.json")
    parser.add_argument("--csv", type=Path, required=True,
                        help="Experiment log CSV to update in place")
    args = parser.parse_args(argv)

    agg = aggregate_results(args.results_root)
    updated, appended = update_csv(args.csv, agg)
    print(f"[update_ablation_log] groups found: {len(agg)}, "
          f"rows updated: {updated}, rows appended: {appended}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
