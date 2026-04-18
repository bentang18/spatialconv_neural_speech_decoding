#!/usr/bin/env python3
"""Aggregate v14-core `.result.json` files into the first-run report (Task E3).

Usage:
    python scripts/v14_core/aggregate_v14_core.py \
        --results-dir /work/ht203/results/v14core \
        --out-md docs/results/v14_core_first_run.md \
        --out-json docs/results/v14_core_first_run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.v14.aggregate import (  # noqa: E402
    aggregate,
    load_results,
    render_markdown,
)


CORE_PATIENTS = ("S14", "S26", "S33", "S62")
CORE_DEPTHS = (1, 2)
CORE_FOLDS = (0, 1, 2, 3, 4)
CORE_SEEDS = (0, 1, 2)


def _to_json(report) -> dict:
    return {
        "actual_n": report.actual_n,
        "expected_n": report.expected_n,
        "missing": [
            {"patient": p, "fold": f, "seed": s, "depth": d}
            for (p, f, s, d) in report.missing
        ],
        "patient_cells": [
            {
                "patient": pc.patient,
                "depth": pc.depth,
                "fold_means": pc.fold_means,
                "mean": pc.mean,
                "std": pc.std,
            }
            for pc in report.patient_cells.values()
        ],
        "population_cells": [
            {
                "depth": pc.depth,
                "patient_means": pc.patient_means,
                "mean": pc.mean,
            }
            for pc in report.population_cells.values()
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--patients", nargs="+", default=list(CORE_PATIENTS))
    parser.add_argument("--depths", nargs="+", type=int, default=list(CORE_DEPTHS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(CORE_FOLDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(CORE_SEEDS))
    args = parser.parse_args(argv)

    rows = load_results(args.results_dir)
    report = aggregate(
        rows,
        patients=args.patients,
        depths=args.depths,
        folds=args.folds,
        seeds=args.seeds,
    )
    md = render_markdown(report, patients=args.patients, depths=args.depths)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(_to_json(report), indent=2))

    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
