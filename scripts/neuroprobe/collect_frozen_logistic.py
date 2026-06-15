#!/usr/bin/env python
"""Aggregate frozen-encoder probe per-cell JSONs into a comparison table.

Pairs with ``probe_v14_frozen_logistic.py`` (sweep discipline). Globs the
per-cell sidecars, writes a tidy CSV, and prints mean AUROC by
(reduction × eval_mode) so the downstream-readout methods can be compared.

    .venv/bin/python scripts/neuroprobe/collect_frozen_logistic.py \
        --dir /work/ht203/probe_out --csv docs/experiments/frozen_probe_cells.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

_MODE_ORDER = ["WithinSession", "CrossSession", "CrossSubject"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/work/ht203/probe_out")
    ap.add_argument("--csv", default=None, help="optional tidy-CSV output path")
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.dir).glob("*.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {p.name}: {e}")
    if not rows:
        print(f"[empty] no JSON cells under {args.dir}")
        return 0
    df = pd.DataFrame(rows)
    print(f"[cells] {len(df)} cells, "
          f"{df['reduction'].nunique()} reductions, {df['task'].nunique()} tasks")

    df["eval_mode"] = pd.Categorical(df["eval_mode"], _MODE_ORDER, ordered=True)
    # mean AUROC over (task, subject, trial, fold) cells, per reduction × mode
    headline = (df.groupby(["reduction", "eval_mode"], observed=True)["auroc"]
                .agg(["mean", "count"]).round(4))
    print("\n=== mean AUROC by reduction × eval_mode (cells averaged) ===")
    print(headline.to_string())

    pertask = (df.pivot_table(index=["eval_mode", "task"], columns="reduction",
                              values="auroc", aggfunc="mean", observed=True).round(4))
    print("\n=== per-task AUROC (rows = mode×task, cols = reduction) ===")
    print(pertask.to_string())

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.sort_values(["reduction", "eval_mode", "task", "subject"]).to_csv(
            args.csv, index=False)
        print(f"\n[csv] {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
