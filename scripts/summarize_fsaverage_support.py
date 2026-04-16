#!/usr/bin/env python3
"""Summarize fsaverage baked-atlas parcel support for projected electrodes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.v14.fsaverage_atlas import (
    DEFAULT_BAKE_DIR,
    DEFAULT_BNA_N_ROIS,
    DEFAULT_BNA_TREE,
    assert_full_bake,
    build_cohort_ranking_rows,
    compute_any_coverage,
    compute_argmax_assignments,
    compute_argmax_wins,
    compute_token_support,
    load_baked_atlas,
    sample_baked_support,
)
from speech_decoding.v14.fsaverage_projection import load_fsaverage_cache


DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "fsaverage_coords"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "atlas" / "fsaverage_parity"
DEFAULT_PATIENTS = ["S14", "S16", "S23", "S26", "S33", "S39", "S62"]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", nargs="+", default=DEFAULT_PATIENTS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--bake-dir", type=Path, default=DEFAULT_BAKE_DIR)
    parser.add_argument("--bna-tree", type=Path, default=DEFAULT_BNA_TREE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    atlas = load_baked_atlas(args.bake_dir, kind="smoothed", tree_path=args.bna_tree)
    assert_full_bake(atlas, expected_n_rois=DEFAULT_BNA_N_ROIS)
    electrode_rows: list[dict[str, object]] = []
    patient_rows: list[dict[str, object]] = []
    support_by_patient: dict[str, object] = {}

    for patient_id in args.patient:
        cache = load_fsaverage_cache(patient_id, args.cache_dir)
        support = sample_baked_support(cache, atlas)
        support_by_patient[patient_id] = support
        argmax_idx, argmax_support, valid = compute_argmax_assignments(support)
        token_support = compute_token_support(support)
        argmax_wins = compute_argmax_wins(support)
        any_cov = compute_any_coverage(support)
        for row_idx, name in enumerate(cache.names):
            parcel_name = (
                atlas.parcel_names[int(argmax_idx[row_idx]) - 1]
                if int(argmax_idx[row_idx]) > 0
                else ""
            )
            electrode_rows.append(
                {
                    "patient_id": patient_id,
                    "electrode": name,
                    "hemisphere": cache.hemisphere[row_idx],
                    "fsaverage_vertex": int(cache.vertex_ids[row_idx]),
                    "x": float(cache.coords[row_idx, 0]),
                    "y": float(cache.coords[row_idx, 1]),
                    "z": float(cache.coords[row_idx, 2]),
                    "any_support": bool(valid[row_idx]),
                    "argmax_idx": int(argmax_idx[row_idx]),
                    "argmax_name": parcel_name,
                    "argmax_support": float(argmax_support[row_idx]),
                }
            )

        for parcel_idx, parcel_name in enumerate(atlas.parcel_names, start=1):
            patient_rows.append(
                {
                    "patient_id": patient_id,
                    "parcel_idx": parcel_idx,
                    "parcel_name": parcel_name,
                    "token_support": float(token_support[parcel_idx - 1]),
                    "argmax_wins": int(argmax_wins[parcel_idx - 1]),
                    "any_coverage": int(any_cov[parcel_idx - 1]),
                    "max_support": float(support[:, parcel_idx - 1].max()),
                }
            )

    cohort_rows = build_cohort_ranking_rows(
        support_by_patient=support_by_patient,
        parcel_names=atlas.parcel_names,
    )

    out_dir = args.out_dir
    _write_csv(
        out_dir / "fsaverage_electrode_support.csv",
        electrode_rows,
        [
            "patient_id",
            "electrode",
            "hemisphere",
            "fsaverage_vertex",
            "x",
            "y",
            "z",
            "any_support",
            "argmax_idx",
            "argmax_name",
            "argmax_support",
        ],
    )
    _write_csv(
        out_dir / "fsaverage_patient_parcels.csv",
        patient_rows,
        [
            "patient_id",
            "parcel_idx",
            "parcel_name",
            "token_support",
            "argmax_wins",
            "any_coverage",
            "max_support",
        ],
    )
    _write_csv(
        out_dir / "fsaverage_cohort_ranking.csv",
        cohort_rows,
        [
            "parcel_idx",
            "parcel_name",
            "token_support",
            "argmax_wins",
            "any_coverage",
            "n_patients",
            "max_support",
        ],
    )
    summary = {
        "patients": args.patient,
        "bake_dir": str(args.bake_dir),
        "cache_dir": str(args.cache_dir),
        "parcel_count": len(atlas.parcel_names),
        "electrode_count": len(electrode_rows),
    }
    (out_dir / "fsaverage_support_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote fsaverage support summaries under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
