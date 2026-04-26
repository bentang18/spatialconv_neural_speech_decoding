#!/usr/bin/env python3
"""Summarize cvs_avg35 baked-atlas parcel support for bridged pial coordinates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.v14.cvsavg_projection import DEFAULT_CACHE_DIR, load_cvsavg_pial_cache
from speech_decoding.v14.fsaverage_atlas import (
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


DEFAULT_BAKE_DIR = PROJECT_ROOT / "data" / "atlas" / "cvsavg35_bake"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "atlas" / "cvsavg35_parity"
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
    support_by_patient: dict[str, np.ndarray] = {}

    for patient_id in args.patient:
        cache = load_cvsavg_pial_cache(patient_id, args.cache_dir)
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
                    "outer_vertex": int(cache.outer_vertex_ids[row_idx]),
                    "pial_vertex": int(cache.pial_vertex_ids[row_idx]),
                    "outer_x": float(cache.outer_coords[row_idx, 0]),
                    "outer_y": float(cache.outer_coords[row_idx, 1]),
                    "outer_z": float(cache.outer_coords[row_idx, 2]),
                    "pial_x": float(cache.pial_coords[row_idx, 0]),
                    "pial_y": float(cache.pial_coords[row_idx, 1]),
                    "pial_z": float(cache.pial_coords[row_idx, 2]),
                    "d_outer_mm": float(cache.outer_snap_mm[row_idx]),
                    "d_outer_to_pial_mm": float(cache.outer_to_pial_mm[row_idx]),
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
        out_dir / "cvsavg35_electrode_support.csv",
        electrode_rows,
        [
            "patient_id",
            "electrode",
            "hemisphere",
            "outer_vertex",
            "pial_vertex",
            "outer_x",
            "outer_y",
            "outer_z",
            "pial_x",
            "pial_y",
            "pial_z",
            "d_outer_mm",
            "d_outer_to_pial_mm",
            "any_support",
            "argmax_idx",
            "argmax_name",
            "argmax_support",
        ],
    )
    _write_csv(
        out_dir / "cvsavg35_patient_parcels.csv",
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
        out_dir / "cvsavg35_cohort_ranking.csv",
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
    (out_dir / "cvsavg35_support_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote cvs_avg35 support summaries under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
