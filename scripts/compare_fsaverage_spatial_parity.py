#!/usr/bin/env python3
"""Compare the strict fsaverage prototype against the current cvs_avg35 oracle."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.v14.fsaverage_atlas import (
    DEFAULT_BAKE_DIR,
    DEFAULT_BNA_N_ROIS,
    DEFAULT_BNA_TREE,
    DEFAULT_ORACLE_PM_PATH,
    DEFAULT_ORACLE_SIGMA_MM,
    assert_full_bake,
    build_cohort_ranking_rows,
    compute_argmax_assignments,
    compute_token_support,
    load_baked_atlas,
    load_mni_cache,
    load_oracle_volume_support,
    load_bna_names,
    sample_baked_support,
)
from speech_decoding.v14.fsaverage_projection import load_fsaverage_cache
from speech_decoding.v14.token_spec import DEFAULT_BASE_PARCELS


DEFAULT_CORE_PATIENTS = ["S14", "S26", "S33", "S62"]
DEFAULT_ORACLE_CACHE_DIR = PROJECT_ROOT / "data" / "mni_coords"
DEFAULT_FSAVG_CACHE_DIR = PROJECT_ROOT / "data" / "fsaverage_coords"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "atlas" / "fsaverage_parity"
DEFAULT_SUBJECTS_DIR = Path("/Applications/freesurfer/8.2.0/subjects")
DEFAULT_TARGET_SUBJECT = "fsaverage"
MIN_ARGMAX_AGREEMENT = 0.85
MIN_TOKENSUPPORT_SPEARMAN = 0.90


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_patient_overlay(
    *,
    patient_id: str,
    cache,
    agreement_mask: np.ndarray,
    valid_mask: np.ndarray,
    verts: np.ndarray,
    out_path: Path,
) -> None:
    point_colors = []
    for valid, agree in zip(valid_mask, agreement_mask):
        if not valid:
            point_colors.append("0.65")
        elif agree:
            point_colors.append("tab:green")
        else:
            point_colors.append("tab:red")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sample = verts[::25]
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c="0.88", s=0.4, alpha=0.4)
    ax.scatter(cache.coords[:, 0], cache.coords[:, 1], cache.coords[:, 2], c=point_colors, s=18)
    ax.set_title(patient_id)
    ax.set_xlabel("R")
    ax.set_ylabel("A")
    ax.set_zlabel("S")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _tier1_set(rows: list[dict[str, object]]) -> set[str]:
    return {str(row["parcel_name"]) for row in rows if int(row["argmax_wins"]) >= 10}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", nargs="+", default=DEFAULT_CORE_PATIENTS)
    parser.add_argument("--oracle-cache-dir", type=Path, default=DEFAULT_ORACLE_CACHE_DIR)
    parser.add_argument("--fsaverage-cache-dir", type=Path, default=DEFAULT_FSAVG_CACHE_DIR)
    parser.add_argument("--bake-dir", type=Path, default=DEFAULT_BAKE_DIR)
    parser.add_argument("--bna-tree", type=Path, default=DEFAULT_BNA_TREE)
    parser.add_argument("--oracle-pm-path", type=Path, default=DEFAULT_ORACLE_PM_PATH)
    parser.add_argument("--oracle-sigma-mm", type=float, default=DEFAULT_ORACLE_SIGMA_MM)
    parser.add_argument("--subjects-dir", type=Path, default=DEFAULT_SUBJECTS_DIR)
    parser.add_argument("--target-subject", default=DEFAULT_TARGET_SUBJECT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    atlas = load_baked_atlas(args.bake_dir, kind="smoothed", tree_path=args.bna_tree)
    assert_full_bake(atlas, expected_n_rois=DEFAULT_BNA_N_ROIS)
    fsavg_pial, _faces = nib.freesurfer.io.read_geometry(
        str(args.subjects_dir / args.target_subject / "surf" / "lh.pial")
    )
    fsavg_pial = np.asarray(fsavg_pial, dtype=np.float64)

    parcel_names = load_bna_names(args.bna_tree, len(atlas.parcel_names))
    prototype_support_by_patient: dict[str, np.ndarray] = {}
    oracle_support_by_patient: dict[str, np.ndarray] = {}
    comparison_rows: list[dict[str, object]] = []
    patient_rows: list[dict[str, object]] = []

    tier1_indices = np.array([idx - 1 for _name, idx in DEFAULT_BASE_PARCELS], dtype=np.int64)

    for patient_id in args.patient:
        oracle_cache = load_mni_cache(patient_id, args.oracle_cache_dir)
        fsavg_cache = load_fsaverage_cache(patient_id, args.fsaverage_cache_dir)
        if oracle_cache.names != fsavg_cache.names:
            raise ValueError(f"{patient_id}: oracle and fsaverage electrode names differ")

        oracle_support = load_oracle_volume_support(
            points_mni=oracle_cache.coords,
            pm_path=args.oracle_pm_path,
            sigma_mm=args.oracle_sigma_mm,
        )
        prototype_support = sample_baked_support(fsavg_cache, atlas)
        oracle_support_by_patient[patient_id] = oracle_support
        prototype_support_by_patient[patient_id] = prototype_support

        oracle_argmax, oracle_max, oracle_valid = compute_argmax_assignments(oracle_support)
        proto_argmax, proto_max, proto_valid = compute_argmax_assignments(prototype_support)
        valid = oracle_valid | proto_valid
        agreement = (oracle_argmax == proto_argmax) & valid
        agreement_rate = float(agreement.sum() / valid.sum()) if valid.any() else float("nan")

        oracle_token_support = compute_token_support(oracle_support)
        proto_token_support = compute_token_support(prototype_support)
        spearman = float(
            spearmanr(
                oracle_token_support[tier1_indices],
                proto_token_support[tier1_indices],
            ).statistic
        )

        for idx, name in enumerate(fsavg_cache.names):
            comparison_rows.append(
                {
                    "patient_id": patient_id,
                    "electrode": name,
                    "hemisphere": fsavg_cache.hemisphere[idx],
                    "fsaverage_vertex": int(fsavg_cache.vertex_ids[idx]),
                    "oracle_argmax_idx": int(oracle_argmax[idx]),
                    "oracle_argmax_name": parcel_names[int(oracle_argmax[idx]) - 1] if int(oracle_argmax[idx]) > 0 else "",
                    "prototype_argmax_idx": int(proto_argmax[idx]),
                    "prototype_argmax_name": parcel_names[int(proto_argmax[idx]) - 1] if int(proto_argmax[idx]) > 0 else "",
                    "oracle_argmax_support": float(oracle_max[idx]),
                    "prototype_argmax_support": float(proto_max[idx]),
                    "valid": bool(valid[idx]),
                    "argmax_match": bool(agreement[idx]),
                }
            )

        patient_rows.append(
            {
                "patient_id": patient_id,
                "n_electrodes": len(fsavg_cache.names),
                "n_valid": int(valid.sum()),
                "argmax_agreement": agreement_rate,
                "tier1_token_support_spearman": spearman,
            }
        )
        _plot_patient_overlay(
            patient_id=patient_id,
            cache=fsavg_cache,
            agreement_mask=agreement,
            valid_mask=valid,
            verts=fsavg_pial,
            out_path=out_dir / "figures" / f"{patient_id}_fsaverage_overlay.png",
        )

    oracle_rows = build_cohort_ranking_rows(
        support_by_patient=oracle_support_by_patient,
        parcel_names=parcel_names,
    )
    prototype_rows = build_cohort_ranking_rows(
        support_by_patient=prototype_support_by_patient,
        parcel_names=parcel_names,
    )
    ranking_rows: list[dict[str, object]] = []
    proto_by_name = {str(row["parcel_name"]): row for row in prototype_rows}
    for oracle_row in oracle_rows:
        name = str(oracle_row["parcel_name"])
        proto_row = proto_by_name[name]
        ranking_rows.append(
            {
                "parcel_name": name,
                "oracle_rank": int(next(i for i, row in enumerate(oracle_rows, start=1) if str(row["parcel_name"]) == name)),
                "prototype_rank": int(next(i for i, row in enumerate(prototype_rows, start=1) if str(row["parcel_name"]) == name)),
                "oracle_argmax_wins": int(oracle_row["argmax_wins"]),
                "prototype_argmax_wins": int(proto_row["argmax_wins"]),
                "oracle_token_support": float(oracle_row["token_support"]),
                "prototype_token_support": float(proto_row["token_support"]),
            }
        )

    _write_csv(
        out_dir / "parity_electrodes.csv",
        comparison_rows,
        [
            "patient_id",
            "electrode",
            "hemisphere",
            "fsaverage_vertex",
            "oracle_argmax_idx",
            "oracle_argmax_name",
            "prototype_argmax_idx",
            "prototype_argmax_name",
            "oracle_argmax_support",
            "prototype_argmax_support",
            "valid",
            "argmax_match",
        ],
    )
    _write_csv(
        out_dir / "parity_patients.csv",
        patient_rows,
        ["patient_id", "n_electrodes", "n_valid", "argmax_agreement", "tier1_token_support_spearman"],
    )
    _write_csv(
        out_dir / "parity_ranking.csv",
        ranking_rows,
        [
            "parcel_name",
            "oracle_rank",
            "prototype_rank",
            "oracle_argmax_wins",
            "prototype_argmax_wins",
            "oracle_token_support",
            "prototype_token_support",
        ],
    )

    oracle_tier1 = _tier1_set(oracle_rows)
    prototype_tier1 = _tier1_set(prototype_rows)
    tier1_diff = sorted(oracle_tier1.symmetric_difference(prototype_tier1))
    status = {
        "min_argmax_agreement": MIN_ARGMAX_AGREEMENT,
        "min_tier1_tokensupport_spearman": MIN_TOKENSUPPORT_SPEARMAN,
        "tier1_symdiff_count": len(tier1_diff),
        "tier1_symdiff": tier1_diff,
        "patients": patient_rows,
    }
    (out_dir / "parity_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    decision_lines = [
        "# Pure fsaverage parity report",
        "",
        "## Cohort rule",
        f"- Tier-1 symmetric difference: {len(tier1_diff)}",
        f"- Tier-1 differing parcels: {', '.join(tier1_diff) if tier1_diff else '(none)'}",
        "",
        "## Per-patient results",
    ]
    for row in patient_rows:
        agreement_ok = float(row["argmax_agreement"]) >= MIN_ARGMAX_AGREEMENT
        spearman_ok = float(row["tier1_token_support_spearman"]) >= MIN_TOKENSUPPORT_SPEARMAN
        decision_lines.append(
            f"- {row['patient_id']}: argmax_agreement={float(row['argmax_agreement']):.3f} "
            f"({'pass' if agreement_ok else 'fail'}), "
            f"tier1_spearman={float(row['tier1_token_support_spearman']):.3f} "
            f"({'pass' if spearman_ok else 'fail'})"
        )
    decision_lines.extend(
        [
            "",
            "## Gate",
            "- Manual QC still required for parcel-frame figures and electrode overlays.",
            "- If any patient-level metric fails badly, stop and try the next geometry option.",
        ]
    )
    (out_dir / "decision_memo.md").write_text("\n".join(decision_lines) + "\n")
    print(f"wrote parity report under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
