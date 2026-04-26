#!/usr/bin/env python3
"""Build the 3D-Euclidean Tier-1 support cache for each Phase-1 patient.

Alternative to `scripts/v14_core/build_support_cache.py`, which samples the
baked atlas at a single snap-to-pial vertex per electrode. This script pools
atlas support over pial vertices within a 3D Euclidean radius, Gaussian-weighted
and optionally crown-biased, via `speech_decoding.v14.electrode_pool`.

Writes to `data/atlas/support_cache_euclidean/<pt>_support_tier1.csv` — schema
identical to the snap-based cache. The v14 loader is unchanged; only the cache
path differs, so the loader config can swap to this path once verification
passes.

Phase-1 cohort (7 LH patients). Hemisphere-agnostic internally: RH electrodes
are handled if present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.v14.electrode_pool import euclidean_support
from speech_decoding.v14.fsaverage_atlas import (
    DEFAULT_BNA_N_ROIS,
    DEFAULT_BNA_TREE,
    assert_full_bake,
    load_baked_atlas,
)
from speech_decoding.v14.fsaverage_projection import load_fsaverage_cache
from speech_decoding.v14.support_cache import (
    PatientQC,
    TIER1_BNA_INDICES_1BASED,
    TIER1_COLUMNS,
    qc_row_for_patient,
    write_support_cache,
)
from speech_decoding.v14.token_spec import DEFAULT_BASE_PARCELS

_HIST_BIN_LABELS = ("[0,1)", "[1,10)", "[10,25)", "[25,50)", "[50,75)", "[75,100]")

DEFAULT_FS_HOME = Path("/Applications/freesurfer/8.2.0")
DEFAULT_SUBJECTS_DIR = DEFAULT_FS_HOME / "subjects"
DEFAULT_COORD_CACHE_DIR = PROJECT_ROOT / "data" / "fsaverage_coords"
DEFAULT_BAKE_DIR = PROJECT_ROOT / "data" / "atlas" / "fsaverage_bake_v2"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "atlas" / "support_cache_euclidean"
DEFAULT_QC_PATH = PROJECT_ROOT / "docs" / "qc" / "support_cache_euclidean_qc_report.md"
DEFAULT_PATIENTS = ("S14", "S16", "S23", "S26", "S33", "S39", "S62")

DEFAULT_SIGMA_MM = 1.3
DEFAULT_RADIUS_MM = 5.0


def _load_fsaverage_surface(subjects_dir: Path, hemi: str) -> tuple[np.ndarray, np.ndarray]:
    surf_dir = subjects_dir / "fsaverage" / "surf"
    pial_path = surf_dir / f"{hemi}.pial"
    curv_path = surf_dir / f"{hemi}.curv"
    geom = nib.freesurfer.io.read_geometry(str(pial_path))
    verts = geom[0]
    curv = nib.freesurfer.io.read_morph_data(str(curv_path))
    return np.asarray(verts, dtype=np.float64), np.asarray(curv, dtype=np.float64)


def _tier1_slice(atlas_lh_rh: np.ndarray) -> np.ndarray:
    col_idx = np.array([i - 1 for i in TIER1_BNA_INDICES_1BASED], dtype=np.int64)
    return atlas_lh_rh[:, col_idx]


def build_patient_support_euclidean(
    patient_id: str,
    *,
    coord_cache_dir: Path,
    bake_dir: Path,
    subjects_dir: Path,
    sigma_mm: float,
    radius_mm: float,
    crown_bias: bool,
    bna_tree: Path = DEFAULT_BNA_TREE,
) -> tuple[tuple[str, ...], np.ndarray]:
    cache = load_fsaverage_cache(patient_id, coord_cache_dir)
    atlas = load_baked_atlas(bake_dir, kind="smoothed", tree_path=bna_tree)
    assert_full_bake(atlas, expected_n_rois=DEFAULT_BNA_N_ROIS)

    n_e = len(cache.names)
    n_tier1 = len(TIER1_BNA_INDICES_1BASED)
    support = np.zeros((n_e, n_tier1), dtype=np.float32)

    hemis_present = tuple(sorted(set(cache.hemisphere)))
    hemi_map = {"L": "lh", "R": "rh"}
    for hemi_letter in hemis_present:
        if hemi_letter not in hemi_map:
            raise ValueError(f"{patient_id}: unexpected hemisphere code {hemi_letter!r}")
        hemi = hemi_map[hemi_letter]
        pial_xyz, curv = _load_fsaverage_surface(subjects_dir, hemi)
        tier1 = _tier1_slice(atlas.values_by_hemi[hemi]).astype(np.float32, copy=False)

        # Rows in `cache` for this hemisphere.
        mask = np.array([h == hemi_letter for h in cache.hemisphere], dtype=bool)
        elec_xyz = cache.coords[mask]
        pooled = euclidean_support(
            elec_xyz,
            pial_xyz,
            tier1,
            curv,
            sigma_mm=sigma_mm,
            radius_mm=radius_mm,
            crown_bias=crown_bias,
        )
        support[mask] = pooled

    return cache.names, support


def _write_qc_report_euclidean(
    qc_path: Path,
    qc_rows: list[PatientQC],
    *,
    bake_dir: Path,
    sigma_mm: float,
    radius_mm: float,
    crown_bias: bool,
) -> None:
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Tier-1 support cache QC report (Euclidean pool)\n")
    lines.append(
        f"Per-patient summary of `{qc_path.parent.parent.name}/"
        f"support_cache_euclidean/<pt>_support_tier1.csv`.\n"
    )
    lines.append(
        f"Source: 3D Euclidean pool over fsaverage pial vertices, "
        f"sigma={sigma_mm} mm, radius={radius_mm} mm, "
        f"crown_bias={'on' if crown_bias else 'off'}, "
        f"atlas={bake_dir.name}/smoothed, sliced to 15 "
        f"`DEFAULT_BASE_PARCELS` columns (raw [0, 100] BNA probability).\n"
    )
    lines.append("\n## Tier-1 column order\n")
    for col, (name, bna_idx) in zip(TIER1_COLUMNS, DEFAULT_BASE_PARCELS):
        lines.append(f"- `{col}` <- `{name}`, BNA index {bna_idx}")
    lines.append("\n## Per-patient summary\n")
    header_bins = " | ".join(f"max {lbl}" for lbl in _HIST_BIN_LABELS)
    lines.append(
        "| patient | n_elec | argmax_in_tier1 | low_tier1_mass (<1) | " + header_bins + " |"
    )
    lines.append("|" + "---|" * (4 + len(_HIST_BIN_LABELS)))
    for row in qc_rows:
        hist_cells = " | ".join(str(c) for c in row.hist_counts)
        lines.append(
            f"| {row.patient} | {row.n_electrodes} | {row.argmax_in_tier1} | "
            f"{row.low_tier1_mass} | {hist_cells} |"
        )
    lines.append("")
    qc_path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", nargs="+", default=list(DEFAULT_PATIENTS))
    parser.add_argument("--coord-cache-dir", type=Path, default=DEFAULT_COORD_CACHE_DIR)
    parser.add_argument("--bake-dir", type=Path, default=DEFAULT_BAKE_DIR)
    parser.add_argument("--subjects-dir", type=Path, default=DEFAULT_SUBJECTS_DIR)
    parser.add_argument("--bna-tree", type=Path, default=DEFAULT_BNA_TREE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--qc-path", type=Path, default=DEFAULT_QC_PATH)
    parser.add_argument("--sigma-mm", type=float, default=DEFAULT_SIGMA_MM)
    parser.add_argument("--radius-mm", type=float, default=DEFAULT_RADIUS_MM)
    parser.add_argument("--no-crown-bias", action="store_true",
                        help="Disable sigmoid crown preference (test-only)")
    args = parser.parse_args(argv)

    qc_rows: list[PatientQC] = []
    for patient_id in args.patient:
        names, support = build_patient_support_euclidean(
            patient_id,
            coord_cache_dir=args.coord_cache_dir,
            bake_dir=args.bake_dir,
            subjects_dir=args.subjects_dir,
            sigma_mm=args.sigma_mm,
            radius_mm=args.radius_mm,
            crown_bias=not args.no_crown_bias,
            bna_tree=args.bna_tree,
        )
        out_path = args.out_dir / f"{patient_id}_support_tier1.csv"
        write_support_cache(out_path, names, support)
        qc_rows.append(qc_row_for_patient(patient_id, support))
        print(f"[{patient_id}] wrote {out_path} ({support.shape[0]} electrodes)")

    _write_qc_report_euclidean(
        args.qc_path, qc_rows,
        bake_dir=args.bake_dir,
        sigma_mm=args.sigma_mm,
        radius_mm=args.radius_mm,
        crown_bias=not args.no_crown_bias,
    )
    print(f"wrote {args.qc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
