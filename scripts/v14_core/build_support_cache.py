#!/usr/bin/env python3
"""CLI wrapper for materializing the per-electrode Tier-1 support cache.

Library logic lives in `speech_decoding.atlas.support`. This script just
wires paths and iterates patients.

See docs/plans/v14-core.md Task A1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.atlas.fsaverage import DEFAULT_BNA_TREE
from speech_decoding.atlas.support import (
    PatientQC,
    build_patient_support,
    qc_row_for_patient,
    write_qc_report,
    write_support_cache,
)


DEFAULT_COORD_CACHE_DIR = PROJECT_ROOT / "data" / "fsaverage_coords"
DEFAULT_BAKE_DIR = PROJECT_ROOT / "data" / "atlas" / "fsaverage_bake_v2c"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "atlas" / "support_cache_v2c_snap"
DEFAULT_QC_PATH = PROJECT_ROOT / "docs" / "qc" / "support_cache_qc_report.md"
DEFAULT_PATIENTS = ("S14", "S26", "S33", "S62")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", nargs="+", default=list(DEFAULT_PATIENTS))
    parser.add_argument("--coord-cache-dir", type=Path, default=DEFAULT_COORD_CACHE_DIR)
    parser.add_argument("--bake-dir", type=Path, default=DEFAULT_BAKE_DIR)
    parser.add_argument("--bna-tree", type=Path, default=DEFAULT_BNA_TREE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--qc-path", type=Path, default=DEFAULT_QC_PATH)
    args = parser.parse_args(argv)

    qc_rows: list[PatientQC] = []
    for patient_id in args.patient:
        names, support = build_patient_support(
            patient_id,
            coord_cache_dir=args.coord_cache_dir,
            bake_dir=args.bake_dir,
            bna_tree=args.bna_tree,
        )
        out_path = args.out_dir / f"{patient_id}_support_tier1.csv"
        write_support_cache(out_path, names, support)
        qc_rows.append(qc_row_for_patient(patient_id, support))
        print(f"[{patient_id}] wrote {out_path} ({support.shape[0]} electrodes)")

    write_qc_report(args.qc_path, qc_rows)
    print(f"wrote {args.qc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
