#!/usr/bin/env python3
"""Precompute strict fsaverage pial electrode coordinates for one or more patients."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.v14.fsaverage_projection import (
    DEFAULT_BOX_ROOT,
    DEFAULT_SUBJECTS_DIR,
    project_patient_to_fsaverage,
)


DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "fsaverage_coords"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", nargs="+", required=True, help="Patient IDs to project")
    parser.add_argument("--box-root", type=Path, default=DEFAULT_BOX_ROOT)
    parser.add_argument("--subjects-dir", type=Path, default=DEFAULT_SUBJECTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args(argv)

    failed: list[tuple[str, str]] = []
    for patient_id in args.patient:
        try:
            out_path = project_patient_to_fsaverage(
                patient_id=patient_id,
                box_root=args.box_root,
                cache_dir=args.cache_dir,
                subjects_dir=args.subjects_dir,
            )
            print(f"{patient_id}: wrote {out_path}")
        except Exception as err:  # noqa: BLE001
            print(f"{patient_id}: FAILED — {err}", file=sys.stderr)
            failed.append((patient_id, str(err)))

    if failed:
        print(f"\n{len(failed)} patient(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
