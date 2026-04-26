#!/usr/bin/env python3
"""Bridge Zac's cvs_avg35 outer-surface cache onto cvs_avg35 pial vertices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.v14.cvsavg_projection import (
    DEFAULT_BOX_ROOT,
    DEFAULT_CACHE_DIR,
    DEFAULT_ORACLE_CACHE_DIR,
    DEFAULT_SUBJECTS_DIR,
    ensure_cvsavg35_subject,
    project_oracle_cache_to_cvsavg_pial,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", nargs="+", required=True)
    parser.add_argument("--oracle-cache-dir", type=Path, default=DEFAULT_ORACLE_CACHE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--subjects-dir", type=Path, default=DEFAULT_SUBJECTS_DIR)
    parser.add_argument("--box-root", type=Path, default=DEFAULT_BOX_ROOT)
    args = parser.parse_args(argv)

    ensure_cvsavg35_subject(subjects_dir=args.subjects_dir, box_root=args.box_root)

    failed: list[tuple[str, str]] = []
    for patient_id in args.patient:
        try:
            out_path = project_oracle_cache_to_cvsavg_pial(
                patient_id=patient_id,
                oracle_cache_dir=args.oracle_cache_dir,
                cache_dir=args.cache_dir,
                subjects_dir=args.subjects_dir,
                box_root=args.box_root,
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
