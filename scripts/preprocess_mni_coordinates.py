"""Precompute MNI152 electrode coordinates for one or more patients.

Reads FreeSurfer reconstructions from Box, runs the Python port of
``sub2AvgBrainClinical.m`` (see ``speech_decoding.v14.coordinates``), and
writes a per-patient CSV into ``data/mni_coords/<pt>_MNI152.csv`` plus a
shared sidecar metadata file ``_projection_meta.json``.

Usage::

    .venv/bin/python scripts/preprocess_mni_coordinates.py --patient S14
    .venv/bin/python scripts/preprocess_mni_coordinates.py --patient S14 S26 S33 S62

The default Box root is the macOS Box Drive mount. Pass ``--box-root`` to
override (e.g., on DCC where Box is not mounted and surface files have been
rsynced into a staging directory).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from speech_decoding.v14.coordinates import project_patient


DEFAULT_BOX_ROOT = Path("/Users/bentang/Library/CloudStorage/Box-Box")
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "mni_coords"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patient",
        nargs="+",
        required=True,
        help="Patient IDs to project (e.g. S14 S26 S33 S62)",
    )
    parser.add_argument(
        "--box-root",
        type=Path,
        default=DEFAULT_BOX_ROOT,
        help=f"Box mount root (default: {DEFAULT_BOX_ROOT})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Output directory for cached CSVs (default: {DEFAULT_CACHE_DIR})",
    )
    args = parser.parse_args(argv)

    if not args.box_root.exists():
        print(f"error: box root does not exist: {args.box_root}", file=sys.stderr)
        return 2

    failed: list[tuple[str, str]] = []
    for patient_id in args.patient:
        try:
            out_path = project_patient(
                patient_id=patient_id,
                box_root=args.box_root,
                cache_dir=args.cache_dir,
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
    sys.exit(main())
