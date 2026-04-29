#!/usr/bin/env python3
"""Bake Brainnetome PM onto cvs_avg35_inMNI152 pial via the accepted compromise path."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.v14.cvsavg_projection import (
    DEFAULT_BOX_ROOT,
    DEFAULT_SUBJECTS_DIR,
    DEFAULT_TARGET_SUBJECT,
    ensure_cvsavg35_subject,
)


DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "atlas" / "cvsavg35_bake"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-dir", type=Path, default=DEFAULT_SUBJECTS_DIR)
    parser.add_argument("--box-root", type=Path, default=DEFAULT_BOX_ROOT)
    parser.add_argument("--target-subject", default=DEFAULT_TARGET_SUBJECT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--atlas-subject", default="ICBM152_fs")
    parser.add_argument("--atlas-volume", type=Path, default=PROJECT_ROOT / "data" / "atlas" / "BNA_PM_4D.nii.gz")
    parser.add_argument("--smooth-fwhm", type=float, default=3.5)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--frames", nargs="+", type=int)
    parser.add_argument("--hemi", action="append", choices=("lh", "rh"), dest="hemis")
    args = parser.parse_args(argv)

    ensure_cvsavg35_subject(
        subjects_dir=args.subjects_dir,
        box_root=args.box_root,
        target_subject=args.target_subject,
    )

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "bake_bna_on_fsaverage.py"),
        "--subjects-dir",
        str(args.subjects_dir),
        "--target-subject",
        args.target_subject,
        "--out-dir",
        str(args.out_dir),
        "--atlas-subject",
        args.atlas_subject,
        "--atlas-volume",
        str(args.atlas_volume),
        "--smooth-fwhm",
        str(args.smooth_fwhm),
    ]
    if args.check:
        cmd.append("--check")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.keep_intermediates:
        cmd.append("--keep-intermediates")
    if args.frames:
        cmd.extend(["--frames", *[str(v) for v in args.frames]])
    for hemi in args.hemis or []:
        cmd.extend(["--hemi", hemi])

    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
