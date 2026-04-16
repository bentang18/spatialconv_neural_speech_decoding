#!/usr/bin/env python3
"""Prepare the atlas-side ICBM152_fs FreeSurfer subject for the fsaverage bake.

The accepted `#36` bake path needs a real surface subject for the atlas side:
`ICBM152_fs`. FreeSurfer 8.2.0 already ships the nonlinear ICBM152 2009c T1
volume, but not a completed surface reconstruction. This helper prints or runs
the `recon-all` command needed to create that subject.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


DEFAULT_FS_HOME = Path("/Applications/freesurfer/8.2.0")
DEFAULT_SUBJECTS_DIR = DEFAULT_FS_HOME / "subjects"
DEFAULT_SOURCE_VOLUME = (
    DEFAULT_FS_HOME
    / "average"
    / "mni_icbm152_nlin_asym_09c"
    / "mni_icbm152_t1_tal_nlin_asym_09c.nii.gz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fs-home",
        type=Path,
        default=Path(os.environ.get("FREESURFER_HOME", DEFAULT_FS_HOME)),
        help="FreeSurfer install root",
    )
    parser.add_argument(
        "--subjects-dir",
        type=Path,
        default=Path(os.environ.get("SUBJECTS_DIR", DEFAULT_SUBJECTS_DIR)),
        help="FreeSurfer SUBJECTS_DIR",
    )
    parser.add_argument(
        "--source-volume",
        type=Path,
        default=DEFAULT_SOURCE_VOLUME,
        help="Input nonlinear ICBM152 T1 volume",
    )
    parser.add_argument(
        "--subject-name",
        default="ICBM152_fs",
        help="Output FreeSurfer subject name",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run recon-all instead of only printing the command",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recon_all = args.fs_home / "bin" / "recon-all"
    subject_dir = args.subjects_dir / args.subject_name

    if not recon_all.exists():
        raise FileNotFoundError(recon_all)
    if not args.source_volume.exists():
        raise FileNotFoundError(args.source_volume)
    if subject_dir.exists():
        print(f"Subject already exists: {subject_dir}")
        return 0

    cmd = [
        str(recon_all),
        "-i",
        str(args.source_volume),
        "-s",
        args.subject_name,
        "-sd",
        str(args.subjects_dir),
        "-all",
    ]

    print("Command:")
    print(" ".join(shlex.quote(part) for part in cmd))

    if not args.run:
        print()
        print("Use --run to execute this recon-all job.")
        return 0

    env = os.environ.copy()
    env["FREESURFER_HOME"] = str(args.fs_home)
    env["SUBJECTS_DIR"] = str(args.subjects_dir)
    env["PATH"] = f"{args.fs_home / 'bin'}:{env.get('PATH', '')}"
    subprocess.run(cmd, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
