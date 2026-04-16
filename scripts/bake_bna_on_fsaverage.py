#!/usr/bin/env python3
"""Bake Brainnetome PM atlas onto fsaverage via FreeSurfer surface tools.

This script encodes the accepted `#36` path:

1. Project the 4D Brainnetome PM volume onto an atlas-side surface subject
   with `mri_vol2surf --projfrac-avg 0 1 0.1`.
2. Transfer the resulting per-vertex values onto `fsaverage` with
   `mri_surf2surf` via `sphere.reg`.
3. Optionally apply geodesic smoothing on the fsaverage mesh.

It is designed to be useful before the full bake is runnable:

- `--check` reports whether local prerequisites exist.
- `--dry-run` prints the exact FreeSurfer commands that would be executed.
- `--frames` allows a smoke test on a subset of atlas channels.

The intended atlas-side source subject is `ICBM152_fs`. If that subject does
not yet exist, the script fails clearly and points at the bundled nonlinear
ICBM152 2009c T1 volume that can be used to build it.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import nibabel as nib
except Exception:  # pragma: no cover - handled at runtime
    nib = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.v14.fsaverage_atlas import consolidate_baked_atlas


DEFAULT_FS_HOME = Path("/Applications/freesurfer/8.2.0")
DEFAULT_SUBJECTS_DIR = DEFAULT_FS_HOME / "subjects"
DEFAULT_ATLAS_VOLUME = Path("data/atlas/BNA_PM_4D.nii.gz")
DEFAULT_ATLAS_SUBJECT = "ICBM152_fs"
DEFAULT_TARGET_SUBJECT = "fsaverage"
DEFAULT_HEMIS = ("lh", "rh")
DEFAULT_SMOOTH_FWHM = 3.5


@dataclass(frozen=True)
class FsEnv:
    fs_home: Path
    subjects_dir: Path

    @property
    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["FREESURFER_HOME"] = str(self.fs_home)
        env["SUBJECTS_DIR"] = str(self.subjects_dir)
        env["PATH"] = f"{self.fs_home / 'bin'}:{env.get('PATH', '')}"
        return env

    def bin_path(self, name: str) -> Path:
        return self.fs_home / "bin" / name


@dataclass(frozen=True)
class PrereqStatus:
    ok: bool
    checks: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas-volume",
        type=Path,
        default=DEFAULT_ATLAS_VOLUME,
        help="4D Brainnetome PM NIfTI volume",
    )
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
        "--atlas-subject",
        default=DEFAULT_ATLAS_SUBJECT,
        help="Atlas-side source subject for the nonlinear bake",
    )
    parser.add_argument(
        "--target-subject",
        default=DEFAULT_TARGET_SUBJECT,
        help="Target surface subject, usually fsaverage",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/atlas/fsaverage_bake"),
        help="Output directory for intermediate and final bake files",
    )
    parser.add_argument(
        "--hemi",
        action="append",
        choices=DEFAULT_HEMIS,
        dest="hemis",
        help="Hemisphere(s) to process; default is both",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="*",
        help="Atlas frames to process; default is all frames",
    )
    parser.add_argument(
        "--smooth-fwhm",
        type=float,
        default=DEFAULT_SMOOTH_FWHM,
        help="Geodesic surface smoothing FWHM on fsaverage",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check local prerequisites and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep per-frame intermediate files under out-dir/intermediates",
    )
    return parser.parse_args()


def canonicalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.hemis:
        args.hemis = list(DEFAULT_HEMIS)
    return args


def check_prereqs(fsenv: FsEnv, atlas_volume: Path, atlas_subject: str, target_subject: str) -> PrereqStatus:
    checks: dict[str, str] = {}

    for tool in ("mri_vol2surf", "mri_surf2surf", "mris_info"):
        tool_path = fsenv.bin_path(tool)
        checks[f"tool:{tool}"] = "ok" if tool_path.exists() else f"missing: {tool_path}"

    checks["atlas_volume"] = "ok" if atlas_volume.exists() else f"missing: {atlas_volume}"

    atlas_subject_dir = fsenv.subjects_dir / atlas_subject
    checks["atlas_subject"] = "ok" if atlas_subject_dir.exists() else f"missing: {atlas_subject_dir}"

    target_subject_dir = fsenv.subjects_dir / target_subject
    checks["target_subject"] = "ok" if target_subject_dir.exists() else f"missing: {target_subject_dir}"

    for hemi in DEFAULT_HEMIS:
        surf = target_subject_dir / "surf" / f"{hemi}.sphere.reg"
        checks[f"target_sphere_reg:{hemi}"] = "ok" if surf.exists() else f"missing: {surf}"
        pial = target_subject_dir / "surf" / f"{hemi}.pial"
        checks[f"target_pial:{hemi}"] = "ok" if pial.exists() else f"missing: {pial}"

    icbm_seed = fsenv.fs_home / "average" / "mni_icbm152_nlin_asym_09c" / "mni_icbm152_t1_tal_nlin_asym_09c.nii.gz"
    checks["icbm_seed_volume"] = "ok" if icbm_seed.exists() else f"missing: {icbm_seed}"

    mni152_reg = fsenv.fs_home / "average" / "mni152.register.dat"
    checks["mni152_register_dat"] = "ok" if mni152_reg.exists() else f"missing: {mni152_reg}"

    ok = all(value == "ok" for value in checks.values() if not value.startswith("missing:"))
    # Special handling: missing atlas subject should fail the overall check even if everything else exists.
    if checks["atlas_subject"] != "ok":
        ok = False
    if checks["atlas_volume"] != "ok":
        ok = False
    return PrereqStatus(ok=ok, checks=checks)


def detect_n_frames(atlas_volume: Path) -> int:
    if nib is None:
        raise RuntimeError("nibabel is required to inspect the 4D Brainnetome atlas volume")
    img = nib.load(str(atlas_volume))
    if img.ndim != 4:
        raise ValueError(f"expected 4D atlas volume, got shape {img.shape}")
    return int(img.shape[3])


def format_cmd(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run(cmd: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print(format_cmd(cmd))
    if dry_run:
        return
    subprocess.run(cmd, env=env, check=True)


def ensure_subject_surface(fsenv: FsEnv, subject: str, hemi: str, surf_name: str) -> Path:
    path = fsenv.subjects_dir / subject / "surf" / f"{hemi}.{surf_name}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def vol2surf_cmd(
    fsenv: FsEnv,
    *,
    atlas_volume: Path,
    atlas_subject: str,
    hemi: str,
    frame: int,
    out_path: Path,
) -> list[str]:
    ensure_subject_surface(fsenv, atlas_subject, hemi, "white")
    ensure_subject_surface(fsenv, atlas_subject, hemi, "pial")
    return [
        str(fsenv.bin_path("mri_vol2surf")),
        "--mov",
        str(atlas_volume),
        "--srcsubject",
        atlas_subject,
        "--trgsubject",
        atlas_subject,
        "--hemi",
        hemi,
        "--surf",
        "white",
        "--projfrac-avg",
        "0",
        "1",
        "0.1",
        "--frame",
        str(frame),
        "--interp",
        "trilinear",
        "--o",
        str(out_path),
    ]


def surf2surf_cmd(
    fsenv: FsEnv,
    *,
    atlas_subject: str,
    target_subject: str,
    hemi: str,
    in_path: Path,
    out_path: Path,
) -> list[str]:
    ensure_subject_surface(fsenv, atlas_subject, hemi, "sphere.reg")
    ensure_subject_surface(fsenv, target_subject, hemi, "sphere.reg")
    return [
        str(fsenv.bin_path("mri_surf2surf")),
        "--srcsubject",
        atlas_subject,
        "--trgsubject",
        target_subject,
        "--hemi",
        hemi,
        "--surfreg",
        "sphere.reg",
        "--sval",
        str(in_path),
        "--tval",
        str(out_path),
    ]


def smooth_cmd(
    fsenv: FsEnv,
    *,
    target_subject: str,
    hemi: str,
    in_path: Path,
    out_path: Path,
    smooth_fwhm: float,
) -> list[str]:
    ensure_subject_surface(fsenv, target_subject, hemi, "sphere.reg")
    return [
        str(fsenv.bin_path("mri_surf2surf")),
        "--srcsubject",
        target_subject,
        "--trgsubject",
        target_subject,
        "--hemi",
        hemi,
        "--surfreg",
        "sphere.reg",
        "--sval",
        str(in_path),
        "--tval",
        str(out_path),
        "--fwhm-trg",
        str(smooth_fwhm),
    ]


def write_manifest(out_dir: Path, payload: dict[str, object]) -> None:
    path = out_dir / "bake_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = canonicalize_args(parse_args())
    fsenv = FsEnv(fs_home=args.fs_home, subjects_dir=args.subjects_dir)
    status = check_prereqs(fsenv, args.atlas_volume, args.atlas_subject, args.target_subject)

    if args.check:
        print(json.dumps(status.checks, indent=2, sort_keys=True))
        if status.checks.get("atlas_subject", "").startswith("missing:"):
            icbm_seed = fsenv.fs_home / "average" / "mni_icbm152_nlin_asym_09c" / "mni_icbm152_t1_tal_nlin_asym_09c.nii.gz"
            print()
            print("Next step:")
            print(f"- build {args.atlas_subject} from {icbm_seed}")
            print("- then rerun this script without --check for a smoke-test bake")
        return 0 if status.ok else 1

    if not status.ok:
        print(json.dumps(status.checks, indent=2, sort_keys=True))
        print()
        print("Prerequisite check failed. Run with --check to inspect the missing pieces.")
        return 1

    n_frames = detect_n_frames(args.atlas_volume)
    frames = args.frames if args.frames else list(range(n_frames))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediates = out_dir / "intermediates"
    intermediates.mkdir(parents=True, exist_ok=True)

    manifest = {
        "atlas_volume": str(args.atlas_volume),
        "atlas_subject": args.atlas_subject,
        "target_subject": args.target_subject,
        "hemis": args.hemis,
        "frames": frames,
        "smooth_fwhm": args.smooth_fwhm,
        "projfrac_avg": [0, 1, 0.1],
        "notes": [
            "This bake assumes atlas-side nonlinear surface alignment via mri_surf2surf sphere.reg.",
            "The accepted #36 contract requires ICBM152_fs (or equivalent) to exist as a real surface subject.",
        ],
    }
    write_manifest(out_dir, manifest)

    for hemi in args.hemis:
        hemi_dir = out_dir / hemi
        hemi_dir.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            proj_path = intermediates / f"{hemi}.frame{frame:03d}.atlas.mgh"
            fsavg_path = hemi_dir / f"{hemi}.frame{frame:03d}.fsaverage.mgh"
            smooth_path = hemi_dir / f"{hemi}.frame{frame:03d}.fsaverage.fwhm{args.smooth_fwhm:g}.mgh"

            run(
                vol2surf_cmd(
                    fsenv,
                    atlas_volume=args.atlas_volume,
                    atlas_subject=args.atlas_subject,
                    hemi=hemi,
                    frame=frame,
                    out_path=proj_path,
                ),
                env=fsenv.env,
                dry_run=args.dry_run,
            )
            run(
                surf2surf_cmd(
                    fsenv,
                    atlas_subject=args.atlas_subject,
                    target_subject=args.target_subject,
                    hemi=hemi,
                    in_path=proj_path,
                    out_path=fsavg_path,
                ),
                env=fsenv.env,
                dry_run=args.dry_run,
            )
            run(
                smooth_cmd(
                    fsenv,
                    target_subject=args.target_subject,
                    hemi=hemi,
                    in_path=fsavg_path,
                    out_path=smooth_path,
                    smooth_fwhm=args.smooth_fwhm,
                ),
                env=fsenv.env,
                dry_run=args.dry_run,
            )

    if not args.dry_run:
        consolidate_baked_atlas(out_dir)
    if not args.keep_intermediates and not args.dry_run:
        for path in intermediates.glob("*.mgh"):
            path.unlink()

    print()
    print(f"Prepared fsaverage bake under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
