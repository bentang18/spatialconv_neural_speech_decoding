"""Dilate BNA PM outward by Euclidean nearest-neighbor up to d_max mm.

Rationale (2026-04-14):
cvs_avg35_inMNI152/lh.pial extends 2-5 mm beyond where BNA's cortical
ribbon is defined at lateral gyral crowns. The cvs_avg35 surface was
built from a different FreeSurfer CVS average and nonlinearly warped
into MNI152 after the fact, so its pial mesh does not line up with
the MNI152 voxel grid on which BNA_PM_4D.nii.gz was defined. At lateral
high-curvature crowns the mismatch reaches ~8 mm in an ~1 cm wide pocket.
Electrodes snapped to that pial land in all-zero PM voxels even though
the cortex is anatomically legitimate (S26 loses ~90/128 electrodes to
this pocket; S14 loses almost none because its grid happens to sit over
the dorsal-frontal strip which is well covered).

Fix: a one-shot preprocessing step that propagates each unlabeled voxel
within d_max mm of the ribbon to inherit the PM vector of its nearest
in-ribbon voxel (straight-line Euclidean). Voxels beyond d_max stay zero
so their electrodes remain unsupported and get token_mask=False at load
time. The dilated volume is treated identically to the original by every
downstream stage -- Gaussian PSF smoothing, trilinear sampling, parcel
ranking -- because the forward physics (G * PM) does not care whether
a given voxel got its PM value from the original probability map or from
nearest-neighbor propagation; it only cares that the PM field is now
defined on a cortical ribbon that actually matches the pial surface we
sample from.

Why this instead of electrode-snapping:
- electrodes stay at their actual projected positions
- one-shot precomputation caches to disk as a versioned artifact
- composes trivially with the existing sigma=1.5 mm Gaussian smoothing
  in rank_parcels_by_support.py (just point the loader at the dilated file)
- visualizable: re-render plot_phase1_parcels.m against the dilated
  volume and look for wrong-side-of-sulcus propagation

Sharp edge: distance_transform_edt is straight-line Euclidean, not
geodesic along the cortex. At very thin sulci a propagating voxel may
inherit from the opposite bank instead of the same gyrus. The production
default (d_max = 8 mm) is the minimum radius that recovers S26's far-
lateral pocket. A safer fallback (d_max = 5 mm) is available via
``--d-max 5`` for cases where visual inspection of ``plot_phase1_parcels.m``
shows cross-sulcal leakage at 8 mm; the 5 mm variant leaves ~16 S26
electrodes below the 10% token_mask threshold. Debug by rendering the
dilated volume on cvs_avg35 pial and spot-checking STS, IFS, CS.

Output: data/atlas/BNA_PM_dilated_{d_max}mm.nii.gz

Run::

    .venv/bin/python scripts/dilate_pm.py               # production default: 8 mm
    .venv/bin/python scripts/dilate_pm.py --d-max 5     # safer fallback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib  # type: ignore[import-untyped]
import numpy as np
from scipy.ndimage import distance_transform_edt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PM_PATH = PROJECT_ROOT / "data" / "atlas" / "BNA_PM_4D.nii.gz"


def voxel_sizes_mm(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def dilate_pm(
    pm: np.ndarray, voxel_mm: np.ndarray, d_max_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate PM vectors outward into unlabeled voxels within d_max mm.

    Parameters
    ----------
    pm
        4D (X, Y, Z, n_rois) probability map with background as all-zero
        in the parcel axis.
    voxel_mm
        Per-axis voxel spacing in world (MNI) millimeters.
    d_max_mm
        Maximum outward propagation distance in millimeters. Voxels
        further than this from the ribbon keep their all-zero PM vector.

    Returns
    -------
    (dilated, d_field) where ``dilated`` is a new float32 array the same
    shape as ``pm`` and ``d_field`` is the per-voxel distance in mm from
    the nearest in-ribbon voxel (zero inside the ribbon).
    """
    any_pm = pm.sum(axis=3) > 0  # [X, Y, Z] ribbon mask
    d_mm, idx = distance_transform_edt(  # type: ignore[misc]
        ~any_pm, sampling=tuple(float(v) for v in voxel_mm), return_indices=True
    )
    d_mm = np.asarray(d_mm, dtype=np.float32)
    idx = np.asarray(idx)  # [3, X, Y, Z]

    propagate = (d_mm > 0) & (d_mm <= d_max_mm)
    dilated = pm.astype(np.float32, copy=True)
    ii = idx[0][propagate]
    jj = idx[1][propagate]
    kk = idx[2][propagate]
    dilated[propagate] = pm[ii, jj, kk]
    return dilated, d_mm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--d-max", type=float, default=8.0,
        help="maximum outward propagation distance in mm (default 8, the "
             "production value; pass --d-max 5 for the safer fallback)",
    )
    parser.add_argument("--in-path", type=Path, default=PM_PATH)
    parser.add_argument(
        "--out-path", type=Path, default=None,
        help="output NIfTI path; default data/atlas/BNA_PM_dilated_{d_max}mm.nii.gz",
    )
    args = parser.parse_args(argv)

    if not args.in_path.exists():
        print(f"error: PM file not found at {args.in_path}", file=sys.stderr)
        return 2

    out_path = args.out_path or (
        PROJECT_ROOT / "data" / "atlas"
        / f"BNA_PM_dilated_{args.d_max:g}mm.nii.gz"
    )

    print(f"loading {args.in_path}")
    img = nib.load(str(args.in_path))  # type: ignore[attr-defined]
    pm = img.get_fdata(dtype=np.float32)  # type: ignore[attr-defined]
    affine = np.asarray(img.affine)  # type: ignore[attr-defined]
    vox_mm = voxel_sizes_mm(affine)
    print(
        f"  shape={pm.shape}  voxel_mm=({vox_mm[0]:.3f},{vox_mm[1]:.3f},{vox_mm[2]:.3f})"
    )

    any_pm_in = pm.sum(axis=3) > 0
    n_in = int(any_pm_in.sum())
    print(
        f"  pre-dilation ribbon voxels: {n_in:>8d}  "
        f"({n_in / any_pm_in.size * 100:.2f}% of bbox)"
    )

    print(f"dilating, d_max={args.d_max:g} mm")
    dilated, d_field = dilate_pm(pm, vox_mm, args.d_max)

    any_pm_out = dilated.sum(axis=3) > 0
    n_out = int(any_pm_out.sum())
    added = int((any_pm_out & ~any_pm_in).sum())
    print(
        f"  post-dilation ribbon voxels: {n_out:>8d}  "
        f"(+{added} voxels added, +{added / max(n_in, 1) * 100:.1f}%)"
    )

    prop_mask = (d_field > 0) & (d_field <= args.d_max)
    if prop_mask.any():
        dv = d_field[prop_mask]
        print("  propagation distances (mm):")
        print(
            f"    mean={dv.mean():.2f}  median={np.median(dv):.2f}  "
            f"p90={np.percentile(dv, 90):.2f}  max={dv.max():.2f}"
        )
        bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        for lo, hi in zip([0.0] + bins[:-1], bins):
            n = int(((dv > lo) & (dv <= hi)).sum())
            print(f"    ({lo:4.1f}, {hi:4.1f}]  {n:>7d}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_img = nib.Nifti1Image(  # type: ignore[attr-defined]
        dilated, affine=affine, header=img.header  # type: ignore[attr-defined]
    )
    nib.save(new_img, str(out_path))  # type: ignore[attr-defined]
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
