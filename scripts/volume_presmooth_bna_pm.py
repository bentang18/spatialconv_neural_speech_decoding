#!/usr/bin/env python3
"""Apply an isotropic 3D Gaussian to the BNA PM volume before projection.

Motivation (2026-04-17 kernel-ablation session): any residual uncertainty we
want to add on top of the cohort PM — ICBM152 -> fsaverage registration
residual, ribbon-sampling noise — is physically 3D and isotropic. Doing that
smoothing on the fsaverage SURFACE imports fsaverage's (fictional average)
fold geometry as a structural prior. Doing it in the VOLUME, BEFORE
projection, keeps the geometry clean.

Input: 4D NIfTI (X, Y, Z, K).
Output: 4D NIfTI with a 3D Gaussian (per-K independently) applied, same
affine and header, but dtype float32.

Default FWHM matches v2b (1.5 mm) so the bake that follows is a direct
apples-to-apples swap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ~0.4246


def presmooth(
    in_path: Path,
    out_path: Path,
    fwhm_mm: float,
) -> None:
    img = nib.load(str(in_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"expected 4D PM volume, got {data.shape}")

    zooms = img.header.get_zooms()[:3]
    sigma_mm = fwhm_mm * _FWHM_TO_SIGMA
    sigma_vox = tuple(float(sigma_mm) / float(z) for z in zooms)

    smoothed = np.empty_like(data)
    for k in range(data.shape[3]):
        smoothed[..., k] = gaussian_filter(data[..., k], sigma=sigma_vox, mode="constant", cval=0.0)
        if k % 50 == 0:
            print(f"  frame {k}/{data.shape[3]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(smoothed, img.affine, img.header)
    out_img.set_data_dtype(np.float32)
    nib.save(out_img, str(out_path))
    print(f"wrote {out_path}  (fwhm={fwhm_mm} mm, sigma_mm={sigma_mm:.3f}, sigma_vox={sigma_vox})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-path", type=Path,
                        default=Path("data/atlas/BNA_PM_4D.nii.gz"))
    parser.add_argument("--out-path", type=Path,
                        default=Path("data/atlas/BNA_PM_4D_vol_fwhm1.5.nii.gz"))
    parser.add_argument("--fwhm-mm", type=float, default=1.5)
    args = parser.parse_args()
    presmooth(args.in_path, args.out_path, args.fwhm_mm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
