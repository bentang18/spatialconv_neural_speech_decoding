"""Rank Brainnetome parcels by Phase-1 electrode support (blocker #5 stat).

Support statistic (blocker #5, agreed 2026-04-13; kernel revised 2026-04-14):

    support(e, p) = integral of  G(x - x_e) * PM_p(x) dx
                  = (G * PM_p)(x_e)

where G is a 3D Gaussian matched to Cogan micro-ECoG high-gamma point
spread function (sigma ~= 1.5 mm, FWHM ~= 3.5 mm). This is the forward
physics model: each cortical voxel emits, and the electrode's response
is a Gaussian-weighted sum of those emissions. Both slicing directions
(per-electrode parcel membership, per-parcel effective_N) are marginals
of the same (n_electrodes x n_parcels) matrix M[e, p] = support(e, p).

    effective_N(parcel p) = sum over electrodes e of support(e, p) / 100

This replaces the earlier snap-plus-trilinear version (one point sample
per electrode) with a physically grounded spatial integration. The change
is mathematically equivalent to convolving PM with G once and then
trilinear-sampling the smoothed volume at each electrode position.

Kernel rationale (see Cogan lab microECoG physics):
- Trumpis 2020 JNE (Cogan-lab 1 mm uECoG): high-gamma PSF FWHM ~= 2 mm
- Chiang 2020:                              FWHM ~= 1.5 mm
- Viventi 2011 (500 um):                    FWHM ~= 0.5 mm
- Default here sigma=1.5 mm (FWHM 3.5 mm)   slightly wider than pure
  physics to absorb 1-3 mm snap-distance uncertainty without crossing
  into macro-ECoG territory (~5-10 mm FWHM).

Tune via ``--sigma-mm``; pass 0 to fall back to pure trilinear sampling
for A/B comparison.

Pipeline:
1. Read raw MNI electrode CSVs from ``data/mni_coords/<pt>_MNI152.csv``
   (the Python port of ``sub2AvgBrainClinical.m``). Coordinates sit on
   cvs_avg35_inMNI152's dural envelope (pial-outer-smoothed). NO pial
   snap — the ~2-4 mm envelope-to-ribbon offset is absorbed by the
   d_max=8 mm nearest-neighbor dilation in the PM volume itself
   (see ``scripts/dilate_pm.py``), so mesh-based repositioning at sample
   time is unnecessary. This keeps the sampling step identical for
   Phase-1 ECoG and Phase-2 sEEG.
2. Spatially smooth the 4D PM volume once with a Gaussian of sigma_mm,
   expressed in voxel units per axis via the PM affine. No smoothing
   along the parcel axis. Default PM is the dilated volume
   (``data/atlas/BNA_PM_dilated_8mm.nii.gz``).
3. Trilinear-sample the smoothed PM at each raw electrode position.

Reports:
- effective_N per parcel
- n_patients covering the parcel (how many of the LH patients have any)
- any_cov    (# electrodes with any nonzero PM at this parcel)
- argmax_wins (# electrodes where this parcel is the dominant one over all 246)
- max_pm     (peak smoothed PM value at any Phase-1 electrode)
- per-patient effective_N breakdown for the top-N parcels

Run::

    .venv/bin/python scripts/rank_parcels_by_support.py
    .venv/bin/python scripts/rank_parcels_by_support.py --sigma-mm 1.0
    .venv/bin/python scripts/rank_parcels_by_support.py --sigma-mm 0   # trilinear only
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib  # type: ignore[import-untyped]
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Default to the dilated PM volume. See scripts/dilate_pm.py for rationale:
# cvs_avg35's pial/dural-envelope sits 2-5 mm outside BNA's MNI152 ribbon at
# lateral gyral crowns, so raw trilinear sampling on BNA_PM_4D.nii.gz drops
# ~half of S26's electrodes into all-zero voxels. The dilated volume
# propagates PM vectors outward via scipy.ndimage.distance_transform_edt.
PM_PATH = PROJECT_ROOT / "data" / "atlas" / "BNA_PM_dilated_8mm.nii.gz"
CACHE_DIR = PROJECT_ROOT / "data" / "mni_coords"
TREE_PATH = Path.home() / "nilearn_data" / "bnatlas_tree.csv"
OUT_CSV = PROJECT_ROOT / "data" / "atlas" / "parcel_support.csv"

PATIENTS = ["S14", "S16", "S23", "S26", "S33", "S39", "S62"]
NONZERO_PCT = 1.0  # below this (%), treat as numerical noise

# Default Gaussian PSF sigma, in MNI millimeters. See module docstring
# for the physical rationale (Trumpis 2020 / Chiang 2020 / Viventi 2011).
DEFAULT_SIGMA_MM = 1.5


@dataclass(frozen=True)
class BnaLabel:
    idx: int
    region: str
    desc: str
    hemisphere: str  # "L" or "R"


def load_bna_tree(path: Path) -> dict[int, BnaLabel]:
    if not path.exists():
        raise FileNotFoundError(f"BNA tree not found at {path}")
    out: dict[int, BnaLabel] = {}
    with path.open() as f:
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            idx = int(row[0])
            region = row[1].strip()
            desc = row[4].strip()
            hem = "L" if "_L_" in region else ("R" if "_R_" in region else "?")
            out[idx] = BnaLabel(idx=idx, region=region, desc=desc, hemisphere=hem)
    return out


def load_raw_mni(patient_id: str, cache_dir: Path) -> np.ndarray:
    """Load raw MNI coordinates from the sub2AvgBrainClinical Python port.

    No pial snap. Raw coords sit on cvs_avg35's pial-outer-smoothed (dural
    envelope), ~2-4 mm outside the true pial. That outer offset is absorbed
    by the d_max=8 mm nearest-neighbor PM dilation (see scripts/dilate_pm.py),
    so no mesh-based repositioning is needed at sample time. Dropping the
    snap keeps the sampling step identical for ECoG and Phase-2 sEEG, neither
    of which depend on a pial mesh at query time.
    """
    path = cache_dir / f"{patient_id}_MNI152.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing raw MNI cache {path}; "
            f"run scripts/preprocess_mni_coordinates.py first"
        )
    coords: list[list[float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return np.array(coords, dtype=np.float64)


def voxel_sizes_mm(affine: np.ndarray) -> np.ndarray:
    """Per-axis voxel spacing in world (MNI) mm, extracted from the affine."""
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def smooth_pm_gaussian(
    pm: np.ndarray, affine: np.ndarray, sigma_mm: float
) -> np.ndarray:
    """Spatially Gaussian-smooth each of the n_rois PM channels.

    Applies a 3D Gaussian with ``sigma_mm`` (world mm) along the three
    spatial axes only; the parcel axis is left alone. Boundary mode is
    ``constant`` with ``cval=0`` so out-of-brain voxels act like zeros
    (correct for a probability atlas) rather than being reflected.

    Returns a new float32 array the same shape as ``pm``.
    """
    if sigma_mm <= 0:
        return pm
    vox_mm = voxel_sizes_mm(affine)
    sigma_vox = (sigma_mm / vox_mm).astype(np.float64)
    sigma_tuple = (float(sigma_vox[0]), float(sigma_vox[1]),
                   float(sigma_vox[2]), 0.0)
    print(f"  smoothing PM with sigma_mm={sigma_mm:.2f}  "
          f"(sigma_vox=({sigma_vox[0]:.2f}, {sigma_vox[1]:.2f}, "
          f"{sigma_vox[2]:.2f})  FWHM_mm={2.355 * sigma_mm:.2f})")
    return gaussian_filter(pm, sigma=sigma_tuple, mode="constant", cval=0.0)


def mni_to_voxel(mni_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    n = mni_xyz.shape[0]
    h = np.concatenate([mni_xyz, np.ones((n, 1))], axis=1)
    return (h @ np.linalg.inv(affine).T)[:, :3]


def sample_pm_trilinear(
    pm: np.ndarray, affine: np.ndarray, mni_xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Trilinear sample the 4D PM volume at each MNI point.

    Returns ``(samples [N, n_rois], in_bounds [N])``. Out-of-bounds points
    get zero rows and ``in_bounds[i] = False``.
    """
    vox = mni_to_voxel(mni_xyz, affine)
    shape = np.array(pm.shape[:3])
    in_bounds = np.all((vox >= 0) & (vox <= shape - 1), axis=1)
    n = mni_xyz.shape[0]
    n_rois = pm.shape[3]
    samples = np.zeros((n, n_rois), dtype=np.float64)
    if not in_bounds.any():
        return samples, in_bounds
    coords = vox[in_bounds].T  # (3, M)
    for r in range(n_rois):
        samples[in_bounds, r] = map_coordinates(
            pm[..., r], coords, order=1, mode="constant", cval=0.0
        )
    return samples, in_bounds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", nargs="+", default=PATIENTS)
    parser.add_argument("--top", type=int, default=50,
                        help="how many top LH parcels to print (default: 50)")
    parser.add_argument("--per-patient-top", type=int, default=30,
                        help="how many parcels to show in the per-patient breakdown")
    parser.add_argument("--sigma-mm", type=float, default=DEFAULT_SIGMA_MM,
                        help="Gaussian PSF sigma in MNI mm (default: 1.5). "
                             "Pass 0 to disable smoothing (pure trilinear).")
    parser.add_argument("--pm-path", type=Path, default=PM_PATH,
                        help="path to 4D BNA PM volume (default: BNA_PM_4D.nii.gz). "
                             "Point at data/atlas/BNA_PM_dilated_Xmm.nii.gz to rank "
                             "against the nearest-neighbor-dilated version.")
    args = parser.parse_args(argv)

    if not args.pm_path.exists():
        print(f"error: PM file not found at {args.pm_path}", file=sys.stderr)
        return 2

    print(f"loading PM volume: {args.pm_path}")
    pm_img = nib.load(str(args.pm_path))  # type: ignore[attr-defined]
    pm = pm_img.get_fdata(dtype=np.float32)  # type: ignore[attr-defined]
    affine = np.asarray(pm_img.affine)  # type: ignore[attr-defined]
    n_rois = pm.shape[3]
    vox_mm = voxel_sizes_mm(affine)
    print(f"  shape={pm.shape}  voxel_mm=({vox_mm[0]:.3f}, {vox_mm[1]:.3f}, "
          f"{vox_mm[2]:.3f})  affine_det={np.linalg.det(affine[:3, :3]):.3f}")

    pm = smooth_pm_gaussian(pm, affine, args.sigma_mm)

    labels = load_bna_tree(TREE_PATH)
    if len(labels) != n_rois:
        print(
            f"warning: tree has {len(labels)} labels vs PM has {n_rois} ROIs",
            file=sys.stderr,
        )

    n_patients = len(args.patients)
    per_patient_support = np.zeros((n_patients, n_rois), dtype=np.float64)
    any_cov = np.zeros(n_rois, dtype=np.int64)
    argmax_wins = np.zeros(n_rois, dtype=np.int64)
    max_pm = np.zeros(n_rois, dtype=np.float64)
    patient_n = np.zeros(n_patients, dtype=np.int64)

    for pt_idx, pt in enumerate(args.patients):
        try:
            mni = load_raw_mni(pt, CACHE_DIR)
        except FileNotFoundError as e:
            print(f"  {pt}: {e}", file=sys.stderr)
            continue
        samples, in_bounds = sample_pm_trilinear(pm, affine, mni)
        n_elec = samples.shape[0]
        patient_n[pt_idx] = n_elec
        print(
            f"  {pt}: N={n_elec}  in_bounds={in_bounds.sum()}  "
            f"any_cov={int((samples.sum(1) > 0).sum())}  "
            f"mean_max_pm={samples.max(1).mean():.2f}"
        )

        per_patient_support[pt_idx] = samples.sum(axis=0) / 100.0
        any_cov += (samples > NONZERO_PCT).sum(axis=0)
        max_pm = np.maximum(max_pm, samples.max(axis=0))

        top_parcel = samples.argmax(axis=1)
        top_vals = samples.max(axis=1)
        for i in range(n_elec):
            if top_vals[i] > NONZERO_PCT:
                argmax_wins[top_parcel[i]] += 1

    total_effective = per_patient_support.sum(axis=0)
    n_patients_covering = (per_patient_support > 0.01).sum(axis=0)

    rows: list[dict] = []
    for r in range(n_rois):
        idx = r + 1
        lbl = labels.get(idx)
        rows.append(
            {
                "bna_idx": idx,
                "region": lbl.region if lbl else "?",
                "desc": lbl.desc if lbl else "?",
                "hemisphere": lbl.hemisphere if lbl else "?",
                "effective_N": float(total_effective[r]),
                "n_patients": int(n_patients_covering[r]),
                "any_cov": int(any_cov[r]),
                "argmax_wins": int(argmax_wins[r]),
                "max_pm": float(max_pm[r]),
                **{f"eff_{pt}": float(per_patient_support[i, r])
                   for i, pt in enumerate(args.patients)},
            }
        )

    lh = [r for r in rows if r["hemisphere"] == "L"]
    rh = [r for r in rows if r["hemisphere"] == "R"]
    lh.sort(key=lambda r: -r["effective_N"])
    rh.sort(key=lambda r: -r["effective_N"])

    total_elec = int(patient_n.sum())
    print()
    print(f"Phase-1 LH electrodes: {total_elec} across {n_patients} patients")
    if args.sigma_mm > 0:
        print(f"Support stat: effective_N = sum_e (G * PM)(x_e) / 100   "
              f"[sigma={args.sigma_mm:.2f} mm, FWHM={2.355 * args.sigma_mm:.2f} mm]")
    else:
        print("Support stat: effective_N = sum_e PM(x_e) / 100   "
              "[trilinear, no smoothing]")
    print()
    print(
        f"{'rk':>3}  {'idx':>4}  {'region':<22}  {'eff_N':>7}  "
        f"{'pts':>3}  {'any':>5}  {'argmx':>5}  {'max%':>5}  desc"
    )
    print("-" * 118)
    for rk, r in enumerate(lh[: args.top], start=1):
        print(
            f"{rk:>3}  {r['bna_idx']:>4}  {r['region']:<22}  "
            f"{r['effective_N']:>7.2f}  {r['n_patients']:>3}  "
            f"{r['any_cov']:>5}  {r['argmax_wins']:>5}  "
            f"{r['max_pm']:>5.1f}  {r['desc']}"
        )

    print()
    print(f"Per-patient effective_N breakdown (top {args.per_patient_top} LH parcels)")
    print(f"{'rk':>3}  {'idx':>4}  {'region':<22}  {'total':>7}  "
          + "  ".join(f"{pt:>6}" for pt in args.patients))
    print("-" * (36 + 7 + 2 + 8 * n_patients))
    for rk, r in enumerate(lh[: args.per_patient_top], start=1):
        per_pt = "  ".join(f"{r[f'eff_{pt}']:>6.2f}" for pt in args.patients)
        print(
            f"{rk:>3}  {r['bna_idx']:>4}  {r['region']:<22}  "
            f"{r['effective_N']:>7.2f}  {per_pt}"
        )

    rh_active = [r for r in rh if r["effective_N"] > 0.05]
    if rh_active:
        print()
        print(
            f"WARNING: {len(rh_active)} RH parcels have nonzero support "
            f"(all Phase-1 patients are LH; expected near zero)"
        )
        for r in rh_active[:15]:
            print(
                f"  {r['bna_idx']:>4}  {r['region']:<22}  "
                f"eff_N={r['effective_N']:>6.2f}  any={r['any_cov']:>4}"
            )

    out_csv = OUT_CSV
    if args.pm_path != PM_PATH:
        stem = args.pm_path.name.replace(".nii.gz", "").replace(".nii", "")
        out_csv = OUT_CSV.parent / f"parcel_support_{stem}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bna_idx", "region", "desc", "hemisphere",
        "effective_N", "n_patients", "any_cov", "argmax_wins", "max_pm",
        *[f"eff_{pt}" for pt in args.patients],
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in lh + rh:
            writer.writerow(r)
    print()
    print(f"wrote full ranking -> {out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
