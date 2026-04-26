"""ORACLE / MIGRATION REFERENCE — NOT PHASE-1 ACTIVE.

This sanity check operates on the retired cvs_avg35 MNI cache and the raw
volumetric BNA_PM_4D volume. Per `#36` the Phase-1 representation is the
baked fsaverage surface atlas. Kept as an oracle spot-check only.

Brainnetome PM coverage spot check (blocker #1 follow-up).

For every Phase-1 LH patient with a cached MNI152 electrode file, samples the
4D Brainnetome probability map and reports:

- how many electrodes have any PM coverage (sum over ROIs > 0)
- how many have strong coverage (max ROI probability >= 25%)
- for electrodes with zero PM, their MNI coordinate and hemisphere
- the top-ranked ROI per electrode (diagnostic)

This is a one-shot verification that the surface projection from
``sub2AvgBrainClinical.m`` lands on points where the Brainnetome atlas is
defined. Zac's script snaps electrodes to the outer-smoothed pial envelope
of ``cvs_avg35_inMNI152``, not into the cortical ribbon; PM is defined in
gray matter, so nonzero coverage at those surface points is not automatic.

Usage::

    .venv/bin/python scripts/check_pm_coverage.py
    .venv/bin/python scripts/check_pm_coverage.py --patient S14 S26

PM file: ``data/atlas/BNA_PM_4D.nii.gz``
MNI cache: ``data/mni_coords/<pt>_MNI152.csv`` (produced by
``scripts/preprocess_mni_coordinates.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from speech_decoding.v14.coordinates import load_mni_cache


DEFAULT_PATIENTS = ["S14", "S26", "S33", "S62", "S16", "S23", "S39"]
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "mni_coords"
DEFAULT_PM_PATH = Path(__file__).resolve().parent.parent / "data" / "atlas" / "BNA_PM_4D.nii.gz"

STRONG_THRESHOLD = 25.0  # percent, "electrode clearly inside one parcel"


def mni_to_voxel(mni_points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(affine)
    n = mni_points.shape[0]
    homogeneous = np.concatenate([mni_points, np.ones((n, 1))], axis=1)
    return (homogeneous @ inv.T)[:, :3]


def sample_pm_nearest(
    pm_data: np.ndarray, affine: np.ndarray, mni_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    voxels = mni_to_voxel(mni_points, affine)
    ix = np.round(voxels).astype(int)
    shape = np.array(pm_data.shape[:3])
    in_bounds = np.all((ix >= 0) & (ix < shape), axis=1)
    samples = np.zeros((mni_points.shape[0], pm_data.shape[3]), dtype=np.float64)
    for i, ok in enumerate(in_bounds):
        if ok:
            samples[i] = pm_data[ix[i, 0], ix[i, 1], ix[i, 2]]
    return samples, in_bounds


def summarize_patient(
    patient_id: str,
    samples: np.ndarray,
    in_bounds: np.ndarray,
    names: tuple[str, ...],
    coords: np.ndarray,
) -> dict:
    n = samples.shape[0]
    totals = samples.sum(axis=1)
    maxes = samples.max(axis=1)
    top_roi = samples.argmax(axis=1)

    any_cov = totals > 0
    strong_cov = maxes >= STRONG_THRESHOLD

    return {
        "patient_id": patient_id,
        "n": n,
        "n_in_bounds": int(in_bounds.sum()),
        "n_any_coverage": int(any_cov.sum()),
        "n_strong_coverage": int(strong_cov.sum()),
        "max_prob_median": float(np.median(maxes)),
        "max_prob_mean": float(maxes.mean()),
        "uncovered": [
            (names[i], coords[i].tolist(), int(top_roi[i]))
            for i in np.where(~any_cov)[0]
        ],
        "weak": [
            (names[i], float(maxes[i]), int(top_roi[i]))
            for i in np.where(any_cov & ~strong_cov)[0]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", nargs="+", default=DEFAULT_PATIENTS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--pm-path", type=Path, default=DEFAULT_PM_PATH)
    args = parser.parse_args(argv)

    if not args.pm_path.exists():
        print(f"error: PM file not found at {args.pm_path}", file=sys.stderr)
        return 2

    pm_img = nib.load(str(args.pm_path))
    pm = pm_img.get_fdata()
    print(
        f"PM volume: shape={pm.shape}  voxels_with_prob>0={int((pm.sum(axis=3) > 0).sum())}\n"
    )

    print(
        f"{'patient':>8}  {'N':>4}  {'in_bounds':>9}  "
        f"{'any_cov':>7}  {'strong':>6}  {'max_med':>7}  {'max_mean':>8}"
    )
    print("-" * 65)

    all_results: list[dict] = []
    for patient_id in args.patient:
        try:
            cache = load_mni_cache(patient_id, args.cache_dir)
        except FileNotFoundError:
            print(f"{patient_id:>8}  (no cache)", file=sys.stderr)
            continue
        samples, in_bounds = sample_pm_nearest(pm, pm_img.affine, cache.coords)
        summary = summarize_patient(
            patient_id, samples, in_bounds, cache.names, cache.coords
        )
        all_results.append(summary)
        print(
            f"{summary['patient_id']:>8}  {summary['n']:>4}  "
            f"{summary['n_in_bounds']:>9}  {summary['n_any_coverage']:>7}  "
            f"{summary['n_strong_coverage']:>6}  "
            f"{summary['max_prob_median']:>7.2f}  {summary['max_prob_mean']:>8.2f}"
        )

    print("\n--- Detail: electrodes with zero PM coverage ---")
    any_uncovered = False
    for summary in all_results:
        if summary["uncovered"]:
            any_uncovered = True
            print(f"\n{summary['patient_id']} ({len(summary['uncovered'])} uncovered):")
            for name, coord, _ in summary["uncovered"]:
                print(f"  {name}: ({coord[0]:7.2f}, {coord[1]:7.2f}, {coord[2]:7.2f})")
    if not any_uncovered:
        print("  (none)")

    print("\n--- Detail: electrodes with weak coverage (0 < max_prob < 25%) ---")
    any_weak = False
    for summary in all_results:
        if summary["weak"]:
            any_weak = True
            print(f"\n{summary['patient_id']} ({len(summary['weak'])} weak):")
            for name, max_prob, roi in summary["weak"][:15]:
                print(f"  {name}: max_prob={max_prob:5.2f}%  top_roi={roi}")
            if len(summary["weak"]) > 15:
                print(f"  ... and {len(summary['weak']) - 15} more")
    if not any_weak:
        print("  (none)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
