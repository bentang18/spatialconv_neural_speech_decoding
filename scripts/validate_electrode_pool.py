#!/usr/bin/env python3
"""Validate whether the 3D Euclidean pool is physically load-bearing.

Hypothesis: if the pool is a physically meaningful correction over snap,
its argmax-changes and support-vector deltas should concentrate at
electrodes whose snap vertex is on a high-curvature region (sulcal walls,
fundi) — i.e., exactly the places where Zac's "uECoG doesn't conform to
sulci" observation predicts snap is wrong.

If pool changes are uniformly distributed across |curv|, the pool is
adding noise, not physics.

Two validations:

1. Curvature stratification. Bin electrodes by |curv(snap_vertex)|.
   Report per-bin: n_electrodes, argmax-change rate, mean L1 delta of
   support vectors (pool - snap).

2. Sigma sweep. Recompute pool at sigma in {0.3, 0.5, 1.0, 1.3, 2.0, 3.0, 5.0}.
   Report argmax-change rate vs snap. Monotonic + smooth = stable parameter.
   Erratic jumps = free-parameter problem.

Runs over all Phase-1 LH patients unless --patient restricts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.v14.electrode_pool import euclidean_support
from speech_decoding.v14.fsaverage_atlas import (
    DEFAULT_BNA_N_ROIS,
    DEFAULT_BNA_TREE,
    assert_full_bake,
    load_baked_atlas,
)
from speech_decoding.v14.fsaverage_projection import load_fsaverage_cache
from speech_decoding.v14.support_cache import TIER1_BNA_INDICES_1BASED, read_support_cache

DEFAULT_FS_HOME = Path("/Applications/freesurfer/8.2.0")
DEFAULT_SUBJECTS_DIR = DEFAULT_FS_HOME / "subjects"
DEFAULT_PATIENTS = ("S14", "S16", "S23", "S26", "S33", "S39", "S62")
DEFAULT_COORD_DIR = PROJECT_ROOT / "data" / "fsaverage_coords"
DEFAULT_BAKE_DIR = PROJECT_ROOT / "data" / "atlas" / "fsaverage_bake_v2d"
DEFAULT_SNAP_DIR = PROJECT_ROOT / "data" / "atlas" / "support_cache_v2d_snap"

CURV_BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0)]
SIGMA_SWEEP = (0.3, 0.5, 1.0, 1.3, 2.0, 3.0, 5.0)


def _load_lh_pial_curv(subjects_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    surf_dir = subjects_dir / "fsaverage" / "surf"
    geom = nib.freesurfer.io.read_geometry(str(surf_dir / "lh.pial"))
    curv = nib.freesurfer.io.read_morph_data(str(surf_dir / "lh.curv"))
    return np.asarray(geom[0], dtype=np.float64), np.asarray(curv, dtype=np.float64)


def _tier1_slice(atlas_lh: np.ndarray) -> np.ndarray:
    col_idx = np.array([i - 1 for i in TIER1_BNA_INDICES_1BASED], dtype=np.int64)
    return atlas_lh[:, col_idx].astype(np.float32, copy=False)


def _curvature_stratification(
    *,
    snap_support: np.ndarray,
    pool_support: np.ndarray,
    curv_at_snap: np.ndarray,
) -> list[dict]:
    snap_argmax = snap_support.argmax(axis=1)
    pool_argmax = pool_support.argmax(axis=1)
    changed = snap_argmax != pool_argmax
    l1 = np.abs(pool_support - snap_support).sum(axis=1)

    rows: list[dict] = []
    abs_curv = np.abs(curv_at_snap)
    for lo, hi in CURV_BINS:
        mask = (abs_curv >= lo) & (abs_curv < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": 0,
                         "change_rate": float("nan"), "mean_l1": float("nan"),
                         "mean_curv": float("nan"), "frac_positive_curv": float("nan")})
            continue
        rows.append({
            "bin": f"[{lo:.2f},{hi:.2f})",
            "n": n,
            "change_rate": float(changed[mask].mean()),
            "mean_l1": float(l1[mask].mean()),
            "mean_curv": float(curv_at_snap[mask].mean()),
            "frac_positive_curv": float((curv_at_snap[mask] > 0).mean()),
        })
    return rows


def _sigma_sweep(
    *,
    elec_xyz: np.ndarray,
    pial_xyz: np.ndarray,
    tier1_support: np.ndarray,
    curv: np.ndarray,
    snap_support: np.ndarray,
) -> list[dict]:
    snap_argmax = snap_support.argmax(axis=1)
    rows: list[dict] = []
    for sigma in SIGMA_SWEEP:
        radius = max(4.0, 4.0 * sigma)  # 4σ tail
        pool = euclidean_support(
            elec_xyz, pial_xyz, tier1_support, curv,
            sigma_mm=sigma, radius_mm=radius, crown_bias=False,
        )
        pool_argmax = pool.argmax(axis=1)
        l1 = np.abs(pool - snap_support).sum(axis=1).mean()
        rows.append({
            "sigma_mm": sigma,
            "radius_mm": radius,
            "change_rate": float((pool_argmax != snap_argmax).mean()),
            "mean_l1": float(l1),
        })
    return rows


def _report_patient(patient: str, curv_rows: list[dict], sigma_rows: list[dict]) -> str:
    out: list[str] = []
    out.append(f"\n### {patient}\n")
    out.append("Curvature stratification (bin by |curv| at snap vertex):\n")
    out.append("| |curv| bin | n | change_rate | mean L1 | mean curv | frac curv>0 |")
    out.append("|---|---|---|---|---|---|")
    for r in curv_rows:
        out.append(
            f"| {r['bin']} | {r['n']} | "
            f"{r['change_rate']:.3f} | {r['mean_l1']:.2f} | "
            f"{r['mean_curv']:+.3f} | {r['frac_positive_curv']:.2f} |"
        )
    out.append("\nSigma sweep (change-rate vs snap, L1 support delta):\n")
    out.append("| sigma (mm) | radius | change_rate | mean L1 |")
    out.append("|---|---|---|---|")
    for r in sigma_rows:
        out.append(
            f"| {r['sigma_mm']:.1f} | {r['radius_mm']:.1f} | "
            f"{r['change_rate']:.3f} | {r['mean_l1']:.2f} |"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", nargs="+", default=list(DEFAULT_PATIENTS))
    parser.add_argument("--coord-cache-dir", type=Path, default=DEFAULT_COORD_DIR)
    parser.add_argument("--bake-dir", type=Path, default=DEFAULT_BAKE_DIR)
    parser.add_argument("--snap-dir", type=Path, default=DEFAULT_SNAP_DIR)
    parser.add_argument("--subjects-dir", type=Path, default=DEFAULT_SUBJECTS_DIR)
    parser.add_argument("--out", type=Path,
                        default=DEFAULT_BAKE_DIR.parent / "support_cache_v2d_euclidean_nobias" / "validation_report.md")
    args = parser.parse_args(argv)

    pial_xyz, curv = _load_lh_pial_curv(args.subjects_dir)
    atlas = load_baked_atlas(args.bake_dir, kind="smoothed", tree_path=DEFAULT_BNA_TREE)
    assert_full_bake(atlas, expected_n_rois=DEFAULT_BNA_N_ROIS)
    tier1_lh = _tier1_slice(atlas.values_by_hemi["lh"])

    lines: list[str] = []
    lines.append("# Electrode pool validation report\n")
    lines.append(f"Atlas: `{args.bake_dir}` (smoothed)")
    lines.append(f"Snap cache: `{args.snap_dir}`")
    lines.append("")
    lines.append("## Hypothesis\n")
    lines.append(
        "If the 3D Euclidean pool is physically load-bearing, argmax-changes "
        "and support deltas concentrate at electrodes whose snap vertex has "
        "high |curv| (i.e., on sulcal walls/fundi rather than on gyral crowns). "
        "If distributions are uniform across |curv|, pool is noise.\n"
    )

    # Aggregate across patients, plus per-patient.
    agg_curv_rows = [
        {"bin": f"[{lo:.2f},{hi:.2f})", "n": 0, "changed": 0, "l1_sum": 0.0,
         "curv_sum": 0.0, "pos_curv_count": 0} for lo, hi in CURV_BINS
    ]
    agg_sigma_rows = {sigma: {"changed": 0, "n": 0, "l1_sum": 0.0}
                      for sigma in SIGMA_SWEEP}

    for patient in args.patient:
        cache = load_fsaverage_cache(patient, args.coord_cache_dir)
        mask = np.array([h == "L" for h in cache.hemisphere], dtype=bool)
        if not mask.all():
            print(f"[{patient}] skipping RH electrodes (not Phase-1)")
        elec_xyz = cache.coords[mask]
        vertex_ids = cache.vertex_ids[mask]
        curv_at_snap = curv[vertex_ids]

        snap_path = args.snap_dir / f"{patient}_support_tier1.csv"
        names, snap_support = read_support_cache(snap_path)
        lh_names = tuple(n for n, h in zip(names, cache.hemisphere) if h == "L")
        assert lh_names == tuple(n for n, m in zip(cache.names, mask) if m), \
            f"{patient}: name order mismatch"

        # Canonical pool (sigma=1.3 mm, no crown bias).
        pool_support = euclidean_support(
            elec_xyz, pial_xyz, tier1_lh, curv,
            sigma_mm=1.3, radius_mm=5.0, crown_bias=False,
        )
        curv_rows = _curvature_stratification(
            snap_support=snap_support, pool_support=pool_support,
            curv_at_snap=curv_at_snap,
        )
        sigma_rows = _sigma_sweep(
            elec_xyz=elec_xyz, pial_xyz=pial_xyz,
            tier1_support=tier1_lh, curv=curv,
            snap_support=snap_support,
        )
        lines.append(_report_patient(patient, curv_rows, sigma_rows))

        # Aggregate.
        abs_curv = np.abs(curv_at_snap)
        snap_argmax = snap_support.argmax(axis=1)
        pool_argmax = pool_support.argmax(axis=1)
        changed = snap_argmax != pool_argmax
        l1 = np.abs(pool_support - snap_support).sum(axis=1)
        for row, (lo, hi) in zip(agg_curv_rows, CURV_BINS):
            m = (abs_curv >= lo) & (abs_curv < hi)
            row["n"] += int(m.sum())
            row["changed"] += int(changed[m].sum())
            row["l1_sum"] += float(l1[m].sum())
            row["curv_sum"] += float(curv_at_snap[m].sum())
            row["pos_curv_count"] += int((curv_at_snap[m] > 0).sum())

        # Sigma aggregate.
        for r in sigma_rows:
            d = agg_sigma_rows[r["sigma_mm"]]
            d["changed"] += int(r["change_rate"] * len(elec_xyz))
            d["n"] += len(elec_xyz)
            d["l1_sum"] += r["mean_l1"] * len(elec_xyz)

    lines.append("\n## Pooled across patients\n")
    lines.append("\nCurvature stratification:\n")
    lines.append("| |curv| bin | n | change_rate | mean L1 | mean curv | frac curv>0 |")
    lines.append("|---|---|---|---|---|---|")
    for row in agg_curv_rows:
        n = row["n"]
        if n == 0:
            lines.append(f"| {row['bin']} | 0 | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {row['bin']} | {n} | "
            f"{row['changed']/n:.3f} | {row['l1_sum']/n:.2f} | "
            f"{row['curv_sum']/n:+.3f} | {row['pos_curv_count']/n:.2f} |"
        )
    lines.append("\nSigma sweep pooled:\n")
    lines.append("| sigma (mm) | change_rate | mean L1 |")
    lines.append("|---|---|---|")
    for sigma in SIGMA_SWEEP:
        d = agg_sigma_rows[sigma]
        if d["n"] == 0:
            continue
        lines.append(f"| {sigma:.1f} | {d['changed']/d['n']:.3f} | {d['l1_sum']/d['n']:.2f} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
