"""ORACLE / MIGRATION REFERENCE — NOT PHASE-1 ACTIVE.

This cross-tab runs on the retired cvs_avg35 + 8 mm dilation pipeline
(envelope-vs-pial chaining, volumetric BNA sampling). Per `#36` the active
equivalent samples the baked fsaverage atlas at each electrode's fsaverage
vertex. Keep for diagnostic sanity checks on the oracle; do not use as the
Phase-1 representation.

Per-electrode atlas cross-tab: Destrieux + Brainnetome, for every Phase-1 LH electrode.

For each electrode in ``data/mni_coords/<pt>_MNI152.csv``:
  - nearest vertex on ``cvs_avg35_inMNI152/lh.pial-outer-smoothed`` (the dural
    envelope that electrodes are projected onto). The envelope is the correct
    matching target because it is the surface the projection lands on; using
    ``lh.pial`` gives false labels at high-curvature crowns where the envelope
    sits 2-9 mm outside the true pial and the nearest-pial vertex can sit on
    the wrong bank of a sulcus.
  - Destrieux (``aparc.a2009s``) annotation is defined on ``lh.pial``, so after
    we find the nearest envelope vertex, we match it back to the nearest
    ``lh.pial`` vertex (which is usually within 3 mm) and read the annot there.
  - Brainnetome argmax across all 246 parcels, on the frozen dilated +
    Gaussian-smoothed volume (matches the ``token_mask`` rule exactly).

Outputs a CSV at ``data/atlas/electrode_atlas_cross.csv`` with one row per
electrode and prints a cohort-level cross-tab plus any electrodes that land
in disputed parietal/temporal territory.

Run::

    .venv/bin/python scripts/query_atlas_at_electrodes.py
    .venv/bin/python scripts/query_atlas_at_electrodes.py --patient S23

Requires ``SUBJECTS_DIR`` to contain ``cvs_avg35_inMNI152``.
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import os
import sys
from pathlib import Path

import nibabel as nib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PM_PATH = PROJECT_ROOT / "data" / "atlas" / "BNA_PM_dilated_8mm.nii.gz"
COORDS_DIR = PROJECT_ROOT / "data" / "mni_coords"
BNA_TREE = Path.home() / "nilearn_data" / "bnatlas_tree.csv"
OUT_CSV = PROJECT_ROOT / "data" / "atlas" / "electrode_atlas_cross.csv"

PATIENTS = ["S14", "S16", "S23", "S26", "S33", "S39", "S62"]
DEFAULT_SIGMA_MM = 1.5


def load_bna_tree(path: Path) -> dict[int, str]:
    """1-indexed BNA idx -> region name (e.g. 157 -> 'PoG_L_4_2')."""
    out: dict[int, str] = {}
    with path.open() as f:
        for row in csv_mod.reader(f):
            if len(row) >= 5:
                out[int(row[0])] = row[1].strip()
    return out


def load_cvs_avg35_surfaces(subjects_dir: Path) -> dict:
    """Load pial, pial-outer-smoothed, and Destrieux annot for cvs_avg35_inMNI152.

    Returns a dict with ``pial_verts``, ``env_verts`` (envelope pial-outer-smoothed),
    ``annot_labels`` (per-pial-vertex), and ``annot_names`` (label table).
    Both vertex arrays are converted from tkrRAS to scanner RAS (= MNI152)
    via the cras offset from ``mri/orig.mgz``.
    """
    avg = subjects_dir / "cvs_avg35_inMNI152"
    pial_path = avg / "surf" / "lh.pial"
    env_path = avg / "surf" / "lh.pial-outer-smoothed"
    annot_path = avg / "label" / "lh.aparc.a2009s.annot"
    orig_path = avg / "mri" / "orig.mgz"

    for p in (pial_path, env_path, annot_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    pial_verts = np.asarray(
        nib.freesurfer.io.read_geometry(str(pial_path))[0], dtype=np.float64  # type: ignore[attr-defined]
    )
    env_verts = np.asarray(
        nib.freesurfer.io.read_geometry(str(env_path))[0], dtype=np.float64  # type: ignore[attr-defined]
    )

    if orig_path.exists():
        orig = nib.load(str(orig_path))  # type: ignore[attr-defined]
        cras = np.asarray(orig.header["Pxyz_c"], dtype=np.float64)  # type: ignore[attr-defined]
    else:
        cras = np.zeros(3)
        print(f"warning: no {orig_path}; assuming cras=0", file=sys.stderr)

    pial_verts = pial_verts + cras
    env_verts = env_verts + cras

    labels, _, names = nib.freesurfer.io.read_annot(str(annot_path))  # type: ignore[attr-defined]
    names_s = [n.decode() if isinstance(n, bytes) else n for n in names]

    return dict(
        pial_verts=pial_verts,
        env_verts=env_verts,
        annot_labels=labels,
        annot_names=names_s,
        cras=cras,
    )


def smooth_pm(pm: np.ndarray, affine: np.ndarray, sigma_mm: float) -> np.ndarray:
    if sigma_mm <= 0:
        return pm
    vox_mm = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    sigma_vox = (sigma_mm / vox_mm).astype(np.float64)
    return gaussian_filter(
        pm, sigma=(float(sigma_vox[0]), float(sigma_vox[1]), float(sigma_vox[2]), 0.0),
        mode="constant", cval=0.0,
    )


def sample_bna(
    pm: np.ndarray, affine: np.ndarray, xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (argmax_bna [N, 1-indexed], max_support [N])."""
    inv = np.linalg.inv(affine)
    n = xyz.shape[0]
    h = np.c_[xyz, np.ones(n)].T
    vox = (inv @ h)[:3]
    n_rois = pm.shape[3]
    support = np.zeros((n_rois, n), dtype=np.float32)
    for pi in range(n_rois):
        support[pi] = map_coordinates(pm[..., pi], vox, order=1, mode="constant", cval=0)
    argmax = support.argmax(axis=0) + 1
    maxs = support.max(axis=0)
    return argmax.astype(int), maxs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", nargs="+", default=PATIENTS)
    parser.add_argument("--patient", default=None,
                        help="shortcut for --patients <single> (detailed dump)")
    parser.add_argument("--sigma-mm", type=float, default=DEFAULT_SIGMA_MM)
    parser.add_argument("--pm-path", type=Path, default=PM_PATH)
    parser.add_argument("--subjects-dir", type=Path,
                        default=Path(os.environ.get("SUBJECTS_DIR", "")))
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = parser.parse_args(argv)

    if args.patient:
        args.patients = [args.patient]
    if not args.subjects_dir or not args.subjects_dir.exists():
        print(f"error: SUBJECTS_DIR not set or missing: {args.subjects_dir}",
              file=sys.stderr)
        return 2

    print(f"loading cvs_avg35_inMNI152 surfaces + annot from {args.subjects_dir}")
    surf = load_cvs_avg35_surfaces(args.subjects_dir)
    pial_verts = surf["pial_verts"]
    env_verts = surf["env_verts"]
    annot_labels = surf["annot_labels"]
    annot_names = surf["annot_names"]
    print(f"  lh.pial: {len(pial_verts)} verts  y=[{pial_verts[:,1].min():.1f}, "
          f"{pial_verts[:,1].max():.1f}]  cras={surf['cras']}")
    print(f"  lh.pial-outer-smoothed: {len(env_verts)} verts")
    env_tree = cKDTree(env_verts)
    pial_tree = cKDTree(pial_verts)

    print(f"loading BNA PM: {args.pm_path}")
    pm_img = nib.load(str(args.pm_path))  # type: ignore[attr-defined]
    pm = pm_img.get_fdata(dtype=np.float32)  # type: ignore[attr-defined]
    affine = np.asarray(pm_img.affine)  # type: ignore[attr-defined]
    pm_sm = smooth_pm(pm, affine, args.sigma_mm)
    bna = load_bna_tree(BNA_TREE)

    rows: list[dict] = []
    for pt in args.patients:
        csv_path = COORDS_DIR / f"{pt}_MNI152.csv"
        if not csv_path.exists():
            print(f"  {pt}: missing {csv_path}", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        xyz = df[["x", "y", "z"]].values.astype(np.float64)
        names_col = df["name"].tolist() if "name" in df.columns else [
            str(i) for i in range(len(xyz))
        ]

        # Nearest vertex on envelope (what electrodes were projected to)
        d_env, env_vid = env_tree.query(xyz, k=1)
        # Chain envelope vertex -> nearest pial vertex (annot lives on pial)
        _, pial_vid = pial_tree.query(env_verts[env_vid], k=1)
        # Also direct nearest pial (for comparison)
        d_pial_direct, _ = pial_tree.query(xyz, k=1)

        dest_labels = [
            annot_names[annot_labels[v]] if annot_labels[v] >= 0 else "unknown"
            for v in pial_vid
        ]

        argmax_bna, max_support = sample_bna(pm_sm, affine, xyz)

        for i, name in enumerate(names_col):
            rows.append(dict(
                pt=pt, electrode=name,
                x=float(xyz[i, 0]), y=float(xyz[i, 1]), z=float(xyz[i, 2]),
                d_env_mm=float(d_env[i]),
                d_pial_direct_mm=float(d_pial_direct[i]),
                destrieux=dest_labels[i],
                bna_idx=int(argmax_bna[i]),
                bna_name=bna.get(int(argmax_bna[i]), "?"),
                bna_support=float(max_support[i]),
            ))

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}  ({len(out)} electrodes)")

    print(f"\nEnvelope-match distance: median={out['d_env_mm'].median():.2f} mm, "
          f"max={out['d_env_mm'].max():.2f} mm")

    print("\n=== Destrieux label distribution ===")
    print(out["destrieux"].value_counts().to_string())

    interesting = out["destrieux"].str.contains(
        "Supramar|parietal|Angular|pariet", case=False, na=False
    )
    print(f"\nElectrodes labeled parietal/SMG/angular by Destrieux: {int(interesting.sum())}")
    if interesting.any():
        print(out[interesting].sort_values(by="y").to_string(index=False))  # type: ignore[call-overload]

    # Cross-tab at the top level
    print("\n=== (Destrieux x BNA) cross-tab ===")
    ct = out.groupby(["destrieux", "bna_name"]).size().reset_index(name="n")  # type: ignore[call-overload]
    ct = ct.sort_values(by="n", ascending=False).head(25)  # type: ignore[call-overload]
    print(ct.to_string(index=False))

    # Per-patient focused dump if --patient was used
    if len(args.patients) == 1:
        pt = args.patients[0]
        print(f"\n=== {pt} electrode-by-electrode (sorted by y) ===")
        sub = out[out["pt"] == pt].sort_values(by="y")  # type: ignore[call-overload]
        print(sub[["electrode", "x", "y", "z", "d_env_mm",
                   "destrieux", "bna_name", "bna_support"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
