"""BNA parity check: pre-baked patient-space CSV vs fsaverage-vertex snap.

Answers: for Stage-3 D-patients, do pre-baked `D<N>_elec_location_radius_3mm_
aparc.BN_atlas+aseg.mgz.csv` argmax labels agree with our v2c fsaverage bake
(snap-to-pial) argmax on the same electrodes?

If they agree on Tier-1-classified electrodes, the Stage-3 loader can read
pre-baked CSVs directly. If they disagree materially, we project fresh via
``fsaverage_projection.py``.

Run (from repo root):
    .venv/bin/python scripts/bna_parity_dpatients.py \\
        --patients D40,D73,D96,D79 \\
        --out reports/bna_parity_dpatients_2026_04_21
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from speech_decoding.v14.fsaverage_atlas import (
    DEFAULT_BAKE_DIR,
    load_baked_atlas,
    sample_baked_support,
)
from speech_decoding.v14.fsaverage_projection import (
    DEFAULT_BOX_ROOT,
    DEFAULT_SUBJECTS_DIR,
    FsaverageCoordinateCache,
    project_to_fsaverage,
    read_lepto,
)
from speech_decoding.v14.token_spec import DEFAULT_BASE_PARCELS

TREE_CSV = Path.home() / "nilearn_data" / "bnatlas_tree.csv"
TIER1_AREAS = {name for name, _ in DEFAULT_BASE_PARCELS}
# Snap-distance gate for sEEG depth contacts. Contacts farther than this from
# the nearest pial vertex are labeled Unknown — the snap is physically
# meaningless when the contact sits in white matter or subcortical gray.
# 3 mm matches the pre-baked CSV's voxel-count sphere radius, so the two
# pipelines compare apples-to-apples (same physical support region).
SNAP_MM_THRESHOLD = 3.0


def load_bna_area_names() -> list[str]:
    """Return 246 area-shortname_hemi labels aligned to BNA index 1..246."""

    names: list[str] = []
    with TREE_CSV.open() as f:
        for row in csv.reader(f):
            gyrus = row[1]
            area_full = row[4] if len(row) > 4 else ""
            area_short = area_full.split(",")[0].strip()
            hemi = "L" if "_L_" in gyrus else "R"
            names.append(f"{area_short}_{hemi}")
    return names


def load_prebaked_csv(path: Path) -> dict[str, str]:
    """Parse D-patient 3mm BNA CSV -> {electrode_name: argmax_label}."""

    out: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2 or not parts[1]:
                continue
            argmax, name = parts[0], parts[1]
            out[name] = argmax
    return out


def snap_distance(points: np.ndarray, surf_verts: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(surf_verts)
    d, _ = tree.query(points, k=1)
    return np.asarray(d, dtype=np.float64)


def project_dpatient(
    patient_id: str,
    box_root: Path,
    subjects_dir: Path,
) -> tuple[FsaverageCoordinateCache, np.ndarray]:
    """Project D-patient to fsaverage, allowing depth contacts.

    Returns (cache, snap_distance_mm) — per-electrode distance from raw RAS
    to nearest patient pial vertex (pre-snap).
    """

    patient_dir = box_root / "ECoG_Recon" / patient_id
    sub_surf = patient_dir / "surf"
    elec = read_lepto(patient_dir, patient_id)

    vids, xyz = project_to_fsaverage(
        sub_coords=elec.coords,
        is_left=elec.is_left,
        sub_surf_dir=sub_surf,
        subjects_dir=subjects_dir,
    )

    import nibabel as nib

    snap_d = np.full((len(elec.names),), np.nan, dtype=np.float64)
    for hem, mask in (("lh", elec.is_left), ("rh", ~elec.is_left)):
        if not mask.any():
            continue
        pial = np.asarray(
            nib.freesurfer.io.read_geometry(str(sub_surf / f"{hem}.pial"))[0],
            dtype=np.float64,
        )
        snap_d[mask] = snap_distance(elec.coords[mask], pial)

    cache = FsaverageCoordinateCache(
        names=tuple(elec.names),
        hemisphere=tuple("L" if lh else "R" for lh in elec.is_left),
        vertex_ids=vids,
        coords=xyz,
    )
    return cache, snap_d


def audit_patient(
    patient_id: str,
    box_root: Path,
    subjects_dir: Path,
    atlas_area_names: list[str],
    bake,
) -> list[dict]:
    csv_path = (
        box_root
        / "ECoG_Recon"
        / patient_id
        / "elec_recon"
        / f"{patient_id}_elec_location_radius_3mm_aparc.BN_atlas+aseg.mgz.csv"
    )
    prebaked = load_prebaked_csv(csv_path)

    cache, snap_d = project_dpatient(patient_id, box_root, subjects_dir)
    support = sample_baked_support(cache, bake)

    totals = support.sum(axis=1)
    our_argmax_idx = support.argmax(axis=1)

    rows: list[dict] = []
    for i, name in enumerate(cache.names):
        raw_ours = "Unknown" if totals[i] <= 0 else atlas_area_names[int(our_argmax_idx[i])]
        # Correct sEEG projection: gate the snap on physical proximity. Contacts
        # >SNAP_MM_THRESHOLD from the nearest pial vertex are not near cortex —
        # the snap label is noise, not signal.
        gated_ours = raw_ours if snap_d[i] <= SNAP_MM_THRESHOLD else "Unknown"
        theirs = prebaked.get(name, "<missing>")
        rows.append(
            {
                "patient": patient_id,
                "electrode": name,
                "hemisphere": cache.hemisphere[i],
                "snap_mm": float(snap_d[i]),
                "ours_argmax_raw": raw_ours,
                "ours_argmax": gated_ours,
                "theirs_argmax": theirs,
                "ours_area_raw": _area_part(raw_ours),
                "ours_area": _area_part(gated_ours),
                "theirs_area": _area_part(theirs),
                "ours_tier1_raw": _area_part(raw_ours) in TIER1_AREAS,
                "ours_tier1": _area_part(gated_ours) in TIER1_AREAS,
                "theirs_tier1": _area_part(theirs) in TIER1_AREAS,
                "exact_match_raw": raw_ours == theirs,
                "exact_match": gated_ours == theirs,
                "area_match": _area_part(gated_ours) == _area_part(theirs) and theirs != "<missing>",
            }
        )
    return rows


def _area_part(label: str) -> str:
    if label in {"Unknown", "<missing>", ""}:
        return label
    if label.endswith("_L") or label.endswith("_R"):
        return label[:-2]
    return label


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    snap = np.array([float(r["snap_mm"]) for r in rows], dtype=np.float64)

    theirs_tier1 = [r for r in rows if r["theirs_tier1"]]
    # Gated pipeline (snap ≤ threshold): this is the physically correct
    # fsaverage projection for sEEG. Compare against the pre-baked CSV on
    # contacts where BOTH pipelines say cortical.
    both_cortical_gated = [
        r for r in rows
        if r["theirs_argmax"] not in {"Unknown", "<missing>"} and r["ours_argmax"] != "Unknown"
    ]
    both_tier1_gated = [r for r in rows if r["theirs_tier1"] and r["ours_tier1"]]
    # Raw pipeline (no gate): shows the inflation we get without the snap filter.
    raw_tier1 = [r for r in rows if r["ours_tier1_raw"]]

    return {
        "n_electrodes": n,
        # --- physical proximity ---
        "mean_snap_mm": float(snap.mean()),
        "median_snap_mm": float(np.median(snap)),
        "frac_snap_le_3mm": float((snap <= 3.0).mean()),
        "frac_snap_le_5mm": float((snap <= 5.0).mean()),
        # --- raw snap pipeline (broken for sEEG depth) ---
        "raw_ours_tier1_count": len(raw_tier1),
        "raw_exact_match_overall": sum(1 for r in rows if r["exact_match_raw"]),
        # --- gated snap pipeline (correct for sEEG) ---
        "theirs_tier1_count": len(theirs_tier1),
        "gated_ours_tier1_count": sum(1 for r in rows if r["ours_tier1"]),
        "gated_both_tier1_count": len(both_tier1_gated),
        "gated_exact_match_overall": sum(1 for r in rows if r["exact_match"]),
        "gated_exact_match_both_cortical": sum(1 for r in both_cortical_gated if r["exact_match"]),
        "gated_exact_match_theirs_tier1": sum(1 for r in theirs_tier1 if r["exact_match"]),
        "gated_area_match_theirs_tier1": sum(1 for r in theirs_tier1 if r["area_match"]),
        "gated_exact_match_both_tier1": sum(1 for r in both_tier1_gated if r["exact_match"]),
        "n_both_cortical_gated": len(both_cortical_gated),
    }


def top_disagreements(rows: list[dict], k: int = 15) -> list[tuple[str, str, int]]:
    c: Counter[tuple[str, str]] = Counter()
    for r in rows:
        if r["theirs_argmax"] == "<missing>":
            continue
        if r["ours_argmax"] != r["theirs_argmax"]:
            c[(str(r["theirs_argmax"]), str(r["ours_argmax"]))] += 1
    return [(t, o, n) for (t, o), n in c.most_common(k)]


def main() -> None:
    global SNAP_MM_THRESHOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", default="D40,D73,D96,D79")
    ap.add_argument("--out", type=Path, default=Path("reports/bna_parity_dpatients_2026_04_21"))
    ap.add_argument("--box-root", type=Path, default=DEFAULT_BOX_ROOT)
    ap.add_argument("--subjects-dir", type=Path, default=DEFAULT_SUBJECTS_DIR)
    ap.add_argument("--bake-dir", type=Path, default=DEFAULT_BAKE_DIR)
    ap.add_argument("--snap-mm", type=float, default=SNAP_MM_THRESHOLD,
                    help="Snap-distance threshold (mm) to gate our fsaverage projection")
    args = ap.parse_args()
    SNAP_MM_THRESHOLD = args.snap_mm

    atlas_area_names = load_bna_area_names()
    bake = load_baked_atlas(args.bake_dir, kind="raw")
    assert len(atlas_area_names) == len(bake.frame_ids) == 246

    args.out.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    per_patient: dict[str, dict] = {}
    for pt in args.patients.split(","):
        pt = pt.strip()
        print(f"[{pt}] auditing...")
        rows = audit_patient(pt, args.box_root, args.subjects_dir, atlas_area_names, bake)
        per_patient[pt] = summarize(rows)
        all_rows.extend(rows)

    per_csv = args.out / "per_electrode.csv"
    fieldnames = [
        "patient", "electrode", "hemisphere", "snap_mm",
        "ours_argmax_raw", "ours_argmax", "theirs_argmax",
        "ours_area_raw", "ours_area", "theirs_area",
        "ours_tier1_raw", "ours_tier1", "theirs_tier1",
        "exact_match_raw", "exact_match", "area_match",
    ]
    with per_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print("\n=== per-patient summary ===")
    for pt, s in per_patient.items():
        print(f"\n[{pt}] N={s['n_electrodes']}")
        for k, v in s.items():
            print(f"  {k}: {v}")

    # Pooled summary
    pooled = summarize(all_rows)
    print("\n=== pooled across all D-patients ===")
    for k, v in pooled.items():
        print(f"  {k}: {v}")

    # Top disagreements
    print("\n=== top disagreements (theirs -> ours) ===")
    for t, o, n in top_disagreements(all_rows):
        print(f"  {n:4d}  {t}  ->  {o}")

    # Write summary json
    import json

    (args.out / "summary.json").write_text(
        json.dumps({"per_patient": per_patient, "pooled": pooled}, indent=2, sort_keys=True) + "\n"
    )
    print(f"\nWrote: {per_csv} and {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
