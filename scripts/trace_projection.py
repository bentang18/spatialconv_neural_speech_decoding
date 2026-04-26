"""ORACLE / MIGRATION REFERENCE — NOT PHASE-1 ACTIVE.

Debug/diagnostic for the retired `sub2AvgBrainClinical.m` path. Per `#36`
the Phase-1 patient-side projection is `fsaverage_projection.py`. Keep for
investigating cvs_avg35 parity questions; do not reuse as an active path.

Walk a handful of S14 and S26 electrodes through sub2AvgBrainClinical
step by step, printing the intermediate position at every stage.

sub2AvgBrainClinical has four steps per hemisphere:

    Step 1.  subject RAS -> nearest subject lh.pial-outer-smoothed vertex
    Step 2.  same vertex -> corresponding subject lh.sphere-outer-mni.reg
    Step 3.  subject sphere point -> nearest avg lh.sphere-outer.reg vertex
    Step 4.  that avg vertex -> avg lh.pial-outer-smoothed

For an electrode whose grid is anatomically fine but whose projection
lands in the wrong place, one of these four steps must be where the
drift appears. This script prints the world-space position at every
step so we can see exactly which one is broken.

S14 is oracle-matched against Zac's MATLAB, so it's the control. S26
is the suspect.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
from scipy.spatial import cKDTree


BOX = Path("/Users/bentang/Library/CloudStorage/Box-Box/ECoG_Recon")
AVG = BOX / "cvs_avg35_inMNI152"


def load_surf(path: Path) -> np.ndarray:
    result = mne.read_surface(str(path), verbose=False)
    return np.asarray(result[0], dtype=np.float64)


def trace(patient: str) -> None:
    pdir = BOX / patient

    sub_pial_outer = load_surf(pdir / "surf" / "lh.pial-outer-smoothed")
    sub_sph = load_surf(pdir / "surf" / "lh.sphere-outer-mni.reg")
    avg_sph = load_surf(AVG / "surf" / "lh.sphere-outer.reg")
    avg_pial_outer = load_surf(AVG / "surf" / "lh.pial-outer-smoothed")

    lepto = pdir / "elec_recon" / f"{patient}.LEPTO"
    lines = lepto.read_text().splitlines()
    names = [
        f"{ln.split()[0]}{ln.split()[1]}"
        for ln in (pdir / "elec_recon" / f"{patient}.electrodeNames")
        .read_text().splitlines()[2:]
    ]
    coords = np.array(
        [[float(v) for v in line.split()] for line in lines[2:]],
        dtype=np.float64,
    )

    print(f"\n{'='*90}\n{patient}\n{'='*90}")
    print(f"  sub_pial_outer  verts={len(sub_pial_outer):>7d}  "
          f"x=[{sub_pial_outer[:, 0].min():+6.1f},{sub_pial_outer[:, 0].max():+6.1f}]  "
          f"y=[{sub_pial_outer[:, 1].min():+6.1f},{sub_pial_outer[:, 1].max():+6.1f}]  "
          f"z=[{sub_pial_outer[:, 2].min():+6.1f},{sub_pial_outer[:, 2].max():+6.1f}]")
    print(f"  sub_sph         verts={len(sub_sph):>7d}  "
          f"mean_radius={np.linalg.norm(sub_sph, axis=1).mean():.2f}")
    print(f"  avg_sph         verts={len(avg_sph):>7d}  "
          f"mean_radius={np.linalg.norm(avg_sph, axis=1).mean():.2f}")
    print(f"  avg_pial_outer  verts={len(avg_pial_outer):>7d}  "
          f"x=[{avg_pial_outer[:, 0].min():+6.1f},{avg_pial_outer[:, 0].max():+6.1f}]  "
          f"y=[{avg_pial_outer[:, 1].min():+6.1f},{avg_pial_outer[:, 1].max():+6.1f}]  "
          f"z=[{avg_pial_outer[:, 2].min():+6.1f},{avg_pial_outer[:, 2].max():+6.1f}]")

    n = coords.shape[0]
    test_idx = np.linspace(0, n - 1, 5, dtype=int)

    sub_tree = cKDTree(sub_pial_outer)
    avg_tree = cKDTree(avg_sph)

    print(f"\n  Per-electrode trace (5 representative electrodes):")
    for idx in test_idx:
        x_sub = coords[idx]
        d1, i1 = sub_tree.query(x_sub, k=1)
        v_sub_pial = sub_pial_outer[i1]
        v_sub_sph = sub_sph[i1]
        d2, i2 = avg_tree.query(v_sub_sph, k=1)
        v_avg_sph = avg_sph[i2]
        v_avg_pial = avg_pial_outer[i2]

        name = names[idx] if idx < len(names) else f"idx{idx}"
        print()
        print(f"    [{name}]")
        print(f"      0 raw LEPTO           ({x_sub[0]:+7.2f},{x_sub[1]:+7.2f},{x_sub[2]:+7.2f})")
        print(f"      1 sub pial-outer   {i1:>7d}  "
              f"({v_sub_pial[0]:+7.2f},{v_sub_pial[1]:+7.2f},{v_sub_pial[2]:+7.2f})   "
              f"snap_mm={d1:.2f}")
        print(f"      2 sub sphere       {i1:>7d}  "
              f"({v_sub_sph[0]:+7.2f},{v_sub_sph[1]:+7.2f},{v_sub_sph[2]:+7.2f})")
        print(f"      3 avg sphere       {i2:>7d}  "
              f"({v_avg_sph[0]:+7.2f},{v_avg_sph[1]:+7.2f},{v_avg_sph[2]:+7.2f})   "
              f"nearest_gap={d2:.3f}")
        print(f"      4 avg pial-outer   {i2:>7d}  "
              f"({v_avg_pial[0]:+7.2f},{v_avg_pial[1]:+7.2f},{v_avg_pial[2]:+7.2f})")

    # Check: for each electrode, does the subject sphere vertex direction
    # match the avg sphere vertex direction? (both are on radius-100 spheres
    # registered to the same target; the *unit vector* from origin should be
    # nearly parallel if the registration is good.)
    sub_vids = sub_tree.query(coords, k=1)[1]
    sub_sph_pts = sub_sph[sub_vids]
    avg_sph_ids = avg_tree.query(sub_sph_pts, k=1)[1]
    avg_sph_pts = avg_sph[avg_sph_ids]

    sub_unit = sub_sph_pts / np.linalg.norm(sub_sph_pts, axis=1, keepdims=True)
    avg_unit = avg_sph_pts / np.linalg.norm(avg_sph_pts, axis=1, keepdims=True)
    cos_angle = (sub_unit * avg_unit).sum(axis=1)
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    print(f"\n  Sphere correspondence (angle between subject-sphere and avg-sphere unit vectors):")
    print(f"    mean   = {angle_deg.mean():.2f} deg")
    print(f"    median = {np.median(angle_deg):.2f} deg")
    print(f"    p95    = {np.percentile(angle_deg, 95):.2f} deg")
    print(f"    max    = {angle_deg.max():.2f} deg")

    # What region of the avg brain do the projected electrodes land in?
    avg_pial_pts = avg_pial_outer[avg_sph_ids]
    print(f"\n  Projected avg-brain world position:")
    print(f"    x mean={avg_pial_pts[:, 0].mean():+6.1f}  "
          f"range=[{avg_pial_pts[:, 0].min():+6.1f},{avg_pial_pts[:, 0].max():+6.1f}]")
    print(f"    y mean={avg_pial_pts[:, 1].mean():+6.1f}  "
          f"range=[{avg_pial_pts[:, 1].min():+6.1f},{avg_pial_pts[:, 1].max():+6.1f}]")
    print(f"    z mean={avg_pial_pts[:, 2].mean():+6.1f}  "
          f"range=[{avg_pial_pts[:, 2].min():+6.1f},{avg_pial_pts[:, 2].max():+6.1f}]")


def main() -> int:
    trace("S14")
    trace("S26")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
