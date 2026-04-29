"""ORACLE / MIGRATION REFERENCE — NOT PHASE-1 ACTIVE.

Volumetric MNI152 visualization on the retired cvs_avg35 pipeline. Per
`#36` the Phase-1 visualization target is the fsaverage surface. Kept for
oracle-side QC only.

Plot Brainnetome PM coverage under every Phase-1 LH patient's electrodes.

Uses ``nilearn.plotting.plot_stat_map`` to render MNI152 template + Brainnetome
PM overlay + electrode markers on orthogonal slices. One PNG per patient under
``data/mni_coords/viz/<pt>_pm_coverage.png``.

Why nilearn and not freeview: everything operates in MNI152 scanner RAS, so
there is no tkrRAS/cras bookkeeping to get wrong. The PM volume, the MNI152
template, and the electrode CSVs are all in the same frame by construction.

Run::

    .venv/bin/python scripts/viz_pm_coverage.py
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import datasets, plotting

from speech_decoding.v14.coordinates import load_mni_cache


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "mni_coords"
VIZ_DIR = CACHE_DIR / "viz"
PM_4D_PATH = PROJECT_ROOT / "data" / "atlas" / "BNA_PM_4D.nii.gz"
PM_TOTAL_PATH = PROJECT_ROOT / "data" / "atlas" / "BNA_PM_total.nii.gz"

PATIENTS = ["S14", "S16", "S23", "S26", "S33", "S39", "S62"]


def write_pm_total() -> Path:
    if PM_TOTAL_PATH.exists():
        return PM_TOTAL_PATH
    print(f"computing total PM volume -> {PM_TOTAL_PATH}")
    img = nib.load(str(PM_4D_PATH))  # type: ignore[attr-defined]
    data = img.get_fdata()  # type: ignore[attr-defined]
    total = data.sum(axis=3).astype(np.float32)
    out = nib.Nifti1Image(total, img.affine, img.header)  # type: ignore[attr-defined]
    nib.save(out, str(PM_TOTAL_PATH))  # type: ignore[attr-defined]
    return PM_TOTAL_PATH


def plot_patient(patient_id: str, pm_path: Path, template_path: str) -> Path:
    cache = load_mni_cache(patient_id, CACHE_DIR)
    out_path = VIZ_DIR / f"{patient_id}_pm_coverage.png"

    display = plotting.plot_stat_map(  # type: ignore[no-untyped-call]
        stat_map_img=str(pm_path),
        bg_img=template_path,
        display_mode="ortho",
        cut_coords=cache.coords.mean(axis=0).tolist(),
        threshold=1.0,
        cmap="hot",
        vmax=200.0,
        colorbar=True,
        title=f"{patient_id} | Brainnetome PM total | N={len(cache.names)}",
        draw_cross=False,
    )
    display.add_markers(  # type: ignore[union-attr]
        marker_coords=cache.coords,
        marker_color="lime",
        marker_size=12,
    )
    display.savefig(str(out_path), dpi=150)  # type: ignore[union-attr]
    display.close()  # type: ignore[union-attr]
    print(f"  wrote {out_path}")
    return out_path


def plot_all_patients_glass(pm_path: Path) -> Path:
    out_path = VIZ_DIR / "all_patients_glass.png"
    display = plotting.plot_glass_brain(
        stat_map_img=str(pm_path),
        display_mode="lzry",
        threshold=1.0,  # pyright: ignore[reportArgumentType]
        cmap="hot",
        vmax=200.0,
        colorbar=True,
        title="All Phase-1 LH patients | Brainnetome PM",
    )
    colors = {
        "S14": "red", "S16": "orange", "S23": "yellow", "S26": "green",
        "S33": "cyan", "S39": "magenta", "S62": "white",
    }
    for patient_id in PATIENTS:
        cache = load_mni_cache(patient_id, CACHE_DIR)
        display.add_markers(  # pyright: ignore[reportOptionalMemberAccess]
            marker_coords=cache.coords,
            marker_color=colors[patient_id],
            marker_size=8,
        )
    display.savefig(str(out_path), dpi=150)  # pyright: ignore[reportOptionalMemberAccess]
    display.close()  # pyright: ignore[reportOptionalMemberAccess]
    print(f"  wrote {out_path}")
    return out_path


def main() -> int:
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    pm_total = write_pm_total()

    print("loading MNI152 template")
    template = datasets.load_mni152_template(resolution=1)
    template_path = str(VIZ_DIR / "_mni152_template.nii.gz")
    nib.save(template, template_path)  # type: ignore[arg-type]

    print("rendering per-patient PM coverage plots:")
    for patient_id in PATIENTS:
        plot_patient(patient_id, pm_total, template_path)

    print("rendering all-patient glass brain:")
    plot_all_patients_glass(pm_total)

    print(f"\ndone. outputs in {VIZ_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
