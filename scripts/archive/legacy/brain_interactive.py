#!/usr/bin/env python3
"""Interactive 3D brain viewer — rotate, zoom, pan with mouse.

Uses PyVista (VTK) interactive window. Controls:
  - Left click + drag: rotate
  - Right click + drag / scroll: zoom
  - Middle click + drag: pan
  - 'r': reset camera
  - 'q': quit
  - 'p': pick point (shows coordinates)
  - 's': screenshot (saves to current directory)
"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from speech_decoding.data.atlas import (
    SPEECH_ROIS_CORE,
    _SPEECH_ROIS_EXTENDED,
    get_roi_labels,
    get_virtual_electrode_positions,
)
from speech_decoding.data.coordinates import (
    apply_talairach_transform,
    build_electrode_coordinates,
    load_talairach_transform,
)

DATA_DIR = project_root / "data"
COORDS_DIR = DATA_DIR / "mni_coords"
CHANMAP_DIR = DATA_DIR / "channel_maps"
TRANSFORM_DIR = DATA_DIR / "transforms"

CORE_PATIENTS = ["S14", "S26", "S33", "S62"]
EXTENDED_PATIENTS = ["S16", "S22", "S23", "S39", "S58"]
EXCLUDED_PATIENTS = ["S32", "S57"]
PATIENTS_128CH = {"S14", "S16", "S22", "S23", "S26"}
ALL_PATIENTS = CORE_PATIENTS + EXTENDED_PATIENTS + EXCLUDED_PATIENTS

PATIENT_COLORS = {
    "S14": [0.12, 0.47, 0.71], "S26": [0.17, 0.63, 0.17],
    "S33": [1.00, 0.50, 0.05], "S62": [0.58, 0.40, 0.74],
    "S16": [0.55, 0.34, 0.29], "S22": [0.89, 0.47, 0.76],
    "S23": [0.50, 0.50, 0.50], "S39": [0.74, 0.74, 0.13],
    "S58": [0.09, 0.75, 0.81], "S32": [0.83, 0.83, 0.83],
    "S57": [0.75, 0.75, 0.75],
}

VE_CATEGORY_COLORS = {
    "Motor": [0.84, 0.15, 0.16], "Sensory": [1.00, 0.50, 0.05],
    "Broca": [0.17, 0.63, 0.17], "Executive": [0.58, 0.40, 0.74],
    "Auditory": [0.09, 0.75, 0.81],
}

VE_CATEGORIES = {
    "A6cvl": "Motor", "A4tl": "Motor", "A4hf": "Motor",
    "A1/2/3tonIa": "Sensory", "A1/2/3ulhf": "Sensory",
    "A44d": "Broca", "A44v": "Broca", "A45c": "Broca",
    "A45i": "Broca", "A45r": "Broca",
    "MFG": "Executive", "STGpp": "Auditory",
}


def load_patient_mni(patient_id):
    ras_path = COORDS_DIR / f"{patient_id}_RAS.txt"
    xfm_path = TRANSFORM_DIR / f"{patient_id}_talairach.xfm"
    chanmap_path = (
        CHANMAP_DIR / f"{patient_id}_channelMap.mat"
        if patient_id in PATIENTS_128CH else None
    )
    elec = build_electrode_coordinates(patient_id, ras_path, chanmap_path=chanmap_path)
    if elec.hemisphere == "R":
        elec = elec.mirror_to_left()
    ch_names = list(elec.coords.keys())
    acpc = elec.to_array(ch_names)
    valid = ~np.isnan(acpc).any(axis=1)
    affine = load_talairach_transform(xfm_path)
    return apply_talairach_transform(acpc[valid], affine)


def main():
    import mne

    # ---- Load data ----
    print("Loading patient coordinates...")
    patient_mni = {}
    for pid in ALL_PATIENTS:
        try:
            patient_mni[pid] = load_patient_mni(pid)
        except Exception as e:
            print(f"  WARN {pid}: {e}")

    print("Loading fsaverage pial surface...")
    subjects_dir = str(Path(mne.datasets.fetch_fsaverage(verbose=False)).parent)
    lh_coords, lh_faces = mne.read_surface(
        str(Path(subjects_dir) / "fsaverage" / "surf" / "lh.pial")
    )

    # Build PyVista mesh
    n_faces = len(lh_faces)
    pv_faces = np.column_stack([np.full(n_faces, 3), lh_faces]).ravel()
    brain_mesh = pv.PolyData(lh_coords, pv_faces)

    # Load sulcal depth for surface shading (gyri vs sulci)
    has_sulc = False
    try:
        import nibabel.freesurfer as fs
        sulc_path = Path(subjects_dir) / "fsaverage" / "surf" / "lh.sulc"
        sulc_data = fs.read_morph_data(str(sulc_path))
        brain_mesh["sulc"] = sulc_data
        has_sulc = True
        print(f"  Loaded sulcal depth ({len(sulc_data)} vertices)")
    except Exception as e:
        print(f"  Sulcal depth not loaded: {e}")

    # ---- Build plotter ----
    print("\nOpening interactive viewer...")
    print("  Left drag: rotate | Right drag/scroll: zoom | Middle drag: pan")
    print("  'r': reset camera | 'q': quit | 's': screenshot\n")

    pl = pv.Plotter(window_size=(1600, 1200), title="uECoG Electrode + VE Viewer")

    # Brain surface with sulcal depth coloring
    if has_sulc:
        pl.add_mesh(
            brain_mesh, scalars="sulc", cmap="bone", opacity=0.3,
            smooth_shading=True, show_scalar_bar=False,
            clim=[-1, 2],
        )
    else:
        pl.add_mesh(
            brain_mesh, color="#e8d0b8", opacity=0.3,
            smooth_shading=True,
        )

    # Patient electrodes as spheres
    for pid in ALL_PATIENTS:
        if pid not in patient_mni:
            continue
        mni = patient_mni[pid]
        is_core = pid in CORE_PATIENTS
        is_excluded = pid in EXCLUDED_PATIENTS
        radius = 1.0 if is_core else (0.5 if is_excluded else 0.7)
        opacity = 0.95 if is_core else (0.15 if is_excluded else 0.35)

        cloud = pv.PolyData(mni)
        glyphs = cloud.glyph(geom=pv.Sphere(radius=radius), orient=False, scale=False)
        pl.add_mesh(glyphs, color=PATIENT_COLORS[pid], opacity=opacity,
                     smooth_shading=True, label=pid)

    # Core VEs — large spheres with labels
    ve_pos = get_virtual_electrode_positions("core")
    ve_labels = get_roi_labels("core")

    for i, label in enumerate(ve_labels):
        cat = VE_CATEGORIES[label]
        color = VE_CATEGORY_COLORS[cat]

        sphere = pv.Sphere(radius=2.5, center=ve_pos[i])
        pl.add_mesh(sphere, color=color, opacity=0.9, smooth_shading=True)

        # Black outline ring
        outline = pv.Sphere(radius=2.7, center=ve_pos[i])
        pl.add_mesh(outline, color="black", opacity=0.25, style="wireframe",
                     line_width=0.5)

        # Label
        pl.add_point_labels(
            [ve_pos[i] + np.array([3, 2, 2])], [label],
            font_size=14, text_color=color,
            point_size=0, shape=None, render_points_as_spheres=False,
            always_visible=True, bold=True,
        )

    # Extended VEs — wireframe
    ve_ext_pos = np.array([[x, y, z] for _, _, x, y, z in _SPEECH_ROIS_EXTENDED])
    ve_ext_labels = [l for l, _, _, _, _ in _SPEECH_ROIS_EXTENDED]
    for i, label in enumerate(ve_ext_labels):
        sphere = pv.Sphere(radius=1.5, center=ve_ext_pos[i])
        pl.add_mesh(sphere, color="gray", opacity=0.25, style="wireframe", line_width=1)
        pl.add_point_labels(
            [ve_ext_pos[i] + np.array([2, 1, 1])], [f"({label})"],
            font_size=9, text_color="gray",
            point_size=0, shape=None, render_points_as_spheres=False,
            always_visible=True,
        )

    # Legend
    legend_entries = []
    for pid in CORE_PATIENTS:
        legend_entries.append([f"{pid} (core)", PATIENT_COLORS[pid]])
    for pid in EXTENDED_PATIENTS:
        legend_entries.append([pid, PATIENT_COLORS[pid]])
    legend_entries.append(["", "white"])
    for cat, color in VE_CATEGORY_COLORS.items():
        legend_entries.append([f"VE: {cat}", color])
    legend_entries.append(["(ext) = dropped", "gray"])
    pl.add_legend(legend_entries, bcolor="white", face="circle", size=(0.15, 0.35))

    # Title
    pl.add_text(
        "uECoG Electrodes + 12 Core VEs on fsaverage Pial Surface\n"
        "Drag to rotate | Scroll to zoom | 's' to screenshot",
        font_size=10, position="upper_edge",
    )

    # Initial camera: left lateral
    pl.camera_position = [(-250, 0, 30), (0, 0, 0), (0, 0, 1)]
    pl.set_background("white")
    pl.enable_anti_aliasing("ssaa")

    pl.show()


if __name__ == "__main__":
    main()
