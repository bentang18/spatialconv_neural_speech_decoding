#!/usr/bin/env python3
"""Interactive 3D brain viewer with proposed VE expansion.

Shows current 16 core VEs + 3 proposed new VEs (Jaw M1, dLMC, Jaw S1).
New VEs shown as gold stars. Controls:
  - Left click + drag: rotate
  - Right click + drag / scroll: zoom
  - Middle click + drag: pan
  - 'r': reset camera  |  'q': quit  |  's': screenshot
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
    "Motor": [0.84, 0.15, 0.16],
    "Sensory": [1.00, 0.50, 0.05],
    "Broca": [0.17, 0.63, 0.17],
    "Executive": [0.55, 0.34, 0.29],
    "Auditory": [0.09, 0.75, 0.81],
    "Insula": [0.58, 0.40, 0.74],
}

VE_CATEGORIES = {
    "A6cvl": "Motor", "A4tl": "Motor", "A4hf": "Motor",
    "A1/2/3tonIa": "Sensory", "A1/2/3ulhf": "Sensory", "A2": "Sensory",
    "A44d": "Broca", "A44v": "Broca", "A45c": "Broca",
    "A45i": "Broca", "A45r": "Broca", "A44op": "Broca",
    "MFG": "Executive",
    "STGpp": "Auditory", "STGa": "Auditory",
    "INSa": "Insula",
}

# Proposed new VEs
PROPOSED_NEW_VES = [
    ("Jaw_M1", "M1 jaw (interpolated)", -50, -4, 24, "Motor"),
    ("dLMC", "Dorsal larynx M1", -48, -10, 48, "Motor"),
    ("Jaw_S1", "S1 jaw (interpolated)", -55, -12, 28, "Sensory"),
]


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

    # Load sulcal depth
    has_sulc = False
    try:
        import nibabel.freesurfer as fs
        sulc_path = Path(subjects_dir) / "fsaverage" / "surf" / "lh.sulc"
        sulc_data = fs.read_morph_data(str(sulc_path))
        brain_mesh["sulc"] = sulc_data
        has_sulc = True
    except Exception:
        pass

    # ---- Build plotter ----
    print("\nOpening interactive viewer...")
    print("  Left drag: rotate | Right drag/scroll: zoom | Middle drag: pan")
    print("  'r': reset | 'q': quit | 's': screenshot\n")

    pl = pv.Plotter(window_size=(1800, 1200),
                     title="VE Expansion: 16 Current + 3 Proposed")

    # Brain surface
    if has_sulc:
        pl.add_mesh(brain_mesh, scalars="sulc", cmap="bone", opacity=0.25,
                     smooth_shading=True, show_scalar_bar=False, clim=[-1, 2])
    else:
        pl.add_mesh(brain_mesh, color="#e8d0b8", opacity=0.25, smooth_shading=True)

    # Patient electrodes
    for pid in ALL_PATIENTS:
        if pid not in patient_mni:
            continue
        mni = patient_mni[pid]
        is_core = pid in CORE_PATIENTS
        is_excluded = pid in EXCLUDED_PATIENTS
        radius = 1.0 if is_core else (0.5 if is_excluded else 0.7)
        opacity = 0.9 if is_core else (0.15 if is_excluded else 0.3)

        cloud = pv.PolyData(mni)
        glyphs = cloud.glyph(geom=pv.Sphere(radius=radius), orient=False, scale=False)
        pl.add_mesh(glyphs, color=PATIENT_COLORS[pid], opacity=opacity,
                     smooth_shading=True)

    # ---- Current 16 Core VEs ----
    ve_pos = get_virtual_electrode_positions("core")
    ve_labels = get_roi_labels("core")

    for i, label in enumerate(ve_labels):
        cat = VE_CATEGORIES.get(label, "Motor")
        color = VE_CATEGORY_COLORS.get(cat, [0.5, 0.5, 0.5])

        # Solid sphere
        sphere = pv.Sphere(radius=2.5, center=ve_pos[i])
        pl.add_mesh(sphere, color=color, opacity=0.85, smooth_shading=True)

        # Black outline
        outline = pv.Sphere(radius=2.7, center=ve_pos[i])
        pl.add_mesh(outline, color="black", opacity=0.2, style="wireframe", line_width=0.5)

        # Label (compact)
        short = label.replace("1/2/3", "")
        pl.add_point_labels(
            [ve_pos[i] + np.array([3.5, 0, 3])], [short],
            font_size=10, text_color=color,
            point_size=0, shape=None, render_points_as_spheres=False,
            always_visible=True, bold=True,
        )

    # ---- NEW Proposed VEs (gold with black edge, larger) ----
    for label, name, x, y, z, cat in PROPOSED_NEW_VES:
        pos = np.array([x, y, z], dtype=float)
        base_color = VE_CATEGORY_COLORS.get(cat, [0.5, 0.5, 0.5])

        # Larger gold sphere
        sphere = pv.Sphere(radius=3.5, center=pos)
        pl.add_mesh(sphere, color=[1.0, 0.84, 0.0], opacity=0.95, smooth_shading=True)

        # Category-colored outline ring
        outline = pv.Sphere(radius=3.8, center=pos)
        pl.add_mesh(outline, color=base_color, opacity=0.6, style="wireframe", line_width=2)

        # Bold label with star prefix
        pl.add_point_labels(
            [pos + np.array([5, 0, 4])], [f"* {label}"],
            font_size=13, text_color=[0.8, 0.0, 0.0],
            point_size=0, shape=None, render_points_as_spheres=False,
            always_visible=True, bold=True,
        )

        # Distance lines to nearest existing motor VEs
        if cat == "Motor":
            for j, ve_label in enumerate(ve_labels):
                if VE_CATEGORIES.get(ve_label) == "Motor":
                    dist = np.linalg.norm(pos - ve_pos[j])
                    if dist < 25:  # only show close connections
                        line = pv.Line(pos, ve_pos[j])
                        pl.add_mesh(line, color="red", line_width=2, opacity=0.4)
                        midpt = (pos + ve_pos[j]) / 2
                        pl.add_point_labels(
                            [midpt], [f"{dist:.0f}mm"],
                            font_size=10, text_color="red",
                            point_size=0, shape=None,
                            render_points_as_spheres=False,
                            always_visible=True,
                        )

    # ---- Extended VEs (faint wireframe) ----
    for label, name, x, y, z in _SPEECH_ROIS_EXTENDED:
        sphere = pv.Sphere(radius=1.5, center=[x, y, z])
        pl.add_mesh(sphere, color="gray", opacity=0.15, style="wireframe", line_width=0.5)

    # ---- Legend (bottom-left corner, compact) ----
    legend_entries = []
    for cat_name, color in VE_CATEGORY_COLORS.items():
        legend_entries.append([f"VE: {cat_name}", color])
    legend_entries.append(["NEW (proposed)", [1.0, 0.84, 0.0]])
    legend_entries.append(["", [1, 1, 1]])
    for pid in CORE_PATIENTS:
        legend_entries.append([f"{pid}", PATIENT_COLORS[pid]])

    pl.add_legend(legend_entries, bcolor=[1, 1, 1], face="circle",
                   size=(0.12, 0.25), loc="lower left")

    # Title (compact, single line)
    pl.add_text(
        "VE Expansion: 16 Current + 3 NEW (gold)",
        font_size=9, position="upper_left", color="black",
    )

    # Camera: left lateral view
    pl.camera_position = [(-250, 0, 30), (-45, 0, 25), (0, 0, 1)]
    pl.set_background("white")
    pl.enable_anti_aliasing("ssaa")

    pl.show()


if __name__ == "__main__":
    main()
