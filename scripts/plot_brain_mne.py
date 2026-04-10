#!/usr/bin/env python3
"""Plot electrodes + VEs on fsaverage pial surface using MNE Brain (PyVista/VTK).

Generates high-quality 3D brain renders — the gold standard for ECoG visualization.
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
OUTPUT_DIR = project_root / "docs" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

CORE_PATIENTS = ["S14", "S26", "S33", "S62"]
EXTENDED_PATIENTS = ["S16", "S22", "S23", "S39", "S58"]
EXCLUDED_PATIENTS = ["S32", "S57"]
PATIENTS_128CH = {"S14", "S16", "S22", "S23", "S26"}
ALL_PATIENTS = CORE_PATIENTS + EXTENDED_PATIENTS + EXCLUDED_PATIENTS

PATIENT_COLORS = {
    "S14": [0.12, 0.47, 0.71],   # blue
    "S26": [0.17, 0.63, 0.17],   # green
    "S33": [1.00, 0.50, 0.05],   # orange
    "S62": [0.58, 0.40, 0.74],   # purple
    "S16": [0.55, 0.34, 0.29],   # brown
    "S22": [0.89, 0.47, 0.76],   # pink
    "S23": [0.50, 0.50, 0.50],   # gray
    "S39": [0.74, 0.74, 0.13],   # olive
    "S58": [0.09, 0.75, 0.81],   # cyan
    "S32": [0.83, 0.83, 0.83],   # light gray
    "S57": [0.75, 0.75, 0.75],   # silver
}

VE_CATEGORY_COLORS = {
    "Motor": [0.84, 0.15, 0.16],     # red
    "Sensory": [1.00, 0.50, 0.05],   # orange
    "Broca": [0.17, 0.63, 0.17],     # green
    "Executive": [0.58, 0.40, 0.74], # purple
    "Auditory": [0.09, 0.75, 0.81],  # cyan
}

VE_CATEGORIES = {
    "A6cvl": "Motor", "A4tl": "Motor", "A4hf": "Motor",
    "A1/2/3tonIa": "Sensory", "A1/2/3ulhf": "Sensory",
    "A44d": "Broca", "A44v": "Broca", "A45c": "Broca",
    "A45i": "Broca", "A45r": "Broca",
    "MFG": "Executive",
    "STGpp": "Auditory",
}


def load_patient_mni(patient_id: str) -> np.ndarray:
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
    acpc = acpc[valid]
    affine = load_talairach_transform(xfm_path)
    return apply_talairach_transform(acpc, affine)


def get_fsaverage_pial():
    """Load fsaverage pial surface as PyVista mesh."""
    import mne
    subjects_dir = str(Path(mne.datasets.fetch_fsaverage(verbose=False)).parent)
    # Read FreeSurfer surface
    lh_path = Path(subjects_dir) / "fsaverage" / "surf" / "lh.pial"
    coords, faces = mne.read_surface(str(lh_path))
    # Convert to MNI by applying the tkr-to-scanner transform
    # fsaverage surface coords are in "surface RAS" — for fsaverage they
    # are already in MNI305 space (Talairach), which is close enough to MNI152
    # for visualization purposes.

    # Build PyVista mesh
    n_faces = len(faces)
    pv_faces = np.column_stack([np.full(n_faces, 3), faces]).ravel()
    mesh = pv.PolyData(coords, pv_faces)
    return mesh


def render_view(plotter, patient_mni, mesh, view_name, camera_position,
                show_labels=True, show_extended=False):
    """Add brain + electrodes + VEs to a plotter at a specific view."""
    # Brain surface — translucent
    plotter.add_mesh(
        mesh, color="#e8d0b8", opacity=0.25,
        smooth_shading=True, specular=0.2,
    )

    # Patient electrodes
    for pid in ALL_PATIENTS:
        if pid not in patient_mni:
            continue
        mni = patient_mni[pid]
        is_core = pid in CORE_PATIENTS
        is_excluded = pid in EXCLUDED_PATIENTS
        radius = 1.2 if is_core else (0.6 if is_excluded else 0.8)
        opacity = 0.95 if is_core else (0.15 if is_excluded else 0.35)
        color = PATIENT_COLORS[pid]

        cloud = pv.PolyData(mni)
        glyphs = cloud.glyph(geom=pv.Sphere(radius=radius), orient=False, scale=False)
        plotter.add_mesh(glyphs, color=color, opacity=opacity, smooth_shading=True)

    # Core VEs — large colored spheres with labels
    ve_pos = get_virtual_electrode_positions("core")
    ve_labels = get_roi_labels("core")

    for i, label in enumerate(ve_labels):
        cat = VE_CATEGORIES[label]
        color = VE_CATEGORY_COLORS[cat]
        sphere = pv.Sphere(radius=2.5, center=ve_pos[i])
        plotter.add_mesh(sphere, color=color, opacity=0.9, smooth_shading=True)

        # Black outline
        outline = pv.Sphere(radius=2.7, center=ve_pos[i])
        plotter.add_mesh(outline, color="black", opacity=0.3, style="wireframe",
                         line_width=0.5)

        if show_labels:
            # Label offset to avoid overlap
            offset = np.array([3, 2, 2])
            plotter.add_point_labels(
                [ve_pos[i] + offset], [label],
                font_size=10, text_color=color,
                point_size=0, shape=None, render_points_as_spheres=False,
                always_visible=True, bold=True,
            )

    # Extended VEs — small hollow markers
    if show_extended:
        ve_ext_pos = np.array([[x, y, z] for _, _, x, y, z in _SPEECH_ROIS_EXTENDED])
        ve_ext_labels = [l for l, _, _, _, _ in _SPEECH_ROIS_EXTENDED]
        for i, label in enumerate(ve_ext_labels):
            sphere = pv.Sphere(radius=1.5, center=ve_ext_pos[i])
            plotter.add_mesh(sphere, color="gray", opacity=0.3, style="wireframe",
                             line_width=1)

    plotter.camera_position = camera_position
    plotter.set_background("white")


def main():
    # Use offscreen rendering for screenshots
    pv.OFF_SCREEN = True

    print("Loading patient coordinates...")
    patient_mni = {}
    for pid in ALL_PATIENTS:
        try:
            patient_mni[pid] = load_patient_mni(pid)
            print(f"  {pid}: {len(patient_mni[pid])} electrodes")
        except Exception as e:
            print(f"  WARN {pid}: {e}")

    print("Loading fsaverage pial surface...")
    mesh = get_fsaverage_pial()
    print(f"  {mesh.n_points} vertices, {mesh.n_cells} faces")

    # --- Multi-view panel ---
    views = {
        "lateral": [(-250, 0, 30), (0, 0, 0), (0, 0, 1)],
        "lateral_tilted": [(-220, 50, 80), (-10, 0, 20), (0, 0, 1)],
        "ventral_lateral": [(-200, -60, -30), (-10, 0, 10), (0, 0, 1)],
        "anterior": [(0, 250, 30), (0, 0, 0), (0, 0, 1)],
    }

    print("Rendering 4-panel view...")
    pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(2400, 1800))

    for idx, (name, cam) in enumerate(views.items()):
        row, col = divmod(idx, 2)
        pl.subplot(row, col)
        show_labels = name in ("lateral", "lateral_tilted")
        show_ext = name == "lateral_tilted"
        render_view(pl, patient_mni, mesh, name, cam,
                    show_labels=show_labels, show_extended=show_ext)
        pl.add_text(name.replace("_", " ").title(), font_size=12, position="upper_left")

    out_path = OUTPUT_DIR / "brain_mne_4panel.png"
    pl.screenshot(str(out_path))
    pl.close()
    print(f"Saved: {out_path}")

    # --- Single high-res lateral view ---
    print("Rendering high-res lateral view...")
    pl2 = pv.Plotter(off_screen=True, window_size=(2400, 1800))
    render_view(pl2, patient_mni, mesh, "lateral",
                views["lateral"], show_labels=True, show_extended=True)

    # Add legend
    legend_entries = []
    for pid in CORE_PATIENTS:
        legend_entries.append([f"{pid} (core)", PATIENT_COLORS[pid]])
    for pid in EXTENDED_PATIENTS:
        legend_entries.append([f"{pid}", PATIENT_COLORS[pid]])
    for cat, color in VE_CATEGORY_COLORS.items():
        legend_entries.append([f"VE: {cat}", color])
    pl2.add_legend(legend_entries, bcolor="white", face="circle", size=(0.15, 0.3))

    pl2.add_text(
        "Electrode Arrays + 12 Core VEs on fsaverage Pial Surface",
        font_size=14, position="upper_edge",
    )

    out_path2 = OUTPUT_DIR / "brain_mne_lateral.png"
    pl2.screenshot(str(out_path2))
    pl2.close()
    print(f"Saved: {out_path2}")

    # --- Zoomed Broca's view ---
    print("Rendering Broca's area zoom...")
    pl3 = pv.Plotter(off_screen=True, window_size=(2400, 1800))
    render_view(pl3, patient_mni, mesh, "broca_zoom",
                [(-180, 120, -20), (-50, 20, 10), (0, 0, 1)],
                show_labels=True, show_extended=True)
    pl3.add_text("Broca's Area Zoom", font_size=14, position="upper_left")
    out_path3 = OUTPUT_DIR / "brain_mne_broca_zoom.png"
    pl3.screenshot(str(out_path3))
    pl3.close()
    print(f"Saved: {out_path3}")


if __name__ == "__main__":
    main()
