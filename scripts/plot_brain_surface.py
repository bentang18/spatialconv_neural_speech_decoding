#!/usr/bin/env python3
"""Plot electrodes + VEs on 3D fsaverage cortical surface using nilearn.

Generates interactive HTML (rotatable in browser) and static PNG views.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    "S14": "#1f77b4", "S26": "#2ca02c", "S33": "#ff7f0e", "S62": "#9467bd",
    "S16": "#8c564b", "S22": "#e377c2", "S23": "#7f7f7f", "S39": "#bcbd22",
    "S58": "#17becf", "S32": "#d3d3d3", "S57": "#c0c0c0",
}

VE_CATEGORIES = {
    "A4hf": "Motor", "A4tl": "Motor",
    "A6cdl": "Premotor", "A6cvl": "Premotor",
    "A1/2/3ulhf": "Sensory", "A1/2/3tonIa": "Sensory", "A2": "Sensory",
    "A44d": "Broca", "A44v": "Broca", "A45c": "Broca",
}

VE_CATEGORY_COLORS = {
    "Motor": "#d62728", "Sensory": "#ff7f0e",
    "Broca": "#2ca02c", "Premotor": "#9467bd",
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


def plot_surface_matplotlib(patient_mni: dict, output_path: Path):
    """Render electrodes on fsaverage pial surface using matplotlib (static)."""
    from nilearn import datasets
    from nilearn.surface import load_surf_mesh

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
    pial_left = load_surf_mesh(fsaverage.pial_left)
    coords_surf = pial_left.coordinates
    faces_surf = pial_left.faces

    ve_pos = get_virtual_electrode_positions("core")
    ve_labels = get_roi_labels("core")

    # Extended VEs (for comparison)
    ve_ext_pos = np.array([[x, y, z] for _, _, x, y, z in _SPEECH_ROIS_EXTENDED])
    ve_ext_labels = [label for label, _, _, _, _ in _SPEECH_ROIS_EXTENDED]

    fig = plt.figure(figsize=(20, 10))

    views = [
        ("Left lateral", -90, 0),
        ("Left lateral (tilted)", -75, 15),
        ("Ventral-lateral", -70, -20),
        ("Anterior", -180, 0),
    ]

    for idx, (title, azim, elev) in enumerate(views):
        ax = fig.add_subplot(1, 4, idx + 1, projection="3d")

        # Draw brain surface (subsample triangles for speed)
        ax.plot_trisurf(
            coords_surf[:, 0], coords_surf[:, 1], coords_surf[:, 2],
            triangles=faces_surf[::4],  # every 4th face for speed
            color="#e8d4c0", alpha=0.15, edgecolor="none",
            linewidth=0,
        )

        # Patient electrodes
        for pid in ALL_PATIENTS:
            if pid not in patient_mni:
                continue
            mni = patient_mni[pid]
            is_core = pid in CORE_PATIENTS
            ax.scatter(
                mni[:, 0], mni[:, 1], mni[:, 2],
                c=PATIENT_COLORS[pid],
                s=8 if is_core else 3,
                alpha=0.8 if is_core else 0.25,
                label=pid if idx == 0 else None,
                edgecolors="none", depthshade=True,
            )

        # Core VEs — large colored spheres
        for i, label in enumerate(ve_labels):
            cat = VE_CATEGORIES[label]
            color = VE_CATEGORY_COLORS[cat]
            ax.scatter(
                [ve_pos[i, 0]], [ve_pos[i, 1]], [ve_pos[i, 2]],
                c=color, s=150, marker="o", alpha=0.95,
                edgecolors="black", linewidths=0.8, depthshade=False,
                zorder=10,
            )
            # Label on select views
            if idx in (0, 1):
                ax.text(
                    ve_pos[i, 0] + 2, ve_pos[i, 1] + 2, ve_pos[i, 2] + 2,
                    label.replace("1/2/3", ""), fontsize=5.5,
                    color=color, fontweight="bold", alpha=0.9,
                )

        # Extended VEs — small hollow diamonds
        for i, label in enumerate(ve_ext_labels):
            ax.scatter(
                [ve_ext_pos[i, 0]], [ve_ext_pos[i, 1]], [ve_ext_pos[i, 2]],
                c="none", s=60, marker="D", alpha=0.6,
                edgecolors="gray", linewidths=1.0, depthshade=False,
            )
            if idx == 1:
                ax.text(
                    ve_ext_pos[i, 0] + 2, ve_ext_pos[i, 1], ve_ext_pos[i, 2],
                    label.replace("1/2/3", ""), fontsize=4.5,
                    color="gray", alpha=0.6,
                )

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=5)

        # Zoom to electrode region
        ax.set_xlim(-80, -25)
        ax.set_ylim(-45, 60)
        ax.set_zlim(-25, 65)

    # Legend
    from matplotlib.lines import Line2D
    handles = []
    for pid in CORE_PATIENTS:
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=PATIENT_COLORS[pid], markersize=7,
                              label=f"{pid} (core)"))
    for pid in EXTENDED_PATIENTS:
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=PATIENT_COLORS[pid], markersize=5,
                              label=pid, alpha=0.4))
    handles.append(Line2D([0], [0], color="w", label=""))
    for cat, color in VE_CATEGORY_COLORS.items():
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=color, markersize=9,
                              markeredgecolor="black", markeredgewidth=0.5,
                              label=f"VE: {cat}"))
    handles.append(Line2D([0], [0], marker="D", color="w",
                          markerfacecolor="none", markeredgecolor="gray",
                          markersize=7, label="VE: extended (dropped)"))

    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=7,
               frameon=True)
    fig.suptitle(
        "Electrode Arrays + Virtual Electrodes on fsaverage Pial Surface\n"
        "Solid circles = 10 core VEs  |  Diamonds = 6 extended VEs (dropped, >30mm from all)",
        fontsize=12, y=0.98,
    )
    plt.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved surface plot: {output_path}")


def plot_interactive_html(patient_mni: dict, output_path: Path):
    """Generate interactive HTML brain view using nilearn view_surf."""
    from nilearn import datasets, plotting, surface

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage")

    pial_mesh = surface.load_surf_mesh(fsaverage.pial_left)
    surf_coords = pial_mesh.coordinates
    n_vertices = len(surf_coords)

    # Stat map: 0 = background, 1-4 = core patients, 5 = VE
    stat_map = np.zeros(n_vertices, dtype=np.float64)

    # Paint surface near electrodes with patient-specific values
    for pid_idx, pid in enumerate(CORE_PATIENTS):
        if pid not in patient_mni:
            continue
        mni = patient_mni[pid]
        # Vectorized: distance from all vertices to all electrodes
        # Process in chunks to avoid memory blowup
        for elec in mni:
            dists = np.sqrt(((surf_coords - elec) ** 2).sum(axis=1))
            nearby = dists < 5  # 5mm radius
            stat_map[nearby] = pid_idx + 1

    # Paint VE locations
    ve_pos = get_virtual_electrode_positions("core")
    for ve in ve_pos:
        dists = np.sqrt(((surf_coords - ve) ** 2).sum(axis=1))
        nearby = dists < 4
        stat_map[nearby] = 6  # distinct from patients

    view = plotting.view_surf(
        fsaverage.infl_left,
        stat_map,
        cmap="tab10",
        symmetric_cmap=False,
        threshold=0.5,
        title="Electrode coverage on fsaverage (core patients + VEs)",
    )
    view.save_as_html(str(output_path))
    print(f"Saved interactive HTML: {output_path}")
    print(f"Open in browser: file://{output_path}")


def main():
    output_dir = project_root / "docs" / "figures"
    output_dir.mkdir(exist_ok=True)

    patient_mni = {}
    for pid in ALL_PATIENTS:
        try:
            patient_mni[pid] = load_patient_mni(pid)
        except Exception as e:
            print(f"WARN: {pid}: {e}")

    # Static matplotlib surface plot
    plot_surface_matplotlib(patient_mni, output_dir / "electrodes_brain_surface.png")

    # Interactive HTML
    try:
        plot_interactive_html(patient_mni, output_dir / "electrodes_brain_interactive.html")
    except Exception as e:
        print(f"Interactive HTML failed: {e}")


if __name__ == "__main__":
    main()
