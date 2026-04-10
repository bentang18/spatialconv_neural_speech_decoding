#!/usr/bin/env python3
"""Visualize proposed VE expansion: current 16 + 3 new motor/sensory VEs.

Shows glass brain views (lateral + dorsal) with:
- Patient electrodes (colored dots)
- Current 16 core VEs (filled circles, color = category)
- Proposed 3 new VEs (star markers)
- Labels for all VEs
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from speech_decoding.data.atlas import SPEECH_ROIS_CORE
from speech_decoding.data.coordinates import (
    apply_talairach_transform,
    build_electrode_coordinates,
    load_talairach_transform,
)

DATA_DIR = project_root / "data"
COORDS_DIR = DATA_DIR / "mni_coords"
CHANMAP_DIR = DATA_DIR / "channel_maps"
TRANSFORM_DIR = DATA_DIR / "transforms"

ALL_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S39", "S57", "S58", "S62"]
CORE_PATIENTS = ["S14", "S26", "S33", "S62"]
PATIENTS_128CH = {"S14", "S16", "S22", "S23", "S26"}

PATIENT_COLORS = {
    "S14": "#1f77b4", "S26": "#2ca02c", "S33": "#ff7f0e", "S62": "#9467bd",
    "S16": "#8c564b", "S22": "#e377c2", "S23": "#7f7f7f", "S39": "#bcbd22",
    "S58": "#17becf", "S32": "#d3d3d3", "S57": "#c0c0c0",
}

# Functional categories with colors
CATEGORY_COLORS = {
    "Motor": "#d62728",
    "Sensory": "#ff7f0e",
    "Broca": "#2ca02c",
    "Auditory": "#1f77b4",
    "Insula": "#9467bd",
    "Executive": "#8c564b",
    "NEW Motor": "#d62728",
    "NEW Sensory": "#ff7f0e",
}

# Current VE categories
VE_CATEGORIES = {
    "A6cvl": "Motor", "A4tl": "Motor", "A4hf": "Motor",
    "A1/2/3tonIa": "Sensory", "A1/2/3ulhf": "Sensory", "A2": "Sensory",
    "A44d": "Broca", "A45c": "Broca", "A44v": "Broca",
    "A45i": "Broca", "A45r": "Broca", "A44op": "Broca",
    "STGpp": "Auditory", "STGa": "Auditory",
    "INSa": "Insula",
    "MFG": "Executive",
}

# Proposed new VEs
PROPOSED_NEW_VES = [
    ("Jaw_M1", "M1 jaw", -50, -4, 24, "NEW Motor"),
    ("dLMC", "Dorsal larynx M1", -48, -10, 48, "NEW Motor"),
    ("Jaw_S1", "S1 jaw", -55, -12, 28, "NEW Sensory"),
]


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


def main():
    from nilearn import plotting

    output_dir = project_root / "docs" / "figures"
    output_dir.mkdir(exist_ok=True)

    # Load patient electrodes
    patient_mni = {}
    for pid in ALL_PATIENTS:
        try:
            patient_mni[pid] = load_patient_mni(pid)
        except Exception as e:
            print(f"WARN: {pid}: {e}")

    # Parse VE positions
    current_positions = []
    current_labels = []
    current_cats = []
    for label, name, x, y, z in SPEECH_ROIS_CORE:
        current_positions.append([x, y, z])
        current_labels.append(label)
        current_cats.append(VE_CATEGORIES[label])
    current_positions = np.array(current_positions)

    new_positions = []
    new_labels = []
    new_cats = []
    for label, name, x, y, z, cat in PROPOSED_NEW_VES:
        new_positions.append([x, y, z])
        new_labels.append(label)
        new_cats.append(cat)
    new_positions = np.array(new_positions)

    # ─── Figure 1: Glass brain with all VEs ───
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax, display, title in zip(
        axes, ["l", "z", "x"],
        ["Left lateral (Y vs Z)", "Dorsal (X vs Y)", "Posterior (X vs Z)"]
    ):
        disp = plotting.plot_glass_brain(
            None, display_mode=display, axes=ax, title=title
        )

        # Patient electrodes
        for pid in ALL_PATIENTS:
            if pid not in patient_mni:
                continue
            mni = patient_mni[pid]
            is_core = pid in CORE_PATIENTS
            disp.add_markers(
                mni,
                marker_color=PATIENT_COLORS[pid],
                marker_size=4 if is_core else 2,
                alpha=0.4 if is_core else 0.15,
            )

        # Current VEs (circles)
        for i, (label, cat) in enumerate(zip(current_labels, current_cats)):
            color = CATEGORY_COLORS[cat]
            disp.add_markers(
                current_positions[i:i + 1],
                marker_color=color,
                marker_size=100,
                alpha=0.9,
            )

        # New VEs (plotted as markers too, but we'll annotate differently)
        for i, (label, cat) in enumerate(zip(new_labels, new_cats)):
            base_cat = cat.replace("NEW ", "")
            color = CATEGORY_COLORS[base_cat]
            disp.add_markers(
                new_positions[i:i + 1],
                marker_color=color,
                marker_size=200,
                alpha=1.0,
            )

        # Annotate labels on inner axes
        inner_ax = disp.axes[display].ax

        # Label current VEs
        for i, label in enumerate(current_labels):
            x, y, z = current_positions[i]
            cat = current_cats[i]
            color = CATEGORY_COLORS[cat]
            short = label.replace("1/2/3", "").replace("STG", "STG")

            if display == "l":
                tx, ty = y, z
            elif display == "z":
                tx, ty = x, y
            else:  # "x" posterior
                tx, ty = x, z

            inner_ax.annotate(
                short, xy=(tx, ty), xytext=(tx + 3, ty + 2),
                fontsize=6, color=color, alpha=0.8,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.3, alpha=0.4),
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=color, alpha=0.5, lw=0.3),
            )

        # Label new VEs (bold, with star prefix)
        for i, label in enumerate(new_labels):
            x, y, z = new_positions[i]
            base_cat = new_cats[i].replace("NEW ", "")
            color = CATEGORY_COLORS[base_cat]

            if display == "l":
                tx, ty = y, z
            elif display == "z":
                tx, ty = x, y
            else:
                tx, ty = x, z

            inner_ax.annotate(
                f"* {label}", xy=(tx, ty), xytext=(tx + 4, ty - 4),
                fontsize=8, fontweight="bold", color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec=color, alpha=0.8, lw=1.0),
            )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CATEGORY_COLORS["Motor"],
               markersize=10, label="Motor VE (current)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CATEGORY_COLORS["Sensory"],
               markersize=10, label="Sensory VE (current)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CATEGORY_COLORS["Broca"],
               markersize=10, label="Broca VE (current)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CATEGORY_COLORS["Auditory"],
               markersize=10, label="Auditory VE (current)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CATEGORY_COLORS["Insula"],
               markersize=10, label="Insula VE (current)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CATEGORY_COLORS["Executive"],
               markersize=10, label="Executive VE (current)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markersize=14, markeredgecolor="black", label="NEW proposed VE"),
        Line2D([0], [0], color="w", label=""),  # spacer
    ]
    for pid in CORE_PATIENTS:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PATIENT_COLORS[pid],
                   markersize=6, label=f"{pid} (core)")
        )

    fig.legend(handles=legend_elements, loc="lower center", ncol=6, fontsize=8,
               frameon=True, fancybox=True)
    fig.suptitle(
        "VE Expansion: 16 Current + 3 Proposed (Jaw M1, dLMC, Jaw S1)\n"
        "Yellow-highlighted = NEW | Patient electrodes shown for spatial context",
        fontsize=13, y=0.99,
    )
    plt.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(output_dir / "ve_expansion_glass_brain.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 've_expansion_glass_brain.png'}")

    # ─── Figure 2: Motor strip zoom ───
    fig2, ax2 = plt.subplots(figsize=(12, 10))

    # Plot patient electrodes in motor strip region (filter to z=-10..60, y=-40..20)
    for pid in ALL_PATIENTS:
        if pid not in patient_mni:
            continue
        mni = patient_mni[pid]
        # Filter to roughly motor cortex region
        mask = (mni[:, 2] > -10) & (mni[:, 2] < 60) & (mni[:, 1] > -40) & (mni[:, 1] < 30)
        pts = mni[mask]
        if len(pts) == 0:
            continue
        is_core = pid in CORE_PATIENTS
        ax2.scatter(
            pts[:, 1], pts[:, 2],  # y vs z (lateral view projection)
            c=PATIENT_COLORS[pid], s=8 if is_core else 3,
            alpha=0.5 if is_core else 0.2, zorder=1,
            label=pid if is_core else None,
        )

    # Plot current motor + sensory VEs
    motor_sensory_labels = ["A6cvl", "A4hf", "A4tl",
                            "A1/2/3tonIa", "A1/2/3ulhf", "A2"]
    for i, (label, cat) in enumerate(zip(current_labels, current_cats)):
        pos = current_positions[i]
        color = CATEGORY_COLORS[cat]
        if label in motor_sensory_labels:
            ax2.scatter(pos[1], pos[2], c=color, s=300, zorder=3,
                        edgecolors="black", linewidths=1.5, marker="o")
            ax2.annotate(
                label, (pos[1], pos[2]),
                xytext=(pos[1] + 2, pos[2] + 2), fontsize=10, fontweight="bold",
                color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=1),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1),
            )
        else:
            # Show non-motor/sensory VEs as small markers for context
            ax2.scatter(pos[1], pos[2], c=color, s=60, zorder=2,
                        edgecolors="gray", linewidths=0.5, marker="o", alpha=0.4)
            ax2.annotate(label.replace("1/2/3", ""), (pos[1], pos[2]),
                         fontsize=6, color="gray", alpha=0.5,
                         xytext=(pos[1] + 1, pos[2] - 2))

    # Plot NEW VEs with star markers
    for i, (label, cat) in enumerate(zip(new_labels, new_cats)):
        pos = new_positions[i]
        base_cat = cat.replace("NEW ", "")
        color = CATEGORY_COLORS[base_cat]
        ax2.scatter(pos[1], pos[2], c="gold", s=500, zorder=4,
                    edgecolors=color, linewidths=2.5, marker="*")
        ax2.annotate(
            f"NEW: {label}", (pos[1], pos[2]),
            xytext=(pos[1] + 3, pos[2] - 4), fontsize=11, fontweight="bold",
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec=color, lw=1.5),
        )

    # Draw the somatotopic gradient arrow
    ax2.annotate(
        "", xy=(-6, 48), xytext=(-6, 9),
        arrowprops=dict(arrowstyle="<->", color="gray", lw=2, linestyle="--"),
    )
    ax2.text(-8, 28, "D-V somatotopic\ngradient\n~30mm", fontsize=9,
             color="gray", ha="right", va="center", style="italic",
             rotation=90)

    # Draw inter-VE gap annotations
    motor_ve_z = {
        "A4tl\n(tongue)": 9, "Jaw_M1\n(NEW)": 24,
        "A4hf\n(lips)": 40, "dLMC\n(NEW)": 48
    }
    prev_z = None
    prev_name = None
    x_line = -20
    for name, z in sorted(motor_ve_z.items(), key=lambda x: x[1]):
        if prev_z is not None:
            gap = z - prev_z
            mid = (z + prev_z) / 2
            ax2.annotate(
                "", xy=(x_line, z), xytext=(x_line, prev_z),
                arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.5),
            )
            ax2.text(x_line - 1, mid, f"{gap}mm", fontsize=8, color="#d62728",
                     ha="right", va="center", fontweight="bold")
        ax2.plot(x_line, z, "o", color="#d62728", markersize=4)
        prev_z = z
        prev_name = name

    ax2.set_xlabel("Y (anterior → posterior, mm)", fontsize=11)
    ax2.set_ylabel("Z (ventral → dorsal, mm)", fontsize=11)
    ax2.set_title(
        "Motor Strip Zoom: Sagittal Projection (Y vs Z)\n"
        "Current motor gap A4tl↔A4hf = 32mm → broken into 15mm, 16mm, 8mm segments",
        fontsize=12,
    )
    ax2.set_xlim(-25, 42)
    ax2.set_ylim(-15, 60)
    ax2.grid(True, alpha=0.2)
    ax2.axhline(0, color="gray", lw=0.5, alpha=0.3)
    ax2.axvline(0, color="gray", lw=0.5, alpha=0.3)

    # Legend
    legend2 = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markersize=12, markeredgecolor="black", label="Current Motor VE"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e",
               markersize=12, markeredgecolor="black", label="Current Sensory VE"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markersize=16, markeredgecolor="black", label="NEW Proposed VE"),
    ]
    for pid in CORE_PATIENTS:
        legend2.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PATIENT_COLORS[pid],
                   markersize=6, label=f"{pid}")
        )
    ax2.legend(handles=legend2, loc="upper right", fontsize=9)

    plt.tight_layout()
    fig2.savefig(output_dir / "ve_expansion_motor_zoom.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 've_expansion_motor_zoom.png'}")

    # ─── Figure 3: Coverage heatmap (19 VEs) ───
    all_ve_labels = current_labels + new_labels
    all_ve_positions = np.vstack([current_positions, new_positions])
    all_ve_cats = current_cats + [c.replace("NEW ", "") for c in new_cats]
    n_ves = len(all_ve_labels)

    patient_order = ALL_PATIENTS
    n_pts = len(patient_order)
    dist_matrix = np.zeros((n_pts, n_ves))

    for i, pid in enumerate(patient_order):
        if pid not in patient_mni:
            dist_matrix[i] = 999
            continue
        mni = patient_mni[pid]
        diffs = mni[:, None, :] - all_ve_positions[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        dist_matrix[i] = dists.min(axis=0)

    fig3, ax3 = plt.subplots(figsize=(20, 8))

    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c", "#7b0023"]
    cmap = LinearSegmentedColormap.from_list("coverage", colors_list, N=256)

    im = ax3.imshow(dist_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=60)

    for i in range(n_pts):
        for j in range(n_ves):
            d = dist_matrix[i, j]
            if d > 900:
                ax3.text(j, i, "?", ha="center", va="center", fontsize=7, color="white")
            else:
                color = "white" if d > 35 else "black"
                weight = "bold" if d <= 25 else "normal"
                ax3.text(j, i, f"{d:.0f}", ha="center", va="center",
                         fontsize=7, color=color, fontweight=weight)
                if d > 25:
                    ax3.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="white", linewidth=1.0, linestyle="--"
                    ))

    # Mark new VE columns
    for j in range(16, n_ves):
        ax3.add_patch(plt.Rectangle(
            (j - 0.5, -0.5), 1, n_pts,
            fill=True, facecolor="lightyellow", alpha=0.15, zorder=0
        ))

    pts_per_ve = (dist_matrix <= 25).sum(axis=0).astype(int)
    ves_per_pt = (dist_matrix <= 25).sum(axis=1).astype(int)

    xlabels = []
    for j, label in enumerate(all_ve_labels):
        cat = all_ve_cats[j]
        prefix = "* " if j >= 16 else ""
        xlabels.append(f"{prefix}{label}\n({cat})\n[{pts_per_ve[j]}pts]")

    ax3.set_xticks(range(n_ves))
    ax3.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax3.set_yticks(range(n_pts))

    ylabels = []
    for i, pid in enumerate(patient_order):
        group = "CORE" if pid in CORE_PATIENTS else "ext"
        ylabels.append(f"{pid} [{group}] ({ves_per_pt[i]}/{n_ves} VEs)")
    ax3.set_yticklabels(ylabels, fontsize=8)

    cbar = plt.colorbar(im, ax=ax3, shrink=0.8, pad=0.01)
    cbar.set_label("Nearest electrode distance (mm)", fontsize=9)
    cbar.ax.axhline(25, color="white", linewidth=2, linestyle="--")

    ax3.set_title(
        "VE Coverage Heatmap — 19 VEs (16 current + 3 NEW)\n"
        "Bold = reachable (<25mm) | Dashed = unreachable | Yellow columns = NEW VEs",
        fontsize=11,
    )
    plt.tight_layout()
    fig3.savefig(output_dir / "ve_expansion_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 've_expansion_heatmap.png'}")


if __name__ == "__main__":
    main()
