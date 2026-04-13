#!/usr/bin/env python3
"""Plot virtual electrode positions with labels + per-VE coverage heatmap.

Two figures:
1. Glass brain with labeled VE positions and patient electrodes
2. Heatmap: patients × VEs showing nearest-electrode distance
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from speech_decoding.data.atlas import (
    SPEECH_ROIS_CORE,
    get_roi_labels,
    get_roi_names,
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

# VE category colors for the glass brain
VE_CATEGORY_COLORS = {
    "Motor": "#d62728",      # red
    "Sensory": "#ff7f0e",    # orange
    "Broca": "#2ca02c",      # green
    "Premotor": "#9467bd",   # purple
}

VE_CATEGORIES = {
    "A4hf": "Motor", "A4tl": "Motor",
    "A6cdl": "Premotor", "A6cvl": "Premotor",
    "A1/2/3ulhf": "Sensory", "A1/2/3tonIa": "Sensory", "A2": "Sensory",
    "A44d": "Broca", "A44v": "Broca", "A45c": "Broca",
}


def load_patient_mni(patient_id: str) -> np.ndarray:
    """Load ACPC coordinates, apply talairach.xfm, return MNI (N, 3)."""
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


def plot_ve_glass_brain(patient_mni: dict, output_path: Path):
    """Glass brain with color-coded, labeled VEs and patient electrodes."""
    from nilearn import plotting

    ve_pos = get_virtual_electrode_positions("core")
    ve_labels = get_roi_labels("core")
    ve_names = get_roi_names("core")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, display, title in zip(axes, ["l", "z"], ["Left lateral", "Dorsal"]):
        disp = plotting.plot_glass_brain(
            None, display_mode=display, axes=ax, title=title
        )

        # Patient electrodes (subdued)
        for pid in ALL_PATIENTS:
            if pid not in patient_mni:
                continue
            mni = patient_mni[pid]
            is_core = pid in CORE_PATIENTS
            disp.add_markers(
                mni,
                marker_color=PATIENT_COLORS[pid],
                marker_size=6 if is_core else 3,
                alpha=0.6 if is_core else 0.2,
            )

        # VEs: large colored markers by category
        for i, (label, name) in enumerate(zip(ve_labels, ve_names)):
            cat = VE_CATEGORIES[label]
            color = VE_CATEGORY_COLORS[cat]
            disp.add_markers(
                ve_pos[i:i+1],
                marker_color=color,
                marker_size=120,
                alpha=0.95,
            )

        # Annotate VE labels — nilearn glass brain axes have specific projections
        # For 'l' (lateral): axes show (y, z) — sagittal view
        # For 'z' (dorsal): axes show (x, y) — axial view
        inner_axes = disp.axes[display].ax
        for i, label in enumerate(ve_labels):
            x, y, z = ve_pos[i]
            cat = VE_CATEGORIES[label]
            color = VE_CATEGORY_COLORS[cat]

            # Short label for display
            short = label.replace("1/2/3", "")

            if display == "l":
                # Lateral: projected to (y, z) plane
                tx, ty = y, z
                # Offset to avoid overlap
                offsets = {
                    "A4hf": (3, 3), "A4tl": (3, -5),
                    "A6cdl": (-18, 3), "A6cvl": (3, 3),
                    "Aulhf": (3, 3), "AtonIa": (3, -5),
                    "A2": (-10, 3),
                    "A44d": (3, 3), "A44v": (3, -5), "A45c": (3, 3),
                }
                dx, dy = offsets.get(short, (3, 2))
            else:
                # Dorsal: projected to (x, y) plane
                tx, ty = x, y
                dx, dy = 2, 2

            inner_axes.annotate(
                short,
                xy=(tx, ty),
                xytext=(tx + dx, ty + dy),
                fontsize=7, fontweight="bold", color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.5, alpha=0.6),
                alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.7, lw=0.5),
            )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for cat, color in VE_CATEGORY_COLORS.items():
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=10, label=f"VE: {cat}")
        )
    legend_elements.append(Line2D([0], [0], color="w", label=""))  # spacer
    for pid in CORE_PATIENTS:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PATIENT_COLORS[pid],
                   markersize=7, label=f"{pid} (core)")
        )
    for pid in EXTENDED_PATIENTS:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PATIENT_COLORS[pid],
                   markersize=5, label=f"{pid}", alpha=0.5)
        )

    fig.legend(handles=legend_elements, loc="lower center", ncol=6, fontsize=8,
               frameon=True, fancybox=True)
    fig.suptitle(
        "10 Core Brainnetome Virtual Electrodes on Glass Brain\n"
        "Color = functional category  |  Patient electrodes shown for spatial context",
        fontsize=12, y=0.98,
    )
    plt.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved annotated glass brain: {output_path}")


def plot_coverage_heatmap(patient_mni: dict, output_path: Path):
    """Heatmap: patients × VEs showing nearest-electrode distance."""
    ve_pos = get_virtual_electrode_positions("core")
    ve_labels = get_roi_labels("core")
    ve_names = get_roi_names("core")

    # Order: core patients first, then extended, then excluded
    patient_order = [p for p in ALL_PATIENTS if p in patient_mni]

    # Build distance matrix (patients × VEs)
    n_pts = len(patient_order)
    n_ves = len(ve_labels)
    dist_matrix = np.zeros((n_pts, n_ves))

    for i, pid in enumerate(patient_order):
        mni = patient_mni[pid]
        # (N_elec, 1, 3) - (1, N_ve, 3) → (N_elec, N_ve)
        diffs = mni[:, None, :] - ve_pos[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        dist_matrix[i] = dists.min(axis=0)  # nearest electrode per VE

    # Create figure with two subplots: heatmap + binary reachability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # --- Left: Distance heatmap ---
    # Custom colormap: green (close) → yellow → red (far) → dark red (unreachable)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c", "#7b0023"]
    cmap = LinearSegmentedColormap.from_list("coverage", colors_list, N=256)

    im = ax1.imshow(dist_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=60)

    # Annotate cells with distance values
    for i in range(n_pts):
        for j in range(n_ves):
            d = dist_matrix[i, j]
            color = "white" if d > 35 else "black"
            weight = "bold" if d <= 25 else "normal"
            ax1.text(j, i, f"{d:.0f}", ha="center", va="center",
                     fontsize=8, color=color, fontweight=weight)

    # Add 25mm threshold line via contour-like visual
    for i in range(n_pts):
        for j in range(n_ves):
            if dist_matrix[i, j] > 25:
                ax1.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="white", linewidth=1.5, linestyle="--"
                ))

    ax1.set_xticks(range(n_ves))
    ax1.set_xticklabels(
        [f"{l}\n({VE_CATEGORIES[l]})" for l in ve_labels],
        fontsize=7, rotation=45, ha="right",
    )
    ax1.set_yticks(range(n_pts))
    ylabels = []
    for pid in patient_order:
        group = "CORE" if pid in CORE_PATIENTS else ("ext" if pid in EXTENDED_PATIENTS else "excl")
        ylabels.append(f"{pid} [{group}]")
    ax1.set_yticklabels(ylabels, fontsize=9)

    # Horizontal lines separating patient groups
    n_core = len([p for p in patient_order if p in CORE_PATIENTS])
    n_ext = len([p for p in patient_order if p in EXTENDED_PATIENTS])
    ax1.axhline(n_core - 0.5, color="black", linewidth=2)
    ax1.axhline(n_core + n_ext - 0.5, color="black", linewidth=1, linestyle="--")

    cbar = plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
    cbar.set_label("Nearest electrode distance (mm)", fontsize=9)
    # Add threshold marker on colorbar
    cbar.ax.axhline(25, color="white", linewidth=2, linestyle="--")
    cbar.ax.text(1.5, 25, "25mm\nthreshold", fontsize=7, va="center",
                 transform=cbar.ax.get_yaxis_transform())

    ax1.set_title("Nearest Electrode → VE Distance (mm)\n"
                  "Bold = reachable (<25mm)  |  Dashed border = unreachable",
                  fontsize=10)

    # --- Right: Binary coverage + summary stats ---
    reachable = (dist_matrix <= 25).astype(float)

    # Count reachable per patient and per VE
    pts_per_ve = reachable.sum(axis=0).astype(int)   # how many patients reach each VE
    ves_per_pt = reachable.sum(axis=1).astype(int)    # how many VEs each patient reaches

    im2 = ax2.imshow(reachable, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    for i in range(n_pts):
        for j in range(n_ves):
            symbol = "●" if reachable[i, j] else "✗"
            color = "#1a6b1a" if reachable[i, j] else "#8b0000"
            ax2.text(j, i, symbol, ha="center", va="center",
                     fontsize=12, color=color, fontweight="bold")

    ax2.set_xticks(range(n_ves))
    ax2.set_xticklabels(ve_labels, fontsize=7, rotation=45, ha="right")
    ax2.set_yticks(range(n_pts))
    ax2.set_yticklabels([f"{p} ({ves_per_pt[i]}/10)" for i, p in enumerate(patient_order)],
                         fontsize=9)

    # Top annotation: patients per VE
    ax2_top = ax2.secondary_xaxis("top")
    ax2_top.set_xticks(range(n_ves))
    ax2_top.set_xticklabels([f"{n}" for n in pts_per_ve], fontsize=9, fontweight="bold")
    ax2_top.set_xlabel("# patients reaching VE", fontsize=8)

    ax2.axhline(n_core - 0.5, color="black", linewidth=2)
    ax2.axhline(n_core + n_ext - 0.5, color="black", linewidth=1, linestyle="--")

    ax2.set_title("Reachable (<25mm)\n● = yes  |  ✗ = no", fontsize=10)

    # --- Summary text below ---
    # Core-only coverage
    core_mask = np.array([p in CORE_PATIENTS for p in patient_order])
    core_reachable = dist_matrix[core_mask] <= 25
    core_any = core_reachable.any(axis=0)

    fig.text(0.5, 0.01,
             f"Core patients (4): {core_any.sum()}/10 VEs covered by at least one patient  |  "
             f"All patients (11): {(reachable.any(axis=0)).sum()}/10 VEs covered  |  "
             f"Best patient: {patient_order[ves_per_pt.argmax()]} ({ves_per_pt.max()}/10)",
             ha="center", fontsize=9, style="italic",
             bbox=dict(boxstyle="round", fc="#f0f0f0", ec="#cccccc"))

    fig.suptitle(
        "Virtual Electrode Coverage by Patient\n"
        "Which VEs can each patient's array reach?",
        fontsize=13, y=1.0,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved coverage heatmap: {output_path}")


def main():
    output_dir = project_root / "docs" / "figures"
    output_dir.mkdir(exist_ok=True)

    patient_mni = {}
    for pid in ALL_PATIENTS:
        try:
            patient_mni[pid] = load_patient_mni(pid)
        except Exception as e:
            print(f"WARN: {pid}: {e}")

    plot_ve_glass_brain(patient_mni, output_dir / "ve_glass_brain_labeled.png")
    plot_coverage_heatmap(patient_mni, output_dir / "ve_coverage_heatmap.png")


if __name__ == "__main__":
    main()
