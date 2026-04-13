#!/usr/bin/env python3
"""Plot electrode positions (ACPC→MNI) on brain surface with virtual electrodes.

Visual validation of the coordinate pipeline:
    ACPC RAS → talairach.xfm → MNI → plot on fsaverage surface

Shows:
- Each patient's electrodes in a unique color
- 10 core Brainnetome virtual electrodes as red stars
- Lateral and dorsal views
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from speech_decoding.data.atlas import (
    SPEECH_ROIS_CORE,
    get_roi_labels,
    get_virtual_electrode_positions,
)
from speech_decoding.data.coordinates import (
    apply_talairach_transform,
    build_electrode_coordinates,
    load_talairach_transform,
)

# ---------- configuration ----------
DATA_DIR = project_root / "data"
COORDS_DIR = DATA_DIR / "mni_coords"
CHANMAP_DIR = DATA_DIR / "channel_maps"
TRANSFORM_DIR = DATA_DIR / "transforms"

# Patient groups
CORE_PATIENTS = ["S14", "S26", "S33", "S62"]
EXTENDED_PATIENTS = ["S16", "S22", "S23", "S39", "S58"]
EXCLUDED_PATIENTS = ["S32", "S57"]

# 128-ch patients need chanMap
PATIENTS_128CH = {"S14", "S16", "S22", "S23", "S26"}

ALL_PATIENTS = CORE_PATIENTS + EXTENDED_PATIENTS + EXCLUDED_PATIENTS

# Colors per patient
PATIENT_COLORS = {
    # Core — bold colors
    "S14": "#1f77b4",  # blue
    "S26": "#2ca02c",  # green
    "S33": "#ff7f0e",  # orange
    "S62": "#9467bd",  # purple
    # Extended — muted
    "S16": "#8c564b",  # brown
    "S22": "#e377c2",  # pink
    "S23": "#7f7f7f",  # gray
    "S39": "#bcbd22",  # olive
    "S58": "#17becf",  # cyan
    # Excluded — dim
    "S32": "#d3d3d3",  # light gray
    "S57": "#c0c0c0",  # silver
}


def load_patient_mni(patient_id: str) -> np.ndarray:
    """Load ACPC coordinates, apply talairach.xfm, return MNI (N, 3)."""
    ras_path = COORDS_DIR / f"{patient_id}_RAS.txt"
    xfm_path = TRANSFORM_DIR / f"{patient_id}_talairach.xfm"
    chanmap_path = (
        CHANMAP_DIR / f"{patient_id}_channelMap.mat"
        if patient_id in PATIENTS_128CH
        else None
    )

    elec = build_electrode_coordinates(
        patient_id, ras_path, chanmap_path=chanmap_path
    )

    # Mirror right hemisphere to left
    if elec.hemisphere == "R":
        elec = elec.mirror_to_left()

    # Get coordinate array (all channels, some might be NaN)
    ch_names = list(elec.coords.keys())
    acpc = elec.to_array(ch_names)

    # Remove any NaN rows
    valid = ~np.isnan(acpc).any(axis=1)
    acpc = acpc[valid]

    # Apply talairach transform → MNI
    affine = load_talairach_transform(xfm_path)
    mni = apply_talairach_transform(acpc, affine)

    return mni


def plot_glass_brain(patient_mni: dict[str, np.ndarray], output_path: Path):
    """Plot all patients on nilearn glass brain with VE positions."""
    from nilearn import plotting

    # Combine all coordinates + marker sizes for nilearn
    all_coords = []
    all_colors = []
    all_sizes = []
    all_labels = []

    for pid in ALL_PATIENTS:
        if pid not in patient_mni:
            continue
        mni = patient_mni[pid]
        all_coords.append(mni)
        n = len(mni)
        all_colors.extend([PATIENT_COLORS[pid]] * n)
        sz = 8 if pid in CORE_PATIENTS else 5
        all_sizes.extend([sz] * n)
        all_labels.append(pid)

    all_coords_arr = np.vstack(all_coords)

    # Plot glass brain with markers
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Views: lateral left, dorsal, lateral right, frontal
    displays = ["l", "z", "r", "y"]
    titles = ["Left lateral", "Dorsal (top-down)", "Right lateral", "Anterior"]

    for ax, display, title in zip(axes.flat, displays, titles):
        disp = plotting.plot_glass_brain(
            None, display_mode=display, axes=ax, title=title
        )
        # Add electrode markers per patient
        for pid in ALL_PATIENTS:
            if pid not in patient_mni:
                continue
            mni = patient_mni[pid]
            sz = 8 if pid in CORE_PATIENTS else 4
            alpha = 0.9 if pid in CORE_PATIENTS else 0.4
            disp.add_markers(
                mni,
                marker_color=PATIENT_COLORS[pid],
                marker_size=sz,
                alpha=alpha,
            )

        # Add virtual electrodes as red stars
        ve_pos = get_virtual_electrode_positions("core")
        disp.add_markers(
            ve_pos,
            marker_color="red",
            marker_size=50,
            alpha=0.8,
        )

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = []
    for pid in CORE_PATIENTS:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PATIENT_COLORS[pid],
                   markersize=8, label=f"{pid} (core)")
        )
    for pid in EXTENDED_PATIENTS:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PATIENT_COLORS[pid],
                   markersize=6, label=f"{pid} (extended)", alpha=0.5)
        )
    for pid in EXCLUDED_PATIENTS:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PATIENT_COLORS[pid],
                   markersize=5, label=f"{pid} (excluded)", alpha=0.3)
        )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=10, label="VE (Brainnetome)")
    )
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9)

    fig.suptitle(
        "Electrode positions (ACPC→MNI via talairach.xfm)\n"
        "Red markers = 10 core Brainnetome virtual electrodes",
        fontsize=13, y=0.98,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved glass brain: {output_path}")
    return fig


def plot_scatter_3d(patient_mni: dict[str, np.ndarray], output_path: Path):
    """3D scatter plot with per-patient colors — simpler, always works."""
    fig = plt.figure(figsize=(16, 12))

    # 4 views
    views = [
        ("Left lateral", -90, 0),    # looking from left
        ("Dorsal", 0, 90),            # looking down
        ("Posterior", 180, 0),         # looking from behind
        ("Oblique", -60, 30),          # angled view
    ]

    for idx, (title, azim, elev) in enumerate(views):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")

        # Plot each patient
        for pid in ALL_PATIENTS:
            if pid not in patient_mni:
                continue
            mni = patient_mni[pid]
            is_core = pid in CORE_PATIENTS
            is_excluded = pid in EXCLUDED_PATIENTS
            sz = 12 if is_core else (3 if is_excluded else 6)
            alpha = 0.9 if is_core else (0.2 if is_excluded else 0.4)
            label = pid if idx == 0 else None  # legend only on first subplot
            ax.scatter(
                mni[:, 0], mni[:, 1], mni[:, 2],
                c=PATIENT_COLORS[pid], s=sz, alpha=alpha,
                label=label, edgecolors="none",
            )

        # Plot virtual electrodes
        ve_pos = get_virtual_electrode_positions("core")
        ve_labels = get_roi_labels("core")
        label_ve = "VE (Brainnetome)" if idx == 0 else None
        ax.scatter(
            ve_pos[:, 0], ve_pos[:, 1], ve_pos[:, 2],
            c="red", s=80, marker="*", alpha=0.9,
            label=label_ve, edgecolors="darkred", linewidths=0.5,
        )

        # Label VEs on oblique view
        if idx == 3:
            for i, lbl in enumerate(ve_labels):
                ax.text(
                    ve_pos[i, 0] + 1, ve_pos[i, 1] + 1, ve_pos[i, 2] + 1,
                    lbl, fontsize=6, color="red", alpha=0.8,
                )

        ax.set_xlabel("X (L→R)")
        ax.set_ylabel("Y (P→A)")
        ax.set_zlabel("Z (I→S)")
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)

    # Global legend
    if True:
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
                   markerscale=1.5)

    fig.suptitle(
        "Electrode positions in MNI space (ACPC → talairach.xfm → MNI)\n"
        "Red stars = 10 core Brainnetome virtual electrodes",
        fontsize=13, y=0.98,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved 3D scatter: {output_path}")
    return fig


def print_summary(patient_mni: dict[str, np.ndarray]):
    """Print per-patient MNI coordinate summary and VE coverage."""
    ve_pos = get_virtual_electrode_positions("core")
    ve_labels = get_roi_labels("core")

    print("\n" + "=" * 80)
    print("ELECTRODE COORDINATE SUMMARY (MNI)")
    print("=" * 80)

    for pid in ALL_PATIENTS:
        if pid not in patient_mni:
            print(f"\n{pid}: MISSING")
            continue

        mni = patient_mni[pid]
        centroid = mni.mean(axis=0)
        span = mni.max(axis=0) - mni.min(axis=0)
        group = (
            "CORE" if pid in CORE_PATIENTS
            else "extended" if pid in EXTENDED_PATIENTS
            else "EXCLUDED"
        )

        # VE distances
        dists = np.sqrt(((mni[:, None, :] - ve_pos[None, :, :]) ** 2).sum(axis=2))
        nn_per_ve = dists.min(axis=0)
        n_reachable = (nn_per_ve <= 25).sum()

        print(f"\n{pid} [{group}] — {len(mni)} electrodes")
        print(f"  Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")
        print(f"  Span:     ({span[0]:.1f}, {span[1]:.1f}, {span[2]:.1f}) mm")
        print(f"  X range:  [{mni[:, 0].min():.1f}, {mni[:, 0].max():.1f}]")
        print(f"  Y range:  [{mni[:, 1].min():.1f}, {mni[:, 1].max():.1f}]")
        print(f"  Z range:  [{mni[:, 2].min():.1f}, {mni[:, 2].max():.1f}]")
        print(f"  VEs reachable (<25mm): {n_reachable}/10")
        for i, (label, d) in enumerate(zip(ve_labels, nn_per_ve)):
            flag = "  <---" if d > 25 else ""
            print(f"    {label:15s}: {d:5.1f} mm{flag}")

    # Inter-patient distances
    print("\n" + "=" * 80)
    print("INTER-PATIENT CENTROID DISTANCES (mm)")
    print("=" * 80)
    pids = [p for p in ALL_PATIENTS if p in patient_mni]
    centroids = {p: patient_mni[p].mean(axis=0) for p in pids}
    print(f"{'':>6s}", end="")
    for p2 in pids:
        print(f" {p2:>6s}", end="")
    print()
    for p1 in pids:
        print(f"{p1:>6s}", end="")
        for p2 in pids:
            d = np.linalg.norm(centroids[p1] - centroids[p2])
            print(f" {d:6.1f}", end="")
        print()


def main():
    output_dir = project_root / "docs" / "figures"
    output_dir.mkdir(exist_ok=True)

    # Load all patient MNI coordinates
    patient_mni = {}
    for pid in ALL_PATIENTS:
        try:
            mni = load_patient_mni(pid)
            patient_mni[pid] = mni
            print(f"Loaded {pid}: {len(mni)} electrodes → MNI")
        except Exception as e:
            print(f"WARN: {pid}: {e}")

    # Print summary statistics
    print_summary(patient_mni)

    # Generate plots
    plot_scatter_3d(patient_mni, output_dir / "electrodes_mni_3d.png")

    try:
        plot_glass_brain(patient_mni, output_dir / "electrodes_glass_brain.png")
    except Exception as e:
        print(f"Glass brain failed (nilearn issue): {e}")
        print("3D scatter plot still saved.")


if __name__ == "__main__":
    main()
