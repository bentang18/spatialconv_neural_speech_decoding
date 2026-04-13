#!/usr/bin/env python3
"""Render Brainnetome parcellations on fsaverage.

Outputs:
- docs/figures/brainnetome_parcellations_surface.png
- docs/figures/brainnetome_parcellations_interactive.html

The preferred source is the official Brainnetome probabilistic atlas
`BNA_PM_4D.nii.gz`. For local development, `--allow-mpm-proxy` falls back to
Gaussian-smoothed masks derived from the MPM atlas.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from nilearn import datasets, image, surface
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from speech_decoding.data.atlas import (  # noqa: E402
    OFFICIAL_CANDIDATE_ROIS,
    ROI_BRAINNETOME_INDEX,
    ROI_CATEGORIES,
    get_brainnetome_roi_index,
    get_roi_categories,
    get_roi_labels,
)
from speech_decoding.data.coordinates import (  # noqa: E402
    apply_talairach_transform,
    build_electrode_coordinates,
    load_talairach_transform,
)

DATA_DIR = project_root / "data"
COORDS_DIR = DATA_DIR / "mni_coords"
CHANMAP_DIR = DATA_DIR / "channel_maps"
TRANSFORM_DIR = DATA_DIR / "transforms"

PATIENTS_128CH = {"S14", "S16", "S22", "S23", "S26"}
ALL_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S39", "S57", "S58", "S62"]

ROI_COLORS = {
    "A6cvl": "#b9352f",
    "A4tl": "#e17c2c",
    "A4hf": "#ef625d",
    "A1/2/3tonIa": "#f1a93b",
    "A1/2/3ulhf": "#f3ca52",
    "A2": "#90c657",
    "A44d": "#3a7bbf",
    "A45c": "#5d9ddb",
    "A44v": "#3756a8",
    "A45i": "#6b66b9",
    "A45r": "#8a6bbf",
    "A44op": "#4ca2a4",
    "STGpp": "#2a9d8f",
    "STGa": "#58c5c8",
    "INSa": "#a05dbd",
    "MFG": "#7d6a58",
    "A6cdl": "#6f3cba",
}

CATEGORY_COLORS = {
    "Motor": "#d1493f",
    "Sensory": "#e9b949",
    "Broca": "#5079d8",
    "Auditory": "#2ca59b",
    "Insula": "#9d5fc5",
    "Executive": "#7d6a58",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pm-atlas",
        type=Path,
        default=Path.home() / "nilearn_data" / "BNA_PM_4D.nii.gz",
        help="Official Brainnetome probabilistic atlas.",
    )
    parser.add_argument(
        "--mpm-atlas",
        type=Path,
        default=Path.home() / "nilearn_data" / "bnatlas.nii.gz",
        help="Brainnetome MPM atlas used only for explicit proxy fallback.",
    )
    parser.add_argument(
        "--allow-mpm-proxy",
        action="store_true",
        help="Fall back to smoothed MPM masks if the PM atlas is missing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "docs" / "figures",
        help="Figure output directory.",
    )
    return parser.parse_args()


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
    affine = load_talairach_transform(xfm_path)
    return apply_talairach_transform(acpc[valid], affine)


def load_roi_volumes(
    roi_labels: list[str],
    pm_path: Path,
    mpm_path: Path,
    allow_mpm_proxy: bool,
) -> tuple[dict[str, nib.spatialimages.SpatialImage], str]:
    if pm_path.exists():
        pm_img = nib.load(str(pm_path))
        if len(pm_img.shape) != 4:
            raise ValueError(f"Expected 4D PM atlas at {pm_path}, got {pm_img.shape}.")
        roi_imgs = {}
        for label in roi_labels:
            roi_idx = get_brainnetome_roi_index(label)
            roi_imgs[label] = image.index_img(pm_img, roi_idx - 1)
        return roi_imgs, "pm"

    if not allow_mpm_proxy:
        raise FileNotFoundError(
            f"Missing {pm_path}. Download the official Brainnetome PM atlas or rerun "
            "with --allow-mpm-proxy for a local proxy render."
        )

    mpm_img = nib.load(str(mpm_path))
    mpm_data = np.nan_to_num(np.asarray(mpm_img.dataobj), nan=0.0)
    roi_imgs = {}
    for label in roi_labels:
        roi_idx = get_brainnetome_roi_index(label)
        mask = (mpm_data == roi_idx).astype(np.float32)
        smooth = gaussian_filter(mask, sigma=5.0)
        if smooth.max() > 0:
            smooth /= smooth.max()
        roi_imgs[label] = nib.Nifti1Image(smooth, mpm_img.affine)
    return roi_imgs, "mpm-proxy"


def project_probabilities(
    roi_imgs: dict[str, nib.spatialimages.SpatialImage],
    mesh_for_sampling: str,
) -> dict[str, np.ndarray]:
    surf_probs: dict[str, np.ndarray] = {}
    for label, img in roi_imgs.items():
        prob = surface.vol_to_surf(
            img,
            mesh_for_sampling,
            interpolation="linear",
            radius=3.0,
            kind="ball",
            n_samples=20,
        )
        prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
        if prob.max() > 1.0:
            scale = 100.0 if prob.max() <= 100.0 else prob.max()
            prob = prob / scale
        surf_probs[label] = np.clip(prob, 0.0, 1.0)
    return surf_probs


def snap_points_to_surface(
    points: np.ndarray,
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    tree: cKDTree,
) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3))
    _, indices = tree.query(points, k=1)
    return target_coords[indices]


def compute_display_colors(
    surf_probs: dict[str, np.ndarray],
    roi_labels: list[str],
    base_gray: tuple[float, float, float] = (0.86, 0.86, 0.86),
) -> np.ndarray:
    stacked = np.stack([surf_probs[label] for label in roi_labels], axis=0).astype(np.float64)
    n_vertices = stacked.shape[1]
    boosted = np.power(np.clip(stacked, 0.0, 1.0), 0.55)
    dominant_idx = boosted.argmax(axis=0)
    dominant_weight = boosted.max(axis=0)

    dominant_rgb = np.tile(np.asarray(base_gray), (stacked.shape[1], 1))
    palette = np.asarray([to_rgb(ROI_COLORS[label]) for label in roi_labels], dtype=np.float64)
    active = dominant_weight > 0
    dominant_rgb[active] = palette[dominant_idx[active]]

    if active.any():
        alpha = dominant_weight / np.percentile(dominant_weight[active], 90)
    else:
        alpha = dominant_weight
    alpha = np.clip(alpha, 0.0, 1.0) * 0.98

    rgb = np.asarray(base_gray)[None, :] * (1.0 - alpha[:, None]) + dominant_rgb * alpha[:, None]
    return np.column_stack([rgb, np.ones(n_vertices)])


def face_colors_from_vertex_rgba(vertex_rgba: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return vertex_rgba[faces].mean(axis=1)


def render_static(
    output_path: Path,
    infl_coords: np.ndarray,
    infl_faces: np.ndarray,
    roi_labels: list[str],
    surf_probs: dict[str, np.ndarray],
    snapped_electrodes: np.ndarray,
    source_mode: str,
) -> None:
    fig = plt.figure(figsize=(10.5, 6.0))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    legend_ax = fig.add_subplot(1, 2, 2)

    vertex_rgba = compute_display_colors(surf_probs, roi_labels)
    face_rgba = face_colors_from_vertex_rgba(vertex_rgba, infl_faces)

    trisurf = ax.plot_trisurf(
        infl_coords[:, 0],
        infl_coords[:, 1],
        infl_coords[:, 2],
        triangles=infl_faces,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )
    trisurf.set_facecolors(face_rgba)
    trisurf.set_edgecolors("none")

    if len(snapped_electrodes) > 0:
        ax.scatter(
            snapped_electrodes[:, 0],
            snapped_electrodes[:, 1],
            snapped_electrodes[:, 2],
            s=2.8,
            c="#444444",
            alpha=0.14,
            depthshade=False,
        )

    ax.view_init(elev=14, azim=-92)
    mins = infl_coords.min(axis=0)
    maxs = infl_coords.max(axis=0)
    pad = np.array([4.0, 3.0, 3.0])
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
    ax.set_axis_off()
    ax.set_box_aspect((1.55, 1.05, 0.85))

    legend_ax.axis("off")
    legend_handles = [Patch(facecolor=ROI_COLORS[label], edgecolor="none", label=label) for label in roi_labels]
    legend = legend_ax.legend(
        handles=legend_handles,
        loc="center left",
        frameon=False,
        fontsize=9,
        ncol=2,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    legend_ax.add_artist(legend)
    legend_ax.text(0.0, 0.12, "Context", fontsize=10, fontweight="bold", transform=legend_ax.transAxes)
    legend_ax.text(
        0.0,
        0.05,
        "Gray points = pooled patient electrodes",
        fontsize=8,
        color="#444444",
        transform=legend_ax.transAxes,
    )
    fig.suptitle("Brainnetome Speech Parcellations on fsaverage", fontsize=14, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_roi_trace(
    label: str,
    probs: np.ndarray,
    coords: np.ndarray,
    visible: bool = True,
    max_points: int = 9000,
    threshold: float = 0.12,
) -> go.Scatter3d:
    idx = np.flatnonzero(probs > threshold)
    if len(idx) > max_points:
        top = np.argsort(probs[idx])[-max_points:]
        idx = idx[top]
    sel_probs = probs[idx]
    color = np.asarray(to_rgb(ROI_COLORS[label]))
    colors = [
        f"rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{0.28 + 0.62*float(p):.3f})"
        for p in sel_probs
    ]
    category = ROI_CATEGORIES[label]
    return go.Scatter3d(
        x=coords[idx, 0],
        y=coords[idx, 1],
        z=coords[idx, 2],
        mode="markers",
        marker=dict(size=2.6, color=colors),
        name=f"{label} ({category})",
        customdata=sel_probs[:, None],
        hovertemplate=f"{label}<br>category={category}<br>p=%{{customdata[0]:.2f}}<extra></extra>",
        visible=visible,
        showlegend=True,
    )


def render_interactive(
    output_path: Path,
    infl_coords: np.ndarray,
    infl_faces: np.ndarray,
    roi_labels: list[str],
    surf_probs: dict[str, np.ndarray],
    snapped_electrodes: np.ndarray,
    candidate_probs: dict[str, np.ndarray],
    candidate_points: dict[str, np.ndarray],
    source_mode: str,
) -> None:
    mesh_trace = go.Mesh3d(
        x=infl_coords[:, 0],
        y=infl_coords[:, 1],
        z=infl_coords[:, 2],
        i=infl_faces[:, 0],
        j=infl_faces[:, 1],
        k=infl_faces[:, 2],
        color="rgba(165,175,190,0.18)",
        opacity=0.16,
        name="fsaverage surface",
        hoverinfo="skip",
        showlegend=False,
        lighting=dict(ambient=0.55, diffuse=0.55, specular=0.12, roughness=0.95),
    )

    traces: list[go.BaseTraceType] = [mesh_trace]

    electrode_trace = go.Scatter3d(
        x=snapped_electrodes[:, 0],
        y=snapped_electrodes[:, 1],
        z=snapped_electrodes[:, 2],
        mode="markers",
        marker=dict(
            size=2.4,
            color="rgba(245,248,252,0.72)",
            line=dict(color="rgba(8,10,14,0.95)", width=1.0),
        ),
        name="Pooled electrodes",
        hoverinfo="skip",
        visible=False,
        showlegend=True,
    )
    traces.append(electrode_trace)

    for label in roi_labels:
        traces.append(build_roi_trace(label, surf_probs[label], infl_coords, visible=True))

    candidate_start = len(traces)
    for label in OFFICIAL_CANDIDATE_ROIS:
        if label not in candidate_probs:
            continue
        traces.append(build_roi_trace(label, candidate_probs[label], infl_coords, visible=False, threshold=0.04))
        point = candidate_points[label]
        traces.append(
            go.Scatter3d(
                x=[point[0]],
                y=[point[1]],
                z=[point[2]],
                mode="markers+text",
                marker=dict(size=5.0, color=ROI_COLORS[label], line=dict(color="black", width=0.5)),
                text=[label],
                textposition="top center",
                name=f"{label} centroid",
                hovertemplate=f"{label}<br>official candidate<extra></extra>",
                visible=False,
                showlegend=False,
            )
        )

    fig = go.Figure(data=traces)

    n_traces = len(traces)
    base_visible = [False] * n_traces
    base_visible[0] = True
    for idx in range(2, candidate_start):
        base_visible[idx] = True

    with_electrodes = base_visible.copy()
    with_electrodes[1] = True

    with_candidate = base_visible.copy()
    for idx in range(candidate_start, n_traces):
        with_candidate[idx] = True

    with_all = with_electrodes.copy()
    for idx in range(candidate_start, n_traces):
        with_all[idx] = True

    title = "Brainnetome Speech Parcellations (pm)"
    if source_mode != "pm":
        title = "Brainnetome Speech Parcellations (proxy)"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=-1.75, y=-0.12, z=0.42)),
            dragmode="orbit",
            bgcolor="rgb(12,14,18)",
        ),
        paper_bgcolor="rgb(12,14,18)",
        plot_bgcolor="rgb(12,14,18)",
        font=dict(color="rgb(232,236,242)"),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(18,22,30,0.82)",
            bordercolor="rgba(255,255,255,0.10)",
            borderwidth=1,
            font=dict(size=11),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.01,
                y=1.08,
                buttons=[
                    dict(label="Active 16", method="update", args=[{"visible": base_visible}]),
                    dict(label="+ Electrodes", method="update", args=[{"visible": with_electrodes}]),
                    dict(label="+ A6cdl", method="update", args=[{"visible": with_candidate}]),
                    dict(label="All layers", method="update", args=[{"visible": with_all}]),
                ],
            )
        ],
        annotations=[
            dict(
                x=0.01,
                y=1.015,
                xref="paper",
                yref="paper",
                text="Drag to rotate. Scroll/right-drag to zoom. Shift-drag to pan. Buttons toggle layers.",
                showarrow=False,
                font=dict(size=11, color="rgb(200,205,214)"),
            )
        ],
    )

    fig.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True,
        },
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roi_labels = get_roi_labels("core")
    patient_mni = {pid: load_patient_mni(pid) for pid in ALL_PATIENTS}

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")
    infl_mesh = surface.load_surf_mesh(fsaverage.infl_left)
    pial_mesh = surface.load_surf_mesh(fsaverage.pial_left)
    infl_coords = infl_mesh.coordinates
    infl_faces = infl_mesh.faces
    pial_coords = pial_mesh.coordinates
    pial_faces = pial_mesh.faces
    pial_tree = cKDTree(pial_coords)

    roi_imgs, source_mode = load_roi_volumes(
        roi_labels,
        pm_path=args.pm_atlas,
        mpm_path=args.mpm_atlas,
        allow_mpm_proxy=args.allow_mpm_proxy,
    )
    surf_probs = project_probabilities(roi_imgs, fsaverage.pial_left)

    candidate_imgs, _ = load_roi_volumes(
        OFFICIAL_CANDIDATE_ROIS,
        pm_path=args.pm_atlas,
        mpm_path=args.mpm_atlas,
        allow_mpm_proxy=args.allow_mpm_proxy,
    )
    candidate_probs = project_probabilities(candidate_imgs, fsaverage.pial_left)

    all_electrodes = np.vstack(list(patient_mni.values()))
    snapped_electrodes = snap_points_to_surface(all_electrodes, pial_coords, pial_coords, pial_tree)

    candidate_points = {}
    for label in OFFICIAL_CANDIDATE_ROIS:
        roi_idx = ROI_BRAINNETOME_INDEX[label]
        if source_mode == "pm" and args.pm_atlas.exists():
            vol = np.asarray(image.index_img(nib.load(str(args.pm_atlas)), roi_idx - 1).dataobj)
        else:
            vol = np.asarray(candidate_imgs[label].dataobj)
        coords = np.argwhere(np.nan_to_num(vol) > (0.30 if source_mode == "pm" else 0.45))
        if len(coords) == 0:
            candidate_points[label] = snap_points_to_surface(
                np.array([[0.0, 0.0, 0.0]]), pial_coords, pial_coords, pial_tree
            )[0]
            continue
        world = nib.affines.apply_affine(candidate_imgs[label].affine, coords)
        centroid = world.mean(axis=0, keepdims=True)
        candidate_points[label] = snap_points_to_surface(centroid, pial_coords, pial_coords, pial_tree)[0]

    render_static(
        args.output_dir / "brainnetome_parcellations_surface.png",
        infl_coords=pial_coords,
        infl_faces=pial_faces,
        roi_labels=roi_labels,
        surf_probs=surf_probs,
        snapped_electrodes=snapped_electrodes,
        source_mode=source_mode,
    )
    render_interactive(
        args.output_dir / "brainnetome_parcellations_interactive.html",
        infl_coords=pial_coords,
        infl_faces=pial_faces,
        roi_labels=roi_labels,
        surf_probs=surf_probs,
        snapped_electrodes=snapped_electrodes,
        candidate_probs=candidate_probs,
        candidate_points=candidate_points,
        source_mode=source_mode,
    )

    print(f"Saved static figure: {args.output_dir / 'brainnetome_parcellations_surface.png'}")
    print(f"Saved interactive figure: {args.output_dir / 'brainnetome_parcellations_interactive.html'}")
    if source_mode != "pm":
        print("WARN: rendered with smoothed MPM proxy; drop in BNA_PM_4D.nii.gz for the true PM atlas.")


if __name__ == "__main__":
    main()
