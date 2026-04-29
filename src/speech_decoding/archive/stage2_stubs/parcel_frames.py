"""ARCHIVED: parcel-frame construction on the baked fsaverage atlas (pre-amendment ``#10``).

The 2026-04-16 late B-1 amendment retired ``parcel_frames.npz`` from the
v14-core pipeline. The B-1 architecture has no within-parcel positional
encoding: registration noise (~3-5 mm RMS) plus inter-individual
anatomical variability (~10 mm RMS) gives ~11 mm joint uncertainty in
localizing sub-parcel functional landmarks, which is larger than the
5-10 mm cluster width we'd want to align. Sub-parcel cross-patient
position-driven transfer is therefore below the SNR floor, and the
intra-parcel ``(u, v)`` chart is no longer used for token construction.

This module and ``scripts/build_fsaverage_parcel_frames.py`` are kept
for archival reference only — they remain useful for visualizing parcel
geometry on fsaverage and for parity comparisons against the cvs_avg35
oracle, but no v14-core code path consumes the resulting ``parcel_frames.npz``.
See ``docs/v14_core_contract_amendment_2026-04-16.md`` for the full record.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class ParcelFrame:
    parcel_name: str
    parcel_idx_1based: int
    hemisphere: str
    origin_vertex: int
    origin_xyz: np.ndarray  # (3,)
    basis: np.ndarray  # (3, 3), columns are u, v, z
    sigma_u: float
    sigma_v: float

    def __post_init__(self) -> None:
        assert self.origin_xyz.shape == (3,)
        assert self.basis.shape == (3, 3)


def load_surface_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing surface file: {path}")
    verts, faces = nib.freesurfer.io.read_geometry(str(path))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(verts, dtype=np.float64)
    tri = verts[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    zero = norms.squeeze(1) <= 1e-12
    if zero.any():
        normals[zero] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        norms[zero] = 1.0
    return normals / norms


def _normalize(vec: np.ndarray, *, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-10:
        raise ValueError(f"{name} is degenerate")
    return vec / norm


def _project_tangent(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return direction - np.dot(direction, normal) * normal


def _choose_u_axis(z_axis: np.ndarray) -> np.ndarray:
    for name, direction in (
        ("anterior", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
        ("superior", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
        ("left_right", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
    ):
        tangent = _project_tangent(direction, z_axis)
        if np.linalg.norm(tangent) > 1e-8:
            return _normalize(tangent, name=name)
    raise ValueError("could not choose a tangent u-axis")


def build_parcel_frame(
    *,
    parcel_name: str,
    parcel_idx_1based: int,
    hemisphere: str,
    verts: np.ndarray,
    faces: np.ndarray,
    weights: np.ndarray,
) -> ParcelFrame:
    if weights.shape != (verts.shape[0],):
        raise ValueError(f"weights shape {weights.shape} incompatible with verts {verts.shape}")
    support_mask = weights > 0
    if support_mask.sum() < 3:
        raise ValueError(f"{parcel_name}: only {int(support_mask.sum())} support vertices")

    support_weights = weights[support_mask]
    support_verts = verts[support_mask]
    centroid = (support_weights[:, None] * support_verts).sum(axis=0) / support_weights.sum()
    support_vids = np.flatnonzero(support_mask)
    origin_local_idx = np.argmin(np.linalg.norm(support_verts - centroid[None, :], axis=1))
    origin_vertex = int(support_vids[int(origin_local_idx)])
    origin_xyz = verts[origin_vertex].astype(np.float64, copy=True)

    normals = compute_vertex_normals(verts, faces)
    weighted_normal = (weights[:, None] * normals).sum(axis=0)
    if np.linalg.norm(weighted_normal) <= 1e-8:
        weighted_normal = normals[origin_vertex]
    z_axis = _normalize(weighted_normal, name=f"{parcel_name} weighted normal")
    u_axis = _choose_u_axis(z_axis)
    v_axis = _normalize(np.cross(z_axis, u_axis), name=f"{parcel_name} v-axis")
    basis = np.column_stack([u_axis, v_axis, z_axis])

    rel = support_verts - origin_xyz[None, :]
    u_coords = rel @ u_axis
    v_coords = rel @ v_axis
    sigma_u = float(np.sqrt(np.average(u_coords ** 2, weights=support_weights)))
    sigma_v = float(np.sqrt(np.average(v_coords ** 2, weights=support_weights)))
    if sigma_u <= 1e-8 or sigma_v <= 1e-8:
        raise ValueError(f"{parcel_name}: degenerate sigma_u={sigma_u} sigma_v={sigma_v}")

    return ParcelFrame(
        parcel_name=parcel_name,
        parcel_idx_1based=parcel_idx_1based,
        hemisphere=hemisphere,
        origin_vertex=origin_vertex,
        origin_xyz=origin_xyz,
        basis=basis,
        sigma_u=sigma_u,
        sigma_v=sigma_v,
    )


def save_parcel_frames(
    frames: list[ParcelFrame],
    out_path: Path,
    *,
    atlas_dir: Path,
    target_subject: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        parcel_names=np.array([frame.parcel_name for frame in frames], dtype=object),
        parcel_indices_1based=np.array([frame.parcel_idx_1based for frame in frames], dtype=np.int64),
        hemisphere=np.array([frame.hemisphere for frame in frames], dtype=object),
        origin_vertex=np.array([frame.origin_vertex for frame in frames], dtype=np.int64),
        origin_xyz=np.stack([frame.origin_xyz for frame in frames], axis=0),
        basis=np.stack([frame.basis for frame in frames], axis=0),
        sigma_u=np.array([frame.sigma_u for frame in frames], dtype=np.float64),
        sigma_v=np.array([frame.sigma_v for frame in frames], dtype=np.float64),
        atlas_dir=np.array([str(atlas_dir)], dtype=object),
        target_subject=np.array([target_subject], dtype=object),
    )


def plot_parcel_frame_qc(
    *,
    frame: ParcelFrame,
    verts: np.ndarray,
    weights: np.ndarray,
    out_path: Path,
) -> None:
    mask = weights > 0
    support_verts = verts[mask]
    support_weights = weights[mask]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        support_verts[:, 0],
        support_verts[:, 1],
        support_verts[:, 2],
        c=support_weights,
        cmap="viridis",
        s=2.0,
        alpha=0.7,
    )
    ax.scatter(
        [frame.origin_xyz[0]],
        [frame.origin_xyz[1]],
        [frame.origin_xyz[2]],
        c="crimson",
        s=40,
        label="origin",
    )

    scales = np.array([frame.sigma_u, frame.sigma_v, max(frame.sigma_u, frame.sigma_v)])
    colors = ("tab:blue", "tab:orange", "tab:green")
    labels = ("u", "v", "z")
    for axis_idx, (scale, color, label) in enumerate(zip(scales, colors, labels)):
        vec = frame.basis[:, axis_idx] * scale
        ax.quiver(
            frame.origin_xyz[0],
            frame.origin_xyz[1],
            frame.origin_xyz[2],
            vec[0],
            vec[1],
            vec[2],
            color=color,
            linewidth=2.0,
            label=label,
        )

    ax.set_title(f"{frame.parcel_name} ({frame.parcel_idx_1based})")
    ax.set_xlabel("R")
    ax.set_ylabel("A")
    ax.set_zlabel("S")
    ax.legend(loc="upper right")
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="raw PM weight")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
