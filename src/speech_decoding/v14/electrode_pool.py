"""3D-Euclidean pial pooling for electrode -> parcel support.

Motivation (Zac 2026-04-16): uECoG grids sit on the pial surface above gyral
crowns and do not conform to sulci. Geodesic smoothing baked into the atlas
therefore conflates two physically distinct kernels — atlas registration
uncertainty (geodesic, on cortex) and electrode LFP catchment (3D Euclidean,
pial-side, crown-biased). Moving the catchment kernel out of the atlas and
onto the electrode side is the fix.

This module expresses electrode catchment as a 3D Gaussian pool over pial
vertices. Each electrode's support vector is

    support(e, k) = sum_v w(e, v) * atlas(v, k)            sum_v w(e, v) = 1

with w(e, v) = 0 outside `radius_mm`, otherwise Gaussian in Euclidean distance
and, when `crown_bias=True`, multiplied by a sigmoid crown-preference on
vertex curvature (curv < 0 = gyral crown -> higher weight).

The function is hemisphere-agnostic. Callers pass the pial mesh and electrodes
already restricted to one hemisphere.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

_CURV_TAU = 0.1  # mm^-1; half-saturation of crown sigmoid


def euclidean_support(
    electrode_xyz: np.ndarray,
    pial_xyz: np.ndarray,
    atlas_support: np.ndarray,
    curv: np.ndarray,
    *,
    sigma_mm: float,
    radius_mm: float,
    crown_bias: bool,
) -> np.ndarray:
    """Pool atlas support around each electrode via a 3D Euclidean Gaussian.

    Args:
        electrode_xyz: (N_e, 3) electrode positions in pial RAS (mm).
        pial_xyz: (V, 3) pial vertex positions in pial RAS (mm).
        atlas_support: (V, K) per-vertex support over K parcels.
        curv: (V,) FreeSurfer curvature at each pial vertex. Ignored when
            crown_bias is False, but the array must still have length V.
        sigma_mm: Gaussian standard deviation in mm.
        radius_mm: Hard cutoff for the pool (weights zero beyond this).
        crown_bias: Multiply Gaussian weights by sigmoid(-curv / tau).

    Returns:
        (N_e, K) float32 support. Electrodes with no pial vertex inside
        `radius_mm` receive a zero row.
    """
    if pial_xyz.shape[0] != atlas_support.shape[0]:
        raise ValueError(
            f"pial_xyz has {pial_xyz.shape[0]} verts, "
            f"atlas_support has {atlas_support.shape[0]}"
        )
    if curv.shape[0] != pial_xyz.shape[0]:
        raise ValueError(
            f"curv has {curv.shape[0]} verts, pial_xyz has {pial_xyz.shape[0]}"
        )

    tree = cKDTree(pial_xyz)
    neighbour_lists = tree.query_ball_point(electrode_xyz, r=radius_mm, p=2.0)

    n_e = electrode_xyz.shape[0]
    k = atlas_support.shape[1]
    out = np.zeros((n_e, k), dtype=np.float32)

    two_sigma_sq = 2.0 * (sigma_mm ** 2)

    for i, vids in enumerate(neighbour_lists):
        if not vids:
            continue
        vids_arr = np.asarray(vids, dtype=np.int64)
        d2 = np.sum((pial_xyz[vids_arr] - electrode_xyz[i]) ** 2, axis=1)
        w = np.exp(-d2 / two_sigma_sq)
        if crown_bias:
            # sigmoid(-curv / tau): crown (curv<0) -> >0.5; sulcus (curv>0) -> <0.5.
            w = w * _sigmoid(-curv[vids_arr] / _CURV_TAU)
        total = w.sum()
        if total <= 0.0:
            continue
        w = w / total
        out[i] = (w[:, None] * atlas_support[vids_arr]).sum(axis=0).astype(np.float32)
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
