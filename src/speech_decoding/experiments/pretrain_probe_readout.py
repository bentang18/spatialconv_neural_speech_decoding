"""Per-cell readouts for the pretrain probe suite — linear ridge (Stage 2a).

Each readout consumes already-forwarded encoder tap grids (the Stage-1 cache) plus
the per-cell split row indices, and returns the held-out **test** AUROC with the
ridge λ selected on the **val** half (upstream 50/50 val/test split). The attentive
readout (Stage 2b) layers onto the same grids in :mod:`pretrain_probe_attentive`.

Tap-space contract (project_pretrain_probe_suite_contract_2026_06_30) for the LINEAR
ridge (fixed feature vector → cross-subject must intersect to consistent dims):

  - frontend / M2  (electrode-space, ``tap_space="electrode"``):
      WS → pool electrode→parcel (mean), keep-S, all supported parcels, flatten.
      CS → pool electrode→parcel (mean), keep-S, intersect supported parcels P∩, flatten.

  WS-M2 pools to parcels (not the all-electrode ``C·S·d`` flatten) so its feature width
  is the ~16 present parcels — the all-electrode matrix is 46 GB on the largest session
  and OOMs the readout. Parcel-mean collapses only the electrode axis (keep-S survives)
  at parcel resolution, matching M3/M4 so the M2→M3→M4 ladder isolates encoder depth at
  fixed spatial resolution, and reuses the CS electrode→parcel reduction (one code path).
  - M3 / M4        (parcel-space, ``tap_space="parcel"``):
      WS → all parcels, keep-S, flatten.
      CS → parcels keep-S, intersect supported parcels P∩, flatten.

A tap grid is ``(N, n, F, 1)`` where ``n`` is the electrode count (M2) or parcel
count (M3/M4) and ``F`` folds the kept time/seed/channel axes (S·d, k·S·d). This is
the exact ``(N, C, D, 1)`` shape :mod:`v2_raw_probe` already pools/intersects, so the
parcel machinery is reused verbatim.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from speech_decoding.experiments.online_probe import (
    auroc,
    dual_ridge_scores,
    feature_matrix,
    parcel_intersection,
)
from speech_decoding.experiments.v2_raw_probe import (
    pool_electrodes_to_parcels,
)

__all__ = [
    "DEFAULT_LAM_GRID",
    "parcel_support",
    "compacted_positions",
    "linear_ws_cell_auroc",
    "linear_cs_cell_auroc",
]

# λ-multiplier grid swept on the val half (dual_ridge_scores' trace rule scales by it).
DEFAULT_LAM_GRID: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)


def parcel_support(
    parcel_per_electrode: Tensor, electrode_mask: Tensor, n_parcels: int
) -> Tensor:
    """Boolean ``(n_parcels,)`` — parcels with ≥1 valid electrode (CS intersection
    support for the parcel-native M3/M4 taps, where there is no electrode pooling)."""
    present = np.zeros(n_parcels, dtype=bool)
    pe = parcel_per_electrode.long().cpu().numpy()
    em = electrode_mask.bool().cpu().numpy()
    for p, valid in zip(pe, em):
        if valid:
            present[int(p)] = True
    return torch.from_numpy(present)


def compacted_positions(parcel_labels: Tensor, atlas_ids: Tensor) -> Tensor:
    """Map DKT atlas ids → positions in a COMPACTED parcel grid whose axis-1 is
    ``parcel_labels`` (the sorted unique present parcels ``encode_clip_taps`` emits).

    The parcel-space M3/M4 grids are stored compacted (P ≈ 16 present parcels, not the
    full ~80-parcel atlas) to avoid a 5× memory/disk blow-up. ``parcel_support`` /
    ``parcel_intersection`` reason in atlas-id space (length ``n_parcels``); this bridges
    those atlas ids to the grid's own positions via ``searchsorted`` (``parcel_labels`` is
    sorted). Every requested id MUST be present — by construction the CS intersection and
    WS support are subsets of ``parcel_labels`` — so a miss is a fail-loud contract bug."""
    lab = parcel_labels.long().cpu()
    ids = torch.as_tensor(atlas_ids, dtype=torch.long).cpu()
    pos = torch.searchsorted(lab, ids)
    if ids.numel() and (pos.max() >= lab.numel() or not torch.equal(lab[pos], ids)):
        raise ValueError(
            "atlas id absent from parcel_labels — compacted grid axis and support disagree"
        )
    return pos


def _parcel_features(grid: Tensor, atlas_ids: Tensor, parcel_labels: Tensor) -> np.ndarray:
    """Select the atlas parcels ``atlas_ids`` from a COMPACTED parcel grid and flatten."""
    pos = compacted_positions(parcel_labels, atlas_ids)
    return feature_matrix(grid, pos).cpu().numpy()


def _pooled_parcel_features_ws(
    grid: Tensor,
    parcel_per_electrode: Tensor,
    electrode_mask: Tensor,
    n_parcels: int,
) -> np.ndarray:
    """WS electrode-space features, parcel-MEAN pooled → ``(N, P_present·F)``.

    Mirrors the CS electrode reduction (electrode→parcel mean via
    :func:`pool_electrodes_to_parcels`, then this session's own supported parcels — a
    self-intersection, since WS is one montage). Keeps WS-M2's width at the ~16 present
    parcels instead of the all-electrode ``C·S·d`` flatten (46 GB on the largest session
    → OOM). Only the electrode axis reduces; keep-S survives."""
    pooled, present = pool_electrodes_to_parcels(
        grid, parcel_per_electrode, electrode_mask, n_parcels
    )
    atlas_ids = parcel_intersection(present, present)
    return feature_matrix(pooled, atlas_ids).cpu().numpy()


def _all_parcel_features(
    grid: Tensor,
    parcel_per_electrode: Tensor,
    electrode_mask: Tensor,
    n_parcels: int,
    parcel_labels: Tensor,
) -> np.ndarray:
    """WS parcel-space features ``(N, P_present·F)`` — all supported parcels, flatten.

    The M3/M4 grid is COMPACTED (axis-1 = ``parcel_labels``). ``parcel_support`` gives the
    supported atlas ids; :func:`compacted_positions` maps them to the grid's own positions
    so an unsupported parcel never contributes a zero column."""
    present = parcel_support(parcel_per_electrode, electrode_mask, n_parcels)
    atlas_ids = parcel_intersection(present, present)
    return _parcel_features(grid, atlas_ids, parcel_labels)


def _select_lambda(
    z_train: np.ndarray, y_train: np.ndarray, z_val: np.ndarray, y_val: np.ndarray,
    lam_grid: Sequence[float],
) -> float:
    """Pick the λ-multiplier maximizing val AUROC (ties → smallest λ)."""
    best_lam, best_auroc = lam_grid[0], -np.inf
    for lam in lam_grid:
        pred = dual_ridge_scores(z_train, y_train, z_val, lam_mult=lam)
        a = auroc(pred, y_val)
        if np.isfinite(a) and a > best_auroc:
            best_auroc, best_lam = a, lam
    return best_lam


def _finite(z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(np.asarray(y, dtype=np.float64))
    return z[m], np.asarray(y, dtype=np.float64)[m]


def linear_ws_cell_auroc(
    grid: Tensor,
    y: np.ndarray,
    *,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
    tap_space: str,
    parcel_per_electrode: Tensor,
    electrode_mask: Tensor,
    n_parcels: int,
    parcel_labels: Tensor | None = None,
    lam_grid: Sequence[float] = DEFAULT_LAM_GRID,
) -> float:
    """WithinSession cell: fit on ``train_rows``, select λ on ``val_rows``, report
    AUROC on ``test_rows`` — all from one session's grid (fixed montage)."""
    if tap_space == "electrode":
        z = _pooled_parcel_features_ws(
            grid, parcel_per_electrode, electrode_mask, n_parcels
        )
    elif tap_space == "parcel":
        if parcel_labels is None:
            raise ValueError("parcel tap_space needs parcel_labels (compacted grid ids)")
        z = _all_parcel_features(
            grid, parcel_per_electrode, electrode_mask, n_parcels, parcel_labels
        )
    else:
        raise ValueError(f"tap_space must be 'electrode'|'parcel'; got {tap_space!r}")

    zt, yt = _finite(z[train_rows], y[train_rows])
    zv, yv = _finite(z[val_rows], y[val_rows])
    ze, ye = _finite(z[test_rows], y[test_rows])
    if len(yt) < 2 or len(ye) < 2:
        return float("nan")
    lam = _select_lambda(zt, yt, zv, yv, lam_grid) if len(yv) >= 2 else lam_grid[0]
    return auroc(dual_ridge_scores(zt, yt, ze, lam_mult=lam), ye)


def linear_cs_cell_auroc(
    grid_anchor: Tensor,
    y_anchor: np.ndarray,
    grid_test: Tensor,
    y_test: np.ndarray,
    *,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
    tap_space: str,
    pe_anchor: Tensor,
    em_anchor: Tensor,
    pe_test: Tensor,
    em_test: Tensor,
    n_parcels: int,
    parcel_labels_anchor: Tensor | None = None,
    parcel_labels_test: Tensor | None = None,
    lam_grid: Sequence[float] = DEFAULT_LAM_GRID,
) -> float:
    """CrossSubject cell: fit on the anchor (train), select λ on the test session's
    val half, report AUROC on its test half. Both taps reduce to a common parcel
    set so the ridge sees consistent feature dims (electrode → pool→parcel→P∩;
    parcel-native → supported-parcel P∩)."""
    if tap_space == "electrode":
        pooled_a, present_a = pool_electrodes_to_parcels(
            grid_anchor, pe_anchor, em_anchor, n_parcels
        )
        pooled_t, present_t = pool_electrodes_to_parcels(
            grid_test, pe_test, em_test, n_parcels
        )
        inter = parcel_intersection(present_a, present_t)
        if inter.numel() == 0:
            return float("nan")
        za_all = feature_matrix(pooled_a, inter).cpu().numpy()
        zt_all = feature_matrix(pooled_t, inter).cpu().numpy()
    elif tap_space == "parcel":
        if parcel_labels_anchor is None or parcel_labels_test is None:
            raise ValueError("parcel tap_space needs anchor + test parcel_labels")
        present_a = parcel_support(pe_anchor, em_anchor, n_parcels)
        present_t = parcel_support(pe_test, em_test, n_parcels)
        inter = parcel_intersection(present_a, present_t)
        if inter.numel() == 0:
            return float("nan")
        za_all = _parcel_features(grid_anchor, inter, parcel_labels_anchor)
        zt_all = _parcel_features(grid_test, inter, parcel_labels_test)
    else:
        raise ValueError(f"tap_space must be 'electrode'|'parcel'; got {tap_space!r}")

    za, ya = _finite(za_all, y_anchor)
    zv, yv = _finite(zt_all[val_rows], y_test[val_rows])
    ze, ye = _finite(zt_all[test_rows], y_test[test_rows])
    if len(ya) < 2 or len(ye) < 2:
        return float("nan")
    lam = _select_lambda(za, ya, zv, yv, lam_grid) if len(yv) >= 2 else lam_grid[0]
    return auroc(dual_ridge_scores(za, ya, ze, lam_mult=lam), ye)
