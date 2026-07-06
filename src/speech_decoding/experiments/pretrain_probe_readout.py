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
    "DEFAULT_LAM",
    "DEFAULT_LAM_GRID",
    "SWEEP_LAM_GRID",
    "parcel_support",
    "compacted_positions",
    "linear_ws_cell_auroc",
    "linear_cs_cell_auroc",
    "linear_ws_cell_scores",
    "linear_cs_cell_scores",
    "linear_ws_cell_scores_all_electrode",
]

# Single fixed λ-multiplier used for scoring BY DEFAULT (dual_ridge_scores' trace rule
# scales by it). 1.0 is the aggregate-val optimum measured across M2/M3/M4 (CS emit-lam,
# 2026-07-02) and the geometric centre of the sweep grid below. Per-cell val-argmax over
# SWEEP_LAM_GRID beats this fixed λ by only ~0.004 test-AUROC — a test-peek optimism on a
# flat-in-λ landscape, not signal — so the default is ONE solve, no sweep. Callers that
# want the diagnostic sweep (run_keepS_ridge, equivalence tests) pass SWEEP_LAM_GRID.
DEFAULT_LAM: float = 1.0
DEFAULT_LAM_GRID: tuple[float, ...] = (DEFAULT_LAM,)
# Opt-in diagnostic sweep grid — pass explicitly to reproduce a per-cell λ pick or to
# re-freeze a tap's λ. Not the default (see above).
SWEEP_LAM_GRID: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)


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


def _lam_scores(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    lam_grid: Sequence[float],
) -> list[tuple[float, float, float]]:
    """Per-λ ``(lam_mult, val_auroc, test_auroc)`` over the whole grid — val drives
    selection, test is the report. Emitting EVERY λ (instead of collapsing to the
    val-argmax inside the cell) lets the caller pick ONE λ per tap across all cells (a
    fixed-HP readout, matching how the attentive path freezes HPs) OR reproduce the
    per-cell pick post-hoc. Dual ridge is feature-dim-free, so the sweep only re-solves
    the N×N system. ``val_auroc`` is NaN when the val half has <2 samples (the old
    ``len(yv) < 2 → lam_grid[0]`` fallback lives in :func:`_selected_test`).

    ``g`` and the cross-kernels are λ-independent, so they are built ONCE and the λ-loop
    re-solves only the N×N system (identical to per-λ ``dual_ridge_scores`` — same
    ``lam = lam_mult·trace(g)/n`` scale rule — but without rebuilding ``ZZᵀ`` per λ; the
    val cross-kernel is also reused, so a fixed λ costs one Gram, not two). This is the
    dual-form analogue already used by :func:`_lam_scores_from_kernels`."""
    z_train = np.asarray(z_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    g = z_train @ z_train.T
    n = g.shape[0]
    base = float(np.trace(g) / max(n, 1))
    eye = np.eye(n)
    has_val = len(y_val) >= 2
    k_val = np.asarray(z_val, dtype=np.float64) @ z_train.T if has_val else None
    k_test = np.asarray(z_test, dtype=np.float64) @ z_train.T
    out: list[tuple[float, float, float]] = []
    for lam in lam_grid:
        alpha = np.linalg.solve(g + (lam * base) * eye, y_train)
        va = auroc(k_val @ alpha, y_val) if has_val else float("nan")
        te = auroc(k_test @ alpha, y_test)
        out.append((float(lam), float(va), float(te)))
    return out


def _lam_scores_from_kernels(
    g: np.ndarray,
    y_train: np.ndarray,
    k_val: np.ndarray,
    y_val: np.ndarray,
    k_test: np.ndarray,
    y_test: np.ndarray,
    lam_grid: Sequence[float],
) -> list[tuple[float, float, float]]:
    """Per-λ ``(lam, val_auroc, test_auroc)`` from PREcomputed dual-ridge kernels — the
    streamed-Gram analogue of :func:`_lam_scores`. ``g = Z_trainᵀ`` Gram (n×n),
    ``k_val = Z_val Z_trainᵀ`` / ``k_test = Z_test Z_trainᵀ`` (built once over electrode
    blocks). ``lam = lam_mult · trace(g)/n`` reproduces ``dual_ridge_scores``' scale-aware
    default exactly, so this returns the SAME list as ``_lam_scores`` on the same features
    (up to summation-order fp noise, irrelevant to rank-based AUROC)."""
    y_train = np.asarray(y_train, dtype=np.float64)
    n = g.shape[0]
    base = float(np.trace(g) / max(n, 1))
    eye = np.eye(n)
    out: list[tuple[float, float, float]] = []
    for lam in lam_grid:
        alpha = np.linalg.solve(g + (lam * base) * eye, y_train)
        va = auroc(k_val @ alpha, y_val) if len(y_val) >= 2 else float("nan")
        te = auroc(k_test @ alpha, y_test)
        out.append((float(lam), float(va), float(te)))
    return out


def _selected_test(
    scores: Sequence[tuple[float, float, float]], lam_grid: Sequence[float]
) -> float:
    """Per-cell λ selection: test AUROC at the λ maximizing val (ties → smallest λ, via
    grid order + strict ``>``). When no λ has a finite val (val < 2 samples), fall back to
    ``lam_grid[0]`` — the exact rule the old ``_select_lambda`` / ``len(yv) < 2`` branch used."""
    best: tuple[float, float, float] | None = None
    for lam, va, te in scores:
        if np.isfinite(va) and (best is None or va > best[1]):
            best = (lam, va, te)
    if best is not None:
        return best[2]
    for lam, _va, te in scores:
        if lam == lam_grid[0]:
            return te
    return scores[0][2] if scores else float("nan")


def _ws_cell_matrices(
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
    parcel_labels: Tensor | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """``(z_train, y_train, z_val, y_val, z_test, y_test)`` for a WS cell, or ``None`` when
    degenerate (<2 train or test samples). Shared by the AUROC and per-λ score paths so
    both build features identically."""
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
        return None
    return zt, yt, zv, yv, ze, ye


# Anchor-pool memo. In a CrossSubject ridge pass the SAME subj-2 anchor grid is pooled to
# parcels once per cell (90×/checkpoint) though the electrode→parcel reduction is
# task-independent. Cache the last anchor's pooled result, keyed on the tensor's id and
# holding a ref to the grid so its id can't be recycled under us. One entry (cleared on a
# new anchor tensor) → bounded to a single pooled grid, which the anchor cache already keeps
# resident, so no extra memory. Test grids are deliberately NOT memoed: LazyCacheMap evicts
# them between subjects, so an id-keyed memo could take a recycled-id false hit; that
# residual (each test grid pooled 15×/subject) is left to a cache-scoped fix if ever needed.
_ANCHOR_POOL: dict[int, tuple[Tensor, Tensor, Tensor]] = {}


def _pool_anchor_cached(
    grid: Tensor, pe: Tensor, em: Tensor, n_parcels: int
) -> tuple[Tensor, Tensor]:
    key = id(grid)
    hit = _ANCHOR_POOL.get(key)
    if hit is not None:
        return hit[0], hit[1]
    pooled, present = pool_electrodes_to_parcels(grid, pe, em, n_parcels)
    _ANCHOR_POOL.clear()
    _ANCHOR_POOL[key] = (pooled, present, grid)
    return pooled, present


def _cs_cell_matrices(
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
    parcel_labels_anchor: Tensor | None,
    parcel_labels_test: Tensor | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """``(z_train=anchor, y_train, z_val, y_val, z_test, y_test)`` for a CS cell, or
    ``None`` when the anchor∩test parcel set is empty or a split is degenerate."""
    if tap_space == "electrode":
        pooled_a, present_a = _pool_anchor_cached(
            grid_anchor, pe_anchor, em_anchor, n_parcels
        )
        pooled_t, present_t = pool_electrodes_to_parcels(
            grid_test, pe_test, em_test, n_parcels
        )
        inter = parcel_intersection(present_a, present_t)
        if inter.numel() == 0:
            return None
        za_all = feature_matrix(pooled_a, inter).cpu().numpy()
        zt_all = feature_matrix(pooled_t, inter).cpu().numpy()
    elif tap_space == "parcel":
        if parcel_labels_anchor is None or parcel_labels_test is None:
            raise ValueError("parcel tap_space needs anchor + test parcel_labels")
        present_a = parcel_support(pe_anchor, em_anchor, n_parcels)
        present_t = parcel_support(pe_test, em_test, n_parcels)
        inter = parcel_intersection(present_a, present_t)
        if inter.numel() == 0:
            return None
        za_all = _parcel_features(grid_anchor, inter, parcel_labels_anchor)
        zt_all = _parcel_features(grid_test, inter, parcel_labels_test)
    else:
        raise ValueError(f"tap_space must be 'electrode'|'parcel'; got {tap_space!r}")

    za, ya = _finite(za_all, y_anchor)
    zv, yv = _finite(zt_all[val_rows], y_test[val_rows])
    ze, ye = _finite(zt_all[test_rows], y_test[test_rows])
    if len(ya) < 2 or len(ye) < 2:
        return None
    return za, ya, zv, yv, ze, ye


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
    m = _ws_cell_matrices(
        grid, y, train_rows=train_rows, val_rows=val_rows, test_rows=test_rows,
        tap_space=tap_space, parcel_per_electrode=parcel_per_electrode,
        electrode_mask=electrode_mask, n_parcels=n_parcels, parcel_labels=parcel_labels,
    )
    if m is None:
        return float("nan")
    return _selected_test(_lam_scores(*m, lam_grid), lam_grid)


def linear_ws_cell_scores(
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
) -> list[tuple[float, float, float]]:
    """Per-λ ``(lam, val_auroc, test_auroc)`` for a WS cell (``[]`` if degenerate). Same
    features as :func:`linear_ws_cell_auroc`; lets one λ be fixed per tap across cells."""
    m = _ws_cell_matrices(
        grid, y, train_rows=train_rows, val_rows=val_rows, test_rows=test_rows,
        tap_space=tap_space, parcel_per_electrode=parcel_per_electrode,
        electrode_mask=electrode_mask, n_parcels=n_parcels, parcel_labels=parcel_labels,
    )
    return _lam_scores(*m, lam_grid) if m is not None else []


def _ws_all_electrode_kernels(
    grid: Tensor,
    y: np.ndarray,
    *,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
    electrode_mask: Tensor,
    target_bytes: int = 2_000_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Dual-ridge kernels ``(g, y_train, k_val, y_val, k_test, y_test)`` for an
    ALL-ELECTRODE keep-S WS cell — every valid electrode's tokens, NO parcel pool — built
    by STREAMING over electrode blocks so peak memory is one block, not the (N, C·S·d)
    flatten (~55 GB on a Lite-120 M2 tap → OOMs a 111 GB node). ``g``/``k`` are additive
    over the feature axis, so block-accumulation equals one BLAS ``Z Zᵀ`` up to summation
    order (the Gram is permutation-invariant in the electrode axis). ``None`` if degenerate
    (<2 finite train or test labels), matching :func:`_ws_cell_matrices`."""
    y = np.asarray(y, dtype=np.float64)

    def _split(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(rows)
        m = np.isfinite(y[r])
        return r[m], y[r][m]

    tr, y_tr = _split(train_rows)
    v, y_v = _split(val_rows)
    te, y_te = _split(test_rows)
    if len(y_tr) < 2 or len(y_te) < 2:
        return None

    valid = electrode_mask.bool().nonzero().flatten()
    B = int(grid.shape[0])
    per_elec = B * int(np.prod([int(s) for s in grid.shape[2:]])) * 8
    block = max(1, target_bytes // max(per_elec, 1))
    n_tr = len(tr)
    g = np.zeros((n_tr, n_tr), dtype=np.float64)
    k_val = np.zeros((len(v), n_tr), dtype=np.float64)
    k_test = np.zeros((len(te), n_tr), dtype=np.float64)
    for b0 in range(0, len(valid), block):
        cs = valid[b0:b0 + block]
        fb = grid.index_select(1, cs).reshape(B, -1).double().cpu().numpy()  # (B, blk·S·d)
        z_tr = fb[tr]
        g += z_tr @ z_tr.T
        if len(v):
            k_val += fb[v] @ z_tr.T
        k_test += fb[te] @ z_tr.T
    return g, y_tr, k_val, y_v, k_test, y_te


def linear_ws_cell_scores_all_electrode(
    grid: Tensor,
    y: np.ndarray,
    *,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
    electrode_mask: Tensor,
    lam_grid: Sequence[float] = DEFAULT_LAM_GRID,
) -> list[tuple[float, float, float]]:
    """Per-λ ``(lam, val_auroc, test_auroc)`` for a WS cell read at FULL electrode
    resolution (no parcel pool) — the un-crippled M2 WS read the pooled ladder can't do,
    via the streamed Gram. ``[]`` if degenerate. Same per-λ contract as
    :func:`linear_ws_cell_scores`, so the fixed-per-tap-λ aggregation is unchanged."""
    k = _ws_all_electrode_kernels(
        grid, y, train_rows=train_rows, val_rows=val_rows, test_rows=test_rows,
        electrode_mask=electrode_mask,
    )
    return _lam_scores_from_kernels(*k, lam_grid) if k is not None else []


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
    m = _cs_cell_matrices(
        grid_anchor, y_anchor, grid_test, y_test, val_rows=val_rows, test_rows=test_rows,
        tap_space=tap_space, pe_anchor=pe_anchor, em_anchor=em_anchor, pe_test=pe_test,
        em_test=em_test, n_parcels=n_parcels, parcel_labels_anchor=parcel_labels_anchor,
        parcel_labels_test=parcel_labels_test,
    )
    if m is None:
        return float("nan")
    return _selected_test(_lam_scores(*m, lam_grid), lam_grid)


def linear_cs_cell_scores(
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
) -> list[tuple[float, float, float]]:
    """Per-λ ``(lam, val_auroc, test_auroc)`` for a CS cell (``[]`` if degenerate)."""
    m = _cs_cell_matrices(
        grid_anchor, y_anchor, grid_test, y_test, val_rows=val_rows, test_rows=test_rows,
        tap_space=tap_space, pe_anchor=pe_anchor, em_anchor=em_anchor, pe_test=pe_test,
        em_test=em_test, n_parcels=n_parcels, parcel_labels_anchor=parcel_labels_anchor,
        parcel_labels_test=parcel_labels_test,
    )
    return _lam_scores(*m, lam_grid) if m is not None else []
