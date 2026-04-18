"""Masked-mean pool primitive for the v14-core baseline-aligned rework (P2).

One scatter-mean kernel services two callers:

* Stage 3 spatial pool: `(B, C, H_p, W_p, T)` grid → `(B, C, n_cells, T)` after
  reshape to `(B, C, H_p·W_p, T)` and pool along `dim=2`.
* Parcel-support pool:  `(B, N_e, P)` → `(B, n_cells, P)` along `dim=1`.

The pool is non-overlapping integer blocks. All Phase-1 grid shapes
(8×16, 16×16, 12×24) divide evenly into the baseline `(4, 8)` pool, so this
matches `AdaptiveAvgPool2d((4, 8))` exactly on all-active inputs. Mixed
divisibility would need adaptive-pool's overlapping bins — we assert instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PoolPrecompute:
    """Per-patient constants. `active_count` depends on `info["bads"]` which is
    trial-invariant, so these are computed once in the dataset and reused."""

    n_cells: int
    pool_shape: tuple[int, int]
    grid_shape: tuple[int, int]
    cell_of_grid: torch.Tensor       # (H_p·W_p,) long
    cell_of_electrode: torch.Tensor  # (N_e,) long
    grid_active: torch.Tensor        # (H_p·W_p,) bool — True at active electrode positions
    active_count: torch.Tensor       # (n_cells,) float — #active electrodes per cell
    cell_active_mask: torch.Tensor   # (n_cells,) bool — True where active_count > 0


def precompute_pool_assignment(
    grid_shape: tuple[int, int],
    pool_shape: tuple[int, int] = (4, 8),
) -> torch.Tensor:
    """Return `cell_of_grid: (H_p·W_p,)` mapping each grid position to a pool cell.

    Row-major flat index: `flat(r, c) = r · W_p + c`.
    Cell index is row-major in the pool grid: `cell = (r // bH) · out_W + (c // bW)`.
    """

    H_p, W_p = grid_shape
    out_H, out_W = pool_shape
    if H_p % out_H != 0 or W_p % out_W != 0:
        raise ValueError(
            f"grid {grid_shape} not divisible by pool {pool_shape}; "
            "add adaptive-pool overlapping bins if this grid shape is real"
        )
    block_H, block_W = H_p // out_H, W_p // out_W

    rs = torch.arange(H_p).repeat_interleave(W_p)  # (H_p·W_p,)
    cs = torch.arange(W_p).repeat(H_p)
    return (rs // block_H) * out_W + (cs // block_W)


def precompute_pool(
    electrode_grid_layout: torch.Tensor,  # (N_e, 2) long
    electrode_active_mask: torch.Tensor,  # (N_e,) bool
    grid_shape: tuple[int, int],
    pool_shape: tuple[int, int] = (4, 8),
) -> PoolPrecompute:
    """Bundle all pool-derived constants for a patient."""

    if electrode_grid_layout.ndim != 2 or electrode_grid_layout.shape[1] != 2:
        raise ValueError(
            f"electrode_grid_layout shape {tuple(electrode_grid_layout.shape)} != (N_e, 2)"
        )
    if electrode_active_mask.shape != (electrode_grid_layout.shape[0],):
        raise ValueError(
            f"active mask shape {tuple(electrode_active_mask.shape)} != "
            f"({electrode_grid_layout.shape[0]},)"
        )

    H_p, W_p = grid_shape
    out_H, out_W = pool_shape
    n_cells = out_H * out_W

    cell_of_grid = precompute_pool_assignment(grid_shape, pool_shape)

    rows = electrode_grid_layout[:, 0].long()
    cols = electrode_grid_layout[:, 1].long()
    flat_e = rows * W_p + cols
    cell_of_electrode = cell_of_grid[flat_e]

    grid_active = torch.zeros(H_p * W_p, dtype=torch.bool)
    active_flat_e = flat_e[electrode_active_mask]
    grid_active[active_flat_e] = True

    active_count = torch.zeros(n_cells, dtype=torch.float32)
    active_count.index_add_(
        0,
        cell_of_electrode[electrode_active_mask],
        torch.ones(int(electrode_active_mask.sum()), dtype=torch.float32),
    )
    cell_active_mask = active_count > 0

    return PoolPrecompute(
        n_cells=n_cells,
        pool_shape=pool_shape,
        grid_shape=grid_shape,
        cell_of_grid=cell_of_grid,
        cell_of_electrode=cell_of_electrode,
        grid_active=grid_active,
        active_count=active_count,
        cell_active_mask=cell_active_mask,
    )


def masked_mean_pool(
    x: torch.Tensor,
    cell_of: torch.Tensor,        # (N,) long
    active: torch.Tensor,         # (N,) bool
    active_count: torch.Tensor,   # (n_cells,) float
    *,
    dim: int,
    n_cells: int,
) -> torch.Tensor:
    """Scatter-mean along `dim`, masked by `active`, normalized by `active_count`.

    * Zeros contributions from inactive positions (via the mask — no branch on `active`).
    * Empty cells (`active_count == 0`) return 0; avoid div-by-zero with a safe denom.
    """

    if cell_of.shape != (x.shape[dim],):
        raise ValueError(
            f"cell_of shape {tuple(cell_of.shape)} incompatible with x.shape[{dim}]={x.shape[dim]}"
        )
    if active.shape != (x.shape[dim],):
        raise ValueError(
            f"active shape {tuple(active.shape)} incompatible with x.shape[{dim}]={x.shape[dim]}"
        )
    if active_count.shape != (n_cells,):
        raise ValueError(
            f"active_count shape {tuple(active_count.shape)} != ({n_cells},)"
        )

    mask_shape = [1] * x.ndim
    mask_shape[dim] = -1
    mask = active.to(x.dtype).view(mask_shape)
    x_masked = x * mask

    out_shape = list(x.shape)
    out_shape[dim] = n_cells
    out = x.new_zeros(out_shape)

    idx_shape = [1] * x.ndim
    idx_shape[dim] = -1
    idx = cell_of.view(idx_shape).expand_as(x_masked)
    out.scatter_add_(dim, idx, x_masked)

    safe_count = torch.where(
        active_count > 0, active_count, torch.ones_like(active_count)
    )
    count_shape = [1] * x.ndim
    count_shape[dim] = -1
    return out / safe_count.view(count_shape)
