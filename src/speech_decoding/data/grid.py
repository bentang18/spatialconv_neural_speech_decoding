"""Grid shape inference and channel-to-grid mapping for uECOG arrays.

uECOG arrays come in multiple physical layouts:
- 128ch: 8×16 (all positions occupied)
- 256ch: 12×22 (8 dead corner positions), 8×32 (fully occupied),
         or 8×34 (16 dead, S57)

Grid shape is inferred from electrode coordinate TSV files, NOT from
channel count alone, because multiple grid shapes share the same count.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GridInfo:
    """Grid mapping for a single patient's electrode array."""

    grid_shape: tuple[int, int]  # (H, W)
    ch_to_pos: dict[str, tuple[int, int]]  # channel name → (row, col)
    dead_mask: np.ndarray  # (H, W) bool — True where no electrode exists


def load_grid_mapping(electrodes_tsv: str | Path) -> GridInfo:
    """Parse electrode coordinate TSV and build grid mapping.

    Electrode coordinates are normalized grid positions (0-1 range, z=0).
    We quantize them to integer (row, col) indices.
    """
    electrodes_tsv = Path(electrodes_tsv)
    with open(electrodes_tsv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [
            (r["name"], float(r["x"]), float(r["y"]))
            for r in reader
            if r["x"] != "n/a" and r["y"] != "n/a"
        ]

    # Unique sorted coordinate values define the grid axes
    xs = sorted(set(x for _, x, _ in rows))
    ys = sorted(set(y for _, _, y in rows))
    n_cols = len(xs)
    n_rows = len(ys)

    # Build coordinate → grid index lookup
    x_to_col = {x: c for c, x in enumerate(xs)}
    y_to_row = {y: r for r, y in enumerate(ys)}

    # Map each channel to its grid position
    ch_to_pos: dict[str, tuple[int, int]] = {}
    occupied = set()
    for name, x, y in rows:
        r, c = y_to_row[y], x_to_col[x]
        ch_to_pos[str(name)] = (r, c)
        occupied.add((r, c))

    # Dead positions: in the full grid but not occupied
    dead_mask = np.zeros((n_rows, n_cols), dtype=bool)
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) not in occupied:
                dead_mask[r, c] = True

    return GridInfo(
        grid_shape=(n_rows, n_cols),
        ch_to_pos=ch_to_pos,
        dead_mask=dead_mask,
    )


def channels_to_grid(
    data: np.ndarray,
    ch_names: list[str],
    grid_info: GridInfo,
) -> np.ndarray:
    """Reshape channel data to spatial grid.

    Args:
        data: (..., n_channels, T) — channel-ordered HGA data.
            Supports (n_ch, T) or (B, n_ch, T).
        ch_names: Channel names matching data's channel dimension.
        grid_info: Grid mapping from load_grid_mapping().

    Returns:
        (..., H, W, T) array with dead positions zeroed.
    """
    H, W = grid_info.grid_shape
    has_batch = data.ndim == 3
    if not has_batch:
        data = data[np.newaxis]  # (1, n_ch, T)

    B, n_ch, T = data.shape
    grid = np.zeros((B, H, W, T), dtype=data.dtype)

    for i, name in enumerate(ch_names):
        if str(name) in grid_info.ch_to_pos:
            r, c = grid_info.ch_to_pos[str(name)]
            grid[:, r, c, :] = data[:, i, :]

    if not has_batch:
        grid = grid[0]  # remove batch dim
    return grid
