"""Tests for the v14 masked-mean pool primitive (P2)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.v14.pool import (
    masked_mean_pool,
    precompute_pool,
    precompute_pool_assignment,
)


class TestPrecomputePoolAssignment:
    def test_matches_adaptive_pool_shape(self) -> None:
        cell_of = precompute_pool_assignment((8, 16), (4, 8))
        assert cell_of.shape == (128,)
        assert int(cell_of.min()) == 0
        assert int(cell_of.max()) == 31  # n_cells - 1

    def test_block_assignment_row_major(self) -> None:
        # 8x16 grid, (4,8) pool → 2-row x 2-col blocks.
        cell_of = precompute_pool_assignment((8, 16), (4, 8)).view(8, 16)
        # Top-left 2x2 block → cell 0.
        assert torch.equal(cell_of[0:2, 0:2], torch.full((2, 2), 0))
        # Next 2 cols along same row → cell 1.
        assert torch.equal(cell_of[0:2, 2:4], torch.full((2, 2), 1))
        # Next 2 rows, first 2 cols → cell 8 (new pool row, first col).
        assert torch.equal(cell_of[2:4, 0:2], torch.full((2, 2), 8))
        # Bottom-right corner → cell 31.
        assert int(cell_of[-1, -1]) == 31

    def test_rejects_grid_smaller_than_pool(self) -> None:
        with pytest.raises(ValueError, match="smaller than pool"):
            precompute_pool_assignment((3, 16), (4, 8))

    def test_non_divisible_grid_has_no_overlap(self) -> None:
        # 12x22 grid, (4, 8) pool: H divides evenly (3 rows per cell); W does
        # not (22 / 8 = 2.75). AdaptiveAvgPool2d-style start-index binning gives
        # col widths 3,3,2,3,3,2,3,3 — uneven but non-overlapping.
        cell_of = precompute_pool_assignment((12, 22), (4, 8)).view(12, 22)
        # Every input position maps to exactly one cell.
        assert int(cell_of.min()) == 0
        assert int(cell_of.max()) == 31
        # Row boundaries are exact: rows 0-2 → pool-row 0, 3-5 → 1, 6-8 → 2, 9-11 → 3.
        assert torch.all(cell_of[0:3, :] // 8 == 0)
        assert torch.all(cell_of[3:6, :] // 8 == 1)
        assert torch.all(cell_of[6:9, :] // 8 == 2)
        assert torch.all(cell_of[9:12, :] // 8 == 3)
        # Expected col-bin widths (start-index rule): 3,3,3,2,3,3,3,2 for
        # W_p=22, out_W=8 → bin_of(c) = (c·8)//22.
        col_of_pool = cell_of[0] % 8
        widths = torch.bincount(col_of_pool)
        assert widths.tolist() == [3, 3, 3, 2, 3, 3, 3, 2]
        assert widths.sum().item() == 22

    def test_deterministic(self) -> None:
        a = precompute_pool_assignment((12, 24), (4, 8))
        b = precompute_pool_assignment((12, 24), (4, 8))
        assert torch.equal(a, b)


class TestPrecomputePool:
    def test_full_grid_all_active(self) -> None:
        H_p, W_p = 8, 16
        N_e = H_p * W_p
        rows = torch.arange(H_p).repeat_interleave(W_p)
        cols = torch.arange(W_p).repeat(H_p)
        layout = torch.stack([rows, cols], dim=1)
        active = torch.ones(N_e, dtype=torch.bool)

        pp = precompute_pool(layout, active, (H_p, W_p), (4, 8))

        assert pp.n_cells == 32
        # 128 electrodes / 32 cells = 4 per cell, uniformly.
        assert torch.equal(pp.active_count, torch.full((32,), 4.0))
        assert torch.all(pp.cell_active_mask)
        assert torch.all(pp.grid_active)

    def test_missing_electrode_excluded_from_count(self) -> None:
        # 128-strip minus one kept electrode at (0, 0).
        H_p, W_p = 8, 16
        rows = torch.arange(H_p).repeat_interleave(W_p)
        cols = torch.arange(W_p).repeat(H_p)
        layout = torch.stack([rows, cols], dim=1)
        keep = torch.ones(H_p * W_p, dtype=torch.bool)
        keep[0] = False  # drop (0,0) entirely — simulates an excluded artifact
        layout = layout[keep]
        active = torch.ones(layout.shape[0], dtype=torch.bool)

        pp = precompute_pool(layout, active, (H_p, W_p), (4, 8))

        # Cell 0 loses one of its 4 slots; others still have 4.
        assert float(pp.active_count[0]) == 3.0
        assert torch.all(pp.active_count[1:] == 4.0)
        # grid_active[flat(0,0)] is False.
        assert not bool(pp.grid_active[0])
        assert bool(pp.grid_active[1])

    def test_inactive_mask_reduces_count(self) -> None:
        # All 128 kept, but one marked inactive.
        H_p, W_p = 8, 16
        rows = torch.arange(H_p).repeat_interleave(W_p)
        cols = torch.arange(W_p).repeat(H_p)
        layout = torch.stack([rows, cols], dim=1)
        active = torch.ones(H_p * W_p, dtype=torch.bool)
        active[0] = False

        pp = precompute_pool(layout, active, (H_p, W_p), (4, 8))

        assert float(pp.active_count[0]) == 3.0
        assert not bool(pp.grid_active[0])


class TestMaskedMeanPool:
    def test_grid_shape_parity_with_adaptive_pool(self) -> None:
        """Full grid, all active, constant-in-time: masked-mean matches AdaptivePool."""
        torch.manual_seed(0)
        B, C, H_p, W_p, T = 2, 8, 8, 16, 11
        N_e = H_p * W_p
        grid = torch.randn(B, C, H_p, W_p, T)

        rows = torch.arange(H_p).repeat_interleave(W_p)
        cols = torch.arange(W_p).repeat(H_p)
        layout = torch.stack([rows, cols], dim=1)
        active = torch.ones(N_e, dtype=torch.bool)
        pp = precompute_pool(layout, active, (H_p, W_p), (4, 8))

        # Pool our way: reshape to (B, C, N_e, T), scatter-mean by cell_of_grid.
        flat = grid.view(B, C, H_p * W_p, T)
        ours = masked_mean_pool(
            flat,
            pp.cell_of_grid,
            pp.grid_active,  # all True here
            pp.active_count,
            dim=2,
            n_cells=pp.n_cells,
        )  # (B, C, 32, T)
        ours_spatial = ours.view(B, C, 4, 8, T)

        # Reference: 2D adaptive avg pool applied per time step.
        ref = torch.nn.functional.adaptive_avg_pool2d(
            grid.permute(0, 1, 4, 2, 3).reshape(B * C * T, 1, H_p, W_p),
            (4, 8),
        ).view(B, C, T, 4, 8).permute(0, 1, 3, 4, 2)

        assert torch.allclose(ours_spatial, ref, atol=1e-6)

    def test_masked_numerator_and_denominator(self) -> None:
        """Inactive electrode must not contribute to its cell's mean, and the
        denominator must drop by one in that cell."""
        B, C, H_p, W_p, T = 1, 1, 8, 16, 1
        grid = torch.zeros(B, C, H_p, W_p, T)
        # Put value 100 at (0, 0). If pooled including it, cell 0 mean = 100/4 = 25.
        grid[0, 0, 0, 0, 0] = 100.0

        rows = torch.arange(H_p).repeat_interleave(W_p)
        cols = torch.arange(W_p).repeat(H_p)
        layout = torch.stack([rows, cols], dim=1)
        active = torch.ones(H_p * W_p, dtype=torch.bool)
        active[0] = False  # mask (0, 0)
        pp = precompute_pool(layout, active, (H_p, W_p), (4, 8))

        # Zero out the grid at the inactive position too — mirrors what a real
        # Conv2d output after masked input would look like only if we pre-zero.
        # The pool must protect us even if we don't.
        flat = grid.view(B, C, H_p * W_p, T)
        out = masked_mean_pool(
            flat,
            pp.cell_of_grid,
            pp.grid_active,
            pp.active_count,
            dim=2,
            n_cells=pp.n_cells,
        )

        # Cell 0 should see 0: only the inactive (0,0) carried the mass.
        assert float(out[0, 0, 0, 0]) == pytest.approx(0.0)
        # Denominator dropped to 3: if we also set an active value in cell 0,
        # it should divide by 3, not 4.
        grid2 = torch.zeros(B, C, H_p, W_p, T)
        grid2[0, 0, 0, 1, 0] = 30.0  # active position (0,1) → cell 0
        out2 = masked_mean_pool(
            grid2.view(B, C, H_p * W_p, T),
            pp.cell_of_grid,
            pp.grid_active,
            pp.active_count,
            dim=2,
            n_cells=pp.n_cells,
        )
        assert float(out2[0, 0, 0, 0]) == pytest.approx(30.0 / 3.0)

    def test_support_pool_shape(self) -> None:
        """Per-electrode support (B, N_e, 15) → (B, n_cells, 15) via dim=1."""
        B, N_e, P = 2, 128, 15
        support = torch.rand(B, N_e, P)

        H_p, W_p = 8, 16
        rows = torch.arange(H_p).repeat_interleave(W_p)
        cols = torch.arange(W_p).repeat(H_p)
        layout = torch.stack([rows, cols], dim=1)
        active = torch.ones(N_e, dtype=torch.bool)
        pp = precompute_pool(layout, active, (H_p, W_p), (4, 8))

        out = masked_mean_pool(
            support,
            pp.cell_of_electrode,
            active,
            pp.active_count,
            dim=1,
            n_cells=pp.n_cells,
        )
        assert out.shape == (B, 32, P)

    def test_empty_cell_returns_zero(self) -> None:
        """A pool cell with zero active electrodes returns 0 without NaN."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # (1, 4)
        cell_of = torch.tensor([0, 0, 0, 0], dtype=torch.long)  # all into cell 0
        active = torch.tensor([True, True, True, True])
        active_count = torch.tensor([4.0, 0.0])  # cell 1 is empty

        out = masked_mean_pool(x, cell_of, active, active_count, dim=1, n_cells=2)
        assert out.shape == (1, 2)
        assert float(out[0, 0]) == pytest.approx(10.0 / 4.0)
        assert float(out[0, 1]) == pytest.approx(0.0)  # no NaN
        assert torch.isfinite(out).all()
