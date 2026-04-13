"""Tests for grid shape inference and channel-to-grid mapping."""
import numpy as np
import pytest
import tempfile
from pathlib import Path

from speech_decoding.data.grid import (
    GridInfo,
    load_grid_mapping,
    channels_to_grid,
)


def _write_electrodes_tsv(path: Path, n_rows: int, n_cols: int,
                           dead_positions: list[tuple[int, int]] | None = None):
    """Write a synthetic electrode TSV for testing."""
    dead = set(dead_positions or [])
    lines = ["name\tx\ty\tz\tsize\n"]
    ch_idx = 1
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) in dead:
                continue
            x = c / n_cols
            y = r / n_rows
            lines.append(f"{ch_idx}\t{x}\t{y}\t0.0\tn/a\n")
            ch_idx += 1
    path.write_text("".join(lines))
    return ch_idx - 1  # n_channels


class TestLoadGridMapping:
    def test_128ch_8x16_no_dead(self, tmp_path):
        tsv = tmp_path / "electrodes.tsv"
        n_ch = _write_electrodes_tsv(tsv, 8, 16)
        assert n_ch == 128

        info = load_grid_mapping(tsv)
        assert info.grid_shape == (8, 16)
        assert len(info.ch_to_pos) == 128
        assert info.dead_mask.sum() == 0  # no dead positions

    def test_256ch_12x22_with_8_dead(self, tmp_path):
        tsv = tmp_path / "electrodes.tsv"
        dead = [(11, 0), (11, 1), (11, 2), (11, 3),
                (11, 18), (11, 19), (11, 20), (11, 21)]
        n_ch = _write_electrodes_tsv(tsv, 12, 22, dead)
        assert n_ch == 256

        info = load_grid_mapping(tsv)
        assert info.grid_shape == (12, 22)
        assert len(info.ch_to_pos) == 256
        assert info.dead_mask.sum() == 8

    def test_256ch_8x32_with_0_dead(self, tmp_path):
        """Some Lexical patients have 8x32=256 grids."""
        tsv = tmp_path / "electrodes.tsv"
        n_ch = _write_electrodes_tsv(tsv, 8, 32)
        assert n_ch == 256

        info = load_grid_mapping(tsv)
        assert info.grid_shape == (8, 32)
        assert len(info.ch_to_pos) == 256
        assert info.dead_mask.sum() == 0

    def test_channel_names_are_strings(self, tmp_path):
        tsv = tmp_path / "electrodes.tsv"
        _write_electrodes_tsv(tsv, 8, 16)
        info = load_grid_mapping(tsv)
        assert all(isinstance(k, str) for k in info.ch_to_pos)

    def test_positions_are_within_bounds(self, tmp_path):
        tsv = tmp_path / "electrodes.tsv"
        _write_electrodes_tsv(tsv, 12, 22,
                              [(11, 0), (11, 1), (11, 2), (11, 3),
                               (11, 18), (11, 19), (11, 20), (11, 21)])
        info = load_grid_mapping(tsv)
        H, W = info.grid_shape
        for ch, (r, c) in info.ch_to_pos.items():
            assert 0 <= r < H, f"ch {ch} row {r} out of bounds"
            assert 0 <= c < W, f"ch {ch} col {c} out of bounds"


class TestChannelsToGrid:
    def test_128ch_reshape(self, tmp_path):
        tsv = tmp_path / "electrodes.tsv"
        _write_electrodes_tsv(tsv, 8, 16)
        info = load_grid_mapping(tsv)

        # Simulate channel data where each channel has a unique value
        n_ch = 128
        T = 10
        data = np.arange(n_ch * T, dtype=np.float32).reshape(n_ch, T)
        ch_names = [str(i + 1) for i in range(n_ch)]

        grid = channels_to_grid(data, ch_names, info)
        assert grid.shape == (8, 16, T)

    def test_256ch_with_dead_positions_zeroed(self, tmp_path):
        tsv = tmp_path / "electrodes.tsv"
        dead = [(11, 0), (11, 1), (11, 2), (11, 3),
                (11, 18), (11, 19), (11, 20), (11, 21)]
        _write_electrodes_tsv(tsv, 12, 22, dead)
        info = load_grid_mapping(tsv)

        n_ch = 256
        T = 10
        data = np.ones((n_ch, T), dtype=np.float32)
        ch_names = [str(i + 1) for i in range(n_ch)]

        grid = channels_to_grid(data, ch_names, info)
        assert grid.shape == (12, 22, T)

        # Dead positions should be zero
        for r, c in dead:
            assert np.all(grid[r, c, :] == 0.0)

        # Live positions should be 1.0
        assert grid[0, 0, 0] == 1.0

    def test_batch_reshape(self, tmp_path):
        """Test with batch dimension: (B, n_ch, T) → (B, H, W, T)."""
        tsv = tmp_path / "electrodes.tsv"
        _write_electrodes_tsv(tsv, 8, 16)
        info = load_grid_mapping(tsv)

        B, n_ch, T = 4, 128, 10
        data = np.ones((B, n_ch, T), dtype=np.float32)
        ch_names = [str(i + 1) for i in range(n_ch)]

        grid = channels_to_grid(data, ch_names, info)
        assert grid.shape == (B, 8, 16, T)


@pytest.mark.slow
class TestRealElectrodeData:
    """Test with actual electrode TSV files."""

    def test_ps_s33_256ch(self):
        tsv = Path(
            "BIDS_1.0_Phoneme_Sequence_uECoG/BIDS_1.0_Phoneme_Sequence_uECoG"
            "/BIDS/sub-S33/ieeg/sub-S33_acq-01_space-ACPC_electrodes.tsv"
        )
        if not tsv.exists():
            pytest.skip("PS BIDS data not available")
        info = load_grid_mapping(tsv)
        assert info.grid_shape == (12, 22)
        assert len(info.ch_to_pos) == 256
        assert info.dead_mask.sum() == 8

    def test_ps_s14_128ch(self):
        tsv = Path(
            "BIDS_1.0_Phoneme_Sequence_uECoG/BIDS_1.0_Phoneme_Sequence_uECoG"
            "/BIDS/sub-S14/ieeg/sub-S14_acq-01_space-ACPC_electrodes.tsv"
        )
        if not tsv.exists():
            pytest.skip("PS BIDS data not available")
        info = load_grid_mapping(tsv)
        assert info.grid_shape == (8, 16)
        # S14 ch 105 has n/a coords → 127 mappable channels, 1 dead position
        assert len(info.ch_to_pos) == 127
        assert info.dead_mask.sum() == 1
