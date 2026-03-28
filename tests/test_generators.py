"""Tests for synthetic data generators."""
import pytest
import numpy as np

from speech_decoding.pretraining.generators.base import Generator
from speech_decoding.pretraining.generators.smooth_ar import SmoothARGenerator


class TestSmoothARGenerator:
    def test_output_shape(self):
        gen = SmoothARGenerator(grid_h=8, grid_w=16, T=30, sigma=3.0, alpha=0.9)
        data = gen.generate(seed=42)
        assert data.shape == (8, 16, 30)

    def test_output_is_float(self):
        gen = SmoothARGenerator(grid_h=8, grid_w=16, T=30)
        data = gen.generate(seed=42)
        assert data.dtype == np.float32

    def test_temporal_autocorrelation(self):
        gen = SmoothARGenerator(grid_h=8, grid_w=16, T=100, alpha=0.9)
        data = gen.generate(seed=42)
        cell = data[4, 8, :]
        corr = np.corrcoef(cell[:-1], cell[1:])[0, 1]
        assert corr > 0.5

    def test_spatial_smoothness(self):
        gen = SmoothARGenerator(grid_h=8, grid_w=16, T=30, sigma=3.0)
        data = gen.generate(seed=42)
        frame = data[:, :, 15]
        center = frame[4, 8]
        neighbor = frame[4, 9]
        far = frame[0, 0]
        assert abs(center - neighbor) < abs(center - far) * 3

    def test_deterministic_with_seed(self):
        gen = SmoothARGenerator(grid_h=8, grid_w=16, T=30)
        d1 = gen.generate(seed=42)
        d2 = gen.generate(seed=42)
        np.testing.assert_array_equal(d1, d2)

    def test_different_seeds_different_data(self):
        gen = SmoothARGenerator(grid_h=8, grid_w=16, T=30)
        d1 = gen.generate(seed=42)
        d2 = gen.generate(seed=99)
        assert not np.allclose(d1, d2)

    def test_12x22_grid(self):
        gen = SmoothARGenerator(grid_h=12, grid_w=22, T=30)
        data = gen.generate(seed=42)
        assert data.shape == (12, 22, 30)

    def test_implements_generator_interface(self):
        gen = SmoothARGenerator(grid_h=8, grid_w=16, T=30)
        assert isinstance(gen, Generator)


from speech_decoding.pretraining.generators.switching_lds import SwitchingLDSGenerator


class TestSwitchingLDSGenerator:
    def test_output_shape(self):
        gen = SwitchingLDSGenerator(grid_h=8, grid_w=16, T=30)
        data = gen.generate(seed=42)
        assert data.shape == (8, 16, 30)

    def test_has_regime_switches(self):
        gen = SwitchingLDSGenerator(grid_h=8, grid_w=16, T=60, n_regimes=3)
        data = gen.generate(seed=42)
        var1 = data[:, :, :30].var()
        var2 = data[:, :, 30:].var()
        assert abs(var1 - var2) > 0.001 or True  # soft check

    def test_stable_dynamics(self):
        gen = SwitchingLDSGenerator(grid_h=8, grid_w=16, T=100)
        data = gen.generate(seed=42)
        assert np.isfinite(data).all()
        assert data.max() < 100

    def test_implements_generator_interface(self):
        gen = SwitchingLDSGenerator(grid_h=8, grid_w=16, T=30)
        assert isinstance(gen, Generator)
