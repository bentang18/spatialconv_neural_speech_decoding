"""Level 0: Spatially-smoothed Gaussian AR process.

AR(1): x_{t+1} = alpha * x_t + (1-alpha) * smooth(noise)
Tests whether ANY smooth synthetic movie helps, regardless of structure.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from speech_decoding.pretraining.generators.base import Generator


class SmoothARGenerator(Generator):
    def __init__(self, grid_h=8, grid_w=16, T=30, alpha=0.9, sigma=3.0):
        super().__init__(grid_h, grid_w, T)
        self.alpha = alpha
        self.sigma = sigma

    def generate(self, seed=None):
        rng = np.random.RandomState(seed)
        frames = np.zeros((self.grid_h, self.grid_w, self.T), dtype=np.float32)
        noise = rng.randn(self.grid_h, self.grid_w).astype(np.float32)
        frames[:, :, 0] = gaussian_filter(noise, sigma=self.sigma)
        for t in range(1, self.T):
            innovation = rng.randn(self.grid_h, self.grid_w).astype(np.float32)
            smoothed = gaussian_filter(innovation, sigma=self.sigma)
            frames[:, :, t] = self.alpha * frames[:, :, t-1] + (1-self.alpha) * smoothed
        return frames
