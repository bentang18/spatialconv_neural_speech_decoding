"""Level 1: Switching Linear State-Space Field.

Piecewise-linear dynamics: x_{t+1} = K_{s_t} * x_t + b_{s_t} + eps
3-6 regimes per sequence with stable local 3×3 convolutional kernels.
Ref: spec §4.1 Level 1, RD-80.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

from speech_decoding.pretraining.generators.base import Generator


class SwitchingLDSGenerator(Generator):
    def __init__(self, grid_h=8, grid_w=16, T=30, n_regimes=4, noise_std=0.1, damping=0.95):
        super().__init__(grid_h, grid_w, T)
        self.n_regimes = n_regimes
        self.noise_std = noise_std
        self.damping = damping

    def _make_stable_kernel(self, rng):
        kernel = rng.randn(3, 3).astype(np.float32)
        kernel = kernel / (np.abs(kernel).sum() + 1e-6) * self.damping
        return kernel

    def generate(self, seed=None):
        rng = np.random.RandomState(seed)
        kernels = [self._make_stable_kernel(rng) for _ in range(self.n_regimes)]
        biases = [rng.randn(self.grid_h, self.grid_w).astype(np.float32) * 0.05
                  for _ in range(self.n_regimes)]

        regime_schedule = []
        t = 0
        while t < self.T:
            regime = rng.randint(self.n_regimes)
            dwell = rng.randint(5, max(6, self.T // self.n_regimes + 1))
            regime_schedule.extend([regime] * min(dwell, self.T - t))
            t += dwell

        frames = np.zeros((self.grid_h, self.grid_w, self.T), dtype=np.float32)
        frames[:, :, 0] = rng.randn(self.grid_h, self.grid_w).astype(np.float32) * 0.5

        for t in range(1, self.T):
            regime = regime_schedule[t]
            k = kernels[regime]
            b = biases[regime]
            eps = rng.randn(self.grid_h, self.grid_w).astype(np.float32) * self.noise_std
            frames[:, :, t] = convolve(frames[:, :, t-1], k, mode="constant") + b + eps

        return frames
