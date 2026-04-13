"""Synthetic data pipeline: generator → augmentation → batch.

Wraps generators with nuisance augmentation matching real data statistics.
Ref: spec §4.1 nuisance realism, RD-25 (real dead templates), RD-81 (noise calibration).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from speech_decoding.pretraining.generators import GENERATORS

DEAD_TEMPLATES = {
    (12, 22): [
        (0, 0), (0, 21), (0, 1), (0, 20),
        (11, 0), (11, 21), (11, 1), (11, 20),
    ],
    (8, 16): [],
    (8, 32): [],
    (8, 34): [(r, c) for r in range(8) for c in [0, 33]],
}


@dataclass
class SyntheticConfig:
    generator: str = "smooth_ar"
    generator_kwargs: dict = field(default_factory=dict)
    grid_shapes: list[tuple[int, int]] = field(default_factory=lambda: [(8, 16)])
    iid_noise_range: tuple[float, float] = (0.3, 0.8)
    apply_dead_mask: bool = True
    flip_prob: float = 0.5
    rotate180_prob: float = 0.5


class SyntheticDataPipeline:
    def __init__(self, config: SyntheticConfig):
        self.config = config

    def generate_batch(self, batch_size, T=30, seed=None):
        rng = np.random.RandomState(seed)
        samples = []

        for i in range(batch_size):
            shape_idx = rng.randint(len(self.config.grid_shapes))
            grid_h, grid_w = self.config.grid_shapes[shape_idx]

            gen_cls = GENERATORS[self.config.generator]
            gen = gen_cls(grid_h=grid_h, grid_w=grid_w, T=T, **self.config.generator_kwargs)
            data = gen.generate(seed=rng.randint(2**31))

            # Z-score per trial
            std = data.std()
            if std > 1e-8:
                data = (data - data.mean()) / std

            # IID noise
            lo, hi = self.config.iid_noise_range
            if hi > 0:
                sigma = rng.uniform(lo, hi)
                data = data + rng.randn(*data.shape).astype(np.float32) * sigma

            # Dead electrode mask
            if self.config.apply_dead_mask:
                template = DEAD_TEMPLATES.get((grid_h, grid_w), [])
                for r, c in template:
                    if r < grid_h and c < grid_w:
                        data[r, c, :] = 0.0

            # Flips and 180° rotation
            if rng.random() < self.config.flip_prob:
                data = data[::-1, :, :].copy()
            if rng.random() < self.config.flip_prob:
                data = data[:, ::-1, :].copy()
            if rng.random() < self.config.rotate180_prob:
                data = data[::-1, ::-1, :].copy()

            samples.append(data)

        # Pad to largest grid in batch
        max_h = max(s.shape[0] for s in samples)
        max_w = max(s.shape[1] for s in samples)
        padded = np.zeros((batch_size, max_h, max_w, T), dtype=np.float32)
        for i, s in enumerate(samples):
            padded[i, :s.shape[0], :s.shape[1], :] = s

        return torch.tensor(padded)
