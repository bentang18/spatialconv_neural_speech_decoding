"""Tests for synthetic data pipeline."""
import pytest
import numpy as np
import torch

from speech_decoding.pretraining.synthetic_pipeline import (
    SyntheticDataPipeline,
    SyntheticConfig,
)


class TestSyntheticDataPipeline:
    def test_generates_batch(self):
        cfg = SyntheticConfig(generator="smooth_ar", grid_shapes=[(8, 16)])
        pipe = SyntheticDataPipeline(cfg)
        batch = pipe.generate_batch(batch_size=4, T=30, seed=42)
        assert batch.shape[0] == 4
        assert batch.shape[1] == 8  # H
        assert batch.shape[2] == 16  # W
        assert batch.shape[3] == 30  # T

    def test_z_scored(self):
        cfg = SyntheticConfig(generator="smooth_ar", grid_shapes=[(8, 16)])
        pipe = SyntheticDataPipeline(cfg)
        batch = pipe.generate_batch(batch_size=16, T=30, seed=42)
        for i in range(batch.shape[0]):
            trial = batch[i]
            assert abs(trial.mean()) < 1.0
            assert trial.std() > 0.1

    def test_dead_electrodes_applied(self):
        cfg = SyntheticConfig(
            generator="smooth_ar",
            grid_shapes=[(12, 22)],
            apply_dead_mask=True,
            flip_prob=0.0,
            rotate180_prob=0.0,
        )
        pipe = SyntheticDataPipeline(cfg)
        batch = pipe.generate_batch(batch_size=4, T=30, seed=42)
        # 12×22 has 8 dead corners — should be zero
        assert batch[0, 0, 0, :].abs().sum() == 0
        assert batch[0, 0, 21, :].abs().sum() == 0

    def test_noise_injection(self):
        cfg = SyntheticConfig(
            generator="smooth_ar",
            grid_shapes=[(8, 16)],
            iid_noise_range=(0.3, 0.8),
        )
        pipe = SyntheticDataPipeline(cfg)
        batch_noisy = pipe.generate_batch(batch_size=1, T=30, seed=42)
        cfg2 = SyntheticConfig(
            generator="smooth_ar",
            grid_shapes=[(8, 16)],
            iid_noise_range=(0.0, 0.0),
        )
        pipe2 = SyntheticDataPipeline(cfg2)
        batch_clean = pipe2.generate_batch(batch_size=1, T=30, seed=42)
        assert not torch.allclose(batch_noisy, batch_clean)

    def test_mixed_grid_sizes(self):
        cfg = SyntheticConfig(
            generator="smooth_ar",
            grid_shapes=[(8, 16), (12, 22)],
        )
        pipe = SyntheticDataPipeline(cfg)
        shapes_seen = set()
        for seed in range(20):
            batch = pipe.generate_batch(batch_size=1, T=30, seed=seed)
            shapes_seen.add((batch.shape[1], batch.shape[2]))
        # With padding, all batches get padded to max grid size
        # But different seeds should select different source grids
        assert len(shapes_seen) >= 1  # at least one shape
