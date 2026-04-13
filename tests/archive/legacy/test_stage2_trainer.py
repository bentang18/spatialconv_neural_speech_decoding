"""Tests for Stage 2 neural adaptation training loop."""
import pytest
import torch
import numpy as np
from pathlib import Path

from speech_decoding.pretraining.stage2_trainer import (
    Stage2Trainer,
    Stage2Config,
)
from speech_decoding.pretraining.pretrain_model import PretrainModel


class TestStage2Trainer:
    @pytest.fixture
    def model_config(self):
        return {
            "spatial_mode": "collapse",
            "d": 64,
            "gru_hidden": 32,
            "gru_layers": 2,
            "temporal_stride": 10,
            "mask_ratio": [0.4, 0.6],
            "mask_spans": [3, 6],
            "spatial_conv": {"channels": 8, "pool_h": 4, "pool_w": 8},
        }

    @pytest.fixture
    def synthetic_trials(self):
        """Synthetic trial data mimicking real patient data."""
        patients = {}
        for pid in ["S_A", "S_B", "S_C"]:
            grids = np.random.randn(50, 8, 16, 300).astype(np.float32)
            patients[pid] = torch.tensor(grids)
        return patients

    def test_trainer_runs_one_step(self, model_config, synthetic_trials):
        model = PretrainModel(model_config, grid_shape=(8, 16))
        cfg = Stage2Config(lr=1e-3, steps=1, batch_size=4)
        trainer = Stage2Trainer(model, cfg, device="cpu")
        metrics = trainer.train_step(synthetic_trials)
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_loss_decreases_over_steps(self, model_config, synthetic_trials):
        model = PretrainModel(model_config, grid_shape=(8, 16))
        cfg = Stage2Config(lr=1e-3, steps=20, batch_size=4)
        trainer = Stage2Trainer(model, cfg, device="cpu")
        losses = []
        for _ in range(20):
            metrics = trainer.train_step(synthetic_trials)
            losses.append(metrics["loss"])
        assert np.mean(losses[-5:]) < np.mean(losses[:5])

    def test_checkpoint_saving(self, model_config, synthetic_trials, tmp_path):
        model = PretrainModel(model_config, grid_shape=(8, 16))
        cfg = Stage2Config(lr=1e-3, steps=5, batch_size=4,
                           checkpoint_dir=str(tmp_path), checkpoint_every=2)
        trainer = Stage2Trainer(model, cfg, device="cpu")
        trainer.train(synthetic_trials)
        checkpoints = list(tmp_path.glob("*.pt"))
        assert len(checkpoints) >= 2

    def test_excludes_dev_and_target(self, model_config, synthetic_trials):
        model = PretrainModel(model_config, grid_shape=(8, 16))
        cfg = Stage2Config(lr=1e-3, steps=1, batch_size=4)
        trainer = Stage2Trainer(model, cfg, device="cpu")
        filtered = trainer._filter_patients(synthetic_trials, exclude={"S_A", "S_C"})
        assert "S_B" in filtered
        assert "S_A" not in filtered
        assert "S_C" not in filtered
