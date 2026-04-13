"""Tests for Stage 1 synthetic pretraining loop."""
import pytest
import torch
import numpy as np

from speech_decoding.pretraining.stage1_trainer import Stage1Trainer, Stage1Config
from speech_decoding.pretraining.pretrain_model import PretrainModel
from speech_decoding.pretraining.synthetic_pipeline import SyntheticDataPipeline, SyntheticConfig


class TestStage1Trainer:
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

    def test_runs_one_step(self, model_config):
        model = PretrainModel(model_config, grid_shape=(8, 16))
        synth_cfg = SyntheticConfig(generator="smooth_ar", grid_shapes=[(8, 16)])
        pipeline = SyntheticDataPipeline(synth_cfg)
        cfg = Stage1Config(steps=1, batch_size=4, T=200)
        trainer = Stage1Trainer(model, pipeline, cfg, device="cpu")
        metrics = trainer.train_step()
        assert "loss" in metrics

    def test_loss_decreases(self, model_config):
        model = PretrainModel(model_config, grid_shape=(8, 16))
        synth_cfg = SyntheticConfig(generator="smooth_ar", grid_shapes=[(8, 16)])
        pipeline = SyntheticDataPipeline(synth_cfg)
        cfg = Stage1Config(steps=20, batch_size=4, lr=1e-3, T=200)
        trainer = Stage1Trainer(model, pipeline, cfg, device="cpu")
        losses = []
        for _ in range(20):
            m = trainer.train_step()
            losses.append(m["loss"])
        assert np.mean(losses[-5:]) < np.mean(losses[:5])
