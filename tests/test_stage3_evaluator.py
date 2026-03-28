"""Tests for Stage 3 fine-tuning evaluator."""
import pytest
import torch
import numpy as np

from speech_decoding.pretraining.stage3_evaluator import (
    Stage3Evaluator,
    Stage3Config,
)
from speech_decoding.pretraining.pretrain_model import PretrainModel


class TestStage3Evaluator:
    @pytest.fixture
    def pretrained_model(self):
        config = {
            "spatial_mode": "collapse",
            "d": 64,
            "gru_hidden": 32,
            "gru_layers": 2,
            "temporal_stride": 10,
            "mask_ratio": [0.4, 0.6],
            "mask_spans": [3, 6],
            "spatial_conv": {"channels": 8, "pool_h": 4, "pool_w": 8},
        }
        return PretrainModel(config, grid_shape=(8, 16))

    @pytest.fixture
    def synthetic_dataset(self):
        """Synthetic labeled data for one patient."""
        n_trials = 100
        grids = np.random.randn(n_trials, 8, 16, 300).astype(np.float32)
        labels = [[np.random.randint(1, 10) for _ in range(3)]
                  for _ in range(n_trials)]
        return torch.tensor(grids), labels

    def test_freezes_backbone(self, pretrained_model, synthetic_dataset):
        cfg = Stage3Config(lr=1e-3, epochs=1, n_folds=3)
        evaluator = Stage3Evaluator(pretrained_model, cfg, device="cpu")
        evaluator._freeze_backbone()
        for name, p in pretrained_model.named_parameters():
            if "readin" not in name and "decoder" not in name:
                if "backbone" in name:
                    assert not p.requires_grad, f"{name} should be frozen"

    def test_evaluate_returns_per(self, pretrained_model, synthetic_dataset):
        grids, labels = synthetic_dataset
        cfg = Stage3Config(lr=1e-3, epochs=2, n_folds=3)
        evaluator = Stage3Evaluator(pretrained_model, cfg, device="cpu")
        results = evaluator.evaluate(grids, labels, patient_id="S_test")
        assert "mean_per" in results
        assert 0.0 <= results["mean_per"] <= 1.0
        assert "fold_pers" in results
        assert len(results["fold_pers"]) == 3

    def test_evaluate_returns_collapse_report(self, pretrained_model, synthetic_dataset):
        grids, labels = synthetic_dataset
        cfg = Stage3Config(lr=1e-3, epochs=2, n_folds=3)
        evaluator = Stage3Evaluator(pretrained_model, cfg, device="cpu")
        results = evaluator.evaluate(grids, labels, patient_id="S_test")
        assert "content_collapse" in results
        assert "entropy" in results["content_collapse"]
