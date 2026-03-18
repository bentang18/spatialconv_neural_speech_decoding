"""Tests for per-patient trainer."""
import numpy as np
import pytest
import torch

from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.training.trainer import train_per_patient
from speech_decoding.evaluation.metrics import (
    evaluate_predictions,
    per_position_balanced_accuracy,
    ctc_length_accuracy,
)


def _make_synthetic_dataset(n_trials=50, grid_h=8, grid_w=16, T=100):
    """Create a synthetic dataset for quick training tests."""
    np.random.seed(42)
    grid_data = np.random.randn(n_trials, grid_h, grid_w, T).astype(np.float32)
    # 9 phonemes (1-9), random 3-phoneme sequences
    ctc_labels = [
        [np.random.randint(1, 10) for _ in range(3)]
        for _ in range(n_trials)
    ]
    return BIDSDataset(
        grid_data=grid_data,
        ctc_labels=ctc_labels,
        patient_id="S_test",
        grid_shape=(grid_h, grid_w),
    )


def _quick_config():
    """Minimal config for fast training."""
    return {
        "model": {
            "readin_type": "spatial_conv",
            "head_type": "articulatory",
            "d_shared": 64,
            "hidden_size": 64,
            "gru_layers": 2,
            "gru_dropout": 0.2,
            "temporal_stride": 5,
            "num_classes": 10,
            "spatial_conv": {
                "channels": 8,
                "num_layers": 1,
                "kernel_size": 3,
                "pool_h": 2,
                "pool_w": 4,
            },
        },
        "training": {
            "stage1": {
                "epochs": 30,
                "lr": 1e-3,
                "readin_lr_mult": 3.0,
                "weight_decay": 1e-4,
                "batch_size": 16,
                "grad_clip": 5.0,
                "patience": 3,
                "eval_every": 10,
                "val_fraction": 0.2,
            },
            "augmentation": {
                "time_shift_frames": 5,
                "amp_scale_std": 0.1,
                "channel_dropout_max": 0.1,
                "noise_frac": 0.01,
                "feat_dropout_max": 0.1,
                "time_mask_min": 2,
                "time_mask_max": 4,
            },
            "stage2": {
                "epochs": 50,
                "lr": 1e-3,
                "weight_decay": 1e-3,
                "source_replay_frac": 0.3,
                "patience": 10,
            },
        },
        "evaluation": {
            "seeds": [42],
            "cv_folds": 3,
            "primary_metric": "per",
        },
    }


class TestMetrics:
    def test_evaluate_predictions(self):
        preds = [[1, 2, 3], [1, 2, 4], [1, 2, 3]]
        tgts = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        result = evaluate_predictions(preds, tgts)
        assert "per" in result
        assert "bal_acc_mean" in result
        assert "length_accuracy" in result
        assert result["per"] >= 0
        assert result["length_accuracy"] == 1.0  # all correct length

    def test_per_position_accuracy(self):
        # Need multiple classes per position for balanced_accuracy_score
        preds = [[1, 2, 3], [4, 5, 6], [1, 2, 3]]
        tgts = [[1, 2, 3], [4, 5, 6], [1, 2, 3]]
        accs = per_position_balanced_accuracy(preds, tgts)
        assert all(a == 1.0 for a in accs)

    def test_ctc_length_accuracy(self):
        preds = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
        assert ctc_length_accuracy(preds, target_length=3) == pytest.approx(1 / 3)


class TestPerPatientTrainer:
    def test_runs_and_returns_metrics(self):
        """Trainer runs without errors on synthetic data."""
        ds = _make_synthetic_dataset(n_trials=30, T=50)
        config = _quick_config()
        config["evaluation"]["cv_folds"] = 2  # minimal

        result = train_per_patient(ds, config, seed=42, device="cpu")
        assert "per_mean" in result
        assert "per_std" in result
        assert "fold_results" in result
        assert result["patient_id"] == "S_test"

    def test_loss_is_finite(self):
        """All fold losses should be finite (no NaN/Inf from CTC)."""
        ds = _make_synthetic_dataset(n_trials=30, T=50)
        config = _quick_config()
        config["evaluation"]["cv_folds"] = 2

        result = train_per_patient(ds, config, seed=42, device="cpu")
        for fold in result["fold_results"]:
            assert np.isfinite(fold["per"])

    def test_per_is_valid(self):
        """PER should be between 0 and some reasonable upper bound."""
        ds = _make_synthetic_dataset(n_trials=30, T=50)
        config = _quick_config()
        config["evaluation"]["cv_folds"] = 2

        result = train_per_patient(ds, config, seed=42, device="cpu")
        assert 0.0 <= result["per_mean"] <= 2.0  # PER > 1.0 is possible (insertions)
