"""Tests for regression training utilities."""
from __future__ import annotations

import numpy as np
import torch

from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.evaluation.metrics import framewise_r2_diagnostics
from speech_decoding.evaluation.metrics import segment_r2_diagnostics
from speech_decoding.models.regression_head import RegressionHead
from speech_decoding.training.regression_loss import masked_mse_loss, segment_mse_loss
from speech_decoding.training.regression_trainer import train_per_patient_regression


def _make_synthetic_regression_data(n_trials=30, grid_h=8, grid_w=16, n_times=500, n_frames=50, d_raw=96):
    rng = np.random.default_rng(42)
    grid_data = rng.normal(size=(n_trials, grid_h, grid_w, n_times)).astype(np.float32)
    labels = []
    embeddings = np.zeros((n_trials, n_frames, d_raw), dtype=np.float32)
    speech_mask = np.zeros((n_trials, n_frames), dtype=np.float32)

    for i in range(n_trials):
        seq = [1 + (i % 3), 4 + (i % 3), 7 + (i % 3)]
        labels.append(seq)
        speech_mask[i, 15:30] = 1.0
        class_signal = np.zeros(d_raw, dtype=np.float32)
        class_signal[seq[0] - 1] = 2.0
        class_signal[seq[1] - 1] = 1.5
        class_signal[seq[2] - 1] = 1.0
        embeddings[i, 15:30] = class_signal
        grid_data[i, seq[0] % grid_h, seq[1] % grid_w, 150:300] += 2.0
        grid_data[i, seq[2] % grid_h, (seq[0] + seq[1]) % grid_w, 200:350] += 1.5

    ds = BIDSDataset(grid_data=grid_data, ctc_labels=labels, patient_id="S_test", grid_shape=(grid_h, grid_w))
    return ds, embeddings, speech_mask


def _quick_regression_config():
    return {
        "model": {
            "readin_type": "spatial_conv",
            "head_type": "flat",
            "d_shared": 64,
            "hidden_size": 32,
            "gru_layers": 2,
            "gru_dropout": 0.2,
            "temporal_stride": 10,
            "num_classes": 10,
            "d_emb": 16,
            "spatial_conv": {
                "channels": 8,
                "num_layers": 1,
                "kernel_size": 3,
                "pool_h": 2,
                "pool_w": 4,
            },
        },
        "training": {
            "loss_type": "ce",
            "ce_segments": 3,
            "regression_lambda": 0.3,
            "stage1": {
                "epochs": 20,
                "lr": 1e-3,
                "warmup_epochs": 0,
                "readin_lr_mult": 3.0,
                "weight_decay": 1e-4,
                "batch_size": 8,
                "grad_clip": 5.0,
                "patience": 3,
                "eval_every": 5,
                "val_fraction": 0.2,
            },
            "augmentation": {
                "time_shift_frames": 0,
                "amp_scale_std": 0.0,
                "channel_dropout_max": 0.0,
                "noise_frac": 0.0,
                "feat_dropout_max": 0.0,
                "time_mask_min": 2,
                "time_mask_max": 4,
                "temporal_stretch": False,
            },
        },
        "evaluation": {
            "seeds": [42],
            "cv_folds": 2,
            "primary_metric": "per",
        },
    }


class TestRegressionHead:
    def test_forward_shape(self):
        head = RegressionHead(64, 16)
        x = torch.randn(4, 50, 64)
        y = head(x)
        assert y.shape == (4, 50, 16)


class TestMaskedMSE:
    def test_masked_mse_respects_mask(self):
        pred = torch.tensor([[[1.0], [3.0]]])
        target = torch.tensor([[[0.0], [1.0]]])
        mask = torch.tensor([[1.0, 0.0]])
        loss = masked_mse_loss(pred, target, mask)
        assert loss.item() == 1.0

    def test_segment_mse(self):
        pred = torch.tensor([[[1.0], [3.0], [5.0], [7.0]]])
        target = torch.tensor([[[2.0], [6.0]]])
        seg_mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
        loss = segment_mse_loss(pred, target, seg_mask)
        assert loss.item() == 0.0


class TestDiagnostics:
    def test_framewise_r2_perfect_prediction(self):
        target = np.random.randn(10, 50, 8).astype(np.float32)
        mask = np.ones((10, 50), dtype=np.float32)
        result = framewise_r2_diagnostics(target, target, mask)
        assert result["r2_speech"] == 1.0
        assert result["r2_all"] == 1.0

    def test_segment_r2_perfect_prediction(self):
        target = np.random.randn(10, 3, 8).astype(np.float32)
        result = segment_r2_diagnostics(target, target)
        assert result["r2_segment"] == 1.0


class TestRegressionTrainer:
    def test_runs_and_returns_metrics(self):
        ds, embeddings, speech_mask = _make_synthetic_regression_data()
        result = train_per_patient_regression(
            ds,
            embeddings,
            speech_mask,
            _quick_regression_config(),
            seed=42,
            device="cpu",
        )
        assert "per_mean" in result
        assert "bal_acc_mean_mean" in result
        assert "r2_speech_mean" in result

    def test_ce_only_control_runs(self):
        ds, embeddings, speech_mask = _make_synthetic_regression_data()
        config = _quick_regression_config()
        config["training"]["regression_lambda"] = 0.0
        result = train_per_patient_regression(ds, embeddings, speech_mask, config, seed=42, device="cpu")
        assert "per_mean" in result

    def test_segment_target_mode_runs(self):
        ds, embeddings, speech_mask = _make_synthetic_regression_data()
        config = _quick_regression_config()
        config["training"]["regression_target_mode"] = "segment"
        segment_mask = np.zeros((len(ds), 3, embeddings.shape[1]), dtype=np.float32)
        segment_mask[:, 0, 15:20] = 1.0
        segment_mask[:, 1, 20:25] = 1.0
        segment_mask[:, 2, 25:30] = 1.0
        result = train_per_patient_regression(
            ds,
            embeddings,
            speech_mask,
            config,
            seed=42,
            device="cpu",
            segment_mask=segment_mask,
        )
        assert "r2_segment_mean" in result
