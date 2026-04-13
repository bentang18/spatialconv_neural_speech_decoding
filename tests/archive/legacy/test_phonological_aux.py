"""Tests for phonological auxiliary targets and trainer."""
from __future__ import annotations

import numpy as np

from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.training.phonological_aux import (
    build_feature_targets,
    per_position_feature_bce_loss,
)
from speech_decoding.training.phonological_aux_trainer import train_per_patient_phonological_aux


def _make_dataset(n_trials=30, grid_h=8, grid_w=16, n_times=300):
    rng = np.random.default_rng(42)
    grid_data = rng.normal(size=(n_trials, grid_h, grid_w, n_times)).astype(np.float32)
    labels = []
    for i in range(n_trials):
        seq = [1 + (i % 3), 4 + (i % 3), 7 + (i % 3)]
        labels.append(seq)
        grid_data[i, seq[0] % grid_h, seq[1] % grid_w, 50:110] += 2.0
        grid_data[i, seq[2] % grid_h, (seq[0] + seq[1]) % grid_w, 140:220] += 1.5
    return BIDSDataset(grid_data=grid_data, ctc_labels=labels, patient_id="S_aux", grid_shape=(grid_h, grid_w))


def _config():
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
            "phonological_aux_lambda": 0.3,
            "phonological_num_features": 15,
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
        "evaluation": {"seeds": [42], "cv_folds": 2, "primary_metric": "per"},
    }


def test_feature_targets_shape():
    target = build_feature_targets([[1, 5, 9], [2, 6, 8]])
    assert target.shape == (2, 3, 15)


def test_feature_loss_finite():
    import torch

    logits = torch.randn(4, 10, 45)
    labels = [[1, 5, 9], [2, 6, 8], [3, 7, 4], [4, 8, 5]]
    loss = per_position_feature_bce_loss(logits, labels)
    assert torch.isfinite(loss)


def test_aux_trainer_runs():
    ds = _make_dataset()
    result = train_per_patient_phonological_aux(ds, _config(), seed=42, device="cpu")
    assert "per_mean" in result
    assert "bal_acc_mean_mean" in result
    assert "feature_acc_mean" in result
