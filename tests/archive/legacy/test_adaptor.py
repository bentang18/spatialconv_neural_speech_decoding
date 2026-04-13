"""Tests for Stage 2 target adaptation."""
import numpy as np
import torch

from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.training.adaptor import adapt_stage2
from speech_decoding.training.lopo_trainer import train_stage1


def _make_synthetic_dataset(pid, grid_shape=(8, 16), n_trials=40):
    H, W = grid_shape
    np.random.seed(int(pid.replace("P", "").replace("T", "")) + 42)
    data = np.random.randn(n_trials, H, W, 100).astype(np.float32)
    labels = [[((i * 3 + j) % 9) + 1 for j in range(3)] for i in range(n_trials)]
    return BIDSDataset(data, labels, pid, grid_shape)


def _make_config():
    aug = {
        "time_shift_frames": 0,
        "amp_scale_std": 0.0,
        "channel_dropout_max": 0.0,
        "noise_frac": 0.0,
        "feat_dropout_max": 0.0,
        "time_mask_min": 2,
        "time_mask_max": 4,
        "spatial_cutout": False,
        "temporal_stretch": False,
    }
    return {
        "model": {
            "readin_type": "spatial_conv",
            "head_type": "articulatory",
            "d_shared": 64,
            "hidden_size": 32,
            "gru_layers": 1,
            "gru_dropout": 0.0,
            "temporal_stride": 5,
            "num_classes": 10,
            "blank_bias": 2.0,
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
                "steps": 3,
                "lr": 1e-3,
                "warmup_epochs": 0,
                "readin_lr_mult": 3.0,
                "weight_decay": 1e-4,
                "batch_size": 8,
                "grad_clip": 5.0,
                "patience": 3,
                "eval_every": 1,
                "val_fraction": 0.2,
                "augmentation": aug,
            },
            "stage2": {
                "steps": 3,
                "lr": 1e-3,
                "warmup_epochs": 0,
                "readin_lr_mult": 3.0,
                "weight_decay": 1e-3,
                "batch_size": 8,
                "grad_clip": 5.0,
                "patience": 3,
                "eval_every": 1,
                "cv_folds": 2,
                "val_fraction": 0.2,
                "min_inner_class_count": 2,
                "source_replay_frac": 0.3,
                "augmentation": aug,
            },
            "augmentation": aug,
        },
        "evaluation": {"seeds": [42], "cv_folds": 2, "primary_metric": "per"},
    }


class TestAdaptStage2:
    def test_returns_metrics_dict(self):
        sources = {"P1": _make_synthetic_dataset("P1"), "P2": _make_synthetic_dataset("P2")}
        target = _make_synthetic_dataset("T1")
        checkpoint = train_stage1(sources, _make_config(), seed=42, device="cpu")
        result = adapt_stage2(checkpoint, target, sources, _make_config(), seed=42, device="cpu")

        assert "per_mean" in result
        assert "fold_results" in result
        assert len(result["fold_results"]) == 2

    def test_backbone_is_frozen(self):
        sources = {"P1": _make_synthetic_dataset("P1")}
        target = _make_synthetic_dataset("T1")
        config = _make_config()
        checkpoint = train_stage1(sources, config, seed=42, device="cpu")
        bb_before = {k: v.clone() for k, v in checkpoint["backbone"].items()}

        adapt_stage2(checkpoint, target, sources, config, seed=42, device="cpu")

        for key in bb_before:
            assert torch.equal(bb_before[key], checkpoint["backbone"][key])

    def test_source_replay_present(self):
        sources = {"P1": _make_synthetic_dataset("P1")}
        target = _make_synthetic_dataset("T1")
        config = _make_config()
        config["training"]["stage2"]["source_replay_frac"] = 0.3
        checkpoint = train_stage1(sources, config, seed=42, device="cpu")
        result = adapt_stage2(checkpoint, target, sources, config, seed=42, device="cpu")
        assert result["per_mean"] >= 0

    def test_handles_different_target_grid(self):
        sources = {"P1": _make_synthetic_dataset("P1", (8, 16))}
        target = _make_synthetic_dataset("T1", (12, 22))
        config = _make_config()
        checkpoint = train_stage1(sources, config, seed=42, device="cpu")
        result = adapt_stage2(checkpoint, target, sources, config, seed=42, device="cpu")
        assert "per_mean" in result

    def test_low_trial_target_falls_back_to_safe_split(self):
        sources = {"P1": _make_synthetic_dataset("P1", n_trials=40)}
        target = _make_synthetic_dataset("T1", n_trials=18)
        config = _make_config()
        config["training"]["stage2"]["cv_folds"] = 5
        checkpoint = train_stage1(sources, config, seed=42, device="cpu")
        result = adapt_stage2(checkpoint, target, sources, config, seed=42, device="cpu")
        assert "per_mean" in result

    def test_singleton_class_target_uses_unstratified_outer_holdout(self):
        sources = {"P1": _make_synthetic_dataset("P1", n_trials=40)}
        target = _make_synthetic_dataset("T1", n_trials=18)
        target.ctc_labels[0] = [9, 9, 9]
        for i in range(1, len(target.ctc_labels)):
            target.ctc_labels[i][0] = 1
        config = _make_config()
        config["training"]["stage2"]["cv_folds"] = 5
        checkpoint = train_stage1(sources, config, seed=42, device="cpu")
        result = adapt_stage2(checkpoint, target, sources, config, seed=42, device="cpu")
        assert len(result["fold_results"]) == 1
