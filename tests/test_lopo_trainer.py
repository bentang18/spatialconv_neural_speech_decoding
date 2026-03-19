"""Tests for Stage 1 LOPO training."""
import numpy as np

from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.training.lopo_trainer import train_stage1


def _make_synthetic_dataset(pid: str, grid_shape: tuple[int, int], n_trials: int = 40) -> BIDSDataset:
    H, W = grid_shape
    T = 100
    np.random.seed(int(pid.replace("P", "").replace("T", "")) + 42)
    data = np.random.randn(n_trials, H, W, T).astype(np.float32)
    labels = [[((i * 3 + j) % 9) + 1 for j in range(3)] for i in range(n_trials)]
    return BIDSDataset(data, labels, pid, grid_shape)


def _make_config() -> dict:
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
                "steps": 5,
                "lr": 1e-3,
                "warmup_epochs": 0,
                "readin_lr_mult": 3.0,
                "weight_decay": 1e-4,
                "batch_size": 8,
                "grad_clip": 5.0,
                "patience": 3,
                "eval_every": 2,
                "val_fraction": 0.2,
                "augmentation": {
                    "time_shift_frames": 0,
                    "amp_scale_std": 0.0,
                    "channel_dropout_max": 0.0,
                    "noise_frac": 0.0,
                    "feat_dropout_max": 0.0,
                    "time_mask_min": 2,
                    "time_mask_max": 4,
                    "spatial_cutout": False,
                    "temporal_stretch": False,
                },
            },
            "augmentation": {
                "time_shift_frames": 0,
                "amp_scale_std": 0.0,
                "channel_dropout_max": 0.0,
                "noise_frac": 0.0,
                "feat_dropout_max": 0.0,
                "time_mask_min": 2,
                "time_mask_max": 4,
            },
        },
        "evaluation": {"seeds": [42], "cv_folds": 5, "primary_metric": "per"},
    }


class TestTrainStage1:
    def test_returns_checkpoint_dict(self):
        sources = {
            "P1": _make_synthetic_dataset("P1", (8, 16)),
            "P2": _make_synthetic_dataset("P2", (8, 16)),
        }
        checkpoint = train_stage1(sources, _make_config(), seed=42, device="cpu")

        assert "backbone" in checkpoint
        assert "head" in checkpoint
        assert "read_ins" in checkpoint
        assert "P1" in checkpoint["read_ins"]
        assert "P2" in checkpoint["read_ins"]

    def test_loss_decreases(self):
        sources = {
            "P1": _make_synthetic_dataset("P1", (8, 16), n_trials=20),
            "P2": _make_synthetic_dataset("P2", (8, 16), n_trials=20),
        }
        config = _make_config()
        config["training"]["stage1"]["steps"] = 10
        config["training"]["stage1"]["eval_every"] = 5
        checkpoint = train_stage1(sources, config, seed=42, device="cpu")

        assert len(checkpoint["train_losses"]) > 1
        assert checkpoint["train_losses"][-1] < checkpoint["train_losses"][0]

    def test_handles_different_grid_shapes(self):
        sources = {
            "P1": _make_synthetic_dataset("P1", (8, 16)),
            "P2": _make_synthetic_dataset("P2", (12, 22)),
        }
        checkpoint = train_stage1(sources, _make_config(), seed=42, device="cpu")
        assert "P1" in checkpoint["read_ins"]
        assert "P2" in checkpoint["read_ins"]

    def test_val_split_applied(self):
        sources = {"P1": _make_synthetic_dataset("P1", (8, 16), n_trials=50)}
        config = _make_config()
        config["training"]["stage1"]["steps"] = 3
        config["training"]["stage1"]["eval_every"] = 1
        checkpoint = train_stage1(sources, config, seed=42, device="cpu")
        assert "val_losses" in checkpoint
        assert len(checkpoint["val_losses"]) > 0

    def test_gradient_accumulation_across_patients(self):
        sources = {
            "P1": _make_synthetic_dataset("P1", (8, 16), n_trials=20),
            "P2": _make_synthetic_dataset("P2", (8, 16), n_trials=20),
            "P3": _make_synthetic_dataset("P3", (8, 16), n_trials=20),
        }
        config = _make_config()
        config["training"]["stage1"]["steps"] = 2
        checkpoint = train_stage1(sources, config, seed=42, device="cpu")
        assert len(checkpoint["read_ins"]) == 3

    def test_low_count_source_uses_unstratified_val_split(self):
        data = np.random.randn(18, 8, 16, 100).astype(np.float32)
        labels = [[1, 1, 1]] + [[((i * 3 + j) % 9) + 1 for j in range(3)] for i in range(1, 18)]
        sources = {"P1": BIDSDataset(data, labels, "P1", (8, 16))}
        config = _make_config()
        config["training"]["stage1"]["steps"] = 2
        config["training"]["stage1"]["eval_every"] = 1
        checkpoint = train_stage1(sources, config, seed=42, device="cpu")
        assert "val_losses" in checkpoint
