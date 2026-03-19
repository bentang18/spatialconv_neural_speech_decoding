"""Tests for LOPO orchestration."""
import numpy as np

from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.training.lopo import run_lopo


def _make_ds(pid, grid=(8, 16), n=30):
    H, W = grid
    np.random.seed(int(pid.replace("P", "")) + 42)
    data = np.random.randn(n, H, W, 100).astype(np.float32)
    labels = [[((i * 3 + j) % 9) + 1 for j in range(3)] for i in range(n)]
    return BIDSDataset(data, labels, pid, grid)


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


class TestRunLopo:
    def test_returns_results_for_all_patients(self):
        datasets = {"P1": _make_ds("P1"), "P2": _make_ds("P2"), "P3": _make_ds("P3")}
        results = run_lopo(datasets, _make_config(), seeds=[42], device="cpu")

        assert "per_patient" in results
        for pid in ["P1", "P2", "P3"]:
            assert pid in results["per_patient"]

    def test_population_stats(self):
        datasets = {"P1": _make_ds("P1"), "P2": _make_ds("P2"), "P3": _make_ds("P3")}
        results = run_lopo(datasets, _make_config(), seeds=[42], device="cpu")
        assert "population_per_mean" in results
        assert "population_per_std" in results

    def test_wilcoxon_with_baseline(self):
        datasets = {"P1": _make_ds("P1"), "P2": _make_ds("P2"), "P3": _make_ds("P3")}
        baseline_pers = {"P1": 1.0, "P2": 1.0, "P3": 1.0}
        results = run_lopo(
            datasets,
            _make_config(),
            seeds=[42],
            device="cpu",
            baseline_pers=baseline_pers,
        )
        assert "wilcoxon_stat" in results
        assert "wilcoxon_p" in results
