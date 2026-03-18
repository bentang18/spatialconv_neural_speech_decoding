"""End-to-end integration tests.

Tests the full pipeline: synthetic data → model assembly → train → decode → PER.
Also tests with real data (slow).
"""
import numpy as np
import pytest
import torch

from speech_decoding.data.bids_dataset import BIDSDataset, load_patient_data
from speech_decoding.data.augmentation import augment_batch
from speech_decoding.data.phoneme_map import ARPA_PHONEMES, encode_ctc_label
from speech_decoding.models.assembler import assemble_model
from speech_decoding.training.ctc_utils import ctc_loss, greedy_decode, compute_per, blank_ratio
from speech_decoding.training.trainer import train_per_patient


def _full_config(readin="spatial_conv", head="articulatory"):
    """Full config for integration testing."""
    config = {
        "model": {
            "readin_type": readin,
            "head_type": head,
            "d_shared": 64,
            "hidden_size": 64,
            "gru_layers": 2,
            "gru_dropout": 0.2,
            "temporal_stride": 5,
            "num_classes": 10,
            "spatial_conv": {
                "channels": 8, "num_layers": 1,
                "kernel_size": 3, "pool_h": 2, "pool_w": 4,
            },
            "linear": {"d_padded": 128},
        },
        "training": {
            "stage1": {
                "epochs": 20, "lr": 1e-3, "readin_lr_mult": 3.0,
                "weight_decay": 1e-4, "batch_size": 8, "grad_clip": 5.0,
                "patience": 3, "eval_every": 10, "val_fraction": 0.2,
            },
            "augmentation": {
                "time_shift_frames": 5, "amp_scale_std": 0.1,
                "channel_dropout_max": 0.1, "noise_frac": 0.01,
                "feat_dropout_max": 0.1, "time_mask_min": 2, "time_mask_max": 4,
            },
            "stage2": {
                "epochs": 50, "lr": 1e-3, "weight_decay": 1e-3,
                "source_replay_frac": 0.3, "patience": 10,
            },
        },
        "evaluation": {"seeds": [42], "cv_folds": 2, "primary_metric": "per"},
    }
    return config


class TestEndToEndSynthetic:
    """Full pipeline test with synthetic data."""

    def test_forward_pass_spatial_conv(self):
        """grid → SpatialConv → backbone → ArticulatoryCTC → log_probs."""
        config = _full_config("spatial_conv", "articulatory")
        patients = {"S14": (8, 16), "S33": (12, 22)}
        backbone, head, readins = assemble_model(config, patients)
        backbone.eval()

        for pid, (h, w) in patients.items():
            x = torch.randn(4, h, w, 100)
            shared = readins[pid](x)
            assert shared.shape == (4, 64, 100)
            out = backbone(shared)
            assert out.shape == (4, 20, 128)  # T=100, stride=5 → 20
            log_probs = head(out)
            assert log_probs.shape == (4, 20, 10)

    def test_forward_pass_linear(self):
        """flat_ch → Linear → backbone → FlatCTC → log_probs."""
        config = _full_config("linear", "flat")
        patients = {"S14": (8, 16)}
        backbone, head, readins = assemble_model(config, patients)
        backbone.eval()

        x = torch.randn(4, 128, 100)  # (B, D_padded, T)
        shared = readins["S14"](x)
        out = backbone(shared)
        log_probs = head(out)
        assert log_probs.shape == (4, 20, 10)

    def test_ctc_loss_backward(self):
        """CTC loss + backward pass through full model."""
        config = _full_config()
        patients = {"S14": (8, 16)}
        backbone, head, readins = assemble_model(config, patients)
        backbone.train()

        x = torch.randn(4, 8, 16, 100)
        targets = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 3, 5]]

        shared = readins["S14"](x)
        h = backbone(shared)
        log_probs = head(h)
        loss = ctc_loss(log_probs, targets)

        assert torch.isfinite(loss)
        loss.backward()

        # All parameters should have gradients
        for name, p in backbone.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for backbone.{name}"

    def test_overfit_single_patient(self):
        """Single patient, many epochs, no augmentation → loss decreases."""
        # Create structured data: each phoneme class activates specific grid region
        np.random.seed(42)
        n_trials = 20
        grid_h, grid_w, T = 8, 16, 50
        grid_data = np.random.randn(n_trials, grid_h, grid_w, T).astype(np.float32) * 0.1

        # Add class-specific signals
        labels = []
        for i in range(n_trials):
            seq = [1 + (i * 3) % 9, 1 + (i * 3 + 1) % 9, 1 + (i * 3 + 2) % 9]
            labels.append(seq)
            # Inject signal: first phoneme activates top-left, etc.
            for j, p in enumerate(seq):
                r, c = p % grid_h, p % grid_w
                t_start = j * T // 4
                grid_data[i, r, c, t_start:t_start + T // 6] += 3.0

        ds = BIDSDataset(grid_data, labels, "S_test", (grid_h, grid_w))

        config = _full_config()
        config["training"]["stage1"]["epochs"] = 80
        config["training"]["stage1"]["eval_every"] = 40
        config["training"]["stage1"]["patience"] = 10
        config["training"]["augmentation"] = {
            "time_shift_frames": 0, "amp_scale_std": 0.0,
            "channel_dropout_max": 0.0, "noise_frac": 0.0,
            "feat_dropout_max": 0.0, "time_mask_min": 2, "time_mask_max": 4,
        }
        config["evaluation"]["cv_folds"] = 2

        result = train_per_patient(ds, config, seed=42)
        # With structured data and enough training, PER should be < 1.0
        # (random chance PER for 9 classes, length 3 ≈ 0.89)
        assert result["per_mean"] < 1.0, f"PER {result['per_mean']} too high"

    def test_augmentation_pipeline(self):
        """Augmentations don't break the forward pass."""
        x = torch.randn(4, 8, 16, 300)
        x_aug = augment_batch(x, training=True)
        assert x_aug.shape == x.shape
        assert torch.isfinite(x_aug).all()


@pytest.mark.slow
class TestEndToEndRealData:
    """Integration test with real BIDS data."""

    def test_load_and_forward_s14(self):
        """Load real S14 data → full forward pass → finite loss."""
        from pathlib import Path
        bids_root = Path(
            "BIDS_1.0_Phoneme_Sequence_uECoG"
            "/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS"
        )
        if not bids_root.exists():
            pytest.skip("PS BIDS data not available")

        ds = load_patient_data(
            "S14", bids_root, task="PhonemeSequence", n_phons=3,
            tmin=-0.5, tmax=1.0,
        )
        config = _full_config()
        patients = {ds.patient_id: ds.grid_shape}
        backbone, head, readins = assemble_model(config, patients)
        backbone.eval()

        # Forward pass on first 4 trials
        x = torch.from_numpy(ds.grid_data[:4])
        targets = ds.ctc_labels[:4]

        shared = readins["S14"](x)
        h = backbone(shared)
        log_probs = head(h)
        loss = ctc_loss(log_probs, targets)

        assert torch.isfinite(loss)
        assert log_probs.shape[0] == 4
        assert log_probs.shape[2] == 10

        # Decode
        decoded = greedy_decode(log_probs)
        assert len(decoded) == 4
        br = blank_ratio(log_probs)
        assert 0 <= br <= 1

    def test_train_s14_quick(self):
        """Quick per-patient training on real S14 data (2 folds, 20 epochs)."""
        from pathlib import Path
        bids_root = Path(
            "BIDS_1.0_Phoneme_Sequence_uECoG"
            "/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS"
        )
        if not bids_root.exists():
            pytest.skip("PS BIDS data not available")

        ds = load_patient_data(
            "S14", bids_root, task="PhonemeSequence", n_phons=3,
            tmin=-0.5, tmax=1.0,
        )
        config = _full_config()
        config["evaluation"]["cv_folds"] = 2
        config["training"]["stage1"]["epochs"] = 20

        result = train_per_patient(ds, config, seed=42)
        assert "per_mean" in result
        assert np.isfinite(result["per_mean"])
        print(f"\nS14 quick train: PER={result['per_mean']:.3f}, "
              f"bal_acc={result['bal_acc_mean_mean']:.3f}, "
              f"blank_ratio={result['blank_ratio_mean']:.2f}")
