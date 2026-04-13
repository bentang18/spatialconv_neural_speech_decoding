"""Tests for MFA-guided per-position CE loss and trainer."""
import numpy as np
import pytest
import torch

from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.training.ctc_utils import (
    mfa_guided_ce_decode,
    mfa_guided_ce_loss,
)
from speech_decoding.training.mfa_guided_trainer import train_per_patient_mfa_guided


class TestMFAGuidedCELoss:
    """Test CE loss with MFA-derived per-position temporal windows."""

    def test_output_is_scalar(self):
        """Loss should be a scalar tensor."""
        B, T, n_pos, n_phon = 4, 30, 3, 9
        logits = torch.randn(B, T, n_phon)
        targets = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 3, 5]]
        # (B, n_pos, T) segment masks
        seg_masks = torch.zeros(B, n_pos, T)
        seg_masks[:, 0, 5:10] = 1.0
        seg_masks[:, 1, 10:18] = 1.0
        seg_masks[:, 2, 18:25] = 1.0

        loss = mfa_guided_ce_loss(logits, targets, seg_masks)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_gradient_flows(self):
        """Backward pass should produce gradients."""
        B, T, n_phon = 4, 30, 9
        logits = torch.randn(B, T, n_phon, requires_grad=True)
        targets = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 3, 5]]
        seg_masks = torch.zeros(B, 3, T)
        seg_masks[:, 0, 5:10] = 1.0
        seg_masks[:, 1, 10:18] = 1.0
        seg_masks[:, 2, 18:25] = 1.0

        loss = mfa_guided_ce_loss(logits, targets, seg_masks)
        loss.backward()
        assert logits.grad is not None
        # Gradients should be nonzero in phoneme windows, zero outside
        assert logits.grad[:, 5:10, :].abs().sum() > 0
        assert logits.grad[:, 0:5, :].abs().sum() == 0.0

    def test_per_trial_masks(self):
        """Different trials can have different phoneme windows."""
        B, T, n_phon = 2, 30, 9
        logits = torch.randn(B, T, n_phon)
        targets = [[1, 2, 3], [4, 5, 6]]
        seg_masks = torch.zeros(B, 3, T)
        # Trial 0: phonemes at [5:10], [10:15], [15:20]
        seg_masks[0, 0, 5:10] = 1.0
        seg_masks[0, 1, 10:15] = 1.0
        seg_masks[0, 2, 15:20] = 1.0
        # Trial 1: phonemes at [8:14], [14:20], [20:28]
        seg_masks[1, 0, 8:14] = 1.0
        seg_masks[1, 1, 14:20] = 1.0
        seg_masks[1, 2, 20:28] = 1.0

        loss = mfa_guided_ce_loss(logits, targets, seg_masks)
        assert torch.isfinite(loss)

    def test_empty_mask_handled(self):
        """If a segment mask is all zeros, loss should still be finite."""
        B, T, n_phon = 2, 30, 9
        logits = torch.randn(B, T, n_phon)
        targets = [[1, 2, 3], [4, 5, 6]]
        seg_masks = torch.zeros(B, 3, T)
        # Only position 0 has a mask
        seg_masks[:, 0, 5:10] = 1.0

        loss = mfa_guided_ce_loss(logits, targets, seg_masks)
        assert torch.isfinite(loss)


class TestMFAGuidedCEDecode:
    """Test decoding with MFA-guided per-position pooling."""

    def test_decode_shape(self):
        """Should return one prediction per trial, 3 phonemes each."""
        B, T, n_phon = 4, 30, 9
        logits = torch.randn(B, T, n_phon)
        seg_masks = torch.zeros(B, 3, T)
        seg_masks[:, 0, 5:10] = 1.0
        seg_masks[:, 1, 10:18] = 1.0
        seg_masks[:, 2, 18:25] = 1.0

        decoded = mfa_guided_ce_decode(logits, seg_masks)
        assert len(decoded) == B
        assert all(len(seq) == 3 for seq in decoded)

    def test_decode_picks_correct_class(self):
        """With strong signal in the right window, should decode correctly."""
        T, n_phon = 30, 9
        logits = torch.zeros(1, T, n_phon)
        # Position 0 (frames 5-10): phoneme 3 (0-indexed 2) has highest logit
        logits[0, 5:10, 2] = 10.0
        # Position 1 (frames 10-18): phoneme 7 (0-indexed 6)
        logits[0, 10:18, 6] = 10.0
        # Position 2 (frames 18-25): phoneme 1 (0-indexed 0)
        logits[0, 18:25, 0] = 10.0

        seg_masks = torch.zeros(1, 3, T)
        seg_masks[0, 0, 5:10] = 1.0
        seg_masks[0, 1, 10:18] = 1.0
        seg_masks[0, 2, 18:25] = 1.0

        decoded = mfa_guided_ce_decode(logits, seg_masks)
        assert decoded == [[3, 7, 1]]  # 1-indexed


class TestMFAGuidedTrainer:
    """Test the MFA-guided trainer end-to-end on synthetic data."""

    def test_returns_metrics(self):
        """Trainer should return dict with per, bal_acc keys."""
        np.random.seed(42)
        n_trials = 20
        grid_h, grid_w, T = 8, 16, 100
        n_backbone_frames = T // 10  # stride=10 → 10 frames
        n_positions = 3

        grid_data = np.random.randn(n_trials, grid_h, grid_w, T).astype(np.float32)
        labels = [
            [1 + i % 9, 1 + (i + 1) % 9, 1 + (i + 2) % 9]
            for i in range(n_trials)
        ]
        ds = BIDSDataset(grid_data, labels, "S_test", (grid_h, grid_w))

        # Segment masks: phonemes at frames [2:4], [4:7], [7:9]
        seg_masks = np.zeros(
            (n_trials, n_positions, n_backbone_frames), dtype=np.float32
        )
        seg_masks[:, 0, 2:4] = 1.0
        seg_masks[:, 1, 4:7] = 1.0
        seg_masks[:, 2, 7:9] = 1.0

        config = {
            "model": {
                "readin_type": "spatial_conv",
                "head_type": "flat",
                "d_shared": 64,
                "hidden_size": 32,
                "gru_layers": 2,
                "gru_dropout": 0.2,
                "temporal_stride": 10,
                "num_classes": 10,
                "blank_bias": 0.0,
                "spatial_conv": {
                    "channels": 8, "num_layers": 1,
                    "kernel_size": 3, "pool_h": 2, "pool_w": 4,
                },
            },
            "training": {
                "loss_type": "ce",
                "ce_segments": 3,
                "phonological_aux_lambda": 0.0,
                "stage1": {
                    "epochs": 10, "lr": 1e-3, "readin_lr_mult": 3.0,
                    "weight_decay": 1e-4, "batch_size": 8, "grad_clip": 5.0,
                    "patience": 5, "eval_every": 5, "warmup_epochs": 0,
                },
                "augmentation": {
                    "time_shift_frames": 0, "amp_scale_std": 0.0,
                    "channel_dropout_max": 0.0, "noise_frac": 0.0,
                    "feat_dropout_max": 0.0, "time_mask_min": 2, "time_mask_max": 4,
                },
            },
            "evaluation": {"seeds": [42], "cv_folds": 2, "primary_metric": "per"},
        }

        result = train_per_patient_mfa_guided(
            ds, seg_masks, config, seed=42, device="cpu"
        )
        assert "per_mean" in result
        assert np.isfinite(result["per_mean"])
