"""Tests for Phase 0 baseline evaluation pipeline."""
import pytest
import torch
import numpy as np

from speech_decoding.models.spatial_conv import SpatialConvReadIn
from speech_decoding.models.backbone import SharedBackbone


class TestCEMeanPoolBaseline:
    """Validate the 27-way CE mean-pool readout pipeline."""

    def test_mean_pool_head_shape(self):
        """Temporal mean pool → Linear(2H, 27) → (B, 27)."""
        head = torch.nn.Linear(128, 27)  # 2H=128 for H=64
        features = torch.randn(4, 30, 128)  # (B, T, 2H)
        pooled = features.mean(dim=1)  # (B, 2H)
        logits = head(pooled)  # (B, 27)
        assert logits.shape == (4, 27)

    def test_27way_to_per_position_decode(self):
        """27-way logits → 3 positions × 9 phonemes → PER."""
        logits = torch.randn(4, 27)
        per_pos = logits.view(4, 3, 9)
        preds = per_pos.argmax(dim=-1)  # (B, 3) in [0, 8]
        assert preds.shape == (4, 3)
        assert (preds >= 0).all() and (preds < 9).all()

    def test_spatial_only_baseline_no_gru(self):
        """Spatial-only: Conv2d → temporal mean pool → CE head. No BiGRU."""
        readin = SpatialConvReadIn(grid_h=8, grid_w=16, pool_h=4, pool_w=8)
        head = torch.nn.Linear(readin.out_dim, 27)
        x = torch.randn(4, 8, 16, 60)  # (B, H, W, T) at 200Hz
        spatial_feats = readin(x)  # (B, D, T)
        pooled = spatial_feats.mean(dim=2)  # (B, D) temporal mean pool
        logits = head(pooled)  # (B, 27)
        assert logits.shape == (4, 27)

    def test_method_e_frozen_forward(self):
        """Method E: frozen random init backbone → train only CE head."""
        backbone = SharedBackbone(D=64, H=64, temporal_stride=10)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        head = torch.nn.Linear(128, 27)
        trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
        assert trainable == 128 * 27 + 27  # 3,483
        assert frozen > 0

    def test_method_d_end_to_end(self):
        """Method D: all params trainable, end-to-end CE."""
        readin = SpatialConvReadIn(grid_h=8, grid_w=16)
        backbone = SharedBackbone(D=64, H=64, temporal_stride=10)
        head = torch.nn.Linear(128, 27)
        x = torch.randn(2, 8, 16, 300)
        shared = readin(x)
        h = backbone(shared)  # (B, T//10, 128)
        pooled = h.mean(dim=1)
        logits = head(pooled)
        loss = torch.nn.functional.cross_entropy(
            logits, torch.randint(0, 27, (2,))
        )
        loss.backward()
        assert readin.convs[0].weight.grad is not None
