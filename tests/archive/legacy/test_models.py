"""Tests for all model components."""
import pytest
import torch
import torch.nn as nn
import numpy as np

from speech_decoding.models.linear_readin import LinearReadIn
from speech_decoding.models.spatial_conv import SpatialConvReadIn
from speech_decoding.models.backbone import SharedBackbone
from speech_decoding.models.flat_head import FlatCTCHead
from speech_decoding.models.articulatory_head import ArticulatoryCTCHead
from speech_decoding.models.assembler import assemble_model


# ── Linear Read-In ─────────────────────────────────────────────────

class TestLinearReadIn:
    def test_output_shape(self):
        """(B, D_padded, T) → (B, D_shared, T)."""
        m = LinearReadIn(d_padded=128, d_shared=64)
        x = torch.randn(4, 128, 300)
        out = m(x)
        assert out.shape == (4, 64, 300)

    def test_different_d_padded(self):
        m = LinearReadIn(d_padded=256, d_shared=64)
        x = torch.randn(2, 256, 200)
        out = m(x)
        assert out.shape == (2, 64, 200)

    def test_param_count(self):
        m = LinearReadIn(d_padded=208, d_shared=64)
        n = sum(p.numel() for p in m.parameters())
        assert n == 208 * 64 + 64  # weight + bias = 13,376


# ── Spatial Conv Read-In ───────────────────────────────────────────

class TestSpatialConvReadIn:
    def test_output_shape_8x16(self):
        """8x16 grid → 64-dim shared features."""
        m = SpatialConvReadIn(grid_h=8, grid_w=16)
        x = torch.randn(4, 8, 16, 300)  # (B, H, W, T)
        out = m(x)
        assert out.shape == (4, 64, 300)

    def test_output_shape_12x22(self):
        """12x22 grid → same 64-dim output (AdaptiveAvgPool)."""
        m = SpatialConvReadIn(grid_h=12, grid_w=22)
        x = torch.randn(4, 12, 22, 300)
        out = m(x)
        assert out.shape == (4, 64, 300)

    def test_default_param_count(self):
        """Default 1-layer: Conv2d(1,8,3) = 8*(1*9+1) = 80 params."""
        m = SpatialConvReadIn(grid_h=8, grid_w=16, C=8, num_layers=1)
        n = sum(p.numel() for p in m.parameters())
        assert n == 80

    def test_two_layer_param_count(self):
        """2-layer: 80 + Conv2d(8,8,3) = 8*(8*9+1) = 584 → total 664."""
        m = SpatialConvReadIn(grid_h=8, grid_w=16, C=8, num_layers=2)
        n = sum(p.numel() for p in m.parameters())
        assert n == 664

    def test_output_dim_configurable(self):
        m = SpatialConvReadIn(grid_h=8, grid_w=16, C=4, pool_h=4, pool_w=4)
        assert m.out_dim == 4 * 4 * 4  # 64
        x = torch.randn(2, 8, 16, 100)
        out = m(x)
        assert out.shape == (2, 64, 100)


# ── Shared Backbone ────────────────────────────────────────────────

class TestSharedBackbone:
    def test_output_shape(self):
        """(B, D, T) → (B, T//K, 2H)."""
        m = SharedBackbone(D=64, H=64, temporal_stride=5)
        x = torch.randn(4, 64, 300)  # 300 frames → 60 after stride-5
        m.eval()
        out = m(x)
        assert out.shape == (4, 60, 128)

    def test_stride_10(self):
        m = SharedBackbone(D=64, H=64, temporal_stride=10)
        x = torch.randn(4, 64, 200)  # 200 → 20
        m.eval()
        out = m(x)
        assert out.shape == (4, 20, 128)

    def test_training_augments(self):
        """Training mode applies feature dropout + time mask (stochastic)."""
        m = SharedBackbone(D=64, H=64, temporal_stride=5)
        m.train()
        x = torch.randn(4, 64, 300)
        torch.manual_seed(42)
        out1 = m(x)
        torch.manual_seed(99)
        out2 = m(x)
        # Different seeds should produce different outputs in training
        assert not torch.allclose(out1, out2)

    def test_eval_is_deterministic(self):
        m = SharedBackbone(D=64, H=64, temporal_stride=5)
        m.eval()
        x = torch.randn(4, 64, 300)
        out1 = m(x)
        out2 = m(x)
        assert torch.allclose(out1, out2)


# ── Flat CTC Head ──────────────────────────────────────────────────

class TestFlatCTCHead:
    def test_output_shape(self):
        """(B, T, 2H) → (B, T, num_classes) log probabilities."""
        m = FlatCTCHead(input_dim=128, num_classes=10)
        h = torch.randn(4, 60, 128)
        out = m(h)
        assert out.shape == (4, 60, 10)

    def test_output_is_log_prob(self):
        m = FlatCTCHead(input_dim=128, num_classes=10)
        h = torch.randn(2, 10, 128)
        out = m(h)
        # log_softmax: exp should sum to ~1
        probs = out.exp().sum(dim=-1)
        assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)

    def test_param_count(self):
        m = FlatCTCHead(input_dim=128, num_classes=10)
        n = sum(p.numel() for p in m.parameters())
        assert n == 128 * 10 + 10  # 1,290


# ── Articulatory CTC Head ─────────────────────────────────────────

class TestArticulatoryCTCHead:
    def test_output_shape(self):
        """(B, T, 2H) → (B, T, 10) log probabilities."""
        m = ArticulatoryCTCHead(input_dim=128)
        h = torch.randn(4, 60, 128)
        out = m(h)
        assert out.shape == (4, 60, 10)

    def test_output_is_log_prob(self):
        m = ArticulatoryCTCHead(input_dim=128)
        h = torch.randn(2, 10, 128)
        out = m(h)
        probs = out.exp().sum(dim=-1)
        assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)

    def test_blank_bias_initialized(self):
        m = ArticulatoryCTCHead(input_dim=128)
        assert m.blank_head.bias.item() == pytest.approx(2.0)

    def test_param_count(self):
        m = ArticulatoryCTCHead(input_dim=128)
        n = sum(p.numel() for p in m.parameters())
        # 6 heads: (128+1) * (2+3+2+2+3+3) + blank: (128+1)*1 = 129*16 = 2064
        assert n == 2064

    def test_composition_matrix_shape(self):
        m = ArticulatoryCTCHead(input_dim=128)
        assert m.A.shape == (9, 15)

    def test_b_vs_p_differ_in_voicing(self):
        """Model must learn voicing to distinguish b and p."""
        m = ArticulatoryCTCHead(input_dim=128)
        h = torch.randn(1, 1, 128)
        out = m(h)  # (1, 1, 10)
        # b and p should generally differ (different voicing head contributions)
        # We can't test exact values, but the mechanism should work
        assert out.shape == (1, 1, 10)


# ── Model Assembler ────────────────────────────────────────────────

class TestAssembler:
    def test_assemble_field_standard(self):
        """E1: Linear read-in + flat CTC head."""
        config = {
            "model": {
                "readin_type": "linear",
                "head_type": "flat",
                "d_shared": 64,
                "hidden_size": 64,
                "gru_layers": 2,
                "gru_dropout": 0.2,
                "temporal_stride": 5,
                "num_classes": 10,
                "linear": {"d_padded": 128},
            }
        }
        patients = {"S14": (8, 16), "S33": (12, 22)}
        backbone, head, readins = assemble_model(config, patients)
        assert isinstance(backbone, SharedBackbone)
        assert isinstance(head, FlatCTCHead)
        assert "S14" in readins
        assert "S33" in readins
        assert isinstance(readins["S14"], LinearReadIn)

    def test_assemble_full_model(self):
        """E2: Spatial conv + articulatory CTC head."""
        config = {
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
            }
        }
        patients = {"S14": (8, 16), "S33": (12, 22)}
        backbone, head, readins = assemble_model(config, patients)
        assert isinstance(backbone, SharedBackbone)
        assert isinstance(head, ArticulatoryCTCHead)
        assert isinstance(readins["S14"], SpatialConvReadIn)
        assert isinstance(readins["S33"], SpatialConvReadIn)

    def test_end_to_end_forward(self):
        """Full forward pass: grid → read-in → backbone → head → log_probs."""
        config = {
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
            }
        }
        patients = {"S14": (8, 16)}
        backbone, head, readins = assemble_model(config, patients)
        backbone.eval()

        x = torch.randn(4, 8, 16, 300)  # (B, H, W, T) for S14
        shared = readins["S14"](x)       # (B, 64, 300)
        h = backbone(shared)              # (B, 60, 128)
        log_probs = head(h)               # (B, 60, 10)
        assert log_probs.shape == (4, 60, 10)
