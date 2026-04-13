"""Tests for LeWM-style pretraining (SIGReg + next-embedding prediction)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.pretraining.sigreg import sigreg
from speech_decoding.pretraining.lewm_model import LeWMModel


# ---------------------------------------------------------------------------
# SIGReg tests
# ---------------------------------------------------------------------------

class TestSIGReg:
    def test_gaussian_input_low_loss(self):
        """SIGReg should be near zero for isotropic Gaussian embeddings."""
        torch.manual_seed(42)
        Z = torch.randn(256, 64)  # N(0, I)
        loss = sigreg(Z, M=512)
        assert loss.item() < 0.5, f"SIGReg on Gaussian data should be low, got {loss.item()}"

    def test_constant_input_high_loss(self):
        """SIGReg should be high for collapsed (constant) embeddings."""
        Z = torch.ones(256, 64) * 3.0
        loss = sigreg(Z, M=512)
        # Constant input → ECF is a delta → far from Gaussian CF
        assert loss.item() > 1.0, f"SIGReg on constant data should be high, got {loss.item()}"

    def test_non_gaussian_higher_than_gaussian(self):
        """Non-Gaussian distributions should have higher SIGReg than Gaussian."""
        torch.manual_seed(42)
        Z_gauss = torch.randn(256, 32)
        # Uniform distribution — not Gaussian
        Z_uniform = torch.rand(256, 32) * 6 - 3  # U[-3, 3], same range as ~N(0,1)

        loss_gauss = sigreg(Z_gauss, M=256)
        loss_uniform = sigreg(Z_uniform, M=256)
        assert loss_uniform > loss_gauss, (
            f"Uniform ({loss_uniform.item():.4f}) should have higher SIGReg "
            f"than Gaussian ({loss_gauss.item():.4f})"
        )

    def test_differentiable(self):
        """SIGReg loss should be differentiable w.r.t. input."""
        Z = torch.randn(64, 16, requires_grad=True)
        loss = sigreg(Z, M=128, n_nodes=20)
        loss.backward()
        assert Z.grad is not None
        assert not torch.isnan(Z.grad).any()

    def test_output_shape(self):
        """SIGReg should return a scalar."""
        Z = torch.randn(32, 8)
        loss = sigreg(Z, M=64)
        assert loss.shape == ()

    def test_small_batch(self):
        """SIGReg should not crash with small batch sizes."""
        Z = torch.randn(4, 16)
        loss = sigreg(Z, M=32, n_nodes=10)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# LeWMModel tests
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    cfg = {
        "spatial_mode": "collapse",
        "d": 32,
        "gru_hidden": 16,
        "temporal_stride": 10,
        "spatial_conv": {"channels": 4, "pool_h": 2, "pool_w": 4},
        "sigreg_lambda": 0.1,
        "sigreg_M": 64,
        "predictor_hidden": 64,
    }
    cfg.update(overrides)
    return cfg


class TestLeWMModel:
    def test_encode_shape(self):
        """encode() should return (B, T', d)."""
        cfg = _make_config()
        model = LeWMModel(cfg, grid_shape=(8, 16))
        x = torch.randn(2, 8, 16, 100)
        z = model.encode(x)
        T_prime = 100 // cfg["temporal_stride"]
        assert z.shape == (2, T_prime, cfg["d"])

    def test_forward_loss_keys(self):
        """forward() with compute_loss should return loss, pred_loss, sigreg_loss."""
        cfg = _make_config()
        model = LeWMModel(cfg, grid_shape=(8, 16))
        model.train()
        x = torch.randn(4, 8, 16, 100)
        out = model(x, compute_loss=True)
        assert "loss" in out
        assert "pred_loss" in out
        assert "sigreg_loss" in out
        assert "embeddings" in out
        assert torch.isfinite(out["loss"])
        assert isinstance(out["pred_loss"], float)
        assert isinstance(out["sigreg_loss"], float)

    def test_forward_no_loss(self):
        """forward() without compute_loss should only return embeddings."""
        cfg = _make_config()
        model = LeWMModel(cfg, grid_shape=(8, 16))
        model.eval()
        x = torch.randn(2, 8, 16, 100)
        out = model(x, compute_loss=False)
        assert "embeddings" in out
        assert "loss" not in out

    def test_embeddings_shape(self):
        """Embeddings should be (B, T', d)."""
        cfg = _make_config()
        model = LeWMModel(cfg, grid_shape=(8, 16))
        model.train()
        x = torch.randn(2, 8, 16, 100)
        out = model(x, compute_loss=True)
        T_prime = 100 // cfg["temporal_stride"]
        assert out["embeddings"].shape == (2, T_prime, cfg["d"])

    def test_backward(self):
        """Loss should be differentiable through entire model."""
        cfg = _make_config()
        model = LeWMModel(cfg, grid_shape=(8, 16))
        model.train()
        x = torch.randn(2, 8, 16, 100)
        out = model(x, compute_loss=True)
        out["loss"].backward()
        # Encoder should get gradients (end-to-end, no stop-gradient)
        for name, p in model.readin.named_parameters():
            assert p.grad is not None, f"readin.{name} has no gradient"
        for name, p in model.predictor.named_parameters():
            assert p.grad is not None, f"predictor.{name} has no gradient"

    def test_no_stop_gradient(self):
        """Verify gradients flow through BOTH predictor input and target."""
        cfg = _make_config()
        model = LeWMModel(cfg, grid_shape=(8, 16))
        model.train()
        x = torch.randn(2, 8, 16, 60)
        out = model(x, compute_loss=True)
        out["loss"].backward()
        # LayerNorm and temporal_conv should get gradients from both
        # the prediction (via z_t) and target (via z_{t+1}) paths
        conv_weight = list(model.temporal_conv.parameters())[0]
        assert conv_weight.grad is not None
        assert conv_weight.grad.abs().sum() > 0

    def test_different_grid_shapes(self):
        """Model should work with different grid shapes."""
        cfg = _make_config()
        for shape in [(8, 16), (12, 22), (8, 32)]:
            model = LeWMModel(cfg, grid_shape=shape)
            x = torch.randn(2, shape[0], shape[1], 100)
            out = model(x, compute_loss=True)
            assert torch.isfinite(out["loss"])


class TestWeightTransfer:
    def test_transfer_to_pretrain_model(self):
        """Encoder weights should transfer correctly to PretrainModel."""
        from speech_decoding.pretraining.pretrain_model import PretrainModel

        cfg = _make_config()
        lewm = LeWMModel(cfg, grid_shape=(8, 16))
        pretrain = PretrainModel(cfg, grid_shape=(8, 16))

        # Conv2d weights should differ before transfer (randomly initialized)
        lewm_conv = lewm.readin.convs[0].weight.data.clone()
        pretrain_conv = pretrain.readin.convs[0].weight.data.clone()
        assert not torch.equal(lewm_conv, pretrain_conv)

        # Transfer
        lewm.transfer_encoder_weights(pretrain)

        # Encoder weights should match after transfer
        assert torch.equal(lewm.readin.convs[0].weight.data,
                           pretrain.readin.convs[0].weight.data)
        assert torch.equal(lewm.ln.weight.data,
                           pretrain.backbone.layernorm.weight.data)
        assert torch.equal(lewm.ln.bias.data,
                           pretrain.backbone.layernorm.bias.data)

        # temporal_conv is Sequential(Conv1d, GELU) — Conv1d weight should match
        lewm_params = list(lewm.temporal_conv.parameters())
        pretrain_params = list(pretrain.backbone.temporal_conv.parameters())
        for lw, pw in zip(lewm_params, pretrain_params):
            assert torch.equal(lw.data, pw.data)

    def test_transferred_model_forward(self):
        """PretrainModel with transferred weights should produce valid output."""
        from speech_decoding.pretraining.pretrain_model import PretrainModel

        cfg = _make_config()
        lewm = LeWMModel(cfg, grid_shape=(8, 16))
        pretrain = PretrainModel(cfg, grid_shape=(8, 16))
        lewm.transfer_encoder_weights(pretrain)

        x = torch.randn(2, 8, 16, 100)
        features = pretrain.encode(x)
        assert features.shape == (2, 10, cfg["gru_hidden"] * 2)


class TestTrainerCompatibility:
    def test_works_with_stage1_trainer(self):
        """LeWMModel should work as drop-in replacement in Stage1Trainer."""
        from speech_decoding.pretraining.stage1_trainer import Stage1Trainer, Stage1Config
        from speech_decoding.pretraining.synthetic_pipeline import (
            SyntheticDataPipeline, SyntheticConfig,
        )

        cfg = _make_config()
        model = LeWMModel(cfg, grid_shape=(8, 16))
        synth_cfg = SyntheticConfig(generator="smooth_ar", grid_shapes=[(8, 16)])
        pipeline = SyntheticDataPipeline(synth_cfg)
        s1_cfg = Stage1Config(steps=2, batch_size=2, T=100)
        trainer = Stage1Trainer(model, pipeline, s1_cfg, device="cpu")

        metrics = trainer.train()
        assert len(metrics) == 2
        assert all("loss" in m for m in metrics)
        assert all(np.isfinite(m["loss"]) for m in metrics)

    def test_loss_decreases(self):
        """Loss should generally decrease during training (smoke test)."""
        cfg = _make_config()
        model = LeWMModel(cfg, grid_shape=(8, 16))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        torch.manual_seed(42)
        x = torch.randn(4, 8, 16, 100)
        losses = []
        for _ in range(10):
            model.train()
            out = model(x, compute_loss=True)
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()
            losses.append(out["loss"].item())

        # Loss should decrease over 10 steps on same data
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )
