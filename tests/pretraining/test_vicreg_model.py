"""Tests for VICReg contrastive SSL pretraining model."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.pretraining.vicreg_model import (
    VICRegModel,
    off_diagonal,
    vicreg_loss,
)


def _make_config(**overrides):
    cfg = {
        "spatial_mode": "collapse",
        "d": 32,
        "gru_hidden": 16,
        "gru_layers": 2,
        "temporal_stride": 10,
        "spatial_conv": {"channels": 4, "pool_h": 2, "pool_w": 4},
        "vicreg_proj_dim": 64,  # smaller for tests
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# VICReg loss function tests
# ---------------------------------------------------------------------------

class TestOffDiagonal:
    def test_identity_matrix(self):
        """Off-diagonal of identity should sum to 0."""
        M = torch.eye(3)
        assert off_diagonal(M).sum() == 0.0

    def test_ones_matrix(self):
        """Off-diagonal of ones(3,3) should be 6 ones."""
        M = torch.ones(3, 3)
        assert off_diagonal(M).sum() == 6.0

    def test_count(self):
        """Off-diagonal of (n,n) should have n^2 - n elements."""
        for n in [2, 4, 8]:
            M = torch.randn(n, n)
            assert off_diagonal(M).numel() == n * n - n


class TestVICRegLoss:
    def test_loss_finite(self):
        """Loss should be finite for random inputs."""
        z1 = torch.randn(8, 64)
        z2 = torch.randn(8, 64)
        result = vicreg_loss(z1, z2)
        assert result["loss"].isfinite()
        assert "inv_loss" in result
        assert "var_loss" in result
        assert "cov_loss" in result

    def test_identical_inputs_zero_invariance(self):
        """Identical views should have zero invariance loss."""
        z = torch.randn(8, 64)
        result = vicreg_loss(z, z.clone())
        assert result["inv_loss"] < 1e-6

    def test_collapsed_inputs_high_variance_loss(self):
        """Constant embeddings (all same) should have high variance loss."""
        z = torch.ones(8, 64)  # std=0 per dim -> var_loss = relu(1-0) = 1
        result = vicreg_loss(z, z.clone())
        assert result["var_loss"] > 0.5

    def test_correlated_dims_high_cov_loss(self):
        """Perfectly correlated dimensions should produce nonzero cov loss."""
        z = torch.randn(16, 64)
        z[:, 1] = z[:, 0]  # perfectly correlated
        result = vicreg_loss(z, z.clone())
        assert result["cov_loss"] > 0.0

    def test_component_types(self):
        """Component losses should be Python floats, total should be tensor."""
        z1 = torch.randn(8, 32)
        z2 = torch.randn(8, 32)
        result = vicreg_loss(z1, z2)
        assert isinstance(result["loss"], torch.Tensor)
        assert isinstance(result["inv_loss"], float)
        assert isinstance(result["var_loss"], float)
        assert isinstance(result["cov_loss"], float)

    def test_differentiable(self):
        """Loss should be differentiable w.r.t. inputs."""
        z1 = torch.randn(8, 32, requires_grad=True)
        z2 = torch.randn(8, 32, requires_grad=True)
        result = vicreg_loss(z1, z2)
        result["loss"].backward()
        assert z1.grad is not None
        assert z2.grad is not None
        assert not torch.isnan(z1.grad).any()
        assert not torch.isnan(z2.grad).any()

    def test_lambda_scaling(self):
        """Doubling lambda_inv should roughly double invariance contribution."""
        z1 = torch.randn(16, 32)
        z2 = torch.randn(16, 32)
        r1 = vicreg_loss(z1, z2, lambda_inv=25.0, lambda_var=0.0, lambda_cov=0.0)
        r2 = vicreg_loss(z1, z2, lambda_inv=50.0, lambda_var=0.0, lambda_cov=0.0)
        # With only inv term, doubling lambda should double loss
        ratio = r2["loss"].item() / r1["loss"].item()
        assert abs(ratio - 2.0) < 0.01, f"Expected ratio ~2.0, got {ratio:.4f}"


# ---------------------------------------------------------------------------
# VICRegModel tests
# ---------------------------------------------------------------------------

class TestVICRegModel:
    def test_encode_shape(self):
        """encode() returns (B, T', 2*gru_hidden)."""
        cfg = _make_config()
        model = VICRegModel(cfg, grid_shape=(8, 16))
        x = torch.randn(4, 8, 16, 300)
        out = model.encode(x)
        T_prime = 300 // cfg["temporal_stride"]
        assert out.shape == (4, T_prime, cfg["gru_hidden"] * 2)

    def test_forward_returns_loss(self):
        """forward(compute_loss=True) returns loss + component keys."""
        cfg = _make_config()
        model = VICRegModel(cfg, grid_shape=(8, 16))
        model.train()
        x = torch.randn(4, 8, 16, 300)
        result = model.forward(x, compute_loss=True)
        assert "loss" in result
        assert result["loss"].isfinite()
        assert "inv_loss" in result
        assert "var_loss" in result
        assert "cov_loss" in result

    def test_forward_no_loss(self):
        """forward(compute_loss=False) returns embeddings only."""
        cfg = _make_config()
        model = VICRegModel(cfg, grid_shape=(8, 16))
        model.eval()
        x = torch.randn(4, 8, 16, 300)
        result = model.forward(x, compute_loss=False)
        assert "embeddings" in result
        assert "loss" not in result
        # Embeddings: mean-pooled to (B, 2*gru_hidden)
        assert result["embeddings"].shape == (4, cfg["gru_hidden"] * 2)

    def test_transfer_encoder_weights(self):
        """Encoder weights transfer correctly to PretrainModel."""
        from speech_decoding.pretraining.pretrain_model import PretrainModel

        cfg = _make_config()
        vicreg = VICRegModel(cfg, grid_shape=(8, 16))
        pretrain = PretrainModel(cfg, grid_shape=(8, 16))

        # Weights differ before transfer
        vicreg_conv = vicreg.readin.convs[0].weight.data.clone()
        pretrain_conv = pretrain.readin.convs[0].weight.data.clone()
        assert not torch.equal(vicreg_conv, pretrain_conv)

        vicreg.transfer_encoder_weights(pretrain)

        # Readin weights match
        assert torch.equal(vicreg.readin.convs[0].weight.data,
                           pretrain.readin.convs[0].weight.data)
        # LayerNorm matches
        assert torch.equal(vicreg.ln.weight.data,
                           pretrain.backbone.layernorm.weight.data)
        assert torch.equal(vicreg.ln.bias.data,
                           pretrain.backbone.layernorm.bias.data)
        # GRU weights match
        for vp, pp in zip(vicreg.gru.parameters(),
                          pretrain.backbone.gru.parameters()):
            assert torch.equal(vp.data, pp.data)
        # temporal_conv matches
        for vp, pp in zip(vicreg.temporal_conv.parameters(),
                          pretrain.backbone.temporal_conv.parameters()):
            assert torch.equal(vp.data, pp.data)

    def test_transferred_model_forward(self):
        """PretrainModel with transferred weights produces valid output."""
        from speech_decoding.pretraining.pretrain_model import PretrainModel

        cfg = _make_config()
        vicreg = VICRegModel(cfg, grid_shape=(8, 16))
        pretrain = PretrainModel(cfg, grid_shape=(8, 16))
        vicreg.transfer_encoder_weights(pretrain)

        x = torch.randn(2, 8, 16, 300)
        out = pretrain.encode(x)
        T_prime = 300 // cfg["temporal_stride"]
        assert out.shape == (2, T_prime, cfg["gru_hidden"] * 2)
        assert torch.isfinite(out).all()

    def test_different_grid_shapes(self):
        """Model works with (8,16), (12,22), (8,32) grids."""
        cfg = _make_config()
        for shape in [(8, 16), (12, 22), (8, 32)]:
            m = VICRegModel(cfg, grid_shape=shape)
            m.train()
            x = torch.randn(4, shape[0], shape[1], 300)
            result = m(x, compute_loss=True)
            assert result["loss"].isfinite(), f"Non-finite loss for grid {shape}"

    def test_gradients_flow(self):
        """Gradients flow through encoder and projector."""
        cfg = _make_config()
        model = VICRegModel(cfg, grid_shape=(8, 16))
        model.train()
        x = torch.randn(4, 8, 16, 300)
        result = model(x, compute_loss=True)
        result["loss"].backward()

        # Encoder should get gradients
        for name, p in model.readin.named_parameters():
            assert p.grad is not None, f"readin.{name} has no gradient"
        for name, p in model.gru.named_parameters():
            assert p.grad is not None, f"gru.{name} has no gradient"
        # Projector should get gradients
        has_proj_grad = False
        for p in model.projector.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_proj_grad = True
                break
        assert has_proj_grad, "No projector parameter has gradient"


class TestTraining:
    def test_loss_decreases(self):
        """VICReg loss should decrease over a few steps on same data."""
        torch.manual_seed(42)
        cfg = _make_config()
        model = VICRegModel(cfg, grid_shape=(8, 16))
        x = torch.randn(8, 8, 16, 300)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(10):
            model.train()
            result = model(x, compute_loss=True)
            optimizer.zero_grad()
            result["loss"].backward()
            optimizer.step()
            losses.append(result["loss"].item())

        assert losses[-1] < losses[0], (
            f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


class TestTrainerCompatibility:
    def test_works_with_stage2_trainer(self):
        """VICRegModel works in Stage2Trainer with patient data dict."""
        from speech_decoding.pretraining.stage2_trainer import Stage2Trainer, Stage2Config

        cfg = _make_config()
        model = VICRegModel(cfg, grid_shape=(8, 16))
        patient_data = {"P1": torch.randn(20, 8, 16, 300)}
        s2_cfg = Stage2Config(steps=2, batch_size=4)
        trainer = Stage2Trainer(model, s2_cfg, device="cpu")

        metrics = trainer.train(patient_data)
        assert len(metrics) == 2
        assert all(np.isfinite(m["loss"]) for m in metrics)

    def test_works_with_stage1_trainer(self):
        """VICRegModel works in Stage1Trainer with synthetic data."""
        from speech_decoding.pretraining.stage1_trainer import Stage1Trainer, Stage1Config
        from speech_decoding.pretraining.synthetic_pipeline import (
            SyntheticDataPipeline, SyntheticConfig,
        )

        cfg = _make_config()
        model = VICRegModel(cfg, grid_shape=(8, 16))
        synth_cfg = SyntheticConfig(generator="smooth_ar", grid_shapes=[(8, 16)])
        pipeline = SyntheticDataPipeline(synth_cfg)
        s1_cfg = Stage1Config(steps=2, batch_size=4, T=300)
        trainer = Stage1Trainer(model, pipeline, s1_cfg, device="cpu")

        metrics = trainer.train()
        assert len(metrics) == 2
        assert all("loss" in m for m in metrics)
        assert all(np.isfinite(m["loss"]) for m in metrics)

    def test_cli_vicreg_branch(self):
        """Integration: construct from config, forward, transfer, encode."""
        import yaml
        from speech_decoding.pretraining.pretrain_model import PretrainModel

        with open("configs/pretrain_base.yaml") as f:
            config = yaml.safe_load(f)

        vicreg = VICRegModel(config, grid_shape=(8, 16))
        x = torch.randn(4, 8, 16, 200)

        # Forward pass (training)
        vicreg.train()
        out = vicreg(x, compute_loss=True)
        assert out["loss"].isfinite()

        # Transfer to PretrainModel
        pretrain = PretrainModel(config, grid_shape=(8, 16))
        vicreg.transfer_encoder_weights(pretrain)

        # Downstream encode
        features = pretrain.encode(x)
        T_prime = 200 // config["temporal_stride"]
        assert features.shape == (4, T_prime, config["gru_hidden"] * 2)
