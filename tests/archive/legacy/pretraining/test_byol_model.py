"""Tests for BYOL pretraining model."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.pretraining.byol_model import (
    BYOLModel,
    byol_loss,
)


def _make_config(**overrides):
    cfg = {
        "spatial_mode": "collapse",
        "d": 32,
        "gru_hidden": 16,
        "gru_layers": 2,
        "temporal_stride": 10,
        "spatial_conv": {"channels": 4, "pool_h": 2, "pool_w": 4},
        "byol_proj_dim": 64,
        "byol_pred_hidden": 64,
        "ema_momentum": 0.996,
        "ema_momentum_end": 1.0,
        "ema_total_steps": 100,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# BYOL loss function tests
# ---------------------------------------------------------------------------


class TestBYOLLoss:
    def test_identical_inputs_zero_loss(self):
        """Identical normalized vectors should have zero loss."""
        z = torch.randn(8, 64)
        z = torch.nn.functional.normalize(z, dim=-1)
        loss = byol_loss(z, z.clone())
        assert loss.item() < 1e-5

    def test_orthogonal_inputs_loss_two(self):
        """Orthogonal vectors should have loss = 2."""
        # Construct orthogonal pair
        p = torch.zeros(4, 2)
        p[:, 0] = 1.0
        z = torch.zeros(4, 2)
        z[:, 1] = 1.0
        loss = byol_loss(p, z)
        assert abs(loss.item() - 2.0) < 1e-5

    def test_differentiable(self):
        """Loss should be differentiable w.r.t. first argument."""
        p = torch.randn(8, 32, requires_grad=True)
        z = torch.randn(8, 32)
        loss = byol_loss(p, z)
        loss.backward()
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()

    def test_finite(self):
        """Loss should be finite for random inputs."""
        loss = byol_loss(torch.randn(8, 64), torch.randn(8, 64))
        assert loss.isfinite()

    def test_range(self):
        """BYOL loss is in [0, 4] (since 2 - 2*cos, cos in [-1,1])."""
        for _ in range(10):
            loss = byol_loss(torch.randn(8, 32), torch.randn(8, 32))
            assert 0 <= loss.item() <= 4.0


# ---------------------------------------------------------------------------
# BYOLModel tests
# ---------------------------------------------------------------------------


class TestBYOLModel:
    def test_encode_shape(self):
        """encode() returns (B, T', 2*gru_hidden)."""
        cfg = _make_config()
        model = BYOLModel(cfg, grid_shape=(8, 16))
        x = torch.randn(4, 8, 16, 300)
        out = model.encode(x)
        T_prime = 300 // cfg["temporal_stride"]
        assert out.shape == (4, T_prime, cfg["gru_hidden"] * 2)

    def test_forward_returns_loss(self):
        """forward(compute_loss=True) returns loss + components."""
        cfg = _make_config()
        model = BYOLModel(cfg, grid_shape=(8, 16))
        model.train()
        x = torch.randn(4, 8, 16, 300)
        result = model(x, compute_loss=True)
        assert "loss" in result
        assert result["loss"].isfinite()
        assert "loss_12" in result
        assert "loss_21" in result

    def test_forward_no_loss(self):
        """forward(compute_loss=False) returns embeddings only."""
        cfg = _make_config()
        model = BYOLModel(cfg, grid_shape=(8, 16))
        model.eval()
        x = torch.randn(4, 8, 16, 300)
        result = model(x, compute_loss=False)
        assert "embeddings" in result
        assert "loss" not in result
        assert result["embeddings"].shape == (4, cfg["gru_hidden"] * 2)

    def test_target_no_gradients(self):
        """Target encoder + projector params should have requires_grad=False."""
        cfg = _make_config()
        model = BYOLModel(cfg, grid_shape=(8, 16))
        for p in model._target_params():
            assert not p.requires_grad

    def test_online_gets_gradients(self):
        """Online encoder + projector + predictor should get gradients."""
        cfg = _make_config()
        model = BYOLModel(cfg, grid_shape=(8, 16))
        model.train()
        x = torch.randn(4, 8, 16, 300)
        result = model(x, compute_loss=True)
        result["loss"].backward()

        # Encoder
        for name, p in model.readin.named_parameters():
            assert p.grad is not None, f"readin.{name} has no gradient"
        for name, p in model.gru.named_parameters():
            assert p.grad is not None, f"gru.{name} has no gradient"
        # Projector
        has_proj_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.projector.parameters()
        )
        assert has_proj_grad, "No projector parameter has gradient"
        # Predictor
        has_pred_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.predictor.parameters()
        )
        assert has_pred_grad, "No predictor parameter has gradient"

    def test_ema_update_changes_target(self):
        """EMA update should move target toward online."""
        cfg = _make_config(ema_momentum=0.5)
        model = BYOLModel(cfg, grid_shape=(8, 16))

        # Perturb online weights
        with torch.no_grad():
            for p in model.readin.parameters():
                p.add_(torch.randn_like(p) * 10)

        target_before = model.target_readin.convs[0].weight.data.clone()
        model.ema_update()
        target_after = model.target_readin.convs[0].weight.data.clone()

        assert not torch.equal(target_before, target_after)

    def test_ema_momentum_schedule(self):
        """Momentum should follow cosine from start to end."""
        cfg = _make_config(ema_momentum=0.9, ema_momentum_end=1.0, ema_total_steps=10)
        model = BYOLModel(cfg, grid_shape=(8, 16))

        taus = []
        for _ in range(12):
            taus.append(model._get_ema_momentum())
            model.ema_step += 1

        # Should increase monotonically
        for i in range(1, len(taus) - 1):
            if i <= 10:
                assert taus[i] >= taus[i - 1] - 1e-6
        # Start near 0.9, end near 1.0
        assert abs(taus[0] - 0.9) < 0.01
        assert abs(taus[10] - 1.0) < 0.01

    def test_ema_step_increments(self):
        """ema_update() should increment step counter."""
        cfg = _make_config()
        model = BYOLModel(cfg, grid_shape=(8, 16))
        assert model.ema_step == 0
        model.ema_update()
        assert model.ema_step == 1
        model.ema_update()
        assert model.ema_step == 2


class TestWeightTransfer:
    def test_transfer_encoder_weights(self):
        """Encoder weights transfer correctly to PretrainModel."""
        from speech_decoding.pretraining.pretrain_model import PretrainModel

        cfg = _make_config()
        byol = BYOLModel(cfg, grid_shape=(8, 16))
        pretrain = PretrainModel(cfg, grid_shape=(8, 16))

        # Weights differ before transfer
        assert not torch.equal(
            byol.readin.convs[0].weight.data,
            pretrain.readin.convs[0].weight.data,
        )

        byol.transfer_encoder_weights(pretrain)

        # Readin
        assert torch.equal(
            byol.readin.convs[0].weight.data,
            pretrain.readin.convs[0].weight.data,
        )
        # LayerNorm
        assert torch.equal(byol.ln.weight.data, pretrain.backbone.layernorm.weight.data)
        # GRU
        for bp, pp in zip(byol.gru.parameters(), pretrain.backbone.gru.parameters()):
            assert torch.equal(bp.data, pp.data)
        # temporal_conv
        for bp, pp in zip(
            byol.temporal_conv.parameters(),
            pretrain.backbone.temporal_conv.parameters(),
        ):
            assert torch.equal(bp.data, pp.data)

    def test_transferred_model_forward(self):
        """PretrainModel with transferred weights produces valid output."""
        from speech_decoding.pretraining.pretrain_model import PretrainModel

        cfg = _make_config()
        byol = BYOLModel(cfg, grid_shape=(8, 16))
        pretrain = PretrainModel(cfg, grid_shape=(8, 16))
        byol.transfer_encoder_weights(pretrain)

        x = torch.randn(2, 8, 16, 300)
        out = pretrain.encode(x)
        T_prime = 300 // cfg["temporal_stride"]
        assert out.shape == (2, T_prime, cfg["gru_hidden"] * 2)
        assert torch.isfinite(out).all()


class TestDifferentGrids:
    def test_different_grid_shapes(self):
        """Model works with all grid shapes."""
        cfg = _make_config()
        for shape in [(8, 16), (12, 22), (8, 32)]:
            m = BYOLModel(cfg, grid_shape=shape)
            m.train()
            x = torch.randn(4, shape[0], shape[1], 300)
            result = m(x, compute_loss=True)
            assert result["loss"].isfinite(), f"Non-finite loss for grid {shape}"


class TestTraining:
    def test_loss_decreases(self):
        """BYOL loss should decrease over a few steps on same data."""
        torch.manual_seed(42)
        cfg = _make_config()
        model = BYOLModel(cfg, grid_shape=(8, 16))
        x = torch.randn(8, 8, 16, 300)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(15):
            model.train()
            result = model(x, compute_loss=True)
            optimizer.zero_grad()
            result["loss"].backward()
            optimizer.step()
            model.ema_update()
            losses.append(result["loss"].item())

        assert losses[-1] < losses[0], (
            f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


class TestTrainerCompatibility:
    def test_works_with_stage2_trainer(self):
        """BYOLModel works in Stage2Trainer with patient data dict."""
        from speech_decoding.pretraining.stage2_trainer import Stage2Trainer, Stage2Config

        cfg = _make_config()
        model = BYOLModel(cfg, grid_shape=(8, 16))
        patient_data = {"P1": torch.randn(20, 8, 16, 300)}
        s2_cfg = Stage2Config(steps=2, batch_size=4)
        trainer = Stage2Trainer(model, s2_cfg, device="cpu")

        metrics = trainer.train(patient_data)
        assert len(metrics) == 2
        assert all(np.isfinite(m["loss"]) for m in metrics)
        assert model.ema_step == 2

    def test_works_with_stage1_trainer(self):
        """BYOLModel works in Stage1Trainer with synthetic data."""
        from speech_decoding.pretraining.stage1_trainer import Stage1Trainer, Stage1Config
        from speech_decoding.pretraining.synthetic_pipeline import (
            SyntheticDataPipeline, SyntheticConfig,
        )

        cfg = _make_config()
        model = BYOLModel(cfg, grid_shape=(8, 16))
        synth_cfg = SyntheticConfig(generator="smooth_ar", grid_shapes=[(8, 16)])
        pipeline = SyntheticDataPipeline(synth_cfg)
        s1_cfg = Stage1Config(steps=2, batch_size=4, T=300)
        trainer = Stage1Trainer(model, pipeline, s1_cfg, device="cpu")

        metrics = trainer.train()
        assert len(metrics) == 2
        assert all(np.isfinite(m["loss"]) for m in metrics)
        assert model.ema_step == 2

    def test_works_with_semi_supervised_trainer(self):
        """BYOLModel works with SemiSupervisedStage2Trainer."""
        from speech_decoding.pretraining.semi_supervised_trainer import (
            SemiSupervisedStage2Trainer, SemiSupervisedConfig,
        )

        cfg = _make_config()
        model = BYOLModel(cfg, grid_shape=(8, 16))
        patient_data = {"P1": torch.randn(20, 8, 16, 300)}
        labeled_grids = torch.randn(30, 8, 16, 300)
        labeled_labels = [[np.random.randint(1, 10) for _ in range(3)]
                          for _ in range(30)]

        ss_cfg = SemiSupervisedConfig(steps=2, batch_size=4, ce_batch_size=4)
        trainer = SemiSupervisedStage2Trainer(model, ss_cfg, device="cpu")

        metrics = trainer.train(patient_data, labeled_grids, labeled_labels)
        assert len(metrics) == 2
        assert all(np.isfinite(m["loss"]) for m in metrics)
        assert all("ssl_loss" in m for m in metrics)
        assert all("ce_loss" in m for m in metrics)
        assert model.ema_step == 2

    def test_cli_byol_branch(self):
        """Integration: construct from config, forward, transfer, encode."""
        import yaml
        from speech_decoding.pretraining.pretrain_model import PretrainModel

        with open("configs/pretrain_base.yaml") as f:
            config = yaml.safe_load(f)

        config["ema_total_steps"] = 100
        byol = BYOLModel(config, grid_shape=(8, 16))
        x = torch.randn(4, 8, 16, 200)

        # Forward pass (training)
        byol.train()
        out = byol(x, compute_loss=True)
        assert out["loss"].isfinite()

        # Transfer to PretrainModel
        pretrain = PretrainModel(config, grid_shape=(8, 16))
        byol.transfer_encoder_weights(pretrain)

        # Downstream encode
        features = pretrain.encode(x)
        T_prime = 200 // config["temporal_stride"]
        assert features.shape == (4, T_prime, config["gru_hidden"] * 2)
