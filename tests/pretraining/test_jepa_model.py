"""Tests for JEPA pretraining model (EMA target encoder + masked prediction)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from speech_decoding.pretraining.jepa_model import JEPAModel


def _make_config(**overrides):
    cfg = {
        "spatial_mode": "collapse",
        "d": 32,
        "gru_hidden": 16,
        "gru_layers": 2,
        "temporal_stride": 10,
        "spatial_conv": {"channels": 4, "pool_h": 2, "pool_w": 4},
        "predictor_hidden": 64,
        "ema_momentum": 0.996,
        "ema_momentum_end": 1.0,
        "ema_total_steps": 100,
        "mask_ratio": [0.4, 0.6],
        "mask_spans": [3, 6],
    }
    cfg.update(overrides)
    return cfg


class TestJEPAForward:
    def test_forward_shapes(self):
        """predictions/targets are (B, T', 2H), mask is (T',) bool."""
        cfg = _make_config()
        model = JEPAModel(cfg, grid_shape=(8, 16))
        model.train()
        x = torch.randn(2, 8, 16, 100)
        out = model(x, compute_loss=True)

        T_prime = 100 // cfg["temporal_stride"]
        gru_out_dim = cfg["gru_hidden"] * 2
        assert out["predictions"].shape == (2, T_prime, gru_out_dim)
        assert out["targets"].shape == (2, T_prime, gru_out_dim)
        assert out["mask"].shape == (T_prime,)
        assert out["mask"].dtype == torch.bool

    def test_loss_on_masked_only(self):
        """Loss is finite scalar, mask has both True and False."""
        cfg = _make_config()
        model = JEPAModel(cfg, grid_shape=(8, 16))
        model.train()
        x = torch.randn(4, 8, 16, 100)
        out = model(x, compute_loss=True)

        assert "loss" in out
        assert torch.isfinite(out["loss"])
        assert out["loss"].shape == ()
        # Mask should have both masked and unmasked positions
        assert out["mask"].any(), "No positions masked"
        assert not out["mask"].all(), "All positions masked"

    def test_encode_shape(self):
        """encode() returns (B, T', 2*gru_hidden), no predictor involved."""
        cfg = _make_config()
        model = JEPAModel(cfg, grid_shape=(8, 16))
        model.eval()
        x = torch.randn(2, 8, 16, 100)
        z = model.encode(x)
        T_prime = 100 // cfg["temporal_stride"]
        assert z.shape == (2, T_prime, cfg["gru_hidden"] * 2)


class TestEMA:
    def test_ema_update_changes_target(self):
        """Target weights move toward online after EMA update."""
        cfg = _make_config(ema_momentum=0.5)  # aggressive for visibility
        model = JEPAModel(cfg, grid_shape=(8, 16))

        # Snapshot target before update
        target_before = model.target_readin.convs[0].weight.data.clone()
        online_weight = model.readin.convs[0].weight.data.clone()

        # Perturb online to create a gap
        with torch.no_grad():
            model.readin.convs[0].weight.data.add_(torch.randn_like(online_weight) * 10)

        model.ema_update()

        target_after = model.target_readin.convs[0].weight.data
        # Target should have moved from its initial value
        assert not torch.equal(target_before, target_after), "Target did not change after EMA"

    def test_target_no_gradients(self):
        """Target params have requires_grad=False, grad=None after backward."""
        cfg = _make_config()
        model = JEPAModel(cfg, grid_shape=(8, 16))
        model.train()
        x = torch.randn(2, 8, 16, 100)
        out = model(x, compute_loss=True)
        out["loss"].backward()

        for name, p in model.target_readin.named_parameters():
            assert not p.requires_grad, f"target_readin.{name} has requires_grad=True"
            assert p.grad is None, f"target_readin.{name} has gradient"
        for name, p in model.target_gru.named_parameters():
            assert not p.requires_grad, f"target_gru.{name} has requires_grad=True"
            assert p.grad is None, f"target_gru.{name} has gradient"

    def test_online_gets_gradients(self):
        """Online encoder + predictor params get gradients."""
        cfg = _make_config()
        model = JEPAModel(cfg, grid_shape=(8, 16))
        model.train()
        x = torch.randn(2, 8, 16, 100)
        out = model(x, compute_loss=True)
        out["loss"].backward()

        for name, p in model.readin.named_parameters():
            assert p.grad is not None, f"readin.{name} has no gradient"
        for name, p in model.predictor.named_parameters():
            assert p.grad is not None, f"predictor.{name} has no gradient"
        for name, p in model.gru.named_parameters():
            assert p.grad is not None, f"gru.{name} has no gradient"

    def test_ema_momentum_schedule(self):
        """tau progression follows cosine from start to end."""
        cfg = _make_config(ema_momentum=0.9, ema_momentum_end=1.0, ema_total_steps=100)
        model = JEPAModel(cfg, grid_shape=(8, 16))

        taus = []
        for _ in range(100):
            taus.append(model._get_ema_momentum())
            model.ema_step += 1

        # Should start near 0.9 and end near 1.0
        assert abs(taus[0] - 0.9) < 0.01, f"Initial tau {taus[0]} not near 0.9"
        assert abs(taus[-1] - 1.0) < 0.01, f"Final tau {taus[-1]} not near 1.0"
        # Monotonically non-decreasing (cosine schedule)
        for i in range(1, len(taus)):
            assert taus[i] >= taus[i - 1] - 1e-7, (
                f"tau decreased at step {i}: {taus[i-1]:.6f} -> {taus[i]:.6f}"
            )


class TestWeightTransfer:
    def test_transfer_to_pretrain_model(self):
        """TARGET encoder weights should transfer correctly to PretrainModel."""
        from speech_decoding.pretraining.pretrain_model import PretrainModel

        cfg = _make_config(ema_momentum=0.5)  # aggressive EMA so target != online
        jepa = JEPAModel(cfg, grid_shape=(8, 16))
        pretrain = PretrainModel(cfg, grid_shape=(8, 16))

        # Perturb online encoder to create divergence, then EMA update
        with torch.no_grad():
            jepa.readin.convs[0].weight.data.add_(torch.randn_like(jepa.readin.convs[0].weight) * 5)
        jepa.ema_update()

        jepa.transfer_encoder_weights(pretrain)

        # Should match TARGET encoder (not online)
        assert torch.equal(jepa.target_readin.convs[0].weight.data,
                           pretrain.readin.convs[0].weight.data)
        assert not torch.equal(jepa.readin.convs[0].weight.data,
                               pretrain.readin.convs[0].weight.data), \
            "Transferred online encoder instead of target"

        assert torch.equal(jepa.target_ln.weight.data,
                           pretrain.backbone.layernorm.weight.data)

        # temporal_conv params match target
        for tp, pp in zip(jepa.target_temporal_conv.parameters(),
                          pretrain.backbone.temporal_conv.parameters()):
            assert torch.equal(tp.data, pp.data)

        # GRU params match target
        for tp, pp in zip(jepa.target_gru.parameters(),
                          pretrain.backbone.gru.parameters()):
            assert torch.equal(tp.data, pp.data)

    def test_transferred_model_forward(self):
        """PretrainModel with transferred weights produces valid output."""
        from speech_decoding.pretraining.pretrain_model import PretrainModel

        cfg = _make_config()
        jepa = JEPAModel(cfg, grid_shape=(8, 16))
        pretrain = PretrainModel(cfg, grid_shape=(8, 16))
        jepa.transfer_encoder_weights(pretrain)

        x = torch.randn(2, 8, 16, 100)
        features = pretrain.encode(x)
        assert features.shape == (2, 10, cfg["gru_hidden"] * 2)


class TestDifferentGridShapes:
    def test_different_grid_shapes(self):
        """Model works with (8,16), (12,22), (8,32) grids."""
        cfg = _make_config()
        for shape in [(8, 16), (12, 22), (8, 32)]:
            model = JEPAModel(cfg, grid_shape=shape)
            model.train()
            x = torch.randn(2, shape[0], shape[1], 100)
            out = model(x, compute_loss=True)
            assert torch.isfinite(out["loss"]), f"Non-finite loss for grid {shape}"


class TestTraining:
    def test_loss_decreases(self):
        """Loss decreases over 10 steps on same data (fixed seed + frozen EMA)."""
        cfg = _make_config()
        model = JEPAModel(cfg, grid_shape=(8, 16))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        torch.manual_seed(42)
        x = torch.randn(4, 8, 16, 100)

        # Fix the mask by using a deterministic seed for each step
        # and freeze EMA to remove those nondeterminism sources
        losses = []
        for step in range(10):
            model.train()
            # Use fixed mask by monkey-patching np.random.RandomState
            T_prime = 100 // cfg["temporal_stride"]
            rng = np.random.RandomState(0)  # same mask every step
            from speech_decoding.pretraining.masking import generate_span_mask
            mask = generate_span_mask(T_prime, rng=rng)
            mask_tensor = torch.from_numpy(mask)

            # Manual forward with fixed mask
            online_out = model._encode_online(x, mask=mask_tensor)
            B, T_p, D = online_out.shape
            predictions = model.predictor(online_out.reshape(B * T_p, D)).reshape(B, T_p, D)
            with torch.no_grad():
                targets = model._encode_target(x)
            pred_masked = predictions[:, mask_tensor]
            tgt_masked = targets[:, mask_tensor]
            loss = torch.nn.functional.mse_loss(pred_masked, tgt_masked)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Don't call ema_update — freeze target for determinism
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


class TestTrainerCompatibility:
    def test_works_with_stage1_trainer(self):
        """JEPAModel works in Stage1Trainer, ema_step advances correctly."""
        from speech_decoding.pretraining.stage1_trainer import Stage1Trainer, Stage1Config
        from speech_decoding.pretraining.synthetic_pipeline import (
            SyntheticDataPipeline, SyntheticConfig,
        )

        cfg = _make_config(ema_total_steps=10)
        model = JEPAModel(cfg, grid_shape=(8, 16))
        synth_cfg = SyntheticConfig(generator="smooth_ar", grid_shapes=[(8, 16)])
        pipeline = SyntheticDataPipeline(synth_cfg)
        s1_cfg = Stage1Config(steps=2, batch_size=2, T=100)
        trainer = Stage1Trainer(model, pipeline, s1_cfg, device="cpu")

        metrics = trainer.train()
        assert len(metrics) == 2
        assert all("loss" in m for m in metrics)
        assert all(np.isfinite(m["loss"]) for m in metrics)
        assert model.ema_step == 2

    def test_works_with_stage2_trainer(self):
        """JEPAModel works in Stage2Trainer with patient data dict."""
        from speech_decoding.pretraining.stage2_trainer import Stage2Trainer, Stage2Config

        cfg = _make_config(ema_total_steps=10)
        model = JEPAModel(cfg, grid_shape=(8, 16))
        patient_data = {
            "S14": torch.randn(20, 8, 16, 100),
            "S22": torch.randn(15, 8, 16, 100),
        }
        s2_cfg = Stage2Config(steps=2, batch_size=4)
        trainer = Stage2Trainer(model, s2_cfg, device="cpu")

        metrics = trainer.train(patient_data)
        assert len(metrics) == 2
        assert all(np.isfinite(m["loss"]) for m in metrics)
        assert model.ema_step == 2

    def test_cli_jepa_branch(self):
        """Integration: construct from config, forward, transfer, encode."""
        import yaml
        from speech_decoding.pretraining.pretrain_model import PretrainModel

        with open("configs/pretrain_base.yaml") as f:
            config = yaml.safe_load(f)

        # CLI injects ema_total_steps at runtime
        config["ema_total_steps"] = 5000

        jepa = JEPAModel(config, grid_shape=(8, 16))
        x = torch.randn(2, 8, 16, 200)

        # Forward pass (training)
        out = jepa(x, compute_loss=True)
        assert torch.isfinite(out["loss"])

        # Transfer to PretrainModel
        pretrain = PretrainModel(config, grid_shape=(8, 16))
        jepa.transfer_encoder_weights(pretrain)

        # Downstream encode
        features = pretrain.encode(x)
        T_prime = 200 // config["temporal_stride"]
        assert features.shape == (2, T_prime, config["gru_hidden"] * 2)
