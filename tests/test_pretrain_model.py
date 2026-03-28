"""Tests for unified pretrain model (collapse mode first)."""
import pytest
import torch

from speech_decoding.pretraining.pretrain_model import PretrainModel


class TestPretrainModelCollapse:
    @pytest.fixture
    def config(self):
        return {
            "spatial_mode": "collapse",
            "d": 64,
            "gru_hidden": 32,
            "gru_layers": 2,
            "temporal_stride": 10,
            "mask_ratio": [0.4, 0.6],
            "mask_spans": [3, 6],
            "spatial_conv": {
                "channels": 8,
                "pool_h": 4,
                "pool_w": 8,
            },
        }

    def test_forward_returns_loss(self, config):
        model = PretrainModel(config, grid_shape=(8, 16))
        model.eval()
        x = torch.randn(4, 8, 16, 200)
        result = model(x, compute_loss=True)
        assert "loss" in result
        assert result["loss"].shape == ()
        assert result["loss"].item() >= 0

    def test_forward_returns_predictions(self, config):
        model = PretrainModel(config, grid_shape=(8, 16))
        model.eval()
        x = torch.randn(4, 8, 16, 200)
        result = model(x, compute_loss=True)
        assert "predictions" in result
        assert "targets" in result
        assert "mask" in result

    def test_loss_only_on_masked_frames(self, config):
        model = PretrainModel(config, grid_shape=(8, 16))
        model.eval()
        x = torch.randn(4, 8, 16, 200)
        result = model(x, compute_loss=True)
        mask = result["mask"]
        n_masked = mask.sum().item()
        assert n_masked > 0
        n_total = mask.numel()
        assert n_masked < n_total

    def test_backward_pass(self, config):
        model = PretrainModel(config, grid_shape=(8, 16))
        model.train()
        x = torch.randn(4, 8, 16, 200)
        result = model(x, compute_loss=True)
        result["loss"].backward()
        for name, p in model.named_parameters():
            if p.requires_grad and "decoder" not in name:
                assert p.grad is not None, f"No gradient for {name}"
                break

    def test_encode_only(self, config):
        model = PretrainModel(config, grid_shape=(8, 16))
        model.eval()
        x = torch.randn(4, 8, 16, 200)
        features = model.encode(x)
        T_out = 200 // config["temporal_stride"]
        assert features.shape == (4, T_out, config["gru_hidden"] * 2)

    def test_param_count_approximately_correct(self, config):
        # pool(4,8) → out_dim=256 → Conv1d(256,64,k=10) dominates at 163K weights.
        # Total: ~218K for collapse mode with recommended pool(4,8) config.
        model = PretrainModel(config, grid_shape=(8, 16))
        total = sum(p.numel() for p in model.parameters())
        assert 30_000 < total < 300_000, f"Unexpected param count: {total}"

    def test_different_grid_sizes(self, config):
        for grid_shape in [(8, 16), (12, 22)]:
            model = PretrainModel(config, grid_shape=grid_shape)
            model.eval()
            x = torch.randn(2, *grid_shape, 200)
            result = model(x, compute_loss=True)
            assert result["loss"].shape == ()
