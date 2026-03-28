"""Tests for reconstruction decoder."""
import pytest
import torch

from speech_decoding.pretraining.decoder import ReconstructionDecoder


class TestReconstructionDecoder:
    def test_collapse_mode_shape(self):
        """collapse: (B, T, 2H) → (B, T, d_flat)."""
        dec = ReconstructionDecoder(input_dim=128, output_dim=256, mode="collapse")
        h = torch.randn(4, 30, 128)
        out = dec(h)
        assert out.shape == (4, 30, 256)

    def test_preserve_mode_shape(self):
        """preserve: (B, T, 2H) → (B, T, 1) per position."""
        dec = ReconstructionDecoder(input_dim=128, output_dim=1, mode="preserve")
        h = torch.randn(4, 30, 128)
        out = dec(h)
        assert out.shape == (4, 30, 1)

    def test_param_count_collapse(self):
        dec = ReconstructionDecoder(input_dim=128, output_dim=256, mode="collapse")
        n = sum(p.numel() for p in dec.parameters())
        assert n == 128 * 256 + 256

    def test_param_count_preserve(self):
        dec = ReconstructionDecoder(input_dim=128, output_dim=1, mode="preserve")
        n = sum(p.numel() for p in dec.parameters())
        assert n == 128 * 1 + 1

    def test_gradient_flows(self):
        dec = ReconstructionDecoder(input_dim=128, output_dim=256, mode="collapse")
        h = torch.randn(4, 30, 128, requires_grad=True)
        out = dec(h)
        out.sum().backward()
        assert h.grad is not None
