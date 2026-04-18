"""Unit tests for the combined spatiotemporal attention backbone (Task C3)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.v14.backbone import Backbone, _apply_rope
from speech_decoding.v14.config import BackboneConfig


def _time_ids(n_e: int, T: int) -> torch.Tensor:
    """Flat order `(electrode, time)` -> time_ids repeat the time axis per electrode."""
    return torch.arange(T).repeat(n_e)


class TestBackbone:
    def test_shape_preserved(self) -> None:
        bb = Backbone().eval()
        B, N_e, T = 2, 4, 8
        S = N_e * T
        x = torch.randn(B, S, 64)
        active = torch.ones(B, S, dtype=torch.bool)
        y = bb(x, active, _time_ids(N_e, T))
        assert y.shape == (B, S, 64)

    def test_three_blocks_by_default(self) -> None:
        bb = Backbone()
        assert bb.num_blocks == 3
        assert len(bb.blocks) == 3

    def test_inactive_rows_stay_zero_after_forward(self) -> None:
        bb = Backbone().eval()
        B, N_e, T = 1, 6, 4
        S = N_e * T
        x = torch.randn(B, S, 64)
        active = torch.ones(B, S, dtype=torch.bool)
        # Mark electrode 2 (time-rows S_e=8..11) inactive
        active[0, 2 * T : 3 * T] = False
        y = bb(x, active, _time_ids(N_e, T))
        assert torch.all(y[0, 2 * T : 3 * T] == 0)

    def test_rope_is_identity_at_time_zero(self) -> None:
        """RoPE rotation at t=0 must be identity (cos=1, sin=0)."""
        head_dim = 16
        cos = torch.ones(1, head_dim // 2)
        sin = torch.zeros(1, head_dim // 2)
        x = torch.randn(1, 1, 4, head_dim)  # (B=1, S=1, H=4, D_h=16)
        rotated = _apply_rope(x, cos, sin)
        torch.testing.assert_close(rotated, x)

    def test_rope_preserves_norm(self) -> None:
        """A pure rotation must preserve per-feature-pair norm."""
        head_dim = 16
        S = 5
        half = head_dim // 2
        angles = torch.rand(S, half) * 3.14
        cos, sin = torch.cos(angles), torch.sin(angles)
        x = torch.randn(1, S, 4, head_dim)
        rotated = _apply_rope(x, cos, sin)
        # Per-pair magnitude unchanged
        orig_mag = x[..., 0::2] ** 2 + x[..., 1::2] ** 2
        new_mag = rotated[..., 0::2] ** 2 + rotated[..., 1::2] ** 2
        torch.testing.assert_close(orig_mag, new_mag, atol=1e-5, rtol=1e-5)

    def test_no_sc_fc_params_in_baseline(self) -> None:
        bb = Backbone()
        banned = ("sc_fc", "sc_gain", "fc_gain")
        for name, _ in bb.named_parameters():
            for b in banned:
                assert b not in name.lower(), f"found {b!r} in param {name!r}"

    def test_backward_finite_gradients(self) -> None:
        bb = Backbone()
        bb.train()
        B, N_e, T = 1, 3, 4
        S = N_e * T
        x = torch.randn(B, S, 64, requires_grad=True)
        active = torch.ones(B, S, dtype=torch.bool)
        y = bb(x, active, _time_ids(N_e, T))
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        for name, p in bb.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"

    def test_rejects_wrong_d_model(self) -> None:
        bb = Backbone()
        with pytest.raises(ValueError, match="64"):
            bb(torch.zeros(1, 8, 32), torch.ones(1, 8, dtype=torch.bool), _time_ids(2, 4))

    def test_rejects_head_dim_d_model_mismatch(self) -> None:
        with pytest.raises(ValueError, match="num_heads \\* head_dim"):
            Backbone(BackboneConfig(d_model=64, num_heads=3, head_dim=16))
