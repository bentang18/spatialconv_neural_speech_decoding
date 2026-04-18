"""Unit tests for the whole-grid Conv2d mixer (Task C2, Stage 2)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.v14.config import GridMixerConfig
from speech_decoding.v14.grid_mixer import GridMixer


def _layout_and_mask(B: int, N_e: int, _H_p: int, W_p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Place electrodes at (i // W_p, i % W_p), all active."""
    layout = torch.tensor(
        [[[i // W_p, i % W_p] for i in range(N_e)] for _ in range(B)], dtype=torch.long
    )
    active = torch.ones(B, N_e, dtype=torch.bool)
    return layout, active


class TestGridMixer:
    def test_output_shape_preserved(self) -> None:
        mix = GridMixer()
        B, N_e, d, T = 2, 12, 64, 28
        H_p, W_p = 4, 8
        tokens = torch.randn(B, N_e, d, T)
        layout, mask = _layout_and_mask(B, N_e, H_p, W_p)
        out = mix(tokens, layout, (H_p, W_p), mask)
        assert out.shape == (B, N_e, d, T)

    def test_inactive_electrode_output_is_zero(self) -> None:
        mix = GridMixer()
        B, N_e, d, T = 1, 8, 64, 4
        H_p, W_p = 4, 8
        tokens = torch.randn(B, N_e, d, T)
        layout, mask = _layout_and_mask(B, N_e, H_p, W_p)
        mask[0, 3] = False  # electrode 3 inactive
        out = mix(tokens, layout, (H_p, W_p), mask)
        assert torch.all(out[0, 3] == 0)

    def test_backward_produces_finite_gradients(self) -> None:
        mix = GridMixer()
        B, N_e, d, T = 2, 6, 64, 4
        H_p, W_p = 3, 4  # 12 cells, 6 electrodes, 6 padded
        tokens = torch.randn(B, N_e, d, T, requires_grad=True)
        layout, mask = _layout_and_mask(B, N_e, H_p, W_p)
        out = mix(tokens, layout, (H_p, W_p), mask)
        out.sum().backward()
        for c in mix.convs:
            assert c.weight.grad is not None
            assert torch.isfinite(c.weight.grad).all()
        assert tokens.grad is not None
        assert torch.isfinite(tokens.grad).all()

    def test_num_layers_2_works_and_has_two_conv_blocks(self) -> None:
        mix = GridMixer(GridMixerConfig(num_layers=2))
        assert len(mix.convs) == 2 and len(mix.norms) == 2
        B, N_e, d, T = 1, 6, 64, 4
        H_p, W_p = 3, 4
        tokens = torch.randn(B, N_e, d, T)
        layout, mask = _layout_and_mask(B, N_e, H_p, W_p)
        out = mix(tokens, layout, (H_p, W_p), mask)
        assert out.shape == (B, N_e, d, T)

    def test_rejects_wrong_channel_dim(self) -> None:
        mix = GridMixer()
        tokens = torch.zeros(1, 4, 32, 4)  # d=32 != 64
        layout, mask = _layout_and_mask(1, 4, 2, 4)
        with pytest.raises(ValueError, match="channel dim"):
            mix(tokens, layout, (2, 4), mask)
