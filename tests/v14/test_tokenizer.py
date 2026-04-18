"""Unit tests for the shared per-electrode temporal tokenizer (Task C1)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.v14.config import TemporalTokenizerConfig
from speech_decoding.v14.dataset import T_SAMPLES
from speech_decoding.v14.tokenizer import TemporalTokenizer


class TestTemporalTokenizer:
    def test_default_kernel_stride_and_output_shape(self) -> None:
        tok = TemporalTokenizer()
        assert tok.kernel_size == 30 and tok.stride == 10
        x = torch.zeros(2, 4, T_SAMPLES)
        y = tok(x)
        assert y.shape == (2, 4, 64, 28)
        assert y.dtype == torch.float32

    def test_variable_N_e_per_forward(self) -> None:
        tok = TemporalTokenizer()
        y4 = tok(torch.zeros(1, 4, T_SAMPLES))
        y7 = tok(torch.zeros(1, 7, T_SAMPLES))
        assert y4.shape == (1, 4, 64, 28)
        assert y7.shape == (1, 7, 64, 28)

    def test_weights_shared_across_electrodes(self) -> None:
        """Same input on two electrodes must produce identical output slices
        — proves weight sharing (no per-electrode parameters)."""

        tok = TemporalTokenizer()
        trace = torch.randn(1, 1, T_SAMPLES)
        stacked = trace.expand(1, 4, T_SAMPLES)
        out = tok(stacked)
        # (1, 4, 64, 28) — all 4 electrode rows identical
        for i in range(1, 4):
            torch.testing.assert_close(out[0, 0], out[0, i])

    def test_parameter_count_matches_single_conv1d(self) -> None:
        tok = TemporalTokenizer()
        # nn.Conv1d(1 -> 64, kernel=30) has 1*64*30 weights + 64 biases
        expected = 1 * 64 * 30 + 64
        assert sum(p.numel() for p in tok.parameters()) == expected

    def test_backward_produces_gradients(self) -> None:
        tok = TemporalTokenizer()
        y = tok(torch.randn(2, 3, T_SAMPLES, requires_grad=False))
        y.sum().backward()
        assert tok.conv.weight.grad is not None
        assert not torch.isnan(tok.conv.weight.grad).any()

    def test_rejects_wrong_input_rank(self) -> None:
        tok = TemporalTokenizer()
        with pytest.raises(ValueError, match="B, N_e, T"):
            tok(torch.zeros(4, T_SAMPLES))

    def test_hard_asserts_kernel_stride_via_config(self) -> None:
        bad = TemporalTokenizerConfig(patch_ms=100, stride_ms=50, sample_rate_hz=200)
        with pytest.raises(ValueError, match="kernel=30 and stride=10"):
            TemporalTokenizer(bad)
