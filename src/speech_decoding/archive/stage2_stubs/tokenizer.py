"""Shared per-electrode temporal tokenizer for v14-core (Task C1).

Frozen by `#2` / `#6`: one Conv1d applied to every electrode's z-scored
high-gamma trace with shared weights. Weight sharing is implemented by
folding `N_e` into the batch dim.

    input:  (B, N_e, T=300)   float32 at 200 Hz (#29)
    output: (B, N_e, d, 28)   d = cfg.d_model

Kernel `30` samples (150 ms) and stride `10` samples (50 ms) are hard
defaults from `TemporalTokenizerConfig` (#2). `N_e` is variable per
forward call.
"""

from __future__ import annotations

import torch
from torch import nn

from speech_decoding.v14.config import TemporalTokenizerConfig


class TemporalTokenizer(nn.Module):
    """Shared per-electrode Conv1d patch tokenizer."""

    def __init__(self, cfg: TemporalTokenizerConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or TemporalTokenizerConfig()
        kernel = cfg.patch_ms * cfg.sample_rate_hz // 1000
        stride = cfg.stride_ms * cfg.sample_rate_hz // 1000
        if kernel != 30 or stride != 10:
            raise ValueError(
                f"#2/#6 freezes kernel=30 and stride=10 at 200 Hz; got "
                f"kernel={kernel}, stride={stride}"
            )
        self.d_model = cfg.d_model
        self.kernel_size = kernel
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=cfg.d_model,
            kernel_size=kernel,
            stride=stride,
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3:
            raise ValueError(f"expected (B, N_e, T), got {tuple(signal.shape)}")
        B, N_e, T = signal.shape
        flat = signal.reshape(B * N_e, 1, T)
        out = self.conv(flat)
        return out.reshape(B, N_e, self.d_model, out.shape[-1])
