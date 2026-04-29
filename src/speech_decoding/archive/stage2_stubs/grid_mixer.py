"""Whole-grid Conv2d spatial mixer (Stage 2 of the B-1 architecture, Task C2).

Replaces the deprecated within-parcel Perceiver summarizer (`#26`). Per the
2026-04-16 amendment, Stage 2 is a per-time-step 2D convolution over the
patient's bounding-box grid `(H_p, W_p)` with shared weights across `t`.

    tokens:                (B, N_e, d, T)   from the temporal tokenizer
    electrode_grid_layout: (B, N_e, 2)       long, (row, col)
    electrode_grid_shape:  (H_p, W_p)        shared across the batch (#31)
    electrode_active_mask: (B, N_e)          bool, non-artifact AND non-pad

    output:                (B, N_e, d, T)

Per block, pre-norm LayerNorm over the channel dim, then `Conv2d(d -> d,
k=3, s=1, p=1, bias=True)`, GELU, masked to zero at pad/artifact cells,
added as a residual to the scattered grid. `num_layers` = 1 baseline, 2
ablation (A3).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from speech_decoding.v14.config import GridMixerConfig


class GridMixer(nn.Module):
    """Per-time-step 2D Conv residual block over the patient grid."""

    def __init__(self, cfg: GridMixerConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or GridMixerConfig()
        if cfg.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {cfg.num_layers}")
        self.d_model = cfg.d_model
        self.num_layers = cfg.num_layers
        self.norms = nn.ModuleList(
            [nn.LayerNorm(cfg.d_model) for _ in range(cfg.num_layers)]
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    cfg.d_model,
                    cfg.d_model,
                    kernel_size=cfg.kernel_size,
                    stride=cfg.stride,
                    padding=cfg.padding,
                    bias=True,
                )
                for _ in range(cfg.num_layers)
            ]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        electrode_grid_layout: torch.Tensor,
        electrode_grid_shape: tuple[int, int],
        electrode_active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(f"expected (B, N_e, d, T), got {tuple(tokens.shape)}")
        B, N_e, d, T = tokens.shape
        if d != self.d_model:
            raise ValueError(f"tokens channel dim {d} != d_model {self.d_model}")
        H_p, W_p = electrode_grid_shape
        HW = H_p * W_p
        rows = electrode_grid_layout[..., 0]
        cols = electrode_grid_layout[..., 1]
        flat_idx = rows * W_p + cols  # (B, N_e) long

        active = electrode_active_mask.to(tokens.dtype)  # (B, N_e)

        # Scatter tokens onto a (B, d, T, HW) zero grid. Inactive electrodes
        # contribute zero; non-electrode cells remain zero.
        masked = tokens * active.view(B, N_e, 1, 1)
        masked_perm = masked.permute(0, 2, 3, 1).contiguous()  # (B, d, T, N_e)
        grid = masked.new_zeros(B, d, T, HW)
        scatter_idx = flat_idx.view(B, 1, 1, N_e).expand(B, d, T, N_e)
        grid.scatter_(3, scatter_idx, masked_perm)

        # Active-cell mask on the grid: 1 where a non-artifact electrode sits.
        cell_mask = active.new_zeros(B, HW)
        cell_mask.scatter_(1, flat_idx, active)
        cell_mask = cell_mask.view(B, 1, H_p, W_p)
        cell_mask_bt = (
            cell_mask.unsqueeze(1).expand(B, T, 1, H_p, W_p).reshape(B * T, 1, H_p, W_p)
        )

        # Fold T into batch: (B, d, T, HW) -> (B*T, d, H_p, W_p)
        x = grid.view(B, d, T, H_p, W_p).permute(0, 2, 1, 3, 4).reshape(B * T, d, H_p, W_p)

        for norm, conv in zip(self.norms, self.convs):
            # LayerNorm over channel dim: move channel to last.
            h = x.permute(0, 2, 3, 1)  # (B*T, H_p, W_p, d)
            h = norm(h)
            h = h.permute(0, 3, 1, 2)  # (B*T, d, H_p, W_p)
            h = conv(h)
            h = F.gelu(h)
            h = h * cell_mask_bt
            x = x + h

        # Reshape back and gather to (B, N_e, d, T).
        x = x.view(B, T, d, H_p, W_p).permute(0, 2, 3, 4, 1).contiguous()  # (B, d, H_p, W_p, T)
        x = x.view(B, d, HW, T)
        gather_idx = flat_idx.view(B, 1, N_e, 1).expand(B, d, N_e, T)
        gathered = x.gather(2, gather_idx)  # (B, d, N_e, T)
        return gathered.permute(0, 2, 1, 3).contiguous()  # (B, N_e, d, T)
