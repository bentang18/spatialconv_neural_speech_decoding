"""Per-phoneme model for the v14 baseline-aligned rework (plan P4).

Full stack:
    signal (B, N_e, 130)
    → grid-scatter (B, 1, H_p, W_p, 130)
    → Conv2d(1→8, k=3, pad=1) per time-step + GELU
    → masked-mean pool to (4, 8) cells → (B, 8, 32_cells, 130)
    → per-cell Conv1d(8→32, k=30, stride=10) → (B, 32_cells, 32_dim, 11_tokens)
    → + pooled_support @ P_emb[15, 32]  (broadcast across tokens)
    → flatten to (B, 352, 32) and run 3-block combined attention backbone
    → D1 decoder (mean-pool + prev_emb + Linear) → (B, 9)

Total params ~45k. Matches Ben's 0.734 baseline param budget.
"""

from __future__ import annotations

import torch
from torch import nn

from speech_decoding.v14.backbone import Backbone
from speech_decoding.v14.config import PerPhonemeConfig
from speech_decoding.v14.phoneme_decoder import D1Decoder
from speech_decoding.v14.pool import masked_mean_pool, precompute_pool


class NeuralFieldPerceiverPerPhoneme(nn.Module):
    """Per-phoneme model at `d_model=32`."""

    def __init__(self, cfg: PerPhonemeConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PerPhonemeConfig()
        d = self.cfg.d_model

        self.conv2d = nn.Conv2d(
            self.cfg.conv2d_in_channels,
            self.cfg.conv2d_out_channels,
            kernel_size=self.cfg.conv2d_kernel_size,
            padding=self.cfg.conv2d_padding,
        )
        self.conv2d_act = nn.GELU()

        t_cfg = self.cfg.per_cell_temporal
        if t_cfg.in_channels != self.cfg.conv2d_out_channels:
            raise ValueError(
                "per_cell_temporal.in_channels must equal conv2d_out_channels"
            )
        if t_cfg.out_channels != d:
            raise ValueError(
                f"per_cell_temporal.out_channels ({t_cfg.out_channels}) != "
                f"d_model ({d})"
            )
        self.conv1d = nn.Conv1d(
            t_cfg.in_channels,
            t_cfg.out_channels,
            kernel_size=t_cfg.kernel_size,
            stride=t_cfg.stride,
        )

        self.parcel_embedding = nn.Parameter(
            torch.empty(self.cfg.parcel_embedding_n_parcels, d)
        )
        nn.init.xavier_uniform_(self.parcel_embedding)

        if self.cfg.backbone.d_model != d:
            raise ValueError(
                f"backbone.d_model ({self.cfg.backbone.d_model}) != d_model ({d})"
            )
        self.backbone = Backbone(self.cfg.backbone)

        if self.cfg.decoder.d_model != d:
            raise ValueError(
                f"decoder.d_model ({self.cfg.decoder.d_model}) != d_model ({d})"
            )
        self.decoder = D1Decoder(self.cfg.decoder)

    def forward(self, batch: dict) -> torch.Tensor:
        """Return per-phoneme logits `(B, 9)` in train mode."""

        return self._forward_logits(batch)

    def encode_memory(self, batch: dict) -> torch.Tensor:
        """Run everything up to (but not including) the decoder.

        Exposed so exhaustive-AR eval can reuse one backbone pass across all
        9³ hypotheses for a given phoneme window. Returns `(B, S, d)`.
        """

        return self._forward_memory(batch)

    def decode(self, memory: torch.Tensor, prev_phoneme: torch.Tensor) -> torch.Tensor:
        return self.decoder(memory, prev_phoneme)

    def _forward_memory(self, batch: dict) -> torch.Tensor:
        signal: torch.Tensor = batch["signal"]               # (B, N_e, 130)
        layout: torch.Tensor = batch["electrode_grid_layout"]  # (B, N_e, 2) long
        grid_shape: tuple[int, int] = batch["electrode_grid_shape"]
        active: torch.Tensor = batch["electrode_active_mask"]  # (B, N_e) bool
        support: torch.Tensor = batch["support"]             # (B, N_e, 15) float32

        B, _, T_raw = signal.shape
        H_p, W_p = grid_shape

        # Batch-invariant pool precompute: layout/active are the same for every
        # element when batched per patient (#31), so derive from index 0.
        pp = precompute_pool(
            layout[0].to("cpu"),
            active[0].to("cpu"),
            grid_shape,
            self.cfg.pool.pool_shape,
        )
        cell_of_grid = pp.cell_of_grid.to(signal.device)
        cell_of_electrode = pp.cell_of_electrode.to(signal.device)
        grid_active = pp.grid_active.to(signal.device)
        active_count = pp.active_count.to(signal.device)
        cell_active_mask = pp.cell_active_mask.to(signal.device)
        n_cells = pp.n_cells

        # -- Stage 1: scatter to grid (B, 1, H_p, W_p, T_raw) --------------------
        grid = signal.new_zeros((B, 1, H_p, W_p, T_raw))
        rows = layout[0, :, 0]
        cols = layout[0, :, 1]
        # Zero-out contributions from inactive electrodes before writing.
        act_f = active.to(signal.dtype).unsqueeze(-1)  # (B, N_e, 1)
        grid[:, 0, rows, cols, :] = signal * act_f

        # -- Stage 2: Conv2d per time-step ---------------------------------------
        x = grid.permute(0, 4, 1, 2, 3).reshape(
            B * T_raw, 1, H_p, W_p
        )                                       # (B·T, 1, H_p, W_p)
        x = self.conv2d(x)                      # (B·T, 8, H_p, W_p)
        x = self.conv2d_act(x)
        Cout = x.shape[1]
        x = x.view(B, T_raw, Cout, H_p, W_p).permute(0, 2, 3, 4, 1)
        # (B, 8, H_p, W_p, T_raw)

        # -- Stage 3: masked-mean pool to (4, 8) cells --------------------------
        x_flat = x.reshape(B, Cout, H_p * W_p, T_raw)
        pooled = masked_mean_pool(
            x_flat, cell_of_grid, grid_active, active_count,
            dim=2, n_cells=n_cells,
        )                                       # (B, 8, n_cells, T_raw)

        # -- Stage 4: per-cell Conv1d (8→32, k=30, s=10) ------------------------
        # Treat each of the `n_cells` cells as an independent sequence; shared weights.
        y = pooled.permute(0, 2, 1, 3).reshape(B * n_cells, Cout, T_raw)
        y = self.conv1d(y)                      # (B·n_cells, d, T_tokens)
        T_tokens = y.shape[-1]
        y = y.view(B, n_cells, self.cfg.d_model, T_tokens)

        # -- Stage 5: + parcel embedding at d -----------------------------------
        # Same patient per batch (#31) → active[0] is the batch-invariant mask.
        pooled_support = masked_mean_pool(
            support, cell_of_electrode, active[0], active_count,
            dim=1, n_cells=n_cells,
        )                                       # (B, n_cells, 15)
        cell_emb = pooled_support @ self.parcel_embedding  # (B, n_cells, d)
        y = y + cell_emb.unsqueeze(-1)

        # -- Stage 6: flatten + combined-attention backbone ---------------------
        # Cell-major flatten: idx = cell * T_tokens + t.
        # Active mask broadcasts cell_active_mask across T_tokens.
        y_flat = y.permute(0, 1, 3, 2).reshape(B, n_cells * T_tokens, self.cfg.d_model)
        active_flat = (
            cell_active_mask.unsqueeze(-1)
            .expand(n_cells, T_tokens)
            .reshape(-1)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
        time_ids = (
            torch.arange(T_tokens, device=y_flat.device)
            .unsqueeze(0)
            .expand(n_cells, -1)
            .reshape(-1)
        )

        memory = self.backbone(y_flat, active_flat, time_ids)
        return memory

    def _forward_logits(self, batch: dict) -> torch.Tensor:
        memory = self._forward_memory(batch)
        return self.decoder(memory, batch["prev_tokens"])
