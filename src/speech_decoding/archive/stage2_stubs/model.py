"""Top-level Neural Field Perceiver v14 (B-1) assembly for v14-core (Task C5).

Composes the five Phase-1 stages:

    tokenizer          → (B, N_e, d, 28)
    grid_mixer         → (B, N_e, d, 28)
    soft_parcel_embed  → (B, N_e, d), broadcast across T=28
    backbone           → (B, N_e × 28, d)
    decoder            → train: (B, 3, 9)  |  eval: (B, 3) long

`forward(batch, mode)` consumes the keys produced by `collate_v14_batch`:
`signal`, `electrode_grid_layout`, `electrode_grid_shape`,
`electrode_active_mask`, `support`, `prev_tokens`.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from speech_decoding.v14.backbone import Backbone
from speech_decoding.v14.config import V14Config
from speech_decoding.v14.decoder import ARDecoder
from speech_decoding.v14.grid_mixer import GridMixer
from speech_decoding.v14.parcel_embedding import SoftParcelEmbedding
from speech_decoding.v14.tokenizer import TemporalTokenizer


Mode = Literal["train", "eval"]


class NeuralFieldPerceiver(nn.Module):
    """End-to-end v14-core model (B-1 amendment)."""

    def __init__(self, cfg: V14Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg or V14Config()
        self.tokenizer = TemporalTokenizer(self.cfg.tokenizer)
        self.grid_mixer = GridMixer(self.cfg.grid_mixer)
        self.parcel_embedding = SoftParcelEmbedding(self.cfg.parcel_embedding)
        self.backbone = Backbone(self.cfg.backbone)
        self.decoder = ARDecoder(self.cfg.decoder, self.cfg.backbone)

    def _encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        signal: torch.Tensor = batch["signal"]
        electrode_grid_layout: torch.Tensor = batch["electrode_grid_layout"]
        electrode_grid_shape: tuple[int, int] = batch["electrode_grid_shape"]
        electrode_active_mask: torch.Tensor = batch["electrode_active_mask"]
        support: torch.Tensor = batch["support"]

        temporal = self.tokenizer(signal)  # (B, N_e, d, T)
        mixed = self.grid_mixer(
            temporal, electrode_grid_layout, electrode_grid_shape, electrode_active_mask
        )
        parcel_emb = self.parcel_embedding(support)  # (B, N_e, d)
        enriched = mixed + parcel_emb.unsqueeze(-1)  # broadcast across T

        B, N_e, d, T = enriched.shape
        tokens = enriched.permute(0, 1, 3, 2).reshape(B, N_e * T, d)
        token_active_mask = (
            electrode_active_mask.unsqueeze(-1).expand(B, N_e, T).reshape(B, N_e * T)
        )
        time_ids = torch.arange(T, device=signal.device).repeat(N_e)
        memory = self.backbone(tokens, token_active_mask, time_ids)
        return memory, token_active_mask

    def forward(
        self, batch: dict, mode: Mode = "train"
    ) -> torch.Tensor:
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
        memory, _ = self._encode(batch)
        if mode == "train":
            return self.decoder.forward_teacher(memory, batch["prev_tokens"])
        return self.decoder.exhaustive_decode(memory)
