"""Tests for the D1 minimal AR decoder (plan P4 Stage 7)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.v14.config import D1DecoderConfig
from speech_decoding.v14.phoneme_decoder import BOS_SENTINEL, D1Decoder


class TestD1Decoder:
    def test_output_shape(self) -> None:
        dec = D1Decoder(D1DecoderConfig(d_model=32))
        mem = torch.randn(4, 352, 32)
        prev = torch.tensor([0, 3, BOS_SENTINEL, 8], dtype=torch.long)
        out = dec(mem, prev)
        assert out.shape == (4, 9)

    def test_bos_index_distinct_from_phoneme_indices(self) -> None:
        """prev=-1 (BOS) and prev=0 (AA) must produce different embeddings."""
        torch.manual_seed(0)
        dec = D1Decoder(D1DecoderConfig(d_model=32))
        mem = torch.zeros(2, 4, 32)  # mean-pool → 0
        prev_bos = torch.tensor([BOS_SENTINEL, BOS_SENTINEL], dtype=torch.long)
        prev_aa = torch.tensor([0, 0], dtype=torch.long)
        out_bos = dec(mem, prev_bos)
        out_aa = dec(mem, prev_aa)
        assert not torch.allclose(out_bos, out_aa)

    def test_param_count_is_small(self) -> None:
        """Plan target: ~617 params for D1."""
        dec = D1Decoder(D1DecoderConfig(d_model=32))
        total = sum(p.numel() for p in dec.parameters())
        # 10*32 (prev_emb) + 32*9 + 9 (Linear) = 320 + 297 = 617
        assert total == 617

    def test_rejects_mismatched_d_model(self) -> None:
        dec = D1Decoder(D1DecoderConfig(d_model=32))
        with pytest.raises(ValueError, match="memory shape"):
            dec(torch.randn(2, 10, 16), torch.tensor([0, 1], dtype=torch.long))

    def test_rejects_mismatched_batch(self) -> None:
        dec = D1Decoder(D1DecoderConfig(d_model=32))
        with pytest.raises(ValueError, match="prev_phoneme shape"):
            dec(torch.randn(2, 10, 32), torch.tensor([0, 1, 2], dtype=torch.long))
