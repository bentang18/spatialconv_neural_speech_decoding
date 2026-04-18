"""Unit tests for the 3-slot AR decoder (Task C4)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.v14.config import BackboneConfig, DecoderConfig
from speech_decoding.v14.decoder import ARDecoder


class TestBuildQueries:
    def test_bos_is_separate_parameter_not_vocab_row(self) -> None:
        dec = ARDecoder()
        # prev_emb is vocab-sized (9 rows), BOS is a dedicated parameter
        assert dec.prev_emb.weight.shape == (9, 64)
        assert dec.bos_emb.shape == (64,)

    def test_bos_sentinel_replaces_prev_emb(self) -> None:
        dec = ARDecoder().eval()
        # prev_tokens with BOS in all slots
        prev = torch.full((2, 3), -1, dtype=torch.long)
        q_bos = dec._build_queries(prev)
        # Every slot's prev contribution should equal bos_emb
        expected_bos = dec.bos_emb.view(1, 1, -1).expand(2, 3, -1)
        base_plus_slot = dec.base_query.view(1, 1, -1) + dec.slot_emb.unsqueeze(0)
        torch.testing.assert_close(q_bos, base_plus_slot + expected_bos)

    def test_rejects_wrong_prev_tokens_shape(self) -> None:
        dec = ARDecoder()
        with pytest.raises(ValueError, match=r"\(B, 3\)"):
            dec._build_queries(torch.zeros(2, 2, dtype=torch.long))


class TestForwardTeacher:
    def test_output_shape_is_B_3_9(self) -> None:
        dec = ARDecoder().eval()
        B, M, d = 2, 12, 64
        memory = torch.randn(B, M, d)
        prev = torch.tensor([[-1, 2, 0], [-1, 3, 7]], dtype=torch.long)
        logits = dec.forward_teacher(memory, prev)
        assert logits.shape == (B, 3, 9)

    def test_backward_finite_gradients(self) -> None:
        dec = ARDecoder()
        dec.train()
        memory = torch.randn(1, 8, 64, requires_grad=True)
        prev = torch.tensor([[-1, 2, 0]], dtype=torch.long)
        logits = dec.forward_teacher(memory, prev)
        logits.sum().backward()
        assert memory.grad is not None and torch.isfinite(memory.grad).all()
        for name, p in dec.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"

    def test_rejects_d_model_mismatch_with_backbone(self) -> None:
        with pytest.raises(ValueError, match="d_model"):
            ARDecoder(DecoderConfig(d_model=32), BackboneConfig(d_model=64))


class TestExhaustiveDecode:
    def test_output_shape_and_range(self) -> None:
        dec = ARDecoder().eval()
        memory = torch.randn(2, 10, 64)
        with torch.no_grad():
            out = dec.exhaustive_decode(memory)
        assert out.shape == (2, 3)
        assert out.dtype == torch.long
        assert int(out.min()) >= 0 and int(out.max()) < 9

    def test_argmax_matches_hand_scored_sweep(self) -> None:
        """exhaustive_decode's pick matches a brute-force log-prob sweep."""

        dec = ARDecoder().eval()
        torch.manual_seed(0)
        memory = torch.randn(1, 6, 64)
        with torch.no_grad():
            chosen = dec.exhaustive_decode(memory)[0]
            # Brute-force: score every hypothesis via forward_teacher directly
            V, L = 9, 3
            seqs = torch.cartesian_prod(*([torch.arange(V)] * L))
            prev = torch.full((seqs.shape[0], L), -1, dtype=torch.long)
            prev[:, 1:] = seqs[:, :-1]
            logits = dec.forward_teacher(memory.expand(seqs.shape[0], -1, -1), prev)
            lp = torch.log_softmax(logits, dim=-1).gather(
                2, seqs.unsqueeze(-1)
            ).squeeze(-1)
            brute = seqs[lp.sum(dim=1).argmax()]
        assert torch.equal(chosen, brute)

    def test_rejects_wrong_memory_rank(self) -> None:
        dec = ARDecoder().eval()
        with pytest.raises(ValueError, match=r"\(B, M, d\)"):
            dec.exhaustive_decode(torch.zeros(12, 64))
