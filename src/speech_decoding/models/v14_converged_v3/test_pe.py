"""v14_converged_v3 Phase 3 — positional encoding (TDD).

Two mechanisms, disjoint by design (memo project-v14-converged-v3-sensor-arch):

  L1RoPE          — within-sensor JOINT spatiotemporal rotary. EQUAL split of
                    head_dim across (contact-index, time), standard freq schedule
                    per axis. Its load-bearing property is RELATIVITY: the q·k
                    score depends only on (Δindex, Δtime), never absolute position
                    — which is exactly why a global per-shaft depth flip / offset
                    needs no correction.
  ParcelIdentity  — L2 cross-sensor identity: a LEARNED embedding indexed by the
                    DKT hard tag (cross-subject-meaningful), added to tokens. NO
                    geometric RoPE across sensors (no shared metric).
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v3.pe import (
    L1RoPE,
    ParcelIdentityEmbed,
)


def _qk(heads: int, seq: int, hd: int):
    torch.manual_seed(0)
    q = torch.randn(1, heads, seq, hd)
    k = torch.randn(1, heads, seq, hd)
    return q, k


def _score(rope: L1RoPE, q, k, idx, t):
    qr, kr = rope(q, k, idx, t)
    return torch.einsum("bhid,bhjd->bhij", qr, kr)


def test_head_dim_must_be_divisible_by_four() -> None:
    L1RoPE(head_dim=64)  # ok
    with pytest.raises(ValueError):
        L1RoPE(head_dim=62)  # 62 % 4 != 0


def test_rope_preserves_shape() -> None:
    rope = L1RoPE(head_dim=64)
    q, k = _qk(4, 6, 64)
    idx = torch.arange(6).unsqueeze(0)
    t = torch.arange(6).unsqueeze(0)
    qr, kr = rope(q, k, idx, t)
    assert qr.shape == q.shape and kr.shape == k.shape


def test_score_is_translation_invariant_on_both_axes() -> None:
    # The RoPE relativity property: shifting index and/or time by a constant
    # leaves every pairwise q·k score unchanged.
    rope = L1RoPE(head_dim=64)
    q, k = _qk(2, 5, 64)
    idx = torch.tensor([[0, 1, 2, 4, 5]])  # note the gap (dropped contact 3)
    t = torch.tensor([[0, 1, 2, 3, 4]])
    base = _score(rope, q, k, idx, t)
    for da, db in [(3, 0), (0, 7), (3, 7), (-2, 4)]:
        shifted = _score(rope, q, k, idx + da, t + db)
        assert torch.allclose(base, shifted, atol=1e-4), f"shift {(da, db)} changed score"


def test_score_is_not_position_degenerate() -> None:
    # Different relative offsets must give different scores (RoPE actually encodes
    # position, not a no-op).
    rope = L1RoPE(head_dim=64)
    q, k = _qk(1, 3, 64)
    t = torch.zeros(1, 3, dtype=torch.long)
    s_close = _score(rope, q, k, torch.tensor([[0, 1, 2]]), t)
    s_far = _score(rope, q, k, torch.tensor([[0, 10, 20]]), t)
    assert not torch.allclose(s_close, s_far, atol=1e-3)


def test_spatial_and_temporal_axes_are_independent() -> None:
    # Varying only time (index fixed) must leave the score identical to a pure
    # temporal-only rotation, and vice-versa — the two axes occupy disjoint
    # head_dim halves.
    rope = L1RoPE(head_dim=64)
    q, k = _qk(1, 4, 64)
    idx0 = torch.zeros(1, 4, dtype=torch.long)
    t = torch.tensor([[0, 1, 2, 3]])
    # Shift index alone while time fixed: with identical index, the spatial half
    # contributes a constant (Δindex=0) so only the temporal half varies.
    s_a = _score(rope, q, k, idx0, t)
    s_b = _score(rope, q, k, idx0 + 5, t)  # still all-equal index → Δindex=0
    assert torch.allclose(s_a, s_b, atol=1e-4)


def test_default_base_makes_every_pair_position_live() -> None:
    # Audit L12/#8: base must fit OUR ranges, not the LLM-inherited 10000. A pair
    # with frequency f turns R·f radians across a range R; a pair turning <<1 rad
    # end-to-end is position-DEAD. The default base_index/base_time must leave EVERY
    # rotary pair turning a meaningful angle across index (~20 contacts) and time
    # (128 slots) — and the old 10000 must NOT (regression witness).
    R_index, R_time = 20.0, 128.0
    rope = L1RoPE(head_dim=64)  # encoder width; pairs = 16 per axis
    idx_turn = R_index * rope.idx_freq  # (pairs,) radians across the index range
    t_turn = R_time * rope.t_freq
    assert (idx_turn >= 1.0).all(), f"dead index pairs: {idx_turn.tolist()}"
    assert (t_turn >= 1.0).all(), f"dead time pairs: {t_turn.tolist()}"
    # lowest-freq pair stays under a full turn (≤2π) ⇒ no wrap-around ambiguity
    assert idx_turn.min() <= 2 * torch.pi and t_turn.min() <= 2 * torch.pi

    dead = L1RoPE(head_dim=64, base_index=10_000.0, base_time=10_000.0)
    assert (R_index * dead.idx_freq < 1.0).sum() >= 10  # most index pairs were dead
    assert (R_time * dead.t_freq < 1.0).sum() >= 6


def test_cos_sin_and_rotate_reconstruct_forward_exactly() -> None:
    # B5 hoist mechanism: forward(q,k,idx,t) must be BIT-IDENTICAL to
    # rotate(q,k,*cos_sin(idx,t)). This is what lets the tower build the table ONCE
    # and feed every L1 block instead of each block recomputing it.
    rope = L1RoPE(head_dim=64)
    q, k = _qk(4, 6, 64)
    idx = torch.tensor([[0, 1, 2, 4, 5, 6]])
    t = torch.tensor([[0, 1, 2, 3, 4, 5]])
    q_ref, k_ref = rope(q, k, idx, t)
    cos, sin = rope.cos_sin(idx, t)
    q_hoist, k_hoist = rope.rotate(q, k, cos, sin)
    assert torch.equal(q_ref, q_hoist)  # exact, not allclose
    assert torch.equal(k_ref, k_hoist)


def test_cos_sin_is_fp32_regardless_of_input_dtype() -> None:
    # The table must stay fp32 (the rope math runs in fp32, result down-cast AFTER)
    # — a bf16 table would change the rounding and break equivalence.
    rope = L1RoPE(head_dim=64)
    idx = torch.tensor([[0, 1, 2]])
    t = torch.tensor([[0, 1, 2]])
    cos, sin = rope.cos_sin(idx, t)
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32


def test_shared_table_matches_per_instance_recompute() -> None:
    # Two L1RoPE instances with the same head_dim/bases produce the SAME table, so a
    # table built by block[0].rope applied via block[1].rope.rotate is exact — the
    # invariant the tower relies on to share one table across all L1 blocks.
    a, b = L1RoPE(head_dim=64), L1RoPE(head_dim=64)
    q, k = _qk(2, 5, 64)
    idx = torch.tensor([[0, 1, 2, 3, 4]])
    t = torch.tensor([[0, 1, 2, 3, 4]])
    cos, sin = a.cos_sin(idx, t)  # built by instance a
    q_shared, k_shared = b.rotate(q, k, cos, sin)  # applied via instance b
    q_ref, k_ref = b(q, k, idx, t)  # b recomputes its own
    assert torch.equal(q_ref, q_shared)
    assert torch.equal(k_ref, k_shared)


def test_parcel_identity_shape_and_indexing() -> None:
    emb = ParcelIdentityEmbed(n_parcels=74, d_model=256)
    parcel_id = torch.tensor([[3, 3, 7, 0]])
    out = emb(parcel_id)
    assert out.shape == (1, 4, 256)
    # same parcel id → identical vector; different id → different.
    assert torch.allclose(out[0, 0], out[0, 1])
    assert not torch.allclose(out[0, 0], out[0, 2])


def test_parcel_identity_is_single_learned_embedding() -> None:
    emb = ParcelIdentityEmbed(n_parcels=74, d_model=256)
    embeds = [m for m in emb.modules() if isinstance(m, torch.nn.Embedding)]
    assert len(embeds) == 1
    assert embeds[0].weight.shape == (74, 256)
    assert embeds[0].weight.requires_grad  # learned
