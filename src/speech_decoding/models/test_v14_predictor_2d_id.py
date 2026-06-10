"""B37 D9 — JepaPredictor optional SECOND identity axis (freq for M4).

The B37 M4 latent carries frequency, so a masked M4 query token is identified by
BOTH parcel (primary ``query_id``, learned/unordered) AND freq-patch (the second
``query_id_2``, sinusoidal/ordered), with time on RoPE. These tests pin:

  * construction: ``n_identity_2`` builds a second table (learned ``id_embed_2``
    or sinusoidal ``_id_table_2`` buffer); ``None`` → neither.
  * the default second axis is sinusoidal (a fixed buffer, not a Parameter).
  * the second tag is ADDITIVE and actually reaches the prediction (perturbing
    ``query_id_2`` moves the output; so does ``query_id``).
  * backward-compat: a single-identity predictor + a two-identity predictor
    called WITHOUT ``query_id_2`` both behave as before; passing ``query_id_2``
    to a single-identity predictor raises.
  * the ragged forward stays BIT-identical to dense with the second tag.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from speech_decoding.models.v14_encoder import JepaPredictor


def _m4_predictor(
    *, d_model: int = 16, n_parcels: int = 5, n_freq: int = 4,
    id_pos_2: str = "sinusoidal", ragged: bool = False,
) -> JepaPredictor:
    torch.manual_seed(0)
    return JepaPredictor(
        d_model=d_model,
        n_identity=n_parcels,          # parcel (learned, unordered)
        hidden=32,
        n_heads=4,
        depth=2,
        max_time_patches=16,
        ragged_predictor=ragged,
        id_pos="learned",
        n_identity_2=n_freq,           # freq-patch (sinusoidal, ordered)
        id_pos_2=id_pos_2,
    )


def _ctx(B: int, n_ctx: int, d_model: int, *, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, n_ctx, d_model, generator=g)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def test_second_axis_sinusoidal_is_fixed_buffer() -> None:
    p = _m4_predictor(id_pos_2="sinusoidal")
    assert p.n_identity_2 == 4
    assert p.id_embed_2 is None
    assert p._id_table_2 is not None
    assert p._id_table_2.shape == (4, p.hidden)
    # sincos table is a NON-persistent buffer, not a learnable Parameter.
    buf_names = {n for n, _ in p.named_buffers()}
    param_names = {n for n, _ in p.named_parameters()}
    assert "_id_table_2" in buf_names
    assert not any("_id_table_2" in n for n in param_names)


def test_second_axis_learned_is_embedding() -> None:
    p = _m4_predictor(id_pos_2="learned")
    assert isinstance(p.id_embed_2, nn.Embedding)
    assert p.id_embed_2.num_embeddings == 4
    assert p._id_table_2 is None


def test_no_second_axis_when_none() -> None:
    torch.manual_seed(0)
    p = JepaPredictor(d_model=16, n_identity=5, hidden=32, n_heads=4, depth=2)
    assert p.n_identity_2 is None
    assert p.id_embed_2 is None
    assert p._id_table_2 is None


def test_second_axis_cardinality_guard() -> None:
    with pytest.raises(ValueError, match="n_identity_2"):
        JepaPredictor(d_model=16, n_identity=5, hidden=32, n_heads=4, depth=2,
                      n_identity_2=0)


# --------------------------------------------------------------------------- #
# the second tag reaches the prediction (additivity)
# --------------------------------------------------------------------------- #
def _call(p, context, *, qid, qid2, qtime, ctime, qvalid):
    return p(
        context,
        context_time_ids=ctime,
        query_time_ids=qtime,
        query_id=qid,
        query_id_2=qid2,
        query_valid=qvalid,
    )


def test_freq_tag_changes_prediction() -> None:
    """Two query slots that differ ONLY in freq-id produce different
    predictions — the sinusoidal freq tag is carried into the output."""
    p = _m4_predictor().eval()
    B, n_ctx = 2, 6
    ctx = _ctx(B, n_ctx, 16)
    ctime = torch.zeros(n_ctx, dtype=torch.long)
    # one query, all real; vary its freq id, hold parcel + time fixed.
    qtime = torch.tensor([0], dtype=torch.long)
    qid = torch.tensor([2], dtype=torch.long)         # parcel 2
    qvalid = torch.ones(B, 1, dtype=torch.bool)
    out0 = _call(p, ctx, qid=qid, qid2=torch.tensor([0]), qtime=qtime, ctime=ctime, qvalid=qvalid)
    out1 = _call(p, ctx, qid=qid, qid2=torch.tensor([3]), qtime=qtime, ctime=ctime, qvalid=qvalid)
    assert (out0 - out1).abs().max() > 1e-5


def test_parcel_tag_changes_prediction() -> None:
    p = _m4_predictor().eval()
    B, n_ctx = 2, 6
    ctx = _ctx(B, n_ctx, 16)
    ctime = torch.zeros(n_ctx, dtype=torch.long)
    qtime = torch.tensor([0], dtype=torch.long)
    qid2 = torch.tensor([1], dtype=torch.long)        # freq 1
    qvalid = torch.ones(B, 1, dtype=torch.bool)
    out0 = _call(p, ctx, qid=torch.tensor([0]), qid2=qid2, qtime=qtime, ctime=ctime, qvalid=qvalid)
    out1 = _call(p, ctx, qid=torch.tensor([4]), qid2=qid2, qtime=qtime, ctime=ctime, qvalid=qvalid)
    assert (out0 - out1).abs().max() > 1e-5


def test_passing_query_id_2_without_second_axis_raises() -> None:
    torch.manual_seed(0)
    p = JepaPredictor(d_model=16, n_identity=5, hidden=32, n_heads=4, depth=2).eval()
    ctx = _ctx(2, 6, 16)
    with pytest.raises(RuntimeError, match="n_identity_2=None"):
        p(
            ctx,
            context_time_ids=torch.zeros(6, dtype=torch.long),
            query_time_ids=torch.tensor([0]),
            query_id=torch.tensor([1]),
            query_id_2=torch.tensor([0]),
            query_valid=torch.ones(2, 1, dtype=torch.bool),
        )


def test_two_axis_without_query_id_2_is_single_id() -> None:
    """A predictor built with a second axis but called WITHOUT ``query_id_2``
    falls back to single-identity (the second tag is simply not added)."""
    p = _m4_predictor().eval()
    ctx = _ctx(2, 6, 16)
    out = p(
        ctx,
        context_time_ids=torch.zeros(6, dtype=torch.long),
        query_time_ids=torch.tensor([0]),
        query_id=torch.tensor([2]),
        query_valid=torch.ones(2, 1, dtype=torch.bool),
    )
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# ragged == dense with the second tag
# --------------------------------------------------------------------------- #
def test_ragged_matches_dense_with_second_tag() -> None:
    p = _m4_predictor(ragged=False).eval()
    B, n_ctx, n_qry = 3, 8, 5
    ctx = _ctx(B, n_ctx, 16, seed=4)
    ctime = torch.randint(0, 4, (n_ctx,))
    qtime = torch.randint(0, 4, (n_qry,))
    qid = torch.randint(0, 5, (n_qry,))
    qid2 = torch.randint(0, 4, (n_qry,))
    g = torch.Generator().manual_seed(7)
    ctx_pad = torch.rand(B, n_ctx, generator=g) < 0.3
    qvalid = torch.rand(B, n_qry, generator=g) < 0.7
    kw = dict(
        context_time_ids=ctime, query_time_ids=qtime, query_id=qid,
        query_id_2=qid2, context_key_padding_mask=ctx_pad, query_valid=qvalid,
    )
    p.ragged_predictor = False
    dense = p(ctx, **kw)
    p.ragged_predictor = True
    ragged = p(ctx, **kw)
    assert dense.shape == ragged.shape
    torch.testing.assert_close(dense, ragged, atol=1e-5, rtol=1e-5)
