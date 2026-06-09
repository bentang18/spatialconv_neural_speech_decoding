"""#94 — JepaPredictor visible-only (ragged) context+query drop.

The V-JEPA-2 predictor should consume ONLY the visible context cells and ONLY
the real masked query slots, dropping the padded positions instead of feeding
the full grid and key-padding the rest. Because a padded key is masked out of
the SDPA softmax (``exp(NEG_INF)`` underflows to exactly 0 in float32), the kept
tokens' attention — and hence every real prediction — is unchanged. These tests
pin that the ragged forward is BIT-identical (up to ~1e-6 matmul reassociation,
the #91/#93 standard) to the dense key-padded forward, for both the P1
(freq-patch identity) and P2 (parcel-slot identity) call patterns, and that the
NaN guard holds for fully-padded rows.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import JepaPredictor


def _make_predictor(*, d_model: int, n_identity: int, max_time: int) -> JepaPredictor:
    torch.manual_seed(0)
    return JepaPredictor(
        d_model=d_model,
        n_identity=n_identity,
        hidden=32,
        n_heads=4,
        depth=2,
        max_time_patches=max_time,
    )


def _run_both(pred, context, *, ctx_time, qry_time, qry_id, ctx_pad, qry_valid):
    """Return (dense_out, ragged_out) for the SAME predictor weights."""
    pred.ragged_predictor = False
    dense = pred(
        context,
        context_time_ids=ctx_time,
        query_time_ids=qry_time,
        query_id=qry_id,
        context_key_padding_mask=ctx_pad,
        query_valid=qry_valid,
    )
    pred.ragged_predictor = True
    ragged = pred(
        context,
        context_time_ids=ctx_time,
        query_time_ids=qry_time,
        query_id=qry_id,
        context_key_padding_mask=ctx_pad,
        query_valid=qry_valid,
    )
    return dense, ragged


def test_ragged_predictor_default_off() -> None:
    pred = _make_predictor(d_model=16, n_identity=4, max_time=5)
    assert pred.ragged_predictor is False


def test_p1_pattern_bit_identical() -> None:
    """P1: identity axis = freq-patch (n_identity = F_p)."""
    B, N, d, F_p, max_time = 3, 12, 16, 4, 6
    pred = _make_predictor(d_model=d, n_identity=F_p, max_time=max_time)
    torch.manual_seed(1)
    context = torch.randn(B, N, d)
    ctx_time = torch.randint(0, max_time, (N,))
    qry_time = torch.randint(0, max_time, (N,))
    qry_id = torch.randint(0, F_p, (N,))
    # Per-row visibility/mask: some context padded, some query slots real.
    ctx_pad = torch.rand(B, N) < 0.4            # True = ignore
    qry_valid = torch.rand(B, N) < 0.5          # True = real masked slot

    dense, ragged = _run_both(
        pred, context, ctx_time=ctx_time, qry_time=qry_time,
        qry_id=qry_id, ctx_pad=ctx_pad, qry_valid=qry_valid,
    )
    assert dense.shape == ragged.shape == (int(qry_valid.sum()), d)
    torch.testing.assert_close(dense, ragged, rtol=1e-5, atol=1e-5)


def test_p2_pattern_bit_identical() -> None:
    """P2: identity axis = parcel-slot (n_identity = K·M), larger grid."""
    B, N, d, n_id, max_time = 2, 20, 24, 8, 5
    pred = _make_predictor(d_model=d, n_identity=n_id, max_time=max_time)
    torch.manual_seed(2)
    context = torch.randn(B, N, d)
    ctx_time = torch.randint(0, max_time, (N,))
    qry_time = torch.randint(0, max_time, (N,))
    qry_id = torch.randint(0, n_id, (N,))
    ctx_pad = torch.rand(B, N) < 0.3
    qry_valid = torch.rand(B, N) < 0.6

    dense, ragged = _run_both(
        pred, context, ctx_time=ctx_time, qry_time=qry_time,
        qry_id=qry_id, ctx_pad=ctx_pad, qry_valid=qry_valid,
    )
    torch.testing.assert_close(dense, ragged, rtol=1e-5, atol=1e-5)


def test_no_visible_context_row_bit_identical() -> None:
    """A row with ALL context padded but real query slots: predict from pure
    mask tokens. Both paths un-pad nothing on the context side; ragged keeps
    only the query slots. Must stay finite and match dense."""
    B, N, d, n_id, max_time = 3, 10, 16, 4, 5
    pred = _make_predictor(d_model=d, n_identity=n_id, max_time=max_time)
    torch.manual_seed(3)
    context = torch.randn(B, N, d)
    ctx_time = torch.randint(0, max_time, (N,))
    qry_time = torch.randint(0, max_time, (N,))
    qry_id = torch.randint(0, n_id, (N,))
    ctx_pad = torch.ones(B, N, dtype=torch.bool)     # all context padded
    qry_valid = torch.zeros(B, N, dtype=torch.bool)
    qry_valid[:, :3] = True                          # 3 real query slots / row

    dense, ragged = _run_both(
        pred, context, ctx_time=ctx_time, qry_time=qry_time,
        qry_id=qry_id, ctx_pad=ctx_pad, qry_valid=qry_valid,
    )
    assert torch.isfinite(ragged).all()
    torch.testing.assert_close(dense, ragged, rtol=1e-5, atol=1e-5)


def test_fully_padded_row_no_nan() -> None:
    """A row with NO visible context AND NO real query slot contributes nothing
    (gathered away), but must not poison the shared qkv grads with NaN."""
    B, N, d, n_id, max_time = 3, 8, 16, 4, 5
    pred = _make_predictor(d_model=d, n_identity=n_id, max_time=max_time)
    torch.manual_seed(4)
    context = torch.randn(B, N, d, requires_grad=True)
    ctx_time = torch.randint(0, max_time, (N,))
    qry_time = torch.randint(0, max_time, (N,))
    qry_id = torch.randint(0, n_id, (N,))
    ctx_pad = torch.rand(B, N) < 0.4
    qry_valid = torch.rand(B, N) < 0.5
    # Force row 0 fully padded on BOTH sides.
    ctx_pad[0] = True
    qry_valid[0] = False
    # Guarantee at least one real query elsewhere so the loss is non-empty.
    qry_valid[1, 0] = True

    pred.ragged_predictor = True
    out = pred(
        context,
        context_time_ids=ctx_time,
        query_time_ids=qry_time,
        query_id=qry_id,
        context_key_padding_mask=ctx_pad,
        query_valid=qry_valid,
    )
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(context.grad).all()


def test_no_query_valid_falls_back_to_dense() -> None:
    """query_valid=None → ragged path is skipped (returns full (B, N_qry, d))."""
    B, N, d, n_id, max_time = 2, 6, 16, 4, 5
    pred = _make_predictor(d_model=d, n_identity=n_id, max_time=max_time)
    pred.ragged_predictor = True
    torch.manual_seed(5)
    context = torch.randn(B, N, d)
    out = pred(
        context,
        context_time_ids=torch.randint(0, max_time, (N,)),
        query_time_ids=torch.randint(0, max_time, (N,)),
        query_id=torch.randint(0, n_id, (N,)),
        context_key_padding_mask=None,
        query_valid=None,
    )
    assert out.shape == (B, N, d)
