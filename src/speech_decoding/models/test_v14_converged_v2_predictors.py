"""P2.5 tests for the converged-v2 M2/M4 JEPA predictors."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v2 import JepaPredictorV2

N_PARCELS = 62
K = 2


def _m2(d=32, pred=24, n_heads=4, n_layers=2):
    return JepaPredictorV2(d, pred, n_heads, n_layers)  # no parcel/seed


def _m4(d=32, pred=24, n_heads=4, n_layers=2):
    return JepaPredictorV2(d, pred, n_heads, n_layers, n_parcels=N_PARCELS, k=K)


def _meta(B, Lc, Lq, *, with_pc_seed=False, n_freq=4, max_slot=15):
    torch.manual_seed(0)
    ctx = torch.randn(B, Lc, 32)
    ctx_slot = torch.randint(0, max_slot + 1, (B, Lc))
    q_slot = torch.randint(0, max_slot + 1, (B, Lq))
    q_freq = torch.randint(0, n_freq, (B, Lq))
    extra = {}
    if with_pc_seed:
        extra["q_parcel"] = torch.randint(0, N_PARCELS, (B, Lq))
        extra["q_seed"] = torch.randint(0, K, (B, Lq))
    return ctx, ctx_slot, q_slot, q_freq, extra


def test_m2_predictor_output_shape():
    pred = _m2()
    ctx, ctx_slot, q_slot, q_freq, _ = _meta(B=6, Lc=14, Lq=8)
    out = pred(ctx, ctx_slot, q_slot, q_freq)
    assert out.shape == (6, 8, 32)   # (B, Lq, d_model target)


def test_m4_predictor_output_shape():
    pred = _m4()
    ctx, ctx_slot, q_slot, q_freq, extra = _meta(B=2, Lc=40, Lq=20, with_pc_seed=True)
    out = pred(ctx, ctx_slot, q_slot, q_freq, **extra)
    assert out.shape == (2, 20, 32)


def test_m4_requires_parcel_and_seed():
    pred = _m4()
    ctx, ctx_slot, q_slot, q_freq, _ = _meta(B=2, Lc=10, Lq=5)
    with pytest.raises(ValueError, match="pass q_parcel"):
        pred(ctx, ctx_slot, q_slot, q_freq)


def test_m2_rejects_parcel_tag():
    pred = _m2()
    ctx, ctx_slot, q_slot, q_freq, extra = _meta(B=2, Lc=10, Lq=5, with_pc_seed=True)
    with pytest.raises(ValueError, match="no parcel tag"):
        pred(ctx, ctx_slot, q_slot, q_freq, q_parcel=extra["q_parcel"])


def test_predictions_depend_on_context():
    """Queries cross-attend the context ⇒ changing context changes predictions."""
    pred = _m2()
    pred.eval()
    ctx, ctx_slot, q_slot, q_freq, _ = _meta(B=4, Lc=12, Lq=6)
    with torch.no_grad():
        out1 = pred(ctx, ctx_slot, q_slot, q_freq)
        out2 = pred(ctx + 3.0, ctx_slot, q_slot, q_freq)
    assert not torch.allclose(out1, out2)


def test_context_key_mask_leak_free():
    """Masked context tokens never key ⇒ predictions invariant to their values."""
    pred = _m2()
    pred.eval()
    ctx, ctx_slot, q_slot, q_freq, _ = _meta(B=3, Lc=12, Lq=6)
    mask = torch.ones(3, 12, dtype=torch.bool)
    mask[:, 8:] = False                       # hide the last 4 context tokens
    with torch.no_grad():
        out1 = pred(ctx, ctx_slot, q_slot, q_freq, ctx_key_mask=mask)
        ctx2 = ctx.clone()
        ctx2[:, 8:] += 9.0                    # corrupt the masked context
        out2 = pred(ctx2, ctx_slot, q_slot, q_freq, ctx_key_mask=mask)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_m4_gradient_flows_to_all_embeds():
    pred = _m4()
    ctx, ctx_slot, q_slot, q_freq, extra = _meta(B=2, Lc=20, Lq=10, with_pc_seed=True)
    ctx = ctx.requires_grad_(True)
    pred(ctx, ctx_slot, q_slot, q_freq, **extra).sum().backward()
    assert torch.isfinite(ctx.grad).all()
    assert pred.mask_token.grad is not None
    assert pred.freq_embed.grad is not None
    assert pred.parcel_embed.weight.grad is not None
    assert pred.seed_embed.grad is not None


def test_rope_table_sized_for_5s():
    pred = _m4()
    # 5 s HGA reaches slot 79 ⇒ table ≥ 80.
    assert pred.key_rope.shape[1] >= 80
