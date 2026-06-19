"""S3 (#248): the query-only attention path returns the full pass sliced to the
last n_query rows. SDPA is a fused kernel whose reduction order can differ with
the query-tile count, so in fp32 the two differ by a sub-ULP round-off (~6e-8,
below bf16 precision). Under the **actual bf16 autocast training dtype** of the
M4 predictor, the outputs are BITWISE IDENTICAL — zero science change on the
training path. We assert both: tight fp32 closeness AND bf16 bitwise equality."""
from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import (
    _PlainMultiHeadSelfAttentionRoPE,
    _rope_freqs,
)

_FP32_TOL = 1e-5  # >> the observed ~6e-8 SDPA reduction-order round-off


def _attn(d=32, h=4, seed=0):
    torch.manual_seed(seed)
    return _PlainMultiHeadSelfAttentionRoPE(d, h).eval()


def _assert_query_only(a, x, rope, n_q, *, key_mask=None):
    T = x.shape[1]
    with torch.no_grad():
        full = a(x, rope, key_mask=key_mask)
        q_only = a(x, rope, key_mask=key_mask, n_query=n_q)
    assert q_only.shape == (x.shape[0], n_q, x.shape[2])
    # fp32: identical to kernel round-off.
    assert torch.allclose(q_only, full[:, T - n_q :], atol=_FP32_TOL, rtol=_FP32_TOL)
    # bf16 (real M4 dtype): bitwise identical.
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        full_b = a(x, rope, key_mask=key_mask)
        q_only_b = a(x, rope, key_mask=key_mask, n_query=n_q)
    assert torch.equal(q_only_b, full_b[:, T - n_q :])


def test_query_only_equals_full_slice_dense_rope():
    d, h, T, n_q = 32, 4, 20, 7
    _assert_query_only(_attn(d, h), torch.randn(3, T, d), _rope_freqs(d // h, T), n_q)


def test_query_only_equals_full_slice_with_key_mask():
    d, h, T, n_q = 32, 4, 16, 5
    rope = _rope_freqs(d // h, T)
    key_mask = torch.ones(2, T, dtype=torch.bool)
    key_mask[0, 13:] = False  # drop some keys (ragged padding)
    _assert_query_only(_attn(d, h), torch.randn(2, T, d), rope, n_q, key_mask=key_mask)


def test_query_only_equals_full_slice_per_row_rope():
    d, h, T, n_q, B = 32, 4, 18, 6, 3
    base = _rope_freqs(d // h, T)  # (2, T, hd)
    rope = base[:, None].expand(2, B, T, d // h).contiguous()  # per-row (2, B, T, hd)
    _assert_query_only(_attn(d, h), torch.randn(B, T, d), rope, n_q)
