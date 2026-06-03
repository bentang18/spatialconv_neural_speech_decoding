"""bf16-safe sentinel for masked attention logits (IE12).

``NEG_INF_MASK_VALUE = -1e4`` is the value used by every ``masked_fill`` on
attention logits in the v14 encoder. It must satisfy two constraints:

  1. Drive softmax weight at masked positions below 1e-6 in fp32/fp16/bf16
     (otherwise invalid-channel / invalid-parcel positions leak into
     downstream signal).
  2. Not overflow when added to typical bias magnitudes (so we don't get
     NaN/Inf in attn_logits = logits + bias).

``torch.finfo(dtype).min`` (the HuggingFace default) satisfies (1) in
practice because softmax upcasts internally, but it can overflow when
*added* to other large negatives — the cited HF/transformers issue. The
``-1e4`` constant avoids both pitfalls without losing masking strength.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import NEG_INF_MASK_VALUE


def test_neg_inf_mask_drives_softmax_below_threshold_in_all_dtypes() -> None:
    """A row where every position is masked-out except one must give a
    softmax distribution where the masked positions are <1e-6 across
    fp32/fp16/bf16."""
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        logits = torch.zeros(64, dtype=dtype)
        masked = torch.ones(64, dtype=torch.bool)
        masked[0] = False
        logits = logits.masked_fill(masked, NEG_INF_MASK_VALUE)
        p = logits.softmax(dim=-1)
        valid_p = p[0].item()
        max_masked_p = p[1:].max().item()
        assert valid_p > 0.999, f"{dtype}: valid position got prob {valid_p}"
        assert max_masked_p < 1e-6, (
            f"{dtype}: masked positions got prob {max_masked_p} > 1e-6"
        )


def test_neg_inf_mask_does_not_overflow_when_added_to_negative_bias() -> None:
    """B36's default hard pool masks straight to ``NEG_INF_MASK_VALUE`` (no
    additive anatomy bias), but the gated ``R-bna-soft`` sister still adds
    ``log(support+eps)`` to the masked logits, where for zero-support
    electrodes that bias is itself a small negative. Adding the sentinel to it
    must not produce -Inf in any dtype — the HF/transformers `dtype.min`
    pitfall. This pins the sentinel's headroom for that sister and in general."""
    bias = torch.tensor(-10.0)  # log(1e-2 + 1e-2) ≈ -3.9, well within typical range
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        sentinel = torch.tensor(NEG_INF_MASK_VALUE, dtype=dtype)
        bias_d = bias.to(dtype)
        s = sentinel + bias_d
        assert torch.isfinite(s), f"{dtype}: sentinel + bias = {s} not finite"

    # By contrast, finfo(bf16).min added to itself overflows (a documented
    # pitfall in the HF/transformers code that motivated IE12 in the first
    # place):
    bf16_min = torch.tensor(torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16)
    assert not torch.isfinite(bf16_min + bf16_min), (
        "test premise: finfo(bf16).min + finfo(bf16).min should overflow"
    )
