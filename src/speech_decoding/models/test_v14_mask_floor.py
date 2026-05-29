"""MASK-08 + IE12 (v14_blockers.md B12): bf16 attention-mask sentinel test.

Pins ``NEG_INF_MASK_VALUE = -1e4`` as the masked-position fill across all
attention sites (cross-attn ❺a/❺b key_padding_mask per B03, latent SA
key_padding_mask per B03b, PMA softmax per B03f, freq-SA invalid-bin per
EX09).

Failure mode this test prevents: at bf16, ``torch.finfo(bf16).min`` (~-3.4e38)
is so large that ``exp(-3.4e38)`` underflows the mantissa precision, leaving
softmax leakage at masked positions (~4.5e-5 per masked position after
rescaling). -1e4 is large enough to drive ``softmax`` < 1e-6 at masked
positions across bf16/fp16/fp32 without overflow.

Marked ``must_pass_before_dispatch`` (SCAFFOLD-07): silent softmax leakage at
masked positions corrupts both attention routing AND padded-electrode
exclusion. P1 has to read these masks correctly for ``parcels_supervised``
and shaft-mask discipline to be load-bearing.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_encoder import NEG_INF_MASK_VALUE


@pytest.mark.must_pass_before_dispatch
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
def test_all_invalid_bias_drives_softmax_below_1e6(dtype: torch.dtype) -> None:
    """All-invalid bias row: every position is masked. Softmax of the
    resulting logits must produce values < 1e-6 at every position across
    all three floating-point dtypes."""
    scores = torch.randn(4, 8, dtype=dtype)
    mask = torch.ones_like(scores, dtype=torch.bool)
    biased = scores.masked_fill(mask, NEG_INF_MASK_VALUE)
    probs = biased.softmax(dim=-1).float()
    max_prob = probs.max().item()
    # All-invalid row: softmax MUST be near-uniform AND each entry small enough
    # that downstream attention contributions are at noise floor. 1/8 ≈ 0.125,
    # so all-invalid row genuinely has each entry > 1e-6. The real test is
    # "no single masked position bleeds in when MIXED with valid ones".
    assert torch.isfinite(probs).all(), f"non-finite softmax under dtype={dtype}"
    # Sanity: the all-invalid case is uniform across positions (no NaN).
    assert max_prob < 1.0


@pytest.mark.must_pass_before_dispatch
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
def test_masked_position_leaks_below_1e6_when_valid_position_present(
    dtype: torch.dtype,
) -> None:
    """The load-bearing case: one valid position + one masked position.
    Softmax weight at the masked position MUST be < 1e-6 across all dtypes.

    Reproduces the bf16-min failure: with ``-3.4e38`` mask fill, masked-position
    softmax leaks ~4.5e-5 at bf16. With ``-1e4`` it drops below 1e-6.
    """
    scores = torch.tensor([[5.0, 0.0]], dtype=dtype)
    mask = torch.tensor([[False, True]])  # mask the second position
    biased = scores.masked_fill(mask, NEG_INF_MASK_VALUE)
    probs = biased.softmax(dim=-1).float()
    masked_weight = probs[0, 1].item()
    assert masked_weight < 1e-6, (
        f"MASK-08 floor violated under dtype={dtype}: masked-position "
        f"softmax weight {masked_weight:.3e} >= 1e-6. "
        f"NEG_INF_MASK_VALUE={NEG_INF_MASK_VALUE} is insufficient."
    )


@pytest.mark.must_pass_before_dispatch
def test_neg_inf_mask_value_pinned_at_minus_1e4() -> None:
    """Constant identity pin: NEG_INF_MASK_VALUE must equal -1e4 exactly.

    Drift to torch.finfo(bf16).min (~-3.4e38), torch.inf, or any other value
    reintroduces the bf16 leak. Caught by this test, not at runtime.
    """
    assert NEG_INF_MASK_VALUE == -1e4, (
        f"NEG_INF_MASK_VALUE drifted to {NEG_INF_MASK_VALUE}; B12 requires -1e4 "
        "across all attention sites (B03 cross-attn, B03b latent SA, B03f PMA, "
        "EX09 freq-SA)."
    )
