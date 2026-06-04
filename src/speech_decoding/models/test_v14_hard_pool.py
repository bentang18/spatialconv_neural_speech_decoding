"""WS-A — B36 hard block-diagonal per-parcel pool.

The central B36 architectural claim: the soft additive
``λ_anat·log(support+ε)`` Graphormer routing bias is replaced by a HARD
one-hot DK assignment mask. A parcel-slot query attends ONLY to its own
parcel's electrode-freq tokens; every off-parcel pooling weight is EXACTLY
0.0 (not within a tolerance, in every dtype), on-parcel rows renormalise to
sum 1.0, and a no-coverage parcel gets an all-zero row.

  * A1 — the exact-0.0 / sum-1.0 pooling invariant (``pool_weights``).
  * A2 — source-hygiene: the soft-bias machinery is gone from non-test src,
    and the encoder forward signature carries no ``lambda_anat`` kwarg.

These fail against ``HEAD`` (pre-B36 soft bias) and pass after.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

from speech_decoding.models.v14_encoder import (
    V14ParcelPerceiverModel,
    _MultiHeadCrossAttention,
)

_SRC_ROOT = Path(__file__).resolve().parents[1]  # .../speech_decoding


# ---------------------------------------------------------------------------
# A1 — hard block-diagonal pool: off-parcel == 0.0 exactly; on-parcel sum 1.0
# ---------------------------------------------------------------------------


def _attn_with_block_diag_mask(dtype: torch.dtype = torch.float32):
    """A tiny cross-attn + a known block-diagonal key_mask.

    L=3 parcel slots, N=5 electrode-freq tokens:
      * parcel 0 ← tokens {0, 1}
      * parcel 1 ← tokens {2, 3, 4}
      * parcel 2 ← {} (no-coverage parcel — all keys blocked)
    """
    torch.manual_seed(0)
    d_model, n_heads = 16, 4
    B, L, N = 2, 3, 5
    attn = _MultiHeadCrossAttention(d_model, n_heads).eval().to(dtype)
    latents = torch.randn(B, L, d_model, dtype=dtype)
    electrodes = torch.randn(B, N, d_model, dtype=dtype)
    key_mask = torch.zeros(B, L, N, dtype=torch.bool)
    key_mask[:, 0, 0:2] = True
    key_mask[:, 1, 2:5] = True
    # parcel 2 stays all-False (no coverage)
    return attn, latents, electrodes, key_mask, (B, L, N)


def test_a1_off_parcel_pool_weight_is_exactly_zero() -> None:
    """Every blocked (slot, token) pooling weight is 0.0 EXACTLY (not ≤ tol)."""
    attn, latents, electrodes, key_mask, (B, L, N) = _attn_with_block_diag_mask()
    with torch.no_grad():
        weights = attn.pool_weights(latents, electrodes, key_mask)  # (B, H, L, N)
    H = attn.n_heads
    assert weights.shape == (B, H, L, N)
    # Broadcast key_mask over the head axis; blocked == ~on-parcel.
    blocked = ~key_mask.unsqueeze(1).expand(B, H, L, N)
    off_parcel = weights[blocked]
    assert torch.equal(off_parcel, torch.zeros_like(off_parcel)), (
        "off-parcel pooling weights must be exactly 0.0; got max "
        f"{off_parcel.abs().max().item():.3e}"
    )


def test_a1_on_parcel_rows_sum_to_one() -> None:
    """Covered parcel rows renormalise to sum 1.0; no-coverage rows sum 0.0."""
    attn, latents, electrodes, key_mask, (B, L, N) = _attn_with_block_diag_mask()
    with torch.no_grad():
        weights = attn.pool_weights(latents, electrodes, key_mask)  # (B, H, L, N)
    row_sums = weights.sum(dim=-1)  # (B, H, L)
    covered = key_mask.any(dim=-1)  # (B, L) — True where the parcel has ≥1 key
    covered_h = covered.unsqueeze(1).expand_as(row_sums)
    torch.testing.assert_close(
        row_sums[covered_h], torch.ones_like(row_sums[covered_h]),
        atol=1e-6, rtol=0,
    )
    # The no-coverage parcel (slot 2) sums to exactly 0 (all-zero row).
    nocov = row_sums[~covered_h]
    assert torch.equal(nocov, torch.zeros_like(nocov)), (
        f"no-coverage parcel row must sum to 0.0; got max {nocov.abs().max().item():.3e}"
    )


def test_a1_off_parcel_exact_zero_holds_under_bf16() -> None:
    """The exact-0.0 claim is dtype-independent: in the hand-rolled
    ``pool_weights`` path the post-softmax ``masked_fill(0)`` forces a
    structural zero off-parcel in bf16, not merely a near-zero from the -1e4
    sentinel. (``forward`` reaches the same zero via SDPA's -inf mask; this
    pins the monitor/test path that SDPA cannot return weights for.)"""
    attn, latents, electrodes, key_mask, (B, L, N) = _attn_with_block_diag_mask(
        dtype=torch.bfloat16
    )
    with torch.no_grad():
        weights = attn.pool_weights(latents, electrodes, key_mask)
    H = attn.n_heads
    blocked = ~key_mask.unsqueeze(1).expand(B, H, L, N)
    off_parcel = weights[blocked]
    assert torch.equal(off_parcel, torch.zeros_like(off_parcel)), (
        "bf16 off-parcel pooling weights must be exactly 0.0; got max "
        f"{off_parcel.abs().max().item():.3e}"
    )


def test_a1_forward_matches_pool_weights_contraction() -> None:
    """``forward`` pools with the same masked weights ``pool_weights`` reports:
    ctx = Σ_n w[b,h,l,n] · v[b,n,h,:]. A no-coverage slot → zero context →
    out_proj(0) == 0 (out has bias=False)."""
    attn, latents, electrodes, key_mask, (B, L, N) = _attn_with_block_diag_mask()
    with torch.no_grad():
        out = attn(latents, electrodes, key_mask)  # (B, L, d)
        weights = attn.pool_weights(latents, electrodes, key_mask)  # (B, H, L, N)
        kv = attn.kv_proj(electrodes).reshape(
            B, N, 2, attn.n_heads, attn.head_dim
        )
        v = kv.unbind(dim=2)[1]  # (B, N, H, head_dim)
        ctx = torch.einsum("bhln,bnhd->blhd", weights, v).reshape(B, L, -1)
        expected = attn.out(ctx)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)
    # No-coverage slot 2 → zero context → exactly-zero out row.
    nocov_out = out[:, 2]
    assert torch.equal(nocov_out, torch.zeros_like(nocov_out)), (
        "no-coverage parcel slot must produce an exactly-zero pooled output"
    )


def test_a1_no_coverage_row_keeps_forward_and_backward_finite() -> None:
    """A no-coverage parcel slot is FULLY blocked in the cross-attn. ``forward``
    must use an *additive* -1e4 mask (not a boolean mask): a boolean mask makes
    SDPA softmax over an empty set → NaN in forward AND backward on CUDA's
    mem-efficient/flash kernels, and the forward no-coverage zeroing cannot
    block the backward NaN (0·NaN = NaN poisons the q/k/v grads). The -1e4
    sentinel makes a fully-blocked row a uniform finite softmax instead, so
    forward + backward stay finite on every backend.

    NOTE: the CPU math backend already returns 0 (not NaN) for an empty-set
    softmax, so this CPU test cannot *reproduce* the CUDA NaN — it pins the
    structural requirement (finite output, finite grads, exact-zero no-coverage
    row) that the -1e4 sentinel guarantees by construction, and would catch a
    regression to a boolean/-inf mask on a backend that NaNs.
    """
    attn, latents, electrodes, key_mask, (B, L, N) = _attn_with_block_diag_mask()
    latents = latents.clone().detach().requires_grad_(True)
    electrodes = electrodes.clone().detach().requires_grad_(True)
    out = attn(latents, electrodes, key_mask)   # slot 2 = no coverage
    out.sum().backward()

    # No NaN/Inf anywhere: outputs, input grads, parameter grads.
    assert torch.isfinite(out).all(), "forward produced non-finite output"
    assert latents.grad is not None and torch.isfinite(latents.grad).all()
    assert electrodes.grad is not None and torch.isfinite(electrodes.grad).all()
    for name, p in attn.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), (
            f"non-finite grad in parameter {name!r} (no-coverage row NaN leak)"
        )
    # The no-coverage slot output is still exactly zero.
    nocov_out = out[:, 2]
    assert torch.equal(nocov_out, torch.zeros_like(nocov_out))


def test_a1_model_forward_uses_one_hot_support_as_hard_assignment() -> None:
    """End-to-end: with a one-hot DK ``support``, ablating an electrode's
    input changes ONLY its own parcel's pre-self-attn latent — off-parcel
    latents are bit-identical (hard partition at the cross-attn). Tested at
    ``depth_self_attn=0`` so latent self-attn does not diffuse the routing.
    """
    torch.manual_seed(0)
    kw = dict(
        n_freq_bins=6, n_time_bins=4, k_parcels=6,
        d_model=32, n_heads=4, depth_self_attn=0, m_sub_slots=1,
        patch_kernel_freq=3,  # FE-RAW-1: kernel-3 path for the tiny F=6 config
    )
    model = V14ParcelPerceiverModel(**kw).eval()
    B, C = 1, 4
    electrodes = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"])
    support = torch.zeros(B, C, kw["k_parcels"])
    for c in range(C):
        support[0, c, c] = 1.0  # electrode c → parcel c (one-hot)

    with torch.no_grad():
        base = model(electrodes, support)
        ablated = electrodes.clone()
        ablated[0, 0] = 0.0  # perturb electrode 0 → only parcel 0
        pert = model(ablated, support)

    # Parcel 0 latent must change; parcels 1..3 must be bit-identical.
    delta_p0 = (base[:, 0] - pert[:, 0]).abs().max().item()
    delta_off = (base[:, 1:4] - pert[:, 1:4]).abs().max().item()
    assert delta_p0 > 0.0, "perturbing electrode 0 must move its own parcel latent"
    assert delta_off == 0.0, (
        f"off-parcel latents must be bit-identical under the hard pool; "
        f"got max |Δ| = {delta_off:.3e}"
    )


# ---------------------------------------------------------------------------
# A2 — source hygiene: soft-bias machinery removed from non-test src
# ---------------------------------------------------------------------------

_FORBIDDEN_LITERALS = ("compute_gated_log_support_bias", "lambda_anat")


def _non_test_py_files() -> list[Path]:
    return [
        p for p in _SRC_ROOT.rglob("*.py")
        if not p.name.startswith("test_")
    ]


@pytest.mark.parametrize("literal", _FORBIDDEN_LITERALS)
def test_a2_soft_bias_literal_absent_from_non_test_src(literal: str) -> None:
    """The B28/B29 soft-routing machinery (``compute_gated_log_support_bias``,
    the ``lambda_anat`` kwarg) must not survive anywhere in non-test ``src/``
    — name, kwarg, or comment. The unicode ``λ_anat`` may appear in
    historical-note prose; only the ASCII forms are forbidden."""
    offenders = []
    for path in _non_test_py_files():
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if literal in line:
                offenders.append(f"{path.relative_to(_SRC_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        f"{literal!r} still present in non-test src:\n" + "\n".join(offenders)
    )


def test_a2_encoder_forward_signature_has_no_lambda_anat() -> None:
    """Neither the model nor the head-wrapper forward exposes ``lambda_anat``."""
    from speech_decoding.models.v14_encoder import V14ParcelPerceiverWithHead

    for cls in (V14ParcelPerceiverModel, V14ParcelPerceiverWithHead):
        params = inspect.signature(cls.forward).parameters
        assert "lambda_anat" not in params, (
            f"{cls.__name__}.forward must not take a lambda_anat kwarg; "
            f"got {list(params)}"
        )


def test_a2_no_gated_log_support_bias_call_in_ast() -> None:
    """AST-level guard: no call to a name/attribute ``compute_gated_log_support_bias``
    anywhere in non-test src (catches re-imports the line grep might miss)."""
    bad: list[str] = []
    for path in _non_test_py_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name)
                    else None
                )
                if name == "compute_gated_log_support_bias":
                    bad.append(f"{path.relative_to(_SRC_ROOT)}:{node.lineno}")
    assert not bad, "compute_gated_log_support_bias is still called: " + ", ".join(bad)
