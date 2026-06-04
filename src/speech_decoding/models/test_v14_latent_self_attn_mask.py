"""Latent self-attn bidirectional ``attn_mask`` tests.

Canonical Perceiver-IO + DETR leave self-attn over the latent / query array
unmasked because their latent slots are uniformly valid by construction. v14's
parcel-id-tagged latents have per-subject variable validity (no-coverage
parcels). Without masking, invalid latents serve as keys/values in latent
self-attn and contaminate covered latents — a subject-shape side-channel that
breaks the zero-per-subject-params commitment.

**B30 lock 2026-05-28** ([[project_v14_anatomy_gated_symmetric_2026_05_28]]):
latent self-attn applies a **bidirectional** ``attn_mask = latent_valid[:, :, None]
& latent_valid[:, None, :]`` derived from ``_compute_latent_valid(support,
valid_mask)``. Invalid slots fully bypass — they are masked as both queries
AND keys, so covered latents are invariant to perturbations of invalid-parcel
embeddings AND invalid query positions get zero SA contribution (preserved via
the residual). The pre-B30 key-only ``key_padding_mask`` path is gone.

Tests:
1. ``_LatentSelfAttnBlock`` masks invalid keys (valid outputs unchanged when
   invalid inputs are perturbed). Bidirectional implies key-side masking.
2. B30 bidirectional bypass: perturbing valid inputs leaves invalid query
   outputs unchanged (invalid queries can't attend, residual preserves them).
3. B30 zero-active: all-invalid mask doesn't NaN (post-softmax row-zeroing).
4. ``V14ParcelPerceiverModel`` covered-latent outputs invariant to invalid
   parcels' ``parcel_embedding`` rows.
5. ``V14ParcelPerceiverWithHead`` end-to-end output invariant to the same
   perturbation (latent SA mask + DETR memory mask compose).
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import (
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
    _LatentSelfAttnBlock,
    _rope_freqs,
)


D_MODEL = 32
N_HEADS = 4


def _rope(T: int) -> torch.Tensor:
    """RoPE cos/sin table sized to ``T``; used to feed the factorized
    latent block (T1.9) in unit tests that work over a flat (L,d) tensor."""
    return _rope_freqs(D_MODEL // N_HEADS, max_seq_len=max(T, 1))


def test_latent_self_attn_block_masks_invalid_keys() -> None:
    """Unit level: ``_LatentSelfAttnBlock(x, latent_valid=mask)`` makes the
    output at covered positions invariant to perturbations at invalid key
    positions. This is the standard `nn.MultiheadAttention(key_padding_mask=...)`
    contract: mask invalid keys, valid positions don't see them.

    With T1.9 the block is factorized (t-SA × parcel-SA × FFN). Setting T=1
    collapses the time half to a trivial per-position transform; the
    parcel-SA mask semantics tested here are unchanged.
    """
    torch.manual_seed(0)
    block = _LatentSelfAttnBlock(d_model=D_MODEL, n_heads=N_HEADS)
    block.eval()

    B, T, L, d = 1, 1, 6, D_MODEL
    x = torch.randn(B * T, L, d)
    latent_valid = torch.tensor([[True, True, False, False, True, False]])
    rope = _rope(T)

    with torch.no_grad():
        out_ref = block(x, B=B, T=T, L=L, latent_valid=latent_valid, rope_t=rope)

        perturbed = x.clone()
        invalid = ~latent_valid[0]
        perturbed[0, invalid, :] = torch.randn(int(invalid.sum().item()), d) * 100.0
        out_perturbed = block(
            perturbed, B=B, T=T, L=L, latent_valid=latent_valid, rope_t=rope,
        )

    valid = latent_valid[0]
    assert torch.allclose(
        out_ref[0, valid, :], out_perturbed[0, valid, :], atol=1e-5
    ), "covered positions must be invariant to perturbations at masked keys"


def test_latent_self_attn_block_invalid_queries_bypass_via_residual() -> None:
    """B30 bidirectional-mask contract ([[project_v14_anatomy_gated_symmetric_2026_05_28]]):
    inactive slots are masked as BOTH queries and keys. That means
    perturbing VALID keys must NOT affect the output at INVALID query
    positions (because invalid queries can't attend to anything, so the
    SA contribution is zero and the residual leaves the input unchanged).

    The pre-B30 key-only mask would have failed this test: invalid query
    rows would still mix valid keys in the attention output."""
    torch.manual_seed(0)
    block = _LatentSelfAttnBlock(d_model=D_MODEL, n_heads=N_HEADS)
    block.eval()

    B, T, L, d = 1, 1, 6, D_MODEL
    x = torch.randn(B * T, L, d)
    latent_valid = torch.tensor([[True, True, False, False, True, False]])
    rope = _rope(T)
    invalid = ~latent_valid[0]

    with torch.no_grad():
        out_ref = block(x, B=B, T=T, L=L, latent_valid=latent_valid, rope_t=rope)

        # Perturb only VALID positions of x. With a key-only mask, the
        # invalid query rows would still mix these — output would change.
        # With B30 bidirectional masking, invalid query rows produce zero
        # attention contribution → residual preserves the input.
        perturbed = x.clone()
        valid = latent_valid[0]
        perturbed[0, valid, :] = perturbed[0, valid, :] + torch.randn(
            int(valid.sum().item()), d
        ) * 100.0
        out_perturbed = block(
            perturbed, B=B, T=T, L=L, latent_valid=latent_valid, rope_t=rope,
        )

    # Invalid query positions: out = (residual of perturbed.x at those rows)
    # which equals perturbed.x at those rows; since we didn't touch x at
    # invalid rows, out_perturbed[invalid] must equal out_ref[invalid] AND
    # equal x[invalid] (modulo whatever time-SA does at T=1 — identity).
    assert torch.allclose(
        out_ref[0, invalid, :], out_perturbed[0, invalid, :], atol=1e-5
    ), "invalid query positions must bypass parcel-SA via residual under B30 bidirectional mask"


def test_latent_self_attn_block_zero_active_does_not_nan() -> None:
    """B30 degenerate case ([[project_v14_anatomy_gated_symmetric_2026_05_28]]):
    a SWEC clip with ``latent_valid`` all-False would, under the
    bidirectional mask, drive softmax over a row of finite
    ``NEG_INF_MASK_VALUE = -1e4`` to a uniform ``1/L`` distribution (NOT
    NaN — the mask value is finite). The
    ``attn.masked_fill(all_masked.unsqueeze(1), 0.0)`` guard inside
    ``_PlainMultiHeadSelfAttention`` must zero those rows post-softmax so
    the SA output at fully-masked queries is exactly zero and the
    residual leaves x unchanged. This test asserts no NaN under the
    all-False degenerate."""
    torch.manual_seed(0)
    block = _LatentSelfAttnBlock(d_model=D_MODEL, n_heads=N_HEADS)
    block.eval()

    B, T, L, d = 2, 1, 4, D_MODEL
    x = torch.randn(B * T, L, d)
    latent_valid = torch.zeros(B * T, L, dtype=torch.bool)

    with torch.no_grad():
        out = block(x, B=B, T=T, L=L, latent_valid=latent_valid, rope_t=_rope(T))

    assert torch.isfinite(out).all(), "all-invalid latent SA must not produce NaN"


def test_latent_self_attn_block_unmasked_falls_back_to_full_attention() -> None:
    """Backward compat: when ``latent_valid`` is omitted, the parcel-SA half
    is unmasked self-attn (perturbing any latent affects others)."""
    torch.manual_seed(0)
    block = _LatentSelfAttnBlock(d_model=D_MODEL, n_heads=N_HEADS)
    block.eval()

    B, T, L, d = 1, 1, 4, D_MODEL
    x = torch.randn(B * T, L, d)
    rope = _rope(T)

    with torch.no_grad():
        out_ref = block(x, B=B, T=T, L=L, rope_t=rope)
        perturbed = x.clone()
        perturbed[0, 0, :] = perturbed[0, 0, :] + torch.randn(d) * 10.0
        out_perturbed = block(perturbed, B=B, T=T, L=L, rope_t=rope)

    assert not torch.allclose(out_ref, out_perturbed, atol=1e-5), (
        "without mask, perturbing any latent must affect others via self-attn"
    )


def test_encoder_covered_latents_invariant_to_invalid_parcel_embeddings() -> None:
    """Encoder integration: when parcel ``p`` has no coverage, perturbations
    to the parcel-specific latent contribution must not change the covered
    parcels' encoder outputs. Holds only if latent self-attn masks invalid
    keys.

    LAT-01 (B21 lock 2026-05-25): the per-parcel slot contribution lives in
    two places — ``learnable_parcel_embed[p]`` (gradient-receiving) and
    ``latent_init_noise[p, :, :]`` (frozen ε buffer). Both rows are
    parcel-specific, so perturbing rows ``[1:]`` of each changes only the
    invalid parcels' slot contributions.

    Setup: K=3 parcels, M=2 → L=6 latents; depth_self_attn=2 so the mask is
    applied twice. Only parcel 0 covered (one valid electrode at index 0;
    others padded). Parcels 1, 2 have no coverage anywhere.
    """
    torch.manual_seed(0)
    encoder = V14ParcelPerceiverModel(
        n_freq_bins=3, n_time_bins=5, k_parcels=3,
        d_model=D_MODEL, n_heads=N_HEADS, depth_self_attn=2, m_sub_slots=2,
        patch_kernel_freq=3,  # FE-RAW-1: kernel-3 path for the tiny F=3 config
    )
    encoder.eval()

    B, C, T, F = 1, 4, 5, 3
    electrodes = torch.randn(B, C, T, F)
    support = torch.zeros(B, C, 3)
    support[0, 0, 0] = 1.0  # only electrode 0 → parcel 0
    valid_mask = torch.tensor([[True, False, False, False]])

    M = 2
    with torch.no_grad():
        out_ref = encoder(electrodes, support, valid_mask)
        original_parcel = encoder.learnable_parcel_embed.data.clone()
        original_noise = encoder.latent_init_noise.data.clone()
        encoder.learnable_parcel_embed.data[1:, :] = (
            torch.randn_like(encoder.learnable_parcel_embed.data[1:, :]) * 100.0
        )
        encoder.latent_init_noise.data[1:, :, :] = (
            torch.randn_like(encoder.latent_init_noise.data[1:, :, :]) * 100.0
        )
        out_perturbed = encoder(electrodes, support, valid_mask)
        encoder.learnable_parcel_embed.data.copy_(original_parcel)
        encoder.latent_init_noise.data.copy_(original_noise)

    out_ref_covered = out_ref[:, :M, :]
    out_perturbed_covered = out_perturbed[:, :M, :]

    assert torch.allclose(
        out_ref_covered, out_perturbed_covered, atol=1e-5
    ), "covered latents must be invariant to invalid parcel-embedding perturbations"


def test_wrapper_output_invariant_to_invalid_parcel_embeddings() -> None:
    """End-to-end: ``V14ParcelPerceiverWithHead`` logits are invariant to
    perturbations of the per-parcel latent contribution for no-coverage
    parcels. Composes the latent-self-attn mask (this commit) with the DETR
    memory mask (prior commit 0a79ff5). Both must be in place for the
    wrapper to be invariant.

    LAT-01 (B21 lock 2026-05-25): perturbs ``learnable_parcel_embed[1:, :]``
    + ``latent_init_noise[1:, :, :]`` (parcel-specific rows of the two
    parcel-axis tables). ``learnable_subslot_embed`` is shared across
    parcels and is left untouched."""
    torch.manual_seed(0)
    cfg = V14ParcelPerceiver(
        n_freq_bins=3, n_time_bins=5, k_parcels=3,
        d_model=D_MODEL, n_heads=N_HEADS, depth_self_attn=2, m_sub_slots=2,
        patch_kernel_freq=3,  # FE-RAW-1: kernel-3 path for the tiny F=3 config
    )
    model = cfg.build(n_outputs=3)
    model.eval()

    B, C, T, F = 1, 4, 5, 3
    electrodes = torch.randn(B, C, T, F)
    support = torch.zeros(B, C, 3)
    support[0, 0, 0] = 1.0
    valid_mask = torch.tensor([[True, False, False, False]])

    with torch.no_grad():
        out_ref = model(electrodes, support, valid_mask)
        original_parcel = model.encoder.learnable_parcel_embed.data.clone()
        original_noise = model.encoder.latent_init_noise.data.clone()
        model.encoder.learnable_parcel_embed.data[1:, :] = (
            torch.randn_like(model.encoder.learnable_parcel_embed.data[1:, :]) * 100.0
        )
        model.encoder.latent_init_noise.data[1:, :, :] = (
            torch.randn_like(model.encoder.latent_init_noise.data[1:, :, :]) * 100.0
        )
        out_perturbed = model(electrodes, support, valid_mask)
        model.encoder.learnable_parcel_embed.data.copy_(original_parcel)
        model.encoder.latent_init_noise.data.copy_(original_noise)

    assert torch.allclose(out_ref, out_perturbed, atol=1e-5), (
        "wrapper output must be invariant to invalid parcel-embedding perturbations"
    )
