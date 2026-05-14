"""Latent self-attn key-padding mask tests.

Canonical Perceiver-IO + DETR leave self-attn over the latent / query array
unmasked because their latent slots are uniformly valid by construction. v14's
parcel-id-tagged latents have per-subject variable validity (no-coverage
parcels). Without masking, invalid latents serve as keys/values in latent
self-attn and contaminate covered latents — a subject-shape side-channel that
breaks the zero-per-subject-params commitment.

Fix: latent self-attn applies a key-padding mask derived from
``_compute_latent_valid(support, valid_mask)``. Covered latents become
genuinely invariant to perturbations of invalid-parcel embeddings.

Three tests:
1. ``_LatentSelfAttnBlock`` masks invalid keys at the unit level.
2. ``V14ParcelPerceiverModel`` covered-latent outputs are invariant to
   perturbations of invalid parcels' ``parcel_embedding`` rows.
3. ``V14ParcelPerceiverWithHead`` end-to-end output is invariant to the same
   perturbation (composition of latent self-attn mask + DETR memory mask).
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import (
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
    _LatentSelfAttnBlock,
)


D_MODEL = 32
N_HEADS = 4


def test_latent_self_attn_block_masks_invalid_keys() -> None:
    """Unit level: ``_LatentSelfAttnBlock(x, latent_valid=mask)`` makes the
    output at covered positions invariant to perturbations at invalid key
    positions. This is the standard `nn.MultiheadAttention(key_padding_mask=...)`
    contract: mask invalid keys, valid positions don't see them."""
    torch.manual_seed(0)
    block = _LatentSelfAttnBlock(d_model=D_MODEL, n_heads=N_HEADS)
    block.eval()

    B, L, d = 1, 6, D_MODEL
    x = torch.randn(B, L, d)
    latent_valid = torch.tensor([[True, True, False, False, True, False]])

    with torch.no_grad():
        out_ref = block(x, latent_valid=latent_valid)

        perturbed = x.clone()
        invalid = ~latent_valid[0]
        perturbed[0, invalid, :] = torch.randn(int(invalid.sum().item()), d) * 100.0
        out_perturbed = block(perturbed, latent_valid=latent_valid)

    valid = latent_valid[0]
    assert torch.allclose(
        out_ref[0, valid, :], out_perturbed[0, valid, :], atol=1e-5
    ), "covered positions must be invariant to perturbations at masked keys"


def test_latent_self_attn_block_unmasked_falls_back_to_full_attention() -> None:
    """Backward compat: when ``latent_valid`` is omitted, the block behaves
    identically to plain self-attn (perturbing any position changes others)."""
    torch.manual_seed(0)
    block = _LatentSelfAttnBlock(d_model=D_MODEL, n_heads=N_HEADS)
    block.eval()

    B, L, d = 1, 4, D_MODEL
    x = torch.randn(B, L, d)

    with torch.no_grad():
        out_ref = block(x)
        perturbed = x.clone()
        perturbed[0, 0, :] = perturbed[0, 0, :] + torch.randn(d) * 10.0
        out_perturbed = block(perturbed)

    assert not torch.allclose(out_ref, out_perturbed, atol=1e-5), (
        "without mask, perturbing any latent must affect others via self-attn"
    )


def test_encoder_covered_latents_invariant_to_invalid_parcel_embeddings() -> None:
    """Encoder integration: when parcel ``p`` has no coverage, perturbations
    to ``parcel_embedding[p, :, :]`` must not change the covered parcels'
    encoder outputs. Holds only if latent self-attn masks invalid keys.

    Setup: K=3 parcels, M=2 → L=6 latents; depth_self_attn=2 so the mask is
    applied twice. Only parcel 0 covered (one valid electrode at index 0;
    others padded). Parcels 1, 2 have no coverage anywhere.
    """
    torch.manual_seed(0)
    encoder = V14ParcelPerceiverModel(
        n_freq_bins=3, n_time_bins=5, k_parcels=3,
        d_model=D_MODEL, n_heads=N_HEADS, depth_self_attn=2, m_sub_slots=2,
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
        original = encoder.parcel_embedding.data.clone()
        encoder.parcel_embedding.data[1:, :, :] = (
            torch.randn_like(encoder.parcel_embedding.data[1:, :, :]) * 100.0
        )
        out_perturbed = encoder(electrodes, support, valid_mask)
        encoder.parcel_embedding.data.copy_(original)

    out_ref_covered = out_ref[:, :M, :]
    out_perturbed_covered = out_perturbed[:, :M, :]

    assert torch.allclose(
        out_ref_covered, out_perturbed_covered, atol=1e-5
    ), "covered latents must be invariant to invalid parcel-embedding perturbations"


def test_wrapper_output_invariant_to_invalid_parcel_embeddings() -> None:
    """End-to-end: ``V14ParcelPerceiverWithHead`` logits are invariant to
    perturbations of ``parcel_embedding`` for no-coverage parcels. Composes
    the latent-self-attn mask (this commit) with the DETR memory mask (prior
    commit 0a79ff5). Both must be in place for the wrapper to be invariant."""
    torch.manual_seed(0)
    cfg = V14ParcelPerceiver(
        n_freq_bins=3, n_time_bins=5, k_parcels=3,
        d_model=D_MODEL, n_heads=N_HEADS, depth_self_attn=2, m_sub_slots=2,
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
        original = model.encoder.parcel_embedding.data.clone()
        model.encoder.parcel_embedding.data[1:, :, :] = (
            torch.randn_like(model.encoder.parcel_embedding.data[1:, :, :]) * 100.0
        )
        out_perturbed = model(electrodes, support, valid_mask)
        model.encoder.parcel_embedding.data.copy_(original)

    assert torch.allclose(out_ref, out_perturbed, atol=1e-5), (
        "wrapper output must be invariant to invalid parcel-embedding perturbations"
    )
