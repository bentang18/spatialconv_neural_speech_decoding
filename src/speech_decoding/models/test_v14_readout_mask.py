"""Phase-4 readout memory-padding-mask tests (B35, 2026-05-31).

The P4 default readout is :class:`V14PmaReadout`
([[project_v14_b35_p4_frozen_pma_mean_linear_2026_05_31]]): the FROZEN
P3-PMA collapses the parcel/slot axis (key-masked by ``latent_valid``) →
``(B, T_p, d)``, then mean-over-time → ``Linear``. The B34 sisters are kept
selectable: ``V14PerTaskAttentivePooler`` (``readout="attentive"``, one
trainable query over the full ``(L*T_p)`` field, key-masked by
``latent_valid``) and ``V14MeanPoolLinearHead`` (``readout="meanpool"``,
masked-mean over the same valid cells). The mask semantics carry over:
parcel slots with no electrode coverage in this subject are padded memory and
must not contribute to the readout.

Pins:

1. The default ``V14PmaReadout`` is invariant to the activations of latent
   rows marked invalid (the frozen PMA key-masks them).
2. ``V14PerTaskAttentivePooler(..., latent_valid=mask)`` is invariant to the
   activations of latent rows marked invalid (key-masking).
3. ``latent_valid=None`` reduces to plain attention (perturbing any latent
   affects the output).
4. ``V14MeanPoolLinearHead`` masked mean ignores invalid latent rows.
5. The wrapper computes ``latent_valid`` from ``support`` and ``valid_mask``
   and forwards it to the readout (default + sister readouts).
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import (
    V14MeanPoolLinearHead,
    V14ParcelCollapsePMA,
    V14ParcelPerceiver,
    V14ParcelPerceiverWithHead,
    V14PerTaskAttentivePooler,
    V14PmaReadout,
)


D_MODEL = 32
N_CLASSES = 3
N_HEADS = 4


def test_pma_readout_is_invariant_to_perturbations_of_invalid_latents() -> None:
    """B35 default: the frozen-PMA readout key-masks invalid latent rows, so
    perturbing them must not change the collapsed (B, T_p, d) field nor the
    logits."""
    torch.manual_seed(0)
    pma = V14ParcelCollapsePMA(d_model=D_MODEL, n_heads=N_HEADS)
    readout = V14PmaReadout(
        pma=pma, temporal="mean", d_model=D_MODEL, n_classes=N_CLASSES,
        n_time_patches=4,
    )
    readout.eval()

    B, L, T = 2, 8, 4
    latents = torch.randn(B, L, T, D_MODEL)
    latent_valid = torch.tensor(
        [
            [True, True, False, False, True, True, False, True],
            [False, True, True, True, False, True, True, False],
        ]
    )

    with torch.no_grad():
        out_ref = readout(latents, latent_valid=latent_valid)
        perturbed = latents.clone()
        for b in range(B):
            for r in (~latent_valid[b]).nonzero(as_tuple=False).flatten().tolist():
                perturbed[b, r] = torch.randn(T, D_MODEL) * 100.0
        out_perturbed = readout(perturbed, latent_valid=latent_valid)

    assert torch.allclose(out_ref, out_perturbed, atol=1e-5), (
        "frozen-PMA readout output must be invariant to perturbations of "
        "invalid latent rows"
    )


def test_attentive_pooler_is_invariant_to_perturbations_of_invalid_latents() -> None:
    """Key-masking semantics: perturbing invalid latent rows must not change
    the attentive pooler logits."""
    torch.manual_seed(0)
    pooler = V14PerTaskAttentivePooler(
        d_model=D_MODEL, n_heads=N_HEADS, n_classes=N_CLASSES
    )
    pooler.eval()

    B, L, T = 2, 8, 4
    latents = torch.randn(B, L, T, D_MODEL)
    latent_valid = torch.tensor(
        [
            [True, True, False, False, True, True, False, True],
            [False, True, True, True, False, True, True, False],
        ]
    )

    with torch.no_grad():
        out_ref = pooler(latents, latent_valid=latent_valid)

        perturbed = latents.clone()
        for b in range(B):
            invalid_rows = (~latent_valid[b]).nonzero(as_tuple=False).flatten()
            for r in invalid_rows.tolist():
                perturbed[b, r] = torch.randn(T, D_MODEL) * 100.0
        out_perturbed = pooler(perturbed, latent_valid=latent_valid)

    assert torch.allclose(out_ref, out_perturbed, atol=1e-5), (
        "attentive pooler output must be invariant to perturbations of "
        "invalid latent rows"
    )


def test_attentive_pooler_unmasked_attends_to_every_position() -> None:
    """When ``latent_valid`` is omitted, every parcel×time token can be
    attended — perturbing any row changes the logits."""
    torch.manual_seed(0)
    pooler = V14PerTaskAttentivePooler(
        d_model=D_MODEL, n_heads=N_HEADS, n_classes=N_CLASSES
    )
    pooler.eval()

    B, L, T = 1, 4, 3
    latents = torch.randn(B, L, T, D_MODEL)

    with torch.no_grad():
        out_ref = pooler(latents)
        perturbed = latents.clone()
        perturbed[0, 0] = perturbed[0, 0] + torch.randn(T, D_MODEL) * 10.0
        out_perturbed = pooler(perturbed)

    assert not torch.allclose(out_ref, out_perturbed, atol=1e-5), (
        "without a mask, perturbing any latent must change the pooler output"
    )


def test_meanpool_head_ignores_invalid_latents() -> None:
    """The meanpool baseline masked-mean only over valid ``(parcel, time)``
    cells — invalid rows do not affect the logits."""
    torch.manual_seed(0)
    head = V14MeanPoolLinearHead(d_model=D_MODEL, n_classes=N_CLASSES)
    head.eval()

    B, L, T = 2, 6, 3
    latents = torch.randn(B, L, T, D_MODEL)
    latent_valid = torch.tensor(
        [
            [True, True, False, True, False, True],
            [False, True, True, False, True, True],
        ]
    )

    with torch.no_grad():
        out_ref = head(latents, latent_valid=latent_valid)
        perturbed = latents.clone()
        for b in range(B):
            for r in (~latent_valid[b]).nonzero(as_tuple=False).flatten().tolist():
                perturbed[b, r] = torch.randn(T, D_MODEL) * 100.0
        out_perturbed = head(perturbed, latent_valid=latent_valid)

    assert torch.allclose(out_ref, out_perturbed, atol=1e-5), (
        "meanpool head must ignore invalid latent rows"
    )


def test_wrapper_passes_latent_valid_computed_from_support_and_mask() -> None:
    """Integration: the wrapper computes ``latent_valid`` from ``support`` and
    ``valid_mask`` and forwards it to the readout. Captured by monkey-patching
    the readout's forward and inspecting kwargs.

    Setup: B=1, C=4, K=3 parcels, M=2 sub-slots → L=6 latents.
      * electrode 0 → parcel 0, electrode 1 → parcel 1, electrodes 2,3 padded.
      * Parcel 2 has no coverage anywhere → both sub-slots invalid.
    Expected latent_valid: [[T,T, T,T, F,F]].
    """
    torch.manual_seed(0)
    cfg = V14ParcelPerceiver(
        n_freq_bins=3, n_time_bins=5, k_parcels=3,
        d_model=D_MODEL, n_heads=N_HEADS, depth_self_attn=1, m_sub_slots=2,
        patch_kernel_freq=3,  # FE-RAW-1: kernel-3 path for the tiny F=3 config
    )
    model = cfg.build(n_outputs=N_CLASSES)
    assert isinstance(model, V14ParcelPerceiverWithHead)
    model.eval()

    captured: dict[str, torch.Tensor] = {}
    orig_forward = model.readout.forward

    def _capturing_forward(latents, latent_valid=None):
        if latent_valid is not None:
            captured["latent_valid"] = latent_valid.detach().clone()
        return orig_forward(latents, latent_valid=latent_valid)

    model.readout.forward = _capturing_forward  # type: ignore[assignment]

    B, C, T, F = 1, 4, 5, 3
    electrodes = torch.randn(B, C, T, F)
    support = torch.zeros(B, C, 3)
    support[0, 0, 0] = 1.0
    support[0, 1, 1] = 1.0
    valid_mask = torch.tensor([[True, True, False, False]])

    _ = model(electrodes, support, valid_mask)

    assert "latent_valid" in captured, "wrapper must pass latent_valid to readout"
    assert captured["latent_valid"].shape == (B, 6)
    expected = torch.tensor([[True, True, True, True, False, False]])
    assert torch.equal(captured["latent_valid"], expected), (
        f"expected {expected.tolist()}, got {captured['latent_valid'].tolist()}"
    )


def test_wrapper_passes_latent_valid_when_valid_mask_absent() -> None:
    """If ``valid_mask`` is not provided, the wrapper falls back to
    support-only coverage and still passes a ``latent_valid`` (every parcel
    with any nonzero support row is valid)."""
    torch.manual_seed(0)
    cfg = V14ParcelPerceiver(
        n_freq_bins=3, n_time_bins=5, k_parcels=3,
        d_model=D_MODEL, n_heads=N_HEADS, depth_self_attn=1, m_sub_slots=2,
        patch_kernel_freq=3,  # FE-RAW-1: kernel-3 path for the tiny F=3 config
    )
    model = cfg.build(n_outputs=N_CLASSES)
    model.eval()

    captured: dict[str, torch.Tensor | None] = {}
    orig_forward = model.readout.forward

    def _capturing_forward(latents, latent_valid=None):
        captured["latent_valid"] = (
            latent_valid.detach().clone() if latent_valid is not None else None
        )
        return orig_forward(latents, latent_valid=latent_valid)

    model.readout.forward = _capturing_forward  # type: ignore[assignment]

    B, C, T, F = 1, 3, 5, 3
    electrodes = torch.randn(B, C, T, F)
    support = torch.zeros(B, C, 3)
    support[0, 0, 0] = 1.0
    support[0, 1, 1] = 1.0

    _ = model(electrodes, support)

    assert captured["latent_valid"] is not None
    expected = torch.tensor([[True, True, True, True, False, False]])
    assert torch.equal(captured["latent_valid"], expected)
