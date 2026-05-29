"""Readout memory-padding-mask tests.

Replaces the original DETR-style ``V14ClassifierHead`` mask tests after T1.10:
the readout is now a frozen ``V14ParcelCollapsePMA`` (k=1 over the K*M parcel
axis, per timestep) followed by ``V14Phase4FlatHead``. The mask semantics
carry over — parcel slots with no electrode coverage in this subject are
padded memory and must not be attended to by the PMA query.

Pins:

1. ``V14ParcelCollapsePMA.forward(..., latent_valid=mask)`` is invariant to
   the activations of latent rows marked invalid.
2. ``latent_valid=None`` reduces to plain PMA (perturbing any latent affects
   the pooled output).
3. The wrapper computes ``latent_valid`` from ``support`` and ``valid_mask``
   and forwards it to the PMA.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import (
    V14ParcelCollapsePMA,
    V14ParcelPerceiver,
    V14ParcelPerceiverWithHead,
)


D_MODEL = 32
N_CLASSES = 3
N_HEADS = 4


def test_parcel_collapse_pma_is_invariant_to_perturbations_of_invalid_latents() -> None:
    """Standard memory-padding semantics: perturbing invalid latent rows must
    not change the PMA output. ``freeze=False`` so we can train if needed in
    a follow-up; the freeze-by-default smoke is exercised separately."""
    torch.manual_seed(0)
    pma = V14ParcelCollapsePMA(d_model=D_MODEL, n_heads=N_HEADS, freeze=False)
    pma.eval()

    B, L, T = 2, 8, 4
    latents = torch.randn(B, L, T, D_MODEL)
    latent_valid = torch.tensor(
        [
            [True, True, False, False, True, True, False, True],
            [False, True, True, True, False, True, True, False],
        ]
    )

    with torch.no_grad():
        out_ref = pma(latents, latent_valid=latent_valid)

        perturbed = latents.clone()
        for b in range(B):
            invalid_rows = (~latent_valid[b]).nonzero(as_tuple=False).flatten()
            for r in invalid_rows.tolist():
                perturbed[b, r] = torch.randn(T, D_MODEL) * 100.0
        out_perturbed = pma(perturbed, latent_valid=latent_valid)

    assert torch.allclose(out_ref, out_perturbed, atol=1e-5), (
        "PMA output must be invariant to perturbations of invalid latent rows"
    )


def test_parcel_collapse_pma_unmasked_attends_to_every_position() -> None:
    """Backward-compat: when ``latent_valid`` is omitted, every position
    contributes to the pooled output — perturbing any row changes it."""
    torch.manual_seed(0)
    pma = V14ParcelCollapsePMA(d_model=D_MODEL, n_heads=N_HEADS, freeze=False)
    pma.eval()

    B, L, T = 1, 4, 3
    latents = torch.randn(B, L, T, D_MODEL)

    with torch.no_grad():
        out_ref = pma(latents)
        perturbed = latents.clone()
        perturbed[0, 0] = perturbed[0, 0] + torch.randn(T, D_MODEL) * 10.0
        out_perturbed = pma(perturbed)

    assert not torch.allclose(out_ref, out_perturbed, atol=1e-5), (
        "without a mask, perturbing any latent must change the PMA output"
    )


def test_wrapper_passes_latent_valid_computed_from_support_and_mask() -> None:
    """Integration: the wrapper computes ``latent_valid`` from ``support`` and
    ``valid_mask`` and forwards it to the PMA. Captured by monkey-patching the
    PMA's forward and inspecting kwargs.

    Setup: B=1, C=4, K=3 parcels, M=2 sub-slots → L=6 latents.
      * electrode 0 → parcel 0, electrode 1 → parcel 1, electrodes 2,3 padded.
      * Parcel 2 has no coverage anywhere → both sub-slots invalid.
    Expected latent_valid: [[T,T, T,T, F,F]].
    """
    torch.manual_seed(0)
    cfg = V14ParcelPerceiver(
        n_freq_bins=3, n_time_bins=5, k_parcels=3,
        d_model=D_MODEL, n_heads=N_HEADS, depth_self_attn=1, m_sub_slots=2,
    )
    model = cfg.build(n_outputs=N_CLASSES)
    assert isinstance(model, V14ParcelPerceiverWithHead)
    model.eval()

    captured: dict[str, torch.Tensor] = {}
    orig_pma_forward = model.parcel_pma.forward

    def _capturing_forward(latents, latent_valid=None):
        if latent_valid is not None:
            captured["latent_valid"] = latent_valid.detach().clone()
        return orig_pma_forward(latents, latent_valid=latent_valid)

    model.parcel_pma.forward = _capturing_forward  # type: ignore[assignment]

    B, C, T, F = 1, 4, 5, 3
    electrodes = torch.randn(B, C, T, F)
    support = torch.zeros(B, C, 3)
    support[0, 0, 0] = 1.0
    support[0, 1, 1] = 1.0
    valid_mask = torch.tensor([[True, True, False, False]])

    _ = model(electrodes, support, valid_mask)

    assert "latent_valid" in captured, "wrapper must pass latent_valid to PMA"
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
    )
    model = cfg.build(n_outputs=N_CLASSES)
    model.eval()

    captured: dict[str, torch.Tensor] = {}
    orig_pma_forward = model.parcel_pma.forward

    def _capturing_forward(latents, latent_valid=None):
        captured["latent_valid"] = (
            latent_valid.detach().clone() if latent_valid is not None else None
        )
        return orig_pma_forward(latents, latent_valid=latent_valid)

    model.parcel_pma.forward = _capturing_forward  # type: ignore[assignment]

    B, C, T, F = 1, 3, 5, 3
    electrodes = torch.randn(B, C, T, F)
    support = torch.zeros(B, C, 3)
    support[0, 0, 0] = 1.0
    support[0, 1, 1] = 1.0

    _ = model(electrodes, support)

    assert captured["latent_valid"] is not None
    expected = torch.tensor([[True, True, True, True, False, False]])
    assert torch.equal(captured["latent_valid"], expected)
