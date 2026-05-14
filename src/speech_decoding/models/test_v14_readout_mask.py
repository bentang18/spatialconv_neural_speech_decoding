"""DETR-readout memory-padding-mask tests.

Canonical DETR (Carion 2020 §3.3) passes ``memory_key_padding_mask`` into the
decoder so object queries cannot attend to padded encoder features. The v14
analog: parcel slots with no electrode coverage for this subject are "padded
memory" — the task query must not attend to them.

These tests pin three behaviors:

1. ``V14ClassifierHead.forward(..., latent_valid=mask)`` is invariant to the
   activations of latent rows marked invalid.
2. ``latent_valid=None`` reproduces the pre-mask behavior (backward compat).
3. ``V14ParcelPerceiverWithHead`` computes ``latent_valid`` from ``support``
   and ``valid_mask`` and forwards it to the head.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import (
    V14ClassifierHead,
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
    V14ParcelPerceiverWithHead,
)


D_MODEL = 32
N_CLASSES = 3
N_HEADS = 4


def test_classifier_head_is_invariant_to_perturbations_of_invalid_latents() -> None:
    """Standard memory-padding semantics: perturbing invalid latent rows must
    not change the readout output. Tests the head in isolation with a hand-built
    latent tensor and validity mask."""
    torch.manual_seed(0)
    head = V14ClassifierHead(d_model=D_MODEL, n_classes=N_CLASSES, n_heads=N_HEADS)
    head.eval()

    B, L = 2, 8
    latents = torch.randn(B, L, D_MODEL)
    latent_valid = torch.tensor(
        [
            [True, True, False, False, True, True, False, True],
            [False, True, True, True, False, True, True, False],
        ]
    )

    out_ref = head(latents, latent_valid=latent_valid)

    perturbed = latents.clone()
    invalid = ~latent_valid                                 # (B, L)
    n_invalid = int(invalid.sum().item())
    perturbed[invalid] = torch.randn(n_invalid, D_MODEL) * 100.0

    out_perturbed = head(perturbed, latent_valid=latent_valid)

    assert torch.allclose(out_ref, out_perturbed, atol=1e-5), (
        "head output must be invariant to perturbations of invalid latent rows"
    )


def test_classifier_head_defaults_to_no_mask_for_backward_compat() -> None:
    """When ``latent_valid`` is omitted, the head reduces to its pre-mask form:
    perturbing any latent row changes the output. Ensures existing call sites
    that don't pass the mask still see the original behavior."""
    torch.manual_seed(0)
    head = V14ClassifierHead(d_model=D_MODEL, n_classes=N_CLASSES, n_heads=N_HEADS)
    head.eval()

    B, L = 1, 4
    latents = torch.randn(B, L, D_MODEL)
    out_ref = head(latents)

    perturbed = latents.clone()
    perturbed[0, 0, :] = perturbed[0, 0, :] + torch.randn(D_MODEL) * 10.0
    out_perturbed = head(perturbed)

    assert not torch.allclose(out_ref, out_perturbed, atol=1e-5), (
        "without a mask, perturbing any latent must change the head output"
    )


def test_wrapper_passes_latent_valid_computed_from_support_and_mask() -> None:
    """Integration: the wrapper computes ``latent_valid`` from ``support`` and
    ``valid_mask`` and forwards it to the head. Captured by monkey-patching the
    head's forward and inspecting kwargs.

    Setup: B=1, C=4, K=3 parcels, M=2 sub-slots → L=6 latents.
      * electrode 0 → parcel 0, electrode 1 → parcel 1, electrodes 2,3 padded
        (valid_mask all-False).
      * Parcel 2 has no coverage anywhere → both of its sub-slots invalid.
      * Padded electrodes 2,3 have all-zero support rows by construction.
    Expected latent_valid: parcel 0 valid (×M), parcel 1 valid (×M), parcel 2
    invalid (×M) → (B, 6) = [[T,T, T,T, F,F]].
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
    orig_head_forward = model.head.forward

    def _capturing_forward(latents, latent_valid=None):
        if latent_valid is not None:
            captured["latent_valid"] = latent_valid.detach().clone()
        return orig_head_forward(latents, latent_valid=latent_valid)

    model.head.forward = _capturing_forward  # type: ignore[assignment]

    B, C, T, F = 1, 4, 5, 3
    electrodes = torch.randn(B, C, T, F)
    support = torch.zeros(B, C, 3)
    support[0, 0, 0] = 1.0
    support[0, 1, 1] = 1.0
    valid_mask = torch.tensor([[True, True, False, False]])

    _ = model(electrodes, support, valid_mask)

    assert "latent_valid" in captured, "wrapper must pass latent_valid to head"
    assert captured["latent_valid"].shape == (B, 6)
    expected = torch.tensor([[True, True, True, True, False, False]])
    assert torch.equal(captured["latent_valid"], expected), (
        f"expected {expected.tolist()}, got {captured['latent_valid'].tolist()}"
    )


def test_wrapper_passes_no_mask_when_valid_mask_absent() -> None:
    """If ``valid_mask`` is not provided, the wrapper falls back to support-only
    coverage and still passes a ``latent_valid`` (every parcel with any nonzero
    support row is valid). This keeps the readout consistent even when the
    caller — e.g. legacy tests — omits ``valid_mask``."""
    torch.manual_seed(0)
    cfg = V14ParcelPerceiver(
        n_freq_bins=3, n_time_bins=5, k_parcels=3,
        d_model=D_MODEL, n_heads=N_HEADS, depth_self_attn=1, m_sub_slots=2,
    )
    model = cfg.build(n_outputs=N_CLASSES)
    model.eval()

    captured: dict[str, torch.Tensor] = {}
    orig_head_forward = model.head.forward

    def _capturing_forward(latents, latent_valid=None):
        captured["latent_valid"] = (
            latent_valid.detach().clone() if latent_valid is not None else None
        )
        return orig_head_forward(latents, latent_valid=latent_valid)

    model.head.forward = _capturing_forward  # type: ignore[assignment]

    B, C, T, F = 1, 3, 5, 3
    electrodes = torch.randn(B, C, T, F)
    support = torch.zeros(B, C, 3)
    support[0, 0, 0] = 1.0
    support[0, 1, 1] = 1.0
    # parcel 2 has no coverage; valid_mask omitted.

    _ = model(electrodes, support)

    assert captured["latent_valid"] is not None
    expected = torch.tensor([[True, True, True, True, False, False]])
    assert torch.equal(captured["latent_valid"], expected)
