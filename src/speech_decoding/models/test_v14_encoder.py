"""Tests for the v14 Perceiver-IO encoder.

Validates shape, dtype, finiteness, param budget, and the load-bearing
invariants: anatomy bias actually routes attention through `log(support+eps)`,
parcel-id-tagged latents are not shared across parcels, and `valid_mask`
zeros out padded electrodes from the cross-attn pool.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import (
    V14ClassifierHead,
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
    V14ParcelPerceiverWithHead,
)
from speech_decoding.studies.braintreebank.anatomy import DEFAULT_SUPPORT_BIAS_EPS


def _tiny_kwargs() -> dict:
    return {
        "n_freq_bins": 4,
        "n_time_bins": 5,
        "k_parcels": 6,
        "d_model": 32,
        "n_heads": 4,
        "depth_self_attn": 2,
        "m_sub_slots": 2,
    }


def test_v14_encoder_forward_shape_dtype_finite() -> None:
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    B, C = 2, 7
    electrodes = torch.randn(B, C, 5, 4)
    support = torch.zeros(B, C, 6)
    support[..., 0] = 1.0  # one-hot at parcel 0

    out = model(electrodes, support)

    assert out.shape == (B, 6 * 2, 32)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_v14_encoder_rejects_mismatched_support_shape() -> None:
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    electrodes = torch.randn(1, 4, 5, 4)
    bad_support = torch.zeros(1, 4, 7)  # K mismatch (7 vs 6)
    try:
        model(electrodes, bad_support)
    except ValueError as exc:
        assert "support shape" in str(exc)
    else:
        raise AssertionError("expected ValueError on K mismatch")


def test_v14_encoder_anatomy_bias_routes_attention() -> None:
    """When support is one-hot at parcel p, the (p, m) latent must attend
    only to electrodes whose support row is also p — not other electrodes.

    Tested with ``depth_self_attn=0`` so the cross-attn output is not diffused
    by the latent-side self-attn stack. (Self-attn mixes information across
    all latents by design; testing routing requires bypassing it.)
    """
    torch.manual_seed(0)
    kw = dict(_tiny_kwargs(), depth_self_attn=0)
    model = V14ParcelPerceiverModel(**kw)
    model.eval()

    B, C = 1, 4
    electrodes = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"])
    support = torch.zeros(B, C, kw["k_parcels"])
    support[0, 0, 0] = 1.0
    support[0, 1, 0] = 1.0
    support[0, 2, 3] = 1.0
    support[0, 3, 3] = 1.0

    out_baseline = model(electrodes, support, eps=1e-6)

    ablated = electrodes.clone()
    ablated[0, 2:4] = 0.0
    out_ablated = model(ablated, support, eps=1e-6)

    M = kw["m_sub_slots"]
    parcel0_slice = slice(0 * M, 1 * M)
    parcel3_slice = slice(3 * M, 4 * M)
    delta_parcel0 = (out_baseline[:, parcel0_slice] - out_ablated[:, parcel0_slice]).abs().mean()
    delta_parcel3 = (out_baseline[:, parcel3_slice] - out_ablated[:, parcel3_slice]).abs().mean()
    assert delta_parcel3 > delta_parcel0 * 100, (
        f"anatomy bias did not route: parcel0 delta {delta_parcel0:.4e} "
        f"should be ≪ parcel3 delta {delta_parcel3:.4e}"
    )


def test_v14_encoder_routing_diffuses_through_latent_self_attn() -> None:
    """With the full latent self-attn stack on, routing-through-eps is not
    a hard partition: ablating electrodes for parcel p will affect latents for
    other parcels too (by design — cross-parcel functional connectivity).

    Sanity-checks the architecture's claimed behavior: clean routing in the
    cross-attn followed by deliberate cross-parcel mixing in the self-attn stack.
    """
    torch.manual_seed(0)
    kw = _tiny_kwargs()  # depth_self_attn=2
    model = V14ParcelPerceiverModel(**kw)
    model.eval()

    B, C = 1, 4
    electrodes = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"])
    support = torch.zeros(B, C, kw["k_parcels"])
    support[0, 0, 0] = 1.0
    support[0, 1, 0] = 1.0
    support[0, 2, 3] = 1.0
    support[0, 3, 3] = 1.0

    out_baseline = model(electrodes, support, eps=1e-6)
    ablated = electrodes.clone()
    ablated[0, 2:4] = 0.0
    out_ablated = model(ablated, support, eps=1e-6)

    M = kw["m_sub_slots"]
    delta_parcel0 = (out_baseline[:, : M] - out_ablated[:, : M]).abs().mean()
    delta_parcel3 = (out_baseline[:, 3 * M : 4 * M] - out_ablated[:, 3 * M : 4 * M]).abs().mean()
    # parcel-3 still moves more than parcel-0, but mixing is allowed.
    assert delta_parcel3 > delta_parcel0, (
        f"directional routing broke under self-attn mixing: "
        f"parcel0 {delta_parcel0:.4e} parcel3 {delta_parcel3:.4e}"
    )


def test_v14_encoder_valid_mask_excludes_padded_electrodes() -> None:
    """A padded electrode (valid_mask=False) must not affect the encoder output."""
    torch.manual_seed(0)
    kw = _tiny_kwargs()
    model = V14ParcelPerceiverModel(**kw)
    model.eval()

    B, C = 1, 4
    electrodes_a = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"])
    electrodes_b = electrodes_a.clone()
    electrodes_b[0, 3] = 999.0  # arbitrary garbage in padded slot

    support = torch.zeros(B, C, kw["k_parcels"])
    support[0, 0, 0] = 1.0
    support[0, 1, 1] = 1.0
    support[0, 2, 2] = 1.0
    support[0, 3, 3] = 1.0

    valid = torch.tensor([[True, True, True, False]])

    out_a = model(electrodes_a, support, valid_mask=valid)
    out_b = model(electrodes_b, support, valid_mask=valid)
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)


def test_v14_encoder_default_eps_is_anatomy_prior_strength() -> None:
    """`forward()` default for `eps` matches `DEFAULT_SUPPORT_BIAS_EPS=1e-2`."""
    assert DEFAULT_SUPPORT_BIAS_EPS == 1e-2
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    model.eval()
    electrodes = torch.randn(1, 3, 5, 4)
    support = torch.eye(6)[None, :3, :].float()
    a = model(electrodes, support)
    b = model(electrodes, support, eps=DEFAULT_SUPPORT_BIAS_EPS)
    torch.testing.assert_close(a, b)


def test_v14_classifier_head_shape() -> None:
    head = V14ClassifierHead(d_model=32, n_classes=4, n_heads=4)
    latents = torch.randn(2, 12, 32)
    logits = head(latents)
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()


def test_v14_config_build_param_budget_at_first_pass_defaults() -> None:
    """First-pass defaults (d=128, depth=6, M=4, K=80, T=17, F=38, n_classes=10):
    parameter count must sit safely under the 30M cap."""
    cfg = V14ParcelPerceiver(
        n_freq_bins=38,
        n_time_bins=17,
        k_parcels=80,
    )
    model = cfg.build(n_classes=10)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 30_000_000, f"v14 first-pass over 30M cap: {n_params:,}"
    assert n_params < 10_000_000, (
        f"v14 first-pass unexpectedly large: {n_params:,} — design proposal "
        f"projected ~1.7-3M"
    )


def test_v14_config_build_returns_callable_module() -> None:
    cfg = V14ParcelPerceiver(
        n_freq_bins=4, n_time_bins=5, k_parcels=6,
        d_model=32, n_heads=4, depth_self_attn=2, m_sub_slots=2,
    )
    model = cfg.build(n_classes=3)
    assert isinstance(model, V14ParcelPerceiverWithHead)
    electrodes = torch.randn(2, 7, 5, 4)
    support = torch.zeros(2, 7, 6)
    support[..., 0] = 1.0
    logits = model(electrodes, support)
    assert logits.shape == (2, 3)
