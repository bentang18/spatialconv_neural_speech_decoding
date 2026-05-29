"""Tests for the v14 Perceiver-IO encoder.

Validates shape, dtype, finiteness, param budget, and the load-bearing
invariants: anatomy bias actually routes attention through `log(support+eps)`,
parcel-id-tagged latents are not shared across parcels, and `valid_mask`
zeros out padded electrodes from the cross-attn pool.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_encoder import (
    Predictor2Block,
    V14ParcelCollapsePMA,
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
    V14ParcelPerceiverWithHead,
    V14Phase3DistillHead,
    V14Phase3TimePoolTriangular,
    V14Phase4FlatHead,
    _JointTokenBlock,
    _PatchStem,
)
from speech_decoding.studies.braintreebank.anatomy import DEFAULT_SUPPORT_BIAS_EPS


def _tiny_kwargs() -> dict:
    # FE-02: defaults give F_p=2, T_p=2 with kernel=(3,2), stride=(3,2).
    # n_freq_bins=6: F_p = (6-3)/3 + 1 = 2.  n_time_bins=4: T_p = (4-2)/2 + 1 = 2.
    return {
        "n_freq_bins": 6,
        "n_time_bins": 4,
        "k_parcels": 6,
        "d_model": 32,
        "n_heads": 4,
        "depth_self_attn": 2,
        "m_sub_slots": 2,
    }


def _tiny_time_patches() -> int:
    # T_p for _tiny_kwargs(): (4-2)/2 + 1 = 2
    return 2


def _tiny_freq_patches() -> int:
    # F_p for _tiny_kwargs(): (6-3)/3 + 1 = 2
    return 2


def test_v14_fe02_patch_stem_shape_and_non_overlap() -> None:
    """FE-02 (B20 v4 lock 2026-05-24): non-overlap Conv2d patch stem.

    ``_PatchStem`` consumes ``(B, C, F, T)`` and emits ``(B, C, F_p, T_p, d)``
    with ``F_p = (F - k_f) // k_f + 1`` (stride = kernel ⇒ non-overlap).
    """
    stem = _PatchStem(d_model=32, kernel_freq=3, kernel_time=2)
    B, C, F, T = 2, 5, 30, 16
    x = torch.randn(B, C, F, T)
    out = stem(x)
    F_p = stem.n_freq_patches(F)
    T_p = stem.n_time_patches(T)
    assert F_p == (F - 3) // 3 + 1, "FE-02: non-overlap F-patch count"
    assert T_p == (T - 2) // 2 + 1, "FE-02: non-overlap T-patch count"
    assert out.shape == (B, C, F_p, T_p, 32), (
        f"_PatchStem must emit (B, C, F_p, T_p, d); got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all()
    # Conv2d stride == kernel: the spatial halves of each output cell receive
    # gradient from one non-overlapping (k_f × k_t) input window only.
    assert stem.conv.stride == (3, 2)
    assert stem.conv.kernel_size == (3, 2)
    assert stem.conv.padding == (0, 0)


def test_v14_fe03_per_patch_freq_embedding() -> None:
    """FE-03 (B20 v4 lock 2026-05-24): per-PATCH learnable freq embedding
    of shape ``(F_p, d)`` — NOT the v3 per-bin ``(F, d)`` table.
    """
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    F_p = _tiny_freq_patches()
    d = _tiny_kwargs()["d_model"]
    assert model.freq_embed.shape == (F_p, d), (
        f"FE-03: freq_embed must be (F_p={F_p}, d={d}); got "
        f"{tuple(model.freq_embed.shape)}"
    )
    assert model.n_freq_patches == F_p


def test_v14_fe04_joint_token_block_attends_across_freq_and_time() -> None:
    """FE-04 (B20 v4 lock 2026-05-24): ``_JointTokenBlock`` does single
    multi-head self-attention over the flat ``F_p · T_p`` token sequence
    (joint, not factorized) with RoPE on the time-axis only.

    Verifies:
      1. Forward preserves shape ``(B*C, F_p · T_p, d)``.
      2. Changing a single token at index (t_p, f_p) perturbs OTHER tokens
         too — i.e. attention mixes across both axes.
      3. The model exposes ``token_blocks`` of type ``_JointTokenBlock``.
    """
    torch.manual_seed(0)
    d = 32
    head_dim = d // 4
    F_p, T_p = 2, 3
    block = _JointTokenBlock(d_model=d, n_heads=4)
    block.eval()
    # RoPE table tiled: position i has time-patch (i // F_p) rotation.
    from speech_decoding.models.v14_encoder import _rope_freqs
    base_rope = _rope_freqs(head_dim, T_p)
    flat_pos = torch.arange(F_p * T_p)
    time_idx = flat_pos // F_p
    rope_time = base_rope[:, time_idx, :]

    x = torch.randn(2, F_p * T_p, d)
    out = block(x, rope_time)
    assert out.shape == x.shape

    # Perturb one token; verify others move (cross-token mixing).
    x_pert = x.clone()
    x_pert[:, 0] = 999.0
    out_pert = block(x_pert, rope_time)
    delta_other = (out[:, 1:] - out_pert[:, 1:]).abs().mean()
    assert delta_other > 1e-6, "FE-04: joint SA must mix across other tokens"

    # Sanity: model uses _JointTokenBlock, not factorized _TokenBlock.
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    for blk in model.token_blocks:
        assert isinstance(blk, _JointTokenBlock), (
            f"FE-04: token blocks must be _JointTokenBlock; got {type(blk)}"
        )


def test_v14_fe04_token_blocks_default_to_six() -> None:
    """FE-04 (B20 v4 lock 2026-05-24): N=6 joint token blocks at v4 defaults.

    Verifies the V14ParcelPerceiver config + the model defaults agree on
    n_token_blocks=6 (was N=4 under v3 factorized t×f).
    """
    cfg = V14ParcelPerceiver(n_freq_bins=12, n_time_bins=8, k_parcels=6)
    assert cfg.n_token_blocks == 6, (
        f"FE-04: V14ParcelPerceiver default n_token_blocks=6; got "
        f"{cfg.n_token_blocks}"
    )
    # Constructed model carries the same default.
    model = V14ParcelPerceiverModel(n_freq_bins=12, n_time_bins=8, k_parcels=6)
    assert len(model.token_blocks) == 6


def test_v14_mask03_shaft_mask_drops_electrodes_from_cross_attn() -> None:
    """MASK-03 (B03 mask-discipline lock 2026-05-25 PM): shaft_mask DROPs
    electrodes via key_padding_mask. Output must be invariant to changes
    in the values of dropped electrodes.
    """
    torch.manual_seed(0)
    kw = dict(_tiny_kwargs(), depth_self_attn=0)
    model = V14ParcelPerceiverModel(**kw)
    model.eval()

    B, C = 1, 4
    electrodes_a = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"])
    electrodes_b = electrodes_a.clone()
    electrodes_b[0, 0] = 999.0  # garbage in electrode 0
    electrodes_b[0, 2] = -999.0  # garbage in electrode 2

    support = torch.eye(kw["k_parcels"])[None, :C, :].float()  # full-rank
    shaft_mask = torch.tensor([[True, False, True, False]])    # drop 0 + 2

    out_a = model(electrodes_a, support, shaft_mask=shaft_mask)
    out_b = model(electrodes_b, support, shaft_mask=shaft_mask)
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)


def test_v14_mask03_shaft_mask_rejects_wrong_shape() -> None:
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    kw = _tiny_kwargs()
    electrodes = torch.randn(2, 3, kw["n_time_bins"], kw["n_freq_bins"])
    support = torch.zeros(2, 3, kw["k_parcels"])
    support[..., 0] = 1.0
    bad = torch.zeros(2, 5, dtype=torch.bool)  # C=5 mismatches actual C=3
    with pytest.raises(ValueError, match="shaft_mask"):
        model(electrodes, support, shaft_mask=bad)


def test_v14_mask03_shaft_mask_combines_with_valid_mask() -> None:
    """Combined drop = pad (~valid_mask) | shaft_mask. Output must be
    invariant to changes in the union of dropped electrodes."""
    torch.manual_seed(0)
    kw = dict(_tiny_kwargs(), depth_self_attn=0)
    model = V14ParcelPerceiverModel(**kw)
    model.eval()

    B, C = 1, 4
    electrodes_a = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"])
    electrodes_b = electrodes_a.clone()
    electrodes_b[0, 0] = 555.0  # padded via valid_mask
    electrodes_b[0, 2] = 777.0  # shaft-blocked
    support = torch.eye(kw["k_parcels"])[None, :C, :].float()
    valid_mask = torch.tensor([[False, True, True, True]])  # electrode 0 padded
    shaft_mask = torch.tensor([[False, False, True, False]])  # electrode 2 shaft

    out_a = model(electrodes_a, support, valid_mask=valid_mask, shaft_mask=shaft_mask)
    out_b = model(electrodes_b, support, valid_mask=valid_mask, shaft_mask=shaft_mask)
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)


def test_v14_mask04_predictor2block_shape_and_param_budget() -> None:
    """MASK-04 (B03c Paradigm-B predictor 2026-05-25 PM): standalone
    2-block transformer predictor. Shape preservation + param budget."""
    torch.manual_seed(0)
    predictor = Predictor2Block(d_model=256, hidden=128, n_heads=4, depth=2)
    x = torch.randn(2, 50, 256)
    out = predictor(x)
    assert out.shape == x.shape, f"got {tuple(out.shape)}, expected {tuple(x.shape)}"
    assert torch.isfinite(out).all()
    # Param budget: spec says ~0.2M; the standard nn.TransformerEncoderLayer
    # default of dim_feedforward = 4 × hidden gives ~0.5M at hidden=128.
    # Spec is approximate — we pin "lightweight" with a generous upper bound
    # and a hard cap below the encoder backbone (15M).
    n_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    assert 100_000 < n_params < 800_000, (
        f"Predictor2Block param count out of lightweight scale: {n_params:,}"
    )
    # Trainable end-to-end (gradient flow check).
    target = torch.randn_like(out)
    loss = (out - target).pow(2).mean()
    loss.backward()
    for name, p in predictor.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), (
            f"Predictor2Block {name} did not receive finite gradient"
        )


def test_v14_b29_item12_supervised_slot_mask_kwarg_dropped() -> None:
    """B29 Item 12 (5/27 PM-late) drops the pre-B29 ``supervised_slot_mask``
    override. Loss-side gating (L_mid_slot / L_post_frame) AND the latent-SA
    key_padding_mask now always operate on the support-derived slot bank —
    no per-subject ``parcels_supervised[subject]`` override path remains.
    Passing the retired kwarg must raise ``TypeError`` (unknown argument)."""
    kw = _tiny_kwargs()
    model = V14ParcelPerceiverModel(**kw)
    K, M = kw["k_parcels"], kw["m_sub_slots"]
    B, C = 1, 4
    electrodes = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"])
    support = torch.ones(B, C, K) / K
    bogus_mask = torch.ones(B, K * M, dtype=torch.bool)
    with pytest.raises(TypeError):
        model(electrodes, support, supervised_slot_mask=bogus_mask)


def test_v14_lat02_03_04_three_dedicated_layer_norms() -> None:
    """LAT-02 / LAT-03 / LAT-04 (B21+B22 collapse-prevention lock 2026-05-25):
    encoder owns three dedicated LayerNorm(d=d_model) modules, one per
    SSL loss head — ``ln_mid`` (M3), ``ln_frame`` (M4 frame head),
    ``ln_utt`` (M4 utterance head).

    They are NOT inserted in the default forward path; the loss heads apply
    them externally. Verifying only existence + dim here; usage is exercised
    by the Phase-1/2 loss tests once those land (Wave 6, LOSS-01).
    """
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    d = _tiny_kwargs()["d_model"]
    for name in ("ln_mid", "ln_frame", "ln_utt"):
        assert hasattr(model, name), f"{name} missing on encoder (LAT-02..04)"
        ln = getattr(model, name)
        assert isinstance(ln, torch.nn.LayerNorm), (
            f"{name} must be a torch.nn.LayerNorm; got {type(ln)}"
        )
        assert ln.normalized_shape == (d,), (
            f"{name} normalized_shape must be (d={d},); got {ln.normalized_shape}"
        )


def test_v14_lat05_return_taps_yields_m2_m3_m4_dict() -> None:
    """LAT-05 (B22 collapse-prevention dense-features 2026-05-25):
    ``forward(..., return_taps=True)`` returns a dict with three intermediate
    streams used by the SSL loss heads.

    Contract:
      - ``M2``: per-electrode-patch state, shape ``(B, C, F_p, T_p, d)``.
      - ``M3``: post cross-attn-0 / pre LN_mid, shape ``(B, K*M, T_p, d)``.
      - ``M4``: post encoder_ln, pre task-head LN, shape ``(B, K*M, T_p, d)``.

    The default ``return_taps=False`` path is unchanged: returns a single
    ``(B, K*M, T_p, d)`` tensor.
    """
    torch.manual_seed(0)
    kw = _tiny_kwargs()
    model = V14ParcelPerceiverModel(**kw)
    model.eval()

    B, C = 2, 4
    T, F = kw["n_time_bins"], kw["n_freq_bins"]
    electrodes = torch.randn(B, C, T, F)
    support = torch.zeros(B, C, kw["k_parcels"])
    support[..., 0] = 1.0

    out_default = model(electrodes, support)
    out_taps = model(electrodes, support, return_taps=True)

    assert isinstance(out_taps, dict), "return_taps=True must return a dict"
    assert set(out_taps.keys()) == {"M2", "M3", "M4"}, (
        f"return_taps=True must return keys {{M2, M3, M4}}; got "
        f"{set(out_taps.keys())}"
    )

    K, M = kw["k_parcels"], kw["m_sub_slots"]
    T_p = _tiny_time_patches()
    F_p = _tiny_freq_patches()
    d = kw["d_model"]

    assert out_taps["M2"].shape == (B, C, F_p, T_p, d), (
        f"LAT-05 M2 shape mismatch: got {tuple(out_taps['M2'].shape)}, "
        f"expected (B={B}, C={C}, F_p={F_p}, T_p={T_p}, d={d})"
    )
    assert out_taps["M3"].shape == (B, K * M, T_p, d), (
        f"LAT-05 M3 shape mismatch: got {tuple(out_taps['M3'].shape)}, "
        f"expected (B={B}, L={K * M}, T_p={T_p}, d={d})"
    )
    assert out_taps["M4"].shape == (B, K * M, T_p, d)

    # Default path is byte-identical to taps["M4"].
    torch.testing.assert_close(out_default, out_taps["M4"], atol=0.0, rtol=0.0)


def test_v14_encoder_lat01_identity_anchored_init() -> None:
    """LAT-01 (B21 collapse-prevention lock 2026-05-25): the 320-slot tensor
    is NOT a single free Parameter — it is reconstructed each forward from
    LearnableParcelEmbed + LearnableSubSlotEmbed + frozen ε.

    Verifies:
      1. ``learnable_parcel_embed`` exists with shape (K, d).
      2. ``learnable_subslot_embed`` exists with shape (M, d).
      3. ``latent_init_noise`` buffer exists with shape (K, M, d) and is NOT
         a Parameter (no gradient).
      4. Latents from two slots in the same parcel share the parcel embedding
         (so their difference equals subslot[s1] - subslot[s2] + ε noise term).
      5. The old single-tensor ``parcel_embedding`` attribute is gone.
    """
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    K = _tiny_kwargs()["k_parcels"]
    M = _tiny_kwargs()["m_sub_slots"]
    d = _tiny_kwargs()["d_model"]

    assert hasattr(model, "learnable_parcel_embed"), "LearnableParcelEmbed missing (LAT-01)"
    assert hasattr(model, "learnable_subslot_embed"), "LearnableSubSlotEmbed missing (LAT-01)"
    assert hasattr(model, "latent_init_noise"), "frozen ε buffer missing (LAT-01)"

    assert model.learnable_parcel_embed.shape == (K, d)
    assert model.learnable_subslot_embed.shape == (M, d)
    assert model.latent_init_noise.shape == (K, M, d)

    # ε is a buffer (broken-symmetry noise frozen at construction), not a
    # Parameter — must not receive gradient.
    assert not isinstance(model.latent_init_noise, torch.nn.Parameter)
    buffer_names = {n for n, _ in model.named_buffers()}
    assert "latent_init_noise" in buffer_names

    # Both learnable tables are real Parameters.
    param_names = {n for n, _ in model.named_parameters()}
    assert "learnable_parcel_embed" in param_names
    assert "learnable_subslot_embed" in param_names

    # The old single-tensor parcel_embedding attribute should be gone.
    assert not hasattr(model, "parcel_embedding"), (
        "LAT-01: old single-tensor parcel_embedding must be replaced by "
        "the identity-anchored construction; current attribute breaks the "
        "collapse-prevention contract."
    )


def test_v14_encoder_forward_shape_dtype_finite() -> None:
    """v4 contract (5/19 amendment §1 + B20 5/24 FE-02 lock): latents keep
    the time axis, but at the POST-PATCH frame count ``T_p``, not the raw
    bin count ``T``.

    Output shape is ``(B, K*M, T_p, d)`` per the FE-02 patch stem.
    """
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    B, C = 2, 7
    T, F = _tiny_kwargs()["n_time_bins"], _tiny_kwargs()["n_freq_bins"]  # T=4, F=6
    electrodes = torch.randn(B, C, T, F)
    support = torch.zeros(B, C, 6)
    support[..., 0] = 1.0  # one-hot at parcel 0

    out = model(electrodes, support)

    T_p = _tiny_time_patches()
    assert out.shape == (B, 6 * 2, T_p, 32), (
        "v14 encoder must return (B, K*M, T_p, d) per FE-02 (B20 5/24 lock); "
        f"got {tuple(out.shape)}"
    )
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_v14_encoder_rejects_mismatched_support_shape() -> None:
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    kw = _tiny_kwargs()
    electrodes = torch.randn(1, 4, kw["n_time_bins"], kw["n_freq_bins"])
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
    kw = _tiny_kwargs()
    model = V14ParcelPerceiverModel(**kw)
    model.eval()
    electrodes = torch.randn(1, 3, kw["n_time_bins"], kw["n_freq_bins"])
    support = torch.eye(6)[None, :3, :].float()
    a = model(electrodes, support)
    b = model(electrodes, support, eps=DEFAULT_SUPPORT_BIAS_EPS)
    torch.testing.assert_close(a, b)


def test_v14_phase4_head_shape() -> None:
    """T1.10 Phase-4 head: (B, T, d) → (B, n_classes) via flatten + linear."""
    head = V14Phase4FlatHead(n_time_bins=5, d_model=32, n_classes=4)
    pooled = torch.randn(2, 5, 32)
    logits = head(pooled)
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()


def test_v14_parcel_collapse_pma_shape() -> None:
    """T1.10 PMA k=1: (B, L, T, d) → (B, T, d) per the 5/22 spec."""
    pma = V14ParcelCollapsePMA(d_model=32, n_heads=4, freeze=False)
    latents = torch.randn(2, 12, 5, 32)
    out = pma(latents)
    assert out.shape == (2, 5, 32)
    assert torch.isfinite(out).all()


def test_v14_parcel_collapse_pma_is_frozen_by_default() -> None:
    """5/22 spec §3: PMA is frozen so Phase-3 SSL distillation and Phase-4
    downstream evaluation share a non-trainable parcel-collapse readout."""
    pma = V14ParcelCollapsePMA(d_model=32, n_heads=4)
    for name, p in pma.named_parameters():
        assert not p.requires_grad, f"{name} should be frozen but requires grad"


def test_v14_phase3_triangular_pool_shape_and_normalization() -> None:
    """5/22 spec §3 Phase-3 readout: triangular-window pool from T_in to T_out
    buckets. Output shape ``(B, T_out, d)``; each output bucket is a row-
    normalized weighted average of input bins (no information bleed-out)."""
    pool = V14Phase3TimePoolTriangular(t_in=73, t_out=50)
    x = torch.randn(2, 73, 32)
    out = pool(x)
    assert out.shape == (2, 50, 32)
    row_sums = pool.weights.sum(dim=1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)


def test_v14_phase3_distill_head_shape() -> None:
    """Phase-3 distillation head: ``(B, T_in, d) → (B, T_out, d_teacher)``
    via triangular pool + linear. Whisper-L8 teacher dim 1280."""
    head = V14Phase3DistillHead(t_in=73, t_out=50, d_model=32, d_teacher=1280)
    x = torch.randn(2, 73, 32)
    out = head(x)
    assert out.shape == (2, 50, 1280)
    assert torch.isfinite(out).all()


def test_v14_config_build_param_budget_at_first_pass_defaults() -> None:
    """v4 defaults (5/19 amendment §3): d=256, heads=8, depth=6, M=4, K=80,
    T=17, F=38, n_classes=10 → ~13M params, comfortably under the 30M cap
    (also under the 25M Stage-2-mid-sweep cell of the 13/25/40M sizing sweep
    per project_v14_scaling_law_param_sizing_2026_05_20)."""
    cfg = V14ParcelPerceiver(
        n_freq_bins=38,
        n_time_bins=17,
        k_parcels=80,
    )
    model = cfg.build(n_classes=10)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 30_000_000, f"v14 first-pass over 30M cap: {n_params:,}"
    assert n_params < 20_000_000, (
        f"v14 first-pass unexpectedly large: {n_params:,} — v4 spec projects "
        f"~13M at d=256, heads=8"
    )


def test_v14_config_build_returns_callable_module() -> None:
    # Use the same tiny shape contract as _tiny_kwargs() so the flat head's
    # T_p sizing matches the encoder output.
    cfg = V14ParcelPerceiver(
        n_freq_bins=6, n_time_bins=4, k_parcels=6,
        d_model=32, n_heads=4, depth_self_attn=2, m_sub_slots=2,
    )
    model = cfg.build(n_classes=3)
    assert isinstance(model, V14ParcelPerceiverWithHead)
    electrodes = torch.randn(2, 7, 4, 6)
    support = torch.zeros(2, 7, 6)
    support[..., 0] = 1.0
    logits = model(electrodes, support)
    assert logits.shape == (2, 3)


def test_v14_head_wrapper_forwards_b29_conditioning_to_encoder() -> None:
    """The head wrapper's ``forward`` must thread ``subject_subtype``,
    ``ref_idx``, ``lambda_anat``, and ``shaft_mask`` through to the inner
    encoder; otherwise Phase-4 downstream + the dispatch ``cfg.build()``
    path can't exercise the B29 conditioning."""
    cfg = V14ParcelPerceiver(
        n_freq_bins=6, n_time_bins=4, k_parcels=6,
        d_model=32, n_heads=4, depth_self_attn=2, m_sub_slots=2,
        subtype_embed_enabled=True,                  # post 5/28 flip: opt in
    )
    model = cfg.build(n_classes=3).eval()
    B, C, T, F = 2, 7, 4, 6
    electrodes = torch.randn(B, C, T, F)
    support = torch.zeros(B, C, 6)
    support[..., 0] = 1.0
    with torch.no_grad():
        # Different subject_subtype ids must produce different logits.
        zero_subtype = model(
            electrodes, support, subject_subtype=torch.zeros(B, dtype=torch.long),
        )
        one_subtype = model(
            electrodes, support, subject_subtype=torch.ones(B, dtype=torch.long),
        )
    assert (zero_subtype - one_subtype).abs().max().item() > 1e-4, (
        "head wrapper must forward subject_subtype into the encoder; "
        "different ids produced identical logits."
    )
    with torch.no_grad():
        # Different ref_idx ids must also produce different logits.
        zero_ref = model(electrodes, support, ref_idx=torch.zeros(B, dtype=torch.long))
        two_ref = model(electrodes, support, ref_idx=torch.full((B,), 2, dtype=torch.long))
    assert (zero_ref - two_ref).abs().max().item() > 1e-4, (
        "head wrapper must forward ref_idx into the encoder; "
        "different ids produced identical logits."
    )
    # Build graded support (every electrode covers multiple parcels) so the
    # anatomy-bias gate has actual contrast to scale.
    graded_support = torch.rand(B, C, 6) + 1e-3
    graded_support = graded_support / graded_support.sum(dim=-1, keepdim=True)
    with torch.no_grad():
        bias_on = model(
            electrodes, graded_support, lambda_anat=torch.tensor([1.0, 1.0]),
        )
        bias_mixed = model(
            electrodes, graded_support, lambda_anat=torch.tensor([1.0, 0.0]),
        )
    assert (bias_on - bias_mixed).abs().max().item() > 1e-4, (
        "head wrapper must forward lambda_anat into the encoder; "
        "per-clip gate had no effect."
    )


# --- B28-cross-attn (2026-05-27 PM) -----------------------------------
# Default ``cross_attn_positions=[0]`` is the Perceiver IO canonical
# (Jaegle 2021, arXiv:2107.14795). The 2-cross-attn variant is opt-in
# via the ``R-perceiver-original-2-cross-attns`` sister flag.


def test_v14_b28_cross_attn_default_is_single_block_at_position_0() -> None:
    """B28 default: a single cross-attn at position 0 (Perceiver IO)."""
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    assert model._cross_attn_at_block == (0,)
    assert len(model.cross_attns) == 1
    assert model.cross_attn is model.cross_attns[0]


def test_v14_b28_cross_attn_two_block_sister_at_zero_and_three() -> None:
    """R-perceiver-original-2-cross-attns sister: cross-attns at {0, 3}."""
    kwargs = _tiny_kwargs() | {"depth_self_attn": 4, "cross_attn_positions": [0, 3]}
    model = V14ParcelPerceiverModel(**kwargs)
    assert model._cross_attn_at_block == (0, 3)
    assert len(model.cross_attns) == 2


def test_v14_b28_cross_attn_positions_deduped_and_sorted() -> None:
    """Duplicates collapse; order is normalized so downstream looks deterministic."""
    kwargs = _tiny_kwargs() | {
        "depth_self_attn": 4,
        "cross_attn_positions": [3, 0, 0, 3],
    }
    model = V14ParcelPerceiverModel(**kwargs)
    assert model._cross_attn_at_block == (0, 3)


def test_v14_b28_cross_attn_requires_position_zero() -> None:
    """Position 0 (pre-stack routing) is non-optional under B28."""
    with pytest.raises(ValueError, match="must include 0"):
        V14ParcelPerceiverModel(
            **(_tiny_kwargs() | {"cross_attn_positions": [1]})
        )


def test_v14_b28_cross_attn_interior_position_must_have_latent_block() -> None:
    """Interior positions ``p > 0`` must satisfy ``p < depth_self_attn``."""
    # depth_self_attn=2 from _tiny_kwargs → only positions 0, 1 are valid.
    with pytest.raises(ValueError, match="requires depth_self_attn"):
        V14ParcelPerceiverModel(
            **(_tiny_kwargs() | {"cross_attn_positions": [0, 3]})
        )


def test_v14_b28_cross_attn_rejects_negative_positions() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        V14ParcelPerceiverModel(
            **(_tiny_kwargs() | {"cross_attn_positions": [0, -1]})
        )


def test_v14_b28_cross_attn_config_propagates_through_build() -> None:
    """``V14ParcelPerceiver.cross_attn_positions`` flows through to the model."""
    cfg = V14ParcelPerceiver(
        n_freq_bins=6, n_time_bins=4, k_parcels=6,
        d_model=32, n_heads=4, depth_self_attn=4, m_sub_slots=2,
        cross_attn_positions=[0, 3],
    )
    model = cfg.build(n_classes=3)
    encoder = model.encoder  # type: ignore[attr-defined]
    assert encoder._cross_attn_at_block == (0, 3)
    assert len(encoder.cross_attns) == 2


# --- B28-anatomy-warmup (2026-05-27 PM) ------------------------------
# ``lambda_anat`` is a forward-time scalar multiplier on the
# ``log(support+eps)`` cross-attn bias. The training-loop scheduler in
# ``ssl/warmup.py::anatomy_bias_warmup_schedule`` ramps it 0 → 1 over the
# last 25% of P1 ∪ first 25% of P2.


def test_v14_b28_lambda_anat_default_one_matches_legacy() -> None:
    """``lambda_anat=1.0`` (default) is the pre-B28 behavior — calling
    without the kwarg must produce the same output as λ=1.0 explicit."""
    torch.manual_seed(0)
    model = V14ParcelPerceiverModel(**_tiny_kwargs()).eval()
    electrodes = torch.randn(2, 5, 4, 6)
    support = torch.rand(2, 5, 6)
    with torch.no_grad():
        out_default = model(electrodes, support)
        out_explicit = model(electrodes, support, lambda_anat=1.0)
    torch.testing.assert_close(out_default, out_explicit)


def test_v14_b28_lambda_anat_zero_decouples_from_anatomy() -> None:
    """``lambda_anat=0`` zeros the anatomy bias; varying support no longer
    changes the encoder output (uniform-attention regime)."""
    torch.manual_seed(0)
    model = V14ParcelPerceiverModel(**_tiny_kwargs()).eval()
    electrodes = torch.randn(2, 5, 4, 6)
    support_a = torch.rand(2, 5, 6)
    support_b = torch.rand(2, 5, 6)  # different anatomy
    with torch.no_grad():
        out_a = model(electrodes, support_a, lambda_anat=0.0)
        out_b = model(electrodes, support_b, lambda_anat=0.0)
    # With λ=0, the only support-dependent path is _compute_latent_valid
    # (the latent SA key_padding_mask), which depends on support>0 not its
    # magnitude. Both support_a and support_b have all-positive entries
    # under torch.rand, so the masks match and outputs must be bit-equal.
    torch.testing.assert_close(out_a, out_b)


def test_v14_b28_lambda_anat_changes_output_when_bias_loadbearing() -> None:
    """At λ=1, support magnitudes route attention; at λ=0 they don't.
    The two outputs must differ for a non-trivial support distribution."""
    torch.manual_seed(0)
    model = V14ParcelPerceiverModel(**_tiny_kwargs()).eval()
    electrodes = torch.randn(2, 5, 4, 6)
    # Skewed support: each electrode strongly favors one parcel. Without
    # the bias, the cross-attn cannot recover this routing.
    support = torch.zeros(2, 5, 6)
    for c in range(5):
        support[:, c, c % 6] = 0.99
        support[:, c, (c + 1) % 6] = 0.01
    with torch.no_grad():
        out_on = model(electrodes, support, lambda_anat=1.0)
        out_off = model(electrodes, support, lambda_anat=0.0)
    diff = (out_on - out_off).abs().max().item()
    assert diff > 1e-3, (
        f"lambda_anat had no observable effect on the encoder output "
        f"(max |Δ| = {diff:.2e}); the bias-scaling wiring is dead"
    )


def test_v14_b28_lambda_anat_rejects_negative() -> None:
    model = V14ParcelPerceiverModel(**_tiny_kwargs()).eval()
    electrodes = torch.randn(1, 3, 4, 6)
    support = torch.zeros(1, 3, 6)
    support[..., 0] = 1.0
    with pytest.raises(ValueError, match="lambda_anat"):
        model(electrodes, support, lambda_anat=-0.5)


# --- B29 Item 12: per-clip λ_anat tensor (anatomy-rich vs SWEC) ----------


def test_v14_b29_lambda_anat_per_clip_tensor_accepted() -> None:
    """A ``(B,)`` ``lambda_anat`` tensor must run cleanly — B29 Item 12 spec."""
    torch.manual_seed(0)
    model = V14ParcelPerceiverModel(**_tiny_kwargs()).eval()
    electrodes = torch.randn(2, 5, 4, 6)
    support = torch.rand(2, 5, 6)
    lambda_anat = torch.tensor([1.0, 0.0])  # anatomy-rich, SWEC
    with torch.no_grad():
        out = model(electrodes, support, lambda_anat=lambda_anat)
    assert out.shape[0] == 2


def test_v14_b29_lambda_anat_per_clip_tensor_matches_scalar_for_uniform() -> None:
    """Per-clip tensor ``[1.0, 1.0]`` must equal scalar ``1.0`` for both clips."""
    torch.manual_seed(0)
    model = V14ParcelPerceiverModel(**_tiny_kwargs()).eval()
    electrodes = torch.randn(2, 5, 4, 6)
    support = torch.rand(2, 5, 6)
    with torch.no_grad():
        out_scalar = model(electrodes, support, lambda_anat=1.0)
        out_tensor = model(
            electrodes, support, lambda_anat=torch.tensor([1.0, 1.0]),
        )
    torch.testing.assert_close(out_scalar, out_tensor)


def test_v14_b29_lambda_anat_per_clip_tensor_gates_independently() -> None:
    """Two-clip batch with ``λ_anat = [1.0, 0.0]`` — the λ=0 clip's output
    must equal what you'd get from a single-clip scalar λ=0 run."""
    torch.manual_seed(0)
    model = V14ParcelPerceiverModel(**_tiny_kwargs()).eval()
    # Use skewed support so the λ-gating actually moves the output.
    support_one = torch.zeros(5, 6)
    for c in range(5):
        support_one[c, c % 6] = 0.99
        support_one[c, (c + 1) % 6] = 0.01
    electrodes_one = torch.randn(5, 4, 6)
    electrodes = electrodes_one.unsqueeze(0).repeat(2, 1, 1, 1)
    support = support_one.unsqueeze(0).repeat(2, 1, 1)
    with torch.no_grad():
        out_batch = model(
            electrodes, support, lambda_anat=torch.tensor([1.0, 0.0]),
        )
        out_off = model(
            electrodes[:1], support[:1], lambda_anat=0.0,
        )
        out_on = model(
            electrodes[:1], support[:1], lambda_anat=1.0,
        )
    torch.testing.assert_close(out_batch[1:2], out_off)
    torch.testing.assert_close(out_batch[0:1], out_on)


def test_v14_b29_lambda_anat_per_clip_tensor_rejects_wrong_shape() -> None:
    model = V14ParcelPerceiverModel(**_tiny_kwargs()).eval()
    electrodes = torch.randn(2, 5, 4, 6)
    support = torch.rand(2, 5, 6)
    with pytest.raises(ValueError, match="must have shape"):
        model(electrodes, support, lambda_anat=torch.tensor([1.0, 0.0, 0.5]))


def test_v14_b29_lambda_anat_per_clip_tensor_rejects_negative() -> None:
    model = V14ParcelPerceiverModel(**_tiny_kwargs()).eval()
    electrodes = torch.randn(2, 5, 4, 6)
    support = torch.rand(2, 5, 6)
    with pytest.raises(ValueError, match="must be >= 0"):
        model(electrodes, support, lambda_anat=torch.tensor([1.0, -0.1]))


# ---------------------------------------------------------------------------
# B29 Item 13: M=1 default (drop SubSlotEmbed plurality)
# ---------------------------------------------------------------------------


def test_v14_b29_m_sub_slots_default_is_one() -> None:
    """B29 Item 13 lock 2026-05-27 PM-late: default M=1 (was M=4). Latent
    count K*M = K when M=1.
    """
    kw_no_m = {k: v for k, v in _tiny_kwargs().items() if k != "m_sub_slots"}
    model = V14ParcelPerceiverModel(**kw_no_m)
    assert model.m_sub_slots == 1, (
        f"B29 Item 13: m_sub_slots default must be 1; got {model.m_sub_slots}"
    )
    K = kw_no_m["k_parcels"]
    d = kw_no_m["d_model"]
    # Latent reconstruction is K*M-shaped — with M=1 the cross-attn fires over
    # exactly K latent slots.
    assert model.learnable_subslot_embed.shape == (1, d)
    assert model.latent_init_noise.shape == (K, 1, d)


def test_v14_b29_m_sub_slots_default_forward_runs() -> None:
    """M=1 default must produce non-NaN encoder output."""
    kw_no_m = {k: v for k, v in _tiny_kwargs().items() if k != "m_sub_slots"}
    model = V14ParcelPerceiverModel(**kw_no_m).eval()
    B, C, T, F = 2, 3, kw_no_m["n_time_bins"], kw_no_m["n_freq_bins"]
    K = kw_no_m["k_parcels"]
    electrodes = torch.randn(B, C, T, F)
    support = torch.rand(B, C, K)
    out = model(electrodes, support)
    # Encoder returns (B, K*M, T_p, d) where M=1 ⇒ (B, K, T_p, d)
    assert out.shape[0] == B
    assert out.shape[1] == K, (
        f"M=1 default: latent dim must be K (not K*M); got {out.shape[1]}"
    )
    assert torch.isfinite(out).all()


def test_v14_b29_r_m4_slots_sister_restores_320_stack() -> None:
    """Sister ``R-m4-slots`` P0 sets m_sub_slots=4 via dispatch to falsify."""
    kw = _tiny_kwargs() | {"m_sub_slots": 4}
    model = V14ParcelPerceiverModel(**kw).eval()
    assert model.m_sub_slots == 4
    K = kw["k_parcels"]
    # K*M latents.
    B, C, T, F = 2, 3, kw["n_time_bins"], kw["n_freq_bins"]
    electrodes = torch.randn(B, C, T, F)
    support = torch.rand(B, C, K)
    out = model(electrodes, support)
    assert out.shape[1] == K * 4


def test_v14_b29_config_default_m_is_one() -> None:
    """V14ParcelPerceiver config: default m_sub_slots=1 (B29 Item 13)."""
    cfg = V14ParcelPerceiver(
        n_freq_bins=_tiny_kwargs()["n_freq_bins"],
        n_time_bins=_tiny_kwargs()["n_time_bins"],
        k_parcels=_tiny_kwargs()["k_parcels"],
        d_model=_tiny_kwargs()["d_model"],
        n_heads=_tiny_kwargs()["n_heads"],
        depth_self_attn=_tiny_kwargs()["depth_self_attn"],
    )
    assert cfg.m_sub_slots == 1, (
        f"V14ParcelPerceiver default m_sub_slots must be 1; got {cfg.m_sub_slots}"
    )


# ---------------------------------------------------------------------------
# B29 Item 11: subtype_embed (binary default) + ref_embed (3-cell vocab)
# ---------------------------------------------------------------------------


def test_v14_b29_subtype_embed_default_is_disabled() -> None:
    """5/28 PM precedent-audit flip: subtype_embed default ON → OFF
    (Agent 2 found M3AE precedent net-neutral on iEEG via DIVER-1 §4.1
    ablation). vocab=2 (binary) stays; reuse_kv stays True but is moot
    when the embed is absent."""
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    assert model.subtype_embed_enabled is False
    assert model.subtype_vocab == 2
    assert model.subtype_embed is None
    assert model.subtype_embed_reuse_kv is True


def test_v14_b29_r_subtype_embed_on_with_kv_reuse_sister_enables_module() -> None:
    """Sister ``R-subtype-embed-on-with-kv-reuse`` P0 (NEW, prior default
    → sister): enabling subtype_embed restores the binary embed with K/V
    reuse."""
    kw = _tiny_kwargs() | {"subtype_embed_enabled": True}
    model = V14ParcelPerceiverModel(**kw)
    assert model.subtype_embed_enabled is True
    assert model.subtype_vocab == 2
    assert model.subtype_embed is not None
    assert model.subtype_embed.weight.shape == (2, kw["d_model"])
    assert model.subtype_embed_reuse_kv is True


def test_v14_b29_r_subtype_embed_input_only_sister_enables_without_kv_reuse() -> None:
    """Sister ``R-subtype-embed-input-only`` P0 (PROMOTED M3AE-faithful,
    Geng 2022 §3.1): enabling subtype_embed with reuse_kv=False adds at
    A1 only."""
    kw = _tiny_kwargs() | {
        "subtype_embed_enabled": True,
        "subtype_embed_reuse_kv": False,
    }
    model = V14ParcelPerceiverModel(**kw)
    assert model.subtype_embed_enabled is True
    assert model.subtype_embed is not None
    assert model.subtype_embed_reuse_kv is False


def test_v14_b29_ref_embed_default_is_three_cell_enabled() -> None:
    """Default: ref_embed_enabled=True, fixed 3-cell vocab
    {shaftCAR, bipolar, Laplacian}."""
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    assert model.ref_embed_enabled is True
    assert model.ref_embed.weight.shape == (3, _tiny_kwargs()["d_model"])
    assert model.ref_embed_reuse_kv is True


def test_v14_b29_no_subtype_embed_sister_disables_module() -> None:
    """``R-no-subtype-embed`` P0 sets subtype_embed_enabled=False."""
    kw = _tiny_kwargs() | {"subtype_embed_enabled": False}
    model = V14ParcelPerceiverModel(**kw)
    assert model.subtype_embed is None
    # Forward still runs (ref_embed still active, contributing embed(0)).
    B, C, T, F = 2, 3, kw["n_time_bins"], kw["n_freq_bins"]
    out = model(torch.randn(B, C, T, F), torch.rand(B, C, kw["k_parcels"]))
    assert torch.isfinite(out).all()


def test_v14_b29_no_ref_embed_sister_disables_module() -> None:
    """``R-no-ref-embed`` P1 sets ref_embed_enabled=False."""
    kw = _tiny_kwargs() | {"ref_embed_enabled": False}
    model = V14ParcelPerceiverModel(**kw)
    assert model.ref_embed is None
    B, C, T, F = 2, 3, kw["n_time_bins"], kw["n_freq_bins"]
    out = model(torch.randn(B, C, T, F), torch.rand(B, C, kw["k_parcels"]))
    assert torch.isfinite(out).all()


def test_v14_b29_subtype_embed_three_way_vocab_sister() -> None:
    """``R-subtype-embed-3way`` P2 sets subtype_vocab=3 (DIVER-1).
    Subtype embed is OFF by default post 5/28 PM flip, so the 3-way
    sister must enable it explicitly."""
    kw = _tiny_kwargs() | {"subtype_vocab": 3, "subtype_embed_enabled": True}
    model = V14ParcelPerceiverModel(**kw)
    assert model.subtype_vocab == 3
    assert model.subtype_embed is not None
    assert model.subtype_embed.weight.shape == (3, kw["d_model"])


def test_v14_b29_subtype_vocab_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="subtype_vocab"):
        V14ParcelPerceiverModel(**(_tiny_kwargs() | {"subtype_vocab": 4}))


def test_v14_b29_subtype_embed_changes_output_when_id_changes() -> None:
    """Different subject_subtype ids must produce different encoder
    outputs — proves the embed is wired into the forward, not a dead
    parameter. Subtype embed defaults OFF post 5/28 PM flip, so the
    sister-enabled path is what we test here."""
    kw = _tiny_kwargs() | {"subtype_embed_enabled": True}
    model = V14ParcelPerceiverModel(**kw).eval()
    B, C, T, F = 2, 3, _tiny_kwargs()["n_time_bins"], _tiny_kwargs()["n_freq_bins"]
    K = _tiny_kwargs()["k_parcels"]
    electrodes = torch.randn(B, C, T, F)
    support = torch.rand(B, C, K)
    with torch.no_grad():
        out_zero = model(electrodes, support, subject_subtype=torch.zeros(B, dtype=torch.long))
        out_one = model(electrodes, support, subject_subtype=torch.ones(B, dtype=torch.long))
    diff = (out_zero - out_one).abs().max().item()
    assert diff > 1e-3, (
        f"subject_subtype had no observable effect on encoder output "
        f"(max |Δ| = {diff:.2e}); subtype_embed wiring is dead"
    )


def test_v14_b29_ref_embed_changes_output_when_id_changes() -> None:
    model = V14ParcelPerceiverModel(**_tiny_kwargs()).eval()
    B, C, T, F = 2, 3, _tiny_kwargs()["n_time_bins"], _tiny_kwargs()["n_freq_bins"]
    K = _tiny_kwargs()["k_parcels"]
    electrodes = torch.randn(B, C, T, F)
    support = torch.rand(B, C, K)
    with torch.no_grad():
        out_zero = model(electrodes, support, ref_idx=torch.zeros(B, dtype=torch.long))
        out_two = model(electrodes, support, ref_idx=torch.full((B,), 2, dtype=torch.long))
    diff = (out_zero - out_two).abs().max().item()
    assert diff > 1e-3, (
        f"ref_idx had no observable effect on encoder output "
        f"(max |Δ| = {diff:.2e}); ref_embed wiring is dead"
    )


def test_v14_b29_subtype_embed_input_only_sister_changes_output() -> None:
    """``R-subtype-embed-input-only`` (M3AE-faithful per Geng 2022 §3.1):
    subtype_embed_reuse_kv=False adds at A1 only, NOT in cross-attn K/V.

    The proof: with the same weights and same inputs, the default
    (reuse_kv=True) and the sister (reuse_kv=False) must produce
    distinguishable outputs whenever subject_subtype != 0. (When the id
    is 0, both branches add embed(0) — identical contributions — so the
    falsifier can only fire on non-zero ids.)
    """
    # Subtype embed is OFF by default post 5/28 PM flip; this sister test
    # compares ON+reuse_kv vs ON+input-only.
    kw_default = _tiny_kwargs() | {
        "ref_embed_enabled": False,             # isolate subtype contribution
        "subtype_embed_enabled": True,          # 5/28 flip: must opt in
    }
    kw_input_only = kw_default | {"subtype_embed_reuse_kv": False}
    model_default = V14ParcelPerceiverModel(**kw_default).eval()
    model_input_only = V14ParcelPerceiverModel(**kw_input_only).eval()
    model_input_only.load_state_dict(model_default.state_dict(), strict=True)
    assert model_default.subtype_embed_reuse_kv is True
    assert model_input_only.subtype_embed_reuse_kv is False
    B, C, T, F = 2, 3, kw_default["n_time_bins"], kw_default["n_freq_bins"]
    K = kw_default["k_parcels"]
    electrodes = torch.randn(B, C, T, F)
    support = torch.rand(B, C, K)
    nonzero_subtype = torch.ones(B, dtype=torch.long)
    with torch.no_grad():
        out_default = model_default(electrodes, support, subject_subtype=nonzero_subtype)
        out_io = model_input_only(electrodes, support, subject_subtype=nonzero_subtype)
    diff = (out_default - out_io).abs().max().item()
    assert diff > 1e-4, (
        f"Sister R-subtype-embed-input-only must produce a distinguishable "
        f"output vs the K/V-reuse default on nonzero subtype; max |Δ| = "
        f"{diff:.2e} (effectively identical — K/V reuse branch is dead)"
    )


def test_v14_b29_subtype_embed_invalid_id_rejected() -> None:
    """Subtype embed is OFF by default post 5/28 PM flip; bounds-check
    only fires when the embed is enabled."""
    kw = _tiny_kwargs() | {"subtype_embed_enabled": True}
    model = V14ParcelPerceiverModel(**kw).eval()
    B, C, T, F = 2, 3, kw["n_time_bins"], kw["n_freq_bins"]
    K = kw["k_parcels"]
    bad = torch.tensor([5, 0], dtype=torch.long)  # out of vocab=2
    with pytest.raises(ValueError, match="subject_subtype"):
        model(torch.randn(B, C, T, F), torch.rand(B, C, K), subject_subtype=bad)


def test_v14_b29_ref_idx_invalid_id_rejected() -> None:
    model = V14ParcelPerceiverModel(**_tiny_kwargs()).eval()
    B, C, T, F = 2, 3, _tiny_kwargs()["n_time_bins"], _tiny_kwargs()["n_freq_bins"]
    K = _tiny_kwargs()["k_parcels"]
    bad = torch.tensor([3, 0], dtype=torch.long)  # out of vocab=3
    with pytest.raises(ValueError, match="ref_idx"):
        model(torch.randn(B, C, T, F), torch.rand(B, C, K), ref_idx=bad)


def test_v14_b29_subtype_ref_config_propagates_through_build() -> None:
    cfg = V14ParcelPerceiver(
        n_freq_bins=6, n_time_bins=4, k_parcels=6,
        d_model=32, n_heads=4, depth_self_attn=2,
        subtype_vocab=3, subtype_embed_enabled=False,
        ref_embed_reuse_kv=False,
    )
    wrap = cfg.build(n_classes=5)
    encoder = wrap.encoder
    assert encoder.subtype_vocab == 3
    assert encoder.subtype_embed is None
    assert encoder.ref_embed_reuse_kv is False
    assert encoder.ref_embed_enabled is True
