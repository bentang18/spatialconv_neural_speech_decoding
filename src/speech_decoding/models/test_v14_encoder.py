"""Tests for the v14 Perceiver-IO encoder.

Validates shape, dtype, finiteness, param budget, and the load-bearing
invariants: the B36 hard block-diagonal pool routes attention by the one-hot
DK assignment (parcel-slot ← own-parcel electrodes only), parcel-id-tagged
latents are not shared across parcels, and `valid_mask` drops padded
electrodes from the cross-attn pool. (The exact-0.0 / sum-1.0 pool invariant
and the soft-bias removal live in ``test_v14_hard_pool.py``.)
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_encoder import (
    JepaPredictor,
    V14MeanPoolLinearHead,
    V14ParcelCollapsePMA,
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
    V14ParcelPerceiverWithHead,
    V14PerTaskAttentivePooler,
    V14Phase4FlatHead,
    V14PmaReadout,
    _JointTokenBlock,
    _PatchStem,
    compute_latent_valid_3way,
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


def _n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_v14_b2_jepa_predictor_depth_default_and_param_budget() -> None:
    """B36 B2 — ``JepaPredictor`` default depth = 3; param count at depth 3
    within ±2% of the locked target (~0.66M); depth knob {2,3,4} monotone;
    depth-2 ≠ the B36 default (regression guard against the old
    ``Predictor2Block`` depth-2 spec)."""
    # Analytical target for d_model=256, hidden=128, heads=4, MLP 4×, depth 3.
    # per block (TransformerEncoderLayer, norm_first):
    #   attn in_proj 3·128·128+3·128 + out_proj 128·128+128
    #   + ff 128·512+512 + 512·128+128 + 2·LN(2·128) = 198_272
    # + input_proj(256·128+128) + output_proj(128·256+256) + mask_token(128)
    target = 198_272 * 3 + (256 * 128 + 128) + (128 * 256 + 256) + 128
    assert target == 660_864  # pins the arithmetic

    pred3 = JepaPredictor(d_model=256)          # default depth
    assert pred3.depth == 3, "B36 default predictor depth must be 3 (D2)"
    n3 = _n_params(pred3)
    assert abs(n3 - target) <= 0.02 * target, (
        f"depth-3 param count {n3:,} not within ±2% of target {target:,}"
    )
    assert 500_000 < n3 < 800_000, f"~0.6M budget violated: {n3:,}"

    n2 = _n_params(JepaPredictor(d_model=256, depth=2))
    n4 = _n_params(JepaPredictor(d_model=256, depth=4))
    assert n2 < n3 < n4, f"depth knob not monotone: {n2:,} {n3:,} {n4:,}"
    assert n2 != target, "depth-2 must differ from the B36 default (depth-3)"


def test_v14_b2_jepa_predictor_output_at_masked_positions_only() -> None:
    """B36 B2 — predictor returns ``(n_masked, d_model)`` gathered at the
    masked query slots; the ungathered path returns ``(B, N_qry, d_model)``;
    gradients flow to every param including the learnable mask token."""
    torch.manual_seed(0)
    d_model, hidden = 256, 128
    B, n_ctx, n_qry = 2, 10, 6
    pred = JepaPredictor(d_model=d_model, hidden=hidden, depth=3)
    context = torch.randn(B, n_ctx, d_model)
    query_pos = torch.randn(B, n_qry, hidden)
    query_valid = torch.tensor(
        [[True, True, False, True, False, False],
         [True, False, False, False, False, False]]
    )
    n_masked = int(query_valid.sum())

    out = pred(context, query_pos, query_valid=query_valid)
    assert out.shape == (n_masked, d_model), (
        f"gathered output {tuple(out.shape)} != ({n_masked}, {d_model})"
    )
    assert torch.isfinite(out).all()

    # Ungathered path → (B, N_qry, d_model).
    out_full = pred(context, query_pos)
    assert out_full.shape == (B, n_qry, d_model)

    loss = out.pow(2).mean()
    loss.backward()
    assert pred.mask_token.grad is not None
    for name, p in pred.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), (
            f"JepaPredictor {name} did not receive finite gradient"
        )


def test_v14_b2_jepa_predictor_empty_mask_returns_empty() -> None:
    """B36 B6 precondition — all-padded query rows → gathered ``(0, d_model)``
    (no NaN), so the masked-empty L1 term is exactly 0."""
    pred = JepaPredictor(d_model=256, depth=2)
    context = torch.randn(2, 8, 256)
    query_pos = torch.randn(2, 5, 128)
    query_valid = torch.zeros(2, 5, dtype=torch.bool)
    out = pred(context, query_pos, query_valid=query_valid)
    assert out.shape == (0, 256)
    assert torch.isfinite(out).all()  # vacuously true; no NaN from empty softmax


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


# NOTE: the per-loss-head LayerNorms (ln_frame always-on; ln_mid / ln_utt
# only for the b31_plus_* sisters) live on the SSL student bundle
# (``experiments/v14_joint_module._V14StudentBundle``), NOT on the encoder.
# B31 dropped LN_mid + LN_utt from the default SSL path, and the 5/29 drift
# audit removed the encoder's never-read ln_mid/ln_frame/ln_utt scaffold.
# Their construction + EMA-teacher mirroring is tested in
# ``experiments/test_v14_joint_module.py``.


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
    """LAT-01 (B21 collapse-prevention lock 2026-05-25): the latent-slot tensor
    (80 at the B29 M=1 default, 320 under the M=4 ``R-m4-slots`` sister) is NOT
    a single free Parameter — it is reconstructed each forward from
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

    # _tiny_kwargs() runs at M=2, so the subslot table is instantiated (it is
    # None only at the M=1 B29 default — see test_v14_b29_m_sub_slots_default_is_one).
    assert model.learnable_subslot_embed is not None
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


def test_v14_encoder_eps_is_vestigial_under_b36_hard_pool() -> None:
    """B36: the hard block-diagonal pool consumes the one-hot DK support
    directly — there is no ``log(support + ε)`` smoothing — so ``eps`` is
    vestigial (reserved for the gated ``R-bna-soft`` sister) and varying it
    must NOT change the output. Regression guard: if a future edit reintroduces
    an eps-dependent softening on the default path, this fails."""
    assert DEFAULT_SUPPORT_BIAS_EPS == 1e-2
    kw = _tiny_kwargs()
    model = V14ParcelPerceiverModel(**kw)
    model.eval()
    electrodes = torch.randn(1, 3, kw["n_time_bins"], kw["n_freq_bins"])
    support = torch.eye(6)[None, :3, :].float()
    with torch.no_grad():
        tiny_eps = model(electrodes, support, eps=1e-9)
        big_eps = model(electrodes, support, eps=10.0)
    torch.testing.assert_close(tiny_eps, big_eps)


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


def test_v14_parcel_collapse_pma_default_is_unfrozen_under_b31() -> None:
    """B31 lock 2026-05-28
    ([[project_v14_b31_vjepa2_canonical_loss_2026_05_28]]): PMA's default
    ``freeze=False`` so the P3 cross-modal distillation construction picks
    up an unfrozen PMA (gradient flows from the Whisper distillation loss).
    Under B35 (2026-05-31) the PMA returns to Phase-4 but FROZEN there —
    :class:`V14PmaReadout` re-freezes it via ``requires_grad_(False)``, so
    the bare ``V14ParcelCollapsePMA`` default stays unfrozen (the natural P3
    construction) and the P4 freeze is owned by the readout wrapper."""
    pma = V14ParcelCollapsePMA(d_model=32, n_heads=4)
    for name, p in pma.named_parameters():
        assert p.requires_grad, f"{name} should be trainable under B31 default"


def test_v14_parcel_collapse_pma_freeze_true_kwarg_freezes_all_params() -> None:
    """Explicit ``freeze=True`` freezes every PMA parameter. Under B35 the
    P4 freeze is owned by :class:`V14PmaReadout` (``requires_grad_(False)``)
    rather than this kwarg, but the kwarg path still freezes correctly —
    used by any caller that wants an inert PMA at construction."""
    pma = V14ParcelCollapsePMA(d_model=32, n_heads=4, freeze=True)
    for name, p in pma.named_parameters():
        assert not p.requires_grad, f"{name} should be frozen under freeze=True"


def test_v14_config_build_default_readout_is_pma_mean_linear() -> None:
    """B35 (2026-05-31): the build-config default ``readout="pma_mean_linear"``
    wires a :class:`V14PmaReadout` whose PMA is FROZEN (loaded from P3 at the
    trainer level) and whose ONLY trainable param is the linear classifier.
    The encoder stays trainable in the bare build (the trainer freezes it for
    the real probe)."""
    cfg = V14ParcelPerceiver(
        n_freq_bins=6, n_time_bins=4, k_parcels=6,
        d_model=32, n_heads=4, depth_self_attn=2, m_sub_slots=2,
    )
    assert cfg.readout == "pma_mean_linear"
    model = cfg.build(n_classes=3)
    assert isinstance(model.readout, V14PmaReadout)
    assert model.readout.temporal == "mean"
    # The PMA is frozen; the linear classifier is the only trainable readout param.
    assert not any(p.requires_grad for p in model.readout.pma.parameters())
    assert all(p.requires_grad for p in model.readout.classifier.parameters())
    assert any(p.requires_grad for p in model.encoder.parameters())


def test_v14_config_build_pma_flatten_readout_sister() -> None:
    """B35 ``readout="pma_flatten_linear"`` (R-p4-flatten): frozen PMA collapse
    → flatten T_p·d → Linear. The classifier in-features = n_time_patches·d."""
    cfg = V14ParcelPerceiver(
        n_freq_bins=6, n_time_bins=4, k_parcels=6,
        d_model=32, n_heads=4, depth_self_attn=2, m_sub_slots=2,
        readout="pma_flatten_linear",
    )
    model = cfg.build(n_classes=3)
    assert isinstance(model.readout, V14PmaReadout)
    assert model.readout.temporal == "flatten"
    t_p = model.encoder.patch_stem.n_time_patches(cfg.n_time_bins)
    assert model.readout.classifier.in_features == t_p * cfg.d_model
    assert not any(p.requires_grad for p in model.readout.pma.parameters())


def test_v14_config_build_pma_timeattn_readout_sister() -> None:
    """B35 ``readout="pma_timeattn_linear"`` (R-p4-time-attn-pool): frozen PMA
    collapse → learned convex pool over T_p → Linear. Adds a ``time_score``
    head; the linear in-features = d (pooled to one vector)."""
    cfg = V14ParcelPerceiver(
        n_freq_bins=6, n_time_bins=4, k_parcels=6,
        d_model=32, n_heads=4, depth_self_attn=2, m_sub_slots=2,
        readout="pma_timeattn_linear",
    )
    model = cfg.build(n_classes=3)
    assert isinstance(model.readout, V14PmaReadout)
    assert model.readout.temporal == "timeattn"
    assert hasattr(model.readout, "time_score")
    assert model.readout.classifier.in_features == cfg.d_model


def test_v14_config_build_attentive_readout_sister() -> None:
    """B35 ``readout="attentive"`` (R-p4-attentive, retired B34 default): a
    :class:`V14PerTaskAttentivePooler` over the full parcel×time field; fully
    trainable, no PMA on the readout path."""
    cfg = V14ParcelPerceiver(
        n_freq_bins=6, n_time_bins=4, k_parcels=6,
        d_model=32, n_heads=4, depth_self_attn=2, m_sub_slots=2,
        readout="attentive",
    )
    model = cfg.build(n_classes=3)
    assert isinstance(model.readout, V14PerTaskAttentivePooler)
    assert not hasattr(model.readout, "pma")
    assert any(p.requires_grad for p in model.readout.parameters())


def test_v14_config_build_meanpool_readout_baseline() -> None:
    """B35 (2026-05-31): ``readout="meanpool"`` (R-p4-meanpool-no-pma) wires
    the :class:`V14MeanPoolLinearHead` reference (means over parcel×time,
    skipping the PMA); it produces finite per-class logits."""
    cfg = V14ParcelPerceiver(
        n_freq_bins=6, n_time_bins=4, k_parcels=6,
        d_model=32, n_heads=4, depth_self_attn=2, m_sub_slots=2,
        readout="meanpool",
    )
    model = cfg.build(n_classes=3)
    assert isinstance(model.readout, V14MeanPoolLinearHead)
    assert not hasattr(model.readout, "pma")
    assert any(p.requires_grad for p in model.readout.parameters())


# Phase-3 distillation pool + projection moved to the teacher side under the
# B05/B06 lock (2026-05-25 PM); the student is identity. Their tests live in
# extractors/test_whisper_teacher_pool.py and models/test_whisper_adapter.py.
# The former student-side V14Phase3TimePoolTriangular / V14Phase3DistillHead
# tests (5/22-era, t_out=50 @ 10 Hz, d_teacher=1280) were removed with the
# stale classes.


def test_v14_config_build_param_budget_at_first_pass_defaults() -> None:
    """First-pass defaults: d=256, heads=8, depth=6, M=1 (B29 Item 13 default;
    was M=4 pre-B29), K=80, T=17, F=38, n_classes=10 → ~13M params, comfortably
    under the 30M cap (also under the 25M Stage-2-mid-sweep cell of the
    13/25/40M sizing sweep per project_v14_scaling_law_param_sizing_2026_05_20)."""
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
    # B35 readout accounting (d=256, n_heads=8, n_classes=10): the frozen
    # P3-PMA = 263,424 params (query 256 + ln_q 512 + ln_kv 512 + q_proj
    # 65,536 + kv_proj 131,072 + out_proj 65,536; no rFF MLP), contributing
    # ZERO trainable params at P4. The only trainable readout param is the
    # 10-way linear = 2,570 (256·10 + 10).
    pma_total = sum(p.numel() for p in model.readout.pma.parameters())
    assert pma_total == 263_424, f"frozen PMA param count drifted: {pma_total:,}"
    assert not any(p.requires_grad for p in model.readout.pma.parameters())
    linear_trainable = sum(
        p.numel() for p in model.readout.classifier.parameters() if p.requires_grad
    )
    assert linear_trainable == 2_570, (
        f"trainable P4 linear drifted: {linear_trainable:,} (expected 2,570)"
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
    ``ref_idx``, and ``shaft_mask`` through to the inner encoder; otherwise
    Phase-4 downstream + the dispatch ``cfg.build()`` path can't exercise the
    B29 conditioning. (The B28 ``lambda_anat`` soft-gate was removed at B36 —
    the hard block-diagonal pool consumes the one-hot DK support directly.)"""
    cfg = V14ParcelPerceiver(
        n_freq_bins=6, n_time_bins=4, k_parcels=6,
        d_model=32, n_heads=4, depth_self_attn=2, m_sub_slots=2,
        subtype_embed_enabled=True,                  # post 5/28 flip: opt in
        ref_embed_enabled=True,                      # post B32 flip: opt in
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
    # B29 Item 13: SubSlotEmbed is DROPPED at the M=1 default (z[p] =
    # ParcelEmbed[p] + ε); the per-sub-slot table is instantiated only for
    # the R-m4-slots sister (M>1).
    assert model.learnable_subslot_embed is None, (
        "B29 Item 13: learnable_subslot_embed must be None at the M=1 "
        "default; the per-sub-slot table exists only for M>1."
    )
    # Cross-attn fires over exactly K latent slots; ε noise stays (K, 1, d).
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


def test_v14_b29_ref_embed_default_is_disabled() -> None:
    """B32 5/28 PM-late first-pass-no-input-aug lock: default
    ref_embed_enabled=False; the embedding module is None. Sister
    ``R-ref-aug-3-cell`` re-enables a fixed 3-cell vocab
    {shaftCAR, bipolar, Laplacian} paired with RefAugMultiStftView."""
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    assert model.ref_embed_enabled is False
    assert model.ref_embed is None
    # K/V reuse flag stays at its True default (it gates a code path that
    # is no-op when ``ref_embed_enabled=False``; the flag itself is
    # preserved so opt-in callers don't have to set it).
    assert model.ref_embed_reuse_kv is True


def test_v14_b29_no_subtype_embed_sister_disables_module() -> None:
    """``R-no-subtype-embed`` IS the default post 5/28 PM Item-11 flip,
    so this test just re-confirms that the default leaves the subtype
    embed module unset. Forward still runs end-to-end (B32 also leaves
    ref_embed off by default; the A1 additive branch is fully skipped
    when both flags are False)."""
    kw = _tiny_kwargs() | {"subtype_embed_enabled": False}
    model = V14ParcelPerceiverModel(**kw)
    assert model.subtype_embed is None
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
    """Different ref_idx values must produce different encoder outputs —
    proves the embed is wired into the forward, not a dead parameter.
    Ref embed defaults OFF post B32, so the sister-enabled path is what
    we test here."""
    kw = _tiny_kwargs() | {"ref_embed_enabled": True}
    model = V14ParcelPerceiverModel(**kw).eval()
    B, C, T, F = 2, 3, kw["n_time_bins"], kw["n_freq_bins"]
    K = kw["k_parcels"]
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
    """Vocab-range validation lives on the ref_embed branch — opt the
    embed in (B32 default OFF) so the validation path executes."""
    kw = _tiny_kwargs() | {"ref_embed_enabled": True}
    model = V14ParcelPerceiverModel(**kw).eval()
    B, C, T, F = 2, 3, kw["n_time_bins"], kw["n_freq_bins"]
    K = kw["k_parcels"]
    bad = torch.tensor([3, 0], dtype=torch.long)  # out of vocab=3
    with pytest.raises(ValueError, match="ref_idx"):
        model(torch.randn(B, C, T, F), torch.rand(B, C, K), ref_idx=bad)


def test_v14_b29_subtype_ref_config_propagates_through_build() -> None:
    """Config-level overrides flow through ``build``. Note: B32 sets the
    ``ref_embed_enabled`` default to False, so this test passes
    ``ref_embed_enabled=True`` explicitly to verify that the K/V-reuse
    flag still propagates (i.e. the input-only ref-embed sister wiring
    survives the cfg → encoder boundary)."""
    cfg = V14ParcelPerceiver(
        n_freq_bins=6, n_time_bins=4, k_parcels=6,
        d_model=32, n_heads=4, depth_self_attn=2,
        subtype_vocab=3, subtype_embed_enabled=False,
        ref_embed_enabled=True, ref_embed_reuse_kv=False,
    )
    wrap = cfg.build(n_classes=5)
    encoder = wrap.encoder
    assert encoder.subtype_vocab == 3
    assert encoder.subtype_embed is None
    assert encoder.ref_embed_reuse_kv is False
    assert encoder.ref_embed_enabled is True


# --- #119 activation checkpointing (2026-05-30 speedup audit, Tier-2) -------


def _ckpt_loss_and_grads(model: V14ParcelPerceiverModel) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Deterministic forward+backward on a fixed input; returns the scalar
    loss and a flat list of per-parameter grads (None grads → zeros)."""
    torch.manual_seed(1234)
    kw = _tiny_kwargs()
    B, C = 2, 7
    electrodes = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"])
    support = torch.zeros(B, C, kw["k_parcels"])
    support[..., 0] = 1.0
    model.zero_grad(set_to_none=True)
    out = model(electrodes, support)
    loss = out.float().pow(2).mean()
    loss.backward()
    grads = [
        (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for p in model.parameters()
    ]
    return loss.detach(), grads


def test_v14_gradient_checkpointing_is_bit_identical() -> None:
    """#119 (Tier-2 speedup): activation checkpointing on the encoder block
    stacks must not change numerics. The blocks contain no dropout, and the
    flag is gated on ``training and grad_enabled`` — under those conditions
    the recomputed forward (use_reentrant=False) is exact, so loss + every
    gradient must match the non-checkpointed run.

    Same model object, toggled flag → params are byte-for-byte identical
    across both runs; only the checkpointing path differs.
    """
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    model.train()

    assert model.gradient_checkpointing is False, (
        "#119: gradient_checkpointing must default OFF"
    )
    loss_off, grads_off = _ckpt_loss_and_grads(model)

    model.gradient_checkpointing = True
    loss_on, grads_on = _ckpt_loss_and_grads(model)

    assert torch.equal(loss_off, loss_on), (
        f"#119: checkpointed loss {loss_on.item()!r} != "
        f"non-checkpointed {loss_off.item()!r} (no dropout ⇒ exact)"
    )
    assert len(grads_off) == len(grads_on)
    for i, (g_off, g_on) in enumerate(zip(grads_off, grads_on)):
        assert torch.equal(g_off, g_on), (
            f"#119: grad[{i}] diverged under checkpointing "
            f"(max abs diff {(g_off - g_on).abs().max().item():.3e})"
        )


def test_v14_gradient_checkpointing_skipped_in_eval() -> None:
    """The checkpoint path is gated on ``self.training`` so the no_grad
    EMA-teacher pass (and any eval forward) is never checkpointed even when
    the flag is on. Verifies an eval-mode forward still runs and matches the
    flag-off eval forward exactly."""
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    model.eval()
    torch.manual_seed(7)
    kw = _tiny_kwargs()
    electrodes = torch.randn(1, 4, kw["n_time_bins"], kw["n_freq_bins"])
    support = torch.zeros(1, 4, kw["k_parcels"])
    support[..., 0] = 1.0

    with torch.no_grad():
        out_off = model(electrodes, support)
        model.gradient_checkpointing = True
        out_on = model(electrodes, support)

    assert torch.equal(out_off, out_on), (
        "#119: eval/no_grad forward must be unaffected by the checkpointing flag"
    )


# ─────────────────── B36 WS-B: masked-JEPA encoder seams ──────────────────


def _mask_fixture():
    """Tiny encoder + a 2-clip batch with a known parcel assignment.

    Electrodes 0,1 → parcel 0; electrode 2 → parcel 1; electrode 3 → parcel 2.
    Returns (model, electrodes, support).
    """
    torch.manual_seed(0)
    model = V14ParcelPerceiverModel(**_tiny_kwargs())
    model.eval()
    kw = _tiny_kwargs()
    B, C = 2, 4
    electrodes = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"])
    support = torch.zeros(B, C, kw["k_parcels"])
    support[:, 0, 0] = 1.0
    support[:, 1, 0] = 1.0
    support[:, 2, 1] = 1.0
    support[:, 3, 2] = 1.0
    return model, electrodes, support


def test_v14_b1_frontend_ln_normalizes_m2_tap() -> None:
    """B36 B1 — the M2 tap is post-``frontend_ln``: each token is
    LayerNorm-normalized over the feature axis (mean≈0, var≈1 at init)."""
    model, electrodes, support = _mask_fixture()
    assert isinstance(model.frontend_ln, torch.nn.LayerNorm)
    with torch.no_grad():
        taps = model(electrodes, support, return_taps=True)
    m2 = taps["M2"]  # (B, C, F_p, T_p, d)
    assert m2.shape[-1] == _tiny_kwargs()["d_model"]
    mean = m2.mean(dim=-1)
    var = m2.var(dim=-1, unbiased=False)
    torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-5, rtol=0)
    torch.testing.assert_close(var, torch.ones_like(var), atol=1e-3, rtol=1e-3)


def _patch_mask_to_bin_mask(token_mask: torch.Tensor) -> torch.Tensor:
    """(B, C, F_p, T_p) patch mask → (B, C, T_bins, F_bins) input-bin mask for
    _tiny_kwargs (kernel (3,2), stride (3,2): F 6→2, T 4→2, non-overlap)."""
    # token_mask freq axis (F_p=2) tiles 3 freq bins each; time axis (T_p=2)
    # tiles 2 time bins each. Input layout is (B, C, T_bins, F_bins).
    bin_ft = token_mask.repeat_interleave(3, dim=2).repeat_interleave(2, dim=3)
    # bin_ft is (B, C, F_bins=6, T_bins=4) → transpose to (B, C, T_bins, F_bins)
    return bin_ft.transpose(2, 3)


def test_v14_b5_token_mask_is_leakage_free() -> None:
    """B36 B5 — with a P1 token mask, changing a masked-region INPUT value
    does not change any VISIBLE token's M2 (front-end never encodes masked
    cells: they are zeroed pre-token-block)."""
    model, electrodes, support = _mask_fixture()
    B, C = electrodes.shape[0], electrodes.shape[1]
    F_p, T_p = _tiny_freq_patches(), _tiny_time_patches()
    token_mask = torch.zeros(B, C, F_p, T_p, dtype=torch.bool)
    # mask patch (f_p=0, t_p=1) on electrode 0, and (f_p=1,t_p=0) on electrode 3.
    token_mask[:, 0, 0, 1] = True
    token_mask[:, 3, 1, 0] = True

    with torch.no_grad():
        m2_a = model(electrodes, support, return_taps=True, token_mask=token_mask)["M2"]
        # perturb ONLY the input bins that belong to masked patches.
        bin_mask = _patch_mask_to_bin_mask(token_mask)
        electrodes_b = electrodes.clone()
        electrodes_b[bin_mask] = electrodes_b[bin_mask] + 123.0
        m2_b = model(electrodes_b, support, return_taps=True, token_mask=token_mask)["M2"]

    visible = ~token_mask  # (B, C, F_p, T_p)
    vis = visible.unsqueeze(-1).expand_as(m2_a)
    torch.testing.assert_close(m2_a[vis], m2_b[vis], atol=1e-6, rtol=0)
    # And the masked region's INPUT really differs (test is not vacuous):
    assert not torch.equal(electrodes, electrodes_b)


def test_v14_b8_three_way_latent_valid_partitions_covered() -> None:
    """B36 B8 — ``compute_latent_valid_3way`` splits ``covered`` into
    visible/target with teacher == covered: visible ⊎ target == covered,
    visible ∩ target == ∅, per time patch."""
    _model, _e, support = _mask_fixture()
    m = 2  # m_sub_slots in _tiny_kwargs
    K = _tiny_kwargs()["k_parcels"]
    T_p = _tiny_time_patches()
    B = support.shape[0]
    # mask parcel 0 at t=1 and parcel 2 at t=0 (both covered).
    ptm = torch.zeros(B, K, T_p, dtype=torch.bool)
    ptm[:, 0, 1] = True
    ptm[:, 2, 0] = True

    visible, target, teacher = compute_latent_valid_3way(
        support=support, valid_mask=None, m_sub_slots=m, parcel_time_mask=ptm,
    )
    L = K * m
    assert teacher.shape == (B, L)
    assert visible.shape == (B, L, T_p)
    assert target.shape == (B, L, T_p)
    # teacher == covered (parcels 0,1,2 covered → slots 0..5 True; 6..11 False)
    covered = teacher
    # partition: visible ⊎ target == covered (broadcast over time); disjoint.
    union = visible | target
    torch.testing.assert_close(
        union.float(), covered.unsqueeze(-1).expand(B, L, T_p).float(),
    )
    assert not bool((visible & target).any())
    # target lands exactly on masked covered slot-times (slot l=k*m+s).
    # parcel 0 → slots 0,1 ; parcel 2 → slots 4,5.
    assert bool(target[:, 0, 1].all()) and bool(target[:, 1, 1].all())
    assert bool(target[:, 4, 0].all()) and bool(target[:, 5, 0].all())
    # uncovered parcels (3,4,5 → slots 6..11) never visible/target.
    assert not bool(visible[:, 6:].any()) and not bool(target[:, 6:].any())


def test_v14_b5_parcel_time_mask_is_leakage_free() -> None:
    """B36 B5 (P2 paradigm B) — with a parcel-time mask, perturbing a masked
    parcel's dropped (electrode, masked-time) INPUT does not change any
    VISIBLE parcel token's M4. The visible-only encoder zeroes the dropped
    electrode-times pre-token-block AND excludes the masked parcel-times from
    the latent-SA keys, so visible M4 is independent of the masked region.
    Companion to the P1 ``token_mask`` leakage test above."""
    model, electrodes, support = _mask_fixture()
    K = _tiny_kwargs()["k_parcels"]
    m = _tiny_kwargs()["m_sub_slots"]
    T_p = _tiny_time_patches()
    B = electrodes.shape[0]
    # Mask parcel 0 at t_p=1 (electrodes 0,1 are assigned parcel 0).
    ptm = torch.zeros(B, K, T_p, dtype=torch.bool)
    ptm[:, 0, 1] = True

    with torch.no_grad():
        m4_a = model(
            electrodes, support, return_taps=True, parcel_time_mask=ptm,
        )["M4"]
        # Perturb ONLY parcel-0 electrodes (0,1) at the masked time patch:
        # t_p=1 → input time bins {2,3} (stride-2 non-overlap, T 4→2). Input
        # layout is (B, C, T_bins, F_bins), so the time axis is dim 2.
        electrodes_b = electrodes.clone()
        electrodes_b[:, 0:2, 2:4, :] = electrodes_b[:, 0:2, 2:4, :] + 123.0
        m4_b = model(
            electrodes_b, support, return_taps=True, parcel_time_mask=ptm,
        )["M4"]

    visible, _target, _teacher = compute_latent_valid_3way(
        support=support, valid_mask=None, m_sub_slots=m, parcel_time_mask=ptm,
    )
    vis = visible.unsqueeze(-1).expand_as(m4_a)  # (B, L, T_p, d)
    torch.testing.assert_close(m4_a[vis], m4_b[vis], atol=1e-6, rtol=0)
    # Non-vacuity: the masked-region input really changed.
    assert not torch.equal(electrodes, electrodes_b)


def test_v14_m2_only_early_exit_matches_full_forward() -> None:
    """B36 WS-B P1 efficiency — ``m2_only=True`` returns ONLY the M2 tap and
    that tap is byte-identical to the full forward's M2 (the skipped pool /
    inter-parcel encoder are pure downstream compute P1 never reads)."""
    model, electrodes, support = _mask_fixture()
    B, C = electrodes.shape[0], electrodes.shape[1]
    F_p, T_p = _tiny_freq_patches(), _tiny_time_patches()
    token_mask = torch.zeros(B, C, F_p, T_p, dtype=torch.bool)
    token_mask[:, 0, 0, 1] = True

    with torch.no_grad():
        full = model(
            electrodes, support, return_taps=True, token_mask=token_mask,
        )
        m2o = model(
            electrodes, support, return_taps=True, token_mask=token_mask,
            m2_only=True,
        )
    assert set(m2o.keys()) == {"M2"}
    torch.testing.assert_close(m2o["M2"], full["M2"], atol=0, rtol=0)


def test_v14_m2_only_requires_return_taps() -> None:
    """``m2_only`` is a tap-only short-circuit — it is meaningless without
    ``return_taps=True`` and must raise rather than silently fall through to
    the full (non-taps) tensor return."""
    model, electrodes, support = _mask_fixture()
    with pytest.raises(ValueError, match="return_taps"):
        model(electrodes, support, m2_only=True)


# ---------------------------------------------------------------------------
# B36 C5: per-corpus freq-patch validity (SWEC k0–21 → F-patch 0–6)
# ---------------------------------------------------------------------------


def test_c5_freq_patch_valid_mask_swec_k0_21_maps_to_fpatch_0_6() -> None:
    """``freq_patch_valid_mask`` maps the SWEC per-bin validity (k0–21 valid,
    k22–29 invalid) to per-PATCH validity. With the non-overlap (kernel=3,
    stride=3) stem, patch p covers bins [3p, 3p+3): patch 7 = {21,22,23}
    straddles the k21/k22 boundary → dropped (all-bins-valid rule). So valid
    F-patches = 0–6, exactly the plan's 'SWEC k0–21 → F-patch 0–6'."""
    from speech_decoding.models.v14_encoder import freq_patch_valid_mask

    bin_mask = torch.ones(30, dtype=torch.bool)
    bin_mask[22:] = False  # k22–29 invalid
    fpv = freq_patch_valid_mask(bin_mask, kernel_freq=3, stride_freq=3)
    assert fpv.shape == (10,)
    expected = torch.tensor([True] * 7 + [False] * 3)
    assert torch.equal(fpv, expected), f"got {fpv.tolist()}"


def _c5_fixture():
    """Encoder with F_p=10 (n_freq_bins=30), T_p=4 (n_time_bins=8), a 2-clip
    batch, and a SWEC-style freq_patch_valid masking F-patches 7–9."""
    torch.manual_seed(0)
    kw = {
        "n_freq_bins": 30, "n_time_bins": 8, "k_parcels": 6,
        "d_model": 32, "n_heads": 4, "depth_self_attn": 2, "m_sub_slots": 2,
    }
    model = V14ParcelPerceiverModel(**kw)
    model.eval()
    B, C = 2, 4
    electrodes = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"])
    support = torch.zeros(B, C, kw["k_parcels"])
    support[:, 0, 0] = 1.0
    support[:, 1, 0] = 1.0
    support[:, 2, 1] = 1.0
    support[:, 3, 2] = 1.0
    fpv = torch.tensor([True] * 7 + [False] * 3)  # F-patches 7–9 invalid
    return model, electrodes, support, fpv


def test_c5_freq_patch_excluded_from_keys_protects_valid_outputs() -> None:
    """With freq_patch_valid masking F-patches 7–9, BOTH the parcel latents
    (M4) and the valid-freq-patch M2 tokens are invariant to the CONTENT of
    the invalid freq bins (k22–29) — the masked patches are excluded as keys
    in both the per-electrode token-block SA and the hard-pool cross-attn.
    Patches 0–6 cover bins 0–20 (non-overlap stem), so the perturbation never
    enters a valid patch's Conv window; the only path is via the (masked-out)
    invalid-patch keys."""
    model, electrodes, support, fpv = _c5_fixture()
    pert = electrodes.clone()
    pert[:, :, :, 22:] += 50.0  # perturb invalid freq bins only

    with torch.no_grad():
        a = model(electrodes, support, return_taps=True, freq_patch_valid=fpv)
        b = model(pert, support, return_taps=True, freq_patch_valid=fpv)
    torch.testing.assert_close(a["M4"], b["M4"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        a["M2"][:, :, :7], b["M2"][:, :, :7], atol=1e-5, rtol=1e-5,
    )

    # Control: WITHOUT the mask, the same perturbation DOES move M4 — so it is
    # the freq-patch key mask, not an accident, protecting the valid outputs.
    with torch.no_grad():
        c = model(electrodes, support, return_taps=True)
        d = model(pert, support, return_taps=True)
    assert (c["M4"] - d["M4"]).abs().max() > 1e-4


def test_c5_freq_patch_valid_none_is_byte_identical() -> None:
    """freq_patch_valid=None and an all-True mask both reduce to the pre-C5
    forward — the BT no-op (all 30 bins valid)."""
    model, electrodes, support, _ = _c5_fixture()
    all_true = torch.ones(10, dtype=torch.bool)
    with torch.no_grad():
        none_out = model(electrodes, support, return_taps=True)
        true_out = model(
            electrodes, support, return_taps=True, freq_patch_valid=all_true,
        )
    torch.testing.assert_close(none_out["M4"], true_out["M4"], atol=0, rtol=0)
    torch.testing.assert_close(none_out["M2"], true_out["M2"], atol=0, rtol=0)


def test_c5_freq_patch_valid_rejects_bad_shape() -> None:
    """A freq_patch_valid that is neither (F_p,) nor (B, F_p) is a contract
    violation, not silently broadcast."""
    model, electrodes, support, _ = _c5_fixture()
    with pytest.raises(ValueError, match="freq_patch_valid"):
        model(
            electrodes, support, return_taps=True,
            freq_patch_valid=torch.ones(5, dtype=torch.bool),  # F_p is 10
        )
