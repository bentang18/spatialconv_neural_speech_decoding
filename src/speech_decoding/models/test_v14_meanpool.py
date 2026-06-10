"""B37 Chunk A — mean-before electrode→parcel pool + per-parcel ragged stem.

D1/D2/D4 (reports/b37_meanpool_freq_latent_spec_2026_06_10.md): a HARD masked
mean over each parcel's electrodes, taken on the raw |STFT| BEFORE the token
blocks. The electrode axis is consumed once, in the mean; the stem + token
blocks are per-parcel and frequency is preserved (M2 = (B, K, F_p, T_p, d)).

Pinned here:
  * the mean math: masked, per-parcel, normalized (mean not sum), uncovered →
    zero + latent_valid False, pad electrodes excluded;
  * the bias-carrying LINEARITY EQUIVALENCE — because the stem is a linear
    Conv2d with electrode-shared weights and the per-parcel pooling weights sum
    to 1, ``stem(parcel_raw) ≡ mean_c stem(x_in_c)`` exactly on covered parcels;
  * forward shapes/taps (frequency preserved), identity-latent baseline
    (M4 = encoder_ln(M2) in Chunk A), m2_only;
  * the ragged stem is bit-identical to dense on covered parcels;
  * the ``pool="cross_attn"`` sister (the B36 learned pool) is the default and
    is untouched;
  * the D7 composite mask (token_mask band / parcel_time_mask tube) zeroes
    masked tokens upstream of the token blocks and drops tubed parcels from the
    parcel-SA context (visible-only student);
  * the dropped conditioning args (shaft/subtype/ref/M3/freq_patch_valid) still
    fail loud on the mean path.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_encoder import V14ParcelPerceiverModel


# --------------------------------------------------------------------------- #
# configs
# --------------------------------------------------------------------------- #
def _mp_kw() -> dict:
    # patch_kernel_freq=3 avoids the FE-RAW-1 F=50 guard; d_model even for sincos.
    return {
        "n_freq_bins": 6,
        "n_time_bins": 4,
        "k_parcels": 6,
        "d_model": 32,
        "n_heads": 4,
        "depth_self_attn": 2,
        "m_sub_slots": 1,
        "n_token_blocks": 2,
        "patch_kernel_freq": 3,
        "pool": "mean",
    }


def _make(**over) -> V14ParcelPerceiverModel:
    kw = _mp_kw()
    kw.update(over)
    torch.manual_seed(0)
    return V14ParcelPerceiverModel(**kw)


def _inputs(B: int, C: int, kw: dict, *, seed: int = 1):
    """Default (B, C, T, F) electrode tokens + a one-hot support + valid mask."""
    g = torch.Generator().manual_seed(seed)
    et = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"], generator=g)
    support = torch.zeros(B, C, kw["k_parcels"])
    for c in range(C):
        support[:, c, c % kw["k_parcels"]] = 1.0
    valid = torch.ones(B, C, dtype=torch.bool)
    return et, support, valid


# --------------------------------------------------------------------------- #
# _mean_pool_electrodes math
# --------------------------------------------------------------------------- #
def test_mean_pool_is_masked_normalized_mean() -> None:
    enc = _make()
    B, C, F, T = 2, 6, 6, 4
    x_in = torch.randn(B, C, F, T)
    # electrodes 0,1,2 → parcel 0; 3 → parcel 1; 4,5 → parcel 2; parcels 3-5 empty.
    support = torch.zeros(B, C, 6)
    support[:, 0, 0] = support[:, 1, 0] = support[:, 2, 0] = 1.0
    support[:, 3, 1] = 1.0
    support[:, 4, 2] = support[:, 5, 2] = 1.0
    valid = torch.ones(B, C, dtype=torch.bool)

    parcel_raw, latent_valid = enc._mean_pool_electrodes(x_in, support, valid)
    assert parcel_raw.shape == (B, 6, F, T)
    assert latent_valid.shape == (B, 6)
    # parcel 0 = mean of electrodes 0,1,2 (a true mean, not a sum).
    torch.testing.assert_close(parcel_raw[:, 0], x_in[:, :3].mean(dim=1))
    torch.testing.assert_close(parcel_raw[:, 1], x_in[:, 3])
    torch.testing.assert_close(parcel_raw[:, 2], x_in[:, 4:6].mean(dim=1))
    # coverage mask: 0,1,2 covered; 3,4,5 empty.
    assert latent_valid[:, :3].all()
    assert not latent_valid[:, 3:].any()
    # uncovered parcels are exactly zero.
    torch.testing.assert_close(parcel_raw[:, 3:], torch.zeros(B, 3, F, T))


def test_mean_pool_excludes_pad_electrodes() -> None:
    enc = _make()
    B, C, F, T = 1, 6, 6, 4
    x_in = torch.randn(B, C, F, T)
    support = torch.zeros(B, C, 6)
    support[:, 0, 0] = support[:, 1, 0] = 1.0   # both into parcel 0
    valid = torch.ones(B, C, dtype=torch.bool)
    valid[:, 1] = False                          # electrode 1 is a pad

    parcel_raw, latent_valid = enc._mean_pool_electrodes(x_in, support, valid)
    # parcel 0 = electrode 0 ONLY (pad electrode 1 dropped, not averaged in).
    torch.testing.assert_close(parcel_raw[:, 0], x_in[:, 0])
    assert latent_valid[:, 0].all()


def test_mean_pool_all_pad_parcel_is_uncovered() -> None:
    enc = _make()
    B, C, F, T = 1, 6, 6, 4
    x_in = torch.randn(B, C, F, T)
    support = torch.zeros(B, C, 6)
    support[:, 0, 0] = 1.0
    valid = torch.ones(B, C, dtype=torch.bool)
    valid[:, 0] = False                          # the only electrode in parcel 0 is pad

    parcel_raw, latent_valid = enc._mean_pool_electrodes(x_in, support, valid)
    assert not latent_valid[:, 0].any()
    torch.testing.assert_close(parcel_raw[:, 0], torch.zeros(B, F, T))


def test_mean_pool_none_valid_mask_includes_all() -> None:
    enc = _make()
    B, C, F, T = 1, 6, 6, 4
    x_in = torch.randn(B, C, F, T)
    support = torch.zeros(B, C, 6)
    support[:, 0, 0] = support[:, 1, 0] = 1.0
    pr_none, lv_none = enc._mean_pool_electrodes(x_in, support, None)
    pr_all, lv_all = enc._mean_pool_electrodes(
        x_in, support, torch.ones(B, C, dtype=torch.bool)
    )
    torch.testing.assert_close(pr_none, pr_all)
    torch.testing.assert_close(lv_none.float(), lv_all.float())


# --------------------------------------------------------------------------- #
# D2 linearity equivalence: mean-before ≡ stem-then-mean (bias included)
# --------------------------------------------------------------------------- #
def test_stem_mean_before_equals_mean_after_stem() -> None:
    """The crux of D2: the linear electrode-shared Conv2d stem commutes with
    the per-parcel mean because the pooling weights sum to 1 — so taking the
    mean BEFORE the stem (cheap, per-parcel) is bit-equal to a per-electrode
    stem then mean (the conceptual baseline), bias and all, on covered parcels.
    """
    enc = _make()
    # The stem conv bias inits to zero; inject a non-zero bias so the
    # equivalence exercises the bias term (which cancels iff the per-parcel
    # pooling weights sum to 1), matching a trained model, not just weights.
    torch.nn.init.normal_(enc.patch_stem.conv.bias)
    B, C, F, T = 2, 6, 6, 4
    x_in = torch.randn(B, C, F, T)
    support = torch.zeros(B, C, 6)
    support[:, 0, 0] = support[:, 1, 0] = support[:, 2, 0] = 1.0
    support[:, 3, 1] = 1.0
    support[:, 4, 2] = support[:, 5, 2] = 1.0
    valid = torch.ones(B, C, dtype=torch.bool)

    parcel_raw, latent_valid = enc._mean_pool_electrodes(x_in, support, valid)
    mean_before = enc.patch_stem(parcel_raw)                  # (B, K, F_p, T_p, d)

    # mean-AFTER: stem every electrode, then average over each parcel's set.
    stem_per_elec = enc.patch_stem(x_in)                      # (B, C, F_p, T_p, d)
    w = (support > 0).float()                                 # (B, C, K)
    denom = w.sum(dim=1).clamp(min=1.0)                       # (B, K)
    mean_after = torch.einsum("bck,bcftd->bkftd", w, stem_per_elec)
    mean_after = mean_after / denom[:, :, None, None, None]

    covered = latent_valid                                    # (B, K)
    torch.testing.assert_close(
        mean_before[covered], mean_after[covered], rtol=1e-5, atol=1e-5
    )


# --------------------------------------------------------------------------- #
# forward shapes / taps / identity-latent baseline
# --------------------------------------------------------------------------- #
def test_meanpool_forward_taps_preserve_frequency() -> None:
    enc = _make().eval()
    kw = _mp_kw()
    B, C = 2, 8
    et, support, valid = _inputs(B, C, kw)
    out = enc(et, support, valid_mask=valid, return_taps=True)
    K, F_p, d = kw["k_parcels"], enc.n_freq_patches, kw["d_model"]
    T_p = enc.patch_stem.n_time_patches(kw["n_time_bins"])
    assert set(out) == {"M2", "M4", "latent_valid"}
    assert out["M2"].shape == (B, K, F_p, T_p, d)
    assert out["M4"].shape == (B, K, F_p, T_p, d)        # frequency preserved (D3)
    assert out["latent_valid"].shape == (B, K)
    assert torch.isfinite(out["M2"]).all()
    assert torch.isfinite(out["M4"]).all()


# --------------------------------------------------------------------------- #
# D5 thin parcel-SA-only latent
# --------------------------------------------------------------------------- #
def test_latent_parcel_blocks_only_built_for_mean_pool() -> None:
    # Each pool builds ONLY its own per-parcel pool + inter-parcel latent: the
    # mean path carries the thin parcel-SA stack and NOT the B36 cross_attn
    # routing pool OR latent (no dead EMA/optimizer-tracked params), and the
    # cross_attn sister is byte-identical to before B37.
    mean = _make()
    assert hasattr(mean, "latent_parcel_blocks")
    assert len(mean.latent_parcel_blocks) == 2              # default depth (D5)
    assert not hasattr(mean, "latent_blocks")               # no dead B36 latent on mean
    assert not hasattr(mean, "cross_attns")                 # hard mean replaces the pool
    assert not hasattr(mean, "cross_attn")                  # ...incl. the legacy alias
    # The dead params are GENUINELY gone, not just unused: no cross_attns / B36
    # latent params appear anywhere in the mean model's parameter set.
    mean_param_tops = {n.split(".", 1)[0] for n, _ in mean.named_parameters()}
    assert "cross_attns" not in mean_param_tops
    assert "latent_blocks" not in mean_param_tops
    ca = V14ParcelPerceiverModel(
        n_freq_bins=6, n_time_bins=4, k_parcels=6, d_model=32, n_heads=4,
        depth_self_attn=2, m_sub_slots=1, patch_kernel_freq=3,  # cross_attn default
    )
    assert not hasattr(ca, "latent_parcel_blocks")          # sister param set untouched
    assert hasattr(ca, "latent_blocks")                     # cross_attn keeps its latent
    assert len(ca.latent_blocks) == 2                       # == depth_self_attn
    assert hasattr(ca, "cross_attns")                       # ...and its routing pool
    assert ca.cross_attn is ca.cross_attns[0]               # ...and the legacy alias


def test_partition_parameters_assigns_latent_parcel_to_parcel() -> None:
    """The B37 parcel-SA latent is the mean-path inter-parcel op, so its params
    must land in the PARCEL staging bucket (the D8 discriminative-LR group) —
    and ``partition_parameters_for_staging`` must NOT raise on a mean model."""
    enc = _make()
    frontend, parcel = enc.partition_parameters_for_staging()  # no raise
    parcel_ids = {id(p) for p in parcel}
    frontend_ids = {id(p) for p in frontend}
    lp_params = list(enc.latent_parcel_blocks.parameters())
    assert lp_params                                            # has trainable params
    for p in lp_params:
        assert id(p) in parcel_ids                             # → parcel group
        assert id(p) not in frontend_ids                       # → not front-end
    # Every parameter is partitioned exactly once (the guard's whole point).
    assert len(frontend) + len(parcel) == len(list(enc.parameters()))

    # The shared guard must ALSO still partition the cross_attn pool (the B36
    # latent → parcel) without raising — the fix touched the shared frozenset.
    ca = V14ParcelPerceiverModel(
        n_freq_bins=6, n_time_bins=4, k_parcels=6, d_model=32, n_heads=4,
        depth_self_attn=2, m_sub_slots=1, patch_kernel_freq=3,  # cross_attn default
    )
    ca_frontend, ca_parcel = ca.partition_parameters_for_staging()  # no raise
    ca_parcel_ids = {id(p) for p in ca_parcel}
    for p in ca.latent_blocks.parameters():
        assert id(p) in ca_parcel_ids                          # B36 latent → parcel
    for p in ca.cross_attns.parameters():
        assert id(p) in ca_parcel_ids                          # routing pool → parcel
    assert len(ca_frontend) + len(ca_parcel) == len(list(ca.parameters()))


def test_latent_depth_param_controls_block_count() -> None:
    for depth in (1, 3):
        assert len(_make(latent_parcel_depth=depth).latent_parcel_blocks) == depth


def test_latent_depth_must_be_positive() -> None:
    with pytest.raises(ValueError, match="latent_parcel_depth"):
        _make(latent_parcel_depth=0)


def test_latent_is_no_longer_identity() -> None:
    """The real parcel-SA latent mixes parcels, so M4 != encoder_ln(M2)
    (the Chunk-A identity placeholder is gone)."""
    enc = _make().eval()
    kw = _mp_kw()
    et, support, valid = _inputs(2, 8, kw)
    out = enc(et, support, valid_mask=valid, return_taps=True)
    assert (out["M4"] - enc.encoder_ln(out["M2"])).abs().max() > 1e-4


def _rand_m2(enc: V14ParcelPerceiverModel, *, B: int, T_p: int, seed: int):
    F_p, K, d = enc.n_freq_patches, enc.k_parcels, enc.d_model
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, K, F_p, T_p, d, generator=g), K


def _perturb(t: torch.Tensor, seed: int) -> torch.Tensor:
    # A VARIANCE-bearing perturbation: a constant added to all d channels is
    # invisible to the pre-LN attention (LayerNorm is shift-invariant), so the
    # mixing/exclusion probes must move the post-LN signal, not just the mean.
    g = torch.Generator().manual_seed(seed)
    return t + torch.randn(t.shape, generator=g)


def test_parcel_latent_freq_time_ride_in_batch_dim() -> None:
    """freq×time are folded into the batch dim and NEVER attended: perturbing
    m2 at one (f,t) slice changes z only at that slice (covered parcels)."""
    enc = _make().eval()
    m2, K = _rand_m2(enc, B=2, T_p=5, seed=0)
    lv = torch.ones(2, K, dtype=torch.bool)
    z0 = enc._parcel_latent(m2, lv, use_ckpt=False)
    m2p = m2.clone()
    m2p[:, :, 1, 2, :] = _perturb(m2p[:, :, 1, 2, :], seed=11)   # one (f=1,t=2) slice
    z1 = enc._parcel_latent(m2p, lv, use_ckpt=False)
    delta = (z1 - z0).abs()
    assert delta[:, :, 1, 2, :].max() > 1e-5                    # that slice moved
    delta[:, :, 1, 2, :] = 0.0
    assert delta.max() < 1e-6                                    # every other (f,t) untouched


def test_parcel_latent_mixes_across_covered_parcels() -> None:
    """Perturbing covered parcel j changes a different covered parcel i's z —
    cross-parcel integration is the latent's whole job."""
    enc = _make().eval()
    m2, K = _rand_m2(enc, B=1, T_p=4, seed=1)
    lv = torch.ones(1, K, dtype=torch.bool)
    z0 = enc._parcel_latent(m2, lv, use_ckpt=False)
    m2p = m2.clone()
    m2p[:, 3] = _perturb(m2p[:, 3], seed=12)                     # perturb parcel 3
    z1 = enc._parcel_latent(m2p, lv, use_ckpt=False)
    assert (z1 - z0).abs()[:, 0].max() > 1e-5                  # parcel 0 (≠3) moved


def test_parcel_latent_excludes_uncovered_parcels_as_keys() -> None:
    """An uncovered parcel is bidirectionally excluded: perturbing its m2 does
    NOT change any covered parcel's z (so covered M4 is desync-proof)."""
    enc = _make().eval()
    m2, K = _rand_m2(enc, B=1, T_p=4, seed=2)
    lv = torch.ones(1, K, dtype=torch.bool)
    lv[:, 5] = False                                            # parcel 5 uncovered
    z0 = enc._parcel_latent(m2, lv, use_ckpt=False)
    m2p = m2.clone()
    m2p[:, 5] = _perturb(m2p[:, 5], seed=13)                     # perturb uncovered parcel
    z1 = enc._parcel_latent(m2p, lv, use_ckpt=False)
    delta = (z1 - z0).abs()
    assert delta[0, lv[0]].max() < 1e-6                         # every covered parcel unchanged


def test_parcel_latent_all_uncovered_is_finite() -> None:
    """All parcels masked → fully-masked query rows → zero SA output (bypass),
    no NaN; FFN still applies (don't-care rows)."""
    enc = _make().eval()
    m2, K = _rand_m2(enc, B=2, T_p=3, seed=3)
    lv = torch.zeros(2, K, dtype=torch.bool)
    z = enc._parcel_latent(m2, lv, use_ckpt=False)
    assert z.shape == m2.shape
    assert torch.isfinite(z).all()


def test_parcel_latent_checkpoint_matches_and_backprops() -> None:
    """The ``use_ckpt=True`` branch (gradient-checkpointed parcel blocks, with
    the non-differentiable bool ``attn_mask`` passed positionally) is bit-equal
    to the non-checkpointed path and yields finite, non-zero grads — including
    when a batch element is fully uncovered (softmax over finite NEG_INF)."""
    enc = _make().train()
    m2, K = _rand_m2(enc, B=2, T_p=3, seed=4)
    lv = torch.ones(2, K, dtype=torch.bool)
    lv[1] = False                                              # one all-uncovered element
    base = m2.clone().requires_grad_(True)
    ckpt = m2.clone().requires_grad_(True)
    z_base = enc._parcel_latent(base, lv, use_ckpt=False)
    z_ckpt = enc._parcel_latent(ckpt, lv, use_ckpt=True)
    torch.testing.assert_close(z_ckpt, z_base)                 # recompute is exact
    z_base.sum().backward()
    z_base_grad = base.grad.clone()
    # Reset before the second backward so the block-param-grad check below
    # reflects ONLY the checkpointed path (no cross-path accumulation).
    enc.zero_grad(set_to_none=True)
    z_ckpt.sum().backward()
    assert torch.isfinite(z_base_grad).all() and torch.isfinite(ckpt.grad).all()
    torch.testing.assert_close(ckpt.grad, z_base_grad)         # same input gradient
    # The checkpointed path actually trained the block params (finite grads).
    for p in enc.latent_parcel_blocks.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


# --------------------------------------------------------------------------- #
# #112 ragged parcel latent == dense on VISIBLE parcels (memory/FLOP opt)
# --------------------------------------------------------------------------- #
# Gated on ``ragged_parcel`` (default OFF). The parcel-SA already key-masks
# uncovered/masked parcels out with an exact-0 softmax weight, so gathering the
# visible parcels to a pad-to-batch-max Kk and scattering back is a pure FLOP
# cut: every VISIBLE parcel's z is bit-identical to dense up to float32
# reduction re-tiling (~1e-6, the #91 standard), and dropped parcels go to 0.
def _parcel_latent_dense_vs_ragged(enc, m2, lv, *, use_ckpt: bool = False):
    enc.ragged_parcel = False
    zd = enc._parcel_latent(m2, lv, use_ckpt=use_ckpt)
    enc.ragged_parcel = True
    zr = enc._parcel_latent(m2, lv, use_ckpt=use_ckpt)
    enc.ragged_parcel = False
    return zd, zr


def test_ragged_parcel_default_off_on_mean_model() -> None:
    assert _make().ragged_parcel is False


def test_parcel_latent_ragged_all_covered_bit_identical() -> None:
    """All parcels visible ⇒ identity gather/scatter ⇒ BIT-identical to dense
    (the gather index is the identity permutation, no reduction-size change)."""
    enc = _make().eval()
    m2, K = _rand_m2(enc, B=2, T_p=5, seed=0)
    lv = torch.ones(2, K, dtype=torch.bool)
    zd, zr = _parcel_latent_dense_vs_ragged(enc, m2, lv)
    assert torch.equal(zr, zd)


def test_parcel_latent_ragged_mixed_coverage_at_visible() -> None:
    """Mixed (per-clip-varying) coverage ⇒ visible-parcel z matches dense up to
    float32 reduction re-tiling (~1e-6); dropped parcels go to exactly 0."""
    enc = _make().eval()
    m2, K = _rand_m2(enc, B=3, T_p=4, seed=1)
    lv = torch.ones(3, K, dtype=torch.bool)
    lv[0, 3:] = False                     # clip 0: 3 visible
    lv[1, 5] = False                      # clip 1: 5 visible
    lv[2, 2:] = False                     # clip 2: 2 visible (Kk = per-batch max = 5)
    zd, zr = _parcel_latent_dense_vs_ragged(enc, m2, lv)
    torch.testing.assert_close(zr[lv], zd[lv], atol=1e-5, rtol=1e-5)   # visible parity
    drop = ~lv
    assert drop.any()
    assert (zr[drop] == 0).all()          # dropped → exactly 0 (scratch scatter)
    # dense leaves FFN garbage at dropped parcels (don't-care) ⇒ they differ.
    assert not torch.equal(zr[drop], zd[drop])


def test_parcel_latent_ragged_all_uncovered_is_zero_and_finite() -> None:
    """All parcels masked ⇒ Kk clamps to 1, every row is a pad ⇒ scratch stays
    zero everywhere; finite, correctly-shaped, all-zero (don't-care rows)."""
    enc = _make().eval()
    m2, K = _rand_m2(enc, B=2, T_p=3, seed=3)
    lv = torch.zeros(2, K, dtype=torch.bool)
    enc.ragged_parcel = True
    z = enc._parcel_latent(m2, lv, use_ckpt=False)
    assert z.shape == m2.shape
    assert torch.isfinite(z).all()
    assert (z == 0).all()


def test_parcel_latent_ragged_checkpoint_matches() -> None:
    """The checkpointed ragged path is bit-equal to the non-checkpointed ragged
    path and yields finite, matching input gradients (incl. a fully-uncovered
    batch element exercising the Kk-clamp)."""
    enc = _make().train()
    enc.ragged_parcel = True
    m2, K = _rand_m2(enc, B=2, T_p=3, seed=4)
    lv = torch.ones(2, K, dtype=torch.bool)
    lv[1] = False                          # one all-uncovered element
    base = m2.clone().requires_grad_(True)
    ckpt = m2.clone().requires_grad_(True)
    zb = enc._parcel_latent(base, lv, use_ckpt=False)
    zc = enc._parcel_latent(ckpt, lv, use_ckpt=True)
    torch.testing.assert_close(zc, zb)
    zb.sum().backward()
    gb = base.grad.clone()
    enc.zero_grad(set_to_none=True)
    zc.sum().backward()
    assert torch.isfinite(gb).all() and torch.isfinite(ckpt.grad).all()
    torch.testing.assert_close(ckpt.grad, gb)


def test_parcel_latent_ragged_gradients_match_dense() -> None:
    """Param + (visible) input grads through the ragged latent match the dense
    latent — the visible parcels drive an identical learning signal."""
    enc = _make().train()
    m2, K = _rand_m2(enc, B=2, T_p=4, seed=8)
    lv = torch.ones(2, K, dtype=torch.bool)
    lv[0, 3:] = False
    lv[1, 4:] = False

    def run(ragged: bool):
        enc.ragged_parcel = ragged
        x = m2.clone().requires_grad_(True)
        z = enc._parcel_latent(x, lv, use_ckpt=False)
        enc.zero_grad(set_to_none=True)
        # Sum VISIBLE parcels only — dropped parcels differ (dense FFN garbage
        # vs ragged 0), so including them would inject a spurious dense-only grad.
        z[lv].sum().backward()
        pg = {n: p.grad.detach().clone()
              for n, p in enc.named_parameters() if p.grad is not None}
        return x.grad.detach().clone(), pg

    gxd, pgd = run(False)
    gxr, pgr = run(True)
    enc.ragged_parcel = False
    torch.testing.assert_close(gxr[lv], gxd[lv], atol=1e-5, rtol=1e-4)
    keys = set(pgd) & set(pgr)
    assert keys, "no grad-bearing latent params"
    for k in keys:
        torch.testing.assert_close(pgr[k], pgd[k], atol=1e-5, rtol=1e-4,
                                   msg=f"grad mismatch @ {k}")


def test_meanpool_ragged_parcel_equals_dense_on_covered() -> None:
    """Full forward: ``ragged_parcel`` True vs False ⇒ M2 untouched (pre-latent)
    and M4 bit-identical at covered parcels (mixed coverage, real drop)."""
    kw = _mp_kw()
    B, C = 2, 4
    g = torch.Generator().manual_seed(3)
    et = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"], generator=g)
    support = torch.zeros(B, C, kw["k_parcels"])
    for c in range(C):
        support[:, c, c] = 1.0                 # parcels 0..3 covered, 4-5 empty
    valid = torch.ones(B, C, dtype=torch.bool)
    dense = _make(ragged_parcel=False).eval()
    ragged = _make(ragged_parcel=True).eval()
    ragged.load_state_dict(dense.state_dict())
    od = dense(et, support, valid_mask=valid, return_taps=True)
    orr = ragged(et, support, valid_mask=valid, return_taps=True)
    cov = od["latent_valid"]
    assert not cov.all()                       # the ragged drop is exercised
    assert torch.equal(od["M2"], orr["M2"])    # the latent drop never touches M2
    torch.testing.assert_close(od["M4"][cov], orr["M4"][cov], rtol=1e-5, atol=1e-5)


def test_meanpool_ragged_parcel_tube_visible_parity() -> None:
    """Full forward under a tube ``parcel_time_mask``: the tubed parcel is the
    one the ragged latent drops, and every VISIBLE parcel's M4 matches dense."""
    kw = _mp_kw()
    B, C, K = 2, 8, kw["k_parcels"]
    et, support, valid = _inputs(B, C, kw, seed=6)
    dense = _make(ragged_parcel=False).eval()
    ragged = _make(ragged_parcel=True).eval()
    ragged.load_state_dict(dense.state_dict())
    T_p = kw_T(dense, kw)
    ptm = torch.zeros(B, K, T_p, dtype=torch.bool)
    ptm[:, 0] = True                           # tube parcel 0
    od = dense(et, support, valid_mask=valid, return_taps=True, parcel_time_mask=ptm)
    orr = ragged(et, support, valid_mask=valid, return_taps=True, parcel_time_mask=ptm)
    visible = od["latent_valid"].clone()
    visible[:, 0] = False                      # parcel 0 tubed ⇒ not visible
    assert visible.any()
    torch.testing.assert_close(od["M4"][visible], orr["M4"][visible],
                               atol=1e-5, rtol=1e-5)


def test_meanpool_ragged_parcel_fully_uncovered_clip_mixed() -> None:
    """End-to-end: a fully-uncovered clip (all-pad) batched with a covered clip
    exercises the Kk-clamp + pad scatter on the real forward (not just the
    _parcel_latent unit) — the covered clip's M4 still matches dense, the
    uncovered clip's don't-care rows stay finite."""
    kw = _mp_kw()
    B, C = 2, 4
    g = torch.Generator().manual_seed(5)
    et = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"], generator=g)
    support = torch.zeros(B, C, kw["k_parcels"])
    for c in range(C):
        support[:, c, c] = 1.0                 # parcels 0..3 covered
    valid = torch.ones(B, C, dtype=torch.bool)
    valid[1] = False                           # clip 1: every electrode pad → uncovered
    dense = _make(ragged_parcel=False).eval()
    ragged = _make(ragged_parcel=True).eval()
    ragged.load_state_dict(dense.state_dict())
    od = dense(et, support, valid_mask=valid, return_taps=True)
    orr = ragged(et, support, valid_mask=valid, return_taps=True)
    cov = od["latent_valid"]
    assert cov[0, :4].all() and not cov[1].any()   # clip0 covered, clip1 all-uncovered
    torch.testing.assert_close(od["M4"][cov], orr["M4"][cov], atol=1e-5, rtol=1e-5)
    assert torch.isfinite(orr["M4"]).all()         # uncovered clip don't-care rows finite


def test_meanpool_m_sub_slots_does_not_leak_into_parcel_latent() -> None:
    """The mean path's parcel axis is ``k_parcels`` (K), NOT the cross-attn
    ``L = K·m_sub_slots`` slot expansion — so m_sub_slots must not change the M4
    axis or the ragged gather. Build a mean model with m_sub_slots=4 (the
    R-m4-slots sister value) and confirm M4 is (B, K, F_p, T_p, d) and the
    ragged latent still matches dense on covered parcels."""
    kw = _mp_kw()
    B, C = 2, 4
    g = torch.Generator().manual_seed(9)
    et = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"], generator=g)
    support = torch.zeros(B, C, kw["k_parcels"])
    for c in range(C):
        support[:, c, c] = 1.0                 # parcels 0..3 covered, 4-5 empty
    valid = torch.ones(B, C, dtype=torch.bool)
    dense = _make(m_sub_slots=4, ragged_parcel=False).eval()
    ragged = _make(m_sub_slots=4, ragged_parcel=True).eval()
    ragged.load_state_dict(dense.state_dict())
    od = dense(et, support, valid_mask=valid, return_taps=True)
    orr = ragged(et, support, valid_mask=valid, return_taps=True)
    K = kw["k_parcels"]
    assert od["M4"].shape[1] == K                  # K = k_parcels, NOT K·4
    assert od["latent_valid"].shape == (B, K)
    cov = od["latent_valid"]
    assert not cov.all()                           # ragged drop exercised
    torch.testing.assert_close(od["M4"][cov], orr["M4"][cov], atol=1e-5, rtol=1e-5)


def test_meanpool_default_return_is_m4_tensor() -> None:
    enc = _make().eval()
    kw = _mp_kw()
    et, support, valid = _inputs(2, 8, kw)
    out = enc(et, support, valid_mask=valid)
    assert torch.is_tensor(out)
    K, F_p, d = kw["k_parcels"], enc.n_freq_patches, kw["d_model"]
    T_p = enc.patch_stem.n_time_patches(kw["n_time_bins"])
    assert out.shape == (2, K, F_p, T_p, d)


def test_meanpool_m2_only_returns_only_m2() -> None:
    enc = _make().eval()
    kw = _mp_kw()
    et, support, valid = _inputs(2, 8, kw)
    out = enc(et, support, valid_mask=valid, return_taps=True, m2_only=True)
    assert set(out) == {"M2", "latent_valid"}
    full = enc(et, support, valid_mask=valid, return_taps=True)
    torch.testing.assert_close(out["M2"], full["M2"])       # byte-identical M2


def test_meanpool_m2_only_requires_return_taps() -> None:
    enc = _make().eval()
    kw = _mp_kw()
    et, support, valid = _inputs(2, 8, kw)
    with pytest.raises(ValueError, match="return_taps"):
        enc(et, support, valid_mask=valid, m2_only=True)


# --------------------------------------------------------------------------- #
# D4 ragged stem == dense on covered parcels
# --------------------------------------------------------------------------- #
def test_meanpool_ragged_equals_dense_on_covered_parcels() -> None:
    # Leave parcels 4,5 uncovered so the ragged path actually drops rows.
    kw = _mp_kw()
    B, C = 2, 4
    g = torch.Generator().manual_seed(3)
    et = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"], generator=g)
    support = torch.zeros(B, C, kw["k_parcels"])
    for c in range(C):
        support[:, c, c] = 1.0                              # parcels 0..3 covered, 4-5 empty
    valid = torch.ones(B, C, dtype=torch.bool)

    dense = _make(ragged_frontend=False).eval()
    ragged = _make(ragged_frontend=True).eval()
    ragged.load_state_dict(dense.state_dict())              # same weights

    od = dense(et, support, valid_mask=valid, return_taps=True)
    orr = ragged(et, support, valid_mask=valid, return_taps=True)
    cov = od["latent_valid"]
    assert not cov.all()                                    # the ragged path is exercised
    torch.testing.assert_close(od["M2"][cov], orr["M2"][cov], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(od["M4"][cov], orr["M4"][cov], rtol=1e-5, atol=1e-5)


def test_meanpool_ragged_all_covered_matches_dense_everywhere() -> None:
    kw = _mp_kw()
    B, C = 2, 6
    et, support, valid = _inputs(B, C, kw)                   # electrodes hit all 6 parcels
    dense = _make(ragged_frontend=False).eval()
    ragged = _make(ragged_frontend=True).eval()
    ragged.load_state_dict(dense.state_dict())
    od = dense(et, support, valid_mask=valid, return_taps=True)
    orr = ragged(et, support, valid_mask=valid, return_taps=True)
    assert od["latent_valid"].all()
    torch.testing.assert_close(od["M2"], orr["M2"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(od["M4"], orr["M4"], rtol=1e-5, atol=1e-5)


def test_meanpool_ragged_per_batch_varying_coverage() -> None:
    """The realistic multi-subject case: coverage DIFFERS across batch
    elements, so the B·K-flattened ragged gather/scatter must index covered
    parcels per-(b,k) — the index arithmetic most prone to a silent bug."""
    kw = _mp_kw()
    B, C, K = 2, 5, kw["k_parcels"]
    g = torch.Generator().manual_seed(7)
    et = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"], generator=g)
    support = torch.zeros(B, C, K)
    # b=0 covers parcels {0,1}; b=1 covers DISJOINT {2,3,4}. Unassigned
    # electrode rows (all-zero support) contribute to no parcel.
    support[0, 0, 0] = support[0, 1, 1] = 1.0
    support[1, 0, 2] = support[1, 1, 3] = support[1, 2, 4] = 1.0
    valid = torch.ones(B, C, dtype=torch.bool)

    dense = _make(ragged_frontend=False).eval()
    ragged = _make(ragged_frontend=True).eval()
    ragged.load_state_dict(dense.state_dict())
    od = dense(et, support, valid_mask=valid, return_taps=True)
    orr = ragged(et, support, valid_mask=valid, return_taps=True)
    cov = od["latent_valid"]
    # coverage genuinely varies across the batch.
    assert cov[0, :2].all() and not cov[0, 2:].any()
    assert cov[1, 2:5].all() and not cov[1, :2].any() and not cov[1, 5:].any()
    torch.testing.assert_close(od["M2"][cov], orr["M2"][cov], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(od["M4"][cov], orr["M4"][cov], rtol=1e-5, atol=1e-5)


def test_meanpool_ragged_whole_batch_empty() -> None:
    """All parcels uncovered → the ragged ``valid_idx.numel()==0`` branch.
    Must not crash; returns finite, correctly-shaped, all-uncovered taps."""
    kw = _mp_kw()
    B, C = 2, 6
    et, support, _ = _inputs(B, C, kw)
    valid = torch.zeros(B, C, dtype=torch.bool)             # every electrode is a pad
    enc = _make(ragged_frontend=True).eval()
    out = enc(et, support, valid_mask=valid, return_taps=True)
    assert not out["latent_valid"].any()
    K, F_p, d = kw["k_parcels"], enc.n_freq_patches, kw["d_model"]
    T_p = enc.patch_stem.n_time_patches(kw["n_time_bins"])
    assert out["M2"].shape == (B, K, F_p, T_p, d)
    assert torch.isfinite(out["M2"]).all() and torch.isfinite(out["M4"]).all()


# --------------------------------------------------------------------------- #
# input validation (hard mean-pool: a support↔valid_mask desync is corrupting)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_shape", [(8,), (2, 1), (2, 7), (1, 8)])
def test_meanpool_valid_mask_wrong_shape_raises(bad_shape: tuple) -> None:
    enc = _make().eval()
    kw = _mp_kw()
    et, support, _ = _inputs(2, 8, kw)
    bad = torch.ones(*bad_shape, dtype=torch.bool)
    with pytest.raises(ValueError, match="valid_mask"):
        enc(et, support, valid_mask=bad, return_taps=True)


# --------------------------------------------------------------------------- #
# pool flag plumbing + sister preservation
# --------------------------------------------------------------------------- #
def test_default_pool_is_cross_attn_sister_unchanged() -> None:
    """The B36 learned cross-attention pool stays the default; its return
    contract ((B, L, T_p, d)) is untouched by the B37 mean-pool addition."""
    enc = V14ParcelPerceiverModel(
        n_freq_bins=6, n_time_bins=4, k_parcels=6, d_model=32, n_heads=4,
        depth_self_attn=2, m_sub_slots=1, patch_kernel_freq=3,
    )
    assert enc.pool == "cross_attn"
    et, support, valid = _inputs(2, 8, {"n_time_bins": 4, "n_freq_bins": 6, "k_parcels": 6})
    out = enc(et, support, valid_mask=valid)
    assert out.ndim == 4                                    # (B, L, T_p, d)
    assert out.shape[0] == 2 and out.shape[1] == 6 * 1


def test_invalid_pool_raises() -> None:
    with pytest.raises(ValueError, match="pool"):
        V14ParcelPerceiverModel(
            n_freq_bins=6, n_time_bins=4, k_parcels=6, d_model=32, n_heads=4,
            depth_self_attn=2, m_sub_slots=1, patch_kernel_freq=3,
            pool="soft",  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# dropped conditioning args fail loud on the mean path (B37 drops these)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arg", ["return_m3", "shaft_mask", "subject_subtype", "ref_idx", "freq_patch_valid"])
def test_meanpool_dropped_args_raise(arg: str) -> None:
    enc = _make().eval()
    kw = _mp_kw()
    B, C = 2, 8
    et, support, valid = _inputs(B, C, kw)
    call: dict = {"return_taps": True}
    if arg == "return_m3":
        call["return_m3"] = True
    elif arg == "shaft_mask":
        call["shaft_mask"] = torch.zeros(B, C, dtype=torch.bool)
    elif arg == "subject_subtype":
        call["subject_subtype"] = torch.zeros(B, dtype=torch.long)
    elif arg == "ref_idx":
        call["ref_idx"] = torch.zeros(B, dtype=torch.long)
    elif arg == "freq_patch_valid":
        call["freq_patch_valid"] = torch.ones(enc.n_freq_patches, dtype=torch.bool)
    with pytest.raises(NotImplementedError):
        enc(et, support, valid_mask=valid, **call)


# --------------------------------------------------------------------------- #
# D7 composite mask: visible-only student (token_mask band / parcel_time tube)
# --------------------------------------------------------------------------- #
def _mask_inputs(enc, kw, *, B=2, C=8, seed=1):
    et, support, valid = _inputs(B, C, kw, seed=seed)
    T_p = enc.patch_stem.n_time_patches(kw["n_time_bins"])
    F_p, K = enc.n_freq_patches, kw["k_parcels"]
    return et, support, valid, F_p, T_p, K


def test_meanpool_masks_thread_and_finite() -> None:
    """Both composite masks thread through the forward → finite, freq-preserving
    taps; the band-only and tube-only sub-cases also run."""
    enc = _make().eval()
    kw = _mp_kw()
    et, support, valid, F_p, T_p, K = _mask_inputs(enc, kw)
    g = torch.Generator().manual_seed(2)
    token_mask = torch.rand(2, K, F_p, T_p, generator=g) < 0.5
    ptm = torch.zeros(2, K, T_p, dtype=torch.bool)
    ptm[:, 0] = True                                            # tube parcel 0
    for tm, pm in [(token_mask, ptm), (token_mask, None), (None, ptm)]:
        out = enc(et, support, valid_mask=valid, return_taps=True,
                  token_mask=tm, parcel_time_mask=pm)
        d = kw["d_model"]
        assert out["M2"].shape == (2, K, F_p, T_p, d)
        assert out["M4"].shape == (2, K, F_p, T_p, d)
        assert torch.isfinite(out["M2"]).all() and torch.isfinite(out["M4"]).all()


def test_meanpool_no_masks_byte_identical() -> None:
    """token_mask=None and parcel_time_mask=None → byte-identical to the plain
    unmasked forward (the backward-compat / no-op contract)."""
    enc = _make().eval()
    kw = _mp_kw()
    et, support, valid, _, _, _ = _mask_inputs(enc, kw)
    base = enc(et, support, valid_mask=valid, return_taps=True)
    nun = enc(et, support, valid_mask=valid, return_taps=True,
              token_mask=None, parcel_time_mask=None)
    torch.testing.assert_close(base["M2"], nun["M2"], atol=0, rtol=0)
    torch.testing.assert_close(base["M4"], nun["M4"], atol=0, rtol=0)


def test_meanpool_tube_zeros_parcel_m2_upstream() -> None:
    """A tube on parcel j zeroes ALL of parcel j's tokens BEFORE the token
    blocks → parcel j's M2 is independent of its own electrode input (the
    upstream-zeroing visible-only contract)."""
    kw = _mp_kw()
    enc = _make(ragged_frontend=False).eval()
    B, C, K = 2, 8, kw["k_parcels"]
    # electrode 0 → parcel 0 (the tubed one); perturbing it must not move M2[0].
    et, support, valid = _inputs(B, C, kw, seed=5)
    ptm = torch.zeros(B, K, kw_T(enc, kw), dtype=torch.bool)
    ptm[:, 0] = True
    out0 = enc(et, support, valid_mask=valid, return_taps=True, parcel_time_mask=ptm)
    et2 = et.clone()
    et2[:, support[0, :, 0] > 0] += 7.0          # bump every electrode in parcel 0
    out1 = enc(et2, support, valid_mask=valid, return_taps=True, parcel_time_mask=ptm)
    torch.testing.assert_close(out0["M2"][:, 0], out1["M2"][:, 0], atol=1e-6, rtol=1e-5)


def test_meanpool_tube_excludes_parcel_from_latent_context() -> None:
    """A tube on parcel j drops it from the parcel-SA context → every surviving
    parcel's M4 is independent of parcel j's electrode input (visible-only M4)."""
    kw = _mp_kw()
    enc = _make(ragged_frontend=False).eval()
    B, C, K = 2, 8, kw["k_parcels"]
    et, support, valid = _inputs(B, C, kw, seed=6)
    ptm = torch.zeros(B, K, kw_T(enc, kw), dtype=torch.bool)
    ptm[:, 0] = True                              # tube parcel 0
    out0 = enc(et, support, valid_mask=valid, return_taps=True, parcel_time_mask=ptm)
    et2 = et.clone()
    et2[:, support[0, :, 0] > 0] += 9.0           # perturb tubed parcel's electrodes
    out1 = enc(et, support, valid_mask=valid, return_taps=True, parcel_time_mask=ptm)
    out1b = enc(et2, support, valid_mask=valid, return_taps=True, parcel_time_mask=ptm)
    # surviving parcels = all covered except the tubed parcel 0.
    surviving = out0["latent_valid"].clone()
    surviving[:, 0] = False
    torch.testing.assert_close(out0["M4"][surviving], out1b["M4"][surviving],
                               atol=1e-6, rtol=1e-5)
    del out1


def test_meanpool_band_mask_is_local_to_its_parcel_m2() -> None:
    """M2 is pre-parcel-SA (per-parcel), so a band token_mask on parcel j moves
    parcel j's M2 but leaves every other parcel's M2 byte-identical."""
    kw = _mp_kw()
    enc = _make(ragged_frontend=False).eval()
    B, C, K = 2, 8, kw["k_parcels"]
    et, support, valid = _inputs(B, C, kw, seed=7)
    F_p, T_p = enc.n_freq_patches, kw_T(enc, kw)
    base = enc(et, support, valid_mask=valid, return_taps=True)
    tm = torch.zeros(B, K, F_p, T_p, dtype=torch.bool)
    tm[:, 1, : F_p // 2, :] = True                # band-mask only parcel 1
    out = enc(et, support, valid_mask=valid, return_taps=True, token_mask=tm)
    assert (out["M2"][:, 1] - base["M2"][:, 1]).abs().max() > 1e-5   # parcel 1 moved
    others = [k for k in range(K) if k != 1]
    torch.testing.assert_close(out["M2"][:, others], base["M2"][:, others],
                               atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("bad", [(2, 8, 99, 99), (2, 5, 99, 99)])
def test_meanpool_token_mask_wrong_shape_raises(bad: tuple) -> None:
    """token_mask must be PARCEL-shaped (B, K, F_p, T_p) — an electrode-shaped
    (B, C, ...) or wrong-K mask is rejected loudly (desync guard)."""
    kw = _mp_kw()
    enc = _make().eval()
    et, support, valid = _inputs(2, 8, kw)
    F_p, T_p = enc.n_freq_patches, kw_T(enc, kw)
    tm = torch.zeros(bad[0], bad[1], F_p, T_p, dtype=torch.bool)
    with pytest.raises(ValueError, match="token_mask"):
        enc(et, support, valid_mask=valid, return_taps=True, token_mask=tm)


def test_meanpool_parcel_time_mask_wrong_shape_raises() -> None:
    kw = _mp_kw()
    enc = _make().eval()
    et, support, valid = _inputs(2, 8, kw)
    bad = torch.zeros(2, kw["k_parcels"] + 1, kw_T(enc, kw), dtype=torch.bool)
    with pytest.raises(ValueError, match="parcel_time_mask"):
        enc(et, support, valid_mask=valid, return_taps=True, parcel_time_mask=bad)


def kw_T(enc, kw) -> int:
    return enc.patch_stem.n_time_patches(kw["n_time_bins"])
