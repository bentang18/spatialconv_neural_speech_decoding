"""B37 Chunk C — freq-preserving Phase-4 readout (D6).

The mean-pool latent keeps frequency (M4 = (B, K, F_p, T_p, d)), so the P4
readout collapses the PARCEL axis ONLY with the frozen P3-PMA — frequency ×
time survive — then mean-over-time, flatten the freq axis, and a per-task
``Linear(F_p·d, n_classes)`` (the only trainable P4 param). See
reports/b37_meanpool_freq_latent_spec_2026_06_10.md §2.1.

Pinned here:
  * output shape (B, n_classes); the classifier is Linear(F_p·d, n_classes);
    binary / 10-way param counts.
  * the PMA is FROZEN, the classifier is the only trainable thing.
  * the PMA collapses PARCEL only — freq × time both survive the collapse
    (intermediate is (B, F_p, T_p, d)); frequency reaches the classifier.
  * uncovered parcels are excluded (latent_valid key-mask) → the readout is
    independent of their (don't-care) M4 content; covered parcels are used.
  * F_p-mismatch fails loud.
  * V14ParcelPerceiverWithHead routes the mean path end-to-end (M4 + the
    pool's own latent_valid), and rejects dropped shaft/subtype/ref args.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from speech_decoding.models.v14_encoder import (
    V14FreqPreservingPmaReadout,
    V14MeanPoolLinearHead,
    V14ParcelCollapsePMA,
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
    V14ParcelPerceiverWithHead,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _enc_kw() -> dict:
    # patch_kernel_freq=2 avoids the FE-RAW-1 F=50 guard; small + even d_model.
    return {
        "n_freq_bins": 6,
        "n_time_bins": 4,
        "k_parcels": 6,
        "d_model": 16,
        "n_heads": 4,
        "depth_self_attn": 2,
        "m_sub_slots": 1,
        "n_token_blocks": 2,
        "patch_kernel_freq": 2,
        "pool": "mean",
    }


def _mean_encoder(**over) -> V14ParcelPerceiverModel:
    kw = _enc_kw()
    kw.update(over)
    torch.manual_seed(0)
    return V14ParcelPerceiverModel(**kw)


def _readout(enc: V14ParcelPerceiverModel, *, n_classes: int = 2):
    torch.manual_seed(1)
    pma = V14ParcelCollapsePMA(d_model=enc.d_model, n_heads=_enc_kw()["n_heads"])
    return V14FreqPreservingPmaReadout(
        pma=pma,
        d_model=enc.d_model,
        n_freq_patches=enc.n_freq_patches,
        n_classes=n_classes,
    )


def _rand_m4(enc: V14ParcelPerceiverModel, *, B: int, T_p: int, seed: int):
    F_p, K, d = enc.n_freq_patches, enc.k_parcels, enc.d_model
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, K, F_p, T_p, d, generator=g), K


def _inputs(B: int, C: int, kw: dict, *, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    et = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"], generator=g)
    support = torch.zeros(B, C, kw["k_parcels"])
    for c in range(C):
        support[:, c, c % kw["k_parcels"]] = 1.0
    valid = torch.ones(B, C, dtype=torch.bool)
    return et, support, valid


# --------------------------------------------------------------------------- #
# shape + param structure (D6)
# --------------------------------------------------------------------------- #
def test_freq_readout_output_shape() -> None:
    enc = _mean_encoder()
    r = _readout(enc, n_classes=3).eval()
    m4, K = _rand_m4(enc, B=2, T_p=5, seed=0)
    lv = torch.ones(2, K, dtype=torch.bool)
    out = r(m4, latent_valid=lv)
    assert out.shape == (2, 3)
    assert torch.isfinite(out).all()


def test_freq_readout_classifier_is_linear_fp_d() -> None:
    enc = _mean_encoder()
    r = _readout(enc, n_classes=2)
    F_p, d = enc.n_freq_patches, enc.d_model
    # D6: the freq axis flattens into the classifier → input is F_p·d, NOT d.
    assert isinstance(r.classifier, nn.Linear)
    assert r.classifier.in_features == F_p * d
    assert r.classifier.out_features == 2


def test_freq_readout_param_counts_binary_and_10way() -> None:
    # Full-scale F_p=10, d=256 (the production token grid): binary ≈5.1k,
    # 10-way ≈25.7k — the spec §2.1 figures (only the linear trains).
    pma = V14ParcelCollapsePMA(d_model=256, n_heads=8)
    r_bin = V14FreqPreservingPmaReadout(
        pma=pma, d_model=256, n_freq_patches=10, n_classes=2,
    )
    r_10 = V14FreqPreservingPmaReadout(
        pma=pma, d_model=256, n_freq_patches=10, n_classes=10,
    )
    bin_trainable = sum(p.numel() for p in r_bin.classifier.parameters())
    ten_trainable = sum(p.numel() for p in r_10.classifier.parameters())
    assert bin_trainable == 2560 * 2 + 2            # 5122
    assert ten_trainable == 2560 * 10 + 10          # 25_610


def test_freq_readout_pma_frozen_classifier_trainable() -> None:
    enc = _mean_encoder()
    r = _readout(enc)
    assert all(not p.requires_grad for p in r.pma.parameters())   # frozen P3-PMA
    assert all(p.requires_grad for p in r.classifier.parameters())  # only linear


# --------------------------------------------------------------------------- #
# D6 — PMA collapses PARCEL only; frequency × time survive
# --------------------------------------------------------------------------- #
def test_freq_readout_collapse_keeps_freq_and_time() -> None:
    """The frozen PMA collapses K only: the intermediate (pre mean-time) is
    (B, F_p, T_p, d) — both frequency AND time survive the parcel collapse."""
    enc = _mean_encoder()
    r = _readout(enc).eval()
    m4, K = _rand_m4(enc, B=2, T_p=5, seed=4)
    B, _, F_p, T_p, d = m4.shape
    lv = torch.ones(B, K, dtype=torch.bool)
    s = m4.reshape(B, K, F_p * T_p, d)
    collapsed = r.pma(s, latent_valid=lv).reshape(B, F_p, T_p, d)
    assert collapsed.shape == (B, F_p, T_p, d)      # freq + time both kept


def test_freq_readout_frequency_reaches_classifier() -> None:
    """Perturbing M4 at one freq patch (covered parcels) changes the logits —
    frequency is a resolved axis carried through to the readout (D3/D6)."""
    enc = _mean_encoder()
    r = _readout(enc).eval()
    m4, K = _rand_m4(enc, B=1, T_p=4, seed=5)
    lv = torch.ones(1, K, dtype=torch.bool)
    base = r(m4, latent_valid=lv)
    g = torch.Generator().manual_seed(99)
    m4p = m4.clone()
    m4p[:, :, 1, :, :] += torch.randn(m4p[:, :, 1, :, :].shape, generator=g)
    moved = r(m4p, latent_valid=lv)
    assert (moved - base).abs().max() > 1e-5


def test_freq_readout_excludes_uncovered_parcels() -> None:
    """Uncovered parcels are key-masked in the PMA, so the readout is
    independent of their (don't-care) M4 content."""
    enc = _mean_encoder()
    r = _readout(enc).eval()
    m4, K = _rand_m4(enc, B=1, T_p=4, seed=6)
    lv = torch.ones(1, K, dtype=torch.bool)
    lv[0, 5] = False                                # parcel 5 uncovered
    base = r(m4, latent_valid=lv)
    g = torch.Generator().manual_seed(7)
    m4p = m4.clone()
    m4p[:, 5] += 100.0 * torch.randn(m4p[:, 5].shape, generator=g)
    moved = r(m4p, latent_valid=lv)
    torch.testing.assert_close(moved, base)         # uncovered parcel ignored


def test_freq_readout_uses_covered_parcels() -> None:
    enc = _mean_encoder()
    r = _readout(enc).eval()
    m4, K = _rand_m4(enc, B=1, T_p=4, seed=8)
    lv = torch.ones(1, K, dtype=torch.bool)
    base = r(m4, latent_valid=lv)
    g = torch.Generator().manual_seed(9)
    m4p = m4.clone()
    m4p[:, 2] += torch.randn(m4p[:, 2].shape, generator=g)   # covered parcel 2
    moved = r(m4p, latent_valid=lv)
    assert (moved - base).abs().max() > 1e-5


def test_freq_readout_fp_mismatch_raises() -> None:
    enc = _mean_encoder()
    r = _readout(enc)                                # built for enc.n_freq_patches
    bad = torch.randn(1, enc.k_parcels, enc.n_freq_patches + 1, 4, enc.d_model)
    lv = torch.ones(1, enc.k_parcels, dtype=torch.bool)
    with pytest.raises(ValueError, match="F_p"):
        r(bad, latent_valid=lv)


# --------------------------------------------------------------------------- #
# end-to-end through V14ParcelPerceiverWithHead (mean path)
# --------------------------------------------------------------------------- #
def test_withhead_mean_path_end_to_end() -> None:
    enc = _mean_encoder()
    r = _readout(enc, n_classes=2)
    head = V14ParcelPerceiverWithHead(enc, r).eval()
    et, support, valid = _inputs(2, 8, _enc_kw())
    out = head(et, support, valid_mask=valid)
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shaft_mask": True},
        {"subject_subtype": True},
        {"ref_idx": True},
    ],
)
def test_withhead_mean_path_rejects_dropped_conditioning(kwargs: dict) -> None:
    enc = _mean_encoder()
    head = V14ParcelPerceiverWithHead(enc, _readout(enc)).eval()
    et, support, valid = _inputs(2, 8, _enc_kw())
    B, C = et.shape[0], et.shape[1]
    payload = {}
    for k in kwargs:
        if k == "shaft_mask":
            payload[k] = torch.zeros(B, C, dtype=torch.bool)
        else:
            payload[k] = torch.zeros(B, C, dtype=torch.long)
    with pytest.raises(NotImplementedError, match="shaft/subtype/ref"):
        head(et, support, valid_mask=valid, **payload)


def test_withhead_cross_attn_path_unaffected() -> None:
    """The cross_attn WithHead path still routes through _compute_latent_valid
    and the (B, L, T, d) readout — Chunk C only added the mean branch."""
    torch.manual_seed(0)
    enc = V14ParcelPerceiverModel(
        n_freq_bins=6, n_time_bins=4, k_parcels=6, d_model=16, n_heads=4,
        depth_self_attn=1, m_sub_slots=1, n_token_blocks=1, patch_kernel_freq=2,
    )  # cross_attn default
    assert enc.pool == "cross_attn"
    # A trivial mean-over-time linear head over (B, L, T, d) for the smoke path.
    head = V14ParcelPerceiverWithHead(enc, V14MeanPoolLinearHead(16, 2)).eval()
    et, support, valid = _inputs(2, 6, dict(_enc_kw(), k_parcels=6))
    out = head(et, support, valid_mask=valid)
    assert out.shape == (2, 2)


# --------------------------------------------------------------------------- #
# config-layer V14ParcelPerceiver.build() — readout selection MUST be
# pool-aware (the Chunk-E CRITICAL fix). A pool='mean' encoder emits 5-D M4
# (B, K, F_p, T_p, d); the B35 4-D readouts would crash one batch into the
# forward (`too many values to unpack`). build() must hand a pool='mean'
# config the freq-preserving head — and refuse the non-wired readouts loud.
# --------------------------------------------------------------------------- #
def _build_cfg_kw() -> dict:
    return {
        "n_freq_bins": 6,
        "n_time_bins": 4,
        "k_parcels": 6,
        "d_model": 16,
        "n_heads": 4,
        "depth_self_attn": 2,
        "m_sub_slots": 1,
        "n_token_blocks": 2,
        "patch_kernel_freq": 2,
    }


def test_build_pool_mean_selects_freq_preserving_readout() -> None:
    cfg = V14ParcelPerceiver(pool="mean", readout="pma_mean_linear", **_build_cfg_kw())
    model = cfg.build(n_classes=3)
    assert isinstance(model, V14ParcelPerceiverWithHead)
    assert isinstance(model.readout, V14FreqPreservingPmaReadout)
    # the classifier flattens the freq axis: in_features == F_p · d (D6).
    assert model.readout.classifier.in_features == model.encoder.n_freq_patches * 16
    assert model.readout.classifier.out_features == 3


def test_build_pool_mean_forward_no_unpack_crash() -> None:
    """The auditor repro: a built pool='mean' probe must forward end-to-end —
    the 4-D B35 readouts crashed here with `too many values to unpack`."""
    cfg = V14ParcelPerceiver(pool="mean", readout="pma_mean_linear", **_build_cfg_kw())
    model = cfg.build(n_classes=2)
    model.eval()
    et, support, valid = _inputs(2, 8, _build_cfg_kw())
    out = model(et, support, valid_mask=valid)
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "readout",
    ["pma_flatten_linear", "pma_timeattn_linear", "attentive", "meanpool"],
)
def test_build_pool_mean_rejects_non_freq_readouts(readout: str) -> None:
    cfg = V14ParcelPerceiver(pool="mean", readout=readout, **_build_cfg_kw())
    with pytest.raises(ValueError, match="pool='mean'.*pma_mean_linear"):
        cfg.build(n_classes=2)


def test_build_cross_attn_default_unaffected_by_fix() -> None:
    """A cross_attn config still routes to the 4-D V14PmaReadout, untouched."""
    cfg = V14ParcelPerceiver(readout="pma_mean_linear", **_build_cfg_kw())
    assert cfg.pool == "cross_attn"
    model = cfg.build(n_classes=2)
    assert not isinstance(model.readout, V14FreqPreservingPmaReadout)
