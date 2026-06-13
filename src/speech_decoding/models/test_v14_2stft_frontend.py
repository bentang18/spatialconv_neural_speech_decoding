"""2STFT dual-band FRONTEND (per_band_stem) — §3b/§4 of the engineer handoff
(reports/fe_2stft_handoff_2026_06_12.md).

The dual-band frontend mean|std-pools two STFT bands at DIFFERENT latent rates
(low 8 Hz / high 16 Hz) into their parcels, stems each band separately, and
UNIONS them into one per-parcel FLAT token sequence ``(B, K, S, d)`` with
dual-rate real-time RoPE. It stops at the M2 tap (the latent stack that would
reconcile the two rates is §9 out of scope).

Pinned here (the danger-zone invariants Ben flagged):
  * acceptance §8.2 — F_p re-point: low → 7 freq-patches, high → 3, F_p = 10;
  * acceptance §8.3 — rates: low → 8 time-tokens / 1 s, high → 16 (high = 2·low);
  * the low-band trim that closes the dual-rate grid (centre-pad gives low 9 raw
    → 8; high robustly 16) and the EVEN-``t_high_p`` guard;
  * acceptance §8.5 — RoPE real-time re-key: a low token at time-patch t and a
    high token at time-patch 2t (same real time) share the SAME rotation;
  * the mean|std 2-channel stem (B37) per band;
  * the union token layout (low block then high block, time-OUTER/freq-INNER)
    and the freq_embed "re-point, do not re-size" split;
  * dispatch guards (high band required/rejected; M2-only; mask deferred to SSL);
  * staging partition includes the two band stems;
  * the single-grid path is untouched (covered by test_v14_meanpool /
    test_v14_encoder, re-asserted here for the cross-check).
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_encoder import JepaPredictor, V14ParcelPerceiverModel
from speech_decoding.ssl.mask import sample_m2_dual_band_mask
from speech_decoding.ssl.masked_jepa import m2_dual_band_loss


# --------------------------------------------------------------------------- #
# configs — geometry mirrors the spec @ a 1 s clip (low raw 9 → 8, high 16)
# --------------------------------------------------------------------------- #
# Band raw-frame counts at 1 s (centre-padded STFT, see view.py §3a tests):
#   low  nperseg=512 hop=256 → 9 frames; conv (fk2,tk1) → 7 freq / 9 time (→8 trim)
#   high nperseg=128 hop=64  → 33 frames; conv (fk3,tk2) → 3 freq / 16 time
_LOW_T_1S = 9
_HIGH_T_1S = 33


def _band_kw(**over) -> dict:
    kw = {
        # single-grid params (unused by the dual stem, but the ctor reads them)
        "n_freq_bins": 14,
        "n_time_bins": _LOW_T_1S,
        "k_parcels": 6,
        "d_model": 32,
        "n_heads": 4,
        "depth_self_attn": 2,
        "m_sub_slots": 1,
        "n_token_blocks": 2,
        "pool": "mean",
        # 2STFT dual band
        "per_band_stem": True,
        "band_low_n_freq_bins": 14,
        "band_high_n_freq_bins": 9,
        "band_low_kernel_freq": 2,
        "band_low_kernel_time": 1,
        "band_high_kernel_freq": 3,
        "band_high_kernel_time": 2,
        "band_low_n_time_bins": _LOW_T_1S,
        "band_high_n_time_bins": _HIGH_T_1S,
    }
    kw.update(over)
    return kw


def _make(**over) -> V14ParcelPerceiverModel:
    torch.manual_seed(0)
    return V14ParcelPerceiverModel(**_band_kw(**over))


def _band_inputs(B: int, C: int, K: int, *, t_low: int, t_high: int, seed: int = 1):
    """Two freq-major bands (B, C, F, T) + one-hot support + valid mask."""
    g = torch.Generator().manual_seed(seed)
    low = torch.randn(B, C, 14, t_low, generator=g)
    high = torch.randn(B, C, 9, t_high, generator=g)
    support = torch.zeros(B, C, K)
    for c in range(C):
        support[:, c, c % K] = 1.0
    valid = torch.ones(B, C, dtype=torch.bool)
    return low, high, support, valid


def _fwd(model, low, high, support, valid):
    return model(
        low, support, valid,
        electrode_tokens_high=high, return_taps=True, m2_only=True,
    )


# --------------------------------------------------------------------------- #
# geometry — F_p re-point + rates (acceptance §8.2 / §8.3)
# --------------------------------------------------------------------------- #
def test_band_stem_freq_patches_are_7_and_3_fp10() -> None:
    m = _make()
    assert m.F_p_low == 7
    assert m.F_p_high == 3
    assert m.n_freq_patches == 10  # F_p re-point, not re-size
    assert m.freq_embed.shape[0] == 10


def test_low_time_patch_count_is_half_high_after_trim() -> None:
    # high robustly 16 @ 1 s; low raw 9 trimmed to 8 (= high // 2).
    m = _make()
    assert m._dual_band_low_t_patches(16) == 8
    assert m._dual_band_low_t_patches(80) == 40  # 5 s


def test_odd_high_time_patch_count_is_rejected() -> None:
    m = _make()
    with pytest.raises(ValueError, match="even"):
        m._dual_band_low_t_patches(15)


# --------------------------------------------------------------------------- #
# forward shapes — flat per-parcel token sequence (B, K, S, d)
# --------------------------------------------------------------------------- #
def test_dual_band_m2_is_flat_token_sequence() -> None:
    m = _make()
    K, d = 6, 32
    low, high, sup, val = _band_inputs(2, 5, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    out = _fwd(m, low, high, sup, val)
    # S = low 7×8 + high 3×16 = 56 + 48 = 104.
    assert out["M2"].shape == (2, K, 104, d)
    assert out["latent_valid"].shape == (2, K)
    assert torch.isfinite(out["M2"]).all()


def test_dual_band_5s_grid_scales_2x() -> None:
    # 5 s: low raw 41 → 40, high 161 → 80. S = 7×40 + 3×80 = 280 + 240 = 520.
    m = _make(band_low_n_time_bins=41, band_high_n_time_bins=161)
    K = 6
    low, high, sup, val = _band_inputs(1, 4, K, t_low=41, t_high=161)
    out = _fwd(m, low, high, sup, val)
    assert out["M2"].shape == (1, K, 520, 32)


def test_dual_band_latent_valid_marks_uncovered_parcels() -> None:
    # K=6, only 4 electrodes → parcels 4,5 never covered → latent_valid False.
    m = _make()
    low, high, sup, val = _band_inputs(2, 4, 6, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    out = _fwd(m, low, high, sup, val)
    lv = out["latent_valid"]
    assert lv[:, :4].all()
    assert not lv[:, 4:].any()


# --------------------------------------------------------------------------- #
# token layout + RoPE real-time re-key (acceptance §8.5 — the danger zone)
# --------------------------------------------------------------------------- #
def test_token_layout_slots_and_basket_split() -> None:
    m = _make()
    lay = m.dual_band_token_layout(8, 16)
    slot, fpi, is_high = lay["slot"], lay["freq_patch_idx"], lay["is_high"]
    assert int(lay["S_low"]) == 56
    assert slot.shape == (104,)
    # low block: t-OUTER/f-INNER → token i has t=i//7, slot=2t, freq-patch i%7.
    for i in range(56):
        assert int(slot[i]) == 2 * (i // 7)
        assert int(fpi[i]) == i % 7
        assert not bool(is_high[i])
    # high block: token j (global 56+j) has t=j//3, slot=t, freq-patch 7 + j%3.
    for j in range(48):
        g = 56 + j
        assert int(slot[g]) == j // 3
        assert int(fpi[g]) == 7 + (j % 3)
        assert bool(is_high[g])


def test_low_and_high_token_at_same_real_time_share_rotation() -> None:
    # §8.5: low time-patch t → high grid slot 2t; a low token and a high token
    # at the same real time receive IDENTICAL RoPE rotation.
    m = _make()
    lay = m.dual_band_token_layout(8, 16)
    slot = lay["slot"]
    rope = m.key_rope[:, slot, :]                       # (2, S, head_dim)
    # low token with t_low=3 (freq-patch 0) → flat index 3*7 = 21, slot 6.
    low_tok = 3 * 7
    # high token with t_high=6 (freq-patch 0) → global 56 + 6*3 = 74, slot 6.
    high_tok = 56 + 6 * 3
    assert int(slot[low_tok]) == 6 and int(slot[high_tok]) == 6
    assert torch.equal(rope[:, low_tok, :], rope[:, high_tok, :])
    # and a DIFFERENT real time gives a different rotation.
    other = 56 + 7 * 3                                  # high t=7 → slot 7
    assert not torch.equal(rope[:, high_tok, :], rope[:, other, :])


def test_dual_band_rope_uses_common_grid_table_only() -> None:
    # The single-grid pre-tiled buffer must NOT exist in 2STFT mode (the
    # per-token gather replaces it); key_rope sizes the common 16-slot grid.
    m = _make()
    assert m.rope_joint_token is None
    assert m.key_rope.shape[1] == 16  # max_T_high_p @ the 1 s default geometry


# --------------------------------------------------------------------------- #
# mean|std 2-channel stem (B37)
# --------------------------------------------------------------------------- #
def test_band_stems_are_two_channel_under_mean_pool_std() -> None:
    m = _make(mean_pool_std=True)
    assert m.patch_stem_low.conv.in_channels == 2
    assert m.patch_stem_high.conv.in_channels == 2
    # std channel zero-init (B37): the second input channel contributes nothing
    # at init, so a std-on forward matches the mean-only stem at step 0.
    low, high, sup, val = _band_inputs(2, 5, 6, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    out = _fwd(m, low, high, sup, val)
    assert out["M2"].shape == (2, 6, 104, 32)
    assert torch.isfinite(out["M2"]).all()


def test_band_stems_are_one_channel_without_mean_pool_std() -> None:
    m = _make()  # mean_pool_std default OFF
    assert m.patch_stem_low.conv.in_channels == 1
    assert m.patch_stem_high.conv.in_channels == 1


def test_dual_band_is_deterministic() -> None:
    m = _make().eval()
    low, high, sup, val = _band_inputs(2, 5, 6, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    a = _fwd(m, low, high, sup, val)["M2"]
    b = _fwd(m, low, high, sup, val)["M2"]
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# staging partition + construction guards
# --------------------------------------------------------------------------- #
def test_partition_assigns_band_stems_to_frontend() -> None:
    m = _make()
    frontend, _parcel = m.partition_parameters_for_staging()
    names = {n for n, _ in m.named_parameters()}
    assert any(n.startswith("patch_stem_low") for n in names)
    assert any(n.startswith("patch_stem_high") for n in names)
    # band-stem params must all land in the FRONTEND bucket (none unassigned —
    # partition raises if any param top is in neither group).
    fe_ids = {id(p) for p in frontend}
    for n, p in m.named_parameters():
        if n.startswith("patch_stem_low") or n.startswith("patch_stem_high"):
            assert id(p) in fe_ids


def test_no_dead_single_grid_stem_in_dual_mode() -> None:
    m = _make()
    assert m.patch_stem is None  # no dead, undecayed-but-untrained params
    names = {n.split(".", 1)[0] for n, _ in m.named_parameters()}
    assert "patch_stem" not in names


def test_per_band_stem_requires_mean_pool() -> None:
    with pytest.raises(ValueError, match="requires pool='mean'"):
        V14ParcelPerceiverModel(**_band_kw(pool="cross_attn"))


# --------------------------------------------------------------------------- #
# forward dispatch guards
# --------------------------------------------------------------------------- #
def test_high_band_required_in_dual_mode() -> None:
    m = _make()
    low, _high, sup, val = _band_inputs(1, 4, 6, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    with pytest.raises(ValueError, match="requires electrode_tokens_high"):
        m(low, sup, val, return_taps=True, m2_only=True)


def test_high_band_rejected_on_single_grid_model() -> None:
    sg = V14ParcelPerceiverModel(
        n_freq_bins=6, n_time_bins=4, k_parcels=6, d_model=32, n_heads=4,
        depth_self_attn=2, m_sub_slots=1, n_token_blocks=2,
        patch_kernel_freq=3, pool="mean",
    )
    et = torch.randn(1, 4, 4, 6)
    sup = torch.zeros(1, 4, 6)
    for c in range(4):
        sup[:, c, c] = 1.0
    val = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="only valid with per_band_stem"):
        sg(et, sup, val, electrode_tokens_high=torch.randn(1, 4, 4, 6),
           return_taps=True, m2_only=True)


def test_dual_band_m2_only_requires_return_taps() -> None:
    m = _make()
    low, high, sup, val = _band_inputs(1, 4, 6, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    with pytest.raises(ValueError, match="m2_only=True requires return_taps"):
        m(low, sup, val, electrode_tokens_high=high, return_taps=False, m2_only=True)


# --------------------------------------------------------------------------- #
# M4 latent — parcel-SA-only flat-S middle + whole-parcel tube (Ben 2026-06-13)
# --------------------------------------------------------------------------- #
def _fwd_m4(model, low, high, support, valid, tubed=None):
    return model(
        low, support, valid, electrode_tokens_high=high,
        return_taps=True, m2_only=False, dual_band_parcel_mask=tubed,
    )


def test_dual_band_m4_tap_shape_and_finite() -> None:
    """m2_only=False runs the parcel-SA-only latent on the flat dual-band tap and
    returns the (B, K, S, d) M4 tap (same flat shape as M2), finite."""
    m = _make().eval()
    K, d = 6, 32
    low, high, sup, val = _band_inputs(2, 6, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    out = _fwd_m4(m, low, high, sup, val)
    assert out["M4"].shape == (2, K, 104, d)
    assert out["M2"].shape == (2, K, 104, d)
    assert out["latent_valid"].shape == (2, K)
    assert torch.isfinite(out["M4"]).all()


def test_dual_band_m4_bare_tap_without_return_taps() -> None:
    """return_taps=False, m2_only=False → bare M4 tensor (the readout input)."""
    m = _make().eval()
    K, d = 6, 32
    low, high, sup, val = _band_inputs(1, 6, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    m4 = m(low, sup, val, electrode_tokens_high=high, return_taps=False, m2_only=False)
    assert isinstance(m4, torch.Tensor)
    assert m4.shape == (1, K, 104, d)


def test_dual_band_m4_precision_stats_shape() -> None:
    """With mean_pool_std ON the M4 tap carries flat-S precision σ (B, K, S) and
    per-parcel n_k (B, K) for the heteroscedastic loss; σ aligns 1:1 with M4."""
    m = _make(mean_pool_std=True).eval()
    K = 6
    low, high, sup, val = _band_inputs(2, 6, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    out = _fwd_m4(m, low, high, sup, val)
    assert out["M4_precision_std"].shape == (2, K, 104)
    assert out["M4_precision_n"].shape == (2, K)
    assert torch.isfinite(out["M4_precision_std"]).all()
    # 6 electrodes 1:1 onto 6 parcels → n_k == 1 for every covered parcel.
    assert out["M4_precision_n"].eq(1).all()


def test_dual_band_m4_precision_std_none_without_mean_pool_std() -> None:
    """mean_pool_std OFF → no σ channel → precision_std None (loss gated upstream)."""
    m = _make().eval()
    K = 6
    low, high, sup, val = _band_inputs(2, 6, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    out = _fwd_m4(m, low, high, sup, val)
    assert out["M4_precision_std"] is None
    assert out["M4_precision_n"].shape == (2, K)


def test_dual_band_m4_tube_changes_survivor_reps() -> None:
    """The M4 tube is applied: tubing a parcel changes the SURVIVING parcels' M4
    (they no longer attend to the tubed parcel as a parcel-latent key)."""
    m = _make().eval()
    K = 6
    low, high, sup, val = _band_inputs(2, 6, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    tubed = torch.zeros(2, K, dtype=torch.bool)
    tubed[:, 0] = True
    m4_tube = _fwd_m4(m, low, high, sup, val, tubed=tubed)["M4"]
    m4_full = _fwd_m4(m, low, high, sup, val)["M4"]
    assert not torch.allclose(m4_tube[:, 1:], m4_full[:, 1:])


def test_dual_band_m4_tubed_parcel_independent_of_own_input() -> None:
    """A tubed parcel's tokens are dropped from its within-parcel token block AND
    it is excluded from the parcel latent — so a SURVIVING parcel's M4 is
    invariant to perturbing the TUBED parcel's own band input (it is the masked
    target, reconstructed from survivors, never an attendable key)."""
    m = _make().eval()
    K = 6
    low, high, sup, val = _band_inputs(2, 6, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    tubed = torch.zeros(2, K, dtype=torch.bool)
    tubed[:, 0] = True                                       # parcel 0 = electrode 0
    base = _fwd_m4(m, low, high, sup, val, tubed=tubed)["M4"]
    low2 = low.clone()
    low2[:, 0, :, :] += 50.0                                 # perturb parcel-0's electrode
    pert = _fwd_m4(m, low2, high, sup, val, tubed=tubed)["M4"]
    # survivors 1..5 never attend to tubed parcel 0 → their M4 is invariant.
    torch.testing.assert_close(pert[:, 1:], base[:, 1:])


def test_dual_band_m4_ragged_matches_dense_on_visible() -> None:
    """#112 ragged-parcel gather is a pure FLOP cut: every VISIBLE parcel's M4
    equals the dense path up to float32 re-tiling (~1e-5)."""
    K = 6
    low, high, sup, val = _band_inputs(2, 6, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    tubed = torch.zeros(2, K, dtype=torch.bool)
    tubed[:, 0] = True                                       # parcel 0 tubed
    dense = _fwd_m4(_make(ragged_parcel=False).eval(), low, high, sup, val, tubed)["M4"]
    rag = _fwd_m4(_make(ragged_parcel=True).eval(), low, high, sup, val, tubed)["M4"]
    vis = slice(1, None)                                     # surviving parcels
    torch.testing.assert_close(dense[:, vis], rag[:, vis], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("ragged", [False, True])
def test_dual_band_m4_tube_finite_and_grads_clean(ragged: bool) -> None:
    """A tubed parcel is an all-masked query row in the parcel attention; its M4
    must be FINITE (not NaN) and backprop through the survivors must not poison
    the shared latent-block gradients — in BOTH the dense and #112 ragged paths."""
    m = _make(ragged_parcel=ragged, mean_pool_std=True).train()
    K = 6
    low, high, sup, val = _band_inputs(2, 6, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    tubed = torch.zeros(2, K, dtype=torch.bool)
    tubed[:, 0] = True
    out = _fwd_m4(m, low, high, sup, val, tubed=tubed)
    assert torch.isfinite(out["M4"]).all()                  # incl. the tubed parcel
    out["M4"][:, 1:].pow(2).mean().backward()
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_dual_band_m4_joint_latent_mode_rejected() -> None:
    """The dual-rate flat-S tap has no clean (F_p, T_p) grid, so the joint
    parcel×time latent is unsupported on the 2STFT path."""
    m = _make(latent_mode="joint").eval()
    K = 6
    low, high, sup, val = _band_inputs(1, 6, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    with pytest.raises(NotImplementedError, match="latent_mode='parcel' only"):
        _fwd_m4(m, low, high, sup, val)


def test_dual_band_m4_parcel_mask_wrong_shape_raises() -> None:
    m = _make().eval()
    K = 6
    low, high, sup, val = _band_inputs(1, 6, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    bad = torch.zeros(1, K + 1, dtype=torch.bool)            # K mismatch
    with pytest.raises(ValueError, match=r"tubed shape.*\(B, K\)"):
        _fwd_m4(m, low, high, sup, val, tubed=bad)


def test_dual_band_rejects_single_grid_token_mask() -> None:
    # The single-grid (B, K, F_p, T_p) token_mask / parcel_time_mask do not apply
    # to the dual-rate flat tap; the per_band path redirects to the FLAT
    # (B, K, S) dual_band_token_mask (§5/§6). The flat mask itself is exercised by
    # the §6 drop-token tests below.
    m = _make()
    low, high, sup, val = _band_inputs(1, 4, 6, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    tok_mask = torch.zeros(1, 6, 7, 8, dtype=torch.bool)
    with pytest.raises(NotImplementedError, match="FLAT.*dual_band_token_mask"):
        m(low, sup, val, electrode_tokens_high=high, token_mask=tok_mask,
          return_taps=True, m2_only=True)


def test_dual_band_freq_bin_mismatch_raises() -> None:
    m = _make()
    _low, high, sup, val = _band_inputs(1, 4, 6, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    bad_low = torch.randn(1, 4, 13, _LOW_T_1S)  # 13 != 14
    with pytest.raises(ValueError, match="freq-bin mismatch"):
        m(bad_low, sup, val, electrode_tokens_high=high,
          return_taps=True, m2_only=True)


def test_dual_band_mismatched_high_time_grid_raises() -> None:
    # high band with an ODD post-conv time count breaks the dual-rate grid.
    # high T=32 → conv (32-2)//2+1 = 16 (even, OK). high T=34 → 17 (odd).
    m = _make()
    low, _h, sup, val = _band_inputs(1, 4, 6, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    bad_high = torch.randn(1, 4, 9, 34)  # → T_high_p = 17 (odd)
    with pytest.raises(ValueError, match="even"):
        m(low, sup, val, electrode_tokens_high=bad_high,
          return_taps=True, m2_only=True)


# --------------------------------------------------------------------------- #
# single-grid path untouched (cross-check)
# --------------------------------------------------------------------------- #
def test_single_grid_meanpool_still_has_patch_stem() -> None:
    sg = V14ParcelPerceiverModel(
        n_freq_bins=6, n_time_bins=4, k_parcels=6, d_model=32, n_heads=4,
        depth_self_attn=2, m_sub_slots=1, n_token_blocks=2,
        patch_kernel_freq=3, pool="mean",
    )
    assert sg.patch_stem is not None
    assert sg.rope_joint_token is not None
    assert sg.per_band_stem is False


# --------------------------------------------------------------------------- #
# §6 — M2 drop-token SSL contract (visible-only student + ONE predictor)
# --------------------------------------------------------------------------- #
def _fwd_masked(model, low, high, support, valid, mask):
    return model(
        low, support, valid, electrode_tokens_high=high,
        dual_band_token_mask=mask, return_taps=True, m2_only=True,
    )


def test_drop_token_forward_shape_and_finite() -> None:
    """A masked (student) forward returns the same flat (B, K, S, d) M2 and is
    finite — the drop-token key-pad does not change the tap shape."""
    m = _make().eval()
    K = 6
    low, high, sup, val = _band_inputs(2, 5, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    g = torch.Generator().manual_seed(0)
    mask = sample_m2_dual_band_mask(
        2, K, F_p_low=7, T_low_p=8, F_p_high=3, T_high_p=16, generator=g,
    )
    out = _fwd_masked(m, low, high, sup, val, mask)
    assert out["M2"].shape == (2, K, 104, 32)
    assert torch.isfinite(out["M2"]).all()


def test_drop_token_mask_changes_reps_vs_unmasked() -> None:
    """The mask is actually applied: a masked forward differs from the unmasked
    forward (key-padding masked tokens changes the visible tokens' attention)."""
    m = _make().eval()
    K = 6
    low, high, sup, val = _band_inputs(2, 5, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    g = torch.Generator().manual_seed(1)
    mask = sample_m2_dual_band_mask(
        2, K, F_p_low=7, T_low_p=8, F_p_high=3, T_high_p=16, generator=g,
    )
    masked = _fwd_masked(m, low, high, sup, val, mask)["M2"]
    unmasked = _fwd(m, low, high, sup, val)["M2"]
    assert not torch.allclose(masked, unmasked)


def test_drop_token_is_true_drop_not_zerofill() -> None:
    """THE drop-vs-zero-fill distinction (§6 'dropped, not zero-filled'):

    Mask low freq-patches {0,1} across all time (input freq bins [0,4), since the
    low stem is non-overlapping kernel/stride 2). A visible token's M2 must be
    INVARIANT to perturbing those masked-only input bins — visible tokens attend
    ONLY to visible keys, never the (key-padded) masked tokens. Sanity: the SAME
    perturbation in an UNMASKED forward DOES move the visible reps, proving the
    perturbation is non-inert and the invariance is genuinely the drop."""
    m = _make().eval()
    K = 6
    low, high, sup, val = _band_inputs(2, 5, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    low_masked = [i for i in range(56) if i % 7 in (0, 1)]   # low f∈{0,1}, all 8 time
    vis = [i for i in range(104) if i not in low_masked]     # vis low {2..6} + all high
    mask = torch.zeros(2, K, 104, dtype=torch.bool)
    mask[:, :, low_masked] = True

    base = _fwd_masked(m, low, high, sup, val, mask)["M2"]
    low2 = low.clone()
    low2[:, :, 0:4, :] += 50.0                               # bins feeding patches 0,1 only
    perturbed = _fwd_masked(m, low2, high, sup, val, mask)["M2"]
    # visible-token M2 invariant to the masked-only perturbation → true drop.
    torch.testing.assert_close(perturbed[:, :, vis], base[:, :, vis])

    # sanity: unmasked, the identical perturbation DOES move the visible reps.
    um = _fwd(m, low, high, sup, val)["M2"]
    um_p = _fwd(m, low2, high, sup, val)["M2"]
    assert not torch.allclose(um_p[:, :, vis], um[:, :, vis]), (
        "perturbation was inert — the drop-invariance test is vacuous"
    )


def test_drop_token_fully_masked_parcel_is_finite() -> None:
    """A fully-masked parcel (every token dropped) must not NaN — the empty-key
    SDPA softmax is guarded (slot-0 kept, its output discarded downstream)."""
    m = _make().eval()
    K = 6
    low, high, sup, val = _band_inputs(2, 5, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    mask = torch.zeros(2, K, 104, dtype=torch.bool)
    mask[:, 0, :] = True                                     # parcel 0 fully masked
    out = _fwd_masked(m, low, high, sup, val, mask)
    assert torch.isfinite(out["M2"]).all()


def test_dual_band_token_mask_rejected_off_per_band() -> None:
    """dual_band_token_mask is only valid on the 2STFT (per_band_stem) path."""
    sg = V14ParcelPerceiverModel(
        n_freq_bins=6, n_time_bins=4, k_parcels=4, d_model=16, n_heads=2,
        depth_self_attn=1, m_sub_slots=1, n_token_blocks=1,
        patch_kernel_freq=3, pool="mean",
    )
    x = torch.randn(2, 3, 6, 4)
    sup = torch.zeros(2, 3, 4); sup[:, :, 0] = 1.0
    with pytest.raises(ValueError, match="dual_band_token_mask is only valid"):
        sg(x, sup, None, dual_band_token_mask=torch.zeros(2, 4, 10, dtype=torch.bool),
           return_taps=True, m2_only=True)


def test_dual_band_token_mask_wrong_shape_raises() -> None:
    m = _make().eval()
    K = 6
    low, high, sup, val = _band_inputs(2, 5, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    bad = torch.zeros(2, K, 103, dtype=torch.bool)           # S should be 104
    with pytest.raises(ValueError, match=r"token_mask.*\(B, K, S\)"):
        _fwd_masked(m, low, high, sup, val, bad)


def test_m2_dual_band_contract_end_to_end() -> None:
    """Full §6 contract: drop-token student + EMA-style full teacher + ONE
    predictor (parcel=query_id, freq-patch=query_id_2, slot=RoPE) + L1. The loss
    is finite, n_masked counts the covered masked tokens, and gradient flows to
    BOTH the predictor and the student encoder (paradigm B), not the teacher."""
    m = _make().eval()
    K, d = 6, 32
    low, high, sup, val = _band_inputs(2, 6, K, t_low=_LOW_T_1S, t_high=_HIGH_T_1S)
    g = torch.Generator().manual_seed(3)
    mask = sample_m2_dual_band_mask(
        2, K, F_p_low=7, T_low_p=8, F_p_high=3, T_high_p=16, generator=g,
    )
    # student sees visible tokens only; teacher sees the full set (no grad).
    student = _fwd_masked(m, low, high, sup, val, mask)
    with torch.no_grad():
        teacher = _fwd(m, low, high, sup, val)
    layout = m.dual_band_token_layout(8, 16)

    torch.manual_seed(5)
    predictor = JepaPredictor(
        d_model=d, n_identity=K, hidden=16, n_heads=2, depth=2,
        max_time_patches=16, id_pos="learned",
        n_identity_2=m.n_freq_patches, id_pos_2="sinusoidal",
    )
    bd = m2_dual_band_loss(
        predictor=predictor,
        student_m2=student["M2"],
        teacher_m2=teacher["M2"],
        token_mask=mask,
        slot_ids=layout["slot"],
        freq_patch_ids=layout["freq_patch_idx"],
        latent_valid=student["latent_valid"],
    )
    assert bd.phase == "m2_dual_band"
    covered = student["latent_valid"].unsqueeze(-1).expand(2, K, 104)
    assert bd.n_masked == int((mask & covered).sum())
    assert torch.isfinite(bd.total)

    bd.total.backward()
    pred_grad = [
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in predictor.named_parameters() if "id_embed" not in n
    ]
    assert pred_grad and all(pred_grad), "predictor got no gradient"
    enc_grad = [
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in m.patch_stem_low.parameters()
    ]
    assert any(enc_grad), "student encoder (low stem) got no gradient"
