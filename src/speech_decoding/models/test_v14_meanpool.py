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
  * the not-yet-wired masking / conditioning args fail loud on the mean path.
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


def test_meanpool_identity_latent_m4_is_ln_of_m2() -> None:
    """Chunk A baseline: with the latent identity, M4 = encoder_ln(M2)."""
    enc = _make().eval()
    kw = _mp_kw()
    et, support, valid = _inputs(2, 8, kw)
    out = enc(et, support, valid_mask=valid, return_taps=True)
    torch.testing.assert_close(out["M4"], enc.encoder_ln(out["M2"]))


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
# not-yet-wired args fail loud on the mean path (Chunk A guards)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"token_mask": True},
        {"parcel_time_mask": True},
        {"return_m3": True},
    ],
)
def test_meanpool_unsupported_args_raise(kwargs: dict) -> None:
    enc = _make().eval()
    kw = _mp_kw()
    et, support, valid = _inputs(2, 8, kw)
    T_p = enc.patch_stem.n_time_patches(kw["n_time_bins"])
    F_p = enc.n_freq_patches
    call: dict = {"return_taps": True}
    if kwargs.get("token_mask"):
        call["token_mask"] = torch.zeros(2, 8, F_p, T_p, dtype=torch.bool)
    if kwargs.get("parcel_time_mask"):
        call["parcel_time_mask"] = torch.zeros(2, kw["k_parcels"], T_p, dtype=torch.bool)
    if kwargs.get("return_m3"):
        call["return_m3"] = True
    with pytest.raises(NotImplementedError):
        enc(et, support, valid_mask=valid, **call)
