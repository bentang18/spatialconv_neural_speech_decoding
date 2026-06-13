"""#93 token-drop (V-JEPA-2 visible-only front-end) parity tests — MEAN path.

Companion to ``test_v14_ragged_token.py`` (which pins the B36 cross_attn
per-ELECTRODE path). Here the encoder is ``pool="mean"`` (B37): the
electrode→parcel mean|std pool runs up front, so the per-parcel JOINT token
blocks operate on ``(B·K, S=F_p·T_p, d)`` PARCEL rows and the M2 tap is
``(B, K, F_p, T_p, d)``. The token drop gathers the VISIBLE ``(t_p, f_p)``
tokens per PARCEL before the blocks, runs the blocks over the kept set with a
per-row time-RoPE, then scatters back (dropped → zeros).

Same contract as the cross_attn file, re-proved on the mean path:
  * default OFF on the dataclass + the model.
  * all-False token_mask ⇒ ragged == dense BIT-identical (gather/scatter no-op).
  * full-encoder M2: drop == KEY-MASK reference at VISIBLE tokens (~1e-5), using
    the model's real token blocks (dense input captured via a forward hook).
  * the drop DIFFERS from the zero-in-place dense path at visible tokens (the
    representation change is real, not a no-op); dropped tokens scatter to exact
    zeros (frontend_ln bias is zero-init, so LN(0)=0).
  * fully-masked / tubed parcels do NOT produce NaN (empty-row SDPA guard).
  * the ``ragged_token``/``ragged_frontend`` combined gather is non-overlapping:
    with both ON and an uncovered parcel, the uncovered parcel is dropped and the
    covered parcels' visible M2 still equals the key-mask reference.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import (
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
)


def _kw() -> dict:
    return {
        "n_freq_bins": 6,
        "n_time_bins": 4,
        "k_parcels": 6,
        "d_model": 32,
        "n_heads": 4,
        "depth_self_attn": 2,
        "latent_parcel_depth": 2,
        "m_sub_slots": 1,
        "patch_kernel_freq": 3,
        "pool": "mean",
    }


def _batch(seed: int = 0):
    """Tiny fully-covered batch. C=8 electrodes → K=6 parcels (c % K), so every
    parcel is covered (parcels 0,1 twice)."""
    kw = _kw()
    g = torch.Generator().manual_seed(seed)
    B, C, K = 2, 8, kw["k_parcels"]
    et = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"], generator=g)
    support = torch.zeros(B, C, K)
    vm = torch.ones(B, C, dtype=torch.bool)
    for b in range(B):
        for c in range(C):
            support[b, c, c % K] = 1.0
    return kw, et, support, vm


def _grid(model, et) -> tuple[int, int]:
    """(F_p, T_p) — the per-parcel token grid (pool-independent)."""
    _, F_p, T_p = model.patch_grid_shape(et)
    return F_p, T_p


def _flat_keep(token_mask: torch.Tensor) -> torch.Tensor:
    """``token_mask`` (B,K,F_p,T_p) → flat keep (B·K, S) in the encoder's flat
    token order (i = t_p·F_p + f_p ⇒ permute F_p/T_p before reshape)."""
    B, K, F_p, T_p = token_mask.shape
    return (~token_mask).permute(0, 1, 3, 2).reshape(B * K, F_p * T_p)


# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #
def test_meanpool_ragged_token_default_off() -> None:
    req = {"n_freq_bins": 6, "n_time_bins": 4, "k_parcels": 6,
           "patch_kernel_freq": 3, "pool": "mean"}
    assert V14ParcelPerceiver(**req).ragged_token is False
    assert V14ParcelPerceiverModel(**_kw()).ragged_token is False
    cfg = V14ParcelPerceiver(**req, ragged_token=True)
    assert cfg.build(n_outputs=2).encoder.ragged_token is True


# --------------------------------------------------------------------------- #
# all-False token_mask: gather/scatter no-op ⇒ bit-identical
# --------------------------------------------------------------------------- #
def test_meanpool_token_all_false_mask_bit_identical() -> None:
    kw, et, support, vm = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    F_p, T_p = _grid(model, et)
    B, K = et.shape[0], kw["k_parcels"]
    tm0 = torch.zeros(B, K, F_p, T_p, dtype=torch.bool)
    model.ragged_frontend = False
    with torch.no_grad():
        model.ragged_token = False
        d = model(et, support, valid_mask=vm, return_taps=True,
                  m2_only=True, token_mask=tm0)["M2"]
        model.ragged_token = True
        r = model(et, support, valid_mask=vm, return_taps=True,
                  m2_only=True, token_mask=tm0)["M2"]
    assert torch.equal(r, d), "all-False mask: gather/scatter must be a no-op"


# --------------------------------------------------------------------------- #
# THE core proof: full-encoder drop == key-mask reference at VISIBLE tokens
# --------------------------------------------------------------------------- #
def _keymask_reference_m2(model, et, support, vm, token_mask, *, fold_valid=None):
    """Reference visible-only M2 from the model's REAL token blocks.

    Hook the dense (ragged_token=False) forward to capture the flat per-parcel
    token-block input (B·K, S, d), then re-run ALL blocks with the masked tokens
    KEY-MASKED out, reshape to the M2 grid, frontend_ln. ``fold_valid`` (B,K)
    additionally key-masks whole uncovered parcels (the ragged_frontend fold)."""
    captured = {}

    def _pre_hook(_mod, args):
        captured["x"] = args[0].detach().clone()   # (B·K, S, d)
        captured["rope"] = args[1]
        captured["kpm"] = args[2]

    # Capture the FULL dense (B·K, S, d) token-block input: force BOTH ragged
    # flags off so neither the uncovered-parcel pack (ragged_frontend) nor the
    # token drop (ragged_token) reshapes the rows — the reference re-applies the
    # drop / parcel fold itself as a pure key-mask.
    saved = (model.ragged_token, model.ragged_frontend)
    h = model.token_blocks[0].register_forward_pre_hook(_pre_hook)
    try:
        model.ragged_token = False
        model.ragged_frontend = False
        with torch.no_grad():
            model(et, support, valid_mask=vm, return_taps=True,
                  m2_only=True, token_mask=token_mask)
    finally:
        h.remove()
        model.ragged_token, model.ragged_frontend = saved

    B, K, F_p, T_p = token_mask.shape
    keep = _flat_keep(token_mask)                  # (B·K, S) True=visible
    if fold_valid is not None:
        keep = keep & fold_valid.reshape(B * K, 1)
    kpm = keep
    if captured["kpm"] is not None:
        kpm = kpm & captured["kpm"]
    # empty-row guard mirrors the encoder (force slot-0 so SDPA never sees an
    # all-masked key set); those rows are dropped/zeroed downstream anyway.
    empty = ~kpm.any(dim=1, keepdim=True)
    if empty.any():
        ar = torch.arange(kpm.shape[1]) == 0
        kpm = kpm | (empty & ar.unsqueeze(0))
    x = captured["x"]
    with torch.no_grad():
        for tb in model.token_blocks:
            x = tb(x, captured["rope"], kpm)
        m2 = (x.reshape(B, K, T_p, F_p, model.d_model)
                .permute(0, 1, 3, 2, 4).contiguous())
        m2 = model.frontend_ln(m2)
    return m2


def test_meanpool_token_drop_equals_keymask_reference() -> None:
    kw, et, support, vm = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    F_p, T_p = _grid(model, et)
    B, K = et.shape[0], kw["k_parcels"]
    model.ragged_frontend = False
    g = torch.Generator().manual_seed(3)
    token_mask = torch.rand(B, K, F_p, T_p, generator=g) < 0.4
    fully = token_mask.reshape(B, K, -1).all(dim=2)
    token_mask.reshape(B, K, -1)[fully] = False    # ≥1 visible per parcel

    ref = _keymask_reference_m2(model, et, support, vm, token_mask)
    model.ragged_token = True
    with torch.no_grad():
        drop = model(et, support, valid_mask=vm, return_taps=True,
                     m2_only=True, token_mask=token_mask)["M2"]
    vis = ~token_mask
    torch.testing.assert_close(drop[vis], ref[vis], atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# representation change is real + dropped tokens are exact zeros
# --------------------------------------------------------------------------- #
def test_meanpool_token_drop_differs_from_zerofill() -> None:
    kw, et, support, vm = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    F_p, T_p = _grid(model, et)
    B, K = et.shape[0], kw["k_parcels"]
    model.ragged_frontend = False
    token_mask = torch.zeros(B, K, F_p, T_p, dtype=torch.bool)
    token_mask[:, :, 0, 1] = True
    vis = ~token_mask

    model.ragged_token = True
    with torch.no_grad():
        drop = model(et, support, valid_mask=vm, return_taps=True,
                     m2_only=True, token_mask=token_mask)["M2"]
    model.ragged_token = False
    with torch.no_grad():
        dense = model(et, support, valid_mask=vm, return_taps=True,
                      m2_only=True, token_mask=token_mask)["M2"]
    assert not torch.allclose(drop[vis], dense[vis], atol=1e-4), \
        "drop must differ from zero-in-place at visible tokens (real change)"
    assert torch.count_nonzero(drop[token_mask]) == 0, \
        "masked tokens must scatter back as exact zeros (frontend_ln bias=0)"


# --------------------------------------------------------------------------- #
# fully-masked / tubed parcels: no NaN (empty-row SDPA guard)
# --------------------------------------------------------------------------- #
def test_meanpool_token_fully_masked_parcels_no_nan() -> None:
    kw, et, support, vm = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    F_p, T_p = _grid(model, et)
    B, K = et.shape[0], kw["k_parcels"]
    model.ragged_frontend = False
    token_mask = torch.zeros(B, K, F_p, T_p, dtype=torch.bool)
    token_mask[:, 0] = True       # parcel 0 fully masked
    token_mask[:, 2] = True       # parcel 2 fully masked
    model.ragged_token = True
    with torch.no_grad():
        m2 = model(et, support, valid_mask=vm, return_taps=True,
                   m2_only=True, token_mask=token_mask)["M2"]
    assert torch.isfinite(m2).all(), "fully-masked parcels must not NaN/Inf"
    assert torch.count_nonzero(m2[:, 0]) == 0
    assert torch.count_nonzero(m2[:, 2]) == 0


# --------------------------------------------------------------------------- #
# non-overlap: ragged_token + ragged_frontend combined gather is correct
# --------------------------------------------------------------------------- #
def test_meanpool_token_drop_with_ragged_frontend_combined_gather() -> None:
    """With ragged_frontend ON, an uncovered parcel is folded into the visible
    keep (ONE combined gather, no double-drop). Covered parcels' visible M2 must
    still equal the key-mask reference that folds the same uncovered set."""
    kw = _kw()
    torch.manual_seed(7)
    g = torch.Generator().manual_seed(1)
    B, C, K = 2, 8, kw["k_parcels"]
    et = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"], generator=g)
    # Leave parcel 5 UNCOVERED: route electrodes only to parcels 0..4.
    support = torch.zeros(B, C, K)
    vm = torch.ones(B, C, dtype=torch.bool)
    for b in range(B):
        for c in range(C):
            support[b, c, c % (K - 1)] = 1.0     # parcels 0..4 only
    model = V14ParcelPerceiverModel(**kw).eval()
    F_p, T_p = _grid(model, et)
    model.ragged_frontend = True
    latent_valid = (support.sum(dim=1) > 0)       # (B, K); parcel 5 False

    token_mask = torch.rand(B, K, F_p, T_p, generator=g) < 0.4
    fully = token_mask.reshape(B, K, -1).all(dim=2)
    token_mask.reshape(B, K, -1)[fully] = False

    ref = _keymask_reference_m2(model, et, support, vm, token_mask,
                                fold_valid=latent_valid)
    model.ragged_token = True
    with torch.no_grad():
        drop = model(et, support, valid_mask=vm, return_taps=True,
                     m2_only=True, token_mask=token_mask)["M2"]
    # Compare only at VISIBLE tokens of COVERED parcels (uncovered parcel 5 is
    # dropped → zeros in both).
    vis = ~token_mask
    vis = vis & latent_valid.unsqueeze(-1).unsqueeze(-1)
    torch.testing.assert_close(drop[vis], ref[vis], atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(drop[:, 5]) == 0, "uncovered parcel must drop to 0"
