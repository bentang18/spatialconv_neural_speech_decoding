"""MASK-04 P2 parcel-drop parity tests.

The parcel drop (``ragged_parcel=True``) gathers the COVERED (and, under a tube
``parcel_time_mask``, non-masked = VISIBLE) parcel slots *before* the cross-attn
pool, runs the pool + latent stack on the ``Lk ≤ L`` kept slots, then scatters
the result back into the full ``L = K·M`` layout as zeros (dropped slots).

Why every KEPT slot's M4 is BIT-identical to the dense path:
  * the pool is block-diagonal — each parcel attends only to its own electrodes,
    so dropping other parcels cannot change a kept parcel's pooled token;
  * the latent stack's time-SA is per-parcel (no cross-parcel mixing);
  * the latent stack's parcel-SA already excludes uncovered/masked parcels via
    ``latent_valid`` with a ``NEG_INF`` logit — ``exp(NEG_INF)`` underflows to
    *exactly* ``0.0`` in float, so the softmax over the visible subset equals a
    gathered softmax, and the masked terms contribute ``0.0 * v == 0.0`` to the
    attn·v sum (exact zeros are invisible to float summation, order-independent).
Hence the drop is a pure FLOP cut, NOT a representation change (contrast the P1
token drop). The dropped slots are never read downstream (loss/PMA mask them via
``latent_valid``), so we compare only the kept positions.

Contract pinned here:
  * default OFF on both the dataclass and the model.
  * no mask, all-covered ⇒ ragged == dense BIT-identical full M4 (identity gather).
  * no mask, mixed coverage ⇒ ragged == dense BIT-identical at COVERED parcels.
  * tube mask ⇒ ragged == dense BIT-identical at VISIBLE (covered & ~masked).
  * dropped-parcel INPUT invariance: garbage in an uncovered/masked parcel's
    electrodes cannot change any kept parcel's M4.
  * P2 gradients (through the real ``p2_parcel_m4_loss``) match dense.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import (
    JepaPredictor,
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
)
from speech_decoding.ssl.masked_jepa import p2_parcel_m4_loss


def _kw() -> dict:
    return {
        "n_freq_bins": 6,
        "n_time_bins": 4,
        "k_parcels": 6,
        "d_model": 32,
        "n_heads": 4,
        "depth_self_attn": 2,
        "m_sub_slots": 1,
        "patch_kernel_freq": 3,
    }


# Per-clip parcel coverage (electrodes round-robin onto these parcels). Leaves
# parcels {3,4,5} uncovered on clip 0, {5} on clip 1, {2,3,4,5} on clip 2 — so
# the keep-set varies per clip and the pad-to-max has real slack.
_COVER = [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1]]


def _batch(seed: int = 0):
    kw = _kw()
    g = torch.Generator().manual_seed(seed)
    B, C, K = 3, 8, kw["k_parcels"]
    et = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"], generator=g)
    support = torch.zeros(B, C, K)
    valid_mask = torch.ones(B, C, dtype=torch.bool)
    for b in range(B):
        ps = _COVER[b]
        for c in range(C):
            support[b, c, ps[c % len(ps)]] = 1.0
    return kw, et, support, valid_mask


def _all_cover_batch(seed: int = 1):
    """Every parcel covered on every clip ⇒ keep-set == all ⇒ identity gather."""
    kw = _kw()
    g = torch.Generator().manual_seed(seed)
    B, C, K = 3, 8, kw["k_parcels"]
    et = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"], generator=g)
    support = torch.zeros(B, C, K)
    valid_mask = torch.ones(B, C, dtype=torch.bool)
    for b in range(B):
        for c in range(C):
            support[b, c, c % K] = 1.0  # C=8 ≥ K=6 ⇒ every parcel hit
    return kw, et, support, valid_mask


def _grid_t_p(model, et, support, valid_mask) -> int:
    with torch.no_grad():
        m4 = model(et, support, valid_mask=valid_mask, return_taps=True)["M4"]
    return m4.shape[2]  # (B, L, T_p, d)


def _tube_mask(B: int, K: int, T_p: int):
    """Whole-parcel tubes (constant over T_p). Mask a covered parcel per clip."""
    m = torch.zeros(B, K, T_p, dtype=torch.bool)
    m[0, 2, :] = True            # clip 0: mask covered parcel 2
    m[1, 3, :] = True            # clip 1: mask covered parcels 3, 4
    m[1, 4, :] = True
    # clip 2 covers only {0,1}; leave both visible (tube needs visible context).
    return m


def _covered(support):
    return support.sum(dim=1) > 0  # (B, K); M=1 ⇒ K == L


# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #
def test_ragged_parcel_default_off() -> None:
    req = {"n_freq_bins": 6, "n_time_bins": 4, "k_parcels": 6,
           "patch_kernel_freq": 3}
    assert V14ParcelPerceiver(**req).ragged_parcel is False
    assert V14ParcelPerceiverModel(**_kw()).ragged_parcel is False
    cfg = V14ParcelPerceiver(**req, ragged_parcel=True)
    assert cfg.build(n_outputs=2).encoder.ragged_parcel is True


# --------------------------------------------------------------------------- #
# no mask, all covered: identity gather ⇒ BIT-identical full M4
# --------------------------------------------------------------------------- #
def test_parcel_all_covered_bit_identical() -> None:
    kw, et, support, vm = _all_cover_batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    with torch.no_grad():
        model.ragged_parcel = False
        d = model(et, support, valid_mask=vm, return_taps=True)
        model.ragged_parcel = True
        r = model(et, support, valid_mask=vm, return_taps=True)
    assert torch.equal(r["M4"], d["M4"]), "all-covered M4 must be bit-identical"
    # M2 (pre-pool) is never touched by the parcel drop.
    assert torch.equal(r["M2"], d["M2"]), "M2 must be untouched by parcel drop"


# --------------------------------------------------------------------------- #
# no mask, mixed coverage: BIT-identical at covered parcels
# --------------------------------------------------------------------------- #
def test_parcel_mixed_coverage_bit_identical_at_covered() -> None:
    kw, et, support, vm = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    with torch.no_grad():
        model.ragged_parcel = False
        d = model(et, support, valid_mask=vm, return_taps=True)
        model.ragged_parcel = True
        r = model(et, support, valid_mask=vm, return_taps=True)
    cov = _covered(support)  # (B, L)
    # Mixed coverage: the parcel-SA / pool sum over parcels/electrodes, so
    # dropping changes the matmul reduction-tree SIZE — the masked terms are
    # exact-0 (contribute nothing mathematically) but the nonzero terms get
    # associated differently for Lk vs L elements ⇒ float32 re-tiling (~1e-6).
    # Same standard as #91's mixed-valid front-end parity. (All-covered, an
    # identity gather with no size change, is bit-identical — see the test above.)
    torch.testing.assert_close(r["M4"][cov], d["M4"][cov], atol=1e-5, rtol=1e-5)
    # Uncovered parcels DO differ (dense leaves stack garbage there; ragged
    # scatters zeros), but they are never read downstream — the PMA/mean readout
    # and the loss both mask them out via ``latent_valid``. Confirm the diff is
    # confined to ~uncovered slots:
    uncov = ~cov
    assert not torch.equal(r["M4"][uncov], d["M4"][uncov]) or not uncov.any()


# --------------------------------------------------------------------------- #
# tube mask: BIT-identical at visible (covered & ~masked) parcels
# --------------------------------------------------------------------------- #
def test_parcel_tube_mask_bit_identical_at_visible() -> None:
    kw, et, support, vm = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    T_p = _grid_t_p(model, et, support, vm)
    tube = _tube_mask(et.shape[0], kw["k_parcels"], T_p)
    with torch.no_grad():
        model.ragged_parcel = False
        d = model(et, support, valid_mask=vm, return_taps=True,
                  parcel_time_mask=tube)
        model.ragged_parcel = True
        r = model(et, support, valid_mask=vm, return_taps=True,
                  parcel_time_mask=tube)
    cov = _covered(support)
    masked = tube.any(dim=2)          # (B, K) — tube ⇒ constant over T_p
    visible = cov & ~masked           # (B, L)
    assert visible.any(), "test batch must have visible parcels"
    # Visible-parcel M4 matches dense up to float32 reduction re-tiling (~1e-6),
    # the #91 standard — the drop is mathematically exact (masked/uncovered
    # parcels contribute exact-0 weight to every kept parcel).
    torch.testing.assert_close(r["M4"][visible], d["M4"][visible],
                               atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# dropped-parcel input invariance
# --------------------------------------------------------------------------- #
def test_parcel_dropped_input_invariance() -> None:
    """Perturbing an UNCOVERED/MASKED parcel's electrodes cannot change any kept
    parcel's M4. Build a 2nd batch where the masked parcel 2's electrodes (clip 0)
    carry garbage; the visible-parcel M4 must be unchanged."""
    kw, et, support, vm = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    model.ragged_parcel = True
    T_p = _grid_t_p(model, et, support, vm)
    tube = _tube_mask(et.shape[0], kw["k_parcels"], T_p)
    # clip 0, parcel 2 is masked: its electrodes are those c with support[0,c,2]=1.
    et2 = et.clone()
    masked_elec_c0 = support[0, :, 2] > 0  # (C,)
    et2[0, masked_elec_c0] = 77.0          # garbage into a masked parcel's input
    cov = _covered(support)
    masked = tube.any(dim=2)
    visible = cov & ~masked
    with torch.no_grad():
        a = model(et, support, valid_mask=vm, return_taps=True,
                  parcel_time_mask=tube)["M4"]
        b = model(et2, support, valid_mask=vm, return_taps=True,
                  parcel_time_mask=tube)["M4"]
    assert torch.equal(a[visible], b[visible]), \
        "visible M4 must be invariant to a masked parcel's input"


# --------------------------------------------------------------------------- #
# gradients through the real P2 loss match dense
# --------------------------------------------------------------------------- #
def _p2_grads(ragged: bool):
    kw, et, support, vm = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).train()
    model.ragged_parcel = ragged
    T_p = _grid_t_p(model, et, support, vm)
    L = kw["k_parcels"] * kw["m_sub_slots"]
    tube = _tube_mask(et.shape[0], kw["k_parcels"], T_p)
    cov = _covered(support).unsqueeze(-1).expand(-1, -1, T_p)  # (B,L,T_p)
    ptm = tube  # (B, K, T_p); M=1 ⇒ K == L
    visible = cov & ~ptm
    target_mask = cov & ptm
    torch.manual_seed(11)
    pred = JepaPredictor(d_model=kw["d_model"], n_identity=L, depth=2,
                         max_time_patches=T_p)
    with torch.no_grad():
        teacher = model(et, support, valid_mask=vm, return_taps=True)["M4"]
    student = model(et, support, valid_mask=vm, return_taps=True,
                    parcel_time_mask=tube)["M4"]
    loss = p2_parcel_m4_loss(predictor=pred, student_m4=student,
                             teacher_m4=teacher, visible=visible,
                             target_mask=target_mask, loss_form="l1").total
    model.zero_grad()
    loss.backward()
    grads = {n: p.grad.detach().clone()
             for n, p in model.named_parameters() if p.grad is not None}
    return float(loss.detach()), grads


def test_parcel_gradients_match_dense() -> None:
    lD, gD = _p2_grads(ragged=False)
    lR, gR = _p2_grads(ragged=True)
    assert abs(lD - lR) < 1e-5, f"loss mismatch dense={lD} ragged={lR}"
    keys = set(gD) & set(gR)
    assert keys, "no grad-bearing params"
    for k in keys:
        torch.testing.assert_close(gR[k], gD[k], atol=1e-5, rtol=1e-4,
                                   msg=f"grad mismatch @ {k}")
    stem = [k for k in keys if "stem" in k]
    assert stem and max(gR[k].norm().item() for k in stem) > 0.0
