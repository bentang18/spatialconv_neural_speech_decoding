"""#91 ragged front-end parity tests.

The ragged front-end (``ragged_frontend=True`` + a ``valid_mask``) gathers the
real electrode rows *after* the conv stem and *before* the per-electrode token
blocks, runs the blocks on the ``N_valid`` real rows, then scatters the result
back into the padded ``(B, C)`` layout as zeros. Because the token blocks are
per-electrode (no cross-electrode path), every valid row's output is
mathematically identical to the dense path; pad rows (zeros) are masked out of
the cross-attn pool by its ``key_mask`` and dropped from the P1 loss by
``valid_mask``. The win is compute: at BT-Lite c_max=256 with ~120 real
channels the token blocks (the OOM/FLOP hot spot) run on ~half the rows.

Contract pinned here:
  * default OFF on both the dataclass and the model (the dense path is the
    byte-identical pre-#91 path — proven against a captured reference in the
    #91 verify harness; here we pin the default is OFF).
  * all-valid ⇒ ragged == dense BIT-identical (the gather is an exact identity).
  * mixed-valid ⇒ ragged == dense at the valid rows up to float32 batched-kernel
    re-tiling (~1e-6); pad rows do not affect any pooled output.
  * pad-input-invariance: garbage in pad slots cannot change a ragged output.
  * ragged needs ``valid_mask``: ``ragged_frontend=True`` + ``valid_mask=None``
    ⇒ the dense path (ragged is a no-op without a mask to gather by).
  * gradients match dense (the conv stem still trains through the gather).
  * the P1 loss ``valid_mask`` drops pad terms (lower ``n_masked``) and exactly
    partitions the same per-electrode errors — no valid term changes.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import (
    JepaPredictor,
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
)
from speech_decoding.ssl.masked_jepa import p1_frontend_m2_loss


def _kw() -> dict:
    # Same tiny config family as test_v14_encoder._tiny_kwargs (kernel-3 path).
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


def _batch(seed: int = 0):
    """Mixed-valid batch: B=3 electrodes counts [5, 8, 3], C_max=8.

    Pad electrodes get garbage input and zero support; each valid electrode is
    one-hot assigned to a parcel (round-robin over k_parcels).
    """
    kw = _kw()
    g = torch.Generator().manual_seed(seed)
    B, C, K = 3, 8, kw["k_parcels"]
    valid_counts = [5, 8, 3]
    et = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"], generator=g)
    support = torch.zeros(B, C, K)
    valid_mask = torch.zeros(B, C, dtype=torch.bool)
    for b, nv in enumerate(valid_counts):
        valid_mask[b, :nv] = True
        for c in range(nv):
            support[b, c, c % K] = 1.0
        et[b, nv:] = 99.0  # garbage in pad slots
    return kw, et, support, valid_mask


def _grid(model, et, support, valid_mask):
    """Return (F_p, T_p) from the M2 tap so the predictor is configured to match
    whatever the stem produces, without hard-coding kernel arithmetic."""
    with torch.no_grad():
        m2 = model(et, support, valid_mask=valid_mask, return_taps=True)["M2"]
    return m2.shape[2], m2.shape[3]  # (B, C, F_p, T_p, d)


def _token_mask(B, C, F_p, T_p):
    # Deterministic, identical per electrode: mask cells where (f + t) is odd
    # ⇒ every row has both visible context and masked targets.
    fm = torch.zeros(F_p, T_p, dtype=torch.bool)
    for f in range(F_p):
        for t in range(T_p):
            fm[f, t] = (f + t) % 2 == 1
    return fm[None, None].expand(B, C, F_p, T_p).contiguous()


# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #
def test_ragged_default_off_dataclass_and_model() -> None:
    req = {"n_freq_bins": 6, "n_time_bins": 4, "k_parcels": 6,
           "patch_kernel_freq": 3}
    assert V14ParcelPerceiver(**req).ragged_frontend is False
    assert V14ParcelPerceiverModel(**_kw()).ragged_frontend is False
    # build() threads the flag through to the inner encoder model.
    cfg = V14ParcelPerceiver(**req, ragged_frontend=True)
    assert cfg.build(n_outputs=2).encoder.ragged_frontend is True


# --------------------------------------------------------------------------- #
# all-valid: the gather is an exact identity ⇒ BIT-identical
# --------------------------------------------------------------------------- #
def test_ragged_all_valid_bit_identical() -> None:
    kw, et, support, _ = _batch()
    # All electrodes valid ⇒ honor the support/valid contract: give EVERY
    # electrode (incl. the slots _batch left empty for the mixed-valid case) a
    # one-hot parcel, else the L2 alignment guard correctly rejects a valid-but-
    # empty-support row (electrode_desync_damage_2026_06_09.md).
    B, C, K = support.shape
    support = torch.zeros(B, C, K)
    for b in range(B):
        for c in range(C):
            support[b, c, c % K] = 1.0
    all_valid = torch.ones(et.shape[0], et.shape[1], dtype=torch.bool)
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    with torch.no_grad():
        model.ragged_frontend = False
        d = model(et, support, valid_mask=all_valid, return_taps=True)
        model.ragged_frontend = True
        r = model(et, support, valid_mask=all_valid, return_taps=True)
    assert torch.equal(r["M2"], d["M2"]), "all-valid M2 must be bit-identical"
    assert torch.equal(r["M4"], d["M4"]), "all-valid M4 must be bit-identical"


# --------------------------------------------------------------------------- #
# mixed-valid: valid-row parity up to float32 kernel re-tiling
# --------------------------------------------------------------------------- #
def test_ragged_matches_dense_at_valid_rows() -> None:
    kw, et, support, valid_mask = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    with torch.no_grad():
        model.ragged_frontend = False
        d = model(et, support, valid_mask=valid_mask, return_taps=True)
        d_full = model(et, support, valid_mask=valid_mask)
        model.ragged_frontend = True
        r = model(et, support, valid_mask=valid_mask, return_taps=True)
        r_full = model(et, support, valid_mask=valid_mask)
    # Pooled outputs (M4, full) are identical up to kernel rounding: pad rows are
    # masked from the pool either way; only the valid token-block rows feed it.
    torch.testing.assert_close(r_full, d_full, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(r["M4"], d["M4"], atol=1e-5, rtol=1e-5)
    # M2 valid rows match; pad rows are unused (ragged emits frontend_ln(0)).
    vm = valid_mask
    torch.testing.assert_close(r["M2"][vm], d["M2"][vm], atol=1e-5, rtol=1e-5)


def test_ragged_pad_rows_do_not_affect_pool() -> None:
    """Perturbing pad-electrode INPUT cannot change any ragged pooled output."""
    kw, et, support, valid_mask = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    model.ragged_frontend = True
    et2 = et.clone()
    et2[~valid_mask] = -123.0  # different garbage in pad slots
    with torch.no_grad():
        a = model(et, support, valid_mask=valid_mask)
        b = model(et2, support, valid_mask=valid_mask)
    assert torch.equal(a, b), "ragged output must be invariant to pad input"


def test_ragged_all_pad_batch_does_not_crash() -> None:
    """Whole-batch all-pad (N_valid==0): the gather is empty. The ragged path
    must not crash (the dense path tolerates it) and must match dense's pooled
    no-coverage output. Unreachable in normal loading, but a robustness guard."""
    kw, et, support, _ = _batch()
    none_valid = torch.zeros(et.shape[0], et.shape[1], dtype=torch.bool)
    support = torch.zeros_like(support)  # no coverage anywhere
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    with torch.no_grad():
        model.ragged_frontend = False
        d = model(et, support, valid_mask=none_valid)
        model.ragged_frontend = True
        r = model(et, support, valid_mask=none_valid)
    assert torch.isfinite(r).all(), "ragged all-pad output must be finite"
    torch.testing.assert_close(r, d, atol=1e-5, rtol=1e-5)


def test_ragged_without_valid_mask_is_dense() -> None:
    """ragged_frontend=True but no valid_mask ⇒ nothing to gather by ⇒ dense."""
    kw, et, support, _ = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    with torch.no_grad():
        model.ragged_frontend = False
        d = model(et, support)
        model.ragged_frontend = True
        r = model(et, support)  # valid_mask=None
    assert torch.equal(r, d), "ragged w/o valid_mask must equal the dense path"


# --------------------------------------------------------------------------- #
# gradients: ragged is differentiable and matches dense; stem still trains
# --------------------------------------------------------------------------- #
def _p1_grads(ragged: bool):
    kw, et, support, valid_mask = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).train()
    model.ragged_frontend = ragged
    F_p, T_p = _grid(model, et, support, valid_mask)
    torch.manual_seed(11)
    pred = JepaPredictor(d_model=kw["d_model"], n_identity=F_p, depth=2,
                         max_time_patches=T_p)
    tok = _token_mask(et.shape[0], et.shape[1], F_p, T_p)
    with torch.no_grad():
        teacher = model(et, support, valid_mask=valid_mask, return_taps=True)["M2"]
    student = model(et, support, valid_mask=valid_mask, return_taps=True,
                    m2_only=True, token_mask=tok)["M2"]
    # Apples-to-apples: BOTH paths pass valid_mask so only the front-end differs.
    loss = p1_frontend_m2_loss(predictor=pred, student_m2=student,
                               teacher_m2=teacher, token_mask=tok,
                               loss_form="l1", valid_mask=valid_mask).total
    model.zero_grad()
    loss.backward()
    grads = {n: p.grad.detach().clone()
             for n, p in model.named_parameters() if p.grad is not None}
    return float(loss.detach()), grads


def test_ragged_gradients_match_dense() -> None:
    lD, gD = _p1_grads(ragged=False)
    lR, gR = _p1_grads(ragged=True)
    assert abs(lD - lR) < 1e-5, f"loss mismatch dense={lD} ragged={lR}"
    keys = set(gD) & set(gR)
    assert keys, "no grad-bearing params"
    for k in keys:
        torch.testing.assert_close(gR[k], gD[k], atol=1e-5, rtol=1e-4,
                                   msg=f"grad mismatch @ {k}")
    # The conv stem (computed dense, gathered after) must still receive gradient
    # — valid rows backprop through index_copy → index_select → stem.
    stem = [k for k in keys if "stem" in k]
    assert stem, "stem params must carry gradient"
    assert max(gR[k].norm().item() for k in stem) > 0.0


# --------------------------------------------------------------------------- #
# P1 loss valid_mask: drops pad terms, exact partition of the same errors
# --------------------------------------------------------------------------- #
def test_p1_loss_valid_mask_partitions_exactly() -> None:
    kw, et, support, valid_mask = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    F_p, T_p = _grid(model, et, support, valid_mask)
    torch.manual_seed(11)
    pred = JepaPredictor(d_model=kw["d_model"], n_identity=F_p, depth=2,
                         max_time_patches=T_p)
    tok = _token_mask(et.shape[0], et.shape[1], F_p, T_p)
    with torch.no_grad():
        teacher = model(et, support, valid_mask=valid_mask, return_taps=True)["M2"]
        student = model(et, support, valid_mask=valid_mask, return_taps=True,
                        m2_only=True, token_mask=tok)["M2"]

        def run(vm):
            return p1_frontend_m2_loss(predictor=pred, student_m2=student,
                                       teacher_m2=teacher, token_mask=tok,
                                       loss_form="l1", valid_mask=vm)

        b_all = run(None)              # every B·C row (incl. pad)
        b_val = run(valid_mask)        # real electrodes only
        b_pad = run(~valid_mask)       # pad electrodes only

    assert b_val.n_masked < b_all.n_masked, "valid_mask must drop pad terms"
    assert b_val.n_masked + b_pad.n_masked == b_all.n_masked
    # _l1_or_zero is a plain mean ⇒ the all-rows mean is the n-weighted average
    # of the valid and pad partitions. Equality proves no valid term changed.
    recombined = (b_val.total * b_val.n_masked + b_pad.total * b_pad.n_masked)
    torch.testing.assert_close(
        recombined, b_all.total * b_all.n_masked, atol=1e-5, rtol=1e-5)
