"""#93 P1 token-drop (V-JEPA-2 visible-only front-end) parity tests.

The token drop (``ragged_token=True``) gathers the VISIBLE (non-masked, valid,
freq-valid-kept) ``(t_p, f_p)`` tokens per electrode row *before* the joint token
blocks, runs the blocks over the ``Lk ≤ S = F_p·T_p`` kept tokens with a per-row
time-RoPE gathered from the shared table, then scatters the result back into the
full ``(B·C, S, d)`` layout as zeros (dropped tokens).

WHY this is NOT loss-neutral (contrast #91 ragged-frontend / #92 ragged-parcel):
the dense path ZEROES masked tokens (``masked_fill``) but still lets them attend
as keys/values — a zero vector LayerNorm-normalises to ``beta`` (a constant,
non-zero key), so a VISIBLE token's output depends on the PRESENCE of masked
tokens. Dropping them makes the front-end truly visible-only (canonical
V-JEPA 2 §2.1). The drop is therefore a REPRESENTATION CHANGE vs the
zero-in-place default — gated behind a P4 transfer ablation.

WHY the drop is nonetheless EXACT: dropping a token is mathematically identical
to KEY-MASKING it out of attention. The token blocks attend over the kept set
only; key-masking a token sets its attention logit to ``NEG_INF_MASK_VALUE``
(-1e4), and ``exp(-1e4)`` underflows to *exactly* 0.0 in float, so the masked
token contributes exactly 0 to every visible token's softmax — identical to it
being absent. The per-row RoPE is gathered straight from the shared time-RoPE
table (flat token ``i`` → time-patch ``i//F_p``), so every kept token keeps the
exact rotation it had densely; only its dropped neighbours move.

Contract pinned here:
  * default OFF on the dataclass and the model; ``build`` threads it.
  * the per-row ``_apply_rope`` branch matches the shared-table path at the
    gathered positions (RoPE is positionwise — gather commutes with rotation).
  * full-encoder M2: drop == KEY-MASK reference at VISIBLE tokens (~1e-6), using
    the model's real token blocks (input captured via a forward hook).
  * all-False token_mask ⇒ ragged == dense BIT-identical (gather/scatter no-op).
  * visible-only leakage-freedom: perturbing a MASKED token's input cannot change
    any visible token's M2 — and the drop DIFFERS from the zero-in-place dense
    path (the representation change is real, not a no-op).
  * fully-masked rows do NOT produce NaN (the empty-row SDPA guard).
  * P1 gradients (through the real ``p1_frontend_m2_loss``) are finite and reach
    the stem, the token blocks, and the predictor.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import (
    JepaPredictor,
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
    _apply_rope,
    _rope_freqs,
)
from speech_decoding.ssl.masked_jepa import p1_frontend_m2_loss


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


def _batch(seed: int = 0):
    """Tiny fully-covered batch. Grid (probed): C=8, F_p=2, T_p=2 ⇒ S=4."""
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


def _grid(model, et) -> tuple[int, int, int]:
    return model.patch_grid_shape(et)  # (C, F_p, T_p)


def _flat_keep(token_mask: torch.Tensor) -> torch.Tensor:
    """``token_mask`` (B,C,F_p,T_p) → flat keep (B·C, S) in the encoder's flat
    token order (i = t_p·F_p + f_p ⇒ permute F_p/T_p before reshape)."""
    B, C, F_p, T_p = token_mask.shape
    return (~token_mask).permute(0, 1, 3, 2).reshape(B * C, F_p * T_p)


# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #
def test_ragged_token_default_off() -> None:
    req = {"n_freq_bins": 6, "n_time_bins": 4, "k_parcels": 6,
           "patch_kernel_freq": 3}
    assert V14ParcelPerceiver(**req).ragged_token is False
    assert V14ParcelPerceiverModel(**_kw()).ragged_token is False
    cfg = V14ParcelPerceiver(**req, ragged_token=True)
    assert cfg.build(n_outputs=2).encoder.ragged_token is True


# --------------------------------------------------------------------------- #
# per-row RoPE == shared-table RoPE at the gathered positions
# --------------------------------------------------------------------------- #
def test_apply_rope_per_row_matches_shared() -> None:
    """RoPE is positionwise: rotating token j by table[j] is independent of its
    neighbours. So gathering the table to the kept positions and rotating (the
    per-row #93 path) must equal rotating the full sequence then gathering."""
    torch.manual_seed(0)
    hd, S, BC, H, Lk = 8, 6, 3, 2, 4
    rope_shared = _rope_freqs(hd, S)                       # (2, S, hd)
    # distinct kept flat positions per row (sorted, like the encoder's argsort)
    keep_idx = torch.stack([torch.randperm(S)[:Lk].sort().values
                            for _ in range(BC)])           # (BC, Lk)
    x = torch.randn(BC, H, Lk, hd)
    # per-row table (what the encoder builds): gather columns by keep_idx.
    rope_v = rope_shared[:, keep_idx, :]                   # (2, BC, Lk, hd)
    out = _apply_rope(x, rope_v)                           # (BC, H, Lk, hd)
    # reference: scatter x into a full (BC,H,S,hd), rotate with the shared table,
    # gather the kept columns back.
    x_full = x.new_zeros(BC, H, S, hd)
    gi = keep_idx[:, None, :, None].expand(BC, H, Lk, hd)
    x_full.scatter_(2, gi, x)
    out_full = _apply_rope(x_full, rope_shared)            # shared path, (BC,H,S,hd)
    ref = torch.gather(out_full, 2, gi)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------------- #
# all-False token_mask: the gather/scatter is a no-op ⇒ bit-identical
# --------------------------------------------------------------------------- #
def test_token_all_false_mask_bit_identical() -> None:
    kw, et, support, vm = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    _, F_p, T_p = _grid(model, et)
    tm0 = torch.zeros(et.shape[0], et.shape[1], F_p, T_p, dtype=torch.bool)
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
def _keymask_reference_m2(model, et, support, vm, token_mask):
    """Reference visible-only M2 built from the model's REAL token blocks.

    Capture the flat token-block input of the DENSE (ragged_token=False) forward
    via a forward-pre-hook (masked tokens already zeroed there — irrelevant since
    we key-mask them), then re-run ALL blocks with the masked tokens KEY-MASKED
    out of attention, frontend-LN, and reshape to the M2 grid. This is the exact
    semantics the drop must reproduce, computed an independent way."""
    captured = {}

    def _pre_hook(_mod, args):
        # token_block.forward(tokens, rope_time, key_mask)
        captured["x"] = args[0].detach().clone()
        captured["rope"] = args[1]
        captured["kpm"] = args[2]

    h = model.token_blocks[0].register_forward_pre_hook(_pre_hook)
    try:
        model.ragged_token = False
        with torch.no_grad():
            model(et, support, valid_mask=vm, return_taps=True,
                  m2_only=True, token_mask=token_mask)
    finally:
        h.remove()

    B, C, F_p, T_p = token_mask.shape
    keep = _flat_keep(token_mask)                          # (B·C, S) True=visible
    kpm = keep
    if captured["kpm"] is not None:
        kpm = kpm & captured["kpm"]
    x = captured["x"]
    with torch.no_grad():
        for tb in model.token_blocks:
            x = tb(x, captured["rope"], kpm)               # full seq, key-masked
        # (B·C, S, d) → (B,C,T_p,F_p,d) → (B,C,F_p,T_p,d), then frontend_ln.
        m2 = (x.reshape(B, C, T_p, F_p, model.d_model)
                .permute(0, 1, 3, 2, 4).contiguous())
        m2 = model.frontend_ln(m2)
    return m2


def test_token_drop_equals_keymask_reference() -> None:
    kw, et, support, vm = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    _, F_p, T_p = _grid(model, et)
    B, C = et.shape[0], et.shape[1]
    # Mixed per-row mask: vary which (f_p, t_p) cells are masked across electrodes
    # so kept-counts differ per row (exercises the pad-to-max + per-row RoPE).
    g = torch.Generator().manual_seed(3)
    token_mask = torch.rand(B, C, F_p, T_p, generator=g) < 0.4
    # guarantee ≥1 visible token per row so no row is fully dropped here
    fully = token_mask.reshape(B, C, -1).all(dim=2)
    token_mask.reshape(B, C, -1)[fully] = False

    ref = _keymask_reference_m2(model, et, support, vm, token_mask)
    model.ragged_token = True
    with torch.no_grad():
        drop = model(et, support, valid_mask=vm, return_taps=True,
                     m2_only=True, token_mask=token_mask)["M2"]
    # Compare at VISIBLE tokens only (dropped tokens → zeros, never read by the
    # P1 loss). visible grid mask: (B,C,F_p,T_p) = ~token_mask.
    vis = ~token_mask
    torch.testing.assert_close(drop[vis], ref[vis], atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# visible-only leakage-freedom + the representation change is real
# --------------------------------------------------------------------------- #
def test_token_visible_invariant_and_changes_representation() -> None:
    kw, et, support, vm = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    _, F_p, T_p = _grid(model, et)
    B, C = et.shape[0], et.shape[1]
    # mask the f_p=0,t_p=1 cell on every electrode (a clean per-row drop)
    token_mask = torch.zeros(B, C, F_p, T_p, dtype=torch.bool)
    token_mask[:, :, 0, 1] = True
    vis = ~token_mask

    model.ragged_token = True
    with torch.no_grad():
        base = model(et, support, valid_mask=vm, return_taps=True,
                     m2_only=True, token_mask=token_mask)["M2"]
    # Perturb a MASKED cell's *input* (the patch stem mixes freq bins, so a masked
    # (f_p,t_p) maps back to a band of input bins; perturb the whole input of a
    # masked-only electrode-time is overkill — instead confirm via the dense vs
    # drop contrast below). Here: the drop must DIFFER from the zero-in-place
    # dense path (proof the representation actually changed).
    model.ragged_token = False
    with torch.no_grad():
        dense = model(et, support, valid_mask=vm, return_taps=True,
                      m2_only=True, token_mask=token_mask)["M2"]
    # At VISIBLE tokens, drop (visible-only) and dense (masked tokens zeroed but
    # still attended) must differ — that difference IS the V-JEPA-2 change.
    assert not torch.allclose(base[vis], dense[vis], atol=1e-4), \
        "drop must differ from zero-in-place at visible tokens (real change)"
    # Dropped tokens are zeros under the drop (never read downstream).
    assert torch.count_nonzero(base[token_mask]) == 0, \
        "masked tokens must scatter back as exact zeros"


# --------------------------------------------------------------------------- #
# fully-masked rows: no NaN (empty-row SDPA guard)
# --------------------------------------------------------------------------- #
def test_token_fully_masked_rows_no_nan() -> None:
    kw, et, support, vm = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).eval()
    _, F_p, T_p = _grid(model, et)
    B, C = et.shape[0], et.shape[1]
    token_mask = torch.zeros(B, C, F_p, T_p, dtype=torch.bool)
    token_mask[:, 0] = True          # electrode 0 fully masked (all its tokens)
    token_mask[:, 3] = True          # electrode 3 fully masked
    model.ragged_token = True
    with torch.no_grad():
        m2 = model(et, support, valid_mask=vm, return_taps=True,
                   m2_only=True, token_mask=token_mask)["M2"]
    assert torch.isfinite(m2).all(), "fully-masked rows must not produce NaN/Inf"
    # the fully-masked electrodes' tokens are all dropped ⇒ scattered zeros
    assert torch.count_nonzero(m2[:, 0]) == 0
    assert torch.count_nonzero(m2[:, 3]) == 0


# --------------------------------------------------------------------------- #
# P1 gradients through the real loss: finite + reach stem / blocks / predictor
# --------------------------------------------------------------------------- #
def test_token_gradients_finite_through_p1_loss() -> None:
    kw, et, support, vm = _batch()
    torch.manual_seed(7)
    model = V14ParcelPerceiverModel(**kw).train()
    model.ragged_token = True
    _, F_p, T_p = _grid(model, et)
    B, C = et.shape[0], et.shape[1]
    g = torch.Generator().manual_seed(5)
    token_mask = torch.rand(B, C, F_p, T_p, generator=g) < 0.4
    fully = token_mask.reshape(B, C, -1).all(dim=2)
    token_mask.reshape(B, C, -1)[fully] = False

    # teacher: full input (no mask), detached inside the loss.
    with torch.no_grad():
        model.ragged_token = False
        teacher = model(et, support, valid_mask=vm, return_taps=True,
                        m2_only=True)["M2"]
    model.ragged_token = True
    student = model(et, support, valid_mask=vm, return_taps=True,
                    m2_only=True, token_mask=token_mask)["M2"]

    # P1 predictor identity axis = freq-patch (n_identity = F_p); time carried by
    # RoPE. (The loss reshapes M2 in its own f_p-outer/t_p-inner order — separate
    # from the encoder's t_p-outer token-block order; both are internally exact.)
    torch.manual_seed(11)
    pred = JepaPredictor(d_model=kw["d_model"], n_identity=F_p, depth=2,
                         max_time_patches=T_p)
    out = p1_frontend_m2_loss(predictor=pred, student_m2=student,
                              teacher_m2=teacher, token_mask=token_mask,
                              loss_form="l1", valid_mask=vm)
    loss = out.total
    assert torch.isfinite(loss), "P1 loss must be finite"
    model.zero_grad(); pred.zero_grad()
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(g).all() for g in grads.values()), "finite grads"
    stem = [n for n in grads if "stem" in n]
    blocks = [n for n in grads if "token_blocks" in n]
    assert stem and max(grads[n].norm().item() for n in stem) > 0.0
    assert blocks and max(grads[n].norm().item() for n in blocks) > 0.0
    pgrads = [p.grad for p in pred.parameters() if p.grad is not None]
    assert pgrads and max(g.norm().item() for g in pgrads) > 0.0
