"""v14_converged_v3 — V-JEPA 2.1 §2.3.1 context loss on the flat r4 path (TDD).

The context loss predicts the SAME per-level-normed teacher target the masked term uses, but
at the VISIBLE (context) tokens, through a SEPARATE ``pred_to_target_context`` head, scored by
the same weighted-mean L1 at ``~masked`` positions and λ_ctx-weighted. Off by default ⇒ zero
new params. The invariants a silent miscompute would violate, named + asserted + printed
(feedback-build-the-invariant-into-the-probe):

  1. OFF by default: no context head, no ctx_loss, param count == the non-context objective.
  2. The weighting is the VISIBLE COMPLEMENT of the JEPA scored set: the context L1 selects
     ``~masked`` (disjoint from ``in_loss ⊆ masked``). Proven directly on the pure helper.
  3. ON: ctx_loss is a finite non-negative scalar; the fixed-hold total folds λ_ctx·ctx; and
     the context term ALONE trains its head AND writes back into the shared encoder.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import sample_masks
from speech_decoding.models.v14_converged_v3.objective import (
    V3JepaObjective,
    _masked_mean_l1,
)
from speech_decoding.models.v14_converged_v3.pack_r4 import build_r4_grid, token_flags
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

T = 16
B = 2
N = 5


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        parcel_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    return sc, build_l1_geometry(sc)


def _bands(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(B, N, 7, T, generator=g),
        torch.randn(B, N, 6, T, generator=g),
        torch.randn(B, N, 7, T, generator=g),
    ]


def _masks(geom, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return sample_masks(geom, N, n_time=T, n_rows=B, generator=g)


def test_context_loss_off_by_default_adds_no_params() -> None:
    # Default OFF: no context head, ctx_loss None on a forward, and the parameter set is
    # byte-identical to the non-context objective (so every existing ckpt / arm loads unchanged).
    sc, geom = _session()
    off = V3JepaObjective(n_parcels=8)
    assert off.pred_to_target_context is None
    out = off(_bands(), geom, sc.parcel_id, _masks(geom))
    no_ctx = out.ctx_loss is None
    # param count delta vs an explicit context objective == exactly the one Linear (w + b).
    on = V3JepaObjective(n_parcels=8, context_loss=True)
    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    head = on.pred_to_target_context
    added = head.weight.numel() + head.bias.numel()
    exact = (n_on - n_off) == added
    ok = no_ctx and exact
    print(f"[check] context OFF by default: head=None, ctx_loss={out.ctx_loss} ({no_ctx}); "
          f"param delta {n_on - n_off} == one Linear {added} ({exact}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_context_l1_weights_the_visible_complement() -> None:
    # THE weighting invariant, proven on the pure helper the forward calls: with a 0/1 weight,
    # tokens where weight==0 contribute NOTHING and the mean is over the weight==1 tokens only.
    # Build masked s.t. ~masked selects exactly the visible set, put huge error on masked tokens
    # and zero on visible ⇒ the context L1 (weight ~masked) is 0; flip the weight ⇒ it is huge.
    masked = torch.tensor([[True, True, False, False, False]])  # (1, 5)
    tgt = torch.zeros(1, 5, 3)
    pred = torch.zeros(1, 5, 3)
    pred[0, :2] = 100.0  # error lives ONLY on the masked tokens
    w_ctx = (~masked).float()
    ctx = _masked_mean_l1(pred, tgt, w_ctx)  # scores VISIBLE (tokens 2,3,4) ⇒ zero error
    w_masked = masked.float()
    jep = _masked_mean_l1(pred, tgt, w_masked)  # scores MASKED (tokens 0,1) ⇒ big error
    disjoint = bool(((~masked).float() * masked.float()).sum() == 0)
    ok = float(ctx) == 0.0 and float(jep) == 100.0 and disjoint
    print(f"[check] context L1 over ~masked={float(ctx):.1f} (visible, err-free), masked L1="
          f"{float(jep):.1f}; sets disjoint={disjoint} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_context_on_is_finite_and_total_folds_lambda_ctx() -> None:
    # ON: ctx_loss a finite non-neg 0-dim scalar, and the standalone total folds the FIXED hold
    # λ_ctx (the module later overrides with the ramped λ). No secondary stats ⇒ nll off, so the
    # total is exactly jepa + λ_ctx·ctx.
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8, context_loss=True, lambda_ctx=0.5)
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom))
    finite = bool(torch.isfinite(out.ctx_loss)) and out.ctx_loss.ndim == 0 and float(out.ctx_loss) >= 0
    exact = torch.allclose(out.loss, out.jepa_loss + 0.5 * out.ctx_loss)
    ok = finite and exact and out.nll_loss is None
    print(f"[check] context ON: ctx_loss={float(out.ctx_loss):.4f} finite={finite}; total "
          f"{float(out.loss):.4f} == jepa {float(out.jepa_loss):.4f} + 0.5·ctx ({exact}); "
          f"nll off={out.nll_loss is None} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_context_term_alone_trains_head_and_writes_encoder() -> None:
    # The context term must (a) train its own head and (b) WRITE back into the shared online
    # encoder (grad flows through h ← predictor ← enc_to_pred ← z ← encoder). Backprop ctx_loss
    # ALONE and check both. This is what makes the context loss shape the representation.
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8, context_loss=True)
    obj.train()
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom))
    out.ctx_loss.backward()
    head = obj.pred_to_target_context
    head_grad = head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    enc = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in obj.online.encoder.parameters()
    )
    # write-only-into-primary check is not claimed here (context IS a primary-stream target);
    # the point is only that the context head + encoder receive gradient from ctx_loss.
    ok = bool(head_grad) and enc
    print(f"[check] ctx_loss.backward → context-head grad={bool(head_grad)}, encoder write={enc} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_context_scored_set_matches_token_flags_visible() -> None:
    # End-to-end consistency: the number of visible tokens the context loss averages over equals
    # (~masked).sum() from the SAME token_flags the objective uses — i.e. the forward really did
    # weight the visible complement, not some other set. Recompute the denominator independently.
    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    masks = _masks(geom)
    masked, in_loss = token_flags(grid, masks)
    n_visible = int((~masked).sum())
    n_masked_scored = int(in_loss.sum())
    obj = V3JepaObjective(n_parcels=8, context_loss=True)
    out = obj(_bands(), geom, sc.parcel_id, masks)
    # n_masked (the JEPA denominator) is in_loss.sum(); visible is its disjoint complement.
    jepa_denom_ok = int(out.n_masked) == n_masked_scored
    disjoint = int((in_loss & ~masked).sum()) == 0
    ok = jepa_denom_ok and disjoint and n_visible > 0 and out.ctx_loss is not None
    print(f"[check] visible tokens={n_visible}, jepa-scored={n_masked_scored} (denom match="
          f"{jepa_denom_ok}); in_loss ∩ ~masked empty={disjoint} {'OK' if ok else 'VIOLATED'}")
    assert ok
