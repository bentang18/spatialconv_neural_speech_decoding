"""v14_converged_v3 r4/r5-mod — JEPA objective on the flat per-band path (TDD, #36 unit 3a).

The objective composes the tested primitives (stem → pack_band_tokens → visible-pack
encoder / full-grid teacher / full-grid predictor → margin-gated L1). The invariants that
a silent miscompute would violate, named + asserted + printed
(feedback-build-the-invariant-into-the-probe):

  1. Runs finite; loss is a non-negative scalar; grads reach every TRAINABLE module
     (stem projs, encoder, predictor, enc_to_pred, pred_to_target, mask_token).
  2. The EMA teacher is stop-grad: its params get NO gradient from the loss.
  3. LEAK-SAFETY (the whole point of r4): perturbing the band inputs at MASKED tokens'
     frames leaves the online latent BIT-IDENTICAL — a target cannot leak into a visible
     latent, because masked tokens are physically dropped from the encoder's sequence.
  4. The scored set is the margin gate: n_masked == in_loss.sum(), and in_loss ⊆ masked.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import sample_masks
from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective
from speech_decoding.models.v14_converged_v3.pack_r4 import (
    BAND_STRIDES,
    build_r4_grid,
    build_visible_pack,
    token_flags,
)
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

T = 16  # 32 Hz clock; multiple of SLOW stride 8 ⇒ SLOW 2 / MID 8 / HGA 16 tokens per contact.
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
    # (B, N, F_b, T) with F_b = 7 / 6 / 7 (PER_BAND_SPECS n_bins), on the shared 32 Hz clock.
    return [
        torch.randn(B, N, 7, T, generator=g),
        torch.randn(B, N, 6, T, generator=g),
        torch.randn(B, N, 7, T, generator=g),
    ]


def _masks(geom, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return sample_masks(geom, N, n_time=T, n_rows=B, generator=g)


def test_forward_runs_and_grads_reach_every_trainable_module() -> None:
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8)
    obj.train()
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom))
    loss_val = out.loss.detach()
    out.loss.backward()

    finite = bool(torch.isfinite(loss_val)) and loss_val.ndim == 0 and float(loss_val) >= 0.0
    checks = {
        "stem.projs": obj.online.stem.projs,
        "encoder": obj.online.encoder,
        "predictor": obj.predictor,
        "enc_to_pred": obj.enc_to_pred,
        "pred_to_target": obj.pred_to_target,
    }
    grad_ok = {
        name: all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        for name, m in checks.items()
    }
    mask_grad = obj.mask_token.grad is not None and torch.isfinite(obj.mask_token.grad).all()
    ok = finite and all(grad_ok.values()) and mask_grad
    print(f"[check] loss={float(out.loss):.4f} finite={finite}; grads {grad_ok}; "
          f"mask_token grad={mask_grad} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_teacher_is_stop_grad() -> None:
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8)
    obj.train()
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom))
    out.loss.backward()
    # the EMA teacher is a separate copy (obj.teacher.model); the loss must not touch it.
    teacher_params = list(obj.teacher.model.parameters())
    no_grad = all(p.grad is None for p in teacher_params)
    print(f"[check] teacher ({len(teacher_params)} params) received NO gradient={no_grad} "
          f"{'OK' if no_grad else 'VIOLATED'}")
    assert no_grad


def test_leak_safe_online_latent_ignores_masked_token_inputs() -> None:
    # THE r4 invariant: masked tokens are physically dropped from the encoder, so the online
    # latent must be BIT-IDENTICAL whether or not we corrupt the band inputs that feed the
    # MASKED tokens. Perturb every masked token's own decimated frame (token j of band b
    # reads frame j·stride) and assert the online output does not move.
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8).eval()
    grid = build_r4_grid(geom, n_time=T)
    parcel_packed = sc.parcel_id[grid.contact]
    masks = _masks(geom)
    masked, _ = token_flags(grid, masks)
    pack = build_visible_pack(grid, masked, parcel_packed)

    bands = _bands()
    z1 = obj.online(bands, grid, parcel_packed, pack=pack)

    bands2 = [b.clone() for b in bands]
    for clip in range(B):
        for t in torch.nonzero(masked[clip]).flatten().tolist():
            bnd, c, j = int(grid.band[t]), int(grid.contact[t]), int(grid.bandpos[t])
            bands2[bnd][clip, c, :, j * BAND_STRIDES[bnd]] += 5.0
    z2 = obj.online(bands2, grid, parcel_packed, pack=pack)

    # sanity: the perturbation is real (it DOES change the full-grid teacher, which sees masked).
    full1 = obj.online(bands, grid, parcel_packed)
    full2 = obj.online(bands2, grid, parcel_packed)
    moved_full = not torch.allclose(full1, full2, atol=1e-4)
    leak_safe = torch.allclose(z1, z2, atol=1e-6)
    ok = leak_safe and moved_full
    print(f"[check] online latent invariant to masked-token inputs={leak_safe} (max|Δ|="
          f"{(z1 - z2).abs().max().item():.2e}); full-grid DID move={moved_full} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_scored_set_is_the_margin_gate() -> None:
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8)
    grid = build_r4_grid(geom, n_time=T)
    masks = _masks(geom)
    masked, in_loss = token_flags(grid, masks)
    out = obj(_bands(), geom, sc.parcel_id, masks)
    nm = int(out.n_masked)  # n_masked is now a 0-dim tensor (sync deferred)
    subset = bool(torch.all(in_loss <= masked))
    n_ok = nm == int(in_loss.sum())
    ok = subset and n_ok and nm > 0
    print(f"[check] in_loss ⊆ masked={subset}; n_masked={nm} == Σin_loss="
          f"{int(in_loss.sum())} ({n_ok}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_session_plan_consts_are_session_invariant_and_exact() -> None:
    """The (grid_max_seqlen, m_vis, pack_max_seqlen) the module caches must be (a) IDENTICAL
    across mask realizations — the property that makes caching them sound — and (b) exactly
    what ``model.session_plan`` returns. Exact per-shaft spatial + GLOBAL per-band temporal
    masking makes the per-shaft visible count clip-invariant, so one representative mask
    fixes the shapes for every clip in the session."""
    from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel

    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    parcel_packed = sc.parcel_id[grid.contact]
    seen = set()
    for seed in range(6):
        g = torch.Generator().manual_seed(100 + seed)
        masks = sample_masks(geom, N, n_time=T, n_rows=B, generator=g)
        masked, _ = token_flags(grid, masks)
        pack = build_visible_pack(grid, masked, parcel_packed)
        seen.add((pack.m_vis, pack.max_seqlen))
    invariant = len(seen) == 1
    m_vis, pack_max = next(iter(seen))
    plan = V3ConvergedModel(n_parcels=8).session_plan(geom, sc.parcel_id, T)
    exact = plan == (grid.max_seqlen, m_vis, pack_max)
    ok = invariant and exact
    print(f"[check] shape-consts invariant across 6 mask seeds={invariant} (seen "
          f"{sorted(seen)}); session_plan={plan} == derived "
          f"{(grid.max_seqlen, m_vis, pack_max)} ({exact}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_passed_shape_consts_give_identical_forward() -> None:
    """Passing the cached shape-consts must NOT change the forward — it only removes the
    per-step ``.item()`` host syncs. Same masks + same weights ⇒ identical loss and n_masked
    whether the objective derives the consts (None) or receives them."""
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8)
    grid = build_r4_grid(geom, n_time=T)
    parcel_packed = sc.parcel_id[grid.contact]
    masks = _masks(geom)
    bands = _bands()
    masked, _ = token_flags(grid, masks)
    pack = build_visible_pack(grid, masked, parcel_packed)
    out_eager = obj(bands, geom, sc.parcel_id, masks)
    out_cached = obj(
        bands, geom, sc.parcel_id, masks,
        grid_max_seqlen=grid.max_seqlen, m_vis=pack.m_vis, pack_max_seqlen=pack.max_seqlen,
    )
    same_loss = torch.allclose(out_eager.loss, out_cached.loss, atol=1e-6)
    same_nm = int(out_eager.n_masked) == int(out_cached.n_masked)
    ok = same_loss and same_nm
    print(f"[check] cached-const forward == eager: loss Δ="
          f"{(out_eager.loss - out_cached.loss).abs().item():.2e} ({same_loss}), "
          f"n_masked {int(out_eager.n_masked)}=={int(out_cached.n_masked)} ({same_nm}) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok
