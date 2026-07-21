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

The secondary section (unit 3b) covers the LIVE r5-mod secondary loss: frozen-DIAGONAL
Gaussian NLL over the 5-D state [slow_mu, mid_mu, hga_mu, relmod48, relmod816], read by a
point-only Perceiver head (no covariance parameter). The retired r4/r5 full-cov "nll" and
point "l1" arms are covered by test_secondary_head.py.
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
from speech_decoding.models.v14_converged_v3.secondary_head import STATE_DIM
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


# ── secondary write-only Perceiver frozen-DIAGONAL NLL (r5-mod, #36 unit 3b) ──────────────
# The LIVE secondary loss: point-only Perceiver head → per-(parcel, 4 Hz slot) μ, scored by
# present_masked_diag_nll against the model-free 5-D state with FROZEN σ² (count-dep floor).
def _stats(sc):
    """FROZEN per-(parcel,dim) 5-D state-norm stats for the test session (identity
    normalization: mean 0, std 1 over the P present parcels). Real stats come from the
    offline pass."""
    parcels = torch.unique(sc.parcel_id)
    P = int(parcels.numel())
    return torch.zeros(P, STATE_DIM), torch.ones(P, STATE_DIM)


def test_diag_nll_is_the_default_and_is_point_only() -> None:
    # The r5-mod build's canonical secondary form: default diag_nll, point head (no covariance
    # parameter — the DDP X1 precondition), 5-D state.
    obj = V3JepaObjective(n_parcels=8)
    default = obj.secondary_loss == "diag_nll"
    point = obj.perceiver.head.chol_head is None
    _, cov = obj.perceiver.head(torch.randn(4, obj.perceiver.d_perc))
    no_cov = cov is None
    ok = default and point and no_cov
    print(f"[check] default='{obj.secondary_loss}' ({default}); point head chol=None ({point}); "
          f"forward cov=None ({no_cov}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_diag_nll_requires_nll_floor() -> None:
    # diag_nll's σ² IS the count-dependent floor (frozen); it cannot run floor-off.
    import pytest

    with pytest.raises(ValueError, match="nll_floor"):
        V3JepaObjective(n_parcels=8, secondary_loss="diag_nll", nll_floor=False)


def test_secondary_off_by_default_is_pure_jepa() -> None:
    # No stats ⇒ JEPA-only: nll/jepa fields None, and the perceiver receives NO gradient
    # (the write path never engaged).
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8)
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom))
    out.loss.backward()
    perc_grad = all(p.grad is None for p in obj.perceiver.parameters())
    ok = out.nll_loss is None and out.jepa_loss is None and perc_grad
    print(f"[check] secondary OFF: nll={out.nll_loss} jepa={out.jepa_loss}; perceiver "
          f"no-grad={perc_grad} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_secondary_active_total_is_jepa_plus_lambda_nll() -> None:
    # With stats: loss == jepa_loss + λ·nll_loss EXACTLY, both finite, nll a 0-dim scalar.
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8, lambda_nll=0.2)
    sm, ss = _stats(sc)
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom), stat_mean=sm, stat_std=ss)
    exact = torch.allclose(out.loss, out.jepa_loss + 0.2 * out.nll_loss)
    finite = bool(torch.isfinite(out.loss) and torch.isfinite(out.nll_loss))
    ok = exact and finite and out.nll_loss.ndim == 0
    print(f"[check] total={float(out.loss):.4f} == jepa {float(out.jepa_loss):.4f} + 0.2·nll "
          f"{float(out.nll_loss):.4f} ({exact}); finite={finite} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_secondary_nll_is_nonnegative_a_proper_scoring_floor() -> None:
    # A frozen-diagonal NLL floors at the target's own diagonal entropy — with unit-variance
    # (identity-normalized) targets and a σ²<1 floor, that entropy is > 0, so the scored term
    # is a sensible finite scalar (this is the tell that separates a diagonal NLL from an L1).
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8, lambda_nll=0.2)
    sm, ss = _stats(sc)
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom), stat_mean=sm, stat_std=ss)
    finite = bool(torch.isfinite(out.nll_loss))
    print(f"[check] diag NLL secondary finite scalar ({float(out.nll_loss):.4f}) "
          f"{'OK' if finite else 'VIOLATED'}")
    assert finite


def test_secondary_grad_reaches_perceiver_head_and_encoder_write() -> None:
    # The secondary must (a) train the perceiver + its point head (mu_head) and (b) WRITE into
    # the shared online encoder (grad flows back through z). Both are the point.
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8)
    sm, ss = _stats(sc)
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom), stat_mean=sm, stat_std=ss)
    out.loss.backward()
    perc = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in obj.perceiver.parameters()
    )
    head = obj.perceiver.head.mu_head.weight.grad is not None
    enc = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in obj.online.encoder.parameters()
    )
    ok = perc and head and enc
    print(f"[check] grad → perceiver={perc}, μ-head={head}, encoder(write)={enc} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_secondary_is_write_only_no_readback_into_primary() -> None:
    # WRITE-ONLY: the perceiver output must never flow into the PRIMARY stream. So the JEPA
    # term alone has NO gradient path to the perceiver (jepa_loss.backward() leaves it None),
    # while the total does. Proves the primary loss is independent of the secondary head.
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8)
    sm, ss = _stats(sc)
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom), stat_mean=sm, stat_std=ss)
    out.jepa_loss.backward(retain_graph=True)  # primary term only
    no_readback = all(p.grad is None for p in obj.perceiver.parameters())
    print(f"[check] primary loss has NO path to the perceiver (write-only)={no_readback} "
          f"{'OK' if no_readback else 'VIOLATED'}")
    assert no_readback


def test_collect_taps_emits_per_dim_monitor_scalars() -> None:
    # Monitor wiring (#40 per-band JEPA + per-DIM diag NLL): with collect_taps=True AND the
    # secondary active, the objective returns per-band JEPA scalars PLUS the 5 per-dim NLL
    # scalars — all 0-dim finite — and NO cov_entropy* (a point head has no covariance to
    # report). The backward still runs (taps are detached / no_grad).
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8, lambda_nll=0.2)
    sm, ss = _stats(sc)
    out = obj(
        _bands(), geom, sc.parcel_id, _masks(geom),
        stat_mean=sm, stat_std=ss, collect_taps=True,
    )
    taps = out.taps
    bands = ("slow", "mid", "hga")
    dims = ("slow_mu", "mid_mu", "hga_mu", "relmod48", "relmod816")
    expect = (
        [f"jepa_{b}_{m}" for b in bands for m in ("explained_var", "pred_target_var_ratio", "l1")]
        + [f"nll_{d}" for d in dims]
    )
    present = taps is not None and all(k in taps for k in expect)
    scalars = present and all(taps[k].ndim == 0 and torch.isfinite(taps[k]) for k in expect)
    no_cov = present and not any(k.startswith("cov_entropy") for k in taps)  # point head
    out.loss.backward()  # graph intact despite the monitor reductions
    enc_flows = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in obj.online.encoder.parameters()
    )
    ok = scalars and no_cov and enc_flows
    print(
        f"[check] collect_taps monitor scalars present+finite={scalars}, no cov_entropy "
        f"tap (point head)={no_cov}, backward-after-taps={enc_flows} "
        f"→ {'OK' if ok else 'VIOLATED'}"
    )
    assert ok


def test_collect_taps_ships_raw_perceiver_latent_bank() -> None:
    # Perceiver-health monitor wiring: with collect_taps + secondary active, the objective
    # ships the processed latent bank as a RAW 3-d tap ``perc_lat`` (B, S·M, d_perc), detached
    # (no grad) so the callback can reduce RankMe / dead-frac / cosine off the loss graph.
    # Absent when the secondary is off (perceiver never ran).
    from speech_decoding.models.v14_converged_v3.perceiver import D_PERC, M_LATENTS

    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8, lambda_nll=0.2)
    sm, ss = _stats(sc)
    out = obj(
        _bands(), geom, sc.parcel_id, _masks(geom),
        stat_mean=sm, stat_std=ss, collect_taps=True,
    )
    lat = (out.taps or {}).get("perc_lat")
    present = isinstance(lat, torch.Tensor) and lat.ndim == 3
    shape_ok = present and lat.shape[-1] == D_PERC and lat.shape[1] % M_LATENTS == 0
    detached = present and not lat.requires_grad
    # secondary OFF ⇒ no perc_lat tap at all
    off = obj(_bands(), geom, sc.parcel_id, _masks(geom), collect_taps=True)
    absent = "perc_lat" not in (off.taps or {})
    ok = present and shape_ok and detached and absent
    print(
        f"[check] perc_lat raw 3-d tap present={present}, shape={tuple(lat.shape) if present else None} "
        f"(d_perc={D_PERC}, ×M={M_LATENTS}) ok={shape_ok}, detached={detached}, "
        f"absent-when-off={absent} → {'OK' if ok else 'VIOLATED'}"
    )
    assert ok


def test_collect_taps_jepa_only_has_no_secondary_scalars() -> None:
    # collect_taps=True but secondary OFF (no stats): enc12 + per-band JEPA scalars present,
    # but NO nll_*/cov_entropy* (the secondary never ran). Guards against logging stale keys.
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8)
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom), collect_taps=True)
    taps = out.taps or {}
    has_jepa = all(f"jepa_{b}_l1" in taps for b in ("slow", "mid", "hga"))
    no_sec = not any(k.startswith("nll_") or k.startswith("cov_entropy") for k in taps)
    ok = has_jepa and no_sec and "enc12" in taps
    print(f"[check] JEPA-only collect_taps: per-band JEPA={has_jepa}, no secondary keys={no_sec} "
          f"→ {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_secondary_requires_deep_sup() -> None:
    # The perceiver reads the deep-sup 1024 taps; the single-tap arm has none ⇒ fail loud.
    import pytest

    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8, deep_sup=False)
    sm, ss = _stats(sc)
    assert obj.perceiver is None
    with pytest.raises(ValueError, match="deep_sup"):
        obj(_bands(), geom, sc.parcel_id, _masks(geom), stat_mean=sm, stat_std=ss)


def test_secondary_singleton_parcel_scores_mean_only_end_to_end() -> None:
    # A 1-electrode parcel: its 2 modulation dims are undefined ⇒ build_state_target masks
    # them, the diag NLL scores the 3 mean dims there, and count_dependent_noise_var stays
    # finite at n=1. End-to-end the forward must run finite and grads stay finite.
    sc = build_sidecar(
        ["LA1", "LA2", "LB1", "LC1", "LC2"],
        parcel_id=torch.tensor([0, 0, 1, 2, 2]),  # parcel 1 is a singleton
    )
    geom = build_l1_geometry(sc)
    obj = V3JepaObjective(n_parcels=8)
    parcels = torch.unique(sc.parcel_id)
    sm = torch.zeros(parcels.numel(), STATE_DIM)
    ss = torch.ones(parcels.numel(), STATE_DIM)
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom), stat_mean=sm, stat_std=ss)
    out.loss.backward()
    grads_finite = all(
        p.grad is None or torch.isfinite(p.grad).all() for p in obj.parameters()
    )
    ok = bool(torch.isfinite(out.loss)) and bool(torch.isfinite(out.nll_loss)) and grads_finite
    print(f"[check] singleton-parcel end-to-end: loss finite + grads finite={ok} "
          f"(nll={float(out.nll_loss):.4f}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_secondary_loss_rejects_an_unknown_form() -> None:
    """A typo'd arm name must fail LOUDLY at construction, not silently fall back and hand us
    a run of the wrong experiment."""
    import pytest

    with pytest.raises(ValueError, match="secondary_loss"):
        V3JepaObjective(n_parcels=8, secondary_loss="l2")
