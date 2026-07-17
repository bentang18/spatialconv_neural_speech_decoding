"""v14_converged_v3 r4 — JEPA objective on the flat per-band path (TDD, #36 unit 3a).

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


# ── secondary write-only Perceiver Gaussian-NLL assembly (#36 unit 3b) ───────────────────
def _stats(sc):
    """FROZEN per-(parcel,dim) state-norm stats for the test session (identity normalization:
    mean 0, std 1 over the P present parcels). Real stats come from the offline pass."""
    parcels = torch.unique(sc.parcel_id)
    P = int(parcels.numel())
    return torch.zeros(P, 6), torch.ones(P, 6)


def test_secondary_off_by_default_is_pure_jepa() -> None:
    # No stats ⇒ JEPA-only: nll/jepa fields None, loss == the stats-absent loss, and the
    # perceiver receives NO gradient (write path never engaged).
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
    # With stats: loss == jepa_loss + λ·nll_loss EXACTLY, both terms finite, nll ≥ 0-ish scalar.
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


def test_secondary_nll_grad_reaches_perceiver_and_encoder_write() -> None:
    # The secondary must (a) train the perceiver (its own params + the Gaussian head) and
    # (b) WRITE into the shared online encoder (grad flows back through z). Both are the point.
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
    print(f"[check] grad → perceiver={perc}, gauss-head={head}, encoder(write)={enc} "
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


def test_collect_taps_emits_per_band_monitor_scalars() -> None:
    # Monitor wiring (#40/#41/#42): with collect_taps=True AND the secondary active, the
    # objective returns the enc12 tap PLUS finite per-band JEPA scalars, per-band NLL, and
    # the cov-entropy-vs-floor triple — all 0-dim — and the backward still runs (the taps
    # are detached / no_grad, so they do not consume or extend the loss graph).
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8, lambda_nll=0.2)
    sm, ss = _stats(sc)
    out = obj(
        _bands(), geom, sc.parcel_id, _masks(geom),
        stat_mean=sm, stat_std=ss, collect_taps=True,
    )
    taps = out.taps
    bands = ("slow", "mid", "hga")
    expect = (
        [f"jepa_{b}_{m}" for b in bands for m in ("explained_var", "pred_target_var_ratio", "l1")]
        + [f"nll_{b}" for b in bands]
        + ["cov_entropy", "cov_entropy_floor", "cov_entropy_gap"]
    )
    present = taps is not None and all(k in taps for k in expect)
    scalars = present and all(taps[k].ndim == 0 and torch.isfinite(taps[k]) for k in expect)
    gap_nonneg = present and float(taps["cov_entropy_gap"]) >= -1e-4  # #42 invariant
    out.loss.backward()  # graph intact despite the monitor reductions
    enc_flows = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in obj.online.encoder.parameters()
    )
    ok = scalars and gap_nonneg and enc_flows
    print(
        f"[check] collect_taps monitor scalars present+finite={scalars}, cov gap≥0 "
        f"({float(taps['cov_entropy_gap']):.3f})={gap_nonneg}, backward-after-taps={enc_flows} "
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


def test_secondary_singleton_parcel_scores_mean_marginal_end_to_end() -> None:
    # A 1-electrode parcel: its std is undefined ⇒ build_state_target masks the 3 std dims,
    # the NLL scores the 3-D mean marginal there, and count_dependent_noise_var stays finite
    # at n=1. End-to-end the forward must run finite and grads stay finite (no NaN from the
    # degenerate parcel).
    sc = build_sidecar(
        ["LA1", "LA2", "LB1", "LC1", "LC2"],
        parcel_id=torch.tensor([0, 0, 1, 2, 2]),  # parcel 1 is a singleton
    )
    geom = build_l1_geometry(sc)
    obj = V3JepaObjective(n_parcels=8)
    parcels = torch.unique(sc.parcel_id)
    sm, ss = torch.zeros(parcels.numel(), 6), torch.ones(parcels.numel(), 6)
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom), stat_mean=sm, stat_std=ss)
    out.loss.backward()
    grads_finite = all(
        p.grad is None or torch.isfinite(p.grad).all() for p in obj.parameters()
    )
    ok = bool(torch.isfinite(out.loss)) and bool(torch.isfinite(out.nll_loss)) and grads_finite
    print(f"[check] singleton-parcel end-to-end: loss finite + grads finite={ok} "
          f"(nll={float(out.nll_loss):.4f}) {'OK' if ok else 'VIOLATED'}")
    assert ok


# ---------------------------------------------------------------------------
# r5 Arm 3 — POINT loss (secondary_loss="l1"). L1 not L2 is measured, not assumed:
# M18 (probe_v3_residual_tailweight) reads 5/6 state dims ABOVE the Laplace pole.
# ---------------------------------------------------------------------------


def test_arm3_point_head_has_no_covariance_parameters() -> None:
    """THE DDP INVARIANT — this is the r3 killer, so it is asserted, not assumed.

    A point loss touches no covariance parameter. If ``chol_head`` still EXISTED it would
    receive no gradient, and DDP with find_unused_parameters=False hard-asserts that every
    parameter contributed to the loss (X1). So Arm 3 must not merely ignore the covariance
    — the Linear must not be constructed at all. Equivalently: Arm 3's claim is "there is
    no second moment", not "there is a second moment we declined to use"."""
    sc, geom = _session()
    sm, ss = _stats(sc)
    obj = V3JepaObjective(n_parcels=8, secondary_loss="l1")
    assert obj.perceiver is not None
    head = obj.perceiver.head
    no_chol = head.chol_head is None
    # no parameter anywhere in the head may carry "chol" — catches a future re-add
    chol_params = [n for n, _ in obj.perceiver.named_parameters() if "chol" in n]

    out = obj(_bands(), geom, sc.parcel_id, _masks(geom), stat_mean=sm, stat_std=ss)
    out.loss.backward()
    # EVERY perceiver parameter must receive a gradient — the exact DDP precondition.
    dead = [n for n, p in obj.perceiver.named_parameters()
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()
                                    or float(p.grad.abs().sum()) == 0.0)]
    ok = no_chol and not chol_params and not dead
    print(f"[check] arm3 point head: chol_head is None={no_chol}; params matching 'chol'="
          f"{chol_params}; perceiver params with NO/zero grad={dead} "
          f"(DDP find_unused_parameters=False requires none) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_arm3_l1_loss_is_the_present_masked_l1_of_mu() -> None:
    """The secondary term must BE the point loss (not an NLL), and the head must return
    cov=None so nothing downstream can silently read a covariance that does not exist."""
    from speech_decoding.models.v14_converged_v3.secondary_head import present_masked_l1

    sc, geom = _session()
    sm, ss = _stats(sc)
    obj = V3JepaObjective(n_parcels=8, secondary_loss="l1")
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom), stat_mean=sm, stat_std=ss)
    # total = JEPA_L1 + λ·secondary, same composition as the NLL arms
    recomposed = out.jepa_loss + obj.lambda_nll * out.nll_loss
    ok_total = torch.allclose(out.loss, recomposed, atol=1e-6)
    # the secondary is non-negative — |r| is, an NLL is NOT (that is the tell)
    ok_sign = float(out.nll_loss) >= 0.0
    mu, cov = obj.perceiver.head(torch.randn(4, obj.perceiver.d_perc))
    ok_none = cov is None
    print(f"[check] arm3 secondary={float(out.nll_loss):.4f} >= 0 ({ok_sign}); "
          f"total==jepa+λ·secondary ({ok_total}); head returns cov=None ({ok_none}) "
          f"{'OK' if ok_total and ok_sign and ok_none else 'VIOLATED'}")
    assert ok_total and ok_sign and ok_none


def test_present_masked_l1_ignores_absent_dims_and_matches_hand_sum() -> None:
    """Absent dims must contribute NOTHING: state_target sets the target to 0 there while
    mu stays free, so an unmasked L1 would score the head on a value carrying no
    information (and the std dims are absent exactly at n_elec=1). Also pins the reduction
    — per-position SUM over present dims, then MEAN over positions — which is what makes
    Arm 3 comparable to Arm 1 rather than a differently-normalized run."""
    from speech_decoding.models.v14_converged_v3.secondary_head import present_masked_l1

    mu = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        [0.0, 0.0, 0.0, 9.0, 9.0, 9.0]]])
    x = torch.zeros(1, 2, 6)
    present = torch.tensor([[[True] * 6, [True, True, True, False, False, False]]])
    got = float(present_masked_l1(mu, x, present))
    # pos0: all 6 present -> 1+2+3+4+5+6 = 21;  pos1: only the 3 mean dims -> 0
    want = (21.0 + 0.0) / 2
    # the absent std dims of pos1 hold 9.0 each: an unmasked L1 would add 27 -> 24.0
    unmasked = float((x - mu).abs().sum(-1).mean())
    ok = abs(got - want) < 1e-6 and abs(unmasked - 24.0) < 1e-6
    print(f"[check] present_masked_l1={got:.4f} == hand {want:.4f}; the same tensors "
          f"UNMASKED give {unmasked:.4f} (absent dims would leak 27/2 nats of gibberish) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_l2_would_be_the_gaussian_nll_at_identity_covariance() -> None:
    """The identity that makes L2-vs-L1 a NOISE-MODEL question rather than a taste one:
    at Sigma=I the full-cov Gaussian NLL IS 0.5*||r||^2 + const. Pinning it here is what
    licenses the M18 framing (L2 <=> Gaussian residuals, L1 <=> Laplace residuals) — if
    this identity ever breaks, the reasoning behind Arm 3's loss choice breaks with it."""
    import math

    from speech_decoding.models.v14_converged_v3.secondary_head import _nll_terms

    torch.manual_seed(0)
    mu = torch.randn(5, 6)
    x = torch.randn(5, 6)
    cov = torch.eye(6).expand(5, 6, 6)
    got = _nll_terms(mu, cov, x)
    want = 0.5 * ((x - mu) ** 2).sum(-1) + 0.5 * 6 * math.log(2 * math.pi)
    ok = torch.allclose(got, want, atol=1e-5)
    print(f"[check] gaussian NLL at Sigma=I == 0.5*||r||^2 + const (max diff "
          f"{float((got - want).abs().max()):.2e}) -> L2 IS the gaussian likelihood "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_arm3_per_band_taps_exist_without_a_covariance() -> None:
    """Arm 3 must not go blind: the per-band split is how r4's flatline was found, so it
    must survive the loss swap. The cov-dependent tap (#42 entropy-vs-floor) has no
    referent on this arm and must be ABSENT rather than fabricated."""
    sc, geom = _session()
    sm, ss = _stats(sc)
    obj = V3JepaObjective(n_parcels=8, secondary_loss="l1")
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom),
              stat_mean=sm, stat_std=ss, collect_taps=True)
    keys = set(out.taps or {})
    has_bands = {"nll_slow", "nll_mid", "nll_hga"} <= keys
    no_cov_tap = not any("entropy" in k or "cov" in k for k in keys)
    ok = has_bands and no_cov_tap
    print(f"[check] arm3 taps: per-band present={has_bands}; no cov/entropy tap "
          f"fabricated={no_cov_tap} (keys={sorted(k for k in keys if k != 'perc_lat')}) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_nll_arm_is_untouched_by_the_arm3_plumbing() -> None:
    """Adding Arm 3 must not perturb the arms ALREADY QUEUED. The default stays 'nll', the
    covariance head still exists, and the secondary is still an NLL (sign-indefinite)."""
    sc, geom = _session()
    sm, ss = _stats(sc)
    obj = V3JepaObjective(n_parcels=8)
    default_nll = obj.secondary_loss == "nll"
    has_chol = obj.perceiver.head.chol_head is not None
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom),
              stat_mean=sm, stat_std=ss, collect_taps=True)
    _, cov = obj.perceiver.head(torch.randn(4, obj.perceiver.d_perc))
    ok_cov = cov is not None and cov.shape[-2:] == (6, 6)
    ok = default_nll and has_chol and ok_cov
    print(f"[check] default arm untouched: secondary_loss='{obj.secondary_loss}' "
          f"({default_nll}); chol_head present ({has_chol}); cov {tuple(cov.shape)} "
          f"({ok_cov}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_secondary_loss_rejects_an_unknown_form() -> None:
    """A typo'd arm name must fail LOUDLY at construction, not silently fall back to the
    NLL and hand us a 48h run of the wrong experiment."""
    import pytest

    with pytest.raises(ValueError, match="secondary_loss"):
        V3JepaObjective(n_parcels=8, secondary_loss="l2")
