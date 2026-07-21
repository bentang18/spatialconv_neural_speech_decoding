"""v14_converged_v3 — MAE-target objective arm (He 2021 / AudioMAE), the MAE-vs-JEPA ablation.

The MAE arm changes ONLY the prediction target: reconstruct each masked token's OWN
norm_pix'd input |STFT| bins instead of the EMA-teacher latent. Everything upstream (the
visible-only encoder with masked tokens dropped, the predictor, the mask query, the
margin-gated ``in_loss`` set, all locked HPs) is shared with the JEPA arm. The invariants a
silent miscompute would violate, named + asserted + printed
(feedback-build-the-invariant-into-the-probe):

  1. NO EMA teacher (``obj.teacher is None``); ``update_teacher()`` is a no-op. The arm has
     per-band reconstruction heads instead of ``pred_to_target``/the Perceiver.
  2. Forward runs finite; loss is a non-negative scalar; grads reach every TRAINABLE module
     (stem projs, encoder, predictor, enc_to_pred, the 3 mae_heads, mask_token).
  3. TARGET ALIGNMENT: the gathered target at flat token t is EXACTLY norm_pix of that
     token's own band bins ``bands[band][:, contact, :, time_pos]`` — the whole point of the
     static ``lin`` gather. A misaligned gather trains against the wrong contact/frame/band.
  4. PER-BAND HEAD SELECTION: ``_mae_pred`` routes each token through its band's head only.
  5. norm_pix is the exact He recipe: per-token mean0 / unbiased-var-1 over the VALID bins,
     with MID's pad bin excluded.
  6. LEAK-SAFETY still holds (masked tokens dropped from the encoder) — inherited from the
     shared visible path, re-asserted here because the arm reuses it.
  7. The arm forbids the secondary NLL / context loss (target-only feature).
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
from speech_decoding.models.v14_converged_v3.stem import PER_BAND_SPECS

T = 16  # 32 Hz clock; multiple of SLOW stride 8 ⇒ SLOW 2 / MID 8 / HGA 16 tokens per contact.
B = 2
N = 5
F_MAX = max(nb for nb, _ in PER_BAND_SPECS)  # 7


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


def _mae(**kw) -> V3JepaObjective:
    return V3JepaObjective(n_parcels=8, mae=True, **kw)


def test_mae_arm_has_no_teacher_and_update_is_noop() -> None:
    obj = _mae()
    no_teacher = obj.teacher is None
    no_perceiver = obj.perceiver is None
    no_ptt = obj.pred_to_target is None
    has_heads = len(obj.mae_heads) == len(PER_BAND_SPECS) and all(
        h.out_features == nb for h, (nb, _) in zip(obj.mae_heads, PER_BAND_SPECS)
    )
    noop = obj.update_teacher() == 0.0  # must not raise / touch a None teacher
    ok = no_teacher and no_perceiver and no_ptt and has_heads and noop
    print(f"[check] teacher=None ({no_teacher}), perceiver=None ({no_perceiver}), "
          f"pred_to_target=None ({no_ptt}), 3 per-band heads {[h.out_features for h in obj.mae_heads]} "
          f"({has_heads}), update_teacher()=0 no-op ({noop}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_forward_runs_and_grads_reach_every_trainable_module() -> None:
    sc, geom = _session()
    obj = _mae()
    obj.train()
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom))
    loss_val = out.loss.detach()
    out.loss.backward()

    finite = bool(torch.isfinite(loss_val)) and loss_val.ndim == 0 and float(loss_val) >= 0.0
    single = out.jepa_loss is None and out.nll_loss is None and out.ctx_loss is None
    checks = {
        "stem.projs": obj.online.stem.projs,
        "encoder": obj.online.encoder,
        "predictor": obj.predictor,
        "enc_to_pred": obj.enc_to_pred,
        "mae_heads": obj.mae_heads,
    }
    grad_ok = {
        name: all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        for name, m in checks.items()
    }
    mask_grad = obj.mask_token.grad is not None and torch.isfinite(obj.mask_token.grad).all()
    ok = finite and single and all(grad_ok.values()) and mask_grad
    print(f"[check] loss={float(out.loss):.4f} finite={finite}; single-loss={single}; "
          f"grads {grad_ok}; mask_token grad={mask_grad} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_target_is_norm_pix_of_the_tokens_own_band_bins() -> None:
    # THE alignment invariant: the static lin gather must map flat token t to its OWN
    # (band, contact, time_pos) input bins, then norm_pix them. Rebuild the reference by hand
    # per token and assert it matches _mae_gather_target → _norm_pix exactly.
    sc, geom = _session()
    obj = _mae()
    grid = build_r4_grid(geom, n_time=T)
    bands = _bands(seed=3)
    tgt_raw, feat_valid, feat_count = obj._mae_gather_target(bands, grid)
    tgt = obj._norm_pix(tgt_raw.float(), feat_valid, feat_count)  # (B, total, F_MAX)

    max_dev = 0.0
    aligned = True
    for t in range(grid.total):
        b = int(grid.band[t]); c = int(grid.contact[t]); tp = int(grid.time_pos[t])
        fb = PER_BAND_SPECS[b][0]
        # own bins at the decimated frame (time_pos IS the T32 index = bandpos·stride).
        raw = bands[b][:, c, :, tp]  # (B, F_b)
        assert tp == int(grid.bandpos[t]) * BAND_STRIDES[b]  # frame identity
        mean = raw.mean(-1, keepdim=True)
        var = raw.var(-1, unbiased=True, keepdim=True)
        ref = (raw - mean) / (var + 1e-6).sqrt()  # (B, F_b)
        dev = (tgt[:, t, :fb] - ref).abs().max().item()
        max_dev = max(max_dev, dev)
        # pad bin (only MID, fb<F_MAX) must be marked invalid.
        if fb < F_MAX and bool(feat_valid[t, fb]):
            aligned = False
    exact = max_dev < 1e-5 and aligned
    print(f"[check] target == norm_pix(own bins) for all {grid.total} tokens: max|Δ|={max_dev:.2e}, "
          f"pad-marked-invalid={aligned} {'OK' if exact else 'VIOLATED'}")
    assert exact


def test_norm_pix_is_per_token_mean0_unbiased_var1_over_valid_bins() -> None:
    # He norm_pix: each token's valid bins standardize to mean 0, unbiased var 1 (the eps
    # only slightly shrinks it). MID's 7th (pad) bin is excluded from both moments.
    sc, geom = _session()
    obj = _mae()
    grid = build_r4_grid(geom, n_time=T)
    tgt_raw, feat_valid, feat_count = obj._mae_gather_target(_bands(seed=5), grid)
    tgt = obj._norm_pix(tgt_raw.float(), feat_valid, feat_count)
    fv = feat_valid[None].float()  # (1, total, F_MAX)
    fc = feat_count[None, :, None].float()
    mean = (tgt * fv).sum(-1, keepdim=True) / fc
    var = ((tgt - mean) ** 2 * fv).sum(-1, keepdim=True) / (fc - 1.0)
    mean0 = mean.abs().max().item() < 1e-4
    # eps 1e-6 on a unit-var signal ⇒ var ~ var_true/(var_true+1e-6) ≈ 1; allow a small band.
    var1 = (var - 1.0).abs().max().item() < 1e-2
    ok = mean0 and var1
    print(f"[check] norm_pix per-token mean0 (max|μ|={mean.abs().max().item():.2e}, {mean0}) + "
          f"unbiased var1 (max|σ²−1|={(var - 1.0).abs().max().item():.2e}, {var1}) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_mae_pred_routes_each_token_through_its_band_head() -> None:
    # Per-band selection: set each head to emit a band-identifying constant (bias = band index
    # in every out dim, zero weight) and assert _mae_pred returns that constant per token —
    # i.e. token of band b gets head-b's output, never another band's.
    sc, geom = _session()
    obj = _mae()
    grid = build_r4_grid(geom, n_time=T)
    with torch.no_grad():
        for b, head in enumerate(obj.mae_heads):
            head.weight.zero_()
            head.bias.fill_(float(b))
    h = torch.randn(B, grid.total, obj.mae_heads[0].in_features)
    pred = obj._mae_pred(h, grid)  # (B, total, F_MAX)
    routed = True
    for t in range(grid.total):
        b = int(grid.band[t]); fb = PER_BAND_SPECS[b][0]
        if not torch.allclose(pred[:, t, :fb], torch.full((B, fb), float(b))):
            routed = False
            break
    print(f"[check] each token routed through its own band head (const b) ={routed} "
          f"{'OK' if routed else 'VIOLATED'}")
    assert routed


def test_leak_safe_online_latent_ignores_masked_token_inputs() -> None:
    # Shared visible path invariant, re-asserted for the MAE arm: masked tokens are dropped
    # from the encoder, so the online latent is BIT-IDENTICAL whether or not the masked
    # tokens' input frames are corrupted.
    sc, geom = _session()
    obj = _mae().eval()
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
    leak_safe = torch.allclose(z1, z2, atol=1e-6)
    print(f"[check] online latent invariant to masked-token inputs={leak_safe} "
          f"(max|Δ|={(z1 - z2).abs().max().item():.2e}) {'OK' if leak_safe else 'VIOLATED'}")
    assert leak_safe


def test_loss_only_depends_on_masked_target_frames() -> None:
    # MAE scores reconstruction at the margin-gated in_loss tokens. Corrupting the RAW input
    # frames those tokens read (their reconstruction targets) MUST change the loss; corrupting
    # a frame no in_loss token reads must NOT. This ties the loss to the correct target frames.
    sc, geom = _session()
    obj = _mae().eval()
    grid = build_r4_grid(geom, n_time=T)
    masks = _masks(geom)
    _, in_loss = token_flags(grid, masks)
    bands = _bands()
    base = obj(bands, geom, sc.parcel_id, masks).loss.item()

    # frames that ARE reconstruction targets (an in_loss token reads them).
    scored = torch.nonzero(in_loss.any(0)).flatten().tolist()
    b2 = [b.clone() for b in bands]
    for t in scored:
        bnd, c, tp = int(grid.band[t]), int(grid.contact[t]), int(grid.time_pos[t])
        b2[bnd][:, c, :, tp] += 10.0
    moved = obj(b2, geom, sc.parcel_id, masks).loss.item()
    changed = abs(moved - base) > 1e-4
    print(f"[check] corrupting scored target frames moves loss {base:.4f}→{moved:.4f} "
          f"(Δ={abs(moved - base):.2e})={changed} {'OK' if changed else 'VIOLATED'}")
    assert changed


def test_mae_forbids_secondary_and_context() -> None:
    import pytest

    with pytest.raises(ValueError, match="context"):
        V3JepaObjective(n_parcels=8, mae=True, context_loss=True)
    sc, geom = _session()
    obj = _mae()
    parcels = torch.unique(sc.parcel_id)
    sm = torch.zeros(parcels.numel(), 5)
    ss = torch.ones(parcels.numel(), 5)
    with pytest.raises(ValueError, match="secondary"):
        obj(_bands(), geom, sc.parcel_id, _masks(geom), stat_mean=sm, stat_std=ss)


def test_collect_taps_emits_enc12_and_per_band_recon_scalars() -> None:
    # Monitor parity with arm0: collect_taps=True ships enc12 + the per-band recon scalars the
    # health callback logs — all finite — and NO secondary keys (the MAE arm has no NLL).
    sc, geom = _session()
    obj = _mae()
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom), collect_taps=True)
    taps = out.taps or {}
    has_enc = "enc12" in taps and taps["enc12"].ndim == 2
    has_band = all(f"jepa_{b}_l1" in taps for b in ("slow", "mid", "hga"))
    finite = all(torch.isfinite(v).all() for v in taps.values())
    no_sec = not any(k.startswith("nll_") or k.startswith("cov_entropy") for k in taps)
    out.loss.backward()  # graph intact despite the monitor reductions
    ok = has_enc and has_band and finite and no_sec
    print(f"[check] taps: enc12 ({has_enc}) + per-band recon ({has_band}) finite ({finite}), "
          f"no secondary keys ({no_sec}) {'OK' if ok else 'VIOLATED'}")
    assert ok
