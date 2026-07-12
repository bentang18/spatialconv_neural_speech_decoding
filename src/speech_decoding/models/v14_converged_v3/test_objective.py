"""v14_converged_v3 Phase 6 — plain-JEPA objective (TDD, dual-axis).

Memo project-v14-converged-v3-sensor-architecture (v1 = PLAIN JEPA, KISS):
EMA teacher + 1 predictor + masked-CELL L1 loss ONLY. KEEP target_ln (affine-free
F.layer_norm on teacher targets). Collapse guard = EMA-teacher asymmetry + predictor
bottleneck (NOT dense loss). I-JEPA mechanics: the online tower (stem + encoder) sees
VISIBLE CELLS only (masked cells excluded as keys so targets can't leak); the EMA
teacher sees the FULL grid → targets at masked cells; the predictor re-inserts a
learnable mask-query at each masked (contact, frame).

Dual-axis masking (Ben 2026-07-12): ``contact_mask`` (B,N) drops whole contacts;
``frame_mask`` (B,S,T) drops frames per shaft. A CELL (c,t) is a target iff its contact
is masked OR its shaft's frame t is masked. ``m_vis = N−D`` and ``t_kept = T−T_mask``
are per-session constants the model supplies (the compacted online shapes).

Asserted contracts: scalar finite loss; gradient flows to the online tower + predictor
but NEVER the EMA teacher; the teacher lags then moves; target_ln applied; no NaN when a
whole shaft is masked; loss reads only masked cells; loss reducible by optimization; and
the packed production path reproduces the padded oracle exactly.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import (
    V3MaskConfig,
    V3Masks,
    sample_masks,
)
from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

N_PARCELS = 8
T = 16  # 32 Hz clock; uniform hop=64 → all bands at 32 Hz (T frames each)
SPACE, TIME = 0.4, 0.4  # feasible on the (4,3,3) montage (largest shaft 4 = D)


def _session(shaft_sizes=(4, 3, 3)):
    labels, parcels = [], []
    for s, n in enumerate(shaft_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
            parcels.append(s % N_PARCELS)
    sc = build_sidecar(labels, parcel_id=torch.tensor(parcels, dtype=torch.long))
    return sc, build_l1_geometry(sc)


def _obj():
    torch.manual_seed(0)
    return V3JepaObjective(n_parcels=N_PARCELS)


def _bands(n, B=1):
    slow = torch.randn(B, n, 7, T)
    mid = torch.randn(B, n, 6, T)
    hga = torch.randn(B, n, 7, T)
    return [slow, mid, hga]


def _masks(sc, geom, *, B=1, seed=0, space=SPACE, time=TIME):
    """Sample a fixed dual-axis mask + the derived (m_vis, t_kept) constants."""
    n = len(sc.labels)
    g = torch.Generator().manual_seed(seed)
    cfg = V3MaskConfig(space_frac=space, time_frac=time)
    m = sample_masks(geom, n, n_time=T, n_rows=B, generator=g, cfg=cfg)
    return m, n - round(space * n), T - round(time * T)


def _whole_shaft_masks(sc, geom, *, shaft=0, time=TIME):
    """Deterministic masks with one shaft wholly dropped (space) + contiguous per-shaft
    time blocks (exact T_mask each), for the whole-shaft no-NaN test."""
    n = len(sc.labels)
    contact_mask = torch.zeros(1, n, dtype=torch.bool)
    contact_mask[0, sc.shaft_id == shaft] = True  # whole shaft absent from the encoder
    d = int(contact_mask[0].sum())
    t_mask = round(time * T)
    frame_mask = torch.zeros(1, geom.n_shafts, T, dtype=torch.bool)
    frame_mask[0, :, :t_mask] = True  # exact T_mask/shaft (homogeneous within shaft)
    masks = V3Masks(contact_mask=contact_mask, frame_mask=frame_mask,
                    whole_contact=contact_mask.clone())
    return masks, n - d, T - t_mask


def _fwd(obj, bands, geom, sc, mm, **kw):
    m, m_vis, t_kept = mm
    return obj(bands, geom, sc.parcel_id, m.contact_mask, m.frame_mask,
               m_vis=m_vis, t_kept=t_kept, **kw)


def _fwd_padded(obj, bands, geom, sc, mm, **kw):
    m, _, _ = mm
    return obj._forward_padded(
        bands, geom, sc.parcel_id, m.contact_mask, m.frame_mask, **kw
    )


def _cell_masked(m, geom):
    return m.contact_mask[:, :, None] | m.frame_mask[:, geom.shaft_of_contact, :]


def test_forward_returns_scalar_finite_loss() -> None:
    sc, geom = _session()
    obj = _obj()
    out = _fwd(obj, _bands(len(sc.labels)), geom, sc, _masks(sc, geom))
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.loss.requires_grad


def test_gradient_flows_to_online_not_teacher() -> None:
    sc, geom = _session()
    obj = _obj()
    _fwd(obj, _bands(len(sc.labels)), geom, sc, _masks(sc, geom)).loss.backward()
    # online tower (stem + encoder) and predictor receive grad
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in obj.online.stem.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in obj.online.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in obj.predictor.parameters())
    # the EMA teacher is frozen: no grad ever, on stem OR encoder
    assert all(p.grad is None for p in obj.teacher.parameters())
    assert all(not p.requires_grad for p in obj.teacher.parameters())


def test_teacher_emas_the_stem_too() -> None:
    # V-JEPA contract: the teacher mirrors the patch-embed (our stem), not just the
    # transformer. The teacher must own a stem that lags then moves on update.
    sc, geom = _session()
    obj = _obj()
    tea = dict(obj.teacher.model.named_parameters())
    stem_names = [n for n in tea if n.startswith("stem.")]
    assert stem_names, "teacher has no stem params — stem is not EMA'd"
    online = dict(obj.online.named_parameters())
    name = stem_names[0]
    assert torch.allclose(online[name], tea[name])  # starts equal (deepcopy)
    with torch.no_grad():
        online[name] += 1.0
    before = tea[name].clone()
    obj.update_teacher()
    after = dict(obj.teacher.model.named_parameters())[name]
    assert not torch.allclose(after, before)  # teacher stem moved
    assert not torch.allclose(after, online[name])  # but LAGS


def test_teacher_lags_then_moves_toward_online() -> None:
    sc, geom = _session()
    obj = _obj()
    online = dict(obj.online.named_parameters())
    tea = dict(obj.teacher.model.named_parameters())
    name = next(n for n in online if n.startswith("encoder."))
    assert torch.allclose(online[name], tea[name])
    with torch.no_grad():
        online[name] += 1.0
    before = tea[name].clone()
    obj.update_teacher()
    after = dict(obj.teacher.model.named_parameters())[name]
    assert not torch.allclose(after, before)  # moved
    assert not torch.allclose(after, online[name])  # but LAGS


def test_target_ln_is_applied() -> None:
    # With the per-level affine LayerNorm on the tower, target_ln is ~a no-op at init
    # (both give unit-scale targets). Its real job is to RE-normalize when the affine
    # gamma/beta drift off identity during training. Force the drift on the teacher's
    # deepest-level norm and confirm target_ln then changes the loss.
    sc, geom = _session()
    bands = _bands(len(sc.labels))
    mm = _masks(sc, geom)
    torch.manual_seed(0)
    obj = V3JepaObjective(n_parcels=N_PARCELS, target_ln=True)
    with torch.no_grad():
        obj.teacher.model.encoder.norms_block[-1].weight.mul_(3.0)
        obj.teacher.model.encoder.norms_block[-1].bias.add_(2.0)
    obj.target_ln = True
    lo = _fwd(obj, bands, geom, sc, mm).loss
    obj.target_ln = False
    lf = _fwd(obj, bands, geom, sc, mm).loss
    assert not torch.allclose(lo, lf)


def test_deep_sup_default_wiring() -> None:
    # #61 deep-sup default: the encoder→predictor map is upstream's 2-layer
    # `predictor_embed` fusion MLP Linear(4·256→256)·GELU·Linear(256→128), and
    # `pred_to_target` is the ONE wide Linear(128→4·256) emitting all levels. The
    # teacher/online encoders are deep-sup (4 per-level norms, no norm_out).
    import torch.nn as nn

    from speech_decoding.models.v14_converged_v3.towers import N_LEVELS

    obj = _obj()
    assert obj.deep_sup and obj.n_levels == N_LEVELS == 4
    assert isinstance(obj.enc_to_pred, nn.Sequential)
    lins = [m for m in obj.enc_to_pred if isinstance(m, nn.Linear)]
    assert len(lins) == 2
    assert lins[0].in_features == 4 * 256 and lins[0].out_features == 256
    assert lins[1].in_features == 256 and lins[1].out_features == 128
    assert any(isinstance(m, nn.GELU) for m in obj.enc_to_pred)
    assert isinstance(obj.pred_to_target, nn.Linear)
    assert obj.pred_to_target.in_features == 128
    assert obj.pred_to_target.out_features == 4 * 256
    assert obj.online.encoder.norm_out is None
    assert len(obj.online.encoder.norms_block) == 4


def test_deep_sup_target_is_per_level_double_normed() -> None:
    # The teacher emits 4 concatenated levels (each already affine-`norms_block`'d);
    # `_ln_target` with n_levels=4 applies a SECOND parameter-free LN to EACH 256-chunk
    # independently ⇒ every chunk is zero-mean/unit-var, not the whole 1024 vector.
    from speech_decoding.models.v14_converged_v3.objective import _ln_target

    torch.manual_seed(1)
    t = torch.randn(2, 5, 3, 4 * 256) * 7.0 + 3.0
    out = _ln_target(t, n_levels=4)
    assert out.shape == t.shape
    for lvl in range(4):
        chunk = out[..., lvl * 256 : (lvl + 1) * 256]
        assert chunk.mean(-1).abs().max() < 1e-5
        assert (chunk.var(-1, unbiased=False) - 1.0).abs().max() < 1e-3
    # a single whole-vector LN would NOT leave each chunk unit-var → the two differ
    whole = _ln_target(t, n_levels=1)
    assert not torch.allclose(out, whole, atol=1e-3)


def test_single_tap_arm_wiring() -> None:
    # deep_sup=False = the single-tap ablation arm: plain Linear maps, encoder norm_out.
    import torch.nn as nn

    obj = V3JepaObjective(n_parcels=N_PARCELS, deep_sup=False)
    assert not obj.deep_sup and obj.n_levels == 1
    assert isinstance(obj.enc_to_pred, nn.Linear)
    assert obj.enc_to_pred.in_features == 256 and obj.enc_to_pred.out_features == 128
    assert isinstance(obj.pred_to_target, nn.Linear)
    assert obj.pred_to_target.in_features == 128 and obj.pred_to_target.out_features == 256
    assert obj.online.encoder.norm_out is not None
    assert obj.online.encoder.norms_block is None
    # still trains end-to-end
    sc, geom = _session()
    out = _fwd(obj, _bands(len(sc.labels)), geom, sc, _masks(sc, geom))
    assert torch.isfinite(out.loss) and out.loss.requires_grad


def test_single_tap_packed_matches_padded() -> None:
    # #24 equivalence must also hold in the single-tap arm (the ablation must be a
    # faithful control, not a differently-wired path).
    sc, geom = _session()
    torch.manual_seed(0)
    obj = V3JepaObjective(n_parcels=N_PARCELS, deep_sup=False).eval()
    bands = _bands(len(sc.labels))
    mm = _masks(sc, geom, seed=3)
    packed = _fwd(obj, bands, geom, sc, mm, backend="reference")
    padded = _fwd_padded(obj, bands, geom, sc, mm)
    assert torch.allclose(packed.loss, padded.loss, atol=1e-5)


def test_context_head_shape_both_arms() -> None:
    # #66: pred_to_target_context mirrors pred_to_target's target space (deep-sup
    # 128→4·256; single-tap 128→256).
    import torch.nn as nn

    obj = _obj()  # deep-sup default
    assert isinstance(obj.pred_to_target_context, nn.Linear)
    assert obj.pred_to_target_context.in_features == 128
    assert obj.pred_to_target_context.out_features == 4 * 256
    obj1 = V3JepaObjective(n_parcels=N_PARCELS, deep_sup=False)
    assert obj1.pred_to_target_context.out_features == 256


def test_context_lambda_zero_is_exactly_plain_jepa() -> None:
    # A python 0.0 (default / feature-off) is the static-skip: loss must be BYTE-equal
    # to not passing lambda_context at all — the context head never perturbs the loss.
    sc, geom = _session()
    obj = _obj().eval()
    bands = _bands(len(sc.labels))
    mm = _masks(sc, geom)
    plain = _fwd(obj, bands, geom, sc, mm)
    off = _fwd(obj, bands, geom, sc, mm, lambda_context=0.0)
    assert torch.equal(plain.loss, off.loss)


def test_context_lambda_positive_changes_loss_and_trains_head() -> None:
    # A 0-d λ tensor > 0 adds the visible-CELL context L1 ⇒ loss changes, and the
    # context head receives gradient (it is trained only when the schedule is active).
    sc, geom = _session()
    obj = _obj()
    bands = _bands(len(sc.labels))
    mm = _masks(sc, geom)
    base = _fwd(obj, bands, geom, sc, mm).loss
    lam = torch.tensor(0.5)
    withc = _fwd(obj, bands, geom, sc, mm, lambda_context=lam)
    assert not torch.allclose(base, withc.loss)
    withc.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in obj.pred_to_target_context.parameters()
    )
    # teacher still frozen under the context loss
    assert all(p.grad is None for p in obj.teacher.parameters())


def test_context_head_untouched_when_lambda_zero() -> None:
    # With the static-off path, the context head gets NO gradient (it is skipped).
    sc, geom = _session()
    obj = _obj()
    _fwd(obj, _bands(len(sc.labels)), geom, sc, _masks(sc, geom)).loss.backward()
    assert all(p.grad is None for p in obj.pred_to_target_context.parameters())


def test_packed_matches_padded_with_context_loss() -> None:
    # #24 equivalence must survive the context loss: both paths add λ·(visible-CELL
    # L1), a mean over context cells ⇒ order-invariant like the masked loss.
    sc, geom = _session()
    obj = _obj().eval()
    bands = _bands(len(sc.labels))
    mm = _masks(sc, geom, seed=5)
    lam = torch.tensor(0.5)
    packed = _fwd(obj, bands, geom, sc, mm, backend="reference", lambda_context=lam)
    padded = _fwd_padded(obj, bands, geom, sc, mm, lambda_context=lam)
    assert torch.allclose(packed.loss, padded.loss, atol=1e-5)


def test_no_nan_when_a_whole_shaft_is_masked() -> None:
    sc, geom = _session()
    obj = _obj()
    mm = _whole_shaft_masks(sc, geom, shaft=0)  # whole shaft A absent from the encoder
    out = _fwd(obj, _bands(len(sc.labels)), geom, sc, mm)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in obj.parameters()
    )


def test_loss_reads_only_masked_cells() -> None:
    sc, geom = _session()
    obj = _obj()
    mm = _masks(sc, geom)
    out = _fwd(obj, _bands(len(sc.labels)), geom, sc, mm)
    # n_masked = total masked CELLS = N·T − m_vis·t_kept (the dual-axis count).
    assert out.n_masked == int(_cell_masked(mm[0], geom).sum())


def test_mask_token_is_weight_decayed_like_upstream() -> None:
    # Audit L36 (Ben-confirmed match-upstream): upstream stores the predictor mask
    # token 3-D (1,1,D) so the shared ndim<=1 no-decay rule DECAYS it; a 1-D (D,)
    # store would silently exempt it. v3 must keep it >=2-D → in the decay group.
    from speech_decoding.experiments.optim_param_groups import is_no_decay

    obj = _obj()
    assert obj.mask_token.ndim >= 2
    assert not is_no_decay("mask_token", obj.mask_token)  # decayed, upstream parity
    # still numerically a no-op in the forward (broadcasts to every masked cell)
    sc, geom = _session()
    assert torch.isfinite(
        _fwd(obj, _bands(len(sc.labels)), geom, sc, _masks(sc, geom)).loss
    )


def test_packed_forward_matches_padded_oracle() -> None:
    # THE #24 correctness proof: the packed (varlen) production forward reproduces the
    # padded oracle loss exactly (L1 is a mean over masked cells ⇒ the reorder between
    # contact-order and full_plan-order is invariant).
    sc, geom = _session()
    obj = _obj().eval()
    bands = _bands(len(sc.labels))
    mm = _masks(sc, geom, seed=7)  # sampled dual-axis: whole + intra + per-shaft time
    packed = _fwd(obj, bands, geom, sc, mm, backend="reference")
    padded = _fwd_padded(obj, bands, geom, sc, mm)
    assert torch.allclose(packed.loss, padded.loss, atol=1e-5)
    assert packed.n_masked == padded.n_masked


def test_packed_matches_padded_multi_clip_and_partial() -> None:
    # B>1 with independent per-clip dual-axis masks.
    sc, geom = _session()
    obj = _obj().eval()
    n = len(sc.labels)
    bands = _bands(n, B=3)
    mm = _masks(sc, geom, B=3, seed=11)
    packed = _fwd(obj, bands, geom, sc, mm, backend="reference")
    padded = _fwd_padded(obj, bands, geom, sc, mm)
    m, m_vis, t_kept = mm
    assert torch.allclose(packed.loss, padded.loss, atol=1e-5)
    assert packed.n_masked == padded.n_masked == 3 * (n * T - m_vis * t_kept)


def test_loss_is_reducible_by_optimization() -> None:
    sc, geom = _session()
    obj = _obj()
    bands = _bands(len(sc.labels))
    mm = _masks(sc, geom)  # FIXED mask (fit the frozen teacher targets)
    opt = torch.optim.Adam([p for p in obj.parameters() if p.requires_grad], lr=1e-3)
    first = None
    for i in range(40):
        opt.zero_grad()
        loss = _fwd(obj, bands, geom, sc, mm).loss
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    assert loss.item() < first  # predictor fits fixed teacher targets
