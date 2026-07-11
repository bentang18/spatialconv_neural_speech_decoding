"""v14_converged_v3 Phase 6 — plain-JEPA objective (TDD).

Memo project-v14-converged-v3-sensor-architecture (v1 = PLAIN JEPA, KISS):
EMA teacher + 1 predictor + masked-position L1 loss ONLY. KEEP target_ln
(affine-free F.layer_norm on teacher targets). Collapse guard = EMA-teacher
asymmetry + predictor bottleneck (NOT dense loss). I-JEPA mechanics: the online
tower (stem + encoder) sees VISIBLE electrodes only (masked excluded as keys so
targets can't leak); the EMA teacher sees the FULL grid → targets at masked
positions; the predictor re-inserts a learnable mask-query at each masked
(electrode,slot), its PE supplied by the predictor's own L1/L2.

The EMA teacher mirrors the ENTIRE target-producing path — stem (patch-embed) AND
encoder — matching V-JEPA (whose target encoder EMAs the patch-embed too). The
predictor is online-only (no teacher counterpart). Asserted contracts: scalar
finite loss; gradient flows to the online tower (stem + encoder) and the predictor
but NEVER the EMA teacher; the teacher (incl. its stem) lags then moves toward the
online net; target_ln is applied; no NaN when a whole shaft is masked; the loss
reads only masked positions; and the loss is reducible by optimization.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

N_PARCELS = 8
T = 16  # 32 Hz clock; uniform hop=64 → all bands at 32 Hz (T frames each)


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


def _batch(sc, n_masked_contacts=4, B=1):
    n = len(sc.labels)
    bands = _bands(n, B)
    mask = torch.zeros(B, n, dtype=torch.bool)
    mask[:, :n_masked_contacts] = True  # mask the first few contacts
    return bands, mask


def test_forward_returns_scalar_finite_loss() -> None:
    sc, geom = _session()
    obj = _obj()
    bands, mask = _batch(sc)
    out = obj(bands, geom, sc.parcel_id, mask, m_masked=int(mask[0].sum()))
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.loss.requires_grad


def test_gradient_flows_to_online_not_teacher() -> None:
    sc, geom = _session()
    obj = _obj()
    bands, mask = _batch(sc)
    obj(bands, geom, sc.parcel_id, mask, m_masked=int(mask[0].sum())).loss.backward()
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
    # gamma/beta drift off identity during training — upstream stacks both (affine
    # `norms_block` THEN affine-free per-level target LN). Force the drift on the
    # teacher's deepest-level norm (deep-sup default: no `norm_out`) and confirm
    # target_ln then changes the loss.
    sc, geom = _session()
    bands, mask = _batch(sc)
    torch.manual_seed(0)
    obj = V3JepaObjective(n_parcels=N_PARCELS, target_ln=True)
    with torch.no_grad():
        obj.teacher.model.encoder.norms_block[-1].weight.mul_(3.0)
        obj.teacher.model.encoder.norms_block[-1].bias.add_(2.0)
    obj.target_ln = True
    lo = obj(bands, geom, sc.parcel_id, mask, m_masked=int(mask[0].sum())).loss
    obj.target_ln = False
    lf = obj(bands, geom, sc.parcel_id, mask, m_masked=int(mask[0].sum())).loss
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
    bands, mask = _batch(sc)
    out = obj(bands, geom, sc.parcel_id, mask, m_masked=int(mask[0].sum()))
    assert torch.isfinite(out.loss) and out.loss.requires_grad


def test_single_tap_packed_matches_padded() -> None:
    # #24 equivalence must also hold in the single-tap arm (the ablation must be a
    # faithful control, not a differently-wired path).
    sc, geom = _session()
    torch.manual_seed(0)
    obj = V3JepaObjective(n_parcels=N_PARCELS, deep_sup=False).eval()
    bands, mask = _batch(sc, n_masked_contacts=5)
    m = int(mask[0].sum())
    packed = obj(bands, geom, sc.parcel_id, mask, m_masked=m, backend="reference")
    padded = obj._forward_padded(bands, geom, sc.parcel_id, mask)
    assert torch.allclose(packed.loss, padded.loss, atol=1e-5)


def test_no_nan_when_a_whole_shaft_is_masked() -> None:
    sc, geom = _session()
    obj = _obj()
    n = len(sc.labels)
    bands = _bands(n)
    mask = torch.zeros(1, n, dtype=torch.bool)
    mask[0, sc.shaft_id == 0] = True  # whole shaft A masked (absent from encoder)
    out = obj(bands, geom, sc.parcel_id, mask, m_masked=int(mask[0].sum()))
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in obj.parameters()
    )


def test_loss_reads_only_masked_positions() -> None:
    sc, geom = _session()
    obj = _obj()
    n = len(sc.labels)
    bands = _bands(n)
    mask = torch.zeros(1, n, dtype=torch.bool)
    mask[0, :4] = True
    out = obj(bands, geom, sc.parcel_id, mask, m_masked=int(mask[0].sum()))
    assert out.n_masked == int(mask.sum()) * T


def test_mask_token_is_weight_decayed_like_upstream() -> None:
    # Audit L36 (Ben-confirmed match-upstream): upstream stores the predictor mask
    # token 3-D (1,1,D) so the shared ndim<=1 no-decay rule DECAYS it; a 1-D (D,)
    # store would silently exempt it. v3 must keep it >=2-D → in the decay group.
    from speech_decoding.experiments.optim_param_groups import is_no_decay

    obj = _obj()
    assert obj.mask_token.ndim >= 2
    assert not is_no_decay("mask_token", obj.mask_token)  # decayed, upstream parity
    # still numerically a no-op in the forward (broadcasts to every masked slot)
    sc, geom = _session()
    bands, mask = _batch(sc)
    assert torch.isfinite(obj(bands, geom, sc.parcel_id, mask, m_masked=int(mask[0].sum())).loss)


def test_packed_forward_matches_padded_oracle() -> None:
    # THE #24 correctness proof: the packed (varlen) production forward reproduces
    # the padded oracle loss exactly (L1 is a mean over masked positions ⇒ the row
    # reorder between contact-order and full_plan-order is invariant).
    sc, geom = _session()
    obj = _obj().eval()
    bands, mask = _batch(sc, n_masked_contacts=5)  # shaft A whole + shaft B partial
    m = int(mask[0].sum())
    packed = obj(bands, geom, sc.parcel_id, mask, m_masked=m, backend="reference")
    padded = obj._forward_padded(bands, geom, sc.parcel_id, mask)
    assert torch.allclose(packed.loss, padded.loss, atol=1e-5)
    assert packed.n_masked == padded.n_masked


def test_packed_matches_padded_multi_clip_and_partial() -> None:
    # B>1 with a per-clip-uniform partial mask across all shafts.
    sc, geom = _session()
    obj = _obj().eval()
    n = len(sc.labels)
    bands = _bands(n, B=3)
    mask = torch.zeros(3, n, dtype=torch.bool)
    mask[:, [1, 5, 8]] = True  # one interior contact per shaft, same per clip
    packed = obj(bands, geom, sc.parcel_id, mask, m_masked=3, backend="reference")
    padded = obj._forward_padded(bands, geom, sc.parcel_id, mask)
    assert torch.allclose(packed.loss, padded.loss, atol=1e-5)
    assert packed.n_masked == padded.n_masked == 3 * 3 * T


def test_loss_is_reducible_by_optimization() -> None:
    sc, geom = _session()
    obj = _obj()
    bands, mask = _batch(sc)
    opt = torch.optim.Adam([p for p in obj.parameters() if p.requires_grad], lr=1e-3)
    first = None
    for i in range(40):
        opt.zero_grad()
        loss = obj(bands, geom, sc.parcel_id, mask, m_masked=int(mask[0].sum())).loss
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    assert loss.item() < first  # predictor fits fixed teacher targets
