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
T = 16  # slow ×8→2, mid ×2→8, hga ×1→16


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
    slow = torch.randn(B, n, 7, T // 8)
    mid = torch.randn(B, n, 6, T // 2)
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
    out = obj(bands, geom, sc.parcel_id, mask)
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.loss.requires_grad


def test_gradient_flows_to_online_not_teacher() -> None:
    sc, geom = _session()
    obj = _obj()
    bands, mask = _batch(sc)
    obj(bands, geom, sc.parcel_id, mask).loss.backward()
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
    # With the terminal affine LayerNorm on the tower, target_ln is ~a no-op at init
    # (both give unit-scale targets). Its real job is to RE-normalize when the
    # terminal affine gamma/beta drift off identity during training — upstream stacks
    # both (affine `self.norm` THEN affine-free target LN). Force the drift on the
    # teacher's terminal norm and confirm target_ln then changes the loss.
    sc, geom = _session()
    bands, mask = _batch(sc)
    torch.manual_seed(0)
    obj = V3JepaObjective(n_parcels=N_PARCELS, target_ln=True)
    with torch.no_grad():
        obj.teacher.model.encoder.norm_out.weight.mul_(3.0)
        obj.teacher.model.encoder.norm_out.bias.add_(2.0)
    obj.target_ln = True
    lo = obj(bands, geom, sc.parcel_id, mask).loss
    obj.target_ln = False
    lf = obj(bands, geom, sc.parcel_id, mask).loss
    assert not torch.allclose(lo, lf)


def test_no_nan_when_a_whole_shaft_is_masked() -> None:
    sc, geom = _session()
    obj = _obj()
    n = len(sc.labels)
    bands = _bands(n)
    mask = torch.zeros(1, n, dtype=torch.bool)
    mask[0, sc.shaft_id == 0] = True  # whole shaft A masked (absent from encoder)
    out = obj(bands, geom, sc.parcel_id, mask)
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
    out = obj(bands, geom, sc.parcel_id, mask)
    assert out.n_masked == int(mask.sum()) * T


def test_loss_is_reducible_by_optimization() -> None:
    sc, geom = _session()
    obj = _obj()
    bands, mask = _batch(sc)
    opt = torch.optim.Adam([p for p in obj.parameters() if p.requires_grad], lr=1e-3)
    first = None
    for i in range(40):
        opt.zero_grad()
        loss = obj(bands, geom, sc.parcel_id, mask).loss
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    assert loss.item() < first  # predictor fits fixed teacher targets
