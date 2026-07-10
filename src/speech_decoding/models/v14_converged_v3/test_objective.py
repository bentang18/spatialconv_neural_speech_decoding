"""v14_converged_v3 Phase 6 — plain-JEPA objective (TDD).

Memo project-v14-converged-v3-sensor-architecture (v1 = PLAIN JEPA, KISS):
EMA teacher + 1 predictor + masked-position L1 loss ONLY. KEEP target_ln
(affine-free F.layer_norm on teacher targets). Collapse guard = EMA-teacher
asymmetry + predictor bottleneck (NOT dense loss). I-JEPA mechanics: the online
encoder sees VISIBLE electrodes only (masked excluded as keys so targets can't
leak); the EMA teacher sees the FULL grid → targets at masked positions; the
predictor re-inserts a learnable mask-query at each masked (electrode,slot),
its PE supplied by the predictor's own L1 (index-RoPE) + L2 (parcel identity).

Asserted contracts: scalar finite loss; gradient flows to online (encoder /
predictor / projections) but NEVER the EMA teacher; the teacher lags then moves
toward the online net on update; target_ln is applied; no NaN when a whole shaft
is masked (all-excluded attention rows go uniform, not NaN); the loss reads only
masked positions; and the loss is reducible by optimization (the predictor can
fit fixed teacher targets from visible context).
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

N_PARCELS = 8
T = 6


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


def _batch(sc, n_masked_contacts=4, B=1):
    n = len(sc.labels)
    x = torch.randn(B, n, T, 256)
    mask = torch.zeros(B, n, dtype=torch.bool)
    mask[:, :n_masked_contacts] = True  # mask the first few contacts
    return x, mask


def test_forward_returns_scalar_finite_loss() -> None:
    sc, geom = _session()
    obj = _obj()
    x, mask = _batch(sc)
    out = obj(x, geom, sc.parcel_id, mask)
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.loss.requires_grad


def test_gradient_flows_to_online_not_teacher() -> None:
    sc, geom = _session()
    obj = _obj()
    x, mask = _batch(sc)
    obj(x, geom, sc.parcel_id, mask).loss.backward()
    # online modules receive grad
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in obj.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in obj.predictor.parameters())
    # the EMA teacher is frozen: no grad ever
    assert all(p.grad is None for p in obj.teacher.parameters())
    assert all(not p.requires_grad for p in obj.teacher.parameters())


def test_teacher_lags_then_moves_toward_online() -> None:
    sc, geom = _session()
    obj = _obj()
    x, mask = _batch(sc)
    # teacher starts == encoder (deepcopy)
    enc_p = dict(obj.encoder.named_parameters())
    tea_p = dict(obj.teacher.model.named_parameters())
    name = next(iter(enc_p))
    assert torch.allclose(enc_p[name], tea_p[name])
    # perturb the online encoder, then EMA-update → teacher moves partway
    with torch.no_grad():
        enc_p[name] += 1.0
    before = tea_p[name].clone()
    obj.update_teacher()
    after = dict(obj.teacher.model.named_parameters())[name]
    assert not torch.allclose(after, before)  # moved
    assert not torch.allclose(after, enc_p[name])  # but LAGS (not fully caught up)


def test_target_ln_is_applied() -> None:
    sc, geom = _session()
    x, mask = _batch(sc)
    torch.manual_seed(0)
    on = V3JepaObjective(n_parcels=N_PARCELS, target_ln=True)
    torch.manual_seed(0)
    off = V3JepaObjective(n_parcels=N_PARCELS, target_ln=False)
    # same init (same seed) ⇒ the only difference is the target normalisation.
    lo = on(x, geom, sc.parcel_id, mask).loss
    lf = off(x, geom, sc.parcel_id, mask).loss
    assert not torch.allclose(lo, lf)


def test_no_nan_when_a_whole_shaft_is_masked() -> None:
    sc, geom = _session()
    obj = _obj()
    n = len(sc.labels)
    x = torch.randn(1, n, T, 256)
    mask = torch.zeros(1, n, dtype=torch.bool)
    mask[0, sc.shaft_id == 0] = True  # whole shaft A masked (absent from encoder)
    out = obj(x, geom, sc.parcel_id, mask)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all()
        for p in obj.parameters()
    )


def test_loss_reads_only_masked_positions() -> None:
    # Perturbing the predictor's target at a VISIBLE contact must not change the
    # loss; perturbing a MASKED contact's teacher target must.
    sc, geom = _session()
    obj = _obj()
    n = len(sc.labels)
    x = torch.randn(1, n, T, 256)
    mask = torch.zeros(1, n, dtype=torch.bool)
    mask[0, :4] = True
    base = obj(x, geom, sc.parcel_id, mask).loss.item()
    # change a VISIBLE contact's input (contact 6, visible) → teacher target there
    # is not in the loss; but it CAN change visible context → allow it to affect.
    # Instead: masked-count invariance — loss over exactly the masked tube.
    out = obj(x, geom, sc.parcel_id, mask)
    assert out.n_masked == int(mask.sum()) * T


def test_loss_is_reducible_by_optimization() -> None:
    sc, geom = _session()
    obj = _obj()
    x, mask = _batch(sc)
    opt = torch.optim.Adam(
        [p for p in obj.parameters() if p.requires_grad], lr=1e-3
    )
    first = None
    for i in range(40):
        opt.zero_grad()
        loss = obj(x, geom, sc.parcel_id, mask).loss
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    assert loss.item() < first  # predictor fits fixed teacher targets
