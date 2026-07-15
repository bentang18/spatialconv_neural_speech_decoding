"""v14_converged_v3 r4 — plain-JEPA objective STRUCTURAL invariants (TDD).

This file holds the wiring / EMA / target-norm invariants of ``V3JepaObjective`` that
are INDEPENDENT of the flat-path forward behaviour (which lives in test_objective_r4.py:
finite loss + grads, teacher stop-grad, leak-safety, margin gate). Kept here:

  • target_ln re-normalizes teacher targets when the affine norms drift;
  • deep-sup vs single-tap MAP wiring (enc_to_pred / pred_to_target / encoder norms);
  • ``_ln_target`` per-level DOUBLE norm;
  • the EMA teacher mirrors the STEM too, and lags-then-moves toward the online tower;
  • the predictor mask-token stays ≥2-D (upstream weight-decay parity);
  • the loss is reducible by optimization against a fixed teacher target.

The retired dual-axis PACKED machinery (``_forward_padded`` / ``backend="reference"`` /
``m_vis``/``t_kept`` / context loss / intra-inter split / whole-shaft ``frame_mask``) is
GONE (r4 contract project-r4-contract-2026-07-15), so the packed↔padded oracle,
context-head, and intra/inter tests were deleted with it.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import sample_masks
from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

N_PARCELS = 8
T = 16  # 32 Hz clock; SLOW 2 / MID 8 / HGA 16 tokens per contact.


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


def _masks(sc, geom, *, B=1, seed=0):
    n = len(sc.labels)
    g = torch.Generator().manual_seed(seed)
    return sample_masks(geom, n, n_time=T, n_rows=B, generator=g)


def _fwd(obj, bands, geom, sc, masks, **kw):
    return obj(bands, geom, sc.parcel_id, masks, **kw)


def test_teacher_emas_the_stem_too() -> None:
    # V-JEPA contract: the teacher mirrors the patch-embed (our PerBandStem), not just the
    # transformer. The teacher must own a stem that lags then moves on update.
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
    masks = _masks(sc, geom)
    torch.manual_seed(0)
    obj = V3JepaObjective(n_parcels=N_PARCELS, target_ln=True)
    with torch.no_grad():
        obj.teacher.model.encoder.norms_block[-1].weight.mul_(3.0)
        obj.teacher.model.encoder.norms_block[-1].bias.add_(2.0)
    obj.target_ln = True
    lo = _fwd(obj, bands, geom, sc, masks).loss
    obj.target_ln = False
    lf = _fwd(obj, bands, geom, sc, masks).loss
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
    # still trains end-to-end on the flat r4 path
    sc, geom = _session()
    out = _fwd(obj, _bands(len(sc.labels)), geom, sc, _masks(sc, geom))
    assert torch.isfinite(out.loss) and out.loss.requires_grad


def test_mask_token_is_weight_decayed_like_upstream() -> None:
    # Audit L36 (Ben-confirmed match-upstream): upstream stores the predictor mask
    # token 3-D (1,1,D) so the shared ndim<=1 no-decay rule DECAYS it; a 1-D (D,)
    # store would silently exempt it. v3 must keep it >=2-D → in the decay group.
    from speech_decoding.experiments.optim_param_groups import is_no_decay

    obj = _obj()
    assert obj.mask_token.ndim >= 2
    assert not is_no_decay("mask_token", obj.mask_token)  # decayed, upstream parity
    # still numerically a no-op in the forward (broadcasts to every masked token)
    sc, geom = _session()
    assert torch.isfinite(_fwd(obj, _bands(len(sc.labels)), geom, sc, _masks(sc, geom)).loss)


def test_loss_is_reducible_by_optimization() -> None:
    sc, geom = _session()
    obj = _obj()
    bands = _bands(len(sc.labels))
    masks = _masks(sc, geom)  # FIXED mask (fit the frozen teacher targets)
    opt = torch.optim.Adam([p for p in obj.parameters() if p.requires_grad], lr=1e-3)
    first = None
    for i in range(40):
        opt.zero_grad()
        loss = _fwd(obj, bands, geom, sc, masks).loss
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    assert loss.item() < first  # predictor fits fixed teacher targets
