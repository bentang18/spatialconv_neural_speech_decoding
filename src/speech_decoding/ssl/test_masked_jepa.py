"""Tests for the B36 masked-JEPA loss terms.

Both phases are paradigm B (visible-only student + separate ``JepaPredictor`` +
EMA full-input teacher target + L1). Here we pin (a) the P1 paradigm-B contract
— the loss flows through the predictor, not a bare ``student[masked]`` vs
``teacher[masked]`` self-distill — and (b) the C5 per-corpus freq-patch
exclusion. The end-to-end P1/P2 wiring is exercised in
``experiments/test_v14_joint_module.py``.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import JepaPredictor
from speech_decoding.ssl.masked_jepa import p1_frontend_m2_loss


def _p1_predictor(d: int, f_p: int, t_p: int) -> JepaPredictor:
    """A small P1 predictor: identity axis = freq-patch (F_p), RoPE-time table
    sized for the test's T_p. Eval mode for deterministic forward."""
    torch.manual_seed(7)
    pred = JepaPredictor(
        d_model=d, n_identity=f_p, hidden=16, n_heads=2, depth=2,
        max_time_patches=t_p,
    )
    return pred.eval()


def test_p1_is_paradigm_b_predictor_gets_gradient() -> None:
    """The P1 loss is paradigm B: the prediction comes from the predictor, so
    (a) the predictor's params receive gradient, and (b) the loss is NOT the
    bare ``L1(student[masked], sg teacher[masked])`` self-distill (which would
    be predictor-independent — the regression that caused the P1 collapse)."""
    torch.manual_seed(0)
    B, C, F_p, T_p, d = 2, 3, 4, 4, 8
    student = torch.randn(B, C, F_p, T_p, d, requires_grad=True)
    teacher = torch.randn(B, C, F_p, T_p, d)
    token_mask = torch.rand(B, C, F_p, T_p) < 0.5
    pred = _p1_predictor(d, F_p, T_p)

    bd = p1_frontend_m2_loss(
        predictor=pred, student_m2=student, teacher_m2=teacher,
        token_mask=token_mask,
    )
    bd.total.backward()

    # (a) every predictor param that participates gets a finite gradient.
    got_grad = [
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for n, p in pred.named_parameters()
        if "id_embed" not in n  # id_embed rows only update for sampled freq ids
    ]
    assert got_grad and all(got_grad), "predictor params did not receive gradient"

    # (b) the loss is predictor-mediated: it differs from the paradigm-A
    # self-distill value (student[masked] vs sg teacher[masked]).
    sg_self_distill = (
        (student.detach()[token_mask] - teacher[token_mask]).abs().mean()
    )
    assert not torch.allclose(bd.total.detach(), sg_self_distill), (
        "P1 loss collapsed to the paradigm-A self-distill — predictor inert"
    )


def test_p1_predictor_context_is_time_aware_and_content_sensitive() -> None:
    """Regression guards for (Lens-1) the RoPE-time fix — the old
    ``context_pos=None`` time-blindness — and (Lens-4) the paradigm-B contract
    that the prediction must READ the visible context (a context-blind constant
    predictor is the dead-front-end failure mode the fix prevents):

    (a) changing ONLY ``context_time_ids`` changes the output ⇒ context tokens
        carry their time via RoPE (not time-blind);
    (b) perturbing a context VALUE changes the output ⇒ the prediction reads the
        visible context (a no-op / constant predictor would not).
    """
    torch.manual_seed(0)
    d, F_p, T_p, B = 8, 3, 8, 2
    pred = JepaPredictor(
        d_model=d, n_identity=F_p, hidden=16, n_heads=2, depth=2,
        max_time_patches=T_p,
    ).eval()
    n_ctx = 5
    context = torch.randn(B, n_ctx, d)
    ctx_t = torch.tensor([0, 2, 4, 6, 1])
    kw = dict(query_time_ids=torch.tensor([1, 3, 5, 7]),
              query_id=torch.tensor([0, 1, 2, 0]))

    base = pred(context, context_time_ids=ctx_t, **kw)

    # (a) permute the context times (identical values + order) → different RoPE.
    permuted = pred(context, context_time_ids=torch.tensor([6, 4, 2, 0, 1]), **kw)
    assert not torch.allclose(base, permuted), (
        "predictor is time-blind: changing context_time_ids did not change the "
        "prediction (the context_pos=None regression this build fixed)"
    )

    # (b) perturb one context VALUE → different prediction (context-sensitive).
    bumped = context.clone()
    bumped[:, 0, :] += 5.0
    out_bumped = pred(bumped, context_time_ids=ctx_t, **kw)
    assert not torch.allclose(base, out_bumped), (
        "predictor ignores the visible context (a constant / no-op predictor — "
        "the dead-front-end failure mode paradigm-B prevents)"
    )


def test_c5_p1_loss_excludes_invalid_freq_patches() -> None:
    """With freq_patch_valid masking F-patches 7–9, those cells are excluded
    from BOTH the target and the visible context: ``n_masked`` counts only
    valid-patch masked cells, and garbaging the student/teacher values on the
    invalid patches cannot change the loss (they never participate)."""
    torch.manual_seed(0)
    B, C, F_p, T_p, d = 2, 3, 10, 4, 8
    student = torch.randn(B, C, F_p, T_p, d)
    teacher = torch.randn(B, C, F_p, T_p, d)
    token_mask = torch.rand(B, C, F_p, T_p) < 0.5  # ~half masked
    pred = _p1_predictor(d, F_p, T_p)

    fpv = torch.tensor([True] * 7 + [False] * 3)
    bd = p1_frontend_m2_loss(
        predictor=pred, student_m2=student, teacher_m2=teacher,
        token_mask=token_mask, freq_patch_valid=fpv,
    )
    valid_masked = token_mask & fpv.view(1, 1, F_p, 1)
    assert bd.n_masked == int(valid_masked.sum())
    assert bd.n_masked < int(token_mask.sum()), "some invalid-patch cells dropped"

    # Invalid F-patches 7–9 are neither targets nor context keys → garbaging
    # those student / teacher values must NOT change the loss.
    s2 = student.clone()
    s2[:, :, 7:] = 123.0
    t2 = teacher.clone()
    t2[:, :, 7:] = -77.0
    bd2 = p1_frontend_m2_loss(
        predictor=pred, student_m2=s2, teacher_m2=t2,
        token_mask=token_mask, freq_patch_valid=fpv,
    )
    torch.testing.assert_close(bd2.total, bd.total, atol=0, rtol=0)


def test_c5_p1_loss_none_is_byte_identical() -> None:
    """freq_patch_valid=None and an all-True mask both reduce to the no-C5
    loss — the BT no-op (all freq patches valid). Same predictor → identical."""
    torch.manual_seed(1)
    B, C, F_p, T_p, d = 2, 3, 10, 4, 8
    student = torch.randn(B, C, F_p, T_p, d)
    teacher = torch.randn(B, C, F_p, T_p, d)
    token_mask = torch.rand(B, C, F_p, T_p) < 0.5
    pred = _p1_predictor(d, F_p, T_p)

    base = p1_frontend_m2_loss(
        predictor=pred, student_m2=student, teacher_m2=teacher,
        token_mask=token_mask,
    )
    allt = p1_frontend_m2_loss(
        predictor=pred, student_m2=student, teacher_m2=teacher,
        token_mask=token_mask, freq_patch_valid=torch.ones(F_p, dtype=torch.bool),
    )
    assert base.n_masked == allt.n_masked
    torch.testing.assert_close(base.total, allt.total, atol=0, rtol=0)


def test_c5_p1_loss_accepts_batched_freq_patch_valid() -> None:
    """A per-sample (B, F_p) freq_patch_valid is honored independently per
    clip (so a future mixed-corpus batch masks each clip by its own bins)."""
    torch.manual_seed(2)
    B, C, F_p, T_p, d = 2, 3, 10, 4, 8
    student = torch.randn(B, C, F_p, T_p, d)
    teacher = torch.randn(B, C, F_p, T_p, d)
    token_mask = torch.ones(B, C, F_p, T_p, dtype=torch.bool)
    pred = _p1_predictor(d, F_p, T_p)

    fpv = torch.ones(B, F_p, dtype=torch.bool)
    fpv[0, 7:] = False  # clip 0 = SWEC (7–9 invalid); clip 1 = all valid
    bd = p1_frontend_m2_loss(
        predictor=pred, student_m2=student, teacher_m2=teacher,
        token_mask=token_mask, freq_patch_valid=fpv,
    )
    expected = C * 7 * T_p + C * F_p * T_p  # clip0 valid patches + clip1 all
    assert bd.n_masked == expected
