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

import pytest

from speech_decoding.models.v14_encoder import JepaPredictor
from speech_decoding.ssl.masked_jepa import (
    b37_m4_freq_loss,
    p1_frontend_m2_loss,
    p2_parcel_m4_loss,
)


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


# --------------------------------------------------------------------------- #
# B37 M4 freq-preserving loss (D3/D9) — the freq-carrying parcel reconstruction
# --------------------------------------------------------------------------- #
def _m4_predictor(d: int, K: int, F_p: int, T_p: int) -> JepaPredictor:
    """B37 M4 predictor: parcel id (learned ``query_id``) + freq id (sinusoidal
    ``query_id_2``) + time RoPE. Depth 2 per D9. Eval mode for determinism."""
    torch.manual_seed(11)
    pred = JepaPredictor(
        d_model=d, n_identity=K, hidden=16, n_heads=2, depth=2,
        max_time_patches=T_p, id_pos="learned",
        n_identity_2=F_p, id_pos_2="sinusoidal",
    )
    return pred.eval()


def _p2_predictor(d: int, n_latents: int, t_p: int) -> JepaPredictor:
    """P2 predictor: identity axis = latent slot (L), RoPE-time table for T_p."""
    torch.manual_seed(7)
    pred = JepaPredictor(
        d_model=d, n_identity=n_latents, hidden=16, n_heads=2, depth=2,
        max_time_patches=t_p,
    )
    return pred.eval()


def _tube_one_parcel(B: int, K: int, T_p: int, tubed: int = 0):
    """Tube a single parcel (all covered): that parcel = target (whole field),
    the rest = visible context. Returns ``(visible, target_mask)`` (B, K, T_p)."""
    target_mask = torch.zeros(B, K, T_p, dtype=torch.bool)
    target_mask[:, tubed, :] = True
    visible = torch.zeros(B, K, T_p, dtype=torch.bool)
    visible[:, [k for k in range(K) if k != tubed], :] = True
    return visible, target_mask


def test_b37_m4_reconstructs_full_freq_and_time_field() -> None:
    """A tubed parcel's WHOLE ``(F_p, T_p)`` field is reconstructed — n_masked
    counts every freq × time cell of the tubed parcel, proving the freq axis is
    carried (vs the parcel-pooled :func:`p2_parcel_m4_loss`, which would only
    count T_p cells per tubed parcel)."""
    torch.manual_seed(0)
    B, K, F_p, T_p, d = 2, 5, 4, 4, 8
    student = torch.randn(B, K, F_p, T_p, d)
    teacher = torch.randn(B, K, F_p, T_p, d)
    visible, target_mask = _tube_one_parcel(B, K, T_p, tubed=0)
    pred = _m4_predictor(d, K, F_p, T_p)

    bd = b37_m4_freq_loss(
        predictor=pred, student_m4=student, teacher_m4=teacher,
        visible=visible, target_mask=target_mask,
    )
    assert bd.phase == "m4_freq"
    # one tubed parcel × all freq × all time, for every clip in the batch.
    assert bd.n_masked == B * F_p * T_p
    assert torch.isfinite(bd.total)


def test_b37_m4_loss_predictor_gets_gradient() -> None:
    """Paradigm B: the M4 prediction comes from the predictor, so its params
    receive a finite gradient (and the loss is not a bare self-distill)."""
    torch.manual_seed(0)
    B, K, F_p, T_p, d = 2, 4, 3, 4, 8
    student = torch.randn(B, K, F_p, T_p, d, requires_grad=True)
    teacher = torch.randn(B, K, F_p, T_p, d)
    visible, target_mask = _tube_one_parcel(B, K, T_p, tubed=1)
    pred = _m4_predictor(d, K, F_p, T_p)

    bd = b37_m4_freq_loss(
        predictor=pred, student_m4=student, teacher_m4=teacher,
        visible=visible, target_mask=target_mask,
    )
    bd.total.backward()
    got_grad = [
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for n, p in pred.named_parameters()
        if "id_embed" not in n  # learned id rows only update for sampled ids
    ]
    assert got_grad and all(got_grad), "M4 predictor params did not receive gradient"
    # gradient also reaches the student (the visible context), not just teacher.
    assert student.grad is not None and student.grad.abs().sum() > 0


def test_b37_m4_visible_target_split_is_wired() -> None:
    """The tube split is honored exactly:

    (a) garbaging a TUBED parcel's STUDENT value is a no-op — tubed parcels are
        excluded from the visible context (kpm) and the target is the TEACHER;
    (b) garbaging a VISIBLE parcel's STUDENT value DOES move the loss — visible
        parcels are the predictor's context;
    (c) garbaging the TUBED parcel's TEACHER value moves the loss — it is the
        reconstruction target;
    (d) garbaging a VISIBLE parcel's TEACHER value is a no-op — only tubed cells
        are targets, and the teacher is never used as context.
    """
    torch.manual_seed(0)
    B, K, F_p, T_p, d = 2, 5, 4, 4, 8
    student = torch.randn(B, K, F_p, T_p, d)
    teacher = torch.randn(B, K, F_p, T_p, d)
    visible, target_mask = _tube_one_parcel(B, K, T_p, tubed=0)
    pred = _m4_predictor(d, K, F_p, T_p)

    def run(s, t):
        return b37_m4_freq_loss(
            predictor=pred, student_m4=s, teacher_m4=t,
            visible=visible, target_mask=target_mask,
        ).total

    base = run(student, teacher)

    # (a) tubed-parcel student perturbation → no change.
    s_tube = student.clone(); s_tube[:, 0] += 9.0
    torch.testing.assert_close(run(s_tube, teacher), base, atol=0, rtol=0)
    # (b) visible-parcel student perturbation → change.
    s_vis = student.clone(); s_vis[:, 1] += 9.0
    assert not torch.allclose(run(s_vis, teacher), base)
    # (c) tubed-parcel teacher perturbation → change (it is the target).
    t_tube = teacher.clone(); t_tube[:, 0] += 9.0
    assert not torch.allclose(run(student, t_tube), base)
    # (d) visible-parcel teacher perturbation → no change.
    t_vis = teacher.clone(); t_vis[:, 1] += 9.0
    torch.testing.assert_close(run(student, t_vis), base, atol=0, rtol=0)


def test_b37_m4_empty_tube_is_exact_zero() -> None:
    """No tubed parcels (all visible) → an exact-0 loss graph-connected to the
    predictor (B6 masked-empty contract — no NaN)."""
    torch.manual_seed(0)
    B, K, F_p, T_p, d = 2, 4, 3, 4, 8
    student = torch.randn(B, K, F_p, T_p, d, requires_grad=True)
    teacher = torch.randn(B, K, F_p, T_p, d)
    visible = torch.ones(B, K, T_p, dtype=torch.bool)
    target_mask = torch.zeros(B, K, T_p, dtype=torch.bool)
    pred = _m4_predictor(d, K, F_p, T_p)

    bd = b37_m4_freq_loss(
        predictor=pred, student_m4=student, teacher_m4=teacher,
        visible=visible, target_mask=target_mask,
    )
    assert bd.n_masked == 0
    assert float(bd.total.detach()) == 0.0
    bd.total.backward()  # must not raise / NaN


# ---------------------------------------------------------------------------
# Objective integrity (#12) — null-control / mask↔loss agreement. The loss must
# read EXACTLY the masked target set, disjoint from the visible context. These
# are pre-dispatch gates: a silent visible-leak or wrong-cell gather inverts the
# SSL objective (the predictor can cheat) with no error raised.
# ---------------------------------------------------------------------------


@pytest.mark.must_pass_before_dispatch
def test_p1_loss_reads_exactly_the_masked_target_set() -> None:
    """M2/P1 null-control: hold the student + predictor fixed and corrupt ONLY
    the teacher target. Corruption at MASKED cells must move the loss (the loss
    reads them); corruption at VISIBLE cells must leave the loss BIT-IDENTICAL
    (the loss never reads them). Proves the L1 covers exactly the masked set,
    disjoint from the visible context — a visible-leak or wrong-cell gather fails
    this."""
    torch.manual_seed(0)
    B, C, F_p, T_p, d = 2, 3, 4, 4, 8
    student = torch.randn(B, C, F_p, T_p, d)
    teacher = torch.randn(B, C, F_p, T_p, d)
    token_mask = torch.rand(B, C, F_p, T_p) < 0.5
    token_mask[0, 0, 0, 0] = True   # guarantee ≥1 masked
    token_mask[0, 0, 0, 1] = False  # guarantee ≥1 visible
    pred = _p1_predictor(d, F_p, T_p)

    def loss_of(t: torch.Tensor) -> torch.Tensor:
        return p1_frontend_m2_loss(
            predictor=pred, student_m2=student, teacher_m2=t, token_mask=token_mask,
        ).total.detach()

    base = loss_of(teacher)

    t_vis = teacher.clone()
    t_vis[~token_mask] += 100.0  # corrupt VISIBLE cells only
    torch.testing.assert_close(base, loss_of(t_vis), rtol=0, atol=0)

    t_msk = teacher.clone()
    t_msk[token_mask] += 100.0  # corrupt MASKED cells only
    assert (base - loss_of(t_msk)).abs() > 1.0, (
        "P1 loss did not respond to the masked target — it is not reading the "
        "masked set (wrong-cell gather or visible leak)"
    )


@pytest.mark.must_pass_before_dispatch
def test_p2_loss_reads_exactly_the_masked_target_set() -> None:
    """M4/P2 null-control: same as P1 for the parcel-time tube mask. Corrupting
    teacher_m4 at the target (masked-covered) cells moves the loss; corrupting at
    non-target (visible / uncovered) cells is bit-identical."""
    torch.manual_seed(0)
    B, L, T_p, d = 2, 6, 4, 8
    student = torch.randn(B, L, T_p, d)
    teacher = torch.randn(B, L, T_p, d)
    mask = torch.rand(B, L, T_p) < 0.4
    mask[0, 0, 0] = True   # guarantee ≥1 target
    mask[0, 0, 1] = False  # guarantee ≥1 visible
    covered = torch.ones(B, L, T_p, dtype=torch.bool)
    target_mask = covered & mask
    visible = covered & ~mask
    pred = _p2_predictor(d, L, T_p)

    def loss_of(t: torch.Tensor) -> torch.Tensor:
        return p2_parcel_m4_loss(
            predictor=pred, student_m4=student, teacher_m4=t,
            visible=visible, target_mask=target_mask,
        ).total.detach()

    base = loss_of(teacher)

    t_non = teacher.clone()
    t_non[~target_mask] += 100.0  # corrupt non-target cells only
    torch.testing.assert_close(base, loss_of(t_non), rtol=0, atol=0)

    t_tgt = teacher.clone()
    t_tgt[target_mask] += 100.0  # corrupt target cells only
    assert (base - loss_of(t_tgt)).abs() > 1.0, (
        "P2 loss did not respond to the masked parcel-time target — wrong-cell "
        "gather or visible leak"
    )
