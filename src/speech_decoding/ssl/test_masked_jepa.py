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
    _jepa_pred_target_stats,
    _m4_precision_monitor_stats,
    _m4_precision_weight,
    _precision_weight_downweight,
    b37_m4_freq_loss,
    m2_dual_band_loss,
    m4_dual_band_loss,
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
# Heteroscedastic / inverse-variance precision weighting on M4
# (project_v14_heteroscedastic_ssl_loss). w = n_k^α / (σ²+σ²₀), DETACHED, mean-1,
# σ²₀ = in-batch shrinkage prior (p{floor_pct} of scored σ²), capped.
# ---------------------------------------------------------------------------


def test_m4_precision_weight_is_mean1_detached_and_ordered() -> None:
    """The gathered weight is mean-1, detached, finite, and orders cells by
    precision: high-n/low-σ parcels outweigh low-n/high-σ parcels."""
    B, K, F_p, T_p = 1, 3, 2, 2
    N = K * F_p * T_p
    n = torch.tensor([[10.0, 4.0, 1.0]])          # parcel 0 high n → parcel 2 low n
    std = torch.ones(B, K, F_p, T_p)
    std[0, 0] = 0.1                                # parcel 0 low σ (most reliable)
    std[0, 2] = 2.0                                # parcel 2 high σ (least reliable)
    target_cell = torch.ones(B, N, dtype=torch.bool)  # score every cell
    w = _m4_precision_weight(
        std, n, target_cell, alpha=1.0, eps=1e-6, floor_pct=25.0, cap=10.0,
        mode="mean1_invvar",  # the mean-1 inverse-variance form (R-precision-mean1)
    )

    assert torch.isfinite(w).all()
    assert not w.requires_grad                     # detached — no grad path via σ/n
    assert abs(w.mean().item() - 1.0) < 1e-5       # mean-1 normalized (scale-preserving)
    w_grid = w.reshape(K, F_p, T_p)
    assert w_grid[0].mean() > w_grid[1].mean() > w_grid[2].mean()


def test_m4_precision_weight_alpha_zero_drops_count_term() -> None:
    """α=0 → n^0=1, so the weight depends only on 1/(σ²+σ²₀): two parcels with
    equal σ but different n get equal weight."""
    B, K, F_p, T_p = 1, 2, 1, 1
    N = K * F_p * T_p
    n = torch.tensor([[8.0, 1.0]])                 # different counts
    std = torch.full((B, K, F_p, T_p), 0.5)        # equal σ
    target_cell = torch.ones(B, N, dtype=torch.bool)
    w = _m4_precision_weight(
        std, n, target_cell, alpha=0.0, eps=1e-6, floor_pct=25.0, cap=10.0,
        mode="mean1_invvar",
    )
    assert torch.allclose(w, torch.ones_like(w), atol=1e-6)


def test_m4_precision_shrinkage_floor_tames_degenerate_cells() -> None:
    """THE fix (2026-06-13): single-electrode / near-equal-electrode parcels have
    σ²≈0 — they are the SHAKIEST targets, yet bare 1/(σ²+ε) gives them runaway
    weight that swamps the informative cells. The in-batch shrinkage prior σ²₀
    must bound the max/median weight ratio to O(1), where a bare-ε floor leaves it
    at O(1e4+). Also asserts the cap alone is NOT the fix (bulk mass, not outliers)."""
    B, K, F_p, T_p = 1, 8, 2, 2
    N = K * F_p * T_p
    n = torch.full((B, K), 6.0)                    # all parcels well-covered on n
    std = torch.full((B, K, F_p, T_p), 0.7)        # typical σ (σ²≈0.49)
    std[0, 0] = 1e-4                               # parcel 0: degenerate σ→0 (σ²=1e-8)
    std[0, 1] = 1e-4                               # parcel 1: degenerate too (~12.5% of cells)
    target_cell = torch.ones(B, N, dtype=torch.bool)

    # Bare ε floor (old behavior: floor_pct=0 → σ²₀=min≈ε, cap off): degenerate
    # cells dominate → enormous max/median.
    w_bare = _m4_precision_weight(
        std, n, target_cell, alpha=0.5, eps=1e-6, floor_pct=0.0, cap=0.0,
        mode="mean1_invvar",
    )
    ratio_bare = w_bare.max() / w_bare.median()
    # Cap alone (still bare floor): does NOT fix it — the bulk degenerate mass
    # drives the median to ~0, so max/median stays huge even after capping max.
    w_cap = _m4_precision_weight(
        std, n, target_cell, alpha=0.5, eps=1e-6, floor_pct=0.0, cap=10.0,
        mode="mean1_invvar",
    )
    ratio_cap = w_cap.max() / w_cap.median()
    # Shrinkage floor (the fix): σ²₀ = p25 of in-batch σ² ≈ 0.49 → degenerate
    # cells pulled back to a typical-cell weight → bounded ratio.
    w_fix = _m4_precision_weight(
        std, n, target_cell, alpha=0.5, eps=1e-6, floor_pct=25.0, cap=10.0,
        mode="mean1_invvar",
    )
    ratio_fix = w_fix.max() / w_fix.median()

    assert ratio_bare > 1e3                        # the pathology is real
    assert ratio_cap > 1e3                         # cap alone does NOT fix it
    assert ratio_fix < 20.0                        # shrinkage floor tames it
    assert torch.isfinite(w_fix).all()
    assert abs(w_fix.mean().item() - 1.0) < 1e-5   # still mean-1 after cap+renorm


def test_b37_m4_precision_uniform_stats_recovers_plain_l1() -> None:
    """Uniform σ + uniform n → mean-1 weight is all-ones → the weighted loss is
    byte-identical to the plain-L1 path (the safe-ship uniform-weight limit)."""
    torch.manual_seed(0)
    B, K, F_p, T_p, d = 2, 5, 4, 4, 8
    student = torch.randn(B, K, F_p, T_p, d)
    teacher = torch.randn(B, K, F_p, T_p, d)
    visible, target_mask = _tube_one_parcel(B, K, T_p, tubed=0)
    pred = _m4_predictor(d, K, F_p, T_p)           # eval → deterministic

    plain = b37_m4_freq_loss(
        predictor=pred, student_m4=student, teacher_m4=teacher,
        visible=visible, target_mask=target_mask,
    )
    std = torch.full((B, K, F_p, T_p), 0.37)
    n = torch.full((B, K), 5.0)
    weighted = b37_m4_freq_loss(
        predictor=pred, student_m4=student, teacher_m4=teacher,
        visible=visible, target_mask=target_mask,
        precision_std=std, precision_n=n, precision_alpha=1.0,
        precision_mode="mean1_invvar",  # mean-1 form: uniform stats → all-ones
    )
    assert torch.allclose(plain.total, weighted.total, atol=1e-6)


def test_b37_m4_precision_nonuniform_changes_loss() -> None:
    """Non-uniform σ across the tubed parcel's cells re-weights the L1 → the
    weighted loss differs from plain L1 (the weighting is actually applied)."""
    torch.manual_seed(0)
    B, K, F_p, T_p, d = 2, 5, 4, 4, 8
    student = torch.randn(B, K, F_p, T_p, d)
    teacher = torch.randn(B, K, F_p, T_p, d)
    visible, target_mask = _tube_one_parcel(B, K, T_p, tubed=0)
    pred = _m4_predictor(d, K, F_p, T_p)

    plain = b37_m4_freq_loss(
        predictor=pred, student_m4=student, teacher_m4=teacher,
        visible=visible, target_mask=target_mask,
    )
    std = torch.rand(B, K, F_p, T_p) + 0.1         # non-uniform σ
    n = torch.full((B, K), 5.0)
    weighted = b37_m4_freq_loss(
        predictor=pred, student_m4=student, teacher_m4=teacher,
        visible=visible, target_mask=target_mask,
        precision_std=std, precision_n=n, precision_alpha=1.0,
        precision_mode="mean1_invvar",  # σ re-weighting is a mean1-form property
    )
    assert torch.isfinite(weighted.total)
    assert not torch.allclose(plain.total, weighted.total, atol=1e-4)


def test_b37_m4_precision_requires_both_std_and_n() -> None:
    """Passing only one of (std, n) is a usage error (both or neither)."""
    torch.manual_seed(0)
    B, K, F_p, T_p, d = 1, 3, 2, 2, 8
    student = torch.randn(B, K, F_p, T_p, d)
    teacher = torch.randn(B, K, F_p, T_p, d)
    visible, target_mask = _tube_one_parcel(B, K, T_p, tubed=0)
    pred = _m4_predictor(d, K, F_p, T_p)
    std = torch.ones(B, K, F_p, T_p)
    with pytest.raises(ValueError, match="BOTH"):
        b37_m4_freq_loss(
            predictor=pred, student_m4=student, teacher_m4=teacher,
            visible=visible, target_mask=target_mask, precision_std=std,
        )


# ---------------------------------------------------------------------------
# Downweight-only electrode-dof precision weight — the NEW default
# (reports/m4_precision_downweight_handoff_2026_06_15.md). w = min(1, ((n-1)/
# (n_ref-1))^α): n-only, NOT mean-1 (sub-1 mean by design ≈ a principled λ_m4),
# 0 at n=1, saturates at n_ref. σ is NOT read.
# ---------------------------------------------------------------------------


def test_precision_downweight_weight_table() -> None:
    """The acceptance table (n_ref=11, α=1.0): w = min(1, (n-1)/10) for
    n ∈ {1,2,3,6,11,15} → [0.0, 0.1, 0.2, 0.5, 1.0, 1.0] (n≥n_ref caps at 1)."""
    n_cell = torch.tensor([[1.0, 2.0, 3.0, 6.0, 11.0, 15.0]])
    target = torch.ones_like(n_cell, dtype=torch.bool)
    w = _precision_weight_downweight(n_cell, target, n_ref=11.0, alpha=1.0)
    expected = torch.tensor([0.0, 0.1, 0.2, 0.5, 1.0, 1.0])
    torch.testing.assert_close(w, expected, atol=1e-6, rtol=0.0)
    assert not w.requires_grad                          # detached


def test_precision_downweight_zero_at_n1_and_caps_at_nref() -> None:
    """Passes through 0 at n=1 (single-electrode parcels carry no std → excluded
    upstream) and saturates at exactly 1.0 for n ≥ n_ref."""
    n_cell = torch.tensor([[1.0, 11.0, 50.0]])
    target = torch.ones_like(n_cell, dtype=torch.bool)
    w = _precision_weight_downweight(n_cell, target, n_ref=11.0, alpha=1.0)
    assert w[0].item() == 0.0
    assert w[1].item() == 1.0
    assert w[2].item() == 1.0                           # min(1, 49/10) = 1


def test_precision_downweight_mean_below_one_no_renorm() -> None:
    """The defining contrast with the mean-1 core: a mixed-n batch (all < n_ref)
    yields weight.mean() < 1 — shaky parcels contribute less and are NOT
    compensated by upweighting others (no mean-1 renormalization)."""
    n_cell = torch.tensor([[2.0, 3.0, 4.0, 5.0]])       # all < n_ref → all < 1
    target = torch.ones_like(n_cell, dtype=torch.bool)
    w = _precision_weight_downweight(n_cell, target, n_ref=11.0, alpha=1.0)
    assert w.mean().item() < 1.0
    assert abs(w.mean().item() - 0.25) < 1e-6           # (0.1+0.2+0.3+0.4)/4


def test_precision_downweight_ignores_sigma() -> None:
    """σ-independence: in downweight_dof, two batches with different precision_std
    but identical precision_n give identical weights (n-only form)."""
    B, K, F_p, T_p = 1, 3, 2, 2
    N = K * F_p * T_p
    n = torch.tensor([[10.0, 4.0, 2.0]])
    target = torch.ones(B, N, dtype=torch.bool)
    std_a = torch.rand(B, K, F_p, T_p) + 0.1
    std_b = torch.rand(B, K, F_p, T_p) + 5.0            # wildly different σ
    kw = dict(alpha=1.0, eps=1e-6, floor_pct=25.0, cap=10.0,
              mode="downweight_dof", n_ref=11.0)
    w_a = _m4_precision_weight(std_a, n, target, **kw)
    w_b = _m4_precision_weight(std_b, n, target, **kw)
    torch.testing.assert_close(w_a, w_b)


def test_precision_downweight_differs_from_mean1_and_is_subone() -> None:
    """Mode divergence + mean1 regression sanity: the two forms produce DIFFERENT
    weights; mean1_invvar stays mean-1 (its defining property — the byte-identical
    R-precision-mean1 path), downweight_dof is sub-1."""
    B, K, F_p, T_p = 1, 4, 2, 2
    N = K * F_p * T_p
    n = torch.tensor([[10.0, 6.0, 3.0, 2.0]])
    std = torch.rand(B, K, F_p, T_p) + 0.2
    target = torch.ones(B, N, dtype=torch.bool)
    kw = dict(alpha=1.0, eps=1e-6, floor_pct=25.0, cap=10.0, n_ref=11.0)
    w_dof = _m4_precision_weight(std, n, target, mode="downweight_dof", **kw)
    w_m1 = _m4_precision_weight(std, n, target, mode="mean1_invvar", **kw)
    assert abs(w_m1.mean().item() - 1.0) < 1e-5         # mean1 path stays mean-1
    assert w_dof.mean().item() < 1.0                    # downweight is sub-1
    assert not torch.allclose(w_dof, w_m1)              # the two forms diverge


def test_precision_downweight_alpha_robust_overlay() -> None:
    """α>1 is the risk-averse overlay: α=1.15 at n=2 gives (1/10)^1.15 ≈ 0.0708."""
    n_cell = torch.tensor([[2.0]])
    target = torch.ones_like(n_cell, dtype=torch.bool)
    w = _precision_weight_downweight(n_cell, target, n_ref=11.0, alpha=1.15)
    assert abs(w.item() - 0.1 ** 1.15) < 1e-6


def test_b37_m4_precision_downweight_default_smaller_than_plain() -> None:
    """End-to-end (b37): the DEFAULT downweight_dof on an all-low-n fixture yields a
    finite loss strictly smaller than the unweighted path (sub-1 weights scale the
    L1 down — the principled λ_m4 effect)."""
    torch.manual_seed(0)
    B, K, F_p, T_p, d = 2, 5, 4, 4, 8
    student = torch.randn(B, K, F_p, T_p, d)
    teacher = torch.randn(B, K, F_p, T_p, d)
    visible, target_mask = _tube_one_parcel(B, K, T_p, tubed=0)
    pred = _m4_predictor(d, K, F_p, T_p)

    plain = b37_m4_freq_loss(
        predictor=pred, student_m4=student, teacher_m4=teacher,
        visible=visible, target_mask=target_mask,
    )
    std = torch.full((B, K, F_p, T_p), 0.4)
    n = torch.full((B, K), 3.0)                          # low n → w = (3-1)/10 = 0.2
    weighted = b37_m4_freq_loss(                         # default mode = downweight_dof
        predictor=pred, student_m4=student, teacher_m4=teacher,
        visible=visible, target_mask=target_mask,
        precision_std=std, precision_n=n,
    )
    assert torch.isfinite(weighted.total)
    assert weighted.total < plain.total


def test_m4_dual_band_precision_downweight_default_smaller_than_plain() -> None:
    """End-to-end (2STFT dual-band M4): same as the b37 case — the default
    downweight_dof on a low-n fixture is finite and smaller than the plain L1."""
    torch.manual_seed(0)
    B, K, d = 2, 5, 8
    student = torch.randn(B, K, _M2DB_S, d)
    teacher = torch.randn(B, K, _M2DB_S, d)
    tubed = _tube_bk(B, K, tubed=0)
    lv = torch.ones(B, K, dtype=torch.bool)
    pred = _m4_dual_predictor(d, K)

    plain = _m4db_call(pred, student, teacher, tubed, lv)
    std = torch.full((B, K, _M2DB_S), 0.4)
    n = torch.full((B, K), 3.0)
    weighted = _m4db_call(                               # default mode = downweight_dof
        pred, student, teacher, tubed, lv,
        precision_std=std, precision_n=n,
    )
    assert torch.isfinite(weighted.total)
    assert weighted.total < plain.total


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


# ═══════════════════ FE 2STFT §6 — M2 dual-band drop-token loss ═══════════════
#
# The 2STFT M2 tap is a FLAT per-parcel sequence (B, K, S, d) — the two bands
# carry different time rates, so no shared (F_p, T_p) grid. ONE predictor
# reconstructs all masked slots: parcel = learned query_id, freq-patch = sincos
# query_id_2 (the "Hz tag", basket implicit), slot = dual-rate RoPE time. Loss =
# pure L1 pooled mean over masked tokens; latent_valid gates uncovered parcels.

# Small synthetic dual-band geometry for the loss unit tests:
#   low  F_p_low=2 × T_low_p=2 → S_low=4 (flat t-outer/f-inner)
#   high F_p_high=1 × T_high_p=4 → S_high=4 ; S=8, F_p=3 (low 0,1 / high 2)
# slot: low t→2t, high t→t ; freq-patch: low f∈{0,1}, high f=2.
_M2DB_S = 8
_M2DB_FP = 3
_M2DB_SLOT = torch.tensor([0, 0, 2, 2, 0, 1, 2, 3])        # (S,) dual-rate slots
_M2DB_FREQ = torch.tensor([0, 1, 0, 1, 2, 2, 2, 2])        # (S,) freq-patch / Hz tag


def _m2_dual_predictor(d: int, K: int) -> JepaPredictor:
    """2STFT M2 predictor — built IDENTICALLY to the B37 M4 predictor: parcel
    (learned query_id) + freq-patch (sinusoidal query_id_2) + time RoPE. ONE
    predictor for both baskets. Eval mode for determinism."""
    torch.manual_seed(13)
    pred = JepaPredictor(
        d_model=d, n_identity=K, hidden=16, n_heads=2, depth=2,
        max_time_patches=8, id_pos="learned",
        n_identity_2=_M2DB_FP, id_pos_2="sinusoidal",
    )
    return pred.eval()


def _m2db_call(pred, student, teacher, token_mask, latent_valid):
    return m2_dual_band_loss(
        predictor=pred, student_m2=student, teacher_m2=teacher,
        token_mask=token_mask, slot_ids=_M2DB_SLOT, freq_patch_ids=_M2DB_FREQ,
        latent_valid=latent_valid,
    )


def test_m2_dual_band_n_masked_counts_covered_masked_only() -> None:
    """n_masked = (token_mask & covered) — uncovered parcels contribute no
    targets even where masked; phase tag is ``m2_dual_band``."""
    torch.manual_seed(0)
    B, K, d = 2, 4, 8
    student = torch.randn(B, K, _M2DB_S, d)
    teacher = torch.randn(B, K, _M2DB_S, d)
    token_mask = torch.zeros(B, K, _M2DB_S, dtype=torch.bool)
    token_mask[:, :, [0, 2, 5]] = True                     # 3 tokens / parcel masked
    latent_valid = torch.ones(B, K, dtype=torch.bool)
    latent_valid[:, K - 1] = False                         # last parcel uncovered
    pred = _m2_dual_predictor(d, K)

    bd = _m2db_call(pred, student, teacher, token_mask, latent_valid)
    assert bd.phase == "m2_dual_band"
    # 3 masked tokens × (K-1) covered parcels × B clips.
    assert bd.n_masked == B * (K - 1) * 3
    assert torch.isfinite(bd.total)


def test_m2_dual_band_predictor_and_student_get_gradient() -> None:
    """Paradigm B: the prediction is the predictor's, so its params get a finite
    gradient, and the student (visible context) does too — not a bare distill."""
    torch.manual_seed(0)
    B, K, d = 2, 4, 8
    student = torch.randn(B, K, _M2DB_S, d, requires_grad=True)
    teacher = torch.randn(B, K, _M2DB_S, d)
    token_mask = torch.zeros(B, K, _M2DB_S, dtype=torch.bool)
    token_mask[:, :, [1, 3, 6]] = True
    latent_valid = torch.ones(B, K, dtype=torch.bool)
    pred = _m2_dual_predictor(d, K)

    bd = _m2db_call(pred, student, teacher, token_mask, latent_valid)
    bd.total.backward()
    got = [
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for n, p in pred.named_parameters()
        if "id_embed" not in n  # learned parcel rows only update for sampled ids
    ]
    assert got and all(got), "M2 dual-band predictor params did not get gradient"
    assert student.grad is not None and student.grad.abs().sum() > 0


def test_m2_dual_band_visible_target_split_and_detach() -> None:
    """The drop-token split is honored exactly (mirrors the B37 M4 split test):

    (a) garbaging a MASKED slot's STUDENT value → no-op (masked tokens are
        key-padded out of the predictor context; the target is the TEACHER);
    (b) garbaging a VISIBLE slot's STUDENT value → moves the loss (it is context);
    (c) garbaging a MASKED slot's TEACHER value → moves the loss (it is target);
    (d) garbaging a VISIBLE slot's TEACHER value → no-op (only masked = target,
        teacher never context);
    (e) garbaging an UNCOVERED parcel (either side) → no-op (latent_valid gate).
    """
    torch.manual_seed(0)
    B, K, d = 2, 5, 8
    student = torch.randn(B, K, _M2DB_S, d)
    teacher = torch.randn(B, K, _M2DB_S, d)
    token_mask = torch.zeros(B, K, _M2DB_S, dtype=torch.bool)
    masked_tok = [0, 2, 5]
    vis_tok = [1, 4, 7]
    token_mask[:, :, masked_tok] = True
    latent_valid = torch.ones(B, K, dtype=torch.bool)
    latent_valid[:, K - 1] = False                         # uncovered parcel
    pred = _m2_dual_predictor(d, K)

    def run(s, t):
        return _m2db_call(pred, s, t, token_mask, latent_valid).total

    base = run(student, teacher)
    # (a) masked-slot student perturbation → no change.
    s = student.clone(); s[:, 0, masked_tok] += 9.0
    torch.testing.assert_close(run(s, teacher), base, atol=0, rtol=0)
    # (b) visible-slot student perturbation → change.
    s = student.clone(); s[:, 0, vis_tok] += 9.0
    assert not torch.allclose(run(s, teacher), base)
    # (c) masked-slot teacher perturbation → change (the target).
    t = teacher.clone(); t[:, 0, masked_tok] += 9.0
    assert not torch.allclose(run(student, t), base)
    # (d) visible-slot teacher perturbation → no change.
    t = teacher.clone(); t[:, 0, vis_tok] += 9.0
    torch.testing.assert_close(run(student, t), base, atol=0, rtol=0)
    # (e) uncovered parcel, either side → no change.
    s = student.clone(); s[:, K - 1] += 9.0
    torch.testing.assert_close(run(s, teacher), base, atol=0, rtol=0)
    t = teacher.clone(); t[:, K - 1] += 9.0
    torch.testing.assert_close(run(student, t), base, atol=0, rtol=0)


def test_m2_dual_band_empty_mask_is_exact_graph_connected_zero() -> None:
    """No masked slot → exact 0 loss still connected to the predictor graph
    (B6 masked-empty contract — no NaN, optimizer sees a real zero)."""
    torch.manual_seed(0)
    B, K, d = 2, 3, 8
    student = torch.randn(B, K, _M2DB_S, d, requires_grad=True)
    teacher = torch.randn(B, K, _M2DB_S, d)
    token_mask = torch.zeros(B, K, _M2DB_S, dtype=torch.bool)  # nothing masked
    latent_valid = torch.ones(B, K, dtype=torch.bool)
    pred = _m2_dual_predictor(d, K)

    bd = _m2db_call(pred, student, teacher, token_mask, latent_valid)
    assert bd.n_masked == 0
    assert float(bd.total.detach()) == 0.0
    bd.total.backward()  # must not raise / NaN


def test_m2_dual_band_shape_guards() -> None:
    torch.manual_seed(0)
    B, K, d = 2, 3, 8
    student = torch.randn(B, K, _M2DB_S, d)
    teacher = torch.randn(B, K, _M2DB_S, d)
    lv = torch.ones(B, K, dtype=torch.bool)
    pred = _m2_dual_predictor(d, K)
    good = torch.zeros(B, K, _M2DB_S, dtype=torch.bool)

    with pytest.raises(ValueError, match="token_mask"):
        _m2db_call(pred, student, teacher, torch.zeros(B, K, _M2DB_S + 1, dtype=torch.bool), lv)
    with pytest.raises(ValueError, match="latent_valid"):
        _m2db_call(pred, student, teacher, good, torch.ones(B, K + 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="teacher_m2"):
        m2_dual_band_loss(
            predictor=pred, student_m2=student, teacher_m2=teacher[:, :, :-1],
            token_mask=good, slot_ids=_M2DB_SLOT, freq_patch_ids=_M2DB_FREQ,
            latent_valid=lv,
        )
    with pytest.raises(ValueError, match="slot_ids"):
        m2_dual_band_loss(
            predictor=pred, student_m2=student, teacher_m2=teacher,
            token_mask=good, slot_ids=_M2DB_SLOT[:-1], freq_patch_ids=_M2DB_FREQ,
            latent_valid=lv,
        )


# ---------------------------------------------------------------------------
# 2STFT M4 latent loss (m4_dual_band_loss): WHOLE-PARCEL tube on the flat-S latent.
# Sibling of b37_m4_freq_loss (F_p×T_p grid) and m2_dual_band_loss (flat-S M2).
# SEPARATE M4 predictor (depth 1 in the locked config) + opt-in heteroscedastic
# precision weight (project_v14_heteroscedastic_ssl_loss).
# ---------------------------------------------------------------------------


def _m4_dual_predictor(d: int, K: int) -> JepaPredictor:
    """2STFT M4 predictor — built like the M2 dual-band predictor (parcel learned
    query_id + freq-patch sinusoidal query_id_2 + slot RoPE) but DEPTH 1 (the
    locked-config M4 predictor depth; M2 is depth 3). Eval for determinism."""
    torch.manual_seed(17)
    pred = JepaPredictor(
        d_model=d, n_identity=K, hidden=16, n_heads=2, depth=1,
        max_time_patches=8, id_pos="learned",
        n_identity_2=_M2DB_FP, id_pos_2="sinusoidal",
    )
    return pred.eval()


def _tube_bk(B: int, K: int, tubed: int = 0) -> torch.Tensor:
    """Whole-parcel tube: parcel ``tubed`` is the target (all S), the rest survive.
    Returns the per-parcel ``(B, K)`` bool tube mask (no time axis — the parcel is
    the atomic masking unit)."""
    m = torch.zeros(B, K, dtype=torch.bool)
    m[:, tubed] = True
    return m


def _m4db_call(pred, student, teacher, tubed, latent_valid, **kw):
    # Default token_mask = all-False (nothing M2-band-masked) → visible_cell =
    # visible_parcel & ~False = visible_parcel, i.e. the pre-"never read zeros"
    # per-parcel context. Tests that exercise the band drop pass their own.
    kw.setdefault(
        "token_mask",
        torch.zeros(student.shape[0], student.shape[1], _M2DB_S, dtype=torch.bool),
    )
    return m4_dual_band_loss(
        predictor=pred, student_m4=student, teacher_m4=teacher,
        tubed=tubed, latent_valid=latent_valid,
        slot_ids=_M2DB_SLOT, freq_patch_ids=_M2DB_FREQ, **kw,
    )


def test_m4_dual_band_n_masked_counts_tubed_covered_only() -> None:
    """A tubed parcel's WHOLE flat-S field is the target — n_masked counts every S
    slot of each covered tubed parcel; an uncovered parcel contributes nothing even
    if tubed. Phase tag is ``m4_dual_band``."""
    torch.manual_seed(0)
    B, K, d = 2, 4, 8
    student = torch.randn(B, K, _M2DB_S, d)
    teacher = torch.randn(B, K, _M2DB_S, d)
    tubed = torch.zeros(B, K, dtype=torch.bool)
    tubed[:, 0] = True                                     # parcel 0 tubed
    tubed[:, K - 1] = True                                 # last parcel tubed but...
    latent_valid = torch.ones(B, K, dtype=torch.bool)
    latent_valid[:, K - 1] = False                         # ...uncovered → no target
    pred = _m4_dual_predictor(d, K)

    bd = _m4db_call(pred, student, teacher, tubed, latent_valid)
    assert bd.phase == "m4_dual_band"
    # one covered tubed parcel × all S slots × B clips (the uncovered tubed parcel
    # is gated out).
    assert bd.n_masked == B * _M2DB_S
    assert torch.isfinite(bd.total)


def test_m4_dual_band_predictor_and_student_get_gradient() -> None:
    """Paradigm B: the tubed-parcel reconstruction is the predictor's, so its
    params get a finite gradient and the surviving-parcel student context does
    too (not a bare self-distill)."""
    torch.manual_seed(0)
    B, K, d = 2, 5, 8
    student = torch.randn(B, K, _M2DB_S, d, requires_grad=True)
    teacher = torch.randn(B, K, _M2DB_S, d)
    tubed = _tube_bk(B, K, tubed=1)
    latent_valid = torch.ones(B, K, dtype=torch.bool)
    pred = _m4_dual_predictor(d, K)

    bd = _m4db_call(pred, student, teacher, tubed, latent_valid)
    bd.total.backward()
    got = [
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for n, p in pred.named_parameters()
        if "id_embed" not in n  # learned parcel rows only update for sampled ids
    ]
    assert got and all(got), "M4 dual-band predictor params did not get gradient"
    assert student.grad is not None and student.grad.abs().sum() > 0


def test_m4_dual_band_visible_target_split_and_detach() -> None:
    """The whole-parcel tube split is honored exactly (mirrors the B37 M4 split):

    (a) garbaging a TUBED parcel's STUDENT value → no-op (tubed parcels are
        key-padded out of the predictor context; the target is the TEACHER);
    (b) garbaging a SURVIVING parcel's STUDENT value → moves the loss (context);
    (c) garbaging a TUBED parcel's TEACHER value → moves the loss (the target);
    (d) garbaging a SURVIVING parcel's TEACHER value → no-op (only tubed = target,
        teacher never context);
    (e) garbaging an UNCOVERED parcel (either side) → no-op (latent_valid gate).
    """
    torch.manual_seed(0)
    B, K, d = 2, 6, 8
    student = torch.randn(B, K, _M2DB_S, d)
    teacher = torch.randn(B, K, _M2DB_S, d)
    tubed = _tube_bk(B, K, tubed=0)
    latent_valid = torch.ones(B, K, dtype=torch.bool)
    latent_valid[:, K - 1] = False                         # uncovered parcel
    pred = _m4_dual_predictor(d, K)

    def run(s, t):
        return _m4db_call(pred, s, t, tubed, latent_valid).total

    base = run(student, teacher)
    # (a) tubed-parcel student perturbation → no change.
    s = student.clone(); s[:, 0] += 9.0
    torch.testing.assert_close(run(s, teacher), base, atol=0, rtol=0)
    # (b) surviving-parcel student perturbation → change.
    s = student.clone(); s[:, 1] += 9.0
    assert not torch.allclose(run(s, teacher), base)
    # (c) tubed-parcel teacher perturbation → change (the target).
    t = teacher.clone(); t[:, 0] += 9.0
    assert not torch.allclose(run(student, t), base)
    # (d) surviving-parcel teacher perturbation → no change.
    t = teacher.clone(); t[:, 1] += 9.0
    torch.testing.assert_close(run(student, t), base, atol=0, rtol=0)
    # (e) uncovered parcel, either side → no change.
    s = student.clone(); s[:, K - 1] += 9.0
    torch.testing.assert_close(run(s, teacher), base, atol=0, rtol=0)
    t = teacher.clone(); t[:, K - 1] += 9.0
    torch.testing.assert_close(run(student, t), base, atol=0, rtol=0)


def test_m4_dual_band_empty_tube_is_exact_graph_connected_zero() -> None:
    """No tubed parcel → exact 0 loss still connected to the predictor graph
    (B6 masked-empty contract — no NaN)."""
    torch.manual_seed(0)
    B, K, d = 2, 3, 8
    student = torch.randn(B, K, _M2DB_S, d, requires_grad=True)
    teacher = torch.randn(B, K, _M2DB_S, d)
    tubed = torch.zeros(B, K, dtype=torch.bool)            # nothing tubed
    latent_valid = torch.ones(B, K, dtype=torch.bool)
    pred = _m4_dual_predictor(d, K)

    bd = _m4db_call(pred, student, teacher, tubed, latent_valid)
    assert bd.n_masked == 0
    assert float(bd.total.detach()) == 0.0
    bd.total.backward()  # must not raise / NaN


def test_m4_dual_band_precision_uniform_recovers_plain_l1() -> None:
    """Uniform σ + uniform n → mean-1 weight is all-ones → the weighted loss is
    byte-identical to the plain-L1 path (the safe-ship uniform-weight limit)."""
    torch.manual_seed(0)
    B, K, d = 2, 5, 8
    student = torch.randn(B, K, _M2DB_S, d)
    teacher = torch.randn(B, K, _M2DB_S, d)
    tubed = _tube_bk(B, K, tubed=0)
    latent_valid = torch.ones(B, K, dtype=torch.bool)
    pred = _m4_dual_predictor(d, K)                        # eval → deterministic

    plain = _m4db_call(pred, student, teacher, tubed, latent_valid)
    std = torch.full((B, K, _M2DB_S), 0.37)
    n = torch.full((B, K), 5.0)
    weighted = _m4db_call(
        pred, student, teacher, tubed, latent_valid,
        precision_std=std, precision_n=n, precision_alpha=0.75,
        precision_mode="mean1_invvar",  # mean-1 form: uniform stats → all-ones
    )
    torch.testing.assert_close(weighted.total, plain.total, atol=1e-6, rtol=1e-5)


def test_m4_dual_band_precision_nonuniform_changes_loss() -> None:
    """Non-uniform σ/n → the weight is not all-ones → the weighted loss differs
    from plain L1 (the heteroscedastic term is actually doing something)."""
    torch.manual_seed(0)
    B, K, d = 2, 5, 8
    student = torch.randn(B, K, _M2DB_S, d)
    teacher = torch.randn(B, K, _M2DB_S, d)
    # tube TWO parcels so there are >1 scored parcels with different σ/n.
    tubed = torch.zeros(B, K, dtype=torch.bool)
    tubed[:, [0, 2]] = True
    latent_valid = torch.ones(B, K, dtype=torch.bool)
    pred = _m4_dual_predictor(d, K)

    plain = _m4db_call(pred, student, teacher, tubed, latent_valid)
    std = torch.full((B, K, _M2DB_S), 0.5)
    std[:, 0] = 0.05                                       # parcel 0 reliable
    std[:, 2] = 3.0                                        # parcel 2 shaky
    n = torch.full((B, K), 6.0)
    n[:, 0] = 12.0                                         # parcel 0 well covered
    n[:, 2] = 2.0                                          # parcel 2 low coverage
    weighted = _m4db_call(
        pred, student, teacher, tubed, latent_valid,
        precision_std=std, precision_n=n, precision_alpha=0.75,
        precision_mode="mean1_invvar",  # σ/n re-weighting is a mean1-form property
    )
    assert not torch.allclose(weighted.total, plain.total)
    assert torch.isfinite(weighted.total)


def test_m4_dual_band_precision_requires_both_std_and_n() -> None:
    """Precision weighting needs BOTH precision_std and precision_n — passing one
    without the other is a hard error (no silent half-weighting)."""
    torch.manual_seed(0)
    B, K, d = 2, 4, 8
    student = torch.randn(B, K, _M2DB_S, d)
    teacher = torch.randn(B, K, _M2DB_S, d)
    tubed = _tube_bk(B, K, tubed=0)
    lv = torch.ones(B, K, dtype=torch.bool)
    pred = _m4_dual_predictor(d, K)
    std = torch.full((B, K, _M2DB_S), 0.4)
    n = torch.full((B, K), 5.0)
    with pytest.raises(ValueError, match="needs BOTH"):
        _m4db_call(pred, student, teacher, tubed, lv, precision_std=std)
    with pytest.raises(ValueError, match="needs BOTH"):
        _m4db_call(pred, student, teacher, tubed, lv, precision_n=n)


def test_m4_dual_band_shape_guards() -> None:
    torch.manual_seed(0)
    B, K, d = 2, 3, 8
    student = torch.randn(B, K, _M2DB_S, d)
    teacher = torch.randn(B, K, _M2DB_S, d)
    tubed = _tube_bk(B, K, tubed=0)
    lv = torch.ones(B, K, dtype=torch.bool)
    pred = _m4_dual_predictor(d, K)

    with pytest.raises(ValueError, match="student_m4 must be"):
        _m4db_call(pred, student[..., 0], teacher, tubed, lv)
    with pytest.raises(ValueError, match="teacher_m4"):
        _m4db_call(pred, student, teacher[:, :, :-1], tubed, lv)
    with pytest.raises(ValueError, match="tubed"):
        _m4db_call(pred, student, teacher, torch.zeros(B, K + 1, dtype=torch.bool), lv)
    with pytest.raises(ValueError, match="latent_valid"):
        _m4db_call(pred, student, teacher, tubed, torch.ones(B, K + 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="token_mask"):
        _m4db_call(
            pred, student, teacher, tubed, lv,
            token_mask=torch.zeros(B, K, _M2DB_S + 1, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="slot_ids"):
        m4_dual_band_loss(
            predictor=pred, student_m4=student, teacher_m4=teacher,
            tubed=tubed, latent_valid=lv,
            token_mask=torch.zeros(B, K, _M2DB_S, dtype=torch.bool),
            slot_ids=_M2DB_SLOT[:-1], freq_patch_ids=_M2DB_FREQ,
        )
    with pytest.raises(ValueError, match="precision_std shape"):
        _m4db_call(
            pred, student, teacher, tubed, lv,
            precision_std=torch.full((B, K, _M2DB_S + 1), 0.4),
            precision_n=torch.full((B, K), 5.0),
        )
    with pytest.raises(ValueError, match="precision_n shape"):
        _m4db_call(
            pred, student, teacher, tubed, lv,
            precision_std=torch.full((B, K, _M2DB_S), 0.4),
            precision_n=torch.full((B, K + 1), 5.0),
        )


def test_m4_dual_band_predictor_reads_only_unmasked_untubed() -> None:
    """L1 / "NEVER READ ZEROS" (Ben 2026-06-13): the M4 predictor reads ONLY the
    UNMASKED tokens of the UNTUBED (surviving) parcels as context, and predicts the
    TUBED parcels at ALL S=(freq×time) spots. So corrupting ``student_m4`` at
    (a) M2-band-masked cells of a surviving parcel, or (b) ANY tubed-parcel cell,
    leaves the loss UNCHANGED (neither is read as context); corrupting an UNMASKED
    surviving cell MOVES it (non-vacuous)."""
    torch.manual_seed(0)
    B, K, d = 2, 4, 8
    S = _M2DB_S
    student = torch.randn(B, K, S, d)
    teacher = torch.randn(B, K, S, d)
    tubed = _tube_bk(B, K, tubed=0)                          # parcel 0 = tube target
    lv = torch.ones(B, K, dtype=torch.bool)
    token_mask = torch.zeros(B, K, S, dtype=torch.bool)
    token_mask[:, :, [1, 3]] = True                          # band-mask slots {1,3}
    pred = _m4_dual_predictor(d, K)

    base = _m4db_call(pred, student, teacher, tubed, lv, token_mask=token_mask).total
    # (a) corrupt the M2-masked cells of a surviving parcel → NOT context → no move.
    s_a = student.clone(); s_a[:, 1, [1, 3]] += 100.0
    torch.testing.assert_close(
        _m4db_call(pred, s_a, teacher, tubed, lv, token_mask=token_mask).total, base
    )
    # (b) corrupt the TUBED parcel's student cells → NOT context → no move.
    s_b = student.clone(); s_b[:, 0] += 100.0
    torch.testing.assert_close(
        _m4db_call(pred, s_b, teacher, tubed, lv, token_mask=token_mask).total, base
    )
    # (c) corrupt an UNMASKED surviving cell → IS context → loss moves.
    s_c = student.clone(); s_c[:, 1, 2] += 100.0             # slot 2 visible on parcel 1
    assert not torch.allclose(
        _m4db_call(pred, s_c, teacher, tubed, lv, token_mask=token_mask).total, base
    )


def test_m4_dual_band_target_is_tubed_at_all_s() -> None:
    """The M4 target = the TUBED parcels at ALL S spots (whole-parcel tube), drawn
    from the EMA teacher. n_masked = (#tubed covered parcels)·S; corrupting the
    teacher at a tubed cell moves the loss, at a surviving cell does not."""
    torch.manual_seed(1)
    B, K, d = 2, 4, 8
    S = _M2DB_S
    student = torch.randn(B, K, S, d)
    teacher = torch.randn(B, K, S, d)
    tubed = _tube_bk(B, K, tubed=0)
    lv = torch.ones(B, K, dtype=torch.bool)
    token_mask = torch.zeros(B, K, S, dtype=torch.bool)
    token_mask[:, :, [1, 3]] = True
    pred = _m4_dual_predictor(d, K)

    bd = _m4db_call(pred, student, teacher, tubed, lv, token_mask=token_mask)
    assert bd.n_masked == B * S                              # parcel 0 at all S spots
    base = bd.total
    # teacher corruption at a tubed cell (a target) moves the loss.
    t_t = teacher.clone(); t_t[:, 0, 4] += 100.0
    assert not torch.allclose(
        _m4db_call(pred, student, t_t, tubed, lv, token_mask=token_mask).total, base
    )
    # teacher corruption at a surviving (non-target) cell does NOT.
    t_s = teacher.clone(); t_s[:, 2, 4] += 100.0
    torch.testing.assert_close(
        _m4db_call(pred, student, t_s, tubed, lv, token_mask=token_mask).total, base
    )


# ---------------------------------------------------------------------------
# Collapse/quality monitor scalars — target_var / target_norm / pred_var /
# explained_var. RankMe is scale-INVARIANT so it can't see target-magnitude
# collapse; these can (monitor-stats handoff). Pure observability, detached.
# ---------------------------------------------------------------------------


def test_jepa_stats_target_var_detects_constant_target() -> None:
    """A constant-across-tokens target (the magnitude-collapse signature) has
    ~0 target_var; a varied target has a clearly positive target_var."""
    d = 8
    pred = torch.randn(32, d)
    const_target = torch.ones(32, d) * 0.5            # identical across tokens
    varied_target = torch.randn(32, d)
    tv_const, _, _, _ = _jepa_pred_target_stats(pred, const_target)
    tv_varied, _, _, _ = _jepa_pred_target_stats(pred, varied_target)
    assert float(tv_const) < 1e-6
    assert float(tv_varied) > 0.1


def test_jepa_stats_explained_var_one_for_perfect_pred() -> None:
    """Perfect prediction → explained_var == 1 (zero error)."""
    t = torch.randn(40, 8)
    _, _, _, ev = _jepa_pred_target_stats(t.clone(), t)
    assert abs(float(ev) - 1.0) < 1e-5


def test_jepa_stats_explained_var_decreases_as_pred_worsens() -> None:
    """EV is monotone in prediction quality: a closer prediction scores higher
    than a worse one on the same target."""
    torch.manual_seed(0)
    t = torch.randn(64, 8)
    close = t + 0.1 * torch.randn(64, 8)
    far = t + 1.0 * torch.randn(64, 8)
    _, _, _, ev_close = _jepa_pred_target_stats(close, t)
    _, _, _, ev_far = _jepa_pred_target_stats(far, t)
    assert float(ev_close) > float(ev_far)


def test_jepa_stats_pred_var_zero_when_predictor_hedges_to_mean() -> None:
    """A predictor that outputs a constant (hedges to one value) has ~0
    pred_var even when the target varies — the soft-collapse readout that
    target_var alone misses."""
    torch.manual_seed(0)
    t = torch.randn(50, 8)
    const_pred = torch.full((50, 8), 0.3)
    tv, _, pv, _ = _jepa_pred_target_stats(const_pred, t)
    assert float(pv) < 1e-6
    assert float(tv) > 0.1


def test_jepa_stats_nan_when_fewer_than_two_tokens() -> None:
    """< 2 scored tokens → variance undefined → all NaN (logger skips)."""
    for n in (0, 1):
        tv, tn, pv, ev = _jepa_pred_target_stats(
            torch.randn(n, 8), torch.randn(n, 8)
        )
        assert all(torch.isnan(x) for x in (tv, tn, pv, ev))


def test_jepa_stats_target_norm_tracks_magnitude() -> None:
    """target_norm is the mean per-token L2 norm — scaling the target scales it."""
    t = torch.randn(30, 8)
    _, tn1, _, _ = _jepa_pred_target_stats(torch.zeros(30, 8), t)
    _, tn2, _, _ = _jepa_pred_target_stats(torch.zeros(30, 8), t * 3.0)
    assert abs(float(tn2) - 3.0 * float(tn1)) < 1e-4


# ---------------------------------------------------------------------------
# M4 precision-weight monitor stats — the observable proof the downweight-dof
# weight down-weights low-n parcels (low-n: higher raw loss, lower weighted
# contribution).
# ---------------------------------------------------------------------------


def test_m4_precision_monitor_weight_stats_and_tier_split() -> None:
    """weight mean/min/max are the applied-weight summary; the low-n tier
    (n < n_ref) carries a strictly smaller weighted contribution-per-cell than
    its raw loss implies, vs the high-n (full-trust) tier."""
    torch.manual_seed(0)
    n_target, d = 100, 8
    pred = torch.randn(n_target, d)
    target = torch.randn(n_target, d)
    # half the cells are low-n (n=3, downweighted), half full-trust (n=11).
    n_at_target = torch.cat([torch.full((50,), 3.0), torch.full((50,), 11.0)])
    # downweight-dof weights: n=3 → 0.2, n=11 → 1.0.
    weight = torch.where(n_at_target < 11.0, torch.tensor(0.2), torch.tensor(1.0))
    stats = _m4_precision_monitor_stats(
        pred=pred, target=target, weight=weight,
        n_at_target=n_at_target, n_ref=11.0, loss_form="l1",
    )
    assert abs(float(stats["weight_max"]) - 1.0) < 1e-6
    assert abs(float(stats["weight_min"]) - 0.2) < 1e-6
    # low-n weighted contribution is its raw loss scaled by 0.2; high-n by 1.0,
    # so the low-n weighted contribution is well below its raw loss.
    assert float(stats["wcontrib_lown"]) < float(stats["loss_lown"])
    assert abs(float(stats["wcontrib_highn"]) - float(stats["loss_highn"])) < 1e-5
    assert torch.isfinite(
        torch.tensor([float(stats[k]) for k in stats])
    ).all()


def test_m4_precision_monitor_tier_nan_when_a_tier_is_empty() -> None:
    """If every cell is full-trust, the low-n tier mean is NaN (skipped) and the
    high-n tier is finite."""
    n_target, d = 20, 4
    pred = torch.randn(n_target, d)
    target = torch.randn(n_target, d)
    n_at_target = torch.full((n_target,), 11.0)       # all full-trust
    weight = torch.ones(n_target)
    stats = _m4_precision_monitor_stats(
        pred=pred, target=target, weight=weight,
        n_at_target=n_at_target, n_ref=11.0, loss_form="l1",
    )
    assert torch.isnan(stats["loss_lown"])
    assert torch.isfinite(stats["loss_highn"])


# ---------------------------------------------------------------------------
# End-to-end: every per-term loss fn surfaces the collapse scalars on its
# breakdown; the weighted M4 fns additionally surface the precision-tier stats
# only when precision weighting is ON.
# ---------------------------------------------------------------------------


def test_b37_m4_breakdown_carries_collapse_stats() -> None:
    """The M4 freq loss fills the four collapse scalars (finite) and leaves the
    precision-tier fields None when precision weighting is OFF."""
    torch.manual_seed(0)
    B, K, F_p, T_p, d = 2, 5, 4, 4, 8
    student = torch.randn(B, K, F_p, T_p, d)
    teacher = torch.randn(B, K, F_p, T_p, d)
    visible, target_mask = _tube_one_parcel(B, K, T_p, tubed=0)
    bd = b37_m4_freq_loss(
        predictor=_m4_predictor(d, K, F_p, T_p), student_m4=student,
        teacher_m4=teacher, visible=visible, target_mask=target_mask,
    )
    for f in ("target_var", "target_norm", "pred_var", "explained_var"):
        assert getattr(bd, f) is not None and torch.isfinite(getattr(bd, f))
    assert bd.weight_mean is None and bd.loss_lown is None


def test_b37_m4_breakdown_carries_precision_stats_when_weighted() -> None:
    """With precision weighting ON, the breakdown also carries the weight
    distribution + the low/high-n tier split."""
    torch.manual_seed(0)
    B, K, F_p, T_p, d = 2, 6, 4, 4, 8
    student = torch.randn(B, K, F_p, T_p, d)
    teacher = torch.randn(B, K, F_p, T_p, d)
    visible, target_mask = _tube_one_parcel(B, K, T_p, tubed=0)
    std = torch.rand(B, K, F_p, T_p) + 0.1
    n = torch.full((B, K), 3.0)                        # low-n → downweighted
    bd = b37_m4_freq_loss(
        predictor=_m4_predictor(d, K, F_p, T_p), student_m4=student,
        teacher_m4=teacher, visible=visible, target_mask=target_mask,
        precision_std=std, precision_n=n,              # default downweight_dof
    )
    assert bd.weight_mean is not None and torch.isfinite(bd.weight_mean)
    assert bd.weight_max is not None and float(bd.weight_max) <= 1.0
    # the tubed parcel has n=3 < n_ref=11 → every scored cell is low-n.
    assert bd.loss_lown is not None and torch.isfinite(bd.loss_lown)
    assert bd.loss_highn is not None and torch.isnan(bd.loss_highn)


def test_p2_and_m2_dual_band_breakdowns_carry_collapse_stats() -> None:
    """The staged P2 and the 2STFT M2 dual-band losses both fill the collapse
    scalars — so the joint logger surfaces them for every active term."""
    torch.manual_seed(0)
    B, K, T_p, d = 2, 5, 4, 8
    student = torch.randn(B, K, T_p, d)
    teacher = torch.randn(B, K, T_p, d)
    visible, target_mask = _tube_one_parcel(B, K, T_p, tubed=0)
    bd = p2_parcel_m4_loss(
        predictor=_p2_predictor(d, K, T_p), student_m4=student,
        teacher_m4=teacher, visible=visible, target_mask=target_mask,
    )
    for f in ("target_var", "target_norm", "pred_var", "explained_var"):
        assert getattr(bd, f) is not None and torch.isfinite(getattr(bd, f))
