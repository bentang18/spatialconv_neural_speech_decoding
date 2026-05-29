"""B25 amendment (2026-05-27): tests for the V-JEPA 2.1 §2.3.1 Eq 2
context-self-supervision loss at ``L_pre_frame``.

Companion to the masked-patch ``L_pre_frame`` loss: applies a distance-
weighted Smooth-L1 to the *visible* (non-masked) M2 patches with per-
position weight ``λ_i = λ_ctx / √d_min(i, M)`` where ``d_min`` is the
Chebyshev distance on the per-electrode ``(F_p × T_p)`` grid from
visible patch ``i`` to the nearest masked patch ``M``.

``λ_ctx = 0.5`` is the default per the B25 lock. The loss reuses the
visible-only encoder forward, so no extra forward pass is required.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.ssl.context_loss import (
    LAMBDA_CTX_DEFAULT,
    LAMBDA_CTX_WARMUP_FRACTION,
    chebyshev_dmin_to_masked,
    context_self_supervision_loss,
    lambda_ctx_p1_warmup_schedule,
)


def test_lambda_ctx_default_locked_at_half() -> None:
    """B25 lock 2026-05-27 (B26 unchanged): ``λ_ctx`` peak = 0.5."""
    assert LAMBDA_CTX_DEFAULT == 0.5


def test_lambda_ctx_warmup_fraction_locked_at_25pct() -> None:
    """B26 lock 2026-05-27 PM: warmup fraction = 0.25 of P1 steps
    (V-JEPA 2.1 Table 2 best schedule)."""
    assert LAMBDA_CTX_WARMUP_FRACTION == 0.25


# ---------- λ_ctx warmup schedule (B26) ------------------------------


def test_lambda_ctx_warmup_starts_at_zero() -> None:
    """B26: λ_ctx(0) = 0 — V-JEPA 2.1 Table 2 best schedule starts the
    context loss off at zero before linearly ramping in."""
    sched = lambda_ctx_p1_warmup_schedule(total_p1_steps=400_000)
    assert sched(0) == 0.0


def test_lambda_ctx_warmup_hits_peak_at_25pct() -> None:
    """B26: λ_ctx reaches the 0.5 peak at exactly 25% of total P1 steps."""
    total = 400_000
    sched = lambda_ctx_p1_warmup_schedule(total_p1_steps=total)
    warmup_end = int(round(0.25 * total))
    assert abs(sched(warmup_end) - LAMBDA_CTX_DEFAULT) < 1e-9


def test_lambda_ctx_warmup_holds_peak_after_25pct() -> None:
    """B26: past the warmup endpoint the value clamps at peak."""
    total = 400_000
    sched = lambda_ctx_p1_warmup_schedule(total_p1_steps=total)
    assert abs(sched(total) - LAMBDA_CTX_DEFAULT) < 1e-9
    assert abs(sched(total * 10) - LAMBDA_CTX_DEFAULT) < 1e-9


def test_lambda_ctx_warmup_linear_midpoint() -> None:
    """B26: at half the warmup span, λ_ctx is half the peak."""
    total = 400_000
    sched = lambda_ctx_p1_warmup_schedule(total_p1_steps=total)
    midpoint = int(round(0.125 * total))
    assert abs(sched(midpoint) - 0.5 * LAMBDA_CTX_DEFAULT) < 1e-6


def test_lambda_ctx_warmup_rejects_non_positive_total_steps() -> None:
    """B26: schedule must guard against undefined warmup spans."""
    with pytest.raises(ValueError, match="total_p1_steps"):
        lambda_ctx_p1_warmup_schedule(total_p1_steps=0)


def test_lambda_ctx_warmup_custom_peak_and_fraction() -> None:
    """B26: schedule honours custom peak / warmup_fraction (used by the
    ``R-ctx-lambda-{0.0, 0.25, 0.5, 1.0}`` and ``R-no-warmup`` sisters)."""
    total = 100
    sched = lambda_ctx_p1_warmup_schedule(
        total_p1_steps=total, peak=1.0, warmup_fraction=0.5,
    )
    assert sched(0) == 0.0
    assert abs(sched(25) - 0.5) < 1e-9
    assert abs(sched(50) - 1.0) < 1e-9
    assert abs(sched(75) - 1.0) < 1e-9


def test_lambda_ctx_r_no_warmup_sister_via_constant_callable() -> None:
    """``R-no-warmup`` sister: replace the schedule with
    ``lambda _step: LAMBDA_CTX_DEFAULT`` to falsify the warmup-is-load-
    bearing claim. Validate via the public construction pattern."""

    def constant_no_warmup(_step: int) -> float:
        return LAMBDA_CTX_DEFAULT

    assert constant_no_warmup(0) == LAMBDA_CTX_DEFAULT
    assert constant_no_warmup(10_000) == LAMBDA_CTX_DEFAULT


def test_chebyshev_dmin_on_4x4_grid_hand_constructed() -> None:
    """Hand-crafted 4×4 mask grid. Masked = (1,1) and (2,2).

    For each ``(f, t)`` cell, ``d_min = min over masked m of max(|f-m_f|, |t-m_t|)``::

        f\\t  0  1  2  3
         0   1  1  1  2
         1   1  0  1  1     (mask cell at (1, 1))
         2   1  1  0  1     (mask cell at (2, 2))
         3   2  1  1  1
    """
    F_p, T_p = 4, 4
    mask = torch.zeros(F_p, T_p, dtype=torch.bool)
    mask[1, 1] = True
    mask[2, 2] = True
    expected = torch.tensor([
        [1, 1, 1, 2],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [2, 1, 1, 1],
    ], dtype=torch.float32)
    got = chebyshev_dmin_to_masked(mask)
    assert torch.equal(got, expected)


def test_chebyshev_dmin_all_visible_returns_sentinel_inf() -> None:
    """If no patch is masked, d_min is undefined; the function must
    signal this with +∞ so callers can drop the term cleanly."""
    F_p, T_p = 4, 4
    all_visible = torch.zeros(F_p, T_p, dtype=torch.bool)
    got = chebyshev_dmin_to_masked(all_visible)
    assert torch.isinf(got.float()).all()


def test_chebyshev_dmin_all_masked_is_zero_everywhere() -> None:
    """When every cell is masked, d_min = 0 everywhere (a cell is at
    distance 0 from itself; masked cells are excluded from the loss but
    the helper should not error)."""
    F_p, T_p = 3, 3
    all_masked = torch.ones(F_p, T_p, dtype=torch.bool)
    got = chebyshev_dmin_to_masked(all_masked)
    assert torch.equal(got, torch.zeros(F_p, T_p, dtype=torch.float32))


def test_context_loss_zero_when_student_equals_teacher() -> None:
    """When the visible-patch student and teacher reps are identical,
    the loss is exactly 0 regardless of the mask layout."""
    torch.manual_seed(0)
    B, C, F_p, T_p, d = 2, 3, 4, 5, 16
    s = torch.randn(B, C, F_p, T_p, d)
    # Random mask with at least one masked cell so d_min is defined.
    patch_mask = torch.zeros(B, C, F_p, T_p, dtype=torch.bool)
    patch_mask[:, :, 1, 2] = True
    loss = context_self_supervision_loss(
        student=s, teacher=s, patch_mask=patch_mask,
    )
    assert torch.allclose(loss, torch.zeros(()), atol=1e-7)


def test_context_loss_gradient_flows_to_student_only() -> None:
    """End-to-end gradient check: detached teacher receives no grad."""
    torch.manual_seed(0)
    B, C, F_p, T_p, d = 1, 2, 3, 3, 8
    s = torch.randn(B, C, F_p, T_p, d, requires_grad=True)
    t = torch.randn(B, C, F_p, T_p, d, requires_grad=True)
    patch_mask = torch.zeros(B, C, F_p, T_p, dtype=torch.bool)
    patch_mask[:, :, 1, 1] = True
    loss = context_self_supervision_loss(
        student=s, teacher=t.detach(), patch_mask=patch_mask,
    )
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert t.grad is None


def test_context_loss_excludes_masked_patches_from_loss() -> None:
    """The loss is computed over *visible* (non-masked) patches only.
    Perturbing teacher values at masked positions must not change the
    loss."""
    torch.manual_seed(0)
    B, C, F_p, T_p, d = 1, 1, 4, 4, 8
    s = torch.randn(B, C, F_p, T_p, d)
    t_a = torch.randn(B, C, F_p, T_p, d)
    t_b = t_a.clone()
    patch_mask = torch.zeros(B, C, F_p, T_p, dtype=torch.bool)
    patch_mask[:, :, 1:3, 1:3] = True  # 2×2 masked block
    # Perturb teacher only at masked positions.
    t_b[:, :, patch_mask[0, 0]] = 999.0
    loss_a = context_self_supervision_loss(student=s, teacher=t_a, patch_mask=patch_mask)
    loss_b = context_self_supervision_loss(student=s, teacher=t_b, patch_mask=patch_mask)
    torch.testing.assert_close(loss_a, loss_b, atol=1e-6, rtol=1e-6)


def test_context_loss_weights_decay_with_distance() -> None:
    """λ_i = λ_ctx / √d_min(i, M): closer visible patches get larger
    weight. Test: hold mask fixed, place a unit-amplitude error at a
    near-mask cell and another at a far-mask cell of equal magnitude;
    the loss should be dominated by the near-mask contribution."""
    torch.manual_seed(0)
    B, C, F_p, T_p, d = 1, 1, 4, 4, 1
    s = torch.zeros(B, C, F_p, T_p, d)
    t = torch.zeros(B, C, F_p, T_p, d)
    patch_mask = torch.zeros(B, C, F_p, T_p, dtype=torch.bool)
    patch_mask[:, :, 0, 0] = True  # one mask cell at corner
    # d_min: (0,1) → 1 (near); (3,3) → 3 (far).
    # Place unit error at near (0, 1).
    s_near = s.clone()
    s_near[0, 0, 0, 1, 0] = 1.0
    # Place unit error at far (3, 3).
    s_far = s.clone()
    s_far[0, 0, 3, 3, 0] = 1.0

    loss_near = context_self_supervision_loss(student=s_near, teacher=t, patch_mask=patch_mask)
    loss_far = context_self_supervision_loss(student=s_far, teacher=t, patch_mask=patch_mask)
    assert loss_near > loss_far, (
        f"near-mask weight should exceed far-mask weight; "
        f"got near={loss_near.item():.4f} far={loss_far.item():.4f}"
    )


def test_context_loss_lambda_ctx_zero_makes_loss_zero() -> None:
    """``R-no-context-loss`` sister cell: setting ``λ_ctx = 0`` zeroes
    every weight, hence the total loss."""
    torch.manual_seed(0)
    B, C, F_p, T_p, d = 1, 1, 3, 3, 4
    s = torch.randn(B, C, F_p, T_p, d)
    t = torch.randn(B, C, F_p, T_p, d)
    patch_mask = torch.zeros(B, C, F_p, T_p, dtype=torch.bool)
    patch_mask[:, :, 1, 1] = True
    loss = context_self_supervision_loss(
        student=s, teacher=t, patch_mask=patch_mask, lambda_ctx=0.0,
    )
    assert torch.allclose(loss, torch.zeros(()), atol=1e-7)


def test_context_loss_rejects_shape_mismatch() -> None:
    s = torch.randn(1, 1, 3, 3, 4)
    t = torch.randn(1, 1, 3, 3, 5)
    patch_mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    patch_mask[:, :, 1, 1] = True
    with pytest.raises(ValueError, match="shape mismatch"):
        context_self_supervision_loss(student=s, teacher=t, patch_mask=patch_mask)


def test_context_loss_rejects_wrong_ndim_patch_mask() -> None:
    s = torch.randn(1, 1, 3, 3, 4)
    t = torch.randn(1, 1, 3, 3, 4)
    patch_mask = torch.zeros(1, 1, 3, 3, dtype=torch.float32)  # not bool, wrong dtype path
    # OK with float (treated as truthy), but a 3-D mask must error.
    bad = torch.zeros(1, 3, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match=r"\(B, C, F_p, T_p\)"):
        context_self_supervision_loss(student=s, teacher=t, patch_mask=bad)


def test_context_loss_no_masked_patches_returns_zero() -> None:
    """When no patch is masked, d_min = +∞ everywhere; every weight is
    1/√∞ = 0; total loss is exactly 0. (Lets the trainer just call the
    primitive without a guard.)"""
    torch.manual_seed(0)
    B, C, F_p, T_p, d = 1, 1, 3, 3, 4
    s = torch.randn(B, C, F_p, T_p, d)
    t = torch.randn(B, C, F_p, T_p, d)
    patch_mask = torch.zeros(B, C, F_p, T_p, dtype=torch.bool)
    loss = context_self_supervision_loss(student=s, teacher=t, patch_mask=patch_mask)
    assert torch.allclose(loss, torch.zeros(()), atol=1e-7)


def test_context_loss_default_loss_form_is_l1() -> None:
    """B26 lock 2026-05-27 PM: default ``loss_form`` is pure L1 (V-JEPA 2
    §2.1 Eq 1 + V-JEPA 2.1 §2.3.1 Eqs 1+2). Hand-computed expectation: a
    single visible cell with a single masked neighbour contributes
    ``(λ_ctx/√1) · |s - t|`` per feature, summed over d, divided by the
    same weight × d. Net per-clip = ``|s - t|.mean()`` (the λ and d
    cancel out)."""
    B, C, F_p, T_p, d = 1, 1, 1, 2, 4  # one visible cell + one masked cell
    s = torch.zeros(B, C, F_p, T_p, d)
    t = torch.zeros(B, C, F_p, T_p, d)
    s[0, 0, 0, 0] = torch.tensor([1.0, -2.0, 3.0, -4.0])  # visible cell error
    patch_mask = torch.zeros(B, C, F_p, T_p, dtype=torch.bool)
    patch_mask[0, 0, 0, 1] = True                              # mask the other cell
    loss = context_self_supervision_loss(student=s, teacher=t, patch_mask=patch_mask)
    expected = torch.tensor((1.0 + 2.0 + 3.0 + 4.0) / 4.0)  # mean |L1|
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-6)


def test_context_loss_smooth_l1_opt_in_via_loss_form() -> None:
    """B26: ``loss_form="smooth_l1"`` reproduces the B25 default Smooth-L1
    form for ``R-smoothl1-beta-*`` sister cells."""
    torch.manual_seed(0)
    B, C, F_p, T_p, d = 1, 1, 2, 2, 4
    s = torch.randn(B, C, F_p, T_p, d)
    t = torch.randn(B, C, F_p, T_p, d)
    patch_mask = torch.zeros(B, C, F_p, T_p, dtype=torch.bool)
    patch_mask[0, 0, 0, 0] = True
    loss_l1 = context_self_supervision_loss(
        student=s, teacher=t, patch_mask=patch_mask, loss_form="l1",
    )
    loss_smooth_l1 = context_self_supervision_loss(
        student=s, teacher=t, patch_mask=patch_mask, loss_form="smooth_l1",
    )
    loss_mse = context_self_supervision_loss(
        student=s, teacher=t, patch_mask=patch_mask, loss_form="mse",
    )
    # Three forms produce different values; loud sanity rather than
    # closed-form (closed form covered by test_context_loss_default_loss_form_is_l1).
    assert torch.isfinite(loss_l1) and loss_l1 > 0
    assert torch.isfinite(loss_smooth_l1) and loss_smooth_l1 > 0
    assert torch.isfinite(loss_mse) and loss_mse > 0
    assert not torch.allclose(loss_l1, loss_smooth_l1, atol=1e-3, rtol=1e-3)
    assert not torch.allclose(loss_l1, loss_mse, atol=1e-3, rtol=1e-3)


def test_context_loss_rejects_unknown_loss_form() -> None:
    """B26: unknown ``loss_form`` must fail loudly."""
    s = torch.randn(1, 1, 2, 2, 4)
    t = torch.randn(1, 1, 2, 2, 4)
    patch_mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    patch_mask[0, 0, 0, 0] = True
    with pytest.raises(ValueError, match="loss_form"):
        context_self_supervision_loss(
            student=s, teacher=t, patch_mask=patch_mask,
            loss_form="huber",  # type: ignore[arg-type]
        )
