"""Tests for the EMA teacher + StopGrad (T2.2) + phase-aware schedules (EMA-01)
+ layer-avg target (LOSS-06) + B26 fixed-τ lock (2026-05-27 PM) + B26 teacher
full-input contract assert.

B26 amendment 2026-05-27 PM: EMA momentum is locked at fixed τ=0.99925 across
P1 and P2 per V-JEPA 2 §2.4 (drops the V-JEPA 1-style 0.996 → 1.0 ramp).
The ramp is preserved as the ``R-ema-ramp-v-jepa1`` sister falsifier via
:func:`v_jepa1_ema_schedule`.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from speech_decoding.ssl.ema import (
    EmaTeacher,
    P1_EMA_TAU,
    P1_TOTAL_STEPS,
    P2_EMA_TAU,
    P2_TOTAL_STEPS,
    assert_teacher_full_input,
    fixed_ema_schedule,
    layer_avg_with_instance_norm,
    linear_ema_schedule,
    p1_ema_schedule,
    p2_ema_schedule,
    phase_ema_schedule,
    stop_grad,
    v_jepa1_ema_schedule,
)


def test_stop_grad_detaches_tensor() -> None:
    """``stop_grad`` is just a readable alias for ``.detach()``; ensure no
    surprise transformations beyond that."""
    x = torch.randn(3, requires_grad=True)
    y = stop_grad(x)
    assert torch.equal(y, x.detach())
    assert not y.requires_grad


def test_linear_ema_schedule_ramps_correctly() -> None:
    sched = linear_ema_schedule(start=0.99, end=0.9999, total_steps=1_000)
    assert abs(sched(0) - 0.99) < 1e-9
    assert abs(sched(500) - (0.99 + 0.5 * (0.9999 - 0.99))) < 1e-9
    assert abs(sched(1_000) - 0.9999) < 1e-9
    # Past the end clamps.
    assert abs(sched(10_000) - 0.9999) < 1e-9


def test_ema_teacher_params_never_require_grad() -> None:
    """Teacher must be frozen at construction AND remain frozen after updates."""
    student = nn.Linear(4, 4)
    teacher = EmaTeacher(student)
    for p in teacher.model.parameters():
        assert not p.requires_grad
    # After an update, still frozen.
    teacher.update_from(student)
    for p in teacher.model.parameters():
        assert not p.requires_grad


def test_ema_teacher_forward_runs_under_no_grad() -> None:
    """Teacher forward must not produce gradient-tracking tensors even if
    the caller forgets to wrap the call site in ``with torch.no_grad()``."""
    student = nn.Linear(4, 4)
    teacher = EmaTeacher(student)
    x = torch.randn(2, 4, requires_grad=True)
    y = teacher(x)
    assert not y.requires_grad


def test_ema_teacher_tracks_student_under_constant_coefficient() -> None:
    """Under a constant coefficient α, after k steps the teacher's distance
    from a stationary student decays by α^k. With α=0.5 the teacher should
    closely track a stationary student after a handful of steps."""
    torch.manual_seed(0)
    student = nn.Linear(4, 4)
    teacher = EmaTeacher(student, coeff_schedule=lambda _step: 0.5)
    # Move teacher away from student so the test isn't trivially satisfied.
    with torch.no_grad():
        for p in teacher.model.parameters():
            p.add_(torch.randn_like(p) * 5.0)
    # Run 20 update steps without touching the student.
    for _ in range(20):
        teacher.update_from(student)
    # With α=0.5 and 20 steps, residual ≈ 0.5^20 ≈ 1e-6 of the initial offset.
    student_state = student.state_dict()
    teacher_state = teacher.model.state_dict()
    for name, s in student_state.items():
        if not s.dtype.is_floating_point:
            continue
        diff = (teacher_state[name] - s).abs().max().item()
        assert diff < 1e-3, f"{name}: teacher drifted, max abs diff {diff}"


def test_ema_teacher_step_counter_advances() -> None:
    student = nn.Linear(4, 4)
    teacher = EmaTeacher(student)
    assert int(teacher._step.item()) == 0
    teacher.update_from(student)
    assert int(teacher._step.item()) == 1
    teacher.update_from(student)
    teacher.update_from(student)
    assert int(teacher._step.item()) == 3


def test_ema_teacher_update_uses_schedule_at_current_step() -> None:
    """Verify the schedule callable is invoked with the current step."""
    seen_steps: list[int] = []

    def tracking_schedule(step: int) -> float:
        seen_steps.append(step)
        return 0.5

    student = nn.Linear(4, 4)
    teacher = EmaTeacher(student, coeff_schedule=tracking_schedule)
    teacher.update_from(student)
    teacher.update_from(student)
    teacher.update_from(student)
    assert seen_steps == [0, 1, 2]


# --- #135 aliased-tensor double-update fix (2026-05-30 speedup audit) -------


class _AliasModule(nn.Module):
    """A module that registers one block under two attribute names — the same
    shape as the v14 encoder's legacy ``self.cross_attn = self.cross_attns[0]``
    handle. ``state_dict()`` then yields both keys pointing at one storage."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False)])
        self.alias = self.blocks[0]  # same object → duplicate state_dict key
        self.solo = nn.Linear(4, 4, bias=False)  # non-aliased control


def test_ema_update_applies_coeff_once_to_aliased_params() -> None:
    """#135: a tensor that appears under two state_dict keys (module aliasing)
    must receive the EMA coefficient EXACTLY ONCE, not twice. The pre-fix loop
    applied ``mul_(coeff).add_(s, 1-coeff)`` per key → effective coeff² on the
    aliased params, violating the B26 uniform-τ contract."""
    torch.manual_seed(0)
    student = _AliasModule()
    coeff = 0.9
    teacher = EmaTeacher(student, coeff_schedule=lambda _s: coeff)
    # Move the teacher off the student so single vs double application differ
    # (they collapse to ``s`` while teacher == student from the deepcopy).
    with torch.no_grad():
        for p in teacher.model.parameters():
            p.add_(torch.randn_like(p) * 5.0)

    t0_alias = teacher.model.blocks[0].weight.detach().clone()
    s_alias = student.blocks[0].weight.detach().clone()
    t0_solo = teacher.model.solo.weight.detach().clone()
    s_solo = student.solo.weight.detach().clone()

    teacher.update_from(student)

    single_alias = t0_alias * coeff + s_alias * (1.0 - coeff)
    double_alias = t0_alias * coeff**2 + s_alias * (1.0 - coeff**2)
    t1_alias = teacher.model.blocks[0].weight
    assert torch.allclose(t1_alias, single_alias), "aliased param double-updated"
    assert not torch.allclose(t1_alias, double_alias), "still hitting the coeff² bug"

    # Non-aliased control still follows the single-application formula.
    single_solo = t0_solo * coeff + s_solo * (1.0 - coeff)
    assert torch.allclose(teacher.model.solo.weight, single_solo)


def test_ema_foreach_matches_reference_sequential_loop() -> None:
    """#135: the fused ``_foreach`` update must be bit-identical to a reference
    per-tensor (deduped) ``mul_/add_`` loop on a realistic module."""
    torch.manual_seed(1)
    student = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.Linear(8, 4))
    coeff = 0.99925
    teacher = EmaTeacher(student, coeff_schedule=lambda _s: coeff)
    with torch.no_grad():
        for p in teacher.model.parameters():
            p.add_(torch.randn_like(p) * 2.0)

    # Reference: dedup by storage, then per-tensor mul_/add_.
    ref = {}
    s_state = student.state_dict()
    seen: set[int] = set()
    for name, t in teacher.model.state_dict().items():
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        s = s_state[name]
        if t.dtype.is_floating_point and s.dtype.is_floating_point:
            ref[name] = (t.detach().clone() * coeff + s.detach() * (1.0 - coeff))

    teacher.update_from(student)

    t_state = teacher.model.state_dict()
    for name, expected in ref.items():
        assert torch.equal(t_state[name], expected), f"{name}: foreach != sequential"


# ---- EMA-01 / B26: phase-aware decay schedules ------------------------------


def test_p1_ema_tau_locked_at_0p99925() -> None:
    """B26 lock 2026-05-27 PM: P1 EMA momentum constant ``P1_EMA_TAU = 0.99925``
    per V-JEPA 2 §2.4."""
    assert P1_EMA_TAU == 0.99925


def test_p2_ema_tau_locked_at_0p99925() -> None:
    """B26 lock 2026-05-27 PM: P2 EMA momentum constant ``P2_EMA_TAU = 0.99925``
    per V-JEPA 2 §2.4."""
    assert P2_EMA_TAU == 0.99925


def test_fixed_ema_schedule_returns_constant_tau() -> None:
    """B26: ``fixed_ema_schedule(tau)`` returns ``tau`` at every step."""
    sched = fixed_ema_schedule(tau=0.99925)
    for step in (0, 1, 1_000, P1_TOTAL_STEPS, 10 * P1_TOTAL_STEPS):
        assert abs(sched(step) - 0.99925) < 1e-12


def test_fixed_ema_schedule_rejects_out_of_range_tau() -> None:
    """B26: tau must be in (0, 1)."""
    with pytest.raises(ValueError, match="tau"):
        fixed_ema_schedule(tau=0.0)
    with pytest.raises(ValueError, match="tau"):
        fixed_ema_schedule(tau=1.0)
    with pytest.raises(ValueError, match="tau"):
        fixed_ema_schedule(tau=-0.5)


def test_p1_ema_schedule_is_fixed_0p999_under_b26() -> None:
    """B26 lock: P1 schedule is now constant τ=0.99925 (previously a 0.99 →
    0.9999 ramp). Verify a few step counts return the same value."""
    sched = p1_ema_schedule()
    assert abs(sched(0) - P1_EMA_TAU) < 1e-12
    assert abs(sched(P1_TOTAL_STEPS // 2) - P1_EMA_TAU) < 1e-12
    assert abs(sched(P1_TOTAL_STEPS) - P1_EMA_TAU) < 1e-12
    assert abs(sched(10 * P1_TOTAL_STEPS) - P1_EMA_TAU) < 1e-12


def test_p2_ema_schedule_is_fixed_0p999_under_b26() -> None:
    """B26 lock: P2 schedule is now constant τ=0.99925 (previously a 0.999 →
    0.9999 ramp). Verify a few step counts return the same value."""
    sched = p2_ema_schedule()
    assert abs(sched(0) - P2_EMA_TAU) < 1e-12
    assert abs(sched(P2_TOTAL_STEPS // 2) - P2_EMA_TAU) < 1e-12
    assert abs(sched(P2_TOTAL_STEPS) - P2_EMA_TAU) < 1e-12


def test_phase_ema_schedule_dispatches_to_fixed_0p999_under_b26() -> None:
    """B26 lock: phase dispatch returns fixed-τ=0.99925 for both P1 and P2."""
    p1 = phase_ema_schedule(1)
    p2 = phase_ema_schedule(2)
    assert abs(p1(0) - 0.99925) < 1e-12
    assert abs(p2(0) - 0.99925) < 1e-12
    # And at any later step.
    assert abs(p1(100_000) - 0.99925) < 1e-12
    assert abs(p2(20_000) - 0.99925) < 1e-12


def test_phase_ema_schedule_rejects_phase3() -> None:
    """EMA-01: P3 uses an external frozen Whisper teacher, no EMA. Caller
    must not be able to silently get an EMA schedule for P3."""
    with pytest.raises(ValueError, match="P3"):
        phase_ema_schedule(3)  # type: ignore[arg-type]


# ---- B26 sister: R-ema-ramp-v-jepa1 ----------------------------------------


def test_v_jepa1_ema_schedule_ramps_0p996_to_1p0() -> None:
    """``R-ema-ramp-v-jepa1`` sister cell: V-JEPA 1 ramp 0.996 → 1.0 over
    ``total_steps``. Falsifies the V-JEPA 2 drop-the-ramp decision at
    v14 scale."""
    total = 1_000
    sched = v_jepa1_ema_schedule(total_steps=total)
    assert abs(sched(0) - 0.996) < 1e-9
    assert abs(sched(total) - 1.0) < 1e-9
    # Midpoint = 0.998
    assert abs(sched(total // 2) - (0.996 + 0.5 * (1.0 - 0.996))) < 1e-9


# ---- B26 teacher full-input contract assert --------------------------------


def test_assert_teacher_full_input_accepts_no_masks() -> None:
    """B26 contract: the ideal teacher call passes no masks at all
    (``None``, ``None``)."""
    assert_teacher_full_input()  # should not raise


def test_assert_teacher_full_input_accepts_all_true_masks() -> None:
    """B26 contract: an all-True mask is the equivalent of no mask — the
    teacher sees the full input."""
    patch_mask = torch.ones(2, 4, 10, 50, dtype=torch.bool)
    shaft_mask = torch.ones(8, dtype=torch.bool)
    assert_teacher_full_input(patch_mask=patch_mask, shaft_mask=shaft_mask)


def test_assert_teacher_full_input_rejects_patch_drop() -> None:
    """B26 contract: any False entry in ``patch_mask`` violates the
    full-input contract — assert raises."""
    patch_mask = torch.ones(2, 4, 10, 50, dtype=torch.bool)
    patch_mask[0, 0, 0, 0] = False  # a single dropped patch
    with pytest.raises(AssertionError, match="patch_mask"):
        assert_teacher_full_input(patch_mask=patch_mask)


def test_assert_teacher_full_input_rejects_shaft_drop() -> None:
    """B26 contract: any False entry in ``shaft_mask`` violates the
    full-input contract — assert raises."""
    shaft_mask = torch.ones(8, dtype=torch.bool)
    shaft_mask[3] = False
    with pytest.raises(AssertionError, match="shaft_mask"):
        assert_teacher_full_input(shaft_mask=shaft_mask)


def test_assert_teacher_full_input_message_cites_v_jepa_sections() -> None:
    """B26: error message names the V-JEPA section so a future debugger
    can trace the contract back to the source."""
    shaft_mask = torch.tensor([True, False, True])
    with pytest.raises(AssertionError) as excinfo:
        assert_teacher_full_input(shaft_mask=shaft_mask)
    assert "V-JEPA 2" in str(excinfo.value)


# ---- LOSS-06: per-layer instance-norm + K-layer averaging on EMA target ----


def test_layer_avg_with_instance_norm_normalizes_per_layer() -> None:
    """LOSS-06: each layer is instance-normed before averaging. Output should
    have ~zero mean and unit std along ``instance_norm_dim`` per source layer
    AFTER averaging — averaging of normed layers preserves the moment if all
    layers happen to be aligned, otherwise reduces variance."""
    torch.manual_seed(0)
    # 6 layers (K=6 for v14 latent stack) of (B, T, d) features.
    layers = [torch.randn(2, 3, 16) * 10.0 + 7.0 for _ in range(6)]
    avg = layer_avg_with_instance_norm(layers)
    assert avg.shape == (2, 3, 16)
    # Each *layer* is normed to mu≈0, sigma≈1 along feature dim before averaging;
    # the average then has mu≈0 too (sum of zero-mean vars is zero-mean).
    assert avg.mean(dim=-1).abs().max().item() < 0.1


def test_layer_avg_with_instance_norm_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        layer_avg_with_instance_norm([])


def test_layer_avg_with_instance_norm_passes_through_single_layer() -> None:
    """Single-layer 'average' is just instance-norm of that layer."""
    torch.manual_seed(1)
    layer = torch.randn(2, 4, 8) * 5.0 + 3.0
    avg = layer_avg_with_instance_norm([layer])
    # Instance-norm along last dim → mu≈0, std≈1.
    assert avg.mean(dim=-1).abs().max().item() < 1e-5
    assert (avg.std(dim=-1) - 1.0).abs().max().item() < 0.1
