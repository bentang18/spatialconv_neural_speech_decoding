"""Loss-weight warmup scheduler (T2.6).

Generic primitive — does not lock any per-phase λ value (those are blocked
on B04, M01 in ``docs/neuroprobe/v14_blockers.md``). The training loop
constructs a scheduler per loss term with the (peak_value, warmup_steps,
total_steps) the caller picks.

Three shapes are provided:

  * :func:`linear_warmup` — ramps 0 → peak over ``warmup_steps``, holds
    at peak after. Drop-in for the data2vec 2.0 / DINOv3 λ-warmup pattern.
  * :func:`linear_warmup_then_cosine` — ramps 0 → peak, then cosine-decays
    peak → end over ``decay_steps``. Useful for KoLeo's "fade out late in
    training" pattern if Phase-1 needs it.
  * :func:`anatomy_bias_warmup_schedule` — B28 anatomy-bias linear warmup
    (2026-05-27 PM): λ_anat ramps 0 → 1 over the last ``warmup_fraction_p1``
    of P1 ∪ the first ``warmup_fraction_p2`` of P2; constant 1 in P3/P4.
    Replaces the prior discrete OFF→ON toggle at the P1→P2 boundary.
"""

from __future__ import annotations

import math
import typing as tp


Schedule = tp.Callable[[int], float]


def linear_warmup(*, peak: float, warmup_steps: int) -> Schedule:
    """Returns ``λ(step)``: 0 at step 0, ramps linearly to ``peak`` at
    ``warmup_steps``, stays at ``peak`` after."""
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if warmup_steps == 0:
        return lambda _step: float(peak)

    def schedule(step: int) -> float:
        t = max(0, min(step, warmup_steps))
        return float(peak) * (t / float(warmup_steps))

    return schedule


def linear_warmup_then_cosine(
    *,
    peak: float,
    end: float,
    warmup_steps: int,
    decay_steps: int,
) -> Schedule:
    """0 → peak over ``warmup_steps``, then cosine to ``end`` over ``decay_steps``.

    After ``warmup_steps + decay_steps`` the schedule clamps at ``end``.
    """
    if warmup_steps < 0 or decay_steps <= 0:
        raise ValueError(
            f"need warmup_steps >= 0 (got {warmup_steps}) and "
            f"decay_steps > 0 (got {decay_steps})"
        )

    ramp = linear_warmup(peak=peak, warmup_steps=warmup_steps)

    def schedule(step: int) -> float:
        if step <= warmup_steps:
            return ramp(step)
        t = min(step - warmup_steps, decay_steps)
        progress = t / float(decay_steps)
        # Standard cosine decay: peak at progress=0, end at progress=1.
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(end) + (float(peak) - float(end)) * cos

    return schedule


# --- B28 anatomy-bias warmup (2026-05-27 PM) --------------------------
# Option B from the B28 lock memo: linear ramp 0 → 1 across the contiguous
# window spanning the last ``warmup_fraction_p1`` of P1 and the first
# ``warmup_fraction_p2`` of P2. Replaces the prior discrete OFF→ON toggle
# at the P1→P2 boundary (which produced a step discontinuity in the
# cross-attn routing distribution). Sisters ``R-anatomy-bias-step``
# (discrete) and ``R-anatomy-bias-on-from-p1`` (constant 1 throughout)
# falsify the warmup choice.

Phase = tp.Literal["P1", "P2", "P3", "P4"]


def anatomy_bias_warmup_schedule(
    phase: Phase,
    *,
    total_p1_steps: int,
    total_p2_steps: int,
    warmup_fraction_p1: float = 0.25,
    warmup_fraction_p2: float = 0.25,
) -> Schedule:
    """λ_anat(step) for the given phase under the B28 anatomy-bias warmup.

    The warmup window has length
    ``W = warmup_fraction_p1 * total_p1_steps + warmup_fraction_p2 * total_p2_steps``
    and the ramp is linear 0 → 1 across that window. ``λ_anat`` is 0
    before the window starts (early P1), 0.5 at the P1→P2 boundary, and
    1 after the window ends (P2 mid-and-late + all of P3/P4).

    ``step`` is per-phase (P1 step counter restarts at P1 entry; P2 step
    counter restarts at P2 entry). The returned schedule maps that
    per-phase step → λ_anat.
    """
    if not 0.0 <= warmup_fraction_p1 <= 1.0:
        raise ValueError(
            f"warmup_fraction_p1 must be in [0, 1]; got {warmup_fraction_p1}"
        )
    if not 0.0 <= warmup_fraction_p2 <= 1.0:
        raise ValueError(
            f"warmup_fraction_p2 must be in [0, 1]; got {warmup_fraction_p2}"
        )
    if total_p1_steps < 0 or total_p2_steps < 0:
        raise ValueError(
            f"step totals must be >= 0; got total_p1_steps={total_p1_steps}, "
            f"total_p2_steps={total_p2_steps}"
        )

    p1_warmup_steps = warmup_fraction_p1 * total_p1_steps
    p2_warmup_steps = warmup_fraction_p2 * total_p2_steps
    p1_warmup_start = total_p1_steps - p1_warmup_steps  # window start (P1 step)
    window_length = p1_warmup_steps + p2_warmup_steps

    if phase == "P1":
        def p1_schedule(step: int) -> float:
            if window_length <= 0:
                return 0.0
            if step <= p1_warmup_start:
                return 0.0
            t = min(step, total_p1_steps) - p1_warmup_start
            return float(min(1.0, t / window_length))
        return p1_schedule

    if phase == "P2":
        def p2_schedule(step: int) -> float:
            if window_length <= 0:
                return 1.0
            if step >= p2_warmup_steps:
                return 1.0
            t = p1_warmup_steps + max(0, step)
            return float(min(1.0, t / window_length))
        return p2_schedule

    if phase in ("P3", "P4"):
        return lambda _step: 1.0

    raise ValueError(f"unknown phase {phase!r}; expected one of P1/P2/P3/P4")
