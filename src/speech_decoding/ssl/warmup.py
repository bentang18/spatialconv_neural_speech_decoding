"""Loss-weight warmup scheduler (T2.6).

Generic primitive — does not lock any per-phase λ value (those are blocked
on B04, M01 in ``docs/neuroprobe/v14_blockers.md``). The training loop
constructs a scheduler per loss term with the (peak_value, warmup_steps,
total_steps) the caller picks.

Two shapes are provided:

  * :func:`linear_warmup` — ramps 0 → peak over ``warmup_steps``, holds
    at peak after. Drop-in for the data2vec 2.0 / DINOv3 λ-warmup pattern.
  * :func:`linear_warmup_then_cosine` — ramps 0 → peak, then cosine-decays
    peak → end over ``decay_steps``. Useful for KoLeo's "fade out late in
    training" pattern if Phase-1 needs it.

The B28 ``anatomy_bias_warmup_schedule`` was removed by the B36 hard
per-parcel pool (2026-06-01): the soft ``λ_anat·log(support+ε)`` routing bias
it ramped no longer exists, so there is nothing to warm up. See
[[project_v14_b36_perparcel_pool_structured_jepa_2026_06_01]].
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
