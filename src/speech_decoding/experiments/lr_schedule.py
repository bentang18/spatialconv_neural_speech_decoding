"""Linear-warmup → cosine→0 LR schedule for the v14 SSL/distill phases.

The locked v14 optimizer recipe (training_recipe.md §7, the closed B01 lock)
fixes the LR *shape* as ``linear warmup → cosine → 0`` as a non-swept universal
(the peak-LR value and weight_decay were re-classified to M0-measured sweep
centers on 2026-06-01, but the warmup+cosine *shape*, the AdamW family,
β=(0.9, 0.95), and grad_clip=1.0 stay locked). torch ships no single scheduler
expressible through neuraltrain's ``name``+``kwargs`` config that does linear
warmup *then* cosine decay (``SequentialLR``/``LambdaLR`` need pre-built objects
or callables), so this module supplies a small custom one.

Why a direct ``LRScheduler`` subclass (not ``SequentialLR(LinearLR,
CosineAnnealingLR)``): ``CosineAnnealingLR.eta_min`` is an *absolute* floor
shared across param groups. The moment ``min_lr_ratio > 0`` (any floored sister
run), a single absolute floor silently breaks the P2/P3 discriminative-LR ratio
(the smaller front-end group stops decaying while the larger group keeps going).
``_WarmupCosineLR`` instead multiplies *each group's own* ``base_lr`` by a shared
multiplier, so the ratio is preserved exactly through warmup and the cosine tail
for any number of groups — which P3's 3a (1 group) / 3b (2–3 groups) path needs.
"""

from __future__ import annotations

import math
import typing as tp

from neuraltrain.optimizers.base import BaseLRScheduler
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class _WarmupCosineLR(LRScheduler):
    """Linear warmup 0→peak over ``warmup_steps``, then cosine decay to
    ``min_lr_ratio · peak`` over the remaining steps — applied per param group.

    ``warmup_steps`` is clamped strictly below ``total_steps`` so the cosine
    window is always ≥ 1 step (short smoke runs can request ``warmup ≥ total``).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1; got {total_steps}")
        self.warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))
        self.total_steps = int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        super().__init__(optimizer, last_epoch=last_epoch)

    def _multiplier(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return min(1.0, (step + 1) / self.warmup_steps)
        cos_total = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / cos_total))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def get_lr(self) -> list[float]:  # type: ignore[override]
        mult = self._multiplier(self.last_epoch)
        return [base_lr * mult for base_lr in self.base_lrs]


class WarmupCosine(BaseLRScheduler):
    """Config for :class:`_WarmupCosineLR`, resolved by class name through exca's
    ``DiscriminatedModel`` (``scheduler={"name": "WarmupCosine", ...}``).

    Fields are TOP-LEVEL — this is a plain ``BaseLRScheduler`` subclass, NOT a
    ``BaseTorchLRScheduler`` (those use a ``kwargs`` dict). ``DiscriminatedModel``
    is ``extra="forbid"``, so a stray ``kwargs`` wrapper would fail validation.

    ``total_steps`` is consumed as a build-kwarg from ``configure_optimizers``
    (``trainer.estimated_stepping_batches``); the optional field here is only a
    fallback for the no-trainer unit-test path, where the schedule degrades to a
    constant LR rather than crashing on a ``None`` horizon.
    """

    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    total_steps: int | None = None

    def build(  # type: ignore[override]
        self,
        optimizer: Optimizer,
        *,
        total_steps: int | None = None,
        **_: tp.Any,
    ) -> LRScheduler:
        steps = total_steps if total_steps is not None else self.total_steps
        if steps is None:
            # No horizon (no trainer attached, no explicit total) → constant LR.
            return _WarmupCosineLR(
                optimizer, warmup_steps=0, total_steps=1, min_lr_ratio=1.0
            )
        return _WarmupCosineLR(
            optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=int(steps),
            min_lr_ratio=self.min_lr_ratio,
        )


__all__ = ["WarmupCosine", "_WarmupCosineLR"]
