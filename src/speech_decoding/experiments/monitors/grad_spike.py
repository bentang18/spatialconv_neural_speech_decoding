"""MON-GRAD-SPIKE-DIVERGENCE: AdamW grad-L2 spike + divergence monitor.

Catches the failure mode where AdamW diverges (loss → NaN / blows up)
under joint Phase-1+Phase-2 training (B29) and α=0.3 multi-corpus
sampling — particularly acute when a SWEC / AJILE12 batch lands with
outlier statistics relative to the BT/D-cohort majority.

Precedent: SPAM (Huang et al. 2025, arXiv 2501.06842 §3.1) — exponential
moving average of the per-step grad-L2 + spike detection at
``current / ema > k`` for ``k`` in {3, 5, 8}. Wortsman et al. 2024
("Small-scale proxies for large-scale Transformer training instabilities",
arXiv 2309.14322 §2) document the same protocol on LLM training.

Cost: **free**. AdamW already requires `clip_grad_norm_` to produce a
grad-L2 number; this monitor only reads it.

Wire-point in :class:`V14JointBrainModule`:

* On every train step, after ``loss.backward()`` and before
  ``optimizer.step()``, call :func:`grad_spike_monitor` with the L2 the
  trainer already computed for gradient clipping.
* The module owns a single scalar buffer for the EMA so spike detection
  survives ``trainer.fit_loop`` restarts and the inter-step gap.

The per-step logs are::

  train_mon_grad_l2          # current grad-L2
  train_mon_grad_ema_l2      # EMA of grad-L2 (alpha = 0.99 default)
  train_mon_grad_spike_ratio # current / ema (1.0 at steady state)
  train_mon_grad_spike       # 1.0 when ratio > spike_threshold

Sustain logic lives in the dispatch callback (N consecutive spikes →
checkpoint rollback + LR backoff) per SPAM §3.1. This module returns the
*instantaneous* verdict at one optimiser step.
"""

from __future__ import annotations

from dataclasses import dataclass

import math


# SPAM defaults. ``alpha`` is the EMA decay (higher = longer memory);
# ``spike_threshold`` is the ratio that triggers a spike flag.
GRAD_EMA_ALPHA: float = 0.99
GRAD_SPIKE_THRESHOLD: float = 5.0


@dataclass(frozen=True)
class GradSpikeVerdict:
    """Instantaneous MON-GRAD-SPIKE-DIVERGENCE probe verdict.

    Fields
    ------
    grad_l2:
        Current step's gradient L2 norm.
    grad_ema_l2:
        EMA of the gradient L2 norm prior to this step (the comparison
        baseline). Equal to ``grad_l2`` on the first step (no prior
        history → no spike).
    spike_ratio:
        ``grad_l2 / grad_ema_l2``. 1.0 = steady state; >> 1 = spike.
        Returns ``0.0`` when ``grad_ema_l2 == 0`` (no history) to avoid
        a divide-by-zero noise flag on step 0.
    is_spike:
        ``True`` iff ``spike_ratio > spike_threshold``.
    is_diverged:
        ``True`` iff ``grad_l2`` is non-finite (NaN / Inf). A diverged
        step is also flagged as a spike for cadence purposes.
    new_grad_ema_l2:
        EMA *after* mixing in the current step. The caller writes this
        back to its persistent buffer.
    """

    grad_l2: float
    grad_ema_l2: float
    spike_ratio: float
    is_spike: bool
    is_diverged: bool
    new_grad_ema_l2: float


def grad_spike_monitor(
    grad_l2: float,
    prior_ema_l2: float,
    *,
    alpha: float = GRAD_EMA_ALPHA,
    spike_threshold: float = GRAD_SPIKE_THRESHOLD,
) -> GradSpikeVerdict:
    """Compute the instantaneous MON-GRAD-SPIKE-DIVERGENCE verdict.

    Parameters
    ----------
    grad_l2:
        This step's gradient L2 norm. May be ``nan`` / ``inf`` to flag
        AdamW divergence (the monitor reports the divergence; it does
        not nan-poison the EMA buffer).
    prior_ema_l2:
        The persistent EMA value from the prior step. ``0.0`` on the
        very first step bypasses the spike check (no baseline yet).
    alpha:
        EMA decay coefficient. SPAM default ``0.99``; higher = longer
        history. Must be in ``[0, 1)``.
    spike_threshold:
        Ratio above which a spike is flagged. SPAM default ``5.0``;
        Wortsman et al. 2024 uses 3.0 (more sensitive) — sister-cell
        candidate.

    Returns
    -------
    GradSpikeVerdict — see field docs above.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1); got {alpha}")
    if spike_threshold <= 1.0:
        raise ValueError(
            f"spike_threshold must be > 1.0 (ratio); got {spike_threshold}"
        )

    is_diverged = not math.isfinite(grad_l2)

    if is_diverged:
        # Don't mix Inf/NaN into the persistent EMA — keep it usable for
        # post-spike recovery once the optimiser un-explodes.
        new_ema = prior_ema_l2
        spike_ratio = math.inf
        is_spike = True
    else:
        # First step (no history) → seed EMA to current and skip spike flag.
        if prior_ema_l2 <= 0.0:
            new_ema = grad_l2
            spike_ratio = 0.0
            is_spike = False
        else:
            spike_ratio = grad_l2 / prior_ema_l2
            is_spike = spike_ratio > spike_threshold
            new_ema = alpha * prior_ema_l2 + (1.0 - alpha) * grad_l2

    return GradSpikeVerdict(
        grad_l2=float(grad_l2) if math.isfinite(grad_l2) else float("nan"),
        grad_ema_l2=float(prior_ema_l2),
        spike_ratio=float(spike_ratio),
        is_spike=bool(is_spike),
        is_diverged=bool(is_diverged),
        new_grad_ema_l2=float(new_ema),
    )
