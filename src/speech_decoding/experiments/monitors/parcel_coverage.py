"""MON-PARCEL-COVERAGE-VARIANCE: B30 latent_valid degeneracy monitor.

Catches the failure mode where a corpus + ref-aug + shaft-mask
combination produces degenerate routing — most clips have ≤1 active
parcel slot, the slot-axis losses (``L_mid_slot``, ``L_post_frame``,
``L_post_utterance``) silently collapse onto a single slot, and the
training appears healthy while throwing away the per-parcel inductive
bias.

This is v14-original. The closest published precedent is the load-
balance loss in Switch Transformer (Fedus et al. JMLR'22 Eq. 4),
which targets the same topology: expert/slot fractions per batch and
their variance. v14 doesn't need the loss term — DK routing is
analytical, not learned — but the *same statistics* are an early-
warning signal when ``latent_valid`` degenerates.

The B30 contract (`memory/project_v14_anatomy_gated_symmetric_2026_05_28`)
makes ``latent_valid = (support.sum(over electrodes) > 0)`` the single
source of truth. SWEC clips legitimately yield all-False slot masks
(zero anatomy → front-end-only contribution). The monitor distinguishes
two cohorts:

* **anatomy-bearing corpora** (BT / D-cohort / AJILE12): per-clip
  active-slot count should hover near ``K * coverage_ratio`` with
  modest variance.
* **front-end-only corpora** (SWEC): per-clip active-slot count is 0;
  these are excluded from variance computation so a healthy SWEC
  batch doesn't pollute the BT signal.

Cost: cheap. Sums over a bool tensor, no extra forward pass.

Outputs per batch:

* ``active_slots_per_clip_mean`` — mean across the batch.
* ``active_slots_per_clip_std`` — std across the batch.
* ``active_slots_per_clip_cv`` — coefficient of variation (std / mean);
  the unit-free comparable across batch sizes.
* ``slot_usage_fraction_var`` — variance of the per-slot usage fraction
  across the slot axis (Switch Transformer load-balance analog).
* ``degenerate_clip_fraction`` — fraction of clips with ≤1 active slot
  (post-exclusion of front-end-only clips). Hard alarm above 0.1
  per the v14 sister-escalation table.
* ``front_end_only_clip_fraction`` — fraction of clips in the batch
  with zero active slots (informational; not an alarm).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# Hard alarms (instantaneous; sustain logic is dispatch-side).
DEGENERATE_CLIP_FRACTION_ALARM: float = 0.10
DEGENERATE_SLOT_COUNT_MAX: int = 1
SLOT_USAGE_VARIANCE_ALARM: float = 0.05


@dataclass(frozen=True)
class ParcelCoverageVerdict:
    """Instantaneous MON-PARCEL-COVERAGE-VARIANCE verdict.

    All quantities are computed on the *anatomy-bearing* sub-batch:
    front-end-only clips (latent_valid all-False) are excluded before
    aggregation so SWEC presence doesn't bias the BT-side signal.

    Fields
    ------
    active_slots_per_clip_mean:
        Mean of ``latent_valid.sum(dim=1)`` across anatomy-bearing
        clips. Healthy regime ≈ ``K * subject_coverage_fraction``
        (K=80 default, coverage ≈ 0.4 for BT9 → ~32).
    active_slots_per_clip_std:
        Std of the same per-clip count.
    active_slots_per_clip_cv:
        ``std / mean`` (coefficient of variation). Healthy ≈ 0.1–0.3;
        > 0.5 = some subjects in the batch are vastly under-routed.
    slot_usage_fraction_var:
        Variance of the per-slot usage fraction across L slots. High
        variance = some slots monopolised by a few clips, others
        unused. Switch Transformer load-balance analog.
    degenerate_clip_fraction:
        Fraction of anatomy-bearing clips with
        ``active_slots <= DEGENERATE_SLOT_COUNT_MAX``. Hard alarm at
        ``> DEGENERATE_CLIP_FRACTION_ALARM``.
    front_end_only_clip_fraction:
        Fraction of clips in the full batch with zero active slots
        (SWEC etc.). Informational; not a kill signal.
    is_alarm:
        ``True`` iff ``degenerate_clip_fraction >
        DEGENERATE_CLIP_FRACTION_ALARM``. ``slot_usage_fraction_var`` is
        logged but does NOT trip the alarm: under K=80 sparse per-clip DK
        coverage the per-slot usage fraction is structurally bimodal
        (a parcel is either wired or never wired across the sub-batch),
        so its variance sits ≈ p(1-p) — well above
        ``SLOT_USAGE_VARIANCE_ALARM`` by construction — and is
        non-discriminative. ``degenerate_clip_fraction`` is the
        load-bearing degeneracy signal.
    """

    active_slots_per_clip_mean: float
    active_slots_per_clip_std: float
    active_slots_per_clip_cv: float
    slot_usage_fraction_var: float
    degenerate_clip_fraction: float
    front_end_only_clip_fraction: float
    is_alarm: bool


def parcel_coverage_monitor(
    latent_valid: Tensor,
    *,
    degenerate_slot_count_max: int = DEGENERATE_SLOT_COUNT_MAX,
    degenerate_clip_fraction_alarm: float = DEGENERATE_CLIP_FRACTION_ALARM,
) -> ParcelCoverageVerdict:
    """Compute the instantaneous MON-PARCEL-COVERAGE-VARIANCE verdict.

    Parameters
    ----------
    latent_valid:
        ``(B, L)`` bool tensor; the B30 single source of truth. ``True``
        at ``(b, l)`` iff slot ``l`` of clip ``b`` has ≥1 covered
        electrode under the anatomy-gated symmetric rule.

    Returns
    -------
    ParcelCoverageVerdict — see field docs above. When the batch
    contains no anatomy-bearing clips (e.g. an all-SWEC batch),
    per-clip stats are ``0.0`` and ``is_alarm`` is ``False``.
    """
    if latent_valid.dim() != 2:
        raise ValueError(
            f"latent_valid must be (B, L); got shape {tuple(latent_valid.shape)}"
        )
    if latent_valid.dtype != torch.bool:
        raise ValueError(
            f"latent_valid must be bool; got dtype {latent_valid.dtype}"
        )

    B, L = latent_valid.shape
    counts = latent_valid.sum(dim=1)                              # (B,) int
    is_front_end_only = counts == 0                                # (B,) bool
    front_end_only_clip_fraction = float(
        is_front_end_only.float().mean().item()
    ) if B > 0 else 0.0

    anatomy_mask = ~is_front_end_only
    anatomy_counts = counts[anatomy_mask].to(torch.float32)
    n_anatomy = int(anatomy_counts.numel())

    if n_anatomy == 0:
        return ParcelCoverageVerdict(
            active_slots_per_clip_mean=0.0,
            active_slots_per_clip_std=0.0,
            active_slots_per_clip_cv=0.0,
            slot_usage_fraction_var=0.0,
            degenerate_clip_fraction=0.0,
            front_end_only_clip_fraction=front_end_only_clip_fraction,
            is_alarm=False,
        )

    mean = float(anatomy_counts.mean().item())
    # Unbiased std for n >= 2; population std for n == 1.
    if n_anatomy >= 2:
        std = float(anatomy_counts.std(unbiased=True).item())
    else:
        std = 0.0
    cv = (std / mean) if mean > 0.0 else 0.0

    degenerate = anatomy_counts <= float(degenerate_slot_count_max)
    degenerate_clip_fraction = float(degenerate.float().mean().item())

    # Per-slot usage fraction across the anatomy-bearing sub-batch.
    # (L,) float; variance across L = Switch-Transformer load-balance term.
    anatomy_latent = latent_valid[anatomy_mask].to(torch.float32)  # (n_anat, L)
    slot_usage = anatomy_latent.mean(dim=0)                        # (L,)
    if L >= 2:
        slot_usage_var = float(slot_usage.var(unbiased=True).item())
    else:
        slot_usage_var = 0.0

    # slot_usage_var is computed and reported (below) but is NOT an alarm
    # trigger: under K=80 sparse per-clip DK coverage the per-slot usage
    # fraction is structurally bimodal, so its variance ≈ p(1-p) exceeds
    # SLOT_USAGE_VARIANCE_ALARM by construction and is non-discriminative.
    # degenerate_clip_fraction is the load-bearing degeneracy signal.
    is_alarm = degenerate_clip_fraction > degenerate_clip_fraction_alarm

    return ParcelCoverageVerdict(
        active_slots_per_clip_mean=mean,
        active_slots_per_clip_std=std,
        active_slots_per_clip_cv=cv,
        slot_usage_fraction_var=slot_usage_var,
        degenerate_clip_fraction=degenerate_clip_fraction,
        front_end_only_clip_fraction=front_end_only_clip_fraction,
        is_alarm=bool(is_alarm),
    )
