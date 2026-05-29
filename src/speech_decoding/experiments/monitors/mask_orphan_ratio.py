"""MON-MASK-002: bounded orphan-vs-visible mean-squared-error ratio.

Spec: `docs/neuroprobe/v14_blockers.md §B03d` (registered 2026-05-25 PM) +
the K=1-default rewording in `memory/project_v14_b03_mask_lock_2026_05_25.md`.

For every step in the joint phase (B29 merged P1 + P2; this monitor was
originally a "P2-only" probe but the bug it catches — mean-collapse on the
shaft-mask supervision signal — exists in the joint phase exactly as it did
in standalone P2):

* ``orphan_MSE`` = mean reconstruction MSE on parcels that lost all of
  their electrodes to the shaft mask (the JEPA "prediction target" set).
* ``visible_MSE`` = mean reconstruction MSE on parcels with at least one
  unmasked electrode (the JEPA "context" set).
* ``ratio = orphan_MSE / visible_MSE``. Healthy band ``[0.7, 1.5]``.

Escalation table (under the K=1 default, 2026-05-27 PM):

* ``ratio < 0.7`` — mean-collapse (student is producing the
  orphan-marginal mean and dodging the prediction task). Escalate to
  ``R-stratified-shaft-mask`` (avoids uniform shaft selection making the
  same parcels orphan in nearly every clip).
* ``ratio > 1.5`` — student is failing the orphan task (sparser orphans
  under K=1 may be under-providing gradient). Escalate UP to
  ``R-shaft-K2`` (P0); secondary investigation is
  ``R-stratified-shaft-mask`` if shaft-selection uniformity is the root
  cause.

Sustain rule: more than 50 consecutive out-of-band steps triggers the
escalation. The 50-step sustain logic lives in the training loop — this
module returns the **instantaneous** verdict at one logging step.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import torch
from torch import Tensor


MIN_RATIO: float = 0.7
"""Lower bound of healthy band: ratio < MIN_RATIO → mean-collapse."""

MAX_RATIO: float = 1.5
"""Upper bound of healthy band: ratio > MAX_RATIO → orphan-task failure."""

SUSTAIN_STEPS_DEFAULT: int = 50

ESCALATE_STRATIFIED_SHAFT_MASK: tp.Final[str] = "R-stratified-shaft-mask"
ESCALATE_SHAFT_K2: tp.Final[str] = "R-shaft-K2"


@dataclass(frozen=True)
class MaskOrphanRatioVerdict:
    """Instantaneous MON-MASK-002 probe verdict.

    Fields
    ------
    orphan_mse:
        Mean of the per-parcel reconstruction MSE on parcels that lost
        all of their electrodes to the shaft mask. ``nan`` when the
        batch has no orphan parcels.
    visible_mse:
        Mean of the per-parcel reconstruction MSE on parcels with at
        least one unmasked electrode. ``nan`` when the batch has no
        visible parcels.
    ratio:
        ``orphan_mse / visible_mse``. ``nan`` when either side is
        absent (training-loop ignores; sustain counter does NOT tick).
    in_band:
        ``True`` iff the ratio is finite and in ``[MIN_RATIO, MAX_RATIO]``.
    escalations:
        Tuple of sister labels that the instantaneous ratio triggers.
        Empty when the ratio is in band or undefined. The training loop
        OR's these across a sustain window before flipping a sister.
    """

    orphan_mse: float
    visible_mse: float
    ratio: float
    in_band: bool
    escalations: tuple[str, ...]


def _per_parcel_mse(
    pred: Tensor, target: Tensor, *, parcel_dim: int,
) -> Tensor:
    """Reduce ``(pred - target) ** 2`` over every axis except ``parcel_dim``."""
    sq = (pred - target).pow(2)
    # Move parcel axis to position 0, then mean over everything else.
    sq = sq.movedim(parcel_dim, 0)
    P = sq.shape[0]
    return sq.reshape(P, -1).mean(dim=-1)


def mask_orphan_ratio_monitor(
    student_post_frame: Tensor,
    teacher_post_frame: Tensor,
    orphan_parcels: Tensor,
    *,
    parcel_dim: int = -2,
    min_ratio: float = MIN_RATIO,
    max_ratio: float = MAX_RATIO,
) -> MaskOrphanRatioVerdict:
    """Compute the instantaneous orphan/visible MSE ratio.

    Parameters
    ----------
    student_post_frame: any shape ``(..., P, d)`` (or with ``P`` at
        ``parcel_dim``). Post-frame student tap aligned to parcels.
    teacher_post_frame: same shape as ``student_post_frame``. EMA
        teacher's full-input post-frame tap.
    orphan_parcels: ``(P,) bool`` — ``True`` at every parcel that lost
        all of its electrodes to the shaft mask in the current batch.
        Computed by the caller from ``shaft_mask`` × ``support`` at
        joint-phase step time.
    parcel_dim: which axis of the post-frame taps is the parcel axis.
        Default ``-2`` matches the M4 ``(B, P, T, d)`` layout written by
        the encoder (parcel before time before features).
    min_ratio / max_ratio: healthy band. Defaults: B03d lock 0.7 / 1.5.

    Returns
    -------
    MaskOrphanRatioVerdict
        ``orphan_mse``, ``visible_mse``, ``ratio``, ``in_band``, and the
        escalations triggered by the instantaneous ratio.
    """
    if student_post_frame.shape != teacher_post_frame.shape:
        raise ValueError(
            "student_post_frame and teacher_post_frame must share shape; "
            f"got {tuple(student_post_frame.shape)} vs "
            f"{tuple(teacher_post_frame.shape)}"
        )
    if orphan_parcels.dtype is not torch.bool:
        raise ValueError(
            f"orphan_parcels must be bool; got dtype {orphan_parcels.dtype}"
        )
    if orphan_parcels.dim() != 1:
        raise ValueError(
            f"orphan_parcels must be 1-d (P,); got shape "
            f"{tuple(orphan_parcels.shape)}"
        )
    P = orphan_parcels.shape[0]
    parcel_axis = (
        parcel_dim if parcel_dim >= 0 else parcel_dim + student_post_frame.dim()
    )
    if student_post_frame.shape[parcel_axis] != P:
        raise ValueError(
            f"parcel axis size mismatch: post-frame has "
            f"{student_post_frame.shape[parcel_axis]} parcels at axis "
            f"{parcel_axis} but orphan_parcels has {P}"
        )

    mse_per_parcel = _per_parcel_mse(
        student_post_frame.detach(),
        teacher_post_frame.detach(),
        parcel_dim=parcel_axis,
    )  # (P,)

    visible = ~orphan_parcels
    orphan_n = int(orphan_parcels.sum().item())
    visible_n = int(visible.sum().item())

    if orphan_n == 0 or visible_n == 0:
        return MaskOrphanRatioVerdict(
            orphan_mse=float("nan"),
            visible_mse=float("nan"),
            ratio=float("nan"),
            in_band=False,
            escalations=(),
        )

    orphan_mse = float(mse_per_parcel[orphan_parcels].mean().item())
    visible_mse = float(mse_per_parcel[visible].mean().item())
    if visible_mse <= 0.0:
        # No supervision happening on the visible side — pathological,
        # but don't claim "in_band" by accident.
        return MaskOrphanRatioVerdict(
            orphan_mse=orphan_mse,
            visible_mse=visible_mse,
            ratio=float("nan"),
            in_band=False,
            escalations=(),
        )

    ratio = orphan_mse / visible_mse
    in_band = (min_ratio <= ratio <= max_ratio)
    escalations: tuple[str, ...] = ()
    if ratio < min_ratio:
        escalations = (ESCALATE_STRATIFIED_SHAFT_MASK,)
    elif ratio > max_ratio:
        escalations = (ESCALATE_SHAFT_K2,)

    return MaskOrphanRatioVerdict(
        orphan_mse=orphan_mse,
        visible_mse=visible_mse,
        ratio=ratio,
        in_band=in_band,
        escalations=escalations,
    )


def compute_orphan_parcels(
    shaft_mask: Tensor,
    support: Tensor,
    *,
    threshold: float = 0.0,
) -> Tensor:
    """Identify parcels that lost ALL of their electrodes to the shaft mask.

    Parameters
    ----------
    shaft_mask: ``(B, C) bool`` — ``True`` at every electrode dropped by
        the shaft mask in the current step.
    support: ``(B, C, P) float`` — DK one-hot (or soft) per-electrode
        parcel routing weights. ``support[b, c, p] > threshold`` flags
        electrode ``c`` of clip ``b`` as covering parcel ``p``.
    threshold: support cutoff above which an electrode is considered to
        cover a parcel. ``0.0`` matches the DK one-hot convention.

    Returns
    -------
    ``(P,) bool`` — ``True`` at parcel ``p`` iff EVERY clip in the batch
    has lost all of parcel ``p``'s electrodes to the shaft mask. The
    batch-level reduction (intersection across clips) matches the
    "compute parcel-side orphan once per batch" simplification — finer
    per-clip orphan tracking is reserved for the per-clip variant in
    the joint module's loss path.
    """
    if shaft_mask.dtype is not torch.bool:
        raise ValueError(
            f"shaft_mask must be bool; got dtype {shaft_mask.dtype}"
        )
    if shaft_mask.dim() != 2:
        raise ValueError(
            f"shaft_mask must be (B, C); got shape {tuple(shaft_mask.shape)}"
        )
    if support.dim() != 3:
        raise ValueError(
            f"support must be (B, C, P); got shape {tuple(support.shape)}"
        )
    if support.shape[:2] != shaft_mask.shape:
        raise ValueError(
            f"support[..., :2] must match shaft_mask; got "
            f"{tuple(support.shape[:2])} vs {tuple(shaft_mask.shape)}"
        )

    covers = support > threshold                          # (B, C, P) bool
    visible = ~shaft_mask                                 # (B, C)   bool
    visible_covers = covers & visible.unsqueeze(-1)       # (B, C, P) bool
    parcel_has_visible_electrode = visible_covers.any(dim=1)  # (B, P)
    parcel_orphan_per_clip = ~parcel_has_visible_electrode    # (B, P)
    return parcel_orphan_per_clip.all(dim=0)              # (P,)


__all__: tp.Final[tuple[str, ...]] = (
    "ESCALATE_SHAFT_K2",
    "ESCALATE_STRATIFIED_SHAFT_MASK",
    "MAX_RATIO",
    "MIN_RATIO",
    "MaskOrphanRatioVerdict",
    "SUSTAIN_STEPS_DEFAULT",
    "compute_orphan_parcels",
    "mask_orphan_ratio_monitor",
)
