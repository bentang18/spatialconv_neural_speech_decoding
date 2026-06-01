"""Phase-3 cross-modal distillation loss (T2.4).

Wraps the Phase-3 student → Whisper supervision from the 5/22 iMINDBench pivot.
B33 (2026-05-30) flipped distillation from project-DOWN to **project-UP**: the
student projects its 256-d readout up to the fixed 1280-d Whisper target via the
``StudentWhisperProjector`` head, and the loss lives in 1280-d. The teacher
carries no trainable module — its target is the raw all-layer-mean feature
(``layer_merge="mean_all"``; 2026-05-30) run through a fixed, train-only
per-channel z-score (``TargetStandardizer``):

    student_readout: (B, 40, 256)   ←  encoder PMA-k=1 over parcels (identity
                                        passthrough at 8 Hz native)
    student:         (B, 40, 1280)  ←  StudentWhisperProjector 256→1280 head
    teacher_target:  (B, 40, 1280)  ←  Whisper all-layer mean (B, 250, 1280) →
                                        triangular pool 50→8 Hz
                                        (whisper_teacher_pool) → per-channel
                                        z-score (TargetStandardizer); FIXED, detached

Both meet at d=1280 (40 buckets = 8 Hz × 5 s). The loss math is shape-agnostic;
this module is unchanged from project-down except for the contract it documents.
The teacher target is a fixed cached + standardized tensor, so the ``detach()``
below is trivially correct (no teacher-side trainable module to starve).

Do not conflate the two distinct normalization axes:

  * **Per-channel corpus z-score** (mandatory, B33): one scalar ``(μ, σ)`` per
    feature channel, fit train-only over ``(clips × timesteps)``, applied
    UPSTREAM by ``TargetStandardizer`` before the target reaches this loss —
    NOT done here.
  * **``target_instance_norm``** (opt-in, default ``False``): per-**token**
    instance-norm across the 1280-channel axis (M05 candidate). Different axis,
    computed here when enabled. Orthogonal to the per-channel z-score.

Open blockers (see docs/neuroprobe/v14_blockers.md):

  * **M04** — Smooth-L1 β value not locked. ``beta`` is a required field of
    :class:`PhaseThreeDistillationConfig` (no default that would silently
    pin the decision).
  * **M05** — Whisper instance-norm axes not locked. We do NOT per-token
    instance-norm inside this loss by default; ``target_instance_norm``
    (per-token over the 1280-channel axis) defaults to ``False``.
  * **AB10** — R-loss-family sweep. ``loss_form`` accepts ``mse``,
    ``smooth_l1``, or ``cosine`` to make the sweep one-liner.
"""

from __future__ import annotations

import typing as tp

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor


def phase3_distillation_loss(
    student: Tensor,
    teacher: Tensor,
    *,
    loss_form: tp.Literal["mse", "smooth_l1", "cosine"] = "smooth_l1",
    beta: float = 1.0,
    target_instance_norm: bool = False,
    eps: float = 1e-8,
) -> Tensor:
    """Phase-3 distillation loss.

    The teacher target is ``detach()``-ed inside this function: stop-grad on a
    feature-regression target is a load-bearing collapse-safety property
    (SimSiam, data2vec), so it is enforced here as a code invariant rather than
    left to the caller. The recipe already passes a detached tensor; the extra
    detach is idempotent.

    ``target_instance_norm=True`` activates M05's default candidate: per-token
    instance-norm across the feature dim on the *teacher* side only. Kept
    opt-in until M05 settles.
    """
    if student.shape != teacher.shape:
        raise ValueError(
            f"student.shape ({tuple(student.shape)}) != "
            f"teacher.shape ({tuple(teacher.shape)})"
        )

    teacher = teacher.detach()

    if target_instance_norm:
        mu = teacher.mean(dim=-1, keepdim=True)
        sigma = teacher.std(dim=-1, keepdim=True).clamp(min=eps)
        teacher = (teacher - mu) / sigma

    if loss_form == "mse":
        return torch.nn.functional.mse_loss(student, teacher)
    if loss_form == "smooth_l1":
        return torch.nn.functional.smooth_l1_loss(student, teacher, beta=beta)
    if loss_form == "cosine":
        # 1 - mean cosine similarity over the feature dim, averaged over the
        # leading (batch × time) dims. Matches V-JEPA / data2vec audio
        # cosine-objective convention.
        s_norm = torch.nn.functional.normalize(student, dim=-1)
        t_norm = torch.nn.functional.normalize(teacher, dim=-1)
        return 1.0 - (s_norm * t_norm).sum(dim=-1).mean()
    raise ValueError(f"unknown loss_form={loss_form!r}")


class PhaseThreeDistillationConfig(BaseModel):
    """Pydantic-shaped distillation-loss config; β is required (no implicit
    default that pins M04). Pass to a NeuralTrain Experiment via Exca grid."""

    model_config = ConfigDict(extra="forbid")

    loss_form: tp.Literal["mse", "smooth_l1", "cosine"] = "smooth_l1"
    beta: float = 1.0  # B01 v3 lock (P3-04): Smooth-L1 β=1.0 default. Sister AB10 sweeps {0.5, 1.0, 2.0}.
    target_instance_norm: bool = False
    eps: float = 1e-8

    def __call__(self, student: Tensor, teacher: Tensor) -> Tensor:
        return phase3_distillation_loss(
            student,
            teacher,
            loss_form=self.loss_form,
            beta=self.beta,
            target_instance_norm=self.target_instance_norm,
            eps=self.eps,
        )
