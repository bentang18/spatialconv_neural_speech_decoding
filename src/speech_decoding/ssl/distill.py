"""Phase-3 cross-modal distillation loss (T2.4).

Wraps the Phase-3 student → Whisper-L8 supervision from the 5/22 iMINDBench
pivot, as amended by the B05/B06 lock (2026-05-25 PM) that moved the pool +
projection to the teacher side and made the student an identity passthrough:

    student_readout: (B, 40, 256)  ←  encoder PMA-k=1 over parcels; identity
                                       passthrough at 8 Hz native (NO
                                       student-side pool or projection)
    teacher_target:  (B, 40, 256)  ←  Whisper-L8 (B, 250, 1280) → triangular
                                       pool 50→8 Hz (whisper_teacher_pool) →
                                       WhisperAdapter 2-layer MLP (1280→256)

Both meet at d=256 (40 buckets = 8 Hz × 5 s). The teacher target is expected
to be detached (caller's responsibility).

Open blockers (see docs/neuroprobe/v14_blockers.md):

  * **M04** — Smooth-L1 β value not locked. ``beta`` is a required field of
    :class:`PhaseThreeDistillationConfig` (no default that would silently
    pin the decision).
  * **M05** — Whisper instance-norm axes not locked. We do NOT instance-norm
    inside this loss. If the caller decides M05's default candidate
    (per-token over the 1280-channel axis), they pre-normalize the teacher
    target before calling. Optional ``target_instance_norm`` hook exists for
    AB10 sweep convenience but defaults to ``False``.
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

    Caller is responsible for ``stop_grad`` on teacher; this function does
    NOT detach. (We keep teacher gradient-routable so it's traceable in
    debugging; the recipe just attaches a detached tensor here.)

    ``target_instance_norm=True`` activates M05's default candidate: per-token
    instance-norm across the feature dim on the *teacher* side only. Kept
    opt-in until M05 settles.
    """
    if student.shape != teacher.shape:
        raise ValueError(
            f"student.shape ({tuple(student.shape)}) != "
            f"teacher.shape ({tuple(teacher.shape)})"
        )

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
