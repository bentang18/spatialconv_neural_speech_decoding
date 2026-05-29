"""MON-SLOT-REDUNDANCY: B28 slot-bank collapse monitor.

Lock memo: ``memory/project_v14_loss_design_amendment_b28_2026_05_27.md``
§MON-SLOT-REDUNDANCY (B28 Item 1 + B29 Item 13 rescale).

Probe every 10k steps on a held-out 1024-clip probe batch (B29 Item 13
rescale: 256 → 1024 to compensate for 4× pct95 estimator noise from
16× fewer slot-pairs after the M=4 → M=1 collapse).

Thresholds (post-B29 rescale; pre-preflight):

  * ``per_clip_cos.pct95 > 0.5`` sustained ≥ 50k → escalate
    ``R-dkoleo-intra-clip-slots`` (was 0.7 pre-rescale).
  * ``batch_cos.pct95 > 0.7`` sustained ≥ 50k → escalate
    ``R-dkoleo-batch-cls-unit`` (unchanged — K-invariant).
  * ``per_clip_cos.diag-zeroed.mean > 0.35`` sustained ≥ 50k →
    escalate ``R-vicreg-slot-variance`` (was 0.5 pre-rescale).

The numbers above are *pre-preflight*; a 1-GPU-h BT-Lite calibration
preflight is required before Phase-1 dispatch per B28 §MON-SLOT-REDUNDANCY
threshold rescale.

This module exposes the pure probe function and a small verdict struct.
The 50k-step sustain logic lives in the training loop — this file only
returns the *instantaneous* probe verdict at one logging step.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import torch
from torch import Tensor


# B29 Item 13 (5/27 PM-late) post-M=1-rescale thresholds.
PER_CLIP_COS_PCT95_THRESHOLD: float = 0.5
BATCH_COS_PCT95_THRESHOLD: float = 0.7
DIAG_ZEROED_MEAN_THRESHOLD: float = 0.35
PROBE_BATCH_SIZE_M1_DEFAULT: int = 1024
SUSTAIN_STEPS_DEFAULT: int = 50_000

# Sister labels the verdict can escalate to. The training loop maps each
# label to a dispatch override (e.g. flipping ``dkoleo_mode``).
ESCALATE_INTRA_CLIP_SLOTS: tp.Final[str] = "R-dkoleo-intra-clip-slots"
ESCALATE_BATCH_CLS_UNIT: tp.Final[str] = "R-dkoleo-batch-cls-unit"
ESCALATE_VICREG_SLOT_VARIANCE: tp.Final[str] = "R-vicreg-slot-variance"


@dataclass(frozen=True)
class SlotRedundancyVerdict:
    """Instantaneous MON-SLOT-REDUNDANCY probe verdict.

    Fields
    ------
    per_clip_cos_pct95:
        95th percentile of pairwise cosine similarity between slots
        *within* each clip's slot bank, averaged across the probe batch.
        High = slots collapsing toward each other inside a clip.
    batch_cos_pct95:
        95th percentile of pairwise cosine similarity between
        utterance-pooled vectors (PMA) *across* the probe batch.
        High = utterance-level collapse across clips.
    diag_zeroed_mean:
        Mean of the within-clip cosine-similarity matrix with the
        diagonal zeroed out (so the trivial 1.0 self-similarity entries
        don't dominate). Higher diag-zeroed mean = more uniform
        non-trivial similarity → VICReg-style variance collapse.
    escalations:
        Tuple of sister labels that the instantaneous numbers trigger.
        The training loop OR's these across windows of ``SUSTAIN_STEPS``
        before actually flipping a dispatch override.
    """

    per_clip_cos_pct95: float
    batch_cos_pct95: float
    diag_zeroed_mean: float
    escalations: tuple[str, ...]


def _pairwise_cosine(x: Tensor) -> Tensor:
    """Pairwise cosine similarity matrix for rows of ``x``.

    Input: ``x`` of shape ``(N, d)``. Output: ``(N, N)`` symmetric, with
    1.0 on the diagonal.
    """
    if x.dim() != 2:
        raise ValueError(f"expected (N, d); got shape {tuple(x.shape)}")
    x_n = torch.nn.functional.normalize(x, dim=-1)
    return x_n @ x_n.transpose(0, 1)


def _upper_triangular_pct95(mat: Tensor) -> float:
    """95th percentile of strict-upper-triangular entries.

    Avoids double-counting pairs and excludes the trivial 1.0 diagonal.
    Returns ``0.0`` when ``mat`` has fewer than 2 rows.
    """
    N = mat.shape[0]
    if N < 2:
        return 0.0
    iu = torch.triu_indices(N, N, offset=1)
    vals = mat[iu[0], iu[1]]
    return float(torch.quantile(vals, q=0.95).item())


def _diag_zeroed_mean(mat: Tensor) -> float:
    """Mean of ``mat`` with the diagonal zeroed out.

    For an ``(N, N)`` matrix with ``N >= 2``, returns
    ``(sum(mat) - sum(diag(mat))) / (N * (N - 1))``.
    """
    N = mat.shape[0]
    if N < 2:
        return 0.0
    diag_zeroed = mat - torch.diag(torch.diagonal(mat))
    return float(diag_zeroed.sum().item() / (N * (N - 1)))


def slot_redundancy_monitor(
    slot_bank: Tensor,
    utterance_pma: Tensor,
    *,
    per_clip_cos_pct95_threshold: float = PER_CLIP_COS_PCT95_THRESHOLD,
    batch_cos_pct95_threshold: float = BATCH_COS_PCT95_THRESHOLD,
    diag_zeroed_mean_threshold: float = DIAG_ZEROED_MEAN_THRESHOLD,
) -> SlotRedundancyVerdict:
    """Compute the instantaneous MON-SLOT-REDUNDANCY verdict.

    Parameters
    ----------
    slot_bank: ``(B, L, d)`` Tensor of per-clip slot vectors at M4 (or
        whichever supervised checkpoint the loss head reads from). ``L``
        is the slot count — ``K * M`` (B29 default: 80).
    utterance_pma: ``(B, d)`` Tensor of per-clip utterance-level vectors
        (PMA-pooled M4 latents).

    Threshold kwargs override the locked defaults for sister sweeps and
    the BT-Lite preflight calibration.

    Returns
    -------
    SlotRedundancyVerdict — see field docs above.
    """
    if slot_bank.dim() != 3:
        raise ValueError(
            f"slot_bank must be (B, L, d); got shape {tuple(slot_bank.shape)}"
        )
    if utterance_pma.dim() != 2:
        raise ValueError(
            f"utterance_pma must be (B, d); got shape {tuple(utterance_pma.shape)}"
        )
    if slot_bank.shape[0] != utterance_pma.shape[0]:
        raise ValueError(
            f"slot_bank and utterance_pma must share batch dim; "
            f"got {slot_bank.shape[0]} vs {utterance_pma.shape[0]}"
        )

    B, L, _ = slot_bank.shape
    per_clip_pct95s: list[float] = []
    diag_zeroed_means: list[float] = []
    for b in range(B):
        mat = _pairwise_cosine(slot_bank[b])              # (L, L)
        per_clip_pct95s.append(_upper_triangular_pct95(mat))
        diag_zeroed_means.append(_diag_zeroed_mean(mat))
    per_clip_cos_pct95 = (
        sum(per_clip_pct95s) / max(len(per_clip_pct95s), 1)
    )
    diag_zeroed_mean = (
        sum(diag_zeroed_means) / max(len(diag_zeroed_means), 1)
    )

    batch_mat = _pairwise_cosine(utterance_pma)            # (B, B)
    batch_cos_pct95 = _upper_triangular_pct95(batch_mat)

    escalations: list[str] = []
    if per_clip_cos_pct95 > per_clip_cos_pct95_threshold:
        escalations.append(ESCALATE_INTRA_CLIP_SLOTS)
    if batch_cos_pct95 > batch_cos_pct95_threshold:
        escalations.append(ESCALATE_BATCH_CLS_UNIT)
    if diag_zeroed_mean > diag_zeroed_mean_threshold:
        escalations.append(ESCALATE_VICREG_SLOT_VARIANCE)

    return SlotRedundancyVerdict(
        per_clip_cos_pct95=per_clip_cos_pct95,
        batch_cos_pct95=batch_cos_pct95,
        diag_zeroed_mean=diag_zeroed_mean,
        escalations=tuple(escalations),
    )
