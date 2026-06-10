"""MON-TEACHER-FEATURE-RANK: RankMe dimensional-collapse monitor.

Catches the failure mode that MON-SLOT-REDUNDANCY (B28; cosine between
slots) cannot see: *dimensional* collapse within an individual slot's
feature, where all 256 dims become a low-rank affine of a small
subspace even while inter-slot cosines stay healthy. This is the
canonical V-JEPA / DINOv2 collapse symptom (data2vec-2.0 §3.2; DINOv3
§3.3); the EMA-teacher *target* is what collapses first because it has
no direct gradient pressure away from a degenerate fixed point.

Precedent: RankMe (Garrido et al. ICML'23, arXiv 2210.02885 §3). For a
feature matrix Z ∈ R^{N × d}, let σ_k be the singular values of Z and
``p_k = (σ_k + ε) / Σ(σ_j + ε)`` the normalised spectrum. Then::

    RankMe(Z) = exp(-Σ p_k log p_k)

Range: ``[1, min(N, d)]`` ; ``d`` = no collapse, ``1`` = full
dimensional collapse to a single direction. Smooth, differentiable,
batch-statistic only. DINOv3 uses RankMe as the primary feature-health
metric; data2vec-2.0 ships the same proxy under a different name.

Cost: one SVD per logging cadence. The recommended cadence (every 10k
optimiser steps on a held-out 1024-clip probe batch) makes the
amortised cost negligible. For (N=1024, d=256) the SVD is ~30 ms on
H100; the dispatch wires the cadence.

Outputs per probe call:

* ``rankme`` — RankMe (effective rank) on the EMA teacher's pooled
  feature matrix.
* ``rankme_normalised`` — ``rankme / d``. ``1.0`` = full rank;
  ``< 0.5`` = warning band; ``< 0.25`` = hard collapse alarm.
* ``is_alarm`` — ``True`` iff ``rankme_normalised < 0.25`` (hard
  collapse signature per DINOv3 §3.3 + RankMe §4.2 empirical band).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# DINOv3 §3.3 reports healthy RankMe / d > 0.6 across pretraining; the
# warning band starts around 0.5 (recoverable), hard alarm < 0.25
# (collapse signature; needs intervention per RankMe §4.2). This band fits
# the M2 front-end |STFT| representation, whose floor sits ~0.31 (measured,
# guard-OFF diagnostic job 47723576).
RANKME_NORMALISED_WARN: float = 0.5
RANKME_NORMALISED_ALARM: float = 0.25

# M4 parcel-token band. The M4 representation's effective rank is bounded by
# the active-parcel count (~14 for BT-lite-9), so its normalised RankMe floor
# is structurally ~0.05 (measured: chain 47725245 P2, raw rank oscillating
# 12.3–16.8 / 256 ≈ 0.047–0.066, born flat at init, val_loss falling). The
# DINOv3 0.5/0.25 band sits ~5× above this floor and fires from birth (the
# false-positive that killed that chain at P2 step 452). Anchored to the floor:
# warn 0.04 sits just below the oscillation minimum (advisory, log-only); alarm
# 0.02 (raw rank ~5, a >2.5× crater from the floor) still hard-kills a genuine
# collapse toward rank 1 (normalised ~0.004) with ~2.4× margin below the floor.
# See [[project_v14_gate_cadence_guard_response_lock_2026_06_05]].
RANKME_M4_NORMALISED_WARN: float = 0.04
RANKME_M4_NORMALISED_ALARM: float = 0.02

# B37 joint mean-pool band — used ONLY on the ``ssl_mode == "joint"`` path
# (B37 mean-pool encoder). The B37 mean-pool + thin parcel-SA latent produces a
# COMPACT representation: BOTH the M2 front-end and M4 parcel teacher taps hold a
# stable normalised-RankMe floor ~0.028–0.030 (raw effective rank ~7.3–7.6 / 256),
# MEASURED over a 16-checkpoint guard-OFF nano trajectory (job 20260610_090818 —
# P1 joint, 9-subj BT-lite, bs=4, 400 steps, val every 25) during which val_loss
# fell ~5× (1.573 → 0.31) and the rank held flat / rose SLIGHTLY (0.0278 → 0.0284
# M2; 0.0297 → 0.0299 M4) — the canonical no-collapse signature (collapse would
# crater toward rank 1, normalised ~0.004, with the loss stalling). The B36 bands
# above (M2 0.5/0.25, M4 0.04/0.02) are anchored to the PRE-B37 floors (M2 ~0.31,
# M4 ~0.05) and false-fire from birth on B37 (the M2 alarm tripped at EVERY one of
# the 16 checkpoints — it would insta-kill a guard-ON B37 run at step 24). Anchored
# to the measured B37 floor and unified across M2/M4 (same freq-preserving 5-D
# latent family, same floor): warn 0.020 (~0.7× floor; advisory / log-only — sits
# just under the healthy floor) and alarm 0.010 (~0.36× floor; ~2.6× above a
# genuine rank-1 collapse at 1/256 = 0.0039) hard-kills real collapse with margin
# while never firing on the healthy floor. NANO-DERIVED — confirm the floor holds
# on the full-scale capstone (larger batch / full cohort, ≫400 steps) before
# trusting the alarm to gate a long run; the ~7/256 absolute rank is itself a
# yellow flag (the thin latent may over-compress) worth a longer-run check.
RANKME_JOINT_NORMALISED_WARN: float = 0.020
RANKME_JOINT_NORMALISED_ALARM: float = 0.010

RANKME_EPS: float = 1e-7


@dataclass(frozen=True)
class TeacherRankVerdict:
    """Instantaneous MON-TEACHER-FEATURE-RANK probe verdict.

    Fields
    ------
    rankme:
        Effective rank of the teacher feature matrix; range
        ``[1, min(N, d)]``.
    rankme_normalised:
        ``rankme / d_feature``. ``1.0`` = no collapse; ``< 0.25`` = hard
        alarm.
    n_samples:
        ``N`` — number of feature vectors used in the SVD (e.g. clips
        × parcel slots × time bins, after masking).
    d_feature:
        ``d`` — feature dimensionality (student d_model, default 256).
    is_warn:
        ``True`` iff ``rankme_normalised < RANKME_NORMALISED_WARN``.
    is_alarm:
        ``True`` iff ``rankme_normalised < RANKME_NORMALISED_ALARM``.
    """

    rankme: float
    rankme_normalised: float
    n_samples: int
    d_feature: int
    is_warn: bool
    is_alarm: bool


def _rankme_from_singular_values(
    singular_values: Tensor, *, eps: float = RANKME_EPS,
) -> float:
    """RankMe from a 1-D tensor of non-negative singular values.

    Implements Eq. (1) of Garrido et al. 2023::

        p_k = (σ_k + ε) / Σ(σ_j + ε)
        RankMe = exp(-Σ p_k log p_k)

    The ``ε`` prevents ``log 0`` on zero singular values.
    """
    if singular_values.dim() != 1:
        raise ValueError(
            f"singular_values must be 1-D; got shape "
            f"{tuple(singular_values.shape)}"
        )
    sv = singular_values.to(torch.float64).clamp_min(0.0) + eps
    p = sv / sv.sum()
    entropy = -(p * p.log()).sum()
    return float(entropy.exp().item())


def teacher_rank_monitor(
    teacher_features: Tensor,
    *,
    valid_mask: Tensor | None = None,
    warn_threshold: float = RANKME_NORMALISED_WARN,
    alarm_threshold: float = RANKME_NORMALISED_ALARM,
) -> TeacherRankVerdict:
    """Compute the instantaneous MON-TEACHER-FEATURE-RANK verdict.

    Parameters
    ----------
    teacher_features:
        Teacher feature tensor. Supported shapes::

          (N, d)              — already flattened
          (B, L, d)           — slot-bank features
          (B, L, T, d)        — full M4 tap

        For ``(B, L, T, d)``, the (B, L, T) axes are flattened to N.
        For ``(B, L, d)``, the (B, L) axes are flattened. Returning a
        ``(B, L, d)`` tap with ``valid_mask`` of shape ``(B, L)``
        keeps only the True rows so a degenerate SWEC batch can't
        spoof a low rank.
    valid_mask:
        Optional bool tensor aligning to the *non-d* axes of
        ``teacher_features``: shape ``(B, L)`` for ``(B, L, d)``,
        shape ``(B, L, T)`` for ``(B, L, T, d)``. ``True`` rows are
        kept. ``None`` keeps everything.
    warn_threshold, alarm_threshold:
        ``rankme_normalised`` thresholds (warn > alarm).

    Returns
    -------
    TeacherRankVerdict — see field docs above. Returns a degenerate
    verdict (``rankme = 1.0``, alarms ``False``) if fewer than 2
    samples survive the mask.
    """
    if not 0.0 < alarm_threshold < warn_threshold <= 1.0:
        raise ValueError(
            f"need 0 < alarm < warn <= 1; got alarm={alarm_threshold}, "
            f"warn={warn_threshold}"
        )

    if teacher_features.dim() == 2:
        Z = teacher_features
    elif teacher_features.dim() == 3:
        if valid_mask is not None:
            if valid_mask.shape != teacher_features.shape[:2]:
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} must match "
                    f"teacher_features (B, L) = "
                    f"{tuple(teacher_features.shape[:2])}"
                )
            Z = teacher_features[valid_mask]
        else:
            B, L, d = teacher_features.shape
            Z = teacher_features.reshape(B * L, d)
    elif teacher_features.dim() == 4:
        B, L, T, d = teacher_features.shape
        if valid_mask is not None:
            if valid_mask.shape == (B, L):
                # Expand per-clip slot mask across time.
                expanded = valid_mask.unsqueeze(-1).expand(B, L, T)
            elif valid_mask.shape == (B, L, T):
                expanded = valid_mask
            else:
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} not "
                    f"compatible with teacher_features (B, L, T) = "
                    f"({B}, {L}, {T})"
                )
            flat = teacher_features.reshape(B * L * T, d)
            mask_flat = expanded.reshape(B * L * T)
            Z = flat[mask_flat]
        else:
            Z = teacher_features.reshape(B * L * T, d)
    else:
        raise ValueError(
            f"teacher_features must be 2-/3-/4-D; got shape "
            f"{tuple(teacher_features.shape)}"
        )

    N, d = int(Z.shape[0]), int(Z.shape[-1])
    if N < 2:
        # Not enough samples for a meaningful SVD — return a healthy
        # placeholder so the monitor never trips on an empty mask.
        return TeacherRankVerdict(
            rankme=1.0, rankme_normalised=1.0 / max(d, 1),
            n_samples=N, d_feature=d,
            is_warn=False, is_alarm=False,
        )

    # SVD on float32 for stability; SVD on bf16 occasionally returns
    # NaN singular values on H100.
    Z32 = Z.to(torch.float32)
    sv = torch.linalg.svdvals(Z32)
    rankme = _rankme_from_singular_values(sv)
    rankme_normalised = rankme / d

    return TeacherRankVerdict(
        rankme=rankme,
        rankme_normalised=rankme_normalised,
        n_samples=N,
        d_feature=d,
        is_warn=rankme_normalised < warn_threshold,
        is_alarm=rankme_normalised < alarm_threshold,
    )
