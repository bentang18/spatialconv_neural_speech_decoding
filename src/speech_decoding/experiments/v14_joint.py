"""B29 Item 1: single joint SSL phase (replaces split P1 + P2).

Lock memo: ``memory/project_v14_b29_joint_default_2026_05_27.md`` §Item 1.

Per the 5/27 PM-late lock, v14's pretraining collapses to a single SSL
phase running the canonical 4-term L1 objective with the anatomy bias
gated per-clip by ``λ_anat`` (B29 Item 12). The old split P1/P2 path is
preserved as the ``R-keep-phase-split`` sister via ``V14Experiment``;
this module is the *joint-by-default* surface.

Lineage of the loss form:

  * B19 (5/24): 5-term lock at ``W_PRE_FRAME / W_MID_SLOT / W_POST_FRAME /
    W_POST_UTTERANCE / W_DKOLEO_M4 = 1, 1, 1, 1, 0.1``.
  * B25 (5/27 AM): Smooth-L1 β=1.0 across SSL prediction terms.
  * B26 (5/27 PM): pure L1 (V-JEPA-2 §2.1 Eq 1 canonical) — Smooth-L1
    citation was wrong. EMA τ=0.999 fixed. Teacher full-input contract.
  * B27 (5/27 PM-late): DROP context loss + λ_ctx warmup; keep B26's
    pure L1 + fixed EMA + full-input teacher.
  * B28 (5/27 PM-late): DKoleo @ M4 demoted from default → 3 sisters;
    cross-attn count 2 @ {0, 3} → 1 @ {0}; anatomy-bias linear warmup
    over last 25% of P1 ∪ first 25% of P2.
  * B29 Item 1 (5/27 PM-late): P1 + P2 collapse → single joint phase.
  * B29 Item 12 (5/27 PM-late): replace step-time anatomy-bias warmup
    with per-clip ``λ_anat`` gate from metadata.

So the joint default is::

    L_total = W_PRE_FRAME      · L_pre_frame_masked   @ M2
            + W_MID_SLOT       · L_mid_slot           @ LN_mid(M3)
            + W_POST_FRAME     · L_post_frame         @ LN_frame(M4)
            + W_POST_UTTERANCE · L_post_utterance     @ LN_utt(M4)-PMA

with all four W's = 1.0 and the per-term loss form = **pure L1**. DKoleo
@ M4 is OFF by default (3-sister escalation path lives in dispatch's
``--dkoleo-mode``); context loss is dropped (B27); EMA τ fixed at 0.999.
"""

from __future__ import annotations

from typing import Literal

import pydantic
import torch
from torch import Tensor

from speech_decoding.experiments.v14_experiment import V14Experiment
from speech_decoding.ssl.total_loss import (
    W_MID_SLOT,
    W_POST_FRAME,
    W_POST_UTTERANCE,
    W_PRE_FRAME,
    V14TotalLossBreakdown,
    v14_total_loss,
)


JOINT_PHASE: Literal["joint_b29"] = "joint_b29"
"""Dispatch-side tag for the joint mode (B29 Item 1).

This is the ``phase_mode`` value the CLI emits, NOT the ``phase`` field
on :class:`V14Experiment`. The joint experiment runs at the parent
class's canonical first phase (``phase == 1``) and the dispatch wraps
that single phase under the ``JOINT_PHASE`` label so the trainer's
checkpoint cadence and logging match the P1 ∪ P2 collapse contract.
"""

JOINT_PHASE_VALUE: Literal[1] = 1
"""Single-value ``phase`` that :class:`V14JointExperiment` is pinned to.

B29 Item 1 collapses split P1+P2 into one phase; the parent's pydantic
``phase`` field stays integer-typed (so the broader Experiment hierarchy
doesn't need a schema change), but the joint subclass refuses every
value except this one.
"""


def v14_joint_loss_coefficients() -> tuple[float, float, float, float]:
    """B29 Item 1 + B27 + B28 lock: 4-term joint default.

    Returns the canonical ``(W_PRE_FRAME, W_MID_SLOT, W_POST_FRAME,
    W_POST_UTTERANCE)`` tuple; all four equal ``1.0``. DKoleo @ M4 is
    dispatched via the sister-set machinery and is NOT in the default
    tuple (B28 Item 1 demoted it).
    """
    return (W_PRE_FRAME, W_MID_SLOT, W_POST_FRAME, W_POST_UTTERANCE)


def v14_joint_l1_loss(
    pred: Tensor, target: Tensor, *, reduction: str = "mean",
) -> Tensor:
    """Per-term L1 loss matching V-JEPA-2 §2.1 Eq 1.

    B26 (5/27 PM) corrected the Smooth-L1 citation: V-JEPA 2 / 2.1 /
    data2vec-2.0 §3.1 are pure L1, not Smooth-L1. B27 (same day, late)
    kept B26's loss form decision.

    Reductions: ``"mean"`` (default, term-level averaging) or ``"none"``
    (per-element). ``"sum"`` is also accepted to match ``F.l1_loss``.
    """
    if reduction not in ("mean", "none", "sum"):
        raise ValueError(
            f"reduction must be one of 'mean', 'none', 'sum'; got {reduction!r}"
        )
    return torch.nn.functional.l1_loss(pred, target, reduction=reduction)


def compose_v14_joint_loss(
    *,
    l_pre_frame: Tensor,
    l_mid_slot: Tensor,
    l_post_frame: Tensor,
    l_post_utterance: Tensor,
) -> tuple[Tensor, V14TotalLossBreakdown]:
    """Sum the 4-term joint objective with B19 unit weights.

    Thin wrapper around :func:`speech_decoding.ssl.total_loss.v14_total_loss`
    that preserves the legacy ``(total, breakdown)`` tuple shape. The
    breakdown is constructed in ONE place (``v14_total_loss``) so the
    coefficient bindings and the ``V14TotalLossBreakdown`` field
    population can't drift between two parallel composers (Round 12
    audit finding, 2026-05-28).

    DKoleo / Gram / context loss are OMITTED here — they're sister-only
    arms gated by the dispatch (DKoleo) or removed entirely (B27 context
    loss). Sister cells should call ``v14_total_loss`` directly with the
    extra terms via its ``l_dkoleo_m4`` / ``l_dkoleo_m3_reactive`` /
    ``l_gram_reactive`` kwargs.
    """
    breakdown = v14_total_loss(
        l_pre_frame=l_pre_frame,
        l_mid_slot=l_mid_slot,
        l_post_frame=l_post_frame,
        l_post_utterance=l_post_utterance,
    )
    return breakdown.total, breakdown


class V14JointExperiment(V14Experiment):
    """Joint-by-default v14 SSL experiment (B29 Item 1).

    Pinned to a single ``phase`` value (``"joint_b29"``-shaped). The
    sister ``R-keep-phase-split`` uses the parent :class:`V14Experiment`
    with explicit ``phase=1`` then ``phase=2``.

    The per-step loss composition runs through
    :func:`compose_v14_joint_loss`; the trainer never hard-codes the
    weights. ``λ_anat`` flows in from per-clip metadata via the
    :class:`speech_decoding.extractors.subtype_meta.LambdaAnatExtractor`
    and reaches the encoder forward as a ``(B,)`` tensor (see B29 Item
    12 in ``models/v14_encoder.py``).
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # NOTE: ``phase`` inherits from ``V14Experiment``; restricting to a
    # single-value ``Literal`` here would require redeclaring the field
    # on the subclass. Use a runtime check in ``model_post_init`` so the
    # field type stays consistent across the hierarchy.

    def model_post_init(self, _ctx) -> None:  # type: ignore[override]
        super().model_post_init(_ctx)
        # B29 Item 1: ``V14JointExperiment`` is *only* the joint phase,
        # which is pinned to ``phase == 1`` (the canonical collapsed
        # P1 ∪ P2). Split P1 / P2 must instantiate ``V14Experiment``
        # directly via the ``R-keep-phase-split`` sister.
        if self.phase != JOINT_PHASE_VALUE:
            raise ValueError(
                f"V14JointExperiment models the B29 joint phase only "
                f"(phase={JOINT_PHASE_VALUE}); got phase={self.phase!r}. "
                "Use V14Experiment directly for split P1/P2 "
                "(R-keep-phase-split sister)."
            )

    def loss_coefficients(self) -> tuple[float, ...]:
        """Override: joint phase uses the B29 4-term tuple."""
        return v14_joint_loss_coefficients()

    def _train_and_test(self) -> dict[str, float | None]:
        # B2.1 (#96) wires construction through the dispatch phase-switch,
        # but the SSL training-step itself is gated on the rest of Bucket 2:
        # B2.2 (compose 4-term L_total via ssl/aggregator.py with the B30
        # ``latent_valid`` single-source-of-truth mask), B2.3 (shaft-mask +
        # ref_aug + ref_embed extractors into the forward), B2.4 (B02
        # WRS-over-valid-bin-electrode-hours sampler + StatefulDataLoader),
        # B2.5 (MON-SLOT-REDUNDANCY / MON-MASK-002/004 monitors +
        # best-val probe callback). Until those land, falling through to
        # the parent CE-supervised path would silently mis-train the
        # joint phase as Phase-4. Raise here so the gating is loud.
        raise NotImplementedError(
            "V14JointExperiment._train_and_test is gated on Bucket-2 SSL "
            "wiring: B2.2 (4-term L_total via ssl/aggregator.py with B30 "
            "latent_valid), B2.3 (shaft-mask + ref_aug + ref_embed "
            "extractors into the forward), B2.4 (B02 sampler + "
            "StatefulDataLoader), B2.5 (monitors + best-val probe "
            "callback). See docs/neuroprobe/v14_blockers.md."
        )


__all__ = [
    "JOINT_PHASE",
    "JOINT_PHASE_VALUE",
    "V14JointExperiment",
    "compose_v14_joint_loss",
    "v14_joint_l1_loss",
    "v14_joint_loss_coefficients",
]
