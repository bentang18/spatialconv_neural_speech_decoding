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


LatentValidOverride = Literal["support", "all_true", "parcels_supervised"]
"""B30 sister selector for the latent-validity mask source.

``support`` (default): B30 single source of truth, ``support.sum(over
electrodes) > 0`` expanded across the M sub-slots — the same tensor the
encoder's latent-SA ``attn_mask`` consumes and the aggregator threads
through ``L_mid_slot`` / ``L_post_frame`` / ``L_post_utterance``.

Sister falsifiers (drift row ``B30-dispatch-sister-flags``):

* ``all_true`` (``R-item-12-all-true`` P0) — every slot active for every
  clip; falsifies the per-subject anatomy gating.
* ``parcels_supervised`` (``R-parcels-supervised-gating`` P0-retired-into-default)
  — pre-B30 per-subject ``parcels_supervised[subject]`` override; kept
  as falsifier so the retire-into-default move is empirically defended.

Only the ``support`` value is wired into the SSL trainer today; the
sister branches raise :class:`NotImplementedError` at construction
until the joint-phase aggregator-call path (#97 B2.2) lands.
"""

SaMaskMode = Literal["bidirectional", "key_only"]
"""B30 sister selector for the latent self-attention mask shape.

``bidirectional`` (default): inactive slots fully bypass latent SA —
neither keys nor queries. Encoder applies an ``attn_mask (L, L)`` over
the latent token sequence.

``key_only`` (``R-sa-key-only`` P1 falsifier): pre-B30 key-only
``key_padding_mask`` path; queries from inactive slots still emit
attention but receive zeroed contributions. Falsifies the move from
key-only to bidirectional masking.

Only ``bidirectional`` is wired into the encoder today; ``key_only``
raises :class:`NotImplementedError` at construction until the
encoder-side branch lands.
"""


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

    B30 sister selectors (drift row ``B30-dispatch-sister-flags``,
    surfaced 2026-05-28 by the R12 wiring audit):

    * ``latent_valid_override`` picks the source of the
      :data:`speech_decoding.ssl.aggregator.compute_v14_ssl_losses`
      ``latent_valid`` argument. Default ``"support"`` matches the B30
      lock; ``"all_true"`` / ``"parcels_supervised"`` are
      :class:`NotImplementedError`-gated sister falsifiers.
    * ``sa_mask_mode`` picks the encoder's latent-SA mask shape.
      Default ``"bidirectional"`` matches the B30 lock; ``"key_only"``
      is a :class:`NotImplementedError`-gated sister falsifier.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # B30 sister selectors. Pinned to the locked defaults; sister values
    # are accepted by the field but rejected at construction until the
    # respective runtime branch lands (B2.2 aggregator-call, encoder
    # latent-SA key-only path).
    latent_valid_override: LatentValidOverride = "support"
    sa_mask_mode: SaMaskMode = "bidirectional"

    # NOTE: ``phase`` inherits from ``V14Experiment``; restricting to a
    # single-value ``Literal`` here would require redeclaring the field
    # on the subclass. Use a runtime check in ``model_post_init`` so the
    # field type stays consistent across the hierarchy.

    def model_post_init(self, _ctx) -> None:  # type: ignore[override]
        super().model_post_init(_ctx)
        # B30 sister selectors: default values match the B30 lock and are
        # always accepted. Non-default values flag the drift-table
        # B30-dispatch-sister-flags row as runtime-gated — they parse OK
        # at config time so the run-record YAML records the choice, but
        # the trainer / encoder branch hasn't landed yet.
        if self.latent_valid_override != "support":
            raise NotImplementedError(
                f"latent_valid_override={self.latent_valid_override!r} is a "
                "B30 sister falsifier (R-item-12-all-true / "
                "R-parcels-supervised-gating). Wiring blocked on #97 B2.2 "
                "(joint SSL aggregator call) — see "
                "docs/neuroprobe/v14_blockers.md row B30-dispatch-sister-flags."
            )
        if self.sa_mask_mode != "bidirectional":
            raise NotImplementedError(
                f"sa_mask_mode={self.sa_mask_mode!r} is the B30 sister "
                "falsifier R-sa-key-only. Wiring blocked on the encoder "
                "latent-SA key-only branch — see "
                "docs/neuroprobe/v14_blockers.md row B30-dispatch-sister-flags."
            )
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

    def _build_brain_module(self, train_loader):  # type: ignore[override]
        """B2.2: build the joint SSL Lightning module.

        Replaces the parent's CE-classifier ``BrainModule`` with
        :class:`speech_decoding.experiments.v14_joint_module.V14JointBrainModule`,
        which owns the EMA teacher + 3 LN heads + PMA pair and composes the
        4-term B28/B29 ``L_total`` via the
        :func:`speech_decoding.ssl.aggregator.compute_v14_ssl_losses` path
        (drift row ``B29-joint-collapse`` runtime arm).

        The parent's ``brain_model_config.build(n_in_channels, n_outputs)``
        returns a :class:`V14ParcelPerceiverWithHead` (encoder + frozen
        parcel-PMA + flat head) — meant for Phase-4 supervised. The joint
        SSL trainer only needs the bare encoder; we extract ``.encoder``
        and let the BrainModule construct its own student-side PMA + LN
        heads + EMA teacher mirror.

        ``n_outputs`` is informational here (SSL has no classification
        head); we pass ``1`` so the build call succeeds, then ignore the
        head.
        """
        from speech_decoding.experiments.v14_joint_module import V14JointBrainModule
        from speech_decoding.models.v14_encoder import V14ParcelPerceiverModel

        batch = next(iter(train_loader))
        input_name = self._input_tensor_name()
        x = batch.data[input_name]
        head_model = self.brain_model_config.build(
            n_in_channels=int(x.shape[1]),
            n_outputs=1,
        )
        encoder = getattr(head_model, "encoder", None)
        if not isinstance(encoder, V14ParcelPerceiverModel):
            raise RuntimeError(
                "V14JointExperiment expected brain_model_config.build to "
                "return a model with an ``.encoder`` attribute of type "
                "V14ParcelPerceiverModel (per V14ParcelPerceiverWithHead); "
                f"got {type(head_model).__name__} without a recognized "
                "encoder slot."
            )
        return V14JointBrainModule(
            encoder=encoder,
            optim_config=self.optim,
            pma_n_heads=encoder.d_model // max(1, encoder.d_model // 8),
            ema_tau=0.999,
            loss_form="l1",
            latent_valid_override=self.latent_valid_override,
            sa_mask_mode=self.sa_mask_mode,
        )


__all__ = [
    "JOINT_PHASE",
    "JOINT_PHASE_VALUE",
    "LatentValidOverride",
    "SaMaskMode",
    "V14JointExperiment",
    "compose_v14_joint_loss",
    "v14_joint_l1_loss",
    "v14_joint_loss_coefficients",
]
