"""LOSS-01 (B19 + B21 + B22 + B28 + B31 loss-form lock):
``L_total`` composer with locked coefficients.

**B31 amendment (2026-05-28 PM, supersedes B28's 4-term default):**
The V-JEPA-2-canonical 2-term lock drops ``L_mid_slot`` and
``L_post_utterance`` from the joint SSL default. The PMA query stops
receiving gradient in the joint SSL phase; it gets its first gradient
at P3 (Whisper-L8 distillation) and is frozen at P4. The three dropped
mechanisms remain available as falsifier sisters (``R-add-m3-loss``,
``R-add-utterance-loss``, ``R-add-both``) and are reconstructed only
when ``loss_variant`` is non-default upstream.

Joint default (B31 2-term)::

    L_total = 1.0 · L_pre_frame  @ M2
            + 1.0 · L_post_frame @ LN_frame(M4)

Joint from step 1; no curriculum, no schedule. Both terms are pure L1
(V-JEPA 2 §2.1 Eq 1 canonical) — Ben's authorship locks this combination,
do not sweep without explicit approval.

Sister-only terms (B31 falsifier arms; passed by the SSL aggregator only
when the corresponding ``loss_variant`` is selected)::

    + 1.0 · L_mid_slot       @ LN_mid(M3)         if loss_variant in {b31_plus_m3, b31_plus_both}
    + 1.0 · L_post_utterance @ LN_utt(M4) - PMA   if loss_variant in {b31_plus_utt, b31_plus_both}

Opt-in DKoleo @ M4 (B28 sister ``R-dkoleo-intra-clip-slots`` retains the
B21-original per-clip-320-slot unit; ``R-dkoleo-batch-cls-unit`` swaps
to the DINOv2-faithful per-batch CLS-analog unit upstream of the
composer; only the scalar value is passed here)::

    + 0.1 · L_DKoleo @ M4       if l_dkoleo_m4 is not None

Reactive arms (off-default; armed by collapse monitors)::

    + 0.05 · L_DKoleo @ M3      if MON-MID-DKOLEO fires
    + 0.1  · L_Gram             if MON-M4-GRAM fires

The training loop computes each L term externally (via the primitives in
``ssl/{slot_loss,utterance_loss,dkoleo,recon}.py``) and passes them to
``v14_total_loss`` to apply the fixed coefficients. This keeps the
composer a pure function — easy to unit-test, easy to grep for
coefficient drift in code review.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


# Locked coefficients (B31 2-term default + 2 falsifier-sister terms +
# 1 opt-in DKoleo arm). W_MID_SLOT and W_POST_UTTERANCE survive the B31
# default-collapse because the sister arms reuse the historical 1.0
# locked weight when armed.
W_PRE_FRAME: float = 1.0
W_MID_SLOT: float = 1.0
W_POST_FRAME: float = 1.0
W_POST_UTTERANCE: float = 1.0
# DKoleo @ M4 (B28 demoted to sister-only). Constant is retained at the
# B21 locked weight so the ``R-dkoleo-{intra_clip_slots,batch_cls}``
# sisters share the historical coefficient.
W_DKOLEO_M4: float = 0.1
# Reactive cousins (off-default; only armed by monitor triggers).
W_DKOLEO_M3_REACTIVE: float = 0.05
W_GRAM_REACTIVE: float = 0.1


@dataclass(frozen=True)
class V14TotalLossBreakdown:
    """Per-term breakdown returned by :func:`v14_total_loss` for logging.

    ``total`` is the scalar loss the optimizer should step on. The other
    fields preserve the raw (un-weighted) per-term losses + the locked
    coefficient used, so dashboards can plot both.

    Under the **B31 2-term default**: ``l_mid_slot`` and
    ``l_post_utterance`` are ``None`` (the falsifier-sister arms are
    inactive). Under a sister variant (``b31_plus_m3``,
    ``b31_plus_utt``, ``b31_plus_both``), the corresponding fields are
    populated. ``l_dkoleo_m4`` is ``None`` unless a DKoleo sister passes
    a tensor through (B28 sister-only).
    """

    total: Tensor
    l_pre_frame: Tensor
    l_post_frame: Tensor
    l_mid_slot: Tensor | None
    l_post_utterance: Tensor | None
    l_dkoleo_m4: Tensor | None
    l_dkoleo_m3_reactive: Tensor | None
    l_gram_reactive: Tensor | None
    # 2 always-on coefficients (B31 default) + 2 falsifier-sister
    # coefficients (None when that sister is inactive) + 1 DKoleo
    # coefficient. Order matches the 5 raw fields above.
    coefficients: tuple[
        float, float, float | None, float | None, float | None,
    ] = (
        W_PRE_FRAME, W_POST_FRAME, None, None, None,
    )


def v14_total_loss(
    l_pre_frame: Tensor,
    l_post_frame: Tensor,
    *,
    l_mid_slot: Tensor | None = None,
    l_post_utterance: Tensor | None = None,
    l_dkoleo_m4: Tensor | None = None,
    l_dkoleo_m3_reactive: Tensor | None = None,
    l_gram_reactive: Tensor | None = None,
) -> V14TotalLossBreakdown:
    """Compose the v14 joint SSL loss with locked coefficients.

    **B31 default (2 terms)**: pass only ``l_pre_frame`` + ``l_post_frame``.
    The composer returns ``L_total = L_pre_frame + L_post_frame`` with
    both at locked weight 1.0 (pure L1 V-JEPA-2-canonical).

    **B31 falsifier sisters**: pass ``l_mid_slot`` (for ``R-add-m3-loss``
    / ``R-add-both``) and/or ``l_post_utterance`` (for
    ``R-add-utterance-loss`` / ``R-add-both``); each is weighted at the
    historical 1.0 if provided.

    **B28 DKoleo sister + reactive arms**: all optional, defaulted to
    ``None``; weighted at the locked coefficients when populated.

    Returns a :class:`V14TotalLossBreakdown` carrying ``.total`` and the
    raw per-term losses for logging. Call ``breakdown.total.backward()``
    in the training loop.
    """
    total = W_PRE_FRAME * l_pre_frame + W_POST_FRAME * l_post_frame
    if l_mid_slot is not None:
        total = total + W_MID_SLOT * l_mid_slot
    if l_post_utterance is not None:
        total = total + W_POST_UTTERANCE * l_post_utterance
    if l_dkoleo_m4 is not None:
        total = total + W_DKOLEO_M4 * l_dkoleo_m4
    if l_dkoleo_m3_reactive is not None:
        total = total + W_DKOLEO_M3_REACTIVE * l_dkoleo_m3_reactive
    if l_gram_reactive is not None:
        total = total + W_GRAM_REACTIVE * l_gram_reactive
    coefficients: tuple[
        float, float, float | None, float | None, float | None,
    ] = (
        W_PRE_FRAME,
        W_POST_FRAME,
        W_MID_SLOT if l_mid_slot is not None else None,
        W_POST_UTTERANCE if l_post_utterance is not None else None,
        W_DKOLEO_M4 if l_dkoleo_m4 is not None else None,
    )
    return V14TotalLossBreakdown(
        total=total,
        l_pre_frame=l_pre_frame,
        l_post_frame=l_post_frame,
        l_mid_slot=l_mid_slot,
        l_post_utterance=l_post_utterance,
        l_dkoleo_m4=l_dkoleo_m4,
        l_dkoleo_m3_reactive=l_dkoleo_m3_reactive,
        l_gram_reactive=l_gram_reactive,
        coefficients=coefficients,
    )
