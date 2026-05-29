"""LOSS-01 (B19 + B21 + B22 + B28 loss-form lock 2026-05-27 PM):
``L_total`` composer with locked coefficients.

**B28 amendment (2026-05-27 PM, supersedes the 5-term lock from 5/25):**
DKoleo @ M4 is demoted from the default composer to an opt-in sister
term. The unit-mismatch with DINOv2/v3 KoLeo (per-batch over CLS-analog
tokens vs v14's prior per-clip over 320 parcel slots) made the 0.1
coefficient citation unjustified; the default loss collapses to 4 terms,
and the DKoleo unit choice is selected by the ``dkoleo_mode`` dispatch
field across three sisters (``off`` / ``intra_clip_slots`` / ``batch_cls``).

P1 + P2 default 4-term objective::

    L_total = 1.0 · L_pre_frame  @ M2
            + 1.0 · L_mid_slot   @ LN_mid(M3)
            + 1.0 · L_post_frame @ LN_frame(M4)
            + 1.0 · L_post_utterance @ LN_utt(M4) - PMA

Joint from step 1; no curriculum, no schedule. Coefficients fixed at
``(1, 1, 1, 1.0)`` — Ben's authorship locks this combination, do not
sweep without explicit approval.

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


# Locked coefficients (B28 4-term default + 1 opt-in DKoleo arm).
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
    coefficient used, so dashboards can plot both. ``l_dkoleo_m4`` is
    ``None`` under the B28 4-term default; populated when a DKoleo sister
    passes a tensor through.
    """

    total: Tensor
    l_pre_frame: Tensor
    l_mid_slot: Tensor
    l_post_frame: Tensor
    l_post_utterance: Tensor
    l_dkoleo_m4: Tensor | None
    l_dkoleo_m3_reactive: Tensor | None
    l_gram_reactive: Tensor | None
    # 4 always-on coefficients + 1 optional DKoleo coefficient (``None``
    # iff DKoleo is off; constant ``W_DKOLEO_M4`` otherwise).
    coefficients: tuple[float, float, float, float, float | None] = (
        W_PRE_FRAME, W_MID_SLOT, W_POST_FRAME, W_POST_UTTERANCE, None,
    )


def v14_total_loss(
    l_pre_frame: Tensor,
    l_mid_slot: Tensor,
    l_post_frame: Tensor,
    l_post_utterance: Tensor,
    *,
    l_dkoleo_m4: Tensor | None = None,
    l_dkoleo_m3_reactive: Tensor | None = None,
    l_gram_reactive: Tensor | None = None,
) -> V14TotalLossBreakdown:
    """Compose the B28 4-term v14 P1/P2 loss with locked coefficients.

    All inputs are scalar tensors produced by the SSL loss primitives.
    ``l_dkoleo_m4`` (B28 sister-only) and the reactive terms are optional
    — pass ``None`` (the default) to disable.

    Returns a :class:`V14TotalLossBreakdown` carrying ``.total`` and the
    raw per-term losses for logging. Call ``breakdown.total.backward()``
    in the training loop.
    """
    total = (
        W_PRE_FRAME * l_pre_frame
        + W_MID_SLOT * l_mid_slot
        + W_POST_FRAME * l_post_frame
        + W_POST_UTTERANCE * l_post_utterance
    )
    if l_dkoleo_m4 is not None:
        total = total + W_DKOLEO_M4 * l_dkoleo_m4
    if l_dkoleo_m3_reactive is not None:
        total = total + W_DKOLEO_M3_REACTIVE * l_dkoleo_m3_reactive
    if l_gram_reactive is not None:
        total = total + W_GRAM_REACTIVE * l_gram_reactive
    coefficients: tuple[float, float, float, float, float | None] = (
        W_PRE_FRAME, W_MID_SLOT, W_POST_FRAME, W_POST_UTTERANCE,
        W_DKOLEO_M4 if l_dkoleo_m4 is not None else None,
    )
    return V14TotalLossBreakdown(
        total=total,
        l_pre_frame=l_pre_frame,
        l_mid_slot=l_mid_slot,
        l_post_frame=l_post_frame,
        l_post_utterance=l_post_utterance,
        l_dkoleo_m4=l_dkoleo_m4,
        l_dkoleo_m3_reactive=l_dkoleo_m3_reactive,
        l_gram_reactive=l_gram_reactive,
        coefficients=coefficients,
    )
