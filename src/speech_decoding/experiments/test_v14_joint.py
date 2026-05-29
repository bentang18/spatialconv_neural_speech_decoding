"""Tests for B29 Item 1 joint SSL experiment (``v14_joint``).

Covers:
  * The 4-term coefficient tuple matches the B19/B26/B27/B28 + B29
    cascade lock (all 1.0; DKoleo @ M4 NOT in default).
  * The L1 loss form matches V-JEPA-2 §2.1 Eq 1 (B26 correction over
    B25's Smooth-L1 citation; B27 keeps L1).
  * The composer sums to the same scalar as direct ``W * L`` adds.
  * The composer's :class:`V14TotalLossBreakdown` carries DKoleo / Gram /
    context loss as ``None`` (B27 dropped context loss; B28 demoted
    DKoleo).
  * No context-loss term reachable from the joint path.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.experiments.v14_joint import (
    JOINT_PHASE,
    V14JointExperiment,
    compose_v14_joint_loss,
    v14_joint_l1_loss,
    v14_joint_loss_coefficients,
)
from speech_decoding.ssl.total_loss import (
    W_MID_SLOT,
    W_POST_FRAME,
    W_POST_UTTERANCE,
    W_PRE_FRAME,
    V14TotalLossBreakdown,
)


def test_v14_joint_phase_literal_is_joint_b29() -> None:
    """B29 Item 1: the joint phase is identified by the ``joint_b29`` tag."""
    assert JOINT_PHASE == "joint_b29"


def test_v14_joint_loss_coefficients_are_all_one() -> None:
    """B19/B26/B27/B28 lock: all four W's are 1.0 in the joint default."""
    coeffs = v14_joint_loss_coefficients()
    assert coeffs == (1.0, 1.0, 1.0, 1.0)
    assert coeffs == (W_PRE_FRAME, W_MID_SLOT, W_POST_FRAME, W_POST_UTTERANCE)


def test_v14_joint_loss_coefficients_omit_dkoleo() -> None:
    """B28 Item 1: DKoleo @ M4 demoted to sister-only → NOT in joint default."""
    coeffs = v14_joint_loss_coefficients()
    assert len(coeffs) == 4  # not 5


def test_v14_joint_l1_loss_matches_torch_l1_loss() -> None:
    """L1 loss form per V-JEPA-2 §2.1 Eq 1 (B26 correction)."""
    pred = torch.tensor([0.0, 1.0, 2.0])
    target = torch.tensor([0.5, 0.5, 1.5])
    out = v14_joint_l1_loss(pred, target, reduction="mean")
    expected = torch.nn.functional.l1_loss(pred, target, reduction="mean")
    torch.testing.assert_close(out, expected)
    # = mean(|0-0.5|, |1-0.5|, |2-1.5|) = mean(0.5, 0.5, 0.5) = 0.5
    assert float(out.item()) == pytest.approx(0.5)


def test_v14_joint_l1_loss_supports_per_element_reduction() -> None:
    pred = torch.tensor([0.0, 1.0])
    target = torch.tensor([0.5, 0.5])
    out = v14_joint_l1_loss(pred, target, reduction="none")
    torch.testing.assert_close(out, torch.tensor([0.5, 0.5]))


def test_v14_joint_l1_loss_rejects_unknown_reduction() -> None:
    with pytest.raises(ValueError, match="reduction"):
        v14_joint_l1_loss(
            torch.zeros(2), torch.zeros(2), reduction="bogus",
        )


def test_compose_v14_joint_loss_total_equals_sum_of_terms() -> None:
    l_pre = torch.tensor(0.1)
    l_mid = torch.tensor(0.2)
    l_post_f = torch.tensor(0.3)
    l_post_u = torch.tensor(0.4)
    total, breakdown = compose_v14_joint_loss(
        l_pre_frame=l_pre, l_mid_slot=l_mid,
        l_post_frame=l_post_f, l_post_utterance=l_post_u,
    )
    expected = 1.0 * 0.1 + 1.0 * 0.2 + 1.0 * 0.3 + 1.0 * 0.4
    assert float(total.item()) == pytest.approx(expected)
    assert isinstance(breakdown, V14TotalLossBreakdown)
    torch.testing.assert_close(breakdown.total, total)


def test_compose_v14_joint_loss_breakdown_carries_terms_and_no_dkoleo() -> None:
    """Joint composer returns ``None`` for DKoleo / Gram / context loss
    fields — these are sister-only after B28 / dropped after B27."""
    l_pre = torch.tensor(0.1)
    l_mid = torch.tensor(0.2)
    l_post_f = torch.tensor(0.3)
    l_post_u = torch.tensor(0.4)
    _, breakdown = compose_v14_joint_loss(
        l_pre_frame=l_pre, l_mid_slot=l_mid,
        l_post_frame=l_post_f, l_post_utterance=l_post_u,
    )
    assert breakdown.l_dkoleo_m4 is None
    assert breakdown.l_dkoleo_m3_reactive is None
    assert breakdown.l_gram_reactive is None


def test_compose_v14_joint_loss_breakdown_carries_coefficients() -> None:
    """``V14TotalLossBreakdown.coefficients`` mirrors the joint defaults
    plus ``None`` for the DKoleo slot."""
    _, breakdown = compose_v14_joint_loss(
        l_pre_frame=torch.tensor(0.0), l_mid_slot=torch.tensor(0.0),
        l_post_frame=torch.tensor(0.0), l_post_utterance=torch.tensor(0.0),
    )
    assert breakdown.coefficients == (1.0, 1.0, 1.0, 1.0, None)


def test_compose_v14_joint_loss_no_context_loss_term() -> None:
    """B27 lock 2026-05-27 PM-late: context loss is DROPPED from the
    joint default. The composer's surface must not expose it."""
    import inspect

    sig = inspect.signature(compose_v14_joint_loss)
    assert "l_pre_frame_context" not in sig.parameters
    assert "lambda_ctx" not in sig.parameters
    # Only the 4 canonical terms are accepted.
    assert set(sig.parameters) == {
        "l_pre_frame", "l_mid_slot", "l_post_frame", "l_post_utterance",
    }


def test_v14_joint_experiment_rejects_every_phase_except_canonical_joint() -> None:
    """B29 Item 1 + CR-3: ``V14JointExperiment`` is pinned to
    ``phase == JOINT_PHASE_VALUE`` (1, the canonical collapsed P1 ∪ P2).
    Every other phase value — including the pre-B29 split phase=2 — must
    raise.
    """
    from speech_decoding.experiments.v14_joint import JOINT_PHASE_VALUE

    for bad_phase in ("3a", "3b", 2, 4):
        try:
            V14JointExperiment.model_validate({"phase": bad_phase})
        except Exception as exc:  # noqa: BLE001 — pydantic or our ValueError
            assert "joint" in str(exc).lower() or "phase" in str(exc).lower()
        else:
            pytest.fail(
                f"V14JointExperiment must reject phase={bad_phase!r}; "
                "joint surface is pinned to "
                f"phase={JOINT_PHASE_VALUE} (R-keep-phase-split is the "
                "V14Experiment sister)."
            )


def test_v14_joint_experiment_accepts_b30_sister_flag_defaults() -> None:
    """B30-dispatch-sister-flags: default values match the B30 lock and
    construct without raising."""
    from speech_decoding.experiments.v14_joint import JOINT_PHASE_VALUE

    # Minimal payload: pydantic on V14JointExperiment / V14Experiment /
    # Experiment requires several other fields. Use ``model_construct``
    # to bypass nested validation and just test the sister-flag
    # post-init guard, which is what the row tracks.
    xp = V14JointExperiment.model_construct(
        phase=JOINT_PHASE_VALUE,
        latent_valid_override="support",
        sa_mask_mode="bidirectional",
    )
    xp.model_post_init(None)
    assert xp.latent_valid_override == "support"
    assert xp.sa_mask_mode == "bidirectional"


def test_v14_joint_experiment_default_loss_variant_is_b31_default() -> None:
    """B31 lock 2026-05-28 PM-late: ``V14JointExperiment.loss_variant``
    defaults to ``"b31_default"`` (the V-JEPA-2-canonical 2-term joint
    SSL surface). The three sister variants are accepted by the field
    Literal without raising."""
    from speech_decoding.experiments.v14_joint import (
        JOINT_PHASE_VALUE,
        LossVariant,
    )
    import typing as tp

    xp = V14JointExperiment.model_construct(
        phase=JOINT_PHASE_VALUE,
        latent_valid_override="support",
        sa_mask_mode="bidirectional",
    )
    xp.model_post_init(None)
    assert xp.loss_variant == "b31_default"
    # Schema check: the Literal exposes all 4 B31 arms.
    assert set(tp.get_args(LossVariant)) == {
        "b31_default", "b31_plus_m3", "b31_plus_utt", "b31_plus_both",
    }


@pytest.mark.parametrize(
    "variant",
    ["b31_default", "b31_plus_m3", "b31_plus_utt", "b31_plus_both"],
)
def test_v14_joint_experiment_propagates_loss_variant(variant: str) -> None:
    """The 4 B31 ``loss_variant`` values are all accepted and persisted
    onto the V14JointExperiment instance — the value is what
    ``_build_brain_module`` threads onto :class:`V14JointBrainModule`."""
    from speech_decoding.experiments.v14_joint import JOINT_PHASE_VALUE

    xp = V14JointExperiment.model_construct(
        phase=JOINT_PHASE_VALUE,
        latent_valid_override="support",
        sa_mask_mode="bidirectional",
        loss_variant=variant,
    )
    xp.model_post_init(None)
    assert xp.loss_variant == variant


def test_v14_joint_experiment_rejects_b30_sister_latent_valid_override() -> None:
    """B30 sister falsifier ``R-item-12-all-true`` raises
    :class:`NotImplementedError` until B2.2 lands the aggregator-call
    branch that actually consumes ``latent_valid_override != "support"``.
    Drift-table row ``B30-dispatch-sister-flags``."""
    from speech_decoding.experiments.v14_joint import JOINT_PHASE_VALUE

    # ``model_construct`` triggers ``model_post_init`` in pydantic v2 so
    # the NotImplementedError lands at construction time, not on a later
    # explicit ``model_post_init(None)``.
    with pytest.raises(NotImplementedError, match="B2.2"):
        V14JointExperiment.model_construct(
            phase=JOINT_PHASE_VALUE,
            latent_valid_override="all_true",
            sa_mask_mode="bidirectional",
        )


def test_v14_joint_experiment_rejects_b30_sister_sa_mask_mode() -> None:
    """B30 sister falsifier ``R-sa-key-only`` raises
    :class:`NotImplementedError` until the encoder latent-SA key-only
    branch lands. Drift-table row ``B30-dispatch-sister-flags``."""
    from speech_decoding.experiments.v14_joint import JOINT_PHASE_VALUE

    with pytest.raises(NotImplementedError, match="key-only"):
        V14JointExperiment.model_construct(
            phase=JOINT_PHASE_VALUE,
            latent_valid_override="support",
            sa_mask_mode="key_only",
        )


def test_v14_joint_loss_form_l1_not_smooth_l1() -> None:
    """B26 correction (5/27 PM) over B25 (5/27 AM): the per-term loss
    form is *pure L1*, not Smooth-L1. The encoded loss must not collapse
    to MSE for any nonzero residual, and must agree with ``l1_loss``."""
    pred = torch.tensor([0.0, 3.0])
    target = torch.tensor([1.0, 0.0])
    out_l1 = v14_joint_l1_loss(pred, target, reduction="sum")
    expected_l1 = torch.nn.functional.l1_loss(pred, target, reduction="sum")
    out_smooth_l1 = torch.nn.functional.smooth_l1_loss(
        pred, target, reduction="sum", beta=1.0,
    )
    out_mse = torch.nn.functional.mse_loss(pred, target, reduction="sum")
    torch.testing.assert_close(out_l1, expected_l1)
    # For |residual|=3, Smooth-L1 deviates from L1 (3 vs 2.5).
    assert float(out_l1.item()) != pytest.approx(float(out_smooth_l1.item()))
    assert float(out_l1.item()) != pytest.approx(float(out_mse.item()))
