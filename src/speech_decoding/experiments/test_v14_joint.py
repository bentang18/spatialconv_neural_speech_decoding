"""Tests for B29 Item 1 joint SSL experiment (``v14_joint``).

Covers:
  * The L1 loss form matches V-JEPA-2 §2.1 Eq 1 (B26 correction over
    B25's Smooth-L1 citation; B27 keeps L1).
  * ``V14JointExperiment`` is pinned to the canonical joint phase and
    defaults to the B31 2-term ``loss_variant="b31_default"`` surface;
    the B30 sister flags + B31 sister variants parse but stay gated.

The pre-B31 4-term coefficient tuple + ``compose_v14_joint_loss`` composer
were deleted (the live loss SSOT is
:func:`speech_decoding.ssl.aggregator.compute_v14_ssl_losses`); their
guard-tests were removed with them.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.experiments.v14_joint import (
    JOINT_PHASE,
    V14JointExperiment,
    v14_joint_l1_loss,
)


def test_v14_joint_phase_literal_is_joint_b29() -> None:
    """B29 Item 1: the joint phase is identified by the ``joint_b29`` tag."""
    assert JOINT_PHASE == "joint_b29"


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
    """B36 WS-B: ``loss_variant`` still defaults to ``"b31_default"`` (the
    single-term masked-JEPA surface). The field's Literal retains all 4 B31
    arms for dispatch config-record compatibility, but only the default is
    wired (the multi-term ``b31_plus_*`` sisters are quarantined — B9)."""
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
    # Schema check: the Literal still exposes all 4 B31 arms.
    assert set(tp.get_args(LossVariant)) == {
        "b31_default", "b31_plus_m3", "b31_plus_utt", "b31_plus_both",
    }


@pytest.mark.parametrize("variant", ["b31_plus_m3", "b31_plus_utt", "b31_plus_both"])
def test_v14_joint_experiment_quarantines_multiterm_loss_variants(variant: str) -> None:
    """B9 (B36 WS-B): the masked-JEPA default is single-term per phase, so
    the B31 multi-term aggregator sisters are quarantined — they raise
    :class:`NotImplementedError` at construction until re-added on the
    masked path (R-add-m3-loss / R-add-utterance-loss)."""
    from speech_decoding.experiments.v14_joint import JOINT_PHASE_VALUE

    with pytest.raises(NotImplementedError, match="single-term"):
        V14JointExperiment.model_construct(
            phase=JOINT_PHASE_VALUE,
            latent_valid_override="support",
            sa_mask_mode="bidirectional",
            loss_variant=variant,
        )


def test_v14_joint_experiment_jepa_phase_and_mask_ratio_defaults() -> None:
    """B36 WS-B: the masked-JEPA phase defaults to P1 with the 6/03
    masking-lock defaults (M2 ``bands`` held-out 0.50 / M4 ``tube`` 0.20 of
    covered parcels, ``cross_time`` predictor); WS-E threads the staged
    P1→P2 handoff."""
    from speech_decoding.experiments.v14_joint import JOINT_PHASE_VALUE

    xp = V14JointExperiment.model_construct(
        phase=JOINT_PHASE_VALUE,
        latent_valid_override="support",
        sa_mask_mode="bidirectional",
    )
    xp.model_post_init(None)
    assert xp.jepa_phase == "p1"
    assert xp.m2_mask_type == "bands"
    assert xp.m2_mask_ratio == pytest.approx(0.50)
    assert xp.m4_mask_type == "tube"
    assert xp.m4_mask_ratio == pytest.approx(0.20)
    assert xp.predictor_scope == "cross_time"
    assert xp.mask_seed == 0


def test_v14_joint_experiment_rejects_h1_leak_coupling_at_config_time() -> None:
    """B36 coupling guard fires at the CONFIG layer (``model_post_init``), not
    only in the module: the H1-leak pairing (``time_block`` mask +
    ``cross_time`` predictor — masked-at-t target visible at t±1) must raise
    before dispatch. A regression dropping the ``validate_m4_coupling`` call
    from ``model_post_init`` is caught here. ``model_construct`` triggers
    ``model_post_init`` in pydantic v2 so the raise lands at construction."""
    from speech_decoding.experiments.v14_joint import JOINT_PHASE_VALUE

    with pytest.raises(ValueError, match="H1 leak"):
        V14JointExperiment.model_construct(
            phase=JOINT_PHASE_VALUE,
            latent_valid_override="support",
            sa_mask_mode="bidirectional",
            m4_mask_type="time_block",
            predictor_scope="cross_time",
        )


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
