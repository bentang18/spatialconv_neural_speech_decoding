"""Tests for the B28/B29 SSL-side training-time monitors."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.experiments.monitors import (
    BATCH_COS_PCT95_THRESHOLD,
    DIAG_ZEROED_MEAN_THRESHOLD,
    HEAD_BALANCE_BOUNDS,
    PER_CLIP_COS_PCT95_THRESHOLD,
    PROBE_BATCH_SIZE_M1_DEFAULT,
    REF_TYPE_CANARY_F1_THRESHOLD,
    SENSOR_TYPE_CANARY_F1_THRESHOLD,
    head_balance_monitor,
    ref_type_canary_monitor,
    sensor_type_canary_monitor,
    slot_redundancy_monitor,
)


# ---------------------------------------------------------------------------
# MON-SLOT-REDUNDANCY
# ---------------------------------------------------------------------------


def test_slot_redundancy_thresholds_are_b29_m1_rescaled() -> None:
    """B29 Item 13 (5/27 PM-late) post-M=1 rescale."""
    assert PER_CLIP_COS_PCT95_THRESHOLD == 0.5
    assert BATCH_COS_PCT95_THRESHOLD == 0.7
    assert DIAG_ZEROED_MEAN_THRESHOLD == 0.35
    assert PROBE_BATCH_SIZE_M1_DEFAULT == 1024


def test_slot_redundancy_independent_slots_below_thresholds() -> None:
    """Random orthogonal-ish slots should yield low cosine similarities,
    keeping the verdict below all three thresholds."""
    torch.manual_seed(0)
    B, L, d = 8, 80, 64
    slot_bank = torch.randn(B, L, d)
    utterance_pma = torch.randn(B, d)
    verdict = slot_redundancy_monitor(slot_bank, utterance_pma)
    assert verdict.per_clip_cos_pct95 < PER_CLIP_COS_PCT95_THRESHOLD
    assert verdict.batch_cos_pct95 < BATCH_COS_PCT95_THRESHOLD
    assert verdict.diag_zeroed_mean < DIAG_ZEROED_MEAN_THRESHOLD
    assert verdict.escalations == ()


def test_slot_redundancy_collapsed_slots_escalate_intra_clip_and_vicreg() -> None:
    """All slots = same vector → pct95 cosine = 1.0 + diag-zero mean ≈ 1.0
    → both intra-clip-slots and vicreg sisters escalate."""
    B, L, d = 4, 80, 32
    template = torch.randn(d)
    slot_bank = template.expand(B, L, d).clone()
    utterance_pma = torch.randn(B, d)  # batch axis OK so only intra-clip escalates
    verdict = slot_redundancy_monitor(slot_bank, utterance_pma)
    assert verdict.per_clip_cos_pct95 > PER_CLIP_COS_PCT95_THRESHOLD
    assert verdict.diag_zeroed_mean > DIAG_ZEROED_MEAN_THRESHOLD
    assert "R-dkoleo-intra-clip-slots" in verdict.escalations
    assert "R-vicreg-slot-variance" in verdict.escalations


def test_slot_redundancy_batch_collapse_escalates_batch_cls_unit() -> None:
    """Two clips with identical utterance vectors → batch pct95 > 0.7
    → batch-cls-unit sister escalates."""
    B, L, d = 8, 80, 32
    # Per-clip slot bank stays diverse, but utterance_pma collapses.
    slot_bank = torch.randn(B, L, d)
    template = torch.randn(d)
    utterance_pma = template.expand(B, d).clone()
    verdict = slot_redundancy_monitor(slot_bank, utterance_pma)
    assert verdict.batch_cos_pct95 > BATCH_COS_PCT95_THRESHOLD
    assert "R-dkoleo-batch-cls-unit" in verdict.escalations


def test_slot_redundancy_rejects_wrong_shape_slot_bank() -> None:
    with pytest.raises(ValueError, match="slot_bank"):
        slot_redundancy_monitor(
            torch.zeros(4, 80),       # missing d dim
            torch.zeros(4, 32),
        )


def test_slot_redundancy_rejects_mismatched_batch_dim() -> None:
    with pytest.raises(ValueError, match="batch dim"):
        slot_redundancy_monitor(
            torch.zeros(4, 80, 32),
            torch.zeros(5, 32),
        )


# ---------------------------------------------------------------------------
# MON-SENSOR-TYPE-CANARY
# ---------------------------------------------------------------------------


def test_sensor_type_canary_threshold_is_locked_at_0p05() -> None:
    assert SENSOR_TYPE_CANARY_F1_THRESHOLD == 0.05


def test_sensor_type_canary_passes_when_features_independent_of_label() -> None:
    """Random features → linear probe can't beat chance → no kill."""
    torch.manual_seed(0)
    B, d = 200, 16
    features = torch.randn(B, d)
    labels = torch.randint(0, 2, (B,), dtype=torch.long)
    verdict = sensor_type_canary_monitor(
        features, labels, n_classes=2, baseline_f1=0.5, n_epochs=20, lr=1e-2,
    )
    assert not verdict.kill
    assert verdict.delta_f1 < SENSOR_TYPE_CANARY_F1_THRESHOLD


def test_sensor_type_canary_kills_when_features_leak_label() -> None:
    """Inject a label-correlated feature column → probe finds it → kill."""
    torch.manual_seed(0)
    B, d = 200, 16
    labels = torch.randint(0, 2, (B,), dtype=torch.long)
    features = torch.randn(B, d) + (labels.float() * 5.0).unsqueeze(-1)
    verdict = sensor_type_canary_monitor(
        features, labels, n_classes=2, baseline_f1=0.5, n_epochs=100, lr=5e-2,
    )
    assert verdict.kill, (
        f"label-leaking features should trip the canary; got delta={verdict.delta_f1}"
    )


def test_sensor_type_canary_rejects_wrong_features_shape() -> None:
    with pytest.raises(ValueError, match="features"):
        sensor_type_canary_monitor(
            torch.zeros(5),                   # missing d dim
            torch.zeros(5, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# MON-REF-TYPE-CANARY
# ---------------------------------------------------------------------------


def test_ref_type_canary_threshold_is_locked_at_0p05() -> None:
    assert REF_TYPE_CANARY_F1_THRESHOLD == 0.05


def test_ref_type_canary_three_way_chance_baseline() -> None:
    """Random features over 3-way labels → no kill."""
    torch.manual_seed(0)
    B, d = 300, 16
    features = torch.randn(B, d)
    ref_labels = torch.randint(0, 3, (B,), dtype=torch.long)
    verdict = ref_type_canary_monitor(features, ref_labels, n_epochs=20, lr=1e-2)
    assert not verdict.kill


def test_ref_type_canary_kills_on_label_leak() -> None:
    torch.manual_seed(0)
    B, d = 300, 16
    ref_labels = torch.randint(0, 3, (B,), dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(ref_labels, num_classes=3).float()
    features = torch.cat([one_hot * 5.0, torch.randn(B, d - 3)], dim=-1)
    verdict = ref_type_canary_monitor(features, ref_labels, n_epochs=100, lr=5e-2)
    assert verdict.kill


# ---------------------------------------------------------------------------
# MON-HEAD-BALANCE-005
# ---------------------------------------------------------------------------


def test_head_balance_bounds_locked_at_0p3_3p0() -> None:
    assert HEAD_BALANCE_BOUNDS == (0.3, 3.0)


def test_head_balance_uniform_attention_no_investigation() -> None:
    """All heads firing identically → relative usage = 1.0 → no flag."""
    torch.manual_seed(0)
    B, H, N_q, N_k = 2, 8, 16, 16
    attn = torch.nn.functional.softmax(torch.zeros(B, H, N_q, N_k), dim=-1)
    verdict = head_balance_monitor(attn)
    assert verdict.investigate is False
    assert verdict.starving_heads == ()
    assert verdict.dominating_heads == ()
    assert verdict.kill is False  # B29 demotion: never kill


def test_head_balance_dominating_head_flagged_but_does_not_kill() -> None:
    """One head's attention is sharply concentrated on a single key while
    the other heads stay diffuse → that head's relative concentration
    > 3.0 vs the layer mean. Investigate flag set, but ``kill`` stays
    False per B29 Item 9.
    """
    # H=5 + N_k=8 → diffuse-head concentration = 1/8 = 0.125; sharp head
    # = 1.0. Layer mean = (1 + 4·0.125)/5 = 0.3 → sharp head ratio ≈
    # 3.33 > 3.0 (dominating), diffuse heads ratio ≈ 0.42 > 0.3 (OK).
    B, H, N_q, N_k = 2, 5, 8, 8
    diffuse_logits = torch.zeros(B, H, N_q, N_k)
    # One sharply-concentrated head (large logit on key 0 → softmax ≈ 1.0
    # there) — concentration ≈ 1.0.
    sharp_logits = torch.zeros(B, N_q, N_k)
    sharp_logits[..., 0] = 50.0  # huge logit → softmax pmf ≈ one-hot
    diffuse_logits[:, 0] = sharp_logits
    attn = torch.nn.functional.softmax(diffuse_logits, dim=-1)
    verdict = head_balance_monitor(attn)
    assert verdict.investigate is True
    assert 0 in verdict.dominating_heads
    assert verdict.kill is False


def test_head_balance_rejects_non_4d_input() -> None:
    with pytest.raises(ValueError, match="must be"):
        head_balance_monitor(torch.zeros(2, 4, 8))
