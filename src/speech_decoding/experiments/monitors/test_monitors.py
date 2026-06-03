"""Tests for the B28/B29 SSL-side training-time monitors."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.experiments.monitors import (
    BATCH_COS_PCT95_THRESHOLD,
    CHANCE_F1_BT9,
    DIAG_ZEROED_MEAN_THRESHOLD,
    ESCALATE_SHAFT_K2,
    ESCALATE_STRATIFIED_SHAFT_MASK,
    HEAD_BALANCE_BOUNDS,
    MASK_ORPHAN_MAX_RATIO,
    MASK_ORPHAN_MIN_RATIO,
    PER_CLIP_COS_PCT95_THRESHOLD,
    PROBE_BATCH_SIZE_M1_DEFAULT,
    REF_TYPE_CANARY_F1_THRESHOLD,
    SENSOR_TYPE_CANARY_F1_THRESHOLD,
    SUBJECT_ID_LEAKAGE_F1_THRESHOLD,
    compute_orphan_parcels,
    head_balance_monitor,
    mask_orphan_ratio_monitor,
    ref_type_canary_monitor,
    sensor_type_canary_monitor,
    slot_redundancy_monitor,
    subject_id_leakage_monitor,
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


def test_slot_redundancy_latent_valid_masks_degenerate_uncovered_slots() -> None:
    """K=80 sparse regime: only 10 of 80 slots are covered (diverse
    random vectors); the 70 uncovered slots are identical empty-parcel
    pools. Without the mask the degenerate majority dominates → intra-clip
    + vicreg escalate. With latent_valid the stats see only the 10 diverse
    covered slots → no intra-clip escalation."""
    torch.manual_seed(3)
    B, L, d = 4, 80, 32
    n_cov = 10
    degenerate = torch.randn(d)
    slot_bank = degenerate.expand(B, L, d).clone()
    latent_valid = torch.zeros(B, L, dtype=torch.bool)
    for b in range(B):
        slot_bank[b, :n_cov] = torch.randn(n_cov, d)
        latent_valid[b, :n_cov] = True
    utterance_pma = torch.randn(B, d)

    unmasked = slot_redundancy_monitor(slot_bank, utterance_pma)
    assert "R-dkoleo-intra-clip-slots" in unmasked.escalations
    assert "R-vicreg-slot-variance" in unmasked.escalations

    masked = slot_redundancy_monitor(slot_bank, utterance_pma, latent_valid)
    assert masked.per_clip_cos_pct95 < PER_CLIP_COS_PCT95_THRESHOLD
    assert "R-dkoleo-intra-clip-slots" not in masked.escalations


def test_slot_redundancy_fewer_than_two_covered_contributes_zero() -> None:
    """A batch where every clip covers < 2 slots cannot yield a pairwise
    similarity → per-clip stats are 0.0 and nothing escalates intra-clip."""
    B, L, d = 4, 80, 32
    slot_bank = torch.randn(B, L, d)
    latent_valid = torch.zeros(B, L, dtype=torch.bool)
    latent_valid[:, 0] = True  # exactly 1 covered slot per clip
    utterance_pma = torch.randn(B, d)
    v = slot_redundancy_monitor(slot_bank, utterance_pma, latent_valid)
    assert v.per_clip_cos_pct95 == 0.0
    assert v.diag_zeroed_mean == 0.0
    assert "R-dkoleo-intra-clip-slots" not in v.escalations


def test_slot_redundancy_rejects_wrong_shape_latent_valid() -> None:
    with pytest.raises(ValueError, match=r"latent_valid must be \(B, L\)"):
        slot_redundancy_monitor(
            torch.randn(4, 80, 32),
            torch.randn(4, 32),
            torch.zeros(4, 79, dtype=torch.bool),  # wrong L
        )


def test_slot_redundancy_rejects_non_bool_latent_valid() -> None:
    with pytest.raises(ValueError, match="latent_valid must be bool"):
        slot_redundancy_monitor(
            torch.randn(4, 80, 32),
            torch.randn(4, 32),
            torch.zeros(4, 80),  # float, not bool
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


def test_sensor_type_canary_default_baseline_is_chance_for_n_classes() -> None:
    """baseline_f1=None resolves to chance = 1/n_classes (0.5 binary,
    1/3 three-way), not a hardcoded 0.5 — so the 3-way subtype sister
    compares its probe against the correct chance level."""
    torch.manual_seed(1)
    B, d = 240, 16
    v3 = sensor_type_canary_monitor(
        torch.randn(B, d),
        torch.randint(0, 3, (B,), dtype=torch.long),
        n_classes=3, n_epochs=20, lr=1e-2,
    )
    assert v3.baseline_f1 == pytest.approx(1.0 / 3.0)
    v2 = sensor_type_canary_monitor(
        torch.randn(B, d),
        torch.randint(0, 2, (B,), dtype=torch.long),
        n_classes=2, n_epochs=20, lr=1e-2,
    )
    assert v2.baseline_f1 == 0.5


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


# ---------------------------------------------------------------------------
# MON-MASK-002 (mask_orphan_ratio)
# ---------------------------------------------------------------------------


def test_mask_orphan_ratio_band_locked_at_0p7_1p5() -> None:
    """B03d lock: healthy band is [0.7, 1.5] under K=1 default."""
    assert MASK_ORPHAN_MIN_RATIO == 0.7
    assert MASK_ORPHAN_MAX_RATIO == 1.5


def test_mask_orphan_ratio_balanced_signal_in_band() -> None:
    """When orphan & visible parcels have similar per-parcel MSE, the
    ratio is 1.0 and ``in_band`` is True."""
    torch.manual_seed(0)
    B, P, T, d = 4, 8, 5, 6
    # Student and teacher differ by the same iid noise everywhere → per-
    # parcel MSE is equal in expectation across parcels.
    student = torch.randn(B, P, T, d)
    noise = torch.randn(B, P, T, d) * 0.1
    teacher = student + noise
    orphan = torch.zeros(P, dtype=torch.bool)
    orphan[:3] = True  # first 3 parcels are orphan, rest visible
    verdict = mask_orphan_ratio_monitor(
        student, teacher, orphan, parcel_dim=1,
    )
    assert verdict.in_band is True
    assert verdict.escalations == ()
    assert MASK_ORPHAN_MIN_RATIO <= verdict.ratio <= MASK_ORPHAN_MAX_RATIO


def test_mask_orphan_ratio_below_band_escalates_stratified() -> None:
    """Orphan MSE much smaller than visible MSE (mean-collapse signature)
    → ``R-stratified-shaft-mask`` escalation."""
    B, P, T, d = 2, 6, 4, 4
    student = torch.zeros(B, P, T, d)
    teacher = torch.zeros(B, P, T, d)
    # Orphan parcels: student ≈ teacher (tiny error). Visible parcels:
    # large error.
    orphan = torch.zeros(P, dtype=torch.bool)
    orphan[:3] = True
    teacher[:, orphan] = 0.01    # tiny error on orphan side
    teacher[:, ~orphan] = 1.0    # big error on visible side
    verdict = mask_orphan_ratio_monitor(
        student, teacher, orphan, parcel_dim=1,
    )
    assert verdict.ratio < MASK_ORPHAN_MIN_RATIO
    assert verdict.in_band is False
    assert ESCALATE_STRATIFIED_SHAFT_MASK in verdict.escalations


def test_mask_orphan_ratio_above_band_escalates_shaft_k2() -> None:
    """Orphan MSE much larger than visible MSE (student failing the
    prediction task) → ``R-shaft-K2`` escalation."""
    B, P, T, d = 2, 6, 4, 4
    student = torch.zeros(B, P, T, d)
    teacher = torch.zeros(B, P, T, d)
    orphan = torch.zeros(P, dtype=torch.bool)
    orphan[:3] = True
    teacher[:, orphan] = 5.0     # huge error on orphan side
    teacher[:, ~orphan] = 0.5    # small error on visible side
    verdict = mask_orphan_ratio_monitor(
        student, teacher, orphan, parcel_dim=1,
    )
    assert verdict.ratio > MASK_ORPHAN_MAX_RATIO
    assert verdict.in_band is False
    assert ESCALATE_SHAFT_K2 in verdict.escalations


def test_mask_orphan_ratio_no_orphan_returns_nan_verdict() -> None:
    """Batch with no orphan parcels → ratio = nan, no escalation, training
    loop must not tick the sustain counter."""
    B, P, T, d = 2, 4, 4, 4
    student = torch.randn(B, P, T, d)
    teacher = torch.randn(B, P, T, d)
    orphan = torch.zeros(P, dtype=torch.bool)
    verdict = mask_orphan_ratio_monitor(
        student, teacher, orphan, parcel_dim=1,
    )
    assert verdict.ratio != verdict.ratio  # nan
    assert verdict.in_band is False
    assert verdict.escalations == ()


def test_mask_orphan_ratio_no_visible_returns_nan_verdict() -> None:
    """All parcels orphan (pathological) → no visible side → ratio nan."""
    B, P, T, d = 2, 4, 4, 4
    student = torch.randn(B, P, T, d)
    teacher = torch.randn(B, P, T, d)
    orphan = torch.ones(P, dtype=torch.bool)
    verdict = mask_orphan_ratio_monitor(
        student, teacher, orphan, parcel_dim=1,
    )
    assert verdict.ratio != verdict.ratio
    assert verdict.escalations == ()


def test_mask_orphan_ratio_rejects_shape_mismatch() -> None:
    student = torch.randn(2, 4, 3, 5)
    teacher = torch.randn(2, 4, 3, 6)
    orphan = torch.zeros(4, dtype=torch.bool)
    with pytest.raises(ValueError, match="must share shape"):
        mask_orphan_ratio_monitor(student, teacher, orphan, parcel_dim=1)


def test_mask_orphan_ratio_rejects_non_bool_orphan_mask() -> None:
    student = torch.randn(2, 4, 3, 5)
    teacher = torch.randn(2, 4, 3, 5)
    orphan = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(ValueError, match="must be bool"):
        mask_orphan_ratio_monitor(student, teacher, orphan, parcel_dim=1)


def test_mask_orphan_ratio_rejects_parcel_axis_mismatch() -> None:
    student = torch.randn(2, 4, 3, 5)
    teacher = torch.randn(2, 4, 3, 5)
    orphan = torch.zeros(8, dtype=torch.bool)  # P=8 but tap has P=4
    with pytest.raises(ValueError, match="parcel axis size mismatch"):
        mask_orphan_ratio_monitor(student, teacher, orphan, parcel_dim=1)


def test_compute_orphan_parcels_dk_one_hot_drop_one_shaft() -> None:
    """C=4 electrodes, P=3 parcels, all electrodes cover distinct
    parcels. Shaft-mask electrodes {0,1} → parcels {0,1} lose all
    coverage → both flagged orphan."""
    B, C, P = 2, 4, 3
    support = torch.zeros(B, C, P)
    support[:, 0, 0] = 1.0   # elec 0 covers parcel 0
    support[:, 1, 1] = 1.0   # elec 1 covers parcel 1
    support[:, 2, 2] = 1.0   # elec 2 covers parcel 2
    support[:, 3, 2] = 1.0   # elec 3 also covers parcel 2 (backup)
    shaft = torch.zeros(B, C, dtype=torch.bool)
    shaft[:, :2] = True       # drop electrodes 0 and 1
    orphan = compute_orphan_parcels(shaft, support)
    expected = torch.tensor([True, True, False])
    torch.testing.assert_close(orphan, expected)


def test_compute_orphan_parcels_per_clip_reduction_intersection() -> None:
    """A parcel is orphan only if BOTH clips lose all coverage. Clip 0
    drops electrode 0 only → parcel 0 orphan in clip 0; clip 1 drops
    electrode 1 → parcel 0 NOT orphan in clip 1 → batch-level not
    orphan."""
    B, C, P = 2, 2, 2
    support = torch.zeros(B, C, P)
    support[:, 0, 0] = 1.0   # elec 0 covers parcel 0
    support[:, 1, 1] = 1.0   # elec 1 covers parcel 1
    shaft = torch.zeros(B, C, dtype=torch.bool)
    shaft[0, 0] = True        # clip 0 loses parcel 0
    shaft[1, 1] = True        # clip 1 loses parcel 1
    orphan = compute_orphan_parcels(shaft, support)
    # Batch intersection: no parcel is orphan in BOTH clips.
    expected = torch.tensor([False, False])
    torch.testing.assert_close(orphan, expected)


def test_compute_orphan_parcels_rejects_bad_shapes() -> None:
    support = torch.zeros(2, 4, 3)
    with pytest.raises(ValueError, match="shaft_mask must be \\(B, C\\)"):
        compute_orphan_parcels(torch.zeros(2, dtype=torch.bool), support)
    shaft = torch.zeros(2, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="support must be \\(B, C, P\\)"):
        compute_orphan_parcels(shaft, torch.zeros(2, 4))


# ---------------------------------------------------------------------------
# MON-MASK-004 (subject-ID leakage canary)
# ---------------------------------------------------------------------------


def test_subject_id_leakage_threshold_locked_at_0p5() -> None:
    """§B03f lock: F1 > 0.50 is the kill threshold; chance baseline is
    1 / 9 ≈ 0.111 over the BT 9-subject cohort."""
    assert SUBJECT_ID_LEAKAGE_F1_THRESHOLD == 0.50
    assert abs(CHANCE_F1_BT9 - 1.0 / 9.0) < 1e-9


def test_subject_id_leakage_random_features_below_threshold() -> None:
    """Random features uncorrelated with subject id → probe macro-F1
    well below the 0.50 absolute kill threshold."""
    torch.manual_seed(0)
    B, d, n_sub = 36, 16, 9
    features = torch.randn(B, d)
    subjects = torch.randint(0, n_sub, (B,))
    verdict = subject_id_leakage_monitor(
        features, subjects, n_subjects=n_sub, n_epochs=30,
    )
    assert verdict.probe_f1 < SUBJECT_ID_LEAKAGE_F1_THRESHOLD
    assert verdict.kill is False


def test_subject_id_leakage_id_encoded_features_trip_threshold() -> None:
    """When the encoder leaks subject identity, a single linear probe can
    perfectly recover the subject id and the canary kills."""
    B, n_sub = 36, 9
    subjects = torch.arange(B) % n_sub
    # One-hot per subject as the feature → linearly separable.
    features = torch.zeros(B, n_sub)
    features[torch.arange(B), subjects] = 5.0
    verdict = subject_id_leakage_monitor(
        features, subjects, n_subjects=n_sub, n_epochs=80,
    )
    assert verdict.probe_f1 > SUBJECT_ID_LEAKAGE_F1_THRESHOLD
    assert verdict.kill is True


def test_subject_id_leakage_rejects_out_of_range_id() -> None:
    features = torch.randn(4, 8)
    subjects = torch.tensor([0, 1, 2, 9])  # id 9 outside n=9 (0-indexed)
    with pytest.raises(ValueError, match=">= n_subjects"):
        subject_id_leakage_monitor(features, subjects, n_subjects=9)


def test_subject_id_leakage_rejects_negative_id() -> None:
    features = torch.randn(4, 8)
    subjects = torch.tensor([0, -1, 2, 3])
    with pytest.raises(ValueError, match="negative id"):
        subject_id_leakage_monitor(features, subjects, n_subjects=9)


def test_subject_id_leakage_rejects_small_cohort() -> None:
    features = torch.randn(4, 8)
    subjects = torch.tensor([0, 0, 0, 0])
    with pytest.raises(ValueError, match="n_subjects >= 2"):
        subject_id_leakage_monitor(features, subjects, n_subjects=1)
