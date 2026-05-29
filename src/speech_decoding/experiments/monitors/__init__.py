"""B28/B29 SSL-side training-time monitors (collapse / canary probes).

Each module under ``experiments.monitors`` implements one MON-*
diagnostic from the v14 fix list (``docs/neuroprobe/v14_implementation_fix_list.md``):

  * :mod:`slot_redundancy` — MON-SLOT-REDUNDANCY (B28). Slot-bank
    self-similarity probe. Drives DKoleo-sister escalation gates.
  * :mod:`sensor_type_canary` — MON-SENSOR-TYPE-CANARY (B29 Item 9).
    Subject-subtype linear probe on the encoder output.
  * :mod:`ref_type_canary` — MON-REF-TYPE-CANARY (B29 Item 9). Ref
    operator linear probe analog.
  * :mod:`head_balance` — MON-HEAD-BALANCE-005 (B29 Item 9 demoted).
    Per-head usage ratio; health-canary not kill criterion.
  * :mod:`mask_orphan_ratio` — MON-MASK-002 (B03d). Orphan-vs-visible
    parcel MSE ratio bounded by ``[0.7, 1.5]`` under K=1 default.
  * :mod:`subject_id_leakage` — MON-MASK-004 (B03f). Subject-ID
    linear-probe canary on the encoder pooled output (BT9 cohort
    threshold F1 > 0.50).
  * :mod:`grad_spike` — MON-GRAD-SPIKE-DIVERGENCE (5/28 P0). SPAM
    grad-L2 spike + AdamW divergence probe under joint P1+P2.
  * :mod:`parcel_coverage` — MON-PARCEL-COVERAGE-VARIANCE (5/28 P0).
    Per-batch latent_valid degeneracy + Switch-Transformer load-
    balance variance analog.
  * :mod:`teacher_rank` — MON-TEACHER-FEATURE-RANK (5/28 P0). RankMe
    dimensional-collapse probe on EMA-teacher M4 features.

All monitors are pure functions over encoder outputs (no model state),
so they're trivially testable with synthetic tensors.
"""

from speech_decoding.experiments.monitors.grad_spike import (
    GRAD_EMA_ALPHA,
    GRAD_SPIKE_THRESHOLD,
    GradSpikeVerdict,
    grad_spike_monitor,
)
from speech_decoding.experiments.monitors.head_balance import (
    HEAD_BALANCE_BOUNDS,
    HeadBalanceVerdict,
    head_balance_monitor,
)
from speech_decoding.experiments.monitors.mask_orphan_ratio import (
    ESCALATE_SHAFT_K2,
    ESCALATE_STRATIFIED_SHAFT_MASK,
    MAX_RATIO as MASK_ORPHAN_MAX_RATIO,
    MIN_RATIO as MASK_ORPHAN_MIN_RATIO,
    MaskOrphanRatioVerdict,
    compute_orphan_parcels,
    mask_orphan_ratio_monitor,
)
from speech_decoding.experiments.monitors.parcel_coverage import (
    DEGENERATE_CLIP_FRACTION_ALARM,
    DEGENERATE_SLOT_COUNT_MAX,
    SLOT_USAGE_VARIANCE_ALARM,
    ParcelCoverageVerdict,
    parcel_coverage_monitor,
)
from speech_decoding.experiments.monitors.ref_type_canary import (
    REF_TYPE_CANARY_F1_THRESHOLD,
    RefTypeCanaryVerdict,
    ref_type_canary_monitor,
)
from speech_decoding.experiments.monitors.sensor_type_canary import (
    SENSOR_TYPE_CANARY_F1_THRESHOLD,
    SensorTypeCanaryVerdict,
    sensor_type_canary_monitor,
)
from speech_decoding.experiments.monitors.slot_redundancy import (
    BATCH_COS_PCT95_THRESHOLD,
    DIAG_ZEROED_MEAN_THRESHOLD,
    PER_CLIP_COS_PCT95_THRESHOLD,
    PROBE_BATCH_SIZE_M1_DEFAULT,
    SlotRedundancyVerdict,
    slot_redundancy_monitor,
)
from speech_decoding.experiments.monitors.subject_id_leakage import (
    CHANCE_F1_BT9,
    SUBJECT_ID_LEAKAGE_F1_THRESHOLD,
    SubjectIdLeakageVerdict,
    subject_id_leakage_monitor,
)
from speech_decoding.experiments.monitors.teacher_rank import (
    RANKME_NORMALISED_ALARM,
    RANKME_NORMALISED_WARN,
    TeacherRankVerdict,
    teacher_rank_monitor,
)

__all__ = [
    "BATCH_COS_PCT95_THRESHOLD",
    "CHANCE_F1_BT9",
    "DEGENERATE_CLIP_FRACTION_ALARM",
    "DEGENERATE_SLOT_COUNT_MAX",
    "DIAG_ZEROED_MEAN_THRESHOLD",
    "ESCALATE_SHAFT_K2",
    "ESCALATE_STRATIFIED_SHAFT_MASK",
    "GRAD_EMA_ALPHA",
    "GRAD_SPIKE_THRESHOLD",
    "GradSpikeVerdict",
    "HEAD_BALANCE_BOUNDS",
    "HeadBalanceVerdict",
    "MASK_ORPHAN_MAX_RATIO",
    "MASK_ORPHAN_MIN_RATIO",
    "MaskOrphanRatioVerdict",
    "PER_CLIP_COS_PCT95_THRESHOLD",
    "PROBE_BATCH_SIZE_M1_DEFAULT",
    "ParcelCoverageVerdict",
    "RANKME_NORMALISED_ALARM",
    "RANKME_NORMALISED_WARN",
    "REF_TYPE_CANARY_F1_THRESHOLD",
    "RefTypeCanaryVerdict",
    "SENSOR_TYPE_CANARY_F1_THRESHOLD",
    "SLOT_USAGE_VARIANCE_ALARM",
    "SUBJECT_ID_LEAKAGE_F1_THRESHOLD",
    "SensorTypeCanaryVerdict",
    "SlotRedundancyVerdict",
    "SubjectIdLeakageVerdict",
    "TeacherRankVerdict",
    "compute_orphan_parcels",
    "grad_spike_monitor",
    "head_balance_monitor",
    "mask_orphan_ratio_monitor",
    "parcel_coverage_monitor",
    "ref_type_canary_monitor",
    "sensor_type_canary_monitor",
    "slot_redundancy_monitor",
    "subject_id_leakage_monitor",
    "teacher_rank_monitor",
]
