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

All monitors are pure functions over encoder outputs (no model state),
so they're trivially testable with synthetic tensors.
"""

from speech_decoding.experiments.monitors.head_balance import (
    HEAD_BALANCE_BOUNDS,
    HeadBalanceVerdict,
    head_balance_monitor,
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

__all__ = [
    "BATCH_COS_PCT95_THRESHOLD",
    "DIAG_ZEROED_MEAN_THRESHOLD",
    "HEAD_BALANCE_BOUNDS",
    "HeadBalanceVerdict",
    "PER_CLIP_COS_PCT95_THRESHOLD",
    "PROBE_BATCH_SIZE_M1_DEFAULT",
    "REF_TYPE_CANARY_F1_THRESHOLD",
    "RefTypeCanaryVerdict",
    "SENSOR_TYPE_CANARY_F1_THRESHOLD",
    "SensorTypeCanaryVerdict",
    "SlotRedundancyVerdict",
    "head_balance_monitor",
    "ref_type_canary_monitor",
    "sensor_type_canary_monitor",
    "slot_redundancy_monitor",
]
