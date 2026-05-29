"""MON-REF-TYPE-CANARY: B29 Item 9 reference-operator linear-probe canary.

Symmetric analog of :mod:`sensor_type_canary` but probes for over-reliance
on ``ref_embed`` (B29 Item 11). Same kill threshold ``+0.05`` over a
baseline encoder run.

Lock memo: ``memory/project_v14_b29_joint_default_2026_05_27.md`` §Item 9.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from torch import Tensor

from speech_decoding.experiments.monitors.sensor_type_canary import (
    _fit_linear_probe_macro_f1,
    _validate_probe_inputs,
)


REF_TYPE_CANARY_F1_THRESHOLD: float = 0.05
"""Kill if the canary's F1 exceeds baseline by more than this margin."""

# REF-aug uses 3 cells {shaft_car, bipolar, laplacian} per the 5/27 PM
# lock; the canary classifier is 3-way.
REF_TYPE_CANARY_N_CLASSES: int = 3


@dataclass(frozen=True)
class RefTypeCanaryVerdict:
    canary_f1: float
    baseline_f1: float
    delta_f1: float
    over_threshold: bool
    threshold: float
    kill: bool


def ref_type_canary_monitor(
    features: Tensor,
    ref_labels: Tensor,
    *,
    n_classes: int = REF_TYPE_CANARY_N_CLASSES,
    baseline_f1: float = 1.0 / REF_TYPE_CANARY_N_CLASSES,
    threshold: float = REF_TYPE_CANARY_F1_THRESHOLD,
    n_epochs: int = 50,
    lr: float = 1e-2,
) -> RefTypeCanaryVerdict:
    """Fit a 1-layer linear probe over ``ref_idx`` ids and return verdict.

    Parameters mirror :func:`sensor_type_canary_monitor`. The default
    ``baseline_f1`` is chance for a balanced 3-way classifier (1/3).
    """
    # Both canaries share the same validation + fit body; reuse the
    # helpers in :mod:`sensor_type_canary`.
    _validate_probe_inputs(features, ref_labels, label_name="ref_labels")
    canary_f1 = _fit_linear_probe_macro_f1(
        features, ref_labels, n_classes=n_classes, n_epochs=n_epochs, lr=lr,
    )
    delta = canary_f1 - baseline_f1
    over = delta > threshold
    return RefTypeCanaryVerdict(
        canary_f1=canary_f1,
        baseline_f1=baseline_f1,
        delta_f1=delta,
        over_threshold=over,
        threshold=threshold,
        kill=over,
    )


__all__: tp.Final[tuple[str, ...]] = (
    "REF_TYPE_CANARY_F1_THRESHOLD",
    "REF_TYPE_CANARY_N_CLASSES",
    "RefTypeCanaryVerdict",
    "ref_type_canary_monitor",
)
