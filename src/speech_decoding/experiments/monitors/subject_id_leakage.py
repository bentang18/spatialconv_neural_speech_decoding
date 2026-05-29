"""MON-MASK-004: subject-ID linear-probe leakage canary.

Spec: ``docs/neuroprobe/v14_blockers.md §B03f`` (registered 2026-05-25 PM).

The B03f supervision gate ``parcels_supervised[subject]`` is a deterministic
per-subject set — it encodes subject identity as a side channel that the
encoder can latch onto, which would corrupt the CSubject gate.

This monitor fits a single-layer linear probe from the encoder's pooled
per-clip output (typically the PMA-pooled M4 utterance vector) to the
per-clip subject id, and flags leakage when probe macro-F1 exceeds the
threshold.

Default threshold: ``F1 > 0.50`` against the 9-subject BT cohort's chance
baseline of ``1 / 9 ≈ 0.111``. The 0.50 absolute threshold matches the
"P2 step ~30k subject-ID linear-probe F1 > 0.50" criterion in §B03f.

Probe-fit recipe matches the sensor/ref canaries:

* Single linear layer over the feature dim ``d``.
* SGD optimizer on the cross-entropy objective.
* "Probe on the training set" — this is a *capacity* test ("can these
  features encode subject identity?"), not a held-out classifier. The
  canary's job is to tell us when the encoder makes the label trivially
  separable.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from torch import Tensor

from speech_decoding.experiments.monitors.sensor_type_canary import (
    _fit_linear_probe_macro_f1,
    _validate_probe_inputs,
)


SUBJECT_ID_LEAKAGE_F1_THRESHOLD: float = 0.50
"""Absolute kill threshold (§B03f, 9-subject BT cohort)."""

CHANCE_F1_BT9: float = 1.0 / 9.0
"""Chance-baseline F1 over the 9-subject BT cohort (≈ 0.111)."""


@dataclass(frozen=True)
class SubjectIdLeakageVerdict:
    """Instantaneous MON-MASK-004 probe verdict.

    Fields
    ------
    probe_f1:
        Macro-F1 of the fitted single-layer subject-ID probe over the
        provided ``features``. Single-batch "capacity" estimate.
    chance_f1:
        ``1 / n_classes``. Reported alongside ``probe_f1`` so the
        verdict reads as an absolute number (not "relative to a
        cohort-shifting baseline").
    threshold:
        Absolute F1 cut. Default ``0.50``.
    over_threshold:
        ``True`` iff ``probe_f1 > threshold``.
    kill:
        Same as ``over_threshold``. Named ``kill`` to match the other
        canary verdicts' control surface.
    """

    probe_f1: float
    chance_f1: float
    threshold: float
    over_threshold: bool
    kill: bool


def subject_id_leakage_monitor(
    features: Tensor,
    subject_ids: Tensor,
    *,
    n_subjects: int = 9,
    threshold: float = SUBJECT_ID_LEAKAGE_F1_THRESHOLD,
    n_epochs: int = 50,
    lr: float = 1e-2,
) -> SubjectIdLeakageVerdict:
    """Fit a 1-layer linear probe and return the leakage verdict.

    Parameters
    ----------
    features: ``(B, d)`` per-clip pooled encoder features (e.g. PMA M4).
    subject_ids: ``(B,) int64`` per-clip subject id (0-indexed,
        ``< n_subjects``).
    n_subjects: number of subjects in the cohort. Default ``9`` for the
        BT 9-subject in-distribution cohort. The CSubject leave-one-out
        evaluation uses ``n_subjects = 8`` (held-out subject excluded).
    threshold: absolute F1 cut. Default ``0.50`` per §B03f.
    n_epochs: probe SGD epochs. Default ``50`` matches the sister
        canaries' recipe (sensor-type / ref-type).
    lr: probe SGD learning rate. Default ``1e-2``.
    """
    _validate_probe_inputs(features, subject_ids, label_name="subject_ids")
    max_id = int(subject_ids.max().item())
    if max_id >= n_subjects:
        raise ValueError(
            f"subject_ids contain id {max_id} >= n_subjects={n_subjects}; "
            "0-index the ids and pass the correct n_subjects."
        )
    min_id = int(subject_ids.min().item())
    if min_id < 0:
        raise ValueError(
            f"subject_ids contain negative id {min_id}; expected 0..n-1"
        )
    if n_subjects < 2:
        raise ValueError(f"need n_subjects >= 2; got {n_subjects}")

    probe_f1 = _fit_linear_probe_macro_f1(
        features, subject_ids,
        n_classes=n_subjects, n_epochs=n_epochs, lr=lr,
    )
    over = probe_f1 > threshold
    return SubjectIdLeakageVerdict(
        probe_f1=probe_f1,
        chance_f1=1.0 / n_subjects,
        threshold=threshold,
        over_threshold=over,
        kill=over,
    )


__all__: tp.Final[tuple[str, ...]] = (
    "CHANCE_F1_BT9",
    "SUBJECT_ID_LEAKAGE_F1_THRESHOLD",
    "SubjectIdLeakageVerdict",
    "subject_id_leakage_monitor",
)
