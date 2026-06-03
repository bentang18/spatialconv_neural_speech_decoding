"""MON-SENSOR-TYPE-CANARY: B29 Item 9 subject-subtype linear-probe canary.

Detects over-reliance on the ``subtype_embed`` (B29 Item 11) by running
a linear probe from the encoder output to the per-clip subject-subtype.
Kill criterion: F1 increases by > 0.05 over a baseline encoder run.

Lock memo: ``memory/project_v14_b29_joint_default_2026_05_27.md`` §Item 9.

The full monitor (encoder forward → cache pooled embeddings → fit a
linear classifier on a held-out batch → compare F1 against baseline)
lives in the training-loop wrapper. This module exposes:

  * :data:`SENSOR_TYPE_CANARY_F1_THRESHOLD` — the +0.05 kill threshold.
  * :class:`SensorTypeCanaryVerdict` — the verdict struct.
  * :func:`sensor_type_canary_monitor` — pure function over (features,
    labels, baseline_f1) → verdict.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import torch
from torch import Tensor


SENSOR_TYPE_CANARY_F1_THRESHOLD: float = 0.05
"""Kill if the canary's F1 exceeds baseline by more than this margin."""


@dataclass(frozen=True)
class SensorTypeCanaryVerdict:
    canary_f1: float
    baseline_f1: float
    delta_f1: float
    over_threshold: bool
    threshold: float
    kill: bool


def _macro_f1(preds: Tensor, labels: Tensor, *, n_classes: int) -> float:
    """Vectorized macro-F1 (no sklearn dep)."""
    f1_sum = 0.0
    n_present = 0
    for k in range(n_classes):
        pred_k = preds == k
        true_k = labels == k
        tp_ = int((pred_k & true_k).sum().item())
        fp_ = int((pred_k & ~true_k).sum().item())
        fn_ = int((~pred_k & true_k).sum().item())
        if (tp_ + fp_) == 0 or (tp_ + fn_) == 0:
            # No predictions or no true labels of this class — skip per
            # sklearn's macro-F1 ``zero_division="warn"`` convention.
            continue
        precision = tp_ / (tp_ + fp_)
        recall = tp_ / (tp_ + fn_)
        if precision + recall == 0:
            continue
        f1_sum += 2 * precision * recall / (precision + recall)
        n_present += 1
    return f1_sum / max(n_present, 1)


def _validate_probe_inputs(
    features: Tensor, labels: Tensor, *, label_name: str = "labels",
) -> tuple[int, int]:
    """Shape / size validation shared by all canary probes."""
    if features.dim() != 2:
        raise ValueError(
            f"features must be (B, d); got shape {tuple(features.shape)}"
        )
    if labels.dim() != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError(
            f"{label_name} must be (B={features.shape[0]},); "
            f"got shape {tuple(labels.shape)}"
        )
    B, d = features.shape
    if B < 2:
        raise ValueError(f"need at least 2 clips for a probe; got B={B}")
    return B, d


def _fit_linear_probe_macro_f1(
    features: Tensor,
    labels: Tensor,
    *,
    n_classes: int,
    n_epochs: int,
    lr: float,
) -> float:
    """Fit a single-layer linear probe and return macro-F1 on the fit set.

    Shared by the sensor- and ref-type canaries so they use the same
    gradient update, device handling, and F1 computation. The "probe
    on the training set" pattern is intentional: the canary is a
    *capacity* test ("can these features encode the label?"), not a
    held-out classifier.
    """
    # Probe lives on the feature tensor's device / dtype so the matmul
    # stays device-consistent under Lightning's accelerator="auto".
    feats = features.detach()
    targets = labels.detach().to(device=feats.device, dtype=torch.long)
    d = int(feats.shape[1])
    probe = torch.nn.Linear(d, n_classes).to(device=feats.device, dtype=feats.dtype)
    optim = torch.optim.SGD(probe.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        optim.zero_grad(set_to_none=True)
        logits = probe(feats)
        loss = loss_fn(logits, targets)
        loss.backward()
        optim.step()
    with torch.no_grad():
        preds = probe(feats).argmax(dim=-1)
    return _macro_f1(preds, targets, n_classes=n_classes)


def sensor_type_canary_monitor(
    features: Tensor,
    labels: Tensor,
    *,
    n_classes: int = 2,
    baseline_f1: float | None = None,
    threshold: float = SENSOR_TYPE_CANARY_F1_THRESHOLD,
    n_epochs: int = 50,
    lr: float = 1e-2,
) -> SensorTypeCanaryVerdict:
    """Fit a 1-layer linear probe and return the canary verdict.

    Parameters
    ----------
    features: ``(B, d)`` per-clip encoder features (e.g. PMA-pooled M4).
    labels:   ``(B,)`` per-clip subject-subtype ids (int64).
    n_classes: 2 (binary default) or 3 (three-way sister).
    baseline_f1: macro-F1 of a baseline encoder without ``subtype_embed``.
        The canary's F1 must not exceed ``baseline_f1 + threshold``.
        ``None`` (default) resolves to chance for a balanced classifier,
        ``1 / n_classes`` (0.5 binary, 1/3 three-way) — matching
        :func:`ref_type_canary_monitor`. A hardcoded 0.5 was only correct
        for ``n_classes == 2``.
    threshold: kill margin (default 0.05 per B29 Item 9 lock).
    """
    _validate_probe_inputs(features, labels)
    if baseline_f1 is None:
        baseline_f1 = 1.0 / n_classes
    canary_f1 = _fit_linear_probe_macro_f1(
        features, labels, n_classes=n_classes, n_epochs=n_epochs, lr=lr,
    )
    delta = canary_f1 - baseline_f1
    over = delta > threshold
    return SensorTypeCanaryVerdict(
        canary_f1=canary_f1,
        baseline_f1=baseline_f1,
        delta_f1=delta,
        over_threshold=over,
        threshold=threshold,
        kill=over,
    )


__all__: tp.Final[tuple[str, ...]] = (
    "SENSOR_TYPE_CANARY_F1_THRESHOLD",
    "SensorTypeCanaryVerdict",
    "sensor_type_canary_monitor",
)
