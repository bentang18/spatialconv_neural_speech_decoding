"""v14_converged_v3 — shaft/index sidecar (Phase 1).

Per-session geometry the v3 encoder + masking consume, derived from the ordered
surviving electrode labels (the ``voltage_electrode_order`` survivors) and the
hard atlas support. Memo: project-v14-converged-v3-sensor-architecture.

Three per-contact tensors, all row-aligned to the voltage order:

  shaft_id  (N,) long  — contiguous group id, order of first appearance. Groups
                         contacts into the L1 block-diagonal "sensor" and is the
                         weight for mask sampling ∝ contact-count.
  depth     (N,) long  — CANONICAL along-shaft contact index = the clinical
                         contact number peeled off the label (``parse_shaft[1]``).
                         Off the FULL shaft with drop-gaps PRESERVED: a dropped
                         LTG5 leaves LTG4→4 and LTG6→6 (relative distance 2). This
                         is the index-RoPE coordinate; only |depth_i − depth_j|
                         within a shaft matters, so keeping the raw clinical number
                         (never densely re-numbering) is both correct and
                         flip/offset-invariant. NOT ``_bt_contact_pos_from_labels``,
                         which normalises to a dense [0,1] ramp and destroys gaps.
  parcel_id (N,) long  — DKT/DK hard tag = ``support.argmax(-1)``, the cross-subject
                         query identity (L2 tag + predictor mask-query seed).

Construction iterates the label strings once (unavoidable string parsing, C≤384,
one-time per session at setup — NOT the training hot loop, so the "always
vectorize" rule that governs per-step masking does not apply here).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from speech_decoding.extractors.reference import parse_shaft


@dataclass(frozen=True)
class SensorSidecar:
    """Row-aligned per-contact geometry for one session's surviving electrodes."""

    shaft_id: Tensor  # (N,) long
    depth: Tensor  # (N,) long — canonical clinical contact index, gaps preserved
    parcel_id: Tensor  # (N,) long — hard atlas tag
    n_shafts: int
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.labels)
        for name, t in (("shaft_id", self.shaft_id), ("depth", self.depth),
                        ("parcel_id", self.parcel_id)):
            if t.shape != (n,):
                raise ValueError(f"{name} shape {tuple(t.shape)} != ({n},)")
            if t.dtype != torch.long:
                raise ValueError(f"{name} dtype {t.dtype} != torch.long")


def _parse_shaft_depth(labels: Sequence[str]) -> tuple[Tensor, Tensor, int]:
    """labels → (shaft_id (N,) long, depth (N,) long, n_shafts).

    shaft_id: contiguous, order of first appearance (same alphabetic prefix ⇒ same
    shaft). depth: the trailing clinical contact number, kept verbatim (gaps
    preserved). A label with no trailing number cannot have a canonical depth and
    is rejected (it is a mislabel or a non-neural channel that should have been
    dropped upstream).
    """
    seen: dict[str, int] = {}
    shaft: list[int] = []
    depth: list[int] = []
    for label in labels:
        prefix, num = parse_shaft(str(label))
        if num is None:
            raise ValueError(
                f"label {label!r} has no trailing contact number — cannot assign a "
                "canonical depth index (drop non-neural/trigger channels upstream)"
            )
        if prefix not in seen:
            seen[prefix] = len(seen)
        shaft.append(seen[prefix])
        depth.append(num)
    return (
        torch.tensor(shaft, dtype=torch.long),
        torch.tensor(depth, dtype=torch.long),
        len(seen),
    )


def build_sidecar(
    labels: Sequence[str],
    *,
    parcel_id: Tensor | None = None,
    support: Tensor | None = None,
) -> SensorSidecar:
    """Assemble the sidecar for one session's surviving electrode labels.

    Provide the hard parcel tag either directly (``parcel_id`` (N,) long) or as the
    one-hot ``support`` (N, K) to argmax. Exactly one is required.
    """
    labels = tuple(str(x) for x in labels)
    n = len(labels)
    shaft_id, depth, n_shafts = _parse_shaft_depth(labels)

    if parcel_id is not None:
        pid = parcel_id.long()
        if pid.shape != (n,):
            raise ValueError(f"parcel_id shape {tuple(pid.shape)} != ({n},)")
    elif support is not None:
        if support.ndim != 2 or support.shape[0] != n:
            raise ValueError(
                f"support shape {tuple(support.shape)} incompatible with N={n}"
            )
        pid = support.argmax(dim=-1).long()
    else:
        raise ValueError("provide either parcel_id or support")

    return SensorSidecar(
        shaft_id=shaft_id,
        depth=depth,
        parcel_id=pid,
        n_shafts=n_shafts,
        labels=labels,
    )
