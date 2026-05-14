"""V14 DK-hard-one-hot support extractor.

Reads BT-shipped ``localization/sub_<id>/depth-wm.csv`` and emits per-event
``(n_electrodes, K=80)`` one-hot support over the canonical v14 DK parcel
vocabulary (``V14_DK_PARCEL_LABELS``). Consumed by the v14 encoder cross-attn
``log(support + eps)`` Graphormer-style anatomy bias.

Row order matches depth-wm.csv row order — the canonical per-subject electrode
order that downstream NeuralSet alignment assumes.

Labels falling outside the K=80 vocabulary (e.g. BT btbank4 has
``Left-Inf-Lat-Vent`` ventricle electrodes) raise by default; cohort-loading
layers can pass ``unknown_label_policy="skip"`` to drop those electrodes.
"""

from __future__ import annotations

import re
import typing as tp
from pathlib import Path

import torch
from neuralset.events.etypes import Event
from neuralset.extractors.base import BaseStatic

from speech_decoding.studies.braintreebank.anatomy import (
    V14_DK_PARCEL_LABELS,
    build_hard_public_bt_label_support,
    load_public_bt_anatomy,
)


class V14DKHardSupportExtractor(BaseStatic):
    """Per-event DK-hard-one-hot support tensor.

    Default output shape is ``(n_electrodes, K=80)`` in depth-wm.csv row order.
    Setting ``c_max`` pads to ``(c_max, K=80)`` with zero-rows so per-batch
    collation aligns alongside ``LogStftView`` and ``ElectrodeValidMask``.
    """

    event_types: tp.Literal["Ieeg"] = "Ieeg"
    bt_root: str
    unknown_label_policy: tp.Literal["raise", "skip"] = "raise"
    parcel_labels: tuple[str, ...] = V14_DK_PARCEL_LABELS
    c_max: int | None = None

    def get_static(self, event: Event) -> torch.Tensor:
        subject_id = _coerce_subject_id(getattr(event, "subject"))
        depth_wm_path = (
            Path(self.bt_root) / "localization" / f"sub_{subject_id}" / "depth-wm.csv"
        )
        if not depth_wm_path.exists():
            raise FileNotFoundError(f"depth-wm.csv not found at {depth_wm_path}")

        anatomy = load_public_bt_anatomy(self.bt_root, subject_id)

        if self.unknown_label_policy == "skip":
            mask = anatomy["DesikanKilliany"].isin(self.parcel_labels)
            anatomy = anatomy.loc[mask].reset_index(drop=True)

        electrode_labels = tuple(anatomy["Electrode"].tolist())
        result = build_hard_public_bt_label_support(
            electrode_labels, anatomy, self.parcel_labels,
        )
        support = torch.from_numpy(result.support)
        if self.c_max is not None:
            n_real = support.shape[0]
            if n_real > self.c_max:
                raise ValueError(
                    f"subject {subject_id} has {n_real} electrodes which exceeds c_max={self.c_max}"
                )
            padded = torch.zeros(self.c_max, support.shape[1], dtype=support.dtype)
            padded[:n_real] = support
            support = padded
        return support


_BTBANK_RE = re.compile(r"(?:btbank|sub_)?(\d+)$", re.IGNORECASE)


def _coerce_subject_id(subject: tp.Any) -> int:
    """Normalise an event.subject value to a BT integer subject id.

    Accepts ``int``, ``"7"``, ``"btbank7"``, ``"sub_7"``, and the
    NeuralSet study-qualified form ``"Wang2024Treebank/btbank7"``.
    """
    if isinstance(subject, int):
        return subject
    raw = str(subject).strip()
    tail = raw.rsplit("/", 1)[-1]  # strip "Wang2024Treebank/" study prefix
    match = _BTBANK_RE.match(tail)
    if match is None:
        raise ValueError(f"unrecognised BT subject identifier: {subject!r}")
    return int(match.group(1))
