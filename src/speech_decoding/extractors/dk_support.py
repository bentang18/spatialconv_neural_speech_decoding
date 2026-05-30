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

import functools
import re
import typing as tp

import torch
from neuralset.events.etypes import Event
from neuralset.extractors.base import BaseStatic

from speech_decoding.studies.braintreebank.anatomy import (
    V14_DK_PARCEL_LABELS,
    build_hard_public_bt_label_support,
    load_public_bt_anatomy,
)


@functools.lru_cache(maxsize=64)
def _cached_hard_support(
    bt_root: str,
    subject_id: int,
    unknown_label_policy: str,
    parcel_labels: tuple[str, ...],
) -> torch.Tensor:
    """Per-subject DK-hard one-hot support ``(n_real, K)``, memoized.

    ``load_public_bt_anatomy`` re-reads + re-parses ``depth-wm.csv`` on every
    call (~5 ms); the value is static per subject, so the un-memoized version
    was ~38% of warm-cache ``__getitem__`` cost. ``lru_cache`` does not cache
    exceptions, so the FileNotFoundError / KeyError paths are preserved.

    The returned tensor is shared across callers; ``get_static`` always returns
    a copy (a fresh padded buffer when ``c_max`` is set, else ``.clone()``), so
    callers never mutate the cached array.
    """
    anatomy = load_public_bt_anatomy(bt_root, subject_id)
    if unknown_label_policy == "skip":
        mask = anatomy["DesikanKilliany"].isin(parcel_labels)
        anatomy = anatomy.loc[mask].reset_index(drop=True)
    electrode_labels = tuple(anatomy["Electrode"].tolist())
    result = build_hard_public_bt_label_support(
        electrode_labels, anatomy, parcel_labels,
    )
    return torch.from_numpy(result.support)


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
        # Memoized per-subject lookup: load_public_bt_anatomy re-reads
        # depth-wm.csv on every call, which is static per subject. The
        # FileNotFoundError ("...depth-wm.csv") and KeyError ("absent from
        # parcel vocabulary") paths are preserved inside the cached helper.
        support = _cached_hard_support(
            self.bt_root, subject_id, self.unknown_label_policy, self.parcel_labels,
        )
        if self.c_max is not None:
            n_real = support.shape[0]
            if n_real > self.c_max:
                raise ValueError(
                    f"subject {subject_id} has {n_real} electrodes which exceeds c_max={self.c_max}"
                )
            padded = torch.zeros(self.c_max, support.shape[1], dtype=support.dtype)
            padded[:n_real] = support
            return padded
        # c_max is None: never hit by the live pipeline (dispatch pins
        # c_max=384), but clone so callers can't mutate the shared cached
        # tensor in place and poison _cached_hard_support for the subject.
        return support.clone()


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
