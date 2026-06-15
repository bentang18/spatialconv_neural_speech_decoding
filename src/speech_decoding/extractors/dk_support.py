"""V14 DK-hard-one-hot support extractor.

Reads BT-shipped ``localization/sub_<id>/depth-wm.csv`` and emits per-event
``(n_electrodes, K=80)`` one-hot support over the canonical v14 DK parcel
vocabulary (``V14_DK_PARCEL_LABELS``). Consumed by the v14 encoder's hard
block-diagonal per-parcel pool: the one-hot assignment IS the routing — a
parcel-slot attends ONLY to its own parcel's electrodes (B36 2026-06-01
replaced the soft ``log(support + eps)`` Graphormer bias).

Row order matches the VOLTAGE electrode order — exactly
``BrainTreebankSubject.electrode_labels`` (the cleaned, corrupted/trigger/
missing-coord-filtered ``electrode_labels.json`` order), which is the channel
order the loader feeds the front-end. This is NOT the same as ``depth-wm.csv``
row order: support row ``c`` therefore lines up with ``electrode_tokens[c]``
(C1/C2 fix). See ``anatomy.voltage_electrode_order``.

Labels falling outside the K=80 vocabulary (e.g. BT btbank4 has
``Left-Inf-Lat-Vent`` ventricle electrodes), and voltage electrodes with no
``depth-wm.csv`` row, raise by default; cohort-loading layers can pass
``unmapped_policy="zero"`` to emit a zero support row + ``valid=False`` for
those electrodes **in place** (no re-pack — positions stay aligned).
"""

from __future__ import annotations

import re
import typing as tp

import torch
from neuralset.events.etypes import Event
from neuralset.extractors.base import BaseStatic

from speech_decoding.studies.braintreebank.anatomy import (
    DEFAULT_BT_LABEL_COLUMN,
    V14_DK_PARCEL_LABELS,
    aligned_voltage_support,
)


class V14DKHardSupportExtractor(BaseStatic):
    """Per-event DK-hard-one-hot support tensor.

    Default output shape is ``(n_voltage, K=80)`` in VOLTAGE electrode order
    (``BrainTreebankSubject.electrode_labels``). Setting ``c_max`` pads to
    ``(c_max, K=80)`` with zero-rows so per-batch collation aligns alongside
    ``MultiStftView`` and ``ElectrodeValidMask``.
    """

    event_types: tp.Literal["Ieeg"] = "Ieeg"
    bt_root: str
    unmapped_policy: tp.Literal["raise", "zero"] = "raise"
    parcel_labels: tuple[str, ...] = V14_DK_PARCEL_LABELS
    # Atlas column in depth-wm.csv: "DesikanKilliany" (DK, K=80) or "DKT" (K=74).
    # MUST be the partner of ``parcel_labels`` (dispatch sets both from
    # ``anatomy.atlas_spec`` so they cannot desync).
    label_column: str = DEFAULT_BT_LABEL_COLUMN
    # Ben 2026-06-13: drop parcels covered by exactly one valid electrode — the
    # lone electrode is zeroed + marked invalid (degenerate within-parcel σ poisons
    # the heteroscedastic M4 weight). Global K unchanged. Must match the valid-mask
    # extractor (enforced by ``_assert_support_valid_config_agree``).
    exclude_single_electrode_parcels: bool = False
    c_max: int | None = None
    # Montage selector — must match the loader (study.electrode_set) and the
    # valid-mask extractor (enforced by ``_assert_support_valid_config_agree``).
    # "lite" aligns support to ``lite_voltage_order`` so row ``c`` of support
    # describes the same Lite electrode as Lite voltage row ``c``. As a pydantic
    # field it is part of the extractor uid → a distinct exca cache for "lite".
    electrode_set: tp.Literal["all", "lite"] = "all"

    def get_static(self, event: Event) -> torch.Tensor:
        subject_id = _coerce_subject_id(getattr(event, "subject"))
        # ``aligned_voltage_support`` is memoized per subject and keyed on the
        # voltage electrode order; ``from_numpy`` is a cheap zero-copy view over
        # its shared array. The FileNotFoundError ("...depth-wm.csv" /
        # "...electrode_labels") and KeyError ("absent from parcel vocabulary")
        # paths flow through unchanged (lru_cache does not cache exceptions).
        support = torch.from_numpy(
            aligned_voltage_support(
                self.bt_root,
                subject_id,
                parcel_labels=self.parcel_labels,
                unmapped_policy=self.unmapped_policy,
                label_column=self.label_column,
                exclude_single_electrode_parcels=self.exclude_single_electrode_parcels,
                electrode_set=self.electrode_set,
            ).support
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
        # c_max=384), but clone so callers can't mutate the shared cached array.
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
