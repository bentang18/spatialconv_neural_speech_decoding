"""V14 electrode valid-mask extractor.

NeuralSet's :class:`~neuralset.extractors.neuro.MneRaw` zero-pads its output
to the cohort channel-union dim but does NOT emit a sibling mask telling
downstream consumers which slots are real vs. padding. The v14 encoder
consumes ``valid_mask: (B, C) bool`` to set ``-inf`` cross-attn bias for
invalid slots (see ``V14ParcelPerceiverModel.forward``).

``ElectrodeValidMask`` reconstructs the mask from BT-shipped
``localization/sub_<id>/depth-wm.csv`` (same source as the DK support
extractor), so the first ``n_electrodes`` slots of the cohort-wide C_MAX
tensor are True and the rest are False. The ``unknown_label_policy`` field
mirrors :class:`V14DKHardSupportExtractor` so that filtering decisions stay
aligned between mask + support + electrode-tokens extractors.

Cohort C_MAX default = 384 (CQ12 / B14 lock 2026-05-23 PM: covers
D-cohort max 366, AJILE12 ~200, BT 256, SWEC 128 with 18-electrode
headroom past D-cohort max). Raises ValueError if any subject exceeds.
"""

from __future__ import annotations

import functools
import typing as tp

import torch
from neuralset.events.etypes import Event
from neuralset.extractors.base import BaseStatic

from speech_decoding.extractors.dk_support import _coerce_subject_id
from speech_decoding.studies.braintreebank.anatomy import (
    V14_DK_PARCEL_LABELS,
    load_public_bt_anatomy,
)


@functools.lru_cache(maxsize=64)
def _cached_n_real(
    bt_root: str,
    subject_id: int,
    unknown_label_policy: str,
    parcel_labels: tuple[str, ...],
) -> int:
    """Per-subject real-electrode count after unknown-label policy, memoized.

    ``load_public_bt_anatomy`` re-reads ``depth-wm.csv`` on every call; the
    count is static per subject, so the un-memoized version was ~31% of
    warm-cache ``__getitem__`` cost. ``lru_cache`` does not cache exceptions,
    so the FileNotFoundError ("...depth-wm.csv") and the "absent from parcel
    vocabulary" KeyError raised under ``policy='raise'`` are both preserved.
    The ``c_max`` ValueError is kept in ``get_static`` (out of the cache key)
    because ``c_max`` varies per extractor instance.
    """
    anatomy = load_public_bt_anatomy(bt_root, subject_id)
    if unknown_label_policy == "skip":
        keep = anatomy["DesikanKilliany"].isin(parcel_labels)
        anatomy = anatomy.loc[keep].reset_index(drop=True)
    else:
        unknown = sorted(set(anatomy["DesikanKilliany"]) - set(parcel_labels))
        if unknown:
            raise KeyError(
                "BT anatomy labels absent from parcel vocabulary: "
                f"{unknown[:10]}"
                + (f" (+{len(unknown) - 10} more)" if len(unknown) > 10 else "")
            )
    return len(anatomy)


class ElectrodeValidMask(BaseStatic):
    """Per-event ``(c_max,) bool`` valid-mask aligned with depth-wm.csv row order."""

    event_types: tp.Literal["Ieeg"] = "Ieeg"
    bt_root: str
    c_max: int = 384
    unknown_label_policy: tp.Literal["raise", "skip"] = "raise"
    parcel_labels: tuple[str, ...] = V14_DK_PARCEL_LABELS

    def get_static(self, event: Event) -> torch.Tensor:
        subject_id = _coerce_subject_id(getattr(event, "subject"))
        # Memoized per-subject electrode count (after unknown-label policy):
        # load_public_bt_anatomy re-reads depth-wm.csv per call for a value
        # that is static per subject. FileNotFoundError + "absent from parcel
        # vocabulary" KeyError are preserved inside the cached helper.
        n_real = _cached_n_real(
            self.bt_root, subject_id, self.unknown_label_policy, self.parcel_labels,
        )
        if n_real > self.c_max:
            raise ValueError(
                f"subject {subject_id} has {n_real} electrodes which exceeds c_max={self.c_max}"
            )

        mask = torch.zeros(self.c_max, dtype=torch.bool)
        mask[:n_real] = True
        return mask
