"""V14 electrode valid-mask extractor.

NeuralSet's :class:`~neuralset.extractors.neuro.MneRaw` zero-pads its output
to the cohort channel-union dim but does NOT emit a sibling mask telling
downstream consumers which slots are real vs. padding. The v14 encoder
consumes ``valid_mask: (B, C) bool`` to set ``-inf`` cross-attn bias for
invalid slots (see ``V14ParcelPerceiverModel.forward``).

``ElectrodeValidMask`` reconstructs the mask aligned to the VOLTAGE electrode
order (``BrainTreebankSubject.electrode_labels`` — the same order the DK
support extractor and the front-end tokens use). Slot ``c`` is True iff voltage
electrode ``c`` is mapped to a DK parcel in the vocabulary (equivalently:
``support[c]`` is nonzero); voltage electrodes with no anatomy row or an
out-of-vocab label are False **at their true position** (not dropped), and the
trailing ``c_max - n_voltage`` padding slots are False. The encoder consumes
this as ``effective_support = support * valid_mask`` and ``drop_electrode =
~valid_mask``, so per-row alignment with ``support`` is load-bearing — a
re-packed or depth-wm-ordered mask misroutes every electrode (C1/C2 fix).

The ``unmapped_policy`` field mirrors :class:`V14DKHardSupportExtractor` so
filtering decisions stay aligned between mask + support + electrode-tokens.

Cohort C_MAX default = 384 (CQ12 / B14 lock 2026-05-23 PM: covers
D-cohort max 366, AJILE12 ~200, BT 256, SWEC 128 with 18-electrode
headroom past D-cohort max). Raises ValueError if any subject exceeds.
"""

from __future__ import annotations

import typing as tp

import torch
from neuralset.events.etypes import Event
from neuralset.extractors.base import BaseStatic

from speech_decoding.extractors.dk_support import _coerce_subject_id
from speech_decoding.studies.braintreebank.anatomy import (
    DEFAULT_BT_LABEL_COLUMN,
    V14_DK_PARCEL_LABELS,
    aligned_voltage_support,
    lite_voltage_mask,
)


class ElectrodeValidMask(BaseStatic):
    """Per-event ``(c_max,) bool`` valid-mask aligned to the voltage electrode order.

    ``electrode_set`` selects the electrode subset:

      * ``"all"`` (default) — every parcel-mapped voltage electrode is valid.
        Byte-identical to the pre-Lite extractor; exca cache-uid unchanged.
      * ``"lite"`` — additionally AND in the Neuroprobe-Lite electrode set
        (``lite_voltage_mask``), flipping non-Lite electrodes to ``valid=False``.
        The encoder's per-parcel pool then excludes them entirely (``drop_electrode
        = ~valid_mask`` → ``NEG_INF_MASK_VALUE`` cross-attn bias → exact-zero
        weight; ``v14_encoder.py`` forward). This reproduces upstream's Lite
        electrode subset as a *set* (the pool is permutation-invariant within a
        parcel), so P4 eval runs at 100% electrode parity with the Neuroprobe
        Lite leaderboard. BT-only — raises ``KeyError`` on a non-BT subject.
    """

    event_types: tp.Literal["Ieeg"] = "Ieeg"
    bt_root: str
    c_max: int = 384
    unmapped_policy: tp.Literal["raise", "zero"] = "raise"
    parcel_labels: tuple[str, ...] = V14_DK_PARCEL_LABELS
    # Atlas column partner of ``parcel_labels`` — see V14DKHardSupportExtractor.
    # Must match the support extractor (``_assert_support_valid_config_agree``).
    label_column: str = DEFAULT_BT_LABEL_COLUMN
    # Single-electrode-parcel drop (Ben 2026-06-13): the lone electrode's
    # ``valid`` flips to False here too, so the mask stays row-aligned with the
    # support extractor's zeroed row. Must match the support extractor.
    exclude_single_electrode_parcels: bool = False
    electrode_set: tp.Literal["all", "lite"] = "all"

    def get_static(self, event: Event) -> torch.Tensor:
        subject_id = _coerce_subject_id(getattr(event, "subject"))
        # ``aligned_voltage_support`` is memoized per subject and keyed on the
        # voltage electrode order; ``valid[c]`` is True iff voltage electrode
        # ``c`` was assigned a parcel. ``from_numpy`` is a zero-copy view over
        # its shared array; ``mask[:n] = valid`` copies, so the cached array is
        # never mutated. FileNotFoundError + "absent from parcel vocabulary"
        # KeyError flow through unchanged (lru_cache does not cache exceptions).
        valid = torch.from_numpy(
            aligned_voltage_support(
                self.bt_root,
                subject_id,
                parcel_labels=self.parcel_labels,
                unmapped_policy=self.unmapped_policy,
                label_column=self.label_column,
                exclude_single_electrode_parcels=self.exclude_single_electrode_parcels,
            ).valid
        )
        n_voltage = int(valid.shape[0])
        if n_voltage > self.c_max:
            raise ValueError(
                f"subject {subject_id} has {n_voltage} electrodes which exceeds c_max={self.c_max}"
            )

        if self.electrode_set == "lite":
            # AND in the Neuroprobe-Lite subset. ``lite_voltage_mask`` is over the
            # SAME voltage electrode order as ``valid`` (both from
            # ``voltage_electrode_order``), so the row alignment that
            # ``effective_support = support * valid_mask`` relies on is preserved.
            # A non-Lite electrode flips True→False; a Lite-but-unmapped electrode
            # stays False. ``&`` builds a new tensor — the lru_cached ``valid``
            # array is never mutated.
            lite = torch.from_numpy(lite_voltage_mask(self.bt_root, subject_id))
            valid = valid & lite

        mask = torch.zeros(self.c_max, dtype=torch.bool)
        mask[:n_voltage] = valid
        return mask
