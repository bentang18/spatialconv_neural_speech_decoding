"""V14 electrode native-coordinate extractor.

Emits per-event ``(c_max, 3)`` float32 NATIVE subject-space ``depth-wm.csv``
``(L, I, P)`` coordinates (~1 mm-isotropic voxels), aligned to the VOLTAGE
electrode order (``BrainTreebankSubject.electrode_labels`` — the same order the
DK support / valid-mask extractors and the front-end tokens use). Row ``c`` is
the coordinate of the same physical electrode as ``support[c]`` / ``valid[c]`` /
``electrode_tokens[c]``, so it collates row-for-row alongside them.

Native subject space is deliberate (NOT the fsaverage ``elec_coords_full.csv``):
fsaverage pial registration projects depth contacts onto the surface and
scrambles along-shaft geometry, whereas the relational positional encoding needs
the most accurate INTER-electrode distances (Ben 2026-07-01 — "stay in subject
space where the coordinates are most relationally accurate").

Unlike :class:`V14DKHardSupportExtractor` / :class:`ElectrodeValidMask`, this
extractor is PARCEL-INDEPENDENT: an electrode has a coordinate regardless of
whether it maps to a DK parcel or is a lone-electrode-parcel drop. So there is no
``parcel_labels`` / ``unmapped_policy`` / ``label_column`` /
``exclude_single_electrode_parcels`` field — coords are invariant to all of them.
The row identity is fixed solely by ``electrode_set`` + the per-session trial, so
only those thread through (they must match the loader + support/valid extractors).
Trailing ``c_max - n_voltage`` padding rows are zeros; downstream drops them via
the valid-mask ``keep`` set before any coordinate is used.

Cohort C_MAX default = 384 — matches :class:`ElectrodeValidMask`. Raises
ValueError if any subject exceeds it.
"""

from __future__ import annotations

import typing as tp

import torch
from neuralset.events.etypes import Event
from neuralset.extractors.base import BaseStatic

from speech_decoding.extractors.dk_support import _coerce_subject_id, _coerce_trial_id
from speech_decoding.studies.braintreebank.anatomy import aligned_voltage_coords


class V14ElectrodeCoordsExtractor(BaseStatic):
    """Per-event ``(c_max, 3)`` float32 native ``(L, I, P)`` coords in voltage order."""

    event_types: tp.Literal["Ieeg"] = "Ieeg"
    bt_root: str
    c_max: int = 384
    # Montage selector — must match the loader (study.electrode_set), the DK
    # support extractor, and the valid-mask extractor (all four rebuild over the
    # SAME electrode order). "lite" aligns coords to ``lite_voltage_order``. As a
    # pydantic field it is part of the extractor uid → a distinct exca cache.
    electrode_set: tp.Literal["all", "lite"] = "all"

    def get_static(self, event: Event) -> torch.Tensor:
        subject_id = _coerce_subject_id(getattr(event, "subject"))
        trial_id = _coerce_trial_id(event)
        # ``aligned_voltage_coords`` is memoized per (subject, trial, electrode_set)
        # and keyed on the voltage electrode order; ``trial_id`` threads the
        # per-session STATIC drop so coords row ``c`` aligns to the front-end
        # voltage row ``c`` FOR THIS session (DP4). ``from_numpy`` is a zero-copy
        # view over its shared array; ``coords[:n] = ...`` copies, so the cached
        # array is never mutated. FileNotFoundError ("...depth-wm.csv"), the
        # missing-column KeyError, and the row-desync ValueError flow through
        # unchanged (lru_cache does not cache exceptions).
        native = torch.from_numpy(
            aligned_voltage_coords(
                self.bt_root,
                subject_id,
                trial_id=trial_id,
                electrode_set=self.electrode_set,
            )
        )
        n_voltage = int(native.shape[0])
        if n_voltage > self.c_max:
            raise ValueError(
                f"subject {subject_id} has {n_voltage} electrodes which exceeds c_max={self.c_max}"
            )
        coords = torch.zeros(self.c_max, 3, dtype=native.dtype)
        coords[:n_voltage] = native
        return coords
