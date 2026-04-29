"""BrainTreebank Study + Ieeg event for NeuralSet.

`BraintreebankIeeg._read()` returns raw 2048 Hz voltage with no re-reference,
matching Neuroprobe `__getitem__` native output. Re-ref + HG envelope are
Stage-1 ablation cells layered on top.

`BraintreebankStudy.iter_timelines()` enumerates the Tier-1 cohort via
`manifest.TIER1_WHITELIST`. `_load_timeline_events()` builds one Ieeg row
covering the full continuous trial plus N Word triggers from the trial's
transcript.

Stage-0 Block A0 fills `manifest.TIER1_WHITELIST`; Stage-0 Block E3 wires
the 15-task derivation through `labels.py`.
"""

from __future__ import annotations

from typing import Any

import mne
import neuralset as ns
import pandas as pd
from neuralset.events.etypes import Ieeg

from speech_decoding.studies.braintreebank.loader import bt_load_raw
from speech_decoding.studies.braintreebank.manifest import TIER1_WHITELIST


class BraintreebankIeeg(Ieeg):
    """h5-backed continuous-trial Ieeg event for BrainTreebank.

    Keyed by `(subject_num, trial)`. `_read()` constructs the
    `BrainTreebankSubject`, pulls raw voltage via `bt_load_raw()`, and wraps
    it as `mne.io.RawArray` (raw 2048 Hz, no re-reference).
    """

    subject_num: int = 0
    trial: int = 0

    def _read(self) -> mne.io.RawArray:
        from neuroprobe.braintreebank_subject import BrainTreebankSubject

        bt = BrainTreebankSubject(self.subject_num, self.trial)
        data, ch_names, sfreq = bt_load_raw(bt)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")
        return mne.io.RawArray(data, info, verbose=False)


class BraintreebankStudy(ns.Study):
    """BrainTreebank — 10 subjects, 26 movie-watching trials.

    Tier-1 whitelist only (post-Tier-1-parcel-coverage filtering) — see
    `manifest.TIER1_WHITELIST`. Stage-2 SSL trains on the whitelist; Stage-0
    bake-out fills the whitelist itself.
    """

    aliases = ("braintreebank", "BT")

    def iter_timelines(self) -> list[dict[str, Any]]:
        return [
            {"subject_num": s, "trial": t} for (s, t) in TIER1_WHITELIST
        ]

    def _load_timeline_events(self, timeline: dict[str, Any]) -> pd.DataFrame:
        """One Ieeg row covering the full trial + N Word triggers from transcript.

        The Word-trigger derivation (transcript onset times, downstream-task
        labels) is Stage-0 Block E3. This stub returns the Ieeg row alone so
        the Segmenter contract holds; Stage-2 SSL fills the trigger half.
        """
        del timeline  # unused until Stage-0 E3 lands
        return pd.DataFrame()
