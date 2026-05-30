"""SWEC-iEEG NeuralFetch-style Study (T3.4 scaffold).

Per ``docs/neuroprobe/v14_blockers.md`` DP03 the decision-*candidate* is to
implement ``studies/swec/`` as a Study subclass mirroring
``Wang2024Treebank`` but *anatomy-blind* — SWEC ships zero anatomy metadata
and is band-pass filtered to 0.5-120 Hz on acquisition.

DP03 itself is not yet ratified, so this module is a structural scaffold:
class declaration + ClassVars sourced from the 2026-05-19 SWEC audit memo
(``reference_swec_ieeg_dataset_audit_2026_05_19.md``). Every method that
would commit to a contract decision raises :class:`NotImplementedError`
citing the gating blocker IDs.

Cited blockers (text below):

  - DP03 — subpackage path + contract mirror not yet ratified.
  - M21  — SWEC anatomy confirmation (anatomy-blind vs surface-MNI fallback).
  - DP02 — corpus-balanced sampler weights.
  - DP06 — session-stratified clip extraction.
  - M07  — Phase-1/Phase-2 train/val split policy.
  - M18  — clip extraction strategy.

Audited facts (from the 5/19 SWEC audit memo, safe to encode here):

  - 50 unique patients, 6 672 h total, 460 annotated events.
  - Hardware band-pass: 0.5-120 Hz (constrains trainable filterbank bins
    to k0-k21 of the v14 30-bin log ⅓-octave filterbank).
  - Zero anatomy metadata (no MNI/DK lookup).
"""

from __future__ import annotations

import typing as tp

import mne
import pandas as pd
from neuralset.events import study


_BLOCKER_MSG = (
    "SWECStudy is a T3.4 scaffold gated on docs/neuroprobe/v14_blockers.md "
    "DP03 (studies/swec/ contract not yet ratified). Cited blockers: "
    "DP03, M21 (anatomy), DP02 (sampler weights), DP06 (clip extraction), "
    "M07 (SSL split), M18 (clip strategy)."
)


class SWECStudy(study.Study):
    """SWEC-iEEG (Bern dataset). Anatomy-blind, 0.5-120 Hz band-passed.

    Joint-SSL corpus contributor under v14's cross-corpus pretraining
    strategy (`memory/project_v14_cross_subject_pretraining_data_strategy_2026_05_22.md`).
    Front-end-only: anatomy-blind, so under B30 its ``latent_valid`` is
    all-False (zero slot-axis contribution; front-end ``L_pre_frame_masked``
    only). The class declaration exists so dispatch can reference it; all
    data-loading methods raise until DP03 (subpackage contract) lands."""

    aliases: tp.ClassVar[tuple[str, ...]] = ("SWEC", "SWEC-iEEG", "SWEC_ETHZ")
    bibtex: tp.ClassVar[str] = """
    @misc{ictal-swec_ieeg,
        title = {SWEC iEEG dataset},
        author = {Inselspital Bern},
        url = {http://ieeg-swez.ethz.ch/}
    }
    """
    url: tp.ClassVar[str] = "http://ieeg-swez.ethz.ch/"
    licence: tp.ClassVar[str] = "See http://ieeg-swez.ethz.ch/ for terms."
    description: tp.ClassVar[str] = (
        "iEEG recordings from 50 unique patients (6 672 h total, 460 annotated "
        "events) at Inselspital Bern. Hardware band-pass 0.5-120 Hz; zero "
        "anatomy metadata (anatomy-blind in v14)."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ()
    _info: tp.ClassVar[study.StudyInfo | None] = None

    HARDWARE_BANDPASS_HZ: tp.ClassVar[tuple[float, float]] = (0.5, 120.0)
    N_UNIQUE_PATIENTS: tp.ClassVar[int] = 50
    TOTAL_HOURS: tp.ClassVar[float] = 6672.0
    N_EVENTS: tp.ClassVar[int] = 460
    # CH (Inselspital Bern) mains frequency. v14 dispatch's notch_filter
    # is currently hardcoded to 60 Hz in dispatch_v14.py — must lift to
    # per-corpus before SWEC pretrain (DOC-FIX-08 / MASK-01).
    MAINS_NOTCH_HZ: tp.ClassVar[float] = 50.0
    # Half-open ``(start, stop)`` of the trainable v14 30-bin filterbank.
    # SWEC hardware band-pass 0.5–120 Hz → bins k0..k21 inclusive →
    # ``(0, 22)`` (per the 5/19 SWEC audit + MEMORY.md B1 closure).
    VALID_BIN_RANGE: tp.ClassVar[tuple[int, int]] = (0, 22)

    def _download(self) -> None:
        raise NotImplementedError(_BLOCKER_MSG)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        raise NotImplementedError(_BLOCKER_MSG)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        raise NotImplementedError(_BLOCKER_MSG)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        raise NotImplementedError(_BLOCKER_MSG)
