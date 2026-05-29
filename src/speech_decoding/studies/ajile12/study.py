"""AJILE12 (Peterson 2022) NeuralFetch-style Study scaffold (SCAFFOLD-05).

Per ``docs/neuroprobe/v14_implementation_fix_list.md`` (SCAFFOLD-05) and
``docs/neuroprobe/v14_blockers.md`` DP03 + the cross-corpus pretraining
strategy memo (``project_v14_cross_subject_pretraining_data_strategy_2026_05_22``),
AJILE12 is an ECoG-heavy upper-limb-movement BCI cohort. The v14 cross-
corpus pretraining strategy treats it as a **Phase-1-only** SSL
contributor because the corpus is 89.7% ECoG per Peterson 2022 Table 2
— including it in Phase-2 would dilute the sEEG cohort that the
anatomy-ON pretrain depends on.

DP03 itself is not yet ratified, so this module is a structural
scaffold: class declaration + ClassVars. Every method that would commit
to a contract decision raises :class:`NotImplementedError`.

Audited facts (ClassVars sourced from Peterson 2022 + the
cross-subject-pretraining strategy memo):

  * 12 patients, ~50 days of recording, motor BCI task.
  * 1000 Hz acquisition → 500 Hz Nyquist → v14 30-bin log ⅓-octave
    filterbank trainable bin range = ``k0..k20`` (inclusive).
  * Mains notch = 60 Hz (US site).
  * 89.7% ECoG electrode share (Peterson 2022 Table 2).
  * Phase-1 corpus contributor only (no Phase-2 use).

Cited blockers (text below):

  * DP03 — subpackage path + contract mirror not yet ratified.
  * DP02 — corpus-balanced sampler weights (the 50-day AJILE12 tail
    is balanced against the BT 6-hour + D-cohort 224-hour tails via
    Phase-1 hierarchical sampling).
  * DP06 — session-stratified clip extraction.
  * M07  — Phase-1/Phase-2 train/val split policy.
  * M18  — clip extraction strategy.
"""

from __future__ import annotations

import typing as tp

import mne
import pandas as pd
from neuralset.events import study


_BLOCKER_MSG = (
    "AJILE12Study is a SCAFFOLD-05 scaffold gated on "
    "docs/neuroprobe/v14_blockers.md DP03 (studies/ajile12/ contract not "
    "yet ratified). Cited blockers: DP03, DP02 (sampler weights), "
    "DP06 (clip extraction), M07 (SSL split), M18 (clip strategy)."
)


class AJILE12Study(study.Study):
    """AJILE12 (Peterson 2022) ECoG-heavy motor BCI cohort.

    Phase-1-only Phase-1 SSL contributor under v14's cross-corpus
    pretraining strategy. The class declaration exists so dispatch can
    reference it; all data-loading methods raise until DP03 ratifies."""

    aliases: tp.ClassVar[tuple[str, ...]] = (
        "AJILE12", "AJILE-12", "Peterson2022",
    )
    bibtex: tp.ClassVar[str] = """
    @article{peterson2022ajile12,
        title = {AJILE12: Long-term naturalistic human intracranial neural
                 recordings and pose},
        author = {Peterson, Steven M. and others},
        journal = {Scientific Data},
        year = {2022},
        doi = {10.1038/s41597-022-01280-y}
    }
    """
    url: tp.ClassVar[str] = "https://doi.org/10.6080/K0KW5D4V"
    licence: tp.ClassVar[str] = "See DANDI/Open-Data terms (CC BY 4.0 per dandiset)."
    description: tp.ClassVar[str] = (
        "Long-term naturalistic human intracranial recordings + pose tracking "
        "(Peterson 2022). 12 patients, ~50 days of recording, 1000 Hz "
        "acquisition, 89.7% ECoG electrode share. Phase-1-only SSL "
        "contributor in v14."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ()
    _info: tp.ClassVar[study.StudyInfo | None] = None

    PHASE_SCOPE: tp.ClassVar[tuple[str, ...]] = ("p1",)
    SAMPLE_RATE_HZ: tp.ClassVar[float] = 1000.0
    MAINS_NOTCH_HZ: tp.ClassVar[float] = 60.0
    # Half-open ``(start, stop)`` of the trainable v14 30-bin filterbank.
    # 500 Hz Nyquist → bins k0..k20 inclusive → ``(0, 21)``.
    VALID_BIN_RANGE: tp.ClassVar[tuple[int, int]] = (0, 21)
    ECOG_FRACTION: tp.ClassVar[float] = 0.897
    N_UNIQUE_PATIENTS: tp.ClassVar[int] = 12

    def _download(self) -> None:
        raise NotImplementedError(_BLOCKER_MSG)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        raise NotImplementedError(_BLOCKER_MSG)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        raise NotImplementedError(_BLOCKER_MSG)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        raise NotImplementedError(_BLOCKER_MSG)
