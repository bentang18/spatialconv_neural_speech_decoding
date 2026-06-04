"""D-cohort (Cogan sEEG) NeuralFetch-style Study (T3.5 scaffold).

Per ``docs/neuroprobe/v14_blockers.md`` DP03 the decision-*candidate* is to
implement ``studies/cogan_dcohort/`` as a Study subclass mirroring
``Wang2024Treebank`` and anatomy-bearing via native FS recons (DK
``aparc+aseg`` is present directly in the recon CSVs). The D-cohort is the Cogan-lab Duke sEEG cohort
referenced in ``CLAUDE.md §Key Files`` (``D<num>`` = Stage-3 scope sEEG
patients, distinct from ``S<num>`` PS uECoG).

DP03 itself is not yet ratified, so this module is a structural scaffold:
class declaration + ClassVars. Every method that would commit to a
contract decision raises :class:`NotImplementedError` citing the gating
blocker IDs.

Cited blockers (text below):

  - DP03 — subpackage path + contract mirror not yet ratified.
  - DP05 — per-subject DK-index-map serialization + load-once cache.
  - DP06 — session-stratified clip extraction with respected boundaries.
  - M07  — Phase-1/Phase-2 train/val split policy.
  - M18  — clip extraction strategy.
"""

from __future__ import annotations

import typing as tp

import mne
import pandas as pd
from neuralset.events import study


_BLOCKER_MSG = (
    "DCohortStudy is a T3.5 scaffold gated on docs/neuroprobe/v14_blockers.md "
    "DP03 (studies/cogan_dcohort/ contract not yet ratified). Cited blockers: "
    "DP03, DP05 (DK-index map), DP06 (clip extraction), "
    "M07 (SSL split), M18 (clip strategy)."
)


class DCohortStudy(study.Study):
    """Cogan-lab Duke sEEG D-cohort (D<num> patients).

    Anatomy-bearing under v14: per-subject DK parcel routing via native
    FreeSurfer recons (DK ``aparc+aseg`` is present directly in the recon
    CSVs — no Pipeline C derivation needed). The class declaration exists
    so dispatch can reference it; all data-loading methods raise until DP03
    (subpackage contract) lands.

    NOTE: D-cohort native rate is MIXED (mostly 2048 Hz; also 2000 / 1024 /
    1000 Hz across runs) — NOT a flat 2000 Hz. Loader reads the per-run rate
    and resamples to the v14 canonical 2048 Hz; after resample all 30
    Multi-STFT bins are valid. Inventory: ``memory/project_d_cohort_data_inventory_2026_06_03.md``.
    """

    aliases: tp.ClassVar[tuple[str, ...]] = (
        "DCohort", "Cogan_DCohort", "Cogan-Duke-sEEG",
    )
    bibtex: tp.ClassVar[str] = ""  # internal cohort; cite Cogan-lab publications.
    url: tp.ClassVar[str] = ""
    licence: tp.ClassVar[str] = "Internal Cogan-lab data; IRB-restricted."
    description: tp.ClassVar[str] = (
        "Cogan-lab Duke sEEG D-cohort (D<num> patients). Anatomy-bearing via "
        "native FS recons (DK aparc+aseg present in recon CSVs). Distinct from "
        "the PS S<num> uECoG cohort."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ()
    _info: tp.ClassVar[study.StudyInfo | None] = None

    # v14 canonical RESAMPLE TARGET. Native rate is mixed (mostly 2048; also
    # 2000 / 1024 / 1000 across runs); loader resamples per-run to this.
    SAMPLE_RATE_HZ: tp.ClassVar[float] = 2048.0
    # US site (Duke); v14 dispatch's 60 Hz notch already correct.
    MAINS_NOTCH_HZ: tp.ClassVar[float] = 60.0
    # Half-open ``(start, stop)`` of the trainable v14 30-bin filterbank.
    # After resample to 2048 Hz → all 30 bins valid → ``(0, 30)``.
    VALID_BIN_RANGE: tp.ClassVar[tuple[int, int]] = (0, 30)
    # Speech-subset FS gate (project_d_cohort_data_inventory_2026_06_03):
    # 87 / 87 D-pts pass — D107/D139 earlier mis-flagged (recons live under
    # suffix variants D107A/B, D139A/B). Full inventory has 113 task subjects,
    # 134 recon-localized.
    N_UNIQUE_PATIENTS: tp.ClassVar[int] = 87

    def _download(self) -> None:
        raise NotImplementedError(_BLOCKER_MSG)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        raise NotImplementedError(_BLOCKER_MSG)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        raise NotImplementedError(_BLOCKER_MSG)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        raise NotImplementedError(_BLOCKER_MSG)
