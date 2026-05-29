"""SCAFFOLD-02 (NT02): ``V14Data`` — phase-parametrized corpus mix and
sampler spec for the v14 cross-subject SSL stack.

Per ``docs/neuroprobe/v14_implementation_fix_list.md`` (SCAFFOLD-02) and
``docs/neuroprobe/v14_blockers.md`` NT02, the v14 training program runs
through five phases (P1, P2, P3a, P3b, P4) where the contributing corpora
and the sampler differ:

* **P1** — anatomy-OFF SSL pretrain across 4 corpora (SWEC + AJILE12 +
  D-cohort + BT); hierarchical α=0.5 sampler weights examples by
  ``count^α`` so the 6 672-h SWEC tail does not dominate.
* **P2** — anatomy-ON SSL pretrain on 224 h sEEG (D-cohort + BT). The
  ECoG-heavy AJILE12 and anatomy-blind SWEC drop out; the sampler is
  uniform-per-subject so the 85 D-cohort subjects + 9 BT subjects each
  contribute equal expected weight to a step.
* **P3a / P3b** — Whisper-L8 distillation, BT-only paired iEEG+audio.
* **P4** — downstream linear probe, BT-only supervised.

This module is structural: the resolvers + the pydantic discriminator
are stable; the actual DataLoader construction still runs through the
parent ``Data.build()`` path, which is downstream of the SWEC / DCohort
/ AJILE12 Study scaffolds and the variable-T/C collate (SCAFFOLD-06).
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import pydantic

from speech_decoding.experiments.data import Data
from speech_decoding.experiments.v14_experiment import V14Phase


V14Corpus = Literal["swec", "ajile12", "dcohort", "bt"]


SamplerKind = Literal[
    "hierarchical",            # P1: α-weighted mixture over corpora
    "uniform_per_subject",     # P2: equal expected weight per subject
    "paired_ieeg_audio",       # P3a/P3b: paired with audio for distillation
    "supervised",              # P4: standard per-event probe sampling
]


@dataclasses.dataclass(frozen=True, slots=True)
class SamplerSpec:
    """Per-phase sampler descriptor.

    Frozen — once the phase resolver returns a spec, downstream code must
    not mutate it.

    Attributes
    ----------
    kind
        Sampler family for this phase.
    alpha
        Hierarchical α (only meaningful for ``kind=="hierarchical"``).
        ``None`` for all other kinds.
    paired_audio
        ``True`` for P3 (Whisper distillation needs paired iEEG+audio).
    """

    kind: SamplerKind
    alpha: float | None
    paired_audio: bool


def v14_corpus_mix_for_phase(phase: V14Phase) -> tuple[V14Corpus, ...]:
    """Per-phase corpus mix tuple."""
    if phase == 1:
        return ("swec", "ajile12", "dcohort", "bt")
    if phase == 2:
        return ("dcohort", "bt")
    if phase in ("3a", "3b", 4):
        return ("bt",)
    raise ValueError(f"unknown phase: {phase!r}")


def v14_sampler_spec_for_phase(phase: V14Phase) -> SamplerSpec:
    """Per-phase sampler descriptor."""
    if phase == 1:
        return SamplerSpec(kind="hierarchical", alpha=0.5, paired_audio=False)
    if phase == 2:
        return SamplerSpec(kind="uniform_per_subject", alpha=None, paired_audio=False)
    if phase in ("3a", "3b"):
        return SamplerSpec(kind="paired_ieeg_audio", alpha=None, paired_audio=True)
    if phase == 4:
        return SamplerSpec(kind="supervised", alpha=None, paired_audio=False)
    raise ValueError(f"unknown phase: {phase!r}")


class V14Data(Data):
    """Phase-parametrized Data scaffold.

    Adds ``phase``, optional ``corpus_mix_override``, and
    ``sampler_spec_override``. ``resolved_corpus_mix()`` /
    ``resolved_sampler_spec()`` fall back to the phase resolvers when no
    override is supplied — single source of truth via
    :func:`v14_corpus_mix_for_phase` / :func:`v14_sampler_spec_for_phase`.

    The base :class:`Data.build` path is unchanged; downstream phases
    pass the resolved mix and sampler into the multi-Study orchestration
    layer (SCAFFOLD-03/04/05 + SCAFFOLD-06) when those land.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    phase: V14Phase
    corpus_mix_override: tuple[V14Corpus, ...] | None = None
    sampler_spec_override: SamplerSpec | None = None

    def resolved_corpus_mix(self) -> tuple[V14Corpus, ...]:
        if self.corpus_mix_override is not None:
            return self.corpus_mix_override
        return v14_corpus_mix_for_phase(self.phase)

    def resolved_sampler_spec(self) -> SamplerSpec:
        if self.sampler_spec_override is not None:
            return self.sampler_spec_override
        return v14_sampler_spec_for_phase(self.phase)
