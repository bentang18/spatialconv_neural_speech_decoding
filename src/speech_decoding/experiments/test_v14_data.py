"""SCAFFOLD-02 tests: ``V14Data`` phase + corpus_mix + sampler_spec.

Per fix-list item SCAFFOLD-02 / NT02, ``V14Data`` parametrizes the active
phase, the corpus mix that contributes to it, and the sampler spec
(α=0.5 hierarchical for P1, uniform-per-subject for P2, paired iEEG+audio
for P3).

These tests pin the *structural* contract — actual DataLoader construction
still runs through the parent ``Data.build()`` path; only the phase-gated
corpus / sampler resolution is unit-tested here.
"""

from __future__ import annotations

import pytest

from speech_decoding.experiments.v14_data import (
    SamplerSpec,
    V14Data,
    v14_corpus_mix_for_phase,
    v14_sampler_spec_for_phase,
)


# ---------- per-phase corpus mix ----------------------------------------


def test_corpus_mix_p1_includes_all_four_corpora() -> None:
    """P1: SWEC + AJILE12 + D-cohort + BT."""
    mix = v14_corpus_mix_for_phase(1)
    assert set(mix) == {"swec", "ajile12", "dcohort", "bt"}


def test_corpus_mix_p2_drops_ajile12_and_swec() -> None:
    """P2 = D-cohort + BT (224h sEEG-only per cross-subject strategy memo)."""
    mix = v14_corpus_mix_for_phase(2)
    assert set(mix) == {"dcohort", "bt"}


def test_corpus_mix_p3_is_bt_only_paired_audio() -> None:
    """P3a / P3b: BT-only (paired iEEG + audio for Whisper distillation)."""
    for phase in ("3a", "3b"):
        assert v14_corpus_mix_for_phase(phase) == ("bt",)


def test_corpus_mix_p4_is_bt_only() -> None:
    """P4: downstream BT linear probe."""
    assert v14_corpus_mix_for_phase(4) == ("bt",)


def test_corpus_mix_unknown_phase_raises() -> None:
    with pytest.raises(ValueError, match="unknown phase"):
        v14_corpus_mix_for_phase("p1")  # type: ignore[arg-type]


# ---------- per-phase sampler spec ----------------------------------------


def test_sampler_spec_p1_is_hierarchical_alpha_half() -> None:
    """P1: α=0.5 hierarchical sampler over the 4-corpus mix."""
    spec = v14_sampler_spec_for_phase(1)
    assert spec.kind == "hierarchical"
    assert spec.alpha == 0.5
    assert spec.paired_audio is False


def test_sampler_spec_p2_is_uniform_per_subject() -> None:
    """P2: uniform-per-subject across the D+BT sEEG mix."""
    spec = v14_sampler_spec_for_phase(2)
    assert spec.kind == "uniform_per_subject"
    assert spec.alpha is None
    assert spec.paired_audio is False


def test_sampler_spec_p3_requires_paired_audio() -> None:
    """P3a/P3b: paired iEEG+audio sampler (Whisper distillation)."""
    for phase in ("3a", "3b"):
        spec = v14_sampler_spec_for_phase(phase)
        assert spec.kind == "paired_ieeg_audio", f"phase {phase} spec.kind={spec.kind!r}"
        assert spec.paired_audio is True


def test_sampler_spec_p4_is_standard_supervised() -> None:
    """P4 = downstream supervised probe: standard per-event sampling."""
    spec = v14_sampler_spec_for_phase(4)
    assert spec.kind == "supervised"
    assert spec.paired_audio is False


# ---------- V14Data pydantic discrimination -----------------------------


def test_v14_data_accepts_phase_field() -> None:
    cfg = V14Data.model_construct(phase=1)
    assert cfg.phase == 1


@pytest.mark.parametrize("bad", [0, 3, "3c", None])
def test_v14_data_rejects_illegal_phase(bad: object) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        V14Data(phase=bad)  # type: ignore[arg-type]


def test_v14_data_resolves_corpus_mix_from_phase() -> None:
    """``corpus_mix`` defaults to the phase-derived tuple."""
    cfg = V14Data.model_construct(phase=2)
    assert cfg.resolved_corpus_mix() == ("dcohort", "bt")


def test_v14_data_resolves_sampler_spec_from_phase() -> None:
    cfg = V14Data.model_construct(phase=1)
    spec = cfg.resolved_sampler_spec()
    assert spec.kind == "hierarchical" and spec.alpha == 0.5


def test_sampler_spec_dataclass_is_frozen() -> None:
    """``SamplerSpec`` is frozen so a phase resolver can't be mutated mid-run."""
    spec = SamplerSpec(kind="hierarchical", alpha=0.5, paired_audio=False)
    with pytest.raises((AttributeError, Exception)):
        spec.alpha = 0.25  # type: ignore[misc]
