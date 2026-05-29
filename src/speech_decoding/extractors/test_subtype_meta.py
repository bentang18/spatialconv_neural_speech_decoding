"""Tests for B29 Item 9/11/12 subject-subtype + λ_anat metadata extractors."""

from __future__ import annotations

import numpy as np
import pytest
from neuralset.base import TimedArray

from speech_decoding.extractors.reference import CARIeegExtractor
from speech_decoding.extractors.subtype_meta import (
    BINARY_SUBTYPE_TO_IDX,
    DEFAULT_CORPUS_LAMBDA_ANAT,
    DEFAULT_CORPUS_SUBTYPE,
    THREE_WAY_SUBTYPE_TO_IDX,
    LambdaAnatExtractor,
    SubjectSubtypeExtractor,
)


class _Event:
    def __init__(
        self,
        corpus: str | None = None,
        subject_id: str | None = None,
    ) -> None:
        if corpus is not None:
            self.corpus = corpus
        if subject_id is not None:
            self.subject_id = subject_id


# ---------------------------------------------------------------------------
# Subject-subtype
# ---------------------------------------------------------------------------


def test_subject_subtype_binary_vocab_constants() -> None:
    assert BINARY_SUBTYPE_TO_IDX == {"depth": 0, "ecog": 1}
    assert THREE_WAY_SUBTYPE_TO_IDX == {"depth": 0, "grid": 1, "strip": 2}


def test_default_corpus_subtype_table_anatomy_lock() -> None:
    """BT/D/SWEC = depth; AJILE12 = ECoG (Peterson 2022 surface grid)."""
    assert DEFAULT_CORPUS_SUBTYPE["braintreebank"] == "depth"
    assert DEFAULT_CORPUS_SUBTYPE["d_cohort"] == "depth"
    assert DEFAULT_CORPUS_SUBTYPE["swec"] == "depth"
    assert DEFAULT_CORPUS_SUBTYPE["ajile12"] == "ecog"


def test_subject_subtype_default_emits_zero_for_bt() -> None:
    extractor = SubjectSubtypeExtractor(event_types="Ieeg")
    out = extractor._get_timed_array(
        _Event(corpus="braintreebank"), start=0.0, duration=1.0,
    )
    assert isinstance(out, TimedArray)
    arr = np.asarray(out.data)
    assert arr.shape == (1,)
    assert arr.dtype == np.int64
    assert int(arr.item()) == 0  # depth


def test_subject_subtype_default_emits_one_for_ajile12() -> None:
    extractor = SubjectSubtypeExtractor(event_types="Ieeg")
    out = extractor._get_timed_array(
        _Event(corpus="ajile12"), start=0.0, duration=1.0,
    )
    assert int(np.asarray(out.data).item()) == 1  # ecog


def test_subject_subtype_three_way_vocab_accepts_grid_and_strip() -> None:
    extractor = SubjectSubtypeExtractor(
        event_types="Ieeg", vocab="three_way",
        corpus_subtype={"ajile12_strip": "strip", "ajile12_grid": "grid"},
    )
    out_strip = extractor._get_timed_array(
        _Event(corpus="ajile12_strip"), start=0.0, duration=1.0,
    )
    out_grid = extractor._get_timed_array(
        _Event(corpus="ajile12_grid"), start=0.0, duration=1.0,
    )
    assert int(np.asarray(out_strip.data).item()) == 2
    assert int(np.asarray(out_grid.data).item()) == 1


def test_subject_subtype_per_subject_override_wins_over_corpus() -> None:
    extractor = SubjectSubtypeExtractor(
        event_types="Ieeg",
        subject_subtype={"S99": "ecog"},  # override the BT default
    )
    out = extractor._get_timed_array(
        _Event(corpus="braintreebank", subject_id="S99"),
        start=0.0, duration=1.0,
    )
    assert int(np.asarray(out.data).item()) == 1


def test_subject_subtype_rejects_invalid_vocab() -> None:
    with pytest.raises(ValueError, match="vocab"):
        SubjectSubtypeExtractor(event_types="Ieeg", vocab="quaternary")  # type: ignore


def test_subject_subtype_rejects_unknown_value_in_table() -> None:
    with pytest.raises(ValueError, match="corpus_subtype"):
        SubjectSubtypeExtractor(
            event_types="Ieeg",
            corpus_subtype={"braintreebank": "not_a_subtype"},
        )


def test_subject_subtype_rejects_unregistered_corpus() -> None:
    extractor = SubjectSubtypeExtractor(event_types="Ieeg")
    with pytest.raises(KeyError, match="no subtype registered"):
        extractor._get_timed_array(
            _Event(corpus="unknown_cohort"), start=0.0, duration=1.0,
        )


def test_subject_subtype_three_way_demotes_grid_to_ecog_under_binary_vocab() -> None:
    """When the corpus default is grid but the extractor vocab is binary,
    fall back to ``ecog``."""
    extractor = SubjectSubtypeExtractor(
        event_types="Ieeg", vocab="binary",
    )

    # Patch DEFAULT_CORPUS_SUBTYPE in the extractor module so we can
    # simulate a grid-default corpus without mutating the real table.
    from speech_decoding.extractors import subtype_meta

    orig = dict(subtype_meta.DEFAULT_CORPUS_SUBTYPE)
    subtype_meta.DEFAULT_CORPUS_SUBTYPE["mock_grid_corpus"] = "grid"  # type: ignore[index]
    try:
        out = extractor._get_timed_array(
            _Event(corpus="mock_grid_corpus"), start=0.0, duration=1.0,
        )
        assert int(np.asarray(out.data).item()) == 1  # ecog binary
    finally:
        # Restore module state.
        subtype_meta.DEFAULT_CORPUS_SUBTYPE.clear()  # type: ignore[union-attr]
        subtype_meta.DEFAULT_CORPUS_SUBTYPE.update(orig)  # type: ignore[union-attr]


def test_subject_subtype_inherits_car_ieeg_extractor() -> None:
    assert issubclass(SubjectSubtypeExtractor, CARIeegExtractor)


# ---------------------------------------------------------------------------
# λ_anat
# ---------------------------------------------------------------------------


def test_default_corpus_lambda_anat_table_b29_item_12() -> None:
    """BT/D/AJILE12 = 1.0 (anatomy-rich); SWEC = 0.0."""
    assert DEFAULT_CORPUS_LAMBDA_ANAT["braintreebank"] == 1.0
    assert DEFAULT_CORPUS_LAMBDA_ANAT["d_cohort"] == 1.0
    assert DEFAULT_CORPUS_LAMBDA_ANAT["ajile12"] == 1.0
    assert DEFAULT_CORPUS_LAMBDA_ANAT["swec"] == 0.0


def test_lambda_anat_anatomy_rich_corpus_gates_on() -> None:
    extractor = LambdaAnatExtractor(event_types="Ieeg")
    out = extractor._get_timed_array(
        _Event(corpus="braintreebank"), start=0.0, duration=1.0,
    )
    assert isinstance(out, TimedArray)
    arr = np.asarray(out.data)
    assert arr.shape == (1,)
    assert arr.dtype == np.float32
    assert float(arr.item()) == 1.0


def test_lambda_anat_swec_corpus_gates_off_by_default() -> None:
    extractor = LambdaAnatExtractor(event_types="Ieeg")
    out = extractor._get_timed_array(
        _Event(corpus="swec"), start=0.0, duration=1.0,
    )
    assert float(np.asarray(out.data).item()) == 0.0


def test_lambda_anat_per_subject_override_wins_over_corpus() -> None:
    extractor = LambdaAnatExtractor(
        event_types="Ieeg",
        subject_lambda_anat={"swec_sub1": 0.5},
    )
    out = extractor._get_timed_array(
        _Event(corpus="swec", subject_id="swec_sub1"),
        start=0.0, duration=1.0,
    )
    assert float(np.asarray(out.data).item()) == 0.5


def test_lambda_anat_rejects_value_out_of_range() -> None:
    with pytest.raises(ValueError, match="must lie in"):
        LambdaAnatExtractor(
            event_types="Ieeg",
            corpus_lambda_anat={"braintreebank": 1.5},
        )
    with pytest.raises(ValueError, match="must lie in"):
        LambdaAnatExtractor(
            event_types="Ieeg",
            subject_lambda_anat={"sub_neg": -0.1},
        )


def test_lambda_anat_rejects_unregistered_corpus() -> None:
    extractor = LambdaAnatExtractor(event_types="Ieeg")
    with pytest.raises(KeyError, match="no λ_anat registered"):
        extractor._get_timed_array(
            _Event(corpus="unknown_cohort"), start=0.0, duration=1.0,
        )


def test_lambda_anat_swec_pseudo_parcel_sister_can_turn_on() -> None:
    """Sister ``R-swec-pseudo-parcel-per-shaft`` flips SWEC λ_anat ON
    via the per-corpus override."""
    extractor = LambdaAnatExtractor(
        event_types="Ieeg",
        corpus_lambda_anat={"swec": 1.0},
    )
    out = extractor._get_timed_array(
        _Event(corpus="swec"), start=0.0, duration=1.0,
    )
    assert float(np.asarray(out.data).item()) == 1.0


def test_lambda_anat_extractor_inherits_car_ieeg_extractor() -> None:
    assert issubclass(LambdaAnatExtractor, CARIeegExtractor)


def test_subject_subtype_corpus_kwarg_fallback_when_event_has_no_attr() -> None:
    """When the event lacks a ``corpus`` attribute, the ``corpus`` kwarg
    on the extractor is used as the lookup key."""
    extractor = SubjectSubtypeExtractor(event_types="Ieeg", corpus="braintreebank")
    evt = _Event(corpus=None, subject_id="anonymous")
    out = extractor._get_timed_array(evt, start=0.0, duration=1.0)
    assert int(np.asarray(out.data).item()) == 0
