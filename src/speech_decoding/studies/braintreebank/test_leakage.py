from __future__ import annotations

import pandas as pd
import pytest

from speech_decoding.studies.braintreebank.leakage import (
    assert_no_eval_leakage,
    study_is_braintreebank,
)
from speech_decoding.studies.braintreebank.manifest import (
    BT_LITE_SESSIONS,
    V14_PRETRAIN_SESSIONS,
)


class Wang2024Treebank:  # name-matched stand-in for the real BT study
    pass


class _OtherStudy:
    name = "SWEC"


class _Chain:
    def __init__(self, *steps):
        self.steps = list(steps)


class _DictChain:
    """ns.Chain also accepts an OrderedDict[str, Step] for ``steps``."""

    def __init__(self, **steps):
        self.steps = dict(steps)


class _FakeDataset:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def triggers(self) -> pd.DataFrame:
        return self._df


class _FakeLoader:
    def __init__(self, df: pd.DataFrame | None):
        self.dataset = _FakeDataset(df) if df is not None else None


def _triggers(sessions: list[tuple[int, int]]) -> pd.DataFrame:
    # Real triggers store subject_id / trial_id as strings (word_events.py).
    return pd.DataFrame(
        {
            "subject_id": [str(s) for s, _ in sessions],
            "trial_id": [str(t) for _, t in sessions],
            "split": ["train"] * len(sessions),
        }
    )


def _loaders(train: list[tuple[int, int]], val: list[tuple[int, int]]):
    return {"train": _FakeLoader(_triggers(train)), "val": _FakeLoader(_triggers(val))}


# --- study detection -------------------------------------------------------

def test_study_is_braintreebank_by_type_name():
    assert study_is_braintreebank(Wang2024Treebank())
    assert study_is_braintreebank(_Chain(_OtherStudy(), Wang2024Treebank()))


def test_study_is_not_braintreebank():
    assert not study_is_braintreebank(_OtherStudy())
    assert not study_is_braintreebank(_Chain(_OtherStudy()))


def test_study_is_braintreebank_dict_steps():
    # ns.Chain accepts an OrderedDict[str, Step]. A naive list(steps) would see
    # the string keys and miss the BT study (fail-open). The guard must unwrap.
    assert study_is_braintreebank(_DictChain(bt=Wang2024Treebank(), we=_OtherStudy()))
    assert not study_is_braintreebank(_DictChain(a=_OtherStudy(), b=_OtherStudy()))


def test_study_is_braintreebank_nested_chain():
    # A BT study two levels deep must still be detected.
    assert study_is_braintreebank(_Chain(_Chain(Wang2024Treebank()), _OtherStudy()))
    assert study_is_braintreebank(_DictChain(inner=_Chain(Wang2024Treebank())))
    assert not study_is_braintreebank(_Chain(_Chain(_OtherStudy())))


def test_dict_steps_leak_trips():
    # End-to-end: the guard must still trip on a leaking corpus when the chain
    # uses the dict-of-steps form (the fail-open hole from the #82 audit).
    bad = BT_LITE_SESSIONS[0]
    loaders = _loaders(train=[bad, *V14_PRETRAIN_SESSIONS[:1]], val=list(V14_PRETRAIN_SESSIONS[:1]))
    with pytest.raises(RuntimeError, match="LEAKAGE GUARD TRIPPED"):
        assert_no_eval_leakage(loaders, study=_DictChain(bt=Wang2024Treebank()))


# --- guard core ------------------------------------------------------------

def test_legal_corpus_passes():
    legal = list(V14_PRETRAIN_SESSIONS)
    loaders = _loaders(train=legal, val=legal[:1])
    # Must NOT raise.
    assert_no_eval_leakage(loaders, study=_Chain(Wang2024Treebank()))


def test_eval_session_in_train_trips():
    bad = BT_LITE_SESSIONS[0]
    loaders = _loaders(train=[bad, *V14_PRETRAIN_SESSIONS[:1]], val=list(V14_PRETRAIN_SESSIONS[:1]))
    with pytest.raises(RuntimeError, match="LEAKAGE GUARD TRIPPED"):
        assert_no_eval_leakage(loaders, study=_Chain(Wang2024Treebank()))


def test_eval_session_in_val_trips():
    bad = BT_LITE_SESSIONS[1]
    loaders = _loaders(train=list(V14_PRETRAIN_SESSIONS[:1]), val=[bad])
    with pytest.raises(RuntimeError, match="LEAKAGE GUARD TRIPPED"):
        assert_no_eval_leakage(loaders, study=_Chain(Wang2024Treebank()))


def test_current_coupling_bug_single_subject_s2_trips():
    # The pre-fix CrossSession coupling pretrains on (2, 0) — an off-limits
    # eval session. The guard must catch exactly this.
    assert (2, 0) in BT_LITE_SESSIONS
    loaders = _loaders(train=[(2, 0)], val=[(2, 4)])
    with pytest.raises(RuntimeError, match="LEAKAGE GUARD TRIPPED"):
        assert_no_eval_leakage(loaders, study=_Chain(Wang2024Treebank()))


def test_fail_closed_on_missing_columns():
    df = pd.DataFrame({"split": ["train"]})  # no subject_id / trial_id
    loaders = {"train": _FakeLoader(df), "val": _FakeLoader(_triggers(list(V14_PRETRAIN_SESSIONS[:1])))}
    with pytest.raises(RuntimeError, match="fail-closed"):
        assert_no_eval_leakage(loaders, study=_Chain(Wang2024Treebank()))


def test_fail_closed_on_missing_dataset():
    loaders = {"train": _FakeLoader(None), "val": _FakeLoader(_triggers(list(V14_PRETRAIN_SESSIONS[:1])))}
    with pytest.raises(RuntimeError, match="fail-closed"):
        assert_no_eval_leakage(loaders, study=_Chain(Wang2024Treebank()))


def test_non_bt_study_is_exempt():
    # Even an off-limits-looking pair passes when the study is not BT — the
    # off-limits set is BrainTreebank-specific.
    loaders = _loaders(train=[BT_LITE_SESSIONS[0]], val=[BT_LITE_SESSIONS[1]])
    assert_no_eval_leakage(loaders, study=_OtherStudy())


def test_missing_phase_loader_is_skipped():
    # A phase absent from the mapping (e.g. no val) is skipped, not an error.
    loaders = {"train": _FakeLoader(_triggers(list(V14_PRETRAIN_SESSIONS[:1])))}
    assert_no_eval_leakage(loaders, study=_Chain(Wang2024Treebank()))
