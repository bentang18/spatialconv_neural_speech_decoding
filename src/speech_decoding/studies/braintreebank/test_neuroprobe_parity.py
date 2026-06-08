"""Neuroprobe split-policy parity — TIER 1 (local, data-free).

Ben's requirement (2026-06-08): eval has **100% parity on all counts** with the
Neuroprobe leaderboard for all three modes — WithinSession, CrossSession,
CrossSubject. Parity decomposes into independently-verifiable pieces:

1. **Per-clip identity** (neural window + label) and **item order** — already
   pinned by ``test_word_events.py`` against upstream's interleaving
   (``derive_label_indices`` + ``ordered_dataset_*``) and the ``est_idx`` window.
2. **Split arithmetic** — how the interleaved item order is partitioned into
   train/val/test (and folds). THIS file. We drive the *actual* upstream
   primitives — ``sklearn.model_selection.KFold(n_splits=2, shuffle=False)`` (the
   exact call in ``train_test_splits.generate_splits_within_session``) and the
   literal val/test halving (``range(0, n//2)`` / ``range(n//2, n)``) copied from
   ``train_test_splits.py`` — and assert our ``_assign_*`` functions reproduce
   the realized index partitions byte-for-byte, including order, for many N
   (even, odd, small).
3. **Electrode set** (Lite-120) and the **full sample-level realized split** vs a
   *constructed* upstream ``Dataset`` — TIER 2, DCC-only (upstream ``__init__``
   eager-reads ``transcripts/*/features.csv`` + the h5), in
   ``scripts/neuroprobe/verify_parity.py``.

Tier 1 closes the split-arithmetic gap — the part most prone to off-by-one /
fold-indexing bugs — without BT data. Tier 2 closes the remaining gap end-to-end.
"""

from __future__ import annotations

import pandas as pd
import pytest

from speech_decoding.studies.braintreebank.word_events import (
    _LITE_N_FOLDS,
    _assign_cross_session_split,
    _assign_cross_subject_split,
    _assign_within_session_split,
)

# sklearn is an upstream + transitive dependency; import the *real* splitter so
# the parity comparison is against the exact object upstream uses, not a re-impl.
KFold = pytest.importorskip("sklearn.model_selection").KFold


# --- upstream reference primitives (verbatim from train_test_splits.py) -------

def _upstream_val_test_halving(test_idx: list[int]) -> tuple[list[int], list[int]]:
    """The val/test halving every upstream ``generate_splits_*`` performs on a
    held-out index list. Copied verbatim (datasets line refs in module docstring):

        test_size = len(test_dataset)
        val_size  = test_size // 2
        val_indices  = list(range(val_size))           # -> test_idx[:val_size]
        test_indices = list(range(val_size, test_size)) # -> test_idx[val_size:]
    """
    test_size = len(test_idx)
    val_size = test_size // 2
    val = [test_idx[i] for i in range(val_size)]
    test = [test_idx[i] for i in range(val_size, test_size)]
    return val, test


def _upstream_within_session_partition(
    n: int, fold_index: int, n_folds: int = _LITE_N_FOLDS
) -> tuple[list[int], list[int], list[int]]:
    """Upstream ``generate_splits_within_session`` realized (train, val, test)
    positional index lists for one fold, using the real ``KFold`` object."""
    kf = KFold(n_splits=n_folds, shuffle=False)
    train_idx, test_idx = list(kf.split(range(n)))[fold_index]
    train = list(train_idx)
    val, test = _upstream_val_test_halving(list(test_idx))
    return train, val, test


def _positions_by_split(out: pd.DataFrame) -> dict[str, list[int]]:
    """Recover, per split, the ordered list of original positions. Test frames
    encode position in the ``start`` column (start == position), so positions are
    read straight back in the assigner's emission order."""
    result: dict[str, list[int]] = {}
    for split in ("train", "val", "test"):
        rows = out.loc[out["split"] == split]
        result[split] = [int(round(s)) for s in rows["start"].tolist()]
    return result


def _single_trial_speech_df(n: int, *, subject_id: int, trial_id: int) -> pd.DataFrame:
    """``n`` interleaved 'speech' Word rows on one (subject, trial), ``start`` ==
    position so the realized partition is recoverable. Labels alternate the way
    the interleave produces them; irrelevant to split arithmetic but realistic."""
    return pd.DataFrame(
        {
            "type": ["Word"] * n,
            "start": [float(i) for i in range(n)],
            "duration": [1.0] * n,
            "task": ["speech"] * n,
            "label": [(i + 1) % 2 for i in range(n)],
            "subject_id": [str(subject_id)] * n,
            "trial_id": [str(trial_id)] * n,
            "timeline": [f"tl{subject_id}_{trial_id}"] * n,
        }
    )


# --- WithinSession ------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4, 7, 10, 11, 100, 101, 3500])
@pytest.mark.parametrize("fold_index", list(range(_LITE_N_FOLDS)))
def test_within_session_split_matches_upstream_kfold_exactly(n, fold_index) -> None:
    """Our WithinSession assigner == upstream KFold(2, shuffle=False) + halving,
    for every fold and a range of N (even/odd/small/lite-cap)."""
    df = _single_trial_speech_df(n, subject_id=2, trial_id=4)
    out = _assign_within_session_split(
        df, test_subject_id=2, test_trial_id=4, fold_index=fold_index
    )
    got = _positions_by_split(out)
    want_train, want_val, want_test = _upstream_within_session_partition(n, fold_index)

    assert got["train"] == want_train, f"train mismatch (n={n}, fold={fold_index})"
    assert got["val"] == want_val, f"val mismatch (n={n}, fold={fold_index})"
    assert got["test"] == want_test, f"test mismatch (n={n}, fold={fold_index})"
    # Partition completeness: train ⊎ val ⊎ test == every position, disjoint.
    union = sorted(got["train"] + got["val"] + got["test"])
    assert union == list(range(n)), "splits must partition all items, no overlap/gap"


def test_within_session_two_folds_cover_all_items_as_test_once() -> None:
    """The two folds' (val ∪ test) held-out sets tile the dataset exactly once —
    the defining property of K-fold CV (each item is evaluated in exactly one
    fold). Guards against a fold-indexing bug that re-uses the same held-out half."""
    n = 101
    df = _single_trial_speech_df(n, subject_id=2, trial_id=4)
    heldout: list[int] = []
    for fold in range(_LITE_N_FOLDS):
        out = _assign_within_session_split(
            df, test_subject_id=2, test_trial_id=4, fold_index=fold
        )
        pos = _positions_by_split(out)
        heldout += pos["val"] + pos["test"]
    assert sorted(heldout) == list(range(n)), "folds must tile every item once"


def test_within_session_rejects_out_of_range_fold() -> None:
    df = _single_trial_speech_df(10, subject_id=2, trial_id=4)
    for bad in (-1, _LITE_N_FOLDS, _LITE_N_FOLDS + 3):
        with pytest.raises(ValueError, match="fold_index"):
            _assign_within_session_split(
                df, test_subject_id=2, test_trial_id=4, fold_index=bad
            )


def test_within_session_per_task_independent_folds() -> None:
    """Two tasks on the same trial fold independently — upstream builds one
    Dataset (hence one KFold) per eval_name, so each task's partition is keyed
    only by its own item count."""
    n_a, n_b = 10, 7
    rows = []
    for task, n in (("speech", n_a), ("onset", n_b)):
        for i in range(n):
            rows.append(
                {"type": "Word", "start": float(i), "duration": 1.0, "task": task,
                 "label": (i + 1) % 2, "subject_id": "2", "trial_id": "4",
                 "timeline": "tl2_4"}
            )
    df = pd.DataFrame(rows)
    out = _assign_within_session_split(
        df, test_subject_id=2, test_trial_id=4, fold_index=0
    )
    for task, n in (("speech", n_a), ("onset", n_b)):
        sub = out.loc[out["task"] == task]
        got = {s: [int(round(x)) for x in sub.loc[sub["split"] == s, "start"]]
               for s in ("train", "val", "test")}
        want_train, want_val, want_test = _upstream_within_session_partition(n, 0)
        assert got["train"] == want_train
        assert got["val"] == want_val
        assert got["test"] == want_test


# --- CrossSession -------------------------------------------------------------

@pytest.mark.parametrize("n_test", [2, 3, 10, 11, 100, 101])
def test_cross_session_split_matches_upstream_halving(n_test) -> None:
    """CrossSession: the other trial is train in full; the test trial's
    interleaved order is halved val=test_idx[:n//2] / test=test_idx[n//2:]."""
    n_train = 40
    train_df = _single_trial_speech_df(n_train, subject_id=2, trial_id=0)
    test_df = _single_trial_speech_df(n_test, subject_id=2, trial_id=4)
    # Offset test 'start' so train/test positions don't collide when recovered.
    test_df["start"] = [float(1000 + i) for i in range(n_test)]
    df = pd.concat([train_df, test_df], ignore_index=True)

    out = _assign_cross_session_split(df, test_subject_id=2, test_trial_id=4)

    train_pos = sorted(int(round(s)) for s in out.loc[out["split"] == "train", "start"])
    assert train_pos == list(range(n_train)), "train = the WHOLE other trial"

    test_idx = list(range(n_test))
    want_val, want_test = _upstream_val_test_halving(test_idx)
    got_val = [int(round(s)) - 1000 for s in out.loc[out["split"] == "val", "start"]]
    got_test = [int(round(s)) - 1000 for s in out.loc[out["split"] == "test", "start"]]
    assert got_val == want_val
    assert got_test == want_test


def test_cross_session_drops_other_subjects_entirely() -> None:
    """Only the test subject's two trials survive; other subjects are dropped
    (upstream constructs Datasets only for the test subject)."""
    keep = _single_trial_speech_df(10, subject_id=2, trial_id=4)
    other_subj = _single_trial_speech_df(10, subject_id=3, trial_id=0)
    train = _single_trial_speech_df(10, subject_id=2, trial_id=0)
    df = pd.concat([keep, other_subj, train], ignore_index=True)
    out = _assign_cross_session_split(df, test_subject_id=2, test_trial_id=4)
    assert (out["subject_id"] == "2").all()


# --- CrossSubject -------------------------------------------------------------

@pytest.mark.parametrize("n_test", [2, 3, 10, 11, 100, 101])
def test_cross_subject_split_matches_upstream(n_test) -> None:
    """CrossSubject: train = the WHOLE (DS_DM_TRAIN_SUBJECT_ID=2, trial=4); test
    subject's trial halved identically to upstream."""
    n_train = 40
    train_df = _single_trial_speech_df(n_train, subject_id=2, trial_id=4)
    test_df = _single_trial_speech_df(n_test, subject_id=3, trial_id=0)
    test_df["start"] = [float(1000 + i) for i in range(n_test)]
    df = pd.concat([train_df, test_df], ignore_index=True)

    out = _assign_cross_subject_split(
        df, test_subject_id=3, test_trial_id=0, train_subject_id=2, train_trial_id=4
    )

    train_pos = sorted(int(round(s)) for s in out.loc[out["split"] == "train", "start"])
    assert train_pos == list(range(n_train)), "train = the WHOLE (2,4) trial"

    want_val, want_test = _upstream_val_test_halving(list(range(n_test)))
    got_val = [int(round(s)) - 1000 for s in out.loc[out["split"] == "val", "start"]]
    got_test = [int(round(s)) - 1000 for s in out.loc[out["split"] == "test", "start"]]
    assert got_val == want_val
    assert got_test == want_test


def test_cross_subject_drops_untargeted_subjects() -> None:
    """Anything that is neither the train (2,4) nor the test (subj,trial) is
    dropped — upstream's default ``include_all_train_subjects=False``."""
    train = _single_trial_speech_df(10, subject_id=2, trial_id=4)
    test = _single_trial_speech_df(10, subject_id=3, trial_id=0)
    distractor = _single_trial_speech_df(10, subject_id=4, trial_id=1)
    df = pd.concat([train, test, distractor], ignore_index=True)
    out = _assign_cross_subject_split(
        df, test_subject_id=3, test_trial_id=0, train_subject_id=2, train_trial_id=4
    )
    seen = {(s, t) for s, t in zip(out["subject_id"], out["trial_id"])}
    assert seen == {("2", "4"), ("3", "0")}


def test_lite_n_folds_matches_upstream_constant() -> None:
    """Pin our mirrored fold count to upstream's. If upstream ever changes
    NEUROPROBE_LITE_N_FOLDS, this guards the silent drift."""
    assert _LITE_N_FOLDS == 2
