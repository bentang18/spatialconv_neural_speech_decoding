"""Append BrainTreebank Word events to a NeuralSet events DataFrame.

Composes downstream of :class:`Wang2024Treebank` inside an ``ns.Chain``. Each
``Ieeg`` row from the study is unpacked into per-task balanced ``Word`` events
matching Neuroprobe-upstream's :class:`BrainTreebankSubjectTrialBenchmarkDataset`
labelling (``derive_label_indices``) and split policy
(``train_test_splits.generate_splits_cross_session``).

Split policy mirrors upstream exactly so v14 numbers compare apples-to-apples
with the v2 paper baselines. Row order within each (subject, trial)
follows upstream's ``BrainTreebankSubjectTrialBenchmarkDataset.__getitem__``
verbatim — items strictly interleave classes via ``(idx + 1) % n_classes``
over chronologically-sorted per-class index lists. The val|test halving
then operates on this interleaved order, NOT on a sort by overall
``start``:

* ``CrossSession`` (the submission gate): the test trial's interleaved
  events are split at ``n // 2`` — first half → ``val``, second half →
  ``test``. The *other* trial of the same subject becomes ``train``.
  Other subjects' timelines are dropped.
* ``CrossSubject``: same per-trial split policy on the test (subject,
  trial); train comes from ``(DS_DM_TRAIN_SUBJECT_ID,
  DS_DM_TRAIN_TRIAL_ID)`` only (``include_all_train_subjects=False``, the
  upstream leaderboard default).

Onset/Speech tasks use ``words_df`` for positives and ``nonverbal_df`` for
negatives; all other tasks pull from ``words_df`` alone. Tasks whose label
column is transcript-derived (``face_num``, ``delta_volume``, ``word_index``,
...) need enrichment from ``$ROOT_DIR_BRAINTREEBANK/transcripts/{movie}/features.csv``
to materialise columns like ``face_num``, ``idx_in_sentence`` and ``delta_rms``;
only ``speech`` works without enrichment.
"""

from __future__ import annotations

import os
import typing as tp
from pathlib import Path

import pandas as pd

from neuralset.events.study import EventsTransform

from speech_decoding.studies.braintreebank.labels import (
    NEUROPROBE_TASKS,
    derive_label_indices,
    enrich_words_with_transcript_features,
    ordered_dataset_labels,
    ordered_dataset_source_indices,
)


EvalMode = tp.Literal["CrossSession", "CrossSubject"]


_UPSTREAM_TRAIN_SUBJECT_ID = 2
_UPSTREAM_TRAIN_TRIAL_ID = 4


def _neural_sample_rate() -> float:
    """BT neural-clock sample rate (2048 Hz). ``est_idx`` is indexed on this
    clock, so word-onset seconds = ``est_idx / _neural_sample_rate()``.

    Mirrors ``loader._sampling_rate``: reads ``neuroprobe.config.SAMPLING_RATE``
    when available, else falls back to 2048.0 (``neuroprobe.config`` reads
    ``os.environ['ROOT_DIR_BRAINTREEBANK']`` at import, so it raises ``KeyError``
    on laptops without BT data mounted — the unit tests' path)."""
    try:
        from neuroprobe.config import SAMPLING_RATE
    except (ImportError, KeyError):
        return 2048.0
    return float(SAMPLING_RATE)


def _vendored_csv_dir() -> Path:
    """Return Neuroprobe's vendored ``braintreebank_features_time_alignment``."""
    from neuroprobe.config import SAVE_SUBJECT_TRIAL_DF_DIR

    return Path(SAVE_SUBJECT_TRIAL_DF_DIR)


def _movie_name(subject_id: int, trial_id: int) -> str:
    from neuroprobe.config import BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING

    return BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"btbank{subject_id}_{trial_id}"]


def _load_words_and_nonverbal(
    subject_id: int,
    trial_id: int,
    *,
    bt_root: str | Path | None,
    enrich: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the vendored words/nonverbal CSVs and optionally enrich.

    ``enrich=True`` joins ``transcripts/{movie}/features.csv`` from the BT data
    root, producing ``is_onset``, ``idx_in_sentence``, ``face_num``,
    ``delta_rms`` etc. ``enrich=False`` returns the raw vendored CSV (enough
    for ``speech`` task; tests).
    """
    csv_dir = _vendored_csv_dir()
    words_df = pd.read_csv(csv_dir / f"subject{subject_id}_trial{trial_id}_words_df.csv")
    nonverbal_df = pd.read_csv(
        csv_dir / f"subject{subject_id}_trial{trial_id}_nonverbal_df.csv"
    )
    if enrich:
        if bt_root is None:
            raise RuntimeError(
                "BTWordEvents: enrich=True requires bt_root or "
                "ROOT_DIR_BRAINTREEBANK env var to locate transcripts/."
            )
        movie = _movie_name(subject_id, trial_id)
        features_path = Path(bt_root) / "transcripts" / movie / "features.csv"
        transcript_features = pd.read_csv(features_path)
        words_df = enrich_words_with_transcript_features(words_df, transcript_features)
    return words_df, nonverbal_df


def _word_event_rows(
    *,
    subject_id: int,
    trial_id: int,
    timeline: str,
    words_df: pd.DataFrame,
    nonverbal_df: pd.DataFrame,
    tasks: tp.Sequence[str],
    binary_tasks: bool,
    lite: bool,
    nano: bool,
    random_seed: int,
    duration: float,
) -> pd.DataFrame:
    """Build per-task balanced Word rows in upstream Dataset item order.

    Mirrors :meth:`BrainTreebankSubjectTrialBenchmarkDataset.__getitem__`
    exactly: items strictly interleave classes via ``(idx + 1) % n_classes``
    over chronologically-sorted per-class index lists. The downstream cut
    ``val = range(0, n//2); test = range(n//2, n)`` then produces
    class-balanced halves — sorting these rows by overall ``start`` instead
    would silently flip val/test majority class whenever one class
    temporally clusters differently from the other, producing the failure
    mode ``test_acc ≈ 1 − val_acc`` regardless of model quality.
    """
    rows: list[dict[str, tp.Any]] = []
    # Word/nonverbal events are sliced from the neural stream at the NEURAL-clock
    # onset `est_idx` (samples @ 2048 Hz), NOT the transcript/movie-relative
    # `start` time. The two diverge by a per-trial neural-vs-movie-clock offset
    # that DRIFTS within a trial (sub_4_trial1: first word est_idx 561472 =
    # 274.16 s vs transcript 39.02 s → 235 s; the gap widens to ~904 s by the
    # last word). Upstream `datasets.py:300-301` windows
    # `[est_idx - before, est_idx + after]`; the leaderboard uses before=0,
    # after=1 s, which is exactly `start=est_idx/SR` with the default
    # `duration=1.0`. Emitting transcript `start` would slice every BT clip
    # 235–900 s off-target (C3).
    sample_rate = _neural_sample_rate()
    for task in tasks:
        label_indices = derive_label_indices(
            words_df=words_df,
            nonverbal_df=nonverbal_df,
            task=task,
            binary_tasks=binary_tasks,
            lite=lite,
            nano=nano,
            random_seed=random_seed,
        )
        ordered_labels = ordered_dataset_labels(label_indices)
        ordered_source_indices = ordered_dataset_source_indices(label_indices)
        for class_id_arr, src_idx_arr in zip(ordered_labels, ordered_source_indices):
            class_id = int(class_id_arr)
            src_idx = int(src_idx_arr)
            is_nonverbal = task in {"onset", "speech"} and class_id == 0
            if is_nonverbal:
                source = nonverbal_df.iloc[src_idx]
                text = "<nonverbal>"
            else:
                source = words_df.iloc[src_idx]
                raw_text = source.get("full_word", "")
                text = str(raw_text) if pd.notna(raw_text) and str(raw_text) else "<word>"
            rows.append(
                {
                    "type": "Word",
                    "start": float(source["est_idx"]) / sample_rate,
                    "duration": float(duration),
                    "text": text,
                    "task": task,
                    "label": class_id,
                    "subject_id": str(subject_id),
                    "trial_id": str(trial_id),
                    "timeline": timeline,
                    # MOVIE-clock onset (seconds into the movie audio), for the
                    # P3 Whisper-teacher join (WS-H / WhisperTargetExtractor). This
                    # is the transcript `start`, NOT `est_idx/SR` above: the two
                    # diverge by the per-trial neural-vs-movie drift (235-904 s,
                    # FLAG 9). The neural window slices at `start` (neural clock);
                    # the audio-keyed teacher cache is indexed by movie time, so
                    # it MUST slice at `movie_onset_s`. Both words_df and
                    # nonverbal_df carry the movie-clock column `start`.
                    "movie_onset_s": float(source["start"]),
                }
            )
    if not rows:
        return pd.DataFrame(
            {col: pd.Series(dtype=object) for col in (
                "type", "start", "duration", "text", "task", "label",
                "subject_id", "trial_id", "timeline", "movie_onset_s",
            )}
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _assign_cross_session_split(
    df: pd.DataFrame,
    *,
    test_subject_id: int,
    test_trial_id: int,
) -> pd.DataFrame:
    """Per-task chronological val/test halves on the test trial; train on
    the other trial of the test subject; drop other subjects."""
    if df.empty:
        df = df.copy()
        df["split"] = pd.Series(dtype=str)
        return df
    out = df.copy()
    out["split"] = ""
    s = out["subject_id"].astype(int)
    t = out["trial_id"].astype(int)
    is_test_st = (s == test_subject_id) & (t == test_trial_id)
    is_train_st = (s == test_subject_id) & (t != test_trial_id)
    out.loc[is_train_st, "split"] = "train"
    for task in out.loc[is_test_st, "task"].unique():
        sub_idx = out.index[is_test_st & (out["task"] == task)]
        n = len(sub_idx)
        if n == 0:
            continue
        cut = n // 2
        out.loc[sub_idx[:cut], "split"] = "val"
        out.loc[sub_idx[cut:], "split"] = "test"
    return out.loc[out["split"] != ""].reset_index(drop=True)


def _assign_cross_subject_split(
    df: pd.DataFrame,
    *,
    test_subject_id: int,
    test_trial_id: int,
    train_subject_id: int,
    train_trial_id: int,
) -> pd.DataFrame:
    """Per-task chronological val/test halves on (test_subject, test_trial);
    train on (train_subject, train_trial) only — upstream leaderboard default
    is ``DS_DM_TRAIN_SUBJECT_ID=2 / DS_DM_TRAIN_TRIAL_ID=4``."""
    if df.empty:
        df = df.copy()
        df["split"] = pd.Series(dtype=str)
        return df
    out = df.copy()
    out["split"] = ""
    s = out["subject_id"].astype(int)
    t = out["trial_id"].astype(int)
    is_test_st = (s == test_subject_id) & (t == test_trial_id)
    is_train_st = (s == train_subject_id) & (t == train_trial_id)
    out.loc[is_train_st, "split"] = "train"
    for task in out.loc[is_test_st, "task"].unique():
        sub_idx = out.index[is_test_st & (out["task"] == task)]
        n = len(sub_idx)
        if n == 0:
            continue
        cut = n // 2
        out.loc[sub_idx[:cut], "split"] = "val"
        out.loc[sub_idx[cut:], "split"] = "test"
    return out.loc[out["split"] != ""].reset_index(drop=True)


class BTWordEvents(EventsTransform):
    """Emit Neuroprobe-parity Word events with chronological splits.

    Consumes the ``Ieeg`` rows from :class:`Wang2024Treebank` and appends one
    ``Word`` row per balanced (task, class, sample) pair. The ``split`` column
    matches the upstream ``train_test_splits.py`` policy exactly so v14
    cold-start numbers compare apples-to-apples with the v2 paper baselines.
    """

    tasks: tuple[str, ...]
    binary_tasks: bool = True
    lite: bool = True
    nano: bool = False
    eval_mode: EvalMode = "CrossSession"
    test_subject_id: int = _UPSTREAM_TRAIN_SUBJECT_ID
    test_trial_id: int = _UPSTREAM_TRAIN_TRIAL_ID
    train_subject_id: int = _UPSTREAM_TRAIN_SUBJECT_ID
    train_trial_id: int = _UPSTREAM_TRAIN_TRIAL_ID
    bt_root: str | None = None
    duration: float = 1.0
    random_seed: int = 42

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["bt_root"]

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        unknown = [t for t in self.tasks if t not in NEUROPROBE_TASKS]
        if unknown:
            raise ValueError(
                f"BTWordEvents: unknown tasks {unknown}; "
                f"valid options = {NEUROPROBE_TASKS}"
            )

    def _enrich_needed(self) -> bool:
        """Only ``speech`` works without transcript enrichment. ``face_num`` is
        itself a transcript-derived column (``labels.py`` reads
        ``words_df['face_num']``, absent from the vendored CSV), so it needs
        enrichment like every continuous-valued task (H7)."""
        return not all(t == "speech" for t in self.tasks)

    def _resolve_bt_root(self) -> str | None:
        return self.bt_root or os.environ.get("ROOT_DIR_BRAINTREEBANK")

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        ieeg = events.loc[events["type"] == "Ieeg"]
        if ieeg.empty:
            raise RuntimeError("BTWordEvents: no Ieeg events to derive Word events from")

        enrich = self._enrich_needed()
        bt_root = self._resolve_bt_root()

        all_word_rows: list[pd.DataFrame] = []
        for _, ieeg_row in ieeg.iterrows():
            subject_id = int(ieeg_row["subject_id"])
            trial_id = int(ieeg_row["trial_id"])
            if not self._timeline_is_used(subject_id, trial_id):
                continue
            words_df, nonverbal_df = _load_words_and_nonverbal(
                subject_id, trial_id, bt_root=bt_root, enrich=enrich,
            )
            timeline_rows = _word_event_rows(
                subject_id=subject_id,
                trial_id=trial_id,
                timeline=str(ieeg_row["timeline"]),
                words_df=words_df,
                nonverbal_df=nonverbal_df,
                tasks=self.tasks,
                binary_tasks=self.binary_tasks,
                lite=self.lite,
                nano=self.nano,
                random_seed=self.random_seed,
                duration=self.duration,
            )
            all_word_rows.append(timeline_rows)

        if not all_word_rows:
            raise RuntimeError(
                f"BTWordEvents: eval_mode={self.eval_mode} "
                f"test=({self.test_subject_id}, {self.test_trial_id}) "
                f"matched zero usable timelines"
            )
        words = pd.concat(all_word_rows, ignore_index=True)

        if self.eval_mode == "CrossSession":
            words = _assign_cross_session_split(
                words,
                test_subject_id=self.test_subject_id,
                test_trial_id=self.test_trial_id,
            )
        else:  # CrossSubject
            words = _assign_cross_subject_split(
                words,
                test_subject_id=self.test_subject_id,
                test_trial_id=self.test_trial_id,
                train_subject_id=self.train_subject_id,
                train_trial_id=self.train_trial_id,
            )
        return pd.concat([events, words], ignore_index=True)

    def _timeline_is_used(self, subject_id: int, trial_id: int) -> bool:
        if self.eval_mode == "CrossSession":
            return subject_id == self.test_subject_id
        is_train = (subject_id == self.train_subject_id and
                    trial_id == self.train_trial_id)
        is_test = (subject_id == self.test_subject_id and
                   trial_id == self.test_trial_id)
        return is_train or is_test
