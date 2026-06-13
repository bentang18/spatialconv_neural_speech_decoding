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

import numpy as np
import pandas as pd

from neuralset.events.study import EventsTransform

from speech_decoding.studies.braintreebank.labels import (
    NEUROPROBE_TASKS,
    NEW_PITCH_VOLUME_COLUMNS,
    derive_label_indices,
    enrich_words_with_transcript_features,
    load_pitch_volume_features,
    ordered_dataset_labels,
    ordered_dataset_source_indices,
    remap_task_column,
)
from speech_decoding.studies.braintreebank.manifest import (
    V14_PRETRAIN_SESSIONS,
    bt_subject_native_rate_hz,
)


# "Pretrain" is the leakage-free SSL/distill split: every legal pretraining
# session (V14_PRETRAIN_SESSIONS, disjoint from the eval set) goes to train,
# minus a deterministic per-session positional tail held out for pretext-loss
# monitoring (val/test). It carries NO Neuroprobe eval session, so the SSL
# corpus is fully decoupled from the leaderboard eval split. The supervised P4
# probe keeps CrossSession/CrossSubject; routed per-phase by dispatch_v14.
EvalMode = tp.Literal["WithinSession", "CrossSession", "CrossSubject", "Pretrain"]


_UPSTREAM_TRAIN_SUBJECT_ID = 2
_UPSTREAM_TRAIN_TRIAL_ID = 4

# Mirrors neuroprobe.config.NEUROPROBE_LITE_N_FOLDS (== NEUROPROBE_NANO_N_FOLDS
# == 2). WithinSession runs KFold(n_splits=_LITE_N_FOLDS, shuffle=False) over
# each task's interleaved item order, exactly as upstream
# generate_splits_within_session.
_LITE_N_FOLDS = 2

# Default fraction of each legal session's clips held out for SSL pretext-loss
# monitoring — applied to val AND to test (so ~2× this is held out per session,
# the remainder is train). NOT a leaderboard quantity: it only carves legal
# pretraining data into train vs. monitoring, never touches eval sessions.
DEFAULT_PRETRAIN_HOLDOUT_FRACTION = 0.05

_PRETRAIN_SESSION_SET = frozenset(V14_PRETRAIN_SESSIONS)


def _neural_sample_rate(subject_id: int) -> float:
    """BT neural-clock sample rate for ``subject_id`` — the NATIVE distributed-h5
    rate (2048 Hz for every subject except S9 = 1024 Hz). Single source =
    ``BT_SUBJECT_NATIVE_RATE_HZ``.

    ``est_idx`` (both ``words_df`` and ``nonverbal_df``) is indexed on this native
    clock — verified by the est_idx span: S9 max est_idx 7.10M (words) / 6.90M
    (nonverbal) both sit inside the 7.376M-sample native h5, i.e. the 1024 grid,
    NOT a 2048 grid (which would reach ~14.7M). So onset seconds =
    ``est_idx / _neural_sample_rate(subject_id)``. ``loader.bt_load_raw`` resamples
    a non-2048 subject UP to 2048 while preserving wall-clock time, so slicing the
    resampled stream at these native-derived seconds lands on the right sample.

    Pure registry lookup (no ``neuroprobe.config`` import) → works on laptops
    without BT data mounted, the unit-test path."""
    return float(bt_subject_native_rate_hz(subject_id))


def _vendored_csv_dir() -> Path:
    """Return Neuroprobe's vendored ``braintreebank_features_time_alignment``."""
    from neuroprobe.config import SAVE_SUBJECT_TRIAL_DF_DIR

    return Path(SAVE_SUBJECT_TRIAL_DF_DIR)


def _movie_name(subject_id: int, trial_id: int) -> str:
    from neuroprobe.config import BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING

    return BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"btbank{subject_id}_{trial_id}"]


def _tasks_need_pitch_volume(tasks: tp.Sequence[str]) -> bool:
    """True if any task's feature column lives in Neuroprobe's packaged
    pitch/volume JSON (``enhanced_pitch`` etc.) rather than the transcript
    ``features.csv``. Among the 15 leaderboard tasks only ``pitch`` qualifies,
    but route generically off the column set so a future ``raw_pitch``-style
    task is covered."""
    return any(remap_task_column(t) in NEW_PITCH_VOLUME_COLUMNS for t in tasks)


def _load_pitch_volume_features(
    subject_id: int, trial_id: int
) -> dict[str, dict[str, tp.Any]]:
    """Load Neuroprobe's packaged ``{movie}_pitch_volume_features.json`` for this
    trial (5-dp ``start``-time keyed). Mirrors upstream ``datasets.py``: the file
    ships inside the neuroprobe package (``PITCH_VOLUME_FEATURES_DIR``), NOT under
    the BT data root."""
    from neuroprobe.config import PITCH_VOLUME_FEATURES_DIR

    movie = _movie_name(subject_id, trial_id)
    path = Path(PITCH_VOLUME_FEATURES_DIR) / f"{movie}_pitch_volume_features.json"
    return load_pitch_volume_features(path)


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


def _load_neural_to_movie_map(
    subject_id: int, trial_id: int, bt_root: str | Path | None
) -> tuple[np.ndarray, np.ndarray] | None:
    """BT trigger-track neural→movie map for ``(subject, trial)``.

    Returns sorted, index-deduplicated ``(neural_sample_index, movie_time_s)``
    arrays for ``np.interp(est_idx) -> movie_onset_s``. This is BT's AUTHORITATIVE
    frame-trigger alignment — ``{bt_root}/subject_timings/sub_{N}_trial{T:03d}_timings.csv``,
    the SAME file BT's own ``trial_data_reader.estimate_sample_index`` inverts to
    derive ``words_df.est_idx`` — so interpolating it is the most authoritative
    neural↔movie source that exists (it reproduces ``words_df.start`` to ~2.5 ms).

    Unlike the sparse ``words_df`` (est_idx→start) map, the trigger track carries
    explicit ``pause``/``unpause`` rows, so ``movie_time`` FREEZES across a
    recording pause instead of being linearly bridged. That freeze is exactly the
    fix for the nonverbal pause-bridging residual (sparse-words interp was up to
    89 s off-movie for ~21% of nonverbal anchors, which live in the word-free
    pause stretches the sparse map cannot bracket).

    ``bt_root is None`` (laptop unit tests, no BT mounted) → ``None``; the caller
    then falls back to the sparse words map. ``bt_root`` set but the file missing
    → raises, so a misconfigured BT data root fails LOUDLY rather than silently
    regressing every nonverbal teacher target to the pause-bridging path.
    """
    if bt_root is None:
        return None
    path = (
        Path(bt_root)
        / "subject_timings"
        / f"sub_{subject_id}_trial{trial_id:03d}_timings.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"BTWordEvents: trigger track missing for (subject {subject_id}, "
            f"trial {trial_id}): {path}. It keys the Whisper teacher on the movie "
            "clock (movie_onset_s); check ROOT_DIR_BRAINTREEBANK/subject_timings/."
        )
    df = pd.read_csv(path)[["index", "movie_time"]].dropna().sort_values("index")
    # np.interp needs strictly-increasing xp. Some rows share an `index` (a pause
    # boundary collapses a trigger and the pause row onto one sample; a few sessions
    # also carry plain trigger-row index ties). keep="last" leaves xp strictly
    # increasing for all 26 sessions; the dropped twin's movie_time differs by at
    # most one trigger spacing (~85 ms), and both bracket the same sample, so either
    # is a valid knot — the choice is sub-frame at the 8 Hz teacher rate.
    df = df.drop_duplicates(subset="index", keep="last")
    xp = df["index"].to_numpy(dtype=float)
    yp = df["movie_time"].to_numpy(dtype=float)
    _assert_trigger_track_sane(xp, yp, path)
    return xp, yp


# This map keys the Whisper teacher on the movie clock (`np.interp(est_idx)->yp`).
# `dropna()` above strips NaN knots but NOT inf, and never checks that the movie
# axis is sane — a content-corrupted, truncated, or wrongly-concatenated trigger
# CSV would otherwise yield a wrong-but-finite `movie_onset_s` that silently
# mis-aligns every P3 teacher target (the alignment S9-class: plausible, no
# crash). These bounds are calibrated against all 26 vendored tracks: every one
# is finite, has strictly-increasing `index`, spans >30 s of movie, and its
# `movie_time` never steps backward by more than ~57 ms (trigger jitter; one
# spacing ~85 ms). We tolerate sub-second jitter but fail loud on seconds-scale
# non-monotonicity / inf / truncation.
_MAX_TRIGGER_BACKSTEP_S = 1.0  # >> 57 ms real jitter, << any real corruption
_MIN_TRIGGER_ROWS = 100
_MIN_TRIGGER_SPAN_S = 30.0


def _assert_trigger_track_sane(xp: np.ndarray, yp: np.ndarray, path: Path) -> None:
    """Fail loud on a content-corrupt trigger track (see ledger LG14)."""
    if len(xp) < _MIN_TRIGGER_ROWS:
        raise ValueError(
            f"BT trigger track {path} has only {len(xp)} usable rows "
            f"(< {_MIN_TRIGGER_ROWS}) — truncated/corrupt; it keys the P3 movie clock."
        )
    if not (np.isfinite(xp).all() and np.isfinite(yp).all()):
        raise ValueError(
            f"BT trigger track {path} has non-finite index/movie_time knots after "
            "dropna (inf survives dropna) — corrupt clock map."
        )
    if not np.all(np.diff(xp) > 0):
        raise ValueError(
            f"BT trigger track {path} 'index' axis is not strictly increasing after "
            "sort+dedup — np.interp would misbehave; the dedup contract broke."
        )
    backstep = float(-np.diff(yp).min()) if len(yp) > 1 else 0.0
    if backstep > _MAX_TRIGGER_BACKSTEP_S:
        raise ValueError(
            f"BT trigger track {path} 'movie_time' steps backward by {backstep:.3f}s "
            f"(> {_MAX_TRIGGER_BACKSTEP_S}s tolerance) — the movie clock is non-monotone "
            "(corrupt/concatenated track); P3 teacher targets would mis-align."
        )
    span = float(yp[-1] - yp[0])
    if span < _MIN_TRIGGER_SPAN_S:
        raise ValueError(
            f"BT trigger track {path} movie_time spans only {span:.1f}s "
            f"(< {_MIN_TRIGGER_SPAN_S}s) — a real film clock is minutes long; truncated."
        )


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
    balance: bool = True,
    pitch_volume_features: dict[str, dict[str, tp.Any]] | None = None,
    neural_to_movie: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Build per-task Word rows.

    ``balance=True`` (P4 eval): class-balanced rows in upstream Dataset item
    order — see the interleaving note below. ``balance=False`` (label-free SSL):
    EVERY word + nonverbal anchor (no cross-class down-sampling), emitted in
    chronological ``start`` order so the per-session positional pretrain split
    (:func:`_assign_pretrain_split`) is a clean temporal holdout. The
    item-order/majority-class hazard below applies only to the balanced eval
    splits, never to the label-free SSL split.

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
    sample_rate = _neural_sample_rate(subject_id)
    # Movie-clock onset for the P3 Whisper-teacher join, computed by ONE mechanism
    # for BOTH verbal and nonverbal anchors. AUTHORITATIVE source = the BT trigger
    # track (`neural_to_movie`): np.interp(est_idx) -> movie_time. It is the same
    # map BT inverts to derive words_df.est_idx, the only source that FREEZES
    # movie_time across recording pauses, and strictly more correct than words_df
    # for the half-rate S9 session. The fallback (no trigger track — laptop tests
    # without BT mounted) is the sparse words_df map: exact for verbal (== start)
    # but it linearly BRIDGES pauses for nonverbal, the residual the trigger track
    # is here to remove.
    if (
        neural_to_movie is None
        and len(words_df)
        and {"est_idx", "start"} <= set(words_df.columns)
    ):
        _wsort = (
            words_df[["est_idx", "start"]]
            .dropna()
            .sort_values("est_idx")
            .drop_duplicates(subset="est_idx", keep="last")
        )
        _fb_x = _wsort["est_idx"].to_numpy(dtype=float)
        _fb_y = _wsort["start"].to_numpy(dtype=float)
    else:
        _fb_x = _fb_y = None

    def _movie_onset(est_idx: float, is_nonverbal: bool, source: pd.Series) -> float:
        if neural_to_movie is not None:  # authoritative trigger track (all anchors)
            nm_idx, nm_t = neural_to_movie
            return float(np.interp(est_idx, nm_idx, nm_t))
        if is_nonverbal and _fb_x is not None:  # fallback: sparse words_df interp
            return float(np.interp(est_idx, _fb_x, _fb_y))
        return float(source["start"])  # fallback verbal: words_df start IS movie clock
    for task in tasks:
        label_indices = derive_label_indices(
            words_df=words_df,
            nonverbal_df=nonverbal_df,
            task=task,
            binary_tasks=binary_tasks,
            lite=lite,
            nano=nano,
            random_seed=random_seed,
            balance=balance,
            pitch_volume_features=pitch_volume_features,
        )
        if balance:
            # Eval parity: interleave classes in upstream Dataset item order.
            ordered_pairs = zip(
                ordered_dataset_labels(label_indices),
                ordered_dataset_source_indices(label_indices),
            )
        else:
            # SSL: every anchor, no balanced interleaving (it assumes equal
            # class sizes). Order is irrelevant here — rows are sorted by
            # ``start`` before return for the temporal pretrain split.
            ordered_pairs = (
                (label, src)
                for label, idxs in label_indices.items()
                for src in idxs
            )
        for class_id_arr, src_idx_arr in ordered_pairs:
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
            est_idx = float(source["est_idx"])
            if not np.isfinite(est_idx):
                # est_idx is the NEURAL clock: it sets both the voltage window onset
                # (`start = est_idx/SR`) and the movie-onset interp. A NaN/inf est_idx
                # (corrupt words_df/nonverbal_df) would silently mis-window every clip
                # for this anchor. Real data has ZERO non-finite est_idx across all 26
                # words_df (202792 rows) + nonverbal_df (115990) so this never false-
                # fires; the upstream `.dropna()` is defensive only (ledger LG14h).
                raise ValueError(
                    f"non-finite est_idx for {'nonverbal' if is_nonverbal else 'word'} "
                    f"row {src_idx} (subject {subject_id}, trial {trial_id}, task {task}): "
                    "a corrupt words_df/nonverbal_df est_idx silently mis-windows the "
                    "neural clip AND the P3 teacher onset (ledger LG14h)."
                )
            movie_onset = _movie_onset(est_idx, is_nonverbal, source)
            rows.append(
                {
                    "type": "Word",
                    "start": est_idx / sample_rate,
                    "duration": float(duration),
                    "text": text,
                    "task": task,
                    "label": class_id,
                    "subject_id": str(subject_id),
                    "trial_id": str(trial_id),
                    "timeline": timeline,
                    # MOVIE-clock onset (seconds into the movie audio), for the P3
                    # Whisper-teacher join (WS-H / WhisperTargetExtractor). The
                    # neural window slices at `start` (neural clock, est_idx/SR);
                    # the audio-keyed teacher cache is indexed by movie time, so it
                    # MUST slice at `movie_onset_s`. Computed for BOTH verbal and
                    # nonverbal anchors by `_movie_onset` above — interp over the BT
                    # trigger track (authoritative; freezes across pauses) when it is
                    # available, else the sparse words_df map. The two clocks diverge
                    # by the per-trial neural-vs-movie drift (235-904 s, FLAG 9).
                    # (Legacy NONVERBAL note: nonverbal_df has no movie
                    # clock, so emitting its own `start` here would key the teacher
                    # 235-904 s off-movie (the 2026-06-08 P3 alignment bug).
                    "movie_onset_s": movie_onset,
                }
            )
    if not rows:
        return pd.DataFrame(
            {col: pd.Series(dtype=object) for col in (
                "type", "start", "duration", "text", "task", "label",
                "subject_id", "trial_id", "timeline", "movie_onset_s",
            )}
        )
    out = pd.DataFrame(rows)
    if not balance:
        # Chronological so the positional pretrain split is a temporal holdout
        # (test = movie tail). Stable sort keeps determinism across tasks.
        out = out.sort_values("start", kind="stable")
    return out.reset_index(drop=True)


def _assign_within_session_split(
    df: pd.DataFrame,
    *,
    test_subject_id: int,
    test_trial_id: int,
    fold_index: int,
    n_folds: int = _LITE_N_FOLDS,
) -> pd.DataFrame:
    """Single-trial K-fold split, emitting ONE fold (``fold_index``).

    Mirrors upstream ``generate_splits_within_session`` exactly: for each task's
    interleaved item order on (``test_subject_id``, ``test_trial_id``), run
    ``KFold(n_splits=n_folds, shuffle=False)`` (the same call upstream makes; the
    shuffle=False contiguity is what avoids correlated train/test). The selected
    fold's held-out indices are then halved — first half → ``val``, second half →
    ``test`` — and the complementary fold indices become ``train``. Dispatch runs
    one cell per fold; the collector means metrics over folds. Other subjects /
    trials are dropped (``_timeline_is_used`` already restricts to the one
    timeline, but we re-mask here for safety).
    """
    from sklearn.model_selection import KFold

    if not 0 <= fold_index < n_folds:
        raise ValueError(
            f"within_session fold_index must be in [0, {n_folds}); got {fold_index}"
        )
    if df.empty:
        df = df.copy()
        df["split"] = pd.Series(dtype=str)
        return df
    out = df.copy()
    out["split"] = ""
    s = out["subject_id"].astype(int)
    t = out["trial_id"].astype(int)
    is_test_st = (s == test_subject_id) & (t == test_trial_id)
    for task in out.loc[is_test_st, "task"].unique():
        sub_idx = out.index[is_test_st & (out["task"] == task)]
        n = len(sub_idx)
        if n < n_folds:
            continue  # too few items to form the requested folds (upstream skips)
        kf = KFold(n_splits=n_folds, shuffle=False)
        train_pos, test_pos = list(kf.split(range(n)))[fold_index]
        if len(train_pos) == 0 or len(test_pos) == 0:
            continue  # upstream `continue`s on empty folds
        val_size = len(test_pos) // 2
        val_pos = test_pos[:val_size]
        test_only_pos = test_pos[val_size:]
        out.loc[sub_idx[train_pos], "split"] = "train"
        out.loc[sub_idx[val_pos], "split"] = "val"
        out.loc[sub_idx[test_only_pos], "split"] = "test"
    return out.loc[out["split"] != ""].reset_index(drop=True)


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


def _assign_pretrain_split(
    df: pd.DataFrame,
    *,
    holdout_fraction: float,
) -> pd.DataFrame:
    """Leakage-free SSL split over the legal pretraining corpus.

    Every row is a clip anchor from a ``V14_PRETRAIN_SESSIONS`` session (the
    study emits only those under ``mode="pretrain"``). For each (subject, trial)
    we carve a deterministic positional tail — the last ``holdout_fraction`` of
    that session's rows → ``test``, the preceding ``holdout_fraction`` → ``val``,
    the rest → ``train``. Unlike the CrossSession/CrossSubject splits this does
    NOT interleave by task or balance classes: SSL is label-free, so the only
    requirements are determinism and that no eval session is present (guaranteed
    by the corpus, re-checked by the leakage guard). Sessions too small to give
    all three splits (< 3 rows) stay entirely in ``train``.
    """
    # Validate before the empty short-circuit so the contract is identical for
    # empty and non-empty input (audit D3).
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError(
            "pretrain holdout_fraction must be in (0, 0.5); got "
            f"{holdout_fraction}"
        )
    if df.empty:
        df = df.copy()
        df["split"] = pd.Series(dtype=str)
        return df
    # reset_index so the positional tail is assigned by position regardless of
    # the caller's index (audit D2: label-based .loc mis-assigns on a non-unique
    # index; production feeds a clean RangeIndex, but the helper must be robust).
    out = df.reset_index(drop=True)
    out["split"] = "train"
    groups = out.groupby(["subject_id", "trial_id"], sort=False).groups
    for positions in groups.values():
        idx = list(positions)
        n = len(idx)
        if n < 3:
            continue  # too small to hold out — keep all as train
        # >=1 each: cap the per-split holdout so train keeps at least one row.
        n_hold = max(1, min(round(n * holdout_fraction), (n - 1) // 2))
        test_rows = idx[n - n_hold:]
        val_rows = idx[n - 2 * n_hold: n - n_hold]
        out.loc[test_rows, "split"] = "test"
        out.loc[val_rows, "split"] = "val"
    # Global non-emptiness: Data.build requires every split to have >=1 segment
    # across the whole corpus; otherwise it raises a misdirected "Split has no
    # segments" far from the cause (audit D1). Fail here with the real reason.
    missing = {"val", "test"} - set(out["split"])
    if missing:
        raise ValueError(
            f"pretrain split produced no {sorted(missing)} rows across the "
            "whole corpus (every session had < 3 clips at "
            f"holdout_fraction={holdout_fraction}). Downstream Data.build "
            "requires non-empty train/val/test; raise holdout_fraction or "
            "check the corpus."
        )
    return out


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
    # Class-balance the emitted anchors. True (P4 eval): Neuroprobe-parity
    # balanced + interleaved. False (label-free SSL P1/P2/P3): keep EVERY word +
    # nonverbal anchor (no cross-class min, no eval-parity interleave) →
    # ~96-99% movie coverage instead of the minority-class-bottlenecked subset.
    balance: bool = True
    eval_mode: EvalMode = "CrossSession"
    # WithinSession only: which KFold fold (0..n_folds-1) this cell emits.
    # Ignored for CrossSession/CrossSubject/Pretrain.
    fold_index: int = 0
    test_subject_id: int = _UPSTREAM_TRAIN_SUBJECT_ID
    test_trial_id: int = _UPSTREAM_TRAIN_TRIAL_ID
    train_subject_id: int = _UPSTREAM_TRAIN_SUBJECT_ID
    train_trial_id: int = _UPSTREAM_TRAIN_TRIAL_ID
    bt_root: str | None = None
    duration: float = 1.0
    random_seed: int = 42
    # eval_mode="Pretrain" only: per-session tail fraction held out for SSL
    # pretext-loss monitoring (val AND test). Ignored for CrossSession/Subject.
    pretrain_holdout_fraction: float = DEFAULT_PRETRAIN_HOLDOUT_FRACTION

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
            neural_to_movie = _load_neural_to_movie_map(subject_id, trial_id, bt_root)
            pitch_volume_features = (
                _load_pitch_volume_features(subject_id, trial_id)
                if _tasks_need_pitch_volume(self.tasks)
                else None
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
                balance=self.balance,
                pitch_volume_features=pitch_volume_features,
                neural_to_movie=neural_to_movie,
            )
            all_word_rows.append(timeline_rows)

        if not all_word_rows:
            raise RuntimeError(
                f"BTWordEvents: eval_mode={self.eval_mode} "
                f"test=({self.test_subject_id}, {self.test_trial_id}) "
                f"matched zero usable timelines"
            )
        words = pd.concat(all_word_rows, ignore_index=True)

        if self.eval_mode == "WithinSession":
            words = _assign_within_session_split(
                words,
                test_subject_id=self.test_subject_id,
                test_trial_id=self.test_trial_id,
                fold_index=self.fold_index,
            )
        elif self.eval_mode == "CrossSession":
            words = _assign_cross_session_split(
                words,
                test_subject_id=self.test_subject_id,
                test_trial_id=self.test_trial_id,
            )
        elif self.eval_mode == "Pretrain":
            words = _assign_pretrain_split(
                words,
                holdout_fraction=self.pretrain_holdout_fraction,
            )
        elif self.eval_mode == "CrossSubject":
            words = _assign_cross_subject_split(
                words,
                test_subject_id=self.test_subject_id,
                test_trial_id=self.test_trial_id,
                train_subject_id=self.train_subject_id,
                train_trial_id=self.train_trial_id,
            )
        else:  # pragma: no cover - guarded by the EvalMode Literal
            raise ValueError(f"BTWordEvents: unknown eval_mode {self.eval_mode!r}")
        return pd.concat([events, words], ignore_index=True)

    def _timeline_is_used(self, subject_id: int, trial_id: int) -> bool:
        if self.eval_mode == "Pretrain":
            # Defense-in-depth: only legal pretraining sessions are usable, even
            # if an eval session somehow reaches this transform. The study's
            # mode="pretrain" already restricts the corpus; this is the second
            # gate before the runtime leakage guard.
            return (subject_id, trial_id) in _PRETRAIN_SESSION_SET
        if self.eval_mode == "WithinSession":
            return (subject_id == self.test_subject_id and
                    trial_id == self.test_trial_id)
        if self.eval_mode == "CrossSession":
            return subject_id == self.test_subject_id
        is_train = (subject_id == self.train_subject_id and
                    trial_id == self.train_trial_id)
        is_test = (subject_id == self.test_subject_id and
                   trial_id == self.test_trial_id)
        return is_train or is_test
