"""Probe-dataset core for the online linear probe (spec §2/§3/§6).

Two concerns, kept apart:

1. **Pure, verifiable logic** (this module, laptop-testable): the dense per-window
   ±1 task labels (matching upstream ``derive_label_indices`` class assignment),
   the deterministic ``min(N_cap, available)`` window selection, and the
   :class:`InMemoryProbeDataset` that assembles per-subject
   :class:`~speech_decoding.experiments.online_probe.SubjectProbeData` (with the
   §6 firewall assert) for :func:`~speech_decoding.experiments.online_probe.run_probe`
   to consume.
2. **DCC-only extraction** (turning word-onset windows into encoder-ready 3STFT
   band tensors via the real study/view) — wired in :func:`build_probe_dataset`,
   verified on DCC, not exercised here.

The shared-window design (spec §2: "multiple labels, one window set"): a subject's
windows are forwarded ONCE; each task labels only the windows it covers. Continuous
tasks (``delta_volume``→``delta_rms``, ``word_length``) keep the top quartile (+1)
and bottom quartile (−1), dropping the middle 50% to NaN; ``word_position`` (=
upstream ``word_index``) is first-word(+1) vs second-word(−1). ``run_probe`` masks
the NaN rows per task.
"""

from __future__ import annotations

import os
import typing as tp

import numpy as np
import pandas as pd

from speech_decoding.experiments.bad_windows import (
    filter_events_by_bad_windows,
    load_bad_windows,
)
from speech_decoding.experiments.online_probe import SubjectProbeData, assert_firewall
from speech_decoding.studies.braintreebank.labels import (
    SINGLE_FLOAT_TASK_COLUMNS,
    _assert_finite_labels,
    remap_task_column,
)
from speech_decoding.studies.braintreebank.manifest import (
    BT_LITE_SESSIONS,
    bt_subject_native_rate_hz,
)

if tp.TYPE_CHECKING:  # avoid importing the heavy Data/ns chain on the laptop test path
    from speech_decoding.experiments.data import Data

# The three probe tasks (spec §2). ``word_position`` is the spec's name for the
# upstream ``word_index`` task (first-vs-second word in sentence) — NEUROPROBE has
# no literal "word_position" key; the decision-log "abstract linguistic (position)"
# (spec §11) is word-index. Flagged for Ben; trivially re-pointed via this alias.
PROBE_TASK_ALIASES: dict[str, str] = {"word_position": "word_index"}
PROBE_TASKS: tuple[str, ...] = ("delta_volume", "word_length", "word_position")

# Cohort (spec §9): 7 firewall-legal pretrain subjects; sub2-anchored CS.
WS_SUBJECTS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 9)
CS_ANCHOR: int = 2
CS_TEST_SUBJECTS: tuple[int, ...] = (1, 3, 4, 6, 8, 9)
N_CAP: int = 3500  # NEUROPROBE_LITE_MAX_SAMPLES

# Probe clip contract (Ben 2026-06-18): 1 s word-onset clips, NO neural lag. The
# converged arch is 1 s-fixed (38 tokens), so the probe forwards exactly the clip
# the encoder is built for. These also set the width of the bad-window overlap test
# (``filter_probe_events``) — the probe respects the CLIP sidecars at ITS clip width.
PROBE_CLIP_START_S: float = 0.0
PROBE_CLIP_DUR_S: float = 1.0

# Segmenter keys the probe re-materializes at 1 s (the encoder-ready band tensors +
# the anatomy the pooling needs). The run's ``target`` / ``whisper_target`` extractors
# are dropped: probe labels come from the words_df, not the segmenter target.
_PROBE_SEGMENTER_KEYS: tuple[str, ...] = (
    "electrode_tokens_slow",
    "electrode_tokens_beta",
    "electrode_tokens_hg",
    "support",
    "valid_mask",
)


def _upstream_task(task: str) -> str:
    return PROBE_TASK_ALIASES.get(task, task)


def pm1_labels(words_df: pd.DataFrame, task: str) -> np.ndarray:
    """Dense ``(N,)`` ±1 / NaN labels for one task over the SHARED window set.

    Matches upstream's binary class assignment exactly (``labels.py``
    ``derive_label_indices``): continuous tasks use the empirical-CDF percentile
    ``mean(values < v)`` and assign **+1** above the 75th pct, **−1** below the 25th,
    **NaN** (excluded from the fit) in the middle 50%; ``word_index`` assigns +1 to
    the first word in a sentence (``idx_in_sentence == 0``), −1 to the second
    (``== 1``), NaN otherwise. Class 1↔+1 / class 0↔−1 mirrors the upstream dict
    keys (verified by :mod:`test_online_probe_dataset` against the real function)."""
    ut = _upstream_task(task)
    out = np.full(len(words_df), np.nan, dtype=np.float64)
    if ut in SINGLE_FLOAT_TASK_COLUMNS:
        col = remap_task_column(ut)
        vals = _assert_finite_labels(
            np.asarray(words_df[col].to_numpy(), dtype=float), col, "features.csv"
        )
        pct = np.array([np.mean(vals < v) for v in vals])
        out[pct > 0.75] = 1.0
        out[pct < 0.25] = -1.0
    elif ut == "word_index":
        idx = np.asarray(words_df["idx_in_sentence"].to_numpy(), dtype=float)
        out[idx == 0.0] = 1.0
        out[idx == 1.0] = -1.0
    else:
        raise KeyError(f"probe task {task!r} -> upstream {ut!r} is not supported")
    return out


def probe_feature_columns(tasks: tp.Sequence[str] = PROBE_TASKS) -> tuple[str, ...]:
    """The transcript-feature columns :func:`pm1_labels` reads, derived from the
    probe tasks (deduped, order-preserving). Continuous tasks map through
    ``remap_task_column`` (``delta_volume`` → ``delta_rms``, ``word_length`` →
    ``word_length``); ``word_position`` → upstream ``word_index`` → ``idx_in_sentence``.
    This is the exact set :func:`attach_probe_features` must re-join onto the
    forwarded windows before labelling."""
    cols: list[str] = []
    for task in tasks:
        ut = _upstream_task(task)
        col = remap_task_column(ut) if ut in SINGLE_FLOAT_TASK_COLUMNS else "idx_in_sentence"
        if col not in cols:
            cols.append(col)
    return tuple(cols)


# Recovered-est_idx tolerance (samples). ``start`` was emitted as the EXACT float
# ``est_idx / native_rate`` (word_events.py), so ``start * native_rate`` round-trips
# to the integer est_idx with float64 error << 1 sample. A larger residual means the
# window's clock disagrees with the words_df clock — a real join bug, fail loud.
_EST_IDX_TOL_SAMPLES: float = 1e-3


def attach_probe_features(
    windows: pd.DataFrame,
    enriched_words: dict[tuple[int, int], pd.DataFrame],
    *,
    feature_columns: tp.Sequence[str] | None = None,
    native_rate_hz: tp.Callable[[int], float] = bt_subject_native_rate_hz,
) -> pd.DataFrame:
    """Re-attach the probe's transcript-feature columns to each forwarded window.

    The emitted ``Word`` events (``word_events.py``) carry only the binarized
    ``label`` for the RUN's single training task — the raw feature columns the
    probe binarizes itself (``delta_rms`` / ``word_length`` / ``idx_in_sentence``)
    are dropped. This re-joins them WITHOUT touching the live event schema: each
    window's neural-clock ``est_idx`` is recovered from its ``start``
    (``= est_idx / native_rate``; native = 2048 Hz, S9 = 1024 Hz) and matched
    against the per-session enriched ``words_df`` (carrying ``est_idx`` + every
    ``features.csv`` column). Fails loud if a window's est_idx has no match
    (broken join) or the recovered est_idx isn't integral (clock disagreement) —
    a silent NaN here would corrupt the probe label set, exactly the class of bug
    the data-pipeline bug-hunt guards against.

    ``windows``: forwarded clips, one row each (``start`` neural-clock seconds,
    ``subject_id``, ``trial_id``). ``enriched_words``: ``(subject_id, trial_id)`` →
    enriched ``words_df``. Returns ``windows`` with ``feature_columns`` attached,
    row order preserved."""
    if feature_columns is None:
        feature_columns = probe_feature_columns()
    feature_columns = tuple(feature_columns)
    out = windows.reset_index(drop=True).copy()
    sid = _subject_id_int(out["subject_id"])
    tid = np.asarray(out["trial_id"]).astype(int)
    starts = out["start"].to_numpy(dtype=float)
    for col in feature_columns:
        out[col] = np.full(len(out), np.nan, dtype=float)
    for s, t in {(int(a), int(b)) for a, b in zip(sid, tid)}:
        rows = np.flatnonzero((sid == s) & (tid == t))
        rate = float(native_rate_hz(s))
        recovered = starts[rows] * rate
        est_idx = np.rint(recovered).astype(np.int64)
        drift = np.abs(recovered - est_idx)
        if drift.max(initial=0.0) > _EST_IDX_TOL_SAMPLES:
            bad = int(np.argmax(drift))
            raise ValueError(
                f"probe window in session ({s},{t}) has start={starts[rows][bad]} "
                f"that does not round-trip to an integer est_idx at {rate} Hz "
                f"(residual {drift[bad]:.4g} samples) — neural-clock mismatch."
            )
        ew = enriched_words.get((s, t))
        if ew is None:
            raise KeyError(
                f"attach_probe_features: no enriched words_df supplied for session "
                f"({s},{t}); cannot label its {len(rows)} probe windows."
            )
        lut = ew.drop_duplicates(subset="est_idx").set_index("est_idx")
        missing_cols = [c for c in feature_columns if c not in lut.columns]
        if missing_cols:
            raise KeyError(
                f"enriched words_df for session ({s},{t}) is missing probe feature "
                f"columns {missing_cols}; was it loaded with enrich=True?"
            )
        missing = np.setdiff1d(est_idx, lut.index.to_numpy())
        if missing.size:
            raise KeyError(
                f"{missing.size} probe windows in session ({s},{t}) have no matching "
                f"est_idx in the enriched words_df (e.g. {int(missing[0])}) — the "
                "neural→transcript join broke."
            )
        sub = lut.loc[est_idx, list(feature_columns)]
        for col in feature_columns:
            out.loc[rows, col] = sub[col].to_numpy()
    return out


def select_window_indices(n_available: int, n_cap: int, seed: int) -> np.ndarray:
    """Deterministic ``min(n_cap, n_available)`` window selection (spec §2/§3).

    Returns SORTED indices — time order is preserved so the WS contiguous 2-fold
    never straddles the same autocorrelated stretch (spec §5). Seeded from
    ``(seed)`` so the probe set is identical across resume (spec §8)."""
    if n_available <= n_cap:
        return np.arange(n_available)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_available, size=n_cap, replace=False))


def filter_probe_events(
    events: pd.DataFrame, bad_window_dir: str | None
) -> pd.DataFrame:
    """Drop probe windows overlapping a CLIP bad-window span (Ben 2026-06-18:
    "Probe should also respect clip bad-window sidecars").

    Reuses the Layer-2 filter (:func:`bad_windows.filter_events_by_bad_windows`),
    but at the PROBE clip contract — ``start=0.0`` / ``dur=1.0`` — so the overlap
    test matches the 1 s clip the encoder actually sees, NOT the 5 s SSL clip the
    sidecars were scanned for. ``bad_window_dir is None`` (a run with no CLIP layer)
    → no filtering. A session with no sidecar keeps all its windows."""
    if bad_window_dir is None:
        return events.reset_index(drop=True)
    return filter_events_by_bad_windows(
        events,
        load_bad_windows(bad_window_dir),
        clip_start_s=PROBE_CLIP_START_S,
        clip_dur_s=PROBE_CLIP_DUR_S,
        # Only Word anchors are droppable; the continuous Ieeg row (start=0) must
        # survive — see filter_events_by_bad_windows. Matches the probe segmenter.
        trigger_query="type == 'Word'",
    )


def select_subject_window_positions(
    triggers: pd.DataFrame, subject_id: int, *, n_cap: int, seed: int
) -> np.ndarray:
    """Positional indices into ``triggers`` of the windows kept for one subject:
    that subject's rows, deterministically capped to ``min(n_cap, available)``
    (spec §2/§3). Time order is preserved (``select_window_indices`` returns sorted
    indices) so the WS contiguous 2-fold never straddles the same autocorrelated
    stretch. An absent subject → empty array.

    ``subject_id`` is compared as ``int``: the real BT study types the column as
    strings ('2') while the cohort constants are ints — without coercion the
    comparison is silently always-false (DCC #241 anchor-missing crash)."""
    pos = np.flatnonzero(_subject_id_int(triggers["subject_id"]) == int(subject_id))
    keep = select_window_indices(len(pos), n_cap, seed)
    return pos[keep]


def _subject_id_int(subject_id: tp.Any) -> np.ndarray:
    """Subject-id column as an int array. BT subjects are integers 1–10 whether the
    study stored them as ints or strings — one coercion point for every comparison."""
    return np.asarray(subject_id).astype(int)


def present_subject_ids(triggers: pd.DataFrame) -> set[int]:
    """The int subject-ids present in ``triggers`` (dtype-robust). Intersect the int
    cohort sets against THIS so a string-typed column still matches (DCC #241)."""
    if len(triggers) == 0:
        return set()
    return set(np.unique(_subject_id_int(triggers["subject_id"])).tolist())


class InMemoryProbeDataset:
    """Concrete :class:`ProbeDataset` from already-extracted per-subject tensors.

    Each ``per_subject[sid]`` is a dict with the encoder-ready band tensors
    (``slow``/``beta``/``hg``), ``parcel_per_electrode`` / ``electrode_mask``
    ``(C,)``, the ``words_df`` (carrying the task feature columns), and the
    ``sessions`` the windows came from. Task ±1 labels are derived once at
    construction via :func:`pm1_labels`; the §6 firewall asserts no window came
    from a lite/leaderboard cell. ``run_probe`` consumes this directly."""

    def __init__(
        self,
        per_subject: dict[int, dict[str, tp.Any]],
        *,
        n_parcels: int,
        tasks: tp.Sequence[str] = PROBE_TASKS,
        ws_subjects: tp.Sequence[int] = WS_SUBJECTS,
        cs_anchor: int = CS_ANCHOR,
        cs_test_subjects: tp.Sequence[int] = CS_TEST_SUBJECTS,
    ) -> None:
        self.n_parcels = n_parcels
        self.tasks = tuple(tasks)
        self.ws_subjects = tuple(ws_subjects)
        self.cs_anchor = cs_anchor
        self.cs_test_subjects = tuple(cs_test_subjects)
        self._data: dict[int, SubjectProbeData] = {}
        all_sessions: list[tuple[int, int]] = []
        for sid, rec in per_subject.items():
            wdf = rec["words_df"]
            labels = {t: pm1_labels(wdf, t) for t in self.tasks}
            sessions = [tuple(s) for s in rec["sessions"]]
            all_sessions.extend(sessions)
            self._data[sid] = SubjectProbeData(
                subject_id=sid,
                slow=rec["slow"], beta=rec["beta"], hg=rec["hg"],
                parcel_per_electrode=rec["parcel_per_electrode"],
                electrode_mask=rec["electrode_mask"],
                labels=labels, sessions=sessions,
            )
        # §6: hard fail at build time if any probe window is a lite eval cell.
        assert_firewall(all_sessions, BT_LITE_SESSIONS)

    def subject_data(self, subject_id: int) -> SubjectProbeData:
        return self._data[subject_id]


def _probe_segmenter(run_data: "Data") -> tp.Any:
    """A 1 s / zero-lag segmenter that REUSES the run's wired band+anatomy extractors
    (spec §2). Built the same way :class:`Data` coerces its ``segmenter`` field — a
    dict to ``ns.dataloader.Segmenter`` — so the band tensors are byte-identical to
    the run's, only the clip window changes (5 s SSL → 1 s probe, ``start`` → 0.0).
    Drops ``target`` / ``whisper_target`` (probe labels come from the words_df)."""
    import neuralset as ns

    src = run_data.segmenter.extractors
    missing = [k for k in _PROBE_SEGMENTER_KEYS if k not in src]
    if missing:
        raise KeyError(
            f"online probe needs the converged 3STFT segmenter keys {missing} on the "
            f"run's data; got {sorted(src)}. The probe only supports --frontend 3stft."
        )
    return ns.dataloader.Segmenter(
        extractors={k: src[k] for k in _PROBE_SEGMENTER_KEYS},
        # Exclude nonverbal anchors. The `speech`/`onset` SSL task emits its class-0
        # silence anchors as `type=="Word"` with `text=="<nonverbal>"` and an est_idx
        # drawn from `nonverbal_df` (word_events._word_event_rows). The probe's three
        # tasks are all word-level transcript features (delta_volume / word_length /
        # word_position) — nonverbal anchors carry none of them, so they would be
        # all-NaN and masked by run_probe anyway. Forwarding them also trips the
        # attach_probe_features est_idx join (their est_idx is in nonverbal_df, not the
        # enriched words_df) — the "neural→transcript join broke" KeyError that disabled
        # the probe on every --task speech run (#241). Filtering here keeps that loud-fail
        # guard meaningful for REAL verbal-word clock mismatches.
        trigger_query="type == 'Word' and text != '<nonverbal>'",
        start=PROBE_CLIP_START_S,
        duration=PROBE_CLIP_DUR_S,
    )


def _load_enriched_words(
    sessions: tp.Iterable[tuple[int, int]], bt_root: str | None
) -> dict[tuple[int, int], pd.DataFrame]:  # pragma: no cover - DCC-only (needs BT transcripts)
    """Load each session's transcript-enriched ``words_df`` (``est_idx`` + every
    ``features.csv`` column) for :func:`attach_probe_features`. Reuses the study's
    own loader (``enrich=True``) so the feature columns are byte-identical to the
    eval path. DCC-only: needs ``$ROOT_DIR_BRAINTREEBANK/transcripts``."""
    from speech_decoding.studies.braintreebank.word_events import (
        _load_words_and_nonverbal,
    )

    out: dict[tuple[int, int], pd.DataFrame] = {}
    for s, t in {(int(a), int(b)) for a, b in sessions}:
        words_df, _ = _load_words_and_nonverbal(s, t, bt_root=bt_root, enrich=True)
        out[(s, t)] = words_df
    return out


def cartesian_slow_to_5d(slow: "np.ndarray | tp.Any") -> tp.Any:
    """Split the cached cartesian slow band ``(B, C, 2F, T)`` → ``(B, C, 2, F, T)``.

    The 3STFT slow band is cached as the two real components CONCATENATED on the freq
    axis (``[Re(F) ++ Im(F)]``; ``view.py::_single_stft_raw_view`` ``cartesian=True``),
    but the frontend slow stem (``in_channels=2``) and every probe consumer
    (``encode_frontend`` / ``raw_tokens_from_bands``) want a SEPARATE Re/Im channel axis.
    Row-major split → channel 0 = Re(F), channel 1 = Im(F) — byte-identical to the live
    training path (``v14_converged_module._ingest``). ``_materialize_subject`` applies
    this so the stored ``slow`` honors :class:`SubjectProbeData`'s 5-D contract; the
    beta/HG magnitude bands stay 4-D. Fails loud on a non-even freq axis (a sign the
    band was not cached cartesian)."""
    if slow.ndim != 4 or slow.shape[2] % 2 != 0:
        raise ValueError(
            "cartesian slow band must be (B, C, 2F, T) with an even freq axis "
            f"([Re ++ Im]); got shape {tuple(slow.shape)}"
        )
    b, c, f2, t = slow.shape
    return slow.reshape(b, c, 2, f2 // 2, t)


def _materialize_subject(
    dataset: tp.Any,
    triggers: pd.DataFrame,
    positions: np.ndarray,
    *,
    batch_size: int,
    enriched_words: dict[tuple[int, int], pd.DataFrame],
) -> dict[str, tp.Any]:  # pragma: no cover - DCC-only (needs BT voltage)
    """Forward one subject's selected windows through the segmenter and collect the
    encoder-ready band tensors + per-electrode anatomy. Mirrors ``Data.build``'s
    select→DataLoader path (``num_workers=0``: a one-shot in-worker materialization,
    no persistent pool). ``parcel_per_electrode`` / ``electrode_mask`` are constant
    across a subject's windows, so they are read off the first batch."""
    import torch
    from torch.utils.data import DataLoader

    mask = pd.Series(
        np.isin(np.arange(len(triggers)), positions), index=triggers.index
    )
    subset = dataset.select(mask)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=subset.collate_fn,
    )
    bands: dict[str, list[torch.Tensor]] = {"slow": [], "beta": [], "hg": []}
    support0: torch.Tensor | None = None
    valid0: torch.Tensor | None = None
    for batch in loader:
        bands["slow"].append(batch.data["electrode_tokens_slow"])
        bands["beta"].append(batch.data["electrode_tokens_beta"])
        bands["hg"].append(batch.data["electrode_tokens_hg"])
        if support0 is None:
            support0 = batch.data["support"][0]          # (C, K) one-hot
            valid0 = batch.data["valid_mask"][0]         # (C,) bool
    if support0 is None:
        raise RuntimeError("probe subject yielded no windows after selection")
    return {
        "slow": cartesian_slow_to_5d(torch.cat(bands["slow"], dim=0)),
        "beta": torch.cat(bands["beta"], dim=0),
        "hg": torch.cat(bands["hg"], dim=0),
        "parcel_per_electrode": support0.argmax(dim=-1).long(),   # (C,)
        "electrode_mask": valid0.bool(),                          # (C,)
        "words_df": attach_probe_features(
            triggers.iloc[positions].reset_index(drop=True), enriched_words
        ),
        "sessions": sorted(
            {
                (int(s), int(t))
                for s, t in zip(
                    triggers["subject_id"].to_numpy()[positions],
                    triggers["trial_id"].to_numpy()[positions],
                )
            }
        ),
    }


def build_probe_dataset(
    run_data: "Data",
    *,
    n_cap: int = N_CAP,
    seed: int = 0,
    ws_subjects: tp.Sequence[int] = WS_SUBJECTS,
    cs_anchor: int = CS_ANCHOR,
    cs_test_subjects: tp.Sequence[int] = CS_TEST_SUBJECTS,
    batch_size: int = 256,
) -> InMemoryProbeDataset:  # pragma: no cover - DCC-only (needs BT voltage)
    """Extract the cohort's 1 s word-onset probe windows from the RUN's data chain.

    Reuses ``run_data`` (the converged run's :class:`Data`) so nothing is hand-copied:
    the same study (pretrain corpus), the same wired 3STFT band + DK-support + valid
    extractors, and the same ``bad_window_dir`` — re-segmented at the PROBE contract
    (``start=0.0`` / ``duration=1.0``; Ben 2026-06-18). Steps, mirroring
    :meth:`Data.build`: ``study.run()`` → :func:`filter_probe_events` (CLIP sidecars
    at 1 s width) → ``segmenter.apply`` → ``prepare``; then per cohort subject present
    in the triggers, :func:`select_subject_window_positions` caps to ``n_cap``
    (deterministic in ``seed`` — spec §8) and :func:`_materialize_subject` forwards
    them to band tensors + ``parcel_per_electrode`` / ``electrode_mask``. The §6
    firewall (no lite/leaderboard cell) is asserted by :class:`InMemoryProbeDataset`.

    Cohort lists are intersected with the subjects actually present, so a run on a
    sub-corpus simply probes fewer subjects instead of failing. DCC-only: needs BT
    voltage (the ``ROOT_DIR_BRAINTREEBANK`` root) + the 3STFT spec cache; the pure
    label/selection/filter logic it composes is laptop-TDD'd in
    ``test_online_probe_dataset``."""
    segmenter = _probe_segmenter(run_data)
    events = run_data.study.run()
    events = filter_probe_events(events, run_data.bad_window_dir)
    dataset = segmenter.apply(events)
    dataset.prepare()
    triggers = dataset.triggers

    present = present_subject_ids(triggers)
    per_subject: dict[int, dict[str, tp.Any]] = {}
    needed = {cs_anchor, *ws_subjects, *cs_test_subjects} & present
    # The probe binarizes its OWN ±1 labels from the raw feature columns, which the
    # emitted Word events drop — re-load the enriched words_df per session for the
    # neural-clock est_idx join (attach_probe_features). bt_root from the study.
    enriched_words = _load_enriched_words(
        {
            (int(s), int(t))
            for s, t in zip(triggers["subject_id"], triggers["trial_id"])
        },
        bt_root=os.environ.get("ROOT_DIR_BRAINTREEBANK"),
    )
    for sid in sorted(needed):
        positions = select_subject_window_positions(
            triggers, sid, n_cap=n_cap, seed=seed
        )
        if len(positions) == 0:
            continue
        per_subject[sid] = _materialize_subject(
            dataset, triggers, positions, batch_size=batch_size,
            enriched_words=enriched_words,
        )

    if cs_anchor not in per_subject:
        raise RuntimeError(
            f"online-probe CS anchor subject {cs_anchor} has no windows in the run "
            f"corpus (present subjects: {sorted(present)}); cannot build the probe."
        )
    n_parcels = int(per_subject[cs_anchor]["parcel_per_electrode"].max().item()) + 1
    return InMemoryProbeDataset(
        per_subject,
        n_parcels=max(n_parcels, _support_width(per_subject)),
        ws_subjects=[s for s in ws_subjects if s in per_subject],
        cs_anchor=cs_anchor,
        cs_test_subjects=[s for s in cs_test_subjects if s in per_subject],
    )


def _support_width(per_subject: dict[int, dict[str, tp.Any]]) -> int:
    """Parcel-vocabulary width = max parcel id across subjects + 1. The pooling
    indexes parcels into a fixed ``n_parcels`` table, so it must cover EVERY
    subject's ids, not just the anchor's."""
    return max(int(r["parcel_per_electrode"].max().item()) for r in per_subject.values()) + 1


__all__ = [
    "CS_ANCHOR",
    "CS_TEST_SUBJECTS",
    "InMemoryProbeDataset",
    "N_CAP",
    "PROBE_CLIP_DUR_S",
    "PROBE_CLIP_START_S",
    "PROBE_TASKS",
    "PROBE_TASK_ALIASES",
    "WS_SUBJECTS",
    "attach_probe_features",
    "build_probe_dataset",
    "filter_probe_events",
    "pm1_labels",
    "probe_feature_columns",
    "select_subject_window_positions",
    "select_window_indices",
]
