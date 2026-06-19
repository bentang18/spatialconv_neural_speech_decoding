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
from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS

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
    stretch. An absent subject → empty array."""
    pos = np.flatnonzero(np.asarray(triggers["subject_id"]) == subject_id)
    keep = select_window_indices(len(pos), n_cap, seed)
    return pos[keep]


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
        trigger_query="type == 'Word'",
        start=PROBE_CLIP_START_S,
        duration=PROBE_CLIP_DUR_S,
    )


def _materialize_subject(
    dataset: tp.Any, triggers: pd.DataFrame, positions: np.ndarray, *, batch_size: int
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
        "slow": torch.cat(bands["slow"], dim=0),
        "beta": torch.cat(bands["beta"], dim=0),
        "hg": torch.cat(bands["hg"], dim=0),
        "parcel_per_electrode": support0.argmax(dim=-1).long(),   # (C,)
        "electrode_mask": valid0.bool(),                          # (C,)
        "words_df": triggers.iloc[positions].reset_index(drop=True),
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
    voltage (``/work/ht203/data/braintreebank``) + the 3STFT spec cache; the pure
    label/selection/filter logic it composes is laptop-TDD'd in
    ``test_online_probe_dataset``."""
    segmenter = _probe_segmenter(run_data)
    events = run_data.study.run()
    events = filter_probe_events(events, run_data.bad_window_dir)
    dataset = segmenter.apply(events)
    dataset.prepare()
    triggers = dataset.triggers

    present = set(np.unique(np.asarray(triggers["subject_id"])).tolist())
    per_subject: dict[int, dict[str, tp.Any]] = {}
    needed = {cs_anchor, *ws_subjects, *cs_test_subjects} & present
    for sid in sorted(needed):
        positions = select_subject_window_positions(
            triggers, sid, n_cap=n_cap, seed=seed
        )
        if len(positions) == 0:
            continue
        per_subject[sid] = _materialize_subject(
            dataset, triggers, positions, batch_size=batch_size
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
    "build_probe_dataset",
    "filter_probe_events",
    "pm1_labels",
    "select_subject_window_positions",
    "select_window_indices",
]
