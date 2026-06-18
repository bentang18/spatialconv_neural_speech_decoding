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

from speech_decoding.experiments.online_probe import SubjectProbeData, assert_firewall
from speech_decoding.studies.braintreebank.labels import (
    SINGLE_FLOAT_TASK_COLUMNS,
    _assert_finite_labels,
    remap_task_column,
)
from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS

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


def build_probe_dataset(
    *, n_cap: int = N_CAP, seed: int = 0, **kwargs: tp.Any
) -> InMemoryProbeDataset:  # pragma: no cover
    """DCC-only: extract 1 s word-onset windows for every cohort subject into
    encoder-ready 3STFT band tensors (via the real study + MultiStftView 3STFT
    chain, ``clip_len=1.0`` / P4 neural-lag), select ``min(n_cap, available)`` per
    subject (deterministic in ``seed`` — spec §8), derive ``parcel_per_electrode`` /
    ``electrode_mask`` from the DK support + valid-mask extractors, and assemble an
    :class:`InMemoryProbeDataset`.

    Not implemented on the laptop — needs BT voltage (``/work/ht203/data/
    braintreebank``) + the 3STFT spec cache. Wiring + DCC verification is Phase 2b;
    the pure label/selection/assembly logic it depends on is TDD-checked here."""
    raise NotImplementedError(
        "build_probe_dataset is DCC-only (Phase 2b): needs BT voltage + the 3STFT "
        "spec cache. The pure core (pm1_labels / select_window_indices / "
        "InMemoryProbeDataset) is laptop-tested; this wires them to the real study/view."
    )


__all__ = [
    "CS_ANCHOR",
    "CS_TEST_SUBJECTS",
    "InMemoryProbeDataset",
    "N_CAP",
    "PROBE_TASKS",
    "PROBE_TASK_ALIASES",
    "WS_SUBJECTS",
    "build_probe_dataset",
    "pm1_labels",
    "select_window_indices",
]
