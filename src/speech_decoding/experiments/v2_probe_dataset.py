"""v2 probe-dataset port — the 2-band twin of :mod:`online_probe_dataset`.

The dev (pretrain-corpus) probe for the converged-v2 arch. Reuses ALL of the pure,
already-TDD'd logic from :mod:`online_probe_dataset` — the ±1 task labels
(:func:`pm1_labels`), the deterministic ``min(n_cap, available)`` window selection,
the CLIP-sidecar event filter, the transcript-feature re-join
(:func:`attach_probe_features`), and the §6 leakage firewall — and only swaps the
band-specific materialization:

  - segmenter keys ``electrode_tokens_lfs`` / ``electrode_tokens_hga`` (2 magnitude
    bands) instead of the 3STFT ``slow/beta/hg``;
  - no Cartesian→5-D split (v2 bands are pure magnitude, ``in_channels=1``);
  - ``subject_data(s)`` returns a :class:`V2SubjectProbeData` carrying
    ``.bands = [lfs, hga]`` — the contract :mod:`v2_raw_probe` (and the encoder taps)
    consume.

The pure assembly (:class:`InMemoryV2ProbeDataset`, label derivation, firewall) is
laptop-TDD'd in ``test_v2_probe_dataset``; the segmenter/materialize/build path
needs BT voltage + ``neuralset`` and runs on DeltaAI/DCC (pragma-no-cover).
"""

from __future__ import annotations

import dataclasses
import os
import typing as tp

import numpy as np
import pandas as pd
from torch import Tensor

from speech_decoding.experiments.online_probe import assert_firewall
from speech_decoding.experiments.online_probe_dataset import (
    CS_ANCHOR,
    CS_TEST_SUBJECTS,
    N_CAP,
    PROBE_CLIP_DUR_S,
    PROBE_CLIP_START_S,
    PROBE_TASKS,
    WS_SUBJECTS,
    attach_probe_features,
    filter_probe_events,
    pm1_labels,
    present_subject_ids,
    select_subject_window_positions,
)
from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS

if tp.TYPE_CHECKING:
    from speech_decoding.experiments.data import Data

# v2 segmenter keys: 2 magnitude bands + the anatomy the pooling needs. The run's
# target / whisper_target extractors are dropped (probe labels come from words_df).
_PROBE_SEGMENTER_KEYS_V2: tuple[str, ...] = (
    "electrode_tokens_lfs",
    "electrode_tokens_hga",
    "support",
    "valid_mask",
)

__all__ = [
    "V2SubjectProbeData",
    "InMemoryV2ProbeDataset",
    "build_v2_probe_dataset",
]


@dataclasses.dataclass
class V2SubjectProbeData:
    """One subject's encoder-ready v2 probe inputs.

    ``bands`` is ``[lfs, hga]`` — the 1 s magnitude band tensors ``(N,C,28,T_lfs)`` /
    ``(N,C,7,T_hga)`` the v2 encoder is fed (and that :mod:`v2_raw_probe` flattens).
    ``parcel_per_electrode`` / ``electrode_mask`` are ``(C,)`` constant across the
    subject's windows; ``labels`` maps task → ``(N,)`` ±1; ``sessions`` is the set of
    ``(subject,trial)`` the windows came from (firewall)."""

    subject_id: int
    bands: list[Tensor]
    parcel_per_electrode: Tensor
    electrode_mask: Tensor
    labels: dict[str, np.ndarray]
    sessions: list[tuple[int, int]]


class InMemoryV2ProbeDataset:
    """Concrete v2 probe dataset from already-extracted per-subject tensors.

    Each ``per_subject[sid]`` is a dict with ``bands`` (``[lfs, hga]``),
    ``parcel_per_electrode`` / ``electrode_mask`` ``(C,)``, the ``words_df`` (carrying
    the task feature columns), and the ``sessions`` the windows came from. Task ±1
    labels are derived once via :func:`pm1_labels`; the §6 firewall asserts no window
    came from a lite/leaderboard cell. :func:`v2_raw_probe.run_v2_raw_baseline` and
    the encoder taps consume this directly."""

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
        self._data: dict[int, V2SubjectProbeData] = {}
        all_sessions: list[tuple[int, int]] = []
        for sid, rec in per_subject.items():
            wdf = rec["words_df"]
            labels = {t: pm1_labels(wdf, t) for t in self.tasks}
            sessions = [tuple(s) for s in rec["sessions"]]
            all_sessions.extend(sessions)
            self._data[sid] = V2SubjectProbeData(
                subject_id=sid,
                bands=list(rec["bands"]),
                parcel_per_electrode=rec["parcel_per_electrode"],
                electrode_mask=rec["electrode_mask"],
                labels=labels,
                sessions=sessions,
            )
        assert_firewall(all_sessions, BT_LITE_SESSIONS)

    def subject_data(self, subject_id: int) -> V2SubjectProbeData:
        return self._data[subject_id]


def _probe_segmenter_v2(run_data: "Data") -> tp.Any:  # pragma: no cover - needs neuralset
    """A 1 s / zero-lag segmenter reusing the run's wired 2-band + anatomy extractors.

    Built the way :class:`Data` coerces its ``segmenter`` field, so the band tensors
    are byte-identical to the run's — only the clip window changes (5 s SSL → 1 s
    probe, ``start`` → 0.0). Drops ``target`` / ``whisper_target``."""
    import neuralset as ns

    src = run_data.segmenter.extractors
    missing = [k for k in _PROBE_SEGMENTER_KEYS_V2 if k not in src]
    if missing:
        raise KeyError(
            f"v2 probe needs the 2-band segmenter keys {missing} on the run's data; "
            f"got {sorted(src)}. The v2 probe only supports --frontend 2band."
        )
    return ns.dataloader.Segmenter(
        extractors={k: src[k] for k in _PROBE_SEGMENTER_KEYS_V2},
        # Exclude nonverbal anchors (type=='Word' text=='<nonverbal>'): they carry no
        # word-level transcript features and would trip the attach_probe_features
        # est_idx join (#241). See online_probe_dataset._probe_segmenter.
        trigger_query="type == 'Word' and text != '<nonverbal>'",
        start=PROBE_CLIP_START_S,
        duration=PROBE_CLIP_DUR_S,
    )


def _materialize_subject_v2(
    dataset: tp.Any,
    triggers: pd.DataFrame,
    positions: np.ndarray,
    *,
    batch_size: int,
    enriched_words: dict[tuple[int, int], pd.DataFrame],
) -> dict[str, tp.Any]:  # pragma: no cover - needs BT voltage
    """Forward one subject's selected windows through the segmenter, collect the two
    magnitude band tensors + per-electrode anatomy. Mirrors ``Data.build``'s
    select→DataLoader path (``num_workers=0``). ``parcel_per_electrode`` /
    ``electrode_mask`` are constant across windows, read off the first batch. No
    Cartesian split — both v2 bands are magnitude."""
    import torch
    from torch.utils.data import DataLoader

    mask = pd.Series(np.isin(np.arange(len(triggers)), positions), index=triggers.index)
    subset = dataset.select(mask)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=subset.collate_fn,
    )
    lfs: list[torch.Tensor] = []
    hga: list[torch.Tensor] = []
    support0: torch.Tensor | None = None
    valid0: torch.Tensor | None = None
    for batch in loader:
        lfs.append(batch.data["electrode_tokens_lfs"])
        hga.append(batch.data["electrode_tokens_hga"])
        if support0 is None:
            support0 = batch.data["support"][0]          # (C, K) one-hot
            valid0 = batch.data["valid_mask"][0]         # (C,) bool
    if support0 is None:
        raise RuntimeError("v2 probe subject yielded no windows after selection")
    return {
        "bands": [torch.cat(lfs, dim=0), torch.cat(hga, dim=0)],
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


def _support_width(per_subject: dict[int, dict[str, tp.Any]]) -> int:
    """Parcel-vocabulary width = max parcel id across subjects + 1 (the pool indexes a
    fixed table covering EVERY subject's ids, not just the anchor's)."""
    return max(
        int(r["parcel_per_electrode"].max().item()) for r in per_subject.values()
    ) + 1


def build_v2_probe_dataset(
    run_data: "Data",
    *,
    n_cap: int = N_CAP,
    seed: int = 0,
    ws_subjects: tp.Sequence[int] = WS_SUBJECTS,
    cs_anchor: int = CS_ANCHOR,
    cs_test_subjects: tp.Sequence[int] = CS_TEST_SUBJECTS,
    batch_size: int = 256,
) -> InMemoryV2ProbeDataset:  # pragma: no cover - needs BT voltage
    """Extract the cohort's 1 s word-onset probe windows from the v2 run's data chain.

    Reuses ``run_data`` (the v2 run's :class:`Data`) so nothing is hand-copied: same
    study (pretrain corpus), same wired 2-band + DK-support extractors, same
    ``bad_window_dir`` — re-segmented at the PROBE contract (``start=0.0`` /
    ``duration=1.0``). Steps mirror :func:`online_probe_dataset.build_probe_dataset`,
    only the materialization is 2-band. DeltaAI/DCC-only: needs BT voltage
    (``ROOT_DIR_BRAINTREEBANK``) + the 2-band spec cache."""
    from speech_decoding.experiments.online_probe_dataset import _load_enriched_words

    segmenter = _probe_segmenter_v2(run_data)
    events = run_data.study.run()
    events = filter_probe_events(events, run_data.bad_window_dir)
    dataset = segmenter.apply(events)
    dataset.prepare()
    triggers = dataset.triggers

    present = present_subject_ids(triggers)
    needed = {cs_anchor, *ws_subjects, *cs_test_subjects} & present
    enriched_words = _load_enriched_words(
        {(int(s), int(t)) for s, t in zip(triggers["subject_id"], triggers["trial_id"])},
        bt_root=os.environ.get("ROOT_DIR_BRAINTREEBANK"),
    )
    per_subject: dict[int, dict[str, tp.Any]] = {}
    for sid in sorted(needed):
        positions = select_subject_window_positions(triggers, sid, n_cap=n_cap, seed=seed)
        if len(positions) == 0:
            continue
        per_subject[sid] = _materialize_subject_v2(
            dataset, triggers, positions, batch_size=batch_size, enriched_words=enriched_words
        )

    if cs_anchor not in per_subject:
        raise RuntimeError(
            f"v2 probe CS anchor subject {cs_anchor} has no windows in the run corpus "
            f"(present: {sorted(present)}); cannot build the probe."
        )
    n_parcels = int(per_subject[cs_anchor]["parcel_per_electrode"].max().item()) + 1
    return InMemoryV2ProbeDataset(
        per_subject,
        n_parcels=max(n_parcels, _support_width(per_subject)),
        ws_subjects=[s for s in ws_subjects if s in per_subject],
        cs_anchor=cs_anchor,
        cs_test_subjects=[s for s in cs_test_subjects if s in per_subject],
    )
