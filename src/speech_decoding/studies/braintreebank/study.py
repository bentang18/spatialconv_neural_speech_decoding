"""Local NeuralFetch-style Study for BrainTreebank.

The NeuroAI docs advertise `Wang2024Treebank`, but the installable
`neuralfetch==0.1.0` catalog does not ship it yet. This local Study preserves the
public NeuroAI API shape (`ns.Study(name="Wang2024Treebank", ...)`) while keeping
raw h5 loading behind NeuralSet's `SpecialLoader`.

Collision policy (see CLAUDE.md §Environment "Wang2024Treebank upgrade-path"):
NeuralSet routes ``ns.Study(name="X", ...)`` through a Pydantic discriminated
union keyed off the implementation class's ``__name__``. That makes the class
name part of the public API — we cannot rename to a private class and keep the
public dispatch working. Instead, we rely on NeuralSet's own
``__init_subclass__`` collision check: if upstream NeuralFetch ever ships its
own ``Wang2024Treebank``, whichever module imports second will raise
``RuntimeError("Study name collision: 'Wang2024Treebank' is already
registered ...")``. NeuralSet ships its own test for that mechanism — we
don't duplicate it here because exercising the registry on the live module
pollutes the Pydantic discriminated-union state for sibling tests.
"""

from __future__ import annotations

import os
import typing as tp
from pathlib import Path

import h5py
import mne
import pandas as pd
from neuralset.events import study

from speech_decoding.studies.braintreebank.anatomy import extra_bad_electrodes
from speech_decoding.studies.braintreebank.loader import bt_load_raw
from speech_decoding.studies.braintreebank.manifest import (
    BT_DEFAULT_SAMPLE_RATE_HZ,
    BT_FULL_SESSIONS,
    BT_LITE_SESSIONS,
    BT_NANO_SESSIONS,
    V14_P3_DISTILL_SESSIONS,
    V14_PRETRAIN_SESSIONS,
    bt_subject_native_rate_hz,
)


_SESSIONS_BY_MODE: dict[str, tuple[tuple[int, int], ...]] = {
    "lite": BT_LITE_SESSIONS,
    "nano": BT_NANO_SESSIONS,
    "full": BT_FULL_SESSIONS,
    # SSL/distill pretraining corpus: the Neuroprobe pretraining-allowed
    # sessions (BT_FULL − BT_LITE) restricted to the v14 cohort. DISJOINT from
    # BT_LITE_SESSIONS (the 12 leaderboard eval sessions), so pretraining on
    # this corpus never sees eval data. Paired with BTWordEvents
    # eval_mode="Pretrain"; routed by dispatch for P1/P2 (front-end + parcel).
    "pretrain": V14_PRETRAIN_SESSIONS,
    # P3 (Whisper-distill) corpus: "pretrain" minus the no-teacher sessions
    # ((8,0)). P3 needs a teacher target per clip, so it routes here (12
    # sessions / subjects {1,2,3,4,6,9}) instead of "pretrain" — otherwise the
    # distill phase crashes lazily on the first (8,0) clip.
    "p3_distill": V14_P3_DISTILL_SESSIONS,
}


class Wang2024Treebank(study.Study):
    """BrainTreebank: sEEG from 10 participants watching narrated movies."""

    mode: tp.Literal["lite", "nano", "full", "pretrain", "p3_distill"] = "lite"

    # Optional cache-build subset: restrict the emitted timelines to this exact
    # set of ``(subject_id, trial_id)`` pairs (a SUBSET of the ``mode`` corpus).
    # Used ONLY by dispatch ``--cache-only --cache-session-index`` so a SLURM
    # array can build one session's spec cache per task. Like ``mode`` it lives
    # in the timeline list, not the class uid (see ``_cls_kwargs``), so it never
    # changes the per-session spec-cache key (extractor-uid + session-uid).
    session_subset: tp.Optional[tp.Tuple[tp.Tuple[int, int], ...]] = None

    # Electrode montage, applied at the voltage boundary (PRE-CAR). "all"
    # (default) keeps the full montage minus STATIC bad contacts; "lite" subsets
    # to the Neuroprobe-Lite electrodes so the P4 eval references zero
    # out-of-budget contacts (shaft-CAR over Lite electrodes only). Like ``mode``
    # it lives in the per-session timeline (``iter_timelines`` + ``_cls_kwargs``),
    # not the class uid, but unlike ``mode`` it changes timeline CONTENT — so the
    # RAW exca cache uid (= SpecialLoader.to_json over the timeline) distinguishes
    # full-CAR from Lite-CAR. BT-only.
    electrode_set: tp.Literal["all", "lite"] = "all"

    # Optional precomputed trial-duration override, keyed ``"<subject_id>_<trial_id>"``
    # -> native-grid sample count. When set, ``_trial_duration_seconds`` returns from
    # this map instead of opening the raw h5 — the ONLY h5 touch in timeline building —
    # so the ``spec_only`` redeploy (no h5 on the target, e.g. DeltaAI) reads zero
    # neural data at ``study.run()``. Dropped from the class uid (see ``_cls_kwargs``);
    # ``duration`` does not enter the per-session spec-cache key (that is the
    # SpecialLoader timeline uid), so a JSON-sourced duration yields byte-identical
    # caches + clips to the h5 path. Fails loud on a missing key. BT-only.
    trial_durations: tp.Optional[tp.Dict[str, int]] = None

    aliases: tp.ClassVar[tuple[str, ...]] = (
        "BrainTreebank",
        "Braintreebank",
        "BT",
    )
    bibtex: tp.ClassVar[str] = """
    @article{wang2024treebank,
        title={A Brain Treebank},
        author={Wang, Christopher and others},
        year={2024}
    }
    """
    url: tp.ClassVar[str] = "https://braintreebank.dev/"
    licence: tp.ClassVar[str] = "See https://braintreebank.dev/"
    description: tp.ClassVar[str] = (
        "sEEG recordings from 10 participants watching movies while narrating "
        "(syntax treebank)."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("neuroprobe>=0.1.7",)
    _info: tp.ClassVar[study.StudyInfo | None] = None

    def _download(self) -> None:
        raise NotImplementedError(
            "Wang2024Treebank download is not wrapped here yet. Use Neuroprobe's "
            "braintreebank_download_extract.py, then set ROOT_DIR_BRAINTREEBANK."
        )

    def _cls_kwargs(self) -> dict[str, tp.Any]:
        """`mode` selects which timelines are emitted; it doesn't change content
        within a timeline. NeuralSet's default ``_cls_kwargs`` rejects any
        non-default pydantic field as an unsupported "class parameter", which
        blocks dispatch with ``mode != "lite"``. Drop ``mode`` from the class
        descriptor — it lives in the timeline list, not the class uid.
        """
        kwargs: dict[str, tp.Any] = self.model_dump(
            serialize_as_any=True, exclude_defaults=True,
        )
        for p in ("infra", "infra_timelines", "path", "name", "query", "mode",
                  "session_subset", "electrode_set", "trial_durations"):
            kwargs.pop(p, None)
        if kwargs:
            raise RuntimeError(
                f"Wang2024Treebank: unexpected non-default fields {sorted(kwargs)}"
            )
        return kwargs

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        sessions = _SESSIONS_BY_MODE[self.mode]
        if self.session_subset is not None:
            subset = {tuple(p) for p in self.session_subset}
            unknown = subset - set(sessions)
            if unknown:
                raise ValueError(
                    f"session_subset {sorted(unknown)} not in mode={self.mode!r} "
                    f"corpus {sorted(sessions)}"
                )
            sessions = [p for p in sessions if p in subset]
        for subject_id, trial_id in sessions:
            timeline: dict[str, tp.Any] = {
                "subject": f"btbank{subject_id}",
                "subject_id": subject_id,
                "trial_id": trial_id,
            }
            # Fold the montage + bad-electrode set into the per-session timeline so
            # the RAW exca cache uid (= SpecialLoader.to_json over this dict, via
            # _splittable_event_uid) depends on BOTH. Two guarantees: (1) full-CAR
            # vs Lite-CAR can never collide; (2) any edit to
            # _BT_V14_EXTRA_BAD_ELECTRODES auto-invalidates the stale raw cache —
            # the exact class of bug behind the 2026-06-15 static-cache stale-raw
            # trap, now closed because the cache KEY and the data LOAD both derive
            # from this one dict. Injected CONDITIONALLY (electrode_set only when
            # non-default, extra_bad only when non-empty) so a clean subject's
            # existing "all" cache stays valid — absence ⟺ default/empty, read
            # back symmetrically by _load_raw's .get(..., "all").
            if self.electrode_set != "all":
                timeline["electrode_set"] = self.electrode_set
            extra_bad = sorted(extra_bad_electrodes(subject_id, trial_id))
            if extra_bad:
                timeline["extra_bad"] = extra_bad
            yield timeline

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        filepath = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        return pd.DataFrame(
            [
                {
                    "type": "Ieeg",
                    "start": 0.0,
                    "duration": self._trial_duration_seconds(timeline),
                    "frequency": float(BT_DEFAULT_SAMPLE_RATE_HZ),
                    "filepath": filepath,
                }
            ]
        )

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        from neuroprobe.braintreebank_subject import BrainTreebankSubject

        subject_id = int(timeline["subject_id"])
        trial_id = int(timeline["trial_id"])
        bt = BrainTreebankSubject(
            subject_id=subject_id,
            cache=False,
            coordinates_type="cortical",
        )
        data, ch_names, sfreq = bt_load_raw(
            bt, trial_id=trial_id, subject_id=subject_id,
            electrode_set=timeline.get("electrode_set", "all"),
        )
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")
        return mne.io.RawArray(data, info, verbose=False)

    def _trial_duration_seconds(self, timeline: dict[str, tp.Any]) -> float:
        subject_id = int(timeline["subject_id"])
        trial_id = int(timeline["trial_id"])
        # `n_samples` is on the subject's NATIVE grid (h5 is the raw distributed
        # file), so wall-clock duration divides by the native rate — not the global
        # 2048. The loader resamples the voltage to 2048, which preserves this
        # duration; `frequency` in the Ieeg row is 2048 (post-resample) accordingly.
        native_rate = float(bt_subject_native_rate_hz(subject_id))
        if self.trial_durations is not None:
            key = f"{subject_id}_{trial_id}"
            if key not in self.trial_durations:
                raise ValueError(
                    f"trial_durations override is set but missing session {key!r} "
                    f"(have {sorted(self.trial_durations)}); an h5-free / spec_only "
                    "deploy cannot recover this trial's duration from raw h5."
                )
            return int(self.trial_durations[key]) / native_rate
        from neuroprobe.braintreebank_subject import BrainTreebankSubject
        from neuroprobe.config import ROOT_DIR

        bt = BrainTreebankSubject(
            subject_id=subject_id,
            cache=False,
            coordinates_type="cortical",
        )
        trial_path = Path(os.fspath(ROOT_DIR)) / f"sub_{subject_id}_trial{trial_id:03}.h5"
        first_label = bt.electrode_labels[0]
        first_key = bt.h5_neural_data_keys[first_label]
        with h5py.File(trial_path, "r") as h5:
            n_samples = int(h5["data"][first_key].shape[0])
        return n_samples / native_rate
