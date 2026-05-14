"""Local NeuralFetch-style Study for BrainTreebank.

The NeuroAI docs advertise `Wang2024Treebank`, but the installable
`neuralfetch==0.1.0` catalog does not ship it yet. This local Study preserves the
public NeuroAI API shape (`ns.Study(name="Wang2024Treebank", ...)`) while keeping
raw h5 loading behind NeuralSet's `SpecialLoader`.
"""

from __future__ import annotations

import os
import typing as tp
from pathlib import Path

import h5py
import mne
import pandas as pd
from neuralset.events import study

from speech_decoding.studies.braintreebank.loader import bt_load_raw
from speech_decoding.studies.braintreebank.manifest import (
    BT_FULL_SESSIONS,
    BT_LITE_SESSIONS,
    BT_NANO_SESSIONS,
)


_SESSIONS_BY_MODE: dict[str, tuple[tuple[int, int], ...]] = {
    "lite": BT_LITE_SESSIONS,
    "nano": BT_NANO_SESSIONS,
    "full": BT_FULL_SESSIONS,
}


class Wang2024Treebank(study.Study):
    """BrainTreebank: sEEG from 10 participants watching narrated movies."""

    mode: tp.Literal["lite", "nano", "full"] = "lite"

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
        for p in ("infra", "infra_timelines", "path", "name", "query", "mode"):
            kwargs.pop(p, None)
        if kwargs:
            raise RuntimeError(
                f"Wang2024Treebank: unexpected non-default fields {sorted(kwargs)}"
            )
        return kwargs

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for subject_id, trial_id in _SESSIONS_BY_MODE[self.mode]:
            yield {
                "subject": f"btbank{subject_id}",
                "subject_id": subject_id,
                "trial_id": trial_id,
            }

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        filepath = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        return pd.DataFrame(
            [
                {
                    "type": "Ieeg",
                    "start": 0.0,
                    "duration": self._trial_duration_seconds(timeline),
                    "frequency": 2048.0,
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
        data, ch_names, sfreq = bt_load_raw(bt, trial_id=trial_id)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")
        return mne.io.RawArray(data, info, verbose=False)

    def _trial_duration_seconds(self, timeline: dict[str, tp.Any]) -> float:
        from neuroprobe.braintreebank_subject import BrainTreebankSubject
        from neuroprobe.config import ROOT_DIR, SAMPLING_RATE

        subject_id = int(timeline["subject_id"])
        trial_id = int(timeline["trial_id"])
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
        return n_samples / float(SAMPLING_RATE)
