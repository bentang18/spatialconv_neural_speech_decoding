"""RAM (Kahana/UPenn) OpenNeuro iEEG NeuralSet ``Study`` — v3 SSL corpus producer.

Mirrors ``cogan_dcohort.DCohortStudy`` under the same CODE-VERIFIED v3 contract:
the training path never touches a Study — ``iter_timelines()`` IS the session
registry ``dispatch_v14`` enumerates to bake per-session spec caches, and
``dispatch_v3`` reads those caches by ``--session``. This class only yields one
timeline per run and hands back clean 2048 Hz voltage; the front-end
(``MultiStftView`` via ``CARIeegExtractor``) owns notch / HPF / group-CAR / STFT.

``manifest_path`` is the run MANIFEST CSV (``build_ram_manifest.py``) — one row per
(subject, task, session, run), carrying the globally-disjoint ``global_subject_id``
(``corpus_ids.allocate_ram_ids``, block 3000-3999), the per-subject ``trial_id``,
the MONOPOLAR EDF + channels.tsv paths, and ``duration_s`` (from ieeg.json ⇒ zero
EDF opens at timeline build). Voltage comes from ``ram_load_raw``. Names are clean
``<group><contact>`` so the extractor's name-parse CAR groups ~98% correctly; the
authoritative channels.tsv ``group`` (via ``loader.ram_car_groups``) feeds the #94
explicit-group-CAR path for the ~2% compound ``_``-names.

Guard-1 static-bad + micro-electrode contacts fold into the timeline via
``ram_extra_bad`` (so they enter the cache key and ``_load_raw`` drops them
pre-CAR); absent map ⟺ empty ⟺ no-op until the scan runs.
"""

from __future__ import annotations

import csv
import os
import typing as tp

import mne
import pandas as pd
from neuralset.events import study

from speech_decoding.studies.ram_cohort.guard1_static import ram_extra_bad
from speech_decoding.studies.ram_cohort.loader import (
    TARGET_RATE_HZ,
    ram_car_groups,
    ram_load_raw,
    read_channels_tsv,
)

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "subject_bids",
    "global_subject_id",
    "trial_id",
    "edf_path",
    "channels_tsv_path",
    "duration_s",
)


class RamCohortStudy(study.Study):
    """RAM (Kahana/UPenn) OpenNeuro iEEG cohort — v3 SSL producer (voltage+timeline).

    Anatomy-bearing via the shipped ``electrodes.tsv`` ``ind.region`` (native DK
    aparc); the v3 ``parcel_fn`` consumes it downstream (DK→DKT policy Ben-gated).
    This class is purely the voltage + timeline boundary.
    """

    manifest_path: str = ""
    session_subset: tp.Optional[tp.Tuple[tp.Tuple[int, int], ...]] = None

    aliases: tp.ClassVar[tuple[str, ...]] = ("RAM", "RAM_Kahana", "RAM-UPenn-iEEG")
    bibtex: tp.ClassVar[str] = ""  # cite the per-task OpenNeuro accessions.
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds004789"
    licence: tp.ClassVar[str] = "CC0 (OpenNeuro RAM datasets)."
    description: tp.ClassVar[str] = (
        "RAM (Restoring Active Memory, Kahana/UPenn) OpenNeuro iEEG — FR1/catFR1/"
        "RepFR1(/PAL1/pyFR) memory tasks, CC0. Anatomy via electrodes.tsv ind.region "
        "(DK aparc). SSL label-agnostic."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ()
    _info: tp.ClassVar[study.StudyInfo | None] = None

    # v3 canonical rate. RAM native rate varies per subject; ram_load_raw resamples
    # every run to this before it leaves the loader.
    SAMPLE_RATE_HZ: tp.ClassVar[float] = TARGET_RATE_HZ

    def _cls_kwargs(self) -> dict[str, tp.Any]:
        kwargs: dict[str, tp.Any] = self.model_dump(
            serialize_as_any=True, exclude_defaults=True,
        )
        for p in ("infra", "infra_timelines", "path", "name", "query",
                  "manifest_path", "session_subset"):
            kwargs.pop(p, None)
        if kwargs:
            raise RuntimeError(
                f"RamCohortStudy: unexpected non-default fields {sorted(kwargs)}"
            )
        return kwargs

    def _download(self) -> None:
        raise NotImplementedError(
            "RAM is CC0 on OpenNeuro (ds004789/ds004809/ds005411/…); the raw pull is "
            "a separate orchestration step (Ben-gated). Set RamCohortStudy.manifest_path "
            "to a build_ram_manifest.py CSV over the staged tree."
        )

    def _resolved_manifest_path(self) -> str:
        mp = self.manifest_path or os.environ.get("RAM_MANIFEST", "")
        if not mp:
            raise ValueError(
                "RamCohortStudy.manifest_path is unset and RAM_MANIFEST env is empty; "
                "a reconstructed study restores the manifest path from RAM_MANIFEST "
                "(set by the RAM cache-build experiment from --ram-manifest)."
            )
        return mp

    def _manifest_rows(self) -> list[dict[str, str]]:
        manifest_path = self._resolved_manifest_path()
        with open(manifest_path, newline="") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in _REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(
                    f"manifest {manifest_path} missing columns {missing}; "
                    f"regenerate with build_ram_manifest.py"
                )
            return list(reader)

    def _row_for(self, timeline: dict[str, tp.Any]) -> dict[str, str]:
        sid, tid = int(timeline["subject_id"]), int(timeline["trial_id"])
        for row in self._manifest_rows():
            if int(row["global_subject_id"]) == sid and int(row["trial_id"]) == tid:
                return row
        raise KeyError(
            f"session (subject_id={sid}, trial_id={tid}) not in manifest "
            f"{self._resolved_manifest_path()}"
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        subset = (
            {tuple(p) for p in self.session_subset}
            if self.session_subset is not None
            else None
        )
        seen: set[tuple[int, int]] = set()
        for row in self._manifest_rows():
            sid, tid = int(row["global_subject_id"]), int(row["trial_id"])
            if subset is not None and (sid, tid) not in subset:
                continue
            seen.add((sid, tid))
            timeline: dict[str, tp.Any] = {
                "subject": row["subject_bids"],
                "subject_id": sid,
                "trial_id": tid,
            }
            extra_bad = sorted(ram_extra_bad(sid, tid))
            if extra_bad:
                timeline["extra_bad"] = extra_bad
            yield timeline
        if subset is not None:
            unknown = subset - seen
            if unknown:
                raise ValueError(
                    f"session_subset {sorted(unknown)} not in manifest {self.manifest_path}"
                )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        row = self._row_for(timeline)
        filepath = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        return pd.DataFrame(
            [
                {
                    "type": "Ieeg",
                    "start": 0.0,
                    "duration": float(row["duration_s"]),
                    "frequency": float(TARGET_RATE_HZ),
                    "filepath": filepath,
                }
            ]
        )

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        row = self._row_for(timeline)
        channels_tsv = row["channels_tsv_path"]
        if not channels_tsv:
            raise ValueError(
                f"run {row['edf_path']} has no channels.tsv; cannot select neural "
                "channels by type (this run should have been skipped at manifest build)"
            )
        extra_bad = tuple(timeline.get("extra_bad", ()))
        data, ch_names, sfreq = ram_load_raw(
            row["edf_path"], channels_tsv, extra_bad=extra_bad,
            target_rate=TARGET_RATE_HZ,
        )
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        # #94 group-CAR: stash the authoritative per-run {name: group} map (from the
        # channels.tsv ``group`` column) so ``_apply_car`` (car="group") can reference
        # each contact against its own shaft/group. Must cover every surviving name.
        meta = read_channels_tsv(channels_tsv)
        groups = ram_car_groups(ch_names, meta)
        car_groups = {nm: g for nm, g in zip(ch_names, groups)}
        assert set(car_groups) == set(ch_names), (
            "car_groups map does not cover all surviving ch_names"
        )
        raw.info["temp"] = {"car_groups": car_groups}
        return raw
