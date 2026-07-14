"""Cogan D-cohort (Duke sEEG) NeuralSet ``Study`` — v3 SSL corpus producer.

Mirrors ``braintreebank.Wang2024Treebank`` under the CODE-VERIFIED v3 contract
(memory: project-cogan-seeg-ingestion-contract): the training path never touches
a Study — ``iter_timelines()`` IS the session registry that ``dispatch_v14``
enumerates when it bakes the per-session spec caches, and ``dispatch_v3`` then
reads those caches by ``--session``. So this class only has to yield one timeline
per run and hand back clean 2048 Hz voltage; the front-end (``MultiStftView`` via
``CARIeegExtractor``) owns notch / HPF / shaft-CAR / STFT.

``manifest_path`` is the run MANIFEST CSV emitted by
``scripts/neuroprobe/build_cogan_manifest.py`` — one row per (subject, task, run),
each carrying the globally-disjoint ``global_subject_id`` (``corpus_ids``), the
per-subject ``trial_id``, the EDF + channels.tsv paths, and ``duration_s`` (from
``ieeg.json``, so timeline building opens ZERO EDFs). The cache key embeds the
row's ``subject_id`` + ``trial_id`` as bare ints (``parse_key_session`` parses
them back). Voltage comes from ``cogan_load_raw`` (EDF → neural-type select +
contact-index filter → polyphase resample to 2048), returned as an
``mne.io.RawArray`` whose ``ch_names`` are clean ``<shaft><contact>`` so the
extractor's name-based ``parse_shaft`` groups the CAR correctly.

Guard-1 static-bad contacts (per run) fold into the yielded timeline via
``cogan_extra_bad`` (as BT does with ``extra_bad``) so they enter the cache key
(any edit auto-invalidates the stale raw cache) and ``_load_raw`` drops them
pre-CAR. The drop map is absent until the guard-1 scan+collector runs, so until
then the fold-in is a no-op (see ``guard1_static``).
"""

from __future__ import annotations

import csv
import typing as tp

import mne
import pandas as pd
from neuralset.events import study

from speech_decoding.studies.cogan_dcohort.guard1_static import cogan_extra_bad
from speech_decoding.studies.cogan_dcohort.loader import (
    TARGET_RATE_HZ,
    cogan_load_raw,
)

# Manifest columns this Study consumes (a subset of build_cogan_manifest's
# ManifestRow). Kept as an explicit contract so the CSV format and the reader
# can't silently drift apart.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "subject_bids",
    "global_subject_id",
    "trial_id",
    "edf_path",
    "channels_tsv_path",
    "duration_s",
)


class DCohortStudy(study.Study):
    """Cogan-lab Duke sEEG D-cohort (D<num> patients) — v3 SSL producer.

    Anatomy-bearing via native FreeSurfer recons (per-electrode DKT parcels are
    produced by the #97 localization pipeline into ``depth-wm.csv``, consumed
    downstream by the v3 ``parcel_fn`` — not here). This class is purely the
    voltage + timeline boundary.
    """

    # Run manifest CSV (build_cogan_manifest.py). The base ``path`` field is a
    # study DATA DIRECTORY (a validator mkdir's it), so the manifest — a file —
    # gets its own field. Enumeration-only ⇒ dropped in ``_cls_kwargs``.
    manifest_path: str = ""

    # SLURM-array cache build: restrict emitted timelines to this exact set of
    # ``(subject_id, trial_id)`` pairs (a subset of the manifest). Like BT's, it
    # lives in the timeline list, not the class uid — dropped in ``_cls_kwargs``
    # so it never perturbs a per-session cache key.
    session_subset: tp.Optional[tp.Tuple[tp.Tuple[int, int], ...]] = None

    aliases: tp.ClassVar[tuple[str, ...]] = (
        "DCohort", "Cogan_DCohort", "Cogan-Duke-sEEG",
    )
    bibtex: tp.ClassVar[str] = ""  # internal cohort; cite Cogan-lab publications.
    url: tp.ClassVar[str] = ""
    licence: tp.ClassVar[str] = "Internal Cogan-lab data; IRB-restricted."
    description: tp.ClassVar[str] = (
        "Cogan-lab Duke sEEG D-cohort (D<num> patients), 7-task speech+cognitive "
        "SSL corpus. Anatomy-bearing via native FS recons (DKT). Distinct from "
        "the PS S<num> uECoG cohort."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ()
    _info: tp.ClassVar[study.StudyInfo | None] = None

    # v3 canonical rate. Native rate is mixed across runs (2048/2000/1024/1000);
    # cogan_load_raw resamples every run to this before it leaves the loader.
    SAMPLE_RATE_HZ: tp.ClassVar[float] = TARGET_RATE_HZ

    def _cls_kwargs(self) -> dict[str, tp.Any]:
        """Drop enumeration-only fields from the class uid.

        The base ``_cls_kwargs`` rejects any non-default pydantic field as an
        unsupported class parameter, which would block dispatch whenever
        ``session_subset`` (or the manifest ``path``) is set. Neither changes the
        CONTENT of a per-session timeline, so both are excluded — the per-session
        cache key is the SpecialLoader uid over the timeline, not the class uid.
        """
        kwargs: dict[str, tp.Any] = self.model_dump(
            serialize_as_any=True, exclude_defaults=True,
        )
        for p in ("infra", "infra_timelines", "path", "name", "query",
                  "manifest_path", "session_subset"):
            kwargs.pop(p, None)
        if kwargs:
            raise RuntimeError(
                f"DCohortStudy: unexpected non-default fields {sorted(kwargs)}"
            )
        return kwargs

    def _download(self) -> None:
        raise NotImplementedError(
            "Cogan D-cohort is IRB-restricted lab data already on DCC at "
            "/hpc/group/coganlab/Data; there is no download step. Set "
            "DCohortStudy.manifest_path to a build_cogan_manifest.py CSV."
        )

    def _manifest_rows(self) -> list[dict[str, str]]:
        """Read the manifest CSV at ``self.manifest_path`` → list of row dicts.

        Validates the required columns are present (fail loud on a stale CSV
        schema) but does no type coercion beyond what the callers do inline.
        """
        if not self.manifest_path:
            raise ValueError("DCohortStudy.manifest_path is unset")
        with open(self.manifest_path, newline="") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in _REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(
                    f"manifest {self.manifest_path} missing columns {missing}; "
                    f"regenerate with build_cogan_manifest.py"
                )
            return list(reader)

    def _row_for(self, timeline: dict[str, tp.Any]) -> dict[str, str]:
        sid, tid = int(timeline["subject_id"]), int(timeline["trial_id"])
        for row in self._manifest_rows():
            if int(row["global_subject_id"]) == sid and int(row["trial_id"]) == tid:
                return row
        raise KeyError(
            f"session (subject_id={sid}, trial_id={tid}) not in manifest {self.manifest_path}"
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
            # Fold the guard-1 STATIC drop into the timeline so the raw exca cache
            # uid depends on it (edit → auto-invalidate) and _load_raw drops it
            # pre-CAR. CONDITIONAL (absence ⟺ empty) so a clean run's cache stays
            # valid — read back symmetrically by _load_raw's .get("extra_bad", ()).
            extra_bad = sorted(cogan_extra_bad(sid, tid))
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
                    # Wall-clock seconds from ieeg.json — resample preserves it, so
                    # it is consistent with frequency = post-resample 2048.
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
        data, ch_names, sfreq = cogan_load_raw(
            row["edf_path"], channels_tsv, extra_bad=extra_bad,
            target_rate=TARGET_RATE_HZ,
        )
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")
        return mne.io.RawArray(data, info, verbose=False)
