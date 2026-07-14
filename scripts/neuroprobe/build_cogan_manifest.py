"""Build the Cogan D-cohort run manifest — one row per (subject, task, run) = one cache.

Enumerates the 7 planned-set BIDS task trees on DCC, pairs each ``_ieeg.edf`` with
its ``_ieeg.json`` (native rate, mains, duration) and its per-subject-per-task
``_channels.tsv`` (channel types), and assigns a globally-disjoint
``global_subject_id`` (``corpus_ids.cogan_global_id``) plus a per-subject
``trial_id`` (runs enumerated across tasks in sorted order). The v3 ``DCohortStudy``
consumes this manifest in ``iter_timelines``.

Deliberately METADATA-ONLY: reads ``ieeg.json`` and checks ``channels.tsv``
existence; never opens an EDF. So it runs on a login node and unit-tests on a
synthetic tree with no ``mne`` and no neural data.

Two honesty flags travel in the manifest rather than being silently resolved:
  * ``channels_tsv`` empty  → the run has no BIDS ``channels.tsv`` (seen on TIMIT
    D38/D39/D54, some Sternberg). Downstream must DECIDE drop-vs-name-heuristic;
    the loader's neural-type whitelist needs this file.
  * ``localized`` is NOT set here — the ~91-brain union is emitted whole; the
    #97 localization join filters to the 86-set. Keeps the two steps decoupled.

Run on DCC:
    python build_cogan_manifest.py --data-root /hpc/group/coganlab/Data \\
        --out /work/ht203/cogan_v3/manifest/cogan_run_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import asdict, dataclass

from speech_decoding.models.v14_converged_v3.corpus_ids import cogan_global_id

# Planned SSL set, 7 tasks (memory: reference-cogan-seeg-bids-corpus-audit).
# Values are the substrings of the ``BIDS-<ver>_<TASK>`` dir name.
PLANNED_TASKS: tuple[str, ...] = (
    "SentenceRep",
    "Phoneme_sequencing",
    "TIMIT",
    "LexicalDecRepDelay",
    "Uniqueness_point",
    "Neighborhood_Sternberg",
    "GlobalLocal",
)

# sub-D0019_task-PhonemeSequence_acq-01_run-01_ieeg.edf
# Subject may carry a re-implant suffix letter (sub-D0107A) — a DISTINCT brain
# from a bare D0107 or a D0107B, so it must survive into the manifest identity.
_IEEG_RE = re.compile(
    r"^(?P<subject>sub-D\d+[A-Z]?)_task-(?P<ftask>[A-Za-z0-9]+)"
    r"(?:_acq-(?P<acq>[A-Za-z0-9]+))?(?:_run-(?P<run>[A-Za-z0-9]+))?_ieeg\.edf$"
)
_SUBJECT_RE = re.compile(r"sub-D0*(?P<dnum>\d+)(?P<implant>[A-Z]?)$")


@dataclass(frozen=True)
class ManifestRow:
    subject_bids: str      # sub-D0019 (or sub-D0107A)
    d_num: int             # 19 (107 for D0107A)
    implant: str           # "" or the re-implant suffix letter ("A"/"B")
    global_subject_id: int  # 1019
    task: str              # canonical dir task (Phoneme_sequencing)
    acq: str               # "01" or ""
    run: str               # "01" or ""
    trial_id: int          # per-subject run index (assigned later)
    edf_path: str
    json_path: str
    channels_tsv_path: str  # "" when absent (honesty flag)
    native_sfreq: float
    power_line_hz: float
    duration_s: float


def parse_subject(subject_bids: str) -> tuple[int, str]:
    """``sub-D0019`` → ``(19, "")``; ``sub-D0107A`` → ``(107, "A")``."""
    m = _SUBJECT_RE.fullmatch(subject_bids)
    if not m:
        raise ValueError(f"unrecognised Cogan BIDS subject {subject_bids!r}")
    return int(m.group("dnum")), m.group("implant")


def find_bids_roots(data_root: str) -> dict[str, str]:
    """``{task: <data_root>/BIDS-<ver>_<task>/BIDS}`` for each planned task present."""
    roots: dict[str, str] = {}
    for task in PLANNED_TASKS:
        hits = sorted(glob.glob(os.path.join(data_root, f"BIDS-*_{task}", "BIDS")))
        if hits:
            roots[task] = hits[0]
    return roots


def _channels_tsv_for(ieeg_dir: str, subject: str, ftask: str) -> str:
    """Per-subject-per-task ``channels.tsv`` path if it exists, else ``""``.

    BIDS names it without acq/run (``sub-D0019_task-PhonemeSequence_channels.tsv``),
    shared across that subject's runs of the task.
    """
    cand = os.path.join(ieeg_dir, f"{subject}_task-{ftask}_channels.tsv")
    return cand if os.path.exists(cand) else ""


def enumerate_runs(bids_roots: dict[str, str]) -> list[ManifestRow]:
    """Walk each task tree → one un-trial-id'd ``ManifestRow`` per ``_ieeg.edf``.

    Runs whose ``_ieeg.json`` sidecar is absent (incomplete/practice EDFs — seen
    on TIMIT D38/D39/D54) are SKIPPED with a warning: without the sidecar we have
    no native rate/duration, and those same runs also lack ``channels.tsv``.
    """
    rows: list[ManifestRow] = []
    skipped: list[str] = []
    for task, root in bids_roots.items():
        for edf in sorted(glob.glob(os.path.join(root, "sub-D*", "ieeg", "*_ieeg.edf"))):
            fname = os.path.basename(edf)
            m = _IEEG_RE.match(fname)
            if not m:
                skipped.append(fname)
                continue
            subject = m.group("subject")
            ftask = m.group("ftask")
            acq = m.group("acq") or ""
            run = m.group("run") or ""
            ieeg_dir = os.path.dirname(edf)
            json_path = edf[: -len("_ieeg.edf")] + "_ieeg.json"
            if not os.path.exists(json_path):
                skipped.append(fname)
                continue
            with open(json_path) as fh:
                meta = json.load(fh)
            d_num, implant = parse_subject(subject)
            rows.append(
                ManifestRow(
                    subject_bids=subject,
                    d_num=d_num,
                    implant=implant,
                    global_subject_id=cogan_global_id(d_num, implant),
                    task=task,
                    acq=acq,
                    run=run,
                    trial_id=-1,  # filled by assign_trial_ids
                    edf_path=edf,
                    json_path=json_path,
                    channels_tsv_path=_channels_tsv_for(ieeg_dir, subject, ftask),
                    native_sfreq=float(meta["SamplingFrequency"]),
                    power_line_hz=float(meta.get("PowerLineFrequency", 60.0)),
                    duration_s=float(meta["RecordingDuration"]),
                )
            )
    if skipped:
        print(f"[enumerate_runs] skipped {len(skipped)} runs w/o json/unparseable: {skipped[:8]}")
    return rows


def assign_trial_ids(rows: list[ManifestRow]) -> list[ManifestRow]:
    """Per BIDS-subject, enumerate runs 0..N in sorted ``(task, acq, run)`` order.

    A subject appears across multiple tasks; ``trial_id`` must be unique per
    subject (it pairs with ``global_subject_id`` in the cache key), so it spans
    tasks, not per-task.
    """
    by_subject: dict[str, list[ManifestRow]] = {}
    for r in rows:
        by_subject.setdefault(r.subject_bids, []).append(r)
    out: list[ManifestRow] = []
    for subject in sorted(by_subject):
        ordered = sorted(by_subject[subject], key=lambda r: (r.task, r.acq, r.run))
        for tid, r in enumerate(ordered):
            out.append(ManifestRow(**{**asdict(r), "trial_id": tid}))
    return out


def _assert_ids_disjoint(rows: list[ManifestRow]) -> None:
    """One ``global_subject_id`` ↔ one distinct BIDS subject.

    A re-implant (``sub-D0107A``) collapses to the same ``d_num`` as a bare
    ``D0107`` / ``D0107B`` under ``cogan_global_id`` — distinct brains, colliding
    cache-key ids. If the corpus ever grows such a pair, fail LOUD here rather
    than let two subjects silently share a key namespace.
    """
    owner: dict[int, str] = {}
    for r in rows:
        prev = owner.setdefault(r.global_subject_id, r.subject_bids)
        if prev != r.subject_bids:
            raise ValueError(
                f"global_subject_id {r.global_subject_id} shared by distinct "
                f"subjects {prev!r} and {r.subject_bids!r} — re-implant/d_num "
                "collision; extend cogan_global_id to fold the implant suffix"
            )


def build(data_root: str) -> list[ManifestRow]:
    rows = assign_trial_ids(enumerate_runs(find_bids_roots(data_root)))
    _assert_ids_disjoint(rows)
    return rows


def write_csv(rows: list[ManifestRow], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = list(ManifestRow.__dataclass_fields__)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/hpc/group/coganlab/Data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = build(args.data_root)
    write_csv(rows, args.out)
    n_sub = len({r.subject_bids for r in rows})
    n_missing = sum(1 for r in rows if not r.channels_tsv_path)
    print(
        f"wrote {len(rows)} runs / {n_sub} BIDS subjects → {args.out} "
        f"({n_missing} runs missing channels.tsv)"
    )


if __name__ == "__main__":
    main()
