"""Build the RAM (UPenn/Kahana OpenNeuro) run manifest — one row per
(subject, session, task, acq) = one cache — for the v3 ``RamCohortStudy``.

RAM is 5 CC0 OpenNeuro task-datasets (accessions), each a BIDS tree:

    <data_root>/<accession>/sub-R####X/ses-N/ieeg/
        sub-R####X_ses-N_task-<TASK>_acq-<monopolar|bipolar>_ieeg.edf
        sub-R####X_ses-N_task-<TASK>_acq-<monopolar|bipolar>_ieeg.json
        sub-R####X_ses-N_task-<TASK>_acq-<monopolar|bipolar>_channels.tsv
        sub-R####X_ses-N_task-<TASK>_space-MNI152NLin6ASym_electrodes.tsv

Unlike the Cogan tree, the run axis is ``ses-N`` (not acq/run) and every session
ships BOTH an ``acq-monopolar`` and an ``acq-bipolar`` re-referencing of the SAME
neural data as separate EDFs. We ingest exactly ONE acq (default ``monopolar``):
the v3 front-end (``CARIeegExtractor``) does its own shaft-CAR, so pre-``bipolar``
data would be double-differenced. ``--acq`` overrides.

RAM subject ids (``R1001P``) have no small-int core, so ``global_subject_id`` is
INDEX-assigned by ``corpus_ids.allocate_ram_ids`` over the sorted-unique R-id set
of the ENUMERATED cohort. That id is therefore cohort-dependent: change ``--datasets``
and the ids shift. To keep a built manifest reproducible, the ``rid -> global_subject_id``
map is written next to the CSV as ``ram_subject_id_map.json``.

Default cohort = the three fully-localizable datasets (``ind.region`` coverage
>=93%; memory: project-ram-sidecar-recon-qc-gate): FR1 + catFR1 + RepFR1. pyFR
(ds004865, 0% localized) and PAL1 (ds005059, no ``electrodes.tsv`` sidecar) are
EXCLUDED by default because their contacts would collapse to the DKT sentinel;
pass ``--datasets`` to include them.

Deliberately METADATA-ONLY: reads ``ieeg.json`` (native rate, mains, duration)
and checks ``channels.tsv`` existence; never opens an EDF. Runs on a login node;
unit-tests on a synthetic tree (no mne, no neural data).

Run on DCC (after the raw pull):
    python build_ram_manifest.py --data-root /work/ht203/ram_raw \\
        --out /work/ht203/ram_v3/manifest/ram_run_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import asdict, dataclass

from speech_decoding.models.v14_converged_v3.corpus_ids import allocate_ram_ids

# OpenNeuro accession -> canonical RAM task name. Default cohort is the three
# fully-localizable datasets; the two localization-poor ones are known but
# opt-in (see module docstring).
RAM_DATASETS: dict[str, str] = {
    "ds004789": "FR1",
    "ds004809": "catFR1",
    "ds005411": "RepFR1",
    # opt-in (localization-poor): must be named explicitly via --datasets
    "ds005059": "PAL1",    # no electrodes.tsv sidecar -> unlocalizable
    "ds004865": "pyFR",    # 0% ind.region coverage
}
DEFAULT_DATASETS: tuple[str, ...] = ("ds004789", "ds004809", "ds005411")

# sub-R1001P_ses-0_task-FR1_acq-monopolar_ieeg.edf  (ses optional for safety)
_IEEG_RE = re.compile(
    r"^(?P<subject>sub-R[A-Za-z0-9]+?)"
    r"(?:_ses-(?P<ses>[A-Za-z0-9]+))?"
    r"_task-(?P<task>[A-Za-z0-9]+)"
    r"_acq-(?P<acq>[A-Za-z0-9]+)_ieeg\.edf$"
)
_SUBJECT_RE = re.compile(r"^sub-(?P<rid>R[A-Za-z0-9]+)$")


@dataclass(frozen=True)
class ManifestRow:
    subject_bids: str        # sub-R1001P
    rid: str                 # R1001P
    global_subject_id: int   # 3000 + index (filled after id allocation)
    dataset: str             # OpenNeuro accession (ds004789)
    task: str                # canonical task (FR1)
    ses: str                 # "0" (or "" when the tree is session-less)
    acq: str                 # "monopolar"
    trial_id: int            # per-subject run index (assigned later)
    edf_path: str
    json_path: str
    channels_tsv_path: str   # "" when absent (honesty flag)
    native_sfreq: float
    power_line_hz: float
    duration_s: float


def parse_subject(subject_bids: str) -> str:
    """``sub-R1001P`` -> ``R1001P``."""
    m = _SUBJECT_RE.fullmatch(subject_bids)
    if not m:
        raise ValueError(f"unrecognised RAM BIDS subject {subject_bids!r}")
    return m.group("rid")


def _iter_edfs(dataset_root: str, acq: str) -> list[str]:
    """All ``_acq-<acq>_ieeg.edf`` under a dataset, session-nested or flat."""
    patterns = (
        os.path.join(dataset_root, "sub-R*", "ses-*", "ieeg", f"*_acq-{acq}_ieeg.edf"),
        os.path.join(dataset_root, "sub-R*", "ieeg", f"*_acq-{acq}_ieeg.edf"),
    )
    hits: list[str] = []
    for pat in patterns:
        hits.extend(glob.glob(pat))
    return sorted(set(hits))


def enumerate_runs(
    data_root: str, datasets: tuple[str, ...], acq: str
) -> list[ManifestRow]:
    """Walk each selected dataset -> one un-id'd/un-trial'd ``ManifestRow`` per EDF.

    EDFs whose ``_ieeg.json`` sidecar is absent are SKIPPED (no native rate /
    duration). ``global_subject_id`` is left -1 here and filled by
    ``assign_ids`` once the full cohort R-id set is known.
    """
    rows: list[ManifestRow] = []
    skipped: list[str] = []
    for ds in datasets:
        task = RAM_DATASETS[ds]
        for edf in _iter_edfs(os.path.join(data_root, ds), acq):
            fname = os.path.basename(edf)
            m = _IEEG_RE.match(fname)
            if not m:
                skipped.append(fname)
                continue
            subject = m.group("subject")
            ses = m.group("ses") or ""
            json_path = edf[: -len("_ieeg.edf")] + "_ieeg.json"
            if not os.path.exists(json_path):
                skipped.append(fname)
                continue
            with open(json_path) as fh:
                meta = json.load(fh)
            channels = edf[: -len("_ieeg.edf")] + "_channels.tsv"
            rows.append(
                ManifestRow(
                    subject_bids=subject,
                    rid=parse_subject(subject),
                    global_subject_id=-1,
                    dataset=ds,
                    task=task,
                    ses=ses,
                    acq=m.group("acq"),
                    trial_id=-1,
                    edf_path=edf,
                    json_path=json_path,
                    channels_tsv_path=channels if os.path.exists(channels) else "",
                    native_sfreq=float(meta["SamplingFrequency"]),
                    power_line_hz=float(meta.get("PowerLineFrequency", 60.0)),
                    duration_s=float(meta["RecordingDuration"]),
                )
            )
    if skipped:
        print(f"[enumerate_runs] skipped {len(skipped)} EDFs w/o json/unparseable: {skipped[:8]}")
    return rows


def assign_ids(rows: list[ManifestRow]) -> tuple[list[ManifestRow], dict[str, int]]:
    """Allocate cohort-wide ``global_subject_id`` (index-based) into every row.

    Returns the updated rows and the ``rid -> global_subject_id`` map (persisted
    for reproducibility — the ids depend on the enumerated cohort).
    """
    rids = sorted({r.rid for r in rows})
    id_map = allocate_ram_ids(rids)
    out = [ManifestRow(**{**asdict(r), "global_subject_id": id_map[r.rid]}) for r in rows]
    return out, id_map


def assign_trial_ids(rows: list[ManifestRow]) -> list[ManifestRow]:
    """Per subject, enumerate runs 0..N in sorted ``(dataset, task, ses, acq)`` order.

    A subject appears across multiple datasets/tasks/sessions; ``trial_id`` must be
    unique per subject (it pairs with ``global_subject_id`` in the cache key), so
    it spans everything, not per-dataset.
    """
    by_subject: dict[str, list[ManifestRow]] = {}
    for r in rows:
        by_subject.setdefault(r.subject_bids, []).append(r)
    out: list[ManifestRow] = []
    for subject in sorted(by_subject):
        ordered = sorted(
            by_subject[subject], key=lambda r: (r.dataset, r.task, r.ses, r.acq)
        )
        for tid, r in enumerate(ordered):
            out.append(ManifestRow(**{**asdict(r), "trial_id": tid}))
    return out


def _assert_ids_disjoint(rows: list[ManifestRow]) -> None:
    """One ``global_subject_id`` <-> one distinct BIDS subject."""
    owner: dict[int, str] = {}
    for r in rows:
        prev = owner.setdefault(r.global_subject_id, r.subject_bids)
        if prev != r.subject_bids:
            raise ValueError(
                f"global_subject_id {r.global_subject_id} shared by distinct "
                f"subjects {prev!r} and {r.subject_bids!r}"
            )


def build(
    data_root: str, datasets: tuple[str, ...], acq: str
) -> tuple[list[ManifestRow], dict[str, int]]:
    rows, id_map = assign_ids(enumerate_runs(data_root, datasets, acq))
    rows = assign_trial_ids(rows)
    _assert_ids_disjoint(rows)
    return rows, id_map


def write_outputs(
    rows: list[ManifestRow], id_map: dict[str, int], out_path: str
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = list(ManifestRow.__dataclass_fields__)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    map_path = os.path.join(os.path.dirname(out_path), "ram_subject_id_map.json")
    with open(map_path, "w") as fh:
        json.dump(id_map, fh, indent=2, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="dir containing the OpenNeuro accession subdirs (ds004789, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--acq", default="monopolar", choices=["monopolar", "bipolar"])
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                    help=f"accessions to include (known: {sorted(RAM_DATASETS)})")
    args = ap.parse_args()
    bad = [d for d in args.datasets if d not in RAM_DATASETS]
    if bad:
        ap.error(f"unknown datasets {bad}; known: {sorted(RAM_DATASETS)}")
    rows, id_map = build(args.data_root, tuple(args.datasets), args.acq)
    write_outputs(rows, id_map, args.out)
    n_sub = len(id_map)
    n_missing = sum(1 for r in rows if not r.channels_tsv_path)
    print(
        f"wrote {len(rows)} runs / {n_sub} subjects (acq={args.acq}, "
        f"datasets={args.datasets}) -> {args.out} "
        f"({n_missing} runs missing channels.tsv)"
    )


if __name__ == "__main__":
    main()
