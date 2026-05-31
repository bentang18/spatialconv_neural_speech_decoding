"""Shared constants + manifest helpers for the SWEC sourcing scripts.

SWEC-iEEG dataset = ``NeuroTec/SWEC_iEEG_Dataset`` on HuggingFace (released
with MVPFormer, arXiv:2506.20354). See
``memory/reference_swec_ieeg_dataset_audit_2026_05_19.md`` for the full audit.

The committed per-subject manifest
``reports/swec_dataset_audit_2026_05_19/swec_per_subject_manifest.csv`` is the
single source of truth for *expected* metadata; the live HF revision is
re-verified against it by ``verify_swec_dedup.py`` before any download skips a
folder.
"""

from __future__ import annotations

import csv
import pathlib
import typing as tp

HF_REPO_ID = "NeuroTec/SWEC_iEEG_Dataset"
HF_REPO_TYPE = "dataset"
# HF commit pinned + re-verified by verify_swec_dedup.py on 2026-05-30 (50/18
# dedup reproduced from live data + bit-identical spot-checks). Fetch + audit
# bind to this so we source exactly the bytes we proved.
PINNED_REVISION = "584e9d29313ad6d2ed675b5d5202240f4ff75970"

# /work (75-day purge, bulk tier) per the 2026-05-30 storage decision. Mirrors
# the BrainTreebank precedent /work/ht203/data/braintreebank/. This script IS
# the documented regenerate spec the CLAUDE.md /work-cache discipline requires.
DCC_DEST = "/work/ht203/data/swec"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST_CSV = (
    REPO_ROOT
    / "reports"
    / "swec_dataset_audit_2026_05_19"
    / "swec_per_subject_manifest.csv"
)


class SubjectRow(tp.NamedTuple):
    folder: str
    channels: int
    sampling_rate_hz: int
    n_samples: int
    hours: float
    n_seizures: int
    duplicate_of: str  # "" if not a duplicate
    counts_as_unique_subject: bool


def load_manifest(path: pathlib.Path = MANIFEST_CSV) -> list[SubjectRow]:
    rows: list[SubjectRow] = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(
                SubjectRow(
                    folder=r["folder"],
                    channels=int(r["channels"]),
                    sampling_rate_hz=int(r["sampling_rate_hz"]),
                    n_samples=int(r["n_samples"]),
                    hours=float(r["hours"]),
                    n_seizures=int(r["n_seizures"]),
                    duplicate_of=r["duplicate_of"].strip(),
                    counts_as_unique_subject=r["counts_as_unique_subject"].strip()
                    == "True",
                )
            )
    return rows


def unique_folders(rows: list[SubjectRow] | None = None) -> list[str]:
    """The 50 unique download targets (skips the 18 re-export duplicates)."""
    rows = rows or load_manifest()
    return [r.folder for r in rows if r.counts_as_unique_subject]


def duplicate_map(rows: list[SubjectRow] | None = None) -> dict[str, str]:
    """folder -> folder-it-duplicates, for the 18 redundant re-exports."""
    rows = rows or load_manifest()
    return {r.folder: r.duplicate_of for r in rows if r.duplicate_of}
