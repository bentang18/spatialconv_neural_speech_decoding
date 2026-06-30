"""Long-format results ledger for the pretrain probe suite (Ben's tracking spreadsheet).

One CSV row per scored cell-or-aggregate: ``(stamp, ckpt, readout, tap, eval_mode, task,
split, auroc, n, lam, notes)``. Long format so every run APPENDS and the file opens as a
spreadsheet — pivot ckpt×tap×readout, filter by task, track over time. Lives under
``reports/`` (gitignored): it carries BT-derived numbers, never the public repo.

The writer is pure + unit-tested; the dispatch runner stamps ``stamp`` (Date.now is
unavailable in workflows but this is plain Python) and appends.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass

__all__ = [
    "FIELDS",
    "ResultRow",
    "append_results",
    "read_results",
]

FIELDS = (
    "stamp",       # ISO timestamp of the run
    "ckpt",        # e.g. "ladder-60000", or "raw_371" for the model-free floor
    "readout",     # "ridge" | "attentive"
    "tap",         # "raw_371" | "raw_tok" | "M2" | "M3" | "M4"
    "eval_mode",   # "WithinSession" | "CrossSubject"
    "task",        # one of NEUROPROBE_TASKS, or "MEAN" for an aggregate row
    "split",       # "test" | "val"
    "auroc",       # float
    "n",           # cells averaged (aggregate) or test rows (cell)
    "lam",         # ridge λ used (or "" for attentive)
    "notes",       # free text (e.g. anchor, hp combo, firewall tag)
)


@dataclass(frozen=True)
class ResultRow:
    stamp: str
    ckpt: str
    readout: str
    tap: str
    eval_mode: str
    task: str
    split: str
    auroc: float
    n: int
    lam: str = ""
    notes: str = ""


def append_results(rows: list[ResultRow], path: str) -> None:
    """Append rows to the ledger CSV, writing the header iff the file is new/empty."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def read_results(path: str) -> list[dict[str, str]]:
    """Read the ledger back as a list of dict rows (strings, as CSV stores them)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
