"""Tests for the Cogan guard-1 scan glue (manifest parse + array-index resolution).

The scan itself (``scan_run``) is DCC-only (needs mne + real EDF), so it is not
covered here — only the pure-python row resolution that decides WHICH runs a
Slurm array task scans, which is where an off-by-one or a column-name typo would
silently mis-scan the corpus.
"""

from __future__ import annotations

import csv

import pytest

import precompute_guard1_static_cogan as g1c

_COLS = [
    "subject_bids", "d_num", "implant", "global_subject_id", "task", "acq",
    "run", "trial_id", "edf_path", "json_path", "channels_tsv_path",
    "native_sfreq", "power_line_hz", "duration_s",
]


def _write_manifest(path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _COLS})


def _row(gsid: int, tid: int, **kw) -> dict:
    base = {
        "global_subject_id": gsid,
        "trial_id": tid,
        "edf_path": f"/data/sub{gsid}_r{tid}.edf",
        "channels_tsv_path": f"/data/sub{gsid}_r{tid}_channels.tsv",
        "power_line_hz": 60.0,
    }
    base.update(kw)
    return base


def test_read_manifest_keeps_typed_fields_in_row_order(tmp_path):
    p = tmp_path / "m.csv"
    _write_manifest(p, [_row(1005, 0), _row(1066, 15, power_line_hz=50.0)])
    m = g1c.read_manifest(str(p))
    assert [(r["subject_id"], r["trial_id"]) for r in m] == [(1005, 0), (1066, 15)]
    assert m[0]["subject_id"] == 1005 and isinstance(m[0]["subject_id"], int)
    assert m[0]["trial_id"] == 0 and isinstance(m[0]["trial_id"], int)
    assert m[0]["edf_path"].endswith("sub1005_r0.edf")
    assert m[0]["channels_tsv_path"].endswith("sub1005_r0_channels.tsv")
    # power_line_hz is read per-run (60 for Cogan; a 50 Hz corpus would carry 50).
    assert m[0]["power_line_hz"] == 60.0 and m[1]["power_line_hz"] == 50.0


def test_resolve_rows_explicit_row_overrides_array(monkeypatch):
    m = [_row_d(1, 0), _row_d(2, 1), _row_d(3, 2)]
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")  # would pick index 2...
    assert g1c.resolve_rows(m, row_arg=1) == [m[1]]  # ...but --row wins


def test_resolve_rows_array_index_selects_one(monkeypatch):
    m = [_row_d(1, 0), _row_d(2, 1), _row_d(3, 2)]
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    assert g1c.resolve_rows(m, row_arg=None) == [m[0]]
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    assert g1c.resolve_rows(m, row_arg=None) == [m[2]]


def test_resolve_rows_no_array_no_row_scans_all(monkeypatch):
    m = [_row_d(1, 0), _row_d(2, 1)]
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    assert g1c.resolve_rows(m, row_arg=None) == m


def test_resolve_rows_out_of_range_array_raises(monkeypatch):
    m = [_row_d(1, 0)]
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "5")
    with pytest.raises(IndexError):
        g1c.resolve_rows(m, row_arg=None)


def _row_d(sid: int, tid: int) -> dict:
    """Resolved-row dict (post read_manifest), for resolve_rows tests."""
    return {"subject_id": sid, "trial_id": tid, "edf_path": "", "channels_tsv_path": "",
            "power_line_hz": 60.0}
