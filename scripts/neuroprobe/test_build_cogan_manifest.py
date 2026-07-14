"""Tests for build_cogan_manifest on a synthetic BIDS tree (no mne, no DCC)."""

from __future__ import annotations

import json
import os

import pytest

import build_cogan_manifest as bm


def _mk_run(ieeg_dir, subject, ftask, acq, run, sfreq, dur,
            with_channels=True, with_json=True):
    os.makedirs(ieeg_dir, exist_ok=True)
    stem = f"{subject}_task-{ftask}_acq-{acq}_run-{run}"
    open(os.path.join(ieeg_dir, stem + "_ieeg.edf"), "w").close()
    if with_json:
        with open(os.path.join(ieeg_dir, stem + "_ieeg.json"), "w") as fh:
            json.dump(
                {"SamplingFrequency": sfreq, "RecordingDuration": dur,
                 "PowerLineFrequency": 60}, fh
            )
    if with_channels:
        ch = os.path.join(ieeg_dir, f"{subject}_task-{ftask}_channels.tsv")
        if not os.path.exists(ch):
            with open(ch, "w") as fh:
                fh.write("name\ttype\n ROG1\tECOG\n")


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "Data"
    # Two tasks, dir name BIDS-<ver>_<task>; filename ftask differs (camelCase).
    ps = root / "BIDS-1.4_Phoneme_sequencing" / "BIDS" / "sub-D0019" / "ieeg"
    _mk_run(str(ps), "sub-D0019", "PhonemeSequence", "01", "01", 2000, 893)
    _mk_run(str(ps), "sub-D0019", "PhonemeSequence", "01", "02", 2000, 900)
    sr = root / "BIDS-1.4_SentenceRep" / "BIDS" / "sub-D0019" / "ieeg"
    _mk_run(str(sr), "sub-D0019", "SentenceRep", "01", "01", 2048, 4225)
    # A second subject on TIMIT with NO channels.tsv (the D38/D39/D54 case).
    tm = root / "BIDS-1.0_TIMIT" / "BIDS" / "sub-D0038" / "ieeg"
    _mk_run(str(tm), "sub-D0038", "TIMIT", "01", "01", 2048, 1785, with_channels=False)
    return str(root)


def test_find_roots(tree):
    roots = bm.find_bids_roots(tree)
    assert set(roots) == {"Phoneme_sequencing", "SentenceRep", "TIMIT"}


def test_enumerate_and_fields(tree):
    rows = bm.enumerate_runs(bm.find_bids_roots(tree))
    assert len(rows) == 4
    ps01 = next(r for r in rows if r.task == "Phoneme_sequencing" and r.run == "01")
    assert ps01.d_num == 19
    assert ps01.global_subject_id == 1019
    assert ps01.native_sfreq == 2000.0
    assert ps01.power_line_hz == 60.0
    assert ps01.duration_s == 893.0
    assert ps01.channels_tsv_path.endswith("_channels.tsv")


def test_missing_channels_tsv_flagged_empty(tree):
    rows = bm.enumerate_runs(bm.find_bids_roots(tree))
    timit = next(r for r in rows if r.task == "TIMIT")
    assert timit.channels_tsv_path == ""


def test_trial_ids_span_tasks_sorted(tree):
    rows = bm.assign_trial_ids(bm.enumerate_runs(bm.find_bids_roots(tree)))
    d19 = sorted(
        (r for r in rows if r.subject_bids == "sub-D0019"), key=lambda r: r.trial_id
    )
    # sorted by (task, acq, run): Phoneme_sequencing run01, run02, then SentenceRep
    assert [r.trial_id for r in d19] == [0, 1, 2]
    assert [(r.task, r.run) for r in d19] == [
        ("Phoneme_sequencing", "01"),
        ("Phoneme_sequencing", "02"),
        ("SentenceRep", "01"),
    ]
    # trial_id unique per subject
    assert len({r.trial_id for r in d19}) == 3


def test_subject_parsing():
    assert bm.parse_subject("sub-D0019") == (19, "")
    assert bm.parse_subject("sub-D0107") == (107, "")
    assert bm.parse_subject("sub-D0107A") == (107, "A")
    with pytest.raises(ValueError):
        bm.parse_subject("sub-S0001")


def test_reimplant_suffix_survives_into_manifest(tmp_path):
    root = tmp_path / "Data"
    gl = root / "BIDS-1.0_GlobalLocal" / "BIDS" / "sub-D0107A" / "ieeg"
    _mk_run(str(gl), "sub-D0107A", "GlobalLocal", "01", "01", 2048, 500)
    rows = bm.enumerate_runs(bm.find_bids_roots(str(root)))
    assert len(rows) == 1
    assert rows[0].subject_bids == "sub-D0107A"
    assert rows[0].d_num == 107
    assert rows[0].implant == "A"


def test_run_missing_json_is_skipped(tmp_path):
    root = tmp_path / "Data"
    tm = root / "BIDS-1.0_TIMIT" / "BIDS" / "sub-D0038" / "ieeg"
    _mk_run(str(tm), "sub-D0038", "TIMIT", "01", "01", 2048, 100)  # complete
    _mk_run(str(tm), "sub-D0038", "TIMIT", "01", "02", 2048, 100,
            with_channels=False, with_json=False)  # incomplete
    rows = bm.enumerate_runs(bm.find_bids_roots(str(root)))
    assert [r.run for r in rows] == ["01"]


def test_reimplant_and_bare_get_distinct_ids(tmp_path):
    # The real corpus case: bare D0107 AND re-implant D0107A coexist. They are
    # distinct brains, so they must get DISTINCT global ids (1107 vs 1607) and
    # build() must NOT trip the disjointness guard.
    root = tmp_path / "Data"
    a = root / "BIDS-1.0_GlobalLocal" / "BIDS" / "sub-D0107A" / "ieeg"
    b = root / "BIDS-1.0_GlobalLocal" / "BIDS" / "sub-D0107" / "ieeg"
    _mk_run(str(a), "sub-D0107A", "GlobalLocal", "01", "01", 2048, 500)
    _mk_run(str(b), "sub-D0107", "GlobalLocal", "01", "01", 2048, 500)
    rows = bm.build(str(root))  # no raise
    by_sub = {r.subject_bids: r.global_subject_id for r in rows}
    assert by_sub == {"sub-D0107": 1107, "sub-D0107A": 1607}


def test_write_csv_roundtrip(tree, tmp_path):
    rows = bm.build(tree)
    out = str(tmp_path / "m" / "manifest.csv")
    bm.write_csv(rows, out)
    import csv as _csv
    with open(out) as fh:
        got = list(_csv.DictReader(fh))
    assert len(got) == 4
    assert set(got[0]) == set(bm.ManifestRow.__dataclass_fields__)
