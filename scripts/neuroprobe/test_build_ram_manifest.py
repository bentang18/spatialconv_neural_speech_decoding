"""Tests for build_ram_manifest on a synthetic BIDS tree (no mne, no S3)."""

from __future__ import annotations

import json
import os

import pytest

import build_ram_manifest as bm


def _mk_run(ieeg_dir, subject, ses, task, acq, sfreq, dur,
            with_channels=True, with_json=True):
    os.makedirs(ieeg_dir, exist_ok=True)
    sp = f"_ses-{ses}" if ses else ""
    stem = f"{subject}{sp}_task-{task}_acq-{acq}"
    open(os.path.join(ieeg_dir, stem + "_ieeg.edf"), "w").close()
    if with_json:
        with open(os.path.join(ieeg_dir, stem + "_ieeg.json"), "w") as fh:
            json.dump({"SamplingFrequency": sfreq, "RecordingDuration": dur,
                       "PowerLineFrequency": 60}, fh)
    if with_channels:
        with open(os.path.join(ieeg_dir, stem + "_channels.tsv"), "w") as fh:
            fh.write("name\ttype\tgroup\tsampling_frequency\nLAF1\tSEEG\tLAF\t500\n")


def _ieeg_dir(root, ds, subject, ses):
    sp = f"ses-{ses}" if ses else "."
    return os.path.join(str(root), ds, subject, sp, "ieeg")


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "ram_raw"
    # sub-R1001P: FR1 ses-0 (mono+bi), FR1 ses-1 (mono), catFR1 ses-0 (mono)
    _mk_run(_ieeg_dir(root, "ds004789", "sub-R1001P", "0"), "sub-R1001P", "0", "FR1", "monopolar", 500, 4698)
    _mk_run(_ieeg_dir(root, "ds004789", "sub-R1001P", "0"), "sub-R1001P", "0", "FR1", "bipolar", 500, 4698)
    _mk_run(_ieeg_dir(root, "ds004789", "sub-R1001P", "1"), "sub-R1001P", "1", "FR1", "monopolar", 500, 3000)
    _mk_run(_ieeg_dir(root, "ds004809", "sub-R1001P", "0"), "sub-R1001P", "0", "catFR1", "monopolar", 1000, 2500)
    # sub-R1002P: FR1 ses-0 mono, NO channels.tsv (honesty flag)
    _mk_run(_ieeg_dir(root, "ds004789", "sub-R1002P", "0"), "sub-R1002P", "0", "FR1", "monopolar", 500, 1200,
            with_channels=False)
    return str(root)


DS3 = ("ds004789", "ds004809", "ds005411")


def test_acq_filter_monopolar_only(tree):
    rows = bm.enumerate_runs(tree, DS3, "monopolar")
    # 3 R1001P mono + 1 R1002P mono = 4; the bipolar EDF is excluded.
    assert len(rows) == 4
    assert all(r.acq == "monopolar" for r in rows)


def test_acq_filter_bipolar(tree):
    rows = bm.enumerate_runs(tree, DS3, "bipolar")
    assert len(rows) == 1 and rows[0].acq == "bipolar"


def test_fields(tree):
    rows = bm.enumerate_runs(tree, DS3, "monopolar")
    fr1_s0 = next(r for r in rows if r.dataset == "ds004789" and r.ses == "0"
                  and r.subject_bids == "sub-R1001P")
    assert fr1_s0.rid == "R1001P"
    assert fr1_s0.task == "FR1"
    assert fr1_s0.native_sfreq == 500.0
    assert fr1_s0.power_line_hz == 60.0
    assert fr1_s0.duration_s == 4698.0
    assert fr1_s0.channels_tsv_path.endswith("_channels.tsv")


def test_missing_channels_flagged_empty(tree):
    rows = bm.enumerate_runs(tree, DS3, "monopolar")
    r1002 = next(r for r in rows if r.subject_bids == "sub-R1002P")
    assert r1002.channels_tsv_path == ""


def test_missing_json_skipped(tmp_path):
    root = tmp_path / "ram_raw"
    d = _ieeg_dir(root, "ds004789", "sub-R1001P", "0")
    _mk_run(d, "sub-R1001P", "0", "FR1", "monopolar", 500, 100)              # complete
    _mk_run(d, "sub-R1001P", "1", "FR1", "monopolar", 500, 100,
            with_json=False, with_channels=False)                            # incomplete
    rows = bm.enumerate_runs(str(root), DS3, "monopolar")
    assert [r.ses for r in rows] == ["0"]


def test_subject_parse():
    assert bm.parse_subject("sub-R1001P") == "R1001P"
    assert bm.parse_subject("sub-R1592J") == "R1592J"
    with pytest.raises(ValueError):
        bm.parse_subject("sub-D0019")


def test_index_based_ids_sorted(tree):
    rows, id_map = bm.assign_ids(bm.enumerate_runs(tree, DS3, "monopolar"))
    # Two subjects, sorted -> 3000, 3001; every row carries its subject's id.
    assert id_map == {"R1001P": 3000, "R1002P": 3001}
    got = {r.subject_bids: r.global_subject_id for r in rows}
    assert got == {"sub-R1001P": 3000, "sub-R1002P": 3001}


def test_trial_ids_span_datasets_sorted(tree):
    rows, _ = bm.assign_ids(bm.enumerate_runs(tree, DS3, "monopolar"))
    rows = bm.assign_trial_ids(rows)
    r1001 = sorted((r for r in rows if r.subject_bids == "sub-R1001P"),
                   key=lambda r: r.trial_id)
    # sorted by (dataset, task, ses); dataset<->task is 1:1 and ds004789<ds004809,
    # so: FR1 s0, FR1 s1, catFR1 s0
    assert [(r.dataset, r.ses) for r in r1001] == [
        ("ds004789", "0"), ("ds004789", "1"), ("ds004809", "0")]
    assert [r.trial_id for r in r1001] == [0, 1, 2]
    assert len({r.trial_id for r in r1001}) == 3


def test_ses_less_tree(tmp_path):
    root = tmp_path / "ram_raw"
    d = _ieeg_dir(root, "ds004789", "sub-R1001P", "")  # flat sub-R*/ieeg
    _mk_run(d, "sub-R1001P", "", "FR1", "monopolar", 500, 100)
    rows = bm.enumerate_runs(str(root), DS3, "monopolar")
    assert len(rows) == 1 and rows[0].ses == ""


def test_write_outputs_roundtrip(tree, tmp_path):
    rows, id_map = bm.build(tree, DS3, "monopolar")
    out = str(tmp_path / "m" / "ram_run_manifest.csv")
    bm.write_outputs(rows, id_map, out)
    import csv as _csv
    with open(out) as fh:
        got = list(_csv.DictReader(fh))
    assert len(got) == 4
    assert set(got[0]) == set(bm.ManifestRow.__dataclass_fields__)
    with open(os.path.join(os.path.dirname(out), "ram_subject_id_map.json")) as fh:
        assert json.load(fh) == {"R1001P": 3000, "R1002P": 3001}
