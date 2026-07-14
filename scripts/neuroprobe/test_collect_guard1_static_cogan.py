"""Tests for the Cogan guard-1 static-drop collector.

Covers the pure glue — signature loading + schema guard, manifest key parsing,
full-coverage enforcement, and an end-to-end ``main()`` run whose emitted map is
checked against a hand-built signature (spike via clip_frac, clean run stays []).
The detector thresholds themselves are guard1's contract (tested there); here we
only prove the collector wires scan → ``per_session_static_drops`` → JSON map.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

import collect_guard1_static_cogan as col
from speech_decoding.studies.braintreebank import guard1

_MAN_COLS = [
    "subject_bids", "d_num", "implant", "global_subject_id", "task", "acq",
    "run", "trial_id", "edf_path", "json_path", "channels_tsv_path",
    "native_sfreq", "power_line_hz", "duration_s",
]


def _signature(sid: int, tid: int, labels, clip_frac, rmad) -> dict:
    """A precompute-shaped signature JSON (= guard1.session_signature output)."""
    cls = guard1.classify_from_signature(labels, np.asarray(clip_frac), np.asarray(rmad))
    static = sorted(cls["spike"] | cls["noisy"] | cls["dead"])
    return {
        "subject": sid, "trial": tid, "labels": list(labels),
        "clip_frac": list(map(float, clip_frac)), "rmad": list(map(float, rmad)),
        "spike": sorted(cls["spike"]), "noisy": sorted(cls["noisy"]),
        "dead": sorted(cls["dead"]), "static": static,
    }


def _write_sig(dirpath, sig: dict) -> None:
    p = dirpath / f"cogan{sig['subject']}_t{sig['trial']}.json"
    p.write_text(json.dumps(sig))


def _write_manifest(path, keys: list[tuple[int, int]]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_MAN_COLS)
        w.writeheader()
        for sid, tid in keys:
            w.writerow({c: "" for c in _MAN_COLS}
                       | {"global_subject_id": sid, "trial_id": tid})


def _clean_and_spiked(det):
    # One clean run (uniform rmad, no clip) and one with a single spiked contact.
    labels = ["LA1", "LA2", "LA3", "LA4"]
    _write_sig(det, _signature(1005, 0, labels, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]))
    _write_sig(det, _signature(1005, 1, labels, [0.0, 0.05, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]))
    return labels


def test_load_sessions_requires_full_schema(tmp_path):
    det = tmp_path / "det"; det.mkdir()
    (det / "cogan1_t0.json").write_text(json.dumps({"subject": 1, "trial": 0}))
    with pytest.raises(SystemExit):
        col.load_sessions(det)


def test_load_sessions_empty_dir_raises(tmp_path):
    det = tmp_path / "det"; det.mkdir()
    with pytest.raises(SystemExit):
        col.load_sessions(det)


def test_manifest_run_keys_parses_global_id_and_trial(tmp_path):
    m = tmp_path / "m.csv"
    _write_manifest(m, [(1005, 0), (1066, 15)])
    assert col.manifest_run_keys(m) == {(1005, 0), (1066, 15)}


def test_end_to_end_emits_per_run_static_map(tmp_path, monkeypatch):
    det = tmp_path / "det"; det.mkdir()
    _clean_and_spiked(det)
    man = tmp_path / "m.csv"
    _write_manifest(man, [(1005, 0), (1005, 1)])
    out = tmp_path / "static.json"

    monkeypatch.setattr("sys.argv", [
        "prog", "--detections-dir", str(det), "--manifest", str(man), "--out", str(out),
    ])
    col.main()

    report = json.loads(out.read_text())
    assert report["n_runs"] == 2
    assert report["per_run"]["1005_0"] == []          # clean run
    assert report["per_run"]["1005_1"] == ["LA2"]     # clip_frac 0.05 > 0.01 -> spike
    assert report["n_runs_with_drop"] == 1
    assert report["n_contacts_dropped"] == 1


def test_end_to_end_missing_run_fails_loud(tmp_path, monkeypatch):
    det = tmp_path / "det"; det.mkdir()
    _clean_and_spiked(det)
    man = tmp_path / "m.csv"
    _write_manifest(man, [(1005, 0), (1005, 1), (1005, 2)])  # t2 has no signature
    out = tmp_path / "static.json"
    monkeypatch.setattr("sys.argv", [
        "prog", "--detections-dir", str(det), "--manifest", str(man), "--out", str(out),
    ])
    with pytest.raises(SystemExit):
        col.main()
