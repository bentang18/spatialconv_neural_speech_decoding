"""Tests for the Cogan guard-1 static-bad accessor + its iter_timelines fold-in.

Covers the governance-flexible map lookup (env override, absent→empty, full-report
vs bare-map schema) and that DCohortStudy.iter_timelines injects ``extra_bad``
CONDITIONALLY (only for runs with a drop), so a clean run's cache key is unchanged.
"""

from __future__ import annotations

import csv
import json

import pytest

from speech_decoding.studies.cogan_dcohort import guard1_static
from speech_decoding.studies.cogan_dcohort.study import DCohortStudy


@pytest.fixture(autouse=True)
def _clear_map_cache():
    guard1_static._load_map.cache_clear()
    yield
    guard1_static._load_map.cache_clear()


def _write_map(path, per_run, wrap=True):
    payload = {"per_run": per_run} if wrap else per_run
    path.write_text(json.dumps(payload))
    return str(path)


def test_absent_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv(guard1_static._ENV_VAR, str(tmp_path / "nope.json"))
    assert guard1_static.cogan_extra_bad(1019, 0) == frozenset()


def test_reads_full_report_schema(tmp_path, monkeypatch):
    m = _write_map(tmp_path / "g1.json", {"1019_0": ["LA1", "LA2"], "1019_1": []})
    monkeypatch.setenv(guard1_static._ENV_VAR, m)
    assert guard1_static.cogan_extra_bad(1019, 0) == frozenset({"LA1", "LA2"})
    assert guard1_static.cogan_extra_bad(1019, 1) == frozenset()   # clean run
    assert guard1_static.cogan_extra_bad(1607, 0) == frozenset()   # unmapped run


def test_accepts_bare_map(tmp_path, monkeypatch):
    m = _write_map(tmp_path / "g1.json", {"1019_0": ["LX3"]}, wrap=False)
    monkeypatch.setenv(guard1_static._ENV_VAR, m)
    assert guard1_static.cogan_extra_bad(1019, 0) == frozenset({"LX3"})


# ---- iter_timelines fold-in --------------------------------------------------

_ROWS = [
    dict(subject_bids="sub-D0019", global_subject_id=1019, trial_id=0,
         edf_path="/d/a.edf", channels_tsv_path="/d/a.tsv", duration_s=893.0),
    dict(subject_bids="sub-D0019", global_subject_id=1019, trial_id=1,
         edf_path="/d/b.edf", channels_tsv_path="/d/b.tsv", duration_s=900.0),
]


def _manifest(path):
    fields = ["subject_bids", "global_subject_id", "trial_id", "edf_path",
              "channels_tsv_path", "duration_s"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in _ROWS:
            w.writerow({k: r[k] for k in fields})
    return str(path)


def test_iter_timelines_folds_extra_bad_conditionally(tmp_path, monkeypatch):
    # run (1019,0) drops two contacts; (1019,1) is clean.
    m = _write_map(tmp_path / "g1.json", {"1019_0": ["LB4", "LB5"], "1019_1": []})
    monkeypatch.setenv(guard1_static._ENV_VAR, m)
    manifest = _manifest(tmp_path / "cogan_manifest.csv")
    study = DCohortStudy(path=str(tmp_path), manifest_path=manifest)

    tls = {(t["subject_id"], t["trial_id"]): t for t in study.iter_timelines()}
    assert tls[(1019, 0)]["extra_bad"] == ["LB4", "LB5"]  # sorted, injected
    assert "extra_bad" not in tls[(1019, 1)]               # clean run: absent


def test_iter_timelines_no_map_no_injection(tmp_path, monkeypatch):
    monkeypatch.setenv(guard1_static._ENV_VAR, str(tmp_path / "absent.json"))
    manifest = _manifest(tmp_path / "cogan_manifest.csv")
    study = DCohortStudy(path=str(tmp_path), manifest_path=manifest)
    assert all("extra_bad" not in t for t in study.iter_timelines())
