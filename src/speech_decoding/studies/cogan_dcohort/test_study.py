"""DCohortStudy v3-contract tests: manifest-driven timelines + cache-key round-trip.

Exercises the glue that does NOT need mne/EDF data (iter_timelines,
_load_timeline_events, _cls_kwargs, row lookup). _load_raw's EDF read is covered
on DCC against real runs; here we only assert its pre-read guard.
"""

from __future__ import annotations

import csv

import pytest

from speech_decoding.models.v14_converged_v3.cache_index import parse_key_session
from speech_decoding.studies.cogan_dcohort.study import DCohortStudy

# One subject with two runs (trial 0,1) + a re-implant subject (global id 1607).
_ROWS = [
    dict(subject_bids="sub-D0019", global_subject_id=1019, trial_id=0,
         edf_path="/data/a_ieeg.edf", channels_tsv_path="/data/a_channels.tsv",
         duration_s=893.0),
    dict(subject_bids="sub-D0019", global_subject_id=1019, trial_id=1,
         edf_path="/data/b_ieeg.edf", channels_tsv_path="/data/b_channels.tsv",
         duration_s=900.0),
    dict(subject_bids="sub-D0107A", global_subject_id=1607, trial_id=0,
         edf_path="/data/c_ieeg.edf", channels_tsv_path="/data/c_channels.tsv",
         duration_s=500.0),
]


def _write_manifest(path, rows=_ROWS):
    fields = ["subject_bids", "global_subject_id", "trial_id", "edf_path",
              "channels_tsv_path", "duration_s"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    return str(path)


def _study(tmp_path, manifest_csv, **kw):
    # base `path` is a study DATA DIR (validator mkdir's it); manifest is a file.
    return DCohortStudy(path=str(tmp_path), manifest_path=manifest_csv, **kw)


@pytest.fixture
def manifest(tmp_path):
    return _write_manifest(tmp_path / "cogan_manifest.csv")


def test_classvars():
    assert DCohortStudy.SAMPLE_RATE_HZ == 2048.0
    assert "DCohort" in DCohortStudy.aliases


def test_iter_timelines_shapes_and_types(tmp_path, manifest):
    tls = list(_study(tmp_path, manifest).iter_timelines())
    assert len(tls) == 3
    tl = tls[0]
    assert tl["subject"] == "sub-D0019" and isinstance(tl["subject"], str)
    assert tl["subject_id"] == 1019 and isinstance(tl["subject_id"], int)
    assert tl["trial_id"] == 0 and isinstance(tl["trial_id"], int)
    assert {(t["subject_id"], t["trial_id"]) for t in tls} == {
        (1019, 0), (1019, 1), (1607, 0)
    }


def test_load_timeline_events_row_and_cache_key_roundtrip(tmp_path, manifest):
    s = _study(tmp_path, manifest)
    tl = {"subject": "sub-D0107A", "subject_id": 1607, "trial_id": 0}
    df = s._load_timeline_events(tl)
    assert list(df.columns) >= ["type", "start", "duration", "frequency", "filepath"]
    row = df.iloc[0]
    assert row["type"] == "Ieeg"
    assert row["start"] == 0.0
    assert row["duration"] == 500.0            # from manifest duration_s
    assert row["frequency"] == 2048.0          # post-resample rate, NOT native
    # The consumer regex must recover the exact ids from the cache key.
    assert parse_key_session(row["filepath"]) == (1607, 0)


def test_session_subset_filters(tmp_path, manifest):
    s = _study(tmp_path, manifest, session_subset=((1019, 1),))
    tls = list(s.iter_timelines())
    assert [(t["subject_id"], t["trial_id"]) for t in tls] == [(1019, 1)]


def test_session_subset_unknown_raises(tmp_path, manifest):
    s = _study(tmp_path, manifest, session_subset=((9999, 0),))
    with pytest.raises(ValueError, match="not in manifest"):
        list(s.iter_timelines())


def test_cls_kwargs_drops_enumeration_fields(tmp_path, manifest):
    s = _study(tmp_path, manifest, session_subset=((1019, 0),))
    assert s._cls_kwargs() == {}  # path + session_subset dropped, no raise


def test_manifest_path_resolves_from_env_after_roundtrip(tmp_path, manifest, monkeypatch):
    """A study reconstructed by SpecialLoader.from_json has manifest_path=='' (dropped
    from the uid); _load_raw must still find the CSV via the COGAN_MANIFEST env that
    build_cogan_cache_experiment publishes — otherwise the bake dies at load time."""
    reconstructed = DCohortStudy(path=str(tmp_path), manifest_path="")
    monkeypatch.setenv("COGAN_MANIFEST", manifest)
    assert reconstructed._resolved_manifest_path() == manifest
    # and enumeration off the resolved CSV yields the same rows the field-set study does
    assert [t["subject_id"] for t in reconstructed.iter_timelines()] == [
        t["subject_id"] for t in _study(tmp_path, manifest).iter_timelines()
    ]


def test_manifest_path_unset_and_no_env_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("COGAN_MANIFEST", raising=False)
    with pytest.raises(ValueError, match="COGAN_MANIFEST env is empty"):
        DCohortStudy(path=str(tmp_path), manifest_path="")._resolved_manifest_path()


def test_row_for_missing_session_raises(tmp_path, manifest):
    s = _study(tmp_path, manifest)
    with pytest.raises(KeyError, match="not in manifest"):
        s._row_for({"subject_id": 1234, "trial_id": 7})


def test_manifest_missing_columns_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    with open(bad, "w", newline="") as fh:
        fh.write("subject_bids,trial_id\nsub-D0019,0\n")
    with pytest.raises(ValueError, match="missing columns"):
        list(_study(tmp_path, str(bad)).iter_timelines())


def test_load_raw_empty_channels_tsv_raises(tmp_path):
    rows = [dict(subject_bids="sub-D0019", global_subject_id=1019, trial_id=0,
                 edf_path="/data/a_ieeg.edf", channels_tsv_path="",
                 duration_s=100.0)]
    m = _write_manifest(tmp_path / "m.csv", rows)
    s = _study(tmp_path, m)
    with pytest.raises(ValueError, match="no channels.tsv"):
        s._load_raw({"subject_id": 1019, "trial_id": 0})
