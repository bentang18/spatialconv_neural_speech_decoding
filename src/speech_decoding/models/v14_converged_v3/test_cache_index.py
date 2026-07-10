"""v14_converged_v3 Phase D1/D2 adapter — cache/label file parsers (TDD).

The thin disk-facing helpers the ``load_v3_sessions`` adapter composes: parse the
guard-2 bad-window spans (the DCC scan output), index a band's ``.npy`` spec cache
by scanning its ``{stem}.json`` sidecars (session ``key`` → paths + ch_names +
total_frames), and parse the LOF bad-channel report (guard-1 drop set). Pure
file-parsing — TDD'd on synthetic JSON here; wired to the real braintreebank
anatomy calls (``voltage_electrode_order`` / ``aligned_voltage_support``) inside
``load_v3_sessions``, validated on DeltaAI at F2.

Real ``bad_windows_s`` format confirmed against the DCC scan output
(btbank6_t4.json): a list of ``[start_s, end_s]`` pairs; the file also carries
``subject_id`` / ``trial_id``.
"""

from __future__ import annotations

import json

import numpy as np

from speech_decoding.models.v14_converged_v3.cache_index import (
    index_band_cache,
    index_bad_windows,
    load_bad_window_spans,
    parse_lof_report,
    parse_session_name,
)


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return str(path)


def test_load_bad_window_spans_reads_pairs(tmp_path) -> None:
    p = _write(tmp_path / "btbank6_t4.json", {
        "session": "btbank6_t4", "subject_id": 6, "trial_id": 4,
        "bad_windows_s": [[187.0, 188.0], [374.0, 375.0]], "frac_bad": 0.03,
    })
    spans = load_bad_window_spans(p)
    assert spans == [(187.0, 188.0), (374.0, 375.0)]


def test_load_bad_window_spans_empty(tmp_path) -> None:
    p = _write(tmp_path / "clean.json", {
        "session": "btbank3_t2", "subject_id": 3, "trial_id": 2, "bad_windows_s": [],
    })
    assert load_bad_window_spans(p) == []


def test_index_bad_windows_keys_by_subject_trial(tmp_path) -> None:
    _write(tmp_path / "a.json", {"subject_id": 1, "trial_id": 0,
                                 "bad_windows_s": [[1.0, 2.0]]})
    _write(tmp_path / "b.json", {"subject_id": 2, "trial_id": 1,
                                 "bad_windows_s": [[3.0, 4.0], [5.0, 6.0]]})
    idx = index_bad_windows(str(tmp_path))
    assert idx[(1, 0)] == [(1.0, 2.0)]
    assert idx[(2, 1)] == [(3.0, 4.0), (5.0, 6.0)]


def test_parse_session_name() -> None:
    assert parse_session_name("btbank6_t4") == (6, 4)
    assert parse_session_name("btbank1_t0") == (1, 0)


def test_index_band_cache_scans_sidecars(tmp_path) -> None:
    # two sessions in one band dir; each has {stem}.json + {stem}.npy + {stem}.stats.npz
    for stem, key, ch, tf in (
        ("aaa", "Wang2024Treebank:subject_id=1,trial_id=0", ["LA1", "LA2"], 5000),
        ("bbb", "Wang2024Treebank:subject_id=2,trial_id=1", ["LB1"], 4000),
    ):
        _write(tmp_path / f"{stem}.json", {
            "key": key, "ch_names": ch, "total_frames": tf, "sample_rate": 32,
        })
        np.save(str(tmp_path / f"{stem}.npy"), np.zeros((len(ch), 7, tf), np.float32))
        np.savez(str(tmp_path / f"{stem}.stats.npz"),
                 median=np.zeros((len(ch), 7, 1)), sigma=np.ones((len(ch), 7, 1)))
    idx = index_band_cache(str(tmp_path))
    keys = set(idx)
    assert "Wang2024Treebank:subject_id=1,trial_id=0" in keys
    e = idx["Wang2024Treebank:subject_id=1,trial_id=0"]
    assert e.ch_names == ("LA1", "LA2")
    assert e.total_frames == 5000
    assert e.npy_path.endswith("aaa.npy")
    assert e.stats_path.endswith("aaa.stats.npz")


def test_index_band_cache_ignores_sidecar_without_npy(tmp_path) -> None:
    # a stray .json with no matching .npy must be skipped (not half-indexed)
    _write(tmp_path / "orphan.json", {
        "key": "k", "ch_names": ["X1"], "total_frames": 1, "sample_rate": 32,
    })
    assert index_band_cache(str(tmp_path)) == {}


def test_parse_lof_report_maps_session_to_bad_channels(tmp_path) -> None:
    p = _write(tmp_path / "lof.json", {
        "threshold": 1.5, "n_neighbors": 20, "total_sessions": 2,
        "sessions": [
            {"session": "btbank1_t0", "subject_id": 1, "n_bad": 2,
             "bad_channels": ["LA3", "LB2"]},
            {"session": "btbank2_t1", "subject_id": 2, "n_bad": 0,
             "bad_channels": []},
        ],
    })
    lof = parse_lof_report(p)
    assert lof[(1, 0)] == {"LA3", "LB2"}
    assert lof[(2, 1)] == set()


def test_parse_lof_report_missing_file_is_empty() -> None:
    # LOF is optional (a run may not have fitted a report); absent → no drops.
    assert parse_lof_report(None) == {}
