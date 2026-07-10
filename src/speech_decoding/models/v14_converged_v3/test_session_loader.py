"""v14_converged_v3 Phase D1/D2 orchestrator — load_v3_sessions (TDD).

The glue composing the tested parsers (``cache_index``) + pure cores
(``build_session_setup`` / ``build_session_spec``) + the injectable parcel lookup
into the ``list[V3SessionSpec]`` the datamodule consumes. Exercised here with
synthetic 3-band spec caches + a stub ``parcel_fn`` (one parcel per shaft prefix);
the single real-BT seam (the default ``parcel_fn``) is validated on DeltaAI at F2.

Checks that matter: the 3 bands are matched to ONE session by sidecar ``key`` and
their channel order agreed; guard-1 (LOF) drops the named electrodes; guard-2 spans
are carried through in seconds; stats are sliced to survivors so the frozen
normalizer aligns to the clip rows; and the boundary faults (missing session,
disagreeing bands, wrong band-dir count) fail loud.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from speech_decoding.models.v14_converged_v3.session_loader import (
    _entry_for,
    load_v3_sessions,
)

# F per band in v3 concat order (slow 7, mid 6, hga 7).
_BAND_F = (7, 6, 7)


def _shaft_labels(shaft_sizes):
    labels = []
    for s, n in enumerate(shaft_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
    return labels


def _stub_parcel_fn(subject_id, trial_id, labels):
    """One parcel per shaft prefix — deterministic, no BT anatomy."""
    prefixes: dict[str, int] = {}
    pid = []
    for lab in labels:
        pre = lab.rstrip("0123456789")
        prefixes.setdefault(pre, len(prefixes))
        pid.append(prefixes[pre])
    return torch.tensor(pid, dtype=torch.long)


def _write_band_cache(band_dir, band_idx, sessions, n_frames):
    """Write {stem}.json + .npy + .stats.npz for each session into one band dir."""
    band_dir.mkdir(parents=True, exist_ok=True)
    F = _BAND_F[band_idx]
    for subject_id, trial_id, labels in sessions:
        stem = f"btbank{subject_id}_t{trial_id}"
        key = f"Wang2024Treebank:subject_id={subject_id},trial_id={trial_id}"
        C = len(labels)
        (band_dir / f"{stem}.json").write_text(json.dumps({
            "key": key, "ch_names": labels, "total_frames": n_frames,
            "sample_rate": 32,
        }))
        # distinct per-channel values so a wrong keep-slice would be visible
        arr = (np.arange(C, dtype=np.float32)[:, None, None]
               + np.zeros((C, F, n_frames), np.float32))
        np.save(str(band_dir / f"{stem}.npy"), arr)
        np.savez(
            str(band_dir / f"{stem}.stats.npz"),
            median=np.arange(C, dtype=np.float32)[:, None, None] + np.zeros((C, F, 1)),
            sigma=np.ones((C, F, 1), np.float32) * 2.0,
        )


def _setup_caches(tmp_path, sessions, *, n_frames=2000, bad_windows=None, lof=None):
    band_dirs = []
    for b in range(3):
        d = tmp_path / f"band{b}"
        _write_band_cache(d, b, sessions, n_frames)
        band_dirs.append(str(d))
    span_dir = tmp_path / "spans"
    span_dir.mkdir()
    for subject_id, trial_id, _ in sessions:
        (span_dir / f"btbank{subject_id}_t{trial_id}.json").write_text(json.dumps({
            "session": f"btbank{subject_id}_t{trial_id}",
            "subject_id": subject_id, "trial_id": trial_id,
            "bad_windows_s": (bad_windows or {}).get((subject_id, trial_id), []),
        }))
    lof_path = None
    if lof is not None:
        lof_path = str(tmp_path / "lof.json")
        (tmp_path / "lof.json").write_text(json.dumps({
            "sessions": [
                {"session": f"btbank{s}_t{t}", "subject_id": s,
                 "bad_channels": list(bad)}
                for (s, t), bad in lof.items()
            ]
        }))
    return band_dirs, str(span_dir), lof_path


def _mk_sessions():
    # subject 1 trial 0 (3 shafts of 8) and subject 12 trial 3 (2 shafts of 8) —
    # 1 vs 12 guards the word-boundary key match.
    return [
        (1, 0, _shaft_labels((8, 8, 8))),
        (12, 3, _shaft_labels((8, 8))),
    ]


def test_assembles_one_spec_per_session(tmp_path) -> None:
    sess = _mk_sessions()
    band_dirs, span_dir, _ = _setup_caches(tmp_path, sess)
    specs = load_v3_sessions(
        sessions=[(1, 0), (12, 3)],
        band_cache_dirs=band_dirs, span_dir=span_dir, parcel_fn=_stub_parcel_fn,
    )
    assert [s.session_key for s in specs] == [(1, 0), (12, 3)]
    s0 = specs[0]
    assert len(s0.band_paths) == 3
    assert s0.n_frames == 2000
    # no drops → keep every one of the 24 contacts, in order
    assert s0.keep_idx.tolist() == list(range(24))
    assert s0.setup.geom.n_shafts == 3


def test_word_boundary_match_distinguishes_1_from_12(tmp_path) -> None:
    sess = _mk_sessions()
    band_dirs, span_dir, _ = _setup_caches(tmp_path, sess)
    specs = load_v3_sessions(
        sessions=[(1, 0)],
        band_cache_dirs=band_dirs, span_dir=span_dir, parcel_fn=_stub_parcel_fn,
    )
    # subject 1 has 24 contacts (3×8); subject 12 has 16 — a substring match on
    # "subject_id=1" would have grabbed the wrong entry or raised.
    assert specs[0].keep_idx.numel() == 24


def test_lof_drops_named_electrodes(tmp_path) -> None:
    sess = _mk_sessions()
    band_dirs, span_dir, lof_path = _setup_caches(
        tmp_path, sess, lof={(1, 0): {"LA2", "LB1"}},
    )
    specs = load_v3_sessions(
        sessions=[(1, 0)],
        band_cache_dirs=band_dirs, span_dir=span_dir, parcel_fn=_stub_parcel_fn,
        lof_report_path=lof_path,
    )
    s0 = specs[0]
    assert s0.keep_idx.numel() == 22  # 24 − 2
    # LA2 is full-order row 1, LB1 is row 8 → dropped from the memmap read plan
    assert 1 not in s0.keep_idx.tolist()
    assert 8 not in s0.keep_idx.tolist()


def test_stats_sliced_to_survivors(tmp_path) -> None:
    sess = _mk_sessions()
    band_dirs, span_dir, lof_path = _setup_caches(
        tmp_path, sess, lof={(1, 0): {"LA1"}},  # drop full-order row 0
    )
    specs = load_v3_sessions(
        sessions=[(1, 0)],
        band_cache_dirs=band_dirs, span_dir=span_dir, parcel_fn=_stub_parcel_fn,
        lof_report_path=lof_path,
    )
    s0 = specs[0]
    med, sig = s0.band_stats[0]
    # full-order median was arange(C); survivors start at row 1 → median[0]==1
    assert med.shape[0] == 23
    assert float(med[0, 0, 0]) == 1.0
    assert sig.shape[0] == 23


def test_guard2_spans_carried_in_seconds(tmp_path) -> None:
    sess = _mk_sessions()
    band_dirs, span_dir, _ = _setup_caches(
        tmp_path, sess, bad_windows={(1, 0): [[10.0, 11.0], [20.0, 21.0]]},
    )
    specs = load_v3_sessions(
        sessions=[(1, 0)],
        band_cache_dirs=band_dirs, span_dir=span_dir, parcel_fn=_stub_parcel_fn,
    )
    assert specs[0].bad_spans_s == [(10.0, 11.0), (20.0, 21.0)]


def test_missing_session_fails_loud(tmp_path) -> None:
    sess = _mk_sessions()
    band_dirs, span_dir, _ = _setup_caches(tmp_path, sess)
    with pytest.raises(ValueError, match="found 0"):
        load_v3_sessions(
            sessions=[(9, 9)],
            band_cache_dirs=band_dirs, span_dir=span_dir, parcel_fn=_stub_parcel_fn,
        )


def test_disagreeing_band_channel_order_fails_loud(tmp_path) -> None:
    sess = _mk_sessions()
    band_dirs, span_dir, _ = _setup_caches(tmp_path, sess)
    # corrupt band2's ch order for session (1,0)
    p = tmp_path / "band2" / "btbank1_t0.json"
    meta = json.loads(p.read_text())
    meta["ch_names"] = meta["ch_names"][::-1]
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="disagree on channel order"):
        load_v3_sessions(
            sessions=[(1, 0)],
            band_cache_dirs=band_dirs, span_dir=span_dir, parcel_fn=_stub_parcel_fn,
        )


def test_wrong_band_dir_count_fails_loud(tmp_path) -> None:
    sess = _mk_sessions()
    band_dirs, span_dir, _ = _setup_caches(tmp_path, sess)
    with pytest.raises(ValueError, match="3 band cache dirs"):
        load_v3_sessions(
            sessions=[(1, 0)],
            band_cache_dirs=band_dirs[:2], span_dir=span_dir,
            parcel_fn=_stub_parcel_fn,
        )


def test_entry_for_ambiguous_raises(tmp_path) -> None:
    # two entries both matching the same (S,T) → refuse rather than pick one
    from speech_decoding.models.v14_converged_v3.cache_index import BandCacheEntry
    idx = {
        "subject_id=1,trial_id=0,a": BandCacheEntry("a.npy", "a.npz", ("X",), 1, 32),
        "subject_id=1,trial_id=0,b": BandCacheEntry("b.npy", "b.npz", ("X",), 1, 32),
    }
    with pytest.raises(ValueError, match="found 2"):
        _entry_for(idx, 1, 0)
