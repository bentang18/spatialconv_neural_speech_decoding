"""Tests for the ``spec_only`` lightweight-redeploy path (Option B / DeltaAI).

``spec_only=True`` makes ``prepare`` build the per-session index / positional
channel map / robust-z stats from the on-disk spec sidecars ALONE — no extractor
cache, no raw h5 on the target — and skip ``super().prepare`` (the only
extractor/raw reader at run start). The per-clip read path (``_cached_clip``) is
untouched, so model inputs must be byte-identical to the extractor-resident path.

Strategy: build a real spec cache on disk with a normal view (reusing the
``_cache_view_harness`` from ``test_view``), then drive a SECOND ``spec_only`` view
against the same cache dir and assert (1) byte-identical clips + stats + channels,
(2) zero ``_get_data`` reads, (3) fail-loud on a missing session, (4) construction
rejects LOF / no-cache-dir. The shape-probe ``__call__`` is stubbed to route
through the cache read (fire-after-arm), mirroring production where the probe sets
``_effective_frequency`` / ``_missing_default`` with no extractor read.

Test 5 (golden parity) is the DCC voltage-tier golden + the Phase-E 100-step
parity gate — it needs real BT data, so it is not a laptop unit test.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from speech_decoding.extractors.test_view import (
    _cache_view_harness,
    _make_multi_stft_view,
)
from speech_decoding.extractors.view import MultiStftView


def _build_cache(tmp_path, monkeypatch, rec, ch, *, robust_z, key="s"):
    """Build a real on-disk spec cache for one synthetic session with a normal
    (extractor-resident) view. Returns the building view."""
    view = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=robust_z, c_max=None,
    )
    event, _ = _cache_view_harness(view, rec, ch, monkeypatch, key=key)
    view.prepare([event])
    assert view._spec_ready is True
    return view


def _wire_spec_only(monkeypatch, *, get_data):
    """Wire class-level seams for a spec_only view: ``_event_types_helper`` (identity
    extract), ``_get_data`` (caller-supplied — counter or raise), and a probe
    ``__call__`` stub that fires-after-arm through the cache read path. Returns
    (event factory, probe-call counter)."""
    class _E:
        start = 0.0

        def __init__(self, k="s"):
            self._k = k

        def _splittable_event_uid(self):
            return self._k

    monkeypatch.setattr(
        MultiStftView, "_get_data",
        property(lambda self: types.MethodType(get_data, self)), raising=False,
    )
    monkeypatch.setattr(
        MultiStftView, "_event_types_helper",
        property(lambda self: types.SimpleNamespace(extract=lambda obj: obj)),
        raising=False,
    )
    probe = {"n": 0}

    def _fake_call(self, events, start, duration, trigger=None):
        # Production fires the base __call__ here; with fake events we stub it to
        # the same observable effect: fire-after-arm and route through the cache
        # read (no _get_data). Asserts the arm-before-probe ordering.
        probe["n"] += 1
        assert self._spec_ready is True
        ev = events[0] if isinstance(events, (list, tuple)) else events
        return self._get_timed_array(ev, start, duration)

    monkeypatch.setattr(MultiStftView, "__call__", _fake_call, raising=False)
    return _E, probe


def test_spec_only_clips_and_stats_match_extractor_path(tmp_path, monkeypatch) -> None:
    """Parity: a spec_only view over the same cache yields byte-identical channels,
    robust-z stats, and per-clip tensors as the building (extractor-resident) view."""
    ch = ["e0", "e1", "e2"]
    rng = np.random.default_rng(7)
    n = 2048 * 30
    rec = (rng.standard_normal((3, n)) * np.array([1.0, 4.0, 0.5])[:, None]).astype(
        np.float32
    )
    view1 = _build_cache(tmp_path, monkeypatch, rec, ch, robust_z=True)

    def _raise_get_data(self, evs):
        raise AssertionError("spec_only must not read the extractor cache")
        yield  # pragma: no cover

    make_event, probe = _wire_spec_only(monkeypatch, get_data=_raise_get_data)
    view2 = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=True, c_max=None,
        spec_only=True,
    )
    event2 = make_event("s")
    view2.prepare([event2])

    assert view2._spec_ready is True
    assert view2._stats_ready is True
    assert probe["n"] == 1  # shape-probe fired once, after arm
    assert view2._channels == view1._channels
    np.testing.assert_array_equal(
        view1._session_stats["s"].median.numpy(),
        view2._session_stats["s"].median.numpy(),
    )
    np.testing.assert_array_equal(
        view1._session_stats["s"].sigma.numpy(),
        view2._session_stats["s"].sigma.numpy(),
    )
    event1 = make_event("s")  # uid "s" + start 0.0 → same cache entry as view1 built
    for start, dur in [(0.0, 1.0), (5.0, 1.0), (10.3, 2.0)]:
        out1 = view1._get_timed_array(event1, start=start, duration=dur)
        out2 = view2._get_timed_array(event2, start=start, duration=dur)
        np.testing.assert_array_equal(np.asarray(out1.data), np.asarray(out2.data))


def test_spec_only_never_reads_extractor(tmp_path, monkeypatch) -> None:
    """Isolation: with ``_get_data`` rigged to raise and no raw h5 present, a
    spec_only prepare completes — proving zero extractor/raw read at run start."""
    ch = ["e0", "e1"]
    rng = np.random.default_rng(3)
    rec = rng.standard_normal((2, 2048 * 20)).astype(np.float32)
    _build_cache(tmp_path, monkeypatch, rec, ch, robust_z=True)

    def _raise_get_data(self, evs):
        raise AssertionError("extractor read on the spec_only path")
        yield  # pragma: no cover

    make_event, _ = _wire_spec_only(monkeypatch, get_data=_raise_get_data)
    view = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=True, c_max=None,
        spec_only=True,
    )
    view.prepare([make_event("s")])  # must not raise
    assert view._spec_ready is True


def test_spec_only_missing_session_fails_loud(tmp_path, monkeypatch) -> None:
    """Completeness guard: a session with no cached spec makes prepare raise and
    name the missing key — never a silent mid-run KeyError from _cached_clip."""
    ch = ["e0", "e1"]
    rng = np.random.default_rng(11)
    rec = rng.standard_normal((2, 2048 * 20)).astype(np.float32)
    _build_cache(tmp_path, monkeypatch, rec, ch, robust_z=True, key="s")

    def _raise_get_data(self, evs):
        raise AssertionError("extractor read on the spec_only path")
        yield  # pragma: no cover

    make_event, _ = _wire_spec_only(monkeypatch, get_data=_raise_get_data)
    view = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=True, c_max=None,
        spec_only=True,
    )
    with pytest.raises(ValueError, match="ghost"):
        view.prepare([make_event("s"), make_event("ghost")])


def test_spec_only_missing_stats_sidecar_fails_loud(tmp_path, monkeypatch) -> None:
    """A robust-z spec_only deploy whose .stats.npz sidecar is absent cannot refit
    (no frames/extractor) → prepare fails loud. Build a robust-z cache (so the
    namespace matches and .npy/.json exist), then delete the stats sidecar."""
    ch = ["e0", "e1"]
    rng = np.random.default_rng(5)
    rec = rng.standard_normal((2, 2048 * 20)).astype(np.float32)
    _build_cache(tmp_path, monkeypatch, rec, ch, robust_z=True, key="s")
    stats = list(tmp_path.rglob("*.stats.npz"))
    assert len(stats) == 1
    stats[0].unlink()  # remove the sidecar a spec_only deploy depends on

    def _raise_get_data(self, evs):
        raise AssertionError("extractor read on the spec_only path")
        yield  # pragma: no cover

    make_event, _ = _wire_spec_only(monkeypatch, get_data=_raise_get_data)
    view = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=True, c_max=None,
        spec_only=True,
    )
    with pytest.raises(ValueError, match="stats sidecar"):
        view.prepare([make_event("s")])


def test_spec_only_requires_cache_dir() -> None:
    """Construction rejects spec_only with no spec_cache_dir (nothing to read)."""
    with pytest.raises(ValueError, match="requires spec_cache_dir"):
        _make_multi_stft_view(spec_only=True, c_max=None)


def test_spec_only_rejects_lof_at_construction() -> None:
    """Construction rejects spec_only + LOF: the LOF fit reads a car=None extractor
    sibling a spec_only deploy never stages."""
    with pytest.raises(ValueError, match="lof_bad_channels"):
        _make_multi_stft_view(
            spec_cache_dir="/tmp/spec_only_unit", spec_only=True,
            lof_bad_channels=True, c_max=None,
        )
