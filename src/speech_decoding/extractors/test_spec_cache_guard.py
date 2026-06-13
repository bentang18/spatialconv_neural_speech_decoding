"""Positive controls for the LG4 spec-cache geometry guard (P5 mutation testing).

`MultiStftView._assert_cache_meta` / `._assert_cache_finite` had NO test
exercising them — so the guard that makes an STFT-geometry default edit fail loud
(the LG4 class: a default like `hop_length` 128->256 is invisible to the cache
key because `infra.uid()` uses `exclude_defaults`) was itself unproven. A guard
with no positive control can silently rot into a no-op.

These tests inject each cache-corruption class and assert the guard FIRES, and
confirm it does NOT false-positive on a correct sidecar or a legacy sidecar that
predates the geometry keys (the assert-if-present contract). See ledger LG4.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.extractors.view import MultiStftView

EXPECTED_F = 50  # arbitrary; the guard compares meta["f_bins"] to the passed value
CH = ["e0", "e1", "e2"]


def _view() -> MultiStftView:
    return MultiStftView(event_types="Ieeg")


def _good_meta(view: MultiStftView) -> dict:
    return {"f_bins": EXPECTED_F, "n_channels": len(CH), **view._spec_cache_geometry()}


# --- negative controls: must NOT fire on a valid / legacy sidecar ---------------
def test_cache_meta_passes_on_correct_sidecar() -> None:
    v = _view()
    v._assert_cache_meta("k", _good_meta(v), EXPECTED_F, CH)  # no raise


def test_cache_meta_passes_on_legacy_sidecar_without_geometry_keys() -> None:
    """assert-if-present: a pre-fix sidecar lacking the geometry keys must pass
    (we record+assert rather than invalidate legacy `/work` caches)."""
    v = _view()
    v._assert_cache_meta("k", {"f_bins": EXPECTED_F, "n_channels": len(CH)}, EXPECTED_F, CH)


# --- positive controls: each corruption class must fire -------------------------
def test_cache_meta_fires_on_wrong_f_bins() -> None:
    v = _view()
    meta = _good_meta(v)
    meta["f_bins"] = EXPECTED_F + 1
    with pytest.raises(ValueError, match="f_bins"):
        v._assert_cache_meta("k", meta, EXPECTED_F, CH)


def test_cache_meta_fires_on_wrong_n_channels() -> None:
    v = _view()
    meta = _good_meta(v)
    meta["n_channels"] = len(CH) + 1
    with pytest.raises(ValueError, match="n_channels"):
        v._assert_cache_meta("k", meta, EXPECTED_F, CH)


def test_cache_meta_fires_on_changed_geometry_default() -> None:
    """The LG4 core: a geometry default edit (here hop_length+1) invisible to the
    cache key must fail loud against a sidecar built before the edit."""
    v = _view()
    meta = _good_meta(v)
    meta["hop_length"] = int(v._spec_cache_geometry()["hop_length"]) + 1
    with pytest.raises(ValueError, match="geometry"):
        v._assert_cache_meta("k", meta, EXPECTED_F, CH)


def test_cache_finite_fires_on_nan() -> None:
    t = torch.ones(2, 3)
    t[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        MultiStftView._assert_cache_finite("k", t)


def test_cache_finite_fires_on_inf() -> None:
    t = torch.ones(2, 3)
    t[1, 2] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        MultiStftView._assert_cache_finite("k", t)


def test_cache_finite_passes_on_finite() -> None:
    MultiStftView._assert_cache_finite("k", torch.ones(2, 3))  # no raise
