"""LG14c + LG14g — fail-loud guards on the per-clip spec-cache READ path.

``MultiStftView._assert_cache_finite`` only runs at cache BUILD. A cache HIT
(``_cached_clip``) memmap-slices the ``.npy`` and validates only the ``.json``
sidecar geometry — never the payload values (LG14c) or that the recording is at
least one clip long (LG14g). Both are the S9 class: plausible, non-crashing,
silently-wrong model input. ``_assert_cache_read_sane`` closes both.

These tests exercise the static guard directly (no cache machinery): a healthy
full-length finite slice must PASS (false-positive proof), and each corruption
class must FIRE.
"""

from __future__ import annotations

import numpy as np
import pytest

from speech_decoding.extractors.view import MultiStftView


def _good_slice(c: int = 8, f: int = 50, t: int = 16) -> np.ndarray:
    # A plausible |STFT| clip: non-negative, finite, full clip length.
    return np.abs(np.linspace(0.0, 1.0, c * f * t, dtype=np.float32)).reshape(c, f, t)


# --- false-positive proof: a healthy full-length finite slice must PASS ----------
def test_healthy_slice_passes() -> None:
    sliced = _good_slice(t=16)
    MultiStftView._assert_cache_read_sane(sliced, n_frames=16, total_frames=10_000, key="sub1_t0")


def test_exact_length_slice_passes() -> None:
    # n_frames == slice length is the normal in-bounds case (g0 clamp gives full).
    sliced = _good_slice(t=16)
    MultiStftView._assert_cache_read_sane(sliced, n_frames=16, total_frames=16, key="sub1_t0")


# --- LG14g: whole recording shorter than one clip must FIRE ----------------------
def test_short_recording_fires() -> None:
    sliced = _good_slice(t=12)  # only 12 frames available for a 16-frame window
    with pytest.raises(ValueError, match="shorter than a single clip"):
        MultiStftView._assert_cache_read_sane(sliced, n_frames=16, total_frames=12, key="sub9_t1")


def test_short_recording_message_names_lg14g() -> None:
    sliced = _good_slice(t=1)
    with pytest.raises(ValueError, match="LG14g"):
        MultiStftView._assert_cache_read_sane(sliced, n_frames=16, total_frames=1, key="sub9_t1")


# --- LG14c: non-finite payload on the read path must FIRE ------------------------
def test_nan_payload_fires() -> None:
    sliced = _good_slice(t=16)
    sliced[3, 10, 5] = np.nan  # a single byte-flipped frame
    with pytest.raises(ValueError, match="non-finite"):
        MultiStftView._assert_cache_read_sane(sliced, n_frames=16, total_frames=10_000, key="sub1_t0")


def test_inf_payload_fires() -> None:
    sliced = _good_slice(t=16)
    sliced[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="LG14c"):
        MultiStftView._assert_cache_read_sane(sliced, n_frames=16, total_frames=10_000, key="sub1_t0")


def test_length_checked_before_finiteness() -> None:
    # A slice that is BOTH short and non-finite should report the length problem
    # (LG14g) first — length is the more fundamental corruption signal.
    sliced = _good_slice(t=8)
    sliced[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="LG14g"):
        MultiStftView._assert_cache_read_sane(sliced, n_frames=16, total_frames=8, key="sub9_t1")
