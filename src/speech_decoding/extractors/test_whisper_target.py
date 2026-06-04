"""Tests for WhisperTargetExtractor (WS-H / T20).

The extractor slices a cached whole-movie Whisper feature stream at a clip's
MOVIE-clock onset and returns the ``(250, 1280)`` teacher target the P3
distillation loss consumes (collated to ``(B, 250, 1280)``).

The load-bearing test is ``test_keys_off_movie_onset_not_neural_start`` (FLAG 9):
the join MUST use the movie-audio onset, never the neural-clock window start —
the two clocks diverge hundreds of seconds within a single BT trial. The rest
cover the slice math, the tail clamp (exactly 250 frames always), per-movie
mmap memoization, the layer-merge path keying, and every loud-failure guard
(missing cache / wrong rate / non-zero t0 / wrong d / movie shorter than clip).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from speech_decoding.extractors import whisper_target as wt
from speech_decoding.extractors.whisper_target import WhisperTargetExtractor

_D = 1280
_MOVIE = "the-grand-budapest-hotel"


class _StubEvent:
    """Minimal stand-in for a NeuralSet ``Word`` event.

    ``get_static`` reads only ``event._get_field_or_extra(field)``; this exposes
    exactly that surface (subject_id/trial_id/movie_onset_s in ``extra``)."""

    def __init__(self, **fields: object) -> None:
        self._fields = fields

    def _get_field_or_extra(self, field: str) -> object:
        return self._fields[field]


def _write_cache(
    cache_dir: Path,
    *,
    n_frames: int,
    movie: str = _MOVIE,
    model: str = "openai/whisper-large-v3",
    layer_merge: str = "mean_all",
    rate_hz: int = 50,
    t0_movie_s: float = 0.0,
    d_model: int = _D,
) -> torch.Tensor:
    """Write a whole-movie cache .pt matching ``write_movie_cache``'s schema and
    return the dense fp16 features so a test can assert the exact slice."""
    merge_slug = "mean_all" if layer_merge == "mean_all" else f"L{int(layer_merge)}"
    path = cache_dir / model.replace("/", "_") / merge_slug / f"{movie}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic ramp so a misaligned slice is visible, not just a shape pass.
    features = (
        torch.arange(n_frames, dtype=torch.float32)[:, None].expand(n_frames, d_model)
        + torch.arange(d_model, dtype=torch.float32)[None, :]
    ).to(torch.float16)
    torch.save(
        {
            "features": features,
            "rate_hz": rate_hz,
            "t0_movie_s": t0_movie_s,
            "duration_s": n_frames / float(rate_hz),
            "chunk_s": 30.0,
            "movie": movie,
            "model": model,
            "layer_merge": layer_merge,
        },
        path,
    )
    return features


def _ext(cache_dir: Path, monkeypatch, **kwargs) -> WhisperTargetExtractor:
    # Avoid the lazy neuroprobe.config import (raises off-DCC); every clip in a
    # test maps to the one movie we cached.
    monkeypatch.setattr(wt, "_resolve_movie", lambda subject_id, trial_id: _MOVIE)
    return WhisperTargetExtractor(cache_dir=str(cache_dir), **kwargs)


def _event(movie_onset_s: float, **extra) -> _StubEvent:
    base = {"subject_id": 3, "trial_id": 1, "movie_onset_s": movie_onset_s}
    base.update(extra)
    return _StubEvent(**base)


def test_n_frames_is_250() -> None:
    ext = WhisperTargetExtractor(cache_dir="/unused")
    assert ext.n_frames == 250


def test_happy_path_slices_at_movie_onset(tmp_path: Path, monkeypatch) -> None:
    dense = _write_cache(tmp_path, n_frames=2000)  # 40 s movie
    ext = _ext(tmp_path, monkeypatch)

    out = ext.get_static(_event(10.0))  # frame0 = round(10 * 50) = 500

    assert isinstance(out, torch.Tensor)
    assert out.shape == (250, _D)
    assert out.dtype == torch.float32
    assert out.is_contiguous()
    assert out.device.type == "cpu"
    torch.testing.assert_close(out, dense[500:750].to(torch.float32))
    assert ext._n_clamped == 0


def test_onset_rounds_to_nearest_frame(tmp_path: Path, monkeypatch) -> None:
    dense = _write_cache(tmp_path, n_frames=2000)
    ext = _ext(tmp_path, monkeypatch)
    # 10.007 s -> 500.35 -> 500 ; 10.013 s -> 500.65 -> 501
    torch.testing.assert_close(ext.get_static(_event(10.007)), dense[500:750].float())
    torch.testing.assert_close(ext.get_static(_event(10.013)), dense[501:751].float())


def test_keys_off_movie_onset_not_neural_start(tmp_path: Path, monkeypatch) -> None:
    """FLAG 9 regression: the slice follows the MOVIE-clock ``movie_onset_s``
    even when a (red-herring) neural-clock ``start`` is present and divergent.

    A real BT trial has movie_onset≈235 s while the neural est_idx/2048≈t≈0 at
    the first word; keying off ``start`` would slice the wrong 5 s of the movie."""
    dense = _write_cache(tmp_path, n_frames=60_000)  # 20 min movie
    ext = _ext(tmp_path, monkeypatch)

    # movie_onset_s = 235.0 (frame 11750); neural start = 0.5 (the trap value).
    evt = _event(235.0, start=0.5)
    out = ext.get_static(evt)
    torch.testing.assert_close(out, dense[11750:12000].float())
    # Sanity: it is NOT the neural-clock slice (would be dense[25:275]).
    assert not torch.allclose(out, dense[25:275].float())


def test_tail_clamp_keeps_exactly_250_frames(tmp_path: Path, monkeypatch) -> None:
    dense = _write_cache(tmp_path, n_frames=300)
    ext = _ext(tmp_path, monkeypatch)
    # onset 5.5 s -> frame0 = 275 > total-n = 50 -> clamp back to 50.
    out = ext.get_static(_event(5.5))
    assert out.shape == (250, _D)
    torch.testing.assert_close(out, dense[50:300].float())
    assert ext._n_clamped == 1


def test_last_valid_window_no_clamp(tmp_path: Path, monkeypatch) -> None:
    # frame0 == total - n exactly: the final window that fits without clamping.
    dense = _write_cache(tmp_path, n_frames=300)
    ext = _ext(tmp_path, monkeypatch)
    out = ext.get_static(_event(1.0))  # frame0 = 50 == total(300)-n(250)
    torch.testing.assert_close(out, dense[50:300].float())
    assert ext._n_clamped == 0


def test_clamp_floor_at_zero(tmp_path: Path, monkeypatch) -> None:
    dense = _write_cache(tmp_path, n_frames=2000)
    ext = _ext(tmp_path, monkeypatch)
    out = ext.get_static(_event(0.0))  # frame0 = 0, no clamp
    torch.testing.assert_close(out, dense[0:250].float())
    assert ext._n_clamped == 0


def test_memoizes_dense_per_movie(tmp_path: Path, monkeypatch) -> None:
    _write_cache(tmp_path, n_frames=2000)
    ext = _ext(tmp_path, monkeypatch)
    ext.get_static(_event(1.0))
    first = ext._dense[_MOVIE]
    ext.get_static(_event(2.0))
    # Same tensor object reused — the cache was loaded exactly once.
    assert ext._dense[_MOVIE] is first
    assert len(ext._dense) == 1


def test_layer_merge_keys_cache_path(tmp_path: Path, monkeypatch) -> None:
    dense = _write_cache(tmp_path, n_frames=2000, layer_merge="8")
    ext = _ext(tmp_path, monkeypatch, layer_merge="8")
    out = ext.get_static(_event(4.0))  # frame0 = 200
    torch.testing.assert_close(out, dense[200:450].float())


def test_clip_s_changes_window_length(tmp_path: Path, monkeypatch) -> None:
    dense = _write_cache(tmp_path, n_frames=2000)
    ext = _ext(tmp_path, monkeypatch, clip_s=1.0)  # 1 s -> 50 frames
    assert ext.n_frames == 50
    out = ext.get_static(_event(10.0))
    assert out.shape == (50, _D)
    torch.testing.assert_close(out, dense[500:550].float())


def test_missing_cache_file_raises(tmp_path: Path, monkeypatch) -> None:
    ext = _ext(tmp_path, monkeypatch)  # nothing written
    with pytest.raises(FileNotFoundError, match="teacher cache missing"):
        ext.get_static(_event(1.0))


def test_rate_mismatch_raises(tmp_path: Path, monkeypatch) -> None:
    _write_cache(tmp_path, n_frames=2000, rate_hz=100)
    ext = _ext(tmp_path, monkeypatch)  # extractor default rate_hz=50
    with pytest.raises(ValueError, match="rate"):
        ext.get_static(_event(1.0))


def test_nonzero_t0_raises(tmp_path: Path, monkeypatch) -> None:
    _write_cache(tmp_path, n_frames=2000, t0_movie_s=12.0)
    ext = _ext(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="t0_movie_s=0"):
        ext.get_static(_event(1.0))


def test_d_model_mismatch_raises(tmp_path: Path, monkeypatch) -> None:
    _write_cache(tmp_path, n_frames=2000, d_model=384)  # whisper-tiny width
    ext = _ext(tmp_path, monkeypatch)  # expects 1280
    with pytest.raises(ValueError, match="expected \\(T, 1280\\)"):
        ext.get_static(_event(1.0))


def test_movie_shorter_than_clip_raises(tmp_path: Path, monkeypatch) -> None:
    _write_cache(tmp_path, n_frames=100)  # 2 s movie < 5 s clip
    ext = _ext(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="< clip window"):
        ext.get_static(_event(0.0))
