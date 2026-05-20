"""Tests for the run_stage0_linear_baseline STFT-feature cache key.

The cache memoizes STFT features by a content hash of the sliced raw window.
Correctness rests entirely on the key: identical windows must collide (so the
hit returns the right features) and any feature-affecting difference must not
(so a miss recomputes). These tests pin both directions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    "run_stage0_linear_baseline",
    Path(__file__).with_name("run_stage0_linear_baseline.py"),
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_window_cache_key = _mod._window_cache_key


def _win(seed: int, shape: tuple[int, int] = (8, 64)) -> torch.Tensor:
    return torch.rand(shape, generator=torch.Generator().manual_seed(seed))


def test_identical_window_same_key():
    labels = ["a", "b", "c"]
    assert _window_cache_key(_win(0), labels, "v") == _window_cache_key(_win(0), labels, "v")


def test_non_contiguous_slice_matches_contiguous():
    # Real call path passes a non-contiguous tensor view; the key must equal
    # the one from the same content materialized contiguously.
    full = _win(0, (8, 128))
    view = full[:, 16:80]
    assert _window_cache_key(view, ["a"], "v") == _window_cache_key(view.contiguous(), ["a"], "v")


def test_different_content_different_key():
    labels = ["a", "b"]
    assert _window_cache_key(_win(0), labels, "v") != _window_cache_key(_win(1), labels, "v")


def test_electrode_order_different_key():
    w = _win(0)
    assert _window_cache_key(w, ["a", "b"], "v") != _window_cache_key(w, ["b", "a"], "v")


def test_view_sig_different_key():
    w = _win(0)
    assert _window_cache_key(w, ["a"], "v1") != _window_cache_key(w, ["a"], "v2")


def test_shape_different_key():
    labels = ["a"]
    assert _window_cache_key(_win(0, (8, 64)), labels, "v") != _window_cache_key(_win(0, (8, 32)), labels, "v")


def test_key_is_16_byte_digest():
    k = _window_cache_key(_win(0), ["a"], "v")
    assert isinstance(k, bytes) and len(k) == 16
