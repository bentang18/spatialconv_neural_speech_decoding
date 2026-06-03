"""Test the Stage-0 runner's per-(C,F) robust-z helper (`_mad_normalize_cf`).

per-(C,F) pools the median/MAD over trials AND time, so it removes each band's
level/scale but PRESERVES the within-window temporal envelope (the Nv14 /
iMINDBench recipe). Contrast `_mad_normalize` (per-(C,F,T)), which gives every
(c,f,t) its own stat and whitens the time course.

The runner is a script under scripts/ with heavy deps (neuralset, neuroprobe,
torch). Load it by file path and skip cleanly if those are absent, so
`pytest src/` stays green on a minimal env while still covering the math where
the deps exist (local + DCC).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_RUNNER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "neuroprobe"
    / "run_stage0_linear_baseline.py"
)

_MOD = None


def _runner():
    global _MOD
    if _MOD is not None:
        return _MOD
    if not _RUNNER.exists():
        pytest.skip(f"runner not found at {_RUNNER}")
    spec = importlib.util.spec_from_file_location("_run_stage0_under_test", _RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # heavy optional deps (neuralset/neuroprobe/torch) absent
        pytest.skip(f"runner not importable in this env: {e}")
    _MOD = mod
    return mod


def _toy() -> np.ndarray:
    # 2 trials, C=1, F=1, T=3, flattened row-major over (C,F,T). A temporal peak
    # at t=1 in both trials; nonzero spread elsewhere so the MAD is well-defined.
    return np.array(
        [
            [2.0, 9.0, 1.0],  # trial 0
            [1.0, 7.0, 2.0],  # trial 1
        ]
    )


def test_mad_normalize_cf_preserves_temporal_envelope():
    m = _runner()
    out = m._mad_normalize_cf(_toy(), (1, 1, 3))
    # One median/MAD pooled over BOTH trials and the 3 time bins → the t=1 peak
    # stays well above the t=0 / t=2 baseline (envelope preserved).
    assert out[0, 1] > out[0, 0] + 1.0
    assert out[0, 1] > out[0, 2] + 1.0
    assert out[0, 1] == out[0].max()


def test_mad_normalize_cf_keeps_more_envelope_than_per_cft():
    m = _runner()
    X = _toy()
    cf = m._mad_normalize_cf(X, (1, 1, 3))  # per-(C,F): one stat per band
    cft = m._mad_normalize(X)  # per-(C,F,T): each (c,f,t) its own stat
    # Peak-vs-baseline contrast within trial 0. per-(C,F,T) whitens each column
    # independently and collapses the peak; per-(C,F) preserves it.
    cf_contrast = cf[0, 1] - 0.5 * (cf[0, 0] + cf[0, 2])
    cft_contrast = cft[0, 1] - 0.5 * (cft[0, 0] + cft[0, 2])
    assert cf_contrast > cft_contrast
