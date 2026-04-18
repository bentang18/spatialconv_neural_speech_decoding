"""Tests for src/speech_decoding/v14/electrode_pool.py.

Covers:
- Delta behaviour when only one pial vertex is in range.
- Weighted average across equidistant neighbours (uniform crown).
- Gaussian falloff ordering (closer vertex contributes more).
- Crown bias downweights sulcal-wall vertices vs gyral crowns.
- Out-of-radius electrodes return zeros (no NaN, no crash).
- Output shape and dtype.
"""

from __future__ import annotations

import numpy as np
import pytest

from speech_decoding.v14.electrode_pool import euclidean_support


def _atlas(V: int, K: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 100.0, size=(V, K)).astype(np.float32)


def test_shape_and_dtype():
    pial = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    atlas = _atlas(2, 3)
    curv = np.zeros(2)
    elec = np.array([[0.0, 0.0, 0.0]])
    out = euclidean_support(elec, pial, atlas, curv,
                            sigma_mm=1.3, radius_mm=5.0, crown_bias=False)
    assert out.shape == (1, 3)
    assert out.dtype == np.float32
    assert not np.isnan(out).any()


def test_delta_single_vertex_in_radius():
    # One pial vertex at origin, one far away; electrode at origin.
    # Only the origin vertex should contribute. Output == atlas[0].
    pial = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    atlas = _atlas(2, 4)
    curv = np.zeros(2)
    elec = np.array([[0.0, 0.0, 0.0]])
    out = euclidean_support(elec, pial, atlas, curv,
                            sigma_mm=1.3, radius_mm=5.0, crown_bias=False)
    np.testing.assert_allclose(out[0], atlas[0], rtol=1e-5)


def test_equidistant_uniform_is_average():
    # Two equidistant vertices, no crown bias -> weighted avg equals unweighted
    # mean (both weights equal after normalisation).
    pial = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    atlas = np.array([[10.0, 0.0], [0.0, 20.0]], dtype=np.float32)
    curv = np.zeros(2)
    elec = np.array([[0.0, 0.0, 0.0]])
    out = euclidean_support(elec, pial, atlas, curv,
                            sigma_mm=1.3, radius_mm=5.0, crown_bias=False)
    np.testing.assert_allclose(out[0], [5.0, 10.0], rtol=1e-5)


def test_closer_vertex_dominates():
    # With Gaussian weights, a vertex at 0.5 mm weighs more than one at 2.0 mm.
    pial = np.array([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atlas = np.array([[100.0], [0.0]], dtype=np.float32)
    curv = np.zeros(2)
    elec = np.array([[0.0, 0.0, 0.0]])
    out = euclidean_support(elec, pial, atlas, curv,
                            sigma_mm=1.3, radius_mm=5.0, crown_bias=False)
    # Closer vertex wins -> output > 50.
    assert out[0, 0] > 50.0


def test_crown_bias_downweights_sulcal_wall():
    # Two equidistant vertices with opposite curvature: one crown (curv<0),
    # one sulcal wall (curv>0). Crown bias should pull the pooled support
    # toward the crown vertex.
    pial = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    atlas = np.array([[100.0, 0.0], [0.0, 100.0]], dtype=np.float32)
    curv = np.array([-0.2, +0.2])  # crown, sulcus
    elec = np.array([[0.0, 0.0, 0.0]])
    out_biased = euclidean_support(elec, pial, atlas, curv,
                                   sigma_mm=1.3, radius_mm=5.0, crown_bias=True)
    out_unbiased = euclidean_support(elec, pial, atlas, curv,
                                     sigma_mm=1.3, radius_mm=5.0, crown_bias=False)
    # Crown channel (first) should be larger under crown bias.
    assert out_biased[0, 0] > out_unbiased[0, 0]
    assert out_biased[0, 1] < out_unbiased[0, 1]


def test_out_of_radius_returns_zero():
    # Electrode far from any pial vertex -> all zeros, no NaN.
    pial = np.array([[0.0, 0.0, 0.0]])
    atlas = np.array([[5.0, 7.0]], dtype=np.float32)
    curv = np.zeros(1)
    elec = np.array([[100.0, 0.0, 0.0]])
    out = euclidean_support(elec, pial, atlas, curv,
                            sigma_mm=1.3, radius_mm=5.0, crown_bias=False)
    np.testing.assert_array_equal(out[0], np.zeros(2, dtype=np.float32))


def test_multiple_electrodes_independent():
    # Two electrodes at different pial vertices -> each gets the atlas row at
    # its own vertex.
    pial = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    atlas = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    curv = np.zeros(2)
    elec = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    out = euclidean_support(elec, pial, atlas, curv,
                            sigma_mm=1.3, radius_mm=5.0, crown_bias=False)
    np.testing.assert_allclose(out[0], [1.0, 0.0], rtol=1e-5)
    np.testing.assert_allclose(out[1], [0.0, 1.0], rtol=1e-5)


def test_rejects_shape_mismatch():
    pial = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atlas = _atlas(3, 2)  # wrong vert count
    curv = np.zeros(2)
    elec = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        euclidean_support(elec, pial, atlas, curv,
                          sigma_mm=1.3, radius_mm=5.0, crown_bias=False)
