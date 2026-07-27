"""Pooling and basis code, pinned on hand-computable cases.

The weighted pool is the one place a plausible-looking wrong number is easy to produce: the
cache already holds a parcel MEAN, so an unweighted pool over parcels silently upweights a
1-electrode parcel to match a 40-electrode one. These fix the arithmetic.
"""
from __future__ import annotations

import numpy as np

from scripts.neuroprobe.viz_common import (
    Session, center_per_session, lobe_mean, lobe_of, pca_basis, session_matrix,
    shared_lobes, to_rgb,
)
from speech_decoding.studies.braintreebank.anatomy import V14_DKT_PARCEL_LABELS


def _sess(subject_id=1, trial_id=0, parcels=(4, 6, 7), counts=(1, 39, 10),
          lobes=("temporal", "temporal", "frontal"), vals=(1.0, 2.0, 5.0), t=3, c=2):
    cond = {("enc12", "onset", 1, "all"):
            np.stack([np.full((t, c), v) for v in vals]).astype(np.float32)}
    return Session(subject_id, trial_id, np.asarray(parcels), np.asarray(counts), list(lobes),
                   cond, {"enc12": np.zeros(c)}, {"enc12": np.ones(c)},
                   {"enc12": (len(parcels), t, c)})


def test_lobe_mean_is_electrode_weighted_not_a_flat_parcel_mean() -> None:
    s = _sess()
    got = lobe_mean(s, "enc12", "onset", 1, "all", "temporal", standardize=False)
    # parcels 4 and 6 are temporal with 1 and 39 electrodes, values 1.0 and 2.0
    expected = (1 * 1.0 + 39 * 2.0) / 40
    np.testing.assert_allclose(got, expected)
    assert not np.allclose(got, 1.5), "flat parcel mean would give 1.5 — that is the bug"


def test_lobe_mean_standardizes_with_the_session_channel_moments() -> None:
    s = _sess()
    s.chan_mu["enc12"] = np.array([1.0, 0.0])
    s.chan_sd["enc12"] = np.array([2.0, 1.0])
    raw = lobe_mean(s, "enc12", "onset", 1, "all", "frontal", standardize=False)
    std = lobe_mean(s, "enc12", "onset", 1, "all", "frontal", standardize=True)
    np.testing.assert_allclose(std, (raw - np.array([1.0, 0.0])) / np.array([2.0, 1.0]))


def test_lobe_mean_returns_none_for_a_lobe_the_session_lacks() -> None:
    s = _sess()
    assert lobe_mean(s, "enc12", "onset", 1, "all", "occipital", standardize=False) is None
    assert lobe_mean(s, "enc12", "speech", 1, "all", "temporal", standardize=False) is None


def test_session_matrix_stacks_lobes_and_refuses_a_partial_stack() -> None:
    s = _sess()
    m = session_matrix(s, "enc12", "onset", 1, "all", ["temporal", "frontal"],
                       standardize=False)
    assert m.shape == (2, 3, 2)
    np.testing.assert_allclose(m[1], 5.0)
    assert session_matrix(s, "enc12", "onset", 1, "all", ["temporal", "occipital"],
                          standardize=False) is None


def test_shared_lobes_is_over_subjects_not_sessions() -> None:
    # S1 has two trials, both temporal-only; S2 has temporal + frontal.
    a = _sess(1, 0, (4, 6), (5, 5), ("temporal", "temporal"), (1.0, 1.0))
    b = _sess(1, 1, (4, 6), (5, 5), ("temporal", "temporal"), (1.0, 1.0))
    c = _sess(2, 0, (4, 7), (5, 5), ("temporal", "frontal"), (1.0, 1.0))
    assert shared_lobes([a, b, c]) == ["temporal"]
    # S1's two sessions must not vote twice and turn frontal into a shared lobe
    assert "frontal" not in shared_lobes([a, b, c])


def test_shared_lobes_honours_the_electrode_floor() -> None:
    a = _sess(1, 0, (4, 7), (5, 1), ("temporal", "frontal"), (1.0, 1.0))
    b = _sess(2, 0, (4, 7), (5, 9), ("temporal", "frontal"), (1.0, 1.0))
    assert shared_lobes([a, b], min_elec=2) == ["temporal"]   # S1's frontal has 1 electrode
    assert shared_lobes([a, b], min_elec=1) == ["frontal", "temporal"]


def test_shared_lobes_never_returns_unknown() -> None:
    a = _sess(1, 0, (4, 74), (5, 5), ("temporal", "unknown"), (1.0, 1.0))
    b = _sess(2, 0, (4, 74), (5, 5), ("temporal", "unknown"), (1.0, 1.0))
    assert shared_lobes([a, b]) == ["temporal"]


def test_center_per_session_removes_the_grand_mean_but_keeps_the_pattern() -> None:
    m = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    out = center_per_session([m, m + 100.0])
    np.testing.assert_allclose(out[0], out[1])       # a pure offset difference is gone
    np.testing.assert_allclose(out[0].mean(axis=(0, 1)), 0, atol=1e-12)
    assert not np.allclose(out[0], 0), "centering must not flatten the pattern itself"


def test_pca_basis_finds_the_direction_the_data_actually_varies_along() -> None:
    rng = np.random.default_rng(0)
    t = rng.normal(size=(200, 1))
    stack = t @ np.array([[3.0, 0.0, 0.0]]) + 0.001 * rng.normal(size=(200, 3))
    comps, mu, evr = pca_basis(stack, k=1)
    assert abs(abs(comps[0, 0]) - 1.0) < 1e-2
    assert evr[0] > 0.99
    np.testing.assert_allclose(mu, stack.mean(axis=0))


def test_to_rgb_is_a_shared_stretch_across_the_whole_input() -> None:
    proj = np.stack([np.linspace(0, 1, 100), np.linspace(-5, 5, 100)], axis=-1)
    rgb = to_rgb(proj, lo=0, hi=100)
    assert rgb.min() >= 0 and rgb.max() <= 1
    np.testing.assert_allclose(rgb[0], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(rgb[-1], [1.0, 1.0], atol=1e-9)


def test_lobe_of_indexes_by_raw_parcel_id_including_unknown() -> None:
    ids = np.array([len(V14_DKT_PARCEL_LABELS)])           # the reserved unknown id
    assert lobe_of(ids, pool_hemi=True) == ["unknown"]
    assert lobe_of(ids, pool_hemi=False) == ["unknown"]


def test_lobe_of_pool_hemi_drops_only_the_side() -> None:
    idx = V14_DKT_PARCEL_LABELS.index("ctx-lh-superiortemporal")
    assert lobe_of(np.array([idx]), pool_hemi=False) == ["lh-temporal"]
    assert lobe_of(np.array([idx]), pool_hemi=True) == ["temporal"]
