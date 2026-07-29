"""The split-half statistic's defining property, pinned.

``viz_anatomy``'s whole argument rests on ``d2 = <v_h0, v_h1> / C`` being unbiased for the
squared true contrast, because that is what gives the colour axis a TRUE zero and therefore
what lets "this parcel is blue" mean "no effect here" rather than "fewer trials here". If
that property silently broke, every map would tilt positive and a coverage gradient would
read as anatomy -- the exact failure the figure exists to rule out. So it is tested directly
against synthetic data with a known answer, not just asserted in a docstring.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.neuroprobe.viz_anatomy import (
    OCCIPITAL,
    SPEECH_NET,
    TARGET,
    dcv_rel,
    dkt_tables,
)


def _halves(rng, true_v, n_trials, noise):
    """Two independent half-averages of a contrast whose true value is ``true_v``."""
    T, C = true_v.shape
    out = {}
    for h in ("h0", "h1"):
        out[h] = true_v + rng.normal(0.0, noise / np.sqrt(n_trials), size=(T, C))
    return out


def test_d2_is_unbiased_at_zero_effect() -> None:
    """No true contrast => d_cv straddles zero, and does so however loud the noise is.

    This is the property a single-half magnitude does NOT have: ``||v||^2`` would grow with
    the noise level and never change sign.
    """
    rng = np.random.default_rng(0)
    T, C = 64, 16
    zero = np.zeros((T, C))
    for noise in (0.5, 2.0, 8.0):
        d = np.concatenate([dcv_rel(_halves(rng, zero, 40, noise))[0] for _ in range(200)])
        # mean sits at zero
        assert abs(d.mean()) < 0.35 * d.std(), (noise, d.mean(), d.std())
        # and it is negative about half the time -- the sign test is the real check
        assert 0.4 < (d < 0).mean() < 0.6, (noise, (d < 0).mean())


def test_single_half_magnitude_would_be_biased() -> None:
    """The contrast this design rejects: ``||v_all||^2`` is positive everywhere at zero effect."""
    rng = np.random.default_rng(1)
    T, C = 64, 16
    v = _halves(rng, np.zeros((T, C)), 40, 2.0)
    biased = (((v["h0"] + v["h1"]) / 2) ** 2).sum(axis=1) / C
    assert (biased > 0).all(), "a squared magnitude cannot be negative -- that is the bug"
    d, _ = dcv_rel(v)
    assert (d < 0).any(), "the split-half statistic must be able to go negative"


def test_d2_recovers_the_true_effect_size() -> None:
    """With a real contrast, ``d_cv`` estimates ``||true||/sqrt(C)`` rather than inflating it."""
    rng = np.random.default_rng(2)
    T, C = 64, 16
    true_v = np.zeros((T, C))
    true_v[20:40, :] = 0.8                       # a sustained effect in half the window
    want = np.linalg.norm(true_v[25]) / np.sqrt(C)

    est = []
    for _ in range(400):
        d, _ = dcv_rel(_halves(rng, true_v, 60, 3.0))
        est.append(d[25])
    got = float(np.mean(est))
    assert got == pytest.approx(want, rel=0.06), (got, want)
    # and the pre-effect region stays at zero
    pre = [dcv_rel(_halves(rng, true_v, 60, 3.0))[0][5] for _ in range(400)]
    assert abs(float(np.mean(pre))) < 0.1 * want


def test_reliability_cosine_separates_signal_from_noise() -> None:
    rng = np.random.default_rng(3)
    T, C = 64, 16
    strong = np.tile(np.linspace(0, 1.5, C), (T, 1))
    _, rel_sig = dcv_rel(_halves(rng, strong, 200, 1.0))
    _, rel_noise = dcv_rel(_halves(rng, np.zeros((T, C)), 200, 1.0))
    assert rel_sig.mean() > 0.5, rel_sig.mean()
    assert abs(rel_noise.mean()) < 0.2, rel_noise.mean()


def test_dkt_tables_cover_the_regions_the_figure_asserts_on() -> None:
    """A renamed atlas label would turn every anatomical assert into a silent no-op."""
    base_of, lobe_of_base = dkt_tables()
    bases = set(base_of.values())
    assert TARGET in bases, TARGET
    for r in SPEECH_NET:
        assert r in bases, f"{r} is not a DKT base -- the speech-network report is a no-op"
    for r in OCCIPITAL:
        assert r in bases, f"{r} is not a DKT base -- the occipital zero-check is a no-op"
    assert lobe_of_base[TARGET] == "temporal", lobe_of_base[TARGET]
    # DKT drops these two relative to DK; asserting on them would never fire
    assert "banksts" not in bases and "temporalpole" not in bases
