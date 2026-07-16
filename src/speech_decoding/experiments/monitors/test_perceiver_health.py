"""Invariant tests for MON-PERCEIVER-HEALTH (dead-feature fraction + latent redundancy).

Each test bakes the invariant into a constructed input so a plausible-wrong number can't
pass (the M4/M7 lesson): known dead dims → exact fraction, identical latents → cosine 1,
orthonormal latents → cosine 0.
"""

from __future__ import annotations

import math

import torch

from speech_decoding.experiments.monitors.perceiver_health import (
    dead_feature_fraction,
    latent_redundancy,
    perceiver_latent_health,
)


def test_dead_fraction_counts_exactly_the_constant_dims() -> None:
    # 100 samples, 10 dims; dims 0..2 are constant (std 0 → dead), 3..9 unit-variance.
    torch.manual_seed(0)
    feats = torch.randn(100, 10)
    feats[:, :3] = 5.0  # constant ⇒ std 0
    assert math.isclose(dead_feature_fraction(feats), 3 / 10, abs_tol=1e-6)


def test_dead_fraction_zero_for_isotropic_code() -> None:
    torch.manual_seed(1)
    feats = torch.randn(2000, 64)  # all dims ~unit std ⇒ none below rel·median
    assert dead_feature_fraction(feats) == 0.0


def test_dead_fraction_degenerate_small_n() -> None:
    assert dead_feature_fraction(torch.randn(1, 8)) == 0.0


def test_redundancy_identical_latents_saturate_to_one() -> None:
    # every latent in every clip is the same vector ⇒ pairwise cosine ≡ 1.
    v = torch.randn(1, 1, 16)
    lat = v.expand(4, 12, 16).contiguous()
    cos_mean, cos_pct95 = latent_redundancy(lat)
    assert math.isclose(cos_mean, 1.0, abs_tol=1e-5)
    assert math.isclose(cos_pct95, 1.0, abs_tol=1e-5)


def test_redundancy_orthonormal_latents_near_zero() -> None:
    # L=8 orthonormal rows (identity) in d=8 ⇒ off-diagonal cosine ≡ 0.
    eye = torch.eye(8)[None].expand(3, 8, 8).contiguous()
    cos_mean, cos_pct95 = latent_redundancy(eye)
    assert math.isclose(cos_mean, 0.0, abs_tol=1e-6)
    assert math.isclose(cos_pct95, 0.0, abs_tol=1e-6)


def test_bundle_keys_and_ranges() -> None:
    torch.manual_seed(2)
    lat = torch.randn(4, 24, 128)
    out = perceiver_latent_health(lat)
    assert set(out) == {"dead_frac", "cos_mean", "cos_pct95"}
    assert 0.0 <= out["dead_frac"] <= 1.0
    assert -1.0 <= out["cos_mean"] <= 1.0
    assert -1.0 <= out["cos_pct95"] <= 1.0
