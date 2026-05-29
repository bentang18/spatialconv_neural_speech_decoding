"""T3.2 tests: ridge fit + preflight sweep on synthetic Gaussian features."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.experiments.phase3_preflight import (
    PreflightResult,
    ridge_fit_score,
    run_preflight_sweep,
)


def _planted_features(
    *, n_train: int, n_val: int, d_in: int, d_out: int,
    snr: float, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plant a linear ``Y = X W_true + noise`` setup; return X/Y train+val + W_true."""
    g = torch.Generator().manual_seed(seed)
    W_true = torch.randn(d_in, d_out, generator=g)
    X_train = torch.randn(n_train, d_in, generator=g)
    X_val = torch.randn(n_val, d_in, generator=g)
    Y_train = X_train @ W_true + (1.0 / snr) * torch.randn(n_train, d_out, generator=g)
    Y_val = X_val @ W_true + (1.0 / snr) * torch.randn(n_val, d_out, generator=g)
    return X_train, Y_train, X_val, Y_val, W_true


def test_ridge_fit_recovers_planted_weights_at_low_noise() -> None:
    X_train, Y_train, X_val, Y_val, W_true = _planted_features(
        n_train=500, n_val=100, d_in=8, d_out=4, snr=100.0, seed=0,
    )
    W, _, val_r2, _, val_cos = ridge_fit_score(
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val, alpha=1e-3,
    )
    assert torch.allclose(W, W_true, atol=0.05), "ridge fit should recover W_true at SNR=100"
    assert val_r2 > 0.99, f"val_r2={val_r2} too low at high SNR"
    assert val_cos > 0.99, f"val_cos={val_cos} too low at high SNR"


def test_ridge_alpha_regularization_pulls_weights_toward_zero() -> None:
    X_train, Y_train, X_val, Y_val, _ = _planted_features(
        n_train=200, n_val=50, d_in=6, d_out=3, snr=10.0, seed=1,
    )
    W_low, *_ = ridge_fit_score(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, alpha=1e-3)
    W_high, *_ = ridge_fit_score(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, alpha=1e6)
    assert W_high.norm() < W_low.norm(), "large alpha should shrink weights"


def test_ridge_negative_alpha_rejected() -> None:
    X_train, Y_train, X_val, Y_val, _ = _planted_features(
        n_train=50, n_val=25, d_in=4, d_out=2, snr=10.0, seed=2,
    )
    with pytest.raises(ValueError, match="non-negative"):
        ridge_fit_score(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, alpha=-1.0)


def test_ridge_shape_mismatch_rejected() -> None:
    X_train = torch.randn(20, 4)
    X_val = torch.randn(10, 5)  # mismatched D_in
    Y_train = torch.randn(20, 3)
    Y_val = torch.randn(10, 3)
    with pytest.raises(ValueError, match="D_in"):
        ridge_fit_score(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, alpha=0.1)


def test_preflight_sweep_picks_best_cell_by_val_r2() -> None:
    """Plant 2 layers × 2 rates where layer 8 @ rate 10 has highest SNR;
    the sweep must pick that cell."""

    def feature_fn(layer: int, rate: float):
        snr = 100.0 if (layer == 8 and rate == 10.0) else 1.5
        seed = 1000 * layer + int(rate)
        X_train, Y_train, X_val, Y_val, _ = _planted_features(
            n_train=300, n_val=80, d_in=6, d_out=4, snr=snr, seed=seed,
        )
        return X_train, Y_train, X_val, Y_val

    result = run_preflight_sweep(
        feature_fn=feature_fn,
        whisper_layers=[6, 8],
        target_rates_hz=[5.0, 10.0],
        ridge_alphas=(1e-3, 1.0, 100.0),
        tie_break="val_r2",
    )
    assert isinstance(result, PreflightResult)
    assert len(result.cells) == 4
    assert result.winner.whisper_layer == 8
    assert result.winner.target_rate_hz == 10.0
    assert result.tie_break_metric == "val_r2"


def test_preflight_sweep_picks_best_cell_by_val_cosine() -> None:
    """tie_break='val_cosine' selects the same planted winner."""

    def feature_fn(layer: int, rate: float):
        snr = 100.0 if (layer == 7 and rate == 20.0) else 1.5
        seed = 2000 * layer + int(rate)
        return _planted_features(
            n_train=300, n_val=80, d_in=6, d_out=4, snr=snr, seed=seed,
        )[:4]

    result = run_preflight_sweep(
        feature_fn=feature_fn,
        whisper_layers=[6, 7, 8],
        target_rates_hz=[5.0, 10.0, 20.0],
        ridge_alphas=(1e-3, 0.1),
        tie_break="val_cosine",
    )
    assert result.winner.whisper_layer == 7
    assert result.winner.target_rate_hz == 20.0


def test_empty_grid_raises() -> None:
    def feature_fn(layer: int, rate: float):
        return _planted_features(n_train=10, n_val=5, d_in=3, d_out=2, snr=10.0, seed=0)[:4]

    with pytest.raises(ValueError, match="whisper_layers"):
        run_preflight_sweep(
            feature_fn=feature_fn, whisper_layers=[], target_rates_hz=[10.0],
            ridge_alphas=(0.1,), tie_break="val_r2",
        )
    with pytest.raises(ValueError, match="target_rates_hz"):
        run_preflight_sweep(
            feature_fn=feature_fn, whisper_layers=[8], target_rates_hz=[],
            ridge_alphas=(0.1,), tie_break="val_r2",
        )
    with pytest.raises(ValueError, match="ridge_alphas"):
        run_preflight_sweep(
            feature_fn=feature_fn, whisper_layers=[8], target_rates_hz=[10.0],
            ridge_alphas=(), tie_break="val_r2",
        )


def test_unknown_tie_break_rejected() -> None:
    def feature_fn(layer: int, rate: float):
        return _planted_features(n_train=10, n_val=5, d_in=3, d_out=2, snr=10.0, seed=0)[:4]

    with pytest.raises(ValueError, match="tie_break"):
        run_preflight_sweep(
            feature_fn=feature_fn, whisper_layers=[8], target_rates_hz=[10.0],
            ridge_alphas=(0.1,), tie_break="bogus",  # type: ignore[arg-type]
        )
