"""TDD for the dispatch `--frontend 3stft` converged branch in build_v14_experiment.

Verifies (without BT data — a tmp bt_root constructs the Experiment object but
never builds loaders): the 3stft path returns a V14ConvergedExperiment with a
V14Converged config of the named shape, emits the three band keys + support +
valid_mask into the segmenter, requires the converged shape (no silent run
defaults), and — the regression guard — leaves the raw/2stft V14ParcelPerceiver
paths byte-identical."""

from __future__ import annotations

from typing import Any

import pytest

from speech_decoding.experiments.dispatch_v14 import build_v14_experiment
from speech_decoding.experiments.v14_converged_experiment import (
    V14ConvergedExperiment,
)
from speech_decoding.models.v14_converged_config import V14Converged


def _converged(tmp_path, **kw):
    base: dict[str, Any] = dict(
        bt_root=str(tmp_path), mode="lite", frontend="3stft",
        d_model=64, n_heads=4,
        converged_frontend_layers=2, converged_latent_layers=2,
        converged_m2_pred_dim=32, converged_m2_pred_layers=2,
        converged_m4_pred_dim=32, converged_m4_pred_layers=2,
    )
    base.update(kw)
    return build_v14_experiment(**base)


def test_3stft_returns_converged_experiment(tmp_path) -> None:
    xp = _converged(tmp_path)
    assert isinstance(xp, V14ConvergedExperiment)
    assert isinstance(xp.brain_model_config, V14Converged)


def test_3stft_config_carries_named_shape(tmp_path) -> None:
    xp = _converged(tmp_path, d_model=48)
    cfg = xp.brain_model_config
    assert cfg.d_model == 48
    assert cfg.n_heads == 4
    assert cfg.frontend_layers == 2
    assert cfg.latent_layers == 2
    assert cfg.m2_pred_dim == 32 and cfg.m2_pred_layers == 2
    assert cfg.m4_pred_dim == 32 and cfg.m4_pred_layers == 2
    # n_parcels = DK K=80 (the atlas vocabulary length), derived not passed
    assert cfg.n_parcels == 80
    # freq_pos is the LOCKED converged arch decision (1d learnable), forced
    # regardless of the dispatch's V14ParcelPerceiver "sinusoidal" default
    assert cfg.freq_pos == "learned"


def test_3stft_segmenter_emits_three_band_keys(tmp_path) -> None:
    xp = _converged(tmp_path)
    keys = set(xp.data.segmenter.extractors)
    assert {"electrode_tokens_slow", "electrode_tokens_beta",
            "electrode_tokens_hg"} <= keys
    assert {"support", "valid_mask"} <= keys
    # the live-path single-grid key is NOT present on the converged path
    assert "electrode_tokens" not in keys


def test_3stft_ema_tau_threads(tmp_path) -> None:
    xp = _converged(tmp_path, ema_tau=0.97)
    assert xp.ema_tau == pytest.approx(0.97)


@pytest.mark.parametrize(
    "drop",
    ["converged_frontend_layers", "converged_latent_layers",
     "converged_m2_pred_dim", "converged_m2_pred_layers",
     "converged_m4_pred_dim", "converged_m4_pred_layers"],
)
def test_3stft_requires_converged_shape(tmp_path, drop: str) -> None:
    """No silent run defaults: omitting any converged shape param is a hard
    dispatch error (the shape is Ben's to name at launch)."""
    kw: dict[str, Any] = dict(
        bt_root=str(tmp_path), mode="lite", frontend="3stft",
        d_model=64, n_heads=4,
        converged_frontend_layers=2, converged_latent_layers=2,
        converged_m2_pred_dim=32, converged_m2_pred_layers=2,
        converged_m4_pred_dim=32, converged_m4_pred_layers=2,
    )
    kw[drop] = None
    with pytest.raises(ValueError, match="3stft"):
        build_v14_experiment(**kw)


# ----------------------------------------------------------------- regression
def test_raw_path_unchanged_still_v14parcelperceiver(tmp_path) -> None:
    xp = build_v14_experiment(bt_root=str(tmp_path), mode="lite", joint_phase=True)
    assert type(xp.brain_model_config).__name__ == "V14ParcelPerceiver"
    assert "electrode_tokens" in set(xp.data.segmenter.extractors)


def test_2stft_path_unchanged(tmp_path) -> None:
    xp = build_v14_experiment(
        bt_root=str(tmp_path), mode="lite", joint_phase=True,
        frontend="2stft", pool="mean",
    )
    assert type(xp.brain_model_config).__name__ == "V14ParcelPerceiver"
    keys = set(xp.data.segmenter.extractors)
    assert "electrode_tokens" in keys and "electrode_tokens_high" in keys
