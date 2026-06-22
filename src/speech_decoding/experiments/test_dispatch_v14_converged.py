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


def test_3stft_mask_geometry_defaults(tmp_path) -> None:
    # None at the CLI → the LOCKED experiment defaults (Ben 2026-06-18: HG 20%).
    xp = _converged(tmp_path)
    assert xp.m2_hg_start_rate == pytest.approx(0.20)
    assert xp.m2_hg_span == 3
    assert xp.m2_beta_start_rate == pytest.approx(0.30)  # 2026-06-19 beta coverage dial
    assert xp.m2_beta_span == 2                          # 2026-06-19 span 2 (coarse grid)
    assert xp.m4_parcel_mask_ratio == pytest.approx(0.20)


def test_3stft_mask_geometry_flags_thread(tmp_path) -> None:
    # The §8.7 sister knobs override from dispatch (mask geometry, tuned often).
    xp = _converged(
        tmp_path, converged_m2_hg_start_rate=0.35, converged_m2_hg_span=5,
        converged_m2_beta_start_rate=0.45, converged_m2_beta_span=6,
        converged_m4_parcel_mask_ratio=0.30,
    )
    assert xp.m2_hg_start_rate == pytest.approx(0.35)
    assert xp.m2_hg_span == 5
    assert xp.m2_beta_start_rate == pytest.approx(0.45)
    assert xp.m2_beta_span == 6
    assert xp.m4_parcel_mask_ratio == pytest.approx(0.30)


def test_3stft_m4_precision_defaults_on(tmp_path) -> None:
    cfg = _converged(tmp_path).brain_model_config
    assert cfg.m4_precision_weight is True       # ON by default (Ben 2026-06-18)
    assert cfg.m4_precision_alpha == pytest.approx(1.0)
    assert cfg.m4_precision_n_ref == pytest.approx(11.0)


def test_3stft_m4_precision_off_and_overrides(tmp_path) -> None:
    cfg = _converged(
        tmp_path, converged_m4_precision_off=True,
        converged_m4_precision_alpha=1.15, converged_m4_precision_n_ref=15.0,
    ).brain_model_config
    assert cfg.m4_precision_weight is False       # R-precision-off sister
    assert cfg.m4_precision_alpha == pytest.approx(1.15)
    assert cfg.m4_precision_n_ref == pytest.approx(15.0)


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


# ------------------------------------------------------------- online probe knobs
def test_3stft_online_probe_off_by_default(tmp_path) -> None:
    """The diagnostic probe is OFF unless --online-probe is passed (inert default)."""
    xp = _converged(tmp_path)
    assert xp.online_probe_enabled is False
    assert xp.online_probe_cadence == 1000


def test_3stft_online_probe_flags_thread(tmp_path) -> None:
    """--online-probe / --online-probe-cadence reach the converged experiment."""
    xp = _converged(tmp_path, online_probe_enabled=True, online_probe_cadence=500)
    assert xp.online_probe_enabled is True
    assert xp.online_probe_cadence == 500


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


# --- static-forward throughput-regime cohesion (Ben 2026-06-22) ----------------
def _resolve(argv):
    """Parse a converged argv and apply the static-forward cohesion, returning the
    resolved (compile_dynamic, sdpa_backend)."""
    from speech_decoding.experiments.dispatch_v14 import (
        _parser,
        _resolve_static_forward_cohesion,
    )
    a = _parser().parse_args(argv)
    _resolve_static_forward_cohesion(a)
    return a.compile_dynamic, a.sdpa_backend


_BASE = [
    "--frontend", "3stft",
    "--converged-frontend-layers", "6", "--converged-latent-layers", "6",
    "--converged-m2-pred-dim", "128", "--converged-m2-pred-layers", "4",
    "--converged-m4-pred-dim", "128", "--converged-m4-pred-layers", "6",
]
_STATIC = ["--converged-static-forward", "--converged-tube-ratio", "0.25",
           "--group-by-session"]


def test_cohesion_legacy_path_unchanged() -> None:
    # No static-forward: dynamic compile stays ON, SDPA stays at the dispatcher
    # default — byte-identical to the prior hard defaults.
    assert _resolve(_BASE) == (True, "default")


def test_cohesion_static_forward_autofills_both_wins() -> None:
    # Static-forward auto-resolves to static compile + cuDNN-latent.
    assert _resolve(_BASE + _STATIC) == (False, "cudnn_latent")


def test_cohesion_explicit_flags_override_autofill() -> None:
    # An explicit --compile-dynamic / --sdpa-backend still wins under static-forward.
    assert _resolve(_BASE + _STATIC + ["--compile-dynamic"]) == (True, "cudnn_latent")
    assert _resolve(_BASE + _STATIC + ["--sdpa-backend", "flash"]) == (False, "flash")


def test_cohesion_does_not_autofill_find_unused() -> None:
    # find_unused=False (--ddp-static-graph) is NOT implied by static-forward.
    from speech_decoding.experiments.dispatch_v14 import _parser
    a = _parser().parse_args(_BASE + _STATIC)
    assert a.ddp_static_graph is False
