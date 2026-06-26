"""TDD for the dispatch `--frontend 2band` converged-v2 branch in
build_v14_experiment.

Verifies (without BT data — a tmp bt_root constructs the Experiment object but
never builds loaders): the 2band path returns a V14ConvergedV2Experiment with a
V14ConvergedV2Net config of the named shape, emits the two band keys + support +
valid_mask into the segmenter, requires the v2 shape (no silent run defaults),
threads the science knobs (k / tie_lfs / tube_ratio / ema_tau) onto the MODEL
config, and — the regression guard — leaves the raw / 2stft / 3stft paths
byte-identical."""

from __future__ import annotations

from typing import Any

import pytest

from speech_decoding.experiments.dispatch_v14 import build_v14_experiment
from speech_decoding.experiments.v14_converged_v2_experiment import (
    V14ConvergedV2Experiment,
)
from speech_decoding.models.v14_converged_v2_config import V14ConvergedV2Net


def _v2(tmp_path, **kw):
    base: dict[str, Any] = dict(
        bt_root=str(tmp_path), mode="lite", frontend="2band",
        d_model=64, n_heads=4,
        converged_frontend_layers=2, converged_latent_layers=2,
        converged_v2_pred_dim=32, converged_v2_pred_layers=2,
    )
    base.update(kw)
    return build_v14_experiment(**base)


def test_2band_returns_v2_experiment(tmp_path) -> None:
    xp = _v2(tmp_path)
    assert isinstance(xp, V14ConvergedV2Experiment)
    assert isinstance(xp.brain_model_config, V14ConvergedV2Net)


def test_2band_config_carries_named_shape(tmp_path) -> None:
    cfg = _v2(tmp_path, d_model=48).brain_model_config
    assert cfg.d_model == 48
    assert cfg.n_heads == 4
    assert cfg.frontend_layers == 2
    assert cfg.latent_layers == 2
    assert cfg.pred_dim == 32 and cfg.pred_layers == 2
    # n_parcels = DK K=80 (the atlas vocabulary length), derived not passed
    assert cfg.n_parcels == 80


def test_2band_science_knob_defaults(tmp_path) -> None:
    cfg = _v2(tmp_path).brain_model_config
    assert cfg.k == 2                       # locked first-run seeds/parcel
    assert cfg.tie_lfs is True              # tied n_op=2 default
    assert cfg.tube_ratio == 0.25           # config locked default (none passed)
    # ema_tau is the RUN momentum: the dispatch resolves it to DEFAULT_EMA_TAU
    # (the single source of truth for all converged frontends), which overrides the
    # config's bare-model 0.9992 fallback. So a default 2band run gets 0.99925.
    assert cfg.ema_tau == 0.99925


def test_2band_science_knobs_thread(tmp_path) -> None:
    cfg = _v2(
        tmp_path, converged_v2_k=3, converged_v2_untie_lfs=True,
        converged_tube_ratio=0.30, ema_tau=0.97,
    ).brain_model_config
    assert cfg.k == 3
    assert cfg.tie_lfs is False             # --converged-v2-untie-lfs ⇒ n_op=4
    assert cfg.tube_ratio == 0.30
    assert cfg.ema_tau == pytest.approx(0.97)


def test_2band_clip_len_threads(tmp_path) -> None:
    xp = _v2(tmp_path, clip_len=1.0)
    assert xp.clip_len_s == 1.0


def test_2band_segmenter_emits_two_band_keys(tmp_path) -> None:
    xp = _v2(tmp_path)
    keys = set(xp.data.segmenter.extractors)
    assert {"electrode_tokens_lfs", "electrode_tokens_hga"} <= keys
    assert {"support", "valid_mask"} <= keys
    # the single-grid + 3-band keys are NOT present on the 2band path
    assert "electrode_tokens" not in keys
    assert "electrode_tokens_slow" not in keys
    assert "electrode_tokens_beta" not in keys
    assert "electrode_tokens_hg" not in keys


def test_2band_x_name_is_lfs_peek_key(tmp_path) -> None:
    xp = _v2(tmp_path)
    assert xp.x_name == "electrode_tokens_lfs"


@pytest.mark.parametrize(
    "drop",
    ["converged_frontend_layers", "converged_latent_layers",
     "converged_v2_pred_dim", "converged_v2_pred_layers"],
)
def test_2band_requires_v2_shape(tmp_path, drop: str) -> None:
    """No silent run defaults: omitting any v2 shape param is a hard dispatch
    error (the shape is Ben's to name at launch)."""
    kw: dict[str, Any] = dict(
        bt_root=str(tmp_path), mode="lite", frontend="2band",
        d_model=64, n_heads=4,
        converged_frontend_layers=2, converged_latent_layers=2,
        converged_v2_pred_dim=32, converged_v2_pred_layers=2,
    )
    kw[drop] = None
    with pytest.raises(ValueError, match="2band"):
        build_v14_experiment(**kw)


# ----------------------------------------------------------------- regression
def test_3stft_path_unchanged_still_converged(tmp_path) -> None:
    from speech_decoding.experiments.v14_converged_experiment import (
        V14ConvergedExperiment,
    )
    xp = build_v14_experiment(
        bt_root=str(tmp_path), mode="lite", frontend="3stft",
        d_model=64, n_heads=4,
        converged_frontend_layers=2, converged_latent_layers=2,
        converged_m2_pred_dim=32, converged_m2_pred_layers=2,
        converged_m4_pred_dim=32, converged_m4_pred_layers=2,
    )
    assert isinstance(xp, V14ConvergedExperiment)
    keys = set(xp.data.segmenter.extractors)
    assert {"electrode_tokens_slow", "electrode_tokens_beta",
            "electrode_tokens_hg"} <= keys
    assert "electrode_tokens_lfs" not in keys


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


# --------------------------------------------------------- parser + arg threading
def test_parser_accepts_2band_and_v2_args() -> None:
    from speech_decoding.experiments.dispatch_v14 import _parser

    a = _parser().parse_args([
        "--frontend", "2band",
        "--converged-frontend-layers", "6", "--converged-latent-layers", "8",
        "--converged-v2-pred-dim", "128", "--converged-v2-pred-layers", "4",
        "--converged-v2-k", "2", "--converged-v2-untie-lfs",
        "--group-by-session",
    ])
    assert a.frontend == "2band"
    assert a.converged_v2_pred_dim == 128
    assert a.converged_v2_pred_layers == 4
    assert a.converged_v2_k == 2
    assert a.converged_v2_untie_lfs is True
    # The `--frontend 2band requires --group-by-session` SystemExit guard lives
    # inline in main() (alongside the static-forward guards, which the existing
    # dispatch test also does not unit-test); the AUTHORITATIVE enforcement is the
    # module's runtime fail-loud, covered by
    # test_v14_converged_v2_module::test_v2_inputs_reject_heterogeneous_batch.
