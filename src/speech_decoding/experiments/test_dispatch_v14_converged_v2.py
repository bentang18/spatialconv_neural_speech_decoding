"""TDD for the dispatch `--frontend 2band` converged-v2 branch in
build_v14_experiment.

Verifies (without BT data — a tmp bt_root constructs the Experiment object but
never builds loaders): the 2band path returns a V14ConvergedV2Experiment with a
V14ConvergedV2Net config of the named shape, emits the two band keys + support +
valid_mask into the segmenter, requires the v2 shape (no silent run defaults),
threads the science knobs (k / pool_op / tube_ratio / ema_tau) onto the MODEL
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
        converged_v2_pred_dim=32,
        converged_v2_m2_pred_layers=2, converged_v2_m4_pred_layers=3,
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
    assert cfg.pred_dim == 32
    assert cfg.m2_pred_layers == 2 and cfg.m4_pred_layers == 3
    # n_parcels = DK K=80 (the atlas vocabulary length), derived not passed
    assert cfg.n_parcels == 80


def test_2band_science_knob_defaults(tmp_path) -> None:
    cfg = _v2(tmp_path).brain_model_config
    assert cfg.k == 2                       # locked first-run seeds/parcel
    assert cfg.pool_op is None              # auto (Run-A/legacy ⇒ band n_op=2)
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
    assert cfg.pool_op == "patch"           # --untie-lfs alias ⇒ patch (n_op=4)
    assert cfg.tube_ratio == 0.30
    assert cfg.ema_tau == pytest.approx(0.97)


def test_2band_pool_op_explicit_beats_untie_alias(tmp_path) -> None:
    # explicit --converged-v2-pool-op wins over the --untie-lfs back-compat alias
    cfg = _v2(
        tmp_path, converged_v2_pool_op="band", converged_v2_untie_lfs=True,
    ).brain_model_config
    assert cfg.pool_op == "band"


def test_2band_run_b_default_pool_op_resolves_band(tmp_path) -> None:
    # Run-B (m3_drop_frac set), no pool_op ⇒ Net carries None, model resolves "band"
    # (n_op=2, LFS | HGA — the physiology-backed default).
    cfg = _v2(tmp_path, converged_v2_m3_drop_frac=0.5).brain_model_config
    assert cfg.pool_op is None
    model = cfg.build(n_in_channels=1, n_outputs=1)
    assert model.pool.W_K.shape[0] == 2     # n_op=2 (band-tied LFS | HGA)


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


# --------------------------------------------------------------- Run-B wiring
def test_2band_run_b_config_carries_knobs(tmp_path) -> None:
    """Setting the Run-B master switch (m3_drop_frac) threads the recon knobs into
    the V14ConvergedV2Net config."""
    cfg = _v2(
        tmp_path, converged_v2_m3_drop_frac=0.4, converged_v2_m3_min_keep=3,
        converged_v2_w_melec=0.5, converged_v2_sigma_mm=12.0,
        converged_v2_geom_n_freqs=6,
    ).brain_model_config
    assert cfg.m3_drop_frac == 0.4
    assert cfg.m3_min_keep == 3
    assert cfg.w_melec == 0.5
    assert cfg.sigma_mm == 12.0
    assert cfg.geom_n_freqs == 6


def test_2band_default_is_run_a_no_run_b_knobs(tmp_path) -> None:
    cfg = _v2(tmp_path).brain_model_config
    assert cfg.m3_drop_frac is None                     # Run-B gate off ⇒ Run-A/legacy
    # m2_hetero defaults True (canonical Run-B masking unit = electrodes) but is
    # INERT while m3_drop_frac is None, so a default 2band run is still pure Run-A.
    assert cfg.m2_hetero is True


def test_2band_run_b_m2_hetero_threads(tmp_path) -> None:
    cfg = _v2(
        tmp_path, converged_v2_m3_drop_frac=0.4, converged_v2_m2_hetero=True,
    ).brain_model_config
    assert cfg.m2_hetero is True


def test_2band_run_b_emits_coords_extractor(tmp_path) -> None:
    """Run-B (m3_drop_frac set) auto-registers the native-RAS coords extractor."""
    xp = _v2(tmp_path, converged_v2_m3_drop_frac=0.4)
    keys = set(xp.data.segmenter.extractors)
    assert "electrode_coords" in keys


def test_2band_run_a_omits_coords_extractor(tmp_path) -> None:
    """No coords key on the default (Run-A) path — keeps the segmenter hash / exca
    cache uid unchanged for existing runs."""
    xp = _v2(tmp_path)
    assert "electrode_coords" not in set(xp.data.segmenter.extractors)


@pytest.mark.parametrize(
    "drop",
    ["converged_frontend_layers", "converged_latent_layers",
     "converged_v2_pred_dim", "converged_v2_m2_pred_layers",
     "converged_v2_m4_pred_layers"],
)
def test_2band_requires_v2_shape(tmp_path, drop: str) -> None:
    """No silent run defaults: omitting any v2 shape param is a hard dispatch
    error (the shape is Ben's to name at launch)."""
    kw: dict[str, Any] = dict(
        bt_root=str(tmp_path), mode="lite", frontend="2band",
        d_model=64, n_heads=4,
        converged_frontend_layers=2, converged_latent_layers=2,
        converged_v2_pred_dim=32,
        converged_v2_m2_pred_layers=2, converged_v2_m4_pred_layers=3,
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
        "--converged-v2-pred-dim", "128",
        "--converged-v2-m2-pred-layers", "3", "--converged-v2-m4-pred-layers", "6",
        "--converged-v2-k", "2", "--converged-v2-pool-op", "shared",
        "--group-by-session",
    ])
    assert a.frontend == "2band"
    assert a.converged_v2_pred_dim == 128
    assert a.converged_v2_m2_pred_layers == 3
    assert a.converged_v2_m4_pred_layers == 6
    assert a.converged_v2_k == 2
    assert a.converged_v2_pool_op == "shared"
    assert a.converged_v2_untie_lfs is False
    # The `--frontend 2band requires --group-by-session` SystemExit guard lives
    # inline in main() (alongside the static-forward guards, which the existing
    # dispatch test also does not unit-test); the AUTHORITATIVE enforcement is the
    # module's runtime fail-loud, covered by
    # test_v14_converged_v2_module::test_v2_inputs_reject_heterogeneous_batch.


def test_v2_config_registered_via_models_package() -> None:
    """Worker-context registration guard. On DeltaAI the experiment's
    brain_model_config dict {"name": "V14ConvergedV2Net", ...} is RE-VALIDATED
    against the BaseModelConfig discriminated union in a process that imports only
    `speech_decoding.models` (not the v2 module directly). The discriminator is
    registered as an import side-effect, so models/__init__ MUST import the v2
    config — otherwise the run dies at config validation (observed on DeltaAI
    2026-06-26). A subprocess gives the clean import the bug needs; an in-process
    check would be masked by sibling test imports."""
    import subprocess
    import sys

    code = (
        "import speech_decoding.models\n"
        "from neuraltrain.models.base import BaseModelConfig\n"
        "o = BaseModelConfig.model_validate({'name': 'V14ConvergedV2Net',\n"
        "  'd_model': 256, 'n_heads': 4, 'frontend_layers': 4, 'latent_layers': 8,\n"
        "  'm2_pred_layers': 2, 'm4_pred_layers': 6, 'pred_dim': 128, 'n_parcels': 80})\n"
        "assert type(o).__name__ == 'V14ConvergedV2Net', type(o).__name__\n"
        "assert o.m2_pred_layers == 2 and o.m4_pred_layers == 6\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"v2 config not registered from bare models import:\n{r.stderr}"
