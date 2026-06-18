"""TDD for V14Converged (the BaseModelConfig that registers V14ConvergedSSL).

Quantitative checks of the dispatch seam: ``.build()`` returns a correctly-sized
V14ConvergedSSL, the shape fields are REQUIRED (no silent run defaults), the
neutral/locked defaults thread through, and the discriminator resolves the model
from a ``{"name": "V14Converged", ...}`` dict exactly as the dispatch supplies it.
"""

from __future__ import annotations

import pytest

from speech_decoding.models.v14_converged import V14ConvergedSSL
from speech_decoding.models.v14_converged_config import V14Converged


def _cfg(**kw) -> V14Converged:
    base = dict(
        d_model=16, n_parcels=5, n_heads=4,
        frontend_layers=1, latent_layers=1,
        m2_pred_dim=16, m2_pred_layers=1,
        m4_pred_dim=16, m4_pred_layers=1,
    )
    base.update(kw)
    return V14Converged(**base)


# ---------------------------------------------------------------- build returns
def test_build_returns_converged_ssl() -> None:
    model = _cfg().build(n_in_channels=6, n_outputs=1)
    assert isinstance(model, V14ConvergedSSL)
    # the requested shape reached the model
    assert model.tokens_per_electrode == 38
    assert model.latent.parcel_embed.num_embeddings == 5


def test_build_ignores_n_in_channels_and_n_outputs() -> None:
    """Band geometry is fixed by BANDS, not the electrode count; SSL has no head.
    So different ``n_in_channels``/``n_outputs`` yield structurally identical
    models (same parameter shapes)."""
    a = _cfg().build(n_in_channels=6, n_outputs=1)
    b = _cfg().build(n_in_channels=99, n_outputs=42)
    sa = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    sb = {k: tuple(v.shape) for k, v in b.state_dict().items()}
    assert sa == sb


# ----------------------------------------------------------- required vs default
@pytest.mark.parametrize(
    "missing",
    ["d_model", "n_parcels", "n_heads", "frontend_layers", "latent_layers",
     "m2_pred_dim", "m2_pred_layers", "m4_pred_dim", "m4_pred_layers"],
)
def test_shape_fields_are_required(missing: str) -> None:
    """No pre-committed numeric defaults: omitting any shape field is a hard
    validation error, forcing the dispatch to name every width."""
    full = dict(
        d_model=16, n_parcels=5, n_heads=4,
        frontend_layers=1, latent_layers=1,
        m2_pred_dim=16, m2_pred_layers=1,
        m4_pred_dim=16, m4_pred_layers=1,
    )
    full.pop(missing)
    with pytest.raises(Exception):  # pydantic ValidationError
        V14Converged(**full)


def test_neutral_and_locked_defaults() -> None:
    cfg = _cfg()
    assert cfg.lambda_m2 == 1.0 and cfg.lambda_m4 == 1.0
    assert cfg.freq_pos == "learned"  # locked arch: 1d learnable freq tag


def test_lambda_and_freq_pos_thread_into_model() -> None:
    model = _cfg(lambda_m2=0.3, lambda_m4=0.7, freq_pos="sinusoidal").build(
        n_in_channels=6, n_outputs=1,
    )
    assert model.lambda_m2 == pytest.approx(0.3)
    assert model.lambda_m4 == pytest.approx(0.7)


# ------------------------------------------------------------ discriminator path
def test_resolves_from_name_dict_like_dispatch() -> None:
    """The dispatch supplies ``brain_model_config={"name": "V14Converged", ...}``;
    the BaseModelConfig discriminated union must resolve that to a V14Converged."""
    from neuraltrain.models.base import BaseModelConfig

    import speech_decoding.models  # noqa: F401  # registers the discriminator

    cfg = BaseModelConfig.model_validate({
        "name": "V14Converged",
        "d_model": 16, "n_parcels": 5, "n_heads": 4,
        "frontend_layers": 1, "latent_layers": 1,
        "m2_pred_dim": 16, "m2_pred_layers": 1,
        "m4_pred_dim": 16, "m4_pred_layers": 1,
    })
    assert isinstance(cfg, V14Converged)
    assert isinstance(cfg.build(n_in_channels=6, n_outputs=1), V14ConvergedSSL)
