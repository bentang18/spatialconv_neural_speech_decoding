"""TDD for V14ConvergedExperiment — the thin Philosophy-B experiment that wraps
the self-contained V14ConvergedSSL.

Quantitative checks of the experiment seam: the inert-phase + leakage-guard
ClassVars, the locked mask/EMA defaults, and (the load-bearing part)
``_build_brain_module`` resolving the model from the config and threading the
optimizer / EMA τ / mask config into a V14ConvergedBrainModule — WITHOUT building
a full study chain (a fake one-batch loader supplies the C-peek)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from neuraltrain.optimizers import LightningOptimizer

from speech_decoding.experiments.v14_converged_experiment import (
    V14ConvergedExperiment,
)
from speech_decoding.experiments.v14_converged_module import (
    V14ConvergedBrainModule,
)
from speech_decoding.models.v14_converged import V14ConvergedSSL
from speech_decoding.models.v14_converged_config import V14Converged


def _config(**kw) -> V14Converged:
    base = dict(
        d_model=16, n_parcels=4, n_heads=4,
        frontend_layers=1, latent_layers=1,
        m2_pred_dim=16, m2_pred_layers=1,
        m4_pred_dim=16, m4_pred_layers=1,
    )
    base.update(kw)
    return V14Converged(**base)


def _optim() -> LightningOptimizer:
    return LightningOptimizer(optimizer={"name": "AdamW", "lr": 1e-3})


def _fake_train_loader(C: int = 6):
    """One-batch loader yielding only the slow band (the C-peek key)."""
    batch = SimpleNamespace(data={"electrode_tokens_slow": torch.randn(2, C, 2, 6, 5)})
    return [batch]


# ----------------------------------------------------- inert phase + guard ClassVars
def test_leakage_guard_on_cohort_floor_deferred() -> None:
    assert V14ConvergedExperiment.enforces_pretrain_leakage_guard is True
    assert V14ConvergedExperiment.pretrain_cohort_floor is None


def test_phase_is_inert_default_one() -> None:
    xp = V14ConvergedExperiment.model_construct()
    assert xp.phase == 1


def test_locked_mask_and_ema_defaults() -> None:
    xp = V14ConvergedExperiment.model_construct()
    assert xp.ema_tau == 0.99925
    assert xp.m2_hg_start_rate == 0.20
    assert xp.m2_hg_span == 3
    assert xp.m2_beta_span == 4
    assert xp.m4_parcel_mask_ratio == 0.20
    assert xp.mask_seed == 0
    assert xp.x_name == "electrode_tokens_slow"


# ---------------------------------------------------------- _build_brain_module
def test_build_brain_module_returns_converged_module() -> None:
    xp = V14ConvergedExperiment.model_construct(
        brain_model_config=_config(), optim=_optim(),
    )
    bm = xp._build_brain_module(_fake_train_loader())
    assert isinstance(bm, V14ConvergedBrainModule)
    assert isinstance(bm.model, V14ConvergedSSL)
    # the config's shape reached the model
    assert bm.model.latent.parcel_embed.num_embeddings == 4


def test_build_brain_module_threads_optim_and_ema_and_masks() -> None:
    xp = V14ConvergedExperiment.model_construct(
        brain_model_config=_config(), optim=_optim(),
        ema_tau=0.95, m2_beta_span=5, m4_parcel_mask_ratio=0.3, mask_seed=7,
    )
    bm = xp._build_brain_module(_fake_train_loader())
    assert bm.optim_config is xp.optim
    assert bm.ema_tau == 0.95
    assert bm.m2_cfg.beta_span == 5
    assert bm.m4_cfg.parcel_mask_ratio == 0.3
    assert bm._mask_seed == 7


def test_build_brain_module_rejects_wrong_model_type() -> None:
    """If the config resolves to a non-converged model the build must fail loud
    (guards against a dispatch mis-wire pointing at V14ParcelPerceiver)."""
    import pytest

    class _NotConverged:
        def build(self, n_in_channels: int, n_outputs: int):
            return torch.nn.Linear(3, 3)

    xp = V14ConvergedExperiment.model_construct(
        brain_model_config=_NotConverged(), optim=_optim(),
    )
    with pytest.raises(RuntimeError, match="V14ConvergedSSL"):
        xp._build_brain_module(_fake_train_loader())


def test_build_brain_module_peek_ignores_n_in_channels() -> None:
    """``V14Converged.build`` ignores n_in_channels, so two different electrode
    counts in the peek loader yield structurally identical models."""
    xp = V14ConvergedExperiment.model_construct(
        brain_model_config=_config(), optim=_optim(),
    )
    a = xp._build_brain_module(_fake_train_loader(C=6))
    b = xp._build_brain_module(_fake_train_loader(C=40))
    sa = {k: tuple(v.shape) for k, v in a.model.state_dict().items()}
    sb = {k: tuple(v.shape) for k, v in b.model.state_dict().items()}
    assert sa == sb
