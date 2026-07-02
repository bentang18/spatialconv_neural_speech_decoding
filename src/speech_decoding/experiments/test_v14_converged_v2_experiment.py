"""TDD for V14ConvergedV2Experiment — the thin experiment that wraps the
self-contained V14ConvergedV2.

Quantitative checks of the experiment seam WITHOUT a full study chain (a fake
one-batch loader supplies the C-peek): the inert-phase + leakage-guard ClassVars,
the locked 2-band C-peek key + SSL clip-length default, and (the load-bearing
part) ``_build_brain_module`` resolving the v2 model from the config and threading
the optimizer / clip-len / mask seed into a V14ConvergedV2BrainModule. The v2
model owns ema_tau + the science knobs + masks, so this experiment is THINNER than
the 3STFT one — there is no ema_tau / mask-geometry / tube knob to thread."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from neuraltrain.optimizers import LightningOptimizer

from speech_decoding.experiments.v14_converged_v2_experiment import (
    V14ConvergedV2Experiment,
)
from speech_decoding.experiments.v14_converged_v2_module import (
    V14ConvergedV2BrainModule,
)
from speech_decoding.models.v14_converged_v2 import V14ConvergedV2
from speech_decoding.models.v14_converged_v2_config import V14ConvergedV2Net

N_PARCELS = 8


def _config(**kw) -> V14ConvergedV2Net:
    base = dict(
        d_model=16, n_heads=4, frontend_layers=1, latent_layers=1,
        m2_pred_layers=1, m4_pred_layers=1, pred_dim=16, n_parcels=N_PARCELS,
    )
    base.update(kw)
    return V14ConvergedV2Net(**base)


def _optim() -> LightningOptimizer:
    return LightningOptimizer(optimizer={"name": "AdamW", "lr": 1e-3})


def _fake_train_loader(C: int = 6):
    """One-batch loader yielding only the LFS band (the C-peek key). The shape
    beyond axis 1 (=C) is irrelevant — ``V14ConvergedV2Net.build`` ignores the
    electrode count; only ``x.shape[1]`` is read for the peek."""
    batch = SimpleNamespace(
        data={"electrode_tokens_lfs": torch.randn(2, C, 28, 10)}
    )
    return [batch]


# ----------------------------------------------------- inert phase + guard ClassVars
def test_leakage_guard_on_cohort_floor_deferred() -> None:
    assert V14ConvergedV2Experiment.enforces_pretrain_leakage_guard is True
    assert V14ConvergedV2Experiment.pretrain_cohort_floor is None


def test_phase_is_inert_default_one() -> None:
    xp = V14ConvergedV2Experiment.model_construct()
    assert xp.phase == 1


def test_locked_peek_key_and_clip_len_defaults() -> None:
    xp = V14ConvergedV2Experiment.model_construct()
    assert xp.x_name == "electrode_tokens_lfs"     # 2-band C-peek key
    assert xp.clip_len_s == 5.0                     # locked SSL pretrain clock
    assert xp.mask_seed == 0


# ---------------------------------------------------------- _build_brain_module
def test_build_brain_module_returns_v2_module() -> None:
    xp = V14ConvergedV2Experiment.model_construct(
        brain_model_config=_config(), optim=_optim(),
    )
    bm = xp._build_brain_module(_fake_train_loader())
    assert isinstance(bm, V14ConvergedV2BrainModule)
    assert isinstance(bm.model, V14ConvergedV2)
    # the config's shape reached the model
    assert bm.model.cfg.n_parcels == N_PARCELS


def test_build_brain_module_threads_optim_clip_and_seed() -> None:
    xp = V14ConvergedV2Experiment.model_construct(
        brain_model_config=_config(), optim=_optim(),
        clip_len_s=1.0, mask_seed=7,
    )
    bm = xp._build_brain_module(_fake_train_loader())
    assert bm.optim_config is xp.optim
    assert bm.clip_len_s == 1.0
    assert bm._mask_seed == 7


def test_build_brain_module_threads_science_knobs_via_config() -> None:
    """The science knobs (k / tube_ratio / pool_op / ema_tau) live on the MODEL
    config, not the experiment — confirm they reach the built model so the
    experiment never needs to (or can) re-thread them."""
    xp = V14ConvergedV2Experiment.model_construct(
        brain_model_config=_config(tube_ratio=0.3, ema_tau=0.95), optim=_optim(),
    )
    bm = xp._build_brain_module(_fake_train_loader())
    assert bm.model.cfg.tube_ratio == 0.3
    assert bm.model.cfg.ema_tau == 0.95


def test_build_brain_module_rejects_wrong_model_type() -> None:
    """If the config resolves to a non-v2 model the build must fail loud (guards
    against a dispatch mis-wire pointing at the 3STFT V14Converged)."""
    class _NotV2:
        def build(self, n_in_channels: int, n_outputs: int):  # noqa: ARG002
            return torch.nn.Linear(3, 3)

    xp = V14ConvergedV2Experiment.model_construct(
        brain_model_config=_NotV2(), optim=_optim(),
    )
    with pytest.raises(RuntimeError, match="V14ConvergedV2"):
        xp._build_brain_module(_fake_train_loader())


def test_build_brain_module_peek_ignores_n_in_channels() -> None:
    """``V14ConvergedV2Net.build`` ignores n_in_channels, so two different
    electrode counts in the peek loader yield structurally identical models."""
    xp = V14ConvergedV2Experiment.model_construct(
        brain_model_config=_config(), optim=_optim(),
    )
    a = xp._build_brain_module(_fake_train_loader(C=6))
    b = xp._build_brain_module(_fake_train_loader(C=40))
    sa = {k: tuple(v.shape) for k, v in a.model.state_dict().items()}
    sb = {k: tuple(v.shape) for k, v in b.model.state_dict().items()}
    assert sa == sb
