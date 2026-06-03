"""Pre-dispatch gates — TST03 / TST05 + RT10.

Spec: ``docs/neuroprobe/v14_blockers.md`` §"Pre-Phase-1 unit tests (TST)"
and §"Silent-failure traps (RT)". All tests in this file are marked
``@pytest.mark.must_pass_before_dispatch`` and run by the
``scripts/dcc/dispatch`` pre-flight gate.

The tests below cover (B36 WS-B masked-JEPA surface):

* **TST03** — V14JointBrainModule state-dict round-trip with
  ``strict=True``. The module owns the student encoder + EMA teacher +
  the student-only ``JepaPredictor``; the strict-load contract must cover
  every named parameter across all three.

* **TST05** — the masked-JEPA ``_step`` total is finite under bf16-cast
  inputs. Any silent ``-inf`` / NaN cascade through the front-end +
  terminal-LN + L1 path would invalidate the gradient; TST05 catches it
  before dispatch.

* **RT10** — load_state_dict at phase boundaries uses ``strict=True``;
  ``strict=False`` would silently random-init the predictor / encoder if a
  key drifted, masking the load failure with a working forward pass but
  corrupted weights. The test pins the strict-mode contract.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from neuraltrain.optimizers import LightningOptimizer

from speech_decoding.experiments.v14_joint_module import V14JointBrainModule
from speech_decoding.models.v14_encoder import V14ParcelPerceiverModel


# ---------------------------------------------------------------------------
# Module factories (shared with test_v14_joint_module.py — keeping a local
# copy so the pre-dispatch gate file is self-contained; if either factory
# drifts, the gate must re-verify the new construction surface)
# ---------------------------------------------------------------------------


def _optim_config() -> LightningOptimizer:
    return LightningOptimizer(optimizer={"name": "Adam", "lr": 1e-3})


def _make_tiny_encoder() -> V14ParcelPerceiverModel:
    return V14ParcelPerceiverModel(
        n_freq_bins=4,
        n_time_bins=8,
        k_parcels=5,
        m_sub_slots=1,
        d_model=16,
        n_heads=4,
        depth_self_attn=1,
        n_token_blocks=1,
        patch_kernel_freq=2,
        patch_kernel_time=2,
    )


def _make_module(*, phase: str = "p1") -> V14JointBrainModule:
    """Construct a tiny ``V14JointBrainModule`` for pre-dispatch gates.

    The default ``phase="p1"`` is the front-end masked-JEPA term; ``"p2"``
    exercises the predictor path. The state-dict gates use the union of
    encoder + EMA-teacher + predictor parameters either way.
    """
    return V14JointBrainModule(
        encoder=_make_tiny_encoder(),
        optim_config=_optim_config(),
        phase=phase,  # type: ignore[arg-type]
    )


def _make_synthetic_batch() -> SimpleNamespace:
    torch.manual_seed(0)
    B, C, T_bins, F_bins, K = 2, 5, 8, 4, 5
    electrode_tokens = torch.randn(B, C, T_bins, F_bins)
    # Diagonal support: all K parcels covered so the locked M4 tube default
    # (0.20 of covered, n_min_visible=3) masks 1 of 5 → the P2 bf16 gate
    # exercises a real masked predictor path, not an empty set.
    support = torch.zeros(B, C, K)
    for i in range(min(C, K)):
        support[:, i, i] = 1.0
    valid_mask = torch.ones(B, C, dtype=torch.bool)
    return SimpleNamespace(data={
        "electrode_tokens": electrode_tokens,
        "support": support,
        "valid_mask": valid_mask,
    })


# A load-bearing parameter present in every joint-module state dict — the
# front-end terminal LN (B1). Used by the strict-load drop/round-trip gates.
_LOAD_BEARING_KEY = "student.encoder.frontend_ln.weight"


# ---------------------------------------------------------------------------
# TST03 — V14JointBrainModule state-dict strict round-trip
# ---------------------------------------------------------------------------


@pytest.mark.must_pass_before_dispatch
def test_tst03_joint_module_state_dict_strict_roundtrip() -> None:
    """Save a full joint module state dict, reload it into a fresh module
    with ``strict=True``, and verify every parameter matches — covering the
    encoder, the EMA teacher mirror, and the student-only predictor."""
    src = _make_module()
    # Perturb the source so the round-trip is non-trivial.
    with torch.no_grad():
        for p in src.student.parameters():
            p.add_(0.01)
        for p in src.predictor.parameters():
            p.add_(0.01)
        src.teacher.update_from(src.student)

    state = src.state_dict()
    dst = _make_module()
    missing_unexpected = dst.load_state_dict(state, strict=True)
    assert list(missing_unexpected.missing_keys) == []
    assert list(missing_unexpected.unexpected_keys) == []

    # Spot-check: a perturbed encoder param + a predictor param + the EMA
    # teacher mirror must all match after load.
    torch.testing.assert_close(
        dst.student.encoder.frontend_ln.weight.detach(),
        src.student.encoder.frontend_ln.weight.detach(),
    )
    torch.testing.assert_close(
        dst.predictor.output_proj.weight.detach(),
        src.predictor.output_proj.weight.detach(),
    )
    torch.testing.assert_close(
        dst.teacher.model.encoder.frontend_ln.weight.detach(),
        src.teacher.model.encoder.frontend_ln.weight.detach(),
    )


@pytest.mark.must_pass_before_dispatch
def test_tst03_joint_module_strict_load_rejects_dropped_keys() -> None:
    """A state dict missing a load-bearing key must raise under
    ``strict=True``. Silent fallback to ``strict=False`` would random-init
    the missing tensor and invalidate the resume."""
    src = _make_module()
    state = src.state_dict()
    assert _LOAD_BEARING_KEY in state, "state dict layout drifted; update key"
    del state[_LOAD_BEARING_KEY]
    dst = _make_module()
    with pytest.raises(RuntimeError, match="Missing key"):
        dst.load_state_dict(state, strict=True)


@pytest.mark.must_pass_before_dispatch
def test_tst03_joint_module_strict_load_rejects_extra_keys() -> None:
    """Extra unknown keys also raise — guards against checkpoint drift
    where an upstream module renamed a parameter."""
    src = _make_module()
    state = src.state_dict()
    state["student.ghost_parameter"] = torch.zeros(4)
    dst = _make_module()
    with pytest.raises(RuntimeError, match="Unexpected key"):
        dst.load_state_dict(state, strict=True)


# ---------------------------------------------------------------------------
# TST05 — masked-JEPA loss NaN / inf detector under bf16
# ---------------------------------------------------------------------------


@pytest.mark.must_pass_before_dispatch
@pytest.mark.parametrize("phase", ["p1", "p2"])
def test_tst05_joint_step_finite_under_bf16_inputs(phase: str) -> None:
    """The masked-JEPA ``_step`` total is finite when the batch + module are
    cast to bf16 — catches a silent ``-inf`` / NaN cascade through the
    front-end, terminal LN, predictor and L1 for both phases."""
    module = _make_module(phase=phase).to(torch.bfloat16)
    batch = _make_synthetic_batch()
    cast_batch_data: dict = {}
    for key, tensor in batch.data.items():
        if isinstance(tensor, torch.Tensor) and tensor.dtype.is_floating_point:
            cast_batch_data[key] = tensor.to(torch.bfloat16)
        else:
            cast_batch_data[key] = tensor
    breakdown = module._step(cast_batch_data)
    assert torch.isfinite(breakdown.total).all(), (
        f"masked-JEPA total is non-finite under bf16 (phase={phase})"
    )


@pytest.mark.must_pass_before_dispatch
def test_tst05_joint_step_nan_input_propagates_to_total() -> None:
    """Sanity-check the detector: a NaN-seeded input MUST yield a non-finite
    total. Otherwise TST05 would silently pass on a true NaN cascade."""
    module = _make_module()
    batch = _make_synthetic_batch()
    # NaN every sample of electrode 0 so the masked set (ratio 0.5 over that
    # electrode's front-end cells) is guaranteed to gather a NaN target.
    batch.data["electrode_tokens"][:, 0, :, :] = float("nan")
    breakdown = module._step(batch.data)
    assert not torch.isfinite(breakdown.total).all(), (
        "NaN-seeded input must propagate to total; the gate cannot pass "
        "silently when the cascade is real"
    )


# ---------------------------------------------------------------------------
# RT10 — phase-boundary checkpoint strict=True is the default
# ---------------------------------------------------------------------------


@pytest.mark.must_pass_before_dispatch
def test_rt10_phase_boundary_load_strict_true_is_default() -> None:
    """The phase-boundary load path MUST default to ``strict=True`` so a key
    drift between phases (e.g. a P1 checkpoint carrying a key the P3 module
    does not own) raises instead of silently random-initialising."""
    src = _make_module()
    src_state = src.state_dict()
    src_state["student.fake_phase3_head.weight"] = torch.zeros(4)
    dst = _make_module()
    with pytest.raises(RuntimeError, match="Unexpected key"):
        dst.load_state_dict(src_state)  # strict=True is the default


@pytest.mark.must_pass_before_dispatch
def test_rt10_phase_boundary_strict_false_silently_drops_keys() -> None:
    """Negative control: prove the failure mode RT10 prevents is real. Under
    ``strict=False`` the missing-key path silently leaves the destination's
    parameter at its random init — exactly the pre-dispatch failure mode."""
    src = _make_module()
    src_state = src.state_dict()
    del src_state[_LOAD_BEARING_KEY]
    dst = _make_module()
    dst_pre_load = deepcopy(dst.student.encoder.frontend_ln.weight.detach())
    dst.load_state_dict(src_state, strict=False)
    dst_post_load = dst.student.encoder.frontend_ln.weight.detach()
    # strict=False leaves the dropped param at its random init.
    torch.testing.assert_close(dst_post_load, dst_pre_load)
