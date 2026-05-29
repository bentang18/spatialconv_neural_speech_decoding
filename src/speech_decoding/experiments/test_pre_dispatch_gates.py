"""Pre-dispatch gates — TST03 / TST05 + RT10.

Spec: ``docs/neuroprobe/v14_blockers.md`` §"Pre-Phase-1 unit tests (TST)"
and §"Silent-failure traps (RT)". All tests in this file are marked
``@pytest.mark.must_pass_before_dispatch`` and run by the
``scripts/dcc/dispatch`` pre-flight gate.

The tests below cover:

* **TST03** — V14JointBrainModule state-dict round-trip with
  ``strict=True``. After B29 collapsed P1 + P2 into a single joint
  phase, the "P1 ↔ P2 checkpoint strict-mode" question becomes "joint
  ↔ joint (resume) checkpoint strict-mode" — verifying ``strict=True``
  is the default at the phase-boundary load is the load-bearing piece.

* **TST05** — joint loss is finite under bf16-cast inputs. The
  aggregator chains L1 + masked-MSE + cross-modal sites; any silent
  ``-inf`` / NaN cascade through bf16's narrower exponent would
  invalidate the gradient. TST05 catches the bf16 cascade before
  dispatch.

* **RT10** — load_state_dict at phase boundaries uses ``strict=True``;
  ``strict=False`` would silently random-init the predictor / LN heads
  if a key drifted, masking the load failure with a working forward
  pass but corrupted weights. The test pins the strict-mode contract by
  forcing a key-drift case to raise.
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
        k_parcels=2,
        m_sub_slots=1,
        d_model=16,
        n_heads=4,
        depth_self_attn=1,
        n_token_blocks=1,
        patch_kernel_freq=2,
        patch_kernel_time=2,
    )


def _make_module() -> V14JointBrainModule:
    return V14JointBrainModule(
        encoder=_make_tiny_encoder(),
        optim_config=_optim_config(),
        pma_n_heads=4,
    )


def _make_synthetic_batch() -> SimpleNamespace:
    torch.manual_seed(0)
    B, C, T_bins, F_bins, K = 2, 3, 8, 4, 2
    electrode_tokens = torch.randn(B, C, T_bins, F_bins)
    support = torch.zeros(B, C, K)
    support[:, 0, 0] = 1.0
    support[:, 1, 1] = 1.0
    support[:, 2, 0] = 0.5
    valid_mask = torch.ones(B, C, dtype=torch.bool)
    return SimpleNamespace(data={
        "electrode_tokens": electrode_tokens,
        "support": support,
        "valid_mask": valid_mask,
    })


# ---------------------------------------------------------------------------
# TST03 — V14JointBrainModule state-dict strict round-trip
# ---------------------------------------------------------------------------


@pytest.mark.must_pass_before_dispatch
def test_tst03_joint_module_state_dict_strict_roundtrip() -> None:
    """Save a full joint module state dict, reload it into a fresh
    module with ``strict=True``, and verify every parameter matches."""
    src = _make_module()
    # Perturb the source so the round-trip is non-trivial.
    with torch.no_grad():
        for p in src.student.parameters():
            p.add_(0.01)
        src.teacher.update_from(src.student)

    state = src.state_dict()
    dst = _make_module()
    missing_unexpected = dst.load_state_dict(state, strict=True)
    # Lightning's load_state_dict returns IncompatibleKeys; on a
    # strict-True success both lists are empty.
    assert list(missing_unexpected.missing_keys) == []
    assert list(missing_unexpected.unexpected_keys) == []

    # Spot-check: a perturbed student parameter must match after load.
    src_lnmid = src.student.ln_mid.weight.detach()
    dst_lnmid = dst.student.ln_mid.weight.detach()
    torch.testing.assert_close(dst_lnmid, src_lnmid)
    src_teacher = src.teacher.model.ln_mid.weight.detach()
    dst_teacher = dst.teacher.model.ln_mid.weight.detach()
    torch.testing.assert_close(dst_teacher, src_teacher)


@pytest.mark.must_pass_before_dispatch
def test_tst03_joint_module_strict_load_rejects_dropped_keys() -> None:
    """The pre-dispatch contract: a state dict missing keys against
    the target module must raise under ``strict=True``. Silent fallback
    to ``strict=False`` would random-init the missing tensors and
    invalidate the resume."""
    src = _make_module()
    state = src.state_dict()
    # Drop a load-bearing key (LN_mid weight) and confirm strict load
    # rejects.
    dropped_key = "student.ln_mid.weight"
    assert dropped_key in state, "state dict layout drifted; update key"
    del state[dropped_key]
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
# TST05 — Joint loss NaN / inf detector under bf16
# ---------------------------------------------------------------------------


@pytest.mark.must_pass_before_dispatch
def test_tst05_joint_step_finite_under_bf16_inputs() -> None:
    """The joint module's ``_step`` returns finite per-term + total
    losses when the batch is cast to bf16. Catches silent ``-inf`` /
    NaN cascades through the LN heads + L1 + PMA softmax."""
    module = _make_module()
    batch = _make_synthetic_batch()
    # Cast every float tensor in the batch to bf16 — the encoder is
    # tiny enough to run at bf16 end-to-end on CPU.
    cast_batch_data: dict = {}
    for key, tensor in batch.data.items():
        if isinstance(tensor, torch.Tensor) and tensor.dtype.is_floating_point:
            cast_batch_data[key] = tensor.to(torch.bfloat16)
        else:
            cast_batch_data[key] = tensor
    # The student bundle's parameters are still fp32 — cast them too so
    # the matmul stays homogeneous.
    module = module.to(torch.bfloat16)
    breakdown = module._step(cast_batch_data)
    for label, term in (
        ("l_pre_frame", breakdown.l_pre_frame),
        ("l_mid_slot", breakdown.l_mid_slot),
        ("l_post_frame", breakdown.l_post_frame),
        ("l_post_utterance", breakdown.l_post_utterance),
        ("total", breakdown.total),
    ):
        assert torch.isfinite(term).all(), f"{label} is non-finite under bf16"


@pytest.mark.must_pass_before_dispatch
def test_tst05_joint_step_nan_input_propagates_to_total() -> None:
    """Sanity check the detector itself: a NaN-seeded input MUST yield
    a non-finite total. Otherwise TST05 would silently pass on a true
    NaN cascade because the aggregator absorbed the NaN."""
    module = _make_module()
    batch = _make_synthetic_batch()
    batch.data["electrode_tokens"][0, 0, 0, 0] = float("nan")
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
    """The phase-boundary load path MUST default to ``strict=True`` so
    a key drift between phases (e.g. P1 saves predictor weights;
    Phase-3 module has no predictor module to receive them) raises
    instead of silently random-initialising."""
    src = _make_module()
    src_state = src.state_dict()
    # Inject a key that the destination module DOES NOT own. Under
    # strict=False this would be silently dropped; strict=True must
    # raise. Default is strict=True.
    src_state["student.fake_phase3_head.weight"] = torch.zeros(4)
    dst = _make_module()
    with pytest.raises(RuntimeError, match="Unexpected key"):
        dst.load_state_dict(src_state)  # strict=True is the default


@pytest.mark.must_pass_before_dispatch
def test_rt10_phase_boundary_strict_false_silently_drops_keys() -> None:
    """Negative control: prove the failure mode RT10 prevents is real.
    Under ``strict=False`` the missing-key path silently random-inits
    the destination's parameter, which is exactly the pre-dispatch
    failure mode."""
    src = _make_module()
    src_state = src.state_dict()
    dropped_key = "student.ln_mid.weight"
    del src_state[dropped_key]
    dst = _make_module()
    # The dst module's ln_mid.weight stays at its random init; load
    # under strict=False does NOT raise (this is what the gate forbids
    # in practice).
    dst_pre_load = deepcopy(dst.student.ln_mid.weight.detach())
    dst.load_state_dict(src_state, strict=False)
    dst_post_load = dst.student.ln_mid.weight.detach()
    # strict=False leaves ln_mid at its random init — proving the
    # silent-random-init failure mode RT10 prevents is real.
    torch.testing.assert_close(dst_post_load, dst_pre_load)
