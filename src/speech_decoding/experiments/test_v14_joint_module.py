"""B2.2 tests for :class:`V14JointBrainModule`.

Covers:

* Construction from a tiny encoder; EMA teacher is a frozen mirror.
* End-to-end forward + 4-term L_total composition on a synthetic batch.
* B30 ``latent_valid`` flows from ``support`` → all 3 slot/utterance
  terms.
* B26 EMA step τ=0.999 fixed; teacher params trail the student.
* Predictor fallback path: ``L_pre_frame = F.l1_loss(M2_student,
  detach(M2_teacher))`` when ``predictor is None``.
* B30 sister-flag runtime gates raise ``NotImplementedError`` at
  construction.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from neuraltrain.optimizers import LightningOptimizer

from speech_decoding.experiments.v14_joint_module import (
    V14JointBrainModule,
    _V14StudentBundle,
)
from speech_decoding.models.v14_encoder import V14ParcelPerceiverModel


def _optim_config() -> LightningOptimizer:
    return LightningOptimizer(optimizer={"name": "Adam", "lr": 1e-3})


def _make_tiny_encoder(
    *,
    n_freq_bins: int = 4,
    n_time_bins: int = 8,
    k_parcels: int = 2,
    m_sub_slots: int = 1,
    d_model: int = 16,
    n_heads: int = 4,
    depth_self_attn: int = 1,
    n_token_blocks: int = 1,
    patch_kernel_freq: int = 2,
    patch_kernel_time: int = 2,
    cross_attn_positions=None,
) -> V14ParcelPerceiverModel:
    return V14ParcelPerceiverModel(
        n_freq_bins=n_freq_bins,
        n_time_bins=n_time_bins,
        k_parcels=k_parcels,
        m_sub_slots=m_sub_slots,
        d_model=d_model,
        n_heads=n_heads,
        depth_self_attn=depth_self_attn,
        n_token_blocks=n_token_blocks,
        patch_kernel_freq=patch_kernel_freq,
        patch_kernel_time=patch_kernel_time,
        cross_attn_positions=cross_attn_positions,
    )


def _make_synthetic_batch(
    *,
    B: int = 2,
    C: int = 3,
    T_bins: int = 8,
    F_bins: int = 4,
    K: int = 2,
) -> SimpleNamespace:
    torch.manual_seed(0)
    electrode_tokens = torch.randn(B, C, T_bins, F_bins)
    # Make support have ≥1 covered electrode per parcel for each clip so
    # latent_valid is non-empty everywhere (full-active case).
    support = torch.zeros(B, C, K)
    support[:, 0, 0] = 1.0
    support[:, 1, 1] = 1.0
    support[:, 2, 0] = 0.5
    valid_mask = torch.ones(B, C, dtype=torch.bool)
    data = {
        "electrode_tokens": electrode_tokens,
        "support": support,
        "valid_mask": valid_mask,
    }
    return SimpleNamespace(data=data)


def _make_module(encoder=None) -> V14JointBrainModule:
    if encoder is None:
        encoder = _make_tiny_encoder()
    optim_config = _optim_config()
    return V14JointBrainModule(
        encoder=encoder,
        optim_config=optim_config,
        pma_n_heads=4,
    )


def test_v14_joint_brain_module_constructs_with_frozen_teacher() -> None:
    module = _make_module()
    # Teacher params: every one must be ``requires_grad=False``.
    for p in module.teacher.parameters():
        assert p.requires_grad is False
    # Student bundle has the encoder + 3 LNs + PMA.
    assert isinstance(module.student, _V14StudentBundle)
    assert hasattr(module.student, "ln_mid")
    assert hasattr(module.student, "ln_frame")
    assert hasattr(module.student, "ln_utt")
    assert hasattr(module.student, "pma")


def test_v14_joint_brain_module_rejects_b30_sister_latent_valid_override() -> None:
    optim_config = _optim_config()
    with pytest.raises(NotImplementedError, match="B30"):
        V14JointBrainModule(
            encoder=_make_tiny_encoder(),
            optim_config=optim_config,
            latent_valid_override="all_true",
        )


def test_v14_joint_brain_module_rejects_b30_sister_sa_mask_mode() -> None:
    optim_config = _optim_config()
    with pytest.raises(NotImplementedError, match="key-only"):
        V14JointBrainModule(
            encoder=_make_tiny_encoder(),
            optim_config=optim_config,
            sa_mask_mode="key_only",
        )


def test_v14_joint_brain_module_step_returns_4_term_breakdown() -> None:
    module = _make_module()
    batch = _make_synthetic_batch()
    breakdown = module._step(batch.data)

    # All 4 SSL terms are scalar tensors.
    for term in (
        breakdown.l_pre_frame,
        breakdown.l_mid_slot,
        breakdown.l_post_frame,
        breakdown.l_post_utterance,
    ):
        assert isinstance(term, torch.Tensor)
        assert term.ndim == 0
        assert torch.isfinite(term)

    # B28 4-term default: DKoleo + Gram + reactive arms all None.
    assert breakdown.l_dkoleo_m4 is None
    assert breakdown.l_dkoleo_m3_reactive is None
    assert breakdown.l_gram_reactive is None

    # Total = sum of 4 terms (unit coefficients).
    expected = (
        breakdown.l_pre_frame + breakdown.l_mid_slot
        + breakdown.l_post_frame + breakdown.l_post_utterance
    )
    torch.testing.assert_close(breakdown.total, expected)


def test_v14_joint_brain_module_total_is_non_negative_and_finite() -> None:
    module = _make_module()
    batch = _make_synthetic_batch()
    breakdown = module._step(batch.data)
    # L1 + masked-MSE + utterance terms are all non-negative.
    assert float(breakdown.total.item()) >= 0.0
    assert torch.isfinite(breakdown.total)


def test_v14_joint_brain_module_total_has_grad_through_student() -> None:
    """The aggregator's total must back-prop into every student
    parameter (encoder + LN heads + PMA query)."""
    module = _make_module()
    batch = _make_synthetic_batch()
    breakdown = module._step(batch.data)
    breakdown.total.backward()
    # At least the LN_mid weight and the PMA query must have grads.
    assert module.student.ln_mid.weight.grad is not None
    assert module.student.pma.query.grad is not None


def test_v14_joint_brain_module_predictor_fallback_l1_form() -> None:
    """Predictor-None branch: L_pre_frame = F.l1_loss(M2_s, detach(M2_t))."""
    module = _make_module()
    batch = _make_synthetic_batch()
    breakdown = module._step(batch.data)

    # Recompute the fallback term independently from the encoder taps.
    student_kwargs = {
        "electrode_tokens": batch.data["electrode_tokens"],
        "support": batch.data["support"],
        "valid_mask": batch.data["valid_mask"],
    }
    student_taps = module.student(**student_kwargs)
    with torch.no_grad():
        teacher_taps = module.teacher.model(**student_kwargs)
    expected_l_pre = torch.nn.functional.l1_loss(
        student_taps["M2"], teacher_taps["M2"].detach(),
    )
    torch.testing.assert_close(
        breakdown.l_pre_frame, expected_l_pre, atol=1e-4, rtol=1e-4,
    )


def test_v14_joint_brain_module_ema_step_updates_teacher() -> None:
    """B26 lock: EMA τ=0.999 fixed; ``update_from`` brings teacher
    parameters toward the (post-step) student parameters."""
    module = _make_module()
    # Tweak a student LN weight then EMA-update — teacher should track.
    with torch.no_grad():
        module.student.ln_mid.weight.fill_(2.0)
    pre = module.teacher.model.ln_mid.weight.detach().clone()
    coeff = module.teacher.update_from(module.student)
    post = module.teacher.model.ln_mid.weight.detach().clone()
    # τ=0.999 means the teacher takes 1 - τ = 1e-3 of the student.
    assert coeff == pytest.approx(0.999)
    # The post value should equal τ * pre + (1-τ) * student.
    expected = 0.999 * pre + 0.001 * 2.0
    torch.testing.assert_close(post, expected)


def test_v14_joint_brain_module_teacher_uses_full_input_via_no_grad() -> None:
    """B26 contract: teacher forward runs under no_grad, so teacher
    parameters never accumulate grads from the SSL backward."""
    module = _make_module()
    batch = _make_synthetic_batch()
    breakdown = module._step(batch.data)
    breakdown.total.backward()
    for p in module.teacher.parameters():
        assert p.grad is None


def test_v14_joint_brain_module_routes_shaft_mask_student_only() -> None:
    """B03 + B26 contract: ``shaft_mask`` is included in the student
    encoder kwargs but is NOT forwarded to the EMA teacher (whose forward
    must see the full unmasked input). The kwarg split is enforced by
    :meth:`_extract_student_kwargs`."""
    module = _make_module()
    batch = _make_synthetic_batch()
    # Mark every electrode as shaft-blocked → student forward sees mask;
    # teacher forward sees no shaft input.
    B, C = batch.data["electrode_tokens"].shape[:2]
    batch.data["shaft_mask"] = torch.zeros(B, C, dtype=torch.bool)
    batch.data["shaft_mask"][:, 0] = True

    student_kwargs, shaft_mask = module._extract_student_kwargs(batch.data)
    assert "shaft_mask" not in student_kwargs, (
        "shaft_mask must be carried separately from student_kwargs so the "
        "teacher forward (which reuses student_kwargs) does NOT see it"
    )
    assert shaft_mask is not None
    assert shaft_mask.shape == (B, C)
    assert shaft_mask.dtype == torch.bool
    assert bool(shaft_mask[:, 0].all().item()), (
        "shaft_mask payload must round-trip through _extract_student_kwargs"
    )


def test_v14_joint_brain_module_step_accepts_shaft_mask_in_batch() -> None:
    """End-to-end: with ``shaft_mask`` present, ``_step`` still returns
    a finite 4-term breakdown (the encoder forward accepts the kwarg)."""
    module = _make_module()
    batch = _make_synthetic_batch()
    B, C = batch.data["electrode_tokens"].shape[:2]
    # Sparse, valid-shape mask (one channel masked per clip).
    sm = torch.zeros(B, C, dtype=torch.bool)
    sm[:, 0] = True
    batch.data["shaft_mask"] = sm
    breakdown = module._step(batch.data)
    assert torch.isfinite(breakdown.total)


def test_v14_joint_brain_module_monitor_skips_when_shaft_mask_absent() -> None:
    """MON-MASK-002 is a no-op when ``shaft_mask`` is not in the batch
    (e.g. supervised-phase smoke test or any path without joint masking)."""
    module = _make_module()
    batch = _make_synthetic_batch()
    assert "shaft_mask" not in batch.data
    # Spy on log calls so we can confirm the monitor key never fires.
    logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: logged.update({key: float(value)})  # type: ignore[method-assign]
    module._monitor_from_step(batch.data, step_name="val")
    assert "val_mon_mask_002_ratio" not in logged
    assert "val_mon_mask_002_in_band" not in logged


def test_v14_joint_brain_module_monitor_logs_when_shaft_mask_orphans_some_parcel() -> None:
    """MON-MASK-002 fires when ``shaft_mask`` drives at least one parcel
    to lose every electrode in every clip. The verdict's ratio is finite
    and ``in_band`` is reported.

    The teacher is constructed as a deepcopy of the student so at
    init their M4 taps are identical → both MSE values are zero and the
    monitor returns nan. We perturb the student LN_frame to inject a
    non-zero student/teacher divergence so the ratio is well-defined.
    """
    module = _make_module()
    batch = _make_synthetic_batch()
    B, C = batch.data["electrode_tokens"].shape[:2]
    sm = torch.zeros(B, C, dtype=torch.bool)
    sm[:, 0] = True
    sm[:, 2] = True
    batch.data["shaft_mask"] = sm

    with torch.no_grad():
        module.student.ln_frame.weight.add_(0.5)

    logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: logged.update({key: float(value)})  # type: ignore[method-assign]
    module._monitor_from_step(batch.data, step_name="val")
    assert "val_mon_mask_002_ratio" in logged
    assert "val_mon_mask_002_in_band" in logged
    assert logged["val_mon_mask_002_in_band"] in (0.0, 1.0)
