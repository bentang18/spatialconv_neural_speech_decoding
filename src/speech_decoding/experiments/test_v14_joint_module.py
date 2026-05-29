"""B2.2 + B31 tests for :class:`V14JointBrainModule`.

Covers:

* Construction from a tiny encoder; EMA teacher is a frozen mirror.
* B31 default (``loss_variant="b31_default"``): student bundle ships
  ``ln_frame`` only; ``ln_mid`` / ``ln_utt`` / ``pma`` are ``None``;
  ``_step`` returns a 2-term breakdown.
* B31 sister (``loss_variant="b31_plus_both"``): student bundle ships
  every head; ``_step`` returns a 4-term breakdown matching the legacy
  B19/B22/B28 shape.
* B30 ``latent_valid`` flows from ``support`` → every active slot/
  utterance term.
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


def _make_module(
    encoder=None, *, loss_variant: str = "b31_default",
) -> V14JointBrainModule:
    if encoder is None:
        encoder = _make_tiny_encoder()
    optim_config = _optim_config()
    return V14JointBrainModule(
        encoder=encoder,
        optim_config=optim_config,
        pma_n_heads=4,
        loss_variant=loss_variant,  # type: ignore[arg-type]
    )


def test_v14_joint_brain_module_constructs_with_frozen_teacher() -> None:
    module = _make_module()
    # Teacher params: every one must be ``requires_grad=False``.
    for p in module.teacher.parameters():
        assert p.requires_grad is False
    # Student bundle exists; under the B31 2-term default it carries
    # ``ln_frame`` only and the dropped heads are ``None``.
    assert isinstance(module.student, _V14StudentBundle)
    assert module.student.ln_frame is not None
    assert module.student.ln_mid is None
    assert module.student.ln_utt is None
    assert module.student.pma is None


def test_v14_joint_brain_module_b31_default_omits_dropped_heads() -> None:
    """B31 5/28 PM-late: ``loss_variant="b31_default"`` constructs the
    student bundle WITHOUT ``ln_mid`` / ``ln_utt`` / PMA — the PMA query
    receives no gradient in the joint SSL phase (P3 distillation is the
    first place it sees one)."""
    module = _make_module(loss_variant="b31_default")
    assert module.student.ln_frame is not None
    assert module.student.ln_mid is None
    assert module.student.ln_utt is None
    assert module.student.pma is None
    # Teacher mirror also omits these (it's an EMA-deepcopy of the student).
    assert module.teacher.model.ln_mid is None
    assert module.teacher.model.ln_utt is None
    assert module.teacher.model.pma is None


def test_v14_joint_brain_module_b31_plus_both_builds_full_head_set() -> None:
    """``R-add-both`` sister: every dropped head is reconstructed on
    both student and teacher."""
    module = _make_module(loss_variant="b31_plus_both")
    assert module.student.ln_frame is not None
    assert module.student.ln_mid is not None
    assert module.student.ln_utt is not None
    assert module.student.pma is not None
    assert module.teacher.model.ln_mid is not None
    assert module.teacher.model.ln_utt is not None
    assert module.teacher.model.pma is not None


def test_v14_joint_brain_module_b31_plus_m3_builds_only_mid_head() -> None:
    """``R-add-m3-loss`` sister: only the M3 head reconstructs;
    utterance + PMA stay ``None``."""
    module = _make_module(loss_variant="b31_plus_m3")
    assert module.student.ln_mid is not None
    assert module.student.ln_utt is None
    assert module.student.pma is None


def test_v14_joint_brain_module_b31_plus_utt_builds_only_utterance_heads() -> None:
    """``R-add-utterance-loss`` sister (EAT-faithful comparator): only
    LN_utt + PMA reconstruct; M3 head stays ``None``."""
    module = _make_module(loss_variant="b31_plus_utt")
    assert module.student.ln_mid is None
    assert module.student.ln_utt is not None
    assert module.student.pma is not None


def test_v14_joint_brain_module_rejects_unknown_loss_variant() -> None:
    """Invalid ``loss_variant`` is caught at construction."""
    optim_config = _optim_config()
    with pytest.raises(ValueError, match="loss_variant"):
        V14JointBrainModule(
            encoder=_make_tiny_encoder(),
            optim_config=optim_config,
            loss_variant="bogus",  # type: ignore[arg-type]
        )


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


def test_v14_joint_brain_module_b31_default_step_returns_two_term_breakdown() -> None:
    """B31 default: ``_step`` returns a breakdown with ``L_pre_frame +
    L_post_frame`` only; the dropped term fields are ``None``."""
    module = _make_module(loss_variant="b31_default")
    batch = _make_synthetic_batch()
    breakdown = module._step(batch.data)

    for term in (breakdown.l_pre_frame, breakdown.l_post_frame):
        assert isinstance(term, torch.Tensor)
        assert term.ndim == 0
        assert torch.isfinite(term)
    # B31 dropped terms are None under the default.
    assert breakdown.l_mid_slot is None
    assert breakdown.l_post_utterance is None
    # DKoleo + reactive cousins also None.
    assert breakdown.l_dkoleo_m4 is None
    assert breakdown.l_dkoleo_m3_reactive is None
    assert breakdown.l_gram_reactive is None

    # Total = sum of the 2 active terms (unit coefficients).
    expected = breakdown.l_pre_frame + breakdown.l_post_frame
    torch.testing.assert_close(breakdown.total, expected)


def test_v14_joint_brain_module_b31_plus_both_step_returns_four_term_breakdown() -> None:
    """``R-add-both`` sister: ``_step`` returns the full 4-term legacy
    breakdown (B19/B22/B28 shape)."""
    module = _make_module(loss_variant="b31_plus_both")
    batch = _make_synthetic_batch()
    breakdown = module._step(batch.data)

    for term in (
        breakdown.l_pre_frame,
        breakdown.l_mid_slot,
        breakdown.l_post_frame,
        breakdown.l_post_utterance,
    ):
        assert isinstance(term, torch.Tensor)
        assert term.ndim == 0
        assert torch.isfinite(term)

    assert breakdown.l_dkoleo_m4 is None
    assert breakdown.l_dkoleo_m3_reactive is None
    assert breakdown.l_gram_reactive is None

    expected = (
        breakdown.l_pre_frame + breakdown.l_mid_slot
        + breakdown.l_post_frame + breakdown.l_post_utterance
    )
    torch.testing.assert_close(breakdown.total, expected)


def test_v14_joint_brain_module_total_is_non_negative_and_finite() -> None:
    module = _make_module()
    batch = _make_synthetic_batch()
    breakdown = module._step(batch.data)
    # L1 (default) is non-negative.
    assert float(breakdown.total.item()) >= 0.0
    assert torch.isfinite(breakdown.total)


def test_v14_joint_brain_module_b31_default_total_has_grad_through_ln_frame() -> None:
    """B31 default: backward must populate ``ln_frame`` (the only LN
    head in the bundle) and the encoder."""
    module = _make_module(loss_variant="b31_default")
    batch = _make_synthetic_batch()
    breakdown = module._step(batch.data)
    breakdown.total.backward()
    assert module.student.ln_frame.weight.grad is not None
    assert torch.isfinite(module.student.ln_frame.weight.grad).all()


def test_v14_joint_brain_module_b31_plus_both_total_has_grad_through_all_heads() -> None:
    """``R-add-both`` sister: backward must populate every reconstructed
    head (``ln_mid``, ``ln_utt``, PMA query)."""
    module = _make_module(loss_variant="b31_plus_both")
    batch = _make_synthetic_batch()
    breakdown = module._step(batch.data)
    breakdown.total.backward()
    assert module.student.ln_mid.weight.grad is not None
    assert module.student.ln_utt.weight.grad is not None
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
    parameters toward the (post-step) student parameters. Exercised on
    ``ln_frame`` so the test runs under the B31 default head set."""
    module = _make_module(loss_variant="b31_default")
    with torch.no_grad():
        module.student.ln_frame.weight.fill_(2.0)
    pre = module.teacher.model.ln_frame.weight.detach().clone()
    coeff = module.teacher.update_from(module.student)
    post = module.teacher.model.ln_frame.weight.detach().clone()
    assert coeff == pytest.approx(0.999)
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
    a finite breakdown (the encoder forward accepts the kwarg)."""
    module = _make_module()
    batch = _make_synthetic_batch()
    B, C = batch.data["electrode_tokens"].shape[:2]
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
    monitor returns nan. We perturb the student ``ln_frame`` to inject a
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


# ---------------------------------------------------------------------------
# 5/28 P0 monitors: MON-PARCEL-COVERAGE-VARIANCE, MON-TEACHER-FEATURE-RANK,
# MON-GRAD-SPIKE-DIVERGENCE — wired into V14JointBrainModule.
# ---------------------------------------------------------------------------


def test_v14_joint_brain_module_logs_parcel_coverage_on_every_step() -> None:
    """MON-PARCEL-COVERAGE-VARIANCE has no forward dependency — must fire
    on every train/val/test step regardless of shaft_mask."""
    module = _make_module()
    batch = _make_synthetic_batch()

    logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: logged.update({key: float(value)})  # type: ignore[method-assign]
    module._monitor_from_step(batch.data, step_name="train")
    assert "train_mon_coverage_active_mean" in logged
    assert "train_mon_coverage_active_cv" in logged
    assert "train_mon_coverage_slot_var" in logged
    assert "train_mon_coverage_degenerate_frac" in logged
    assert "train_mon_coverage_swec_frac" in logged
    assert "train_mon_coverage_alarm" in logged
    assert logged["train_mon_coverage_alarm"] in (0.0, 1.0)


def test_v14_joint_brain_module_logs_teacher_rank_on_val_only() -> None:
    """MON-TEACHER-FEATURE-RANK is the expensive SVD probe; wire only on
    val/test steps, not on every train step."""
    module = _make_module()
    batch = _make_synthetic_batch()

    train_logged: dict[str, float] = {}
    module.log = (  # type: ignore[method-assign]
        lambda key, value, **_kw: train_logged.update({key: float(value)})
    )
    module._monitor_from_step(batch.data, step_name="train")
    assert "train_mon_rankme" not in train_logged

    val_logged: dict[str, float] = {}
    module.log = (  # type: ignore[method-assign]
        lambda key, value, **_kw: val_logged.update({key: float(value)})
    )
    module._monitor_from_step(batch.data, step_name="val")
    assert "val_mon_rankme" in val_logged
    assert "val_mon_rankme_normalised" in val_logged
    assert "val_mon_rankme_warn" in val_logged
    assert "val_mon_rankme_alarm" in val_logged


def _perturb_student_for_nonzero_loss(module: V14JointBrainModule) -> None:
    """Perturb the student so the deepcopy-identical EMA teacher no
    longer matches; otherwise all L1 terms are 0 → zero grads → the
    grad-spike hook can't be exercised. ``ln_frame`` is always present
    on the student bundle (B31 2-term default + every sister)."""
    with torch.no_grad():
        module.student.ln_frame.weight.add_(0.3)


def test_v14_joint_brain_module_on_before_optimizer_step_logs_grad_spike() -> None:
    """MON-GRAD-SPIKE-DIVERGENCE fires from the Lightning hook with the
    student grads populated. On the first call (EMA buffer = 0) the
    spike flag is False and the EMA is seeded to the current grad
    norm."""
    module = _make_module()
    _perturb_student_for_nonzero_loss(module)
    batch = _make_synthetic_batch()
    breakdown = module._step(batch.data)
    breakdown.total.backward()

    logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: logged.update({key: float(value)})  # type: ignore[method-assign]
    module.on_before_optimizer_step(optimizer=None)
    assert "train_mon_grad_l2" in logged
    assert "train_mon_grad_ema_l2" in logged
    assert "train_mon_grad_spike_ratio" in logged
    assert "train_mon_grad_spike" in logged
    assert "train_mon_grad_diverged" in logged
    # First step: EMA seeded from current grad (was zero before).
    assert logged["train_mon_grad_ema_l2"] == pytest.approx(0.0)
    assert logged["train_mon_grad_spike"] == 0.0
    assert logged["train_mon_grad_diverged"] == 0.0
    assert logged["train_mon_grad_l2"] > 0.0
    # And the persistent buffer should now hold the seeded value.
    assert float(module._grad_ema_l2.item()) > 0.0


def test_v14_joint_brain_module_grad_ema_buffer_persists_across_calls() -> None:
    """The EMA buffer must persist across hook calls (so the next step
    has a baseline to spike against). Run two steps and verify the
    second sees a non-zero prior EMA."""
    module = _make_module()
    _perturb_student_for_nonzero_loss(module)
    batch = _make_synthetic_batch()

    breakdown = module._step(batch.data)
    breakdown.total.backward()
    module.on_before_optimizer_step(optimizer=None)
    first_ema = float(module._grad_ema_l2.item())
    assert first_ema > 0.0

    # Second backward — fresh grads, but the EMA buffer survives.
    for p in module.student.parameters():
        if p.grad is not None:
            p.grad.zero_()
    breakdown2 = module._step(batch.data)
    breakdown2.total.backward()

    logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: logged.update({key: float(value)})  # type: ignore[method-assign]
    module.on_before_optimizer_step(optimizer=None)
    # Step 2 sees the seeded EMA as its baseline.
    assert logged["train_mon_grad_ema_l2"] == pytest.approx(first_ema)
    assert logged["train_mon_grad_spike_ratio"] > 0.0
