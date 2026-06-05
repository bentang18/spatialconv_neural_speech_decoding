"""B36 WS-B tests for :class:`V14JointBrainModule` (masked-JEPA SSL).

Covers the module-level masked-JEPA contract (the encoder-level B1/B2/B5/B8
unit tests live in ``models/test_v14_encoder.py``):

* Construction: EMA teacher is a frozen mirror; the student bundle is
  encoder-only (no ``ln_frame`` / ``ln_mid`` / ``ln_utt`` / PMA heads); a
  student-only :class:`JepaPredictor` is built and is NOT EMA-mirrored.
* B30 sister-flag runtime gates + invalid ``phase`` raise at construction.
* P1 (``phase="p1"``, paradigm A): ``_step`` returns a single-term
  ``MaskedJepaBreakdown(phase="p1")``; gradient reaches the front-end
  (``frontend_ln``) but NOT the terminal ``encoder_ln`` or the predictor.
* P2 (``phase="p2"``, paradigm B): single-term ``MaskedJepaBreakdown(
  phase="p2")``; gradient reaches both the encoder and the predictor.
* B6: empty mask → exact-0 total (no NaN); target is detached (teacher
  accumulates no grad); the loss is L1, not MSE.
* B7: the EMA teacher always encodes the FULL input — the guard fires if a
  False-containing visibility mask reaches it.
* B9: exactly ONE active loss term per phase; the retired multi-term
  aggregator helpers are not imported by the module.
* B26 EMA step τ=0.99925 fixed.
* 5/28 P0 monitors (coverage / RankMe / grad-spike) still wired.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from neuraltrain.optimizers import LightningOptimizer

from speech_decoding.experiments.v14_joint_module import (
    V14JointBrainModule,
    _V14StudentBundle,
)
from speech_decoding.models.v14_encoder import JepaPredictor, V14ParcelPerceiverModel
from speech_decoding.ssl.masked_jepa import MaskedJepaBreakdown


def _optim_config() -> LightningOptimizer:
    return LightningOptimizer(optimizer={"name": "Adam", "lr": 1e-3})


def _make_tiny_encoder(
    *,
    n_freq_bins: int = 4,
    n_time_bins: int = 8,
    k_parcels: int = 5,
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
    C: int = 5,
    T_bins: int = 8,
    F_bins: int = 4,
    K: int = 5,
) -> SimpleNamespace:
    torch.manual_seed(0)
    electrode_tokens = torch.randn(B, C, T_bins, F_bins)
    # One covered electrode per parcel (diagonal support) so all K parcels are
    # covered → latent_valid is non-empty everywhere AND the locked M4 tube
    # default (0.20 of covered, n_min_visible=3) masks exactly 1 of 5 parcels
    # while keeping ≥3 visible — exercising the real default in P2.
    support = torch.zeros(B, C, K)
    for i in range(min(C, K)):
        support[:, i, i] = 1.0
    valid_mask = torch.ones(B, C, dtype=torch.bool)
    data = {
        "electrode_tokens": electrode_tokens,
        "support": support,
        "valid_mask": valid_mask,
    }
    return SimpleNamespace(data=data)


def _make_module(
    encoder=None, *, phase: str = "p1", **kwargs,
) -> V14JointBrainModule:
    if encoder is None:
        encoder = _make_tiny_encoder()
    return V14JointBrainModule(
        encoder=encoder,
        optim_config=_optim_config(),
        phase=phase,  # type: ignore[arg-type]
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construct_frozen_teacher_and_encoder_only_student() -> None:
    module = _make_module()
    for p in module.teacher.parameters():
        assert p.requires_grad is False
    # B6/B36 §4: the student bundle is encoder-only — no LN/PMA heads.
    assert isinstance(module.student, _V14StudentBundle)
    assert hasattr(module.student, "encoder")
    for dead in ("ln_frame", "ln_mid", "ln_utt", "pma"):
        assert not hasattr(module.student, dead), dead


def test_predictor_is_jepa_predictor_and_not_ema_mirrored() -> None:
    """The predictor is student-only — V-JEPA predictors are never part of
    the teacher. The EMA mirror deepcopies only the student BUNDLE (encoder)."""
    module = _make_module()
    assert isinstance(module.predictor, JepaPredictor)
    # The teacher mirrors the bundle (which holds only the encoder); it must
    # NOT carry a copy of the predictor.
    assert not hasattr(module.teacher.model, "predictor")


def test_rejects_b30_sister_latent_valid_override() -> None:
    with pytest.raises(NotImplementedError, match="B30"):
        _make_module(latent_valid_override="all_true")


def test_rejects_b30_sister_sa_mask_mode() -> None:
    with pytest.raises(NotImplementedError, match="key-only"):
        _make_module(sa_mask_mode="key_only")


def test_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError, match="phase"):
        _make_module(phase="p3")


# ---------------------------------------------------------------------------
# B5/B6/B9: P1 paradigm-A front-end masked JEPA
# ---------------------------------------------------------------------------


def test_p1_step_returns_single_term_p1_breakdown() -> None:
    module = _make_module(phase="p1")
    breakdown = module._step(_make_synthetic_batch().data)
    assert isinstance(breakdown, MaskedJepaBreakdown)
    assert breakdown.phase == "p1"
    assert breakdown.n_masked > 0
    assert breakdown.total.ndim == 0
    assert torch.isfinite(breakdown.total)
    assert float(breakdown.total.detach()) >= 0.0  # L1 is non-negative


def test_p1_grad_reaches_frontend_not_terminal_ln_or_predictor() -> None:
    """B36 §7 P1 grad-scope: the front-end token blocks self-predict the
    masked M2, so gradient reaches ``frontend_ln`` (front-end terminal LN)
    but the downstream pool / inter-parcel encoder (``encoder_ln``) and the
    predictor get NO gradient — the loss is computed entirely at M2."""
    module = _make_module(phase="p1")
    breakdown = module._step(_make_synthetic_batch().data)
    breakdown.total.backward()
    enc = module.student.encoder
    assert enc.frontend_ln.weight.grad is not None
    assert torch.isfinite(enc.frontend_ln.weight.grad).all()
    # Downstream of M2 → off the loss path → no grad.
    assert enc.encoder_ln.weight.grad is None
    for p in module.predictor.parameters():
        assert p.grad is None


# ---------------------------------------------------------------------------
# B6/B8/B9: P2 paradigm-B parcel masked JEPA
# ---------------------------------------------------------------------------


def test_p2_step_returns_single_term_p2_breakdown() -> None:
    module = _make_module(phase="p2")
    breakdown = module._step(_make_synthetic_batch().data)
    assert isinstance(breakdown, MaskedJepaBreakdown)
    assert breakdown.phase == "p2"
    assert breakdown.n_masked > 0
    assert breakdown.total.ndim == 0
    assert torch.isfinite(breakdown.total)
    assert float(breakdown.total.detach()) >= 0.0


def test_p2_grad_reaches_encoder_and_predictor() -> None:
    """P2 paradigm B: the visible-only encoder feeds the separate predictor,
    so gradient reaches both the encoder (``encoder_ln``) and the predictor
    (``output_proj``)."""
    module = _make_module(phase="p2")
    breakdown = module._step(_make_synthetic_batch().data)
    breakdown.total.backward()
    enc = module.student.encoder
    assert enc.encoder_ln.weight.grad is not None
    assert module.predictor.output_proj.weight.grad is not None
    assert torch.isfinite(module.predictor.output_proj.weight.grad).all()


# ---------------------------------------------------------------------------
# B6: masked-empty exact 0, detached target, L1 form
# ---------------------------------------------------------------------------


def test_p1_empty_mask_gives_exact_zero_no_nan() -> None:
    """B6 masked-empty contract: ratio 0 → no masked cell → total is an
    exact 0 (graph-connected, no NaN)."""
    module = _make_module(phase="p1", m2_mask_ratio=0.0)
    breakdown = module._step(_make_synthetic_batch().data)
    assert breakdown.n_masked == 0
    assert float(breakdown.total.detach()) == 0.0
    assert torch.isfinite(breakdown.total)


def test_teacher_accumulates_no_grad_target_is_detached() -> None:
    """B6/B26: the teacher target is ``detach()``ed and the teacher forward
    runs under ``no_grad`` — no teacher parameter accumulates gradient."""
    module = _make_module(phase="p2")
    breakdown = module._step(_make_synthetic_batch().data)
    breakdown.total.backward()
    for p in module.teacher.parameters():
        assert p.grad is None


def test_loss_is_l1_not_mse() -> None:
    """B6: ``loss_form='mse'`` produces a strictly different scalar than the
    default L1 on the same (seeded) masked set — proves the default is L1."""
    batch = _make_synthetic_batch()
    enc = _make_tiny_encoder()
    l1 = _make_module(enc, phase="p1", loss_form="l1")._step(batch.data)
    # A fresh module with an identical encoder + the same mask seed, MSE form.
    enc2 = _make_tiny_encoder()
    enc2.load_state_dict(enc.state_dict())
    mse = _make_module(enc2, phase="p1", loss_form="mse")._step(batch.data)
    assert l1.n_masked == mse.n_masked > 0
    assert not torch.allclose(l1.total, mse.total)


def test_b6_l1_gradient_magnitude_constant_in_error() -> None:
    """B6 (canonical V-JEPA target-norm) — the masked loss is *pure L1*, so
    ``d|s-t|/ds = sign(s-t)``: the per-element gradient magnitude is a
    constant ``1/(n_masked·d)`` regardless of the error scale. (MSE / Smooth-L1
    grads scale with the error and would NOT be constant.) This is the
    "gradient magnitude constant in error" check the B6 TEST clause demands,
    stronger than the L1≠MSE scalar comparison above."""
    from speech_decoding.ssl.masked_jepa import p1_frontend_m2_loss

    B, C, F_p, T_p, d = 1, 1, 1, 1, 4
    token_mask = torch.ones(B, C, F_p, T_p, dtype=torch.bool)  # every cell masked
    teacher_m2 = torch.zeros(B, C, F_p, T_p, d)

    grads = []
    for scale in (0.1, 1.0, 5.0):
        student_m2 = torch.full(
            (B, C, F_p, T_p, d), float(scale), requires_grad=True,
        )
        bd = p1_frontend_m2_loss(
            student_m2=student_m2, teacher_m2=teacher_m2, token_mask=token_mask,
        )
        bd.total.backward()
        g = student_m2.grad[token_mask].abs()  # (n_masked, d)
        # student > teacher ⇒ sign = +1 ⇒ |grad| == 1/(n_masked·d) everywhere.
        expected = 1.0 / g.numel()
        torch.testing.assert_close(g, torch.full_like(g, expected))
        grads.append(g)
    # The discriminator: L1's grad is identical across error scales (MSE's
    # would be 0.1× vs 5×). Constant ⇒ pure L1.
    torch.testing.assert_close(grads[0], grads[-1])


# ---------------------------------------------------------------------------
# B7: teacher full-input guard
# ---------------------------------------------------------------------------


def test_b7_teacher_full_input_guard_is_wired() -> None:
    """B7: the teacher forward must see full input. The module never threads
    a JEPA mask into the teacher; the guard fires if a False-containing
    visibility mask is passed (simulating a leak)."""
    from speech_decoding.ssl.ema import assert_teacher_full_input

    # The wired call: both masks None ⇒ vacuously passes (teacher full-input).
    assert_teacher_full_input(patch_mask=None, shaft_mask=None)
    # A leaked student mask → its visibility (~mask) has False entries → raise.
    token_mask = torch.zeros(2, 3, 2, 4, dtype=torch.bool)
    token_mask[0, 0, 0, 0] = True
    with pytest.raises(AssertionError, match="full-input"):
        assert_teacher_full_input(patch_mask=~token_mask)


def test_b7_step_does_not_pass_mask_to_teacher() -> None:
    """B7 integration: running ``_step`` (which calls the guard at the teacher
    call site) never raises — the teacher truly gets full input."""
    for phase in ("p1", "p2"):
        module = _make_module(phase=phase)
        breakdown = module._step(_make_synthetic_batch().data)
        assert torch.isfinite(breakdown.total)


def test_b7_teacher_forward_call_site_raises_on_leaked_mask() -> None:
    """B7 call-site wiring: ``_teacher_forward`` runs
    ``assert_teacher_full_input`` on the EXACT kwargs the teacher receives.
    Inject a partial ``token_mask`` (simulating a refactor that leaks the
    student mask into the teacher pass) and confirm the guard fires — this
    exercises the live ``_step`` call site, not just the helper in isolation,
    closing the 'guard is vacuously wired' gap."""
    module = _make_module(phase="p1")
    batch = _make_synthetic_batch()
    student_kwargs = module._extract_student_kwargs(batch.data)
    C, F_p, T_p = module.student.encoder.patch_grid_shape(
        student_kwargs["electrode_tokens"],
    )
    B = batch.data["electrode_tokens"].shape[0]
    leaked = torch.zeros(B, C, F_p, T_p, dtype=torch.bool)
    leaked[0, 0, 0, 0] = True  # one masked cell ⇒ ~leaked has a False entry
    teacher_kwargs = dict(student_kwargs, token_mask=leaked)
    with pytest.raises(AssertionError, match="full-input"):
        module._teacher_forward(teacher_kwargs)

    # A parcel-time leak fires the same tripwire.
    K = student_kwargs["support"].shape[-1]
    leaked_ptm = torch.zeros(B, K, T_p, dtype=torch.bool)
    leaked_ptm[0, 0, 0] = True
    with pytest.raises(AssertionError, match="full-input"):
        module._teacher_forward(dict(student_kwargs, parcel_time_mask=leaked_ptm))


def test_b7_teacher_forward_full_input_passes_and_returns_taps() -> None:
    """B7 live path: with no mask key in ``teacher_kwargs`` the guard passes
    and the teacher returns its tap dict. ``m2_only=True`` (the P1 path)
    returns just the M2 tap."""
    module = _make_module(phase="p1")
    student_kwargs = module._extract_student_kwargs(_make_synthetic_batch().data)
    taps = module._teacher_forward(dict(student_kwargs), m2_only=True)
    assert set(taps.keys()) == {"M2"}


# ---------------------------------------------------------------------------
# B9: exactly one term; retired multi-term path not imported
# ---------------------------------------------------------------------------


def test_b9_module_does_not_import_retired_aggregator_helpers() -> None:
    """B9: the retired multi-term aggregator surface is gone from the joint
    module's namespace (the masked-JEPA default is single-term)."""
    mod = importlib.import_module(
        "speech_decoding.experiments.v14_joint_module"
    )
    for dead in (
        "compute_v14_ssl_losses",
        "V14TotalLossBreakdown",
        "LossVariant",
        "_variant_wants_m3",
        "_variant_wants_utt",
        "_compose_l_pre_frame",
    ):
        assert not hasattr(mod, dead), dead


def test_b9_breakdown_exposes_exactly_one_scalar_term() -> None:
    """B9: the breakdown carries a single ``total`` scalar + its phase tag —
    there are no per-term sub-fields (l_mid_slot / l_post_utterance / ...)."""
    breakdown = _make_module(phase="p1")._step(_make_synthetic_batch().data)
    fields = set(vars(breakdown).keys())
    assert fields == {"total", "phase", "n_masked"}


def test_b9_layer_avg_with_instance_norm_retired_from_runtime() -> None:
    """B9: ``layer_avg_with_instance_norm`` (data2vec-2.0 / EAT layer-averaging)
    is explicitly named for quarantine. Under the canonical V-JEPA target-norm
    the target is the encoder's own terminal LN, so this helper builds NO live
    target — neither the joint module nor the masked-JEPA loss module imports
    or calls it (it survives only as the ``R-layer-avg-target`` sister + its
    own unit test)."""
    import inspect

    import speech_decoding.experiments.v14_joint_module as jm
    import speech_decoding.ssl.masked_jepa as mj

    for mod in (jm, mj):
        assert not hasattr(mod, "layer_avg_with_instance_norm"), mod.__name__
        assert "layer_avg_with_instance_norm(" not in inspect.getsource(mod), (
            f"{mod.__name__} must not CALL the retired data2vec helper"
        )


# ---------------------------------------------------------------------------
# B26 EMA + optimizer scope
# ---------------------------------------------------------------------------


def test_ema_step_updates_teacher_fixed_tau() -> None:
    """B26 lock: τ=0.99925 fixed; ``update_from`` pulls the teacher toward
    the (post-step) student. Exercised on the encoder's ``encoder_ln``."""
    module = _make_module()
    with torch.no_grad():
        module.student.encoder.encoder_ln.weight.fill_(2.0)
    pre = module.teacher.model.encoder.encoder_ln.weight.detach().clone()
    coeff = module.teacher.update_from(module.student)
    post = module.teacher.model.encoder.encoder_ln.weight.detach().clone()
    assert coeff == pytest.approx(0.99925)
    torch.testing.assert_close(post, 0.99925 * pre + 0.00075 * 2.0)


def test_ema_fires_once_per_optimizer_step_not_per_microbatch() -> None:
    """#46: the EMA update must live on ``on_before_zero_grad`` — Lightning's
    once-per-optimiser-step hook (after ``optimizer.step()``, before grads are
    zeroed) — NOT ``on_train_batch_end``, which fires once per micro-batch.

    Under ``accumulate_grad_batches=K`` the per-micro-batch placement applied K
    EMA updates per optimiser step, so the effective momentum became τ^K and the
    teacher trailed K× too fast — silently changing the SSL dynamics. This guards
    against a revert to the per-micro-batch hook."""
    module = _make_module()
    calls = {"n": 0}
    orig = module.teacher.update_from

    def _counting_update_from(student, **kw):  # instance attr: no self-binding
        calls["n"] += 1
        return orig(student, **kw)

    module.teacher.update_from = _counting_update_from  # type: ignore[method-assign]

    # The EMA lives on the once-per-optimiser-step hook.
    module.on_before_zero_grad(optimizer=None)
    assert calls["n"] == 1, "on_before_zero_grad must apply exactly one EMA step"

    # It must NOT also fire per micro-batch: the base-class ``on_train_batch_end``
    # is a no-op, so driving it (as Lightning does every micro-batch) leaves the
    # teacher untouched. A revert that re-adds the EMA call here would trip this.
    before = calls["n"]
    module.on_train_batch_end(outputs=None, batch=None, batch_idx=0)
    assert calls["n"] == before, (
        "on_train_batch_end must not apply an EMA step — per-micro-batch updates "
        "break gradient accumulation (#46): K updates/step ⇒ effective τ^K"
    )


def test_trainable_parameters_include_predictor() -> None:
    """The optimizer scope (``_trainable_parameters``) covers the student
    encoder + the predictor, and excludes the frozen teacher."""
    module = _make_module()
    trainable = {id(p) for p in module._trainable_parameters()}
    assert all(id(p) in trainable for p in module.predictor.parameters())
    assert all(id(p) in trainable for p in module.student.parameters())
    assert not any(id(p) in trainable for p in module.teacher.parameters())


# ---------------------------------------------------------------------------
# 5/28 P0 monitors — coverage / RankMe / grad-spike still wired
# ---------------------------------------------------------------------------


def test_monitor_logs_parcel_coverage_on_every_step() -> None:
    module = _make_module()
    logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: logged.update({key: float(value)})  # type: ignore[method-assign]
    module._monitor_from_step(_make_synthetic_batch().data, step_name="train")
    for key in (
        "train_mon_coverage_active_mean",
        "train_mon_coverage_active_cv",
        "train_mon_coverage_slot_var",
        "train_mon_coverage_alarm",
    ):
        assert key in logged
    assert logged["train_mon_coverage_alarm"] in (0.0, 1.0)


def test_monitor_logs_teacher_rank_on_train_from_step0_and_val() -> None:
    """I1 (B36 WS-I): the M4 RankMe fires on the TRAIN loop from step 0 (the
    val/test gate was dropped) so a teacher-feature collapse is caught at the
    start of pretraining, not only at the first val epoch. M4 is the P2 target,
    so this is the P2-phase probe (2026-06-03 phase-scope fix moved M4 RankMe
    out of P1, where M4 is untrained random-init)."""
    module = _make_module(phase="p2")
    train_logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: train_logged.update({key: float(value)})  # type: ignore[method-assign]
    module._monitor_from_step(_make_synthetic_batch().data, step_name="train")
    for key in (
        "train_mon_rankme",
        "train_mon_rankme_normalised",
        "train_mon_rankme_warn",
        "train_mon_rankme_alarm",
    ):
        assert key in train_logged, key

    val_logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: val_logged.update({key: float(value)})  # type: ignore[method-assign]
    module._monitor_from_step(_make_synthetic_batch().data, step_name="val")
    for key in (
        "val_mon_rankme",
        "val_mon_rankme_normalised",
        "val_mon_rankme_warn",
        "val_mon_rankme_alarm",
    ):
        assert key in val_logged


def test_rankme_reads_post_encoder_ln_tap_not_ln_frame() -> None:
    """I1: the RankMe monitor reads the EMA teacher's post-``encoder_ln`` M4 tap
    (the canonical terminal LN, B6) — there is no separate ``ln_frame`` head any
    more. Guards the doc/code claim against an ``ln_frame`` revival."""
    module = _make_module()
    student_enc = module.student.encoder
    teacher_enc = module.teacher.model.encoder
    assert hasattr(student_enc, "encoder_ln")
    assert hasattr(teacher_enc, "encoder_ln")
    assert not hasattr(student_enc, "ln_frame")
    assert not hasattr(teacher_enc, "ln_frame")


def test_training_step_at_batch_idx_0_logs_rankme() -> None:
    """I1 end-to-end: driving ``training_step`` at global step 0 (monitor-due
    with no trainer attached → every-step cadence) emits the PHASE-APPROPRIATE
    feature-rank probe — ``train_mon_frontend_rankme`` in P1 (M2), and
    ``train_mon_rankme`` in P2 (M4)."""
    p1 = _make_module(phase="p1")
    p1_logged: dict[str, float] = {}
    p1.log = lambda key, value, **_kw: p1_logged.update({key: float(value)})  # type: ignore[method-assign]
    p1.training_step(_make_synthetic_batch(), 0)
    assert "train_mon_frontend_rankme" in p1_logged
    assert "train_mon_rankme" not in p1_logged

    p2 = _make_module(phase="p2")
    p2_logged: dict[str, float] = {}
    p2.log = lambda key, value, **_kw: p2_logged.update({key: float(value)})  # type: ignore[method-assign]
    p2.training_step(_make_synthetic_batch(), 0)
    assert "train_mon_rankme" in p2_logged
    assert "train_mon_frontend_rankme" not in p2_logged


def test_monitor_rank_probe_is_phase_scoped() -> None:
    """2026-06-03 mis-scope fix: each phase probes ONLY the representation it
    trains. P1 trains M2 (front-end) and never gradients the hard pool /
    inter-parcel stack that build M4, so probing M4 RankMe in P1 reads
    random-init layers and fires a false collapse alarm from step 0. So P1 ->
    front-end (M2) rank only; P2 -> M4 rank only. The two never alias on the
    same metric key."""
    p1 = _make_module(phase="p1")
    p1_logged: dict[str, float] = {}
    p1.log = lambda key, value, **_kw: p1_logged.update({key: float(value)})  # type: ignore[method-assign]
    p1._monitor_from_step(_make_synthetic_batch().data, step_name="train")
    for key in (
        "train_mon_frontend_rankme",
        "train_mon_frontend_rankme_normalised",
        "train_mon_frontend_rankme_warn",
        "train_mon_frontend_rankme_alarm",
    ):
        assert key in p1_logged, key
    # P1 must NOT emit the M4 probes (random-init M4 -> false alarm) ...
    assert "train_mon_rankme" not in p1_logged
    assert "train_mon_rankme_alarm" not in p1_logged
    # ... nor the M4-based orphan ratio.
    assert "train_mon_mask_002_ratio" not in p1_logged

    p2 = _make_module(phase="p2")
    p2_logged: dict[str, float] = {}
    p2.log = lambda key, value, **_kw: p2_logged.update({key: float(value)})  # type: ignore[method-assign]
    p2._monitor_from_step(_make_synthetic_batch().data, step_name="train")
    assert "train_mon_rankme_alarm" in p2_logged
    assert "train_mon_frontend_rankme" not in p2_logged


def _perturb_student_for_nonzero_grad(module: V14JointBrainModule) -> None:
    """Perturb the student so the deepcopy-identical EMA teacher no longer
    matches; otherwise all L1 terms could be ~0. ``encoder_ln`` is on the
    M4 path (P2) and downstream of M2 (P1)."""
    with torch.no_grad():
        module.student.encoder.frontend_ln.weight.add_(0.3)


def test_on_before_optimizer_step_logs_grad_spike() -> None:
    module = _make_module(phase="p1")
    _perturb_student_for_nonzero_grad(module)
    breakdown = module._step(_make_synthetic_batch().data)
    breakdown.total.backward()
    logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: logged.update({key: float(value)})  # type: ignore[method-assign]
    module.on_before_optimizer_step(optimizer=None)
    for key in (
        "train_mon_grad_l2",
        "train_mon_grad_ema_l2",
        "train_mon_grad_spike_ratio",
        "train_mon_grad_spike",
        "train_mon_grad_diverged",
    ):
        assert key in logged
    assert logged["train_mon_grad_ema_l2"] == pytest.approx(0.0)
    assert logged["train_mon_grad_l2"] > 0.0
    assert float(module._grad_ema_l2.item()) > 0.0


def test_grad_ema_buffer_persists_across_calls() -> None:
    module = _make_module(phase="p1")
    _perturb_student_for_nonzero_grad(module)
    batch = _make_synthetic_batch()
    module._step(batch.data).total.backward()
    module.on_before_optimizer_step(optimizer=None)
    first_ema = float(module._grad_ema_l2.item())
    assert first_ema > 0.0

    for p in module._trainable_parameters():
        if p.grad is not None:
            p.grad.zero_()
    module._step(batch.data).total.backward()
    logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: logged.update({key: float(value)})  # type: ignore[method-assign]
    module.on_before_optimizer_step(optimizer=None)
    assert logged["train_mon_grad_ema_l2"] == pytest.approx(first_ema)


def test_train_monitor_due_fires_on_log_cadence() -> None:
    module = _make_module()
    module.trainer = SimpleNamespace(log_every_n_steps=10)  # type: ignore[assignment]
    for due_idx in (0, 10, 20, 100):
        assert module._train_monitor_due(due_idx) is True, due_idx
    for skip_idx in (1, 5, 9, 11, 19, 99):
        assert module._train_monitor_due(skip_idx) is False, skip_idx


def test_train_monitor_due_falls_back_to_every_step() -> None:
    module = _make_module()  # no trainer attached → property raises
    for idx in (0, 1, 2, 3, 7, 50):
        assert module._train_monitor_due(idx) is True, idx
