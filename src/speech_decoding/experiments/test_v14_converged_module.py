"""TDD for V14ConvergedBrainModule — the Philosophy-B Lightning shell around the
self-contained V14ConvergedSSL. Quantitative checks of every seam: batch ingest
(support→parcel, valid→electrode_mask), the loss path, grad isolation
student-vs-teacher, the EMA tick, optimizer construction (teacher excluded +
no-WD split), the encoder phase-handoff round-trip, and overfit-one-batch."""

from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from neuraltrain.optimizers import LightningOptimizer

from speech_decoding.experiments.v14_converged_module import (
    V14ConvergedBrainModule,
    _apply_sdpa_backend,
    _sdpa_forward_context,
)
from speech_decoding.models.v14_converged import TightPackConfig, V14ConvergedSSL


# --------------------------------------------------------------------- fixtures
def _tiny_model(n_parcels: int = 3) -> V14ConvergedSSL:
    return V14ConvergedSSL(
        16, n_parcels,
        n_heads=4,
        frontend_layers=1,
        latent_layers=1,
        m2_pred_dim=16, m2_pred_layers=1,
        m4_pred_dim=16, m4_pred_layers=1,
    )


def _optim(weight_decay: float = 0.0) -> LightningOptimizer:
    return LightningOptimizer(
        optimizer={"name": "AdamW", "lr": 1e-3,
                   "kwargs": {"weight_decay": weight_decay}}
    )


def _module(n_parcels: int = 3, **kw) -> V14ConvergedBrainModule:
    return V14ConvergedBrainModule(
        model=_tiny_model(n_parcels), optim_config=_optim(kw.pop("wd", 0.0)),
        ema_tau=kw.pop("ema_tau", 0.99), **kw,
    )


def _batch(B: int = 2, C: int = 6, K: int = 3, *, valid: torch.Tensor | None = None):
    """Synthetic 3-band batch on the locked BandSpec geometry, in the REAL cache
    layout: the cartesian slow band arrives as [Re(6) ++ Im(6)] on the freq axis
    (B, C, 12, 5) — `_converged_inputs` splits it to (B, C, 2, 6, 5). ``support`` is
    a one-hot assigning 2 electrodes to each of K parcels (so M4 tubes a real
    multi-electrode parcel and the electrode-mean is non-trivial).

    Note: ``randn(B, C, 12, 5)`` draws the same flat random stream as the old
    ``randn(B, C, 2, 6, 5)``, so the post-split tensor is value-identical and the
    downstream loss/overfit checks are unchanged — they now exercise the adapter."""
    torch.manual_seed(0)
    slow = torch.randn(B, C, 12, 5)
    beta = torch.randn(B, C, 6, 17)
    hg = torch.randn(B, C, 9, 33)
    support = torch.zeros(B, C, K)
    for e in range(C):
        support[:, e, e % K] = 1.0
    data = {
        "electrode_tokens_slow": slow,
        "electrode_tokens_beta": beta,
        "electrode_tokens_hg": hg,
        "support": support,
    }
    if valid is not None:
        data["valid_mask"] = valid
    return SimpleNamespace(data=data)


# ----------------------------------------------------------------- batch ingest
def test_converged_inputs_derive_parcel_and_mask() -> None:
    m = _module()
    data = _batch(B=2, C=6, K=3).data
    slow, beta, hg, ppe, emask = m._converged_inputs(data)
    assert slow.shape == (2, 6, 2, 6, 5)
    assert beta.shape == (2, 6, 6, 17) and hg.shape == (2, 6, 9, 33)
    # support one-hot e%K → parcel ids [0,1,2,0,1,2]
    assert ppe[0].tolist() == [0, 1, 2, 0, 1, 2]
    # no valid_mask → all electrodes real
    assert emask.dtype == torch.bool and bool(emask.all())


def test_converged_inputs_honor_valid_mask() -> None:
    m = _module()
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[0, 5] = False  # one padded electrode
    _, _, _, _, emask = m._converged_inputs(_batch(valid=valid).data)
    assert not bool(emask[0, 5]) and bool(emask[1, 5])


def test_converged_inputs_splits_cartesian_slow_re_im() -> None:
    """REGRESSION (3stft launch 2026-06-18): the cartesian slow band is cached as
    [Re(6) ++ Im(6)] concatenated on the freq axis (B, C, 12, 5); the slow stem is
    built for in_channels=2 and needs the (B, C, 2, 6, 5) channel form. The adapter
    must split the BLOCK [Re then Im] (row-major), NOT interleave — pin the order
    with distinguishable Re/Im fills. Without this split the stem saw a 4-D tensor
    and the first forward died ('in_channels=2 requires a 5-D input; got 4-D')."""
    m = _module()
    B, C = 2, 6
    slow = torch.empty(B, C, 12, 5)
    slow[:, :, :6, :] = 1.0   # Re block (freq 0:6)
    slow[:, :, 6:, :] = 2.0   # Im block (freq 6:12)
    data = _batch(B=B, C=C).data
    data["electrode_tokens_slow"] = slow
    slow_out, *_ = m._converged_inputs(data)
    assert slow_out.shape == (B, C, 2, 6, 5)
    assert torch.all(slow_out[:, :, 0] == 1.0), "channel 0 must be the Re block"
    assert torch.all(slow_out[:, :, 1] == 2.0), "channel 1 must be the Im block"


def test_converged_inputs_rejects_odd_slow_freq_axis() -> None:
    """An odd slow freq axis cannot be a [Re ++ Im] cartesian pair — fail loud
    rather than silently mis-splitting."""
    m = _module()
    data = _batch().data
    data["electrode_tokens_slow"] = torch.randn(2, 6, 11, 5)
    with pytest.raises(ValueError, match="even freq axis"):
        m._converged_inputs(data)


# -------------------------------------------------------------------- loss path
def test_step_returns_finite_nonneg_losses() -> None:
    m = _module()
    out = m._step(_batch().data)
    for k in ("loss", "l_m2", "l_m4"):
        assert torch.isfinite(out[k]) and float(out[k]) >= 0.0


def test_static_forward_matches_legacy_step() -> None:
    # Step B: with the static mask regime active (tube_cfg set), `static_forward=True`
    # routes `_step` through the fixed-K / maskless static forward; on a session-
    # homogeneous batch it must produce losses identical to the legacy variable-count
    # ragged forward (same mask_seed ⇒ same masks ⇒ output-equivalent paths).
    # One module (identical weights) toggled across the two paths, with the mask RNG
    # reset between calls so both draw the SAME masks — isolates the forward routing.
    m = _module(tube_cfg=TightPackConfig(ratio=0.25), static_forward=False, mask_seed=3)
    data = _batch(B=3, C=6).data                       # all-real ⇒ homogeneous
    with torch.no_grad():
        m.static_forward = False
        m._mask_gen.manual_seed(3)
        out_leg = m._step(data)
        m.static_forward = True
        m._mask_gen.manual_seed(3)
        out_stat = m._step(data)
    for k in ("loss", "l_m2", "l_m4"):
        assert torch.allclose(out_stat[k], out_leg[k], atol=1e-5), k
    assert float(out_leg["l_m2"]) > 0.0 and float(out_leg["l_m4"]) > 0.0


def test_step_emits_per_band_diagnostics() -> None:
    # Ben 2026-06-18: per-band M2 (beta/hg) + M4 (slow/beta/hg) loss breakdowns
    # ride in the _step output so _log_losses can surface them.
    m = _module()
    out = m._step(_batch().data)
    for k in ("l_m2_beta", "l_m2_hg", "l_m4_slow", "l_m4_beta", "l_m4_hg"):
        assert k in out, f"missing per-band diagnostic {k}"
        assert torch.isfinite(out[k]) and float(out[k]) >= 0.0
    assert "l_m2_slow" not in out          # slow is M2-exempt


def test_step_emits_ev_tv_and_stem_norms() -> None:
    # Ben 2026-06-18 monitor: the aggregate + per-band explained_var/target_var and
    # the per-band stem-output norms ride in _step's output too.
    m = _module()
    out = m._step(_batch().data)
    for k in ("ev_m2", "tv_m2", "ev_m4", "tv_m4",
              "stem_norm_slow", "stem_norm_beta", "stem_norm_hg"):
        assert k in out, f"missing monitor key {k}"
        assert not out[k].requires_grad
    for k in ("stem_norm_slow", "stem_norm_beta", "stem_norm_hg"):
        assert torch.isfinite(out[k]) and float(out[k]) > 0.0


def test_log_losses_skips_nonfinite_and_nonscalar() -> None:
    # ev/tv are NaN when a band has < 2 scored cells this step — an undefined-variance
    # step must NOT poison the epoch mean, and a non-scalar must never be logged.
    m = _module()
    logged: dict[str, float] = {}
    m.log = lambda name, val, **kw: logged.__setitem__(name, float(val))  # type: ignore[method-assign]
    out = {
        "loss": torch.tensor(1.0),
        "tv_m4_slow": torch.tensor(2.0),
        "ev_m2": torch.tensor(float("nan")),       # undefined variance → skip
        "stem_norm_hg": torch.tensor(float("inf")),  # non-finite → skip
        "vec": torch.zeros(3),                      # non-scalar → skip
    }
    m._log_losses(out, "train", on_step=True, on_epoch=False)
    assert logged == {"train_loss": 1.0, "train_tv_m4_slow": 2.0}


def test_training_step_returns_scalar_loss() -> None:
    m = _module()
    loss = m.training_step(_batch(), 0)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_grad_flows_to_student_not_teacher() -> None:
    m = _module()
    out = m._step(_batch().data)
    out["loss"].backward()
    teacher_ids = {id(p) for p in m.model.teacher_frontend.parameters()}
    got_student_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for _, p in m.model.named_parameters()
        if id(p) not in teacher_ids and p.requires_grad
    )
    assert got_student_grad, "student must receive gradient"
    assert all(p.grad is None for p in m.model.teacher_frontend.parameters()), \
        "frozen EMA teacher must never accumulate gradient"


# --------------------------------------------------------------------- EMA tick
def test_ema_tick_moves_teacher_toward_student() -> None:
    m = _module(ema_tau=0.5)
    # perturb the student frontend so it differs from the (cloned) teacher
    with torch.no_grad():
        for p in m.model.student_frontend.parameters():
            p.add_(torch.randn_like(p))
    t_before = next(iter(m.model.teacher_frontend.parameters())).clone()
    s = next(iter(m.model.student_frontend.parameters())).clone()
    m.on_before_zero_grad(optimizer=None)
    t_after = next(iter(m.model.teacher_frontend.parameters()))
    # τ=0.5 ⇒ teacher = 0.5·teacher + 0.5·student → strictly between, toward student
    assert not torch.allclose(t_after, t_before)
    assert (t_after - s).abs().sum() < (t_before - s).abs().sum()


# ------------------------------------------------------------------- optimizer
def _groups(m: V14ConvergedBrainModule):
    out = m.configure_optimizers()
    opt = out["optimizer"] if isinstance(out, dict) else out
    return opt.param_groups


def test_configure_optimizers_excludes_frozen_teacher() -> None:
    m = _module()
    optimized = {id(p) for g in _groups(m) for p in g["params"]}
    teacher_ids = {id(p) for p in m.model.teacher_frontend.parameters()}
    assert optimized.isdisjoint(teacher_ids), "EMA teacher must not be optimised"
    trainable = {id(p) for p in m.model.parameters() if p.requires_grad}
    assert optimized == trainable


def test_configure_optimizers_no_wd_split_at_positive_wd() -> None:
    m = _module(wd=0.1)
    groups = _groups(m)
    wds = {g["weight_decay"] for g in groups}
    assert 0.0 in wds and 0.1 in wds
    from speech_decoding.experiments.optim_param_groups import no_decay_param_ids
    exempt = no_decay_param_ids(m.model)
    for g in groups:
        for p in g["params"]:
            assert g["weight_decay"] == (0.0 if id(p) in exempt else 0.1)


def test_configure_optimizers_single_group_at_zero_wd() -> None:
    m = _module(wd=0.0)
    groups = _groups(m)
    assert len(groups) == 1 and groups[0]["weight_decay"] == 0.0


# -------------------------------------------------------------- phase handoff
def test_transferable_state_roundtrips_encoder() -> None:
    src = _module()
    # train the source a touch so its encoder weights are non-trivial
    out = src._step(_batch().data)
    out["loss"].backward()
    with torch.no_grad():
        for p in src.model.student_frontend.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-0.1)
    state = src.transferable_state()
    assert set(state) == {"student_frontend", "latent"}

    dst = _module()
    dst.load_transferable_state(state)
    for (na, pa), (_, pb) in zip(
        src.model.student_frontend.named_parameters(),
        dst.model.student_frontend.named_parameters(),
    ):
        assert torch.allclose(pa, pb), f"frontend {na} not transferred"
    # teacher re-synced to the loaded student frontend
    for (_, ps), (nt, pt) in zip(
        dst.model.student_frontend.named_parameters(),
        dst.model.teacher_frontend.named_parameters(),
    ):
        assert torch.allclose(ps, pt), f"teacher not re-synced at {nt}"


def test_load_transferable_state_missing_component_raises() -> None:
    m = _module()
    with pytest.raises(KeyError):
        m.load_transferable_state({"latent": {}})


# ---------------------------------------------------------------- torch.compile
_COMPILE_ENV_KEYS = (
    "V14_COMPILE", "V14_COMPILE_MODE", "V14_COMPILE_DYNAMIC",
    "V14_COMPILE_DDP_OPTIMIZER",
)


@pytest.fixture(autouse=True)
def _eager_unless_opted_in(monkeypatch):
    """Default every test to EAGER construction, independent of suite ordering.

    ``test_ddp_dispatch_wiring`` invokes the dispatch arg path, which sets
    ``os.environ["V14_COMPILE"] = "1"`` directly and does NOT restore it. Without
    this guard that leak makes every later ``_module()`` build a *compiled* model,
    whose forward hits the inductor backend and dies with ``CppCompileError`` on a
    host with no C compiler. The compile tests re-enable compile within their own
    body via ``_compile_env`` (a context manager), so they are unaffected;
    ``monkeypatch`` restores the cleared keys after each test."""
    for k in _COMPILE_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


@contextmanager
def _compile_env(**kv: str):
    """Set the V14_COMPILE* env vars for the body and restore both them AND the
    global ``torch._dynamo.config.optimize_ddp`` afterwards (the module mutates
    the latter in __init__, so it must not leak into sibling tests)."""
    import torch._dynamo as _dyn

    saved = {k: os.environ.get(k) for k in _COMPILE_ENV_KEYS}
    saved_opt = _dyn.config.optimize_ddp
    try:
        for k in _COMPILE_ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(kv)
        yield
    finally:
        for k in _COMPILE_ENV_KEYS:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]
        _dyn.config.optimize_ddp = saved_opt


def test_no_compile_env_is_eager_and_byte_identical() -> None:
    """Unset V14_COMPILE → no OptimizedModule registered; ``_call_model`` is the
    bare ``self.model`` and the loss path is byte-identical to a direct call."""
    with _compile_env():  # all four keys unset
        m = _module()
    assert m._compiled_fwd == {}
    m._mask_gen.manual_seed(7)
    out_call = m._step(_batch().data)
    # re-derive the same masks and call the model directly: must match exactly
    slow, beta, hg, ppe, emask = m._converged_inputs(_batch().data)
    from speech_decoding.models.v14_converged import sample_ssl_masks
    m._mask_gen.manual_seed(7)
    masks = sample_ssl_masks(
        ppe.cpu(), emask.cpu(), m._mask_gen,
        m2_cfg=m.m2_cfg, m4_cfg=m.m4_cfg, bands=m.model.bands,
    )
    out_direct = m.model(slow, beta, hg, ppe, emask, **masks)
    assert torch.equal(out_call["loss"], out_direct["loss"])


def test_compile_env_disables_ddp_optimizer_by_default() -> None:
    """V14_COMPILE truthy → the model forward is compiled AND the DDPOptimizer
    bucket-split is DISABLED (the single-graph fix for the ragged-DDP hang).
    Default-off is the whole point: the production sweep is compile+DDP+dynamic."""
    import torch._dynamo as _dyn

    with _compile_env(V14_COMPILE="1", V14_COMPILE_DYNAMIC="1"):
        m = _module()
        assert "model" in m._compiled_fwd
        assert _dyn.config.optimize_ddp is False


def test_compile_env_ddp_optimizer_is_opt_in() -> None:
    """The bucket-split optimizer is recoverable via V14_COMPILE_DDP_OPTIMIZER=1
    (escape hatch), proving the default-off is a deliberate switch not a constant."""
    import torch._dynamo as _dyn

    with _compile_env(V14_COMPILE="1", V14_COMPILE_DDP_OPTIMIZER="1"):
        _module()
        assert _dyn.config.optimize_ddp is True


def test_compile_dynamic_three_states() -> None:
    """V14_COMPILE_DYNAMIC resolves to THREE states in _compile_spec[1]:
    "1"->True (symbolic, compile once), "0"->False (fully static, no symbolic
    reasoning — the torch-2.10/GH200 storm escape), unset->None (automatic).
    Distinguishing "0" from unset is the whole point: --no-compile-dynamic must
    force static, not fall back to automatic-dynamic (which still goes symbolic)."""
    with _compile_env(V14_COMPILE="1", V14_COMPILE_DYNAMIC="1"):
        assert _module()._compile_spec[1] is True
    with _compile_env(V14_COMPILE="1", V14_COMPILE_DYNAMIC="0"):
        assert _module()._compile_spec[1] is False
    with _compile_env(V14_COMPILE="1"):  # V14_COMPILE_DYNAMIC unset
        assert _module()._compile_spec[1] is None


def test_compiled_forward_matches_eager_loss() -> None:
    """Decisive veracity check: the COMPILED forward (the config Ben launches:
    V14_COMPILE + dynamic) yields the same loss as eager, within fp tolerance.
    Same seed for the build (identical init) and the mask draw (stationary target).

    GPU-GATED (2026-06-19): compile+dynamic equivalence is a CUDA/Triton property.
    The production sweep runs on GPU, where the two paths are bit-exact — verified
    here and out-of-band at C=6..130 (|Δ|=0). The CPU inductor backend under
    ``dynamic=True`` is a different beast: it needs a C++ toolchain AND trips a
    symbolic-shape codegen limit ("cannot determine truth value of Relational")
    on the ragged data-dependent gather — neither of which the CUDA run ever hits.
    So skip on a CPU-only host (there is no GPU compile to verify) rather than
    swallow a real compile failure into a skip (the old try/except masked it)."""
    if not torch.cuda.is_available():
        pytest.skip("compile+dynamic equivalence is a CUDA/Triton property; no GPU here")
    dev = torch.device("cuda")

    def _to(d: dict) -> dict:
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}

    torch.manual_seed(0)
    m_e = _module().to(dev)
    m_e._mask_gen.manual_seed(7)
    out_e = m_e._step(_to(_batch().data))

    with _compile_env(V14_COMPILE="1", V14_COMPILE_DYNAMIC="1"):
        torch.manual_seed(0)
        m_c = _module().to(dev)
        assert "model" in m_c._compiled_fwd
        m_c._mask_gen.manual_seed(7)
        out_c = m_c._step(_to(_batch().data))
    assert torch.isfinite(out_c["loss"])
    assert torch.allclose(out_c["loss"], out_e["loss"], atol=1e-4, rtol=1e-4)


def test_getstate_drops_compiled_forward_and_rebuilds_lazily() -> None:
    """A compiled module must survive the exca job-pickle. ``__getstate__`` drops
    the un-picklable OptimizedModule (its ``torch._dynamo`` guard weakrefs break
    cloudpickle) while keeping ``_compile_spec`` (a plain tuple), so the
    unpickled module rebuilds the compiled forward lazily on the first forward.
    This is what enables ``torch.compile`` on the ``--in-allocation-ddp`` 4-GPU
    path (the pickle previously died with ``cannot pickle 'weakref.ReferenceType'``)."""
    with _compile_env(V14_COMPILE="1", V14_COMPILE_DYNAMIC="1"):
        m = _module()
        assert "model" in m._compiled_fwd
        spec = m._compile_spec
        assert spec is not None
        # __getstate__ returns a copy with the OptimizedModule dropped...
        state = m.__getstate__()
        assert state["_compiled_fwd"] == {}
        assert state["_compile_spec"] == spec
        # ...and the live original is untouched (getstate works on a copy), so a
        # single-GPU / cluster run that never round-trips keeps its compiled fwd.
        assert "model" in m._compiled_fwd
        # Simulate the unpickled side: empty dict, spec retained → lazy rebuild.
        m._compiled_fwd = {}
        m._build_compiled_model()
        assert "model" in m._compiled_fwd


def test_compiled_module_cloudpickles_without_weakref_error() -> None:
    """End-to-end: cloudpickle (what exca uses) round-trips a compiled module
    with no ``weakref.ReferenceType`` crash, and the restored module has an empty
    ``_compiled_fwd`` + a surviving ``_compile_spec`` ready for the lazy rebuild."""
    import cloudpickle

    with _compile_env(V14_COMPILE="1", V14_COMPILE_DYNAMIC="1"):
        m = _module()
        assert "model" in m._compiled_fwd
        blob = cloudpickle.dumps(m)          # must NOT raise on dynamo weakrefs
        assert "model" in m._compiled_fwd    # original still compiled
    m2 = cloudpickle.loads(blob)
    assert m2._compiled_fwd == {}
    assert m2._compile_spec == m._compile_spec


# ---------------------------------------------------------------- overfit check
def test_overfits_one_batch_fixed_masks() -> None:
    """The decisive veracity check: with the mask draw frozen (re-seed each step)
    the shipped default config (M4 precision weight ON) must drive a single
    batch's loss steeply down.

    Seed BEFORE the build so the random init is fixed — otherwise the ratio is
    flaky run-to-run. The M4 precision weight downweights the synthetic montage's
    n=2 parcels to ((2-1)/(11-1))^1 = 0.1, so M4 contributes a 10×-attenuated
    signal and the batch overfits mostly through M2; 150 steps clears 0.5 with a
    comfortable margin (ratio ≈ 0.37)."""
    torch.manual_seed(0)
    m = _module()
    batch = _batch()
    opt = torch.optim.Adam(
        [p for p in m.model.parameters() if p.requires_grad], lr=3e-3,
    )
    losses = []
    for _ in range(150):
        m._mask_gen.manual_seed(0)  # freeze the mask so the target is stationary
        opt.zero_grad()
        out = m._step(batch.data)
        out["loss"].backward()
        opt.step()
        losses.append(float(out["loss"]))
    assert losses[-1] < 0.5 * losses[0], (
        f"one-batch loss did not overfit: {losses[0]:.4f} → {losses[-1]:.4f}"
    )


# ====================================================================
# Monitor instrumentation — the full joint-module suite ported in. Each test
# drives a hook with `self.log` captured and asserts the COMPLETE metric set is
# emitted (so "every monitor is on" is verified, not assumed).
# ====================================================================
def _capture_logs(module) -> dict:
    """Monkeypatch `module.log` to record every (name → value); returns the dict."""
    logged: dict = {}

    def _fake_log(name, value, **_kw):
        logged[name] = float(value) if hasattr(value, "__float__") else value

    module.log = _fake_log  # shadow the bound LightningModule.log
    return logged


def _fake_trainer(**kw):
    base = dict(
        log_every_n_steps=1, gradient_clip_val=1.0,
        accumulate_grad_batches=1, world_size=1,
    )
    base.update(kw)
    return SimpleNamespace(**base)


_GROUPS = ("frontend", "latent", "m2_predictor", "m4_predictor")


def test_grad_routing_splits_predictor_into_m2_and_m4() -> None:
    """The converged routing tiles FOUR groups (not the joint's lumped
    `predictor`): student_frontend / latent / m2_predictor / m4_predictor — and
    the per-group squared grad norms sum to the global (an exact decomposition)."""
    m = _module()
    out = m._step(_batch().data)
    out["loss"].backward()
    grad_sq, weight_sq, total = m._grad_routing_norms()
    assert set(grad_sq) == set(_GROUPS)
    # m2 and m4 are SEPARATE buckets, each carrying real gradient.
    assert float(grad_sq["m2_predictor"]) > 0.0
    assert float(grad_sq["m4_predictor"]) > 0.0
    summed = sum(float(grad_sq[g]) for g in _GROUPS)
    assert summed == pytest.approx(float(total), rel=1e-5)
    # The four weight buckets also tile the trainable weight norm.
    w_global = sum(
        float(p.detach().pow(2).sum()) for p in m._trainable_parameters()
    )
    assert sum(float(weight_sq[g]) for g in _GROUPS) == pytest.approx(
        w_global, rel=1e-5
    )


def test_on_before_optimizer_step_emits_full_grad_suite() -> None:
    m = _module()
    m._trainer = _fake_trainer()
    m._last_micro_bsz = 2
    opt = torch.optim.AdamW(list(m._trainable_parameters()), lr=1e-3)
    # One real step so the AdamW moment state exists (grad-noise-scale source).
    out = m._step(_batch().data)
    out["loss"].backward()
    opt.step()
    opt.zero_grad()
    out = m._step(_batch().data)
    out["loss"].backward()
    logged = _capture_logs(m)
    m.on_before_optimizer_step(opt)
    expected = {
        "train_mon_grad_l2", "train_mon_grad_ema_l2",
        "train_mon_grad_spike_ratio", "train_mon_grad_spike",
        "train_mon_grad_diverged", "train_mon_nonfinite",
        "train_mon_ema_weight_gap", "train_mon_grad_clipped",
        "train_mon_grad_clip_scale",
        "train_mon_grad_noise_scale", "train_mon_grad_noise_signal",
        "train_mon_grad_noise_var",
    }
    for g in _GROUPS:
        expected |= {
            f"train_mon_grad_l2_{g}", f"train_mon_wnorm_{g}",
            f"train_mon_update_ratio_{g}",
        }
    missing = expected - set(logged)
    assert not missing, f"missing grad metrics: {sorted(missing)}"


def test_true_update_ratio_logged_per_group_across_two_steps() -> None:
    """`‖Δθ‖/‖θ‖` per group: the first hook arms the snapshot, the second (after
    an optimizer.step()) logs all four groups."""
    m = _module()
    m._trainer = _fake_trainer()
    m._last_micro_bsz = 2
    opt = torch.optim.AdamW(list(m._trainable_parameters()), lr=1e-2)
    logged = _capture_logs(m)
    # Step 1: backward + arm snapshot.
    out = m._step(_batch().data)
    out["loss"].backward()
    m.on_before_optimizer_step(opt)   # arms _update_snapshot
    opt.step(); opt.zero_grad()
    # Step 2: backward + measure displacement.
    out = m._step(_batch().data)
    out["loss"].backward()
    m.on_before_optimizer_step(opt)   # logs true_update_ratio_*
    for g in _GROUPS:
        name = f"train_mon_true_update_ratio_{g}"
        assert name in logged, f"missing {name}"
        assert logged[name] > 0.0   # params actually moved


def test_on_train_batch_end_emits_throughput() -> None:
    m = _module()
    logged = _capture_logs(m)
    batch = _batch()
    m.on_train_batch_end(None, batch, 0)   # first call: no prior timestamp
    m.on_train_batch_end(None, batch, 1)   # second: emits step time + rate
    assert "train_mon_step_time_s" in logged
    assert "train_mon_steps_per_sec" in logged
    assert "train_mon_samples_per_sec" in logged
    assert logged["train_mon_samples_per_sec"] > 0.0


def test_step_timing_splits_into_data_wait_and_compute(monkeypatch) -> None:
    """V14_STEP_TIMING ⇒ on_train_batch_end emits data_wait_s + compute_s that sum
    to step_time_s — the host-gap split used to attribute the all-GPU-idle bubble.
    OFF (default) ⇒ neither key is logged."""
    monkeypatch.setenv("V14_STEP_TIMING", "1")
    m = _module()
    logged = _capture_logs(m)
    batch = _batch()
    m.on_train_batch_end(None, batch, 0)          # seeds prev-end timestamp
    m.on_train_batch_start(batch, 1)              # marks compute start (after a wait)
    m.on_train_batch_end(None, batch, 1)
    assert "train_mon_data_wait_s" in logged and "train_mon_compute_s" in logged
    split = logged["train_mon_data_wait_s"] + logged["train_mon_compute_s"]
    assert abs(split - logged["train_mon_step_time_s"]) < 1e-6

    m2 = _module()                                # flag unset ⇒ no split logged
    monkeypatch.delenv("V14_STEP_TIMING", raising=False)
    m2._step_timing = False
    logged2 = _capture_logs(m2)
    m2.on_train_batch_end(None, batch, 0)
    m2.on_train_batch_start(batch, 1)
    m2.on_train_batch_end(None, batch, 1)
    assert "train_mon_data_wait_s" not in logged2
    assert "train_mon_step_time_s" in logged2   # base throughput still emitted


def test_monitor_from_step_emits_rankme_coverage_inputstats() -> None:
    m = _module()
    logged = _capture_logs(m)
    m._monitor_from_step(_batch().data, step_name="train")
    expected = set()
    for band in ("slow", "beta", "hg"):
        expected |= {
            f"train_mon_input_{band}_nonfinite_frac",
            f"train_mon_input_{band}_absmax",
            f"train_mon_input_{band}_mean",
            f"train_mon_input_{band}_std",
        }
    expected |= {
        "train_mon_coverage_active_mean", "train_mon_coverage_active_cv",
        "train_mon_coverage_slot_var", "train_mon_coverage_degenerate_frac",
        "train_mon_coverage_swec_frac", "train_mon_coverage_alarm",
    }
    # RankMe on the latent (key="") AND teacher-frontend (key="frontend_").
    for key in ("", "frontend_"):
        expected |= {
            f"train_mon_{key}rankme", f"train_mon_{key}rankme_normalised",
            f"train_mon_{key}rankme_warn", f"train_mon_{key}rankme_alarm",
            f"train_mon_{key}feat_std_mean", f"train_mon_{key}feat_std_min",
            f"train_mon_{key}feat_cov_offdiag",
        }
    missing = expected - set(logged)
    assert not missing, f"missing forward-tap metrics: {sorted(missing)}"


def test_forward_populates_rank_taps() -> None:
    """The forward stashes the teacher-frontend target + student-latent rep (and
    the per-token valid mask) so the monitor reads them instead of re-forwarding
    (#247). Detached — the monitor must never hold the training graph."""
    m = _module()
    m._step(_batch(B=2, C=6).data)
    taps = m.model.last_rank_taps
    assert {"teacher_frontend", "student_latent", "student_latent_valid"} <= set(taps)
    B, C, S, d = taps["teacher_frontend"].shape
    assert (B, C) == (2, 6)
    assert taps["student_latent"].shape == (B, C, S, d)
    assert taps["student_latent_valid"].shape == (B, C, S)
    assert taps["student_latent_valid"].dtype == torch.bool
    assert not taps["teacher_frontend"].requires_grad
    assert not taps["student_latent"].requires_grad
    # The student-latent tap is masked-context: it writes zeros off the visible
    # set, and the valid mask marks exactly the rows the latent wrote.
    assert bool((~taps["student_latent_valid"]).any()), "M2/M4 masking leaves hidden rows"


def _effective_rank(z: torch.Tensor) -> float:
    """RankMe = exp(Shannon entropy of normalised singular values)."""
    sv = torch.linalg.svdvals(z.float())
    p = sv / (sv.sum() + 1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def _strided_subsample(flat: torch.Tensor, n_max: int) -> torch.Tensor:
    """The OLD (buggy) `_rankme_subsample` — kept here only as the artifact
    reference the regression must out-perform."""
    if flat.shape[0] > n_max:
        stride = flat.shape[0] // n_max
        flat = flat[::stride][:n_max]
    return flat


def test_rankme_subsample_is_batchsize_stable_unlike_stride() -> None:
    """Regression for the frontend-RankMe 'needles' (bs32 collapse, bs8 clean).

    The flat tap is ordered (B, C, S, d) with time innermost. A FIXED stride
    ``n_rows // N_MAX`` phase-locks onto a single time-bin whenever it lands on a
    multiple of the per-electrode token period S — i.e. when ``B·C/N_MAX`` hits an
    integer — annihilating temporal rank. That resonance moves with batch size, so
    bs32 needles where bs8 stays clean. The shipped random subsample has no phase
    to lock onto, so its rank readout is batch-size-independent."""
    m = _module()
    torch.manual_seed(0)
    d, S, C, n_max = 64, 8, 128, m._RANKME_N_MAX  # n_max=4096
    time_basis = torch.eye(S, d)  # S orthogonal directions ⇒ full matrix rank = S

    def build(B: int) -> torch.Tensor:
        s_idx = torch.arange(S).repeat(B * C)  # innermost axis = time, period 1
        return time_basis[s_idx] + 1e-4 * torch.randn(B * C * S, d)

    Z32 = build(32)  # N=32768, stride=8 == S  → resonant (locks to one bin)
    Z8 = build(8)    # N=8192,  stride=2       → non-resonant (keeps 4 bins)
    assert Z32.shape[0] // n_max == S, "test must hit the exact phase-lock resonance"

    # The artifact: strided subsample collapses to ~rank-1 at bs32, far less at bs8.
    er_stride_32 = _effective_rank(_strided_subsample(Z32, n_max))
    er_stride_8 = _effective_rank(_strided_subsample(Z8, n_max))
    assert er_stride_32 < 2.0, "strided bs32 should phase-lock to one time-bin"
    assert er_stride_8 > er_stride_32 + 1.0, "needle must vanish at bs8 (the symptom)"

    # The fix: random subsample retains the full temporal rank, batch-size-stable.
    er_rand_32 = _effective_rank(m._rankme_subsample(Z32))
    er_rand_8 = _effective_rank(m._rankme_subsample(Z8))
    assert er_rand_32 > S - 1, "random subsample keeps all time-bins at bs32"
    assert abs(er_rand_32 - er_rand_8) < 1.0, "random subsample is batch-size-stable"


def test_rankme_subsample_is_deterministic_and_count_capped() -> None:
    """Fixed-seed generator ⇒ same input tensor returns the same subset (resume-
    stable), and the cap is honoured exactly while a small input passes through."""
    m = _module()
    torch.manual_seed(1)
    big = torch.randn(5000, 32)
    a = m._rankme_subsample(big)
    b = m._rankme_subsample(big)
    assert a.shape[0] == m._RANKME_N_MAX
    assert torch.equal(a, b), "subsample must be deterministic across calls"
    small = torch.randn(100, 32)
    assert m._rankme_subsample(small) is small, "no-op below the cap"


def _spy(obj, name: str, calls: list[str]):
    """Wrap a bound method/attr so each call records and still delegates."""
    real = getattr(obj, name)

    def wrapper(*a, **k):
        calls.append(name)
        return real(*a, **k)

    setattr(obj, name, wrapper)


def test_monitor_reuses_taps_without_second_forward() -> None:
    """After a forward, the monitor ranks the stashed taps — it must NOT call
    encode_latent or the teacher frontend a second time (the #245 double-forward
    that ~2.1×'d the step). Still emits both rank panels from the reused taps."""
    m = _module()
    m._step(_batch().data)                     # populates last_rank_taps
    calls: list[str] = []
    _spy(m.model, "encode_latent", calls)
    _spy(m.model.teacher_frontend, "forward", calls)
    logged = _capture_logs(m)
    m._monitor_from_step(_batch().data, step_name="train")
    assert calls == [], f"monitor re-forwarded instead of reusing taps: {calls}"
    assert "train_mon_rankme" in logged and "train_mon_frontend_rankme" in logged


def test_monitor_falls_back_to_encode_when_taps_absent() -> None:
    """No prior forward (empty stash) ⇒ the monitor still works by recomputing a
    dense encode — the fallback path keeps unit tests / val-only calls valid."""
    m = _module()
    m.model.last_rank_taps = {}
    calls: list[str] = []
    _spy(m.model, "encode_latent", calls)
    logged = _capture_logs(m)
    m._monitor_from_step(_batch().data, step_name="train")
    assert calls == ["encode_latent"], "absent taps must fall back to a dense encode"
    assert "train_mon_rankme" in logged


def test_monitor_tap_due_none_falls_back_to_log_cadence() -> None:
    # Default (None) ⇒ gate on the log cadence; with no trainer attached the log
    # cadence is 1, so the monitor tap is due every step.
    m = _module()
    assert m._monitor_every_n_steps is None
    assert all(m._monitor_tap_due(i) for i in (0, 1, 2, 5, 49, 50))


def test_monitor_tap_due_decoupled_cadence() -> None:
    # An explicit cadence thins the (cheap, activation-reusing) tap independently
    # of log_every_n_steps (which keeps loss curves per-step).
    m = _module(monitor_every_n_steps=50)
    assert m._monitor_every_n_steps == 50
    assert [i for i in range(101) if m._monitor_tap_due(i)] == [0, 50, 100]


def test_training_step_gates_monitor_on_heavy_cadence() -> None:
    # The gate is actually wired into training_step: with cadence 50 the extra
    # forward (the step-doubling _monitor_from_step) fires at 0 and 50 but NOT at
    # the per-step batches in between — while loss logging is unaffected.
    m = _module(monitor_every_n_steps=50)
    _capture_logs(m)   # stub self.log (no trainer attached)
    fired: list[str] = []
    m._monitor_from_step = lambda data, *, step_name: fired.append(step_name)  # type: ignore[assignment]
    for i in (0, 1, 2, 49, 50, 51):
        m.training_step(_batch(), i)
    assert fired == ["train", "train"]   # exactly batch_idx 0 and 50


def test_log_losses_emits_joint_canonical_panel_names() -> None:
    """The converged forward dict is logged under the JOINT panel names so the
    live 2STFT wandb panels populate: l_m2→loss_m2, ev_m4→m4_explained_var, …"""
    m = _module()
    logged = _capture_logs(m)
    out = m._step(_batch().data)
    m._log_losses(out, "train", on_step=True, on_epoch=False)
    for name in (
        "train_loss", "train_loss_m2", "train_loss_m4",
        "train_m2_explained_var", "train_m2_target_var",
        "train_m4_explained_var", "train_m4_target_var",
        # collapse triad (joint _log_term_stats parity)
        "train_m2_pred_var", "train_m2_target_norm",
        "train_m2_pred_target_var_ratio",
        "train_m4_pred_var", "train_m4_target_norm",
        "train_m4_pred_target_var_ratio",
    ):
        assert name in logged, f"missing panel metric {name}"


def test_ema_weight_gap_tracks_frontend_only() -> None:
    """The EMA gap is over the frontend (the only EMA-mirrored module). Fresh
    deepcopy → gap 0; after a student step the gap grows > 0."""
    m = _module()
    assert m._ema_weight_gap() == pytest.approx(0.0, abs=1e-6)
    opt = torch.optim.AdamW(list(m.model.student_frontend.parameters()), lr=1e-2)
    out = m._step(_batch().data)
    out["loss"].backward()
    opt.step()
    assert m._ema_weight_gap() > 0.0


def test_grad_ema_l2_buffer_persists_in_state_dict() -> None:
    """The spike-monitor EMA buffer is a registered persistent buffer (survives
    checkpoint round-trips)."""
    m = _module()
    assert "_grad_ema_l2" in m.state_dict()


# ----------------------------------------------------- SDPA backend lever (#249)
def _capture_sdpa_toggles(monkeypatch) -> dict[str, bool]:
    """Pretend CUDA is present and record every enable_*_sdp toggle into a dict."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    seen: dict[str, bool] = {}
    for k in ("math", "flash", "mem_efficient", "cudnn"):
        monkeypatch.setattr(
            torch.backends.cuda, f"enable_{k}_sdp",
            (lambda key: lambda v: seen.__setitem__(key, bool(v)))(k),
        )
    return seen


@pytest.mark.parametrize("name", [None, "", "default", "DEFAULT", "  "])
def test_apply_sdpa_backend_noop_is_byte_identical(name, monkeypatch) -> None:
    """Unset / 'default' must touch NOTHING — the live run shares the stock path."""
    seen = _capture_sdpa_toggles(monkeypatch)
    _apply_sdpa_backend(name)
    assert seen == {}


def test_apply_sdpa_backend_rejects_unknown_even_without_cuda(monkeypatch) -> None:
    """A typo fails fast on CPU too (validation precedes the CUDA gate)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unknown V14_SDPA_BACKEND"):
        _apply_sdpa_backend("flsah")


def test_apply_sdpa_backend_noop_without_cuda(monkeypatch) -> None:
    """No CUDA ⇒ silent no-op for a VALID backend (laptop / CI safe)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _apply_sdpa_backend("cudnn")  # must not raise, must not touch backends


def test_apply_sdpa_backend_cudnn_touches_no_global_flags(monkeypatch) -> None:
    """'cudnn' is a NO-OP in _apply_sdpa_backend — it must not flip any global
    flag. cuDNN is forced per-forward by _sdpa_forward_context (the dispatcher
    rejects cuDNN for the masked+grad latent call when only globally enabled, and
    disabling mem-efficient globally would send cuDNN-declined calls to MATH →
    OOM at the latent's L)."""
    seen = _capture_sdpa_toggles(monkeypatch)
    _apply_sdpa_backend("cudnn")
    assert seen == {}


def test_sdpa_forward_context_cudnn_is_priority_kernel(monkeypatch) -> None:
    """'cudnn' ⇒ a real sdpa_kernel priority context (not nullcontext)."""
    import contextlib as _c
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    ctx = _sdpa_forward_context("cudnn")
    assert not isinstance(ctx, _c.nullcontext)
    # entering/exiting must not raise (the priority list is valid for torch ≥2.6)
    with ctx:
        pass


@pytest.mark.parametrize(
    "name", [None, "", "default", "cudnn_latent", "flash", "efficient", "math"]
)
def test_sdpa_forward_context_non_cudnn_is_nullcontext(name, monkeypatch) -> None:
    """Every non-'cudnn' mode ⇒ nullcontext at the WHOLE-forward scope (global
    flags govern those). 'cudnn_latent' in particular is nullcontext HERE — its
    cuDNN force is scoped inside LatentEncoder, not around the full forward."""
    import contextlib as _c
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert isinstance(_sdpa_forward_context(name), _c.nullcontext)


def test_apply_sdpa_backend_cudnn_latent_touches_no_global_flags(monkeypatch) -> None:
    """'cudnn_latent' is a NO-OP in _apply_sdpa_backend (same reasoning as
    'cudnn'): the scoped force is per-call inside LatentEncoder; flipping global
    flags would route cuDNN-declined calls to MATH and OOM at the latent's L."""
    seen = _capture_sdpa_toggles(monkeypatch)
    _apply_sdpa_backend("cudnn_latent")
    assert seen == {}


def test_sdpa_forward_context_cudnn_nullcontext_without_cuda(monkeypatch) -> None:
    """No CUDA ⇒ even 'cudnn' degrades to nullcontext (CPU CI safe)."""
    import contextlib as _c
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert isinstance(_sdpa_forward_context("cudnn"), _c.nullcontext)


def test_apply_sdpa_backend_flash_only(monkeypatch) -> None:
    seen = _capture_sdpa_toggles(monkeypatch)
    _apply_sdpa_backend("flash")
    assert seen == {
        "math": True, "flash": True, "cudnn": False, "mem_efficient": False,
    }


def test_apply_sdpa_backend_efficient_only(monkeypatch) -> None:
    seen = _capture_sdpa_toggles(monkeypatch)
    _apply_sdpa_backend("efficient")
    assert seen == {
        "math": True, "mem_efficient": True, "cudnn": False, "flash": False,
    }


def test_apply_sdpa_backend_math_only(monkeypatch) -> None:
    seen = _capture_sdpa_toggles(monkeypatch)
    _apply_sdpa_backend("math")
    assert seen == {
        "math": True, "cudnn": False, "flash": False, "mem_efficient": False,
    }


# ----------------------------------------------- profiler spec separator parsing
@pytest.mark.parametrize(
    "spec, builds",
    [
        ("1450:20", True),   # colon survives sbatch --export's comma list-split
        ("1450,20", True),   # legacy comma form still parses
        ("1450", False),     # mangled remnant of a comma-split => fail-soft, no profiler
        ("a:b", False),      # non-int => fail-soft
    ],
)
def test_profile_spec_accepts_colon_and_comma(monkeypatch, spec, builds) -> None:
    """V14_PROFILE_STEPS parses both "wait,active" and "wait:active". The colon
    form is what survives `sbatch --export=VAR=1450,20` (the CLI splits the export
    LIST on commas, truncating the env to "1450"); a single/garbage value must
    fail-soft to no profiler rather than raise. Stubs torch.profiler.profile so the
    test exercises only the parse, not a real profiling session."""
    import torch.profiler as tp_prof

    class _StubProf:
        def start(self) -> None: ...
        def stop(self) -> None: ...

    monkeypatch.setenv("V14_PROFILE_STEPS", spec)
    m = _module()
    monkeypatch.setattr(tp_prof, "profile", lambda *a, **k: _StubProf())
    m.on_train_start()
    assert (m._profiler is not None) is builds
