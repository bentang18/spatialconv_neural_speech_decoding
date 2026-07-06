"""TDD for the LEAN SSLHealthMonitor callback + the V14ConvergedV2 `return_taps`
forward seam: the tap path must not perturb the loss, Family A math (grad spike /
ema_weight_gap / true_update_ratio) is correct, Family B + input-stats run finite
on a tiny batch (and input-stats catches an injected outlier), and the module only
requests taps on a monitor-cadence step."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from speech_decoding.experiments.monitors.ssl_health import SSLHealthMonitor
from speech_decoding.experiments.v14_converged_v2_module import (
    V14ConvergedV2BrainModule,
)
from speech_decoding.models.v14_converged_v2 import (
    V14ConvergedV2,
    bands_for_clip_len,
)
from speech_decoding.models.v14_converged_v2_config import V14ConvergedV2Net

from neuraltrain.optimizers import LightningOptimizer

N_PARCELS = 62

_TAP_KEYS = (
    "_tap_teacher_frontend", "_tap_student_latent",
    "_tap_m2_pred", "_tap_m2_target", "_tap_m4_pred", "_tap_m4_target",
)


def _tiny_model() -> V14ConvergedV2:
    return V14ConvergedV2Net(
        d_model=16, n_heads=4, frontend_layers=1, latent_layers=1,
        m2_pred_layers=1, m4_pred_layers=1, pred_dim=16, n_parcels=N_PARCELS,
    ).build(0, 0)


def _module(**kw) -> V14ConvergedV2BrainModule:
    return V14ConvergedV2BrainModule(
        model=_tiny_model(),
        optim_config=LightningOptimizer(
            optimizer={"name": "AdamW", "lr": 1e-3, "kwargs": {"weight_decay": 0.0}}
        ),
        clip_len_s=5.0, **kw,
    )


def _batch(B: int = 3, *, clip_len_s: float = 5.0):
    torch.manual_seed(0)
    lfs_b, hga_b = bands_for_clip_len(clip_len_s)
    C, K = 6, N_PARCELS
    lfs = torch.randn(B, C, lfs_b.n_freq_bins, lfs_b.n_time_frames)
    hga = torch.randn(B, C, hga_b.n_freq_bins, hga_b.n_time_frames)
    support = torch.zeros(B, C, K)
    for e, lab in enumerate([5, 5, 9, 9, 20, 20]):
        support[:, e, lab] = 1.0
    return SimpleNamespace(data={
        "electrode_tokens_lfs": lfs,
        "electrode_tokens_hga": hga,
        "support": support,
    })


# ---------------------------------------------------------- return_taps seam
def test_return_taps_loss_identical_and_taps_finite():
    """`return_taps=True` returns the right detached tap keys with finite values and
    a loss IDENTICAL (same seed) to `return_taps=False` — proves taps don't perturb."""
    m = _module()
    data = _batch(B=3).data

    def _forward(return_taps: bool):
        torch.manual_seed(7)
        m._mask_gen.manual_seed(0)
        lfs, hga, poe, _coords = m._v2_inputs(data)
        from speech_decoding.models.v14_converged_v2 import active_parcels
        _, membership = active_parcels(poe.cpu())
        bands = bands_for_clip_len(m.clip_len_s)
        m2, tube = m.model.sample_masks(lfs.shape[0], membership, bands, m._mask_gen)
        return m.model(lfs, hga, poe, m2, tube, clip_len_s=m.clip_len_s,
                       return_taps=return_taps)

    out_no = _forward(False)
    out_yes = _forward(True)
    assert torch.equal(out_no["loss"], out_yes["loss"])
    for k in _TAP_KEYS:
        t = out_yes[k]
        assert isinstance(t, torch.Tensor)
        assert not t.requires_grad
        assert torch.isfinite(t).all()
        assert t.shape[-1] == m.model.cfg.d_model
    assert all(k not in out_no for k in _TAP_KEYS)


# ------------------------------------------------------ Family A: grad spike
def test_grad_spike_uses_shared_helper():
    m = _module(monitor_every_n_steps=1)
    cb = SSLHealthMonitor(every_n_steps=1)
    m._step(_batch(B=2).data)["loss"].backward()
    logged: dict[str, float] = {}
    m.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    cb.on_before_optimizer_step(trainer=None, pl_module=m, optimizer=None)
    for k in ("train_mon_grad_l2", "train_mon_grad_ema_l2",
              "train_mon_grad_spike_ratio", "train_mon_grad_spike"):
        assert k in logged and logged[k] == logged[k]  # finite/present
    # EMA buffer seeded to the first grad-L2 (shared helper's step-0 behaviour).
    assert cb._grad_ema_l2 == logged["train_mon_grad_l2"]


# -------------------------------------------------- Family A: ema_weight_gap
def test_ema_weight_gap_zero_then_positive():
    m = _module()
    cb = SSLHealthMonitor()
    # Fresh model: teacher is a deepcopy of the student ⇒ gap exactly 0.
    assert cb._ema_weight_gap(m) == 0.0
    with torch.no_grad():
        next(m.model.frontend.parameters()).add_(1.0)
    assert cb._ema_weight_gap(m) > 0.0


# ----------------------------------------------- Family A: true_update_ratio
def test_true_update_ratio_zero_without_step_positive_after():
    m = _module(monitor_every_n_steps=1)
    cb = SSLHealthMonitor(every_n_steps=1)
    logged: dict[str, float] = {}
    m.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    # Step A: snapshot taken, nothing logged yet (no prior snapshot).
    cb._maybe_log_true_update_ratio(m, step=0)
    assert cb._update_snapshot is not None
    assert not any("true_update_ratio" in k for k in logged)
    # No optimizer step between snapshots ⇒ ratio ~0.
    cb._maybe_log_true_update_ratio(m, step=1)
    for g in ("frontend", "latent", "m2_predictor", "m4_predictor"):
        assert logged[f"train_mon_true_update_ratio_{g}"] == 0.0
    # Perturb the frontend, then measure across the next pair ⇒ >0 for frontend.
    logged.clear()
    cb._maybe_log_true_update_ratio(m, step=2)            # snapshot
    with torch.no_grad():
        next(m.model.frontend.parameters()).add_(0.5)
    cb._maybe_log_true_update_ratio(m, step=3)            # measure
    assert logged["train_mon_true_update_ratio_frontend"] > 0.0


# ------------------------------------------------- Family A: update_cosine
def _run_cos_window(cb, m, logged, step0_perturb, step1_perturb):
    """Drive one 3-step cosine window on the frontend's first param. Returns after
    the window closes at step 2 (cosine logged). `step0_perturb`/`step1_perturb`
    are the signed magnitudes added to the whole param before steps 1 and 2 — i.e.
    Δθ₁ and Δθ₂ directions."""
    p = next(m.model.frontend.parameters())
    cb._maybe_log_update_cosine(m, step=0)               # snapshot θ0
    assert cb._cos_snap is not None and cb._cos_delta_a is None
    with torch.no_grad():
        p.add_(step0_perturb)                            # Δθ₁
    cb._maybe_log_update_cosine(m, step=1)               # store Δθ₁, re-snapshot
    assert cb._cos_delta_a is not None
    assert not any("update_cos" in k for k in logged)    # nothing until window closes
    with torch.no_grad():
        p.add_(step1_perturb)                            # Δθ₂
    cb._maybe_log_update_cosine(m, step=2)               # cosine + reset
    assert cb._cos_snap is None and cb._cos_delta_a is None


def test_update_cosine_aligned_updates_positive():
    """Two same-direction consecutive updates ⇒ cos(Δθ₁, Δθ₂) ≈ +1 (smooth descent).
    Only the perturbed group logs; untouched groups (zero update) are skipped."""
    m = _module(monitor_every_n_steps=1)
    cb = SSLHealthMonitor(every_n_steps=1)
    logged: dict[str, float] = {}
    m.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    _run_cos_window(cb, m, logged, step0_perturb=0.1, step1_perturb=0.1)
    assert logged["train_mon_update_cos_frontend"] > 0.99
    # Untouched groups have Δθ = 0 both steps ⇒ undefined cosine ⇒ not logged.
    for g in ("latent", "m2_predictor", "m4_predictor"):
        assert f"train_mon_update_cos_{g}" not in logged


def test_update_cosine_antialigned_updates_negative():
    """Opposite-direction consecutive updates (the edge-of-stability signature:
    iterate bouncing across a sharp valley) ⇒ cos(Δθ₁, Δθ₂) ≈ −1 = too-hot."""
    m = _module(monitor_every_n_steps=1)
    cb = SSLHealthMonitor(every_n_steps=1)
    logged: dict[str, float] = {}
    m.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    _run_cos_window(cb, m, logged, step0_perturb=0.1, step1_perturb=-0.1)
    assert logged["train_mon_update_cos_frontend"] < -0.99


# ------------------------------------------ Family B + input-stats: run finite
def test_tap_monitors_and_input_stats_finite():
    m = _module(monitor_every_n_steps=1)
    cb = SSLHealthMonitor(every_n_steps=1)
    out = m._step(_batch(B=3).data)
    assert m._last_taps is not None
    logged: dict[str, float] = {}
    m.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    cb._run_input_stats(m, _batch(B=3))
    cb._run_tap_monitors(m, m._last_taps)
    for k in (
        "train_mon_frontend_rankme", "train_mon_frontend_rankme_normalised",
        "train_mon_frontend_feat_std_mean", "train_mon_frontend_feat_std_min",
        "train_mon_rankme", "train_mon_feat_std_mean", "train_mon_feat_std_min",
        "train_mon_m2_explained_var", "train_mon_m2_pred_target_var_ratio",
        "train_mon_m4_explained_var", "train_mon_m4_pred_target_var_ratio",
        "train_mon_input_electrode_tokens_lfs_mean",
        "train_mon_input_electrode_tokens_hga_std",
    ):
        assert k in logged, k
        assert logged[k] == logged[k]  # finite
    _ = out


def test_m2_band_tap_shape_and_composition():
    """The model emits a per-query M2 band tag (0=LFS, 1=HGA) matching the
    `_tap_m2_pred` row count, values in {0,1}, both bands present, and — the
    order-integrity check — a CONSTANT per-electrode HGA count (the M2 mask holds
    out a fixed 1-of-3 LFS group + a fixed HGA time-fraction per electrode)."""
    m = _module()
    data = _batch(B=3).data
    torch.manual_seed(7)
    m._mask_gen.manual_seed(0)
    lfs, hga, poe, _coords = m._v2_inputs(data)
    from speech_decoding.models.v14_converged_v2 import active_parcels
    _, membership = active_parcels(poe.cpu())
    bands = bands_for_clip_len(m.clip_len_s)
    m2, tube = m.model.sample_masks(lfs.shape[0], membership, bands, m._mask_gen)
    out = m.model(lfs, hga, poe, m2, tube, clip_len_s=m.clip_len_s, return_taps=True)

    band = out["_tap_m2_band"]
    assert band.shape == (out["_tap_m2_pred"].shape[0],)
    assert set(band.unique().tolist()) <= {0, 1}
    assert (band == 0).any() and (band == 1).any()          # both bands masked
    # (B,C,n_mask): HGA count per electrode is the static-invariant n_mask_hga.
    B, C = lfs.shape[0], lfs.shape[1]
    per_elec_hga = band.reshape(B, C, -1).sum(-1)           # (B,C)
    assert per_elec_hga.min() == per_elec_hga.max()         # constant ⇒ order intact


def test_m2_per_band_explained_var_and_l1_logged():
    """The monitor splits the M2 explained-var + raw-target L1 by band from the tap.
    Both per-band keys log finite for each band, alongside the aggregate."""
    m = _module(monitor_every_n_steps=1)
    cb = SSLHealthMonitor(every_n_steps=1)
    m._step(_batch(B=3).data)
    assert m._last_taps is not None and "_tap_m2_band" in m._last_taps
    logged: dict[str, float] = {}
    m.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    cb._run_tap_monitors(m, m._last_taps)
    for k in (
        "train_mon_m2_explained_var",                        # aggregate unchanged
        "train_mon_m2_explained_var_lfs", "train_mon_m2_explained_var_hga",
        "train_mon_m2_l1_lfs", "train_mon_m2_l1_hga",
    ):
        assert k in logged, k
        assert logged[k] == logged[k]                        # finite


def test_tap_monitors_survive_global_step_advance():
    """Regression: the module stashes taps in training_step at global_step=k (its
    cadence), but on_train_batch_end runs AFTER the optimiser step at k+1. The
    callback must consume the stashed taps by PRESENCE, not re-derive the cadence
    from the advanced global_step — the off-by-one that silently dropped every tap
    monitor (Family A + input-stats logged on DeltaAI, rankme/feat_std/explained_var
    did NOT). Cadence 50 so the +1 step is genuinely off-cadence."""
    m = _module(monitor_every_n_steps=50)
    cb = SSLHealthMonitor(every_n_steps=50)
    m.train()

    class _FakeTrainer:
        def __init__(self, step):
            self.global_step = step

    # Stash taps at the cadence step (global_step=50 ⇒ due).
    m._trainer = _FakeTrainer(50)  # type: ignore[assignment]
    out = m._step(_batch(B=3).data)
    assert m._last_taps is not None
    # Optimiser has since stepped ⇒ global_step is now 51 (NOT a cadence step).
    m._trainer = _FakeTrainer(51)  # type: ignore[assignment]
    logged: dict[str, float] = {}
    m.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    cb.on_train_batch_end(trainer=m._trainer, pl_module=m, outputs=out,
                          batch=_batch(B=3), batch_idx=0)
    # The whole representation-health family fires despite the advanced step.
    for k in ("train_mon_rankme", "train_mon_frontend_rankme",
              "train_mon_feat_std_mean", "train_mon_frontend_feat_std_mean",
              "train_mon_m2_explained_var", "train_mon_m4_explained_var",
              "train_mon_m2_pred_target_var_ratio",
              "train_mon_m4_pred_target_var_ratio",
              "train_mon_input_electrode_tokens_lfs_absmax",
              "train_mon_input_electrode_tokens_hga_std"):
        assert k in logged and logged[k] == logged[k], k  # present + finite


def test_tap_monitors_fire_once_per_accum_window():
    """Under grad-accum the tap monitors fire only on the LAST micro-batch of the
    window (one rankme SVD per optimiser step, not per micro-batch)."""
    m = _module(monitor_every_n_steps=1)
    cb = SSLHealthMonitor(every_n_steps=1)
    m.train()

    class _FakeTrainer:
        global_step = 0
        accumulate_grad_batches = 4

    m._trainer = _FakeTrainer()  # type: ignore[assignment]
    out = m._step(_batch(B=3).data)
    assert m._last_taps is not None
    for bidx in (0, 1, 2):
        logged: dict[str, float] = {}
        m.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
        cb.on_train_batch_end(trainer=m._trainer, pl_module=m, outputs=out,
                              batch=_batch(B=3), batch_idx=bidx)
        assert not any("rankme" in k for k in logged), f"fired on micro-batch {bidx}"
    # Last micro-batch of the window (batch_idx 3, (3+1)%4==0) ⇒ fires.
    logged = {}
    m.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    cb.on_train_batch_end(trainer=m._trainer, pl_module=m, outputs=out,
                          batch=_batch(B=3), batch_idx=3)
    assert "train_mon_rankme" in logged


def test_input_stats_absmax_catches_outlier():
    m = _module()
    cb = SSLHealthMonitor()
    batch = _batch(B=2)
    batch.data["electrode_tokens_lfs"][0, 0, 0, 0] = 1e6
    logged: dict[str, float] = {}
    m.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    cb._run_input_stats(m, batch)
    assert logged["train_mon_input_electrode_tokens_lfs_absmax"] >= 1e6


# ------------------------------------------------ Family A: grad_noise_scale
def test_grad_noise_scale_finite_and_scaled_by_b_eff():
    """GNS reads AdamW's existing moment EMAs (no extra grads): after a real
    opt.step() the four keys log finite, var/signal ≥ 0, and the reported
    ``..._scale`` equals ``ratio · B_eff`` with B_eff = world·accum·per_rank_bs."""
    m = _module(monitor_every_n_steps=1)
    cb = SSLHealthMonitor(every_n_steps=1)
    opt = torch.optim.AdamW(
        [p for p in m.model.parameters() if p.requires_grad], lr=1e-3
    )
    # One real step populates exp_avg / exp_avg_sq / step in optimizer.state.
    m._step(_batch(B=3).data)["loss"].backward()  # sets m._last_batch_size = 3
    opt.step()
    logged: dict[str, float] = {}
    m.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    trainer = SimpleNamespace(world_size=4, accumulate_grad_batches=2)
    cb._grad_noise_scale(trainer, m, opt)
    for k in ("train_mon_grad_noise_signal", "train_mon_grad_noise_var",
              "train_mon_grad_noise_ratio", "train_mon_grad_noise_scale"):
        assert k in logged and logged[k] == logged[k]  # finite/present
    assert logged["train_mon_grad_noise_signal"] > 0.0
    assert logged["train_mon_grad_noise_var"] >= 0.0
    assert logged["train_mon_grad_noise_ratio"] >= 0.0
    b_eff = 4 * 2 * m._last_batch_size  # world · accum · per_rank_bs (=3)
    assert logged["train_mon_grad_noise_scale"] == (
        logged["train_mon_grad_noise_ratio"] * b_eff
    )


def test_grad_noise_scale_noop_without_optimizer_state():
    """Before any opt.step() the moment EMAs are absent ⇒ signal=0 ⇒ the method
    logs nothing (no divide-by-zero), so the cadence path is safe on step 0."""
    m = _module(monitor_every_n_steps=1)
    cb = SSLHealthMonitor(every_n_steps=1)
    opt = torch.optim.AdamW(
        [p for p in m.model.parameters() if p.requires_grad], lr=1e-3
    )
    logged: dict[str, float] = {}
    m.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    trainer = SimpleNamespace(world_size=1, accumulate_grad_batches=1)
    cb._grad_noise_scale(trainer, m, opt)
    assert not any("grad_noise" in k for k in logged)


# -------------------------------------------------------------- tap cadence
def test_taps_only_requested_on_cadence_step():
    m = _module(monitor_every_n_steps=10)
    m.train()

    class _FakeTrainer:
        def __init__(self, step):
            self.global_step = step

    # On a non-cadence step the module must NOT stash taps.
    m._trainer = _FakeTrainer(3)
    m._step(_batch(B=2).data)
    assert m._last_taps is None
    # On a cadence step it stashes the tap subdict and strips the keys from losses.
    m._trainer = _FakeTrainer(10)
    out = m._step(_batch(B=2).data)
    assert m._last_taps is not None
    # feature taps (N,d) + the 1-D per-query M2 band tag for the per-band split.
    assert set(m._last_taps) == set(_TAP_KEYS) | {"_tap_m2_band"}
    assert not any(k.startswith("_tap_") for k in out)
