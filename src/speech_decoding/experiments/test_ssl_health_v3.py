"""TDD for the v3 SSL-health seam: the ``collect_taps`` forward path returns the
right detached tier-split taps without perturbing the loss, and ``SSLHealthMonitorV3``
emits the approved Family-A (grad/param) and Family-B (tap) metric keys with finite
values on a tiny synthetic session.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from speech_decoding.experiments.monitors.ssl_health_v3 import SSLHealthMonitorV3
from speech_decoding.experiments.test_v14_converged_v3_module import (
    _module,
    _session_batch,
)


def _fake_trainer(*, world_size: int = 1, accum: int = 1) -> SimpleNamespace:
    return SimpleNamespace(world_size=world_size, accumulate_grad_batches=accum)


# ----------------------------------------------------------- collect_taps seam
def test_collect_taps_loss_identical_and_taps_finite() -> None:
    """``collect_taps=True`` returns detached taps with finite values and a loss
    IDENTICAL (same mask generator seed) to ``collect_taps=False`` — proves the tap
    path is a pure read that never perturbs the objective."""
    mod = _module()
    batch = _session_batch(n_rows=3)
    N = batch.bands[0].shape[1]

    def _forward(collect: bool):
        gen = torch.Generator().manual_seed(123)
        return mod.model(
            batch.bands, batch.geom, batch.parcel_id,
            generator=gen, collect_taps=collect,
        )

    out_no = _forward(False)
    out_yes = _forward(True)
    assert torch.equal(out_no.loss, out_yes.loss)
    assert out_no.taps is None and out_yes.taps is not None
    taps = out_yes.taps
    for key in ("enc3", "enc12"):
        t = taps[key]
        assert isinstance(t, torch.Tensor) and not t.requires_grad
        assert t.shape[-1] == 256 and torch.isfinite(t).all()
    # intra tier is always populated (whole shafts round to 0 on a 2-shaft synth);
    # pred/target rows must be paired and same-shape. Under deep-sup (#61) the
    # prediction/target space is the n_levels-concatenated vector (4·256=1024), not
    # the single-tap 256 — the monitor's EV/VR read that full target vector.
    tgt_dim = mod.model.objective.n_levels * 256
    assert taps["pred_intra"].shape == taps["tgt_intra"].shape
    assert taps["pred_intra"].shape[-1] == tgt_dim
    assert not taps["pred_intra"].requires_grad
    # whole/intra rows partition the masked (contact,slot) positions exactly.
    total = taps["pred_whole"].shape[0] + taps["pred_intra"].shape[0]
    assert total == out_yes.n_masked
    assert N >= 2


# ----------------------------------------------------- Family A: grad + ema gap
def test_family_a_grad_and_ema_gap_keys() -> None:
    mod = _module()
    cb = SSLHealthMonitorV3(every_n_steps=1)
    loss = mod.training_step(_session_batch(n_rows=2), 0)
    loss.backward()
    logged: dict[str, float] = {}
    mod.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    cb.on_before_optimizer_step(
        trainer=_fake_trainer(), pl_module=mod, optimizer=None
    )
    for k in (
        "train_mon_grad_l2", "train_mon_grad_ema_l2",
        "train_mon_grad_spike_ratio", "train_mon_grad_spike",
        "train_mon_ema_weight_gap",
    ):
        assert k in logged and logged[k] == logged[k]  # present + finite
    assert cb._grad_ema_l2 == logged["train_mon_grad_l2"]  # step-0 EMA seed


def test_ema_weight_gap_zero_then_positive() -> None:
    mod = _module()
    cb = SSLHealthMonitorV3()
    # Fresh: teacher is a deepcopy of online ⇒ gap exactly 0.
    assert cb._ema_weight_gap(mod) == 0.0
    with torch.no_grad():
        next(mod.model.objective.online.parameters()).add_(1.0)
    assert cb._ema_weight_gap(mod) > 0.0


# ---------------------------------------------- Family A: true_update_ratio
def test_true_update_ratio_zero_without_step_positive_after() -> None:
    mod = _module()
    cb = SSLHealthMonitorV3(every_n_steps=1)
    logged: dict[str, float] = {}
    mod.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    cb._maybe_log_true_update_ratio(mod, step=0)  # snapshot only
    assert cb._update_snapshot is not None
    assert not any("true_update_ratio" in k for k in logged)
    cb._maybe_log_true_update_ratio(mod, step=1)  # no step between ⇒ ~0
    for g in ("online", "predictor"):
        assert logged[f"train_mon_true_update_ratio_{g}"] == 0.0
    logged.clear()
    cb._maybe_log_true_update_ratio(mod, step=2)  # snapshot
    with torch.no_grad():
        next(mod.model.objective.online.parameters()).add_(0.5)
    cb._maybe_log_true_update_ratio(mod, step=3)  # measure
    assert logged["train_mon_true_update_ratio_online"] > 0.0


# ------------------------------------------------ Family A: grad_noise_scale
def test_grad_noise_scale_logs_after_optimizer_step() -> None:
    """B_simple needs AdamW moment EMAs; drive one real AdamW step to populate
    ``optimizer.state`` then assert the four grad-noise keys are finite."""
    mod = _module()
    mod._last_batch_size = 2
    opt = torch.optim.AdamW(mod._trainable_parameters(), lr=1e-3, betas=(0.9, 0.95))
    mod.training_step(_session_batch(n_rows=2), 0).backward()
    opt.step()
    logged: dict[str, float] = {}
    mod.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    cb = SSLHealthMonitorV3()
    cb._grad_noise_scale(_fake_trainer(accum=4), mod, opt)
    for k in (
        "train_mon_grad_noise_signal", "train_mon_grad_noise_var",
        "train_mon_grad_noise_ratio", "train_mon_grad_noise_scale",
    ):
        assert k in logged and logged[k] == logged[k]
    # B_eff scaling: scale = ratio * (world 1 * accum 4 * batch 2) = ratio * 8.
    assert abs(logged["train_mon_grad_noise_scale"]
               - logged["train_mon_grad_noise_ratio"] * 8.0) < 1e-4


# --------------------------------------------------------- Family B: tap keys
def test_family_b_tap_keys_finite() -> None:
    """After a monitor-cadence training_step (taps stashed on the module),
    ``on_train_batch_end`` emits the rankme/feat_std depth keys and the intra-tier
    EV/var-ratio/L1 keys, all finite."""
    mod = _module()
    mod.training_step(_session_batch(n_rows=3), 0)
    assert mod._last_taps is not None  # step 0 is a cadence step
    logged: dict[str, float] = {}
    mod.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    cb = SSLHealthMonitorV3()
    cb.on_train_batch_end(
        trainer=_fake_trainer(), pl_module=mod,
        outputs=None, batch=None, batch_idx=0,
    )
    for k in (
        "train_mon_enc3_rankme", "train_mon_enc3_feat_std_mean",
        "train_mon_enc12_rankme", "train_mon_enc12_feat_std_min",
        "train_mon_intra_explained_var", "train_mon_intra_pred_target_var_ratio",
        "train_mon_intra_l1",
    ):
        assert k in logged and logged[k] == logged[k], k


def test_family_b_noop_without_taps() -> None:
    """No stashed taps (off-cadence step) ⇒ ``on_train_batch_end`` logs nothing —
    the <5% budget guarantee: Family B is pure off-cost when not due."""
    mod = _module()
    mod._last_taps = None
    logged: dict[str, float] = {}
    mod.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    cb = SSLHealthMonitorV3()
    cb.on_train_batch_end(
        trainer=_fake_trainer(), pl_module=mod,
        outputs=None, batch=None, batch_idx=0,
    )
    assert not logged
