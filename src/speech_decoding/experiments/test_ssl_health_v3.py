"""TDD for the v3 SSL-health seam: the ``collect_taps`` forward path returns the
right detached tier-split taps without perturbing the loss, and ``SSLHealthMonitorV3``
emits the approved Family-A (grad/param) and Family-B (tap) metric keys with finite
values on a tiny synthetic session.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from speech_decoding.experiments.monitors.ssl_health_v3 import (
    _ROUTING_GROUPS,
    _group,
    SSLHealthMonitorV3,
)
from speech_decoding.experiments.test_v14_converged_v3_module import (
    _module,
    _session_batch,
)


def _fake_trainer(
    *, world_size: int = 1, accum: int = 1, grad_clip: float = 0.0
) -> SimpleNamespace:
    return SimpleNamespace(
        world_size=world_size, accumulate_grad_batches=accum,
        gradient_clip_val=grad_clip,
    )


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
    # block-3 tap removed (2026-07-10) — only the terminal encoder tap remains.
    assert "enc3" not in taps
    t = taps["enc12"]
    assert isinstance(t, torch.Tensor) and not t.requires_grad
    assert t.shape[-1] == 256 and torch.isfinite(t).all()
    # r4 per-band JEPA scalar taps (the whole/intra tier split was retired 2026-07-15):
    # explained-var / pred-target var-ratio / L1 reduced INSIDE the objective, one 0-dim
    # scalar per (band, metric) over the margin-gated scored tokens.
    for band in ("slow", "mid", "hga"):
        for metric in ("explained_var", "pred_target_var_ratio", "l1"):
            v = taps[f"jepa_{band}_{metric}"]
            assert isinstance(v, torch.Tensor) and v.ndim == 0 and not v.requires_grad
            assert torch.isfinite(v).all()
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


def test_grad_clip_hit_tracker() -> None:
    """With ``gradient_clip_val`` set, the monitor logs clip_hit (0/1) + clip_ratio
    (grad_l2/clip); a huge clip is never hit, a tiny clip always is. Off when clip=0."""
    mod = _module()
    cb = SSLHealthMonitorV3(every_n_steps=1)
    mod.training_step(_session_batch(n_rows=2), 0).backward()
    logged: dict[str, float] = {}
    mod.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    cb.on_before_optimizer_step(
        trainer=_fake_trainer(grad_clip=1e9), pl_module=mod, optimizer=None
    )
    assert logged["train_mon_grad_clip_hit"] == 0.0  # norm ≪ 1e9
    assert logged["train_mon_grad_clip_ratio"] > 0.0
    logged.clear()
    cb.on_before_optimizer_step(
        trainer=_fake_trainer(grad_clip=1e-9), pl_module=mod, optimizer=None
    )
    assert logged["train_mon_grad_clip_hit"] == 1.0  # norm ≫ 1e-9
    # clipping off ⇒ no clip keys
    logged.clear()
    cb.on_before_optimizer_step(
        trainer=_fake_trainer(grad_clip=0.0), pl_module=mod, optimizer=None
    )
    assert not any("grad_clip" in k for k in logged)


def test_true_update_ratio_logged_after_one_step() -> None:
    """Snapshot θ at a cadence step, take a real optimizer step, then the next
    ``on_before_optimizer_step`` logs ``‖Δθ‖/‖θ‖`` per group (>0). First call logs
    nothing (no prior snapshot); r5-only ``mae_head`` group is absent on this arm."""
    mod = _module()
    cb = SSLHealthMonitorV3(every_n_steps=1)
    opt = torch.optim.SGD(mod.model.parameters(), lr=0.1)
    mod.training_step(_session_batch(n_rows=2), 0).backward()
    logged: dict[str, float] = {}
    mod.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    cb.on_before_optimizer_step(trainer=_fake_trainer(), pl_module=mod, optimizer=opt)
    assert not any("true_update_ratio" in k for k in logged)  # nothing to diff yet
    opt.step()  # move θ by exactly one update
    logged.clear()
    cb.on_before_optimizer_step(trainer=_fake_trainer(), pl_module=mod, optimizer=opt)
    for name in ("online", "predictor"):
        k = f"train_mon_true_update_ratio_{name}"
        assert k in logged and logged[k] > 0.0, k


def test_band_names_frontend_aware() -> None:
    """r5 (early_fusion) input tripwire labels its two streams hga/lfs; r4 keeps
    slow/mid/hga. A regression guard for the mislabel that read HGA as ``slow``."""
    cb = SSLHealthMonitorV3()
    def _mod(early: bool) -> SimpleNamespace:
        return SimpleNamespace(model=SimpleNamespace(
            objective=SimpleNamespace(early_fusion=early)))
    assert cb._band_names(_mod(False)) == ("slow", "mid", "hga")
    assert cb._band_names(_mod(True)) == ("hga", "lfs")


def test_update_ratio_routes_nofusion_per_stream_heads() -> None:
    """v3r5nf exposes ``mae_head_hga``/``mae_head_lfs`` (r5's ``mae_head_r5`` is None on this
    arm), so the update-ratio monitor must route to the two per-stream heads — otherwise the
    MAE-head update ratio silently vanishes on the no-fusion arm."""
    assert "mae_head_hga" in _ROUTING_GROUPS and "mae_head_lfs" in _ROUTING_GROUPS
    hga, lfs = torch.nn.Linear(4, 8), torch.nn.Linear(4, 2)
    nf = SimpleNamespace(model=SimpleNamespace(objective=SimpleNamespace(
        mae_head_hga=hga, mae_head_lfs=lfs, mae_head_r5=None)))
    assert _group(nf, "mae_head_hga") is hga
    assert _group(nf, "mae_head_lfs") is lfs
    assert _group(nf, "mae_head") is None  # nf has no fused r5 head
    # r5-fused / r4 arms lack the per-stream heads ⇒ those groups resolve to None (skipped).
    fused = SimpleNamespace(model=SimpleNamespace(objective=SimpleNamespace(
        mae_head_r5=torch.nn.Linear(4, 10))))
    assert _group(fused, "mae_head_hga") is None
    assert _group(fused, "mae_head_lfs") is None
    assert _group(fused, "mae_head") is fused.model.objective.mae_head_r5


def test_ema_weight_gap_zero_then_positive() -> None:
    mod = _module()
    cb = SSLHealthMonitorV3()
    # Fresh: teacher is a deepcopy of online ⇒ gap exactly 0.
    assert cb._ema_weight_gap(mod) == 0.0
    with torch.no_grad():
        next(mod.model.objective.online.parameters()).add_(1.0)
    assert cb._ema_weight_gap(mod) > 0.0


# --------------------------------------------------------- Family B: tap keys
def test_family_b_tap_keys_finite() -> None:
    """After a monitor-cadence training_step (taps stashed on the module),
    ``on_train_batch_end`` emits the rankme/feat_std depth keys and the per-band
    JEPA EV/var-ratio/L1 keys, all finite."""
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
    keys = [
        "train_mon_enc12_rankme", "train_mon_enc12_feat_std_mean",
        "train_mon_enc12_feat_std_min",
    ]
    for band in ("slow", "mid", "hga"):
        keys += [
            f"train_mon_jepa_{band}_explained_var",
            f"train_mon_jepa_{band}_pred_target_var_ratio",
            f"train_mon_jepa_{band}_l1",
        ]
    for k in keys:
        assert k in logged and logged[k] == logged[k], k
    # block-3 tap keys are gone (2026-07-10).
    assert not any(key.startswith("train_mon_enc3_") for key in logged)


def test_input_tripwire_flags_nonfinite_band() -> None:
    """The per-band input tripwire emits nonfinite_frac/absmax/mean/std for every band
    and catches a NaN injected into one band (frac>0 there, 0 for the clean bands)."""
    mod = _module()
    batch = _session_batch(n_rows=3)
    batch.bands[2][0, 0, 0, 0] = float("nan")  # corrupt one HGA token
    logged: dict[str, float] = {}
    mod.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    cb = SSLHealthMonitorV3(every_n_steps=1)
    cb.on_train_batch_end(
        trainer=_fake_trainer(), pl_module=mod,
        outputs=None, batch=batch, batch_idx=0,
    )
    for band in ("slow", "mid", "hga"):
        for suffix in ("nonfinite_frac", "absmax", "mean", "std"):
            assert f"train_mon_input_{band}_{suffix}" in logged, f"{band}_{suffix}"
    assert logged["train_mon_input_hga_nonfinite_frac"] > 0.0
    assert logged["train_mon_input_slow_nonfinite_frac"] == 0.0


def test_input_tripwire_noop_without_batch() -> None:
    """batch=None (the Family-B no-tap tests pass it) ⇒ the tripwire logs nothing."""
    mod = _module()
    mod._last_taps = None
    logged: dict[str, float] = {}
    mod.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    SSLHealthMonitorV3().on_train_batch_end(
        trainer=_fake_trainer(), pl_module=mod,
        outputs=None, batch=None, batch_idx=0,
    )
    assert not logged


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
