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
    N_PARCELS,
    V14ConvergedV3Module,
    V3ConvergedModel,
    _module,
    _optim_config,
    _session_batch,
    _session_batch_with_stats,
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
    # block-3 tap removed (2026-07-10) — only the terminal encoder tap remains.
    assert "enc3" not in taps
    t = taps["enc12"]
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
        "train_mon_enc12_rankme", "train_mon_enc12_feat_std_mean",
        "train_mon_enc12_feat_std_min",
        "train_mon_intra_explained_var", "train_mon_intra_pred_target_var_ratio",
        "train_mon_intra_l1",
    ):
        assert k in logged and logged[k] == logged[k], k
    # block-3 tap keys are gone (2026-07-10).
    assert not any(key.startswith("train_mon_enc3_") for key in logged)


def test_family_b_perceiver_latent_health_keys_finite() -> None:
    """Secondary active + monitor-cadence step ⇒ the perceiver processed-latent bank is
    shipped as the ``perc_lat`` tap and the callback logs RankMe / feat_std (shared
    ``_rank_and_std`` path) PLUS dead-frac and latent-latent cosine, all finite and in range.
    Absent from the JEPA-only ``_module()`` run (perceiver frozen, never forwarded)."""
    model = V3ConvergedModel(n_parcels=N_PARCELS)
    mod = V14ConvergedV3Module(
        model=model, optim_config=_optim_config(weight_decay=0.04),
        secondary_active=True,
    )
    mod.training_step(_session_batch_with_stats(n_rows=3), 0)  # step 0 = cadence
    assert "perc_lat" in (mod._last_taps or {})
    logged: dict[str, float] = {}
    mod.log = lambda k, v, **kw: logged.__setitem__(k, float(v))  # type: ignore[assignment]
    SSLHealthMonitorV3().on_train_batch_end(
        trainer=_fake_trainer(), pl_module=mod, outputs=None, batch=None, batch_idx=0,
    )
    for k in (
        "train_mon_perc_lat_rankme", "train_mon_perc_lat_rankme_normalised",
        "train_mon_perc_lat_feat_std_mean", "train_mon_perc_lat_feat_std_min",
        "train_mon_perc_lat_dead_frac",
        "train_mon_perc_lat_cos_mean", "train_mon_perc_lat_cos_pct95",
    ):
        assert k in logged and logged[k] == logged[k], k  # present + finite
    assert 0.0 <= logged["train_mon_perc_lat_dead_frac"] <= 1.0
    assert -1.0 <= logged["train_mon_perc_lat_cos_mean"] <= 1.0

    # JEPA-only module never forwards the perceiver ⇒ no perc_lat tap, no perc keys.
    jepa = _module()
    jepa.training_step(_session_batch(n_rows=3), 0)
    assert "perc_lat" not in (jepa._last_taps or {})


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
