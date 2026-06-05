"""Tests for the collapse/divergence guard (#54).

Driven with lightweight fakes — the callback reads
``trainer.callback_metrics`` / ``global_step`` / ``current_epoch`` /
``sanity_checking`` / ``world_size`` / ``global_rank`` /
``strategy.reduce_boolean_decision`` and never calls ``pl_module``, so no
real Lightning trainer is needed.
"""

from __future__ import annotations

import json
import math
import typing as tp

import pytest
import torch

from speech_decoding.experiments.collapse_guard import (
    CollapseGuardCallback,
    CollapseStop,
)


class _FakeStrategy:
    """Stand-in for ``trainer.strategy``.

    ``reduce_boolean_decision(decision, all=...)`` emulates the real
    collective: with ``all=False`` (OR) the result is the local decision
    OR-ed with ``peer_decision`` (a fixed bool simulating what other ranks
    voted). On a single device ``peer_decision=False`` makes it a
    passthrough, matching the base ``Strategy`` no-op.
    """

    def __init__(self, peer_decision: bool = False) -> None:
        self.peer_decision = peer_decision
        self.calls = 0

    def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool:  # noqa: A002
        self.calls += 1
        return (decision or self.peer_decision) if not all else (
            decision and self.peer_decision
        )


class _FakeTrainer:
    def __init__(
        self,
        metrics: dict[str, tp.Any] | None = None,
        *,
        global_step: int = 0,
        current_epoch: int = 0,
        sanity_checking: bool = False,
        world_size: int = 1,
        global_rank: int = 0,
        strategy: _FakeStrategy | None = None,
    ) -> None:
        self.callback_metrics = metrics or {}
        self.global_step = global_step
        self.current_epoch = current_epoch
        self.sanity_checking = sanity_checking
        self.world_size = world_size
        self.global_rank = global_rank
        self.strategy = strategy or _FakeStrategy()

    def print(self, *args: tp.Any, **kwargs: tp.Any) -> None:  # noqa: A003
        # Real Lightning Trainer.print is rank-zero stdout; the advisory
        # soft-warn log calls it. Tests just no-op.
        pass


def _drive_val(
    guard: CollapseGuardCallback,
    metric_seq: list[dict[str, tp.Any]],
    *,
    strategy: _FakeStrategy | None = None,
    world_size: int = 1,
) -> None:
    """Feed a sequence of per-check metric dicts through the val hook.

    Reuses one strategy across checks so an OR-emulating peer vote is
    stable for the whole sequence.
    """
    strat = strategy or _FakeStrategy()
    for epoch, metrics in enumerate(metric_seq):
        trainer = _FakeTrainer(
            metrics,
            current_epoch=epoch,
            global_step=epoch * 100,
            world_size=world_size,
            strategy=strat,
        )
        guard.on_validation_epoch_end(trainer, None)


# -- per-batch non-finite -----------------------------------------------


def test_nonfinite_train_loss_aborts_immediately() -> None:
    guard = CollapseGuardCallback()
    trainer = _FakeTrainer()
    with pytest.raises(CollapseStop, match="nonfinite_train_loss"):
        guard.on_train_batch_end(
            trainer, None, torch.tensor(float("nan")), None, 0
        )


def test_inf_train_loss_in_dict_output_aborts() -> None:
    guard = CollapseGuardCallback()
    trainer = _FakeTrainer()
    with pytest.raises(CollapseStop, match="nonfinite_train_loss"):
        guard.on_train_batch_end(
            trainer, None, {"loss": torch.tensor(float("inf"))}, None, 0
        )


def test_finite_train_loss_does_not_abort() -> None:
    guard = CollapseGuardCallback()
    trainer = _FakeTrainer()
    # exact-zero loss (the valid n_masked==0 contract) must not trip.
    guard.on_train_batch_end(trainer, None, torch.tensor(0.0), None, 0)
    guard.on_train_batch_end(trainer, None, {"loss": torch.tensor(1.5)}, None, 0)


def test_grad_diverged_flag_aborts() -> None:
    guard = CollapseGuardCallback()
    trainer = _FakeTrainer({"train_mon_grad_diverged": torch.tensor(1.0)})
    with pytest.raises(CollapseStop, match="grad_diverged"):
        guard.on_train_batch_end(trainer, None, torch.tensor(1.0), None, 0)


def test_grad_diverged_writes_diagnosis_json(tmp_path) -> None:
    # The per-batch (train) abort path must also drop a diagnosis artifact,
    # not only the validation path.
    guard = CollapseGuardCallback(artifact_root=tmp_path)
    trainer = _FakeTrainer({"train_mon_grad_diverged": torch.tensor(1.0)})
    with pytest.raises(CollapseStop, match="grad_diverged"):
        guard.on_train_batch_end(trainer, None, torch.tensor(1.0), None, 0)
    out = tmp_path / "collapse_diagnosis.json"  # single-device: no rank suffix
    assert out.is_file()
    payload = json.loads(out.read_text())
    assert payload["verdict"]["criterion"] == "grad_diverged"


def test_grad_healthy_flag_does_not_abort() -> None:
    guard = CollapseGuardCallback()
    trainer = _FakeTrainer({"train_mon_grad_diverged": 0.0})
    guard.on_train_batch_end(trainer, None, torch.tensor(1.0), None, 0)


def test_unknown_output_type_is_ignored() -> None:
    guard = CollapseGuardCallback()
    trainer = _FakeTrainer()
    # A non-tensor/dict output (e.g. None) must not crash the hook.
    guard.on_train_batch_end(trainer, None, None, None, 0)


def test_nonfinite_outputs_loss_ignored_under_ddp() -> None:
    # Under DDP the per-rank outputs loss is rank-local; the grad flag is
    # the rank-safe NaN catch, so a non-finite outputs loss alone must NOT
    # abort (it could deadlock surviving ranks).
    guard = CollapseGuardCallback()
    trainer = _FakeTrainer(world_size=2)
    guard.on_train_batch_end(trainer, None, torch.tensor(float("nan")), None, 0)


def test_grad_diverged_aborts_under_ddp() -> None:
    # The grad flag is DDP-averaged (identical on every rank) → still the
    # authoritative catch under DDP.
    guard = CollapseGuardCallback()
    trainer = _FakeTrainer({"train_mon_grad_diverged": 1.0}, world_size=2)
    with pytest.raises(CollapseStop, match="grad_diverged"):
        guard.on_train_batch_end(trainer, None, torch.tensor(1.0), None, 0)


# -- per-check soft criteria --------------------------------------------


def test_sustained_m4_rankme_alarm_aborts() -> None:
    guard = CollapseGuardCallback(warmup_checks=0, sustain_checks=2)
    with pytest.raises(CollapseStop, match="val_mon_rankme_alarm"):
        _drive_val(
            guard,
            [{"val_mon_rankme_alarm": 1.0}, {"val_mon_rankme_alarm": 1.0}],
        )


def test_sustained_frontend_rankme_alarm_aborts() -> None:
    guard = CollapseGuardCallback(warmup_checks=0, sustain_checks=2)
    with pytest.raises(CollapseStop, match="val_mon_frontend_rankme_alarm"):
        _drive_val(
            guard,
            [
                {"val_mon_frontend_rankme_alarm": 1.0},
                {"val_mon_frontend_rankme_alarm": 1.0},
            ],
        )


def test_sustained_coverage_alarm_aborts() -> None:
    guard = CollapseGuardCallback(warmup_checks=0, sustain_checks=2)
    with pytest.raises(CollapseStop, match="coverage"):
        _drive_val(
            guard,
            [{"val_mon_coverage_alarm": 1.0}, {"val_mon_coverage_alarm": 1.0}],
        )


def test_sustained_no_masking_aborts() -> None:
    guard = CollapseGuardCallback(warmup_checks=0, sustain_checks=2)
    with pytest.raises(CollapseStop, match="no_masking"):
        _drive_val(guard, [{"val_n_masked": 0.0}, {"val_n_masked": 0.0}])


def test_n_masked_positive_does_not_abort() -> None:
    guard = CollapseGuardCallback(warmup_checks=0, sustain_checks=2)
    _drive_val(guard, [{"val_n_masked": 42.0}, {"val_n_masked": 42.0}])


def test_sustained_loss_blowup_aborts() -> None:
    guard = CollapseGuardCallback(warmup_checks=0, sustain_checks=2)
    with pytest.raises(CollapseStop, match="loss_blowup"):
        _drive_val(
            guard,
            # epoch 0 establishes best=1.0; epochs 1-2 are 50× → abort.
            [{"val_loss": 1.0}, {"val_loss": 50.0}, {"val_loss": 50.0}],
        )


def test_loss_within_band_does_not_abort() -> None:
    guard = CollapseGuardCallback(warmup_checks=0, sustain_checks=2)
    # 3× over best is normal fluctuation, well under the 10× factor.
    _drive_val(guard, [{"val_loss": 1.0}, {"val_loss": 3.0}, {"val_loss": 3.0}])


def test_loss_blowup_suppressed_when_best_below_min_floor() -> None:
    # best dips to 1e-4 (below BLOWUP_MIN_FLOOR=1e-3); a 100× jump to 1e-2 is
    # absolutely tiny → the ratio test must stay disarmed (no abort), else a
    # healthy loss that legitimately approaches zero would false-trip.
    guard = CollapseGuardCallback(warmup_checks=0, sustain_checks=2)
    _drive_val(guard, [{"val_loss": 1e-4}, {"val_loss": 1e-2}, {"val_loss": 1e-2}])
    assert guard._streak["loss_blowup"] == 0  # noqa: SLF001 — never armed below floor


def test_loss_blowup_fires_when_best_just_above_min_floor() -> None:
    # best=2e-3 sits just above the 1e-3 floor → the ratio test IS armed and a
    # 50× jump still aborts. Pins the floor as a selective boundary, not a
    # blanket mute of small-loss runs.
    guard = CollapseGuardCallback(warmup_checks=0, sustain_checks=2)
    with pytest.raises(CollapseStop, match="loss_blowup"):
        _drive_val(guard, [{"val_loss": 2e-3}, {"val_loss": 0.1}, {"val_loss": 0.1}])


def test_nonfinite_val_loss_aborts_immediately() -> None:
    guard = CollapseGuardCallback(warmup_checks=5, sustain_checks=5)
    # Even inside warm-up, a NaN val loss is not debounced.
    with pytest.raises(CollapseStop, match="nonfinite_val_loss"):
        _drive_val(guard, [{"val_loss": float("nan")}])


def test_sustained_rankme_warn_is_advisory_by_default() -> None:
    # 2026-06-05 (gate-47722655 diagnosis): the soft warn (<0.5) is ADVISORY
    # by default. A sustained warn that never crosses the 0.25 alarm must NOT
    # abort — its 0.5 line is a DINOv3 image-ViT value, below which this iEEG
    # front-end is born healthy (~0.32). It still streaks (for the diagnosis
    # JSON / advisory log), it just does not raise.
    guard = CollapseGuardCallback(
        warmup_checks=0, sustain_checks=2, warn_sustain_checks=3
    )
    _drive_val(guard, [{"val_mon_rankme_warn": 1.0}] * 5)  # no raise
    assert guard._streak["val_mon_rankme_warn"] == 5  # noqa: SLF001 — still streaks


def test_sustained_rankme_warn_aborts_when_warn_aborts_enabled() -> None:
    # Opt-in: warn_aborts=True restores the early abort on a sustained slow
    # decline (after the longer warn sustain).
    guard = CollapseGuardCallback(
        warmup_checks=0, sustain_checks=2, warn_sustain_checks=3, warn_aborts=True
    )
    with pytest.raises(CollapseStop, match="val_mon_rankme_warn"):
        _drive_val(guard, [{"val_mon_rankme_warn": 1.0}] * 3)


def test_rankme_warn_below_warn_sustain_is_healthy() -> None:
    # With warn_aborts=True, two warn checks < warn_sustain=4 → no abort.
    guard = CollapseGuardCallback(
        warmup_checks=0, sustain_checks=2, warn_sustain_checks=4, warn_aborts=True
    )
    _drive_val(guard, [{"val_mon_rankme_warn": 1.0}, {"val_mon_rankme_warn": 1.0}])


def test_rankme_warn_one_below_sustain_at_same_threshold_is_healthy() -> None:
    # Boundary pin (warn_aborts=True): at warn_sustain=3, 3 warns abort
    # (test_sustained_rankme_warn_aborts_when_warn_aborts_enabled) but exactly
    # 2 must NOT — so a >=/> off-by-one in the warn streak is caught.
    guard = CollapseGuardCallback(
        warmup_checks=0, sustain_checks=2, warn_sustain_checks=3, warn_aborts=True
    )
    _drive_val(guard, [{"val_mon_rankme_warn": 1.0}, {"val_mon_rankme_warn": 1.0}])


# -- debounce semantics --------------------------------------------------


def test_transient_alarm_resets_streak() -> None:
    guard = CollapseGuardCallback(warmup_checks=0, sustain_checks=2)
    # bad, healthy, bad → streak never reaches 2.
    _drive_val(
        guard,
        [
            {"val_mon_rankme_alarm": 1.0},
            {"val_mon_rankme_alarm": 0.0},
            {"val_mon_rankme_alarm": 1.0},
        ],
    )


def test_warmup_checks_are_ignored() -> None:
    guard = CollapseGuardCallback(warmup_checks=2, sustain_checks=1)
    # First two checks are warm-up; alarms there must not count.
    _drive_val(
        guard,
        [{"val_mon_rankme_alarm": 1.0}, {"val_mon_rankme_alarm": 1.0}],
    )


def test_alarm_after_warmup_aborts() -> None:
    guard = CollapseGuardCallback(warmup_checks=2, sustain_checks=1)
    with pytest.raises(CollapseStop, match="val_mon_rankme_alarm"):
        _drive_val(
            guard,
            [
                {"val_mon_rankme_alarm": 1.0},  # warm-up check 1 (n=1)
                {"val_mon_rankme_alarm": 1.0},  # warm-up check 2 (n=2, 2>2 False)
                {"val_mon_rankme_alarm": 1.0},  # n=3 > 2 → streak=1 → abort
            ],
        )


def test_warmup_min_step_suppresses_alarm_during_lr_warmup() -> None:
    # #67: a hard alarm while global_step < warmup_min_step must NOT abort —
    # the representation is legitimately low-rank while the LR ramps.
    # _drive_val sets global_step = epoch*100 → 0, 100, 200 all < 250.
    guard = CollapseGuardCallback(
        warmup_checks=0, sustain_checks=1, warmup_min_step=250
    )
    _drive_val(guard, [{"val_mon_rankme_alarm": 1.0}] * 3)  # no raise


def test_alarm_after_warmup_min_step_aborts() -> None:
    # Past the LR-warmup window (epoch 3 → global_step 300 >= 250) the same
    # alarm counts and aborts. Pins warmup_min_step as a step gate, not a mute.
    guard = CollapseGuardCallback(
        warmup_checks=0, sustain_checks=1, warmup_min_step=250
    )
    with pytest.raises(CollapseStop, match="val_mon_rankme_alarm"):
        _drive_val(guard, [{"val_mon_rankme_alarm": 1.0}] * 4)


def test_alarm_fraction_below_threshold_is_healthy() -> None:
    guard = CollapseGuardCallback(
        warmup_checks=0, sustain_checks=2, alarm_fraction=0.5
    )
    # 0.3 of steps alarmed < 0.5 majority → check is healthy.
    _drive_val(
        guard,
        [{"val_mon_rankme_alarm": 0.3}, {"val_mon_rankme_alarm": 0.3}],
    )


def test_sanity_check_epoch_is_skipped() -> None:
    guard = CollapseGuardCallback(warmup_checks=0, sustain_checks=1)
    trainer = _FakeTrainer({"val_mon_rankme_alarm": 1.0}, sanity_checking=True)
    guard.on_validation_epoch_end(trainer, None)  # must not raise or count
    # A real check with the same alarm now needs its own full streak.
    assert guard._n_checks == 0  # noqa: SLF001 — white-box debounce check


def test_missing_keys_do_not_crash() -> None:
    guard = CollapseGuardCallback(warmup_checks=0, sustain_checks=2)
    # A phase that logs none of the watched keys (e.g. an empty val agg).
    _drive_val(guard, [{}, {}, {}])


# -- DDP rank consensus --------------------------------------------------


def test_peer_rank_abort_propagates_locally() -> None:
    # This rank sees nothing wrong, but a PEER votes collapse → the OR
    # reduction must make THIS rank raise too (else the peer hangs).
    guard = CollapseGuardCallback(warmup_checks=0, sustain_checks=2)
    strat = _FakeStrategy(peer_decision=True)
    with pytest.raises(CollapseStop, match="consensus abort"):
        _drive_val(guard, [{"val_loss": 1.0}], strategy=strat, world_size=2)
    assert strat.calls == 1  # reduce called once per check, symmetrically


def test_local_abort_still_writes_own_verdict_under_ddp(tmp_path) -> None:
    # When THIS rank is the one that detected collapse, it aborts with its
    # own verdict (not the generic consensus message) and rank-suffixes
    # the diagnosis file.
    guard = CollapseGuardCallback(
        artifact_root=tmp_path, warmup_checks=0, sustain_checks=1
    )
    strat = _FakeStrategy(peer_decision=False)
    with pytest.raises(CollapseStop, match="coverage"):
        _drive_val(
            guard,
            [{"val_mon_coverage_alarm": 1.0}],
            strategy=strat,
            world_size=2,
        )
    assert (tmp_path / "collapse_diagnosis_rank0.json").is_file()


def test_reduce_called_every_check_even_when_healthy() -> None:
    # The collective MUST be called on every check on every rank, healthy
    # or not — otherwise ranks desync and NCCL deadlocks.
    guard = CollapseGuardCallback(warmup_checks=0, sustain_checks=2)
    strat = _FakeStrategy(peer_decision=False)
    _drive_val(
        guard,
        [{"val_loss": 1.0}, {"val_loss": 1.1}, {"val_loss": 1.05}],
        strategy=strat,
        world_size=2,
    )
    assert strat.calls == 3


# -- diagnosis artifact --------------------------------------------------


def test_diagnosis_json_written_on_abort(tmp_path) -> None:
    guard = CollapseGuardCallback(
        artifact_root=tmp_path, warmup_checks=0, sustain_checks=1
    )
    with pytest.raises(CollapseStop):
        _drive_val(guard, [{"val_mon_coverage_alarm": 1.0}])
    out = tmp_path / "collapse_diagnosis.json"  # single-device: no suffix
    assert out.is_file()
    payload = json.loads(out.read_text())
    assert payload["verdict"]["criterion"] == "coverage"
    assert "recent_history" in payload
    assert payload["verdict"]["suggested_action"]
    assert payload["world_size"] == 1


def test_no_artifact_root_still_aborts() -> None:
    guard = CollapseGuardCallback(
        artifact_root=None, warmup_checks=0, sustain_checks=1
    )
    with pytest.raises(CollapseStop):
        _drive_val(guard, [{"val_n_masked": 0.0}])


def test_best_val_loss_tracks_min() -> None:
    guard = CollapseGuardCallback(warmup_checks=0, sustain_checks=99)
    _drive_val(guard, [{"val_loss": 5.0}, {"val_loss": 2.0}, {"val_loss": 3.0}])
    assert math.isclose(guard._best_val_loss, 2.0)  # noqa: SLF001


# -- key-drift guard -----------------------------------------------------


def test_watched_keys_are_logged_by_the_joint_module() -> None:
    """Cheap source canary that the module still BUILDS the guard's keys.

    The faithful key-drift catch is
    ``test_guard_watched_keys_appear_in_callback_metrics`` below (it runs a
    real val step and asserts the resolved keys are logged). This is a fast
    static backstop for the same drift, pinning the exact f-string forms that
    assemble each watched key.

    The RankMe keys are NOT contiguous literals — the module composes them as
    ``prefix = f"{step_name}_mon_{key}rankme"`` (key="" for M4,
    key="frontend_" for the M2 front-end) then ``f"{prefix}_alarm"`` /
    ``f"{prefix}_warn"``. The prefix template appears twice in the file (its
    own docstring + the real assignment), so a bare-template check is blind to
    renaming the assignment alone — pin the WHOLE assignment line instead.
    """
    import pathlib

    module_src = (
        pathlib.Path(__file__).parent / "v14_joint_module.py"
    ).read_text()
    required = [
        '{step_name}_loss',                          # → {val,train}_loss
        '{step_name}_n_masked',                      # → val_n_masked
        '{step_name}_mon_coverage_alarm',            # → val_mon_coverage_alarm
        'prefix = f"{step_name}_mon_{key}rankme"',   # the rankme ASSIGNMENT,
        # not the docstring shadow — renaming this alone now breaks the test.
        '{prefix}_alarm',                            # rankme alarm
        '{prefix}_warn',                             # rankme warn
        'key="frontend_"',                           # the M2 front-end variant
        'train_mon_grad_diverged',                   # bare on_step key
    ]
    missing = [s for s in required if s not in module_src]
    assert not missing, (
        f"v14_joint_module.py no longer builds the keys the guard watches "
        f"(missing: {missing}) — the guard would go blind. "
        f"Update both the module and the guard's watched keys."
    )


def test_guard_watched_keys_appear_in_callback_metrics() -> None:
    """Faithful key-drift catch: run a real val step on the joint module and
    assert every key the guard reads is actually logged.

    Stronger than the source canary above — it survives a metric being
    re-routed, gated behind a dropped phase branch, or logged under the wrong
    ``step_name``. Reuses the joint-module test harness so the module geometry
    stays in one place.
    """
    from speech_decoding.experiments.test_v14_joint_module import (
        _make_module,
        _make_synthetic_batch,
    )

    def _logged_keys(phase: str) -> set[str]:
        module = _make_module(phase=phase)
        logged: dict[str, float] = {}
        module.log = lambda key, value, **_kw: logged.update(  # type: ignore[method-assign]
            {key: float(value)}
        )
        module.validation_step(_make_synthetic_batch(), 0)
        return set(logged)

    # P2 trains the M4 parcel target → guard watches the M4 rankme family.
    p2_keys = _logged_keys("p2")
    for key in (
        "val_loss",
        "val_n_masked",
        "val_mon_coverage_alarm",
        "val_mon_rankme_alarm",
        "val_mon_rankme_warn",
    ):
        assert key in p2_keys, f"guard watches {key} but P2 val step no longer logs it"

    # P1 trains the front-end M2 → guard watches the frontend rankme family.
    p1_keys = _logged_keys("p1")
    for key in (
        "val_loss",
        "val_n_masked",
        "val_mon_coverage_alarm",
        "val_mon_frontend_rankme_alarm",
        "val_mon_frontend_rankme_warn",
    ):
        assert key in p1_keys, f"guard watches {key} but P1 val step no longer logs it"
