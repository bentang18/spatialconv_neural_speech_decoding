"""B36 WS-E — staged P1/P2 optimizers, checkpoint handoff, dead-blocker cleanup.

Covers the five WS-E deliverables (``docs/neuroprobe/b36_implementation_plan.md``):

* **E1** P1 optimizer = front-end params + the P1 predictor (paradigm B, exact
  parity with P2's predictor); a P1 backward reaches the front-end AND the
  predictor but leaves the pool / inter-parcel encoder grad-free.
* **E2** P2 optimizer = two param groups with the front-end at base_lr/10.
* **E3** the multi-phase driver wires each phase's snapshot to the next phase's
  ``pretrained_ckpt``; the base ``Experiment`` snapshot↔load round-trips.
* **E4** ``transferable_state`` exports only the shared encoder; the strict load
  raises on a key mismatch and re-syncs a frozen EMA teacher.
* **E5** the dead ``_PHASE1_BLOCKERS`` is gone and ``V14JointExperiment``
  overrides neither ``run`` nor ``_train_and_test`` (the docstrings now match
  the real call graph).

The full 4-phase ``run_phase_pipeline`` over a real loader is exercised by the
WS-I2 synthetic-BT capstone; here the driver wiring + per-boundary load
mechanism are unit-tested without the data pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import exca
import pydantic
import pytest
import torch

from neuraltrain.optimizers import LightningOptimizer
from neuraltrain.optimizers.base import AdamW

from speech_decoding.experiments.experiment import Experiment
from speech_decoding.experiments.v14_joint_module import V14JointBrainModule
from speech_decoding.experiments.v14_phase_pipeline import (
    _phase_ckpt_path,
    configure_phase_handoff,
)
from speech_decoding.models.v14_encoder import V14ParcelPerceiverModel

# Grid sized so the M2 band mask (6/03 lock, held-out 0.50) is PARTIAL —
# F_p=3 / T_p=8 leaves visible cells, so P1's front-end gets real gradient
# through the token-block self-prediction. A 2×2 grid would let one
# floor-width (2) time-band cover the whole axis → 100% masked → zeroed
# front-end output → no front-end gradient (a tiny-grid artifact, not real).
_ENC_KW = dict(
    n_freq_bins=10, n_time_bins=16, k_parcels=6,
    d_model=32, n_heads=4, depth_self_attn=1, m_sub_slots=1,
    patch_kernel_freq=3,  # FE-RAW-1: kernel-3 path for the tiny F=10 config
)
_BASE_LR = 1e-3


def _module(
    phase: str, *, seed: int = 0, frontend_lr_scale: float = 0.1,
) -> V14JointBrainModule:
    torch.manual_seed(seed)
    return V14JointBrainModule(
        encoder=V14ParcelPerceiverModel(**_ENC_KW),
        optim_config=LightningOptimizer(optimizer=AdamW(lr=_BASE_LR)),
        phase=phase,
        frontend_lr_scale=frontend_lr_scale,
    )


def _opt(module: V14JointBrainModule):
    out = module.configure_optimizers()
    return out["optimizer"] if isinstance(out, dict) else out


def _synthetic_batch(B: int = 2, C: int = 5, T: int = 16, Fb: int = 10, K: int = 6):
    support = torch.zeros(B, C, K)
    idx = torch.arange(C) % K
    support[:, torch.arange(C), idx] = 1.0
    return {"electrode_tokens": torch.randn(B, C, T, Fb), "support": support}


# --------------------------------------------------------------------------
# E1 — P1 front-end-only optimizer + grad flow
# --------------------------------------------------------------------------

def test_e1_p1_optimizer_is_frontend_plus_predictor() -> None:
    m = _module("p1")
    opt = _opt(m)
    assert len(opt.param_groups) == 1
    n_opt = sum(len(g["params"]) for g in opt.param_groups)
    frontend, parcel = m.student.encoder.partition_parameters_for_staging()
    predictor_params = list(m.predictor.parameters())
    # Paradigm B: P1 trains the front-end AND its own predictor (exact parity
    # with P2's predictor); the pool / inter-parcel encoder (parcel) is not.
    assert n_opt == len(frontend) + len(predictor_params)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert opt_ids.isdisjoint({id(p) for p in parcel})
    assert {id(p) for p in predictor_params}.issubset(opt_ids)
    assert {id(p) for p in frontend}.issubset(opt_ids)


def test_e1_p1_backward_reaches_frontend_and_predictor_not_parcel() -> None:
    m = _module("p1")
    loss = m._step(_synthetic_batch()).total
    loss.backward()
    frontend, parcel = m.student.encoder.partition_parameters_for_staging()
    # Paradigm B: the visible-only front-end produces the M2 context that feeds
    # the predictor, so BOTH get real gradient; the pool / inter-parcel encoder
    # (parcel) is off the M2 loss path → grad-free. (The predictor-free
    # paradigm-A self-distill here was the P1-collapse regression.)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in frontend)
    assert all(p.grad is None for p in parcel)
    assert m.predictor.output_proj.weight.grad is not None
    assert m.predictor.output_proj.weight.grad.abs().sum() > 0


# --------------------------------------------------------------------------
# E2 — P2 discriminative LR (two param groups, 10:1)
# --------------------------------------------------------------------------

def test_e2_p2_two_param_groups_lr_ratio_10() -> None:
    m = _module("p2")
    opt = _opt(m)
    assert len(opt.param_groups) == 2
    front_lr, parcel_lr = opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]
    assert front_lr == pytest.approx(_BASE_LR * 0.1)
    assert parcel_lr == pytest.approx(_BASE_LR)
    assert parcel_lr / front_lr == pytest.approx(10.0)


def test_e2_p2_frontend_group_holds_exactly_the_frontend_params() -> None:
    m = _module("p2")
    opt = _opt(m)
    frontend, parcel = m.student.encoder.partition_parameters_for_staging()
    front_group_ids = {id(p) for p in opt.param_groups[0]["params"]}
    parcel_group_ids = {id(p) for p in opt.param_groups[1]["params"]}
    assert front_group_ids == {id(p) for p in frontend}
    # The parcel group is the encoder-parcel params PLUS the predictor.
    expected_parcel = {id(p) for p in parcel} | {id(p) for p in m.predictor.parameters()}
    assert parcel_group_ids == expected_parcel


def test_e2_p2_frontend_lr_scale_is_a_hyperparameter() -> None:
    # R-p2-frontend-lr-5: scale 0.2 → front-end at base/5, ratio 5:1.
    m = _module("p2", frontend_lr_scale=0.2)
    opt = _opt(m)
    assert len(opt.param_groups) == 2
    front_lr, parcel_lr = opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]
    assert front_lr == pytest.approx(_BASE_LR * 0.2)
    assert parcel_lr == pytest.approx(_BASE_LR)
    assert parcel_lr / front_lr == pytest.approx(5.0)


def test_e2_p2_frontend_frozen_falsifier_single_group() -> None:
    # R-p2-freeze-frontend: scale 0.0 → front-end frozen + dropped from the
    # optimizer, leaving one base-LR group over the parcel side + predictor.
    m = _module("p2", frontend_lr_scale=0.0)
    frontend, parcel = m.student.encoder.partition_parameters_for_staging()
    assert all(not p.requires_grad for p in frontend)
    assert all(p.requires_grad for p in parcel)
    opt = _opt(m)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == pytest.approx(_BASE_LR)
    group_ids = {id(p) for p in opt.param_groups[0]["params"]}
    expected = {id(p) for p in parcel} | {id(p) for p in m.predictor.parameters()}
    assert group_ids == expected


def test_e2_p2_frontend_frozen_backward_leaves_frontend_grad_free() -> None:
    # The frozen front-end gets no gradient; the parcel side still trains.
    m = _module("p2", frontend_lr_scale=0.0)
    loss = m._step(_synthetic_batch()).total
    loss.backward()
    frontend, parcel = m.student.encoder.partition_parameters_for_staging()
    assert all(p.grad is None for p in frontend)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parcel)


def test_e2_frontend_lr_scale_out_of_range_raises() -> None:
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="frontend_lr_scale"):
            _module("p2", frontend_lr_scale=bad)


def test_partition_raises_on_unassigned_param() -> None:
    # A new top-level encoder param must be consciously assigned to a stage;
    # silently bucketing it (e.g. into the front-end) would corrupt the P1/P2
    # LR split with every test still green. The guard must raise.
    enc = V14ParcelPerceiverModel(**_ENC_KW)
    enc.register_parameter("mystery_param", torch.nn.Parameter(torch.zeros(1)))
    with pytest.raises(RuntimeError, match="unassigned"):
        enc.partition_parameters_for_staging()


# --------------------------------------------------------------------------
# E4 — transferable-state protocol (run first; E3 builds on it)
# --------------------------------------------------------------------------

def test_e4_transferable_state_exports_only_encoder() -> None:
    m = _module("p1")
    state = m.transferable_state()
    assert set(state) == {"encoder"}
    assert set(state["encoder"]) == set(m.student.encoder.state_dict())


def test_e4_load_transfers_encoder_and_resyncs_frozen_teacher() -> None:
    src = _module("p1", seed=0)
    with torch.no_grad():
        for p in src.student.encoder.parameters():
            p.add_(0.5)
    dst = _module("p2", seed=99)
    pre = [torch.equal(a, b) for a, b in zip(
        src.student.encoder.state_dict().values(),
        dst.student.encoder.state_dict().values())]
    assert not all(pre)  # encoders start different

    dst.load_transferable_state(src.transferable_state(), strict=True)

    assert all(torch.equal(a, b) for a, b in zip(
        src.student.encoder.state_dict().values(),
        dst.student.encoder.state_dict().values()))
    # Teacher re-synced to the loaded student and still frozen.
    assert all(torch.equal(dst.student.state_dict()[k], v)
               for k, v in dst.teacher.model.state_dict().items())
    assert all(not p.requires_grad for p in dst.teacher.model.parameters())


def test_e4_strict_load_raises_on_key_mismatch() -> None:
    src = _module("p1")
    dst = _module("p2")
    full = src.transferable_state()["encoder"]
    dropped = {k: v for i, (k, v) in enumerate(full.items()) if i > 0}
    with pytest.raises(RuntimeError):
        dst.load_transferable_state({"encoder": dropped}, strict=True)


def test_e4_missing_encoder_component_raises() -> None:
    dst = _module("p2")
    with pytest.raises(KeyError):
        dst.load_transferable_state({"pma": {}}, strict=True)


# --------------------------------------------------------------------------
# E3 — driver wiring + base-Experiment snapshot/load round-trip
# --------------------------------------------------------------------------

class _StubPhase(pydantic.BaseModel):
    """Minimal exca-decorated stand-in mirroring ``Experiment``'s handoff surface.

    Real exca dispatch (``@infra.apply``) — NOT a duck-typed ``model_copy``
    fake — so the test exercises the actual dispatch path the driver relies
    on. ``run()`` returns the field values *as the dispatched body sees them*;
    that is what catches the ``model_copy``-leaves-infra-bound BLOCKER.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    pretrained_ckpt: str | None = None
    snapshot_ckpt_to: str | None = None
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    def run(self) -> dict:
        return {
            "pretrained_ckpt": self.pretrained_ckpt,
            "snapshot_ckpt_to": self.snapshot_ckpt_to,
        }


def test_e3_configure_phase_handoff_wires_each_phase_to_the_prior_ckpt() -> None:
    phases = [_StubPhase() for _ in range(4)]
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        cfg = configure_phase_handoff(phases, work_dir=work)
        # Phase 0 has no predecessor; every phase snapshots to its own path.
        assert cfg[0].pretrained_ckpt is None
        for i in range(4):
            assert cfg[i].snapshot_ckpt_to == str(_phase_ckpt_path(work, i))
        # Phase i (>0) loads phase i-1's snapshot.
        for i in range(1, 4):
            assert cfg[i].pretrained_ckpt == str(_phase_ckpt_path(work, i - 1))
        # B1 regression: the wired ckpt must survive exca dispatch, not just
        # the pydantic field. ``model_copy`` would set the field but ``run()``
        # would read ``None`` (infra still bound to the pre-copy config).
        for i in range(4):
            dispatched = cfg[i].run()
            assert dispatched["snapshot_ckpt_to"] == str(_phase_ckpt_path(work, i))
            assert dispatched["pretrained_ckpt"] == (
                None if i == 0 else str(_phase_ckpt_path(work, i - 1))
            )
    # Inputs are untouched (clone, not mutate).
    assert all(p.pretrained_ckpt is None and p.snapshot_ckpt_to is None for p in phases)


def test_e3_configure_phase_handoff_empty_raises() -> None:
    with pytest.raises(ValueError):
        configure_phase_handoff([], work_dir="/tmp/never")


def test_e3_base_experiment_snapshot_then_load_roundtrips_encoder() -> None:
    # ``_snapshot`` / ``_load_pretrained`` never touch ``self`` (they delegate
    # to the module protocol), so they are exercised as unbound methods with a
    # ``None`` self — no heavyweight Experiment construction needed.
    src = _module("p1", seed=1)
    with torch.no_grad():
        for p in src.student.encoder.parameters():
            p.add_(0.25)
    dst = _module("p2", seed=2)
    with tempfile.TemporaryDirectory() as td:
        ckpt = str(Path(td) / "sub" / "phase_0.ckpt")  # nested dir is created
        Experiment._snapshot(None, src, ckpt)  # type: ignore[arg-type]
        assert Path(ckpt).exists()
        Experiment._load_pretrained(None, dst, ckpt)  # type: ignore[arg-type]
    assert all(torch.equal(a, b) for a, b in zip(
        src.student.encoder.state_dict().values(),
        dst.student.encoder.state_dict().values()))


def test_e3_snapshot_requires_transferable_protocol() -> None:
    with pytest.raises(TypeError):
        Experiment._snapshot(None, torch.nn.Linear(2, 2), "/tmp/x.ckpt")  # type: ignore[arg-type]


def test_e3_load_requires_transferable_protocol() -> None:
    with pytest.raises(TypeError):
        Experiment._load_pretrained(None, torch.nn.Linear(2, 2), "/tmp/x.ckpt")  # type: ignore[arg-type]


def test_e3_load_missing_snapshot_raises_actionable_error() -> None:
    # Gate-D F4: the snapshot write lives INSIDE the exca-cached run() body, so a
    # cache-hit phase on a purged work_dir leaves no .ckpt for the next phase to
    # warm-start from. A protocol-valid module + a missing ckpt must raise an
    # actionable FileNotFoundError (NOT the opaque bare torch.load one, and NOT
    # the earlier protocol TypeError — the module HAS the protocol here).
    dst = _module("p2", seed=2)  # has load_transferable_state
    with tempfile.TemporaryDirectory() as td:
        missing = str(Path(td) / "never_written" / "phase_0.ckpt")
        with pytest.raises(FileNotFoundError, match="cache-hit"):
            Experiment._load_pretrained(None, dst, missing)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# E5 — dead _PHASE1_BLOCKERS removed + docstrings match the call graph
# --------------------------------------------------------------------------

def test_e5_phase1_blockers_symbol_is_gone() -> None:
    from speech_decoding.experiments import dispatch_v14
    assert not hasattr(dispatch_v14, "_PHASE1_BLOCKERS")


def test_e5_joint_experiment_overrides_neither_run_nor_train_and_test() -> None:
    from speech_decoding.experiments.v14_joint import V14JointExperiment
    # Guards the dispatch docstrings: there is no ``_train_and_test`` override,
    # so any claim that "phase=1 raises from ``V14JointExperiment._train_and_test``"
    # is false. The joint subclass overrides only ``_build_brain_module`` (build
    # the masked-JEPA module) and ``model_post_init`` (construction-time gating).
    assert "run" not in vars(V14JointExperiment)
    assert "_train_and_test" not in vars(V14JointExperiment)
    assert "_build_brain_module" in vars(V14JointExperiment)
    assert "model_post_init" in vars(V14JointExperiment)


# --- #82 leakage-guard call site + per-phase gating (LC1, LC7/EM2) -----------


def test_guard_call_runs_after_build_before_trainer_fit() -> None:
    """LC1: the leakage guard call is ordered AFTER the loaders are built and
    BEFORE the first gradient step in ``Experiment._train_and_test``. Deleting
    the call or moving it after ``trainer.fit`` would let gradients flow on a
    leaking/starved corpus — and currently breaks no other test."""
    import inspect

    src = inspect.getsource(Experiment._train_and_test)
    i_build = src.index("self.data.build(")
    i_guard = src.index("_assert_no_pretrain_leakage(")
    i_fit = src.index("trainer.fit(")
    assert i_build < i_guard < i_fit, (
        "leakage guard must run after data.build and before trainer.fit"
    )


def test_leakage_guard_gating_classvars_per_phase() -> None:
    """LC7 + EM2: the guard is ARMED on the SSL/distill phases and EXEMPT on the
    supervised P4 readout + the base Experiment. P4 legitimately trains/tests on
    the eval split, so arming it there would falsely abort; leaving it off the
    SSL/distill phases would let eval data pretrain unverified."""
    from speech_decoding.experiments.v14_joint import V14JointExperiment
    from speech_decoding.experiments.v14_phase3 import V14Phase3Experiment
    from speech_decoding.experiments.v14_phase4 import (
        V14Phase4ReadoutExperiment,
    )

    assert V14JointExperiment.enforces_pretrain_leakage_guard is True
    assert V14Phase3Experiment.enforces_pretrain_leakage_guard is True
    assert V14Phase4ReadoutExperiment.enforces_pretrain_leakage_guard is False
    assert Experiment.enforces_pretrain_leakage_guard is False


def test_anti_starvation_cohort_floor_per_phase() -> None:
    """The anti-starvation floor is phase-specific: P1/P2 (joint) span the full
    cohort (7 subjects / 13 sessions); P3 drops (8,0) (no teacher) → (6, 12). P4
    + base carry no floor (None) so the guard is inert there."""
    from speech_decoding.experiments.v14_joint import V14JointExperiment
    from speech_decoding.experiments.v14_phase3 import V14Phase3Experiment
    from speech_decoding.experiments.v14_phase4 import (
        V14Phase4ReadoutExperiment,
    )

    assert V14JointExperiment.pretrain_cohort_floor == (7, 13)
    assert V14Phase3Experiment.pretrain_cohort_floor == (6, 12)
    assert V14Phase4ReadoutExperiment.pretrain_cohort_floor is None
    assert Experiment.pretrain_cohort_floor is None


def test_guard_invoked_at_runtime_with_all_realized_split_keys(monkeypatch) -> None:
    """G2-audit finding-2 + LC1(runtime): at execution ``_assert_no_pretrain_leakage``
    passes BOTH guards the FULL realized split tuple (train, val, test), not a
    narrowed ('train',) literal, and forwards the per-phase cohort floor. A
    regression to ``phases=('train',)`` would stop checking the val/test
    monitoring splits for eval leakage (Neuroprobe forbids eval data in ANY
    pretraining split); the source-order test pins WHEN the guard runs, this pins
    WHAT it checks. Spies the lazily-imported guards (``from ... import`` re-reads
    the module attribute at call time, so a module-level monkeypatch is seen)."""
    from types import SimpleNamespace

    from speech_decoding.studies.braintreebank import leakage as lk

    seen: dict[str, object] = {}

    def _spy_leak(loaders, *, study, phases):
        seen["leak_phases"] = tuple(phases)
        seen["leak_study"] = study

    def _spy_cohort(loaders, *, study, min_subjects, min_sessions):
        seen["cohort"] = (min_subjects, min_sessions)
        seen["cohort_study"] = study

    monkeypatch.setattr(lk, "assert_no_eval_leakage", _spy_leak)
    monkeypatch.setattr(lk, "assert_pretrain_corpus_spans_cohort", _spy_cohort)

    study = object()
    stub = SimpleNamespace(
        enforces_pretrain_leakage_guard=True,
        pretrain_cohort_floor=(7, 13),
        data=SimpleNamespace(study=study),
    )
    loaders = {"train": object(), "val": object(), "test": object()}
    Experiment._assert_no_pretrain_leakage(stub, loaders)  # type: ignore[arg-type]

    # The leak guard must see EVERY realized split, not a narrowed literal.
    assert seen["leak_phases"] == ("train", "val", "test")
    assert seen["leak_study"] is study
    # The anti-starvation floor is forwarded verbatim from the per-phase ClassVar.
    assert seen["cohort"] == (7, 13)
    assert seen["cohort_study"] is study


def test_guard_inert_at_runtime_when_phase_does_not_enforce(monkeypatch) -> None:
    """G2-audit finding-1 complement: ``enforces_pretrain_leakage_guard=False``
    (the P4 readout + base Experiment) makes the method return BEFORE importing or
    calling either guard, so P4 can legitimately train+test on the eval split.
    Neither spy fires."""
    from types import SimpleNamespace

    from speech_decoding.studies.braintreebank import leakage as lk

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        lk, "assert_no_eval_leakage",
        lambda *a, **k: seen.setdefault("leak", True),
    )
    monkeypatch.setattr(
        lk, "assert_pretrain_corpus_spans_cohort",
        lambda *a, **k: seen.setdefault("cohort", True),
    )

    stub = SimpleNamespace(
        enforces_pretrain_leakage_guard=False,
        pretrain_cohort_floor=None,
        data=SimpleNamespace(study=object()),
    )
    loaders = {"train": object(), "val": object(), "test": object()}
    Experiment._assert_no_pretrain_leakage(stub, loaders)  # type: ignore[arg-type]

    assert seen == {}, "P4/base must invoke neither guard"
