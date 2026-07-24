"""#33 4-GPU DDP enablement — wiring regression tests.

The 4-GPU hang was a silent default: ``--gpus-per-node 4`` set only
``gpus_per_node`` while exca/submitit kept ``tasks_per_node=1``, so srun
launched ONE task with 4 GPUs visible. Under srun Lightning's
``SLURMEnvironment.detect()`` returns True → ``creates_processes_externally`` →
DDP never spawns ranks 1-3 → world_size=4 waits forever at NCCL init.

The fix is one srun rank per GPU: ``tasks_per_node = gpus_per_node`` and
``slurm_use_srun=True`` (exca hard-requires the latter when tasks_per_node>1).
These tests pin (a) main()'s auto-resolution of that pair, (b) that an explicit
``--tasks-per-node 1`` still forces legacy single-rank, (c) the local / single-GPU
paths stay untouched, (d) the fields thread to EVERY chain phase, and (e) the
rank-0 write gate so per-rank re-execution doesn't corrupt shared artifacts.

These run on the laptop (dry-run + kwargs capture; no BT data, no Experiment
build); the live 4-GPU fast-dev-run on DCC is the end-to-end NCCL gate.
"""
from __future__ import annotations

import os

import pytest

from speech_decoding.experiments.dispatch_v14 import main
from speech_decoding.experiments.experiment import _is_global_zero


@pytest.fixture(autouse=True)
def _restore_v14_env():
    """``main()`` configures the child run by WRITING ``V14_*`` into this process's
    environment, and a dry-run exits before anything unsets them. Left alone, the
    first test here leaves ``V14_COMPILE=1``/``V14_COMPILE_DYNAMIC=1`` set for the
    REST OF THE SESSION, so any later test asserting the eager default
    (``test_speedups_off_by_default_eager_bytewise``: ``_compile_spec is None``)
    fails in a full-suite run while passing in isolation. Snapshot and restore.
    """
    saved = {k: v for k, v in os.environ.items() if k.startswith("V14_")}
    try:
        yield
    finally:
        for k in [k for k in os.environ if k.startswith("V14_")]:
            del os.environ[k]
        os.environ.update(saved)


# --- main() auto-resolution (printed in the dry-run summary) -----------------


def test_four_gpu_slurm_resolves_ddp(capsys) -> None:
    """A bare ``--gpus-per-node 4`` on slurm auto-derives one srun rank per GPU.
    Without this the run hangs at NCCL init (the original bug)."""
    rc = main([
        "--cluster", "slurm", "--gpus-per-node", "4",
        "--slurm-partition", "coganlab-gpu", "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "tasks_per_node=4" in out
    assert "slurm_use_srun=True" in out


def test_single_gpu_slurm_stays_single_rank(capsys) -> None:
    """One GPU must NOT enable srun/multi-task — single-process Lightning, no
    SLURMEnvironment external-launch path."""
    rc = main([
        "--cluster", "slurm", "--gpus-per-node", "1",
        "--slurm-partition", "coganlab-gpu", "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "tasks_per_node=None" in out
    assert "slurm_use_srun=False" in out


def test_explicit_tasks_per_node_1_warns_about_hang(capsys) -> None:
    """``--tasks-per-node 1 --gpus-per-node 4`` is a footgun: 4 GPUs stay in one
    rank → NCCL hang. The dispatch must honour the explicit value (no silent
    override) BUT warn loudly so the operator isn't surprised by a multi-hour
    hang. The true single-GPU path is --gpus-per-node 1."""
    rc = main([
        "--cluster", "slurm", "--gpus-per-node", "4", "--tasks-per-node", "1",
        "--slurm-partition", "coganlab-gpu", "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "tasks_per_node=1" in out
    assert "slurm_use_srun=False" in out
    assert "WARNING" in out and "DDP topology is NOT enabled" in out


def test_cluster_auto_multigpu_warns(capsys) -> None:
    """``--cluster auto`` submits to slurm but the DDP auto-resolve only fires for
    the literal ``slurm`` string, so a bare ``--cluster auto --gpus-per-node 4``
    would silently hang. The safety-net warning must fire."""
    rc = main(["--cluster", "auto", "--gpus-per-node", "4", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARNING" in out and "DDP topology is NOT enabled" in out


def test_four_gpu_resolves_find_unused_parameters_strategy(capsys) -> None:
    """A multi-rank DDP run MUST request the find-unused DDP strategy. The
    staged B36 SSL phases leave whole submodules out of the active loss (P1
    front-end-only; predictor is P2-only) while keeping them grad-requiring, so
    plain DDP crashes on the 2nd iteration. ``--fast-dev-run`` (1 batch) could
    not catch this in the live gate — the reducer only rebuilds buckets from
    the 2nd forward — so it is pinned here at the config layer."""
    rc = main([
        "--cluster", "slurm", "--gpus-per-node", "4",
        "--slurm-partition", "coganlab-gpu", "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ddp_strategy=ddp_find_unused_parameters_true" in out


def test_single_gpu_stays_no_ddp_strategy(capsys) -> None:
    """One GPU must NOT request a DDP strategy — it would force DDP on a
    single-device run. Lightning auto-selects single-device."""
    rc = main([
        "--cluster", "slurm", "--gpus-per-node", "1",
        "--slurm-partition", "coganlab-gpu", "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ddp_strategy=None" in out


def test_build_threads_ddp_strategy_to_experiment() -> None:
    """The find-unused strategy must reach the constructed Experiment (so its
    ``_trainer`` passes ``strategy=`` to the Lightning Trainer). Builds a real
    multi-rank (``tasks_per_node>1``) joint-SSL Experiment (config only, no data
    load) and reads the field off the pydantic object. The build succeeding at
    all also proves the field is accepted (``extra='forbid'`` would reject it)."""
    import speech_decoding.experiments.dispatch_v14 as dv

    exp = dv.build_v14_experiment(
        mode="lite", joint_phase=True, bt_root="/tmp/bt", tasks_per_node=4,
        slurm_use_srun=True, cluster="slurm", slurm_partition="coganlab-gpu",
        exca_folder="/tmp/exca",
    )
    assert exp.ddp_strategy == "ddp_find_unused_parameters_true"


def test_build_single_gpu_no_ddp_strategy() -> None:
    """A single-rank build must leave ``ddp_strategy=None`` so Lightning stays
    on its single-device auto strategy (no forced DDP)."""
    import speech_decoding.experiments.dispatch_v14 as dv

    exp = dv.build_v14_experiment(
        mode="lite", joint_phase=True, bt_root="/tmp/bt", tasks_per_node=None,
        slurm_use_srun=False, cluster="slurm", slurm_partition="coganlab-gpu",
        exca_folder="/tmp/exca",
    )
    assert exp.ddp_strategy is None


def test_four_gpu_slurm_no_hang_warning(capsys) -> None:
    """The happy path (--cluster slurm --gpus-per-node 4) auto-enables real DDP,
    so the hang warning must NOT fire."""
    rc = main([
        "--cluster", "slurm", "--gpus-per-node", "4",
        "--slurm-partition", "coganlab-gpu", "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DDP topology is NOT enabled" not in out


def test_local_run_has_no_slurm_ddp_fields(capsys) -> None:
    """No ``--cluster`` (in-process laptop/dry run) prints no slurm line and the
    DDP resolution is skipped entirely."""
    rc = main(["--gpus-per-node", "4", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "slurm:" not in out


# --- chain threading: the DDP pair must reach every phase --------------------


def _capture_chain_calls(monkeypatch, tmp_path, *, gpus_per_node=4):
    """Build the 5-phase chain with a stubbed ``build_v14_experiment`` and return
    the captured per-phase kwargs. Simulates main()'s post-parse DDP resolution
    for a multi-GPU slurm run."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls: list[dict] = []

    def fake_build(**kw):
        calls.append(kw)

        class _Stub:
            def run(self):
                return {}

        return _Stub()

    monkeypatch.setattr(dv, "build_v14_experiment", fake_build)
    args = dv._parser().parse_args([
        "--chain", "--cluster", "slurm", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
        "--gpus-per-node", str(gpus_per_node),
    ])
    # Mirror main()'s post-parse DDP resolution exactly: only gpus_per_node>1
    # auto-derives one srun rank per GPU; 1-GPU keeps the parser defaults.
    if gpus_per_node > 1:
        args.tasks_per_node = gpus_per_node
        args.slurm_use_srun = True
    phases = dv._build_v14_chain(args)
    return calls, phases


def test_chain_threads_ddp_fields_to_ssl_phases(monkeypatch, tmp_path) -> None:
    """Post-resolution ``tasks_per_node`` / ``slurm_use_srun`` must thread to the
    four SSL/distill phases (P1,P2,P3a,P3b) via _common_build_kwargs; if one
    phase fell back to the exca default it would hang the moment that phase's
    array task launched. (P4 is the deliberate exception — see the next test.)"""
    calls, phases = _capture_chain_calls(monkeypatch, tmp_path)
    assert len(calls) == len(phases) == 5
    for c in calls[:4]:  # P1, P2, P3a, P3b
        assert c["tasks_per_node"] == 4
        assert c["slurm_use_srun"] is True
        assert c["gpus_per_node"] == 4


def test_chain_forces_p4_probe_single_gpu(monkeypatch, tmp_path) -> None:
    """The P4 frozen-probe phase MUST run single-GPU even when the SSL phases run
    4-GPU DDP. Under multi-rank DDP, trainer.test() computes AUROC over only
    rank-0's ~1/N shard (no all_gather) and AUROC is non-decomposable, so the
    reported leaderboard number would be wrong. Single-GPU ⇒ full test set, no
    DistributedSampler, ddp_strategy=None inside build."""
    calls, _ = _capture_chain_calls(monkeypatch, tmp_path)
    p4 = calls[-1]  # P4 is built last in _build_v14_chain
    assert p4["phase4_frozen_probe"] is True
    assert p4["gpus_per_node"] == 1
    assert p4["tasks_per_node"] is None
    assert p4["slurm_use_srun"] is False


def test_chain_single_gpu_leaves_p4_identical(monkeypatch, tmp_path) -> None:
    """A single-GPU chain (no DDP) must NOT special-case P4 — the override only
    fires under real multi-rank DDP, so a 1-GPU chain leaves every phase on the
    same topology (no behaviour change)."""
    calls, _ = _capture_chain_calls(monkeypatch, tmp_path, gpus_per_node=1)
    for c in calls:
        assert c["gpus_per_node"] == 1
        assert c["tasks_per_node"] is None
        assert c["slurm_use_srun"] is False
        # tasks_per_node None ⇒ no find-unused strategy anywhere (incl. P4).


# --- 8-GPU scale-out (C3): the 4->8 path is airtight at the config layer ------


def test_eight_gpu_slurm_resolves_ddp(capsys) -> None:
    """C3 scale-out: ``--gpus-per-node 8`` on the single coganlab node must
    resolve one srun rank per GPU (tasks_per_node=8 + srun) and the find-unused
    strategy, with NO hang warning. The resolution is generic
    (tasks_per_node = gpus_per_node) so 8 is the same code path as 4 — this pins
    it so a future edit can't regress the 8-GPU launch we relaunch on."""
    rc = main([
        "--cluster", "slurm", "--gpus-per-node", "8",
        "--slurm-partition", "coganlab-gpu", "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "tasks_per_node=8" in out
    assert "slurm_use_srun=True" in out
    assert "ddp_strategy=ddp_find_unused_parameters_true" in out
    assert "DDP topology is NOT enabled" not in out


def test_eight_gpu_chain_holds_eff_batch_with_accum_4(monkeypatch, tmp_path) -> None:
    """C3 eff-batch invariant: the 8-GPU recipe holds effective batch = 128 by
    halving accum 8->4 (8 GPU x bs 4 x accum 4 = 128, same as 4 x 4 x 8). The LR
    (1.76e-4) and all opt-step-unit budgets are therefore unchanged → identical
    science. This pins that the SSL phases receive accumulate_grad_batches=4 and
    the 8-GPU topology together, so the copy-paste recipe stays correct."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls: list[dict] = []

    def fake_build(**kw):
        calls.append(kw)

        class _Stub:
            def run(self):
                return {}

        return _Stub()

    monkeypatch.setattr(dv, "build_v14_experiment", fake_build)
    args = dv._parser().parse_args([
        "--chain", "--cluster", "slurm", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
        "--gpus-per-node", "8", "--accumulate-grad-batches", "4",
    ])
    args.tasks_per_node = 8  # mirror main()'s post-parse DDP resolution
    args.slurm_use_srun = True
    dv._build_v14_chain(args)
    assert len(calls) == 5
    for c in calls[:4]:  # P1, P2, P3a, P3b run multi-GPU DDP
        assert c["gpus_per_node"] == 8
        assert c["tasks_per_node"] == 8
        assert c["accumulate_grad_batches"] == 4
    # 8 GPU x (per-rank bs) x accum 4 holds eff-batch == 4-GPU x accum 8 path.


# --- speedup launch flags (C1: torch.compile front-door) ---------------------


def test_compile_flag_sets_env_var(monkeypatch) -> None:
    """The dispatch ALWAYS writes an explicit V14_COMPILE ("1"/"0") so submitit
    captures a definite value into the slurm job env (the module reads it at
    construction). ``--compile`` is DEFAULT-ON since 8a697c4, so a bare run sets
    V14_COMPILE=="1"; ``--no-compile`` selects the eager path (V14_COMPILE=="0").
    """
    monkeypatch.delenv("V14_COMPILE", raising=False)
    monkeypatch.delenv("V14_COMPILE_MODE", raising=False)
    import os

    # Default (no flag) = compile ON (8a697c4) → V14_COMPILE=="1".
    rc = main(["--gpus-per-node", "1", "--dry-run"])
    assert rc == 0
    assert os.environ["V14_COMPILE"] == "1"

    # Explicit --no-compile = eager (what the nano runs use) → V14_COMPILE=="0".
    rc = main(["--no-compile", "--gpus-per-node", "1", "--dry-run"])
    assert rc == 0
    assert os.environ["V14_COMPILE"] == "0"

    # --compile --compile-mode default → V14_COMPILE=="1", mode propagated.
    rc = main([
        "--compile", "--compile-mode", "default",
        "--gpus-per-node", "1", "--dry-run",
    ])
    assert rc == 0
    assert os.environ["V14_COMPILE"] == "1"
    assert os.environ["V14_COMPILE_MODE"] == "default"


def test_static_forward_requires_tube_ratio_and_group_by_session() -> None:
    """``--converged-static-forward`` is fail-fast: the static forward needs the
    tight-pack tube's constant n_vis AND a session-homogeneous batch
    (``compute_static_shapes`` fails loud on a hetero batch). main() must reject a
    launch missing either, BEFORE the dry-run summary — a mid-run abort is worse.
    With both present, the same launch dry-runs clean (rc 0)."""
    import pytest

    # missing tube_ratio → SystemExit
    with pytest.raises(SystemExit):
        main(["--converged-static-forward", "--group-by-session",
              "--gpus-per-node", "1", "--dry-run"])

    # has tube_ratio but missing group-by-session → SystemExit
    with pytest.raises(SystemExit):
        main(["--converged-static-forward", "--converged-tube-ratio", "0.25",
              "--gpus-per-node", "1", "--dry-run"])

    # both present → guard passes, dry-run returns 0
    rc = main(["--converged-static-forward", "--converged-tube-ratio", "0.25",
               "--group-by-session", "--gpus-per-node", "1", "--dry-run"])
    assert rc == 0


def test_in_allocation_ddp_honors_compile(monkeypatch, capsys) -> None:
    """End-to-end: ``--in-allocation-ddp --compile`` now writes ``V14_COMPILE=="1"``
    (the __getstate__ pickle fix makes a compiled module DDP-safe), and does NOT
    print the obsolete force-off override message."""
    monkeypatch.delenv("V14_COMPILE", raising=False)
    import os

    rc = main(["--in-allocation-ddp", "--compile", "--dry-run"])
    assert rc == 0
    assert os.environ["V14_COMPILE"] == "1"
    assert "forcing --no-compile" not in capsys.readouterr().out


# --- rank-0 write gate -------------------------------------------------------


def test_is_global_zero_rank0(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_PROCID", "0")
    assert _is_global_zero() is True


def test_is_global_zero_nonzero_rank(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_PROCID", "3")
    assert _is_global_zero() is False


def test_is_global_zero_rank_env_fallback(monkeypatch) -> None:
    """``RANK`` (torchrun-style) is honoured when SLURM_PROCID is absent."""
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.setenv("RANK", "2")
    assert _is_global_zero() is False


def test_is_global_zero_single_process(monkeypatch) -> None:
    """No rank env at all (laptop / single-GPU) ⇒ this IS the writer."""
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    assert _is_global_zero() is True
