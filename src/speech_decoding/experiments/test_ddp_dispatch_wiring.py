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

from speech_decoding.experiments.dispatch_v14 import main
from speech_decoding.experiments.experiment import _is_global_zero


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


def test_chain_threads_ddp_fields(monkeypatch, tmp_path) -> None:
    """Post-resolution ``tasks_per_node`` / ``slurm_use_srun`` must thread to ALL
    five chain phases via _common_build_kwargs; if one phase fell back to the
    exca default it would hang the moment that phase's array task launched."""
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
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
    ])
    # Simulate main()'s post-parse DDP resolution for a 4-GPU slurm run.
    args.tasks_per_node = 4
    args.slurm_use_srun = True
    phases = dv._build_v14_chain(args, cross_attn_positions=None)
    assert len(calls) == len(phases) == 5
    for c in calls:
        assert c["tasks_per_node"] == 4
        assert c["slurm_use_srun"] is True


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
