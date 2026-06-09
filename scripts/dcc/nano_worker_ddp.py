#!/usr/bin/env python
"""Warm **4-GPU DDP** worker for fast nano iteration on DCC.

The DDP analog of ``nano_worker.py``. Launched as N srun ranks inside one
long-lived allocation (``nano_worker_ddp_start.sh`` → ``srun --ntasks=N
--gres=gpu:N``), each rank imports the heavy v14 dispatch stack ONCE (~50 s off
/work, dominated by torchmetrics→transformers→torchvision) and then runs many
nano experiments in-process. Each subsequent run pays ~0 s import and ~0 s slurm
queue — the workers already hold the GPUs and the live NCCL ranks. This kills the
cold-start wall (~95 s → ~one warm dataset/model build) for the 4-GPU loop.

Why this works (and is byte-identical to the cold submitit path): the COLD 4-GPU
run is exca-submitit → ``srun --ntasks=N`` → N fresh processes each calling
``xp.run()`` with Lightning DDP over the existing srun ranks (see dispatch_v14
build_v14_experiment, comment at ``tasks_per_node``). We replicate exactly that —
N persistent srun ranks each calling ``dispatch_v14.main(argv)`` — but exca runs
LOCAL (no nested submit) via ``--in-allocation-ddp``, and the import is amortized.

Coordination (FS-only; Lightning owns the NCCL process group inside main()):
  - Submit: client writes ``<dir>/queue/<id>.cmd`` (one CLI arg per line),
    atomically via temp+rename. ALL ranks read the SAME file → shared argv.
  - Run:    every rank calls ``dispatch_v14.main(argv)`` together; NCCL collectives
    keep them in lockstep even if they enter at slightly different times.
  - Barrier: each rank writes ``<dir>/done/<id>.<rank>`` when its run returns.
    The leader (SLURM_PROCID 0) waits for all N markers, then writes ``<id>.rc``
    (= max rc; any rank failure → non-zero) — the done-marker the client waits on.
  - Stop: an empty ``<dir>/queue/__stop__`` file (or SIGINT) exits every rank.

The worker injects ``--in-allocation-ddp`` and strips any ``--cluster`` from the
argv so a stray client flag can never make exca try to re-submit.
"""
from __future__ import annotations

import gc
import os
import sys
import time
import traceback
from pathlib import Path

POLL_SECONDS = 0.3
DONE_BARRIER_TIMEOUT_S = 60.0  # how long the leader waits for all ranks per job


def _rank() -> int:
    return int(os.environ.get("SLURM_PROCID", "0"))


def _world() -> int:
    return int(os.environ.get("SLURM_NTASKS", "1"))


def _is_leader() -> bool:
    return _rank() == 0


def _log_status(real_stderr_fd: int, msg: str) -> None:
    """Write a worker-status line to the worker's own console (not a run log)."""
    os.write(real_stderr_fd, f"[nano_ddp r{_rank()}] {msg}\n".encode())


def _sanitize_argv(argv: list[str]) -> list[str]:
    """Force the in-allocation DDP contract: ``--in-allocation-ddp`` present, no
    ``--cluster`` (which would make exca submit a nested job and hang)."""
    out: list[str] = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok == "--cluster":
            skip_next = True  # also drop its value
            continue
        if tok.startswith("--cluster="):
            continue
        out.append(tok)
    if "--in-allocation-ddp" not in out:
        out.append("--in-allocation-ddp")
    return out


def _between_runs() -> None:
    """Best-effort cleanup so the next in-process run starts clean."""
    try:
        import wandb

        if wandb.run is not None:  # a run lightning forgot to finalize
            wandb.finish()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


def main() -> int:
    rank, world = _rank(), _world()
    worker_dir = Path(
        os.environ.get("NANO_DDP_WORKER_DIR", str(Path.home() / ".nano_worker_ddp"))
    )
    queue_dir = worker_dir / "queue"
    logs_dir = worker_dir / "logs"
    done_dir = worker_dir / "done"
    for d in (queue_dir, logs_dir, done_dir):
        d.mkdir(parents=True, exist_ok=True)

    real_stderr_fd = os.dup(2)

    # The leader sweeps stale per-job markers from a previous incarnation BEFORE
    # anyone polls, so a crashed worker's leftovers can't be re-picked.
    if _is_leader():
        for stale in list(done_dir.glob("*")) + list(logs_dir.glob("*.rc")):
            stale.unlink(missing_ok=True)

    _log_status(real_stderr_fd, f"importing dispatch_v14 (one-time) world={world} ...")
    t0 = time.time()
    from speech_decoding.experiments import dispatch_v14  # noqa: E402 — the whole point

    _log_status(
        real_stderr_fd,
        f"import done in {time.time() - t0:.1f}s — READY (pid {os.getpid()})",
    )
    if _is_leader():
        (worker_dir / "READY").write_text(f"{os.getpid()} world={world}\n")

    stop_file = queue_dir / "__stop__"
    seen: set[str] = set()
    try:
        while True:
            if stop_file.exists():
                _log_status(real_stderr_fd, "stop file seen — exiting")
                if _is_leader():
                    stop_file.unlink(missing_ok=True)
                return 0

            cmds = sorted(
                (p for p in queue_dir.glob("*.cmd") if p.stem not in seen),
                key=lambda p: p.stat().st_mtime,
            )
            if not cmds:
                time.sleep(POLL_SECONDS)
                continue

            cmd_path = cmds[0]
            run_id = cmd_path.stem
            seen.add(run_id)
            argv = _sanitize_argv(
                [ln for ln in cmd_path.read_text().splitlines() if ln.strip()]
            )
            _log_status(real_stderr_fd, f"running {run_id}: {' '.join(argv)}")

            # Per-rank log; the client tails rank-0's (Lightning logs + result).
            log_path = logs_dir / f"{run_id}.rank{rank}.log"
            log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            saved_out, saved_err = os.dup(1), os.dup(2)
            os.dup2(log_fd, 1)
            os.dup2(log_fd, 2)
            run_t0 = time.time()
            rc = 0
            try:
                rc = int(dispatch_v14.main(argv) or 0)
            except SystemExit as e:  # argparse etc. — keep the worker alive
                rc = int(e.code) if isinstance(e.code, int) else 1
            except BaseException:  # noqa: BLE001 — one bad run must not kill the worker
                traceback.print_exc()
                rc = 1
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_out, 1)
                os.dup2(saved_err, 2)
                os.close(saved_out)
                os.close(saved_err)
                os.close(log_fd)
                _between_runs()

            # FS barrier: announce this rank is done, then the leader collects.
            (done_dir / f"{run_id}.{rank}").write_text(f"{rc}\n")
            _log_status(
                real_stderr_fd, f"done {run_id} rc={rc} in {time.time() - run_t0:.1f}s"
            )

            if _is_leader():
                deadline = time.time() + DONE_BARRIER_TIMEOUT_S
                worst = rc
                while time.time() < deadline:
                    markers = list(done_dir.glob(f"{run_id}.*"))
                    if len(markers) >= world:
                        for m in markers:
                            try:
                                worst = max(worst, int(m.read_text().strip() or 0))
                            except ValueError:
                                worst = max(worst, 1)
                        break
                    time.sleep(POLL_SECONDS)
                else:
                    _log_status(
                        real_stderr_fd,
                        f"barrier TIMEOUT for {run_id} "
                        f"({len(list(done_dir.glob(f'{run_id}.*')))}/{world} ranks)",
                    )
                    worst = max(worst, 1)
                (logs_dir / f"{run_id}.rc").write_text(f"{worst}\n")
                for m in done_dir.glob(f"{run_id}.*"):
                    m.unlink(missing_ok=True)
    except KeyboardInterrupt:
        _log_status(real_stderr_fd, "interrupted — exiting")
        return 0
    finally:
        if _is_leader():
            (worker_dir / "READY").unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
