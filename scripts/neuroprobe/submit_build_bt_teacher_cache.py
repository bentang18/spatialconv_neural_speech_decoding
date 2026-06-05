"""DCC submitter: build the v14 P3 Whisper teacher cache over BT movies (T14).

One GPU sbatch job. The build script
(``scripts/neuroprobe/build_bt_teacher_cache.py``) loops over every movie WAV
internally and writes one dense ``(T, 1280)`` fp16 stream per movie, so this is
a single job (NOT an array) that runs Whisper-large-v3 once per movie over 30 s
grid chunks. ~20 movies, GPU peak ~3.4 GB, wall-clock ~40 min.

Default partition is ``coganlab-gpu`` (no preemption, RTX 5000 Ada) per the
production-run default — a preempted Whisper build wastes the whole pass.

Usage (sync + sbatch from laptop):

    scripts/dcc/dispatch scripts/neuroprobe/submit_build_bt_teacher_cache.py \\
        --wav-dir /hpc/group/coganlab/ht203/data/braintreebank_wavs \\
        --out-dir /hpc/group/coganlab/ht203/cache_neuroai/bt_teacher_cache

Single-movie smoke (first-light) — note GPU still loads the full large-v3:

    scripts/dcc/dispatch scripts/neuroprobe/submit_build_bt_teacher_cache.py \\
        --limit 1 --out-dir /hpc/group/coganlab/ht203/cache_neuroai/bt_teacher_smoke

Preflight without submitting (writes the .sbatch, prints it):

    scripts/dcc/dispatch scripts/neuroprobe/submit_build_bt_teacher_cache.py --dry-run
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    log_dir = args.log_dir or Path(
        f"reports/bt_teacher_cache_build_{datetime.now():%Y_%m_%d}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    sbatch_path = log_dir / "submit_build.sbatch"
    sbatch_path.write_text(build_sbatch(out_dir=out_dir, log_dir=log_dir, args=args))

    print(f"[submit] wav_dir   = {args.wav_dir}")
    print(f"[submit] out_dir   = {out_dir}")
    print(f"[submit] partition = {args.gpu_partition}")
    print(f"[submit] sbatch    = {sbatch_path}")

    if args.dry_run:
        print("[dry-run] sbatch contents:\n" + sbatch_path.read_text())
        return

    res = subprocess.run(
        ["sbatch", str(sbatch_path)],
        check=True, text=True, capture_output=True,
    )
    print("[sbatch build]", res.stdout.strip())


def build_sbatch(out_dir: Path | None, log_dir: Path, args: argparse.Namespace) -> str:
    # --fit-only fits channel_stats over an already-built cache: no Whisper load,
    # so a CPU job (no GPU) on `common` (there is no coganlab CPU partition).
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {'bt_channel_stats' if args.fit_only else 'bt_teacher_cache'}",
        f"#SBATCH -p {args.cpu_partition if args.fit_only else args.gpu_partition}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines.extend([
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --mem={args.mem}",
    ])
    if not args.fit_only:
        lines.append("#SBATCH --gres=gpu:1")
    lines.extend([
        f"#SBATCH -t {args.time}",
        f"#SBATCH -o {log_dir.resolve()}/build-%j.out",
        f"#SBATCH -e {log_dir.resolve()}/build-%j.err",
        "",
        "set -euo pipefail",
        f"cd {args.repo_root}",
    ])
    if args.pythonpath:
        # Isolated-worktree runs reuse the MAIN clone's .venv binary but must
        # import the worktree's src, so PYTHONPATH has to win over the editable
        # install's path. Lets a separate clone build without its own uv sync.
        lines.append(f"export PYTHONPATH={shlex.quote(str(args.pythonpath))}")
    lines.extend([
        # whisper-large-v3 is pre-cached in ~/.cache/huggingface (shared /hpc/home);
        # pin offline so the compute node never blocks on a HF network HEAD request.
        "export HF_HUB_OFFLINE=1",
        "export TRANSFORMERS_OFFLINE=1",
        "",
    ])
    cmd = [
        f"{shlex.quote(args.python)} scripts/neuroprobe/build_bt_teacher_cache.py \\",
        f"    --wav-dir {shlex.quote(str(args.wav_dir))} \\",
    ]
    if out_dir is not None:
        cmd.append(f"    --out-dir {shlex.quote(str(out_dir))} \\")
    cmd.extend([
        f"    --model {shlex.quote(args.model)} \\",
        f"    --layer-merge {shlex.quote(args.layer_merge)} \\",
        f"    --chunk-s {args.chunk_s} \\",
    ])
    if args.limit is not None:
        cmd.append(f"    --limit {args.limit} \\")
    if args.overwrite:
        cmd.append("    --overwrite \\")
    if args.fit_channel_stats or args.fit_only:
        cmd.append("    --fit-channel-stats \\")
    if args.channel_stats_out is not None:
        cmd.append(f"    --channel-stats-out {shlex.quote(str(args.channel_stats_out))} \\")
    cmd.append(f"    --device {'cpu' if args.fit_only else 'cuda'}")
    lines.extend(cmd)
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wav-dir", type=Path,
        default=Path("/hpc/group/coganlab/ht203/data/braintreebank_wavs"),
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("/hpc/group/coganlab/ht203/cache_neuroai/bt_teacher_cache"),
        help="Teacher-cache root (persistent /hpc tier).",
    )
    p.add_argument(
        "--log-dir", type=Path, default=None,
        help="Where the .sbatch + slurm logs land (default reports/bt_teacher_cache_build_<date>).",
    )
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument(
        "--python", default=".venv/bin/python",
        help="Interpreter for the build (relative to repo-root, or absolute). For "
             "an isolated worktree without its own venv, point at the main clone's "
             "interpreter and pass --pythonpath.",
    )
    p.add_argument(
        "--pythonpath", default=None,
        help="If set, exported as PYTHONPATH so the worktree's src shadows the "
             "editable install (e.g. /work/ht203/repo/speech_bt/src).",
    )
    p.add_argument("--model", default="openai/whisper-large-v3")
    p.add_argument("--layer-merge", default="mean_all")
    p.add_argument("--chunk-s", type=float, default=30.0)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N movies (smoke).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--gpu-partition", default="coganlab-gpu",
                   help="Production default: no preemption, RTX 5000 Ada.")
    p.add_argument("--cpu-partition", default="common",
                   help="Partition for --fit-only (CPU; no coganlab CPU partition exists).")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--mem", default="64G")
    p.add_argument("--time", default="02:00:00")
    p.add_argument(
        "--fit-channel-stats", action="store_true",
        help="Pass --fit-channel-stats to the build (fit + save channel_stats.pt "
             "after the movie cache).",
    )
    p.add_argument(
        "--channel-stats-out", type=Path, default=None,
        help="Pass-through destination for channel_stats.pt.",
    )
    p.add_argument(
        "--fit-only", action="store_true",
        help="Fit channel_stats over an already-complete cache: CPU job (no GPU), "
             "implies --fit-channel-stats. Use when every movie is already cached.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
