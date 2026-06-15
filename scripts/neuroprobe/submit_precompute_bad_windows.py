"""DCC submitter: precompute the Layer-2 bad-time-window sidecars, one SLURM-array
task per pretrain session (#180, bad-electrode defense).

Mirrors ``submit_build_bt_2stft_cache.py``: the per-session glitch scan is CPU-bound
and embarrassingly parallel, so this submits a SLURM ARRAY on the ``common`` (CPU)
partition. Array task ``i`` runs
``precompute_bad_windows.py --out-dir <DIR>`` which (via ``SLURM_ARRAY_TASK_ID``)
scans the i-th session of ``V14_PRETRAIN_SESSIONS`` (13 sessions, indices 0..12):
post-static ``bt_load_raw`` -> notch/HPF -> shaft-CAR -> per-electrode dual-band
|STFT| (production bins) -> robust-z -> 5 s-window glitch scan, and writes one
``{session}.json`` sidecar with the bad-window spans the Layer-2 clip filter reads.

The scan reproduces the encoder's exact dual-band input (wraps the production
``robust_z``), so the sidecars match what the run sees. It needs only the synced
static-exclusion code (post-static ``bt_load_raw``) — NOT the spec cache — so it
can run in parallel with the 2STFT cache build.

The output dir is what a run passes to ``dispatch_v14.py --bad-window-dir`` (SSL
phases only). Sidecars are tiny JSON; pull them back / inspect in place.

Usage (sync + sbatch from laptop, via the dcc dispatch wrapper):

    scripts/dcc/dispatch scripts/neuroprobe/submit_precompute_bad_windows.py \\
        --out-dir /work/ht203/v14_bad_windows

Single-session smoke (first-light):

    scripts/dcc/dispatch scripts/neuroprobe/submit_precompute_bad_windows.py \\
        --array-range 0-0 --out-dir /work/ht203/v14_bad_windows_smoke

Preflight without submitting (writes the .sbatch, prints it):

    scripts/dcc/dispatch scripts/neuroprobe/submit_precompute_bad_windows.py \\
        --out-dir /work/ht203/v14_bad_windows --dry-run
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir or Path(
        f"reports/bt_bad_windows_precompute_{datetime.now():%Y_%m_%d}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    sbatch_path = log_dir / "submit_bad_windows.sbatch"
    sbatch_path.write_text(build_sbatch(log_dir=log_dir, args=args))

    print(f"[submit] out_dir   = {args.out_dir}")
    print(f"[submit] array     = {args.array_range} (one task per session)")
    print(f"[submit] partition = {args.cpu_partition}")
    print(f"[submit] sbatch    = {sbatch_path}")

    if args.dry_run:
        print("[dry-run] sbatch contents:\n" + sbatch_path.read_text())
        return

    res = subprocess.run(
        ["sbatch", str(sbatch_path)],
        check=True, text=True, capture_output=True,
    )
    print("[sbatch bad-windows]", res.stdout.strip())


def build_sbatch(log_dir: Path, args: argparse.Namespace) -> str:
    lines = [
        "#!/bin/bash",
        "#SBATCH -J bt_bad_windows",
        f"#SBATCH -p {args.cpu_partition}",
        f"#SBATCH --array={args.array_range}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines.extend([
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH -t {args.time}",
        f"#SBATCH -o {log_dir.resolve()}/bw-%A_%a.out",
        f"#SBATCH -e {log_dir.resolve()}/bw-%A_%a.err",
        "",
        "set -euo pipefail",
        f"cd {args.repo_root}",
        # The scan imports neuroprobe.config at runtime, which hard-requires
        # ROOT_DIR_BRAINTREEBANK. Export it in the script so the job is hermetic
        # regardless of the submitting shell's env (a bare non-interactive SSH
        # submission won't have it, and sbatch --export=ALL would then drop it).
        f"export ROOT_DIR_BRAINTREEBANK={shlex.quote(str(args.root_dir_braintreebank))}",
    ])
    if args.pythonpath:
        lines.append(f"export PYTHONPATH={shlex.quote(str(args.pythonpath))}")
    lines.append("")
    # scripts/ is NOT a package -> run the precompute by PATH, not -m. The script
    # reads SLURM_ARRAY_TASK_ID to pick its session of V14_PRETRAIN_SESSIONS.
    cmd = [
        f"{shlex.quote(args.python)} scripts/neuroprobe/precompute_bad_windows.py \\",
        f"    --out-dir {shlex.quote(str(args.out_dir))}",
    ]
    if args.sessions:
        cmd[-1] += " \\"
        cmd.append(f"    --sessions {shlex.quote(args.sessions)}")
    lines.extend(cmd)
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir", type=Path, required=True,
        help="Sidecar output dir (= the run's --bad-window-dir). One {session}.json per task.",
    )
    p.add_argument(
        "--array-range", default="0-12",
        help="SLURM array range = session indices in V14_PRETRAIN_SESSIONS "
             "(13 sessions -> 0-12). Use 0-0 for a single-session smoke.",
    )
    p.add_argument(
        "--sessions", default=None,
        help='Optional "S:T,S:T,..." override of the scanned corpus (array indexes into it).',
    )
    p.add_argument(
        "--log-dir", type=Path, default=None,
        help="Where the .sbatch + slurm logs land (default reports/bt_bad_windows_precompute_<date>).",
    )
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument(
        "--root-dir-braintreebank", default="/work/ht203/data/braintreebank",
        help="Exported as ROOT_DIR_BRAINTREEBANK in the job (neuroprobe.config requires it).",
    )
    p.add_argument(
        "--python", default=".venv/bin/python",
        help="Interpreter for the scan (relative to repo-root, or absolute).",
    )
    p.add_argument(
        "--pythonpath", default=None,
        help="If set, exported as PYTHONPATH so a worktree's src shadows the editable install.",
    )
    # SLURM resources (CPU node; no GPU). Mirror the 2STFT cache build — the scan
    # runs the same whole-movie dual-band |STFT| per electrode.
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpu-partition", default="common",
                   help="CPU partition for the array (no coganlab CPU partition exists).")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default="64G")
    p.add_argument("--time", default="02:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
