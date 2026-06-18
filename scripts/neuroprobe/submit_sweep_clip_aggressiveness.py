"""DCC submitter: the CLIP-aggressiveness sensitivity sweep, one SLURM-array task
per pretrain session (#236).

Mirrors ``submit_precompute_bad_windows.py`` — same streaming pass (post-static
``bt_load_raw`` -> notch/HPF -> shaft-CAR -> per-band |STFT| robust-z), but the
sweep re-runs the PURE ``_decide_bad_windows`` over a one-factor-at-a-time fence
grid and writes a counts-only ``{session}.json`` (NOT a clip sidecar). Default
mem is 192G: the 3 largest sessions OOM'd at 64G during the precompute.

Usage (sync + sbatch from laptop, via the dcc dispatch wrapper):

    scripts/dcc/dispatch scripts/neuroprobe/submit_sweep_clip_aggressiveness.py \\
        --out-dir /work/ht203/v14_clip_sweep

Then collect (after the array clears) with the script's --summarize mode:

    .venv/bin/python scripts/neuroprobe/sweep_clip_aggressiveness.py \\
        --summarize --out-dir /work/ht203/v14_clip_sweep
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
        f"reports/bt_clip_sweep_{datetime.now():%Y_%m_%d}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    sbatch_path = log_dir / "submit_clip_sweep.sbatch"
    sbatch_path.write_text(build_sbatch(log_dir=log_dir, args=args))

    print(f"[submit] out_dir   = {args.out_dir}")
    print(f"[submit] array     = {args.array_range} (one task per session)")
    print(f"[submit] partition = {args.cpu_partition}  mem = {args.mem}")
    print(f"[submit] sbatch    = {sbatch_path}")

    if args.dry_run:
        print("[dry-run] sbatch contents:\n" + sbatch_path.read_text())
        return

    res = subprocess.run(
        ["sbatch", str(sbatch_path)],
        check=True, text=True, capture_output=True,
    )
    print("[sbatch clip-sweep]", res.stdout.strip())


def build_sbatch(log_dir: Path, args: argparse.Namespace) -> str:
    lines = [
        "#!/bin/bash",
        "#SBATCH -J bt_clip_sweep",
        f"#SBATCH -p {args.cpu_partition}",
        f"#SBATCH --array={args.array_range}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines.extend([
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH -t {args.time}",
        f"#SBATCH -o {log_dir.resolve()}/cs-%A_%a.out",
        f"#SBATCH -e {log_dir.resolve()}/cs-%A_%a.err",
        "",
        "set -euo pipefail",
        f"cd {args.repo_root}",
        # The scan imports neuroprobe.config at runtime, which hard-requires
        # ROOT_DIR_BRAINTREEBANK; export it so the job is hermetic.
        f"export ROOT_DIR_BRAINTREEBANK={shlex.quote(str(args.root_dir_braintreebank))}",
    ])
    if args.pythonpath:
        lines.append(f"export PYTHONPATH={shlex.quote(str(args.pythonpath))}")
    lines.append("")
    # scripts/ is NOT a package -> run by PATH. The script reads SLURM_ARRAY_TASK_ID.
    cmd = [
        f"{shlex.quote(args.python)} scripts/neuroprobe/sweep_clip_aggressiveness.py \\",
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
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Per-session sweep JSON output dir (counts only, not a sidecar).")
    p.add_argument("--array-range", default="0-12",
                   help="Array range = session indices in V14_PRETRAIN_SESSIONS (0-12).")
    p.add_argument("--sessions", default=None,
                   help='Optional "S:T,S:T,..." override (array indexes into it).')
    p.add_argument("--log-dir", type=Path, default=None,
                   help="Where the .sbatch + slurm logs land (default reports/bt_clip_sweep_<date>).")
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument("--root-dir-braintreebank", default="/work/ht203/data/braintreebank",
                   help="Exported as ROOT_DIR_BRAINTREEBANK in the job.")
    p.add_argument("--python", default=".venv/bin/python")
    p.add_argument("--pythonpath", default=None)
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpu-partition", default="common")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default="192G",
                   help="192G: the 3 largest sessions OOM'd at 64G during precompute.")
    p.add_argument("--time", default="02:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
