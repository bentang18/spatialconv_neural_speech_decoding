"""DCC submitter: GUARD-1 STATIC precompute (#207), one SLURM-array task per BT
session. Pairs ``precompute_guard1_static.py`` (the scan) + ``collect_guard1_static.py``
(the #208 collation).

Mirrors ``submit_precompute_bad_windows.py``: the per-session voltage scan is
CPU-bound and embarrassingly parallel, so this submits a SLURM ARRAY on the
``common`` (CPU) partition. Array task ``i`` runs
``precompute_guard1_static.py --out-dir <DIR>`` which (via ``SLURM_ARRAY_TASK_ID``)
scans the i-th session of the v14 corpus (BT_FULL minus the excluded S5 = 25
sessions, indices 0..24): full-montage ``bt_load_raw(drop_static=False)`` ->
notch/HPF (no CAR) -> ``static_bad_mask`` -> per-session signature JSON.

It needs only the synced code + BT voltage — NOT any spec cache — so it runs
independently of (and before) the 3STFT cache build it ultimately informs.

The output dir holds tiny ``btbank{s}_t{t}.json`` signatures; pull them back to the
laptop and run ``collect_guard1_static.py --detections-dir <DIR>`` to collate into
the would-be per-subject static set and diff it against locked-11.

Usage (sync + sbatch from laptop, via the dcc dispatch wrapper):

    scripts/dcc/dispatch scripts/neuroprobe/submit_precompute_guard1_static.py \\
        --out-dir /work/ht203/v14_guard1_static

Single-session smoke (first-light):

    scripts/dcc/dispatch scripts/neuroprobe/submit_precompute_guard1_static.py \\
        --array-range 0-0 --out-dir /work/ht203/v14_guard1_static_smoke

Preflight without submitting (writes the .sbatch, prints it):

    scripts/dcc/dispatch scripts/neuroprobe/submit_precompute_guard1_static.py \\
        --out-dir /work/ht203/v14_guard1_static --dry-run
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
        f"reports/guard1_static_precompute_{datetime.now():%Y_%m_%d}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    sbatch_path = log_dir / "submit_guard1_static.sbatch"
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
    print("[sbatch guard1-static]", res.stdout.strip())


def build_sbatch(log_dir: Path, args: argparse.Namespace) -> str:
    lines = [
        "#!/bin/bash",
        "#SBATCH -J bt_guard1_static",
        f"#SBATCH -p {args.cpu_partition}",
        f"#SBATCH --array={args.array_range}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines.extend([
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH -t {args.time}",
        f"#SBATCH -o {log_dir.resolve()}/g1-%A_%a.out",
        f"#SBATCH -e {log_dir.resolve()}/g1-%A_%a.err",
        "",
        "set -euo pipefail",
        f"cd {args.repo_root}",
        # The scan imports neuroprobe.braintreebank_subject at runtime, which
        # hard-requires ROOT_DIR_BRAINTREEBANK. Export it in the script so the job
        # is hermetic regardless of the submitting shell's env.
        f"export ROOT_DIR_BRAINTREEBANK={shlex.quote(str(args.root_dir_braintreebank))}",
    ])
    if args.pythonpath:
        lines.append(f"export PYTHONPATH={shlex.quote(str(args.pythonpath))}")
    lines.append("")
    # scripts/ is NOT a package -> run the precompute by PATH, not -m. The script
    # reads SLURM_ARRAY_TASK_ID to pick its session (BT_FULL minus excluded S5).
    cmd = [
        f"{shlex.quote(args.python)} scripts/neuroprobe/precompute_guard1_static.py \\",
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
        help="Signature output dir (= collector --detections-dir). One btbank{s}_t{t}.json per task.",
    )
    p.add_argument(
        "--array-range", default="0-24",
        help="SLURM array range = session indices in the v14 corpus "
             "(BT_FULL minus S5 = 25 sessions -> 0-24). Use 0-0 for a single-session smoke.",
    )
    p.add_argument(
        "--sessions", default=None,
        help='Optional "S:T,S:T,..." override of the scanned corpus (array indexes into it).',
    )
    p.add_argument(
        "--log-dir", type=Path, default=None,
        help="Where the .sbatch + slurm logs land (default reports/guard1_static_precompute_<date>).",
    )
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument(
        "--root-dir-braintreebank", default="/work/ht203/data/braintreebank",
        help="Exported as ROOT_DIR_BRAINTREEBANK in the job (the BT loader requires it).",
    )
    p.add_argument(
        "--python", default=".venv/bin/python",
        help="Interpreter for the scan (relative to repo-root, or absolute).",
    )
    p.add_argument(
        "--pythonpath", default=None,
        help="If set, exported as PYTHONPATH so a worktree's src shadows the editable install.",
    )
    # CPU node; no GPU. The static scan is LIGHTER than the CLIP |STFT| scan (no
    # per-electrode STFT loop) but holds the same full-montage (C,N) float64 +
    # dV/dt copies, so the 2-3 largest sessions (sub_2/sub_10, ~9GB h5) may want
    # rerun-failed --mem 192G. 96G covers the rest.
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpu-partition", default="common",
                   help="CPU partition for the array (no coganlab CPU partition exists).")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default="96G")
    p.add_argument("--time", default="02:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
