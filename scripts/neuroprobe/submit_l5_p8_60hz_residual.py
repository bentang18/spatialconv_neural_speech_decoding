"""Submit Stage-0 L.5.P8 60 Hz residual PSD analyzer on DCC. Single job."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir or Path(
        f"reports/neuroprobe_stage0_l5_p8_60hz_residual_{datetime.now():%Y_%m_%d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ".venv/bin/python",
        "scripts/neuroprobe/run_l5_p8_60hz_residual.py",
        "--bt-root", str(args.bt_root),
        "--neuroprobe-repo", str(args.neuroprobe_repo),
        "--out-dir", str(out_dir),
    ]
    command = ("ROOT_DIR_BRAINTREEBANK=" + shlex.quote(str(args.bt_root))
               + " " + shlex.join(cmd))
    sbatch = "\n".join([
        "#!/bin/bash",
        "#SBATCH -J l5_p8_60hz",
        f"#SBATCH -p {args.partition}",
        f"#SBATCH --account={args.account}" if args.account else "",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH -t {args.time}",
        f"#SBATCH -o {out_dir.resolve()}/slurm-%j.out",
        f"#SBATCH -e {out_dir.resolve()}/slurm-%j.err",
        "",
        "set -euo pipefail",
        "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        "export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        "export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        "export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        "export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        f"cd {args.repo_root}",
        command,
        "",
    ])
    sbatch_path = out_dir / "submit.sbatch"
    sbatch_path.write_text(sbatch)
    if args.dry_run:
        print(f"[dry-run] {sbatch_path}")
    else:
        result = subprocess.run(
            ["sbatch", str(sbatch_path)],
            check=True, text=True, capture_output=True,
        )
        print(result.stdout.strip())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--bt-root", type=Path, default=Path("/work/ht203/data/braintreebank"))
    p.add_argument(
        "--neuroprobe-repo", type=Path,
        default=Path("/work/ht203/repo/neuroprobe_upstream"),
    )
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument("--partition", default="common,scavenger,coganlab-gpu")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpus-per-task", type=int, default=4)
    p.add_argument("--mem", default="64G")
    p.add_argument("--time", default="04:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
