"""Submit Stage-0 L.5.P1 (+P2) cross-subject id nuisance probe on DCC.

Re-runs the existing P2 wrapper with --include-p1 added. P1 pools features
across all 12 BT Lite sessions (truncated to common min feature dim) and
trains a multi-class subject-id probe. Single sbatch job (loops sessions
internally).

Kill: held-out P1 AUROC > 0.95 → drop the L.2-winner view as
subject-discriminating. (P2 was already shown to fire trivially under the
high-dim-binary caveat in chris-doc Section H2; P1 provides the multi-class
cross-subject test that's load-bearing for v14's anatomy-anchored generalization
claim.)

L.2 winner config (R4 × I2 × N1).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir or Path(
        f"reports/neuroprobe_stage0_l5_p1_cross_subject_id_{datetime.now():%Y_%m_%d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ".venv/bin/python",
        "scripts/neuroprobe/run_l5_nuisance_probes.py",
        "--bt-root", str(args.bt_root),
        "--neuroprobe-repo", str(args.neuroprobe_repo),
        "--out-dir", str(out_dir),
        "--ref-kind", "shaft_laplacian",
        "--view-kind", "stft_abs",
        "--window-seconds", "1.0",
        "--seed", str(args.seed),
        "--include-p1",
    ]
    command = ("ROOT_DIR_BRAINTREEBANK=" + shlex.quote(str(args.bt_root))
               + " " + shlex.join(cmd))
    sbatch = build_sbatch("l5_p1_cs_id", out_dir, command, args)
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


def build_sbatch(
    job_name: str, out_dir: Path, command: str, args: argparse.Namespace,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name[:48]}",
        f"#SBATCH -p {args.partition}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines.extend([
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
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--bt-root", type=Path, default=Path("/work/ht203/data/braintreebank"))
    p.add_argument(
        "--neuroprobe-repo", type=Path,
        default=Path("/work/ht203/repo/neuroprobe_upstream"),
    )
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--partition", default="common,scavenger,coganlab-gpu")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpus-per-task", type=int, default=4)
    p.add_argument("--mem", default="96G")
    p.add_argument("--time", default="12:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
