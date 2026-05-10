"""Submit Neuroprobe Stage 0 L.5.P2 within-subject nuisance probe on DCC.

Single CPU sbatch job. For each subject in BT Lite (6 subjects × 2 sessions),
trains a per-subject LogReg classifier to decode session-id (trial within
subject) from features on the L.2 winner view (R4xI2 = shaft_laplacian
× stft_abs). Kill criterion: drop view if any subject's held-out macro
AUROC > 0.95.

Why P2-only (P1 dropped): subject-id is trivially decodable from per-subject
feature width (different channel sets → different F). The meaningful test is
within-subject session drift.

Memory: per-subject features kept on the heap before per-subject probe; peak
~24 GB. `--mem=64G` covers upstream laplacian + stft scratch + StandardScaler
copies.

Usage on DCC:
    .venv/bin/python scripts/neuroprobe/submit_l5_nuisance_probes.py
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_l5_nuisance_probes_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    report_dir = out_root / "L.5.P2"
    report_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        ".venv/bin/python",
        "scripts/neuroprobe/run_l5_nuisance_probes.py",
        "--bt-root", str(args.bt_root),
        "--neuroprobe-repo", str(args.neuroprobe_repo),
        "--out-dir", str(report_dir),
        "--ref-kind", args.ref_kind,
        "--view-kind", args.view_kind,
        "--window-seconds", str(args.window_seconds),
        "--seed", str(args.seed),
    ]
    command = ("ROOT_DIR_BRAINTREEBANK=" + shlex.quote(str(args.bt_root))
               + " " + shlex.join(cmd))

    job_name = f"l5_nuisance_probes_{args.ref_kind}_{args.view_kind}"
    sbatch_path = report_dir / "submit.sbatch"
    sbatch_path.write_text(_build_sbatch(job_name, report_dir, command, args))

    if args.dry_run:
        print(f"[dry-run] {sbatch_path}")
        return

    result = subprocess.run(
        ["sbatch", str(sbatch_path)],
        check=True, text=True, capture_output=True,
    )
    print(result.stdout.strip())


def _build_sbatch(
    job_name: str, report_dir: Path, command: str, args: argparse.Namespace,
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
        f"#SBATCH -o {report_dir.resolve()}/slurm-%j.out",
        f"#SBATCH -e {report_dir.resolve()}/slurm-%j.err",
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
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--bt-root", type=Path, default=Path("/work/ht203/data/braintreebank"))
    p.add_argument(
        "--neuroprobe-repo", type=Path,
        default=Path("/work/ht203/repo/neuroprobe_upstream"),
    )
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument("--ref-kind", default="shaft_laplacian", help="L.2 winner reference")
    p.add_argument("--view-kind", default="stft_abs", help="L.2 winner view")
    p.add_argument("--window-seconds", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--partition", default="common,scavenger,coganlab-gpu")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpus-per-task", type=int, default=4)
    p.add_argument("--mem", default="64G")
    p.add_argument("--time", default="12:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
