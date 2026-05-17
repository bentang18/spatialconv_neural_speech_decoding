"""Submit stft_abs × normalization grid at CrossSubject on DCC.

Mirror of submit_spectral_norm_grid_cross_subject.py (which holds view=log_stft)
but with view=stft_abs (LINEAR magnitude). Together they span the full 2x3x3
view × ref × norm matrix at CSubject.

Mechanism contrast:
- log_stft: gain enters as per-channel ADDITIVE constant in log domain →
  absorbed by LogReg intercept → mean-subtract is a no-op → only σ-divide
  is operative.
- stft_abs: gain enters as per-channel MULTIPLICATIVE scale on the linear
  magnitude → σ-divide IS the principled correction. Pre-spectral voltage z
  (divide voltage by σ_voltage) ≈ post-spectral per-channel scale divide
  by Parseval (Σ|x[t]|² ≈ (1/N) Σ|X[f]|²).

Empirical question: does stft_abs × N8 outperform log_stft × N8 at CSubject?
If yes, the log-compression itself was throwing away signal beyond what
σ-division can recover. If they match, log + N8 captures everything.

3 refs × 3 norms × 10 BT_LITE sessions (subject 2 excluded as TRAIN_SUBJECT_ID)
= 90 jobs at 64G.

Cell list:
  R1 bipolar         × stft_abs × {N1, N2, N8}
  R4 shaft_laplacian × stft_abs × {N1, N2, N8}
  R5 shaft_car       × stft_abs × {N1, N2, N8}
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from speech_decoding.studies.braintreebank.labels import NEUROPROBE_TASKS
from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS


TRAIN_SUBJECT_ID = 2


@dataclass(frozen=True)
class Cell:
    cell_id: str
    ref_kind: str
    normalization: str


REFS: tuple[tuple[str, str], ...] = (
    ("R1_bipolar",         "bipolar"),
    ("R4_shaft_laplacian", "shaft_laplacian"),
    ("R5_shaft_car",       "shaft_car"),
)

NORMS: tuple[tuple[str, str], ...] = (
    ("N1", "train_set_fixed"),
    ("N2", "per_session_fixed"),
    ("N8", "per_channel_train_set_z"),
)


def _build_cells() -> tuple[Cell, ...]:
    return tuple(
        Cell(f"L.SN.{ref_tag}_stft_abs_{norm_tag}", ref_kind, norm)
        for ref_tag, ref_kind in REFS
        for norm_tag, norm in NORMS
    )


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_sn_stft_abs_norm_grid_cross_subject_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    sessions = [(s, t) for (s, t) in BT_LITE_SESSIONS if s != TRAIN_SUBJECT_ID]
    cells = _build_cells()

    rows: list[dict[str, str]] = []
    for cell in cells:
        for sub_id, trial_id in sessions:
            report_dir = out_root / cell.cell_id / f"sub{sub_id}_trial{trial_id}"
            report_dir.mkdir(parents=True, exist_ok=True)
            command = build_command(report_dir, sub_id, trial_id, cell, args)
            sbatch_path = report_dir / "submit.sbatch"
            sbatch_path.write_text(build_sbatch(
                f"sn_{cell.cell_id.split('.')[-1][:18]}_s{sub_id}t{trial_id}",
                report_dir, command, args,
            ))
            job_id = ""
            if args.dry_run:
                print(f"[dry-run] {sbatch_path}")
            else:
                result = subprocess.run(
                    ["sbatch", str(sbatch_path)],
                    check=True, text=True, capture_output=True,
                )
                print(result.stdout.strip())
                job_id = result.stdout.strip().split()[-1]
            rows.append({
                "job_id": job_id,
                "cell_id": cell.cell_id,
                "ref_kind": cell.ref_kind,
                "view_kind": "stft_abs",
                "normalization": cell.normalization,
                "split_type": "CrossSubject",
                "subject_id": str(sub_id),
                "trial_id": str(trial_id),
                "report_dir": str(report_dir),
                "sbatch_path": str(sbatch_path),
            })

    write_manifest(out_root / "launch_manifest.csv", rows)
    print(f"Wrote {out_root / 'launch_manifest.csv'} ({len(rows)} jobs)")


def build_command(
    report_dir: Path, sub_id: int, trial_id: int, cell: Cell,
    args: argparse.Namespace,
) -> str:
    cmd = [
        ".venv/bin/python",
        "scripts/neuroprobe/run_stage0_linear_baseline.py",
        "--bt-root", str(args.bt_root),
        "--neuroprobe-repo", str(args.neuroprobe_repo),
        "--out-dir", str(report_dir),
        "--subject-id", str(sub_id),
        "--trial-id", str(trial_id),
        "--task", ",".join(NEUROPROBE_TASKS),
        "--split-type", "CrossSubject",
        "--binary-tasks", "false",
        "--backend", "neuralset",
        "--ref-kind", cell.ref_kind,
        "--view-kind", "stft_abs",
        "--normalization", cell.normalization,
        "--cell-id", cell.cell_id,
        "--seed", str(args.seed),
    ]
    return ("ROOT_DIR_BRAINTREEBANK=" + shlex.quote(str(args.bt_root))
            + " " + shlex.join(cmd))


def build_sbatch(
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


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--bt-root", type=Path, default=Path("/work/ht203/data/braintreebank"))
    p.add_argument(
        "--neuroprobe-repo", type=Path,
        default=Path("/work/ht203/repo/neuroprobe_upstream"),
    )
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--partition", default="scavenger-gpu,common,scavenger")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpus-per-task", type=int, default=4)
    p.add_argument("--mem", default="64G")
    p.add_argument("--time", default="12:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
