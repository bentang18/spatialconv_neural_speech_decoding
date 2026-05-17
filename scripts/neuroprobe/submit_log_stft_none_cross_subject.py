"""Submit log_stft × normalization=none × CrossSubject on DCC.

Direct test of the additive-gain-in-log-domain hypothesis: for log_stft,
`log(g[c] · |X|) = log g[c] + log|X|` — gain enters as a per-channel CONSTANT
in log-spec. A per-channel constant shift is absorbed by the LogReg intercept
(`score = Σ w[c,f,t]·(log|X| − c[c]) + b = Σ w·log|X| + b'` with b' shifted).

Therefore: if `log_stft × N4 (none)` matches `log_stft × N1/N8` at CSubject,
the additive-gain story holds and σ-divide is doing nothing extra (the LogReg
intercept is the gain corrector). If N8 still beats N4 substantially, then
per-channel σ-of-log-spec carries non-gain signal we'd be missing without it.

Companion to the 90-job spectral × norm grid (R1/R4/R5 × log_stft × N1/N2/N8).

3 refs × log_stft × 10 BT_LITE sessions (subject 2 excluded as TRAIN_SUBJECT_ID)
= 30 jobs at 64G.
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


REFS: tuple[tuple[str, str], ...] = (
    ("R1_bipolar",         "bipolar"),
    ("R4_shaft_laplacian", "shaft_laplacian"),
    ("R5_shaft_car",       "shaft_car"),
)


def _build_cells() -> tuple[Cell, ...]:
    return tuple(
        Cell(f"L.SN.{ref_tag}_log_stft_N4", ref_kind)
        for ref_tag, ref_kind in REFS
    )


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_sn_log_stft_none_cross_subject_{datetime.now():%Y_%m_%d}"
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
                "view_kind": "log_stft",
                "normalization": "none",
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
        "--view-kind", "log_stft",
        "--normalization", "none",
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
