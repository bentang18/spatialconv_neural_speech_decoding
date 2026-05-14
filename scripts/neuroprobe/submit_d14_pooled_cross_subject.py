"""Submit Neuroprobe Stage 0 D.14 pooled multi-source CrossSubject baseline.

Companion to Tier-C (which is 1-to-1 pairwise CrossSubject — DS_DM_TRAIN
single source). D.14 pools training across every NEUROPROBE_LITE source
subject (excluding the held-out test subject) into a single classifier.
This is the "scientific generalization default" multiclass baseline that
v14 must beat for the submission gate, and is absent from the Neuroprobe
v2 leaderboard (so v14 is first-to-report and needs its own linear floor).

Two pool modes per session, dispatched as separate cells:

- D.14a_test_anchored — feature space = held-out test subject's DK regions;
  source channels mean-pooled per region, zero-fill where source missing
  the region. Default. Cleanest test signal.
- D.14b_source_union — feature space = ⋃ DK regions across all source
  subjects; same mean-pool with zero-fill for both sides. Sister cell;
  defensive ablation. Agreement with D.14a → robust. Disagreement → finding.

Held-out test = every (subject_id, trial_id) in NEUROPROBE_LITE_SUBJECT_TRIALS
(12 sessions; subject 2 is permitted as test in pool mode). Per-job:
runs all 16 NEUROPROBE_TASKS in one process, --binary-tasks toggled per
sweep variant.

Usage on DCC:
    .venv/bin/python scripts/neuroprobe/submit_d14_pooled_cross_subject.py
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


@dataclass(frozen=True)
class PoolCell:
    cell_id: str
    region_pool_mode: str
    label: str


@dataclass(frozen=True)
class PoolJob:
    cell: PoolCell
    subject_id: int
    trial_id: int
    binary_tasks: bool

    @property
    def name(self) -> str:
        kind = "bin" if self.binary_tasks else "mc"
        return (
            f"d14_{self.cell.cell_id.lower().replace('.', '_')}_"
            f"sub{self.subject_id}_trial{self.trial_id}_{kind}"
        )


def build_cells(modes: list[str]) -> list[PoolCell]:
    cells: list[PoolCell] = []
    if "test-anchored" in modes:
        cells.append(PoolCell(
            cell_id="D.14a_test_anchored",
            region_pool_mode="test-anchored",
            label="pool train across all source subjects; feature space = test DK regions",
        ))
    if "source-union" in modes:
        cells.append(PoolCell(
            cell_id="D.14b_source_union",
            region_pool_mode="source-union",
            label="pool train across all source subjects; feature space = ⋃ source DK regions",
        ))
    return cells


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_d14_pooled_cross_subject_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    sessions = _parse_sessions(args.sessions) or list(BT_LITE_SESSIONS)
    modes = [m.strip() for m in args.region_pool_modes.split(",") if m.strip()]
    cells = build_cells(modes)
    if not cells:
        raise SystemExit(f"--region-pool-modes selected zero cells from {modes}")

    binary_variants: list[bool] = []
    if "multiclass" in args.binary_variants.split(","):
        binary_variants.append(False)
    if "binary" in args.binary_variants.split(","):
        binary_variants.append(True)
    if not binary_variants:
        raise SystemExit("--binary-variants must include 'multiclass' and/or 'binary'")

    rows: list[dict[str, str]] = []
    for cell in cells:
        for binary_tasks in binary_variants:
            kind = "binary" if binary_tasks else "multiclass"
            for sub_id, trial_id in sessions:
                job = PoolJob(
                    cell=cell, subject_id=sub_id, trial_id=trial_id,
                    binary_tasks=binary_tasks,
                )
                report_dir = (
                    out_root / cell.cell_id / kind / f"sub{sub_id}_trial{trial_id}"
                )
                report_dir.mkdir(parents=True, exist_ok=True)
                sbatch_path = report_dir / "submit.sbatch"
                command = build_command(job, report_dir, args)
                sbatch_path.write_text(build_sbatch(job, report_dir, command, args))
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
                    "label": cell.label,
                    "region_pool_mode": cell.region_pool_mode,
                    "binary_tasks": str(binary_tasks),
                    "split_type": "CrossSubject",
                    "subject_id": str(sub_id),
                    "trial_id": str(trial_id),
                    "report_dir": str(report_dir),
                    "sbatch_path": str(sbatch_path),
                })

    write_manifest(out_root / "launch_manifest.csv", rows)
    print(f"Wrote {out_root / 'launch_manifest.csv'} ({len(rows)} jobs)")


def build_command(job: PoolJob, report_dir: Path, args: argparse.Namespace) -> str:
    cmd = [
        ".venv/bin/python",
        "scripts/neuroprobe/run_stage0_linear_baseline.py",
        "--bt-root", str(args.bt_root),
        "--neuroprobe-repo", str(args.neuroprobe_repo),
        "--out-dir", str(report_dir),
        "--subject-id", str(job.subject_id),
        "--trial-id", str(job.trial_id),
        "--task", ",".join(NEUROPROBE_TASKS),
        "--split-type", "CrossSubject",
        "--binary-tasks", "true" if job.binary_tasks else "false",
        "--backend", "neuralset",
        "--ref-kind", args.ref_kind,
        "--view-kind", args.view_kind,
        "--normalization", args.normalization,
        "--pool-multi-source",
        "--region-pool-mode", job.cell.region_pool_mode,
        "--cell-id", job.cell.cell_id,
        "--seed", str(args.seed),
    ]
    return "ROOT_DIR_BRAINTREEBANK=" + shlex.quote(str(args.bt_root)) + " " + shlex.join(cmd)


def build_sbatch(
    job: PoolJob, report_dir: Path, command: str, args: argparse.Namespace
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job.name[:48]}",
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


def _parse_sessions(spec: str) -> list[tuple[int, int]] | None:
    if not spec:
        return None
    out: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        sub_str, trial_str = chunk.strip().split(":")
        out.append((int(sub_str), int(trial_str)))
    return out


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
    p.add_argument(
        "--region-pool-modes", default="test-anchored,source-union",
        help="Comma-separated pool modes to dispatch. Default = both.",
    )
    p.add_argument(
        "--binary-variants", default="multiclass,binary",
        help="Comma-separated set: 'multiclass', 'binary'. Default = both.",
    )
    p.add_argument(
        "--ref-kind", default="shaft_laplacian",
        choices=("raw", "bipolar", "shaft_laplacian", "global_car", "shaft_car", "median"),
        help="L.2 winner reference (R4 default).",
    )
    p.add_argument(
        "--view-kind", default="stft_abs",
        choices=("raw_voltage", "stft_abs", "hg_envelope", "log_stft"),
        help="L.2 winner view (I2 default).",
    )
    p.add_argument(
        "--normalization", default="train_set_fixed",
        help="L.1 winner (N1 default).",
    )
    p.add_argument(
        "--sessions", default="",
        help="Comma-separated subject:trial pairs; default = all BT_LITE_SESSIONS (12).",
    )
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
