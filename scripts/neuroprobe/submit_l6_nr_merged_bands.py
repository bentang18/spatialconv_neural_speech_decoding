"""Submit Neuroprobe Stage 0 L.6.NR merged-band envelope sweep on DCC.

Three merged-band envelopes that complement the disjoint sub-band sweep
(submit_l6_nr_hg_subbands.py + submit_l6_nr_extra_bands.py +
submit_l6_nr_band_sweep_cross_subject.py):

- envelope_30_500: kitchen sink. Pins down bandwidth-vs-structure
  question. If this matches stft_abs (~0.613 CSession), the I3->I2
  gap is BANDWIDTH; if it stays ~0.558 (broadband HG), the gap is
  spectral STRUCTURE.
- envelope_70_500: HG + above. Tests if supra-HG / MUA add on top of HG.
- envelope_30_150: HG + below. Tests if sub-HG adds on top of HG.

Dispatches BOTH protocols in one launch (CSession 12 sessions + CSubject
10 sessions, subject 2 excluded for CSubject). L.1.N1 winner norm +
L.2 R4 winner ref. 3 cells x 22 sessions = 66 jobs at 64G.
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

# (cell_id, view_kind)
MERGED_BANDS: tuple[tuple[str, str], ...] = (
    ("L.6.NR.B30_500",  "envelope_30_500"),
    ("L.6.NR.B70_500",  "envelope_70_500"),
    ("L.6.NR.B30_150",  "envelope_30_150"),
)


@dataclass(frozen=True)
class Job:
    cell_id: str
    view_kind: str
    split_type: str
    subject_id: int
    trial_id: int


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_l6_nr_merged_bands_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    cs_sessions = list(BT_LITE_SESSIONS)
    csubj_sessions = [(s, t) for (s, t) in BT_LITE_SESSIONS if s != TRAIN_SUBJECT_ID]

    rows: list[dict[str, str]] = []
    for cell_id, view_kind in MERGED_BANDS:
        for split_type, sessions in (("CrossSession", cs_sessions),
                                     ("CrossSubject", csubj_sessions)):
            for sub_id, trial_id in sessions:
                job = Job(cell_id, view_kind, split_type, sub_id, trial_id)
                report_dir = out_root / cell_id / split_type / f"sub{sub_id}_trial{trial_id}"
                report_dir.mkdir(parents=True, exist_ok=True)
                command = build_command(job, report_dir, args)
                sbatch_path = report_dir / "submit.sbatch"
                tag = "cs" if split_type == "CrossSession" else "cu"
                sbatch_path.write_text(build_sbatch(
                    f"l6nr_{tag}_{cell_id.split('.')[-1]}_s{sub_id}t{trial_id}",
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
                    "cell_id": cell_id,
                    "view_kind": view_kind,
                    "split_type": split_type,
                    "subject_id": str(sub_id),
                    "trial_id": str(trial_id),
                    "report_dir": str(report_dir),
                    "sbatch_path": str(sbatch_path),
                })

    write_manifest(out_root / "launch_manifest.csv", rows)
    print(f"Wrote {out_root / 'launch_manifest.csv'} ({len(rows)} jobs)")


def build_command(job: Job, report_dir: Path, args: argparse.Namespace) -> str:
    cmd = [
        ".venv/bin/python",
        "scripts/neuroprobe/run_stage0_linear_baseline.py",
        "--bt-root", str(args.bt_root),
        "--neuroprobe-repo", str(args.neuroprobe_repo),
        "--out-dir", str(report_dir),
        "--subject-id", str(job.subject_id),
        "--trial-id", str(job.trial_id),
        "--task", ",".join(NEUROPROBE_TASKS),
        "--split-type", job.split_type,
        "--binary-tasks", "false",
        "--backend", "neuralset",
        "--ref-kind", "shaft_laplacian",
        "--view-kind", job.view_kind,
        "--normalization", "train_set_fixed",
        "--cell-id", job.cell_id,
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
