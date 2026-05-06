"""Submit Neuroprobe Stage 0 norm × view interaction sanity sweep.

Tag-along to L.1 + L.2: tests whether the L.1 winner survives at the L.2
winner's reference/view, and whether ref × norm × view has hidden three-way
interactions. Each cell = (ref, view, normalization) chosen as the top-2
levels of each factor from the prior sweeps.

Default: 2 refs × 2 views × 2 norms = 8 cells × 12 BT Lite sessions = 96 jobs.

Usage on DCC (after L.1 + L.2 winners identified):
    .venv/bin/python scripts/neuroprobe/submit_l4_norm_view_interaction.py \\
        --refs shaft_laplacian,bipolar \\
        --views hg_envelope,stft_abs \\
        --norms train_set_fixed,train_set_robust_mad
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

from speech_decoding.studies.braintreebank.labels import NEUROPROBE_TASKS
from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS


REF_CODES: dict[str, str] = {
    "raw": "R0", "bipolar": "R3", "shaft_laplacian": "R4",
    "global_car": "R1", "shaft_car": "R2",
}
VIEW_CODES: dict[str, str] = {
    "raw_voltage": "I0", "stft_abs": "I2", "hg_envelope": "I3",
}


@dataclass(frozen=True)
class InterCell:
    cell_id: str
    ref_kind: str
    view_kind: str
    normalization: str


def _norm_short(norm: str) -> str:
    return (
        norm.replace("train_set_fixed", "N1")
            .replace("per_session_fixed", "N2")
            .replace("train_set_scale_only", "N3")
            .replace("none", "N4")
            .replace("per_session_robust_mad", "N5")
            .replace("train_set_robust_mad", "N6")
            .replace("per_session_robust_scale", "N7")
            .replace("per_channel_train_set_z", "N8")
            .replace("per_window_z", "N0")
    )


def build_cells(refs: list[str], views: list[str], norms: list[str]) -> list[InterCell]:
    cells: list[InterCell] = []
    for ref, view, norm in product(refs, views, norms):
        rc = REF_CODES.get(ref, ref)
        vc = VIEW_CODES.get(view, view)
        nc = _norm_short(norm)
        cells.append(InterCell(
            cell_id=f"L.4i.{rc}x{vc}x{nc}",
            ref_kind=ref, view_kind=view, normalization=norm,
        ))
    return cells


@dataclass(frozen=True)
class InterJob:
    cell: InterCell
    subject_id: int
    trial_id: int

    @property
    def name(self) -> str:
        return (
            f"l4i_{self.cell.cell_id.lower().replace('.', '_')}_"
            f"sub{self.subject_id}_trial{self.trial_id}"
        )


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_l4_norm_view_interaction_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    sessions = _parse_sessions(args.sessions) or list(BT_LITE_SESSIONS)
    refs = [r.strip() for r in args.refs.split(",") if r.strip()]
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    norms = [n.strip() for n in args.norms.split(",") if n.strip()]
    cells = build_cells(refs, views, norms)
    if not cells:
        raise SystemExit("No cells selected — check --refs / --views / --norms.")

    rows: list[dict[str, str]] = []
    for cell in cells:
        for sub_id, trial_id in sessions:
            job = InterJob(cell=cell, subject_id=sub_id, trial_id=trial_id)
            report_dir = out_root / cell.cell_id / f"sub{sub_id}_trial{trial_id}"
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
                "cell": cell.cell_id,
                "ref_kind": cell.ref_kind,
                "view_kind": cell.view_kind,
                "normalization": cell.normalization,
                "subject_id": str(sub_id),
                "trial_id": str(trial_id),
                "report_dir": str(report_dir),
                "sbatch_path": str(sbatch_path),
            })

    write_manifest(out_root / "launch_manifest.csv", rows)
    print(f"Wrote {out_root / 'launch_manifest.csv'} ({len(rows)} jobs)")


def build_command(job: InterJob, report_dir: Path, args: argparse.Namespace) -> str:
    cmd = [
        ".venv/bin/python",
        "scripts/neuroprobe/run_stage0_linear_baseline.py",
        "--bt-root", str(args.bt_root),
        "--neuroprobe-repo", str(args.neuroprobe_repo),
        "--out-dir", str(report_dir),
        "--subject-id", str(job.subject_id),
        "--trial-id", str(job.trial_id),
        "--task", ",".join(NEUROPROBE_TASKS),
        "--split-type", "CrossSession",
        "--binary-tasks", "false",
        "--backend", "neuralset",
        "--ref-kind", job.cell.ref_kind,
        "--view-kind", job.cell.view_kind,
        "--normalization", job.cell.normalization,
        "--seed", str(args.seed),
    ]
    return "ROOT_DIR_BRAINTREEBANK=" + shlex.quote(str(args.bt_root)) + " " + shlex.join(cmd)


def build_sbatch(
    job: InterJob, report_dir: Path, command: str, args: argparse.Namespace
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--bt-root", type=Path, default=Path("/work/ht203/data/braintreebank"))
    parser.add_argument(
        "--neuroprobe-repo", type=Path,
        default=Path("/work/ht203/repo/neuroprobe_upstream"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    parser.add_argument(
        "--refs", required=True,
        help="Comma-separated ref-kinds (e.g. shaft_laplacian,bipolar)",
    )
    parser.add_argument(
        "--views", required=True,
        help="Comma-separated view-kinds (e.g. hg_envelope,stft_abs)",
    )
    parser.add_argument(
        "--norms", required=True,
        help="Comma-separated normalization choices (L.1 winner + runner-up)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sessions", default="",
        help="Comma-separated subject:trial pairs; default = all 12 BT Lite sessions.",
    )
    parser.add_argument("--partition", default="common,scavenger,coganlab-gpu")
    parser.add_argument("--account", default="coganlab")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", default="24G")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
