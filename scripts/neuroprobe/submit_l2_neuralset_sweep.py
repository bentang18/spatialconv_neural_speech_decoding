"""Submit Neuroprobe Stage 0 L.2 NeuralSet-backend sweep on DCC.

Replaces `submit_l2_reference_view_sweep.py` (degenerate — 9 cells collapsed
onto 4 upstream `preprocess_type` strings). This submitter dispatches one
sbatch per (cell × session), where cell = (reference, view) wired through
`speech_decoding.views` and the wrapper's
`--backend neuralset --ref-kind ... --view-kind ...` flags.

Tier-A grid: 3 references (raw / bipolar / shaft_laplacian) × 3 views
(raw_voltage / stft_abs / hg_envelope) = 9 distinct cells × 12 BT Lite
sessions = 108 jobs. Normalization is held at the L.1 winner.

Tier-B (R1/R2 CAR via local CARIeegExtractor; I4 multi-band; I5 wavelet) is
gated until Tier-A confirms parity and prioritization.

Usage on DCC:
    .venv/bin/python scripts/neuroprobe/submit_l2_neuralset_sweep.py \\
        --normalization train_set_fixed
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


REFERENCES_TIER_A: tuple[tuple[str, str], ...] = (
    ("R0", "raw"),
    ("R3", "bipolar"),
    ("R4", "shaft_laplacian"),
)
REFERENCES_TIER_B: tuple[tuple[str, str], ...] = (
    ("R1", "global_car"),
    ("R2", "shaft_car"),
    ("R5", "median"),
)
VIEWS_TIER_A: tuple[tuple[str, str], ...] = (
    ("I0", "raw_voltage"),
    ("I2", "stft_abs"),
    ("I3", "hg_envelope"),
)
VIEWS_TIER_B: tuple[tuple[str, str], ...] = (
    ("I1", "low_lfp"),
    ("I2L", "log_stft"),
    ("I3W", "hg_envelope_wide"),
    ("I4", "multi_band_log_power"),
    ("I5", "wavelet_db4"),
    ("I6", "instantaneous_phase"),
)
REFERENCES_ALL = REFERENCES_TIER_A + REFERENCES_TIER_B
VIEWS_ALL = VIEWS_TIER_A + VIEWS_TIER_B


def cell_id(ref_code: str, view_code: str) -> str:
    return f"L.2.{ref_code}x{view_code}"


@dataclass(frozen=True)
class L2Cell:
    cell_id: str
    ref_code: str
    ref_kind: str
    view_code: str
    view_kind: str


def build_cells(
    ref_filter: list[str] | None, view_filter: list[str] | None
) -> list[L2Cell]:
    cells: list[L2Cell] = []
    for (rc, rk), (vc, vk) in product(REFERENCES_ALL, VIEWS_ALL):
        if ref_filter and rc not in ref_filter and rk not in ref_filter:
            continue
        if view_filter and vc not in view_filter and vk not in view_filter:
            continue
        cells.append(L2Cell(cell_id(rc, vc), rc, rk, vc, vk))
    return cells


@dataclass(frozen=True)
class SweepJob:
    cell: L2Cell
    subject_id: int
    trial_id: int
    tasks: tuple[str, ...]
    normalization: str

    @property
    def name(self) -> str:
        return (
            f"l2ns_{self.cell.cell_id.lower().replace('.', '_')}_"
            f"sub{self.subject_id}_trial{self.trial_id}"
        )


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_l2_neuralset_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    sessions = _parse_sessions(args.sessions) or list(BT_LITE_SESSIONS)
    cells = build_cells(
        ref_filter=args.refs.split(",") if args.refs else None,
        view_filter=args.views.split(",") if args.views else None,
    )
    if args.cells:
        wanted = set(args.cells.split(","))
        cells = [c for c in cells if c.cell_id in wanted]
    if not cells:
        raise SystemExit("No cells selected — check --refs / --views / --cells filters")

    rows: list[dict[str, str]] = []
    for cell in cells:
        for sub_id, trial_id in sessions:
            job = SweepJob(
                cell=cell,
                subject_id=sub_id,
                trial_id=trial_id,
                tasks=NEUROPROBE_TASKS,
                normalization=args.normalization,
            )
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
                "ref_code": cell.ref_code,
                "ref_kind": cell.ref_kind,
                "view_code": cell.view_code,
                "view_kind": cell.view_kind,
                "subject_id": str(sub_id),
                "trial_id": str(trial_id),
                "normalization": args.normalization,
                "tasks": ",".join(NEUROPROBE_TASKS),
                "report_dir": str(report_dir),
                "sbatch_path": str(sbatch_path),
            })

    write_manifest(out_root / "launch_manifest.csv", rows)
    print(f"Wrote {out_root / 'launch_manifest.csv'} ({len(rows)} jobs)")


def build_command(job: SweepJob, report_dir: Path, args: argparse.Namespace) -> str:
    cmd = [
        ".venv/bin/python",
        "scripts/neuroprobe/run_stage0_linear_baseline.py",
        "--bt-root", str(args.bt_root),
        "--neuroprobe-repo", str(args.neuroprobe_repo),
        "--out-dir", str(report_dir),
        "--subject-id", str(job.subject_id),
        "--trial-id", str(job.trial_id),
        "--task", ",".join(job.tasks),
        "--split-type", "CrossSession",
        "--binary-tasks", "false",
        "--backend", "neuralset",
        "--ref-kind", job.cell.ref_kind,
        "--view-kind", job.cell.view_kind,
        "--normalization", job.normalization,
        "--seed", str(args.seed),
    ]
    return "ROOT_DIR_BRAINTREEBANK=" + shlex.quote(str(args.bt_root)) + " " + shlex.join(cmd)


def build_sbatch(
    job: SweepJob, report_dir: Path, command: str, args: argparse.Namespace
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job.name[:48]}",
        f"#SBATCH -p {args.partition}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH -t {args.time}",
        f"#SBATCH -o {report_dir.resolve()}/slurm-%j.out",
        f"#SBATCH -e {report_dir.resolve()}/slurm-%j.err",
    ]
    if args.account:
        lines.insert(3, f"#SBATCH --account={args.account}")
    lines.extend([
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
        "--normalization", default="train_set_fixed",
        help="Held fixed at L.1 winner. Default = upstream/L.1 N1 baseline.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--refs", default="",
        help="Comma-separated ref codes (R0/R3/R4) or kinds; default = all 3.",
    )
    parser.add_argument(
        "--views", default="",
        help="Comma-separated view codes (I0/I2/I3) or kinds; default = all 3.",
    )
    parser.add_argument(
        "--cells", default="",
        help="Comma-separated cell IDs (e.g. 'L.2.R4xI3'); overrides --refs/--views.",
    )
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
