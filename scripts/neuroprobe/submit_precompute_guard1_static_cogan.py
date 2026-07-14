"""DCC submitter: GUARD-1 STATIC precompute for the Cogan D-cohort — one SLURM-array
task per manifest run. Cogan sibling of ``submit_precompute_guard1_static.py``.

The per-run voltage scan is CPU-bound and embarrassingly parallel, so this submits
a SLURM ARRAY on the CPU partitions. Array task ``i`` runs
``precompute_guard1_static_cogan.py --row i`` (via ``SLURM_ARRAY_TASK_ID``), scanning
the i-th manifest run: full neural montage ``cogan_load_raw(extra_bad=())`` ->
notch(power_line_hz)/HPF (no CAR) -> ``static_bad_mask`` -> per-run signature JSON.

Reads only the synced code + the Cogan BIDS EDF tree + the run manifest — NOT any
spec cache — so it runs before the 3-band cache build it informs. Unlike the BT
submitter it exports NO ROOT_DIR: ``cogan_load_raw`` takes explicit EDF/channels
paths straight from the manifest row. Uses the probe_bench venv by absolute path
(the repo/speech clone has no ``.venv`` of its own).

Output dir holds tiny ``cogan{sid}_t{tid}.json`` signatures; the collector re-derives
the per-run static drop from ``clip_frac``/``rmad`` via ``guard1.classify_from_signature``.

Usage (run ON DCC after ``git pull`` in the clone):

    /work/ht203/probe_bench/.venv/bin/python \\
      scripts/neuroprobe/submit_precompute_guard1_static_cogan.py \\
        --out-dir /work/ht203/cogan_v3/guard1/detections

Single-run smoke (first-light, one task):

    ... submit_precompute_guard1_static_cogan.py --array-range 0-0 \\
        --out-dir /work/ht203/cogan_v3/guard1/detections_smoke

Preflight without submitting (writes + prints the .sbatch):

    ... submit_precompute_guard1_static_cogan.py --dry-run \\
        --out-dir /work/ht203/cogan_v3/guard1/detections
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

# 938 runs in cogan_manifest.csv (build_cogan_manifest.py, DCC-verified 2026-07-13)
# -> array indices 0-937.
_DEFAULT_ARRAY = "0-937"
_DEFAULT_MANIFEST = "/work/ht203/cogan_v3/manifest/cogan_manifest.csv"


def build_sbatch(log_dir: Path, args: argparse.Namespace) -> str:
    lines = [
        "#!/bin/bash",
        "#SBATCH -J cogan_guard1_static",
        f"#SBATCH -p {args.cpu_partition}",
        f"#SBATCH --array={args.array_range}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines.extend([
        "#SBATCH --requeue",  # scavenger is preemptible -> requeue on preemption
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH -t {args.time}",
        f"#SBATCH -o {log_dir.resolve()}/g1c-%A_%a.out",
        f"#SBATCH -e {log_dir.resolve()}/g1c-%A_%a.err",
        "",
        "set -euo pipefail",
        f"cd {args.repo_root}",
        "",
        # scripts/ is NOT a package -> run by PATH, not -m. The scan reads
        # SLURM_ARRAY_TASK_ID to pick its manifest row (resolve_rows).
        f"{shlex.quote(args.python)} -u scripts/neuroprobe/precompute_guard1_static_cogan.py \\",
        f"    --manifest {shlex.quote(str(args.manifest))} \\",
        f"    --out-dir {shlex.quote(str(args.out_dir))}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir or Path(
        f"reports/guard1_static_cogan_{datetime.now():%Y_%m_%d}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    sbatch_path = log_dir / "submit_guard1_static_cogan.sbatch"
    sbatch_path.write_text(build_sbatch(log_dir=log_dir, args=args))

    print(f"[submit] manifest  = {args.manifest}")
    print(f"[submit] out_dir   = {args.out_dir}")
    print(f"[submit] array     = {args.array_range} (one task per manifest run)")
    print(f"[submit] partition = {args.cpu_partition}")
    print(f"[submit] sbatch    = {sbatch_path}")

    if args.dry_run:
        print("[dry-run] sbatch contents:\n" + sbatch_path.read_text())
        return

    res = subprocess.run(
        ["sbatch", str(sbatch_path)],
        check=True, text=True, capture_output=True,
    )
    print("[sbatch cogan-guard1-static]", res.stdout.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=_DEFAULT_MANIFEST,
                   help="cogan_manifest.csv (row order = array order).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Signature output dir. One cogan{sid}_t{tid}.json per task.")
    p.add_argument("--array-range", default=_DEFAULT_ARRAY,
                   help="SLURM array range = manifest row indices (938 runs -> 0-937). "
                        "Use 0-0 for a single-run smoke; append %%N to throttle concurrency.")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="Where the .sbatch + slurm logs land "
                        "(default reports/guard1_static_cogan_<date>).")
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument("--python", default="/work/ht203/probe_bench/.venv/bin/python",
                   help="Interpreter (absolute — the repo/speech clone has no .venv).")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpu-partition", default="common,scavenger",
                   help="CPU partitions for the array (SLURM picks whichever has room).")
    p.add_argument("--cpus", type=int, default=8)
    # Full-montage (C,N) float64 + dV/dt copies; the longest Cogan runs
    # (SentenceRep/TIMIT, mixed native rates -> 2048) fit comfortably in 96G, the
    # same envelope BT used for its full-montage static scan.
    p.add_argument("--mem", default="96G")
    p.add_argument("--time", default="02:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
