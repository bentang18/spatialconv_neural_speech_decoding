"""DCC submitter: extract Whisper-large-v3 features per (subject, trial), then
run the layer + merge-strategy probe sweep over Neuroprobe Lite tasks.

Two sbatch jobs (chained by dependency):

    extract_array  - array job, one task per (subject, trial), GPU 1
    probe          - single CPU job, waits on the array, loads all NPZs

Default cohort: NEUROPROBE_LITE_SUBJECT_TRIALS (12 sessions, 6 subjects).
Override with --subject-trials sub_3:0 sub_3:1 ...

Usage (sync + sbatch from laptop):

    scripts/dcc/dispatch scripts/neuroprobe/teacher_ceiling/submit_ceiling.py \\
        --out-dir reports/neuroprobe_whisper_ceiling_2026_05_28 \\
        --wav-dir /work/ht203/data/braintreebank_wavs

Single-trial smoke (sub_3 trial0 only) for first-light:

    scripts/dcc/dispatch scripts/neuroprobe/teacher_ceiling/submit_ceiling.py \\
        --out-dir reports/neuroprobe_whisper_ceiling_smoke \\
        --wav-dir /work/ht203/data/braintreebank_wavs \\
        --subject-trials sub_3:0 \\
        --smoke
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


NEUROPROBE_LITE_SUBJECT_TRIALS = [
    (1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1),
    (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1),
]

SUBJECT_TRIAL_MOVIE = {
    (1, 1): "the-martian", (1, 2): "thor-ragnarok",
    (2, 0): "venom", (2, 4): "black-panther",
    (3, 0): "cars-2", (3, 1): "lotr-1",
    (4, 0): "shrek-the-third", (4, 1): "megamind",
    (7, 0): "cars-2", (7, 1): "megamind",
    (10, 0): "cars-2", (10, 1): "ant-man",
}


def parse_subject_trial(s: str) -> tuple[int, int]:
    """Parse 'sub_3:0' or '3:0' into (3, 0)."""
    s = s.replace("sub_", "")
    sid_str, tid_str = s.split(":")
    return int(sid_str), int(tid_str)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or Path(
        f"reports/neuroprobe_whisper_ceiling_{datetime.now():%Y_%m_%d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_cache = args.feature_cache or Path(
        "/hpc/group/coganlab/ht203/cache_neuroai/whisper_ceiling/v3"
    )
    feature_cache.mkdir(parents=True, exist_ok=True)

    if args.subject_trials:
        cohort = [parse_subject_trial(s) for s in args.subject_trials]
    else:
        cohort = list(NEUROPROBE_LITE_SUBJECT_TRIALS)
    if args.smoke:
        cohort = cohort[:1]
    print(f"[submit] cohort = {cohort}")

    # Filter to pairs with a known movie mapping and a present WAV
    valid_cohort = []
    for sid, tid in cohort:
        movie = SUBJECT_TRIAL_MOVIE.get((sid, tid))
        if movie is None:
            print(f"[submit] skip sub_{sid}_trial{tid} (no movie mapping)")
            continue
        wav = args.wav_dir / f"{movie}.wav"
        valid_cohort.append((sid, tid, movie, wav))
    if not valid_cohort:
        sys.exit("[submit] empty cohort after WAV filter")

    # --- extract_array sbatch ---
    array_size = len(valid_cohort)
    cohort_file = out_dir / "cohort.tsv"
    cohort_lines = ["subject_id\ttrial_id\tmovie\twav_path"]
    for sid, tid, movie, wav in valid_cohort:
        cohort_lines.append(f"{sid}\t{tid}\t{movie}\t{wav}")
    cohort_file.write_text("\n".join(cohort_lines) + "\n")

    extract_sbatch_path = out_dir / "submit_extract.sbatch"
    extract_sbatch_path.write_text(
        build_extract_sbatch(
            cohort_file=cohort_file,
            feature_cache=feature_cache,
            out_dir=out_dir,
            array_size=array_size,
            args=args,
        )
    )

    probe_sbatch_path = out_dir / "submit_probe.sbatch"
    probe_sbatch_path.write_text(
        build_probe_sbatch(
            feature_cache=feature_cache,
            out_dir=out_dir,
            args=args,
        )
    )

    if args.dry_run:
        print(f"[dry-run] cohort tsv: {cohort_file}")
        print(f"[dry-run] extract sbatch: {extract_sbatch_path}")
        print(f"[dry-run] probe   sbatch: {probe_sbatch_path}")
        return

    res = subprocess.run(
        ["sbatch", str(extract_sbatch_path)],
        check=True, text=True, capture_output=True,
    )
    print("[sbatch extract]", res.stdout.strip())
    extract_jobid = res.stdout.strip().split()[-1]

    res2 = subprocess.run(
        ["sbatch", "--dependency=afterok:" + extract_jobid, str(probe_sbatch_path)],
        check=True, text=True, capture_output=True,
    )
    print("[sbatch probe]", res2.stdout.strip())


def build_extract_sbatch(
    cohort_file: Path,
    feature_cache: Path,
    out_dir: Path,
    array_size: int,
    args: argparse.Namespace,
) -> str:
    lines = [
        "#!/bin/bash",
        "#SBATCH -J whisper_extract",
        f"#SBATCH -p {args.gpu_partition}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines.extend([
        f"#SBATCH --array=1-{array_size}",
        f"#SBATCH --cpus-per-task={args.extract_cpus}",
        f"#SBATCH --mem={args.extract_mem}",
        "#SBATCH --gres=gpu:1",
        f"#SBATCH -t {args.extract_time}",
        f"#SBATCH -o {out_dir.resolve()}/extract-%A_%a.out",
        f"#SBATCH -e {out_dir.resolve()}/extract-%A_%a.err",
        "",
        "set -euo pipefail",
        f"cd {args.repo_root}",
        "",
        "# Read line ${SLURM_ARRAY_TASK_ID}+1 (skip header) from cohort.tsv",
        f"COHORT={shlex.quote(str(cohort_file))}",
        'LINE=$(awk -v n=$((SLURM_ARRAY_TASK_ID + 1)) "NR==n" "$COHORT")',
        'IFS=$\'\\t\' read -r SID TID MOVIE WAV <<< "$LINE"',
        'echo "[extract] sub_${SID} trial${TID} movie=${MOVIE} wav=${WAV}"',
        "",
        ".venv/bin/python scripts/neuroprobe/teacher_ceiling/run_extract_per_trial.py \\",
        '    --subject-id "$SID" --trial-id "$TID" \\',
        '    --wav-path "$WAV" \\',
        f'    --out-dir {shlex.quote(str(feature_cache))} \\',
        f'    --model {shlex.quote(args.model)} \\',
        "    --device cuda",
        "",
    ])
    return "\n".join(lines)


def build_probe_sbatch(
    feature_cache: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> str:
    lines = [
        "#!/bin/bash",
        "#SBATCH -J whisper_probe",
        f"#SBATCH -p {args.cpu_partition}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines.extend([
        f"#SBATCH --cpus-per-task={args.probe_cpus}",
        f"#SBATCH --mem={args.probe_mem}",
        f"#SBATCH -t {args.probe_time}",
        f"#SBATCH -o {out_dir.resolve()}/probe-%j.out",
        f"#SBATCH -e {out_dir.resolve()}/probe-%j.err",
        "",
        "set -euo pipefail",
        "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        f"cd {args.repo_root}",
        "",
        ".venv/bin/python scripts/neuroprobe/teacher_ceiling/run_probe_ceiling.py \\",
        f"    --features-dir {shlex.quote(str(feature_cache))} \\",
        f"    --out-dir {shlex.quote(str(out_dir))} \\",
        "    --seed 42",
        "",
        ".venv/bin/python scripts/neuroprobe/teacher_ceiling/collect_ceiling.py \\",
        f"    --probe-csv {shlex.quote(str(out_dir / 'ceiling_results.csv'))} \\",
        f"    --out-md {shlex.quote(str(out_dir / 'SUMMARY.md'))}",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--wav-dir", type=Path,
        default=Path("/work/ht203/data/braintreebank_wavs"),
    )
    p.add_argument(
        "--feature-cache", type=Path, default=None,
        help="persistent /hpc tier dir for per-trial NPZ features",
    )
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument("--model", default="openai/whisper-large-v3")
    p.add_argument(
        "--subject-trials", nargs="*", default=None,
        help="override cohort, e.g. sub_3:0 sub_3:1",
    )
    p.add_argument("--smoke", action="store_true",
                   help="single (subject, trial) for first-light run")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--gpu-partition", default="scavenger-gpu,coganlab-gpu")
    p.add_argument("--cpu-partition", default="common,scavenger")
    p.add_argument("--extract-cpus", type=int, default=4)
    p.add_argument("--extract-mem", default="64G")
    p.add_argument("--extract-time", default="04:00:00")
    p.add_argument("--probe-cpus", type=int, default=8)
    p.add_argument("--probe-mem", default="48G")
    p.add_argument("--probe-time", default="02:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
