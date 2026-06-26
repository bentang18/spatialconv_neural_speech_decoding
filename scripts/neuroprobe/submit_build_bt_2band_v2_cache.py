#!/usr/bin/env python
"""Submit the converged-arch-v2 2-band (LFS+HGA) magnitude spec-cache build to
DCC, MASSIVELY PARALLEL across common+scavenger (cache-only, CPU).

Mechanism (identical to the 3STFT per-band builder, new bands): each of the 2
locked v2 bands rides the single-grid ``electrode_tokens`` slot via
``dispatch_v14 --cache-band {lfs,hga}`` (--cache-only exits before the trainer,
so no v2 model is needed). The band view is constructed exactly as a future
``--frontend`` v2 run will build it (shared STFT_2BAND_{LFS,HGA} +
common_fe_kwargs + band_<name> spec-cache subdir), so the run HITs this cache.

  LFS  N=1024 hop=512  2-56 Hz  |STFT| → 28 bins (k1..k28)
  HGA  N=128  hop=64   64-160Hz |STFT| →  7 bins (k4..k10)

Both magnitude (NO phase). STATIC bad-electrode exclusion bakes in (the only
cache-bakeable guard, band-independent via anatomy.extra_bad_electrodes); WINSOR
+ CLIP stay runtime, NOT baked.

PARALLELISM: one independent CPU array task per (band x session) — pretrain
13x2 = 26, eval 12x2 = 24 → 50 tasks. Default partition ``common,scavenger`` so
SLURM lands each task wherever a node is free (scavenger = preemptible but huge);
``--requeue`` re-runs any preempted task from scratch (each task is independent +
idempotent; rerun-failed rescues stragglers). This is the massively-parallel
overnight build.

CACHE DIRS MUST BE FRESH AND PER-CORPUS. A new band geometry already forces a new
extractor uid, but a dedicated EXCA_EXTRACTOR_CACHE_FOLDER guarantees raw
_get_data flows through bt_load_raw so the STATIC drop lands (the 2026-06-15
stale-raw-exca no-op trap). pretrain (--electrode-set all) and eval
(--electrode-set lite, cap<=120) bake DIFFERENT per-session memmaps, so each
corpus needs its OWN --spec-cache-dir + --extractor-cache-folder (run the script
once per corpus). The run then reads BOTH --spec-cache-dir and the extractor dir.

Usage (via scripts/dcc/dispatch so the run is logged) — run PER CORPUS:

    # full pretrain (13 full-montage SSL sessions)
    scripts/dcc/dispatch scripts/neuroprobe/submit_build_bt_2band_v2_cache.py \\
        --corpus pretrain \\
        --spec-cache-dir /work/ht203/cache_neuroai/v14_2band_v2_spec_pretrain \\
        --extractor-cache-folder /work/ht203/cache/v14_extractors_2band_v2_pretrain

    # lite eval (12 leaderboard sessions)
    scripts/dcc/dispatch scripts/neuroprobe/submit_build_bt_2band_v2_cache.py \\
        --corpus eval \\
        --spec-cache-dir /work/ht203/cache_neuroai/v14_2band_v2_spec_lite \\
        --extractor-cache-folder /work/ht203/cache/v14_extractors_2band_v2_lite

Single-band single-session smoke first:

    ... --corpus pretrain --bands hga --array-range 0-0 \\
        --spec-cache-dir /work/ht203/cache_neuroai/v14_2band_v2_smoke \\
        --extractor-cache-folder /work/ht203/cache/v14_extractors_2band_v2_smoke
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

# Mirrors _SESSIONS_BY_MODE['lite'] in dispatch_v14 (array index -> (S, T)) so the
# sbatch can embed the (S, T) map for WithinSession pinning without importing
# dispatch on the login node. main() re-checks it against the live table.
_LITE_SESSIONS: tuple[tuple[int, int], ...] = (
    (1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1),
    (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1),
)
_BANDS = ("lfs", "hga")


def _verify_lite_sessions() -> None:
    """Fail loud if the embedded lite (S, T) map drifts from dispatch's table."""
    try:
        from speech_decoding.experiments.dispatch_v14 import _SESSIONS_BY_MODE
    except Exception as exc:  # pragma: no cover - login-node import guard
        print(f"[warn] could not import dispatch to verify lite sessions: {exc}")
        return
    live = tuple(tuple(int(x) for x in st) for st in _SESSIONS_BY_MODE["lite"])
    if live != _LITE_SESSIONS:
        raise SystemExit(
            "lite (S,T) map drift: submitter has\n"
            f"  {_LITE_SESSIONS}\nbut dispatch _SESSIONS_BY_MODE['lite'] has\n"
            f"  {live}\nUpdate _LITE_SESSIONS to match."
        )


def build_sbatch(log_dir: Path, corpus: str, band: str, args: argparse.Namespace) -> str:
    if corpus == "pretrain":
        n_sessions, job_tag = 13, f"pre_{band}"
    else:
        n_sessions, job_tag = 12, f"lite_{band}"
    array_range = args.array_range or f"0-{n_sessions - 1}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH -J bt2band_{job_tag}",
        f"#SBATCH -p {args.cpu_partition}",
        f"#SBATCH --array={array_range}",
        "#SBATCH --requeue",  # scavenger preemption → re-run the (band,session) task
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines.extend([
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH -t {args.time}",
        f"#SBATCH -o {log_dir.resolve()}/cache-{corpus}-{band}-%A_%a.out",
        f"#SBATCH -e {log_dir.resolve()}/cache-{corpus}-{band}-%A_%a.err",
        "",
        "set -euo pipefail",
        f"cd {args.repo_root}",
        f"export ROOT_DIR_BRAINTREEBANK={shlex.quote(str(args.root_dir_braintreebank))}",
        f"export EXCA_EXTRACTOR_CACHE_FOLDER={shlex.quote(str(args.extractor_cache_folder))}",
    ])
    if args.pythonpath:
        lines.append(f"export PYTHONPATH={shlex.quote(str(args.pythonpath))}")
    lines.append("")

    spec_dir = shlex.quote(str(args.spec_cache_dir))
    py = shlex.quote(args.python)
    common_tail = [
        f"    --atlas {shlex.quote(args.atlas)} \\",
        f"    --clip-len {args.clip_len} \\",
        f"    --spec-cache-dir {spec_dir} \\",
        "    --num-workers 0 \\",
        f"    --cache-only --cache-band {band} \\",
        "    --cache-session-index ${SLURM_ARRAY_TASK_ID}",
    ]
    if corpus == "pretrain":
        # SSL pretrain corpus (study_mode 'pretrain', 13 sess). electrode_set 'all'
        # = full montage; STATIC drop applied pre-CAR by anatomy.extra_bad_electrodes.
        cmd = [
            f"{py} -m speech_decoding.experiments.dispatch_v14 \\",
            "    --phase 1 --mode full --electrode-set all --pool mean \\",
        ] + common_tail
        lines.extend(cmd)
    else:
        # LITE eval corpus (study_mode 'lite', 12 sess). Pin the eval split to the
        # exact (S, T) at this array index so the session materializes (the spec
        # memmap is split-independent → identical to a CrossSession/CrossSubject read).
        subj = " ".join(str(s) for s, _ in _LITE_SESSIONS)
        trial = " ".join(str(t) for _, t in _LITE_SESSIONS)
        lines.extend([
            f"SUBJ=({subj})",
            f"TRIAL=({trial})",
            'i=${SLURM_ARRAY_TASK_ID}',
            "",
            f"{py} -m speech_decoding.experiments.dispatch_v14 \\",
            "    --phase 4 --mode lite --electrode-set lite --frozen-probe --pool mean \\",
            "    --eval-mode WithinSession \\",
            '    --test-subject-id ${SUBJ[$i]} --test-trial-id ${TRIAL[$i]} \\',
        ] + common_tail)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    _verify_lite_sessions()
    log_dir = args.log_dir or Path(
        f"reports/bt_2band_v2_cache_build_{datetime.now():%Y_%m_%d}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    corpora = ["pretrain", "eval"] if args.corpus == "both" else [args.corpus]
    if args.corpus == "both":
        print("[warn] --corpus both reuses ONE pair of cache dirs; pretrain "
              "(all electrodes) + eval (lite cap) bake DIFFERENT per-session "
              "memmaps. For authenticity run the script ONCE PER CORPUS with "
              "distinct --spec-cache-dir/--extractor-cache-folder.")
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    bad = [b for b in bands if b not in _BANDS]
    if bad:
        raise SystemExit(f"unknown band(s) {bad}; choose from {_BANDS}")

    for corpus in corpora:
        for band in bands:
            sbatch_text = build_sbatch(log_dir, corpus, band, args)
            sbatch_path = log_dir / f"cache_{corpus}_{band}.sbatch"
            sbatch_path.write_text(sbatch_text)
            if args.dry_run:
                print(f"\n===== {sbatch_path} =====")
                print(sbatch_text)
                continue
            res = subprocess.run(
                ["sbatch", str(sbatch_path)],
                capture_output=True, text=True, check=True,
            )
            print(f"[sbatch 2band {corpus}/{band}] {res.stdout.strip()}")
    if args.dry_run:
        print("\n(dry-run: no jobs submitted)")
    else:
        print(f"\nlogs + sbatch: {log_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec-cache-dir", type=Path, required=True,
                   help="Spec-cache root. Each band lands in <root>/band_<name>. The "
                        "v2 run MUST pass the SAME --spec-cache-dir to read these. "
                        "Use a FRESH, corpus-specific dir.")
    p.add_argument("--extractor-cache-folder", type=Path, required=True,
                   help="EXCA_EXTRACTOR_CACHE_FOLDER (electrode_tokens raw-waveform "
                        "cache). MUST be fresh so the STATIC drop lands (not a stale "
                        "pre-drop reuse). Corpus-specific.")
    p.add_argument("--corpus", choices=("pretrain", "eval", "both"), default="pretrain",
                   help="pretrain (13 full-montage SSL sessions) OR eval (12 lite "
                        "leaderboard sessions). Run ONCE PER CORPUS with distinct "
                        "dirs (default pretrain). 'both' shares dirs — see warning.")
    p.add_argument("--bands", default="lfs,hga",
                   help="Comma list of v2 bands to build (default both).")
    p.add_argument("--array-range", default=None,
                   help="Override the SLURM array range (default 0-12 pretrain / "
                        "0-11 eval). Use 0-0 for a single-session smoke.")
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument("--root-dir-braintreebank", default="/work/ht203/data/braintreebank")
    p.add_argument("--python", default=".venv/bin/python")
    p.add_argument("--pythonpath", default=None)
    p.add_argument("--atlas", default="dkt", choices=("dk", "dkt"),
                   help="Atlas for the support/valid_mask extractors. Cache-irrelevant "
                        "to the per-band spec memmap; default dkt to match the v2 run.")
    p.add_argument("--clip-len", type=float, default=5.0,
                   help="Cache-irrelevant (whole-movie spec memmap is clip-independent); "
                        "kept for build parity.")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpu-partition", default="common,scavenger",
                   help="Comma list → SLURM lands each task wherever a node is free "
                        "(massively parallel). Default common,scavenger.")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default="160G",
                   help="Per-task memory. 160G clears the 3 largest sessions that "
                        "OOM'd the 3STFT build at 64G (rerun-failed @192G if any die).")
    p.add_argument("--time", default="04:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
