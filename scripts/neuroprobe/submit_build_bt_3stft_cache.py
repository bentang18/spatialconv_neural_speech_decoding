#!/usr/bin/env python
"""Submit the 3STFT (2/2/2) per-band spec-cache build to DCC (cache-only, CPU).

Mechanism (per-band cache-only passes — see the 2026-06-17 cache recommendation):
each of the 3 locked bands (slow | beta | hg, ``reports/fe_3stft_2of2of2_spec_
2026_06_17.md``) is built in its OWN ``--cache-only`` pass, riding the single-grid
``electrode_tokens`` slot via ``dispatch_v14 --cache-band <band>``. No encoder /
Data change is needed — ``--cache-only`` exits before the trainer, and the band
view is constructed identically to a future ``--frontend 3stft`` run, so that run
HITs these caches (verify with analyze_3stft_cache.py).

Two corpora, both baked with STATIC bad-electrode exclusion (the only cache-bakeable
guard; WINSOR + CLIP stay runtime, NOT baked):
  - FULL pretrain : --phase 1 --mode full --electrode-set all, 13 sessions
                    (_SESSIONS_BY_MODE['pretrain']).
  - LITE eval     : --phase 4 --mode lite --electrode-set lite --frozen-probe,
                    12 sessions (_SESSIONS_BY_MODE['lite']). Each session is pinned
                    via --eval-mode WithinSession --test-subject-id/--test-trial-id
                    so the eval split materializes exactly that (S,T) (the whole-
                    movie |STFT| memmap is split-independent → byte-identical to
                    what a CrossSession/CrossSubject run memmap-slices back).

Array = (band x session): pretrain 13x3 = 39, eval 12x3 = 36. Each task is an
independent CPU job; rerun-failed rescues individual band x session cells.

EXCA_EXTRACTOR_CACHE_FOLDER MUST be fresh: a new band geometry already yields a new
extractor uid, but pointing it at a dedicated dir guarantees raw _get_data flows
through bt_load_raw so the STATIC drop actually lands (the 2026-06-15 stale-raw-exca
no-op trap). The run then reads BOTH this and --spec-cache-dir.

Usage (via scripts/dcc/dispatch so the run is logged):

    scripts/dcc/dispatch scripts/neuroprobe/submit_build_bt_3stft_cache.py \\
        --spec-cache-dir /work/ht203/cache_neuroai/v14_3stft_spec_cache \\
        --extractor-cache-folder /work/ht203/cache/v14_extractors_3stft \\
        --corpus both

Single-band single-session smoke first:

    scripts/dcc/dispatch scripts/neuroprobe/submit_build_bt_3stft_cache.py \\
        --spec-cache-dir /work/ht203/cache_neuroai/v14_3stft_smoke \\
        --extractor-cache-folder /work/ht203/cache/v14_extractors_3stft_smoke \\
        --corpus pretrain --bands slow --array-range 0-0
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

# Mirrors _SESSIONS_BY_MODE['lite'] in dispatch_v14 (array index -> (S, T)). Kept
# here so the sbatch can embed the (S, T) map for WithinSession pinning without
# importing dispatch on the login node. A guard in main() re-checks it against the
# live dispatch table so they can never silently drift.
_LITE_SESSIONS: tuple[tuple[int, int], ...] = (
    (1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1),
    (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1),
)
_BANDS = ("slow", "beta", "hg")


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
        f"#SBATCH -J bt3stft_{job_tag}",
        f"#SBATCH -p {args.cpu_partition}",
        f"#SBATCH --array={array_range}",
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
        f"reports/bt_3stft_cache_build_{datetime.now():%Y_%m_%d}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    corpora = ["pretrain", "eval"] if args.corpus == "both" else [args.corpus]
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
            print(f"[sbatch 3stft {corpus}/{band}] {res.stdout.strip()}")
    if args.dry_run:
        print("\n(dry-run: no jobs submitted)")
    else:
        print(f"\nlogs + sbatch: {log_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec-cache-dir", type=Path, required=True,
                   help="Spec-cache root. Each band lands in <root>/band_<name>. The "
                        "training run MUST pass the SAME --spec-cache-dir to read these.")
    p.add_argument("--extractor-cache-folder", type=Path, required=True,
                   help="EXCA_EXTRACTOR_CACHE_FOLDER (electrode_tokens raw-waveform "
                        "cache). MUST be fresh so the STATIC drop lands (not a stale "
                        "pre-drop reuse). New band geometry already forces a new uid; "
                        "a dedicated dir makes it explicit.")
    p.add_argument("--corpus", choices=("pretrain", "eval", "both"), default="both",
                   help="pretrain (13 full-montage SSL sessions) + eval (12 lite "
                        "leaderboard sessions). Default both.")
    p.add_argument("--bands", default="slow,beta,hg",
                   help="Comma list of bands to build (default all 3).")
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
                        "to the per-band spec memmap (atlas affects parcel support, not "
                        "the band STFT); default dkt to match the run.")
    p.add_argument("--clip-len", type=float, default=5.0,
                   help="Cache-irrelevant (whole-movie spec memmap is clip-independent); "
                        "kept for build parity.")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpu-partition", default="common")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default="64G")
    p.add_argument("--time", default="03:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
