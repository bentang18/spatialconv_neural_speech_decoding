#!/usr/bin/env python3
"""Single-patient / single-fold / single-seed v14-core training CLI (Task E1).

Resolves machine-specific paths from `configs/paths.yaml`, builds the per-
patient dataset, and delegates to the right fold runner. One SLURM array
task = one invocation of this script.

Two modes:

* ``--mode slot`` (default, original B-1 path): trial-level `.fif`, slot-CE
  over 3 phoneme positions, ``grid_mixer_depth`` ablation knob.
* ``--mode per-phoneme`` (baseline-aligned rework): phoneme-level `.fif`,
  flat per-phoneme CE, ``backbone_depth`` ablation knob. Full Phase-1 LH
  cohort supported.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.v14.dataset import V14TrialDataset  # noqa: E402
from speech_decoding.v14.phoneme_dataset import V14PhonemeDataset  # noqa: E402
from speech_decoding.v14.phoneme_run_fold import (  # noqa: E402
    run_one_fold as run_one_fold_per_phoneme,
)
from speech_decoding.v14.phoneme_run_fold_pooled import (  # noqa: E402
    run_one_fold_pooled,
)
from speech_decoding.v14.run_fold import run_one_fold  # noqa: E402


CORE_PATIENTS = ("S14", "S26", "S33", "S62")
PHASE1_LH_PATIENTS = ("S14", "S16", "S23", "S26", "S33", "S39", "S62")


def _repo_root() -> Path:
    return PROJECT_ROOT


def _load_paths() -> dict:
    cfg = yaml.safe_load((_repo_root() / "configs" / "paths.yaml").read_text())
    return cfg


def _resolve_fif_path(bids_root: Path, patient: str) -> Path:
    """Trial-level `.fif` per `#34` — authoritative for tokens + onsets."""

    return (
        bids_root
        / "derivatives"
        / "epoch(CAR)"
        / f"sub-{patient}"
        / "epoch(band)(power)"
        / f"sub-{patient}_task-PhonemeSequence_desc-productionZscore_highgamma.fif"
    )


def _resolve_phoneme_fif_path(bids_root: Path, patient: str) -> Path:
    """Phoneme-level `.fif` — 3·n_trials epochs at 9-phoneme event_id."""

    return (
        bids_root
        / "derivatives"
        / "epoch(phonemeLevel)(CAR)"
        / f"sub-{patient}"
        / "epoch(band)(power)"
        / f"sub-{patient}_task-PhonemeSequence_desc-productionZscore_highgamma.fif"
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_repo_root()), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("slot", "per-phoneme", "pooled"),
        default="slot",
        help="Training mode. 'slot' = original B-1 slot-CE; "
        "'per-phoneme' = single-patient baseline-aligned rework; "
        "'pooled' = multi-patient per-phoneme (Q1a).",
    )
    parser.add_argument(
        "--patient",
        default=None,
        help="Required for slot / per-phoneme modes. Ignored for pooled mode.",
    )
    parser.add_argument(
        "--patients",
        default=None,
        help="Comma-separated patient IDs for pooled mode (e.g. S14,S26,S33,S62).",
    )
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument(
        "--grid-mixer-depth",
        type=int,
        default=1,
        choices=(1, 2),
        help="Slot mode only: grid-mixer Conv2d depth.",
    )
    parser.add_argument(
        "--backbone-depth",
        type=int,
        default=3,
        choices=(1, 3, 5),
        help="Per-phoneme mode: number of attention blocks. P8 uses {1, 3}; "
        "depth=5 is the post-P8 capacity ablation.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=32,
        choices=(32, 64),
        help="Per-phoneme mode: backbone width. Default 32 (plan); 64 is the "
        "width ablation. Cascades to per-cell Conv1d + decoder.",
    )
    parser.add_argument(
        "--conv2d-kernel",
        type=int,
        default=3,
        choices=(1, 3, 5),
        help="Per-phoneme mode: Conv2d spatial kernel. Default 3. k=1 removes "
        "spatial context (pure channel-wise); k=5 is a larger receptive field.",
    )
    parser.add_argument(
        "--pool-h",
        type=int,
        default=4,
        help="Per-phoneme mode: pool target rows. Default 4.",
    )
    parser.add_argument(
        "--pool-w",
        type=int,
        default=8,
        help="Per-phoneme mode: pool target cols. Default 8.",
    )
    parser.add_argument(
        "--temporal-frontend",
        type=str,
        default="per_cell",
        choices=("per_cell", "flat"),
        help="Per-phoneme + pooled mode: temporal front-end style. 'per_cell' "
        "(default) is the current 8→d shared Conv1d. 'flat' is the baseline "
        "C·n_cells→d Conv1d — bakes full spatial mixing into the front-end.",
    )
    parser.add_argument(
        "--pool-method",
        type=str,
        default="masked_mean",
        choices=("masked_mean", "adaptive_avg"),
        help="Per-phoneme + pooled: Stage-3 pool implementation. "
        "'masked_mean' (default) uses start-index non-overlapping bins and "
        "divides by active count. 'adaptive_avg' uses torch's "
        "adaptive_avg_pool2d — overlapping bins on non-divisible W, divisor = "
        "fixed bin size. Matches baseline pool.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Per-phoneme + pooled: label smoothing on the CE target. "
        "Baseline recipe uses 0.1 (−4.1pp autoresearch delta). Default 0.0.",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="Per-phoneme + pooled: mixup Beta α on the signal. Baseline "
        "recipe uses 0.2 (−2.9pp autoresearch delta). Default 0.0.",
    )
    parser.add_argument(
        "--no-parcel-embedding",
        action="store_true",
        help="Pooled mode: disable the soft parcel embedding (ablation — "
        "tests whether the atlas anchor actually helps the pooled model).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: 5 epochs, val every epoch, no early-stop. "
        "For pipeline validation, not for real results.",
    )
    args = parser.parse_args(argv)

    pooled_patients: tuple[str, ...] = ()
    if args.mode == "slot":
        if args.patient not in CORE_PATIENTS:
            parser.error(
                f"slot mode supports core patients {CORE_PATIENTS}; got {args.patient}"
            )
    elif args.mode == "per-phoneme":
        if args.patient not in PHASE1_LH_PATIENTS:
            parser.error(
                f"per-phoneme mode supports Phase-1 LH patients {PHASE1_LH_PATIENTS}; "
                f"got {args.patient}"
            )
    else:  # pooled
        if not args.patients:
            parser.error("--mode pooled requires --patients S1,S2,...")
        pooled_patients = tuple(p.strip() for p in args.patients.split(","))
        bad = [p for p in pooled_patients if p not in PHASE1_LH_PATIENTS]
        if bad:
            parser.error(
                f"pooled mode patients must be Phase-1 LH {PHASE1_LH_PATIENTS}; "
                f"got invalid {bad}"
            )
        if len(pooled_patients) < 2:
            parser.error("pooled mode needs ≥2 patients")

    paths = _load_paths()
    bids_root = Path(paths["ps_bids_root"])
    support_cache_dir = Path(
        paths.get(
            "support_cache_dir",
            _repo_root() / "data" / "atlas" / "support_cache_v2c_snap",
        )
    )
    channel_maps_dir = Path(
        paths.get("channel_maps_dir", _repo_root() / "data" / "channel_maps")
    )
    box_root = Path(
        paths.get(
            "box_root",
            Path.home() / "Library" / "CloudStorage" / "Box-Box" / "ECoG_Recon",
        )
    )

    t0 = time.time()
    smoke_kw = (
        {"max_epochs": 5, "val_every": 1, "patience": 100} if args.smoke else {}
    )

    if args.mode == "slot":
        support_cache_path = support_cache_dir / f"{args.patient}_support_tier1.csv"
        fif_path = _resolve_fif_path(bids_root, args.patient)
        dataset = V14TrialDataset(
            args.patient,
            fif_path,
            support_cache_path=support_cache_path,
            channel_maps_dir=channel_maps_dir,
            box_root=box_root,
        )
        result = run_one_fold(
            dataset,
            fold_idx=args.fold,
            seed=args.seed,
            grid_mixer_depth=args.grid_mixer_depth,
            out_dir=args.out_dir,
            patient_id=args.patient,
            **smoke_kw,
        )
        tag = (
            f"{args.patient}_fold{args.fold}_seed{args.seed}"
            f"_depth{args.grid_mixer_depth}"
        )
    elif args.mode == "per-phoneme":
        support_cache_path = support_cache_dir / f"{args.patient}_support_tier1.csv"
        fif_path = _resolve_phoneme_fif_path(bids_root, args.patient)
        dataset = V14PhonemeDataset(
            args.patient,
            fif_path,
            support_cache_path=support_cache_path,
            channel_maps_dir=channel_maps_dir,
            box_root=box_root,
        )
        result = run_one_fold_per_phoneme(
            dataset,
            fold_idx=args.fold,
            seed=args.seed,
            backbone_depth=args.backbone_depth,
            d_model=args.d_model,
            conv2d_kernel=args.conv2d_kernel,
            pool_shape=(args.pool_h, args.pool_w),
            temporal_frontend=args.temporal_frontend,
            pool_method=args.pool_method,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            out_dir=args.out_dir,
            patient_id=args.patient,
            **smoke_kw,
        )
        tag = (
            f"{args.patient}_fold{args.fold}_seed{args.seed}"
            f"_depth{args.backbone_depth}"
        )
    else:  # pooled
        datasets: dict[str, V14PhonemeDataset] = {}
        for pt in pooled_patients:
            fif_path = _resolve_phoneme_fif_path(bids_root, pt)
            datasets[pt] = V14PhonemeDataset(
                pt,
                fif_path,
                support_cache_path=support_cache_dir / f"{pt}_support_tier1.csv",
                channel_maps_dir=channel_maps_dir,
                box_root=box_root,
            )
        result = run_one_fold_pooled(
            datasets,
            fold_idx=args.fold,
            seed=args.seed,
            backbone_depth=args.backbone_depth,
            d_model=args.d_model,
            conv2d_kernel=args.conv2d_kernel,
            pool_shape=(args.pool_h, args.pool_w),
            use_parcel_embedding=not args.no_parcel_embedding,
            temporal_frontend=args.temporal_frontend,
            pool_method=args.pool_method,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            out_dir=args.out_dir,
            **smoke_kw,
        )
        patient_slug = "_".join(sorted(pooled_patients))
        tag = (
            f"pooled_{patient_slug}_fold{args.fold}_seed{args.seed}"
            f"_depth{args.backbone_depth}"
        )

    wall_s = time.time() - t0

    # Augment on-disk result with reproducibility metadata; the runner already
    # wrote the primary fields.
    result["mode"] = args.mode
    result["wall_s"] = wall_s
    result["git_sha"] = _git_sha()
    result["hostname"] = socket.gethostname()
    result["timestamp_unix"] = time.time()
    (args.out_dir / f"{tag}.result.json").write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
