#!/usr/bin/env python3
"""LOPO cross-patient training script."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.training.lopo import run_lopo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PS_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S36", "S39", "S57", "S58", "S62"]


def main() -> None:
    parser = argparse.ArgumentParser(description="LOPO cross-patient training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--patients", nargs="+", default=None)
    parser.add_argument(
        "--baseline-pers",
        default=None,
        help="Optional YAML file mapping patient_id -> baseline PER for Wilcoxon comparison.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tmin", type=float, default=-0.5)
    parser.add_argument("--tmax", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    bids_root = Path(paths["ps_bids_root"])
    patient_ids = args.patients or PS_PATIENTS
    seeds = config["evaluation"]["seeds"]
    baseline_pers = None
    if args.baseline_pers is not None:
        with open(args.baseline_pers) as f:
            baseline_pers = yaml.safe_load(f)

    all_datasets = {}
    for pid in patient_ids:
        try:
            ds = load_patient_data(
                pid,
                bids_root,
                task="PhonemeSequence",
                n_phons=3,
                tmin=args.tmin,
                tmax=args.tmax,
            )
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", pid, exc)
            continue
        all_datasets[pid] = ds
        logger.info("Loaded %s: %d trials, grid %s", pid, len(ds), ds.grid_shape)

    if len(all_datasets) < 3:
        raise SystemExit(f"Need at least 3 patients for LOPO, got {len(all_datasets)}")

    results = run_lopo(
        all_datasets,
        config,
        seeds=seeds,
        device=args.device,
        baseline_pers=baseline_pers,
    )
    print("\n" + "=" * 50)
    print(f"{'Patient':<10} {'Mean PER':<12}")
    print("-" * 50)
    for pid in sorted(results["patient_pers"]):
        print(f"{pid:<10} {results['patient_pers'][pid]:.3f}")
    print("-" * 50)
    print(
        f"{'Population':<10} "
        f"{results['population_per_mean']:.3f} ± {results['population_per_std']:.3f}"
    )
    if "wilcoxon_p" in results:
        print(
            f"Wilcoxon   stat={results['wilcoxon_stat']:.3f} "
            f"p={results['wilcoxon_p']:.4f}"
        )


if __name__ == "__main__":
    main()
