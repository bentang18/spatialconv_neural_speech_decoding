#!/usr/bin/env python3
"""Per-patient training with MFA-guided per-position CE loss.

Uses MFA phoneme boundaries to give each position classifier its own
temporal window instead of global mean-pooling.

Usage:
  python scripts/train_mfa_guided.py --patient S14
  python scripts/train_mfa_guided.py --patient S14 --aux-lambda 0.3
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from speech_decoding.data.audio_features import (
    build_segment_masks,
    load_phoneme_timing,
)
from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.training.mfa_guided_trainer import train_per_patient_mfa_guided

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_patient_segment_masks(
    subject: str,
    bids_root: str | Path,
    n_trials_expected: int,
    tmin: float = -0.5,
    tmax: float = 1.0,
    temporal_stride: int = 10,
    sfreq: int = 200,
) -> np.ndarray:
    """Build per-phoneme segment masks aligned to backbone frame grid.

    Returns (n_trials, 3, n_backbone_frames) array.
    """
    total_samples = int((tmax - tmin) * sfreq)
    n_backbone_frames = total_samples // temporal_stride
    frame_dur = (tmax - tmin) / n_backbone_frames

    timing = load_phoneme_timing(subject, bids_root)

    if len(timing) != n_trials_expected:
        logger.warning(
            "%s: phoneme timing has %d trials but dataset has %d. "
            "Attempting to match by count.",
            subject, len(timing), n_trials_expected,
        )

    all_masks = []
    for trial_info in timing[:n_trials_expected]:
        masks = build_segment_masks(
            trial_info.phoneme_intervals,
            n_frames=n_backbone_frames,
            frame_dur=frame_dur,
            window_start=tmin,
        )
        all_masks.append(masks)

    return np.stack(all_masks)  # (n_trials, 3, n_backbone_frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/per_patient_phonological_aux.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--patient", default="S14")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tmin", type=float, default=-0.5)
    parser.add_argument("--tmax", type=float, default=1.0)
    parser.add_argument("--aux-lambda", type=float, default=None,
                        help="Phonological aux lambda (0=CE only, default=config value)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    if args.aux_lambda is not None:
        config["training"]["phonological_aux_lambda"] = args.aux_lambda

    bids_root = Path(paths["ps_bids_root"])

    # Load neural data
    ds = load_patient_data(
        args.patient, bids_root, task="PhonemeSequence", n_phons=3,
        tmin=args.tmin, tmax=args.tmax,
    )
    logger.info("%s: %d trials, grid %s", args.patient, len(ds), ds.grid_shape)

    # Build segment masks from MFA
    seg_masks = build_patient_segment_masks(
        args.patient, bids_root, n_trials_expected=len(ds),
        tmin=args.tmin, tmax=args.tmax,
        temporal_stride=config["model"]["temporal_stride"],
    )
    logger.info(
        "Segment masks: %s, speech fraction per position: %s",
        seg_masks.shape,
        [f"{seg_masks[:, p, :].mean():.2f}" for p in range(seg_masks.shape[1])],
    )

    # Run matched CE-only control first, then MFA-guided
    for label, use_mfa in [("global-pool CE (control)", False), ("MFA-guided CE", True)]:
        logger.info("=== %s ===", label)

        if use_mfa:
            result = train_per_patient_mfa_guided(
                ds, seg_masks, config, seed=42, device=args.device,
            )
        else:
            # Run global-pool CE with same config for comparison
            from speech_decoding.training.phonological_aux_trainer import (
                train_per_patient_phonological_aux,
            )
            ctrl_config = yaml.safe_load(yaml.safe_dump(config))
            ctrl_config["training"]["phonological_aux_lambda"] = 0.0
            result = train_per_patient_phonological_aux(
                ds, ctrl_config, seed=42, device=args.device,
            )

        logger.info(
            "%s: PER=%.3f bal_acc=%.3f",
            label, result["per_mean"], result["bal_acc_mean_mean"],
        )


if __name__ == "__main__":
    main()
