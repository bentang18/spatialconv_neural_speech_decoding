#!/usr/bin/env python3
"""Validate neural epoch timing against phoneme timing metadata."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from speech_decoding.data.audio_features import (
    extract_audio_segment,
    load_epoch_response_times,
    load_patient_audio,
    load_phoneme_timing,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate response-time alignment.")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--patient", default="S14")
    parser.add_argument("--n-trials", type=int, default=6)
    args = parser.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    bids_root = Path(paths["ps_bids_root"])
    results_dir = Path(paths["results_dir"]) / "alignment_checks"
    results_dir.mkdir(parents=True, exist_ok=True)

    timing = load_phoneme_timing(args.patient, bids_root)
    epoch_times = load_epoch_response_times(args.patient, bids_root)
    response_times = np.asarray([info.response_onset for info in timing], dtype=np.float64)
    diffs = epoch_times[: len(response_times)] - response_times

    logger.info(
        "%s response-onset alignment: n=%d median=%.6fs max_abs=%.6fs",
        args.patient, len(diffs), float(np.median(diffs)), float(np.max(np.abs(diffs))),
    )

    audio, sr = load_patient_audio(args.patient, bids_root)
    trial_indices = np.linspace(0, len(timing) - 1, num=min(args.n_trials, len(timing)), dtype=int)
    fig, axes = plt.subplots(len(trial_indices), 1, figsize=(12, 2.5 * len(trial_indices)), sharex=True)
    if len(trial_indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, trial_indices):
        info = timing[idx]
        segment = extract_audio_segment(audio, sr, center_time=info.response_onset)
        times = np.arange(len(segment)) / sr - 1.0
        ax.plot(times, segment, linewidth=0.6, color="black")
        for start_s, end_s in info.phoneme_intervals:
            ax.axvspan(start_s, end_s, color="tab:orange", alpha=0.25)
        ax.axvline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
        ax.set_title(f"trial={info.trial} syllable={info.syllable} phonemes={'-'.join(info.phonemes)}")
        ax.set_ylabel("amp")
    axes[-1].set_xlabel("time from response onset (s)")

    out_path = results_dir / f"sub-{args.patient}_alignment.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Saved alignment figure to %s", out_path)


if __name__ == "__main__":
    main()
