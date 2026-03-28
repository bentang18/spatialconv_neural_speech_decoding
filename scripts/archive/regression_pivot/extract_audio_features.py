#!/usr/bin/env python3
"""Extract framewise speech targets for regression experiments."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from speech_decoding.data.audio_features import (
    build_segment_masks,
    build_speech_mask,
    extract_audio_segment,
    extract_hubert_embeddings,
    extract_mel_spectrogram,
    load_patient_audio,
    load_phoneme_timing,
    resample_to_backbone_frames,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PS_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S36", "S39", "S57", "S58", "S62"]


def _output_path(results_dir: Path, feature_type: str, patient_id: str) -> Path:
    out_dir = results_dir / "audio_features" / feature_type
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"sub-{patient_id}_{feature_type}.npz"


def extract_patient_features(
    patient_id: str,
    bids_root: Path,
    feature_type: str = "hubert",
) -> dict[str, np.ndarray]:
    timing = load_phoneme_timing(patient_id, bids_root)
    audio, sr = load_patient_audio(patient_id, bids_root)

    embeddings = []
    masks = []
    segment_masks = []
    for info in tqdm(timing, desc=f"{patient_id} {feature_type}", leave=False):
        segment = extract_audio_segment(audio, sr, center_time=info.response_onset)
        if feature_type == "hubert":
            feat = extract_hubert_embeddings(segment, sr)
            feat = resample_to_backbone_frames(feat, n_frames=50)
        elif feature_type == "mel":
            feat = extract_mel_spectrogram(segment, sr, n_mels=40, n_frames=50)
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")
        embeddings.append(feat.astype(np.float32))
        masks.append(build_speech_mask(info.phoneme_intervals, n_frames=50))
        segment_masks.append(build_segment_masks(info.phoneme_intervals, n_frames=50))

    return {
        "embeddings": np.stack(embeddings).astype(np.float32),
        "speech_mask": np.stack(masks).astype(np.float32),
        "segment_mask": np.stack(segment_masks).astype(np.float32),
        "trials": np.asarray([info.trial for info in timing], dtype=np.int32),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract audio-derived regression targets.")
    parser.add_argument("--config", default="configs/per_patient_regression.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--patients", nargs="+", default=None)
    parser.add_argument("--feature-type", choices=["hubert", "mel"], default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    bids_root = Path(paths["ps_bids_root"])
    results_dir = Path(paths["results_dir"])
    patients = args.patients or PS_PATIENTS
    feature_type = args.feature_type or config.get("data", {}).get("feature_type", "hubert")

    for patient_id in patients:
        out_path = _output_path(results_dir, feature_type, patient_id)
        if out_path.exists() and not args.force:
            logger.info("Skipping %s; features already exist at %s", patient_id, out_path)
            continue
        features = extract_patient_features(patient_id, bids_root, feature_type=feature_type)
        np.savez_compressed(out_path, **features)
        logger.info(
            "%s saved: embeddings=%s speech_frac=%.3f path=%s",
            patient_id,
            features["embeddings"].shape,
            float(features["speech_mask"].mean()),
            out_path,
        )


if __name__ == "__main__":
    main()
