#!/usr/bin/env python3
"""Phase-1 diagnostic for speech-embedding regression on one patient."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from speech_decoding.data.bids_dataset import BIDSDataset, load_patient_data
from speech_decoding.training.regression_trainer import train_per_patient_regression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _feature_path(results_dir: Path, feature_type: str, patient_id: str) -> Path:
    return results_dir / "audio_features" / feature_type / f"sub-{patient_id}_{feature_type}.npz"


def _shift_dataset_time(ds: BIDSDataset, shift_frames: int) -> BIDSDataset:
    shifted = np.zeros_like(ds.grid_data)
    if shift_frames >= 0:
        shifted[..., shift_frames:] = ds.grid_data[..., : ds.grid_data.shape[-1] - shift_frames]
    else:
        shifted[..., :shift_frames] = ds.grid_data[..., -shift_frames:]
    return BIDSDataset(
        grid_data=shifted,
        ctc_labels=ds.ctc_labels,
        patient_id=ds.patient_id,
        grid_shape=ds.grid_shape,
    )


def _run(
    label: str,
    ds: BIDSDataset,
    embeddings: np.ndarray,
    speech_mask: np.ndarray,
    config: dict,
    device: str,
    segment_mask: np.ndarray | None,
) -> dict:
    logger.info("Running %s", label)
    result = train_per_patient_regression(
        ds,
        embeddings,
        speech_mask,
        config,
        seed=42,
        device=device,
        segment_mask=segment_mask,
    )
    logger.info(
        "%s: PER=%.3f bal_acc=%.3f r2_main=%s",
        label,
        result["per_mean"],
        result["bal_acc_mean_mean"],
        (
            f"{result['r2_speech_mean']:.3f}"
            if "r2_speech_mean" in result
            else f"{result['r2_segment_mean']:.3f}"
            if "r2_segment_mean" in result
            else "n/a"
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the regression diagnostic on one patient.")
    parser.add_argument("--config", default="configs/per_patient_regression.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--patient", default="S14")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-type", choices=["hubert", "mel"], default=None)
    parser.add_argument("--target-mode", choices=["frame", "segment"], default=None)
    parser.add_argument("--pca-scope", choices=["all_frames", "speech_only"], default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    if args.feature_type is not None:
        config.setdefault("data", {})["feature_type"] = args.feature_type
    if args.target_mode is not None:
        config["training"]["regression_target_mode"] = args.target_mode
    if args.pca_scope is not None:
        config["training"]["regression_pca_scope"] = args.pca_scope

    bids_root = Path(paths["ps_bids_root"])
    results_dir = Path(paths["results_dir"])
    feature_type = config.get("data", {}).get("feature_type", "hubert")
    feat = np.load(_feature_path(results_dir, feature_type, args.patient))
    embeddings = feat["embeddings"]
    speech_mask = feat["speech_mask"]
    segment_mask = feat["segment_mask"] if "segment_mask" in feat else None

    ds = load_patient_data(
        args.patient,
        bids_root,
        task=config["data"]["task"],
        n_phons=config["data"]["n_phons"],
        tmin=config["data"]["tmin"],
        tmax=config["data"]["tmax"],
    )
    logger.info(
        "%s loaded: trials=%d frames=%d feature_dim=%d speech_frac=%.3f target_mode=%s feature=%s pca_scope=%s",
        args.patient,
        len(ds),
        embeddings.shape[1],
        embeddings.shape[2],
        float(speech_mask.mean()),
        config["training"].get("regression_target_mode", "frame"),
        feature_type,
        config["training"].get("regression_pca_scope", "all_frames"),
    )

    cfg_joint = yaml.safe_load(yaml.safe_dump(config))
    cfg_joint["training"]["regression_lambda"] = 0.3
    _run("joint", ds, embeddings, speech_mask, cfg_joint, args.device, segment_mask)

    cfg_ce = yaml.safe_load(yaml.safe_dump(config))
    cfg_ce["training"]["regression_lambda"] = 0.0
    _run("ce_only", ds, embeddings, speech_mask, cfg_ce, args.device, segment_mask)

    cfg_shift = yaml.safe_load(yaml.safe_dump(config))
    cfg_shift["training"]["regression_lambda"] = 0.3
    shifted = _shift_dataset_time(ds, shift_frames=100)
    _run("shifted_500ms", shifted, embeddings, speech_mask, cfg_shift, args.device, segment_mask)


if __name__ == "__main__":
    main()
