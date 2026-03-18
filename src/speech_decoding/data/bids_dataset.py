"""BIDS dataset loading for uECOG HGA data.

Loads preprocessed productionZscore HGA from .fif files, extracts
position-1 epochs for CTC training, reshapes to spatial grids, and
reconstructs 3-phoneme (PS) or 5-phoneme (Lexical) CTC labels.
"""
from __future__ import annotations

import logging
from pathlib import Path

import mne
import numpy as np
from torch.utils.data import Dataset

from speech_decoding.data.grid import GridInfo, channels_to_grid, load_grid_mapping
from speech_decoding.data.phoneme_map import (
    ARPA_PHONEMES,
    encode_ctc_label,
    normalize_label,
)

logger = logging.getLogger(__name__)


class BIDSDataset(Dataset):
    """Dataset of grid-shaped HGA trials with CTC labels.

    Each item is (grid_data, ctc_label, patient_id) where:
    - grid_data: (H, W, T) float32 array
    - ctc_label: list[int] of phoneme indices (1-9)
    - patient_id: str
    """

    def __init__(
        self,
        grid_data: np.ndarray,
        ctc_labels: list[list[int]],
        patient_id: str,
        grid_shape: tuple[int, int],
    ):
        self.grid_data = grid_data  # (n_trials, H, W, T)
        self.ctc_labels = ctc_labels
        self.patient_id = patient_id
        self.grid_shape = grid_shape

    def __len__(self) -> int:
        return len(self.ctc_labels)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, list[int], str]:
        return self.grid_data[idx], self.ctc_labels[idx], self.patient_id


def _find_fif_path(
    bids_root: Path, subject: str, task: str, desc: str = "productionZscore",
) -> Path:
    """Locate the productionZscore HGA .fif file for a patient."""
    deriv = bids_root / "derivatives" / "epoch(phonemeLevel)(CAR)"
    pattern = f"sub-{subject}_task-{task}_desc-{desc}_highgamma.fif"
    fif_path = deriv / f"sub-{subject}" / "epoch(band)(power)" / pattern
    if not fif_path.exists():
        raise FileNotFoundError(f"HGA file not found: {fif_path}")
    return fif_path


def _find_electrodes_tsv(bids_root: Path, subject: str) -> Path:
    """Locate the electrode coordinate TSV for a patient."""
    tsv = (
        bids_root / f"sub-{subject}" / "ieeg"
        / f"sub-{subject}_acq-01_space-ACPC_electrodes.tsv"
    )
    if not tsv.exists():
        raise FileNotFoundError(f"Electrode TSV not found: {tsv}")
    return tsv


def load_patient_data(
    subject: str,
    bids_root: str | Path,
    task: str = "PhonemeSequence",
    n_phons: int = 3,
    desc: str = "productionZscore",
    tmin: float | None = None,
    tmax: float | None = None,
    filter_ps_only: bool = False,
) -> BIDSDataset:
    """Load a single patient's HGA data as a grid-shaped CTC dataset.

    Args:
        subject: Patient ID (e.g., "S14").
        bids_root: Path to BIDS root directory.
        task: BIDS task name ("PhonemeSequence" or "lexical").
        n_phons: Phonemes per trial (3 for PS, 5 for Lexical).
        desc: Description field in BIDS filename.
        tmin: Start of time crop (seconds, relative to epoch onset). None = use full.
        tmax: End of time crop. None = use full.
        filter_ps_only: If True, keep only trials where ALL phonemes are in
            the 9-phoneme PS set (for cross-task with Lexical).

    Returns:
        BIDSDataset with grid-shaped data and CTC labels.
    """
    bids_root = Path(bids_root)

    # Load .fif epochs
    fif_path = _find_fif_path(bids_root, subject, task, desc)
    epochs = mne.read_epochs(str(fif_path), preload=True, verbose=False)

    # Time crop
    if tmin is not None or tmax is not None:
        crop_tmin = tmin if tmin is not None else epochs.tmin
        crop_tmax = tmax if tmax is not None else epochs.tmax
        epochs = epochs.crop(tmin=crop_tmin, tmax=crop_tmax)

    all_data = epochs.get_data()  # (n_total_epochs, n_ch, n_times)
    all_event_ids = epochs.events[:, 2]
    inv_event_id = {v: k for k, v in epochs.event_id.items()}
    ch_names = epochs.ch_names

    # Verify epoch count is divisible by n_phons
    n_total = len(all_data)
    if n_total % n_phons != 0:
        raise ValueError(
            f"{subject}: {n_total} epochs not divisible by {n_phons}"
        )
    n_trials = n_total // n_phons

    # Extract position-1 epochs (every n_phons-th, starting at 0)
    trial_data = all_data[0::n_phons]  # (n_trials, n_ch, T)

    # Reconstruct CTC labels from consecutive epoch triplets/quintuplets
    ps_set = set(ARPA_PHONEMES)
    ctc_labels: list[list[int]] = []
    keep_mask: list[bool] = []

    for i in range(n_trials):
        raw_labels = [
            inv_event_id[all_event_ids[i * n_phons + j]]
            for j in range(n_phons)
        ]
        # Normalize to canonical ARPA
        normed = [normalize_label(lbl) for lbl in raw_labels]

        if filter_ps_only and not all(p in ps_set for p in normed):
            keep_mask.append(False)
            ctc_labels.append([])  # placeholder
            continue

        keep_mask.append(True)
        ctc_labels.append(encode_ctc_label(normed))

    # Apply filter mask
    keep_idx = [i for i, k in enumerate(keep_mask) if k]
    trial_data = trial_data[keep_idx]
    ctc_labels = [ctc_labels[i] for i in keep_idx]

    # Load grid mapping and reshape to (n_trials, H, W, T)
    electrodes_tsv = _find_electrodes_tsv(bids_root, subject)
    grid_info = load_grid_mapping(electrodes_tsv)
    grid_data = channels_to_grid(trial_data, ch_names, grid_info)

    logger.info(
        "%s: %d trials, grid %s, T=%d",
        subject, len(ctc_labels), grid_info.grid_shape, grid_data.shape[-1],
    )

    return BIDSDataset(
        grid_data=grid_data.astype(np.float32),
        ctc_labels=ctc_labels,
        patient_id=subject,
        grid_shape=grid_info.grid_shape,
    )
