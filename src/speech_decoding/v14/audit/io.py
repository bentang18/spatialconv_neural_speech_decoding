"""Data loaders for the audit.

All loaders are read-only and return plain pandas/numpy/MNE objects — the
predicate modules consume these and produce `Check` dicts.
"""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import wavfile

ECOG_RAW_SFREQ = 2000.0  # per BIDS sidecar; used for sample ↔ seconds conversion


def load_trial_epochs(fif_path: Path):
    return mne.read_epochs(str(fif_path), preload=False, verbose="ERROR")


def load_phoneme_epochs(fif_path: Path):
    if not fif_path.exists():
        return None
    return mne.read_epochs(str(fif_path), preload=False, verbose="ERROR")


def load_ps_tokens(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert isinstance(df, pd.DataFrame)
    return df


def load_events(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    assert isinstance(df, pd.DataFrame)
    return df


def response_rows(events: pd.DataFrame) -> pd.DataFrame:
    out = events.loc[events["trial_type"] == "response"].reset_index(drop=True)
    assert isinstance(out, pd.DataFrame)
    return out


def stimulus_rows(events: pd.DataFrame) -> pd.DataFrame:
    out = events.loc[events["trial_type"] == "stimulus"].reset_index(drop=True)
    assert isinstance(out, pd.DataFrame)
    return out


def load_raw_audio(wav_path: Path) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(str(wav_path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    # Convert to float32 in [-1, 1] regardless of int/float input
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float32) / max(abs(info.min), abs(info.max))
    else:
        x = x.astype(np.float32)
    return int(sr), x
