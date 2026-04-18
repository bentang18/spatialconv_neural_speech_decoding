"""Minimal `mne.EpochsArray` factory for B1/B2 tests (plan open-note N8).

Produces a 2-trial fixture with tokens `bak` and `gup`, `N_e = 16`,
`sfreq = 200 Hz`, `tmin = -0.5 s`, `tmax = 1.0 s`. The `event_id` dict
contains all 52 PS tokens (loaded from `data/ps_tokens.csv`) so that the
B1 `V14TrialDataset` 52-key assertion holds, even though only 2 trials
are present.
"""

from __future__ import annotations

import csv
from pathlib import Path

import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PS_TOKENS_CSV = PROJECT_ROOT / "data" / "ps_tokens.csv"


def _load_ps_token_event_id() -> dict[str, int]:
    with PS_TOKENS_CSV.open() as f:
        rows = list(csv.DictReader(f))
    return {row["ps_notation"]: int(row["token_id"]) + 1 for row in rows}


def build_fake_epochs(
    *,
    tokens: tuple[str, ...] = ("bak", "gub"),
    n_electrodes: int = 16,
    seed: int = 0,
) -> mne.EpochsArray:
    """Build a 2-trial `.fif`-compatible `EpochsArray`.

    Parameters match plan N8. Returns an `EpochsArray` whose integer event
    codes track the canonical 1-indexed PS token id.
    """

    sfreq = 200.0
    tmin = -0.5
    n_times = 300  # tmin=-0.5, tmax=1.0 half-open at 5 ms steps (matches upstream)
    event_id = _load_ps_token_event_id()
    for tok in tokens:
        if tok not in event_id:
            raise ValueError(f"token {tok!r} not in ps_tokens.csv manifest")

    ch_names = [f"{i}" for i in range(1, n_electrodes + 1)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    rng = np.random.default_rng(seed)
    data = rng.standard_normal((len(tokens), n_electrodes, n_times)).astype(np.float32)

    events = np.zeros((len(tokens), 3), dtype=np.int64)
    for i, tok in enumerate(tokens):
        events[i, 0] = 1000 * (i + 1)
        events[i, 2] = event_id[tok]

    return mne.EpochsArray(
        data, info, events=events, event_id=event_id, tmin=tmin,
        on_missing="ignore", verbose="ERROR",
    )
