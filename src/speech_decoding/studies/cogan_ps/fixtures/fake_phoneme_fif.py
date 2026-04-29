"""Minimal phoneme-level `mne.EpochsArray` factory for P3 tests.

Event_id is the 9-phoneme PS mapping `{a:1, ae:2, b:3, g:4, i:5, k:6, p:7,
u:8, v:9}`. Given tokens, expands each to 3 phoneme epochs in
`[t0p0, t0p1, t0p2, t1p0, …]` order, matching the production phoneme-level
`.fif` layout. Epoch window `[-0.5, 0.995]` s at 200 Hz (300 samples) so the
dataset's `[-0.15, 0.495]` crop is exercised by the round-trip test.
"""

from __future__ import annotations

import mne
import numpy as np

_PS_EVENT_ID: dict[str, int] = {
    "a": 1, "ae": 2, "b": 3, "g": 4, "i": 5, "k": 6, "p": 7, "u": 8, "v": 9,
}


def _decompose_ps(token: str) -> list[str]:
    parts: list[str] = []
    i = 0
    tok = token.lower()
    while i < len(tok):
        if tok[i : i + 2] == "ae":
            parts.append("ae")
            i += 2
        else:
            parts.append(tok[i])
            i += 1
    if len(parts) != 3:
        raise ValueError(f"token {token!r} does not decompose to 3 phonemes")
    return parts


def build_fake_phoneme_epochs(
    *,
    tokens: tuple[str, ...] = ("bak", "gub"),
    n_electrodes: int = 16,
    seed: int = 0,
) -> mne.EpochsArray:
    """Build a `3·len(tokens)`-epoch phoneme-level fixture."""

    sfreq = 200.0
    tmin = -0.5
    n_times = 300  # [-0.5, 1.0) half-open at 200 Hz; crop to 130 later

    phonemes_flat: list[str] = []
    for t in tokens:
        phonemes_flat.extend(_decompose_ps(t))

    ch_names = [f"{i}" for i in range(1, n_electrodes + 1)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    rng = np.random.default_rng(seed)
    data = rng.standard_normal(
        (len(phonemes_flat), n_electrodes, n_times)
    ).astype(np.float32)

    events = np.zeros((len(phonemes_flat), 3), dtype=np.int64)
    for i, ph in enumerate(phonemes_flat):
        events[i, 0] = 1000 * (i + 1)
        events[i, 2] = _PS_EVENT_ID[ph]

    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id=_PS_EVENT_ID,
        tmin=tmin,
        on_missing="ignore",
        verbose="ERROR",
    )
