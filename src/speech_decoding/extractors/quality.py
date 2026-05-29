"""Channel quality flagging (T1.7) — MNE LOF-based bad-channel detection.

Spec lock (project_v14_preproc_recipe_2026_05_12.md, 5/17 L.0 amendment):

    mne.preprocessing.find_bad_channels_lof(raw, threshold=1.5, n_neighbors=20)

run **per session, pre-shaftCAR** so the bad channel doesn't contaminate the
reference. Emits a per-session ``(C,) bool`` mask of bad channels. Rows
flagged True ("bad") are dropped from the (C, F, T) token tensor before it
reaches the v14 encoder.

This module provides:

  * :func:`lof_bad_channel_mask` — pure-function primitive. Takes a numpy
    voltage matrix ``(C, n_samples)``, runs LOF via MNE, returns
    ``(C,) bool`` where True = bad.
  * :class:`MneLofBadChannelMask` — Pydantic transform skeleton for the
    eventual NeuralFetch wiring. Stores the per-session mask in a cache
    so trial extraction can intersect it with the existing
    :class:`ElectrodeValidMask`.

The wiring point — pre-CAR, post-channel-pick, session-level — is documented
in the task description but the actual substrate plumbing belongs in the
Study / Transform layer once T3.4 (SWEC) and T3.5 (D-cohort) loaders land.
"""

from __future__ import annotations

import typing as tp

import numpy as np
from pydantic import BaseModel, ConfigDict


DEFAULT_LOF_THRESHOLD: float = 1.5
DEFAULT_LOF_N_NEIGHBORS: int = 20


def lof_bad_channel_mask(
    voltage: np.ndarray,
    *,
    sample_rate_hz: float,
    threshold: float = DEFAULT_LOF_THRESHOLD,
    n_neighbors: int = DEFAULT_LOF_N_NEIGHBORS,
    ch_names: tp.Optional[tp.Sequence[str]] = None,
    ch_type: str = "seeg",
) -> np.ndarray:
    """Return a ``(C,) bool`` mask flagging LOF-detected bad channels.

    ``voltage`` is ``(C, n_samples)``. Channels with True in the returned
    mask are considered bad and should be dropped before downstream
    preprocessing (notably the shaft-CAR step, which is sensitive to outliers).

    This is a thin wrapper around
    :func:`mne.preprocessing.find_bad_channels_lof` so the rest of the
    pipeline doesn't have to assemble a transient ``mne.io.Raw`` to access
    the algorithm.
    """
    import mne

    n_channels = int(voltage.shape[0])
    if ch_names is None:
        ch_names = [f"ch{i}" for i in range(n_channels)]
    if len(ch_names) != n_channels:
        raise ValueError(
            f"ch_names has {len(ch_names)} entries, voltage has {n_channels} channels"
        )
    info = mne.create_info(
        ch_names=list(ch_names),
        sfreq=float(sample_rate_hz),
        ch_types=ch_type,
        verbose=False,
    )
    raw = mne.io.RawArray(voltage.astype(np.float64), info, verbose=False)
    bads = mne.preprocessing.find_bad_channels_lof(
        raw,
        threshold=threshold,
        n_neighbors=n_neighbors,
        verbose=False,
    )
    mask = np.zeros(n_channels, dtype=bool)
    bad_idx = {raw.ch_names.index(n) for n in bads if n in raw.ch_names}
    for i in bad_idx:
        mask[i] = True
    return mask


class MneLofBadChannelMask(BaseModel):
    """Pydantic transform skeleton for the NeuralFetch wiring. Holds the LOF
    hyperparameters; the actual per-session computation goes through
    :func:`lof_bad_channel_mask`.

    The flag-then-drop pattern is what the v14 dataset layer should apply:

      1. Pre-CAR session voltage → :func:`lof_bad_channel_mask` → ``bad: (C,)``.
      2. Drop rows where ``bad`` is True before CAR + downstream.

    Step 2 belongs in the Study/Transform layer and is tracked under the
    T3.4 / T3.5 corpus loaders.
    """

    model_config = ConfigDict(extra="forbid")

    threshold: float = DEFAULT_LOF_THRESHOLD
    n_neighbors: int = DEFAULT_LOF_N_NEIGHBORS
    ch_type: str = "seeg"

    def __call__(
        self,
        voltage: np.ndarray,
        *,
        sample_rate_hz: float,
        ch_names: tp.Optional[tp.Sequence[str]] = None,
    ) -> np.ndarray:
        return lof_bad_channel_mask(
            voltage,
            sample_rate_hz=sample_rate_hz,
            threshold=self.threshold,
            n_neighbors=self.n_neighbors,
            ch_names=ch_names,
            ch_type=self.ch_type,
        )
