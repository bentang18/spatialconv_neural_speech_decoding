"""Raw 2048 Hz voltage reader for BrainTreebank h5 trials.

`bt_load_raw()` returns voltage in the shape NeuralSet's `Ieeg._read()` wants:
`(data: np.ndarray (n_ch, n_samples) float32, ch_names: list[str], sfreq: float)`.

Native sample rate is 2048 Hz. **No re-reference applied** — CAR / Laplacian
and HG envelope are Stage-1 ablation cells, not loader behavior. Matches
Neuroprobe `__getitem__` native output exactly so PopT-comparability and
multi-FM SSL hold.

The `BrainTreebankSubject` import stays outside this module so tests can use a
small protocol stub. Real h5 reads only fire on DCC or another machine with
BrainTreebank data and `ROOT_DIR_BRAINTREEBANK` configured.
"""

from __future__ import annotations

import typing as tp

import numpy as np


class _BrainTreebankSubjectLike(tp.Protocol):
    """Subset of `neuroprobe.braintreebank_subject.BrainTreebankSubject` we use."""

    def get_all_electrode_data(self, trial_id: int) -> tp.Any: ...

    @property
    def electrode_labels(self) -> list[str]: ...


def bt_load_raw(
    bt: _BrainTreebankSubjectLike,
    trial_id: int,
) -> tuple[np.ndarray, list[str], float]:
    """Pull `(data, ch_names, sfreq)` from a `BrainTreebankSubject`-shaped object."""

    raw_data = bt.get_all_electrode_data(trial_id)
    if hasattr(raw_data, "detach"):
        raw_data = raw_data.detach().cpu().numpy()
    data = np.asarray(raw_data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected (n_ch, n_samples) array, got shape {data.shape}")
    ch_names = list(bt.electrode_labels)
    if len(ch_names) != data.shape[0]:
        raise ValueError(
            f"electrode_labels len {len(ch_names)} != n_channels {data.shape[0]}"
        )
    sfreq = _sampling_rate()
    return data, ch_names, sfreq


def _sampling_rate() -> float:
    try:
        from neuroprobe.config import SAMPLING_RATE
    except (ImportError, KeyError):  # local unit tests do not set BT data root
        return 2048.0
    return float(SAMPLING_RATE)
