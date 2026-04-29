"""Raw 2048 Hz voltage reader for BrainTreebank h5 trials.

`bt_load_raw()` returns voltage in the shape NeuralSet's `Ieeg._read()` wants:
`(data: np.ndarray (n_ch, n_samples) float32, ch_names: list[str], sfreq: float)`.

Native sample rate is 2048 Hz. **No re-reference applied** — CAR / Laplacian
and HG envelope are Stage-1 ablation cells, not loader behavior. Matches
Neuroprobe `__getitem__` native output exactly so PopT-comparability and
multi-FM SSL hold.

The `BrainTreebankSubject` import is lazy so this module loads cleanly on the
laptop (no neuroprobe / no h5 data); the actual h5 read only fires on DCC.
"""

from __future__ import annotations

import typing as tp

import numpy as np


class _BrainTreebankSubjectLike(tp.Protocol):
    """Subset of `neuroprobe.braintreebank_subject.BrainTreebankSubject` we use."""

    def get_electrode_data(self) -> np.ndarray: ...

    @property
    def electrode_labels(self) -> list[str]: ...

    @property
    def sampling_rate(self) -> float: ...


def bt_load_raw(
    bt: _BrainTreebankSubjectLike,
) -> tuple[np.ndarray, list[str], float]:
    """Pull `(data, ch_names, sfreq)` from a `BrainTreebankSubject`-shaped object."""

    data = np.asarray(bt.get_electrode_data(), dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected (n_ch, n_samples) array, got shape {data.shape}")
    ch_names = list(bt.electrode_labels)
    if len(ch_names) != data.shape[0]:
        raise ValueError(
            f"electrode_labels len {len(ch_names)} != n_channels {data.shape[0]}"
        )
    sfreq = float(bt.sampling_rate)
    return data, ch_names, sfreq
