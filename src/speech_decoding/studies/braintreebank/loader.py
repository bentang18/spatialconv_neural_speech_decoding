"""Voltage reader for BrainTreebank h5 trials, resampled to the pipeline rate.

`bt_load_raw()` returns voltage in the shape NeuralSet's `Ieeg._read()` wants:
`(data: np.ndarray (n_ch, n_samples) float32, ch_names: list[str], sfreq: float)`.

Most BT subjects are distributed at 2048 Hz; a few are not (S9 = 1024 Hz — see
`BT_SUBJECT_NATIVE_RATE_HZ`). The whole downstream pipeline (est_idx→seconds,
STFT Δf, FE-RAW-1 bin↔Hz) assumes a single `BT_DEFAULT_SAMPLE_RATE_HZ`, so any
subject recorded at a different native rate is polyphase-resampled UP to it here,
at the loader boundary — the one place voltage enters. Returned `sfreq` is
therefore always `BT_DEFAULT_SAMPLE_RATE_HZ`. **No re-reference applied** — CAR /
Laplacian and HG envelope are Stage-1 ablation cells, not loader behavior.

The `BrainTreebankSubject` import stays outside this module so tests can use a
small protocol stub. Real h5 reads only fire on DCC or another machine with
BrainTreebank data and `ROOT_DIR_BRAINTREEBANK` configured.
"""

from __future__ import annotations

import typing as tp
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from speech_decoding.studies.braintreebank.anatomy import (
    clean_bt_electrode_label,
    extra_bad_electrodes,
)
from speech_decoding.studies.braintreebank.manifest import (
    BT_DEFAULT_SAMPLE_RATE_HZ,
    bt_subject_native_rate_hz,
)


class _BrainTreebankSubjectLike(tp.Protocol):
    """Subset of `neuroprobe.braintreebank_subject.BrainTreebankSubject` we use."""

    def get_all_electrode_data(self, trial_id: int) -> tp.Any: ...

    @property
    def electrode_labels(self) -> list[str]: ...


def bt_load_raw(
    bt: _BrainTreebankSubjectLike,
    trial_id: int,
    *,
    subject_id: int,
) -> tuple[np.ndarray, list[str], float]:
    """Pull `(data, ch_names, sfreq)` from a `BrainTreebankSubject`-shaped object,
    resampled to `BT_DEFAULT_SAMPLE_RATE_HZ` when the subject's native rate differs.

    `subject_id` selects the native rate from `BT_SUBJECT_NATIVE_RATE_HZ`. It is a
    required keyword (no silent default) so a non-2048 subject can never slip
    through un-resampled — the exact failure that left S9 2× time-stretched.
    """

    raw_data = bt.get_all_electrode_data(trial_id)
    if hasattr(raw_data, "detach"):
        raw_data = raw_data.detach().cpu().numpy()
    data = np.asarray(raw_data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected (n_ch, n_samples) array, got shape {data.shape}")
    # Clean BT `*`/`#` annotation marks at the loader boundary (H6 / WS-D1).
    # The real upstream `BrainTreebankSubject._get_all_electrode_names` already
    # strips these, so on the production path this is defense-in-depth — but
    # `_BrainTreebankSubjectLike` is only a Protocol and cannot guarantee the
    # caller pre-cleaned. Were a raw mark to reach `reference.parse_shaft`
    # (e.g. sub_2's 34 `RT1aIa*` contacts), the trailing `*` defeats the
    # numeric-suffix match, making each its own singleton shaft that shaft-CAR
    # self-subtracts to zero. Idempotent on already-cleaned labels.
    ch_names = [clean_bt_electrode_label(name) for name in bt.electrode_labels]
    if len(ch_names) != data.shape[0]:
        raise ValueError(
            f"electrode_labels len {len(ch_names)} != n_channels {data.shape[0]}"
        )
    # v14 flaky-contact static exclusion (DP4-safe). Drop the bad contacts HERE —
    # at the voltage boundary, BEFORE shaft-CAR in the view — so they can't
    # contaminate their shaft's CAR reference, and so the front-end voltage rows
    # stay in lockstep with support/valid_mask (both consult the same
    # extra_bad_electrodes() source; voltage_electrode_order folds in the identical
    # set). Done on axis-0 before resample; the two axes are independent.
    extra_bad = extra_bad_electrodes(subject_id)
    if extra_bad:
        keep = [i for i, name in enumerate(ch_names) if name not in extra_bad]
        if len(keep) != len(ch_names):
            data = data[keep]
            ch_names = [ch_names[i] for i in keep]
    native_rate = bt_subject_native_rate_hz(subject_id)
    if native_rate != BT_DEFAULT_SAMPLE_RATE_HZ:
        # Polyphase-resample native → pipeline rate (S9: 1024→2048, up/down = 2/1).
        # Wall-clock duration is preserved, so `est_idx` (native grid) → seconds
        # still divides by the NATIVE rate (word_events._neural_sample_rate); a
        # window at native sample i then lands at resampled sample i·(target/native).
        g = gcd(BT_DEFAULT_SAMPLE_RATE_HZ, native_rate)
        up, down = BT_DEFAULT_SAMPLE_RATE_HZ // g, native_rate // g
        data = resample_poly(data, up, down, axis=1).astype(np.float32, copy=False)
    return data, ch_names, float(BT_DEFAULT_SAMPLE_RATE_HZ)
