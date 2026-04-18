"""v14-core per-patient dataset + sample dataclass (Task B1).

Implements the loader contract amended 2026-04-16 late (B-1): no `coords`,
no `token_mask`, no `token_support`. A sample is per-electrode:

    signal[N_e, 300]               float32
    patient_id: str
    label[3]                       long, alphabetical ARPABET index (#16)
    electrode_grid_layout[N_e, 2]  int, (row, col) on patient device grid
    electrode_grid_shape           tuple (H_p, W_p)
    electrode_active_mask[N_e]     bool, non-artifact AND not pad
    support[N_e, 15]               float32, raw BNA prob over Tier-1 (#5)

`V14TrialDataset` binds one `.fif` (per #34, authoritative for tokens and
response onsets) to the #12 channel-map bridge and the A1 per-electrode
support cache. Token -> 3-slot label index decomposition uses the frozen
alphabetical ARPABET ordering in `phoneme_map.ARPA_PHONEMES` (#16 / #17).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import torch

from speech_decoding.data.phoneme_map import ARPA_PHONEMES, PS2ARPA, normalize_label
from speech_decoding.v14.channel_map import (
    GRID_SHAPES,
    expected_grid_shape_for_patient,
    resolve_physical_names_for_patient,
)
from speech_decoding.v14.support_cache import lookup_support_for_kept_channels


T_SAMPLES: int = 300  # #29: tmin=-0.5 s, tmax=1.0 s half-open, 200 Hz (upstream fif is [-1.5, 1.0) exclusive-upper)
N_PS_TOKENS: int = 52  # #18: event_id must enumerate all 52 PS tokens
N_TIER1_PARCELS: int = 15  # #4 / token_spec.DEFAULT_BASE_PARCELS


@dataclass(frozen=True)
class V14Sample:
    """One trial's worth of inputs to the v14-core model."""

    signal: torch.Tensor               # (N_e, T=300) float32
    patient_id: str
    label: torch.Tensor                # (3,) long
    electrode_grid_layout: torch.Tensor  # (N_e, 2) long
    electrode_grid_shape: tuple[int, int]
    electrode_active_mask: torch.Tensor  # (N_e,) bool
    support: torch.Tensor              # (N_e, 15) float32


def decompose_token_to_slot_indices(token: str) -> list[int]:
    """Greedy left-to-right PS-notation decomposition into 3 ARPABET indices.

    PS has one 2-character symbol: `ae` (vowel /æ/). Every other PS symbol is
    a single character. The token must decompose to exactly 3 phonemes.
    Raises `ValueError` otherwise. The returned indices follow the frozen
    alphabetical ARPABET ordering in `ARPA_PHONEMES` (#16).
    """

    lowered = token.lower()
    parts: list[str] = []
    i = 0
    while i < len(lowered):
        if lowered[i : i + 2] == "ae":
            parts.append("ae")
            i += 2
        else:
            parts.append(lowered[i])
            i += 1
    if len(parts) != 3:
        raise ValueError(
            f"token {token!r} decomposes to {len(parts)} phonemes, expected 3"
        )
    for p in parts:
        if p not in PS2ARPA:
            raise ValueError(f"token {token!r}: unknown PS symbol {p!r}")
    return [ARPA_PHONEMES.index(normalize_label(p)) for p in parts]


def build_sample_from_epoch(
    *,
    signal: np.ndarray,
    token: str,
    patient_id: str,
    electrode_grid_layout: np.ndarray,
    electrode_grid_shape: tuple[int, int],
    electrode_active_mask: np.ndarray,
    support: np.ndarray,
) -> V14Sample:
    """Assemble a `V14Sample` from a single-trial slice + metadata.

    `signal` must be `(N_e, 300)` float32 after the upstream crop to
    `[-0.5, 1.0) s` at 200 Hz (#29, half-open to match Zac's upstream epoching). `electrode_grid_layout`, `support`, and
    `electrode_active_mask` must all share the leading `N_e` dimension and
    match the order of the kept `.fif` channels.
    """

    if signal.ndim != 2 or signal.shape[1] != T_SAMPLES:
        raise ValueError(f"signal shape {signal.shape} != (N_e, {T_SAMPLES})")
    n_e = signal.shape[0]
    if electrode_grid_layout.shape != (n_e, 2):
        raise ValueError(
            f"electrode_grid_layout shape {electrode_grid_layout.shape} != ({n_e}, 2)"
        )
    if electrode_active_mask.shape != (n_e,):
        raise ValueError(
            f"electrode_active_mask shape {electrode_active_mask.shape} != ({n_e},)"
        )
    if support.shape != (n_e, N_TIER1_PARCELS):
        raise ValueError(
            f"support shape {support.shape} != ({n_e}, {N_TIER1_PARCELS})"
        )

    H_p, W_p = electrode_grid_shape
    rows = electrode_grid_layout[:, 0]
    cols = electrode_grid_layout[:, 1]
    if np.any(rows < 0) or np.any(rows >= H_p) or np.any(cols < 0) or np.any(cols >= W_p):
        raise ValueError(
            f"electrode_grid_layout has cells outside grid ({H_p}, {W_p})"
        )

    label = decompose_token_to_slot_indices(token)
    return V14Sample(
        signal=torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32)),
        patient_id=patient_id,
        label=torch.tensor(label, dtype=torch.long),
        electrode_grid_layout=torch.from_numpy(
            np.ascontiguousarray(electrode_grid_layout, dtype=np.int64)
        ),
        electrode_grid_shape=electrode_grid_shape,
        electrode_active_mask=torch.from_numpy(
            np.ascontiguousarray(electrode_active_mask, dtype=bool)
        ),
        support=torch.from_numpy(np.ascontiguousarray(support, dtype=np.float32)),
    )


def _crop_to_phase1_window(epochs: mne.BaseEpochs) -> mne.BaseEpochs:
    """Crop an `.fif` epochs object to `[-0.5, 1.0) s` per #29 and assert `T == 300`.

    Upstream fif uses half-open time intervals (tmax is exclusive); the
    crop below asks for `tmax=0.995` inclusive, which yields the same 300
    samples on the discrete sample grid at 200 Hz.
    """

    cropped = epochs.copy().crop(tmin=-0.5, tmax=0.995, include_tmax=True)
    if cropped.times.size != T_SAMPLES:
        raise ValueError(
            f"after crop expected {T_SAMPLES} samples, got {cropped.times.size}"
        )
    return cropped


class V14TrialDataset(torch.utils.data.Dataset[V14Sample]):
    """Per-patient `.fif` trial dataset for v14-core.

    Loads the `.fif`, runs the #12 channel-map bridge on kept channels
    (filtering `info["bads"]` per #11), and reads the per-electrode Tier-1
    support row from the A1 cache. Tokens + response onsets come from the
    `.fif` `event_id` per the #34 closure (the `.fif` is authoritative).

    Per the soft #3 rule, electrodes with zero Tier-1 mass are still emitted;
    the downstream soft parcel embedding carries the suppression naturally.
    """

    def __init__(
        self,
        patient_id: str,
        fif_path: Path,
        *,
        support_cache_path: Path,
        channel_maps_dir: Path,
        box_root: Path,
    ) -> None:
        self.patient_id = patient_id
        self.fif_path = Path(fif_path)
        self.support_cache_path = Path(support_cache_path)

        epochs = mne.read_epochs(str(self.fif_path), preload=True, verbose="ERROR")
        if len(epochs.event_id) != N_PS_TOKENS:
            raise ValueError(
                f"{self.fif_path}: event_id has {len(epochs.event_id)} keys, "
                f"expected {N_PS_TOKENS} (#18)"
            )
        epochs = _crop_to_phase1_window(epochs)

        all_names = tuple(epochs.ch_names)
        bads = set(epochs.info["bads"])
        kept_names = tuple(n for n in all_names if n not in bads)
        if len(kept_names) == 0:
            raise ValueError(f"{self.fif_path}: all channels in info['bads']")

        resolutions = resolve_physical_names_for_patient(
            patient_id,
            kept_names,
            channel_maps_dir=Path(channel_maps_dir),
            box_root=Path(box_root),
        )
        grid_shape = expected_grid_shape_for_patient(patient_id)
        declared = GRID_SHAPES.get(patient_id)
        if declared != grid_shape:
            raise ValueError(
                f"{patient_id}: registered grid {declared} != expected {grid_shape}"
            )

        layout = np.array([[r.row, r.col] for r in resolutions], dtype=np.int64)
        phys_names = tuple(r.phys_name for r in resolutions)
        support = lookup_support_for_kept_channels(
            self.support_cache_path, phys_names
        )
        active_mask = np.ones(len(kept_names), dtype=bool)

        id_to_token = {code: tok for tok, code in epochs.event_id.items()}
        trial_codes = epochs.events[:, 2]
        trial_tokens = [id_to_token[int(c)] for c in trial_codes]

        self._signals = epochs.get_data(copy=False).astype(np.float32)
        self._tokens: list[str] = trial_tokens
        self._kept_name_index = {name: i for i, name in enumerate(kept_names)}
        self._kept_names = kept_names
        self._phys_names = phys_names
        self._grid_shape = grid_shape
        self._electrode_grid_layout = layout
        self._electrode_active_mask = active_mask
        self._support = support

    @property
    def tokens(self) -> list[str]:
        """Per-trial PS-notation tokens, in the same order as `__getitem__`."""

        return list(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, index: int) -> V14Sample:
        return build_sample_from_epoch(
            signal=self._signals[index],
            token=self._tokens[index],
            patient_id=self.patient_id,
            electrode_grid_layout=self._electrode_grid_layout,
            electrode_grid_shape=self._grid_shape,
            electrode_active_mask=self._electrode_active_mask,
            support=self._support,
        )


def collate_v14_batch(samples: list[V14Sample]) -> dict[str, object]:
    """Stack per-trial `V14Sample`s into a grouped-by-patient batch (#31 + B-1).

    Per `#31`, a batch is a single patient. `patient_id`, `N_e`, and
    `electrode_grid_shape` are shared across the batch and asserted. The
    collator also constructs teacher-forcing `prev_tokens` with BOS sentinel
    `-1`: `prev_tokens[b] = [-1, label[b, 0], label[b, 1]]`.

    Output dict keys (consumed by the model):
        signal, electrode_grid_layout, electrode_grid_shape,
        electrode_active_mask, support, labels, prev_tokens, patient_id.
    """

    if len(samples) == 0:
        raise ValueError("collate_v14_batch received empty batch")
    first = samples[0]
    patient_id = first.patient_id
    n_e = first.signal.shape[0]
    grid_shape = first.electrode_grid_shape
    for i, s in enumerate(samples[1:], start=1):
        if s.patient_id != patient_id:
            raise ValueError(
                f"batch[{i}] patient_id {s.patient_id!r} != {patient_id!r} (#31)"
            )
        if s.signal.shape[0] != n_e:
            raise ValueError(
                f"batch[{i}] N_e {s.signal.shape[0]} != {n_e}"
            )
        if s.electrode_grid_shape != grid_shape:
            raise ValueError(
                f"batch[{i}] grid_shape {s.electrode_grid_shape} != {grid_shape}"
            )

    signal = torch.stack([s.signal for s in samples], dim=0)
    electrode_grid_layout = torch.stack(
        [s.electrode_grid_layout for s in samples], dim=0
    )
    electrode_active_mask = torch.stack(
        [s.electrode_active_mask for s in samples], dim=0
    )
    support = torch.stack([s.support for s in samples], dim=0)
    labels = torch.stack([s.label for s in samples], dim=0)
    bos = torch.full((labels.shape[0], 1), -1, dtype=labels.dtype)
    prev_tokens = torch.cat([bos, labels[:, :-1]], dim=1)

    return {
        "signal": signal,
        "electrode_grid_layout": electrode_grid_layout,
        "electrode_grid_shape": grid_shape,
        "electrode_active_mask": electrode_active_mask,
        "support": support,
        "labels": labels,
        "prev_tokens": prev_tokens,
        "patient_id": patient_id,
    }
