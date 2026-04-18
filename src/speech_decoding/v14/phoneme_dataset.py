"""Per-phoneme dataset for the v14-core baseline-aligned rework (P3).

Supersedes `V14TrialDataset` for the per-phoneme training path. Loads the
phoneme-level `.fif` (event_id = 9 PS phonemes, 3·n_trials epochs ordered
`[t0p0, t0p1, t0p2, t1p0, …]`) and emits one sample per phoneme with
`prev_phoneme` teacher-forcing context and `trial_id` for grouped-by-token CV.

Window `[-0.15, 0.5)` at 200 Hz → 130 raw samples per phoneme, matching the
baseline MFA window Ben's 0.734 result used. Crop is inclusive-tmax=0.495
(half-open upstream convention; yields 130 samples on the discrete grid).

Label index is the alphabetical ARPA position (`#16` / `#17`), computed by
translating PS event codes → ARPA via `normalize_label` → `ARPA_PHONEMES.index`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import torch

from speech_decoding.data.phoneme_map import ARPA_PHONEMES, normalize_label
from speech_decoding.v14.channel_map import (
    GRID_SHAPES,
    expected_grid_shape_for_patient,
    resolve_physical_names_for_patient,
)
from speech_decoding.v14.support_cache import lookup_support_for_kept_channels


T_RAW_SAMPLES: int = 130  # tmin=-0.15, tmax=0.495 inclusive at 200 Hz
N_PHONEMES: int = 9        # ARPA phoneme vocabulary size
BOS_TOKEN: int = -1        # prev_phoneme sentinel at phoneme_pos == 0
N_TIER1_PARCELS: int = 15  # #4 / token_spec.DEFAULT_BASE_PARCELS

_PS_EVENT_ID: dict[str, int] = {
    "a": 1, "ae": 2, "b": 3, "g": 4, "i": 5, "k": 6, "p": 7, "u": 8, "v": 9,
}


@dataclass(frozen=True)
class V14PhonemeSample:
    """One phoneme's worth of inputs to the per-phoneme model."""

    signal: torch.Tensor                 # (N_e, 130) float32
    patient_id: str
    label: torch.Tensor                  # () long 0..8
    prev_phoneme: torch.Tensor           # () long 0..8 or BOS_TOKEN=-1
    trial_id: int
    phoneme_pos: int                     # 0, 1, or 2
    electrode_grid_layout: torch.Tensor  # (N_e, 2) long
    electrode_grid_shape: tuple[int, int]
    electrode_active_mask: torch.Tensor  # (N_e,) bool
    support: torch.Tensor                # (N_e, 15) float32


def _build_code_to_arpa_index(event_id: dict[str, int]) -> dict[int, int]:
    """Reverse the 9-phoneme event_id to `{code: alphabetical-ARPA-index}`.

    Asserts the phoneme-level event_id matches the canonical PS mapping
    (#18 audit). ARPA index follows `ARPA_PHONEMES` (#16).
    """

    if dict(event_id) != _PS_EVENT_ID:
        raise ValueError(
            f"phoneme event_id {dict(event_id)} != canonical {_PS_EVENT_ID} (#18)"
        )
    out: dict[int, int] = {}
    for ps, code in event_id.items():
        arpa = normalize_label(ps)
        out[int(code)] = ARPA_PHONEMES.index(arpa)
    return out


def _crop_to_phoneme_window(epochs: mne.BaseEpochs) -> mne.BaseEpochs:
    """Crop to `[-0.15, 0.5)` s at 200 Hz → exactly 130 samples."""

    cropped = epochs.copy().crop(tmin=-0.15, tmax=0.495, include_tmax=True)
    if cropped.times.size != T_RAW_SAMPLES:
        raise ValueError(
            f"after crop expected {T_RAW_SAMPLES} samples, got {cropped.times.size}"
        )
    return cropped


class V14PhonemeDataset(torch.utils.data.Dataset[V14PhonemeSample]):
    """Per-phoneme `.fif` dataset. One dataset == one patient == one `.fif`."""

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
        code_to_arpa = _build_code_to_arpa_index(dict(epochs.event_id))
        epochs = _crop_to_phoneme_window(epochs)

        if len(epochs) == 0 or len(epochs) % 3 != 0:
            raise ValueError(
                f"{self.fif_path}: {len(epochs)} epochs; expected 3·n_trials"
            )

        all_names = tuple(epochs.ch_names)
        bads = set(epochs.info["bads"])
        kept_names = tuple(n for n in all_names if n not in bads)
        if not kept_names:
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

        # Pull only kept channels from the `.fif` data matrix. `get_data(picks=...)`
        # returns `(n_epochs, n_kept, T_RAW)` in `kept_names` order.
        signals = epochs.get_data(picks=list(kept_names), copy=False).astype(np.float32)

        codes = epochs.events[:, 2]
        arpa_idx = np.fromiter(
            (code_to_arpa[int(c)] for c in codes), dtype=np.int64, count=len(codes)
        )

        self._signals = signals
        self._labels = arpa_idx
        self._n_trials = len(epochs) // 3
        self._kept_names = kept_names
        self._phys_names = phys_names
        self._grid_shape = grid_shape
        self._electrode_grid_layout = layout
        self._electrode_active_mask = active_mask
        self._support = support

    @property
    def n_trials(self) -> int:
        return self._n_trials

    def trial_tokens(self) -> list[str]:
        """Per-trial identity strings for grouped-by-token CV.

        Each trial's 3 consecutive phoneme labels are joined into a single
        string (`"{AA}_{B}_{K}"`). Two trials with the same 3-phoneme sequence
        share an identity — equivalent to grouping by PS token because every
        PS token has a unique ARPA triplet.
        """

        out: list[str] = []
        for t in range(self._n_trials):
            p0 = int(self._labels[3 * t])
            p1 = int(self._labels[3 * t + 1])
            p2 = int(self._labels[3 * t + 2])
            out.append(f"{ARPA_PHONEMES[p0]}_{ARPA_PHONEMES[p1]}_{ARPA_PHONEMES[p2]}")
        return out

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def __getitem__(self, index: int) -> V14PhonemeSample:
        if not 0 <= index < len(self):
            raise IndexError(index)
        trial_id = index // 3
        pos = index % 3
        prev = BOS_TOKEN if pos == 0 else int(self._labels[index - 1])

        return V14PhonemeSample(
            signal=torch.from_numpy(
                np.ascontiguousarray(self._signals[index], dtype=np.float32)
            ),
            patient_id=self.patient_id,
            label=torch.tensor(int(self._labels[index]), dtype=torch.long),
            prev_phoneme=torch.tensor(prev, dtype=torch.long),
            trial_id=trial_id,
            phoneme_pos=pos,
            electrode_grid_layout=torch.from_numpy(
                np.ascontiguousarray(self._electrode_grid_layout, dtype=np.int64)
            ),
            electrode_grid_shape=self._grid_shape,
            electrode_active_mask=torch.from_numpy(
                np.ascontiguousarray(self._electrode_active_mask, dtype=bool)
            ),
            support=torch.from_numpy(
                np.ascontiguousarray(self._support, dtype=np.float32)
            ),
        )


def collate_v14_phoneme_batch(samples: list[V14PhonemeSample]) -> dict[str, object]:
    """Stack per-phoneme samples into a grouped-by-patient batch dict (#31).

    Same patient per batch. Mixed `phoneme_pos` allowed. Returns:
        signal[B, N_e, 130], labels[B], prev_tokens[B], phoneme_pos[B],
        trial_id[B], electrode_grid_layout[B, N_e, 2], electrode_grid_shape,
        electrode_active_mask[B, N_e], support[B, N_e, 15], patient_id.
    """

    if not samples:
        raise ValueError("collate_v14_phoneme_batch received empty batch")
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
            raise ValueError(f"batch[{i}] N_e {s.signal.shape[0]} != {n_e}")
        if s.electrode_grid_shape != grid_shape:
            raise ValueError(
                f"batch[{i}] grid_shape {s.electrode_grid_shape} != {grid_shape}"
            )

    signal = torch.stack([s.signal for s in samples], dim=0)
    labels = torch.stack([s.label for s in samples], dim=0)
    prev_tokens = torch.stack([s.prev_phoneme for s in samples], dim=0)
    phoneme_pos = torch.tensor(
        [s.phoneme_pos for s in samples], dtype=torch.long
    )
    trial_id = torch.tensor(
        [s.trial_id for s in samples], dtype=torch.long
    )
    electrode_grid_layout = torch.stack(
        [s.electrode_grid_layout for s in samples], dim=0
    )
    electrode_active_mask = torch.stack(
        [s.electrode_active_mask for s in samples], dim=0
    )
    support = torch.stack([s.support for s in samples], dim=0)

    return {
        "signal": signal,
        "labels": labels,
        "prev_tokens": prev_tokens,
        "phoneme_pos": phoneme_pos,
        "trial_id": trial_id,
        "electrode_grid_layout": electrode_grid_layout,
        "electrode_grid_shape": grid_shape,
        "electrode_active_mask": electrode_active_mask,
        "support": support,
        "patient_id": patient_id,
    }
