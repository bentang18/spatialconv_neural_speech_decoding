"""V14 per-electrode atlas-token metadata extractor.

For each Ieeg event emit a `(n_channels, 5)` tensor with columns:

    [:, 0]   = Tier-1 parcel id (int, 0..N_TIER1-1) — argmax-of-support cell
    [:, 1]   = support weight    (float, [0, 1])    — peak Tier-1 mass
    [:, 2:5] = fsaverage_xyz     (mm)               — strict snap-to-pial

Source of truth: per-patient Tier-1 support CSV under `support_cache_dir`,
file pattern `<event.subject>_support_tier1.csv`. Cohort-agnostic — the same
extractor runs over NeuralSet `Ieeg` events from BrainTreebank and Cogan cohorts.

`fsaverage_xyz` is read from a sibling per-patient CSV at
`fsaverage_coords_dir/<event.subject>_fsaverage_pial.csv` (canonical strict
snap-to-pial coordinates). Raises if either file is missing — parcel tokens
are load-bearing, never silently substituted.
"""

from __future__ import annotations

import csv
import typing as tp
from pathlib import Path

import numpy as np
import torch
from neuralset.events.etypes import Event
from neuralset.extractors.base import BaseStatic

from speech_decoding.atlas.support import read_support_cache


class V14ParcelMetadataExtractor(BaseStatic):
    """Per-electrode (parcel_id, support, fsaverage_xyz) for an Ieeg event."""

    event_types: tp.Literal["Ieeg"] = "Ieeg"
    support_cache_dir: str
    fsaverage_coords_dir: str

    def get_static(self, event: Event) -> torch.Tensor:
        subject = str(getattr(event, "subject"))
        support_path = Path(self.support_cache_dir) / f"{subject}_support_tier1.csv"
        coords_path = (
            Path(self.fsaverage_coords_dir) / f"{subject}_fsaverage_pial.csv"
        )

        names, support_pct = read_support_cache(support_path)
        coords = _read_fsaverage_coords(coords_path, names)

        support_pct = support_pct.astype(np.float32, copy=False)
        parcel_id = support_pct.argmax(axis=1).astype(np.float32)
        peak_support = (support_pct.max(axis=1) / 100.0).astype(np.float32)

        out = np.concatenate(
            [parcel_id[:, None], peak_support[:, None], coords], axis=1
        )
        return torch.from_numpy(out)


def _read_fsaverage_coords(
    path: Path, kept_names: tuple[str, ...]
) -> np.ndarray:
    """Return `(N_e, 3)` xyz in `kept_names` order. Raises on any missing name."""

    if not path.exists():
        raise FileNotFoundError(f"fsaverage coords cache missing: {path}")
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = {row["name"]: row for row in reader}
    missing = [n for n in kept_names if n not in rows]
    if missing:
        raise KeyError(
            f"{path}: missing electrodes in fsaverage coords cache: "
            f"{missing[:10]}"
            + (f" (+{len(missing) - 10} more)" if len(missing) > 10 else "")
        )
    coords = np.empty((len(kept_names), 3), dtype=np.float32)
    for i, name in enumerate(kept_names):
        row = rows[name]
        coords[i] = (float(row["x"]), float(row["y"]), float(row["z"]))
    return coords
