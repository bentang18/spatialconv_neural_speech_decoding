"""Per-patient calibration modules for Neural Field Perceiver v14.

Implementation intentionally deferred. This file will own:

- atlas-resource loading for spatial membership lookup
- gain/offset calibration
- rigid correction
- parcel offsets
- parcel temperature logic

Current default atlas target:

- real Brainnetome 4D probability map: ``/Users/bentang/Documents/Code/speech/data/atlas/BNA_PM_4D.nii.gz``

Previous proxy construction, kept only for historical context:

- start from the Brainnetome MPM/label map
- build a binary mask for each ROI
- smooth each ROI mask with a Gaussian kernel (the docs previously assumed
  ``sigma=5 mm``)
- treat the smoothed values as pseudo-probabilities

That proxy was useful for quick prototyping, but the Phase 1 implementation
target should now use the real probability map instead. The MPM file is still
allowed for ROI indexing and sanity checks, but Phase 1 should not silently
fall back to smoothed pseudo-probabilities.
"""

from __future__ import annotations

from pathlib import Path

from .config import AtlasConfig


def get_brainnetome_probability_map_path(atlas: AtlasConfig) -> Path:
    """Return the configured path to the real Brainnetome 4D PM file."""

    return Path(atlas.probability_map_path)


def get_brainnetome_label_map_path(atlas: AtlasConfig) -> Path:
    """Return the configured path to the Brainnetome label/MPM volume."""

    return Path(atlas.label_map_path)


def brainnetome_probability_map_exists(atlas: AtlasConfig) -> bool:
    """Return whether the configured real PM file exists locally."""

    return get_brainnetome_probability_map_path(atlas).exists()
