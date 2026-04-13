"""Neural Field Perceiver v14 implementation boundary.

This package is the clean home for the current atlas-calibrated architecture.
Legacy baseline modules remain elsewhere in the repo.
"""

from .config import (
    AtlasConfig,
    BackboneConfig,
    DecoderConfig,
    LocalSummarizerConfig,
    PatientCalibrationConfig,
    TemporalTokenizerConfig,
    V14Config,
)
from .token_spec import DEFAULT_BASE_PARCELS, DEFAULT_SPLIT_COUNTS, default_token_count

__all__ = [
    "AtlasConfig",
    "BackboneConfig",
    "DecoderConfig",
    "LocalSummarizerConfig",
    "PatientCalibrationConfig",
    "TemporalTokenizerConfig",
    "V14Config",
    "DEFAULT_BASE_PARCELS",
    "DEFAULT_SPLIT_COUNTS",
    "default_token_count",
]
