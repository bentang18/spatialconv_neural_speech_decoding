"""Neural Field Perceiver v14 implementation boundary.

This package is the clean home for the current atlas-calibrated architecture.
Legacy baseline modules remain elsewhere in the repo.

Note: `DEFAULT_BASE_PARCELS`, `DEFAULT_SPLIT_COUNTS`, and `default_token_count`
are intentionally not re-exported at package level. They live in `token_spec`
and are still provisional under blocker #4 (Phase-1 parcel re-derivation);
any v14 code path that consumes them must import from `token_spec` directly
and call `assert_token_spec_frozen()` first.
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

__all__ = [
    "AtlasConfig",
    "BackboneConfig",
    "DecoderConfig",
    "LocalSummarizerConfig",
    "PatientCalibrationConfig",
    "TemporalTokenizerConfig",
    "V14Config",
]
