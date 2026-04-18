"""Neural Field Perceiver v14 (B-1) implementation boundary.

This package is the clean home for the current atlas-calibrated architecture.
Legacy baseline modules remain elsewhere in the repo.

`DEFAULT_BASE_PARCELS`, `DEFAULT_SPLIT_COUNTS`, and `default_token_count` live
in `token_spec` and were frozen 2026-04-14 by blocker #4 closure, revised
later that day to the argmax-centric rule `argmax_wins >= 10` (`N_tok = 15`,
uniform `k_parcel = 1`). With the 2026-04-16 late B-1 amendment, the
`DEFAULT_BASE_PARCELS` tuple is no longer used to build per-parcel tokens;
it is the embedding-lookup table for the soft parcel embedding (Stage 3).
Any v14 code path that consumes it should still call
`assert_token_spec_frozen()` first in case the provisional flag is ever
flipped back.
"""

from .config import (
    AtlasConfig,
    BackboneConfig,
    DecoderConfig,
    GridMixerConfig,
    PatientCalibrationConfig,
    SoftParcelEmbeddingConfig,
    TemporalTokenizerConfig,
    V14Config,
)

__all__ = [
    "AtlasConfig",
    "BackboneConfig",
    "DecoderConfig",
    "GridMixerConfig",
    "PatientCalibrationConfig",
    "SoftParcelEmbeddingConfig",
    "TemporalTokenizerConfig",
    "V14Config",
]
