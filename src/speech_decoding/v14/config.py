from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _default_brainnetome_probability_map_path() -> str:
    """Return the local path to the real 4D Brainnetome probability map."""

    return str(Path("/Users/bentang/Documents/Code/speech/data/atlas/BNA_PM_4D.nii.gz"))


def _default_brainnetome_label_map_path() -> str:
    """Return the local path to the Brainnetome MPM/label map."""

    return str(Path.home() / "nilearn_data" / "bnatlas.nii.gz")


@dataclass(frozen=True)
class AtlasConfig:
    """Atlas resources and loading defaults for v14.

    Phase 1 should target the real Brainnetome probabilistic map rather than
    the older smoothed-MPM proxy. The MPM file remains useful for ROI indices
    and sanity checks only. Phase 1 should not silently fall back to a proxy
    map if the real PM file is missing or misconfigured.
    """

    probability_map_path: str = field(default_factory=_default_brainnetome_probability_map_path)
    label_map_path: str = field(default_factory=_default_brainnetome_label_map_path)
    use_real_probability_maps: bool = True
    allow_proxy_fallback: bool = False


@dataclass(frozen=True)
class PatientCalibrationConfig:
    """Small per-patient parameters that define the calibration problem."""

    # Defaults are intentionally conservative for the first implementation pass.
    # The initial `v14-core` target uses fixed atlas mapping before any learned
    # per-patient calibration is enabled.
    learn_gain_offset: bool = False
    learn_rigid_correction: bool = False
    learn_parcel_offsets: bool = False
    learn_parcel_temperature: bool = False
    max_translation_mm: float = 15.0
    max_rotation_rad: float = 0.15
    max_parcel_offset_mm: float = 15.0
    min_temperature: float = 0.3
    max_temperature: float = 3.0


@dataclass(frozen=True)
class TemporalTokenizerConfig:
    """Shared temporal front-end before atlas pooling."""

    d_model: int = 64
    patch_ms: int = 250
    stride_ms: int = 50
    sample_rate_hz: int = 200
    hidden_channels: tuple[int, ...] = (16, 32, 32)


@dataclass(frozen=True)
class LocalSummarizerConfig:
    """Within-parcel Perceiver-style local summarizer."""

    d_model: int = 64
    point_mlp_hidden: int = 64
    parcel_embedding_dim: int = 16
    support_feature_dim: int = 1


@dataclass(frozen=True)
class BackboneConfig:
    """Shared relational-temporal model after atlas-token formation."""

    d_model: int = 64
    num_blocks: int = 2
    num_heads: int = 4
    ffn_hidden: int = 256
    dropout: float = 0.2


@dataclass(frozen=True)
class DecoderConfig:
    """Autoregressive phoneme decoder."""

    d_model: int = 64
    num_queries: int = 3
    vocab_size: int = 9
    ar_embedding_dim: int = 64


@dataclass(frozen=True)
class V14Config:
    """Top-level configuration for the uECoG-first v14 implementation."""

    atlas: AtlasConfig = field(default_factory=AtlasConfig)
    calibration: PatientCalibrationConfig = field(default_factory=PatientCalibrationConfig)
    tokenizer: TemporalTokenizerConfig = field(default_factory=TemporalTokenizerConfig)
    summarizer: LocalSummarizerConfig = field(default_factory=LocalSummarizerConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
