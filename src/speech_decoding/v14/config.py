"""Configuration dataclasses for Neural Field Perceiver v14.

=============================================================================
PROVISIONAL — DISCUSS BEFORE USE (2026-04-13)
=============================================================================

Every numeric default in this file was written before the 2026-04-13
discussion-first rule was locked. None of these values have a written
justification tied to data scale, token rate, or a named paper. Under
the working principle recorded in `CLAUDE.md`, `docs/current_direction.md`,
and `docs/implementation_tasks.md` (blocker #15), none of these numbers may
enter a training run until each has been explicitly discussed and agreed.

Specifically unresolved:

- PatientCalibrationConfig leashes: max_translation_mm, max_rotation_rad,
  max_parcel_offset_mm, min_temperature, max_temperature. These are
  inherited from the pre-Phase-1 v14 draft.
- TemporalTokenizerConfig: d_model, patch_ms, stride_ms, sample_rate_hz,
  hidden_channels. The temporal front-end is itself a top blocker (#2/#6).
- LocalSummarizerConfig: d_model, point_mlp_hidden, parcel_embedding_dim,
  support_feature_dim. Gated on #26.
- BackboneConfig: d_model, num_blocks, num_heads, ffn_hidden, dropout.
  Current values match the written expectation in #27 (d=256, B=2,
  head_dim=64, ~1.2M–2M params) but are not yet frozen.
- DecoderConfig: d_model, num_queries, vocab_size, ar_embedding_dim. The
  label->integer index contract is a separate blocker (#16) and the full
  decoder contract is #28.

The `AtlasConfig` and the `learn_*` flags on `PatientCalibrationConfig`
encode the Phase-1 contract (real PM volume, no learned calibration) and
are therefore consistent with today's decisions. Everything else is a
placeholder skeleton.

If you are reading this before a field has been discussed and agreed,
surface the discussion with Ben rather than importing the default.
=============================================================================
"""

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
    """Shared temporal front-end before atlas pooling.

    `d_model = 256` matches the written expectation in blocker #27 so the
    parameter-budget math there is consistent. Every field is still
    provisional under #15 / #2 / #6 — see the module banner.
    """

    d_model: int = 256
    patch_ms: int = 250
    stride_ms: int = 50
    sample_rate_hz: int = 200
    hidden_channels: tuple[int, ...] = (16, 32, 32)


@dataclass(frozen=True)
class LocalSummarizerConfig:
    """Within-parcel Perceiver-style local summarizer.

    `d_model = 256` matches the shared width used by the tokenizer and
    backbone. Provisional under #15 / #26.
    """

    d_model: int = 256
    point_mlp_hidden: int = 256
    parcel_embedding_dim: int = 16
    support_feature_dim: int = 1


@dataclass(frozen=True)
class BackboneConfig:
    """Shared relational-temporal model after atlas-token formation.

    `d_model = 256`, `head_dim = 64`, `B = 2` factored blocks are the
    written defaults in blocker #27 (~1.2M–2M parameters). All values
    still provisional under #15 / #27 until the discussion freezes.
    """

    d_model: int = 256
    num_blocks: int = 2
    num_heads: int = 4
    ffn_hidden: int = 1024
    dropout: float = 0.1


@dataclass(frozen=True)
class DecoderConfig:
    """Autoregressive phoneme decoder.

    `d_model = 256` matches the backbone width so cross-attention keys
    do not need an extra projection. `num_queries = 3` and `vocab_size = 9`
    follow the 3-phoneme target and the 9-phoneme PS set but the exact
    label→index contract is blocker #16. All values still provisional
    under #15 / #28.
    """

    d_model: int = 256
    num_queries: int = 3
    vocab_size: int = 9
    ar_embedding_dim: int = 256


@dataclass(frozen=True)
class V14Config:
    """Top-level configuration for the uECoG-first v14 implementation."""

    atlas: AtlasConfig = field(default_factory=AtlasConfig)
    calibration: PatientCalibrationConfig = field(default_factory=PatientCalibrationConfig)
    tokenizer: TemporalTokenizerConfig = field(default_factory=TemporalTokenizerConfig)
    summarizer: LocalSummarizerConfig = field(default_factory=LocalSummarizerConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
