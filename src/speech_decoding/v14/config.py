"""Configuration dataclasses for Neural Field Perceiver v14 (B-1).

Every numeric default here tracks a frozen Phase-1 contract in
``docs/implementation_tasks.md`` (with the 2026-04-16 late ``[AMENDED]``
markers) and ``docs/v14_core_contract_amendment_2026-04-16.md``. Blockers
cited per field. No speculative defaults; no Phase-2 leash constants
carried as live Phase-1 values (per ``#15``).

The B-1 amendment retired the within-parcel Perceiver summarizer
(``LocalSummarizerConfig``) in favor of a two-stage spatial path:
``GridMixerConfig`` (whole-grid Conv2d) and ``SoftParcelEmbeddingConfig``
(soft probability over Tier-1 via raw ``support[N_e, 15]``). The factored
backbone was replaced by a combined spatiotemporal attention backbone with
``B = 3`` blocks, 4 heads × ``head_dim = 16``, RoPE on temporal only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _default_brainnetome_probability_map_path() -> str:
    """Path to the real 4D Brainnetome probability map (atlas-side source)."""

    return str(Path("/Users/bentang/Documents/Code/speech/data/atlas/BNA_PM_4D.nii.gz"))


def _default_brainnetome_label_map_path() -> str:
    """Path to the Brainnetome MPM/label map (ROI indexing / sanity only)."""

    return str(Path.home() / "nilearn_data" / "bnatlas.nii.gz")


@dataclass(frozen=True)
class AtlasConfig:
    """Atlas resources for Phase 1.

    Phase 1 bakes BNA PM onto fsaverage once per ``#36``; the raw 4D PM volume
    remains the source of truth for the bake. The MPM file is allowed for
    ROI indexing and sanity checks only. No silent fallback to the MPM proxy.
    """

    probability_map_path: str = field(default_factory=_default_brainnetome_probability_map_path)
    label_map_path: str = field(default_factory=_default_brainnetome_label_map_path)
    use_real_probability_maps: bool = True
    allow_proxy_fallback: bool = False


@dataclass(frozen=True)
class PatientCalibrationConfig:
    """Per-patient calibration flags.

    Phase 1 keeps every learn_* flag False per the fixed-atlas contract.
    Leash constants (translation/rotation/offset/temperature bounds) are
    deliberately absent — they would re-enter only when Phase 2 turns
    learned calibration on, and ``#15`` forbids them as live Phase-1 values.
    """

    learn_gain_offset: bool = False
    learn_rigid_correction: bool = False
    learn_parcel_offsets: bool = False
    learn_parcel_temperature: bool = False


@dataclass(frozen=True)
class TemporalTokenizerConfig:
    """Shared per-electrode Conv1d patch tokenizer, frozen per ``#2`` / ``#6``.

    Input: 200 Hz, trial window ``(B, N_e, 300)`` from ``#29``.
    Kernel 150 ms = 30 samples, stride 50 ms = 10 samples.
    Output: ``(B, N_e, d, 28)`` at a 20 Hz token rate.
    """

    d_model: int = 64
    patch_ms: int = 150
    stride_ms: int = 50
    sample_rate_hz: int = 200


@dataclass(frozen=True)
class GridMixerConfig:
    """Whole-grid Conv2d (Stage 2 of the B-1 architecture).

    Per-time-step 2D convolution with shared weights across ``t``, full
    channel mixing. Pre-norm LayerNorm (channel dim) + GELU + residual
    masked by ``electrode_active_mask``. Default depth is 1 layer; depth
    2 is ablation A3 in ``docs/plans/v14-core.md``.

    Justified by the B-1 amendment: kernel 3×3 with stride 1 gives a
    ±1.33–3 mm receptive field, matching the HGA correlation length on
    the rigid uECoG grid.
    """

    d_model: int = 64
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    num_layers: int = 1


@dataclass(frozen=True)
class SoftParcelEmbeddingConfig:
    """Soft parcel embedding (Stage 3 of the B-1 architecture).

    Learnable lookup ``P_emb: (n_parcels, d_model)`` (Xavier-uniform init).
    Per-electrode embedding is ``support @ P_emb``, where ``support`` is
    the raw BNA probability over Tier-1 in ``[0, 100]`` (``#5``). Raw —
    not row-normalized — so electrodes with low Tier-1 mass naturally
    contribute weakly.

    ``n_parcels = 15`` matches ``DEFAULT_BASE_PARCELS`` in ``token_spec``
    and is fixed by ``#4``.
    """

    d_model: int = 64
    n_parcels: int = 15


@dataclass(frozen=True)
class BackboneConfig:
    """Combined spatiotemporal backbone, amended ``#27`` (B-1 amendment).

    Each block is one combined attention layer over the flat sequence of
    ``(N_e × T)`` tokens, then a pre-norm FFN. ``B = 3`` blocks, 4 heads
    × ``head_dim = 16`` (``= d_model // num_heads``), FFN width
    ``4d = 256``, GELU, dropout 0.1, pre-norm. RoPE applied to the
    temporal axis only.

    No SC/FC bias in baseline (deferred to Phase F ablation A4 per the
    amended ``#8``).
    """

    d_model: int = 64
    num_blocks: int = 3
    num_heads: int = 4
    head_dim: int = 16
    ffn_hidden: int = 256
    dropout: float = 0.1


@dataclass(frozen=True)
class DecoderConfig:
    """AR phoneme decoder, frozen per ``#9`` / ``#15`` / ``#28``.

    One AR block, 3 fixed output slots, shared base query + one slot
    embedding per position, one causal self-attention over the 3 slot
    queries + one cross-attention over flattened backbone memory
    ``(N_e × T, d)``, shared linear vocab head. Vocab follows the ``#16``
    alphabetical ARPABET index (AA=0 … V=8).
    """

    d_model: int = 64
    num_queries: int = 3
    vocab_size: int = 9
    ar_embedding_dim: int = 64


@dataclass(frozen=True)
class V14Config:
    """Top-level configuration for the Phase-1 v14 (B-1) implementation."""

    atlas: AtlasConfig = field(default_factory=AtlasConfig)
    calibration: PatientCalibrationConfig = field(default_factory=PatientCalibrationConfig)
    tokenizer: TemporalTokenizerConfig = field(default_factory=TemporalTokenizerConfig)
    grid_mixer: GridMixerConfig = field(default_factory=GridMixerConfig)
    parcel_embedding: SoftParcelEmbeddingConfig = field(default_factory=SoftParcelEmbeddingConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


# --------------------------------------------------------------------------- #
# Per-phoneme baseline-aligned configs (plan: docs/plans/v14-core-baseline-aligned.md)
#
# These are used by the `NeuralFieldPerceiverPerPhoneme` stack. They coexist
# with the B-1 configs above so the slot-CE path continues to work while the
# per-phoneme path ramps up.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PoolConfig:
    """Masked-mean pool to the baseline `(4, 8)` cell grid.

    All Phase-1 patient grids (8×16, 16×16, 12×24) divide evenly into this
    pool shape; divisibility is asserted at precompute time.
    """

    pool_shape: tuple[int, int] = (4, 8)


@dataclass(frozen=True)
class PerCellTemporalConfig:
    """Per-cell Conv1d temporal compressor (Stage 4 of the per-phoneme stack).

    Kernel 30 samples = 150 ms at 200 Hz; stride 10 samples = 50 ms. Matches
    Ben's 0.734 baseline temporal front-end. Output token rate 20 Hz.
    """

    in_channels: int = 8
    out_channels: int = 32
    kernel_size: int = 30
    stride: int = 10


@dataclass(frozen=True)
class D1DecoderConfig:
    """Minimum AR decoder for the per-phoneme path.

    Reduces backbone memory to a (B, d) summary per `readout_mode`, adds
    BOS-aware `prev_phoneme` embedding, projects to vocab. Vocab index 9 is
    the BOS slot in the embedding table.

    ``readout_mode``:
    * "mean_pool" — current baseline. ``memory.mean(dim=1)``. 0 added params.
    * "cls" — expects the top-level model to prepend a learnable CLS token
      at position 0. Decoder reads ``memory[:, 0]``. +d params live on the
      model, not here.
    * "hierarchical" — expects memory shape ``(B, n_cells * t_tokens, d)``.
      Reshape to ``(B, n_cells, t_tokens, d)``; attention-pool across
      ``t_tokens`` via learned query ``q_temporal ∈ R^d`` → ``(B, n_cells, d)``;
      then attention-pool across ``n_cells`` via learned ``q_cell ∈ R^d`` →
      ``(B, d)``. +2d params. ``n_cells`` / ``t_tokens`` default to the
      canonical per_cell path at pool (4,8) + tmin=-0.15, tmax=0.5, fs=200,
      k=30, s=10 → 32 cells, 11 tokens.
    """

    d_model: int = 32
    vocab_size: int = 9  # 9 ARPA phonemes
    prev_embedding_size: int = 10  # 9 phonemes + 1 BOS
    readout_mode: str = "mean_pool"
    n_cells: int = 32
    t_tokens: int = 11


@dataclass(frozen=True)
class PerPhonemeConfig:
    """Top-level config for the baseline-aligned per-phoneme v14 stack.

    Backbone defaults to `d_model=32, num_heads=2, head_dim=16, ffn_hidden=128`
    — half the B-1 width, matched to the 45k total-param budget the plan calls
    for.
    """

    d_model: int = 32
    conv2d_in_channels: int = 1
    conv2d_out_channels: int = 8
    conv2d_kernel_size: int = 3
    conv2d_padding: int = 1
    pool: PoolConfig = field(default_factory=PoolConfig)
    per_cell_temporal: PerCellTemporalConfig = field(default_factory=PerCellTemporalConfig)
    parcel_embedding_n_parcels: int = 15
    use_parcel_embedding: bool = True
    # "per_cell": shared Conv1d(8→d, k, s) applied to each cell independently
    # (~7.7k temporal params). Canonical. Permutation-equivariant across cells;
    # cross-patient alignment via the atlas parcel embedding; scales natively
    # to variable-shape arrays and per-electrode tokens (required for sEEG /
    # external corpora). This is the long-term research direction — cross-
    # patient / cross-sensor generalization. Do not swap the default to `flat`
    # chasing per-subject peaks; flat is a rigid-grid local optimum that
    # breaks the moment we touch sEEG or non-uECoG arrays.
    # "flat": Conv1d(8·n_cells → d, k, s) on flattened (B, C·n_cells, T) —
    # baseline 2026-04-04 front-end that bakes full spatial mixing into the
    # temporal conv (~245k params, position-specific weights). Retained as
    # ablation / single-dataset reference; NOT the canonical path.
    temporal_frontend: str = "per_cell"
    # "masked_mean": our Stage-3 pool — start-index-only non-overlapping bins,
    # divisor = #active electrodes per cell. Inactives drop out of both ends.
    # "adaptive_avg": torch.nn.functional.adaptive_avg_pool2d over the scattered
    # grid — overlapping bins on non-divisible W, divisor = fixed bin size.
    # Matches the 2026-04-04 baseline pool. Only materially differs from
    # masked_mean on non-divisible grids (12×22) or with inactive electrodes.
    pool_method: str = "masked_mean"
    # "zero_fill": artifact / pad positions are zero-filled before Conv2d; the
    # k=3 kernel then spreads those zeros into active neighbors, attenuating
    # their output by ~(k²-n_artifact)/k². Baseline behavior.
    # "partial_conv": Liu et al. 2018 renormalization —
    #     out = W · (X ⊙ M) · (k² / sum(M in RF))    if sum(M) > 0, else 0
    # Zero learnable params, purely arithmetic correction. Layout-agnostic;
    # transfers identically to sEEG / external arrays / future grid shapes.
    # No mask propagation needed (only one Conv2d layer before the pool).
    masking_mode: str = "zero_fill"
    # "mean_pool": current baseline — decoder does memory.mean(dim=1).
    # "cls": prepend a learnable CLS token at position 0 before the backbone
    #   (runs through all 3 combined-attention blocks like any other token).
    #   Decoder reads position 0. +d_model params. No-regret integration test
    #   — CLS has been the standard ViT/BERT readout since 2017.
    # "hierarchical": keep the flat (B, n_cells*T_tokens, d) memory; the
    #   decoder reshapes to (B, n_cells, T_tokens, d) and runs two learned-
    #   query attention pools — first across time per cell, then across cells.
    #   +2·d_model params. Tests the factored-structure hypothesis — whether
    #   summarizing time-within-cell before pooling across cells beats a flat
    #   single-pool. per_cell-only (flat temporal_frontend has no cell axis).
    readout_mode: str = "mean_pool"
    # "none": no added spatial PE (baseline). Cells carry only parcel + time info.
    # "factorized_2d": rigid-grid ViT-style PE — learnable row_emb (H_pool, d/2)
    #   and col_emb (W_pool, d/2), concatenated to (n_cells, d) and added to
    #   every cell broadcast across time tokens. +H_pool·d/2 + W_pool·d/2 params
    #   (at pool (4,8), d=32 → 192 params). Does NOT transfer to sEEG (no grid
    #   axis); retained as rigid-grid bucket ablation. per_cell-only.
    spatial_pe_mode: str = "none"
    # "pool": current baseline — scatter to grid, Conv2d, masked-mean pool to
    #   (H_pool, W_pool) cells, per-cell temporal Conv1d. Grid-dependent.
    # "per_electrode": skips scatter + Conv2d + pool; runs a shared per-
    #   electrode Conv1d(1 → d, k, s) directly on the raw per-electrode signal,
    #   then adds per-electrode atlas embedding (support @ P_emb). Tokens are
    #   (N_e × T_tokens). Grid-agnostic; canonical path for sEEG / external
    #   arrays. Requires temporal_frontend='per_cell' (flat bakes in spatial
    #   mixing). Incompatible with factorized_2d spatial PE and hierarchical
    #   readout (both grid-only).
    spatial_path: str = "pool"
    # Per-electrode path extra PE.
    # "none": atlas embedding only (default).
    # "fourier_mni": random-Fourier PE on electrode xyz. A frozen random matrix
    #   W ~ N(0, fourier_pe_std^2) with shape (3, d/2) projects the 3D coord
    #   into d/2 phases; PE = concat(sin(xyz @ W), cos(xyz @ W)) ∈ R^d. Added
    #   per electrode broadcast across time. per_electrode-only.
    # "distance_bias": pairwise Euclidean-distance additive bias in backbone
    #   attention (added separately in task #51).
    electrode_pe_mode: str = "none"
    # Frequency scale for Fourier PE (cycles per mm). Default 0.2 mm^-1 gives
    # per-coord phase excursion ~25 at 130 mm extent — enough cycles to be
    # informative without aliasing into noise.
    fourier_pe_std: float = 0.2
    backbone: BackboneConfig = field(
        default_factory=lambda: BackboneConfig(
            d_model=32, num_heads=2, head_dim=16, ffn_hidden=128,
            num_blocks=3, dropout=0.1,
        )
    )
    decoder: D1DecoderConfig = field(default_factory=D1DecoderConfig)
