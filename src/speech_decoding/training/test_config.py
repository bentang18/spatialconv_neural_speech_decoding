"""Smoke tests for `src/speech_decoding/v14/config.py` against frozen `#15`
plus the 2026-04-16 late B-1 amendment
(`docs/v14_core_contract_amendment_2026-04-16.md`).

`#15` freezes the width budget and module shapes. The whole point of these
assertions is that a silent regression (e.g. someone flipping `d_model` back
to 256, or re-introducing the within-parcel summarizer config) fails loudly
in CI rather than producing a quietly wrong run.
"""

from __future__ import annotations

from speech_decoding.training.config import (
    AtlasConfig,
    BackboneConfig,
    DecoderConfig,
    GridMixerConfig,
    PatientCalibrationConfig,
    SoftParcelEmbeddingConfig,
    TemporalTokenizerConfig,
    V14Config,
)


class TestTemporalTokenizerConfig:
    def test_frozen_defaults_match_blocker_2_and_6(self) -> None:
        cfg = TemporalTokenizerConfig()
        assert cfg.d_model == 64           # #15 baseline width
        assert cfg.patch_ms == 150         # #2 kernel = 30 samples at 200 Hz
        assert cfg.stride_ms == 50         # #2 stride = 10 samples at 200 Hz
        assert cfg.sample_rate_hz == 200   # #2, #6


class TestGridMixerConfig:
    def test_frozen_defaults_match_b1_amendment_stage_2(self) -> None:
        """Stage 2 of the B-1 amendment: per-time-step Conv2d, k=3, 1 layer baseline."""
        cfg = GridMixerConfig()
        assert cfg.d_model == 64           # shared width per #15
        assert cfg.kernel_size == 3        # ±1.33-3 mm receptive field
        assert cfg.stride == 1
        assert cfg.padding == 1
        assert cfg.num_layers == 1         # depth-2 is ablation A3


class TestSoftParcelEmbeddingConfig:
    def test_frozen_defaults_match_b1_amendment_stage_3(self) -> None:
        """Stage 3 of the B-1 amendment: support @ P_emb soft routing."""
        cfg = SoftParcelEmbeddingConfig()
        assert cfg.d_model == 64           # shared width per #15
        assert cfg.n_parcels == 15         # matches DEFAULT_BASE_PARCELS per #4

    def test_n_parcels_matches_default_base_parcels(self) -> None:
        """Soft embedding lookup width must equal the Tier-1 cardinality (#4)."""
        from speech_decoding.atlas.tokens import DEFAULT_BASE_PARCELS
        cfg = SoftParcelEmbeddingConfig()
        assert cfg.n_parcels == len(DEFAULT_BASE_PARCELS)


class TestBackboneConfig:
    def test_frozen_defaults_match_amended_blocker_27(self) -> None:
        """Combined spatiotemporal attention per the B-1 amendment.

        Walked back from factored (B = 2 block pairs, 2 heads × head_dim = 32)
        to combined (B = 3 blocks, 4 heads × head_dim = 16) — see
        ``docs/v14_core_contract_amendment_2026-04-16.md``.
        """
        cfg = BackboneConfig()
        assert cfg.d_model == 64
        assert cfg.num_blocks == 3         # B-1: combined blocks, fewer needed
        assert cfg.num_heads == 4          # B-1: 4 heads
        assert cfg.head_dim == 16          # = d_model // num_heads
        assert cfg.ffn_hidden == 256       # 4d, unchanged
        assert cfg.dropout == 0.1          # unchanged

    def test_no_sc_fc_baseline_fields(self) -> None:
        """SC/FC bias is deferred to Phase F ablation A4 per the amended #8.

        The baseline backbone must not carry SC/FC parameters.
        """
        fields = set(BackboneConfig.__dataclass_fields__.keys())
        forbidden = {"sc_gain", "fc_gain", "alpha_sc", "alpha_fc", "use_sc_fc_bias"}
        leaked = fields & forbidden
        assert not leaked, f"SC/FC bias fields leaked into baseline backbone: {leaked}"


class TestDecoderConfig:
    def test_frozen_defaults_match_blocker_9_and_28(self) -> None:
        cfg = DecoderConfig()
        assert cfg.d_model == 64
        assert cfg.num_queries == 3        # #9 fixed 3-slot target
        assert cfg.vocab_size == 9         # #16 9 PS phonemes
        assert cfg.ar_embedding_dim == 64  # matches backbone width


class TestAtlasConfig:
    def test_real_pm_not_proxy(self) -> None:
        cfg = AtlasConfig()
        assert cfg.use_real_probability_maps is True
        assert cfg.allow_proxy_fallback is False
        assert cfg.probability_map_path.endswith("BNA_PM_4D.nii.gz")


class TestPatientCalibrationConfig:
    def test_all_learn_flags_off_in_phase1(self) -> None:
        """Phase 1 is fixed-atlas: every learn_* flag must default False."""
        cfg = PatientCalibrationConfig()
        assert cfg.learn_gain_offset is False
        assert cfg.learn_rigid_correction is False
        assert cfg.learn_parcel_offsets is False
        assert cfg.learn_parcel_temperature is False

    def test_no_phase2_leash_fields_leak_into_phase1(self) -> None:
        """#15 forbids Phase-2 leash constants as live Phase-1 defaults."""
        fields = set(PatientCalibrationConfig.__dataclass_fields__.keys())
        forbidden = {
            "max_translation_mm",
            "max_rotation_rad",
            "max_parcel_offset_mm",
            "min_temperature",
            "max_temperature",
        }
        leaked = fields & forbidden
        assert not leaked, f"Phase-2 leash fields leaked into Phase-1 config: {leaked}"


class TestV14Config:
    def test_top_level_assembly_uses_frozen_subconfigs(self) -> None:
        cfg = V14Config()
        # Width consistency across the shared interface (B-1 amendment).
        assert (
            cfg.tokenizer.d_model
            == cfg.grid_mixer.d_model
            == cfg.parcel_embedding.d_model
            == cfg.backbone.d_model
            == cfg.decoder.d_model
            == 64
        )
        # Parcel-embedding width must match the Tier-1 cardinality.
        from speech_decoding.atlas.tokens import DEFAULT_BASE_PARCELS
        assert cfg.parcel_embedding.n_parcels == len(DEFAULT_BASE_PARCELS)

    def test_no_local_summarizer_config_field(self) -> None:
        """The B-1 amendment retired LocalSummarizerConfig.

        It must not reappear on V14Config — the within-parcel Perceiver
        summarizer is replaced by GridMixer + SoftParcelEmbedding.
        """
        fields = set(V14Config.__dataclass_fields__.keys())
        assert "summarizer" not in fields
        assert "local_summarizer" not in fields

    def test_config_is_hashable_and_immutable(self) -> None:
        """Dataclasses are frozen, so accidental mutation during a run raises."""
        cfg = V14Config()
        try:
            cfg.tokenizer.d_model = 256  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("V14Config tokenizer.d_model mutated without raising")
