# v14 Package Layout

This package is the implementation boundary for the current Phase-1 architecture. The B-1 architecture amendment (2026-04-16 late) retired the within-parcel Perceiver summarizer and per-parcel pooled tokens in favor of per-electrode tokens with soft parcel embedding; the subsequent per-phoneme rework (2026-04-17) replaced the trial-level slot-CE path with a per-phoneme MFA-aligned pipeline for Phase 1. See `docs/strategy/stage_1.md` for the current Stage-1 default architecture, frozen contract, and live scoreboard, and `CLAUDE.md` → Code Structure for the per-file breakdown. The shipped P1–P9 implementation record lives at `docs/archive/plans/v14-core-current_implemented_2026_04_17.md`.

## Active modules (per-phoneme Phase-1 path)

- `config.py`: dataclass configs. Per-phoneme stack uses `PerPhonemeConfig` (which composes `PoolConfig`, `PerCellTemporalConfig`, `BackboneConfig`, `D1DecoderConfig`). B-1-full stack still carries `AtlasConfig`, `PatientCalibrationConfig`, `TemporalTokenizerConfig`, `GridMixerConfig`, `SoftParcelEmbeddingConfig`, `V14Config` for the Phase-2+ per-electrode-token target.
- `token_spec.py`: `DEFAULT_BASE_PARCELS` (15 LH Brainnetome parcels). Used as the embedding-lookup table for the soft parcel embedding, not as fixed token slots.
- `phoneme_dataset.py`: `V14PhonemeDataset` — active loader (phoneme-level `.fif`, `[-0.15, 0.5)` s, grouped-by-token CV).
- `phoneme_model.py`: `NeuralFieldPerceiverPerPhoneme` — active model.
- `phoneme_decoder.py`: D1 minimum AR decoder (mean-pool memory + prev-phoneme embedding + Linear).
- `phoneme_run_fold.py`: per-phoneme fold runner (P1–P6).
- `pool.py`: masked-mean pool primitive with divisibility assertion.
- `backbone.py`: combined spatiotemporal attention (RoPE on temporal axis only).
- `train.py`, `eval.py`, `cv.py`: training loop + loss wrappers, per-phoneme + exhaustive 9³ AR eval, grouped-by-token CV splitter.
- `fsaverage_projection.py`: strict patient → fsaverage snap-to-pial.
- `fsaverage_atlas.py`: baked fsaverage atlas loader and support helpers.
- `support_cache.py`: per-electrode Tier-1 support cache I/O.
- `channel_map.py`: `#12` amp-to-physical bridge per patient.
- `audit/`: `#34` phoneme-loading audit (closed 2026-04-16).

## Parity / exploration modules

- `coordinates.py`, `cvsavg_projection.py`: cvs_avg35 parity oracle only (not on active path).
- `electrode_pool.py`: exploration tooling from the 2026-04-17 kernel ablation; not on the canonical path.
- `dataset.py`, `run_fold.py`, `model.py`, `tokenizer.py`, `decoder.py`, `grid_mixer.py`, `parcel_embedding.py`, `aggregate.py`: trial-level B-1-full path and Phase-2+ per-electrode-token stubs — kept but not on the per-phoneme path.

## Deprecated / archival modules

- `local_summarizer.py`: DEPRECATED (was the within-parcel Perceiver summarizer for `#26`); importing raises `ImportError`.
- `parcel_frames.py`: ARCHIVED. The intra-parcel `(u, v, z)` chart (`#10`) is no longer in the pipeline.

## Atlas defaults for Phase 1

- Real Brainnetome 4D probability map (atlas-side bake source): `data/atlas/BNA_PM_4D.nii.gz`.
- Baked fsaverage atlas (active): `data/atlas/fsaverage_bake_v2c/`.
- Per-electrode Tier-1 support cache: `data/atlas/support_cache_v2c_snap/<pt>_support_tier1.csv`.

The first implementation target is `uECoG` only. `sEEG`, SSL, and external datasets come after the Phase-1 path is stable.
