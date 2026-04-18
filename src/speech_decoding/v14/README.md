# v14 Package Layout (B-1 amendment, 2026-04-16 late)

This package is the clean implementation boundary for the current architecture.

The 2026-04-16 late B-1 amendment retired the within-parcel Perceiver
summarizer and the per-parcel pooled token tensor in favor of per-electrode
tokens with soft parcel embedding. See
`docs/v14_core_contract_amendment_2026-04-16.md` and
`docs/plans/v14-core.md` for the full architecture and implementation plan.

## Active modules

- `config.py`: dataclass configs (`AtlasConfig`, `PatientCalibrationConfig`,
  `TemporalTokenizerConfig`, `GridMixerConfig`, `SoftParcelEmbeddingConfig`,
  `BackboneConfig`, `DecoderConfig`, `V14Config`).
- `token_spec.py`: `DEFAULT_BASE_PARCELS` (15 LH Brainnetome parcels,
  argmax_wins ≥ 10 rule). Now used as the embedding-lookup table for the
  soft parcel embedding (Stage 3), not as fixed token slots.
- `fsaverage_projection.py`: strict patient → fsaverage snap-to-pial.
- `fsaverage_atlas.py`: baked fsaverage atlas loader and support helpers.
- `coordinates.py`, `cvsavg_projection.py`: cvs_avg35 parity oracle only.
- `audit/`: `#34` phoneme-loading audit (closed 2026-04-16).

## Planned modules (Phase B+ of `docs/plans/v14-core.md`)

- `dataset.py`: v14-core loader + collator (amended `#13`).
- `tokenizer.py` (Stage 1): per-electrode Conv1d patch projection.
- `grid_mixer.py` (Stage 2): whole-grid Conv2d, k=3, residual, masked.
- `parcel_embedding.py` (Stage 3): `support @ P_emb` soft routing.
- `backbone.py` (Stage 4): combined spatiotemporal attention, B = 3 blocks.
- `decoder.py` (Stage 5): 3-query AR decoder.
- `model.py`: top-level assembly.
- `cv.py`, `eval.py`, `train.py`: CV splitter, slot-averaged PER, training loop.

## Deprecated / archival modules

- `local_summarizer.py`: DEPRECATED. The within-parcel Perceiver summarizer
  (`#26`) was replaced by `GridMixer` + `SoftParcelEmbedding`. Importing
  the module raises `ImportError`.
- `parcel_frames.py`: ARCHIVED. The intra-parcel `(u, v, z)` chart (`#10`)
  is no longer in the pipeline. Builder + cache are retained for
  visualization / parity-oracle work only; no v14-core code path loads
  `parcel_frames.npz`.

## Atlas defaults for Phase 1

- Real Brainnetome 4D probability map (atlas-side bake source):
  `/Users/bentang/Documents/Code/speech/data/atlas/BNA_PM_4D.nii.gz`
- Brainnetome label/MPM map for ROI indexing and sanity checks:
  `~/nilearn_data/bnatlas.nii.gz`
- Baked fsaverage atlas (active):
  `data/atlas/fsaverage_bake_fast2/`
- Per-electrode Tier-1 support cache (planned, Phase A1):
  `data/atlas/support_cache/<pt>_support_tier1.csv`

## Phase 1 rule

- Use the baked fsaverage atlas as the active membership source.
- Do **not** silently fall back to the old smoothed-MPM proxy.
- Use the MPM file only for ROI indexing / sanity checks.

The first implementation target is `uECoG` only. `sEEG`, SSL, and external
datasets should be added only after the end-to-end `uECoG` path is verified.
