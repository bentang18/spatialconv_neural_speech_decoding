# Current Direction

Updated: 2026-04-05 (post-pipeline-audit + data discovery)

## Active Priority: Neural Field Perceiver (v12)

**Design doc**: `docs/neural_field_perceiver_v12.tex`

Cross-patient architecture using ACPC electrode coordinates:
1. **Spatial tokenization** — Fourier PE on ACPC coordinates (3 frequency levels)
2. **Spatial aggregation** — Cross-attention to fixed virtual electrodes (Brainnetome atlas speech ROIs)
3. **Spatial supervision** — Reconstruction loss (predict HGA from latent + position)
4. **Per-phoneme MFA epochs** — mean-pool temporal readout (no learned attention needed)

Implementation spec not yet written — next step before coding.

## Immediate Next Steps

1. **Continuous HGA extraction** — Raw EDF files have 456 min across 29 patients (7.6h). Need HGA pipeline (CAR → filterbank → Hilbert → 200Hz). Ask Zac or run Python `ieeg.timefreq.gamma.extract()`. Biggest unlock for SSL.
2. **Brainnetome atlas virtual electrodes** — Initialize cross-attention targets at speech motor cortex ROIs. Debug: compute overlap with actual uECOG electrode positions.
3. **v12 implementation spec** — Architecture details, training protocol, evaluation plan.

## Data Readiness (audited 2026-04-05)

**Electrode coordinates**: ACPC space (not MNI), 11/11 patients. Implemented in `coordinates.py`: `build_electrode_coordinates()` handles both 128-ch (chanMap) and 256-ch (direct) paths.

**Artifact exclusion**: Integrated in `bids_dataset.py` via `exclude_artifacts=True`. Zeroes chronic artifact channels (>10 std in >5% trials). S39=20, S57=15, S58=37 channels excluded.

**Channel maps**: 11/11 patients. Without chanMap: 8.5mm error for 128-ch patients.

**Raw continuous data**: 456 min across 29 unique patients (13 PS + 17 Lexical, zero overlap). EDF format at 2kHz. Need HGA extraction to match existing productionZscore features.

**Hemisphere**: S22 and S58 are right hemisphere. `mirror_to_left()` implemented.

**MFA quality**: Good for 8/11 PS patients. Poor for S16, S22, S39 (300-500ms offsets). Start v12 with 8 good-MFA patients.

**Blocking for Zac**: S32/S57 sigChannel (non-blocking for v12), continuous HGA extraction.

## Per-Patient Baseline (v12 must beat this)

**PER 0.734 ± 0.007** (S14, grouped-by-token CV, 3-seed). Per-phoneme MFA flat head + full recipe.
**Population: 0.825 mean** across 11 patients (simplified recipe).

Per-patient Conv2d pipeline is complete and fully audited. No further optimization needed.

## Completed Exploration

- **LOPO** (55 experiments): PER 0.750 wall. Measurement ceiling, not model ceiling.
- **SSL / NCA-JEPA**: All methods near-chance on ~11 min epoched data. 456 min raw continuous available — changes the SSL picture entirely.
- **Per-phoneme MFA**: Wins 8/11 patients (+4.0pp). Note: 85% temporal overlap between consecutive phoneme epochs — "3× samples" is misleading.
- **Per-patient Conv2d**: Fully audited, artifact exclusion integrated. Done.

## Practical Rules

- Active design = `neural_field_perceiver_v12.tex`. Old versions archived.
- Per-phoneme MFA is the default input mode. Use 8 good-MFA patients initially; add S16/S22/S39 later with full-trial fallback.
- Always `exclude_artifacts=True` when loading data.
- Always grouped-by-token CV. Never stratified (10pp inflation from token leakage).
- All training on DCC. See `docs/dcc_setup.md`.
- If a doc references CTC, `pool(2,4)`, SSL as failed, or learned temporal attention as default — it's stale.
