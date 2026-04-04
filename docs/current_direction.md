# Current Direction

Updated: 2026-04-04 (post-sweep)

## Active Priority: Neural Field Perceiver (v12)

**Design doc**: `docs/neural_field_perceiver_v12.tex`

Cross-patient architecture using MNI electrode coordinates:
1. **Spatial tokenization** — Fourier PE on MNI coordinates (3 frequency levels)
2. **Spatial aggregation** — Cross-attention to fixed virtual electrodes (Brainnetome atlas)
3. **Spatial supervision** — Reconstruction loss (predict HGA from latent + position)
4. **Per-phoneme MFA epochs** — mean-pool temporal readout (no learned attention needed)

Implementation spec not yet written — next step before coding.

## Per-Patient Baseline (v12 must beat this)

**PER 0.734 ± 0.007** (S14, grouped-by-token CV, 3-seed). Per-phoneme MFA flat head + full recipe.
**Population: 0.825 mean** across 11 patients (simplified recipe).

Optimal config established from 5 DCC sweeps:
- Per-phoneme MFA epochs (tmin=-0.15, tmax=0.5) with global mean pool
- Flat Linear(64→9) head — not articulatory
- Focal CE + mixup + weighted k-NN + TTA
- See `experiment_log.md` findings 86-101 for full results

## Completed Exploration

- **LOPO** (55 experiments): PER 0.750 wall. Measurement ceiling, not model ceiling.
- **SSL / NCA-JEPA**: All methods near-chance. Data too small for SSL.
- **Temporal readout sweep**: Per-phoneme > learned attention > mean pool > equal windows.
- **Multi-patient validation**: Per-phoneme wins 8/11 patients (+4.0pp population mean).

## Practical Rules

- Active design = `neural_field_perceiver_v12.tex`. Old versions archived.
- Per-phoneme MFA is the default input mode. Full-trial only as fallback for bad MFA patients.
- Always grouped-by-token CV. Never stratified (10pp inflation from token leakage).
- All training on DCC. See `docs/dcc_setup.md`.
- If a doc references CTC, `pool(2,4)`, SSL, or learned temporal attention as default — it's stale.
