# Current Direction

Updated: 2026-04-04

This file is the active guardrail against stale implementation drift. Use it before treating older configs, plans, or code paths as the default direction.

## Active Priority: Neural Field Perceiver (v12)

The next phase is **implementing the Neural Field Perceiver** — a cross-patient speech decoding architecture that uses MNI electrode coordinates to handle inter-patient anatomical variability.

**Design doc**: `docs/neural_field_perceiver_v12.tex`

**One core idea**: Spatially-grounded electrode encoding with three coupled components:
1. **Spatial tokenization** — Fourier PE on MNI coordinates (3 frequency levels: 30mm, 15mm, 7.5mm wavelengths)
2. **Spatial aggregation** — Cross-attention from electrodes to fixed virtual electrodes (L=16, Brainnetome atlas centroids) with distance bias
3. **Spatial supervision** — Reconstruction loss (predict electrode HGA from latent + position + patient embedding)

**Temporal readout**: Learned attention queries per phoneme position (from existing `CEPositionHead`), NOT equal windows.

**Implementation spec**: Not yet written. v12 is the design document; a concrete implementation plan with file-by-file changes is the next step before coding.

## Completed Exploration (Archived)

### Supervised LOPO (55 experiments)
- Best PER: **0.750** on S14 grouped-by-token CV (multi-scale temporal + MaxPool read-in)
- Converged to 0.750-0.780 wall across all paradigms (supervised, SSL, domain adaptation, self-training)
- Root cause: measurement ceiling from fixed CV folds (30 val samples/fold) + ~120 target samples
- Archived to: `docs/archive/lopo_autoresearch/`, `scripts/archive/autoresearch_lopo/`

### SSL / NCA-JEPA (Phase 0 complete)
- ALL SSL methods near-chance under fair grouped CV: BYOL, JEPA, masked span, LeWM, VICReg
- ~1 min utterance/patient is 30× below minimum successful SSL corpus
- Archived to: `docs/archive/nca_jepa/`, `scripts/archive/ssl_exploration/`

### Per-Patient Baselines
- **Grouped-by-token CV**: PER 0.737 (full recipe: CE + focal + mixup + k-NN + TTA + articulatory head)
- **Stratified CV**: PER 0.700 (leaky, NOT comparable to grouped)
- These remain the baselines to beat

## Repository Structure

### Active Code
- `src/speech_decoding/` — Core library (data loading, models, training, evaluation)
- `scripts/train_per_patient.py` — Per-patient training CLI
- `scripts/train_lopo.py` — LOPO cross-patient training CLI
- `scripts/sweep_tmin_perpos.py` — Temporal windowing sweep (pending results)
- `configs/per_patient_ce_s10_pool48.yaml` — Current recommended supervised config
- `configs/lopo_ce.yaml` — LOPO config

### Active Docs
- `docs/neural_field_perceiver_v12.tex` — Active design document
- `docs/experiment_log.md` — Full experiment history
- `docs/training_log.md` — Per-patient tuning history
- `docs/dcc_setup.md` — Duke DCC cluster documentation
- `docs/reading_list.md` — Essential papers

### Archived
- `docs/archive/neural_field_perceiver_versions/` — v2-v11, beta (design doc evolution)
- `docs/archive/nca_jepa/` — SSL pretraining design, specs, plans
- `docs/archive/lopo_autoresearch/` — LOPO brainstorm and autoresearch plans
- `docs/archive/historical_supervised/` — Old supervised design docs, implementation plans
- `scripts/archive/ssl_exploration/` — SSL evaluation and pretraining scripts
- `scripts/archive/diagnostics/` — One-off diagnostic scripts
- `scripts/archive/autoresearch_lopo/` — 83 LOPO experiments

## Pending Experiments

1. **tmin sweep** (`sweep_tmin_perpos.py`): 6 conditions comparing temporal windowing on S14. Lost to Python buffering — re-run with `PYTHONUNBUFFERED=1`.
2. **Population-level LOPO**: Run best architecture on ALL patients, not just S14.

## Practical Rules

- The active design document is `neural_field_perceiver_v12.tex`. All older versions are archived.
- If a config or plan references CTC loss, `pool(2,4)`, or SSL pretraining as the default path, it is stale.
- `per_patient_ce_s10_pool48.yaml` is the current supervised starting point.
- Use DCC for all training. See `docs/dcc_setup.md` for complete setup.
- The `src/speech_decoding/pretraining/` module exists on disk but is not actively used.
