# Configs Guide

## Active

- `per_patient_ce_s10_pool48.yaml` — Current recommended per-patient config (CE, stride=10, pool(4,8))
- `lopo_ce.yaml` — LOPO cross-patient config
- `paths.yaml` — Machine-specific data paths (gitignored)

## Archived

- `archive/historical_baselines/` — Old CTC / LOPO baselines kept for reproducibility
- `archive/sweeps/` — One-off sweep configs and exploratory variants
- `archive/negative_pivots/` — Configs from directions that are not current defaults (phonological aux, regression)
- `archive/pretrain_*.yaml` — NCA-JEPA pretraining configs
