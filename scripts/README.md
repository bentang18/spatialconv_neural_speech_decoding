# Scripts Guide

## Active

- `train_per_patient.py` — Per-patient training CLI (all PS patients × seeds)
- `train_lopo.py` — LOPO cross-patient training CLI
- `sweep_tmin_perpos.py` — 6-condition temporal windowing sweep (tmin × head_type × per-phoneme)
- `sweep_tmin_dcc.sh` — SBATCH wrapper for DCC
- `analyze_real_data_stats.py` — Data statistics analysis
- `visualize_ecog.py` — ECoG visualization

## Archived

- `archive/ssl_exploration/` — SSL evaluation scripts, pretraining CLI, Phase 0 baselines
- `archive/diagnostics/` — One-off diagnostic scripts (epoch alignment, LOPO diagnosis, etc.)
- `archive/autoresearch_lopo/` — 83 LOPO experiments from autoresearch
- `archive/sweeps/` — Historical sweep shells
- `archive/regression_pivot/` — Speech-embedding regression pivot scripts
