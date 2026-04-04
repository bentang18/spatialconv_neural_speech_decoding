# Cross-Patient Speech Decoding from Intra-Op uECOG

## Project

Ben Tang, Greg Cogan Lab, Duke. Collaborating with Zac Spalding.
Extending Spalding 2025 (PCA+CCA, SVM/Seq2Seq, 8 patients, 9 phonemes, 0.31 bal. acc.).

**Task**: Non-word repetition (52 CVC/VCV tokens, 3 phonemes each, e.g. /abe/; 9 phonemes). Intra-operative, left sensorimotor cortex, 128/256-ch uECOG arrays. ~1 min utterance/patient.

**Patients**: 11 unique PS patients (S18 excluded — no preprocessed; S36 excluded — duplicate of S32): S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62. 8 are Spalding's published set. 46–178 trials/pt, 63–201 sig channels.

## Current Direction: Neural Field Perceiver (v12)

**Design doc**: `docs/neural_field_perceiver_v12.tex`

Cross-patient architecture using MNI electrode coordinates instead of per-patient Conv2d:
- **Fourier PE** on MNI coords → spatial tokenization
- **Cross-attention** to fixed virtual electrodes (Brainnetome atlas) → shared spatial representation
- **Reconstruction loss** → spatial supervision (predict HGA from latent + position)
- **Per-phoneme MFA epochs** as input (tmin=-0.15, tmax=0.5) with mean-pool temporal readout

Implementation spec not yet written. See `docs/current_direction.md` for full context.

## Best Per-Patient Results (baseline for v12 to beat)

**Per-phoneme MFA + flat head + full recipe = PER 0.734 ± 0.007** (S14, grouped-by-token CV, 3-seed).

Optimal config from 5 DCC sweeps (2026-04-04):
```
Input: Per-phoneme MFA epochs (tmin=-0.15, tmax=0.5) — 3× more samples than full-trial
Spatial: Conv2d(1→8, k=3, pad=1) + AdaptiveAvgPool2d(4,8) → d=256
Temporal: Conv1d(256→32, stride=10) + BiGRU(32, 32, 2L, bidirectional)
Head: Flat Linear(64→9) — NOT articulatory (bottleneck hurts single-phoneme classification)
Readout: Global mean pool → single phoneme prediction (no learned attention needed)
Training: Focal CE (γ=2) + label smoothing (0.1) + mixup (α=0.2)
Eval: Weighted k-NN (k=10) + TTA (n=16)
```

Key sweep findings (see `docs/experiment_log.md` findings 86-101):
- Per-phoneme beats learned attention by 6pp in fair head-to-head (0.734 vs 0.797)
- Per-phoneme wins 8/11 patients, population mean +4.0pp over full-trial
- Flat head > articulatory for single-phoneme (0.734 vs 0.772)
- Padding not critical: tmin=-0.10 to -0.15 optimal, tmax=0.5

Previous baselines: LOPO best 0.750, per-patient full-trial 0.737, LOPO pilot 0.846.

## Key Files

### Active
- `docs/neural_field_perceiver_v12.tex` — Active design document
- `docs/current_direction.md` — Current priorities and what's archived
- `docs/dcc_setup.md` — Complete DCC documentation
- `docs/experiment_log.md` — Full experiment history (101 findings)
- `docs/research_synthesis.md` — 18-paper literature synthesis
- `docs/reading_list.md` — 10 essential papers

### Configs
- `configs/per_patient_ce_s10_pool48.yaml` — Per-patient config (CE, stride=10, pool(4,8))
- `configs/lopo_ce.yaml` — LOPO cross-patient config
- `configs/paths.yaml` — Machine-specific BIDS paths (gitignored)

### Scripts
- `scripts/train_per_patient.py` — Per-patient training CLI
- `scripts/train_lopo.py` — LOPO cross-patient training CLI
- `scripts/sweep_head_to_head.py` — Fair comparison: learned attn vs per-phoneme (full recipe)
- `scripts/sweep_multipatient.py` — Per-phoneme vs full-trial across all 11 patients
- `scripts/sweep_full_recipe.py` — Full recipe (mixup + k-NN + TTA) sweep
- `scripts/sweep_tmin_perpos.py` — Temporal windowing sweep
- `scripts/sweep_padding_grid.py` — tmin/tmax fine-tuning

### Archived
- `docs/archive/` — Old NFP versions, NCA-JEPA specs, LOPO plans, historical design docs
- `scripts/archive/` — SSL eval, diagnostics, 83 autoresearch experiments
- `configs/archive/` — Historical baselines, sweeps, pretrain configs
- `src/speech_decoding/pretraining/` — NCA-JEPA code (on disk, not actively used)

## Code Structure

```
src/speech_decoding/
├── data/
│   ├── phoneme_map.py      # 9 PS phonemes, PS2ARPA, articulatory matrix (9×15)
│   ├── grid.py             # Electrode TSV → grid shape + channel-to-grid mapping
│   ├── bids_dataset.py     # load_patient_data() + load_per_position_data()
│   ├── augmentation.py     # Time shift, amplitude scale, channel dropout, noise
│   └── collate.py          # Group samples by patient_id for multi-grid batching
├── models/
│   ├── spatial_conv.py     # Per-patient Conv2d read-in: (B,H,W,T)→(B,256,T)
│   ├── backbone.py         # LayerNorm → Conv1d(s=10) → BiGRU(32,32,2L)
│   ├── articulatory_head.py # 15-dim bottleneck → fixed A → 9 phonemes
│   ├── flat_head.py        # Linear(128,10) → log_softmax
│   └── assembler.py        # YAML config → model components
├── training/
│   ├── ctc_utils.py        # CTC loss, greedy decode, PER
│   ├── trainer.py          # Per-patient CV training
│   ├── lopo_trainer.py     # Stage 1: multi-patient SGD
│   ├── adaptor.py          # Stage 2: target adaptation
│   └── lopo.py             # LOPO orchestrator
└── evaluation/
    ├── metrics.py          # PER, balanced accuracy
    ├── grouped_cv.py       # Grouped-by-token CV splitter
    └── content_collapse.py # Collapse diagnostics
```

Run: `pytest tests/ -v -m "not slow"` (fast) or `pytest tests/ -v` (all, needs BIDS data)

## Data

### Loading
```python
# Per-phoneme MFA epochs (recommended — 3× more samples):
from speech_decoding.data.bids_dataset import load_per_position_data
ds = load_per_position_data("S14", bids_root, task="PhonemeSequence", n_phons=3, tmin=-0.15, tmax=0.5)
# ds[i] → (grid_data[H,W,T], label[list[int]], patient_id)  — 459 samples for S14

# Full-trial epochs (all 3 phonemes in one window):
from speech_decoding.data.bids_dataset import load_patient_data
ds = load_patient_data("S14", bids_root, task="PhonemeSequence", n_phons=3, tmin=0.0, tmax=1.0)
# ds[i] → (grid_data[H,W,T], ctc_label[list[int]], patient_id)  — 153 trials for S14
```

### Grid Layouts
| Channels | Grid | Dead positions | Patients |
|----------|------|----------------|----------|
| 128 | 8×16 | 0–1 | S14, S16, S22, S23, S26 |
| 256 | 12×22 | 8 (corners) | S32, S33, S39, S58, S62 |
| 256 | 8×32 | 0–1 | (Lexical patients only) |
| 256 | 8×34 | 16 | S57 |

Grid inferred from electrode TSVs, NOT channel count. TSVs have BOM (`\ufeff`). Dead positions zeroed in Conv2d input.

### .fif Path
`{bids_root}/derivatives/epoch(phonemeLevel)(CAR)/sub-{id}/epoch(band)(power)/sub-{id}_task-PhonemeSequence_desc-productionZscore_highgamma.fif`

PS labels: `{'a':1, 'ae':2, 'b':3, 'g':4, 'i':5, 'k':6, 'p':7, 'u':8, 'v':9}` — `phoneme_map.normalize_label()` handles conversion.

## Compute: Duke DCC Cluster

**Use DCC for all training.** See `docs/dcc_setup.md` for complete documentation.

- **SSH**: `ssh ht203@dcc-login.oit.duke.edu`
- **GPU**: 8× RTX 5000 Ada (32 GB) on `coganlab-gpu`
- **Python**: `/work/ht203/miniconda3/envs/speech/bin/python` (PyTorch 2.10.0+cu126; do NOT `conda activate`)
- **Repo**: `/work/ht203/repo/speech`
- **Data**: `/work/ht203/data/BIDS` — all 11 PS patients (.fif + electrode TSV)
- **Submit**: `sbatch scripts/<script>_dcc.sh` | Monitor: `squeue -u ht203`
- **CAUTION**: `/work/ht203` auto-purges after 75 days. Copy results to `/hpc/group/coganlab/ht203/`.

## Completed Exploration (summary — details in experiment_log.md)

- **LOPO** (55 experiments): Converged to PER 0.750-0.780 on S14. Measurement ceiling from fixed CV folds.
- **SSL / NCA-JEPA**: All methods near-chance. ~1 min utterance/patient is 30× below minimum for SSL.
- **Per-patient tuning**: CTC→CE (+7.8pp), pool(2,4)→pool(4,8), stride=10, H=32 sufficient.
- **Per-phoneme MFA sweep** (2026-04-04): Per-phoneme flat (0.734) beats learned attention (0.797) and full-trial (0.807). Generalizes 8/11 patients.

## Established Findings (from literature)

- **Field consensus**: per-patient input → shared backbone (GRU) → CTC/CE. Used by Willett, Metzger, Singh, Boccato, Levin, BIT.
- **Alignment**: uECOG arrays span 15-25mm in MNI with variable rotation. Per-channel scaling is inapplicable (Boccato's diagonal finding is Utah-specific). Per-patient Conv2d matches rigid-array physics.
- **Transfer**: Singh — freeze shared backbone, fine-tune per-patient layers. Levin — 30% source replay prevents forgetting.
- **SSL**: Advantage is cross-subject only (BIT Table 9). Minimum corpus ~30 min; we have ~1 min/pt.
- **Data regime**: Small — 46-178 trials/pt, ~121K shared params, 50-100× smaller than field.

## Preprocessing Pipeline (do not change)

Decimate 2kHz → CAR → impedance exclusion (log10>6) → 70-150Hz Gaussian filterbank (8 bands) → Hilbert envelope → sum → 200Hz → z-score → significant channel selection. Implemented in `coganlab/IEEG_Pipelines`.

## Conventions

- Explain every design choice with precedent and tradeoffs
- Batch work in smaller chunks to prevent context rot
- Keep markdown lean and information-dense
- Report per-patient results, not just population means
- Always use grouped-by-token CV (never stratified — token leakage inflates by ~10pp)
- All training on DCC, never local
