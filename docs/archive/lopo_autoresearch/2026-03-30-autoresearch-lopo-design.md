# Autoresearch LOPO: Cross-Patient Speech Decoding Optimization

## Goal

Automated optimization of cross-patient (LOPO) speech decoding from intra-operative uECOG. An AI agent iteratively modifies a self-contained training script, running full LOPO experiments (train on 9 source patients → adapt+eval on target S14) to minimize Phoneme Error Rate.

## Context

### Current Results

| Method | PER | Notes |
|---|---|---|
| Autoresearch single-patient (best) | 0.737 | CE + full recipe, S14 only, ~120 trials |
| LOPO pilot (baseline) | 0.846 | CTC, no recipe, 8 patients × 1 seed |
| Supervised CE single-patient | 0.837 | Grouped CV, no recipe |
| All SSL methods | 0.85-0.89 | Dead end at this data scale |
| Chance | 0.889 | Random 1/9 per position |

The LOPO pilot (0.846) used basic CTC with zero recipe improvements. Single-patient autoresearch improved from 0.872→0.737 (13.5pp) through recipe engineering. Applying the recipe to LOPO is the highest-leverage experiment: more data (~1300 trials from 9 source patients) + proven recipe.

### What Worked (Single-Patient, ~120 trials)

| Improvement | Gain | Cumulative PER |
|---|---|---|
| Baseline (CE, grouped CV) | — | 0.872 |
| + Label smoothing 0.1 | -4.1pp | 0.831 |
| + Per-position heads + dropout 0.3 | -1.2pp | 0.819 |
| + Mixup α=0.2 | -1.7pp | 0.802 |
| + k-NN eval (best-of per fold) | -2.1pp | 0.781 |
| + TTA 8 copies | -1.4pp | 0.767 |
| + Articulatory head | -0.4pp | 0.763 |
| + Dual head ensemble | -0.7pp | 0.756 |
| + Focal loss γ=2 | -0.1pp | 0.755 |
| + Weighted k-NN + TTA 16 | -1.8pp | 0.737 |

### What Failed (Single-Patient)

- SupCon / contrastive auxiliary losses: all hurt CE optimization
- EMA model averaging: slower, no improvement
- Attention pooling: too few timesteps (~20 frames) for learned queries
- H=64 or H=128: overfits more than H=32 at N=120
- All SSL methods (JEPA, DINO, BYOL, VICReg, LeWM): near-chance (~0.86-0.89)
- LP-FT on JEPA: 0.797, barely improved over frozen features
- CTC loss: wasted capacity on alignment with small data

### What's Untested in Cross-Patient

- CE loss for LOPO (only CTC tested)
- Any recipe improvements on LOPO
- Whether backbone freezing helps or hurts in Stage 2
- Source replay ratio during adaptation
- Domain adversarial / gradient reversal for patient alignment
- Patient-conditional encoding (embeddings, FiLM)
- Different Stage 1/Stage 2 step allocation
- Multi-patient k-NN (source embeddings as extra neighbors)
- Progressive unfreezing during Stage 2

## Architecture

### Directory Structure

```
scripts/autoresearch_lopo/
├── program.md      # Agent instructions (read-only)
├── prepare.py      # Data loading, CV splits, PER metric (DO NOT MODIFY)
├── train.py        # Agent modifies: model, training, evaluation
├── results.tsv     # Experiment results log
└── run.log         # Latest run output
```

### `prepare.py` — Fixed Evaluation Harness

Provides:

```python
# Data — NEW functions (not reusing existing prepare.py)
load_target_data()      → (grids, labels, token_ids)
                        # S14: (153, 8, 16, 201) float32, tmin=0.0 tmax=1.0
load_all_patients()     → {pid: {"grids": tensor, "labels": list, "grid_shape": (H,W)}}
                        # 10 patients total: S14 (target) + 9 source
                        # Excludes S26 (dev patient) and S36 (duplicate of S32)
                        # Labels included for ALL patients — source labels needed for Stage 1

# Evaluation
create_cv_splits(token_ids) → [(train_idx, val_idx), ...]  # Grouped-by-token 5-fold
compute_per(preds, refs)    → float                         # Edit-distance PER
compute_content_collapse(preds) → dict                      # Entropy, stereotypy

# Constants
TARGET_PATIENT = "S14"
SOURCE_PATIENTS = ["S16", "S22", "S23", "S32", "S33", "S39", "S57", "S58", "S62"]  # 9 source
PATIENT_GRIDS = {
    "S14": (8, 16), "S16": (8, 16), "S22": (8, 16), "S23": (8, 16),
    "S32": (12, 22), "S33": (12, 22), "S39": (12, 22),
    "S57": (8, 34),
    "S58": (12, 22), "S62": (12, 22),
}
N_POSITIONS = 3
N_CLASSES = 9
TIME_BUDGET = 900  # 15 minutes (up from 5 min in single-patient — LOPO needs multi-patient Stage 1)
```

This is a **new** `prepare.py` in `scripts/autoresearch_lopo/`, not the existing `scripts/autoresearch/prepare.py`. Key differences from the original:
- `load_all_patients()` returns labels for all patients (original only returned grids for source patients)
- **tmin=0.0** (production only, 201 frames) — 2.2pp better than tmin=-0.5 under grouped CV
- Uses a separate cache directory (`.cache/autoresearch_lopo/`) to avoid stale cache conflicts
- TIME_BUDGET increased to 900s for multi-patient training

Source patients: 9 total (~1300 trials). S32 is included; S36 is excluded as its duplicate. S26 excluded as dev patient.

### `train.py` — Pre-baked Baseline

Starting point with all known wins applied:

**Architecture:**
- Per-patient `SpatialReadIn`: Conv2d(1,8,k=3,pad=1) + ReLU + AdaptiveAvgPool2d(4,8) → d=256
- Shared `Backbone`: LN → feat_dropout(0.3) → Conv1d(256,32,stride=10) → GELU → time_mask(2-5) → BiGRU(32,32,2L,dropout=0.3) → out_dim=64
- Per-position CE head: mean-pool → dropout(0.3) → 3× Linear(64, 9)

**Stage 1** (train on 9 source patients, ~1300 trials):
- Multi-patient SGD: sample patient → sample batch → forward through patient's read-in + shared backbone + head
- CE loss with label smoothing 0.1, focal γ=2, mixup α=0.2
- AdamW, lr=1e-3, readin_lr_mult=3.0, weight_decay=1e-4
- Cosine LR with warmup (20 epochs)
- 20% held-out source validation (stratified by patient — each patient contributes 20% to val), early stopping (patience 7)
- Full augmentation: time_shift(30), temporal_stretch(0.15), amp_scale(0.3), channel_dropout(0.2), noise(0.05)

**Stage 2** (adapt to S14, per CV fold):
- Initialize target read-in fresh (S14's grid shape)
- Train full model (backbone NOT frozen — agent can change this)
- Differential LR: backbone lr × 0.1, read-in + head at full lr
- Same loss recipe as Stage 1
- Early stopping on val fold loss

**Evaluation** (per fold):
- Linear head predictions with TTA (16 augmented copies, average logits)
- Weighted k-NN (k=10) on backbone embeddings with TTA on val embeddings
- Best-of (linear vs k-NN) per fold
- Content collapse diagnostics

**The agent owns everything in `train.py`.** It can change architecture, merge stages, add gradient reversal, replace BiGRU, redesign the head, etc.

### `program.md` — Agent Instructions

Follows the original autoresearch pattern with additions:

1. **Context loading**: experiment_log.md, CLAUDE.md, train.py, results.tsv
2. **Per-experiment loop**: reason → modify → commit → run → analyze → record → compact
3. **"What We Already Know" section**: full table of single-patient results + failures
4. **Cross-patient research directions**: source replay, gradient reversal, patient embeddings, progressive unfreezing, multi-patient k-NN
5. **Rules**: never stop, never modify prepare.py, track everything, compact proactively

Time budget: **15 minutes** per experiment.

## Constraints

- **prepare.py is sacred**: data loading, CV splits, PER metric cannot be modified
- **Evaluation**: grouped-by-token 5-fold CV on S14, PER as primary metric
- **Device**: MPS (Apple Silicon)
- **Seeds**: 1 seed (42) during search. Winning recipe validated with 3 seeds on all 8 patients afterward.
- **Data**: tmin=0.0, tmax=1.0 (production only, 201 frames at 200Hz). Agent can explore tmin=-0.5 as an experiment.

## Success Criteria

1. **Primary**: Beat LOPO pilot PER of 0.846 on S14 (the pre-baked baseline should already do this)
2. **Stretch**: Beat single-patient autoresearch PER of 0.737 (would prove cross-patient data helps)
3. **Validation**: Winning recipe generalizes to all 8 patients (run after search completes)
