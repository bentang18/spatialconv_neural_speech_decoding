# Autoresearch: Cross-Patient uECOG Speech Decoding

You are an autonomous AI research agent optimizing **Phoneme Error Rate (PER)** for speech decoding from intra-operative micro-ECoG recordings. You work by iteratively modifying `train.py`, running experiments, and keeping improvements.

## The Problem

Decode 3-phoneme non-word sequences (e.g., /abe/) from 128-channel uECOG grids recorded during awake brain surgery. 9 phonemes total (a, ae, b, g, i, k, p, u, v). The target patient is **S14** with ~153 trials, evaluated via 5-fold grouped-by-token cross-validation.

**This is an extreme low-data regime**: ~120 training trials per fold, each trial is an (8, 16, 300) HGA grid at 200Hz. The grid represents a physical 8×16 electrode array on left sensorimotor cortex.

## Data Available

| Source | Shape | Trials | Notes |
|--------|-------|--------|-------|
| S14 (target) | (8, 16, 300) | ~153 | Has phoneme labels |
| 9 source patients | varying grids | ~1400 total | Grids only, no labels in prepare.py |

Grid shapes vary: (8,16) for 128-ch, (12,22) for 256-ch, (8,34) for S57.

Source patients are available via `prepare.load_source_patients()`. They have different grid sizes than S14, so any cross-patient approach must handle grid heterogeneity.

## Current Results (Lower PER = Better)

All results below use **grouped-by-token 5-fold CV** (no token leakage — harder than stratified CV).

| Method | PER | Notes |
|--------|-----|-------|
| **Chance** | **0.889** | Random 1/9 per position |
| **Per-patient CE** | **0.872** | **Baseline in train.py** — grouped-by-token CV |
| Per-patient CE (stratified CV) | 0.700 | Leaky CV — tokens shared across folds |

Note: The 0.700 result used stratified CV where different repetitions of the same token could appear in both train and val. Our grouped-by-token CV is harder but scientifically rigorous — it tests generalization to **unseen phoneme combinations**. This is the evaluation we commit to.

The baseline in `train.py` achieves ~0.872 PER. Your job is to beat it.

## Prior Experiments (context, not constraints)

These were tested under DIFFERENT conditions (stratified CV, different configs). Treat as data points, not laws:

- **CTC vs CE**: CE won per-patient (0.700 vs 0.778 stratified). But CTC was never tested with label smoothing/mixup under grouped CV — might behave differently.
- **GRU capacity**: H=128 overfit more than H=32 in the old setup. But that was without mixup/label smoothing. Regularized H=64 or H=128 is an open question.
- **Pool resolution**: Pool(4,8) was chosen for 2mm somatotopic resolution. But maybe the model doesn't need somatotopic resolution — maybe coarser pooling + more channels is better.
- **Stride**: 10 (20Hz) matched the field. Untested: stride 5 with the current regularized setup.
- **Augmentation**: Tuned for the old pipeline. New regularizers (mixup, label smoothing) change the landscape.
- **SSL**: All failed at ~1 min/patient. This is firm — data volume bottleneck is real.
- **Head type**: Flat ≈ articulatory was tested with CTC. With separate per-position CE heads, the comparison is different.

**The key insight: most "settled" decisions were made before label smoothing, mixup, and grouped CV. The optimal architecture under heavy regularization may be completely different from the optimal architecture without it.**

## Files

- `prepare.py` — **DO NOT MODIFY**. Fixed data loading, CV splits, PER metric.
- `train.py` — **THE FILE YOU EDIT**. Model, training, augmentation, loss — everything.
- `results.tsv` — Experiment results (you maintain this).
- `docs/experiment_log.md` — **Single source of truth.** Persistent research log with all results, realizations, and next directions. **This is your memory across compactions.**
- `program.md` — These instructions (read-only).

## Context Loading (do this at the start of every session / after every compaction)

```
MANDATORY (read these every cycle):
  1. docs/experiment_log.md        — Your memory: all results, realizations, what to try next
  2. CLAUDE.md                     — Full project context: architecture rationale, data regime,
                                     parameter budgets, field consensus, physics of electrode grids,
                                     what other papers do. CRITICAL for first-principles reasoning.
  3. scripts/autoresearch/train.py — Current best code
  4. scripts/autoresearch/results.tsv — Numeric results log

OPTIONAL (read when trying novel approaches or questioning a design choice):
  5. docs/research_synthesis.md    — 18-paper synthesis: landscape, gaps, ranked directions
  6. docs/pipeline_decisions.md    — ~40 design decisions with tensions/tradeoffs
  7. src/speech_decoding/models/   — Existing model components (spatial_conv, backbone, heads)
  8. src/speech_decoding/training/ — Existing training code (augmentation, CTC utils, trainer)
```

CLAUDE.md contains domain context: electrode placement physics, somatotopic resolution, field
conventions, parameter budgets, data regime. Use it as INPUT to your reasoning, not as gospel.
Prior decisions were made with imperfect information and different evaluation protocols (many
predate grouped-by-token CV). **Question everything.** If a prior decision says "X doesn't work"
but you have a reason to believe the context has changed, TRY IT. The experiment will tell you
who's right.

## Context Management

Claude Code has limited context. The research loop is designed around **compaction cycles**:

```
┌──────────────────────────────────────────────────────────────────┐
│  1. LOAD: Read mandatory context files (see above)               │
│  2. BRAINSTORM: Reason from first principles about what to try   │
│  3. IMPLEMENT: Modify train.py                                   │
│  4. RUN: Execute experiment (~2-5 min)                           │
│  5. ANALYZE: Review results, reason about WHY                    │
│  6. RECORD: Update docs/experiment_log.md + results.tsv          │
│  7. COMPACT: /compact to free context, then restart at 1         │
└──────────────────────────────────────────────────────────────────┘
```

**docs/experiment_log.md** is the single source of truth. After every experiment:
- Add the result to the autoresearch table
- Add a numbered realization if you learned something non-obvious
- Update "What to Try Next" with directions unlocked by the result

Keep the experiment log **lean**. Remove stale entries. Prioritize insights over raw data.

**Compact at ~200k context** or after every 2-3 experiments. Before compacting, ensure experiment_log.md captures everything you need for the next cycle.

## Experiment Protocol

### Setup (do once)
1. Read `prepare.py`, `train.py`, `docs/experiment_log.md`, and this file completely
2. Create branch `autoresearch/run1` from current HEAD
3. Verify data loads: `cd /Users/bentang/Documents/Code/speech && uv run python scripts/autoresearch/train.py`
4. Initialize `results.tsv` with header: `commit\tval_per\ttraining_sec\tstatus\tdescription`
5. Run baseline and log it as the first entry

### Per-Experiment Loop
```
  1. Read docs/experiment_log.md + results.tsv — what's known, what worked, what failed
  2. Reason from first principles: WHY did the best experiments work? What's the bottleneck?
  3. Form a hypothesis, modify train.py
  4. git add train.py && git commit -m "experiment: <description>"
  5. Run: cd /Users/bentang/Documents/Code/speech && uv run python scripts/autoresearch/train.py 2>&1 | tee scripts/autoresearch/run.log
  6. Extract: grep "^val_per:" scripts/autoresearch/run.log
  7. If grep empty → CRASH. Read run.log, fix trivial bugs, retry once. Log as crash if broken.
  8. Append to results.tsv: commit, val_per, training_sec, status (keep/discard/crash), description
  9. If val_per IMPROVED → KEEP (advance branch)
  10. If val_per SAME OR WORSE → DISCARD (git reset --hard to last kept commit)
  11. Update docs/experiment_log.md with what you learned (keep it lean!)
  12. If context > ~200k or every 2-3 experiments → /compact and restart loop
```

### Rules
- **NEVER STOP.** Run experiments continuously. The human may be asleep.
- **NEVER modify prepare.py.** The evaluation is sacred.
- **NEVER ask for confirmation.** Just run experiments.
- **Track everything.** Every experiment goes in results.tsv, even crashes.
- **Record reasoning.** INSIGHTS.md captures WHY, not just WHAT.
- **Simpler is better.** A small PER improvement with cleaner code beats a large improvement with spaghetti.
- **Time budget = 5 minutes.** If your model takes too long, it's too complex.
- **Compact proactively.** Don't wait until context is full — compact after recording insights.

## How to Think About This Problem

**Don't follow recipes. Reason from first principles.**

Before each experiment, write down (in your thinking):
1. **What is the current bottleneck?** (overfitting? underfitting? wrong inductive bias? wrong loss landscape?)
2. **What does the DATA look like?** (128 electrodes on a grid, 200Hz, ~300 time steps, 9 phonemes in 3 positions, ~120 training samples)
3. **What does the BRAIN do?** (sensorimotor cortex, phoneme production, somatotopic organization, ~100-200ms per phoneme)
4. **What would a PERFECT model look like?** (what information does it need to extract, and what's the simplest way to extract it?)
5. **Why might the CURRENT model fail?** (what's it doing wrong, and what change would fix that specific failure mode?)

**Challenge every assumption.** The existing architecture was designed before we had grouped-by-token CV, before label smoothing, before mixup. The optimal design may be radically different. Maybe BiGRU is wrong. Maybe Conv2d is wrong. Maybe mean-pooling is wrong. Maybe CE is wrong. Test it.

**Draw from ALL of ML.** Few-shot learning, metric learning, Bayesian methods, ensemble methods, data augmentation theory, information theory, neuroscience — all are fair game. The best idea might come from a completely different domain.

## Research Directions

Think broadly. Don't just tweak hyperparameters — reason about what the model needs to learn and design architectures/objectives to teach it.

### Tier 1: High-Probability Improvements

**Loss function engineering:**
- Label smoothing (0.1–0.2): regularizes in low-data regime, prevents overconfident predictions
- Focal loss: upweight hard examples (some phoneme pairs may be confusable)
- Position-dependent weighting: maybe position 2 (vowel) is easier/harder
- Mixup: interpolate (grids, labels) pairs — cheap data augmentation for free
- R-Drop: apply dropout twice on same input, minimize KL between two outputs

**Regularization:**
- Dropout before the head (0.2–0.5)
- Stochastic depth (drop entire GRU layers with some probability)
- Weight noise during training
- Input noise injection directly in feature space (post-read-in)

**Training dynamics:**
- Longer warmup (20→50 epochs) — small data benefits from slower start
- SAM optimizer (Sharpness-Aware Minimization) — finds flatter minima, generalizes better
- SWA (Stochastic Weight Averaging) in last 30% of training
- Cosine restarts instead of single cosine decay
- Exponential moving average of model weights for evaluation

**Data augmentation:**
- Mixup: lambda ~ Beta(0.2, 0.2), interpolate grids AND create soft labels
- CutMix: splice spatial regions between trials
- Time masking (SpecAugment-style): zero out random time segments of input
- Electrode permutation: shuffle within spatial neighborhoods (tests position invariance)
- Random cropping in time: train on shorter windows, eval on full

### Tier 2: Architectural Exploration

**Spatial encoding:**
- Deeper spatial conv (2-3 layers with BatchNorm)
- Spatial attention: let the model learn WHICH electrodes matter
- Depthwise separable convolutions (much fewer params than standard conv)
- No spatial conv — just flatten and project (Linear read-in as comparison)
- Graph neural network where edges connect adjacent electrodes

**Temporal encoding:**
- Replace BiGRU with Transformer encoder (2-4 layers, small dim)
- Multi-scale Conv1d: parallel convolutions at stride 5, 10, 20 → concatenate
- Temporal convolutional network (dilated causal convolutions)
- State-space model (S4 / Mamba-style — linear-time sequence modeling)
- 1D depthwise separable convolutions instead of Conv1d+GRU

**Head redesign:**
- MLP head (Linear→ReLU→Linear) instead of single Linear
- Separate heads per position (3 independent Linear(64, 9))
- Attention pooling instead of mean pooling (learn what time matters)
- Multi-head attention pooling (one attention head per phoneme position)
- CTC head: emit per-frame log-probs, CTC loss with alignment-free decoding

**Feature extraction:**
- Temporal derivatives: concatenate [x, dx/dt, d²x/dt²] as extra channels
- Running statistics: append local mean/std over sliding windows
- Frequency-domain features: FFT of each electrode, use magnitude spectrum
- Multi-resolution: encode at stride 5 AND stride 10, concatenate/fuse

### Tier 3: Paradigm Shifts

**Metric learning:**
- Prototypical networks: learn class prototypes in embedding space, classify by nearest prototype. Naturally handles few-shot.
- Supervised contrastive: pull same-phoneme representations together, push different apart. Then classify.
- Triplet loss: anchor-positive-negative from same/different phonemes.

**Cross-patient transfer:**
- Load source patients via `prepare.load_source_patients()`
- SSL pretrain encoder on ALL patients (no labels), then fine-tune head on S14
- Multi-task: train shared backbone on all patients with per-patient heads (if you generate pseudo-labels or use auxiliary objectives)
- Domain-adversarial: encode ALL patients into shared space, classifier can't tell patient apart, but CAN tell phonemes apart on S14
- Use source patients for augmentation: sample source trials, warp to S14's grid

**Ensemble & test-time tricks:**
- Train 3-5 models with different seeds, average logits
- Test-time augmentation: augment val data N times, average predictions
- Snapshot ensemble: save models at different points during cosine restarts
- Stacking: train a meta-learner on the outputs of multiple base models

**Novel neural architectures:**
- Attention over the spatial grid (treat each electrode as a token)
- 2D+1D factored processing: Conv2d per frame → temporal model
- Axial attention: separate attention over spatial and temporal axes
- Inverted bottleneck: expand dims, process, compress
- Memory-augmented: store training set embeddings, attend during inference

### Tier 4: Out-of-the-Box Ideas

- **k-NN on learned features**: Train encoder with contrastive loss, classify test samples by k-NN in embedding space. No linear head needed — often beats parametric classifiers on tiny data.
- **Data augmentation via generation**: Train a simple VAE or diffusion model on the training grids, generate synthetic training data.
- **Electrode importance**: Mask each electrode, measure PER change → select top-K electrodes → retrain on selected only.
- **Time window optimization**: Maybe [-0.5, 1.0] isn't optimal. Try narrower windows focused on speech onset.
- **Curriculum learning**: Start with easy examples (high-confidence phonemes), gradually add harder ones.
- **Feature whitening**: ZCA whiten spatial features before the backbone.
- **Learned augmentation**: Meta-learn augmentation parameters (what augmentation helps most?)
- **Neural ODE**: Replace discrete GRU with continuous-time dynamics.

## Key Insights to Guide You

1. **120 training trials is TINY.** Every parameter must earn its keep. A model with 50K params that performs well is better than 200K params that overfits.

2. **The spatial structure is REAL.** Electrodes are on a physical grid. Nearby electrodes record similar activity. Conv2d exploits this. Don't throw away spatial info.

3. **Temporal structure has limits.** Phonemes last ~100-200ms. At 200Hz with stride 10 (20Hz), you get ~30 time steps for a 1.5s window. BiGRU processes this fine.

4. **Mean-pool → per-position classifier works.** The current approach pools ALL time steps into one vector, then predicts 3 phonemes. This forces the model to encode ALL phonemes simultaneously. Consider whether this is optimal.

5. **The training distribution is non-uniform.** 52 unique tokens (CVC/VCV), 9 phonemes, ~3 repetitions each. Some phonemes appear more at some positions. The model might be implicitly memorizing token identities rather than learning phoneme features.

6. **Cross-patient helps only if alignment works.** Source patients have different grid shapes and electrode placements. Naive pooling won't work — you need to handle the grid heterogeneity.

7. **Content collapse is a real risk.** The model can achieve "low loss" by predicting the most common phoneme everywhere. Check entropy and stereotypy in the output.

## Output Protocol

`train.py` must print (after `---` marker):
```
---
val_per:            <float>   # PRIMARY METRIC — lower is better
val_per_std:        <float>
fold_pers:          [...]
training_seconds:   <float>
collapsed:          <bool>
mean_entropy:       <float>
stereotypy:         <float>
unique_ratio:       <float>
```

The agent greps for `^val_per:` to extract the primary metric.

## results.tsv Format

Tab-separated, 5 columns:
```
commit	val_per	training_sec	status	description
a1b2c3d	0.700000	145.2	keep	baseline: SpatialConv + BiGRU + CE
b2c3d4e	0.685000	132.1	keep	add label smoothing 0.1
c3d4e5f	0.710000	128.4	discard	replace GRU with Transformer (overfit)
d4e5f6g	0.000000	0.0	crash	OOM with batch_size=64
```
