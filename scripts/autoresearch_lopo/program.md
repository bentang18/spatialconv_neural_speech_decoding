# Autoresearch: LOPO Cross-Patient uECOG Speech Decoding

You are an autonomous AI research agent optimizing **Phoneme Error Rate (PER)** for cross-patient speech decoding from intra-operative micro-ECoG recordings. You work by iteratively modifying `train.py`, running experiments, and keeping improvements.

## The Problem

Decode 3-phoneme non-word sequences (e.g., /abe/) from uECOG grids recorded during awake brain surgery. 9 phonemes total (a, ae, b, g, i, k, p, u, v). The setting is **Leave-One-Patient-Out (LOPO)**: train a shared backbone on 9 source patients (Stage 1), then adapt to target patient **S14** (Stage 2), evaluated via 5-fold grouped-by-token cross-validation on S14.

**This is a low-data regime**: S14 has ~153 trials, each an (8, 16, 201) HGA grid at 200Hz. But Stage 1 has ~1300 trials across source patients — 10× more data than single-patient. This changes what architectures and regularizers are optimal.

## Data Available

| Patient | Grid Shape | Approx Trials | Role |
|---------|-----------|---------------|------|
| S14 | (8, 16) | ~153 | **Target** — adapt + evaluate here |
| S16 | (8, 16) | ~148 | Source |
| S22 | (8, 16) | ~144 | Source |
| S23 | (8, 16) | ~151 | Source |
| S32 | (12, 22) | ~151 | Source |
| S33 | (12, 22) | ~46 | Source (fewest trials) |
| S39 | (12, 22) | ~137 | Source |
| S57 | (8, 34) | ~141 | Source (unique layout) |
| S58 | (12, 22) | ~178 | Source |
| S62 | (12, 22) | ~178 | Source |

Grid shapes vary: (8,16) for 128-ch, (12,22) for 256-ch, (8,34) for S57. Each patient needs its own spatial read-in layer to project their grid to a shared representation. S26 and S36 are excluded (S26 is dev hold-out; S36 duplicates S32).

## Current Results (Lower PER = Better)

All results use **grouped-by-token 5-fold CV** (no token leakage — harder than stratified CV).

| Method | PER | Notes |
|--------|-----|-------|
| **Chance** | **0.889** | Random 1/9 per position |
| **All SSL methods** | **0.85–0.89** | Dead end — data volume too small |
| **LOPO pilot (baseline)** | **0.846** | CTC, no recipe, 8 patients × 1 seed |
| Supervised CE single-patient | 0.837 | Grouped CV, no recipe |
| **Autoresearch single-patient (best)** | **0.737** | CE + full recipe, S14 only, ~120 trials |

The LOPO pilot (0.846) is **worse** than the single-patient baseline (0.837) and far worse than the single-patient best (0.737). Your job is to beat 0.837 with cross-patient training, then push toward 0.737 and below.

Note: The 0.700 result sometimes cited used stratified CV (leaky — tokens shared across folds). Our grouped-by-token CV is harder but scientifically rigorous. This is the evaluation we commit to.

## What We Already Know

### Single-Patient Autoresearch Progression (completed)

These are improvements found on S14 alone with ~120 training trials per fold. Understand WHAT and WHY — many of these carry over to LOPO.

| Improvement | Gain | Cumulative PER |
|---|---|---|
| Baseline (CE, grouped CV, mean-pool head) | — | 0.872 |
| + Label smoothing 0.1 | −4.1 pp | 0.831 |
| + Per-position heads + dropout 0.3 | −1.2 pp | 0.819 |
| + Mixup α=0.2 | −1.7 pp | 0.802 |
| + k-NN eval (best-of per fold) | −2.1 pp | 0.781 |
| + TTA 8 copies | −1.4 pp | 0.767 |
| + Articulatory head | −0.4 pp | 0.763 |
| + Dual head ensemble (linear + articulatory) | −0.7 pp | 0.756 |
| + Focal loss γ=2 | −0.1 pp | 0.755 |
| + Weighted k-NN + TTA 16 | −1.8 pp | 0.737 |

### What Failed in Single-Patient

- **SupCon / contrastive auxiliary losses**: all hurt CE optimization — loss conflict
- **EMA model averaging**: slower convergence, no PER improvement
- **Attention pooling**: too few timesteps (~20 frames) for learned queries to find signal
- **H=64 or H=128 GRU**: overfit more than H=32 at N=120 training trials
- **All SSL methods** (JEPA, DINO, BYOL, VICReg, LeWM): near-chance (~0.86–0.89) — data volume bottleneck is real, do not retry SSL
- **CTC loss**: wasted capacity on alignment with small data

### LOPO-Specific Context

- **H=32 was optimal at N=120, but LOPO Stage 1 has N≈1300** — larger backbone (H=64 or H=128) may now help without overfitting
- **CTC was tested for LOPO (0.846) but CE was never tested cross-patient** — CE is the first thing to test
- **Source replay during Stage 2** (Levin 2026): 30% source data replay prevents catastrophic forgetting during target adaptation. Critical for LOPO.
- **All SSL failed** — don't waste experiments on SSL at this data scale. This is firm.
- **Contrastive losses fought CE in single-patient** — may behave differently with multi-patient data and shared backbone, but approach cautiously
- **Stage 1 early-stopped at ~step 800 of 2000** in the LOPO pilot — the current architecture (CTC, no recipe) diverges. A better objective + recipe may allow longer Stage 1 training
- **Stage 2 folds completed in seconds** (3-5s, early-stop 13-46 steps) in the pilot — Stage 2 is currently doing almost nothing. Fixing this is likely the highest-leverage change.

## Files

- `prepare.py` — **DO NOT MODIFY**. Fixed data loading, CV splits, PER metric.
- `train.py` — **THE FILE YOU EDIT**. Model, training, augmentation, loss — everything.
- `results.tsv` — Experiment results (you maintain this).
- `docs/experiment_log.md` — **Single source of truth.** Persistent research log with all results, realizations, and next directions. **This is your memory across compactions.**
- `program.md` — These instructions (read-only).

```
scripts/autoresearch_lopo/
├── program.md      # THIS FILE (read-only)
├── prepare.py      # Data loading, CV splits, PER metric (DO NOT MODIFY)
├── train.py        # Agent modifies: model, training, evaluation
├── results.tsv     # Experiment results log
└── run.log         # Latest run output
```

## Context Loading (do this at the start of every session / after every compaction)

```
MANDATORY (read these every cycle):
  1. docs/experiment_log.md              — Your memory: all results, realizations, what to try next
  2. CLAUDE.md                           — Full project context: architecture rationale, data regime,
                                           parameter budgets, field consensus, physics of electrode
                                           grids, what other papers do. CRITICAL for reasoning.
  3. scripts/autoresearch_lopo/train.py  — Current best code
  4. scripts/autoresearch_lopo/results.tsv — Numeric results log

OPTIONAL (read when trying novel approaches or questioning a design choice):
  5. docs/research_synthesis.md          — 18-paper synthesis: landscape, gaps, ranked directions
  6. docs/pipeline_decisions.md          — ~40 design decisions with tensions/tradeoffs
  7. src/speech_decoding/models/         — Existing model components (spatial_conv, backbone, heads)
  8. src/speech_decoding/training/       — Existing training code (augmentation, CTC utils, trainer)
```

CLAUDE.md contains domain context: electrode placement physics, somatotopic resolution, field conventions, parameter budgets, data regime. Use it as INPUT to your reasoning, not as gospel. Prior decisions were made under single-patient conditions. **Question everything.** If a prior decision says "X doesn't work" but you have a reason to believe the context has changed (e.g., more data, different objective), TRY IT.

## Context Management

Claude Code has limited context. The research loop is designed around **compaction cycles**:

```
┌──────────────────────────────────────────────────────────────────┐
│  1. LOAD: Read mandatory context files (see above)               │
│  2. BRAINSTORM: Reason from first principles about what to try   │
│  3. IMPLEMENT: Modify train.py                                   │
│  4. RUN: Execute experiment (~5-15 min with multi-patient Stage 1)│
│  5. ANALYZE: Review results, reason about WHY                    │
│  6. RECORD: Update docs/experiment_log.md + results.tsv          │
│  7. COMPACT: /compact to free context, then restart at 1         │
└──────────────────────────────────────────────────────────────────┘
```

**docs/experiment_log.md** is the single source of truth. After every experiment:
- Add the result to the autoresearch LOPO table
- Add a numbered realization if you learned something non-obvious
- Update "What to Try Next" with directions unlocked by the result

Keep the experiment log **lean**. Remove stale entries. Prioritize insights over raw data.

**Compact at ~200k context** or after every 2-3 experiments. Before compacting, ensure experiment_log.md captures everything you need for the next cycle.

## Experiment Protocol

### Setup (do once)
1. Read `prepare.py`, `train.py`, `docs/experiment_log.md`, and this file completely
2. Create branch `autoresearch-lopo/run1` from current HEAD
3. Verify data loads: `cd /Users/bentang/Documents/Code/speech && PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/autoresearch_lopo/train.py`
4. Initialize `results.tsv` with header: `commit\tval_per\ttraining_sec\tstatus\tdescription`
5. Run baseline and log it as the first entry

### Run Command

```bash
cd /Users/bentang/Documents/Code/speech && PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/autoresearch_lopo/train.py 2>&1 | tee scripts/autoresearch_lopo/run.log
```

### Per-Experiment Loop
```
  1. Read docs/experiment_log.md + results.tsv — what's known, what worked, what failed
  2. Reason from first principles: WHY did the best experiments work? What's the bottleneck now?
  3. Form a hypothesis, modify train.py
  4. git add train.py && git commit -m "experiment: <description>"
  5. Run the command above
  6. Extract: grep "^val_per:" scripts/autoresearch_lopo/run.log
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
- **Record reasoning.** experiment_log.md captures WHY, not just WHAT.
- **Time budget = 15 minutes.** LOPO Stage 1 is slower than single-patient. If an experiment takes longer, simplify.
- **Compact proactively.** Don't wait until context is full — compact after recording insights.
- **If an experiment crashes or NaNs**, record as "fail/crash" and move on.

## Output Protocol

`train.py` must print (after `---` marker):
```
---
val_per:            <float>   # PRIMARY METRIC — lower is better
val_per_std:        <float>
fold_pers:          [...]
linear_pers:        [...]     # per-fold PER from linear head
knn_pers:           [...]     # per-fold PER from k-NN eval
stage1_seconds:     <float>   # Stage 1 (multi-patient) training time
training_seconds:   <float>   # total training time
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
a1b2c3d	0.846000	312.5	keep	baseline: SpatialConv + BiGRU + CE LOPO
b2c3d4e	0.821000	298.1	keep	Stage 2 source replay 30%
c3d4e5f	0.850000	287.4	discard	H=128 overfit in Stage 2
d4e5f6g	0.000000	0.0	crash	OOM with batch_size=64
```

## How to Think About This Problem

**Don't follow recipes. Reason from first principles.**

Before each experiment, write down (in your thinking):
1. **What is the current bottleneck?** (Stage 1 diverging? Stage 2 doing nothing? Overfitting to source? Underfitting on target?)
2. **What does the DATA look like?** (~1300 source trials across 9 grids of varying shape + 120 target trials per fold, 200Hz, ~200 timesteps after stride, 9 phonemes in 3 positions)
3. **What does the BRAIN do?** (sensorimotor cortex, phoneme production, somatotopic organization, ~100-200ms per phoneme, variable placement across patients)
4. **What would a PERFECT LOPO model look like?** (Stage 1 learns a patient-agnostic phoneme representation; Stage 2 fast-adapts the spatial read-in to S14's grid without forgetting)
5. **Why might the CURRENT model fail?** (Stage 2 adapter too small? Stage 1 CE on diverse grids is noisier than single-patient? Backbone too small for multi-patient signal?)

**The key insight from the LOPO pilot**: Stage 2 was near-trivial (3-5s, early-stop at 13-46 steps). The adaptation is doing almost nothing. Either the Stage 1 backbone is already well-calibrated (unlikely given 0.846 PER) or Stage 2 needs a stronger signal / more epochs / source replay to stabilize.

**Challenge every assumption.** The single-patient optimal (H=32, CE, label smoothing, dropout, mixup) was tuned for N=120. With N≈1300 in Stage 1:
- Maybe H=64 or H=128 now helps instead of hurts
- Maybe stronger augmentation is needed to prevent Stage 1 from memorizing source patients
- Maybe a different Stage 1/Stage 2 epoch allocation is needed

## Research Directions

Think broadly. Don't just tweak hyperparameters — reason about what the model needs to learn and design architectures/objectives to teach it.

### Tier 1: High-Confidence (Try First)

**Fix the obvious failures:**
- **CE loss for LOPO Stage 1**: Pilot used CTC (0.846). CE was never tested cross-patient. All the single-patient gains came from CE. This is the #1 experiment.
- **Apply full single-patient recipe to Stage 2**: Label smoothing 0.1, per-position heads, dropout 0.3, mixup, TTA — none of these were in the pilot. Apply the recipe to Stage 2 adaptation.
- **Source replay during Stage 2** (Levin 2026 precedent): Interleave 30% source patient data during Stage 2. Prevents catastrophic forgetting of shared representations.

**Backbone capacity:**
- **H=64 or H=128 in Stage 1**: At N=1300, H=32 may now underfit the multi-patient signal. H=128 overfit at N=120 but with 10× data it might be the right size.
- **Unfreezing vs freezing backbone in Stage 2**: The pilot froze the backbone. Try partially unfreezing (e.g., last GRU layer only) during Stage 2 for faster adaptation.

**Stage 1/Stage 2 allocation:**
- More Stage 1 epochs: pilot early-stopped at ~800 of 2000 steps. Better objective + recipe may sustain training longer.
- More Stage 2 epochs + lower LR: Stage 2 needs enough steps to actually adapt without forgetting.

### Tier 2: Medium Confidence

**Patient alignment:**
- **Gradient reversal / domain adversarial**: Add a patient-ID classifier with reversed gradients on the backbone output. Forces the backbone to learn patient-invariant phoneme representations. Classic domain adaptation.
- **Patient-conditional encoding**: Learned patient ID embeddings added (FiLM-style: scale + shift) to the backbone's hidden state. Lets the model be aware of which patient it's processing.
- **Per-patient batch/instance normalization**: Normalize backbone activations per-patient before the shared GRU.

**k-NN with source patients:**
- At eval time, add source patient embeddings as extra k-NN neighbors. The backbone (trained on source patients) likely encodes source trials in the same space as S14 trials.
- Multi-patient k-NN: retrieve from all patients, not just the current fold's training set.

**Progressive unfreezing:**
- Stage 2: unfreeze from head → GRU → Conv1d in sequence over training. Prevents early overfitting of the backbone.

### Tier 3: Exploratory

**Attention-based approaches:**
- Attention pooling over time (LOPO has more source training data — learned attention queries may now work for Stage 1)
- Cross-patient attention: query S14 representations against source patient representations
- Multi-scale temporal convolution: parallel Conv1d at stride 5, 10, 20 — multi-resolution phoneme encoding

**Training dynamics:**
- Curriculum learning: start Stage 1 with patients that share S14's grid shape (8×16: S16, S22, S23), then add 12×22 and 8×34 patients
- Cosine restarts in Stage 1 to avoid local minima from grid heterogeneity
- SWA in last 30% of Stage 1 for a flatter, more generalizable minimum
- SAM optimizer: finds flatter minima — might help with cross-patient generalization

**Ensemble of Stage 1 checkpoints:**
- Save backbone at multiple Stage 1 checkpoints, run Stage 2 from each, ensemble predictions

### Tier 4: Low Confidence (Try If Stuck)

- **CTC loss for LOPO with improved recipe**: Was 0.846 baseline. With label smoothing + longer training, might improve — but CE is almost certainly better.
- **Contrastive auxiliary losses**: Fought CE in single-patient. With multi-patient data, within-class same-phoneme pairs are now across-patient — might provide useful invariance signal. But treat as risky.
- **Meta-learning (Reptile/MAML)**: Learn initializations that adapt quickly to new patients. Principled for LOPO, but adds complexity and training instability. Try only if simpler approaches plateau.
- **SSL pretraining on source patients**: All SSL failed at ~1 min/patient. Source patients collectively have ~15 min — still below minimum (~30 min). Only try if a fundamentally different SSL formulation is available.

## Key Insights to Guide You

1. **Stage 2 doing nothing is the most actionable failure mode.** The pilot's Stage 2 completed in seconds. Fix this first — source replay, more epochs, lower LR, or partial unfreezing.

2. **1300 source trials is a different regime.** The single-patient constraints (H=32, heavy regularization) may not apply. Design Stage 1 as if it's a normal supervised problem on a medium dataset.

3. **Grid heterogeneity is the core challenge.** Patients have (8,16), (12,22), and (8,34) grids. The spatial read-in (per-patient Conv2d) handles this, but the shared backbone must generalize across very different spatial representations.

4. **CE > CTC per-patient for every tested condition.** CTC wasted capacity on alignment. CE is per-position supervised signal. Use CE for Stage 2. Stage 1 CE requires phoneme labels — all source patients have them.

5. **Do not retry SSL.** All SSL methods (JEPA, DINO, BYOL, VICReg, LeWM) failed at this data scale. This is a firm finding. Data volume is the bottleneck.

6. **Source patients are NOT noisy augmentation.** They are real patients with real phoneme labels. Stage 1 should use them as a supervised multi-patient training problem, not as unlabeled SSL data.

7. **Content collapse is a real risk in multi-patient training.** With heterogeneous grids, the model might learn patient-specific statistics rather than phoneme features. Check collapse diagnostics on every run.

8. **The 15-minute budget is real.** Stage 1 is slower than single-patient training. Keep Stage 1 under 600s and Stage 2 under 300s. If an architecture requires longer, simplify.
