# SSL Solution Space Exploration Plan

## Context

**Best supervised result**: PER 0.737 on S14 (grouped-by-token CV) from autoresearch session — label smoothing + mixup + focal loss + dual CE/articulatory head + weighted k-NN (k=10) + TTA 16. All on CE-trained (supervised) backbone features.

**Best SSL result**: PER 0.811 on S14 from JEPA with basic unweighted k-NN eval. The improved eval recipe (weighted k-NN, TTA, articulatory head) was **never applied to JEPA features** — this is the #1 gap.

**Key autoresearch insight**: Evaluation improvements (k-NN → weighted k-NN → TTA) gave 6.5pp. Training improvements (label smoothing → mixup → focal) gave 7.0pp. At N=120, eval dominates. ALL contrastive auxiliary losses failed per-patient (SupCon, center loss, projection SupCon) — but cross-patient (N≈2400) is a fundamentally different regime where contrastive can work (more negatives, cross-patient positives, articulatory targets).

**Data regime**: ~120 trials/patient, 11 unique patients (S32=S36 are duplicates), 9 phonemes, 3/trial (CVC/VCV). ~2400 pooled trials across ~11 patients for SSL. Full-trial CTC on position-1 epochs.

## Data Quality Findings

### .fif event markers are MFA-derived
Verified by exact match (0.0-0.2ms) between .fif event sample times and CSV phoneme onset times for S14 and S22. Both the epoch boundaries and CSV timing come from Montreal Forced Aligner on paired audio. If MFA fails for a patient, both are wrong. Phoneme LABELS (event_id) are always correct — they come from the known stimulus transcript.

### Per-patient alignment quality
- **Good MFA alignment**: S22, S39, S58 (confirmed by visual inspection of sweep_v2 videos)
- **Poor alignment**: Others have 300-500ms offsets or misaligned phoneme boundaries
- **S32 = S36**: Same recording, different channel subsets (234MB/256ch vs 117MB/128ch). 11 unique patients, not 12.

### Decision: Full-Trial CTC, Not Per-Position
Per-phoneme MFA alignment is unreliable for ~50% of patients. Per-position epochs require accurate alignment. CTC handles temporal alignment natively — the 1.0s window is wide enough to contain all 3 phonemes regardless of MFA quality. All SSL work uses full-trial (3-phoneme) CTC evaluation.

### Evaluation approach: CTC primary, k-NN secondary
**Why not k-NN as primary?** The task is decoding [blank][p1][p2][p3][blank] from ~20 temporal frames. Mean-pooled k-NN collapses all 20 frames into a single 64-dim vector and votes on 3 positions independently from that same vector — this throws away the temporal sequence structure. It works at all only because different 3-phoneme combinations produce different temporal averages, but it's an information bottleneck.

CTC classifies each frame into {9 phonemes + blank} and decodes the sequence — it preserves temporal structure and handles variable phoneme timing. CE per-position (dividing frames into 3 segments) also works but assumes fixed segmentation.

**k-NN remains a useful secondary metric**: it measures pure feature geometry quality without any training. Good k-NN = well-clustered embedding space.

## Theoretical Foundation: What Makes Representations k-NN-Friendly?

**Wang & Isola 2020** ("Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere"):
- **Alignment**: Same-class samples map close together
- **Uniformity**: Features spread evenly on the unit hypersphere (no mode collapse, no dimensional collapse)

k-NN needs BOTH. A collapsed representation has perfect alignment but zero uniformity. A random projection has uniformity but no alignment. The sweet spot is compact, well-separated clusters.

This framework explains:
- Why DINO (centering + sharpening) excels at k-NN — centering directly targets uniformity
- Why JEPA's MSE loss doesn't guarantee k-NN-friendly geometry — it optimizes prediction accuracy, not cluster separation
- Why VICReg's variance + covariance terms directly prevent both forms of collapse

## Augmentation Space for Neural Data

Critical constraint for all augmentation-dependent SSL methods (VICReg, Barlow Twins, BYOL, SimCLR):

| Augmentation | Preserves phoneme identity? | Evidence |
|---|---|---|
| Amplitude scale (±30%) | Yes | Simulates impedance variation |
| Gaussian noise (σ=0.05) | Yes | Recording noise |
| Channel dropout (10-20%) | Probably | Redundant channels |
| Time jitter (±50ms) | Probably | Onset variability |
| Temporal stretch (±15%) | Probably | Speaking rate variation |
| Time warp (nonlinear) | Unknown | May distort dynamics |
| Spatial permutation | No | Destroys somatotopy |
| Random crop | Depends on length | May cut phoneme boundaries |

This is **much narrower** than vision (SimCLR uses color jitter, random crop, blur, grayscale, flip — each highly informative). With only 4-5 safe augmentations, contrastive methods that rely heavily on augmentation diversity are at a disadvantage compared to masking-based methods (JEPA, DINO) that don't need augmentations.

## Phase A: Data & Re-baseline (prerequisite)

### 0A: Mask Auditory Stimulus
- **Change**: `tmin=-0.5` → `tmin=0.0` in all data loading
- **Effect**: 300→200 frames (1.5s→1.0s). Removes pre-production planning/auditory activation. After stride=10: 20 frames instead of 30.
- **Status**: train_pretrain.py updated. train_per_patient.py accepts --tmin 0.0.
- **Early result (stratified CV)**: tmin=0.0 gave PER=0.727 vs tmin=-0.5 PER=0.700. Production-only is slightly worse — pre-production planning frames carry discriminative information. Grouped CV comparison running.

### 0C: Trial Averaging for Evaluation (deferred)
Average all repetitions of same token (~3 reps) to improve SNR by ~√3 ≈ 1.7×. Use averaged trials as k-NN prototypes. **Deferred** until k-NN is established as secondary eval metric.

### 0D: Non-Sig Channel Interpolation (deferred)
Replace non-significant channels with spatial average of neighbors instead of zeros. **Deferred** — requires neighbor graph from electrode grid, moderate implementation effort. Revisit if spatial conv struggles.

### 1A: Re-baseline Supervised (Grouped-by-Token CV)
- Added `cv_type: grouped` to trainer config (uses `grouped_cv.py` module)
- Running two comparisons on S14:
  - Grouped CV + tmin=0.0 (production-only)
  - Grouped CV + tmin=-0.5 (original window)
- Previous baselines for reference:
  - PER 0.700: CE, stratified CV, tmin=-0.5 (token leakage)
  - PER 0.737: autoresearch, grouped CV, tmin=-0.5 (full recipe)
  - PER 0.778: CTC, stratified CV, tmin=-0.5
- **Extends to**: S22, S39, S58 (confirmed clean MFA alignment)

## Phase B: JEPA Assessment (gate for Phase C)

### 2A: JEPA + Improved Eval
- Load existing JEPA checkpoint (from train_pretrain.py Method B run)
- Stage 3 evaluation with weighted k-NN (k=10), TTA 16, dual articulatory head
- Use production-only window (tmin=0.0) and grouped-by-token CV
- **Compare**: JEPA (improved eval) vs supervised (improved eval, PER 0.737)
- **Decision gate**:
  - JEPA < 0.737 → SSL features are better when properly evaluated → optimize JEPA (Phase E)
  - JEPA ≈ 0.737 → features equivalent → try alternative SSL objectives (Phase C)
  - JEPA > 0.737 → supervised genuinely better → focus on SSL objectives that target k-NN geometry (Phase C, prioritize 2B/2C)

## Phase C: SSL Objective Exploration

Each method uses the same backbone (SpatialConv → LN → Conv1d(s=10) → BiGRU) and same evaluation protocol (Stage 3: freeze backbone → CE fine-tune → PER + collapse diagnostics + weighted k-NN as secondary). All trained on production-only window with grouped-by-token CV.

### 2B: DINO Self-Distillation (best k-NN geometry)

**Why**: DINO (Caron et al. 2021) produces the best k-NN features in all SSL literature. k-NN on DINO ViT features matches supervised linear probes on ImageNet. Three collapse-prevention mechanisms directly target the alignment + uniformity properties (Wang & Isola 2020) that k-NN needs:
1. **EMA teacher** — prevents trivial collapse (alignment without representation)
2. **Centering** — subtracts running mean of teacher outputs → prevents mode collapse → targets uniformity
3. **Sharpening** — low temperature on teacher softmax → encourages peaked predictions → better cluster separation

JEPA only uses mechanism (1). DINO's extra mechanisms specifically target the uniformity component that makes k-NN work.

**Architecture delta from JEPA** (~50 lines):
```
Student (online encoder): masked input → BiGRU → mean-pool → projector → P-dim logits
Teacher (EMA encoder):    full input   → BiGRU → mean-pool → projector → P-dim logits

Loss: CE(sharpen(center(teacher_logits), τ_t), softmax(student_logits / τ_s))
Centering: running mean of teacher outputs, subtracted before softmax
Sharpening: teacher temperature τ_t < student temperature τ_s
```

Key difference from JEPA: **cross-entropy loss in probability space**, not MSE in latent space. Operates on a discrete distribution over P prototypes, not continuous frame predictions.

**Views** (multi-crop adapted for temporal data):
- Global view 1: full trial (20 frames after stride)
- Global view 2: full trial + augmentation (amp scale, noise, channel dropout)
- Local views (2-4): random 8-frame temporal crops
- Note: masking-based views avoid the narrow augmentation space problem

**Hyperparameters**: P=256 (prototype dim, << ImageNet 65536 due to 9 classes), τ_t=0.04, τ_s=0.1, EMA momentum 0.996→1.0 cosine.

**Ref**: Caron et al. 2021 (DINO), Zhou et al. 2022 (iBOT adds patch-level distillation)

### 2C: CEBRA Articulatory Contrastive (domain-specific)

**Why**: Uses the one thing we know is patient-invariant: articulatory motor commands. The 9×15 articulatory feature matrix (from `phoneme_map.py`) provides cross-patient-invariant behavioral variables. InfoNCE with articulatory similarity as the positive criterion directly optimizes for phoneme-separable embeddings.

**Key regime shift from failed autoresearch SupCon**: SupCon failed at N=120 (single patient, 9 classes, ~13/class). CEBRA operates **cross-patient** at N≈2400 (11 patients pooled) with **soft articulatory labels** (not hard class labels). More negatives, more positive diversity, richer similarity structure. Fundamentally different from per-patient contrastive.

**Architecture**:
```
Encoder: input → BiGRU → mean-pool → 64-dim embedding
Projection: Linear(64, 64) → L2-normalize

For each anchor z_i with phoneme label y_i:
  art_i = articulatory_matrix[y_i]  # 15-dim binary vector
  Positive set: {z_j : cosine(art_i, art_j) > 0.5}  # same or articulatorily similar phoneme
  Negative set: rest of batch

Loss: InfoNCE = -log(Σ_pos exp(sim(z_i, z_j)/τ) / Σ_all exp(sim(z_i, z_k)/τ))
```

**Cross-patient training**: Pool all patients' labeled trials. Each batch samples across patients. Articulatory similarity is patient-invariant by construction — no spatial alignment needed.

**Ref**: Schneider et al. 2023 (CEBRA)

### 2D: VICReg (fixed, augmentation baseline)

**Why**: Simplest principled SSL approach. No momentum encoder, no asymmetry, no architectural tricks. Variance + Invariance + Covariance losses directly prevent both failure modes identified by Wang & Isola 2020:
- Variance term → prevents complete collapse → targets alignment
- Covariance term → prevents dimensional collapse → targets uniformity
- Invariance term → augmentation invariance

Previous VICReg attempt failed only due to proj_dim > batch_size bug — never properly tested.

**Architecture**:
```
Encoder: input → BiGRU → mean-pool → 64-dim
Projector: Linear(64, 64) → BN → ReLU → Linear(64, 64)

View 1: trial + augmentation A
View 2: trial + augmentation B

Loss = λ_inv * MSE(z1, z2)                    # invariance
     + λ_var * max(0, 1-std(z))               # variance (per-dim)
     + λ_cov * off_diagonal(cov(z))^2 / d     # covariance
```

**Constraint**: Augmentation-dependent — limited by narrow neural augmentation space. With batch_size=8 and 64-dim projector, covariance matrix (64×64) estimated from 8 samples is marginal. May need larger batches.

**Ref**: Bardes et al. 2022 (VICReg)

### 2E-2H: Lower Priority (skip unless 2B-2D fail)

| Method | When to try | Key uncertainty | Ref |
|--------|------------|-----------------|-----|
| **Barlow Twins** | If VICReg works but batch norm helps | Batch size sensitivity | Zbontar et al. 2021 |
| **BYOL** | If augmentation-based > masking-based | Narrow augmentation space for neural data | Grill et al. 2020 |
| **CPC/wav2vec 1.0** | If temporal prediction + contrastive > JEPA MSE | LeWM (MSE temporal prediction) already failed at PER 0.911 — but CPC uses contrastive loss, not MSE, which may learn content rather than smoothness | Schneider et al. 2019 |
| **wav2vec 2.0** | If discrete codebook helps | Codebook needs enough data to populate; ~2400 trials may not suffice | Baevski et al. 2020 |

## Phase D: Fine-Tuning & Transfer (after best SSL found)

### 3A: LP-FT (Linear Probe then Fine-Tune)
- Freeze best SSL backbone → train CE head (linear probe) until convergence
- Unfreeze backbone with 10× lower LR → fine-tune end-to-end
- Prevents destroying pretrained features (Kumar et al. 2022)

### 3B: Prototypical Fine-Tuning
- Sample N-way K-shot episodes from training data
- Classify by Euclidean distance to class prototypes (9-way, K-shot where K = train set size / 9)
- Backprop through backbone — directly optimizes the metric space for few-shot classification
- Tian et al. 2020: SSL pretraining + prototypical fine-tuning > end-to-end meta-learning
- Snell et al. 2017: Prototypical Networks

### 3C: Feature Concatenation / Multi-View
- JEPA(64d) + supervised(64d) → 128d → k-NN
- Or: separate k-NN on each feature set, average class probabilities (multi-view ensemble)
- Tests whether SSL and supervised features capture complementary information

### 3D: Full Autoresearch Recipe on SSL Backbone
- Best SSL pretrain → fine-tune with full recipe: label smoothing 0.1, mixup α=0.2, focal γ=2, dual CE/articulatory head, weighted k-NN k=10, TTA 16
- Maximum supervised optimization on top of SSL features

## Phase E: Scaling (conditional on Phase C/D signal)

| Direction | When | Expected impact |
|-----------|------|-----------------|
| Include dev patients (S16, S26, S57) + Lexical patients in SSL corpus | If SSL shows any signal | 11→20+ patients for pretraining |
| Longer pretraining (5K→20-50K steps) | If loss still decreasing at 5K | More capacity utilization |
| Masking curriculum (easy→hard) | If JEPA/DINO converge to mediocre features | Progressive difficulty prevents early collapse |
| Patient-conditional JEPA (patient embedding conditions predictor) | If cross-patient features noisy | Patient-aware prediction |
| Larger backbone (H=64, deeper GRU) | If SSL features good but low-capacity | SSL justifies more params (more data → less overfitting) |

## Execution Order

```
Phase A (1 day) ─── prerequisite for everything
│
├─ 0A: tmin=0.0 ✓ (done, but early results suggest tmin=-0.5 may be better)
├─ 1A: Re-baseline S14 CE grouped CV (running: both tmin=0.0 and tmin=-0.5)
│
Phase B (half day) ─── gate decision
│
├─ 2A: JEPA + improved eval (weighted k-NN, TTA, articulatory head)
├─ Compare → decide Phase C priority
│
Phase C (2-3 days) ─── parallel where possible
│
├─ 2B: DINO adaptation (~50 lines delta from JEPA)
├─ 2C: CEBRA articulatory contrastive (cross-patient, semi-supervised)
├─ 2D: VICReg fixed (augmentation baseline)
├─ Compare all on same eval protocol (grouped CV, PER + k-NN)
│
Phase D (1-2 days) ─── on best SSL method
│
├─ 3A: LP-FT
├─ 3B: Prototypical fine-tuning
├─ 3D: Full autoresearch recipe
│
Phase E (ongoing) ─── if signal exists
│
├─ Scale data, steps, architecture
```

## Success Criteria

- **Minimum**: Any SSL method beats supervised PER 0.737 on S14 grouped-by-token CV
- **Strong**: SSL method generalizes to S22/S39/S58 (clean MFA patients) with consistent improvement
- **Home run**: Cross-patient SSL (train on all 11 patients, eval on held-out) beats per-patient supervised

## Key Files

| File | Role |
|------|------|
| `scripts/train_pretrain.py` | SSL training CLI (Methods B/C/A) |
| `src/speech_decoding/pretraining/jepa_model.py` | JEPA (to create per existing plan) |
| `src/speech_decoding/pretraining/dino_model.py` | DINO (to create, Phase C) |
| `src/speech_decoding/pretraining/cebra_model.py` | CEBRA (to create, Phase C) |
| `src/speech_decoding/pretraining/vicreg_model.py` | VICReg (to create, Phase C) |
| `src/speech_decoding/pretraining/stage3_evaluator.py` | Unified SSL evaluation |
| `src/speech_decoding/evaluation/grouped_cv.py` | Grouped-by-token CV splitter |
| `src/speech_decoding/training/trainer.py` | Per-patient trainer (now supports cv_type: grouped) |
| `scripts/autoresearch/train.py` | Supervised baseline (branch autoresearch/run1) |
| `docs/experiment_log.md` | All results |

## References

- Assran et al. 2023 — I-JEPA: Self-supervised learning from images with a joint-embedding predictive architecture
- Bardes et al. 2022 — VICReg: Variance-Invariance-Covariance Regularization
- Bardes et al. 2024 — V-JEPA: Video Joint Embedding Predictive Architecture
- Baevski et al. 2020 — wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
- Caron et al. 2021 — DINO: Emerging Properties in Self-Supervised Vision Transformers
- Grill et al. 2020 — BYOL: Bootstrap Your Own Latent
- Kumar et al. 2022 — Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution (LP-FT)
- Schneider et al. 2019 — wav2vec: Unsupervised Pre-training for Speech Recognition
- Schneider et al. 2023 — CEBRA: Learnable latent embeddings for joint behavioural and neural analysis
- Snell et al. 2017 — Prototypical Networks for Few-shot Learning
- Tian et al. 2020 — Rethinking Few-Shot Image Classification
- Wang & Isola 2020 — Understanding Contrastive Representation Learning through Alignment and Uniformity
- Zbontar et al. 2021 — Barlow Twins: Self-Supervised Learning via Redundancy Reduction
- Zhou et al. 2022 — iBOT: Image BERT Pre-Training with Online Tokenizer
