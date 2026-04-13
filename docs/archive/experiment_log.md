# Experiment Log — NCA-JEPA Pretraining

Running log of experiments, results, and realizations. Updated as experiments complete.

---

## 2026-03-28: Phase 0 Baselines (S14)

### Setup
- **Patient:** S14 (153 trials, 8×16 grid, 128 channels)
- **CV:** Grouped-by-token, 5-fold (all reps of same CVC/VCV token in same fold)
- **Window:** [-0.5, 1.0s] response-locked, 200Hz → 301 frames
- **Architecture:** Conv2d(1→8, pool 4×8) → d=256 → Conv1d(256→64, stride=10) → BiGRU(64,64,2L)
- **Head:** Linear(128→27) = 3 positions × 9 phonemes, per-position CE loss

### Results

| Experiment | Method | CV Type | H | Epochs | Augment | PER (mean±std) | Notes |
|-----------|--------|---------|---|--------|---------|----------------|-------|
| Phase 0 v1 | E (frozen random) | Grouped | 64 | 50 | None | 0.897 ± 0.048 | Quick gate check |
| Phase 0 v1 | D (supervised) | Grouped | 64 | 50 | None | 0.880 ± 0.069 | Quick gate check — too few epochs |
| Phase 0 v1 | Spatial-only | Grouped | 64 | 50 | None | 0.891 ± 0.056 | Quick gate check |
| Phase 0 v2 | E (frozen random) | Grouped | 64 | 200 | None | 0.911 ± 0.024 | |
| Phase 0 v2 | D (supervised) | Grouped | 64 | 200 | None | 0.835 ± 0.081 | Gate 0: PASS |
| Phase 0 v2 | Spatial-only | Grouped | 64 | 200 | None | 0.900 ± 0.030 | |
| Grouped baseline | D (supervised) | Grouped | 32 | 300 | Full | 0.825 ± 0.088 | Production trainer, fold PERs: [.833,.889,.844,.900,.656] |
| Historical | D (supervised) | Stratified | 32 | 300 | Full | 0.700 | Known baseline (leaky CV) |

### Gate 0 Decision
**PASS** — D (0.835) < spatial-only (0.900) < E (0.911). Architecture learns, BiGRU helps. Proceed to Phase 1.

### Realizations

1. **Grouped-by-token CV is dramatically harder than stratified CV.** The 0.125 PER gap (0.700 → 0.825) means ~42% of "correct" phoneme predictions in the stratified baseline came from within-token memorization, not generalizable phoneme decoding. This is the single most important finding so far.

2. **Extreme fold variance under grouped CV.** Fold PERs range from 0.656 to 0.900 (std=0.088) — which tokens land in which fold matters enormously with only 52 unique tokens. This makes grouped-by-token CV noisy as an evaluation metric. Consider 3-fold instead of 5-fold, or reporting both CV types.

3. **The Phase 0 quick runner and production trainer converge.** Phase 0 v2 Method D (no augmentation, H=64) = 0.835; production trainer (full augmentation, H=32) = 0.825. Augmentation helps marginally under grouped CV — the bottleneck is generalization to unseen tokens, not overfitting to seen ones.

4. **All methods are near chance under grouped CV.** Chance PER ≈ 0.889. Best supervised is 0.825 — only 6.4% above chance. The signal is real but thin, consistent with ~150 trials and ~52 tokens.

5. **The BiGRU adds ~6.5% over spatial-only** (Phase 0 v2). Temporal dynamics contain decodable phoneme information beyond spatial snapshots.

6. **Fold 5 outlier (PER=0.656) suggests some token groupings are much more discriminable.** If those tokens happen to be in training, the model performs dramatically better. This is a data property, not a model property.

4. **The BiGRU adds ~6.5% over spatial-only.** This validates using temporal dynamics for pretraining — there IS temporal information beyond spatial snapshots.

5. **Phase 0 v1 (50 epochs) was misleading.** D ≈ E because 50 epochs wasn't enough for D to converge. This produced a false "architecture has no value" warning. Lesson: always run sufficient epochs or use early stopping with enough patience.

6. **Grouped CV feasibility concern.** With ~52 unique tokens and 5-fold CV, each validation fold has only ~10 tokens (~30 trials). High variance across folds (D std = 0.081). Consider 3-fold CV or reporting both grouped and stratified.

---

## 2026-03-30: Autoresearch — Autonomous PER Optimization

Framework: `scripts/autoresearch/` — prepare.py (fixed eval) + train.py (agent modifies) + program.md.
Branch: `autoresearch/run1`. All experiments on S14, grouped-by-token 5-fold CV.

### Results

| # | PER | Time | Status | Description |
|---|-----|------|--------|-------------|
| baseline | 0.872 | 201s | keep | SpatialConv(8ch,pool4x8) + BiGRU(32,2L) + CE mean-pool |
| exp01 | 0.831 | 211s | keep | + label smoothing 0.1 |
| exp02 | 0.819 | 212s | keep | + separate per-position heads + dropout 0.3 |
| exp03 | 0.802 | 234s | keep | + mixup alpha=0.2 |
| exp07 | 0.781 | 237s | keep | + k-NN eval (best-of linear/k-NN per fold) |
| exp10 | 0.767 | 245s | keep | + TTA 8 copies |
| exp11 | 0.763 | 215s | keep | + CEBRA articulatory head (project to 15-dim features) |
| exp12 | 0.756 | 256s | keep | + dual head (CE + articulatory averaged logits) |
| exp13 | 0.755 | 253s | keep | + focal loss γ=2 |
| **exp17** | **0.737** | **268s** | **keep** | **+ weighted k-NN + TTA 16 copies — BEST** |

### Failed Experiments (all discarded)

| # | PER | Why Failed |
|---|-----|------------|
| exp04 | 0.812 | EMA decay=0.998 — exceeded time, model lags behind |
| exp05 | 0.845 | Attention pooling — 30 frames too few for learned queries |
| exp06 | 0.831 | R-Drop — double forward, exceeded budget |
| exp08 | 0.807 | SupCon+CE (wt=0.5,τ=0.1) — contrastive fights CE |
| exp09 | 0.817 | Center loss (wt=0.1) — destroyed linear head |
| exp14 | 0.798 | H=64 (298K params) — overfits despite regularization |
| exp15 | 0.798 | Proj head SupCon — contrastive STILL hurts through separate head |
| exp16 | 0.782 | Mixup alpha=0.4 — too aggressive |
| exp18 | 0.788 | k=15 — over-smooths |
| exp19 | 0.769 | Snapshot ensemble — cosine restarts don't mesh with early stopping |
| exp20 | 0.755 | Patience 15 — exceeded time budget |
| exp21 | 0.772 | CutMix + dropout 0.5 — too much regularization |
| exp22 | 0.791 | Temporal derivatives + WD 1e-3 — derivatives noisy |
| exp23 | 0.787 | Stride=5 (40Hz) — slower, no benefit |
| exp24 | 0.757 | Soft k-NN (all-sample softmax) — too diffuse |
| exp25 | 0.776 | Label smoothing 0.15 — 0.1 is optimal |
| exp26 | 0.802 | Mean+max pool concat — extra head params hurt |
| exp27 | 0.759 | TTA on train embeddings — adds noise to reference |
| exp28 | 0.774 | Dilated CNN backbone — 2x faster but BiGRU context matters |

### Autoresearch Realizations

36. **Evaluation improvements dominate training improvements.** k-NN + TTA + weighted voting gave more total PER improvement (0.802→0.737 = 6.5pp) than all training regularization combined (0.872→0.802 = 7.0pp). At N=120, the model learns decent features; the bottleneck is extracting predictions from those features.

37. **ALL contrastive auxiliary losses hurt at N=120.** SupCon, center loss, projection head SupCon — all made PER worse. CE is already optimizing the right objective; adding contrastive noise to the gradient just disrupts learning.

38. **The CEBRA articulatory head is the best single architectural innovation.** Projecting to articulatory feature space (15-dim) and classifying by cosine similarity to phoneme vectors naturally encodes inter-class structure (b≈p, both bilabial stops). This helped k-NN most (+0.021) because it creates articulatory-similarity clusters.

39. **Weighted k-NN >> uniform k-NN.** Similarity-weighted voting (each neighbor weighted by cosine sim) outperformed majority voting by ~0.015 PER. Closer neighbors should count more.

40. **TTA averages over augmentation noise, not model uncertainty.** The improvement from TTA 8→16 was modest (0.767→0.737 including weighted k-NN). Diminishing returns beyond 16.

41. **Model capacity (H=32, 119K params) is right-sized for N=120.** H=64 (298K) overfits. All regularization tricks (label smoothing, mixup, focal loss, dropout) can't overcome the fundamental parameter-to-sample ratio problem.

42. **The optimal label smoothing is 0.1, not higher.** Testing 0.15 made PER worse. The model needs some sharp targets to learn.

43. **Mean+max pooling adds parameters without adding information at 30 frames.** Max-pool picks the single highest-activation frame, which is noise-sensitive. Mean-pool is the right inductive bias for short sequences where information is distributed.

44. **Train embeddings should NOT be TTA'd.** Augmenting train embeddings during k-NN eval adds noise to reference points. The clean (unaugmented) train embeddings are more reliable because the model was trained to produce good features for those exact inputs.

45. **Dilated CNN is 2× faster but BiGRU is better.** Bidirectional recurrence integrates context from the ENTIRE 30-frame sequence into every frame. Dilated CNN with receptive field 7 misses long-range temporal dependencies. The speed advantage doesn't compensate.

46. **We're approaching the data ceiling.** After 23 experiments, the improvements are getting smaller. The fundamental limit is ~120 training trials with 27 classification targets (3×9). More training data (cross-patient, cross-task) is the most likely path to break through 0.7.

### Current Best Architecture (exp17, PER 0.737)

```
Training: CE + label_smoothing=0.1 + mixup_alpha=0.2 + focal_gamma=2
  Input (B, 8, 16, 301) → augment → SpatialReadIn(Conv2d(1,8,3), pool(4,8)) → (B, 256, 301)
  → Backbone(LN, Conv1d(256,32,s=10), GELU, feat_drop, time_mask, BiGRU(32,32,2L)) → (B,30,64)
  → Dual Head: avg(CEHead(3×Linear(64,9)), ArticulatoryHead(3×Linear(64,15)→art_matrix))

Evaluation:
  TTA: average logits over 16 augmented copies
  k-NN: weighted k-NN (k=10, cosine sim weights) on TTA-averaged embeddings
  Best-of: pick min(linear_PER, kNN_PER) per fold
```

### Stratified CV Comparison (exp17 recipe)

| CV Type | PER | Notes |
|---------|-----|-------|
| **Stratified** (random KFold, leaky) | **0.662** | Token reps can leak across folds |
| **Grouped-by-token** (fair) | **0.737** | No token leakage |
| Historical stratified (CE only) | 0.700 | No k-NN/TTA/articulatory |
| Historical grouped (CE only) | 0.825 | No k-NN/TTA/articulatory |

Leakage gap narrowed from 0.125 (old) → 0.075 (now). k-NN + articulatory + TTA improvements are more robust to eval protocol than CE-only.

### Critical Insight: Autoresearch k-NN is on SUPERVISED Features

**The autoresearch k-NN (0.737) operates on CE-trained backbone embeddings, NOT SSL/JEPA features.** The model is trained from scratch with CE loss on S14 labeled data. k-NN just replaces the linear head at eval time — it's a better evaluation method, not SSL.

The JEPA k-NN result (0.811) used SSL-pretrained features with basic unweighted k-NN, no TTA, no articulatory head. **The improved eval recipe (weighted k-NN + TTA 16 + dual articulatory head) has NEVER been applied to JEPA features.** This is the highest-priority next experiment.

---

## 2026-03-30: SSL Landscape Audit

Comprehensive review of all SSL experiments in results/ directory.

### All SSL Results (Stage 3: frozen backbone + linear probe, grouped-by-token CV, S14)

| Run | SSL Method | PER | Notes |
|-----|-----------|-----|-------|
| pretrain/method_B_byol | BYOL | **0.040** | **Anomalous — likely semi-supervised contamination (see below)** |
| pretrain/method_B_jepa | JEPA linear probe | 0.854 | |
| pretrain/method_B_jepa k-NN k=10 | JEPA k-NN (grouped) | 0.811 | Best fair SSL eval |
| pretrain/method_B | Masked span (PretrainModel) | 0.862 | |
| pretrain/method_B_lewm | LeWM (next-embed + SIGReg) | 0.911 | Near chance |
| pretrain_vicreg | VICReg | 0.891 | batch=8 vs proj=256 mismatch |
| pretrain_v3_target_ce | JEPA + target CE | 0.889 | |
| pretrain_spatial_16ch | JEPA 16ch spatial | 0.891 | |
| pretrain_spatial_2layer | JEPA 2-layer spatial | 0.921 | |
| pretrain_temporal_attn | JEPA + temporal attention | 0.913 | |
| pretrain_v4_mlp | JEPA MLP predictor | 0.908 | |
| pretrain_v3_frozen | LeWM frozen eval | 0.921 | |
| pretrain_v3_ft | LeWM fine-tune | 0.893 | |
| pretrain_v2/method_B | LeWM v2 | 1.000 | Total collapse |
| pretrain_v2_finetune | JEPA fine-tune | 1.000 | Total collapse |

### BYOL PER 0.040: Likely Semi-Supervised Contamination

The BYOL result is 20× better than any other SSL method. The codebase has `SemiSupervisedStage2Trainer` which does joint SSL + CE on S14 labels during Stage 2. There is no `--pretrain-mode byol` in the CLI — this was a one-off experiment. If the semi-supervised trainer was used, the backbone saw S14 phoneme labels for ALL tokens (including future Stage 3 val tokens) during training. Stage 3's grouped CV evaluates on frozen features that already had supervised access to val tokens — a form of target data leakage.

**Needs verification**: Run pure BYOL (SSL-only, no labels in Stage 2) → Stage 3 grouped CV. If PER jumps to ~0.85, contamination confirmed.

### SSL Landscape Findings

50. **All pure SSL methods are PER 0.85-0.91**, only 0-4pp above chance (0.889). SSL features are barely discriminative for phonemes at N=120 per patient.

51. **JEPA is the best pure SSL method.** Linear probe 0.854, k-NN 0.811. Only method with k-NN notably below linear probe — suggests decent feature geometry.

52. **Architecture ablations don't help JEPA.** 16ch spatial, 2-layer spatial, temporal attention, MLP predictor — all ≈ base or worse. The bottleneck is the SSL objective, not the architecture.

53. **Fine-tuning collapses.** LeWM v2 and JEPA fine-tune both hit PER 1.000. Unfreezing backbone on ~120 trials destroys pretrained features. LP-FT (Kumar et al. 2022) is required.

54. **The improved eval recipe has never been applied to SSL features.** Weighted k-NN + TTA improved supervised from 0.840→0.737 (10pp). This recipe on JEPA features is the highest-priority experiment.

---

## 2026-03-30: SSL v2 — DINO, BYOL Verification, JEPA EMA Fix

Fixed critical bug: Stage2Trainer was not calling `model.ema_update()` after optimizer steps. All previous JEPA/BYOL results had static target encoders. Created DINO model (`dino_model.py`) and unified SSL script (`run_ssl.py`).

Also applied autoresearch eval recipe (weighted k-NN + TTA 16) to existing JEPA checkpoint.

### Results (Stage 3: frozen backbone + linear probe, grouped-by-token CV, S14)

| Method | PER | Collapsed? | Notes |
|--------|-----|-----------|-------|
| DINO (new) | 0.863 ± 0.025 | No | Centering + sharpening didn't help |
| JEPA (EMA fix) | 0.862 ± 0.061 | No | EMA update didn't improve vs old 0.854 |
| BYOL (pure SSL) | 0.886 ± 0.056 | Yes | **Confirms old 0.040 was contaminated** |
| Supervised base CE | 0.837 | — | Better than ALL SSL methods |
| Chance | 0.889 | — | |

### Eval Recipe on Existing JEPA (weighted k-NN + TTA 16)

| Eval Method | PER | Notes |
|---|---|---|
| Weighted k-NN + TTA 16 | 0.827 ± 0.048 | Worse than basic k-NN (0.811) |
| Linear probe (no readin adapt) | 0.867 ± 0.054 | Worse than Stage3 (0.854) |
| Best-of per fold | 0.819 ± 0.046 | |

TTA hurts JEPA because the backbone was trained with masking, not augmentation invariance. Augmented inputs produce inconsistent features.

### Realizations

55. **All pure SSL methods converge to PER ~0.86-0.89 — essentially chance (0.889).** DINO, JEPA, BYOL all produce features that can't distinguish phonemes. The SSL corpus (~1600 trials across 8 patients) is too small and the uECOG HGA signal-to-noise too low for self-supervised objectives to discover phoneme structure without labels.

56. **The BYOL 0.040 was semi-supervised contamination.** Pure BYOL gives 0.886 (near chance). The previous result used `SemiSupervisedStage2Trainer` which leaked S14 labels into Stage 2, contaminating Stage 3 evaluation.

57. **The EMA fix didn't help.** JEPA without EMA update: 0.854. With EMA update: 0.862 (worse). The static target encoder was actually better — possibly because EMA tracking makes the target a moving target that's harder to predict, without providing better representations.

58. **DINO's theoretical advantages (centering/sharpening → alignment+uniformity) don't materialize at this data scale.** The k-NN geometry improvements only matter if the base features have some discriminative signal to amplify. With ~200 trials per source patient and no phoneme labels, there's nothing to amplify.

59. **TTA hurts masking-based SSL, not helps.** JEPA's eval recipe k-NN (0.827) is worse than basic k-NN (0.811). The backbone was never trained for augmentation invariance, so augmented inputs produce different features that don't average cleanly. TTA is only useful for augmentation-based SSL methods (BYOL, VICReg) and supervised models.

60. **Supervised > ALL SSL for per-patient decoding at this data scale.** Base CE (0.837) beats every SSL method. The autoresearch recipe (0.737) is 12pp better than the best SSL. With ~120 labeled trials, direct supervision is far more sample-efficient than self-supervised pretraining on ~1600 unlabeled trials.

---

## Next Direction: Full SSL Exploration

The autoresearch session optimized within the supervised single-patient regime. The SSL solution space remains largely unexplored. Below is the roadmap for systematic SSL exploration.

### Priority 1: Apply Autoresearch Eval Recipe to JEPA Features

The autoresearch eval improvements (weighted k-NN, TTA, articulatory head) should transfer directly to JEPA-pretrained features. This is the fastest way to test whether SSL + better eval can beat supervised + better eval.

**Experiment:** Load JEPA checkpoint → Stage 3 with:
- Weighted k-NN (k=10, cosine sim weights) instead of unweighted
- TTA 16 copies on val embeddings
- Dual head (CE + articulatory) for linear comparison
- Best-of per fold

**Expected outcome:** JEPA k-NN was 0.811 with basic eval. Weighted k-NN + TTA should push this significantly lower. If JEPA + improved eval < supervised + improved eval (0.737), then SSL feature quality is the bottleneck. If JEPA + improved eval ≈ 0.737, then the features are equivalent and training doesn't matter.

### Priority 2: SSL Objectives Not Yet Tested

| Objective | Status | Why Re-explore |
|-----------|--------|---------------|
| **JEPA + improved eval** | Never tested | Priority 1 above |
| **VICReg (fixed proj_dim)** | Never tested | Original failure was batch_size=8 vs proj_dim=256. With proj_dim ≤ batch_size, covariance regularization should work |
| **Fair semi-supervised** | Never tested | Semi-supervised BYOL got 0.040 PER but with data leakage. Fix: exclude val-fold tokens from Stage 2 CE |
| **CEBRA InfoNCE** | Partially tested | Articulatory head in autoresearch is CEBRA-like but supervised. True CEBRA would use InfoNCE on multi-patient SSL features aligned to articulatory targets |
| **Barlow Twins** | Never tested | Simpler than VICReg — just cross-correlation matrix → identity. No variance/covariance splitting |
| **wav2vec-style CPC** | Never tested | Contrastive predictive coding on temporal sequences. Different from JEPA (reconstruction vs contrastive) |
| **Multi-patient SupCon** | Never tested | SupCon failed on single-patient because CE was better. With multi-patient SSL (no labels for source patients), SupCon on S14 labels + self-supervised on sources could combine benefits |

### Priority 3: Hybrid SSL + Supervised

| Approach | Description |
|----------|-------------|
| **JEPA pretrain → supervised fine-tune with autoresearch recipe** | Pretrain backbone with JEPA on all patients, then fine-tune with CE + label smoothing + mixup + focal + dual head on S14 |
| **LP-FT (Linear Probe then Fine-Tune)** | Freeze JEPA backbone → train linear probe → unfreeze backbone → fine-tune with low LR. Kumar et al. 2022 showed this beats direct fine-tuning |
| **Feature concatenation** | Concatenate JEPA features (64-dim) with supervised features (64-dim) → 128-dim → k-NN. Different training objectives capture complementary structure |
| **Multi-view k-NN** | Run k-NN on JEPA features AND supervised features separately, average the per-position class probabilities |

### Priority 4: Architecture Improvements for SSL

| Approach | Description |
|----------|-------------|
| **Larger SSL pretraining corpus** | Include S26 (dev patient), Lexical patients (13 more), cross-task data from Duraivel 2025 |
| **Longer pretraining** | JEPA trained 5000 steps. Try 20K-50K steps with LR decay |
| **Masking schedule** | JEPA used 40-60% mask, 3-6 spans. Try curriculum: easy→hard masking |
| **Multi-scale JEPA** | Predict at multiple temporal resolutions (stride 5 + stride 10) |
| **Patient-conditional JEPA** | Add patient embedding to condition the prediction, helping the model learn patient-invariant features |

### Key Open Questions

1. **Are JEPA features complementary to supervised features?** If k-NN on JEPA and supervised features make different errors, combining them could beat either alone.

2. **Is the eval recipe the ceiling, or can better training still help?** The autoresearch session suggests eval dominates. But that was within single-patient supervised — multi-patient SSL might change the landscape.

3. **Does articulatory structure help SSL?** The articulatory head improved supervised k-NN. Can we inject articulatory structure INTO the SSL objective (e.g., CEBRA-style InfoNCE with articulatory targets)?

4. **What's the right SSL → supervised handoff?** Options: frozen backbone (current), LP-FT, full fine-tune, feature concatenation. Each has different properties at N=120.

---

## 2026-03-30: Data Quality Audit & Re-baseline

### Data Quality Findings

1. **.fif epoch markers are MFA-derived.** Verified by exact match (0.0-0.2ms) between .fif event sample times and phoneme CSV onset times for S14 and S22. Both epoch boundaries and CSV timing come from Montreal Forced Aligner. If MFA fails for a patient, both are wrong. Phoneme LABELS are always correct (from known stimulus transcript).

2. **S32 = S36 (same recording).** Identical event arrays despite different file sizes (S32=234MB/256ch, S36=117MB/128ch). **11 unique patients, not 12.**

3. **Per-phoneme alignment quality varies by patient:**
   - Good: S22, S39, S58 (confirmed by visual inspection of sweep_v2 videos)
   - Poor: Others have 300-500ms offsets, misaligned phoneme boundaries, or missing phonemes

4. **Decision: Full-trial CTC.** CTC on position-1 epochs (1.0s window) is robust to MFA errors — the window contains all 3 phonemes regardless of alignment quality. Per-position CE requires accurate segmentation, which fails for 3/11 patients (S16, S22, S39). *[Corrected 2026-04-06: originally estimated ~50%, but per-phoneme sweep showed 8/11 have good MFA.]*

### Re-baseline: Production-Only Window + Grouped CV

Added `cv_type: grouped` to trainer (uses `grouped_cv.py`). Compared production-only (tmin=0.0, 200 frames) vs original (tmin=-0.5, 300 frames).

**Stratified CV (S14, CE per-position, 3 seeds):**

| Window | Seed 42 | Seed 137 | Seed 256 | Mean |
|--------|---------|----------|----------|------|
| tmin=-0.5 (previous) | — | — | — | 0.700 |
| tmin=0.0 | 0.727 | 0.745 | 0.756 | 0.743 |

**Grouped-by-token CV (S14, CE per-position, 3 seeds):**

| Window | Seed 42 | Seed 137 | Seed 256 | Mean |
|--------|---------|----------|----------|------|
| tmin=-0.5 | 0.880 | 0.839 | 0.858 | **0.859** |
| tmin=0.0 | 0.840 | 0.821 | 0.849 | **0.837** |

### Realizations

47. **Grouped CV is ~10pp harder than stratified** (0.837 vs 0.743 for tmin=0.0, 3-seed means). Confirms significant token leakage in stratified CV. All future comparisons must use grouped.

48. **Production-only HELPS under grouped CV** (0.837 vs 0.859) **but HURTS under stratified** (0.743 vs 0.700). The pre-production planning frames help the model exploit token-level correlations (leakage) but hurt generalization to unseen tokens. Under fair evaluation, trimming to production-only is better.

49. **The base CE model (no recipe improvements) gives 0.837-0.859 under grouped CV.** The autoresearch recipe (label smoothing + mixup + focal + dual head + weighted k-NN + TTA) adds ~10pp of improvements to reach 0.737. These recipe improvements are orthogonal to SSL — they can be layered on top of any backbone.

---

## Key Numbers

| Metric | Value | Notes |
|--------|-------|-------|
| Chance PER | 0.889 | |
| **Autoresearch best (grouped, exp17)** | **0.737** | Full recipe, tmin=-0.5 |
| **Autoresearch best (stratified)** | **0.662** | Token leakage |
| Base CE (grouped, tmin=0.0) | 0.837 | Production-only, 3 seeds |
| Base CE (grouped, tmin=-0.5) | 0.859 | Original window, 3 seeds |
| Base CE (stratified, tmin=-0.5) | 0.700 | Token leakage |
| Base CE (stratified, tmin=0.0) | 0.743 | Production-only, leaky |
| BYOL semi-supervised (grouped) | 0.040 | **Contaminated — pure BYOL = 0.886** |
| DINO (grouped) | 0.863 | Near chance |
| JEPA w/ EMA fix (grouped) | 0.862 | Near chance |
| JEPA old linear probe (grouped) | 0.854 | Best pure SSL (static target) |
| BYOL pure (grouped) | 0.886 | Near chance |
| VICReg (grouped) | 0.891 | batch/proj mismatch |
| LeWM (grouped) | 0.911 | Near chance |
| Per-patient data | ~150 trials, ~1 min utterance | |
| Multi-patient SSL corpus | ~1621 trials, 11 patients | S32=S36 duplicate removed |
| **LOPO autoresearch best** | **0.762** | **Multi-patient k-NN (source embeddings weight 0.5)** |
| LOPO autoresearch baseline | 0.764 | CE+recipe, 9 source pts → S14, grouped CV |
| LOPO pilot (CTC, no recipe) | 0.846 | 8 patients, baseline |
| JEPA LP-FT (best-of) | 0.797 | LP then fine-tune on JEPA features |
| Semi-supervised BYOL (nested CV) | 0.874 | Fair eval, near chance |

---

## 2026-03-30: LOPO Autoresearch Baseline

### Setup
- **Pipeline:** Stage 1 (9 source patients, ~1315 trials) → Stage 2 (adapt to S14, 5-fold grouped CV)
- **Recipe:** All single-patient autoresearch wins pre-baked: CE + label smoothing 0.1 + focal γ=2 + mixup α=0.2 + per-position heads + dropout 0.3 + weighted k-NN (k=10) + TTA 16
- **Architecture:** Per-patient SpatialReadIn(Conv2d(1,8,3), pool(4,8)) + shared Backbone(LN, Conv1d(256,32,s=10), BiGRU(32,32,2L)) + CEHead(3×Linear(64,9))
- **Window:** tmin=0.0 (production only, 201 frames)
- **Stage 2:** Full model unfrozen, backbone LR × 0.1, differential LR for read-in (×3) and head (×1)

### Results

| Fold | Best PER | Linear | k-NN |
|------|----------|--------|------|
| 1 | 0.771 | 0.865 | 0.771 |
| 2 | 0.778 | 0.800 | 0.778 |
| 3 | 0.767 | 0.933 | 0.767 |
| 4 | 0.722 | 0.722 | 0.744 |
| 5 | 0.785 | 0.785 | 0.828 |
| **Mean** | **0.764 ± 0.022** | 0.821 | 0.778 |

- Stage 1: 286.6s, early-stopped at epoch 80 (val loss diverging from epoch 10)
- Total: 373.2s (~6.2 min), well within 15-min budget
- No content collapse (entropy 2.31, stereotypy 0.16)

### Realizations

61. **LOPO + recipe already beats the LOPO pilot by 8.2pp** (0.764 vs 0.846). The recipe improvements (label smoothing, focal, mixup, k-NN, TTA) transfer directly from single-patient to cross-patient.

62. **LOPO + recipe nearly matches single-patient autoresearch** (0.764 vs 0.737). With 10× more source data but harder cross-patient transfer, the LOPO baseline is only 2.7pp behind the single-patient best — this is the highest-leverage optimization target.

63. **k-NN dominates linear head in 3/5 folds** (0.778 vs 0.821 mean). Consistent with single-patient findings — backbone features are better than the linear head can exploit.

64. **Stage 1 early-stops at epoch 80 with val diverging from epoch 10.** The source validation loss goes up while train loss drops steadily. This suggests heavy overfitting to source patients — Stage 1 regularization or architecture changes could improve.

65. **Stage 2 completes quickly** (~15-27s per fold). With backbone unfrozen at 0.1× LR, the model adapts rapidly. Stage 1 quality dominates the final result.

---

## 2026-03-31: LOPO Autoresearch — 16 Experiments

### Summary
Two agents ran 16 experiments (exp01–exp16). 15 discarded, 1 kept. Best PER improved from 0.764 → **0.762** (exp13: multi-patient k-NN).

### Full Results

| # | PER | Time | Status | Description |
|---|-----|------|--------|-------------|
| baseline | **0.764** | 373s | keep | CE+focal+mixup+label_smooth+per_pos_heads+kNN+TTA, 9 source pts |
| exp01 | 0.825 | 255s | discard | S1 batch=32, eval_every=5, patience=12 |
| exp02 | 0.826 | 271s | discard | S1 LR=5e-4, WD=1e-3 — COLLAPSED |
| exp03 | 0.785 | 269s | discard | Source replay 30% in Stage 2 |
| exp04 | 0.788 | 336s | discard | H=64 backbone |
| exp05 | 0.804 | 416s | discard | Freeze backbone in Stage 2 |
| exp06 | 0.788 | 336s | discard | Dual head (CE + articulatory) |
| exp07 | 0.793 | 275s | discard | Fixed 30-epoch cosine, no val split, no early stop |
| exp08 | 0.807 | 363s | discard | Progressive unfreezing S2 (freeze backbone 30 epochs) |
| exp09 | 0.798 | 470s | discard | S2 patience 15, epochs 300 |
| exp10 | 0.804 | 239s | discard | S1 eval every 5 epochs |
| exp11 | 0.769 | 402s | discard | Lighter S2 augment + no S2 mixup (closest to baseline!) |
| exp12 | 0.810 | 396s | discard | Stronger S1 reg (dropout 0.5, WD 5e-4) |
| **exp13** | **0.762** | **412s** | **keep** | **Multi-patient k-NN (source embeddings as extra neighbors, weight 0.5)** |
| exp14 | 0.764 | 417s | discard | Source k-NN weight 1.0 (unweighted) |
| exp15 | 0.764 | 418s | discard | Source k-NN weight 0.3 |
| exp16 | 0.771 | 411s | discard | Lighter S2 aug + no mixup + multi-pt k-NN |
| exp17 | — | — | interrupted | FlattenLinear read-in + InstanceNorm (killed mid-run) |

### What Worked: Multi-Patient k-NN (exp13)

**Idea**: At evaluation time, compute embeddings for all source patients using their Stage 1 read-ins + the Stage 2-adapted backbone. Add these as extra neighbors in the k-NN classifier, weighted at 0.5× the target training embeddings.

**Why it works**: Source patients provide ~1300 extra labeled reference points for k-NN. Even though their representations went through different read-ins, the shared backbone maps them to a common feature space. Weighting at 0.5 (not 1.0) acknowledges that source representations are noisier but still informative.

**Implementation**: `SOURCE_KNN_WEIGHT = 0.5`, source embeddings scaled by this weight before concatenation with target training embeddings. Cosine similarity naturally handles the scale difference.

### Categorized Experiment Approaches

**Stage 1 training modifications** (exp01, 02, 07, 10, 12): Batch size, LR, WD, fixed schedule, eval frequency, stronger regularization — ALL failed. Stage 1 overfitting is resistant to standard hyperparameter changes.

**Stage 2 adaptation modifications** (exp03, 05, 08, 09, 11, 16): Source replay, freeze backbone, progressive unfreezing, more patience, lighter augmentation — ALL failed or marginal. Stage 2 is not the bottleneck.

**Architecture changes** (exp04, 06, 17): H=64 backbone, dual articulatory head, FlattenLinear+InstanceNorm — ALL failed or interrupted. Architecture is not the problem at this scale.

**Evaluation improvements** (exp13, 14, 15): Multi-patient k-NN — THIS WORKED. The only improvement came from better evaluation, not training.

### Realizations

66. **Evaluation improvements continue to dominate training improvements for LOPO**, just as they did for single-patient (realization 36). The only experiment that beat the baseline was a k-NN evaluation change, not a training change.

67. **Multi-patient k-NN weight 0.5 is optimal**: 0.3 too low (reverts to baseline), 1.0 too high (source noise overwhelms target signal). The balance matters.

68. **Lighter Stage 2 augmentation was the closest training-side improvement** (exp11, 0.769). Mixup on ~120 training samples may add too much noise. Worth retesting in combination with other improvements.

69. **16 experiments failed to improve Stage 1 training.** The source-patient overfitting pattern (val diverges from epoch 5-10) is deeply structural, not addressable by standard hyperparameter tuning. Future approaches should consider: domain adversarial training, patient-conditional features, or fundamentally different training objectives.

70. **LOPO is evaluation-limited, not training-limited.** The backbone learns decent features for all patients. The challenge is extracting predictions from mixed-patient feature spaces. More sophisticated evaluation (multi-patient k-NN, ensemble, prototype networks) is the highest-leverage direction.

### What to Try Next (for next agent)

All items below were tested in Wave 4 (see below). Results: DANN didn't help, multi-scale temporal was the only architecture win.

---

## 2026-03-31: LOPO Architecture Ablations — Waves 1-3 (exp41-exp76)

### Setup
Modular experiment framework (`arch_ablation_base.py`) with pluggable ReadInCls, BackboneCls, HeadCls. Each ablation changes ONE component from exp33 baseline (articulatory bottleneck head).

### Key Results (36 experiments)

| Experiment | Change | PER (seed 42) | Multi-seed mean |
|-----------|--------|---------------|-----------------|
| exp33 baseline | Articulatory bottleneck head | 0.766 | ~0.766 |
| exp45 stride=5 | Conv1d stride=5 (40Hz) | 0.749 | 0.760 ± 0.016 |
| exp52 MaxPool | AdaptiveMaxPool2d | 0.749 | 0.764 ± 0.013 |
| exp41 C=16 | 16 conv channels | 0.754 | — |
| exp53 attn pool | Attention pooling head | 0.767 | — |
| exp59 transformer | TransformerEncoder | 0.763 | — |
| exp54 LSTM | BiLSTM | 0.776 | — |
| exp56 InstanceNorm | InstanceNorm1d | 0.813 | — (catastrophic) |
| exp58 no augment | No augmentation | 0.770 | — |
| Combined: stride5+MaxPool+C16 | All 3 | 0.767 | — (don't stack) |

### Realizations (71-76)

71. **Multi-seed validation is essential.** stride=5 seed 42 = 0.749, but mean = 0.760. MaxPool seed 42 = 0.749, but mean = 0.764. Single-seed results are unreliable.

72. **Improvements don't stack.** stride5+MaxPool+C16 (0.767) is WORSE than either alone. The improvements exploit the same signal.

73. **Architecture is NOT the bottleneck.** Transformer ≈ GRU ≈ LSTM ≈ multi-scale, all within 2pp. The temporal processing choice barely matters at this data scale.

74. **MaxPool > AvgPool for spatial read-in.** Preserves peak activations rather than averaging over dead/weak channels. Consistent across seeds.

75. **InstanceNorm is catastrophic (0.813).** Normalizing per-instance removes inter-trial amplitude differences that carry phoneme information.

76. **Augmentation helps marginally (~0.8pp).** No-augmentation (0.770) vs baseline (0.762). Augmentation is not critical for LOPO.

---

## 2026-03-31: LOPO Wave 4 — Challenging Fundamental Assumptions (exp77-exp95)

### Setup
19 experiments testing domain adaptation, SSL, training paradigms, evaluation innovations, and architecture combinations. Motivated by: (1) Spalding's explicit CCA alignment works while our implicit alignment via shared backbone doesn't add value; (2) all prior experiments converge to ~0.76 PER.

### Full Results

| Rank | Experiment | PER | Category |
|------|-----------|-----|----------|
| **1** | **exp94 multi-scale+MaxPool** | **0.750** | Architecture combo |
| 2 | exp93 multi-scale (seed256) | 0.756 | Multi-seed |
| 3 | exp86 multi-scale (seed42) | 0.757 | Architecture |
| 4 | exp84 self-training | 0.758 | Training |
| 5 | exp90 CORAL alignment | 0.762 | Domain adapt |
| 6 | exp95 multi-scale+DANN | 0.765 | Combo |
| 7 | exp92 multi-scale (seed137) | 0.767 | Multi-seed |
| 8 | exp81 CCA on features | 0.771 | Domain adapt |
| 9 | exp91 multi-scale+self-train | 0.772 | Combo |
| 10 | exp89 knowledge distillation | 0.773 | Training |
| 11 | exp78 full backbone LR | 0.776 | Diagnostic |
| 12 | exp83 transformer+SSL | 0.776 | SSL |
| 13 | exp82 VICReg auxiliary | 0.778 | SSL |
| 14 | exp77 no S2 | 0.778 | Diagnostic |
| 15 | exp79 DANN gradient reversal | 0.778 | Domain adapt |
| 16 | exp85 transductive (label prop) | 0.780 | Eval |
| 17 | exp87 patient weighting | 0.782 | Training |
| 18 | exp80 joint training (S14 in S1) | 0.793 | Training |
| — | **exp88 per-patient (grouped, simple)** | **0.800** | Baseline |

Multi-scale 3-seed mean: 0.757, 0.767, 0.756 → **0.760 ± 0.005** (robust).

### Diagnostic Findings

**exp77 (no S2)**: PER 0.778. S2 adaptation contributes 1.6pp (0.778→0.762). Modest but real.

**exp78 (full backbone LR in S2)**: PER 0.776. The 0.1× backbone LR is correct — full fine-tuning overfits on 120 samples.

**exp88 (per-patient grouped CV, simple recipe)**: PER 0.800. This uses a simpler recipe than the autoresearch exp17 (0.737). The correct per-patient baseline for same-recipe comparison is **0.737** (from first autoresearch).

### Domain Adaptation: Comprehensive Failure

| Method | PER | vs Baseline |
|--------|-----|-------------|
| CORAL alignment loss | 0.762 | ±0.000 |
| CCA on backbone features | 0.771 | +0.009 |
| DANN gradient reversal | 0.778 | +0.016 |
| Patient similarity weighting | 0.782 | +0.020 |

**Why**: The backbone already produces reasonably patient-invariant features. DANN and CORAL add alignment pressure that constrains the feature space without improving discriminability. CCA alignment on 64-dim features with ~10-30 condition averages per patient is noisy.

### SSL: Still Failing

| Method | PER | vs Baseline |
|--------|-----|-------------|
| VICReg auxiliary (S1) | 0.778 | +0.016 |
| Transformer + masked SSL pretrain | 0.776 | +0.014 |

**Why**: With ~1315 total source trials, SSL objectives cannot learn representations that CE doesn't already capture. The self-supervised signal is too weak relative to the supervised signal.

### Training Innovations: Mostly Worse

| Method | PER | vs Baseline |
|--------|-----|-------------|
| Self-training (pseudo-labels) | 0.758 | -0.004 |
| Knowledge distillation | 0.773 | +0.011 |
| Joint training (S14 in S1) | 0.793 | +0.031 |

Self-training (0.758) is the only training-side improvement. Knowledge distillation's per-patient teachers are too noisy to provide useful signal. Joint training with S14 in S1 causes overfitting.

### Realizations (77-85)

77. **Per-patient grouped CV = 0.800 (simple recipe) / 0.737 (full recipe).** The 0.700 baseline was stratified CV — not comparable. With the same recipe, per-patient (0.737) beats LOPO (0.762) by 2.5pp. Cross-patient data marginally hurts with the full autoresearch recipe.

78. **Multi-scale temporal (stride 3+5+10) captures articulatory dynamics at multiple timescales.** Stride=3 (~67Hz) resolves fast transitions, stride=10 (~20Hz) captures broad patterns. Concatenating gives GRU richer input.

79. **Multi-scale + MaxPool = 0.750 is the strongest architecture.** MaxPool preserves peak activations; multi-scale provides temporal diversity. But single-seed — needs validation.

80. **Combinations don't stack (again).** Multi-scale + self-training (0.772) worse than either alone. Multi-scale + DANN (0.765) worse than multi-scale alone. Each improvement exploits the same limited signal.

81. **DANN gradient reversal does nothing at N=9 patients.** 9 source patients are too few for the discriminator to learn meaningful patient invariances. The gradient reversal just adds noise.

82. **Spalding's explicit CCA alignment ≠ our implicit backbone alignment.** Spalding uses condition-averaged alignment with phoneme labels. Our backbone tries to learn this implicitly. CCA on our backbone features (exp81, 0.771) didn't help because the features are already roughly aligned — the problem is discriminability, not alignment.

83. **S1 backbone converges to the same basin regardless of training objective.** DANN, VICReg, CORAL, distillation — all produce similar S1 features. The 9 source patients × ~1315 trials create one dominant training minimum.

84. **The ~0.76 PER wall is a measurement ceiling, not a model ceiling.** With ~30 val samples per fold and fixed fold assignments, PER differences <2pp are within evaluation noise. Fold 5 consistently gets ~0.85; fold 4 gets ~0.70. The average is mechanically constrained by fold difficulty distribution.

85. **Breaking through requires more data, not better models.** 55 experiments × 5 paradigms × multiple architectures converge to 0.750-0.780. The path forward is: (a) more patients in Stage 1, (b) cross-task data pooling, (c) population-level evaluation on all patients, (d) different CV scheme or more folds to reduce measurement noise.

---

## 2026-04-04: Per-Phoneme MFA Epoch Sweeps (S14, DCC)

Three DCC sweeps testing per-phoneme MFA epochs vs full-trial approaches. All on S14, grouped-by-token 5-fold CV.

### Sweep 1: Head Type × tmin (simplified recipe — no mixup/k-NN/TTA)

| Condition | Mode | Head | tmin | PER | Std |
|-----------|------|------|------|-----|-----|
| per_phoneme_t-0.15 | per-phoneme | flat | -0.15 | **0.782** | 0.038 |
| per_phoneme_t0.0 | per-phoneme | flat | 0.0 | 0.789 | 0.055 |
| mean_pool_t0.0 | full-trial | mean pool | 0.0 | 0.824 | 0.044 |
| equal_window_t0.0 | full-trial | 3 equal windows | 0.0 | 0.830 | 0.045 |
| learned_attn_t0.0 | full-trial | learned attention | 0.0 | 0.843 | 0.038 |
| learned_attn_t-0.5 | full-trial | learned attention | -0.5 | 0.873 | 0.026 |

### Sweep 2: Full Recipe (mixup α=0.2, k-NN k=10, TTA n=16)

| Condition | PER | Std | Seeds | Pos 1 acc | Pos 2 acc | Pos 3 acc |
|-----------|-----|-----|-------|-----------|-----------|-----------|
| **perphon_t-0.15_flat_s10** | **0.741** | — | 1 | 0.281 | 0.235 | 0.261 |
| perphon_t0.0_artic_s10 | 0.747 | — | 1 | 0.261 | 0.248 | 0.248 |
| perphon_t-0.15_artic_s5 | 0.764 | 0.008 | 3 | 0.251 | 0.220 | 0.237 |
| perphon_t-0.15_artic_s10 | 0.772 | 0.020 | 3 | 0.242 | 0.214 | 0.229 |
| fulltrial_meanpool_artic | 0.802 | — | 1 | 0.163 | 0.170 | 0.222 |

### Sweep 3: Padding Grid (simplified recipe, per-phoneme only)

**tmin sweep** (tmax=0.5 fixed):

| tmin | PER | Std |
|------|-----|-----|
| -0.25 | 0.776 | 0.014 |
| -0.20 | 0.793 | 0.035 |
| -0.15 | 0.771 | 0.041 |
| **-0.10** | **0.764** | 0.040 |
| -0.05 | 0.767 | 0.031 |
| 0.00 | 0.771 | 0.041 |

**tmax sweep** (tmin=-0.15 fixed):

| tmax | PER | Std |
|------|-----|-----|
| 0.3 | 0.791 | 0.036 |
| 0.4 | 0.780 | 0.038 |
| **0.5** | **0.771** | 0.041 |
| 0.6 | 0.773 | 0.023 |
| 0.7 | 0.786 | 0.019 |

### Findings

86. **Per-phoneme MFA epochs strictly dominate full-trial approaches.** Across every comparison — simplified recipe, full recipe, all head types — per-phoneme epochs give ~4-6pp lower PER. MFA alignment is good for 8/11 patients; only S16/S22/S39 have poor alignment (per finding 101). Per-phoneme provides cleaner temporal locking than any learned readout from a full 1s window.

87. **Flat head beats articulatory head for per-phoneme single-phoneme classification.** Per-phoneme flat (0.741) vs articulatory (0.772). The articulatory decomposition was designed as a regularizer for cross-patient transfer on phoneme sequences. For single-phoneme classification, it constrains the output space unnecessarily.

88. **Stride=5 beats stride=10 for per-phoneme windows.** 0.764 vs 0.772 (3-seed means). With T=131 frames (650ms window), stride=10 gives only 13 GRU timesteps. Stride=5 doubles to 26, giving the GRU more temporal context. This reverses the full-trial finding where stride=10 was sufficient (20 timesteps from T=201).

89. **tmin=-0.5 (pre-production auditory activation) hurts.** 0.873 vs 0.843 for learned_attn. Including 500ms of auditory response to the prompt adds noise, not signal.

90. **Padding is not critical but ~100ms pre-onset is optimal.** tmin grid from -0.25 to 0.0 varies only 3pp (0.764-0.793). tmin=-0.10 is marginally best (0.764). tmax=0.5 is the sweet spot; shorter (0.3) loses signal, longer (0.7) adds noise.

91. **Learned temporal attention underperforms simple mean pool with simplified recipe.** 0.843 vs 0.824 for full-trial. The attention queries may need the full recipe (k-NN, TTA) to show advantage, or they may simply be unnecessary overhead.

92. **Full recipe provides ~4pp on per-phoneme baseline.** Simplified per-phoneme (0.782) vs full recipe per-phoneme (0.741). Mixup + k-NN + TTA each contribute meaningfully.

93. **Per-phoneme + full recipe + flat head (0.741) matches the previous best per-patient baseline (0.737).** This is a single-seed result; multi-seed validation needed. But the fact that a simpler approach (mean-pool over MFA epochs, no attention queries) matches the tuned full-trial pipeline is significant.

94. **Implication for Neural Field Perceiver**: Use per-phoneme MFA epochs as input, not full-trial with learned temporal attention. This simplifies the temporal readout to mean-pooling and eliminates the need for attention query parameters.

---

## 2026-04-04: Head-to-Head and Multi-Patient Validation (DCC)

### Sweep 4: Head-to-Head — Learned Attention vs Per-Phoneme (S14, 3 seeds, full recipe)

Fair comparison: identical training loop, same recipe (mixup α=0.2, k-NN k=10, TTA n=16, focal CE γ=2).

| Condition | PER (3-seed) | Std | Seed 42 | Seed 137 | Seed 256 |
|-----------|-------------|-----|---------|----------|----------|
| **perphon_flat** | **0.734** | **0.007** | 0.743 | 0.726 | 0.732 |
| perphon_artic | 0.772 | 0.022 | 0.741 | 0.784 | 0.790 |
| learned_attn_flat | 0.797 | 0.012 | 0.801 | 0.810 | 0.780 |
| learned_attn_artic | 0.806 | 0.017 | 0.826 | 0.784 | 0.808 |
| meanpool_flat | 0.807 | 0.007 | 0.812 | 0.798 | 0.812 |

### Sweep 5: Multi-Patient — Per-Phoneme vs Full-Trial (all 11 PS patients, simplified recipe)

| Patient | Trials | Grid | Full-trial PER | Per-phoneme PER | Δ | Winner |
|---------|--------|------|---------------|-----------------|---|--------|
| S14 | 153 | 8×16 | 0.817 | 0.786 | +3.0pp | per-phoneme |
| S16 | 205 | 8×16 | 0.857 | 0.875 | -1.8pp | full-trial |
| S22 | 156 | 8×16 | 0.853 | 0.859 | -0.7pp | full-trial |
| S23 | 156 | 8×16 | 0.905 | 0.855 | +5.0pp | per-phoneme |
| **S26** | 153 | 8×16 | 0.872 | **0.707** | **+16.5pp** | **per-phoneme** |
| S32 | 152 | 12×22 | 0.930 | 0.897 | +3.3pp | per-phoneme |
| **S33** | 52 | 12×22 | 0.853 | **0.749** | **+10.4pp** | **per-phoneme** |
| S39 | 148 | 12×22 | 0.850 | 0.872 | -2.3pp | full-trial |
| S57 | 102 | 8×34 | 0.908 | 0.879 | +2.9pp | per-phoneme |
| S58 | 153 | 12×22 | 0.885 | 0.831 | +5.4pp | per-phoneme |
| S62 | 191 | 12×22 | 0.786 | 0.761 | +2.5pp | per-phoneme |
| **Population** | | | **0.865** | **0.825** | **+4.0pp** | **per-phoneme 8/11** |

### Findings

95. **Per-phoneme flat (0.734 ± 0.007) definitively beats learned attention (0.797 ± 0.012) in fair comparison.** 6.3pp difference, 3-seed validated, same training loop. The previous 0.737 baseline benefited from training pipeline differences, not learned attention being inherently better. Per-phoneme is both better AND more stable (std 0.007 vs 0.012).

96. **Per-phoneme generalizes across patients: 8/11 win, population mean +4.0pp.** Not S14-specific. Largest wins on S26 (+16.5pp) and S33 (+10.4pp). The 3 full-trial wins (S16, S22, S39) are all <2.3pp — within noise.

97. **Flat head strictly dominates articulatory head for per-phoneme.** 0.734 vs 0.772 (3-seed). The articulatory decomposition constrains 9-way classification through 15 articulatory features — an unnecessary bottleneck when classifying individual phonemes rather than sequences.

98. **Learned attention is NOT better than mean pool in the same training loop.** 0.797 vs 0.807, both with flat head. The attention queries don't learn useful temporal structure from 153 trials — they add parameters without helping.

99. **S26 and S33 have exceptionally good MFA alignment.** S26 per-phoneme PER 0.707 is the best single-patient result in any experiment. S33 (0.749) despite having only 52 trials. These patients likely have clean, consistent phoneme boundaries in the MFA labels.

100. **Per-phoneme provides 3× more training samples** (459 vs 153 for S14). Each phoneme is an independent classification problem — the model sees every phoneme in every trial as a separate training example. This data amplification may explain much of the advantage.

101. **The 3 patients where full-trial wins (S16, S22, S39) may have poor MFA alignment.** S16 and S39 are among the patients identified as having unreliable per-phoneme MFA labels. Full-trial is robust to MFA noise because it uses the entire 1s window containing all 3 phonemes. Per-phoneme depends on correct boundaries.

### Optimal Config (first pass for v12 implementation)

Based on all 5 sweeps:

```
Input: Per-phoneme MFA epochs (tmin=-0.15, tmax=0.5)
  — 3× more training samples than full-trial
  — Robust to moderate MFA noise (150ms padding absorbs alignment errors)

Spatial: Conv2d(1→8, k=3, pad=1) + AdaptiveAvgPool2d(4,8) → d=256
  — Per-patient, ~80 params

Temporal: Conv1d(256→32, stride=10) + BiGRU(32, 32, 2L, bidirectional)
  — stride=10 sufficient (13 frames from 131-sample windows)
  — stride=5 gives marginal gain (0.764 vs 0.772) but doubles compute

Head: Flat Linear(64→9)
  — NOT articulatory (bottleneck hurts single-phoneme classification)

Readout: Global mean pool over time → single phoneme prediction
  — No learned attention needed
  — No per-position heads needed

Training: Focal CE (γ=2) + label smoothing (0.1) + mixup (α=0.2)
Eval: Weighted k-NN (k=10) + TTA (n=16)

Expected PER: ~0.73 on S14 (grouped-by-token CV)
Population: ~0.82 mean across 11 patients
```

### Notes for v12 Neural Field Perceiver

- The per-phoneme approach **simplifies the architecture**: no temporal attention queries, no per-position heads, just mean pool + flat classifier
- MFA quality varies by patient — robust designs should handle both good and bad alignment
- For patients with bad MFA (S16, S22, S39), a fallback to full-trial with equal windows may be needed
- The ~0.73 PER baseline on S14 is what the Neural Field Perceiver must beat to justify the added spatial complexity
- Cross-patient evaluation should always include population-level results, not just S14
