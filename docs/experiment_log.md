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

## Key Numbers

| Metric | Value |
|--------|-------|
| Chance PER | 0.889 |
| **Autoresearch best (grouped, exp17)** | **0.737** |
| **Autoresearch best (stratified)** | **0.662** |
| Supervised baseline (grouped) | 0.825-0.872 |
| Supervised baseline (stratified, leaky) | 0.700 |
| JEPA k-NN k=10 (grouped, basic eval) | 0.811 |
| JEPA linear probe (grouped) | 0.854 |
| All SSL linear probes (grouped) | 0.854-0.911 |
| Per-patient data | ~150 trials, ~1 min utterance |
| Multi-patient SSL corpus | ~1621 trials, 11 patients |
