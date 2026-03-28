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

## Planned Experiments

### Phase 1: Method B (Neural-only SSL)
- **Stage 2:** Masked span prediction on source patients' unlabeled trials (2,500 steps)
- **Stage 3:** Freeze backbone → CE head on S14 → grouped-CV PER
- **Gate 1:** B < E (0.911) → pretraining helps

### Phase 2: Method C (Smooth AR synthetic → neural)
- **Stage 1:** Synthetic pretraining on smooth AR data (2,500 steps)
- **Stage 2:** Neural adaptation (2,500 steps)
- **Stage 3:** Same evaluation
- **Gate 2:** C < B → synthetic pretraining adds value

### Phase 2: Method A (Switching LDS synthetic → neural)
- Same as C but with switching LDS generator
- **Gate 2b:** A < C → structured dynamics help

---

## Key Numbers for Reference

| Metric | Value | Context |
|--------|-------|---------|
| Chance PER (9-class, 3-pos) | 0.889 | Random guessing |
| Stratified CV baseline | 0.700 | Known, but has token leakage |
| Grouped CV baseline (Phase 0) | 0.835 | Fair evaluation, thin signal |
| Spalding 2025 (PCA+SVM) | ~0.69 bal acc | Different metric, stratified CV |
| Per-patient data | ~150 trials, ~1 min utterance | Extreme data bottleneck |
