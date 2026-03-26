# NCA Pretraining for Data-Scarce Neural Decoding — Design Spec

**Date**: 2026-03-26 (v4 — revised per architectural stress test)
**Author**: Ben Tang, Cogan Lab (Duke)
**Status**: Draft

## 1. Problem Statement

Intraoperative uECOG speech decoding faces an extreme data bottleneck: ~1 minute of utterance per patient (~150 trials), 128-256 channels at 200Hz. Current supervised approaches train ~60K parameters from scratch on this thin data and overfit heavily. No synthetic pretraining method has been tested for acute intra-op uECOG phoneme-sequence decoding at this data scale. SSL methods exist for speech-BCI (wav2vec ECoG, BrainBERT), but none target the specific regime of <200 trials on rigid micro-ECoG arrays with known 2D spatial structure.

## 2. Core Idea

Pretrain an encoder-predictor on synthetic spatiotemporal grid dynamics, then adapt to real cortical recordings, then fine-tune a lightweight classification head per patient.

**The NCA-neural analogy**: Both NCA and uECOG produce 2D grids evolving over time. The model must extract spatial patterns and track their temporal evolution. The analogy is imperfect: NCA rules are deterministic while cortical dynamics are stochastic; NCA rules are fixed per sequence while neural trials have within-trial regime switches across 3 phonemes; NCA grids are homogeneous while real arrays have dead electrodes, patient-specific anatomy, and spatially correlated noise. Stage 1 NCA data is designed to bridge these gaps via nuisance augmentation and regime-switching dynamics (see Section 4).

**Pretraining objective**: Observation-space prediction is the starting point (collapse-proof, simple), but one-step MSE on 20Hz LPF data is vulnerable to a trivial persistence solution. The objective must be made harder: multi-step prediction at k=3-5 frames, masked temporal spans, or frame-skipping. The exact pretext task is an ablation axis, not a locked decision.

**JEPA as future ablation**: We start with observation-space prediction for simplicity and stability. However, the rejection of JEPA should not be categorical — HGA contains substantial patient-specific nuisance variation (impedance differences, dead channels, local smoothness, task timing structure) where JEPA's latent abstraction could provide real value. JEPA remains a planned ablation once the observation-space pipeline is validated.

## 3. Paper Scope

**Title**: *Synthetic Spatiotemporal Pretraining for Data-Scarce Neural Decoding*

**Type**: Standalone methods paper.

**Claims** (limited to three questions):
1. Does SSL pretraining help for acute intra-op uECOG decoding at <200 trials/patient?
2. Does synthetic pre-pretraining add value beyond real-data-only SSL?
3. Does cortex-inspired synthetic structure help beyond generic smooth dynamics?

**Validation**: uECOG dataset (8-12 patients). Second dataset deferred.

**Relationship to existing codebase**: Separate pipeline. Shares data loading utilities but has its own models, training, and evaluation.

## 4. Three-Stage Pipeline

### Stage 1: NCA Pre-pretraining (unlimited synthetic data)

**Goal**: Teach encoder and predictor to process 2D grids evolving over time.

**NCA data generation** — all rule families use continuous scalar values per cell:

1. **Random MLP** (from Lee et al. 2026)
   - Rule: MLP(9 -> 32 -> 1), tanh hidden, random weights per sequence
   - 3x3 neighborhood, synchronous update
   - Maximally diverse: chaos, waves, fixed points, oscillations

2. **Reaction-Diffusion (Gray-Scott)**
   - Two variables per cell (u activator, v inhibitor), model observes u only
   - Parameters (F, k) sampled per sequence from known interesting regime
   - Produces Turing patterns, waves, spots

3. **Excitable Media (FitzHugh-Nagumo)**
   - Two variables per cell (v membrane potential, w recovery), model observes v only
   - Parameters (a, b, epsilon, D) sampled per sequence
   - Produces traveling waves, spiral waves, refractory dynamics
   - An excitable-medium toy model with dynamics qualitatively similar to cortical traveling waves, though not a model of HGA population activity per se

4. **Damped Wave Propagation**
   - Wave equation with random speed, damping, initial impulses
   - Clean traveling waves

5. **Sequential Hotspot Activation**
   - Gaussian bump following random smooth trajectory
   - Models sequential spatial activation

6. **Smooth Gaussian AR** (synthetic control — weaker than NCA)
   - Spatially smoothed Gaussian noise with temporal AR(1) process
   - No local update rules, no emergent dynamics
   - Tests whether ANY smooth synthetic movie helps, or whether structured dynamics are required
   - If NCA ≈ Smooth AR, the paper becomes "smooth pretraining helps" not "NCA dynamics help"

**Within-sequence regime switches**: To bridge the mismatch between fixed NCA rules and within-trial phoneme transitions, Stage 1 sequences include regime-switching variants:
- Switch the NCA rule (or its parameters) 2-3 times within a single sequence at random timepoints
- The model must track dynamics that change mid-sequence, not just infer a single fixed rule
- This better matches neural trials where the driving phoneme changes 3 times

**Nuisance realism** (applied to ALL synthetic data):
- Z-score normalize to zero mean, unit variance
- Add Gaussian noise: std=0.2 per frame
- Temporal smoothing: Butterworth LPF 8Hz before subsampling to 20Hz
- Dead electrode simulation: randomly zero out 0-8 grid positions per sequence (matching real dead electrode rates)
- Spatial rotations/reflections: random 90-degree rotations and flips (arrays are placed at variable orientations)
- Spatially correlated noise: add low-rank common-mode fluctuation (rank 1-3, random spatial pattern per sequence)
- Trial-onset jitter: random temporal offset (0-5 frames) for each sequence start
- Grid sizes: 8x16, 12x22, 8x32, 8x34

**Training objective — multi-step prediction in observation space**:
```
Given frames_{1:t}, predict frames at horizons k in {1, 3, 5}:

  frames_{1:t} -> Encoder -> spatial tokens -> Predictor -> predicted tokens at t+k
  -> Decoder -> predicted_frame_{t+k}

  Loss = sum over k: MSE(predicted_frame_{t+k}, actual_frame_{t+k})
```

Multi-step prediction forces the model to learn actual dynamics, not just temporal smoothness. A persistence baseline (predict frame_t as frame_{t+k}) is the lower bound; the model must beat this to be learning anything.

**Persistence baseline diagnostic**: Compute MSE of the identity predictor (copy current frame) at each horizon k. If the trained model's loss at k=1 is close to persistence, the pretext task is too easy. Increase k or switch to masked spans.

### Stage 2: Neural Pretraining (all patients pooled, no labels)

**Goal**: Adapt encoder and predictor from synthetic to real cortical dynamics.

**Data**: ALL trials from ALL PhonemeSequence patients (~1500+ trials). No labels used. Lexical task patients are a separate ablation (different temporal statistics and trial structure), not included by default.

**Training**:
- Initialize encoder + predictor + decoder from Stage 1 weights
- Per-patient patch embedding and per-patient decoder
- Same multi-step prediction objective
- Monitor per-patient prediction loss: if one patient's loss drops 10x faster than others, the model is memorizing that patient's dynamics rather than learning general cortical structure

**Patient-ID probe diagnostic**: After Stage 2, train a linear probe to predict patient identity from frozen encoder features. If patient-ID is highly separable, the pretext task is learning nuisance structure (patient-specific noise patterns) rather than task-relevant dynamics. Compare patient-ID separability before and after Stage 2 — it should NOT increase dramatically if pretraining is learning dynamics rather than patient fingerprints.

### Stage 3: Per-Patient Fine-tuning (labeled data)

**Goal**: Train a lightweight classification head on pretrained features.

**Training**:
- Freeze encoder + predictor. Decoder discarded.
- For each patient independently: 5-fold stratified CV
- CTC or CE head on frozen features (see loss options below)

**Primary freeze configuration**: Head + per-patient embed + predictor LayerNorms. Head-only is the purest probe of representation quality, but allowing predictor LayerNorm tuning (~3K params) is a reasonable default that adapts temporal statistics without risking overfitting. Full fine-tune is an ablation expected to overfit.

**Loss options** (CTC primary, others as ablations):
- **CTC**: Alignment-free, no MFA dependency, rigorous.
- **CE + learned attention pooling**: 3 position queries cross-attend to temporal features.
- **CE + MFA pooling**: Diagnostic for MFA boundary accuracy.

## 5. Architecture

**Unified dimension**: d = 128. Sweep: {64, 128, 256}.

### Preprocessing

**Neural data** (Stages 2-3):
```
Raw .fif (200Hz, z-scored HGA) -> Butterworth LPF 8Hz -> subsample to 20Hz
  -> dead electrode mask (set to 0.0) -> no threshold clamping
```

**NCA data** (Stage 1):
```
Raw NCA -> z-score normalize -> add noise (std=0.2) -> dead electrode sim
  -> spatial rotations -> correlated noise -> LPF 8Hz -> subsample to 20Hz
```

### Encoder (spatial frame -> spatial tokens, NOT a single vector)

**Key design**: The encoder outputs **per-patch spatial tokens**, not a single mean-pooled vector. Spatial structure is preserved through the temporal model. Mean-pooling before the predictor would destroy local propagation patterns — exactly what NCA pretraining is supposed to teach.

**Default: ViT-Small** (d=128, 4 layers, 4 heads, head_dim=32, FFN 2x)
- Input: single frame (H x W)
- Patch tokenization: 2x2 spatial patches
  - 8x16 -> 32 tokens; 12x22 -> 66 tokens; 8x32 -> 64 tokens; 8x34 -> 68 tokens
- Patch embedding: Linear(4, 128) = 640 params. **Shared** in Stage 1, **per-patient** in Stages 2-3.
- **Coordinate-based positional encoding**: NOT shared absolute token indices (patch 17 in an 8x16 grid is not anatomically comparable to patch 17 in a 12x22 grid). Instead: normalized (x, y) coordinates in [0, 1] for each patch center, projected via Linear(2, 128). Plus a per-patient-grid binary mask indicating dead positions.
- Output: **(N_patches, d)** spatial tokens per frame — NOT collapsed to one vector

**Per-layer**: QKV 49K + out 16K + FFN 66K + LN 0.5K = ~131K
- **4 layers: ~525K. Total encoder: ~535K.**

**Ablation: Conv stem + per-patient spatial read-in**
- For these tiny regular grids, a pure ViT front-end may not be the right inductive bias
- Alternative: Conv2d(1, C, k=3, pad=1) + ReLU stack, then flatten spatial patches for transformer
- Tests whether conv spatial bias helps vs pure attention

### Predictor (temporal dynamics on spatial tokens)

**Factorized space-time modeling**: The predictor operates on spatial tokens across time, not on a single collapsed vector.

**Default: Factorized Transformer** (d=128, 2 layers, 4 heads)
- Each layer has two attention blocks:
  1. **Spatial attention**: for each timestep, attend across all spatial patches (N_patches tokens)
  2. **Temporal attention**: for each spatial patch position, attend across timesteps (T' tokens)
- Causal mask on temporal attention during pretraining; remove during fine-tuning
- This preserves local spatial propagation while modeling temporal dynamics
- Sequence lengths: spatial attention = 32-68 tokens (manageable), temporal attention = 30-50 tokens

**Per factorized layer**: spatial attn ~131K + temporal attn ~131K + FFN ~66K + LN = ~328K
- But Q/K/V are shared across the two attention types → more like ~200K per layer
- **2 layers: ~400K. Total predictor: ~400K.**

**Output**: For pretraining, the predictor outputs predicted spatial tokens at future timesteps. For fine-tuning, it outputs contextualized spatial tokens that are then pooled for classification.

**Stage 3 readout**: After the predictor, spatial tokens at each timestep are mean-pooled to get per-frame vectors h_{1:T'} (128-dim each), then CTC/CE head operates on these. The spatial pooling happens AFTER temporal modeling, not before.

**Ablation: GRU-Small**
- Operates on spatially-pooled frames (collapses spatial tokens first)
- Tests whether preserving spatial tokens through time actually helps

### Decoder (pretraining only)

- Predicts future spatial patches from predicted spatial tokens
- Per-patch: Linear(128, 4) for 2x2 patches. ~520 params per patch, ~34K total for 66 patches.
- Per-patient decoder in Stage 2 (different grid sizes)
- Discarded after pretraining

### Parameter Budget

```
Component                    Params    Phase         Notes
------------------------------------------------------
Encoder (ViT, d=128)         ~535K    Stage 1+2     4 layers, spatial tokens out
Predictor (Factorized Txf)   ~400K    Stage 1+2     2 layers, space-time factorized
Decoder (pretraining only)    ~34K    Stage 1+2     Per-patch Linear, DISCARDED
Per-patient embed (x12)       ~7.7K   Stage 2       640 each
Coord PE projection             256   Shared        Linear(2, 128)
CTC head Linear(128, 10)      1,290   Stage 3       9 phonemes + blank
------------------------------------------------------
Total (pretraining)          ~977K
Total (fine-tuning)          ~944K    (decoder discarded)
  Frozen at Stage 3          ~943K
  Trained from scratch         1,290
  With LN tuning              ~4K
```

### Latent Space

- d = 128 default. Sweep: {64, 128, 256}.
- SIGReg optional (lambda in {0, 0.01, 0.1}). Reconstruction prevents collapse; SIGReg may help latent geometry.

## 6. Comparison Methods (Initial Round)

Five methods. All use the same architecture and Stage 3 setup.

### A. NCA + Neural Pretrain (hypothesis)
Stage 1: NCA multi-step prediction -> Stage 2: Neural multi-step prediction -> Stage 3: head

### B. Neural Pretrain Only
Stage 2: Neural multi-step prediction (random init) -> Stage 3: head

### C. Smooth AR + Neural Pretrain (synthetic control)
Stage 1: Spatially-smoothed Gaussian AR data (no local update rules, no emergent dynamics) -> Stage 2: Neural -> Stage 3: head.
This isolates whether structured NCA dynamics matter or whether any smooth synthetic movie helps.

### D. Supervised Baseline
No pretraining. Encoder + predictor + head trained end-to-end with CTC on labeled data.

### E. Random-Init Frozen (reservoir baseline)
Freeze randomly-initialized encoder+predictor. Train only CTC head.

### Experiment Execution Order

```
Step 1: Method E (random-init frozen)
  Cost:    Minutes.
  Purpose: Noise floor. If frozen random features decode well,
           the architecture itself has useful inductive bias.

Step 2: Method D (supervised from scratch)
  Cost:    ~30 min per patient.
  Purpose: Architecture + supervision baseline. The number to beat.
  Gate:    If D ≈ E, architecture has no value. Rethink before proceeding.

Step 3: Method B (neural-only pretrain)
  Cost:    Hours.
  Purpose: Does SSL on real data help?
  Gate:    If B ≈ E, pretraining objective is broken. Debug before adding NCA.

Step 4: Methods A and C (NCA pretrain and smooth-AR pretrain)
  Cost:    Hours each. Run in parallel.
  Purpose: Does NCA add value? Does structure matter vs smooth noise?
  Interpretation:
    A > C > B  →  Structured dynamics help beyond smooth pretraining. Core claim.
    A ≈ C > B  →  Any smooth pretraining helps. Weaker but publishable.
    A ≈ C ≈ B  →  Synthetic pretraining adds nothing. Real-data SSL is sufficient.
    A ≈ B ≈ D  →  SSL doesn't help at this scale. Null result, document and move on.
```

### Future Round

| Method | Description |
|--------|-------------|
| F. NCA Only | Stage 1 only, skip Stage 2 → quantifies domain gap |
| G. MAE | Masked spatial patches, reconstruct |
| H. JEPA | Latent-space prediction + SIGReg, tests abstraction value |

## 7. Evaluation Framework

### Primary Metrics
- **PER**: via CTC greedy decode
- **Balanced accuracy**: 9-way, per position

Both via 5-fold stratified CV within each patient.

### Evaluation Mode
- **Inductive is the primary result**: Stage 2 excludes target patient
- **Transductive is secondary**: Stage 2 includes target patient's unlabeled data
- Rationale: inductive is the realistic deployment setting. Transductive measures the value of patient-specific adaptation in Stage 2 but represents a different (easier) scenario.

### Required Diagnostics

1. **Persistence baseline**: MSE of identity predictor (copy frame_t as prediction for frame_{t+k}) at each horizon. If trained model barely beats persistence, the pretext task is too easy.
2. **Patient-ID linear probe**: After each stage, train linear classifier to predict patient identity from frozen encoder features. If patient-ID separability increases sharply in Stage 2, pretraining is learning nuisance (patient fingerprints) not dynamics.
3. **Phoneme-ID linear probe**: After each stage, train linear classifier to predict phoneme from frozen features (using labels for evaluation only). Improvement from Stage 1→2→3 shows where phoneme-relevant structure emerges.

### Reporting
- Per-patient table, population mean +/- std
- Wilcoxon signed-rank (paired, N=8-12)
- Bootstrap 95% CIs, effect sizes (Cohen's d)
- Holm-Bonferroni for multiple comparisons

## 8. Ablations (Prioritized)

### Tier 1 (run with initial round)

**A. NCA Rule Family**
- Random MLP vs FitzHugh-Nagumo vs Mixed vs Smooth AR (control)
- If Mixed wins, the paper becomes "diverse synthetic dynamics help" — prepare this interpretation now

**B. Prediction Horizon / Objective**
- k=1 only vs k={1,3,5} vs k=5 only vs masked temporal spans (mask 3-5 consecutive frames, predict from context)
- Determines whether the pretext task learns dynamics or just smoothness

**C. Freeze Level**
- Head only (1.3K) vs Head+embed+LN (4K) vs Full fine-tune (~944K)
- Head-only probes representation quality; head+LN is the practical default

### Tier 2 (run if Tier 1 shows promise)

- Encoder: ViT vs conv stem
- Predictor: factorized transformer vs GRU (spatial tokens collapsed)
- Latent dimension: d in {64, 128, 256}
- Stage 2 data volume: 2, 4, 8, all patients
- Loss type (Stage 3): CTC vs CE+learned attention vs CE+MFA
- Lexical patients in Stage 2 (separate ablation, not default)
- SIGReg: lambda in {0, 0.01, 0.1}
- NCA grid geometry: matching vs square vs mismatched
- Predictor depth: 2 vs 4 layers

## 9. Known Risks

1. **NCA-to-neural domain gap**: Synthetic dynamics may not transfer despite nuisance augmentation. Smooth AR control (Method C) isolates this. Even null results contribute.

2. **Trivial persistence solution**: One-step 20Hz prediction may be too easy. Multi-step prediction and persistence baseline diagnostic address this.

3. **Patient-ID leakage in Stage 2**: Predictor may learn patient fingerprints rather than dynamics. Patient-ID probe diagnostic detects this.

4. **Statistical power**: N=8-12 with noisy per-patient estimates. Bootstrap CIs, effect sizes, per-patient plots.

5. **FHN parameter instability**: Large parameter regions produce trivial dynamics or numerical blow-up. Map interesting regime from known phase diagrams before generating at scale.

6. **Factorized attention compute**: Spatial (32-68 tokens) x temporal (30-50 tokens) at each layer is manageable but 2x the attention ops of a standard transformer. Profile early.

## 9b. Precautionary Lessons from Prior Experiments

### Data scarcity is the wall, not architecture
H=128 GRU showed 55x train/val gap — more capacity = more overfitting with ~150 trials. If pretrained features aren't good enough to freeze, unfreezing will overfit immediately. The freeze-level ablation is make-or-break, not nice-to-have.

### LOPO failed near chance — pooled training is risky
LOPO got PER=0.846 with 7 source patients; Stage 1 multi-patient SGD early-stopped at step 800 (val diverging). Stage 2 neural pretraining faces the same risk. Monitor per-patient prediction loss for dominance by one patient.

### Augmentation tuning is within noise at this data scale
All augmentation sweeps landed within PER 0.78-0.83. Don't spend time fine-tuning NCA preprocessing params. Get a reasonable default, move on. The big question is whether pretraining helps at all.

### Auxiliary targets need negative controls
Acoustic regression R² looked positive but shifted control gave the same number — artifact. Method E (random-init) is the negative control for this pipeline. Run it first.

### CE may have been fitting trial artifacts
CE beat CTC (0.700 vs 0.778) but may have exploited epoch structure. The three-way Stage 3 loss comparison resolves this on frozen features.

### MPS breaks with mixed grid sizes
Test with ALL grid sizes early: 8x16, 12x22, 8x32, 8x34.

### Stratified splits essential
S33 has 52 trials. Use StratifiedKFold on first phoneme label.

### Per-patient layers are non-negotiable
Per-patient patch embedding (640 params each) is the minimum adaptation mechanism. Do not skip.

## 10. References

- Lee, Han, Kumar, Agrawal (2026). "Training Language Models via Neural Cellular Automata." arXiv:2603.10055.
- Maes, Le Lidec, Scieur, LeCun, Balestriero (2026). "LeWorldModel." arXiv:2603.19312.
- Spalding et al. (2025). Cross-patient speech decoding from uECOG.
- Singh et al. (2025). Per-patient layers + frozen shared backbone.
- Duraivel et al. (2025). Pseudo-word repetition with overlapping patients.

## 11. Architecture Diagram

See `docs/nca_jepa_pipeline.svg` (note: diagram predates v4 revision; spatial token preservation and factorized attention not yet reflected).

## 12. Resolved Design Decisions

1. **Preserve spatial tokens through predictor**: Mean-pool AFTER temporal modeling, not before. Spatial propagation structure is the whole point of NCA pretraining.
2. **Coordinate-based PE**: Normalized (x, y) coordinates projected to d-dim, not shared absolute token indices. Patch 17 in an 8x16 grid is not comparable to patch 17 in a 12x22 grid.
3. **Multi-step prediction**: k={1,3,5} default. One-step on 20Hz LPF data is too easy (persistence baseline).
4. **Regime-switching NCA**: Include within-sequence rule/parameter switches to match within-trial phoneme transitions.
5. **Nuisance realism in synthetic data**: Dead electrodes, rotations, correlated noise, onset jitter applied to all NCA data.
6. **Smooth AR synthetic control**: Required to distinguish "structured dynamics help" from "any smooth movie helps."
7. **Inductive evaluation primary**: Stage 2 excludes target patient for the main results.
8. **Lexical patients as ablation, not default**: Different temporal statistics and trial structure.
9. **FHN is a toy model**: Not "cortex." Useful excitable-medium model with qualitatively similar dynamics.
10. **JEPA deferred, not rejected**: Has value for nuisance abstraction. Future ablation once observation-space pipeline validated.
11. **Patient-ID probe required**: Detects if Stage 2 learns nuisance structure.
12. **Head+embed+LN as practical default**: Head-only is purest probe; LN tuning is practical default.

## 13. Open Questions

1. **Causal vs bidirectional predictor**: Causal pretraining then bidirectional fine-tuning is a mismatch. Options: (a) keep causal end-to-end (CTC handles it), (b) pretrain with masked-span/bidirectional objective, (c) accept the mismatch as an ablation. Leaning toward (a) for simplicity.
2. **Factorized attention parameter sharing**: Should spatial and temporal attention share Q/K/V projections? Sharing saves params; separate may specialize better.
3. **FHN parameter ranges**: Need to map interesting regime from known phase diagrams.
4. **Multi-variable observation**: FHN has 2 variables per cell. Expose both (H, W, 2) or only observable (H, W, 1)?
5. **Prediction target**: Predict raw frame values or predict patch tokens (encoder output)? Patch-token prediction is closer to JEPA but still has reconstruction target (the encoder's output for the target frame, with stop-gradient on the target encoder).
