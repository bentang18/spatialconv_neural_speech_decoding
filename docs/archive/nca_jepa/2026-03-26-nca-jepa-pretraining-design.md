# Synthetic Grid-Dynamics Pretraining for Data-Scarce Neural Decoding — Design Spec

**Date**: 2026-03-26 (v5)
**Author**: Ben Tang, Cogan Lab (Duke)
**Status**: Draft

## 1. Problem Statement

Intraoperative uECOG speech decoding faces an extreme data bottleneck: ~1 minute of utterance per patient (~150 trials), 128-256 channels at 200Hz. Current supervised approaches train ~60K parameters from scratch on this thin data and overfit heavily. No synthetic pretraining method has been tested for acute intra-op uECOG phoneme-sequence decoding at this data scale. SSL methods exist for speech-BCI (wav2vec ECoG, BrainBERT), but none target the specific regime of <200 trials on rigid micro-ECoG arrays with known 2D spatial structure.

## 2. Core Idea

Pretrain an encoder-predictor on synthetic spatiotemporal grid dynamics, then adapt to real cortical recordings, then fine-tune a lightweight classification head per patient.

The synthetic data generators span a family: Neural Cellular Automata (local-rule systems), reaction-diffusion PDEs, excitable-medium models, damped wave equations, parametric hotspot trajectories, and spatially-smoothed Gaussian AR processes. NCA is one family within this broader set, not the exclusive method. The shared structure is: a 2D grid of continuous values evolving over time according to a latent rule or parameter set that varies per sequence.

**The synthetic-neural analogy**: Both synthetic generators and uECOG produce 2D grids evolving over time. The model must extract spatial patterns and track their temporal evolution. The analogy is imperfect: synthetic rules are deterministic while cortical dynamics are stochastic; most synthetic rules are fixed per sequence while neural trials have within-trial regime switches across 3 phonemes; synthetic grids are homogeneous while real arrays have dead electrodes, patient-specific anatomy, and spatially correlated noise. Stage 1 data bridges these gaps via nuisance augmentation and regime-switching dynamics (see Section 4).

**Pretraining objective**: Observation-space multi-step prediction at horizons k={1,3,5} frames. One-step MSE on 20Hz LPF data is vulnerable to a trivial persistence solution; multi-step prediction forces learning of actual dynamics. The exact pretext task (multi-step MSE vs masked temporal spans) is an ablation axis.

**JEPA as future ablation**: We start with observation-space prediction for simplicity and stability. JEPA is deferred, not dismissed — HGA contains substantial patient-specific nuisance variation (impedance differences, dead channels, local smoothness, task timing structure) where JEPA's latent abstraction could provide real value. JEPA is a planned ablation once the observation-space pipeline is validated.

## 3. Paper Scope

**Title**: *Synthetic Spatiotemporal Grid Pretraining for Data-Scarce Neural Decoding*

**Claims** (limited to three questions):
1. Does SSL pretraining help for acute intra-op uECOG decoding at <200 trials/patient?
2. Does synthetic pre-pretraining add value beyond real-data-only SSL?
3. Does structured synthetic dynamics help beyond generic smooth dynamics?

**Validation**: uECOG dataset (8-12 patients). Second dataset deferred.

**Relationship to existing codebase**: Separate pipeline. Shares data loading utilities but has its own models, training, and evaluation.

## 4. Three-Stage Pipeline

### Stage 1: Synthetic Pre-pretraining (unlimited synthetic data)

**Goal**: Teach encoder and predictor to process 2D grids evolving over time.

**Synthetic data generator families** — all use continuous scalar values per cell:

1. **Neural Cellular Automata (Random MLP)** (from Lee et al. 2026)
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
   - Excitable-medium toy dynamics qualitatively resembling propagating cortical activity, though not a model of HGA population dynamics per se

4. **Damped Wave Propagation**
   - Wave equation with random speed, damping, initial impulses
   - Clean traveling waves

5. **Sequential Hotspot Activation**
   - Gaussian bump following random smooth trajectory
   - Provides a simple synthetic analog of sequential spatial activation

6. **Spatially-Smoothed Gaussian AR** (synthetic control — weaker than all above)
   - AR(1) process with spatially-smoothed Gaussian innovations: x_{t+1} = alpha * x_t + (1-alpha) * smooth(noise)
   - Spatial smoothing via Gaussian kernel (sigma=2-4 grid cells)
   - No local update rules, no emergent dynamics
   - Tests whether ANY smooth synthetic movie helps, or whether structured dynamics are required

**Within-sequence regime switches**: To bridge the mismatch between fixed synthetic rules and within-trial phoneme transitions, Stage 1 sequences include regime-switching variants: the rule parameters change 2-3 times within a single sequence at random timepoints. The model must track dynamics that change mid-sequence.

**Nuisance realism** (applied to ALL synthetic data):
- Z-score normalize to zero mean, unit variance
- Gaussian noise: std=0.2 per frame
- Temporal smoothing: Butterworth LPF before subsampling
- Dead electrode simulation: randomly zero 0-8 grid positions per sequence
- Flips and 180-degree rotations (default geometry augmentation for rectangular grids; 90-degree rotation swaps H and W dimensions and is deferred to a separate geometry-remapping ablation)
- Spatially correlated noise: low-rank common-mode fluctuation (rank 1-3, random spatial pattern)
- Trial-onset jitter: random temporal offset (0-5 frames)
- Grid sizes: 8x16, 12x22, 8x32, 8x34

**Training objective — multi-step prediction in observation space**:
```
Given spatial tokens for frames_{1:t}, predict frames at horizons k in {1, 3, 5}:

  frames -> Encoder -> spatial tokens -> Predictor -> predicted tokens at t+k
  -> Decoder -> predicted patches at t+k

  Loss = sum over k in {1,3,5}: MSE(predicted_patches_{t+k}, actual_patches_{t+k})
```

**Persistence baseline diagnostic** (required): Compute MSE of the identity predictor (copy frame_t as prediction for frame_{t+k}) at each horizon k. The trained model must substantially beat persistence. If it doesn't at k=1, the pretext is too easy; if it doesn't at k=5, dynamics are too stochastic to predict.

### Stage 2: Neural Pretraining (all patients pooled, no labels)

**Goal**: Adapt encoder and predictor from synthetic to real cortical dynamics.

**Data**: ALL trials from ALL PhonemeSequence patients (~1500+ trials). No labels used. Lexical task patients are a separate ablation (different temporal statistics and trial structure), not included by default.

**Training**:
- Initialize from Stage 1 weights
- Per-patient patch embedding and per-patient decoder
- Same multi-step prediction objective
- Monitor per-patient prediction loss variance and dominance ratios: if a single patient's loss drops dramatically faster, the model may be memorizing that patient's noise structure

**Patient-ID probe diagnostic** (required): After Stage 2, train linear probes to predict both patient identity and phoneme identity from frozen encoder features. A sharp rise in patient-ID separability WITHOUT a corresponding rise in phoneme-ID separability signals that the pretext task is learning nuisance (patient fingerprints) rather than task-relevant dynamics. High patient-ID separability alone is not concerning if phoneme-ID also improves — useful neural features can preserve patient identity.

### Stage 3: Per-Patient Fine-tuning (labeled data)

**Goal**: Train a lightweight classification head on pretrained features.

**Default freeze configuration**: Freeze all weights except CTC head, per-patient patch embedding, and predictor LayerNorm parameters. This allows task-specific temporal normalization (~4K total trainable) without risking overfitting from unfreezing attention weights.

**Training**: For each patient independently, 5-fold stratified CV (stratification on first phoneme label — an approximation to preserve label balance under small N, not perfect stratification of the 3-phoneme output sequence).

**Loss options** (CTC primary):
- **CTC**: Alignment-free, no MFA dependency.
- **CE + learned attention pooling** (ablation): 3 position queries cross-attend to temporal features.
- **CE + MFA pooling** (ablation): Diagnostic for MFA boundary accuracy.

## 5. Architecture

**Unified dimension**: d = 128. Sweep: {64, 128, 256}.

### Preprocessing

**Neural data** (Stages 2-3):
```
Raw .fif (200Hz, z-scored HGA) -> Butterworth LPF 8Hz -> subsample to 20Hz
  -> dead electrode mask -> no threshold clamping
```

20Hz is the default, chosen as a compute/regularization tradeoff (30 frames per 1.5s trial). It is not a field truth — 40Hz (60 frames per trial) may retain useful dynamics at modest cost. **20 vs 40 Hz is an early ablation** (Tier 1).

**Synthetic data** (Stage 1): same pipeline after generation (z-normalize, noise, dead electrodes, flips/180° rotations, correlated noise, LPF, subsample).

### Encoder (spatial frame -> spatial tokens)

The encoder outputs **per-patch spatial tokens**, not a single mean-pooled vector. Spatial structure is preserved through the predictor.

**Default: ViT-Small** (d=128, 4 layers, 4 heads, head_dim=32, FFN 2x)
- Patch tokenization: 2x2 spatial patches. This is a bias toward smooth local structure and may be too coarse for micro-ECoG where single electrodes carry distinct signal. **1x1 patches (one token per electrode) and a conv-stem alternative are serious baselines, not convenience ablations** (Tier 2).
  - 2x2 patches: 8x16 -> 32 tokens; 12x22 -> 66; 8x32 -> 64; 8x34 -> 68
  - 1x1 patches: 128-272 tokens (more expensive but preserves per-electrode resolution)
- Patch embedding: Linear(4, 128) = 640 params (with bias). **Shared** in Stage 1, **per-patient** in Stages 2-3. For 1x1 patches: Linear(1, 128) = 256 params.
- **Coordinate-based positional encoding**: Each patch's center position as normalized (x, y) in [0, 1], projected via MLP(2 -> 128) = 384 params. Plus a binary dead-position indicator. NOT shared absolute token indices — patch 17 in an 8x16 grid is not anatomically comparable to patch 17 in a 12x22 grid.
- Transformer: 4 layers, d=128, 4 heads, head_dim=32, GELU FFN with 2x expansion
- Output: **(N_patches, d)** spatial tokens per frame

**Per encoder layer**:
- Multi-head self-attention: Q/K/V projections 3 x 128 x 128 = 49,152; output projection 128 x 128 = 16,384; total attn = 65,536
- FFN: Linear(128, 256) + Linear(256, 128) = 65,536
- LayerNorm x2: 2 x 256 = 512
- **Per layer: 131,584**
- **4 layers: 526,336**
- Patch embedding: 640; Coord PE MLP: 384; Spatial PE buffer: 0 (computed, not learned)
- **Total encoder: ~528K**

**Ablation: Conv stem + transformer**
- Conv2d(1, C, k=3, pad=1) stack as spatial front-end, then flatten to patch tokens for transformer
- Tests whether conv spatial bias helps vs pure attention on these tiny regular grids

### Predictor (factorized space-time transformer)

**TimeSformer-style divided space-time attention** (Bertasius et al. 2021). Each layer has three blocks in sequence: temporal self-attention, spatial self-attention, and FFN. All blocks have separate Q/K/V projections (not shared).

d=128, 2 layers, 4 heads, head_dim=32, FFN 2x.

**Per predictor layer** (three blocks):
1. **Temporal self-attention**: For each spatial patch position, attend across T' timesteps.
   - LayerNorm: 256
   - Q/K/V: 3 x 128 x 128 = 49,152
   - Output projection: 16,384
   - Block total: **65,792**
2. **Spatial self-attention**: For each timestep, attend across N_patches spatial tokens.
   - Same structure: **65,792**
3. **FFN**: Linear(128, 256) + GELU + Linear(256, 128).
   - LayerNorm: 256
   - Weights: 128 x 256 + 256 + 256 x 128 + 128 = 65,920
   - Block total: **66,176**

- **Per layer: 197,760**
- **2 layers: 395,520**
- Temporal PE: sinusoidal (0 learnable params)
- **Total predictor: 395,520**

Temporal attention uses a causal mask during pretraining (Stages 1-2). At Stage 3, the predictor is used as a feature extractor with the causal mask retained for consistency (CTC handles unidirectional features; removing the mask mid-pipeline is a mismatch we avoid).

### Decoder (pretraining only, discarded at Stage 3)

**Shared per-patch decoder**: A single Linear(128, 4) applied identically to every predicted spatial token (translation-equivariant across patch positions).
- Parameters: 128 x 4 + 4 = **516 params** (with bias)
- Output per token: 4 values (the 2x2 patch pixel values)
- Reshaped to (H, W) frame for MSE loss
- Same decoder applied to all grid sizes, all patients (patch content is grid-size-independent)
- **Discarded after pretraining**

### Stage 3 Readout

After the predictor, spatial tokens at each timestep are mean-pooled to get per-frame vectors h_{1:T'} (128-dim each). CTC head operates on these. Spatial pooling happens AFTER temporal modeling, preserving local propagation structure through the dynamics model.

### Parameter Budget

```
Component                       Params      Phase
────────────────────────────── ────────── ──────────────
Encoder (ViT, d=128, 4L)        528,384   Stage 1+2
Predictor (Factorized, 2L)      395,520   Stage 1+2
Decoder (shared Linear(128,4))      516   Stage 1+2, DISCARDED
Coord PE MLP(2->128)                384   Shared
Per-patient patch embed (x12)     7,680   Stage 2
CTC head Linear(128, 10)         1,290    Stage 3
────────────────────────────── ────────── ──────────────
Total (pretraining)             933,774
Total (fine-tuning)             933,258   (decoder discarded)

Stage 3 trainable (default):
  CTC head                        1,290
  Per-patient embed                 640
  Predictor LayerNorms           2,048   (4 LN layers x 2 x 128 + 2 x 128)
  ────────────────────────────
  Total trainable at Stage 3:    ~3,978
  Frozen at Stage 3:           ~929,280
```

## 6. Comparison Methods

Five methods in the initial round. All use the same architecture and Stage 3 setup.

**Compute-matched protocol**: Methods A and B must have the same total optimization budget, otherwise A > B is confounded by "more pretraining, not better pretraining." Protocol:
- Fix total pretraining steps to S_total (e.g., 50K steps).
- Method A: S_total/2 steps on synthetic (Stage 1) + S_total/2 steps on neural (Stage 2).
- Method B: S_total steps on neural (Stage 2 only, random init).
- Method C (Smooth AR): S_total/2 steps on smooth AR + S_total/2 steps on neural.
- This ensures A, B, C have identical optimizer budgets. Any A > B difference is attributable to the CONTENT of pretraining, not its DURATION.

### A. Structured Synthetic + Neural (hypothesis)
Stage 1: Multi-step prediction on NCA/FHN/Mixed synthetic data (S_total/2 steps) -> Stage 2: Neural multi-step prediction (S_total/2 steps) -> Stage 3: head

### B. Neural Only (compute-matched)
Stage 2: Neural multi-step prediction, random init (S_total steps) -> Stage 3: head

### C. Smooth AR + Neural (synthetic control)
Stage 1: Multi-step prediction on smooth Gaussian AR data (S_total/2 steps) -> Stage 2: Neural (S_total/2 steps) -> Stage 3: head.
Isolates whether structured dynamics matter or any smooth synthetic movie helps.

### D. Supervised Baseline
No pretraining. Encoder + predictor + head trained end-to-end with CTC.

### E. Random-Init Frozen (reservoir baseline)
Freeze randomly-initialized encoder+predictor. Train only CTC head.

### Experiment Execution Order

```
Step 1: Method E (random-init frozen)
  Cost:    Minutes.
  Purpose: Noise floor.
  Gate:    If E gives PER < 0.85, random features are non-trivial.

Step 2: Method D (supervised from scratch)
  Cost:    ~30 min per patient.
  Purpose: Architecture + supervision baseline.
  Gate:    If D ≈ E, architecture has no value. Rethink.

Step 3: Method B (neural-only pretrain, S_total steps)
  Cost:    Hours.
  Purpose: Does SSL on real data help?
  Gate:    If B ≈ E, pretraining objective is broken. Debug.

Step 4: Methods A and C in parallel (synthetic + neural, compute-matched)
  Cost:    Hours each.
  Purpose: Does synthetic pre-pretraining add value? Does structure matter?
  Interpretation:
    A > C > B  →  Structured dynamics help. Core claim.
    A ≈ C > B  →  Any smooth pretraining helps. Weaker but publishable.
    A ≈ C ≈ B  →  Synthetic pretraining adds nothing beyond real-data SSL.
    A ≈ B ≈ D  →  SSL doesn't help at this scale. Null result.
```

### Future Round

| Method | Description |
|--------|-------------|
| F. Synthetic Only | Stage 1 only, skip Stage 2 → quantifies domain gap |
| G. MAE | Masked spatial patches |
| H. JEPA | Latent prediction + SIGReg, tests abstraction value |
| I. B-extended | Method B with 2x S_total steps → tests if more real-data training alone closes gap |

## 7. Evaluation Framework

### Primary Metrics
- **PER**: via CTC greedy decode
- **Balanced accuracy**: 9-way, per position
- 5-fold stratified CV per patient

### Evaluation Mode
- **Inductive is the primary result**: Stage 2 excludes target patient
- **Transductive is secondary**: Stage 2 includes target patient's unlabeled data

### Required Diagnostics
1. **Persistence baseline**: MSE of identity predictor at each horizon k. Trained model must beat this substantially.
2. **Patient-ID linear probe**: After each stage. A sharp rise in patient-ID separability WITHOUT corresponding phoneme-ID improvement is concerning.
3. **Phoneme-ID linear probe**: After each stage. Tracks where task-relevant structure emerges.
4. **Per-patient loss monitoring**: Track variance and dominance ratios during Stage 2.

### Reporting
- Per-patient table, population mean +/- std
- Wilcoxon signed-rank, bootstrap 95% CIs, Cohen's d
- Holm-Bonferroni correction

## 8. Ablations (Prioritized)

### Tier 1 (run with initial round)

**A. Synthetic Generator Family**
- NCA (Random MLP) vs FitzHugh-Nagumo vs Mixed (all families) vs Smooth AR (control)
- If Mixed wins, the paper becomes "diverse synthetic dynamics help" — not "cortex-like dynamics help." Both interpretations are prepared.

**B. Prediction Horizon / Objective**
- k={1,3,5} (default) vs k=5 only vs masked temporal spans (mask 3-5 frames, predict from context)

**C. Freeze Level**
- Head+embed+LN (~4K trainable, default) vs head only (1.3K) vs full fine-tune (~933K)

**D. Temporal Resolution**
- 20Hz (default, 30 frames/trial) vs 40Hz (60 frames/trial)
- 20Hz is a compute/regularization choice, not a field truth. 40Hz may retain useful dynamics.

### Tier 2 (run if Tier 1 shows promise)

- Patch size: 2x2 (default) vs 1x1 (per-electrode) vs conv stem
- Encoder type: ViT vs conv stem + transformer
- Predictor: factorized transformer vs GRU (spatial tokens collapsed)
- Latent dimension: d in {64, 128, 256}
- Stage 2 data volume: 2, 4, 8, all patients
- Loss type (Stage 3): CTC vs CE+learned attention vs CE+MFA
- Lexical patients in Stage 2 (separate ablation)
- SIGReg: lambda in {0, 0.01, 0.1}
- Predictor depth: 2 vs 4 layers
- 90-degree rotation augmentation (requires geometry remapping for rectangular grids)

## 9. Known Risks

1. **Synthetic-to-neural domain gap**: Nuisance augmentation and regime-switching bridge part of the gap. Smooth AR control (Method C) isolates whether structure matters.
2. **Trivial persistence solution**: Multi-step prediction at k={1,3,5} and persistence diagnostic address this.
3. **Patient-ID leakage in Stage 2**: Patient-ID probe detects this.
4. **Statistical power**: N=8-12 with noisy estimates. Bootstrap CIs, effect sizes, per-patient plots.
5. **FHN parameter instability**: Map interesting regime from phase diagrams before generating.
6. **Factorized attention compute**: 2x attention ops per layer (spatial + temporal). Profile early.

## 9b. Precautionary Lessons from Prior Experiments

*Internal engineering notes from the existing supervised pipeline.*

- **Data scarcity is the wall**: H=128 GRU showed 55x train/val gap. If pretrained features aren't good enough to freeze, unfreezing overfits immediately. The freeze-level ablation is make-or-break.
- **LOPO failed near chance (0.846)**: Pooled training diverged at step 800. Stage 2 faces same risk. Monitor per-patient loss.
- **Augmentation tuning is within noise**: Don't fine-tune NCA preprocessing params. Get defaults, move on.
- **Auxiliary targets need negative controls**: Acoustic regression R² was an artifact. Method E (random-init) is the negative control here. Run it first.
- **CE may fit trial artifacts**: Three-way Stage 3 loss comparison resolves this on frozen features.
- **MPS breaks with mixed grid sizes**: Test ALL grid sizes early.
- **Stratified splits essential**: S33 has 52 trials. StratifiedKFold on first phoneme.
- **Per-patient adaptation is the default assumption**, based on prior transfer literature (Singh 2025) and prior failures without patient-specific adaptation (LOPO pilot PER=0.846).

## 10. References

- Lee, Han, Kumar, Agrawal (2026). "Training Language Models via Neural Cellular Automata." arXiv:2603.10055.
- Maes, Le Lidec, Scieur, LeCun, Balestriero (2026). "LeWorldModel." arXiv:2603.19312.
- Bertasius, Wang, Torresani (2021). "Is Space-Time Attention All You Need for Video Understanding?" ICML 2021. (TimeSformer factorized attention.)
- Spalding et al. (2025). Cross-patient speech decoding from uECOG.
- Singh et al. (2025). Per-patient layers + frozen shared backbone.
- Duraivel et al. (2025). Pseudo-word repetition with overlapping patients.

## 11. Architecture Diagram

See `docs/nca_jepa_pipeline.svg` (note: diagram predates v5; does not yet reflect factorized attention, coordinate PE, or spatial token preservation).

## 12. Resolved Design Decisions

1. **"Synthetic grid-dynamics pretraining"** not "NCA pretraining." NCA is one generator family among six.
2. **Preserve spatial tokens through predictor**: Mean-pool AFTER temporal modeling, not before.
3. **Coordinate-based PE**: MLP(2 -> d) on normalized (x, y), not shared absolute token indices.
4. **Multi-step prediction k={1,3,5}**: One-step on 20Hz LPF is too easy.
5. **Compute-matched comparison**: A/B/C all get S_total optimizer steps. A vs B tests content, not duration.
6. **Shared per-patch decoder**: Linear(128, 4) = 516 params, translation-equivariant, applied identically to all patch tokens regardless of grid size.
7. **Factorized space-time predictor**: TimeSformer-style divided attention with separate Q/K/V per block. Exact param count: 197,760 per layer, 395,520 total.
8. **Causal mask retained at Stage 3**: Avoids pretraining/fine-tuning mismatch. CTC handles unidirectional features.
9. **Flips + 180° default, 90° deferred**: 90° rotation swaps H/W on rectangular grids, requires geometry remapping. Separate ablation.
10. **20Hz default, 40Hz as early ablation**: 20Hz is a compute/regularization choice, not a field truth.
11. **2x2 patches as starting default**: Acknowledged bias toward smooth local structure. 1x1 and conv stem are serious baselines in Tier 2.
12. **Inductive as primary evaluation**: Target patient excluded from Stage 2.
13. **Lexical as ablation, not default**: Different temporal statistics.
14. **Stage 3 default**: Freeze all except CTC head + per-patient embed + predictor LayerNorms (~4K trainable).
15. **FHN is a toy model**: "Excitable-medium toy dynamics qualitatively resembling propagating activity."
16. **JEPA deferred, not rejected**: Acknowledged value for nuisance abstraction.

## 13. Open Questions

1. **Masked spans vs multi-step MSE**: Masked temporal spans (mask 3-5 consecutive frames, predict from context) may be a stronger objective than explicit multi-step prediction. Tier 1 ablation.
2. **FHN parameter ranges**: Map interesting regime from phase diagrams.
3. **Multi-variable observation**: FHN has 2 variables per cell. Expose both (H, W, 2) or only observable (H, W, 1)?
4. **Prediction target space**: Predict raw frame patches or encoder-output patch tokens (with stop-gradient on target encoder)?
5. **Predictor depth**: 2 layers may be insufficient. Tier 2 ablation (2 vs 4 layers).
