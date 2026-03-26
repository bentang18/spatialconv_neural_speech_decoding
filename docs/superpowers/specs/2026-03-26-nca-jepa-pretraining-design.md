# NCA-JEPA Pretraining for Data-Scarce Neural Decoding — Design Spec

**Date**: 2026-03-26
**Author**: Ben Tang, Cogan Lab (Duke)
**Status**: Draft (reviewed, issues fixed)

## 1. Problem Statement

Intraoperative uECOG speech decoding faces an extreme data bottleneck: ~1 minute of utterance per patient (~150 trials), with 128-256 channels at 200Hz. Current supervised approaches train ~60K parameters from scratch on this thin data and overfit heavily (H=128 test: 55x train/val gap; best PER 0.700 with H=32 CE). No self-supervised pretraining method has been tested in this regime.

## 2. Core Idea

Pretrain a latent dynamics model (JEPA) on synthetic Neural Cellular Automata data, then adapt to real cortical recordings, then fine-tune a lightweight classification head per patient.

**The NCA-neural analogy**: Both NCA and neural data are 2D grids evolving over time, driven by a latent cause the model must infer from observations. In NCA, the cause is a fixed transition rule (constant per sequence). In neural data, the cause is the phoneme stimulus (which drives distinct motor activation patterns). While the analogy is imperfect — NCA rules are deterministic and constant, while phoneme-driven dynamics are stochastic and switch 3 times per trial — the core computational skill transfers: extracting spatiotemporal features from a grid and using temporal context to infer what's generating the observed patterns. The model doesn't need NCA and cortical dynamics to be identical; it needs the spatial convolution primitives and temporal tracking abilities that both require.

## 3. Paper Scope

**Title**: *Synthetic Spatiotemporal Pretraining for Data-Scarce Neural Decoding via Latent Predictive Models*

**Type**: Standalone methods paper (separate from current cross-patient architecture paper).

**Contributions**:
1. First application of JEPA (latent predictive world models) to neural decoding
2. NCA pre-pretraining adapted for spatially-structured neural recordings
3. Systematic SSL bake-off on uECOG: NCA+JEPA vs JEPA-only vs MAE vs CPC vs supervised
4. Analysis of what transfers (layers, NCA complexity, latent space structure)

**Validation**: uECOG dataset (8-12 patients). Second dataset deferred until initial results assessed.

**Relationship to existing codebase**: This is a new, standalone codebase. The existing `speech_decoding` pipeline (Conv2d + BiGRU + CTC/CE) remains the current paper's architecture. This spec describes a separate experimental pipeline that shares data loading utilities (`bids_dataset.py`, `audio_features.py`, `grid.py`) but has its own models, training loops, and evaluation.

## 4. Three-Stage Pipeline

### Stage 1: NCA Pre-pretraining (unlimited synthetic data)

**Goal**: Teach encoder and predictor to process 2D grids evolving over time and infer latent rules.

**NCA data generation**:
- All rule families use **continuous scalar values** per cell (floats, not 0/1). This matches uECOG where each electrode produces a continuous z-scored HGA value.
- Initialize an H x W grid with random values
- Apply rule iteratively for 100-500 steps (synchronous update)
- Subsample to ~30-50 frames per sequence (matching 20Hz neural frame rate)
- Grid sizes: 8x16, 12x22, 8x32, 8x34 (all actual uECOG array geometries)
- Volume: ~100K+ sequences per rule family (unlimited, generated on CPU in seconds)

**NCA rule families** (from generic to cortex-specific):

1. **Random MLP** (from Lee et al. 2026)
   - Input: 3x3 neighborhood flattened to 9 continuous values
   - Rule: MLP(9 → 32 → 1), tanh hidden, no output clamping (raw continuous values)
   - Random weights sampled per sequence
   - Produces maximally diverse dynamics: chaos, waves, fixed points, oscillations
   - No biological motivation — forces general spatiotemporal processing

2. **Reaction-Diffusion (Gray-Scott)**
   - Two continuous variables per cell: u (activator), v (inhibitor)
   - Update: du/dt = D_u·nabla²u - uv² + F(1-u); dv/dt = D_v·nabla²v + uv² - (F+k)v
   - Parameters (F, k) sampled randomly per sequence from interesting regime [F∈(0.01,0.06), k∈(0.03,0.07)]
   - Euler integration with dt=1.0, D_u=0.16, D_v=0.08
   - Produces Turing patterns: spots, stripes, waves, splitting blobs
   - Biologically grounded — reaction-diffusion models cortical population dynamics
   - Model sees only u (activator) as the observable per cell

3. **Excitable Media (FitzHugh-Nagumo on grid)**
   - Two continuous variables per cell: v (membrane potential), w (recovery)
   - Update: dv/dt = v - v³/3 - w + D·nabla²v; dw/dt = epsilon·(v + a - b·w)
   - Parameters (a, b, epsilon, D) sampled randomly per sequence
   - Produces traveling waves, spiral waves, refractory dynamics
   - **This IS a model of cortical tissue** — FitzHugh-Nagumo is the standard reduced model of excitable neural populations. Each cell has resting → excited → refractory → resting dynamics, exactly as cortical columns do.
   - The latent parameter (excitability = f(a, b, epsilon)) determines wave behavior, analogous to how the latent phoneme determines cortical activation patterns
   - Model sees only v (membrane potential) as the observable per cell

4. **Damped Wave Propagation**
   - One continuous variable per cell: u (displacement)
   - Update: u_{t+1} = 2u_t - u_{t-1} + c²·nabla²u_t - gamma·(u_t - u_{t-1}) + noise
   - Parameters (c=wave speed, gamma=damping, noise_std) sampled per sequence
   - Random initial impulse locations
   - Clean traveling waves with controllable speed
   - Models cortical wave propagation during speech (posterior → anterior, ~0.1-0.5 m/s on cortex)

5. **Sequential Hotspot Activation** (cortex-specific)
   - One continuous variable per cell: activation level
   - activation(x, y, t) = A · exp(-||(x,y) - pos(t)||² / 2*sigma²) + noise
   - pos(t) follows a random smooth trajectory across the grid (cubic spline through 3-5 waypoints)
   - Parameters (speed, sigma, amplitude, trajectory) sampled per sequence
   - Directly models somatotopic sequential activation during speech (moving activation along motor strip)
   - Very cortex-specific but low diversity (always a moving Gaussian bump)

**NCA rule family ablation** (core question: does cortex similarity or rule diversity matter more?):

| Rule family | Cortex similarity | Diversity | Key dynamics |
|---|---|---|---|
| Random MLP | Low | Very high | Anything: chaos, waves, fixed points |
| Gray-Scott | Medium | Medium | Turing patterns, waves, spots |
| **FitzHugh-Nagumo** | **High** | **Medium** | **Traveling waves, spiral waves, excitable pulses** |
| Damped Wave | Medium-High | Low | Clean propagating waves |
| Sequential Hotspot | Very high | Very low | Moving Gaussian bumps |
| **Mixed (all above)** | **Medium-High** | **Very high** | **Everything** |

Hypothesis: FitzHugh-Nagumo or Mixed wins. FHN because it best matches cortical dynamics (excitable media IS cortex). Mixed because diversity + relevance may outperform either alone. Random MLP may underperform despite diversity because many sequences produce dynamics irrelevant to cortex (chaotic noise, frozen patterns).

**Domain gap mitigation** (applied to ALL rule families):
- Z-score normalize NCA outputs to zero mean, unit variance (matching HGA distribution)
- Add Gaussian noise (std=0.1-0.3) per frame (matching neural noise floor)
- Apply temporal smoothing (Butterworth LPF ~8-10Hz equivalent before subsampling) to match HGA autocorrelation structure — NCA dynamics are "crisp" (deterministic update) while HGA at 20Hz is smooth (envelope + filtering)
- Complexity control within each family via gzip compression ratio of output sequences

**Training**:
- Encoder maps one frame (H x W) to latent z_t (d-dim)
- Predictor maps z_t to predicted next latent z-hat_{t+1}
- Loss: L_pred = MSE(z-hat_{t+1}, z_{t+1}) + lambda * SIGReg(Z)
- SIGReg enforces isotropic Gaussian latent distribution (anti-collapse)
- No action conditioning (NCA dynamics are autonomous; rule is latent)
- No reconstruction decoder, no EMA, no stop-gradient
- Single hyperparameter: lambda (default 0.1, tune via bisection)

**Collapse risk mitigation**: SIGReg is the primary anti-collapse mechanism (following LeWM). However, JEPA without EMA/stop-gradient is a known collapse risk (V-JEPA and I-JEPA both use EMA). We monitor effective rank of the latent space (rank of Z's covariance matrix) during training. If effective rank drops below d/2 (collapse signal), we add an EMA target encoder as fallback. This is a research risk, not an architectural commitment.

**Key insight**: Each NCA sequence has a unique latent rule. The model must infer the rule from context to predict accurately. This teaches in-context rule inference — the same computational primitive needed when the model encounters neural data where the "rule" is the phoneme being produced.

### Stage 2: Neural JEPA Pretraining (all patients pooled, no labels)

**Goal**: Adapt encoder and predictor from synthetic to real cortical dynamics.

**Data**: ALL trials from ALL patients (~1500+ trials from PhonemeSequence task, plus optionally Lexical task patients for additional unsupervised data — Stage 2 uses no labels, so the different phoneme set in Lexical data is irrelevant). No phoneme labels used.

**Training**:
- Initialize encoder + predictor from Stage 1 weights
- Per-patient patch embedding: lightweight Linear layer mapping each patient's grid patches to shared latent space (see Section 5 for exact param counts)
- Same JEPA loss: MSE prediction + SIGReg
- The encoder adapts spatial feature extraction from NCA patterns to real HGA patterns
- The predictor adapts temporal dynamics modeling from NCA evolution to cortical dynamics

**Why pooling is correct**: JEPA is self-supervised (predicts next latent frame, no labels). Using all patients' data causes zero label leakage. The model learns "how do cortical grids evolve?" across the full population. Analogous to BERT pretraining on all text including evaluation text.

### Stage 3: Per-Patient Fine-tuning (labeled data)

**Goal**: Train a lightweight classification head on pretrained features.

**Training**:
- Freeze encoder + predictor (or light fine-tuning, ablated)
- For each patient independently:
  - Encode trial frames and contextualize (see Section 5 for full forward pass)
  - CTC decode: log_softmax(Linear(d, 10)) applied to h_{1:T'}, CTC loss with targets [p1, p2, p3]
  - 5-fold stratified CV within patient
- Trainable params: ~650 (CTC head) + ~320 (per-patient embed) = ~970 total

**Loss options** (CTC primary, others as ablations):
- **CTC** (primary): Alignment-free, no MFA dependency, maximally rigorous. Less prone to boundary artifacts.
- **CE + learned attention pooling** (ablation): 3 learned position queries (d-dim each) cross-attend to h_{1:T'}, pool, then 9-way CE per position. No MFA dependency. Learns WHERE to look. Adds ~3d + attention params (~500 extra).
- **CE + MFA pooling** (ablation): Pool h's within MFA-derived phoneme time windows, 9-way CE per position. Depends on MFA boundary accuracy.

**Diagnostic value of three-way loss comparison**: If CTC ~ CE+learned ~ CE+MFA, features are good and boundaries don't matter. If CE+MFA >> others, MFA provides real signal. If CE+MFA >> CE+learned ~ CTC, MFA is fitting artifacts. This answers "are MFA boundaries accurate enough?" as a side finding.

## 5. Architecture

**Unified dimension**: d = 64 throughout the entire pipeline. This is the latent dimension, the encoder's internal dimension, and the predictor's working dimension. No projection layers needed between components.

### Preprocessing Pipeline

The preprocessing is designed to match the statistical properties of neural data and synthetic NCA data as closely as possible before they reach the encoder.

**Neural data preprocessing** (Stages 2-3):
```
Raw .fif data (200Hz, z-scored HGA, from upstream pipeline)
  → Anti-aliasing LPF: zero-phase Butterworth, cutoff 8Hz, order 4
  → Temporal subsampling: every 10th sample → 20Hz
  → NO threshold clamping (let the encoder learn what's informative)
  → Dead electrode masking: set dead positions to 0.0
  → Output: (n_trials, H, W, T') where T'=30 for [-0.5,1.0s], T'=50 for [-1.0,1.5s]
```

**Why LPF before subsampling but no threshold clamping:**
- The 8Hz LPF is proper anti-aliasing: 200Hz→20Hz subsampling has Nyquist at 10Hz. Without LPF, frequencies 10-100Hz alias into the 0-10Hz band, corrupting the signal. This is standard signal processing, not optional.
- Threshold clamping (z > 2 only) is NOT applied for model input. Subthreshold modulation (z ≈ 0.5-1.5) contains preparatory motor activity and planning signals. The JEPA encoder's job is to learn what's informative; clamping imposes a human prior that may discard signal. The latent bottleneck (H×W → 64-dim) naturally filters noise during compression.
- The existing upstream preprocessing (CAR, HGA extraction, z-scoring) is unchanged. We consume the same .fif files as the current supervised pipeline.

**NCA data preprocessing** (Stage 1):
```
Raw NCA output (continuous values, rule-dependent range)
  → Z-score normalize: per-sequence zero mean, unit variance
  → Add Gaussian noise: std=0.2 per frame (bridges noise gap with neural data)
  → Temporal smoothing: zero-phase Butterworth, cutoff 8Hz equivalent
    (match the autocorrelation structure of real HGA — NCA dynamics are
     "crisp" while HGA envelopes are smooth)
  → Temporal subsampling: ~30-50 frames per sequence
  → Output: (n_sequences, H, W, T') matching neural data format exactly
```

**Why match NCA to neural statistics**: The domain gap between raw NCA and neural data is substantial: NCA is deterministic and crisp; HGA is stochastic and smooth. Without normalization, noise injection, and temporal smoothing, the Stage 1 pretrained weights may be tuned to a completely different data distribution, making Stage 2 adaptation harder (or worse than random init). The goal: after preprocessing, a batch of NCA frames and a batch of neural frames should be statistically indistinguishable to the encoder's input layer.

**For visualization** (separate from model pipeline):
Threshold clamping (z > 2), LPF at ~7Hz, and slower playback remain useful for human inspection. These settings are visualization-only and do not affect model input.

### Encoder (one spatial frame -> latent vector)

**Default: ViT-Tiny** (d=64, 4 layers, 4 heads, head_dim=16, FFN expansion=2x)
- Input: single subsampled frame (H x W), values are z-scored HGA
- Patch tokenization: 2x2 spatial patches
  - 8x16 grid -> 4x8 = 32 tokens per frame
  - 12x22 grid -> 6x11 = 66 tokens per frame (pad W from 22 to 22, one dead column)
  - 8x32 grid -> 4x16 = 64 tokens
  - 8x34 grid -> 4x17 = 68 tokens
- Patch embedding: Linear(4, 64) = 320 params (no bias). **Shared** in Stage 1 (NCA), **per-patient** in Stages 2-3.
- Positional encoding: learned spatial PE (one per grid position, up to 68 positions)
- Transformer: 4 layers, d=64, 4 heads, head_dim=16, GELU FFN with 2x expansion (d->128->d)
- Output: mean-pool over spatial tokens -> z_t (64-dim)

**Per-layer param count**:
- QKV projection: 3 x 64 x 64 = 12,288
- Output projection: 64 x 64 = 4,096
- FFN: 64 x 128 + 128 x 64 = 16,384
- LayerNorm (x2): 2 x 2 x 64 = 256
- **Per layer: ~33K**
- **4 layers: ~132K**
- Patch embedding (shared): 320
- Positional encoding: 68 x 64 = 4,352
- **Total encoder: ~137K**

**Ablation: CNN-Small**
- Conv2d stack: 1->16->32->64, k=3, pad=1, BatchNorm, ReLU, stride-2 on layers 2-3
- AdaptiveAvgPool2d(2, 4) -> flatten(64x2x4=512) -> Linear(512, 64) -> z_t
- Params: ~50K

### Predictor (latent dynamics model)

**Default: Transformer-Small** (d=64, 2 layers, 4 heads, causal mask, FFN 2x)
- Input: sequence of z_{1:t} (latent vectors from encoder, one per subsampled frame)
- Causal attention mask during pretraining (Stages 1-2): can only attend to past/current frames
- **Bidirectional attention during fine-tuning (Stage 3)**: CTC benefits from full temporal context. The predictor is repurposed as a contextualizer, not a next-step predictor. The causal mask is removed.
- Predicts z-hat_{t+1} during pretraining (causal, autoregressive)
- Outputs h_{1:T'} during fine-tuning (bidirectional, used as features for CTC head)

**Per-layer param count**: same structure as encoder = ~33K per layer
- **2 layers: ~66K**
- Positional encoding: reuse sinusoidal temporal PE
- **Total predictor: ~67K**

**Ablation: GRU-Small**
- 2-layer BiGRU, hidden=32 (output dim = 64 to match d)
- Params: ~37K

### Complete Forward Pass

```
STAGES 1-2 (pretraining):
  frame_t (H x W)
    -> [per-patient patch embed] Linear(4, 64)    # 320 params, per-patient in Stage 2
    -> 32-68 spatial tokens, each 64-dim
    -> ViT encoder (4 layers, d=64)                # 137K params, shared
    -> mean-pool -> z_t (64-dim)

  z_{1:t} (sequence of latents)
    -> Transformer predictor (2 layers, d=64, CAUSAL)  # 67K params, shared
    -> z-hat_{t+1} (64-dim)

  Loss: MSE(z-hat_{t+1}, z_{t+1}) + lambda * SIGReg(Z)


STAGE 3 (fine-tuning, e.g., trial of 30 frames at 20Hz):
  frames_{1:30} (each H x W)
    -> [per-patient patch embed] Linear(4, 64)     # 320 params, FROZEN (from Stage 2)
    -> encode each frame -> z_{1:30}               # encoder FROZEN
    -> Transformer predictor (BIDIRECTIONAL)        # predictor FROZEN
    -> h_{1:30} (contextualized, 64-dim each)
    -> CTC head: Linear(64, 10) + log_softmax      # 650 params, TRAINED
    -> CTC loss with targets [p1, p2, p3]
```

### Per-Patient Components

- Patch embedding: Linear(4, 64) no bias = 256 params per patient. With bias = 320 params.
- Using bias (default): **320 params per patient**
- In Stage 1 (NCA): single shared patch embedding (NCA values are synthetic, no patient variation)
- In Stage 2: per-patient patch embeddings initialized from Stage 1 shared embedding
- In Stage 3: frozen per-patient embeddings from Stage 2

### Parameter Budget

```
Component                    Params    Pretrained?   Notes
───────────────────────────  ────────  ───────────── ─────────────────────
Encoder (ViT-Tiny, d=64)     ~137K    Stage 1+2     4 layers, 4 heads
Predictor (Transformer)        ~67K    Stage 1+2     2 layers, causal->bidir
Per-patient embed (x12)        ~3.8K   Stage 2       320 each, 12 patients
CTC head Linear(64, 10)         650    Stage 3 only  9 phonemes + blank
───────────────────────────  ────────  ───────────── ─────────────────────
Total                        ~209K
  Pretrained (Stages 1+2)   ~208K
  Trained from scratch          650
```

Ratio: 650 params trained on ~150 trials = **~4 params/trial** (vs current ~400 params/trial). Even if we unfreeze the per-patient embed (320 params): **~6 params/trial**.

### Latent Space

- Dimension: d = 64 (default). Sweep: d in {32, 64, 128}.
- SIGReg: M=1024 random projections, lambda=0.1 (following LeWM defaults).
- Collapse monitoring: track effective rank of Z's covariance matrix each epoch.

## 6. SSL Bake-off Methods

All methods use the same Stage 3 (CTC head on frozen features) for fair comparison.

### A. NCA + Neural JEPA (hypothesis winner)
Stage 1: NCA JEPA -> Stage 2: Neural JEPA -> Stage 3: CTC head

### B. Neural JEPA only
Stage 2: Neural JEPA (random init) -> Stage 3: CTC head

### C. NCA JEPA only
Stage 1: NCA JEPA -> Stage 3: CTC head (skip Stage 2)

### D. MAE (Masked Autoencoder)
Stage 2: Mask 50-75% of spatial patch tokens per frame, reconstruct masked patch values via lightweight decoder (Linear(64, 4) per patch). Same ViT encoder architecture. Real neural data. -> Stage 3: CTC head.

### E. CPC (Contrastive Predictive Coding)
Stage 2: Encode frames with ViT, predict future frame embeddings z_{t+k} from z_{1:t} using InfoNCE loss (negatives: other frames in batch). Real neural data. -> Stage 3: CTC head.

### F. Supervised baseline
No pretraining. Train full JEPA architecture (encoder + predictor + CTC head) end-to-end with CTC loss on labeled data. Same architecture, same data, no self-supervised stage.

### G. Random-init JEPA architecture (architecture control)
No pretraining. Freeze randomly-initialized encoder+predictor. Train only CTC head. Tests whether the architecture alone provides useful random features (reservoir computing baseline).

### Key Comparisons

| Comparison | Question |
|-----------|----------|
| A vs F | Does the full pipeline beat from-scratch? (headline) |
| A vs B | Does NCA pre-pretraining add value? (core claim) |
| A vs C | Does neural JEPA adaptation matter? (domain gap) |
| A vs D, E | Is JEPA better than MAE/CPC? |
| F vs G | Is the architecture change alone responsible? (control) |

**Phased execution**: Run A, B, F, G first (core claims require only these 4). Expand to C, D, E if A shows promise.

## 7. Evaluation Framework

### Primary Metrics
- **PER** (Phoneme Error Rate): primary, via CTC greedy decode
- **Balanced accuracy**: 9-way, per position, averaged across positions
- Both computed via 5-fold stratified CV within each patient

### Reporting
- Per-patient results table (all patients)
- Population mean +/- std
- Wilcoxon signed-rank test across patients (paired, N=8-12)
- Bootstrap 95% confidence intervals on population mean (10K resamples) to supplement Wilcoxon given small N

### Statistical Power Considerations
With N=8-12 patients and ~150 trials/patient (30 per CV fold), per-patient estimates are noisy. Wilcoxon with N=8 has limited power for small effects. We:
- Report effect sizes (Cohen's d) alongside p-values
- Use bootstrap CIs rather than relying solely on Wilcoxon
- Prioritize per-patient plots showing individual trajectories (not just means)
- For 7 bake-off comparisons, apply Holm-Bonferroni correction

### Inductive Evaluation (required, not optional)
For each patient:
- **Transductive**: Stage 2 includes target patient's unlabeled data
- **Inductive**: Stage 2 excludes target patient
- Compare both. If similar -> no leakage. If transductive better -> patient-specific adaptation matters.
This preemptively addresses the strongest reviewer objection and is a finding either way.

## 8. Ablations

### A. NCA Rule Family (core ablation)
- Random MLP vs Gray-Scott vs FitzHugh-Nagumo vs Damped Wave vs Sequential Hotspot vs Mixed
- Does cortex-similar dynamics (FHN) transfer better than maximally diverse dynamics (Random MLP)?
- Within each family: low / medium / high complexity bands (gzip compression ratio)
- This is the paper's most distinctive ablation — no prior work compares synthetic data generators for neural pretraining

### B. NCA Grid Geometry
- Matching grids (8x16, 12x22) vs square (16x16) vs mismatched
- Does array geometry transfer matter?

### C. Encoder Architecture
- ViT-Tiny vs CNN-Small with same pretraining recipe

### D. Predictor Architecture
- Transformer vs GRU predictor

### E. Latent Dimension
- d in {32, 64, 128}

### F. Freeze Level (Stage 3)

| Level | Trainable | Params | Expected |
|-------|-----------|--------|----------|
| Head only | CTC head | 650 | Safest. Tests representation quality. |
| Head + embed | + per-patient embed | 970 | Patient normalization adapts. |
| + pred LN | + predictor LayerNorms | ~1.5K | Light temporal adaptation. |
| + last layer | + last predictor layer | ~34K | Moderate unfreezing. |
| Full fine-tune | Everything | ~209K | Likely overfits. |

Hypothesis: Head-only or Head+embed wins. Full fine-tune overfits.

### G. Stage 2 Data Volume
- Train Stage 2 on 2, 4, 6, 8, all patients
- How much real neural data does Stage 2 need?

### H. Loss Type (Stage 3)
- CTC vs CE+learned attention vs CE+MFA pooling
- Diagnostic for MFA boundary accuracy

### I. Latent Space Analysis
- t-SNE/UMAP of latent trajectories colored by phoneme
- Do classes separate before Stage 3 (without labels)?
- Compare: NCA+Neural vs Neural-only vs random init

### J. Layer-wise Probing
- After Stage 1 vs after Stage 2: train linear probes at each encoder/predictor layer
- Which layers change most? (shallow = spatial adaptation, deep = temporal)

### K. Temporal Path Straightness
- Cosine similarity of consecutive latent velocity vectors (from LeWM)
- Do phoneme trajectories become straighter than noise during pretraining?

## 9. Known Risks

1. **JEPA collapse without EMA**: SIGReg alone may not prevent representation collapse. Mitigation: monitor effective rank, add EMA fallback if needed. This is a research risk — if SIGReg is insufficient, we adopt LeWM's exact setup.

2. **NCA-to-neural domain gap too large**: Synthetic NCA dynamics may be too different from real cortical dynamics for positive transfer. Mitigation: (a) normalize NCA to Gaussian (matching HGA), (b) add noise to NCA, (c) ablation C (NCA-only) will quantify this directly. Even a null result (NCA doesn't help) contributes to the bake-off.

3. **Statistical power**: N=8-12 patients with noisy per-patient estimates limits ability to detect small effects. Mitigation: bootstrap CIs, effect sizes, per-patient plots, Holm-Bonferroni correction.

4. **ViT on 30-frame trials**: Even at 20Hz, the predictor processes 30-step sequences. The ViT encoder runs 30 times per trial. This is computationally feasible (30 x 32-68 tokens through 4 layers) but Stage 1 with 100K+ NCA sequences requires attention to throughput. Batched encoding will be essential.

## 10. References

- Lee, Han, Kumar, Agrawal (2026). "Training Language Models via Neural Cellular Automata." arXiv:2603.10055. — NCA pre-pretraining for language models.
- Maes, Le Lidec, Scieur, LeCun, Balestriero (2026). "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels." arXiv:2603.19312. — JEPA with SIGReg for world models.
- Spalding et al. (2025). Cross-patient speech decoding from uECOG. — Baseline dataset and evaluation.
- Singh et al. (2025). Per-patient layers + frozen shared backbone. — Transfer protocol.
- Duraivel et al. (2025). Pseudo-word repetition with overlapping patients. — Cross-task data source.

## 11. Architecture Diagram

See `docs/nca_jepa_pipeline.svg` for the full three-stage data flow diagram.

## 12. Resolved Design Decisions

1. **Encoder frame rate**: Anti-aliasing LPF (8Hz Butterworth) then subsample 200Hz -> 20Hz. Field standard, computationally feasible, proper signal processing.
2. **Predictor context**: Full trial sequence (30-50 frames at 20Hz). Causal during pretraining, bidirectional during fine-tuning.
3. **NCA state space**: Continuous scalar values (floats, not 0/1). Z-score normalized, noise-injected, and temporally smoothed to match HGA statistics.
4. **NCA rule families**: Five families from generic (Random MLP) to cortex-specific (FitzHugh-Nagumo, Sequential Hotspot), plus Mixed. Rule family is a core ablation axis.
5. **Cross-task data**: Include Lexical task patients in Stage 2 (unsupervised, label-free). Different phoneme set is irrelevant since Stage 2 uses no labels.
6. **Unified dimension**: d=64 throughout (encoder, predictor, latent space). No projection layers between components.
7. **No threshold clamping for model input**: Subthreshold neural activity contains preparatory/planning signal. The encoder learns what's informative. Clamping is visualization-only.
8. **Preprocessing parity**: NCA and neural data undergo matching preprocessing (z-score, noise, LPF, subsample) so they are statistically similar at the encoder input.

## 13. Open Questions

1. **Multi-step prediction**: Predict z_{t+1} only (1-step) vs z_{t+1:t+k} (multi-step). Multi-step is harder but may learn richer dynamics. Could be an ablation.
2. **Predictor depth**: 2 layers may be insufficient for modeling complex temporal dependencies. Could sweep 2-4 layers.
3. **FHN parameter ranges**: The FitzHugh-Nagumo parameter space has large regions that produce trivial dynamics (fixed points) or numerical instability. Need to map the "interesting" regime before generating at scale. Same for Gray-Scott (F, k) parameter space — known phase diagrams exist in the literature.
4. **Multi-variable NCA observation**: FHN and Gray-Scott have 2 variables per cell (v+w, u+v). The model sees only 1 (v or u). Should we expose both channels? This would make the encoder input (H, W, 2) instead of (H, W, 1), closer to multi-band neural data but departing from the scalar-per-electrode format.
