# NCA Pretraining for Data-Scarce Neural Decoding — Design Spec

**Date**: 2026-03-26
**Author**: Ben Tang, Cogan Lab (Duke)
**Status**: Draft v3

## 1. Problem Statement

Intraoperative uECOG speech decoding faces an extreme data bottleneck: ~1 minute of utterance per patient (~150 trials), with 128-256 channels at 200Hz. Current supervised approaches train ~60K parameters from scratch on this thin data and overfit heavily (H=128 test: 55x train/val gap; best PER 0.700 with H=32 CE). No self-supervised pretraining method has been tested in this regime.

## 2. Core Idea

Pretrain an encoder-predictor-decoder model on synthetic Neural Cellular Automata data via **next-frame prediction in observation space**, then adapt to real cortical recordings, then fine-tune a lightweight classification head per patient.

**The NCA-neural analogy**: Both NCA and neural data are 2D grids evolving over time, driven by a latent cause the model must infer from observations. In NCA, the cause is a fixed transition rule (constant per sequence). In neural data, the cause is the phoneme stimulus (which drives distinct motor activation patterns). While the analogy is imperfect — NCA rules are deterministic and constant, while phoneme-driven dynamics are stochastic and switch 3 times per trial — the core computational skill transfers: extracting spatiotemporal features from a grid and using temporal context to infer what's generating the observed patterns.

**Why observation-space prediction, not JEPA**: JEPA (predicting in latent space) was designed for high-dimensional observations (84x84x3 pixel images) where predicting in observation space wastes capacity on irrelevant pixel details. Our observations are only 128-264 scalar values per frame — there are no irrelevant details to avoid. Meanwhile, JEPA's latent-space prediction is prone to representation collapse (a known failure mode requiring EMA/stop-gradient heuristics). Observation-space prediction is collapse-proof (the target is a fixed, real frame), simpler to implement, and closer to the NCA paper (Lee et al. 2026) which uses observation-space next-token prediction. JEPA remains a future ablation once the core pipeline is validated.

## 3. Paper Scope

**Title**: *Synthetic Spatiotemporal Pretraining for Data-Scarce Neural Decoding*

**Type**: Standalone methods paper (separate from current cross-patient architecture paper).

**Contributions**:
1. NCA pre-pretraining adapted for spatially-structured neural recordings — extending Lee et al. 2026 from language to 2D grid data where the structural match is tighter
2. Systematic SSL comparison on uECOG: NCA pretrain vs real-data pretrain vs MAE vs supervised
3. NCA rule family analysis: does cortex-aligned synthetic data (FitzHugh-Nagumo) transfer better than generic dynamics (Random MLP)?
4. Analysis of what transfers (layers, latent space structure, pretraining data volume)

**Validation**: uECOG dataset (8-12 patients). Second dataset deferred until initial results assessed.

**Relationship to existing codebase**: Separate experimental pipeline. Shares data loading utilities (`bids_dataset.py`, `audio_features.py`, `grid.py`) but has its own models, training loops, and evaluation.

## 4. Three-Stage Pipeline

### Stage 1: NCA Pre-pretraining (unlimited synthetic data)

**Goal**: Teach encoder and predictor to process 2D grids evolving over time and infer latent rules from observed dynamics.

**NCA data generation**:
- All rule families use **continuous scalar values** per cell (floats, not 0/1). This matches uECOG where each electrode produces a continuous z-scored HGA value.
- Initialize an H x W grid with random values
- Apply rule iteratively for 100-500 steps (synchronous update)
- Grid sizes: 8x16, 12x22, 8x32, 8x34 (all actual uECOG array geometries)
- Volume: ~100K+ sequences per rule family (unlimited, generated on CPU in seconds)

**NCA rule families** (from generic to cortex-specific):

1. **Random MLP** (from Lee et al. 2026)
   - Input: 3x3 neighborhood flattened to 9 continuous values
   - Rule: MLP(9 -> 32 -> 1), tanh hidden, no output clamping (raw continuous values)
   - Random weights sampled per sequence
   - Produces maximally diverse dynamics: chaos, waves, fixed points, oscillations
   - No biological motivation — forces general spatiotemporal processing

2. **Reaction-Diffusion (Gray-Scott)**
   - Two continuous variables per cell: u (activator), v (inhibitor)
   - Update: du/dt = D_u nabla^2 u - uv^2 + F(1-u); dv/dt = D_v nabla^2 v + uv^2 - (F+k)v
   - Parameters (F, k) sampled randomly per sequence from interesting regime [F in (0.01,0.06), k in (0.03,0.07)]
   - Euler integration with dt=1.0, D_u=0.16, D_v=0.08
   - Produces Turing patterns: spots, stripes, waves, splitting blobs
   - Biologically grounded — reaction-diffusion models cortical population dynamics
   - Model sees only u (activator) as the observable per cell

3. **Excitable Media (FitzHugh-Nagumo on grid)**
   - Two continuous variables per cell: v (membrane potential), w (recovery)
   - Update: dv/dt = v - v^3/3 - w + D nabla^2 v; dw/dt = epsilon (v + a - bw)
   - Parameters (a, b, epsilon, D) sampled randomly per sequence
   - Produces traveling waves, spiral waves, refractory dynamics
   - **This IS a model of cortical tissue** — FitzHugh-Nagumo is the standard reduced model of excitable neural populations. Each cell has resting -> excited -> refractory -> resting dynamics, exactly as cortical columns do.
   - The latent parameter (excitability = f(a, b, epsilon)) determines wave behavior, analogous to how the latent phoneme determines cortical activation patterns
   - Model sees only v (membrane potential) as the observable per cell

4. **Damped Wave Propagation**
   - One continuous variable per cell: u (displacement)
   - Update: u_{t+1} = 2u_t - u_{t-1} + c^2 nabla^2 u_t - gamma (u_t - u_{t-1}) + noise
   - Parameters (c=wave speed, gamma=damping, noise_std) sampled per sequence
   - Random initial impulse locations
   - Clean traveling waves with controllable speed
   - Models cortical wave propagation during speech (posterior -> anterior, ~0.1-0.5 m/s on cortex)

5. **Sequential Hotspot Activation** (cortex-specific)
   - One continuous variable per cell: activation level
   - activation(x, y, t) = A exp(-||(x,y) - pos(t)||^2 / 2 sigma^2) + noise
   - pos(t) follows a random smooth trajectory across the grid (cubic spline through 3-5 waypoints)
   - Parameters (speed, sigma, amplitude, trajectory) sampled per sequence
   - Directly models somatotopic sequential activation during speech
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

Hypothesis: FitzHugh-Nagumo or Mixed wins. FHN because excitable media IS cortex. Mixed because diversity + relevance may outperform either alone.

**Training objective — next-frame prediction in observation space**:
```
frame_t (H x W) -> Encoder -> z_t (d-dim) -> Predictor(z_t) -> z-hat_{t+1} (d-dim) -> Decoder -> predicted_frame_{t+1} (H x W)

Loss = MSE(predicted_frame_{t+1}, actual_frame_{t+1})
     + lambda * SIGReg(Z)     [optional: encourages structured latent geometry]
```

- The encoder compresses spatial information (H x W -> d-dim)
- The predictor models temporal dynamics in latent space (z_t -> z-hat_{t+1})
- The decoder reconstructs the prediction back to observation space (d-dim -> H x W)
- **The reconstruction target is a real frame — no collapse is possible.** The trivial solution (constant output) produces high MSE against diverse frames.
- The decoder is lightweight (~16K params) and **discarded after pretraining** — not used during fine-tuning
- SIGReg is optional: it encourages nicer latent geometry (useful for analysis) but is not needed for stability
- No EMA, no stop-gradient, no contrastive negatives. Simple end-to-end MSE.

**Key insight**: Each NCA sequence has a unique latent rule. The model must infer the rule from context to predict the next frame accurately. This teaches in-context rule inference — the same skill needed to infer phoneme identity from neural dynamics.

### Stage 2: Neural Pretraining (all patients pooled, no labels)

**Goal**: Adapt encoder and predictor from synthetic to real cortical dynamics.

**Data**: ALL trials from ALL patients (~1500+ trials from PhonemeSequence task, plus optionally Lexical task patients). No phoneme labels used.

**Training**:
- Initialize encoder + predictor + decoder from Stage 1 weights
- Per-patient patch embedding: lightweight Linear layer mapping each patient's grid patches to shared latent space
- Same objective: next-frame prediction MSE (+ optional SIGReg)
- Per-patient decoder: Linear(d, H*W) maps back to patient's specific grid size
- The encoder adapts spatial features from NCA patterns to real HGA patterns
- The predictor adapts temporal dynamics from NCA evolution to cortical dynamics

**Why pooling is correct**: The pretraining objective is self-supervised (predicts next frame, no labels). Using all patients' data causes zero label leakage. The model learns "how do cortical grids evolve?" across the full population.

### Stage 3: Per-Patient Fine-tuning (labeled data)

**Goal**: Train a lightweight classification head on pretrained features.

**Training**:
- Freeze encoder + predictor. Decoder is discarded.
- For each patient independently:
  - Encode trial frames and contextualize (see Section 5 for full forward pass)
  - CTC decode: log_softmax(Linear(d, 10)) applied to h_{1:T'}, CTC loss with targets [p1, p2, p3]
  - 5-fold stratified CV within patient
- Trainable params: ~1.3K (CTC head) + ~640 (per-patient embed) = ~1.9K total

**Loss options** (CTC primary, others as ablations):
- **CTC** (primary): Alignment-free, no MFA dependency, maximally rigorous.
- **CE + learned attention pooling** (ablation): 3 learned position queries cross-attend to h_{1:T'}, pool, then 9-way CE per position. No MFA dependency. Learns WHERE to look.
- **CE + MFA pooling** (ablation): Pool h's within MFA-derived phoneme time windows, 9-way CE per position. Depends on MFA boundary accuracy.

**Diagnostic value of three-way loss comparison**: If CTC ~ CE+learned ~ CE+MFA, features are good and boundaries don't matter. If CE+MFA >> CE+learned ~ CTC, MFA is fitting artifacts.

## 5. Architecture

**Unified dimension**: d = 128 throughout. This gives each attention head a 32-dim subspace (4 heads x 32 = 128), which is the minimum for expressive attention patterns. Sweep: d in {64, 128, 256}.

### Preprocessing Pipeline

**Neural data preprocessing** (Stages 2-3):
```
Raw .fif data (200Hz, z-scored HGA, from upstream pipeline)
  -> Anti-aliasing LPF: zero-phase Butterworth, cutoff 8Hz, order 4
  -> Temporal subsampling: every 10th sample -> 20Hz
  -> NO threshold clamping (let the encoder learn what's informative)
  -> Dead electrode masking: set dead positions to 0.0
  -> Output: (n_trials, H, W, T') where T'=30 for [-0.5,1.0s], T'=50 for [-1.0,1.5s]
```

- The 8Hz LPF is proper anti-aliasing: 200Hz->20Hz subsampling has Nyquist at 10Hz. Without LPF, frequencies 10-100Hz alias into the 0-10Hz band. Standard signal processing.
- No threshold clamping: subthreshold modulation (z ~ 0.5-1.5) contains preparatory motor activity. The encoder learns what's informative; clamping imposes a human prior that may discard signal.
- The existing upstream preprocessing (CAR, HGA extraction, z-scoring) is unchanged.

**NCA data preprocessing** (Stage 1):
```
Raw NCA output (continuous values, rule-dependent range)
  -> Z-score normalize: per-sequence zero mean, unit variance
  -> Add Gaussian noise: std=0.2 per frame (bridges noise gap)
  -> Temporal smoothing: zero-phase Butterworth, cutoff 8Hz equivalent
     (match HGA autocorrelation — NCA is "crisp", HGA is smooth)
  -> Temporal subsampling: ~30-50 frames per sequence
  -> Output: (n_sequences, H, W, T') matching neural data format
```

**Goal**: After preprocessing, NCA frames and neural frames should be statistically similar at the encoder input — both z-scored, both temporally smooth at 20Hz, both noisy.

**Visualization** (separate from model): threshold clamping (z > 2), LPF ~7Hz, slower playback. These are visualization-only settings.

### Encoder (one spatial frame -> latent vector)

**Default: ViT-Small** (d=128, 4 layers, 4 heads, head_dim=32, FFN expansion=2x)
- Input: single subsampled frame (H x W), values are z-scored HGA (or preprocessed NCA)
- Patch tokenization: 2x2 spatial patches
  - 8x16 grid -> 32 tokens; 12x22 -> 66 tokens; 8x32 -> 64 tokens; 8x34 -> 68 tokens
- Patch embedding: Linear(4, 128) = 640 params. **Shared** in Stage 1, **per-patient** in Stages 2-3.
- Positional encoding: learned spatial PE (up to 68 positions)
- Transformer: 4 layers, d=128, 4 heads, head_dim=32, GELU FFN with 2x expansion (128->256->128)
- Output: mean-pool over spatial tokens -> z_t (128-dim)

**Per-layer param count**:
- QKV: 3 x 128 x 128 = 49,152
- Output projection: 128 x 128 = 16,384
- FFN: 128 x 256 + 256 x 128 = 65,536
- LayerNorm (x2): 2 x 2 x 128 = 512
- **Per layer: ~131K**
- **4 layers: ~525K**
- Patch embedding (shared): 640
- Positional encoding: 68 x 128 = 8,704
- **Total encoder: ~535K**

**Ablation: CNN-Small**
- Conv2d stack: 1->16->32->64->128, k=3, pad=1, BatchNorm, ReLU
- AdaptiveAvgPool2d(2, 4) -> flatten -> Linear(1024, 128) -> z_t
- Params: ~160K

### Predictor (latent dynamics model)

**Default: Transformer-Small** (d=128, 2 layers, 4 heads, causal mask, FFN 2x)
- Input: sequence of z_{1:t} (latent vectors from encoder, one per subsampled frame)
- Causal attention mask during pretraining (Stages 1-2): can only attend to past/current
- **Bidirectional during fine-tuning (Stage 3)**: CTC benefits from full temporal context. Predictor is repurposed as a contextualizer, causal mask removed.

**Per-layer**: ~131K (same structure as encoder)
- **2 layers: ~262K**
- Sinusoidal temporal PE
- **Total predictor: ~263K**

**Ablation: GRU-Small**
- 2-layer BiGRU, hidden=64 (output dim = 128 to match d)
- Params: ~100K

### Decoder (latent -> next frame, pretraining only)

**Lightweight, discarded after pretraining.**
- Linear(128, H*W): maps d-dim latent prediction back to observation space
- For 8x16 grid: Linear(128, 128) = 16.5K params
- For 12x22 grid: Linear(128, 264) = 34K params
- In Stage 1 (NCA): single shared decoder per grid size
- In Stage 2: per-patient decoder (same architecture, different grid sizes)
- **NOT used in Stage 3** — only the encoder and predictor are kept

### Complete Forward Pass

```
STAGES 1-2 (pretraining — next-frame prediction):
  frame_t (H x W)
    -> [patch embed] Linear(4, 128)               # 640 params
    -> 32-68 spatial tokens, each 128-dim
    -> ViT encoder (4 layers, d=128)               # 535K params, shared
    -> mean-pool -> z_t (128-dim)

  z_{1:t} (sequence of latents)
    -> Transformer predictor (2 layers, d=128, CAUSAL)  # 263K params, shared
    -> z-hat_{t+1} (128-dim)

  z-hat_{t+1}
    -> Decoder: Linear(128, H*W)                   # ~17-34K params, DISCARDED after pretraining
    -> predicted_frame_{t+1} (H x W)

  Loss: MSE(predicted_frame, actual_frame) + lambda * SIGReg(Z)


STAGE 3 (fine-tuning, e.g., trial of 30 frames at 20Hz):
  frames_{1:30} (each H x W)
    -> [per-patient patch embed] Linear(4, 128)    # 640 params, FROZEN
    -> encode each frame -> z_{1:30}               # encoder FROZEN
    -> Transformer predictor (BIDIRECTIONAL)        # predictor FROZEN
    -> h_{1:30} (contextualized, 128-dim each)
    -> CTC head: Linear(128, 10) + log_softmax     # 1,290 params, TRAINED
    -> CTC loss with targets [p1, p2, p3]
```

### Per-Patient Components

- Patch embedding: Linear(4, 128) with bias = **640 params per patient**
- In Stage 1: single shared patch embedding
- In Stage 2: per-patient embeddings initialized from Stage 1 shared embedding
- In Stage 3: frozen per-patient embeddings from Stage 2
- Per-patient decoder (Stage 2 only): Linear(128, H*W) = ~17-34K. Discarded after Stage 2.

### Parameter Budget

```
Component                    Params    Pretrained?   Notes
------------------------------------------------------
Encoder (ViT, d=128)         ~535K    Stage 1+2     4 layers, 4 heads
Predictor (Transformer)      ~263K    Stage 1+2     2 layers, causal->bidir
Decoder (pretraining only)    ~34K    Stage 1+2     Linear, DISCARDED at Stage 3
Per-patient embed (x12)       ~7.7K   Stage 2       640 each, 12 patients
CTC head Linear(128, 10)      1,290   Stage 3 only  9 phonemes + blank
------------------------------------------------------
Total (pretraining)          ~840K    (including decoder)
Total (fine-tuning)          ~807K    (decoder discarded)
  Frozen at Stage 3          ~806K
  Trained from scratch         1,290
```

Ratio: 1,290 params trained on ~150 trials = **~9 params/trial** (vs current ~400 params/trial).

**Proportional sizing**: LeWM uses ~15M params for 84x84x3 = 21,168-value images. Proportionally, 128-264 value grids need 15M x (200/21168) ~ 142K. Our ~800K is **5-6x larger** than proportional — giving ample capacity for our observation complexity.

### Latent Space

- Dimension: d = 128 (default). Sweep: d in {64, 128, 256}.
- SIGReg: M=1024 random projections, lambda=0.1. Optional (model trains fine without it since reconstruction prevents collapse). Included for nicer latent geometry.

## 6. SSL Bake-off Methods (Initial Round)

All methods use the same Stage 3 (CTC head on frozen features) for fair comparison. Initial round focuses on core claims with 4 methods. Expand later if warranted.

### A. NCA + Neural Pretrain (hypothesis winner)
Stage 1: NCA next-frame prediction -> Stage 2: Neural next-frame prediction -> Stage 3: CTC head

### B. Neural Pretrain Only
Stage 2: Neural next-frame prediction (random init) -> Stage 3: CTC head

### C. Supervised Baseline
No pretraining. Train encoder + predictor + CTC head end-to-end with CTC loss on labeled data per patient. Same architecture, no self-supervised stage.

### D. Random-Init Architecture Control
No pretraining. Freeze randomly-initialized encoder+predictor. Train only CTC head. Tests whether architecture alone provides useful random features (reservoir computing baseline).

### Key Comparisons (initial round)

| Comparison | Question |
|-----------|----------|
| A vs C | Does the full pipeline beat from-scratch? (headline) |
| A vs B | Does NCA pre-pretraining add value beyond real-data SSL? (core claim) |
| C vs D | Is the architecture responsible, or is it the pretraining? (control) |
| B vs C | Does real-data SSL alone help? (SSL value) |

### Experiment Execution Order

Run cheapest experiments first to establish baselines and detect dead ends early.

```
Step 1: Method D (random-init frozen encoder+predictor, CTC head only)
  Cost:    Minutes. No pretraining needed.
  Purpose: Sets the noise floor. If frozen random features already decode
           well, the whole framing changes (reservoir computing baseline).
  Go/no-go: If D gives PER < 0.85, random features are non-trivial.
            Proceed to Step 2 regardless.

Step 2: Method C (supervised from scratch, same architecture)
  Cost:    ~30 min per patient (5-fold CV).
  Purpose: Establishes what the architecture can do with direct supervision.
           This is the number to beat.
  Go/no-go: If C ≈ D, the architecture has no value even with supervision.
            Rethink architecture before proceeding.

Step 3: Method B (neural-only pretrain, no NCA)
  Cost:    Hours (Stage 2 pretraining on ~1500 trials + Stage 3 per patient).
  Purpose: Does SSL on real data help? If B >> C, SSL works and NCA
           pretraining only needs to improve on it. If B ≈ C, SSL on
           ~1500 trials isn't enough — NCA's unlimited data is the
           only path forward.
  Go/no-go: If B ≈ D (pretrained features no better than random),
            something is fundamentally wrong with the pretraining
            objective or architecture. Debug before adding NCA.

Step 4: Method A (NCA + neural pretrain)
  Cost:    Hours (Stage 1 NCA + Stage 2 neural + Stage 3 per patient).
  Purpose: The hypothesis test. Does NCA pre-pretraining add value?
  Interpretation:
    A > B > C > D  →  Full pipeline works, NCA adds value. Write paper.
    A ≈ B > C > D  →  NCA doesn't add value, but real-data SSL works.
                       Pivot paper to "SSL for data-scarce neural decoding."
    A ≈ B ≈ C > D  →  Neither SSL helps. Architecture + supervision is
                       sufficient. Pretraining story doesn't hold.
    A ≈ B ≈ C ≈ D  →  Architecture is wrong. Rethink everything.
```

### Future Round (expand if A shows promise)

| Method | Description |
|--------|-------------|
| E. NCA Only | Stage 1 only, skip Stage 2 → tests domain gap |
| F. MAE | Mask 50-75% of spatial patches, reconstruct. Real neural data. |
| G. CPC | Contrastive predictive coding. InfoNCE on real neural data. |
| H. JEPA | Latent-space prediction + SIGReg (no decoder). Tests if latent prediction helps at our scale. |

## 7. Evaluation Framework

### Primary Metrics
- **PER** (Phoneme Error Rate): primary, via CTC greedy decode
- **Balanced accuracy**: 9-way, per position, averaged across positions
- Both computed via 5-fold stratified CV within each patient

### Reporting
- Per-patient results table (all patients)
- Population mean +/- std
- Wilcoxon signed-rank test across patients (paired, N=8-12)
- Bootstrap 95% confidence intervals (10K resamples)

### Statistical Power
N=8-12 patients with ~150 trials each (30 per fold). Wilcoxon has limited power at N=8. We:
- Report effect sizes (Cohen's d) alongside p-values
- Use bootstrap CIs rather than relying on Wilcoxon alone
- Prioritize per-patient plots (not just means)
- Holm-Bonferroni correction for multiple comparisons

### Inductive Evaluation (required)
For each patient:
- **Transductive**: Stage 2 includes target patient's unlabeled data
- **Inductive**: Stage 2 excludes target patient
- Compare both. If similar -> no leakage concern. If transductive better -> patient-specific adaptation matters. Either result is publishable.

## 8. Ablations

### A. NCA Rule Family (core ablation)
- Random MLP vs Gray-Scott vs FitzHugh-Nagumo vs Damped Wave vs Sequential Hotspot vs Mixed
- Does cortex-aligned dynamics (FHN) transfer better than maximally diverse (Random MLP)?
- Within each family: low / medium / high complexity bands (gzip compression ratio)
- This is the paper's most distinctive ablation

### B. NCA Grid Geometry
- Matching grids (8x16, 12x22) vs square (16x16) vs mismatched
- Does array geometry transfer matter?

### C. Encoder Architecture
- ViT-Small vs CNN-Small with same pretraining recipe

### D. Predictor Architecture
- Transformer vs GRU predictor

### E. Latent Dimension
- d in {64, 128, 256}

### F. Freeze Level (Stage 3)

| Level | Trainable | Params | Expected |
|-------|-----------|--------|----------|
| Head only | CTC head | 1.3K | Safest. Tests representation quality. |
| Head + embed | + per-patient embed | 1.9K | Patient normalization adapts. |
| + pred LN | + predictor LayerNorms | ~3K | Light temporal adaptation. |
| + last layer | + last predictor layer | ~134K | Moderate unfreezing. |
| Full fine-tune | Everything | ~807K | Likely overfits. |

Hypothesis: Head-only or Head+embed wins.

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
- After Stage 1 vs after Stage 2: train linear probes at each layer
- Which layers change most? (shallow = spatial, deep = temporal)

### K. Temporal Path Straightness
- Cosine similarity of consecutive latent velocity vectors (inspired by LeWM)
- Do phoneme trajectories become straighter than noise during pretraining?

## 9. Known Risks

1. **NCA-to-neural domain gap too large**: Synthetic NCA dynamics may be too different from real cortical dynamics for positive transfer. Mitigation: (a) normalize NCA to Gaussian (matching HGA), (b) add noise, (c) temporal smoothing, (d) ablation E (NCA-only) quantifies this. Even a null result contributes to the bake-off.

2. **Statistical power**: N=8-12 patients with noisy per-patient estimates limits ability to detect small effects. Mitigation: bootstrap CIs, effect sizes, per-patient plots, Holm-Bonferroni correction.

3. **Stage 1 throughput**: 100K+ NCA sequences x 30-50 frames x ViT encoding. Computationally feasible with batched encoding but requires attention to throughput. Estimate: ~2-4 hours on single GPU for Stage 1.

4. **FHN parameter instability**: FitzHugh-Nagumo has large parameter regions producing trivial dynamics (fixed points) or numerical blow-up. Need to map the "interesting" regime using known phase diagrams before generating at scale.

## 9b. Precautionary Lessons from Prior Experiments

Hard-won insights from the existing supervised pipeline that directly apply to this pretraining pipeline.

### Data scarcity is the wall, not architecture
H=128 GRU showed 55x train/val gap — more capacity = more overfitting with ~150 trials. **If pretrained features aren't good enough to freeze, unfreezing will overfit immediately.** The freeze-level ablation (Section 8F) is the make-or-break diagnostic, not a nice-to-have. If "head only" doesn't work, the pretraining failed.

### Mean-pooling over spatial tokens collapses spatial information
The current pipeline's Pool(2,4) at 4mm couldn't resolve 3-5mm somatotopy. Our ViT uses 2x2 patches (~1mm, fine), but **mean-pooling all spatial tokens into one z_t vector** may lose spatial structure that the predictor needs. Consider: does the predictor need to know WHERE on the grid activity is occurring, or just WHAT the overall pattern looks like? If spatial position matters for temporal prediction, keep N spatial tokens rather than collapsing to 1 vector. This is an architecture decision worth ablating.

### LOPO failed near chance — pooled training is risky
LOPO got PER=0.846 with 7 source patients. Stage 1 multi-patient SGD early-stopped at step 800 (val diverging). **Stage 2 neural pretraining (all patients pooled) faces the same risk** — if patient-specific noise dominates, the predictor learns to predict noise. Monitor: does prediction loss decrease similarly across patients, or do some patients dominate? If one patient's loss drops 10x faster, the model is memorizing that patient's dynamics.

### Augmentation tuning is within noise at this data scale
Swept time-shift, amplitude, noise, dropout — all within PER 0.78-0.83. **Don't spend time fine-tuning NCA preprocessing params** (noise std, LPF cutoff, subsampling rate). Get a reasonable default, validate it runs, move on. Big gains come from whether NCA pretraining helps at all, not from noise_std=0.2 vs 0.15.

### Auxiliary targets need negative controls — acoustic regression was an artifact
Acoustic regression R² (~0.07 for Mel features) looked positive, but shifted control (+500ms) gave ~0.065 — not alignment-sensitive. The signal was trial structure, not motor→acoustic mapping. **Method D (random-init control) serves the same purpose for this pipeline.** Run it first. If frozen random features give similar PER to pretrained ones, pretraining learned nothing useful.

### CE loss may have been fitting trial artifacts
CE beat CTC (PER 0.700 vs 0.778), but it may have exploited epoch structure rather than neural dynamics. **The three-way Stage 3 loss comparison (CTC vs CE+learned attention vs CE+MFA) resolves this.** If CE+MFA >> CE+learned ≈ CTC on the SAME frozen features, MFA boundaries were leaking non-neural information.

### MPS breaks with mixed grid sizes
AdaptiveAvgPool2d silently falls back to CPU for non-divisible sizes. Our ViT avoids this (variable token count, no adaptive pooling), but **test with ALL grid sizes early**: 8x16, 12x22, 8x32, 8x34. Don't develop only on S14 (8x16) and discover integration bugs at scale.

### Stratified splits are essential
S33 has 52 trials across 52 possible tokens. Some tokens appear 0-1 times. **Use StratifiedKFold on the first phoneme label**, as the existing pipeline does. Inventory class distribution per patient before running any experiment.

### Per-patient layers are non-negotiable for cross-patient work
Singh (25 patients) achieves group PER 0.49 via per-patient Conv1D + shared LSTM. Without per-patient layers, pooling data across patients causes catastrophic interference. **Our per-patient patch embedding (640 params each) is the minimum required adaptation mechanism.** Do not skip it, even if it seems small.

## 10. References

- Lee, Han, Kumar, Agrawal (2026). "Training Language Models via Neural Cellular Automata." arXiv:2603.10055. — NCA pre-pretraining.
- Maes, Le Lidec, Scieur, LeCun, Balestriero (2026). "LeWorldModel." arXiv:2603.19312. — JEPA with SIGReg (architectural reference).
- Spalding et al. (2025). Cross-patient speech decoding from uECOG. — Baseline dataset.
- Singh et al. (2025). Per-patient layers + frozen shared backbone. — Transfer protocol.
- Duraivel et al. (2025). Pseudo-word repetition with overlapping patients. — Cross-task data.

## 11. Architecture Diagram

See `docs/nca_jepa_pipeline.svg` for the pipeline diagram. (Note: diagram shows JEPA framing; the observation-space prediction adds a decoder in Stages 1-2 that is discarded at Stage 3. Core data flow is unchanged.)

## 12. Resolved Design Decisions

1. **Observation-space prediction over JEPA**: Collapse-proof, simpler, closer to the NCA paper. JEPA deferred to future ablation.
2. **d=128 over d=64**: head_dim=32 gives expressive attention patterns. d=64 (head_dim=16) was too small. ~800K total params is 5-6x proportional to LeWM scaled by observation size.
3. **Anti-aliasing LPF 8Hz + subsample to 20Hz**: Proper signal processing before encoding. Field standard temporal resolution.
4. **Causal pretraining, bidirectional fine-tuning**: Predictor uses causal mask for next-frame prediction, then bidirectional for CTC feature extraction.
5. **NCA rule families**: Five families from generic (Random MLP) to cortex-specific (FHN), plus Mixed. Rule family is a core ablation.
6. **Continuous NCA values**: All rules produce continuous scalars, z-normalized with noise to match HGA statistics.
7. **No threshold clamping for model input**: Subthreshold activity contains preparatory signal. Clamping is visualization-only.
8. **Cross-task data in Stage 2**: Include Lexical task patients (unsupervised, different phonemes irrelevant).
9. **Initial round = 4 methods**: A (NCA+Neural), B (Neural only), C (Supervised), D (Random init). Expand if A shows promise.
10. **Decoder is disposable**: ~17-34K params per grid size, discarded after pretraining. Its only role is to make the training objective collapse-proof.

## 13. Open Questions

1. **Multi-step prediction**: Predict frame_{t+1} only (1-step) vs frame_{t+1:t+k} (multi-step). Multi-step is harder but may learn richer dynamics.
2. **Predictor depth**: 2 layers may be insufficient. Could sweep 2-4 layers.
3. **FHN parameter ranges**: Need to map the "interesting" regime before generating at scale. Known phase diagrams exist.
4. **Multi-variable NCA observation**: FHN and Gray-Scott have 2 variables per cell. Should we expose both channels (H, W, 2) or only the observable (H, W, 1)?
5. **SIGReg value with reconstruction**: Since reconstruction prevents collapse, is SIGReg still useful? It may still help by encouraging interpretable latent geometry. Ablate lambda in {0, 0.01, 0.1}.
