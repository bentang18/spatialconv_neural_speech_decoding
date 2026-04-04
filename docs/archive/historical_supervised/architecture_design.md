# Neural Field Perceiver: Architecture Design

*Companion to `theoretical_framing.md`. Assumes familiarity with the problem decomposition, evidence hierarchy, and manifold formulation established there.*

---

## 0. Design Principles

Every architectural choice traces back to the observation function:

$$d_j^{(i)}(t) = g(\mathbf{p}_j^{(i)} + \Delta^{(i)},\ \mathbf{z}(t);\ e^{(i)}) + \epsilon_j^{(i)}(t)$$

| Symbol | Meaning | Architectural component |
|--------|---------|------------------------|
| $d_j^{(i)}(t)$ | Electrode observation | Input token |
| $\mathbf{p}_j^{(i)}$ | MNI position (approximate) | MNI positional encoding |
| $\Delta^{(i)}$ | Per-patient registration correction | Learnable 3D offset |
| $\mathbf{z}(t)$ | Shared articulatory latent state | Spatial encoder output |
| $e^{(i)}$ | Patient-specific residual | Learnable patient embedding |
| $g(\cdot)$ | Shared observation function (somatotopic map) | Observation decoder |
| $\epsilon$ | Noise | Regularization (dropout, augmentation) |

The architecture has four modules. Each solves one subproblem:

1. **Spatial Encoder** — Invert $g$: observations at known positions → latent state $\hat{\mathbf{z}}(t)$
2. **Observation Decoder** — Implement $g$: latent state + position → predicted observation (reprojection)
3. **Temporal Encoder** — Model dynamics: $\hat{\mathbf{z}}(1{:}T) \to$ temporally-coded representations
4. **Phoneme Decoder** — Classify trajectory: temporal representations → phoneme sequence $\hat{\mathbf{y}}$

---

## 1. Spatial Encoder

### Purpose

Per-frame, invert the observation function: given sparse electrode observations $\{d_j(t)\}$ at known positions $\{\mathbf{p}_j\}$, estimate the latent articulatory state $\hat{\mathbf{z}}(t)$.

### Input: Electrode tokens

For each electrode $j$ on patient $i$ at timeframe $t$:

```
activity_embed  = Linear(1 → d)(d_j(t))                           # what it measures
mni_pe          = Linear(3 → d)(p_j + Δ^(i))                      # where it is (cross-patient)
grid_pe         = Linear(2 → d)(row_j, col_j)                     # where it is (within-patient)
patient_embed   = e^(i)                                            # who it belongs to

token_j = activity_embed + mni_pe + grid_pe + patient_embed        # (d,)
```

Dead electrodes are excluded from the token set (not masked, just absent). The transformer sees only live electrodes.

**Per-patient parameters**:
- $\Delta^{(i)} \in \mathbb{R}^3$: learnable MNI offset, initialized to zero. Corrects registration error. Shared across all electrodes of patient $i$.
- $e^{(i)} \in \mathbb{R}^d$: learnable patient embedding, initialized from $\mathcal{N}(0, 0.02)$. Captures signal characteristics, pathology effects, fine-grained neural organization.
- Total: $3 + d$ parameters per patient.

### Perceiver cross-attention

**Virtual electrodes** (shared across all patients):

```
L = 16 learnable queries
  query_i = virtual_embed_i + MNI_PE(v_pos_i)

  virtual_embed_i ∈ R^d   — learnable content embedding
  v_pos_i ∈ R^3           — learnable MNI position, initialized to cover
                             the convex hull of source patient electrode positions
```

**Cross-attention with distance bias**:

```
Q = virtual electrodes                      (L, d)
K = Linear_K(electrode tokens)              (N_i, d)
V = Linear_V(electrode tokens)              (N_i, d)

Attention logits:
  A_{ij} = (Q_i · K_j) / √d  +  β_{ij}

Distance bias (spatial locality prior):
  β_{ij} = -α · ||v_pos_i - (p_j + Δ^(i))||²

  α > 0 is a learnable scalar (initialized to 0.1)
  Controls spatial blur: large α = sharp (precise MNI), small α = fuzzy (robust)

Output:
  out_i = Σ_j softmax(A)_{ij} · V_j         (L, d)
```

Multi-head attention with $H = 4$ heads, each with $d_h = d/H = 16$.

**Optional self-attention** (1 layer): Refine virtual electrode representations by attending to each other. Captures interactions between somatotopic zones (e.g., lip-tongue co-articulation).

**Readout**:
```
s(t) = MaxPool(out)    ∈ R^d       # single spatial summary per frame
```

MaxPool over the $L$ virtual electrode outputs (proven better than AvgPool in experiments).

### Dimensions

| Quantity | Value | Rationale |
|----------|-------|-----------|
| $d$ (embedding dim) | 64 | Matches current pipeline BiGRU output |
| $L$ (virtual electrodes) | 16 | ~1 per articulatory zone, matches manifold dim ~5-8 with redundancy |
| $H$ (attention heads) | 4 | Standard for d=64 |
| $N_i$ (electrode tokens) | 63-201 | Varies by patient, no padding needed |

### Parameter count

| Component | Params |
|-----------|--------|
| Linear(1→d) activity embedding | 128 |
| Linear(3→d) MNI PE | 256 |
| Linear(2→d) Grid PE | 192 |
| Virtual electrode embeddings (16 × d) | 1,024 |
| Virtual MNI positions (16 × 3) | 48 |
| Cross-attention QKV projections (4 heads) | 3 × d × d = 12,288 |
| Output projection | d × d = 4,096 |
| Self-attention (1 layer, same structure) | ~16,384 |
| LayerNorm × 2 | 256 |
| FFN (d → 4d → d) × 2 | 2 × (d × 4d + 4d × d) = 65,536 |
| Distance scale α | 1 |
| **Total spatial encoder** | **~100K** |
| Per-patient Δ (3) + e (64) | 67/patient |

---

## 2. Observation Decoder (Neural Field $g$)

### Purpose

Implement the forward observation model: given latent state $\hat{\mathbf{z}}(t)$ and an electrode position $\mathbf{p}_j$, predict what the electrode should observe. Used for the reprojection loss — not for phoneme classification.

### Bilinear form

```
φ_j = MLP_spatial(p_j + Δ^(i), e^(i))  ∈ R^k     # spatial tuning at position p_j
ψ   = MLP_latent(ŝ(t))                  ∈ R^k     # articulatory activation for state ẑ(t)

d̂_j(t) = φ_j ᵀ · ψ + bias                         # predicted observation (scalar)
```

Where:
- $\text{MLP}_{\text{spatial}}$: Linear(3+d → 128) → GELU → Linear(128 → k). Takes corrected MNI position concatenated with patient embedding.
- $\text{MLP}_{\text{latent}}$: Linear(d → 128) → GELU → Linear(128 → k). Takes the spatial encoder output $s(t)$.
- $k = 32$: dimensionality of the bilinear interaction space.

**Interpretation**: $\phi_j$ is the spatial tuning vector — it encodes what articulatory features brain position $\mathbf{p}_j$ is sensitive to. $\psi$ is the articulatory activation — it encodes which spatial tuning profiles are excited by the current state. Their inner product gives the predicted activity.

### Reprojection loss

```
L_reproj = (1/N_i) Σ_j ||d_j(t) - d̂_j(t)||²
```

Averaged over electrodes, frames, and patients in the batch. This loss:
- Trains the spatial encoder to produce accurate $\hat{\mathbf{z}}(t)$
- Trains the observation decoder to learn the somatotopic map $g$
- Optimizes $\Delta^{(i)}$ to correct MNI registration error
- Optimizes $e^{(i)}$ to capture patient-specific signal characteristics
- **Requires no phoneme labels** — purely geometric/reconstructive

### Electrode masking for regularization

During training, randomly drop 20-40% of electrodes from the encoder input but keep them in the reprojection loss. The model must predict masked electrodes' observations from unmasked context — spatial interpolation through the neural field.

This is the geometric analog of masked image modeling, but grounded in the observation function rather than arbitrary.

### Parameter count

| Component | Params |
|-----------|--------|
| MLP_spatial (3+d → 128 → k) | (67 × 128) + (128 × 32) = 12,672 |
| MLP_latent (d → 128 → k) | (64 × 128) + (128 × 32) = 12,288 |
| Bias | 1 |
| **Total observation decoder** | **~25K** |

---

## 3. Temporal Encoder

### Purpose

Model the dynamics $\dot{\mathbf{z}} = h(\mathbf{z}; \mathbf{y})$: transform the per-frame spatial summaries $s(1{:}T)$ into temporally-coded representations $z(t)$ that capture the articulatory trajectory.

### Multi-scale tokenization (experimentally validated)

```
Input: s(1:T) at 200Hz, s(t) ∈ R^d

Scale 1:  Conv1d(d, d, kernel=3,  stride=3)  + GELU → T₃ ≈ 67 tokens at ~67Hz
Scale 2:  Conv1d(d, d, kernel=5,  stride=5)  + GELU → T₅ ≈ 40 tokens at ~40Hz
Scale 3:  Conv1d(d, d, kernel=10, stride=10) + GELU → T₁₀ ≈ 20 tokens at ~20Hz

Each token gets:
  scale_pe  = learnable per-scale embedding (3 vectors of dim d)
  time_pe   = sinusoidal PE based on the token's center time in seconds

All tokens concatenated: (T₃ + T₅ + T₁₀, d) ≈ (127, d)
```

### Temporal transformer

```
2 layers, 4 heads, d=64, FFN dim=256
Standard pre-norm transformer (LayerNorm → Attention → residual → LayerNorm → FFN → residual)

Input: (127, d) multi-scale tokens
Output: (127, d) temporally-coded representations

Each token attends to all other tokens across all scales and times.
The 67Hz token at t=100ms can attend to the 20Hz token at t=800ms.
```

### Readout

```
For k-NN evaluation:
  z_global = mean_pool(all 127 tokens) ∈ R^d

For per-position classification:
  z_per_pos = mean_pool(tokens within each phoneme window) ∈ R^d per position
  (windows: 0-333ms, 333-666ms, 666-1000ms for 3-phoneme CVC)
```

### Parameter count

| Component | Params |
|-----------|--------|
| Conv1d × 3 branches | 3 × (d × d × k) ≈ 37K |
| Scale PE (3 × d) | 192 |
| Temporal transformer (2 layers) | ~33K |
| **Total temporal encoder** | **~70K** |

---

## 4. Phoneme Decoder

### Purpose

Classify the articulatory trajectory into phoneme sequence $\hat{\mathbf{y}} = (\hat{y}_1, \hat{y}_2, \hat{y}_3)$.

### Per-position CE head (proven)

```
z_pooled = mean_pool(temporal encoder output) ∈ R^d
z_dropped = Dropout(0.3)(z_pooled)

For each position p ∈ {1, 2, 3}:
  logits_p = Linear(d → 9)(z_dropped)           # 9 phoneme classes
```

### Articulatory bottleneck head (proven for LOPO)

```
For each position p ∈ {1, 2, 3}:
  art_proj_p = Linear(d → 15)(z_dropped)         # project to articulatory space
  art_proj_p = normalize(art_proj_p, dim=-1)      # unit norm
  logits_p   = (art_proj_p @ A^T) * τ             # cosine similarity × temperature

A ∈ R^{9×15}: fixed articulatory feature matrix (phonological features)
τ = exp(log_temp): learnable temperature scalar
```

### Dual head

```
logits_p = (ce_logits_p + art_logits_p) / 2
```

### Classification loss

```
For each position p:
  L_p = FocalCE(logits_p, y_p - 1, γ=2.0, label_smoothing=0.1)

L_class = (L_1 + L_2 + L_3) / 3

With mixup (α=0.2): sample λ ~ Beta(0.2, 0.2), mix inputs and average losses.
```

### Parameter count

| Component | Params |
|-----------|--------|
| CE heads (3 × Linear(64→9)) | 1,755 |
| Articulatory heads (3 × Linear(64→15)) | 2,925 |
| Temperature | 1 |
| **Total phoneme decoder** | **~5K** |

---

## 5. Complete Architecture Summary

```
╔══════════════════════════════════════════════════════════════════════╗
║                     NEURAL FIELD PERCEIVER                          ║
║                                                                     ║
║  Input: {(d_j(t), p_j^MNI, row_j, col_j)}  for j=1..N_i           ║
║                                                                     ║
║  ┌─────────────────────────────────────────────────────┐            ║
║  │ SPATIAL ENCODER (per frame)                 ~100K   │            ║
║  │                                                     │            ║
║  │ Electrode tokens:                                   │            ║
║  │   Linear(d_j→64) + MNI_PE(p_j+Δ) + Grid_PE + e^i  │            ║
║  │                                                     │            ║
║  │ Perceiver cross-attention (L=16 virtual electrodes) │            ║
║  │   distance bias: -α·||v_pos - p_j||²               │            ║
║  │ + Self-attention (1 layer)                          │            ║
║  │ → MaxPool → s(t) ∈ R^64                             │            ║
║  └──────────────┬──────────────────────────────────────┘            ║
║                 │                                                    ║
║       ┌─────────┴─────────┐                                         ║
║       │                   │                                         ║
║       ▼                   ▼                                         ║
║  ┌─────────────┐   ┌──────────────────────────────────┐            ║
║  │ OBSERVATION │   │ TEMPORAL ENCODER          ~70K   │            ║
║  │ DECODER     │   │                                  │            ║
║  │        ~25K │   │ Multi-scale Conv1d (s=3,5,10)    │            ║
║  │             │   │ + Temporal transformer (2 layers) │            ║
║  │ φ(p,e)ᵀψ(s)│   │ → z(t) ∈ R^64                    │            ║
║  │ = d̂_j(t)   │   └──────────────┬───────────────────┘            ║
║  │             │                  │                                  ║
║  │ L_reproj    │                  ▼                                  ║
║  │ (self-sup)  │   ┌──────────────────────────────────┐            ║
║  └─────────────┘   │ PHONEME DECODER            ~5K   │            ║
║                    │                                   │            ║
║                    │ Per-position CE + articulatory     │            ║
║                    │ Focal loss + label smooth + mixup  │            ║
║                    │                                   │            ║
║                    │ L_class (supervised)               │            ║
║                    └───────────────────────────────────┘            ║
║                                                                     ║
║  Total loss: L = L_class + λ · L_reproj                            ║
║  Total shared params: ~200K                                         ║
║  Per-patient params: 67 (3D offset + 64D embedding)                ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Total parameter budget

| Module | Params | Role |
|--------|--------|------|
| Spatial encoder | ~100K | Invert observation function |
| Observation decoder | ~25K | Forward observation model (reprojection) |
| Temporal encoder | ~70K | Dynamics modeling |
| Phoneme decoder | ~5K | Classification |
| **Total shared** | **~200K** | |
| Per-patient (× 10 patients) | 670 | Registration correction + patient identity |
| **Grand total** | **~201K** | |

Compare: current pipeline is ~121K shared + ~800 per-patient (80 × 10). The new architecture is ~1.7× larger in shared params, but this is still firmly in the small-model regime appropriate for our data scale.

---

## 6. Training Pipeline

### Phase 1: Joint training on source patients (supervised + reprojection)

**Data**: 9 source patients, ~1315 labeled trials.

```
For each training batch:
  1. Sample a mini-batch of trials from source patients
  2. For each trial (B, N_i, T):

     a. SPATIAL ENCODING (per frame):
        - Construct electrode tokens with dual PE + patient params
        - Mask 20-40% of electrodes (keep for reprojection target)
        - Perceiver cross-attention → s(t) per frame

     b. REPROJECTION (per frame):
        - Observation decoder: s(t) + electrode positions → d̂_j(t)
        - L_reproj = mean ||d_j(t) - d̂_j(t)||² over ALL electrodes (including masked)

     c. TEMPORAL ENCODING:
        - s(1:T) → multi-scale Conv1d → temporal transformer → z(t)

     d. PHONEME CLASSIFICATION:
        - z(t) → pool → dual head → logits
        - L_class = focal CE with label smoothing + mixup

     e. TOTAL LOSS:
        L = L_class + λ · L_reproj
        (λ = 0.5 initially, anneal to 0.1 over training)

Optimizer: AdamW
  - Shared params: lr = 1e-3
  - Virtual electrode positions: lr = 3e-4 (slow — positions should stabilize)
  - Per-patient Δ: lr = 3e-4 (slow — geometric correction)
  - Per-patient e: lr = 1e-3 (can adapt faster)

Schedule: Linear warmup (20 epochs) → cosine decay
Weight decay: 1e-4
Gradient clipping: 5.0
Batch size: 16 trials
Epochs: 200 (early stopping on held-out 20% source validation, patience 7)
```

**Optional extension**: Apply reprojection loss to unlabeled data (continuous recordings, Lexical patients) in addition to labeled trials. This only requires electrode activity + positions, no phoneme labels. Alternate batches: labeled (both losses) / unlabeled (reprojection only).

### Phase 2: Target patient adaptation (LP-FT on S14)

**Data**: S14, 153 trials, 5-fold grouped-by-token CV.

```
Per fold:

  STEP 1 — Linear Probe (freeze encoder, train decoder):
    - Initialize Δ^(S14) = 0, e^(S14) = mean(e^(source))
    - Freeze spatial encoder + temporal encoder
    - Train phoneme decoder + Δ^(S14) + e^(S14)
    - Epochs: 150, lr: 1e-3 (heads), 3e-4 (Δ, e)
    - L = L_class + 0.1 · L_reproj

  STEP 2 — Fine-tune (unfreeze all at low LR):
    - Unfreeze spatial + temporal encoder at 0.1× lr
    - Continue training all params
    - Epochs: 100, lr: 1e-4 (encoder), 1e-3 (heads), 3e-4 (Δ, e)
    - Patience: 7

  EVALUATION:
    - Linear head predictions (with TTA 16 if augmentation-trained)
    - Weighted k-NN (k=10, cosine) on mean-pooled temporal features
    - Best-of per fold: min(linear_per, knn_per)
```

### Phase 3 (optional): Reprojection pre-training on unlabeled data

If continuous recording data and Lexical patients are available:

```
Pre-train spatial encoder + observation decoder using L_reproj ONLY.
Data: continuous recordings from all patients (after VAD curation).
No phoneme labels needed.

Then proceed to Phase 1 (initialize from pre-trained spatial encoder).
```

This is the SSL component — but grounded in the reprojection objective rather than arbitrary masked prediction.

---

## 7. Evaluation Protocol

### Primary metric

PER (Phoneme Error Rate) via edit distance, grouped-by-token 5-fold CV.

### Evaluation recipe (all experimentally validated components)

| Component | Setting | Source |
|-----------|---------|--------|
| Weighted k-NN | k=10, cosine similarity weights | Autoresearch exp9 |
| TTA | 16 augmented copies (if augmentation-trained) | Autoresearch exp16 |
| Dual head | CE + articulatory, averaged logits | Autoresearch exp14-15 |
| Best-of per fold | min(linear, kNN) per fold | Autoresearch exp9 |
| Source k-NN | Source patient embeddings at 0.5× weight | LOPO exp13 |

### Ablation experiments (ordered by priority)

**A1: Does MNI help?** (the central question)
```
  a. Current baseline: Per-patient Conv2d + BiGRU         → PER baseline (0.762)
  b. Perceiver + Grid PE only (no MNI)                    → architecture benefit
  c. Perceiver + MNI PE only (no Grid)                    → MNI benefit
  d. Perceiver + MNI PE + Grid PE                         → combined benefit
  e. (d) + reprojection loss                              → reprojection benefit

  Compare b vs d: MNI effect (grid-controlled)
  Compare c vs d: Grid effect (MNI-controlled)
  Compare d vs e: Reprojection loss effect
```

**A2: Population-level evaluation**
```
  Run best configuration on ALL patients (not just S14).
  Required to determine if 0.76 wall is S14-specific or universal.
```

**A3: Per-patient parameter ablation**
```
  a. No per-patient params (Δ=0, e=0, frozen)
  b. Δ only (3 params)
  c. e only (64 params)
  d. Δ + e (67 params)
  e. Δ + e + per-patient gain/bias (69 params)
```

**A4: Virtual electrode count**
```
  L ∈ {4, 8, 16, 32}
  Expect: too few (L=4) underfits, too many (L=32) provides no benefit
```

**A5: Reprojection loss weight**
```
  λ ∈ {0, 0.01, 0.1, 0.5, 1.0}
  λ=0: no reprojection (pure supervised)
  Determines whether reprojection helps or hurts
```

**A6: Cross-task pooling** (independent of architecture)
```
  Add Duraivel pseudoword patients to source training.
  Can be tested with current pipeline AND new architecture.
  Lowest-risk highest-confidence experiment.
```

---

## 8. MNI Coordinate Sourcing

### Dependency

The architecture requires MNI coordinates for each electrode. Current pipeline uses ACPC-normalized grid positions (0-1 range in TSVs) for grid topology only.

### What we need

Per patient: a mapping from electrode name → (x, y, z) in MNI-152 mm. For 12 PS patients + 13 Lexical patients = 25 patients total.

### Where they likely are

1. **BioImage Suite outputs** (DBS/Parkinson's patients): electrode reconstruction produces MNI coordinates directly.
2. **Brainlab outputs** (tumor patients): electrode localization in patient space, then registration to MNI.
3. **ECoG_Recon data on Box** (referenced in project notes): may contain reconstructed electrode positions in MNI.
4. **BIDS coordsystem JSON files**: Currently say "space: ACPC" — but the actual coordinates in the TSVs are normalized grid positions, not brain coordinates. The true MNI coordinates may be in a different file or format.

### Action items

1. Check ECoG_Recon data on Box for MNI coordinate files
2. Ask Zac Spalding where the MNI-152 coordinates are stored
3. Verify coordinate system and units (mm vs m — the coordsystem JSON notes units may be wrong)
4. For each patient: validate that coordinates are in MNI-152 and make anatomical sense (electrodes should cluster over left perisylvian cortex)

### Fallback if MNI coordinates are unavailable

If true MNI coordinates cannot be sourced, two fallback options:

1. **Use ACPC grid positions as approximate coordinates**: These are normalized (0-1) but could be scaled to approximate mm. This loses cross-patient alignment but retains within-patient spatial structure. The model would use Grid PE only with these scaled positions.

2. **Estimate MNI from grid positions + array metadata**: If we know the array center in MNI (from surgical reports) and the array geometry (grid spacing × grid shape), we can reconstruct approximate MNI coordinates per electrode. Accuracy depends on knowing the array orientation.

---

## 9. Implementation Plan

### Prerequisites

- [ ] Source MNI coordinates for all patients (Section 8)
- [ ] Validate coordinate quality (visualization, anatomical plausibility)
- [ ] Verify continuous recording data availability in .fif files

### Phase 1: Core architecture (supervised LOPO, ~2 weeks)

1. **Spatial encoder** (`src/speech_decoding/models/neural_field_perceiver.py`)
   - Electrode token construction (dual PE + per-patient params)
   - Perceiver cross-attention with distance bias
   - Virtual electrodes with learnable MNI positions
   - Unit tests: variable electrode count, different grid shapes, dead electrodes

2. **Observation decoder** (`src/speech_decoding/models/observation_decoder.py`)
   - Bilinear neural field (MLP_spatial × MLP_latent)
   - Reprojection loss computation
   - Electrode masking during training
   - Unit tests: reconstruction on synthetic data, gradient flow

3. **Temporal encoder** (`src/speech_decoding/models/temporal_transformer.py`)
   - Multi-scale Conv1d tokenization
   - Temporal transformer with scale PE
   - Unit tests: multi-scale token shapes, attention mask

4. **Integration** — Assemble full pipeline, train on source patients, evaluate on S14
   - LOPO training loop (Phase 1 + Phase 2 from Section 6)
   - Grouped-by-token CV evaluation with full recipe
   - Compare against current baseline (PER 0.762)

### Phase 2: Ablations (~1 week)

5. Run ablation experiments A1-A5 from Section 7
6. Run population-level evaluation (A2) on all patients
7. Run cross-task pooling experiment (A6, independent of architecture)

### Phase 3: SSL extension (optional, ~1 week)

8. Continuous recording extraction and VAD curation
9. Reprojection pre-training on unlabeled data (Phase 3 from Section 6)
10. Evaluate whether SSL pre-training improves over supervised-only

---

## 10. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| MNI coordinates unavailable | High | Fallback to scaled ACPC + Grid PE only (Section 8) |
| MNI alignment too imprecise for benefit | High | Ablation A1 detects this; learnable α adapts spatial blur |
| Reprojection loss dominates / hurts classification | Medium | λ ablation A5; can set λ=0 (pure supervised) |
| Perceiver overfits with ~200K params on ~1300 trials | Medium | Reprojection regularizes; electrode masking; early stopping |
| Virtual electrodes collapse to same position | Low | Initialize spread across electrode convex hull; monitor positions |
| Transformer temporal encoder overfits | Medium | Only 2 layers, d=64; proven multi-scale Conv1d is the heavy lifter |
| Per-patient Δ overfits (only ~150 trials for S14) | Low | Regularize Δ toward 0 (L2 penalty); low learning rate |

---

## 11. Success Criteria

### Minimum success (validates approach)

- Ablation A1d (Perceiver + MNI + Grid) outperforms A1b (Perceiver + Grid only) by ≥ 2pp on S14
- This directly answers: "Do MNI coordinates help?"

### Target success (breaks the wall)

- PER < 0.740 on S14 (better than per-patient supervised 0.737)
- This would mean cross-patient transfer with MNI alignment exceeds single-patient training

### Strong success (population-level)

- Mean PER across all patients improves over current LOPO baseline
- Consistent improvement, not just S14-specific

### Interpretability bonus

- Virtual electrode positions cluster over known speech motor regions (vSMC)
- Learned Δ^(i) correlate with known registration quality (larger for tumor patients)
- Spatial tuning vectors φ(p) recover articulatory somatotopy when visualized on cortical surface
