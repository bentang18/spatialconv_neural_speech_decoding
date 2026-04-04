# 3D Vision ↔ Neural Field Perceiver: Reference Survey

*Standalone reference. Maps 2024-2026 advances in 3D Gaussian splatting, NeRF, and multi-view reconstruction to our cross-patient cortical field decoding problem.*

*Last updated: April 2026.*

---

## 1. The Structural Analogy

Our problem — reconstructing a neural field on the cortical surface from sparse electrode observations at known brain positions — maps to multi-view 3D scene reconstruction:

| 3D Vision | Cortical Field Decoding |
|-----------|----------------------|
| Multiple cameras at known poses | Multiple patients with electrodes at known MNI positions |
| Sparse views of a 3D scene | Sparse samples of a cortical activity field |
| Per-scene optimization (classic NeRF/3DGS) | Per-patient spatial encoder (current Conv2d) |
| Feed-forward generalizable model | Cross-patient shared model (our goal) |
| Novel view synthesis | Novel patient prediction |
| Camera extrinsics [R\|t] | Per-patient MNI offset Δ^(i) |
| Camera intrinsics K | Per-patient gain/bias |
| Reprojection loss ‖π(V) - x‖² | Reprojection loss ‖g(p+Δ, ẑ) - d‖² |

---

## 2. Key Papers and Transferable Ideas

### 2.1. Feed-Forward Generalizable Reconstruction

The field's central trajectory 2020-2026: **per-scene optimization → feed-forward generalization → foundation models.** Our architecture sits at step 2.

**VGGT: Visual Geometry Grounded Transformer**
- Wang, Chen et al. CVPR 2025 Best Paper. [arXiv:2503.11651](https://arxiv.org/abs/2503.11651)
- Single transformer predicts ALL 3D attributes (cameras, pointmaps, depth, tracks) from 1-100s of views in <1 second.
- **Transfer**: Joint prediction of multiple geometric quantities from variable-length observation sets. Validates that one backbone can handle both geometry estimation and feature extraction.
- **Our analog**: Perceiver cross-attention jointly estimates latent state + spatial alignment from variable electrode counts.

**pixelSplat / MVSplat**
- pixelSplat: Charatan et al., CVPR 2024 Oral Best Paper Runner-Up.
- MVSplat: Chen et al., ECCV 2024. [arXiv:2403.14627](https://arxiv.org/abs/2403.14627)
- Feed-forward 3DGS from image pairs. MVSplat uses cost volumes for geometry (10× fewer params, 2× faster than pixelSplat). Cross-dataset generalization.
- **Transfer**: The pattern of encode-each-view → aggregate-across-views → decode-representation. MVSplat's cost volumes are the analog of our MNI distance-biased attention.

**LVSM: Large View Synthesis Model**
- Jin, Jiang et al. ICLR 2025 Oral. [arXiv:2410.17242](https://arxiv.org/abs/2410.17242)
- Two architectures: encoder-decoder (compresses inputs into **fixed latent tokens**) vs decoder-only (direct mapping).
- Decoder-only wins by 1.5-3.5 dB with enough data. Key finding: explicit 3D structure is unnecessary at sufficient data scale.
- **Transfer**: Our virtual electrodes ARE the fixed latent tokens. LVSM validates this bottleneck design. The decoder-only result suggests our structured approach (explicit geometry) is correct for our small-data regime — when data is limited, geometric inductive biases help.

**LRM: Large Reconstruction Model**
- Hong, Zhang et al. ICLR 2024. [arXiv:2311.04400](https://arxiv.org/abs/2311.04400)
- 500M-param transformer: single image → triplane NeRF via cross-attention. Trained on ~1M objects.
- **Transfer**: Cross-attention from canonical representation (triplane) to observation tokens. Our virtual electrodes serve the same role as the triplane — a fixed canonical sampling of the field.

**AnySplat**
- Jiang et al. SIGGRAPH Asia 2025. [arXiv:2505.23716](https://arxiv.org/abs/2505.23716)
- Feed-forward from unconstrained, uncalibrated views. Predicts 3DGS + camera intrinsics/extrinsics simultaneously. Geometry transformer + differentiable voxelization.
- **Transfer**: Handles arbitrary view count AND unknown camera poses. Analogous to handling variable electrode count with approximate (noisy) MNI positions.

### 2.2. Sparse-View Reconstruction

Our setting — 63-201 electrodes sampling a cortical field — is analogous to few-view 3D reconstruction.

**SPARF: Neural Radiance Fields from Sparse and Noisy Poses**
- Truong et al. CVPR 2023.
- Multi-view geometric constraints for NeRF from wide-baseline images with **noisy camera poses**.
- **Transfer**: Directly relevant — handles both sparse observations AND noisy position information. Our MNI coordinates are noisy poses; our electrodes are sparse views.

**Sparse-view 3DGS strategies (survey: AI Review, Springer 2025)**
- Three dominant strategies for sparse views: (1) monocular depth priors, (2) diffusion priors to hallucinate unseen views, (3) cost volumes for multi-view matching.
- **Transfer**: Strategy (1) maps to our Grid PE (local structural prior). Strategy (3) maps to MNI distance-biased attention. Strategy (2) requires massive training data and doesn't directly apply.

**GS-GS: Generative Sparse-View Gaussian Splatting**
- Kong et al. CVPR 2025.
- Reconstructs from only 3 views by using LoRA-adapted diffusion models to hallucinate pseudo-views.
- **Transfer (cautious)**: The concept of using a learned prior to fill in unobserved positions is appealing — our reprojection loss with electrode masking is a weaker version. But diffusion priors require millions of training examples; we have ~20 patients.

### 2.3. Dynamic / 4D Representations

Our cortical field evolves over time — this connects to dynamic scene reconstruction.

**4D Gaussian Splatting: Native 4D Primitives**
- Yang et al. ICLR 2024 (extended 2025). [arXiv:2412.20720](https://arxiv.org/abs/2412.20720)
- Native 4D spatiotemporal Gaussians instead of deforming 3D Gaussians over time.
- **Transfer (limited)**: Our cortical field is spatiotemporal, but each trial is independent (not a persistent evolving scene). The spatial-temporal factorization idea transfers; the specific 4DGS representation does not.

**Hybrid 3D-4D Gaussian Splatting**
- Adaptively uses 3D for static regions and 4D for dynamic. Iterative conversion.
- **Transfer**: Some cortical regions may be relatively stable (background activity) while others change with phoneme production. The hybrid idea of adaptive temporal complexity is interesting but may be overengineered for our scale.

### 2.4. Continuous Field Reconstruction from Sparse Sensors

**MMGN: Continuous Field Reconstruction from Sparse Observations with Implicit Neural Networks**
- Luo, Xu, Nadiga, Ren, Yoo. ICLR 2024. [arXiv:2401.11611](https://arxiv.org/abs/2401.11611)
- Reconstructs continuous physical fields (sea surface temperature, climate) from **sparse, irregularly sampled sensor observations** using implicit neural representations.
- **Factorizes spatiotemporal variability** into spatial and temporal components via separation of variables. Learns basis functions from the data.
- **THE most directly relevant paper.** Our problem is almost identical: sparse sensors (electrodes) at known positions (MNI), continuous field (cortical activity), spatiotemporal factorization (our bilinear decoder d̂ = φ(p)ᵀψ(z)).

**FLRONet: Deep Operator Learning for Fluid Flow Field Reconstruction**
- Deep operator network for field reconstruction from sparse sensor measurements. Branch net uses FNO to combine sensor observations across time.
- **Transfer**: Operator learning (learn the mapping from sensor observations → full field) is what our encoder does. The FNO branch net is analogous to our multi-scale temporal Conv1d.

**DeepONet with NDFT (2025)**
- Modified branch network handles **irregular input grids** via Non-uniform Discrete Fourier Transform.
- **Transfer**: Handling irregular sensor positions is exactly our challenge. NDFT could be an alternative to our MNI PE for encoding irregular electrode positions.

### 2.5. Graph-Based and Geometry-Aware Aggregation

**GraphSplat**
- ACM MM 2025.
- Multi-view Aggregate Graph Attention (MAGA): edge-weighted graph attention where edges are determined by spatial proximity between views. Adaptively reweights intra-view vs inter-view connections.
- **Transfer**: Electrodes as graph nodes, edges weighted by MNI distance. Graph attention is an alternative to our Perceiver cross-attention with distance bias. Same principle: use known geometry to structure the aggregation.

**FreeSplat / FreeSplat++**
- NeurIPS 2024/2025.
- Handles variable number of input views with low-cost cross-view aggregation. Pixel-wise triplet fusion eliminates redundant Gaussians.
- **Transfer**: Variable input handling without padding or masking. Our Perceiver naturally handles variable electrode counts via attention, achieving the same goal.

---

## 3. Are We Following Best Practices?

### What we're doing right (aligned with field)

| Best practice (3D vision 2025-2026) | Our implementation | Status |
|------|------|------|
| **Feed-forward generalizable model** (not per-scene optimization) | Shared Perceiver across patients (not per-patient Conv2d) | ✅ Core design |
| **Cross-attention from canonical to observation tokens** (LRM, LVSM, Perceiver) | Virtual electrodes cross-attend to electrode tokens | ✅ Core design |
| **Known geometry as attention bias** (GraphSplat, MVSplat cost volumes) | MNI distance bias: β = -α‖v - p‖² | ✅ Core design |
| **Joint optimization of geometry + representation** (SPARF, AnySplat) | Joint optimization of Δ^(i) + field model via reprojection loss | ✅ Core design |
| **Spatiotemporal factorization** (MMGN, 4DGS) | Bilinear decoder: d̂ = φ(p)ᵀψ(z) separating space from time | ✅ Core design |
| **Reprojection / reconstruction loss** (standard in all NeRF/3DGS) | L_reproj = Σ‖d - ĝ(p+Δ, ẑ)‖² | ✅ Core design |
| **Fixed-size latent bottleneck** (LVSM latent tokens, LRM triplane) | L=16 virtual electrodes as canonical cortical sampling | ✅ Core design |
| **Masking for self-supervised pretraining** (MAE, mask-based NeRF) | Electrode masking in reprojection loss | ✅ Planned |
| **Variable input handling without padding** (FreeSplat, VGGT, AnySplat) | Attention-based aggregation naturally handles 63-201 electrodes | ✅ Core design |
| **Dual local + global spatial encoding** (monocular depth + multi-view matching) | Grid PE (local) + MNI PE (global) | ✅ Core design |
| **Noisy pose correction** (SPARF, RePose-NeRF) | Learnable per-patient Δ^(i) for MNI offset correction | ✅ Core design |
| **Structured inductive bias for data-poor regimes** (LVSM finding: explicit geometry helps when data is limited) | Full geometric pipeline (Perceiver + bilinear decoder + reprojection) for ~1300 trials | ✅ By design |

### What we could consider adding

| Technique | Vision papers | Applicability | Priority |
|-----------|-------------|---------------|----------|
| **Cost-volume for cross-patient electrode matching** | MVSplat, TranSplat | Build explicit feature matching between electrodes at similar MNI positions across patients. Could replace or augment distance-biased attention. | Medium — adds complexity but MVSplat showed it's more robust than regression-based geometry. |
| **Multi-task prediction** | VGGT (joint camera + pointmap + depth) | Jointly predict cortical field + electrode significance + patient metadata from the same backbone. Auxiliary tasks improve the primary representation. | Low — limited auxiliary supervision available. |
| **Graph attention aggregation** | GraphSplat MAGA | Replace or augment Perceiver cross-attention with graph attention, where edges are MNI-distance-weighted. Could handle local vs global information more explicitly. | Low — Perceiver with distance bias achieves similar effect. Consider if attention patterns look wrong. |
| **Depth/confidence-aware attention** | TranSplat (depth confidence maps) | Weight electrode contributions by signal quality (SNR, impedance). Electrodes with poor contact should contribute less. | Medium — we already exclude dead electrodes, but graded confidence weighting could help. |
| **Iterative refinement** | SPARF (iterative pose + NeRF), Bundle-adjusting NeRF | Multiple encode-decode passes, refining Δ^(i) and ẑ(t) iteratively. Could improve registration correction. | Low — adds training complexity; single forward pass may suffice at our scale. |

### What we're correctly NOT doing

| Technique | Why popular in vision | Why wrong for us |
|-----------|----------------------|-----------------|
| **Diffusion priors for hallucination** (GS-GS, RI3D) | Fill in sparse views with generated pseudo-observations | Requires millions of training scenes; we have ~20 patients. The learned prior would overfit. |
| **4D Gaussian primitives** (4DGS) | Native 4D spatiotemporal representation | Our trials are independent events, not a persistent evolving scene. The temporal structure is phoneme-specific, not scene-persistent. |
| **Foundation model scale** (VGGT 500M+, LRM 500M) | Learn universal 3D priors from massive data | ~1300 trials cannot support >200K params without overfitting. Our 200K budget is appropriate. |
| **Decoder-only architecture** (LVSM) | Drop explicit 3D structure, let transformer learn geometry | Only works with massive data. LVSM's own ablation shows encoder-decoder (explicit structure) is better at small scale. Our geometric priors are load-bearing. |
| **Self-supervised contrastive pretraining** (in 3D learning) | Learn representations from view consistency | Failed in our experiments (VICReg, BYOL near chance). Too little data for contrastive learning to work. |

### Uncertainty estimation (critical gap — now addressed)

Modern 3D reconstruction explicitly models uncertainty in sparse regions. Our original architecture had no uncertainty mechanism. This section details the methods and our integration plan.

See **Section 6: Uncertainty Estimation** below for the full treatment.

---

## 4. Architectural Pattern Validation

Our architecture independently arrives at the same patterns that dominate 2025-2026 3D vision. This provides external validation:

| Our component | Same pattern in vision | Papers |
|---------------|----------------------|--------|
| Perceiver cross-attention from virtual → actual tokens | Cross-attention from canonical → observation tokens | LRM, LVSM encoder, Laser-NV |
| Virtual electrodes at learnable positions | Triplane NeRF / fixed latent tokens | LRM triplane, LVSM latent array |
| MNI distance-biased attention | Geometry-aware attention / cost volumes | GraphSplat MAGA, MVSplat cost volumes |
| Bilinear observation decoder d̂ = φ(p)ᵀψ(z) | Spatiotemporal field factorization | MMGN (ICLR 2024), NeRF position→features decomposition |
| Reprojection loss for spatial alignment | Photometric / reprojection loss | ALL NeRF/3DGS methods |
| Learnable per-patient Δ (pose correction) | Learnable camera extrinsics | SPARF, Bundle-adjusting NeRF, AnySplat |
| Patient embedding e^(i) | Per-scene latent code | Auto-decoder NeRF, CodeNeRF |
| Electrode masking in reprojection | View dropout / masked pretraining | MAE, masked NeRF methods |
| Multi-scale temporal tokenization | Multi-resolution feature processing | MViT, multi-scale cost volumes |
| Dual spatial encoding (MNI + grid) | Multi-level feature aggregation (global + local) | MVSplat (cost vol + monocular depth) |

**Every major component has a validated analog in the 3D vision literature.** We are not inventing novel architectural patterns — we are transferring proven patterns to a new domain with appropriate adaptations for our data regime.

---

## 5. The Data Scale Gap (Honest Assessment)

The biggest disconnect between 3D vision and our problem:

| | 3D Vision | Our Problem |
|---|---|---|
| Training scenes/patients | 50K-1M+ | ~20 patients |
| Observations per scene | 10-100+ images (millions of pixels) | 63-201 electrodes (sparse) |
| Total training data | ~TB scale | ~240 MB |
| Parameter budget | 50M-500M+ | ~200K |

This gap means:
- **We cannot use data-hungry techniques** (diffusion priors, decoder-only transformers, foundation models)
- **We MUST use strong geometric inductive biases** (MNI coordinates, bilinear factorization, reprojection loss) to compensate for limited data
- **Our architecture is correctly small** (~200K params) — the vision analogs at our data scale would also use <1M params
- **The reprojection loss is especially important** because it provides per-frame spatial supervision without phoneme labels, effectively multiplying our training signal

The vision field's finding that **explicit geometric structure helps at small scale** (LVSM ablation, SPARF vs unconstrained optimization) directly validates our design choice to invest in structured architecture rather than scale.

---

## 6. Uncertainty Estimation

### 6.1. The connection to existing preprocessing

Our pipeline already has a **binary** uncertainty mechanism: the significant channel selection (permutation cluster test) that keeps or drops entire electrodes. This is equivalent to a hard 0/1 confidence weight per electrode.

Proper uncertainty estimation replaces this with a **continuous** mechanism:

| Current (binary) | Proposed (continuous) |
|---|---|
| Electrode passes significance test → weight 1 | Each electrode gets a learned confidence ∈ (0, 1) |
| Electrode fails → weight 0 (excluded) | Low-quality electrodes downweighted, not dropped |
| Same weight for all significant channels | Edge electrodes, high-impedance contacts weighted less |
| No uncertainty propagated downstream | Per-position phoneme predictions carry confidence scores |

### 6.2. Two types of uncertainty we need

**Aleatoric uncertainty** (irreducible, data-dependent): "This electrode is inherently noisy because of high impedance / poor contact / sub-electrode-scale variability." Per-electrode, does not decrease with more training data.

**Epistemic uncertainty** (reducible, model-dependent): "We haven't seen enough data in this cortical region to make a confident prediction." Per-cortical-position, decreases with more patients/data.

Both matter:
- Aleatoric → downweight noisy electrodes in spatial aggregation
- Epistemic → flag predictions in poorly-covered cortical regions as unreliable

### 6.3. Methods ranked by applicability

#### Tier 1: Minimal implementation, high value

**Beta-NLL loss** (Seitzer et al., ICLR 2022, [arXiv:2203.09168](https://arxiv.org/abs/2203.09168))

Replace the MSE reprojection loss with learned per-electrode variance:

```
# Current: MSE reprojection
loss = (d_observed - d_predicted)**2

# With Beta-NLL (3 lines of code):
mean, log_var = decoder(s_t, p_j)    # predict both mean and log-variance
nll = 0.5 * (log_var + (d_observed - mean)**2 / log_var.exp())
loss = nll * log_var.exp().detach()**beta  # beta=0.5 prevents variance explosion
```

The model learns that electrodes with high impedance, poor contact, or positions far from articulatory-relevant cortex have high aleatoric variance. The `detach()**beta` trick prevents the mean from hiding behind inflated variance.

- **Cost**: Zero extra inference cost. 3 lines of code change.
- **Gives**: Per-electrode aleatoric uncertainty at every timeframe.
- **Integration**: Directly replaces `L_reproj = MSE(...)` with `L_reproj = BetaNLL(...)`.

**MC Dropout at inference** (Gal & Ghahramani 2016; stable output variant, 2025)

Our architecture already has Dropout(0.3). At inference, keep dropout active and run 10-20 forward passes. Variance across passes = epistemic uncertainty.

2025 best practice: remove dropout from the final output layer (keeps predictions sharp while still capturing uncertainty in features).

- **Cost**: 10-20× inference time (but model is small — 200K params).
- **Gives**: Per-position epistemic uncertainty.
- **Integration**: Already have dropout. Just add inference loop + variance computation.

**Conformal prediction wrapper** (Angelopoulos & Bates 2021; CONFIDE for transformers, 2024)

Post-hoc, distribution-free method: on a held-out calibration set, compute nonconformity scores. At test time, produce prediction SETS with guaranteed coverage (e.g., "the true phoneme is in this set with 90% probability").

- **Cost**: Zero additional training. One forward pass on calibration set.
- **Gives**: Per-position phoneme prediction sets with statistical coverage guarantees.
- **Integration**: Completely post-hoc. Wraps the existing phoneme decoder. Use the held-out CV fold as calibration.
- **Clinical value**: "The model thinks position 2 is /b/ or /p/ (both labial stops)" — interpretable uncertainty.

#### Tier 2: Moderate implementation, strong theoretical grounding

**Evidential regression (NIG) for reprojection** (Amini et al. NeurIPS 2018; Flexible EDL, NeurIPS 2025)

Instead of predicting a single field value, predict Normal-Inverse-Gamma parameters (μ, ν, α, β) per electrode. This decomposes uncertainty into aleatoric and epistemic in a **single forward pass**:

```
Aleatoric uncertainty = β / (α - 1)          # data noise
Epistemic uncertainty = β / (ν * (α - 1))    # model uncertainty
```

- **Cost**: Single forward pass. 4× output dim for reprojection decoder.
- **Gives**: Decomposed aleatoric + epistemic per electrode per frame.
- **Integration**: Replace bilinear output d̂ = φᵀψ (scalar) with d̂ = (μ, ν, α, β). Train with evidential loss.

**Faithful heteroscedastic regression** (Stirn & Knowles, AISTATS 2023)

Add a variance head to the reprojection decoder, but **stop gradients** from the variance head to the shared backbone. This prevents the backbone from learning to produce "uncertain" features as an excuse for poor reconstruction.

```
# Forward pass
features = perceiver_encoder(electrode_tokens)
mean = bilinear_decoder(features, positions)           # gradients flow normally
log_var = variance_head(features.detach(), positions)  # stop-grad: features frozen for this head
```

- **Cost**: ~1.1× (small MLP for variance head).
- **Gives**: Per-electrode aleatoric uncertainty without corrupting the main representation.
- **Integration**: Add small MLP parallel to bilinear decoder. Train with Gaussian NLL.

**pixelSplat-style probabilistic latent** (Charatan et al., CVPR 2024)

Instead of deterministic virtual electrode outputs, predict a **distribution** over latent values. Sample via reparameterization trick. Virtual electrodes far from any real electrode produce broad distributions (high uncertainty); those near dense observations produce sharp ones.

```
# Current: deterministic
s(t) = MaxPool(cross_attention_output)     # point estimate

# Probabilistic:
mu, log_var = split(cross_attention_output)  # predict distribution
s(t) = mu + exp(0.5 * log_var) * epsilon     # reparameterized sample
# KL regularization to prevent posterior collapse
```

- **Cost**: ~1.1× (distribution parameters + sampling).
- **Gives**: Per-virtual-electrode uncertainty in the latent space. Propagates through temporal encoder to phoneme predictions.
- **Integration**: Changes Perceiver output from (L, d) to (L, 2d) — mean and log-variance.

#### Tier 3: Higher implementation cost, principled but complex

**Bayesian attention** (Residual Bayesian Attention Networks, 2025)

Replace deterministic attention weights with distributions. Each attention logit QK^T/√d has a learned mean and variance. Uncertainty propagates through softmax → value aggregation.

- **Cost**: ~2× per attention layer.
- **Gives**: Per-token uncertainty directly from the attention mechanism.
- **Integration**: Significant architecture change. Consider for v2 only.

**SWAG** (Stochastic Weight Averaging Gaussian, NeurIPS 2019; ASWA 2024)

Maintain running mean and low-rank covariance of SGD iterates during cosine annealing. At inference, sample weights and average predictions.

- **Cost**: ~1.2× training, 10-30× inference.
- **Gives**: Full epistemic uncertainty with weight-space coverage.
- **Integration**: Compatible with our cosine LR schedule. ASWA only updates running average when validation improves.
- **Note**: No BCI paper has used SWA/SWAG — this would be novel for the domain.

**UQ-SONet (cVAE latent)** (Paul et al., TMLR 2025)

Set transformer over sensors + conditional VAE latent. The VAE latent captures the distribution of cortical field configurations consistent with the sparse observations.

- **Cost**: ~2-3× (VAE encoder + decoder + KL).
- **Gives**: Full generative uncertainty — sample multiple plausible cortical fields.
- **Integration**: Major architecture change. Adds VAE between spatial encoder and temporal encoder.

### 6.4. Recommended integration for Neural Field Perceiver

**Core stack (implement in v1):**

```
1. REPROJECTION: Beta-NLL loss (replaces MSE)
   → Per-electrode aleatoric uncertainty
   → 3 lines of code
   → Model learns which electrodes are inherently noisy

2. CLASSIFICATION: Conformal prediction wrapper (post-hoc)
   → Per-position phoneme prediction sets with coverage guarantees
   → Zero training cost, use CV fold as calibration
   → Clinical interpretability: "model thinks /b/ or /p/, both labial"

3. INFERENCE: MC Dropout (10 passes, dropout off in output layer)
   → Per-position epistemic uncertainty
   → Already have dropout layers
   → High epistemic = poorly-covered cortical region
```

**Enhanced stack (v2):**

```
4. REPROJECTION: Evidential (NIG) head
   → Decomposed aleatoric + epistemic per electrode
   → Single forward pass (replaces MC Dropout)

5. LATENT: Probabilistic virtual electrodes (pixelSplat-style)
   → Uncertainty propagates through full pipeline
   → Virtual electrodes far from real ones have broad distributions
```

### 6.5. How uncertainty flows through the architecture

```
Electrode observations d_j(t)
  ↓
Per-electrode ALEATORIC uncertainty (Beta-NLL / NIG)
  "This electrode is noisy"
  ↓
Perceiver cross-attention (uncertainty-weighted)
  Electrodes with high aleatoric uncertainty contribute less
  ↓
Virtual electrode latent s(t) [± variance if probabilistic]
  Per-position EPISTEMIC uncertainty (MC Dropout / probabilistic sampling)
  "This cortical region is poorly observed"
  ↓
Temporal encoder → z(t)
  ↓
Phoneme decoder → logits
  Per-phoneme PREDICTIVE uncertainty (conformal prediction)
  "This phoneme is ambiguous — could be /b/ or /p/"
```

Each level captures a different uncertainty source:
- **Electrode level**: signal quality (impedance, contact, noise) → aleatoric
- **Field level**: coverage gaps (no electrodes in this cortical region) → epistemic
- **Prediction level**: phoneme ambiguity (articulatory similarity) → predictive


## 7. Key References

### Most relevant to our architecture

1. **MMGN** — Luo et al. ICLR 2024. Continuous field reconstruction from sparse irregular observations. [arXiv:2401.11611](https://arxiv.org/abs/2401.11611)
2. **VGGT** — Wang et al. CVPR 2025 Best Paper. Joint 3D attribute prediction from variable views. [arXiv:2503.11651](https://arxiv.org/abs/2503.11651)
3. **LVSM** — Jin et al. ICLR 2025 Oral. Fixed latent tokens as canonical scene representation. [arXiv:2410.17242](https://arxiv.org/abs/2410.17242)
4. **pixelSplat** — Charatan et al. CVPR 2024 Oral. Feed-forward 3DGS via cross-attention.
5. **MVSplat** — Chen et al. ECCV 2024. Cost-volume geometry for generalizable 3DGS. [arXiv:2403.14627](https://arxiv.org/abs/2403.14627)
6. **SPARF** — Truong et al. CVPR 2023. NeRF from sparse noisy poses.
7. **GraphSplat** — ACM MM 2025. Graph attention with spatial edge weighting.
8. **AnySplat** — Jiang et al. SIGGRAPH Asia 2025. Uncalibrated variable-view feed-forward 3DGS. [arXiv:2505.23716](https://arxiv.org/abs/2505.23716)
9. **LRM** — Hong et al. ICLR 2024. Cross-attention to triplane canonical representation. [arXiv:2311.04400](https://arxiv.org/abs/2311.04400)

### Relevant surveys

10. **A review on 3D Gaussian splatting for sparse view reconstruction** — AI Review, Springer 2025.
11. **Sparse-View 3D Reconstruction: Recent Advances and Open Challenges** — [arXiv:2507.16406](https://arxiv.org/abs/2507.16406)

### Uncertainty estimation

12. **Beta-NLL** — Seitzer et al. ICLR 2022. [arXiv:2203.09168](https://arxiv.org/abs/2203.09168). 3-line fix for heteroscedastic variance learning.
13. **pixelSplat** — Charatan et al. CVPR 2024. Probabilistic depth via reparameterized sampling in feed-forward 3DGS.
14. **Evidential regression** — Amini et al. NeurIPS 2018. Single-pass aleatoric + epistemic decomposition via NIG prior. Flexible EDL: NeurIPS 2025.
15. **Faithful heteroscedastic regression** — Stirn & Knowles. AISTATS 2023. Stop-gradient variance head.
16. **Conformal prediction** — Angelopoulos & Bates 2021. Distribution-free prediction sets with coverage guarantees.
17. **Senseiver** — Janny et al. Nature Machine Intelligence 2023. Perceiver IO for field reconstruction from sparse sensors — structurally identical to our architecture.
18. **UQ-SONet** — Paul et al. TMLR 2025. Set transformer + cVAE for operator learning with built-in UQ.

### For the user's background (3D hand pose)

19. **Look Ma, no markers** — Hewitt et al. 2024. [arXiv:2410.11520](https://arxiv.org/abs/2410.11520). Holistic performance capture without markers. Parametric model fitting to multi-view observations — the direct inspiration for our reprojection loss framing.
