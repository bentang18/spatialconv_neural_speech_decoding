# Transformer-Based Neural Speech Decoding from Surface and Depth Electrode Signals

**Authors:** Junbo Chen, Xupeng Chen*, Ran Wang*, Chenqian Le, Amirhossein Khalilian-Gourtani, Leyao Yu, Patricia Dugan, Daniel Friedman, Werner Doyle, Orrin Devinsky, Yao Wang†, Adeen Flinker†
**Venue/Year:** Journal of Neural Engineering, Volume 22, Issue 1, 016017, January 2025
**DOI:** https://doi.org/10.1088/1741-2552/ada741
**Project page:** https://xc1490.github.io/swinTW/
**Data:** https://data.mendeley.com/datasets/fp4bv9gtwk/1 (v1 — same cohort as Chen 2024 v2 + 4 additional participants). Audio data shared on request to corresponding author.
**Code:** https://github.com/flinkerlab/neural_speech_decoding
**Predecessor:** Chen et al. 2024 (Nature Machine Intelligence) — per-patient 3D ResNet + speech synthesizer

## TL;DR

Follow-up to Chen 2024. Replaces the grid-dependent 3D ResNet/Swin with **SwinTW**, a transformer that uses MNI cortical coordinates instead of grid positions for positional encoding. This removes the requirement for grid-structured electrodes, enabling: (1) sEEG-only decoding (PCC 0.798, N=9), (2) **multi-subject training without per-subject layers** — a single 15-patient model matches 15 individual models (PCC 0.837 vs 0.831, NS), and (3) leave-one-out decoding of unseen patients (PCC 0.765). First paper to demonstrate cross-patient ECoG speech decoding via coordinate-based tokenization with NO per-patient input layers.

## Problem & Motivation

Chen 2024's 3D ResNet and Swin Transformer required electrodes arranged in a regular grid (8×8 ECoG), preventing: (a) use of sEEG/strip electrodes, (b) multi-patient training across heterogeneous layouts, (c) generalization to new patients without retraining. The key insight: replace grid-based spatial structure with anatomical coordinates — the brain doesn't care where on the array an electrode sits, only where in the cortex it samples.

## Method

### SwinTW Architecture

```
ECoG/sEEG input (N_electrodes × T, 125 Hz)
  │
  ├─ Temporal patch partition: T → T/W patches (W=4)
  │   → N_electrodes × T/W tokens, each token = (1 electrode, W timepoints)
  │
  ├─ Positional encoding: MNI coordinates (x,y,z) + brain region index (ROI)
  │   → Positional bias B_{i,j} added to **scaled cosine similarity** attention (not dot product):
  │     SIM(q_i, k_j) = (q_i·k_j)/(||q_i||·||k_j||/τ) + B_{i,j}
  │     τ = learnable per-head, per-layer temperature
  │     B_{i,j} = 2-layer MLP(x_i,y_i,z_i,t_i, x_j,y_j,z_j,t_j, Δx,Δy,Δz,Δt) + r_i·r_j (ROI embedding dot product)
  │     Unknown electrode coords set to (0,0,0), ROI="Unknown" (1-11 per patient)
  │
  ├─ Stage 1: 2 transformer layers (3 heads, spatial-temporal windowed attention, Wt=4)
  ├─ Stage 2: 2 transformer layers (6 heads)
  ├─ Stage 3: 6 transformer layers (12 heads)
  │   Latent dim C=96 throughout, reduced to C'=32 after Stage 3
  │
  ├─ **Spatial max pooling** across all electrodes → reduces N_electrode dim to 1
  │   (all spatial info must flow through attention stages, not decoder)
  │
  ├─ MLP FFN inside each transformer block: 384→196→96→32 (LayerNorm + LeakyReLU)
  │   (this is the block FFN, NOT a separate prediction head)
  │
  └─ 4× transposed Conv1d (stride=2, kernel=3) → upsample to original temporal resolution
      → temporal conv + MLP → 18 speech parameters per timestep
```

### Spatial-Temporal Windowed Attention

Attention is computed within local windows of Wt=4 consecutive temporal patches. **All electrodes attend to each other within each temporal window** — there is NO spatial windowing. Spatial attention is O(N_electrodes²) per window. Shifted windows (from Swin Transformer) enable cross-window information flow in the **temporal** direction only. Uses **SwinV2-style scaled cosine similarity** with learnable temperature, not standard scaled dot-product attention.

**Temporal resolution progression:** Input T → T/4 patches (Stage 1) → T/8 (Stage 2, after merge) → T/16 (Stage 3, after merge) → spatial max pool removes electrode dim → T/16 → 4× transposed conv → T (restored).

### Multi-Subject Training

- **Shared decoder:** Single SwinTW model across all patients. NO per-patient input layers, read-in, or adapters
- **Per-patient components:** Only the Speech Encoder (audio→18 params) and Speech Synthesizer (18 params→spectrogram) are per-patient (trained in Stage 1 on audio only, frozen during decoder training)
- **How variation is handled:** MNI coordinates tell the model where each electrode samples. The model learns position→function mapping rather than electrode-index→function mapping
- **Training set:** 15 male ECoG patients pooled (**4 left + 11 right hemisphere**; gender-separated to reduce acoustic variance). Separate region embeddings learned for L/R hemispheres. Both-hemisphere combined models perform comparably to hemisphere-specific on unseen patients (p > 0.05)

### Loss

Same as Chen 2024: multi-scale spectral loss + STOI+ loss (weight 1.2) + supervision loss (0.1) + reference parameter loss (1.0). Optimizer: Adam (lr=5e-4).

## Data

- **N = 52** total: 43 with ECoG (8×8 subdural, 1cm spacing), 9 with sEEG only, 39 with additional strip/depth electrodes beyond the primary grid
- **Same cohort as Chen 2024** but expanded from 48 to 52 (sEEG-only patients added)
- **Tasks:** 5 speech production tasks × 50 words = 400 trials/patient; 350 train / 50 test
- **Signal:** HGA 70–150 Hz, Hilbert envelope, 125 Hz, CAR, z-scored
- **Coordinates:** MNI-transformed via FreeSurfer + electrode localization; ROI index from brain parcellation (atlas not named in paper — Desikan-Killiany attribution is unconfirmed)

## Key Results

### Subject-Specific SwinTW (vs. Prior Architectures)

| Architecture | Electrodes | N | PCC | STOI+ | MCD |
|---|---|---|---|---|---|
| LSTM (Chen 2024) | 8×8 grid | 43 | 0.745 | — | — |
| 3D Swin (Chen 2024) | 8×8 grid | 43 | 0.785 | — | — |
| **3D ResNet (Chen 2024)** | 8×8 grid | 43 | 0.804 | — | — |
| **SwinTW** | 8×8 grid | 43 | **0.825** | 0.309 | 2.341 |
| SwinTW | 8×8 + extra | 39 | **0.838** | 0.359 | 2.228 |
| SwinTW | sEEG only | 9 | **0.798** | 0.341 | 2.396 |

SwinTW > ResNet (p<0.001) and > 3D Swin (p<0.001) on 8×8 grid.

### Multi-Subject (15 Male ECoG Patients)

| Condition | PCC | STOI+ | MCD |
|---|---|---|---|
| Individual models (15 separate) | 0.831 | 0.334 | 2.313 |
| **Multi-subject model (1 shared)** | **0.837** | 0.352 | 2.307 |
| Statistical difference | NS | NS | NS |

One model trained on all 15 patients matches or slightly exceeds 15 individual models.

### Leave-Group-Out (Unseen Patients)

- **Mean PCC = 0.765** across **5-fold leave-group-out CV** (not true LOO — each fold holds out ~4-5 patients, gender-separated). 20 male + 23 female evaluated separately
- Performance "not consistently high" — some patients decoded well, others poorly. No per-patient PCC values reported
- **Top-N baseline:** 3 best female + 2 best male individual models tested on unseen patients — multi-subject model significantly outperforms these top individual models, showing the advantage isn't just averaging
- Authors note: expanding training set should improve generalization

## Relevance to Cross-Patient uECOG Speech Decoding

### Directly applicable

1. **Coordinate-based tokenization validates cross-patient training without per-patient layers.** This is the strongest evidence that MNI coordinates alone can handle electrode placement variability across patients. Our implementation_plan Phase 3.1 variant D (2D conv on grid) and Phase 3 coordinate PE are both validated by this result.

2. **Multi-subject model matches individual models (PCC 0.837 vs 0.831).** Cross-patient pooling provides NO degradation with coordinate encoding. This is a stronger result than Singh (group PER 0.49 vs single 0.57 — pooling helps but doesn't match individual). The difference: coordinates explicitly encode spatial information that per-patient linear layers must learn implicitly.

3. **LOO PCC 0.765 on unseen patients.** This is genuine zero-shot cross-patient speech decoding — no fine-tuning on the target patient. Even without per-patient adaptation, the model extracts useful speech information from novel electrode placements. With per-patient read-in fine-tuning (our Stage 2), this would likely improve.

4. **sEEG-only PCC 0.798.** The architecture handles non-grid electrodes natively. Relevant if any of our patients have supplementary depth electrodes.

5. **Same lab, same preprocessing, same 18-parameter target as Chen 2024.** The comparison is clean — SwinTW vs ResNet vs LSTM on identical data.

### What doesn't transfer

1. **Standard ECoG (1cm spacing), not uECOG (200μm–1.7mm).** Spatial resolution is 5–50× lower. At 1cm, nearby electrodes sample overlapping cortical populations. At uECOG resolution, adjacent electrodes carry distinct information (Duraivel). Coordinate PE may be less valuable when spatial relationships are already dense and regular.

2. **No per-patient layers tested in combination.** The paper shows coordinates alone work, but doesn't test coordinate PE + per-patient read-in. For our setting (variable surgical placement, different array types), the combination may be better than either alone.

3. **Decoding target is 18 speech parameters, not phonemes.** Frame-level regression with MSE-type losses, not CTC sequence decoding. The tasks are different enough that results don't directly predict phoneme classification performance.

4. **52 patients (large cohort for brain decoding).** We have ~20. The coordinate→function mapping may need more patients to learn robustly. Their LOO result suggests 43 is sufficient; unclear if 20 suffices.

5. **Chronic epilepsy monitoring, not intra-op.** Days of recording vs ~10 minutes. The multi-subject result uses full datasets per patient.

### Critical comparison to our approach

| Dimension | Chen 2025 SwinTW | Our Phase 2 |
|---|---|---|
| Electrode variation handling | MNI coordinates | Per-patient linear read-in |
| Per-patient params | 0 (in decoder) | ~10–21k (read-in layer) |
| Zero-shot capable? | Yes (PCC 0.765) | No (needs Stage 2 fine-tuning) |
| Multi-subject = individual? | Yes (0.837 vs 0.831) | Unknown |
| Grid structure exploited? | No (individual tokens) | Phase 3 ablation (2D conv) |
| Temporal modeling | Windowed attention | BiGRU |
| Sequence loss | None (frame-level MSE) | CTC |

**The hybrid approach — coordinate PE + per-patient read-in — is unexplored.** Coordinates provide anatomical priors; read-in provides patient-specific gain adjustment. This combination is a strong Phase 3 ablation candidate.

## Limitations

1. LOO performance (PCC 0.765) noticeably lower than within-training (0.837) — generalization gap remains
2. Multi-subject training limited to 15 same-gender patients — scalability to mixed cohorts untested
3. No fine-tuning/adaptation on held-out patients — the gap between 0.765 and 0.837 could likely be closed with minimal per-patient adaptation
4. Attention over all electrodes per window — quadratic in electrode count. For 256-ch uECOG, this is 256² = 65k attention entries per window vs 64² = 4k for 8×8 grid
5. Frame-level regression only — no sequence-level decoding or language model integration
6. Training: Adam lr=5×10⁻⁴, β1=0.9, β2=0.999, 60 epochs, 350 train / 50 test per participant (10 per task reserved). SwinTW: C=96, C'=32, patch W=4, window W_t=4, stages 2/2/6 layers with 3/6/12 heads. MLP: 384→196→96→32
7. **Spatial max pooling** after Stage 3 compresses all electrode information to a single vector per timepoint — potentially lossy for uECOG's 128-256 channels carrying distinct spatial info at <2mm

### Future work noted by authors
- Authors explicitly propose **hybrid per-patient + shared architecture**: "train non-subject-specific layers with large multi-subject dataset and refine only subject-specific layer with small amount of data for any new participants" — this is essentially our Stage 2 approach, independently validating our design
- Authors mention exploring **HuBERT latent features** as alternative target — connects to our Speech FM projection idea

## Reusable Ideas

### 1. Coordinate-based tokenization (Phase 3 ablation — high priority)
Replace grid-based 2D conv or linear read-in with MNI coordinate positional encoding. Each electrode becomes a token with position information from BrainLab coordinates. This enables zero-shot generalization and removes the per-patient input layer entirely. **Test as Phase 3.1 variant alongside linear read-in, diagonal+bias, low-rank, and 2D conv.**

### 2. Hybrid: coordinate PE + per-patient read-in
Coordinates provide anatomical priors (position→function); per-patient read-in adjusts for impedance, signal quality, and patient-specific cortical organization. The combination is untested anywhere in the literature — a genuine novel contribution.

### 3. Individual electrode tokenization
Instead of treating the entire channel dimension as one vector (our current approach), tokenize each electrode separately (as SwinTW does). This naturally handles variable channel counts without zero-padding. But: attention is O(N²) in electrode count, which is expensive for 128–256 channels.

### 4. Gender-separated multi-subject training
SwinTW trains multi-subject models within gender. Relevant if acoustic targets are used (formants are gender-dependent). Not relevant for phoneme classification (our primary task).

## Key References

- Chen, X. et al. (2024) A neural speech decoding framework leveraging deep learning and speech synthesis. *Nature Machine Intelligence* 6, 467–480.
- Liu, Z. et al. (2021) Swin Transformer: Hierarchical vision transformer using shifted windows. *ICCV*.
- Desikan, R.S. et al. (2006) An automated labeling system for subdividing the human cerebral cortex. *NeuroImage* 31, 968–980.
