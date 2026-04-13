# Neuro-MoBRE: Exploring Multi-Subject Multi-Task Intracranial Decoding via Explicit Heterogeneity Resolving

## Citation
Wu, D., Li, Z., Liang, M., Gao, C., Si, J., Zhu, J., & He, B. (2025). Neuro-MoBRE: Exploring Multi-Subject Multi-Task Intracranial Decoding via Explicit Heterogeneity Resolving. *Preprint, August 2025*. Same group as MIBRAIN (Wu et al. 2025).

---

## Setup

- **Modality:** Stereoelectroencephalography (sEEG), depth electrodes for epilepsy monitoring. Same 11 patients as MIBRAIN
- **Patients:** 11 epilepsy patients; variable electrode counts and placement; 21 brain regions covered (FreeSurfer parcellation)
- **Tasks:** 5 tasks evaluated -- 23-class initial consonant, 11-class final phoneme, 4-class Mandarin tone, seizure detection, seizure prediction. Multi-task training with task-disentangled CLS tokens
- **Signal:** Raw broadband sEEG preprocessed to 512 Hz (same as MIBRAIN pipeline: low-pass 200 Hz, notch 50/100 Hz, downsample)
- **Data:** Same Mandarin articulation corpus as MIBRAIN (407 monosyllabic characters, 3-5 trials each) plus epilepsy monitoring data for seizure tasks

---

## Architecture

```
Per-region sEEG channels (variable count per region per patient)
  |
-> Brain-Regional-Temporal Tokenizer (BRT)
     5 sequential 1D conv layers: 64->64->64->32->64
     Per-region independent processing (like MIBRAIN)
     Produces region tokens: R^{N_regions x T' x 64}
  |
-> Decoder-only Transformer (4 blocks, d=64, 8 heads)
     Each block contains:
       - BrMoE (Brain-Region Mixture-of-Experts):
           21 expert networks (1 per brain region)
           TopK=2 routing per token
           Router: Linear(64) -> softmax -> top-2 selection
           Replaces standard FFN in transformer block
       - Task-Disentangled CLS tokens:
           J=4 width multiplier (4 CLS tokens per task)
           Each task has dedicated CLS tokens for readout
           Tasks share the same backbone but read from different CLS positions
  |
-> Task-specific MLP heads (one per task)
     Read from task-disentangled CLS tokens
```

- **Total params:** ~2-5M estimated (small by design -- data-limited regime penalizes over-parameterization)
- **Per-patient params:** ZERO. No per-patient layers, normalization, or embeddings. All heterogeneity resolved via region routing + SSL
- **Expert count:** 21 (= number of brain regions). Ablation confirms matching expert count to region count is optimal; fewer experts lose region specificity, more experts fragment sparse data

---

## SSL / Pretraining

**Two-stage pretraining pipeline:**

**Stage 1 -- Per-subject RMAE (Region-structured Masked Autoencoding):**
- Train one model per subject independently
- Region-structured masking: mask ratio r=0.2 (mask entire region tokens, not random individual tokens)
- Reconstruction target: **frequency-domain** representation via DFT -> mask -> reconstruct -> IDFT
- Loss: MSE in frequency domain (not time domain)
- 800 epochs per subject
- This produces N per-subject expert models

**Stage 2 -- Co-upcycling initialization:**
- Merge per-subject models into a single multi-subject model via **ties-merging**:
  1. Compute task vectors (difference from random init) for each per-subject model
  2. Apply 50% magnitude pruning (keep top-50% by absolute value)
  3. Sign consensus: only keep parameters where >50% of subject models agree on sign
  4. Average surviving parameters
- The merged model initializes the shared backbone
- Then supervised multi-task fine-tuning: 200 epochs, all subjects + all tasks jointly

**Why frequency-domain target:** The authors argue frequency components capture oscillatory patterns more relevant to neural coding than raw time-domain reconstruction. Ablation shows frequency-domain RMAE > time-domain RMAE

---

## Cross-Patient Handling

**Mechanism:** FreeSurfer anatomical parcellation -> discrete region embeddings + MoE routing by region

- Each electrode assigned to 1 of 21 brain regions via FreeSurfer parcellation (hard assignment, same as MIBRAIN)
- Region identity determines MoE expert routing: tokens from region R preferentially activate expert R
- NO MNI coordinates used. NO distance bias. NO continuous spatial information
- NO per-patient params at all -- heterogeneity resolved entirely by:
  1. Region-level tokenization (absorbs electrode count variation)
  2. MoE expert specialization (each region gets dedicated computation)
  3. Co-upcycling (merges per-subject knowledge into shared weights)

**Zero-shot LOSO evaluation:** Leave-one-subject-out, no fine-tuning on held-out subject. Above-chance performance achieved for 9/11 subjects. This is a true zero-shot test (unlike Singh who fine-tunes 100 epochs)

**Evolution from MIBRAIN:** MIBRAIN used learnable region prototypes for missing regions + region attention encoder. Neuro-MoBRE replaces prototypes with MoE routing (soft expert selection vs hard prototype substitution) and adds multi-task training. The MoE approach is more parameter-efficient and naturally handles partial coverage via routing weights

---

## I/O Features

| Feature | Detail |
|---------|--------|
| Input representation | Raw broadband sEEG at 512 Hz, per-region tokenized |
| Input spatial info | FreeSurfer region labels (categorical, no coordinates) |
| Temporal resolution | 5-layer conv tokenizer with temporal downsampling |
| Output | Task-specific classification via disentangled CLS tokens |
| Sequence modeling | Decoder-only transformer (4 blocks), causal attention |
| Vocabulary | 23 initial, 11 final, 4 tone, 2 seizure (multi-task) |

---

## Key Results

### Multi-task classification accuracy

| Task | Neuro-MoBRE | Best baseline | Delta | Chance |
|------|------------|--------------|-------|--------|
| 23-class initial | **28.26%** | ~20% (Brant) | +8pp | 4.3% |
| 11-class final | **32.15%** | ~22% | +10pp | 9.1% |
| 4-class tone | **43.41%** | ~32% | +11pp | 25.0% |
| Seizure detection | **89.57%** | ~82% | +7pp | 50.0% |
| Seizure prediction | **81.24%** | ~72% | +9pp | 50.0% |

Average improvement over multi-task baselines: **+17.98%**

### Zero-shot LOSO
- Above chance for 9/11 subjects (no fine-tuning on held-out subject)
- Modest absolute accuracy but demonstrates cross-patient transfer without any adaptation

### Ablation highlights

| Component | Effect |
|-----------|--------|
| Expert count = region count (21) | Optimal; fewer or more hurts |
| Transformer-16-512 (larger) | Collapses -- over-parameterization in data-limited regime |
| BrMoE vs standard FFN | BrMoE significantly better (+4-8pp across tasks) |
| Task-disentangled CLS (J=4) | Better than shared CLS (J=1) or task-specific heads only |
| Frequency-domain RMAE | Better than time-domain RMAE |
| Co-upcycling init | Better than random init or simple averaging |
| RMAE masking ratio r=0.2 | Optimal; higher ratios destabilize |

### Scaling
- Multi-subject training consistently outperforms single-subject for all tasks
- But improvements require sufficient subject count (consistent with MIBRAIN's finding that N<6 can hurt)

---

## v12 Comparison

### What Neuro-MoBRE validates for v12
1. **Small models are correct for this regime.** Transformer-16-512 collapses; d=64 with 4 blocks works. v12's ~171K params is well-calibrated. Over-parameterization is the enemy with <200 trials/patient
2. **Region-based common space works for cross-patient transfer.** Both MIBRAIN and Neuro-MoBRE demonstrate that mapping electrodes to anatomical regions enables pooling. v12's VE cross-attention is the soft continuous version of this same principle
3. **Zero per-patient params can work (modestly).** 9/11 subjects above chance in zero-shot LOSO with no per-patient layers at all. But absolute accuracy is low -- per-patient params likely needed for competitive performance
4. **Multi-task training improves all tasks.** Disentangled CLS tokens allow shared backbone to serve multiple objectives. v12 could benefit from auxiliary tasks (e.g., patient ID prediction, temporal ordering)
5. **Frequency-domain SSL target is novel and importable.** DFT-based reconstruction target outperforms time-domain MSE. Worth testing for v12's temporal masking pretraining

### Where v12 goes beyond Neuro-MoBRE
1. **Continuous coordinates vs categorical regions.** v12 uses MNI + Fourier PE + distance bias for fine-grained spatial resolution. Neuro-MoBRE uses only region labels (21 categories). Electrodes at region boundaries lose spatial information with hard assignment
2. **Per-patient params (134 vs 0).** v12's diagonal normalization + learned delta/omega addresses measurement heterogeneity that zero-per-patient models absorb into backbone noise
3. **Sequence decoding.** v12 produces 3-phoneme CVC/VCV sequences via AR decoder. Neuro-MoBRE does single-class classification per trial
4. **Soft VE assignment vs hard region assignment.** Distance-biased cross-attention with ~25mm receptive fields gracefully handles electrodes near region boundaries. MoE top-2 routing is softer than MIBRAIN's hard parcellation but still operates on categorical region tokens
5. **Explicit registration error correction.** Learned delta/omega corrects systematic MNI alignment error. No equivalent in Neuro-MoBRE

### Evolution from MIBRAIN to Neuro-MoBRE (same group)
- Prototypes -> MoE routing (more parameter-efficient)
- Single-task -> multi-task (disentangled CLS)
- Region masking MAE -> region-structured frequency-domain MAE
- Channel similarity alignment -> co-upcycling initialization
- The trajectory shows the group converging on region-specialized computation as the core mechanism

---

## Regime Comparison Table

| Dimension | Neuro-MoBRE | v12 regime | Implication |
|-----------|------------|-----------|-------------|
| Data volume | ~11 patients, sEEG monitoring (hours/pt) | ~11 patients, ~1 min epoched/pt | Similar patient count; v12 has far less data per patient |
| Trials per subject | ~1200+ per task | 46-178 | Neuro-MoBRE can afford zero per-patient params; v12 likely cannot |
| Electrode coverage | Whole-brain distributed (21 regions) | Focal vSMC patch (~6-12 of 16 VEs reachable) | Neuro-MoBRE's region diversity enables MoE specialization; v12's coverage is sparser |
| Signal type | Raw broadband 512 Hz | HGA envelopes 200 Hz | Different feature spaces; frequency-domain SSL target may not transfer directly to HGA |
| Task structure | Multi-task (5 tasks, shared backbone) | Single-task (phoneme sequence) | Multi-task regularization helps Neuro-MoBRE; v12 could add auxiliary tasks |
| Model scale | d=64, 4 blocks, ~2-5M | d varies, B=1, ~171K | Both appropriately small for regime; v12 is even leaner |
| Per-patient mechanism | Zero (MoE routing absorbs variation) | Diagonal norm + delta/omega (134/pt) | v12's explicit per-patient params target measurement physics directly |
| SSL strategy | Per-subject RMAE -> co-upcycling merge | Temporal span masking (joint across patients) | Neuro-MoBRE's per-subject-then-merge is interesting but requires enough data per subject |

### Key takeaway for v12
Neuro-MoBRE confirms region-based common space is the right abstraction for cross-patient intracranial decoding, and that small models dominate in this data regime. The frequency-domain SSL target and co-upcycling initialization are importable innovations. The zero-per-patient-params result (9/11 above chance) sets a useful floor -- v12's 134 per-patient params should beat this floor substantially. The MoE evolution from MIBRAIN's prototypes shows the field converging on region-specialized computation, which v12's distance-biased VE cross-attention implements in continuous coordinate space.
