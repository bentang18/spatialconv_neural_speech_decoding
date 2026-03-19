# Neuro-MoBRE: Exploring Multi-subject Multi-task Intracranial Decoding via Explicit Heterogeneity Resolving

**Authors:** Di Wu, Yifei Jia, Siyuan Li, Shiqi Zhao, Jie Yang, Mohamad Sawan
**Venue/Year:** arXiv preprint, August 2025 (submitted to ICLR 2026, rejected)
**arXiv:** https://arxiv.org/abs/2508.04128
**OpenReview:** https://openreview.net/forum?id=eTmjZ82hXa

## TL;DR

Multi-subject multi-task intracranial decoder using Mixture-of-Brain-Regional-Experts (MoE) routing, where each expert specializes in a brain region defined by MNI parcellation. Tested on 11 sEEG epilepsy patients across 5 tasks (3 Mandarin speech, 2 seizure). Roughly doubles best baseline accuracy on speech tasks (0.283 vs 0.156 on 23-class initial consonants). Claims above-chance zero-shot decoding on unseen subjects via leave-one-subject-out. **Rejected from ICLR 2026.**

## Problem & Motivation

Multi-subject intracranial decoding faces two heterogeneities: (1) electrode placement varies across subjects (especially sEEG), and (2) different tasks require different neural features. Existing brain foundation models (BIOT, Brant, LaBraM, NeuroBERT) handle neither well — they learn generic temporal features without accounting for regional specialization or task-specific readout. The authors propose resolving both heterogeneities explicitly: brain-regional MoE for spatial variation, task-disentangled CLS tokens for task variation.

## Method

### Architecture (4 components)

**1. Brain-Regional-Temporal Tokenizer**
- Cascade of 5 temporal 1D convolutions: kernel sizes {15, 7, 5, 3, 3}, strides {7, 4, 3, 2, 2}, filters {8, 16, 16, 32, 64}
- Produces per-channel, per-time-patch embeddings
- Final token embedding: e_final(p,c) = e(p,c) + e^t_p + e^r_q
  - e^t_p: learnable temporal positional embedding
  - e^r_q: learnable region embedding (q = brain region index from MNI parcellation)
  - Each sEEG contact assigned to a brain region; contacts in same region share region embedding

**2. Brain-Regional MoE (BrMoE) Blocks**
- N_x experts, where N_x = number of brain regions of interest (tested 4, 8, 16, 21; default 21)
- Top-k routing with k=2
- Router: aggregate channel embeddings across time via summation → multiply by router matrix Φ ∈ R^(d × N_x) → softmax gating
- Load balancing auxiliary loss (Lepikhin et al.)
- Key alignment mechanism: each expert specializes in a brain region, so cross-subject variation is handled by routing channels to appropriate regional expert based on MNI location
- **No explicit per-subject input layers** — alignment is purely through regional routing

**3. Task-Disentangled Information Aggregation (TIA)**
- n task-specific CLS tokens (one per task), each of dimension J×d (J=4, d=64)
- CLS tokens split into J individual d-dim tokens, participate in multi-head self-attention alongside neural tokens
- After attention, task-specific CLS tokens → dedicated FFN heads per task

**4. Default architecture:** d=64, 4 Transformer blocks, 8 attention heads, MLP size 128, prediction head 256 (2 layers)

### Region-Masked Autoencoding (RMAE) Pretraining
- Mask entire brain regions (all tokens from same region masked together), ratio r=0.2
- Prediction target: frequency-domain representation (amplitude + phase) of masked regions, MSE loss in time domain
- 800 epochs, lr=5e-5, AdamW (β1=0.9, β2=0.95), weight decay 0.05, batch size 32 per task

### Co-Upcycling Initialization
Model merging strategy for initializing the multi-subject model:
1. Pre-train per-subject models individually via RMAE
2. Prune lowest-magnitude 50% of parameters to zero
3. Compute consensus sign across subjects
4. Average consensus-sign-agreeing parameters
5. MoE expert weights randomly initialized (not merged)

## Data

- **Modality:** Stereo-EEG (sEEG) — distributed depth electrodes, NOT surface grids
- **Subjects:** 11 epilepsy monitoring patients
- **Electrode placement:** Variable across subjects, mapped to MNI template coordinates
- **Preprocessing:** Low-pass 200 Hz, notch 50/100 Hz, downsample 512 Hz, per-channel z-normalization

### Tasks (5 total)
| Task | Subjects | Classes | Trials | Split |
|---|---|---|---|---|
| Mandarin initial consonant | 5 | 23 | 407 monosyllables × 3 repetitions | 10 per class test |
| Mandarin finals | 5 | 11 (clustered from 35 via k-means on formants) | Same stimuli | 20 per class test |
| Mandarin tone | 5 | 4 | Same stimuli | 80 per class test |
| Seizure prediction | 11 | 2 (preictal/interictal) | 5-min horizon, 35-min preictal | 80/20 train/test |
| Seizure detection | 11 | 2 (ictal/interictal) | — | 80/20 train/test |

## Key Results

### Language Decoding (top-1 accuracy, 5 subjects)

| Method | Initial (23-class) | Finals (11-class) | Tone (4-class) |
|---|---|---|---|
| BIOT | 0.148 | 0.174 | 0.321 |
| Brant | 0.142 | 0.196 | 0.355 |
| NeuroBERT | 0.131 | 0.203 | 0.322 |
| LaBraM | 0.130 | 0.194 | 0.333 |
| H2DiLR | 0.156 | 0.217 | 0.328 |
| **Neuro-MoBRE** | **0.283** | **0.322** | **0.434** |

Roughly doubles best baseline on initial/finals; +22% absolute on tone.

### Seizure Tasks (11 subjects)

| Method | Prediction | Detection |
|---|---|---|
| Best baseline | 0.678 | 0.801 |
| **Neuro-MoBRE** | **0.788** | **0.896** |

### Ablation (initial decoding)

| Component | Accuracy | Δ |
|---|---|---|
| Tokenizer only | 0.180 | — |
| + BrMoE | 0.248 | +6.9 pp |
| + TIA | 0.272 | +2.4 pp |
| + RMAE | 0.283 | +1.1 pp |

BrMoE provides the largest single improvement. Expert count scales monotonically: 4 experts (0.234) → 21 experts (0.283).

### Scaling behavior
4-layer d=64 is near-optimal. Scaling to 16-layer d=512 causes **model collapse** (drops to 0.160), consistent with small data regime. This mirrors our own concern about transformer sizing.

### Zero-shot on unseen subjects
Leave-one-subject-out evaluation. Claims "consistently above chance-level on previously unseen subjects." Specific per-subject numbers in Figure 3 only (not extracted as text). The zero-shot mechanism: new subject's electrodes are mapped to MNI regions, routed to the appropriate experts. No per-subject parameters needed at inference.

## Relevance to Cross-Patient uECOG Speech Decoding

### What's relevant

1. **The core finding that regional specialization helps (BrMoE is the biggest ablation gain).** This is consistent with the broader literature: handling spatial heterogeneity explicitly matters more than generic pretraining.

2. **Scaling collapse at 16-layer d=512.** Independent confirmation that large transformers fail in small intracranial data regimes. Supports our choice of GRU or small transformer.

3. **Co-upcycling initialization** (train per-subject, prune 50%, consensus-sign merge) is a potentially useful model merging strategy independent of MoE. Could be applied to merge per-patient GRU backbones.

4. **Region-masked autoencoding** — masking by spatial region rather than random tokens. For uECOG, this could translate to masking contiguous grid sub-regions (e.g., mask a 4×4 patch of the 8×16 grid).

5. **Brain foundation models (BIOT, Brant, LaBraM, NeuroBERT) all perform poorly** on multi-subject intracranial speech tasks. Further confirms our CLAUDE.md finding that existing pretrained models don't transfer to cross-patient intracranial decoding.

### What doesn't apply

1. **The core MoE routing mechanism doesn't translate.** It relies on sEEG contacts being distributed across **many distinct brain regions** (up to 21). Our uECOG arrays cover a **single contiguous cortical patch** (left sensorimotor cortex). All our channels are in roughly the same region — there's nothing to route between. The "expert per brain region" concept becomes degenerate when all electrodes are in one region.

2. **sEEG ≠ uECOG.** Distributed depth electrodes sampling heterogeneous structures vs. dense surface grid over homogeneous cortex. The spatial heterogeneity problem they solve is fundamentally different from ours.

3. **No per-subject input layers.** Alignment is purely through regional routing, which works for sEEG (each contact's region is its identity) but is insufficient for uECOG (all contacts share the same region, but per-patient gain variation exists). Our per-patient affine layers address a different problem than regional routing.

4. **Mandarin speech + seizure tasks.** Different language, different task structure, different neural representations than English phoneme/word repetition.

5. **Rejected from ICLR 2026.** Methodology may have unresolved issues. Reviewer concerns not examined here.

## Limitations

1. Only 11 subjects — small for learning MoE routing
2. sEEG only — not tested on ECoG or uECOG
3. ICLR rejection suggests potential methodological concerns
4. Zero-shot results shown only in figures, not tabulated — hard to evaluate magnitude
5. Expert count (21) approaches subject count (11) — risk of overfitting routing
6. Speech tasks are classification (consonant/tone ID), not sequence decoding (CTC/PER) — different and easier problem
7. No comparison to the per-patient input layer + shared backbone approach that dominates the ECoG/Utah literature

## Reusable Ideas

### 1. Region-masked autoencoding for uECOG (medium priority)
Mask contiguous spatial patches (e.g., 4×4 sub-grid) rather than random channels during SSL pretraining. Forces the model to reconstruct local spatial patterns from surrounding context — directly exploits the grid topology. More structured than random channel dropout.

### 2. Co-upcycling model merging (low priority, interesting)
Train per-patient models independently, then merge via pruning + consensus sign averaging to initialize a shared backbone. This is an alternative to joint training from scratch — potentially useful if joint training has convergence issues.

### 3. Task-specific CLS tokens (relevant for cross-task extension)
If combining phoneme patients and word/nonword patients (your cross-task gap), per-task CLS tokens allow the shared backbone to produce task-specific readouts without conflicting gradients. Simple to implement: prepend different learned tokens per task, use task-specific linear heads.

## Key References

- Lepikhin, D. et al. (2020) GShard: Scaling giant models with conditional computation. *ICLR*.
- Tang, J. et al. (2023) BIOT: cross-data biosignal learning. *NeurIPS*.
- Yuan, Z. et al. (2024) Brant-2: foundation model for brain signals. *ICLR*.
- Jiang, Z. et al. (2024) LaBraM: large brain model for learning generic representations. *ICLR*.
