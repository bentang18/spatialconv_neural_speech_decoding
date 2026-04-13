# Population Transformer: Learning Population-Level Representations of Neural Activity

## Citation
Chau, C.L.\*, Wang, C.Z.\*, et al. (2025). Population Transformer: Learning Population-Level Representations of Neural Activity. *ICLR 2025*. Code: https://github.com/czlwang/PopulationTransformer

---

## Setup

- **Modality:** Stereoelectroencephalography (sEEG), depth electrodes implanted for epilepsy monitoring
- **Patients:** 10 subjects from the Brain Treebank dataset; 1,688 total electrodes (~167/subject average, variable per patient)
- **Data:** ~55.5 hours total recording; movie watching (naturalistic stimuli)
- **Tasks (downstream probes):** Binary classification -- pitch (high/low), volume (loud/quiet), sentence onset (yes/no), speech vs non-speech. All evaluated via ROC-AUC on single-timepoint snapshots
- **Signal:** Frozen temporal encoder applied per electrode to extract per-electrode feature vectors. Default: BrainBERT (d=768, pretrained on sEEG spectrograms). Also tested: wav2vec 2.0, raw voltage. Temporal encoder is NEVER fine-tuned -- only the population-level transformer trains

---

## Architecture

```
Per-electrode temporal encoding (frozen BrainBERT, d=768)
  each electrode independently -> f_e in R^768
  |
+ 3D sinusoidal PE on MNI coordinates (Left/Posterior/Inferior axes)
  |
-> [CLS] token prepended to electrode token set
  |
-> 6-layer Transformer encoder (d=512, 8 heads, FFN=2048)
     Self-attention over ALL electrodes + [CLS]
     Variable ensemble size N_e per sample (random subsets during training)
  |
-> [CLS] output -> Linear probe -> binary prediction
```

- **Total params:** ~20M (transformer encoder only; BrainBERT frozen, not counted)
- **Per-patient params:** ZERO. No per-patient layers, normalization, or heads
- **Positional encoding:** 3D sinusoidal on MNI coordinates (Left, Posterior, Inferior axes). Added to electrode token embeddings before transformer input. Encodes anatomical location, not electrode index
- **Variable ensemble handling:** During SSL pretraining, random subsets of electrodes are sampled per batch (simulates variable coverage). At inference, any number of electrodes works natively via set-based self-attention

---

## SSL / Pretraining

Two discriminative (NOT reconstructive) objectives applied jointly:

1. **Ensemble-wise temporal discrimination:** Given two sets of electrode embeddings from adjacent or non-adjacent time windows, predict whether they are temporally consecutive. BCE loss on [CLS] token output. Tests whether the population snapshot captures temporal dynamics

2. **Channel-wise swap detection:** 10% of electrode tokens are randomly replaced with tokens from other timepoints (same electrode). Per-token BCE predicts which tokens were swapped. Tests per-electrode temporal consistency within the population context

**Key ablation:** Discriminative SSL >> reconstructive (MAE-style). Reconstructing electrode tokens performs substantially worse. The authors hypothesize discriminative objectives force the model to capture relational structure across electrodes rather than per-electrode statistics

**Training:**
- Optimizer: LAMB, lr=5e-4
- Batch size: 256
- Steps: 500K
- Hardware: 2 days on 1 Titan RTX
- Electrode subset sampling during pretraining for robustness to variable coverage

---

## Cross-Patient Handling

**Mechanism:** Self-attention over ALL electrodes with 3D sinusoidal PE. That's it.

- No atlas mapping, no virtual electrodes, no distance bias
- No per-patient normalization, read-in, or read-out layers
- No domain adaptation, no adversarial loss, no contrastive alignment
- Electrode identity encoded solely by MNI coordinate PE
- Variable electrode counts handled natively by set-based self-attention (no padding needed)

**Hold-one-out evaluation:** Training on 9/10 subjects and evaluating on the held-out subject shows minimal degradation compared to training on all 10. Cross-patient generalization emerges purely from shared MNI coordinate space + self-attention

**Coordinate jitter augmentation (sigma=5mm):** Tested but did NOT help. The authors speculate MNI registration noise is already sufficient natural augmentation

---

## I/O Features

| Feature | Detail |
|---------|--------|
| Input representation | Frozen per-electrode temporal features (BrainBERT d=768) |
| Input spatial info | MNI coordinates (3D sinusoidal PE) |
| Temporal resolution | Single timepoint snapshot (no sequence modeling) |
| Output | Binary classification per timepoint ([CLS] -> Linear -> sigmoid) |
| Sequence modeling | NONE -- single-frame population snapshot only |
| Vocabulary | N/A (binary probes, not phoneme/word decoding) |

---

## Key Results

### Downstream probe accuracy (ROC-AUC)

| Task | PopT (SSL) | PopT (no pretrain) | Delta |
|------|-----------|-------------------|-------|
| Pitch (high/low) | **0.74** | ~0.54 | +0.20 |
| Volume (loud/quiet) | **0.87** | ~0.64 | +0.23 |
| Sentence onset | **0.90** | ~0.75 | +0.15 |
| Speech/non-speech | **0.93** | ~0.80 | +0.13 |

### Ablation study (speech/non-speech AUC)

| Ablation | AUC | Delta from full |
|----------|-----|-----------------|
| Full model | **0.93** | -- |
| - PE (remove positional encoding) | 0.83 | **-0.10** (most damaging) |
| - SSL pretraining | ~0.80 | -0.13 |
| - Ensemble-wise objective | 0.90 | -0.03 |
| - Channel-wise objective | 0.91 | -0.02 |

PE removal is the single most damaging architectural ablation (0.93 -> 0.83 speech, 0.74 -> 0.62 pitch). This contrasts with seegnificant where PE barely helped (delta R^2 = -0.02, p=0.73).

### Data efficiency
- SSL pretraining provides ~5x data efficiency: models pretrained with SSL match the performance of non-pretrained models using 5x more labeled data

### Temporal encoder comparison
- BrainBERT (sEEG-pretrained) > wav2vec 2.0 (audio-pretrained) > raw voltage features
- Domain-matched pretraining matters for the frozen temporal encoder

---

## v12 Comparison

### What PopT validates for v12
1. **Self-attention + 3D PE works for variable electrodes without atlas mapping.** Zero per-patient params, no VEs, no distance bias -- and cross-patient transfer still works. This is the strongest evidence that v12's A_self_attn ablation (self-attention without VEs) is a serious competitor
2. **PE matters more here than in seegnificant.** The discrepancy may be task-dependent: PopT's tasks involve distributed networks (pitch, speech detection) where position disambiguates large-scale regions, similar to how v12's phoneme task involves somatotopic distinctions in focal vSMC
3. **Discriminative > reconstructive SSL.** v12 plans temporal span masking (reconstructive, MSE). PopT found reconstructive SSL substantially worse. However, PopT's reconstructive baseline reconstructs frozen BrainBERT embeddings (already compressed), not raw neural signals -- the comparison may not transfer

### Why PopT cannot do our task
1. **No temporal sequence modeling.** PopT operates on single-timepoint population snapshots. Phoneme decoding requires temporal dynamics (CTC/AR over ~500ms windows). PopT literally cannot produce a phoneme sequence
2. **Binary detection, not 9-way classification.** All PopT tasks are binary (AUC metric). 9-phoneme decoding from 46-178 trials is orders of magnitude harder
3. **No per-patient normalization.** PopT's zero per-patient params works for binary detection on long naturalistic recordings (~5.5h/subject). With 46-178 trials per patient and 9 classes, per-patient layers are likely essential (seegnificant: delta R^2 = -0.18 from removing per-subject heads)
4. **Frozen temporal encoder assumes pretrained representations exist.** BrainBERT was pretrained on sEEG spectrograms. No equivalent exists for uECOG HGA -- we would need to train our own temporal encoder, which is what v12's Conv1d + backbone already does

### Critical comparison

| Dimension | Population Transformer | v12 |
|-----------|----------------------|-----|
| Modality | sEEG (depth, sparse 3D) | uECOG (surface, dense 2D grid) |
| Electrodes/subject | ~167 (3D distributed) | 63-201 (2D focal patch) |
| Subjects | 10 | 11 (4 core) |
| Data per subject | ~5.5 hours | ~1 min epoched |
| Task | Binary detection (AUC) | 9-phoneme sequence (PER) |
| Temporal modeling | None (single snapshot) | Conv1d + self-attention + AR decoder |
| Spatial mechanism | Self-attention + 3D sinusoidal PE | VE cross-attention + distance bias + Fourier PE |
| Per-patient params | 0 | 134 (diagonal + delta/omega) |
| Total params | ~20M (+ frozen BrainBERT) | ~171K |
| SSL objective | Discriminative (temporal + swap) | Reconstructive (temporal span masking, MSE) |
| Atlas / VEs | None | 16 Brainnetome VEs |

---

## Regime Comparison Table

| Dimension | PopT | v12 regime | Implication |
|-----------|------|-----------|-------------|
| Data volume | ~55.5h (10 subjects) | ~11 min epoched, ~456 min raw | PopT operates in data-rich regime; v12 is 100x smaller for SSL |
| Trials per subject | ~20K+ frames | 46-178 trials | PopT can afford zero per-patient params; v12 cannot |
| Electrode coverage | Whole-brain distributed | Focal vSMC patch | PE importance likely differs: distributed networks vs somatotopic |
| Task complexity | Binary detection | 9-way temporal sequence | PopT's simplicity makes zero-patient-params feasible |
| Temporal encoder | Frozen pretrained (BrainBERT) | Must train from scratch | v12 must learn temporal representations jointly |
| Signal type | Broadband sEEG (via BrainBERT) | HGA envelopes at 200 Hz | Different feature spaces; no pretrained temporal encoder for HGA |
| Per-patient calibration | None needed | Essential (seegnificant: delta R^2=-0.18) | Regime difference, not architectural insight |

### Key takeaway for v12
PopT demonstrates that self-attention + coordinate PE alone can achieve cross-patient generalization -- but in a fundamentally different regime (binary detection, 5.5h/subject, frozen pretrained features). The A_self_attn ablation tests whether this simplicity transfers to our harder regime. If it does, VEs may be unnecessary overhead. If it doesn't, VEs earn their keep by providing the atlas prior that self-attention alone cannot learn from N=10 patients with ~1 min each.
