# BrainWhisperer: Leveraging Large-Scale ASR Models for Neural Speech Decoding

**Authors:** Tommaso Boccato, Michal Olak, Matteo Ferrante
**Affiliation:** Tether Evo
**Venue/Year:** arXiv:2603.13321v1, March 2026
**Data:** Card et al. (NEJM 2024), Willett et al. (Nature 2023), Kunz et al. (Cell 2025) — all publicly available intracortical datasets
**Code:** Not yet released (promised in future version)

## TL;DR

Adapts OpenAI's Whisper (pretrained on 680k hours of speech audio) to decode intracortical MEA recordings. Key insight: Whisper's early encoder layers already learn phoneme-selective representations with localized attention — repurpose this for neural data. Achieves 8.7% WER end-to-end (SOTA for E2E decoding, first below 10%), matching heavier n-gram pipelines. Cross-dataset training improves single-dataset performance without fine-tuning. 4th/466 in Brain-to-Text '25 Kaggle.

## Problem & Motivation

Current MEA speech decoders face three limitations:
1. Extreme data requirements (~10^4 sentences/participant)
2. Rapid degradation across recording sessions (non-stationarity)
3. No cross-participant generalization

Current SOTA uses RNN + CTC → n-gram LM rescoring, requiring ~300GB RAM and ~750ms inference. BrainWhisperer hypothesizes that pretrained ASR models provide phonetic representations and linguistic priors that address all three.

## Method

### Neural Embedder

**Month/Day-Specific Projections:** Each temporal frame x_t is projected by:
```
x̃_t = σ([W + A·B] x_t + b_mth + b_day)
```
- W: full-rank matrix shared across sessions within same month (30-day segments)
- A·B: low-rank (R << C) day-specific correction (prevents overfitting with limited per-session trials)
- b_mth, b_day: month/day-specific biases
- Coupled with batch sampler guaranteeing 16 samples per session per batch

**Convolutional Token Generation:** Two conv layers produce token embeddings E ∈ R^{D×T'}.

**Positional Encodings:** Sinusoidal, encoding both session identity and temporal position.

### Modified Whisper Encoder (6 layers)

1. **Windowed attention (layers 1-3):** Non-causal local attention with window w, replacing global attention. Motivated by finding that early Whisper layers encode phonemes with localized receptive fields. Encodes inductive bias that articulatory dynamics are locally constrained.

2. **Phoneme head (after layer 5):** Linear layer on concatenated pairs of embeddings (one prediction per 80ms). Trained with CTC loss.

3. **Remaining layers (4-6):** Feed Whisper decoder for word-level prediction.

### Whisper Decoder

Original pretrained decoder, fine-tuned with LoRA. Cross-attends to encoder output. Generates word tokens autoregressively. Compatible with HuggingFace generation interfaces.

### Training

**Dual loss:** L = L_CTC + L_CE

**Two-phase training step** (novel):
- Phase 1: Forward pass with time masking → compute CTC loss (masking prevents overfitting)
- Phase 2: Forward pass without masking → compute CE loss (masking destabilizes CE)
- Both phases use same batch, update all weights

**Cross-dataset mode:** Different month/day-specific projections per session (and thus per subject). Batch sampler handles multiple datasets, allocating sessions proportional to dataset size.

### Decoding Paths

1. **High-compute:** Phoneme CTC → WFST + 5-gram rescoring (maximum accuracy)
2. **Low-compute E2E:** Whisper decoder greedy generation (<2GB VRAM, ~50ms inference)

## Data

- **Card et al. (NEJM 2024):** 1 participant (ALS), 256-ch Utah arrays in speech motor cortex, ~6,000+ sentences
- **Willett et al. (Nature 2023):** 1 participant (ALS), 256-ch Utah arrays, ~12,100 sentences
- **Kunz et al. (Cell 2025):** Inner speech dataset
- **Cross-dataset:** ~18,000 training trials total across all three

All use spike counts and spike-band power at 50Hz from intracortical microelectrode arrays.

## Key Results

### Card Dataset

| Model | PER | WER_5gram | WER_E2E |
|-------|-----|-----------|---------|
| RNN baseline | — | 6.7% | 16.0% |
| BIT | — | 5.2% | 12.2% |
| BIT (cross) | — | **4.1%** | 11.1% |
| **BrainWhisperer** | 5.9% | 5.9% | 10.2% |
| **BrainWhisperer (cross)** | 5.2% | 5.2% | **8.7%** |

- E2E WER 8.7% is SOTA — first below 10% threshold for E2E decoding
- Cross-dataset training improved Card performance **without fine-tuning**
- Phoneme-based decoding within ~1pp of BIT

### Ablation Contributions
- Windowed attention, dual loss, and low-rank projections each contribute meaningfully
- Time masking during CTC phase important for preventing overfitting

### Brain-to-Text '25 Challenge
- Extended high-compute pipeline: 4th place / 466 teams
- Low-compute E2E branch: SOTA for end-to-end without heavy LM

## Relevance to Cross-Patient uECOG Speech Decoding

### Key ideas to adopt

1. **Pretrained ASR backbone as phoneme prior.** The core insight: don't learn phoneme representations from neural data — bring them from a model trained on 680k hours of audio. Whisper's early layers are phoneme-selective. This addresses our fundamental data scarcity problem. However, our data is ECoG HGA (not spikes), so the modality gap is larger.

2. **Hierarchical non-stationarity handling.** Month/day-specific projections are analogous to our per-patient read-ins, but with a principled low-rank structure for finer temporal granularity. For our setting: patient-specific = their "month-specific" (major shift), session-specific = their "day-specific" (minor shift). We only have one session per patient, so the hierarchy collapses to just per-patient.

3. **Windowed attention for articulatory locality.** Articulatory dynamics are temporally local (~200ms phoneme duration). Global attention is wasteful and overfits. This validates our use of Conv1d (inherently local) over transformer for the temporal backbone.

4. **Cross-dataset training improves without fine-tuning.** Most important result for us. Training on Card+Willett+Kunz jointly improved Card alone. This is the clearest evidence that cross-subject/dataset neural speech pre-training works — the shared Whisper encoder extracts generalizable phoneme features.

5. **Dual CTC+CE objective.** CTC at intermediate layer for phonemes, CE at final layer for words. The hierarchical loss encourages phoneme-discriminative representations early and linguistic coherence later. We could adapt: CTC/CE at backbone output for phonemes, with auxiliary linguistic loss.

6. **Low-compute E2E path.** <2GB VRAM, ~50ms inference. Demonstrates that heavy LM rescoring isn't necessary — the pretrained decoder provides sufficient linguistic prior. Relevant for deployment considerations.

### What doesn't transfer

1. **Intracortical arrays, not ECoG/uECOG.** Spike counts and spike-band power from Utah arrays. Fundamentally different signal (Poisson spikes vs Gaussian HGA field potentials). The Whisper adaptation may not work as well with the noisier, lower-dimensional HGA signal.

2. **Much more data per participant.** ~6,000-12,000 sentences per participant vs our ~150 trials. Even with Whisper's pretrained knowledge, they still train on orders of magnitude more neural data than we have.

3. **Sentence-level decoding.** Open vocabulary sentences with rich linguistic context. Our task is 3-phoneme non-words — no linguistic prior helps (non-words by definition).

4. **No ECoG validation.** The paper only uses intracortical data. Whether Whisper's phoneme representations transfer to ECoG HGA is untested.

## Critical Assessment

**Strengths:**
- First demonstration that pretrained ASR → neural speech decoding is viable
- Cross-dataset generalization without fine-tuning is genuinely novel
- Practical deployment path (<2GB, 50ms) is a real contribution
- Clean architecture with principled modifications (windowed attention, dual loss)

**Weaknesses:**
- Only 2-3 participants (from publicly available BrainGate datasets)
- All intracortical — no validation on field potentials (ECoG, sEEG, MEG)
- The "cross-dataset" training is still within intracortical MEA modality from similar brain regions
- Code not yet released
- From a startup (Tether Evo) — not peer-reviewed

**For us specifically:** The concept (pretrained phoneme representations from audio) is powerful, but the implementation gap to ECoG HGA is large. A more direct path for us is pre-training on the Flinker lab's 48-patient ECoG dataset (same modality, same signal type) rather than trying to bridge from Whisper's audio representations to HGA.

## Key References

- Card et al. (2024) An accurate and rapidly calibrating speech neuroprosthesis. *NEJM* 391(7):609-618.
- Willett et al. (2023) A high-performance speech neuroprosthesis. *Nature* 620:1031-1036.
- Zhang et al. (2026) BIT: Decoding inner speech with an end-to-end brain-to-text neural interface. *ICLR*.
- Radford et al. (2023) Robust speech recognition via large-scale weak supervision (Whisper). *ICML*.
- Hu et al. (2021) LoRA: Low-rank adaptation of large language models.
- Feghhi et al. (2025) Time-masked transformers with lightweight test-time adaptation for neural speech decoding. *NeurIPS*.
