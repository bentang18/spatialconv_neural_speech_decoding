# A Neural Speech Decoding Framework Leveraging Deep Learning and Speech Synthesis

**Authors:** Xupeng Chen*, Ran Wang*, Amirhossein Khalilian-Gourtani, Leyao Yu, Patricia Dugan, Daniel Friedman, Werner Doyle, Orrin Devinsky, Yao Wang†, Adeen Flinker†
**Venue/Year:** Nature Machine Intelligence, Volume 6, April 2024, pp. 467–480
**DOI:** https://doi.org/10.1038/s42256-024-00824-8
**Code:** https://github.com/flinkerlab/neural_speech_decoding
**Data:** https://data.mendeley.com/datasets/fp4bv9gtwk/2

## TL;DR

ECoG-to-speech framework that decodes neural signals into 18 interpretable speech parameters (pitch, formants, voicing, loudness) via a neural decoder, then synthesizes naturalistic spectrograms via a fully differentiable vocoder-inspired speech synthesizer. Tested on N=48 neurosurgical participants with subdural ECoG grids. Best architecture (3D ResNet): mean PCC 0.806 (non-causal) / 0.797 (causal) between decoded and ground-truth spectrograms. Key contributions: (1) compact interpretable intermediate representation, (2) causal decoding validated, (3) right-hemisphere decoding comparable to left, (4) low-density grids sufficient.

## Problem & Motivation

Prior ECoG speech decoding approaches either:
- Map neural signals directly to spectrograms (high-dimensional regression, hard to train with limited data, outputs not interpretable)
- Map to articulatory/kinematic space then synthesize via GAN (non-natural sounding, kinematic representation loses speaker-specific acoustic characteristics)
- Use HuBERT quantized tokens as intermediate (speaker-independent — cannot reconstruct patient's own voice; requires separate speaker model)

The authors propose a compact set of 18 physically interpretable speech parameters as the intermediate representation, paired with a differentiable synthesizer that can backpropagate spectral reconstruction loss through the entire pipeline. This keeps the regression problem low-dimensional while preserving speaker-specific acoustics.

## Method

### Two-Stage Training

**Stage 1 — Speech auto-encoder (no neural data):**
- Speech encoder: temporal conv layers + channel MLPs extract 18 speech parameters per time step from spectrograms
- Speech synthesizer: differentiable vocoder combining voiced (harmonic excitation through formant filters) and unvoiced (white noise through broadband filter) pathways, mixed by a learned voice weight
- Trained on each participant's speech audio only (semi-supervised: MSS spectral loss + STOI+ intelligibility loss + Praat-supervised pitch/formant loss)
- Speaker-specific prototype formant filter shapes are learned during this stage (M varies by config: 20 for Stage 1/female, 80 for male e2a; default 40; N=6 formant filters + 1 broadband)
- Total learnable synthesizer parameters: 834 (female) / 1090 (male), all speaker-dependent but time-independent
- **Gender-dependent differences:** N_FFT=256 (female) vs 512 (male); formant freq limits differ (e.g., F1 upper: 950 Hz female vs 850 Hz male); F0 range: 88–420 Hz; spectrogram channels: 64 (from production config)

**Stage 2 — ECoG decoder (uses neural data):**
- Input: high-gamma ECoG (70–150 Hz, Hilbert envelope, 125 Hz, z-scored, CAR)
- Output: 18 speech parameters per time step
- Loss: MSS spectral loss + STOI+ loss + reference loss (L2 on speech parameters from Stage 1 encoder) + Praat supervision on pitch/formants
- Three decoder architectures tested:

### ECoG Decoder Architectures

| Architecture | Design | PCC (non-causal) | PCC (causal) |
|---|---|---|---|
| **3D ResNet** | Temporal conv → 4 residual blocks (channels 16→32→64→128→256, **InstanceNorm** not GroupNorm, LeakyReLU(0.2), dropout 0.2) → spatial max pool → transposed conv upsample (256→128→64→32) → temporal conv + channel MLP. Electrodes reshaped to sqrt(N)×sqrt(N) grid (8×8 for 64 LD electrodes) | **0.806** | **0.797** |
| **3D Swin Transformer** | 3D patch embedding → shifted 3D window self-attention (2×2×2 patches) → patch merging → transposed conv → temporal conv + channel MLP | 0.792 | 0.798 |
| **LSTM** | 3-layer LSTM + linear layer to parameters | 0.745 | 0.712 |

- All architectures support both causal (past+current only) and non-causal (past+current+future) temporal operations
- Per-participant models (no cross-patient training). **However:** codebase includes a gradient reversal speaker classifier (weight=0.0005) and separate train/test subject args — infrastructure for cross-patient adversarial training exists but is not used in the paper
- 350 train / 50 test trials per participant (50 words × 5 tasks, word-level 5-fold CV: 10 held-out words/fold)

### Training Hyperparameters (from code, not reported in paper)
- **Optimizer:** LREQAdam (custom Adam with learning-rate equalization from StyleGAN)
- **LR:** 0.002; **beta1=0.0** (momentum-free, unusual), beta2=0.99
- **Batch size:** 32; **Epochs:** 60 (both stages)
- **EMA** on model weights for evaluation
- **Loss scaling** (Stage 2) far more complex than paper suggests: loudness MSE×150, F0 MSE×0.3 (weighted by onset×loudness), formant freq MSE×300, formant amp MSE×400, noise freq MSE×12000, linear+mel spectrogram L1×80, STOI+×10, plus smoothness regularization, explosive consonant bias, formant ordering penalty

### The 18 Speech Parameters

Per time step: pitch f₀, loudness L, voice weight a, 6 formant centre frequencies (f₁–f₆), 6 formant amplitudes (a₁–a₆), broadband unvoice filter centre frequency f₀ᵘ, bandwidth b₀ᵘ, and amplitude a₀ᵘ.

### Differentiable Speech Synthesizer

Inspired by classical source-filter vocoder but fully differentiable:
- **Voiced pathway:** harmonic excitation (**K=40** harmonics at f₀, from code; paper/summary previously said K=80) → voice filter (sum of N=6 formant filters, each a learned prototype shape scaled by amplitude/bandwidth/centre frequency). Harmonic bandwidths are **deterministic**: `bw = 0.65 * (0.00625 * relu(freq) + 375)`, not free parameters
- **Unvoiced pathway:** Gaussian white noise → broadband filter (same parametric form as formant filters)
- **Mixing:** S(f) = a·V(f) + (1−a)·U(f), weighted by voice weight a ∈ [0,1]
- **Output:** S'(f) = L·S(f) + B(f), where B(f) is stationary background noise (K frequency bins, learned per speaker)

Key advantage over DDSP: only 18 parameters/timestep vs. 80 harmonic amplitudes, and the representation is disentangled (formant filters are independent of pitch).

## Data

- **N = 48** native English-speaking neurosurgical participants (26 female, 22 male) with refractory epilepsy at NYU Langone
- **Recording:** Subdural ECoG grids over perisylvian cortex (STG, IFG, pre-central, post-central gyri)
- **Grid types:** 43 participants with low-density (LD) standard 8×8 macro grids (1 cm spacing); 5 participants with hybrid-density (HB) grids (64 LD + 64 interleaved micro contacts at 1 mm, total 128 electrodes)
- **Hemisphere:** 32 left, 16 right (placement by clinical need)
- **Tasks:** 5 speech tasks (auditory repetition, auditory naming, sentence completion, visual reading, picture naming), each with 50 unique target words, 400 total trials per participant
- **Signal:** ECoG at **3051.76 Hz** (not 2048 as previously stated) → Hilbert HGA (70–150 Hz) → downsample 125 Hz → CAR → z-score normalize → **Savitzky-Golay filter** (3rd-order polynomial, window 11)
- **Preprocessing:** Reject channels with artefacts, line noise, poor contact, epileptiform activity; apply spectral gating for noisy speech recordings; verified no acoustic contamination per Roussel et al. (2020) procedure
- **Speech duration:** ~500 ms average per trial (~3.3 min total speech per patient)
- **Data augmentation:** Temporal jitter ±64 frames; training samples repeated 128×

## Key Results

### Spectrogram Reconstruction

| Condition | Mean PCC (N=48) | Median PCC |
|---|---|---|
| ResNet non-causal | 0.806 | 0.805 |
| ResNet causal | 0.797 | — |
| Swin non-causal | 0.792 | — |
| Swin causal | 0.798 | — |
| LSTM non-causal | 0.745 | — |
| LSTM causal | 0.712 | — |

- PCC range across 48 participants: 0.62–0.92
- 54% of participants had PCC > 0.80
- Causal vs. non-causal: no significant difference for ResNet (p=0.093) or Swin (p=0.196); significant for LSTM (p=0.009)
- Word-level cross-validation (10 held-out words never seen during training): PCC comparable across 5 folds, confirming generalization to unseen words

### Speech Parameter Reconstruction (ResNet causal)

| Parameter | Mean PCC |
|---|---|
| Voice weight (a) | 0.781 |
| Loudness (L) | 0.571 |
| Pitch (f₀) | 0.889 |
| Formant 1 (f₁) | 0.812 |
| Formant 2 (f₂) | 0.883 |

### Left vs. Right Hemisphere

| Model | Left PCC (N=32) | Right PCC (N=16) | p-value |
|---|---|---|---|
| ResNet | 0.806 | 0.790 | 0.623 |
| Swin | 0.795 | 0.798 | 0.968 |

No significant difference. Right-hemisphere decoding is viable — important for patients with left-hemisphere damage.

### Electrode Density

HB vs. LD grids in the 5 hybrid participants: no significant PCC difference for 4/5 participants. Low-density clinical grids are sufficient for the speech parameter intermediate representation.

### Contribution Analysis (Occlusion)

- **Causal models:** dominant contribution from ventral sensorimotor cortex (pre/post-central gyri) bilaterally
- **Non-causal models:** additional strong contribution from STG (auditory feedback), which causal models correctly exclude
- This validates the causal architecture — it avoids reliance on auditory feedback signals unavailable in real-time BCI

## Relevance to Cross-Patient uECOG Speech Decoding

### Directly applicable

1. **Interpretable intermediate representation.** The 18-parameter speech space is compact, physically grounded, and disentangled. For cross-patient alignment, aligning in this 18-dim space is far simpler than aligning in spectrogram space. Formant frequencies and pitch are inherently more cross-patient-invariant than raw spectral features.

2. **Causal decoding validated.** Causal ResNet/Swin perform on par with non-causal (PCC 0.797 vs 0.806, NS). This is necessary for real-time BCI and means the decoder doesn't depend on auditory feedback signals — important because auditory feedback timing varies across patients.

3. **Low-density grids sufficient.** HB vs. LD grids show no significant PCC difference, suggesting that uECOG's high spatial resolution may not be the primary driver of performance in this framework. The speech parameter representation is robust to electrode density.

4. **Right hemisphere works.** No significant left/right difference. Relevant if some uECOG patients have right-hemisphere coverage.

5. **Large cohort (N=48) validates reproducibility.** Unlike most speech BCI papers (N=1–3), this demonstrates the pipeline works across a heterogeneous population with clinically-determined electrode placements.

6. **Speaker-specific synthesizer from speech audio only.** The synthesizer's speaker-specific parameters (formant filter prototypes, background noise) are learned from the patient's own speech recordings — no neural data required. For intra-op uECOG, the patient's pre-operative speech recordings could be used to pre-train a personalized synthesizer before the surgery.

### What doesn't transfer

1. **No cross-patient training or transfer.** Every model is per-participant. No shared backbone, no cross-patient pooling, no domain adaptation. The 18-parameter intermediate space is suggestive of cross-patient alignment potential, but this is not tested.

2. **Subdural ECoG, not uECOG.** Standard clinical grids (1 cm spacing) and hybrid grids. Not micro-ECoG arrays (200–400 μm spacing). The spatial resolution and cortical sampling are fundamentally different.

3. **Speech production tasks (not repetition of phonemes).** Five tasks all involve producing 50 real English words. Our data involves phoneme repetition and word/nonword repetition — different task structure and potentially different neural representations.

4. **No CTC or sequence-level loss.** The decoder predicts speech parameters frame-by-frame with MSE-type losses. No alignment-free sequence loss (CTC) that would handle variable timing across patients.

## Limitations

1. **Per-participant models only.** No cross-patient evaluation despite N=48 providing ample opportunity
2. **Word-level vocabulary (50 words).** Not sentence-level or open-vocabulary decoding
3. **Intra-operative feasibility not tested.** All participants are chronic epilepsy monitoring patients with days of recording time; not tested under intra-op time constraints
4. **No phoneme-level evaluation.** PCC on spectrograms is the primary metric; no phoneme classification accuracy or PER reported
5. **Formant-based synthesizer assumes vocal tract model.** May not generalize well to non-speech sounds, severely dysarthric speech, or languages with very different phonological inventories
6. **Grid-based architecture assumption.** 3D ResNet/Swin treat electrodes as a spatial grid. Non-grid electrode layouts (sEEG, strips) would require architectural changes
7. **No temporal alignment across patients.** Speech onset alignment is per-trial; no mechanism for handling variable response latencies across patients

## Reusable Ideas

### 1. Speech parameter intermediate representation (high priority for us)
The 18-parameter speech space (pitch, formants, voicing, loudness) is a principled cross-patient-invariant target. Unlike spectrogram regression, formant frequencies have physical meaning that should be preserved across patients despite electrode placement differences. Could serve as an alignment target alongside or instead of articulatory features. Lower-dimensional than HuBERT embeddings (18 vs. 768) and more interpretable.

### 2. Differentiable vocoder-based synthesizer for end-to-end training
The speech synthesizer is fully differentiable, allowing spectral reconstruction loss to flow back through the synthesizer into the decoder. This means the decoder is trained to produce parameters that actually sound right, not just match reference values. The synthesizer architecture (source-filter with soft voiced/unvoiced mixing) is elegant and could be adopted for our pipeline's output stage.

### 3. Causal 3D convolution architecture
The 3D ResNet with causal temporal convolution is a strong baseline for grid-structured neural data. For uECOG arrays (8×16, 12×22), this is directly applicable — treat the grid as a 2D spatial field with a temporal dimension. The causal constraint ensures no future information leakage.

### 4. STOI+ loss for speech intelligibility optimization
Beyond spectral MSE, adding STOI+ (Short-Time Objective Intelligibility) as a differentiable loss improved intelligibility. STOI+ operates on one-third octave band envelopes and is a standard speech quality metric. Easy to add to any spectrogram-based training pipeline.

### 5. Speaker-specific synthesizer pre-training from audio only
Train the speech encoder + synthesizer auto-encoder on each patient's speech recordings before touching neural data. This gives: (a) reference speech parameters for supervised decoder training, (b) a speaker-specific synthesizer ready to convert decoded parameters to spectrograms. For intra-op, pre-operative speech recordings could be used.

### 6. Occlusion-based contribution analysis
Zero out individual electrodes at test time and measure PCC drop. Simple, model-agnostic, and provides interpretable spatial contribution maps. Project onto MNI coordinates for cross-patient comparison. Useful for validating that the decoder is using sensorimotor (not just auditory feedback) channels.

## Key References

- Angrick, M. et al. (2019) Speech synthesis from ECoG using densely connected 3D convolutional neural networks. *J. Neural Eng.* 16, 036019.
- Willett, F.R. et al. (2023) A high-performance speech neuroprosthesis. *Nature* 620, 1031–1036.
- Metzger, S.L. et al. (2023) A high-performance speech neuroprosthesis for speech decoding and avatar control. *Nature* 620, 1037–1046.
- Engel, J. et al. (2020) DDSP: Differentiable digital signal processing. *ICLR*.
- Wang, R. et al. (2023) Distributed feedforward and feedback cortical processing supports human speech production. *PNAS* 120, e2300255120.
- Roussel, P. et al. (2020) Observation and assessment of acoustic contamination of electrophysiological brain signals during speech production and perception. *J. Neural Eng.* 17, 056028.
