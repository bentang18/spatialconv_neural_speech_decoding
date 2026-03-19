# High-Fidelity Neural Speech Reconstruction through an Efficient Acoustic-Linguistic Dual-Pathway Framework

**Citation:** Li, J., Guo, C., Zhang, C., Chang, E.F., & Li, Y. (2025). bioRxiv preprint. https://doi.org/10.1101/2025.09.24.678428

**Code:** https://github.com/CCTN-BCI/Neural2Speech2

---

## Setup

- **Recording modality:** High-density ECoG grids (standard uniform arrays), signals sampled at 3052 Hz, then high-gamma analytic amplitude extracted via 8 Gaussian filters (center frequencies 70–150 Hz), downsampled to 50 Hz for model input
- **Patient count:** 9 monolingual right-handed participants (4 male, 5 female; age 31–55) undergoing clinical neurosurgical evaluation for epilepsy at UCSF
- **Brain region:** Left hemisphere cortex; electrodes placed over the superior temporal gyrus (STG) and surrounding **auditory speech cortex ONLY** — no motor cortex coverage. The near-linear mapping claim is grounded entirely in auditory cortex literature. **No evidence this extends to motor cortex.**
- **Task:** Passive listening — participants listened to naturalistic English sentences from the TIMIT corpus (paper says 499 in Methods, 599 in Results — internal inconsistency) read by 286 male and 116 female speakers; organized into 5 blocks of ~5 minutes each; 0.4 sec silence between sentences
- **Data per participant:** ~20 minutes of labeled neural recordings; z-score normalized **per recording block (~5 min)**, not globally; split 70% train / 20% validation / 10% test
- **Electrode localization:** Chronic monitoring patients: pre-implantation MRI + post-implantation CT. Awake surgery cases: Medtronic neuronavigation + intraoperative photographs. **Some participants may have been intra-operative** (awake surgery), not all chronic epilepsy monitoring.

---

## Problem Statement

Existing neural-to-speech decoding methods face a fundamental trade-off:

- **Acoustic-only** approaches (neural-to-spectrogram regression) capture naturalness, prosody, and speaker identity, but produce low intelligibility when data is limited because the neural-to-acoustic mapping is a high-dimensional regression problem.
- **Linguistic-only** approaches (neural-to-text classification) achieve good word recognition but sacrifice naturalness and paralinguistic features.

Additionally, prior end-to-end methods require hours or days of simultaneous neural and speech recordings per subject, making them impractical in clinical and intraoperative settings. This paper addresses both the acoustic-linguistic trade-off and the data scarcity bottleneck simultaneously.

---

## Method

### Architecture: Acoustic-Linguistic Dual-Pathway

The framework maps ECoG signals into the latent spaces of two pre-trained foundation models in parallel, then fuses the outputs via voice cloning.

#### Acoustic Pathway (naturalness, prosody, speaker identity)

**Stage 1 — Pre-train a speech autoencoder (no neural data required):**
- Encoder: frozen **Wav2Vec2.0** (7 1D-conv layers + 12 transformer blocks; extracts 768-dim features at 50 Hz from 16 kHz audio). **NOT HuBERT** — the paper uses Wav2Vec2.0 exclusively; HuBERT is never mentioned.
- Decoder: HiFi-GAN generator (7 transposed conv layers, upsample rates 2,2,2,2,2,2,5; MRF ResBlock kernel sizes 2,2,3,3,3,10; multi-period discriminator periods 2,3,5,7,11). Trained with Adam lr=1e-5, beta1=0.9, beta2=0.999, max 2000 epochs, 12 days on 6×RTX3090. Loss coefficients: λ_FM=2, λ_Mel=50.
- Trained on LibriSpeech (960 hours) with adversarial loss + mel-spectrogram loss + feature-matching loss
- Defines a brain-aligned acoustic latent space known to correlate with **auditory cortex** activity (refs 22-24 are all auditory pathway papers)

**Stage 2 — Train a lightweight acoustic adaptor (uses neural data):**
- Architecture: 3-layer bidirectional LSTM (~9.4M parameters)
- Input: z-scored high-gamma ECoG snippets (N electrodes × T timepoints at 50 Hz)
- Output: 768-dim Wav2Vec2.0-compatible features fed into the frozen HiFi-GAN
- Loss: mel-spectrogram L1 loss + Laplacian operator loss (L_Lap = ||∇(Φ(x)) − ∇(Φ(x_a))||_1) for phonetic sharpening
- Training: SGD, lr = 3×10⁻³, momentum 0.5, 10% dropout, 500 epochs on one RTX3090 (~12 hours)

#### Linguistic Pathway (word-level intelligibility)

- Architecture: Seq2Seq Transformer adaptor (~10.1M parameters); 3-layer encoder + 3-layer decoder, hidden dim 256, 8 attention heads
- Input: same z-scored high-gamma ECoG, projected via linear layer + positional encoding
- Output: 1024-dim word token sequences (aligned with Parler-TTS token space), decoded autoregressively from a start-of-sequence token
- Loss: KL-divergence on token distributions + Huber loss on sequence length + L2 regularization (weight decay 0.01)
- Synthesized via frozen Parler-TTS (~880M parameters), a natural-language-guided TTS model
- Training: Adam, lr = 10⁻⁴, 500 epochs (~6 hours per participant on one RTX3090)

#### Fusion via Voice Cloning

- Both pathways produce separate speech waveforms: x_a (acoustic fidelity) and x_l (linguistic precision)
- CosyVoice 2.0 performs zero-shot voice cloning using:
  1. A 3-second "rapid cloning" voice reference extracted from the acoustic pathway output
  2. The transcribed word sequence from the linguistic pathway
- CosyVoice 2.0 was fine-tuned on the TIMIT training sentences using a compound loss: STFT-based KL-divergence + MFCC L1 loss
- Final output x_rec preserves both the speaker's vocal identity and the decoded word content

### Training Procedure Summary

| Component | Data required | Training time | Parameters |
|---|---|---|---|
| HiFi-GAN autoencoder (Stage 1) | LibriSpeech (960 hr audio, no neural) | 12 days, 6× RTX3090 | ~pre-trained |
| Acoustic LSTM adaptor (Stage 2) | 20 min neural per subject | ~12 hr, 1× RTX3090 | ~9.4M |
| Linguistic Transformer adaptor | 20 min neural per subject | ~6 hr, 1× RTX3090 | ~10.1M |
| CosyVoice 2.0 fine-tune | TIMIT sentences (text+audio, no neural) | separate pre-training | ~880M |

### Data Efficiency Mechanism

The core insight is that near-linear mappings exist between neural activity and the latent spaces of pre-trained SSL speech models (Wav2Vec2.0) and language models (Parler-TTS). The lightweight adaptors only need to learn this near-linear projection, not a full acoustic synthesis model, which is why 20 minutes suffices. The heavy generative capacity lives entirely in the frozen pre-trained models.

---

## Key Results

### Full Pipeline (dual-pathway + voice cloning)

| Metric | Mean ± SE | Best participant |
|---|---|---|
| Mel-spectrogram R² | 0.824 ± 0.029 | 0.844 ± 0.028 |
| MOS (1–5 scale) | 3.956 ± 0.173 | 4.200 ± 0.119 |
| WER | 18.9% ± 3.3% | 12.1% ± 2.1% |
| PER | 12.0% ± 2.5% | 8.3% ± 1.5% |

- Mel-spectrogram R² is statistically indistinguishable from 0 dB additive noise on original speech (0.771 ± 0.014, p = 0.064)
- WER is statistically indistinguishable from −5 dB additive noise on original speech (14.0 ± 2.1%, p = 0.332)
- MOS improvement of ~37.4% over conventional methods

### Pathway-Specific Ablation

| Condition | Mel-R² | MOS | WER | PER |
|---|---|---|---|---|
| Acoustic pathway alone (Baseline 2) | 0.793 ± 0.016 | 2.878 ± 0.205 | 74.6 ± 5.5% | 28.1 ± 2.2% |
| Linguistic pathway alone (Baseline 3) | 0.627 ± 0.026 | 4.822 ± 0.086 | 17.7 ± 3.2% | 11.0 ± 2.3% |
| MLP regression (Baseline 1) | — | — | 65.4 ± 1.9% | 68.4 ± 1.9% |
| **Full dual-pathway (ours)** | **0.824 ± 0.029** | **3.956 ± 0.173** | **18.9 ± 3.3%** | **12.0 ± 2.5%** |

Key takeaway: WER drops 75% (74.6% → 18.9%) when fusing both pathways vs. acoustic alone, without sacrificing spectral fidelity relative to acoustic-only (R² 0.793 → 0.824, p = 4.486×10⁻³).

**Notable:** Linguistic pathway MOS (4.822 ± 0.086) **exceeds ground truth** MOS (4.234 ± 0.097, p=6.674×10⁻³³). TTS-synthesized speech sounds better than original recordings. Ground truth R² comparison: mel-R² comparable to -5 dB noise (0.864, p=0.161).

### Phoneme Class Clarity (PCC)

- Full model PCC = 2.462 ± 0.201 (significantly above chance level of 1.360, p = 3.424×10⁻⁶)
- Errors are predominantly within-class (vowel confused with vowel, consonant with consonant), not across-class

---

## Cross-Patient Relevance

### What is directly applicable to intra-op uECOG (~20 patients, ~20 min each)

1. **Data budget match is exact.** The paper was designed around the 20-minute constraint arising from intraoperative neurosurgery (explicitly cited as motivation). Each participant's adaptor is trained independently on their own 20 minutes. This is the target regime.

2. **Passive listening paradigm.** No overt speech production is required from patients. The entire training set uses patients passively listening to TIMIT sentences — directly compatible with intraoperative paradigms where patients cannot or should not speak.

3. **Lightweight per-subject adaptors.** The LSTM (~9.4M params) and Transformer (~10.1M params) adaptors are the only components that require neural data. Everything else (Wav2Vec2.0, HiFi-GAN, Parler-TTS, CosyVoice 2.0) is pre-trained on speech corpora only and frozen during neural adaptation. This means the neural-data-dependent compute is small and fast.

4. **STG / auditory cortex coverage ONLY.** The paper uses electrodes over STG and auditory speech cortex — **NOT the same regions as our uECOG** (which targets sensorimotor cortex). The near-linear mapping claim is grounded in auditory cortex literature only. Extension to motor cortex is an untested hypothesis.

5. **SSL latent space as a bridge.** The key mechanism — projecting neural signals into the Wav2Vec2.0 latent space — exploits the empirically established correspondence between auditory cortex activity and SSL speech model representations (refs 22–24 in the paper). This correspondence should hold for uECOG over STG.

6. **No cross-patient generalization tested here, but the architecture supports it.** The paper trains per-subject adaptors. For cross-patient transfer, the frozen SSL backbone provides a common representational space. Projecting multiple subjects' neural data into the same Wav2Vec2.0 latent space is a principled approach to enabling cross-subject pooling or transfer.

### What this enables for cross-patient decoding specifically

- Training one adaptor per patient (subject-specific) on 20 min of passive listening data, then pooling or fine-tuning across patients, is architecturally supported by this framework.
- The shared Wav2Vec2.0 latent space acts as a patient-agnostic anchor — different patients' adaptors are being trained to hit the same target space, which could reduce cross-subject variance.
- Voice cloning means the final synthesis output can be personalized per patient using only a 3-second voice reference, important if the goal is to reconstruct each patient's own voice.

---

## Limitations

1. **Primarily chronic ECoG, not acute uECOG.** Most participants had clinically implanted ECoG grids, though some may have been intra-operative awake surgery cases. Electrode placement followed clinical requirements (seizure focus coverage), not speech-optimized coverage. Intraoperative uECOG grids are transient, higher density, and placed under different anatomical constraints.

2. **Passive listening only.** The system decodes auditory-evoked responses, not speech production or imagined speech. It is a speech reconstruction system from perception, not a speech BCI for communication. Applicability to production-based decoding (e.g., attempted or imagined speech) is not demonstrated.

3. **No cross-patient generalization reported.** Each model is trained and tested on the same participant. No cross-patient transfer learning, domain adaptation, or pooled training is evaluated. The 20-patient cross-patient scenario described in your project is not tested here.

4. **English only.** Stimuli are TIMIT English sentences. Performance in other languages or with non-standard speech is unknown.

5. **Fixed vocabulary / corpus bias.** TIMIT sentences cover a specific phonetic distribution. Generalization to open-vocabulary or conversational speech is not evaluated.

6. **Generative model dependency.** Output quality is bounded by the pre-trained generative models (HiFi-GAN, Parler-TTS, CosyVoice 2.0). Artifacts and synthesis failures in these models propagate into results.

7. **High-gamma band only.** Neural features are high-gamma analytic amplitude (70–150 Hz) at 50 Hz. No evaluation of other frequency bands or raw LFP/spike-based features.

8. **Electrode count variability not reported.** The number of speech-responsive electrodes selected per participant is not explicitly stated in the main text, which matters for cross-patient comparison where electrode counts may differ substantially across uECOG placements.

---

## Reusable Ideas

### 1. SSL latent space projection as the core adaptor target (highest priority)
Train a lightweight per-patient adaptor to project neural signals into the Wav2Vec2.0 latent space rather than directly to spectrograms or phoneme labels. This is the primary reason 20 minutes suffices. The near-linear mapping from auditory cortex activity to Wav2Vec2.0 representations means the regression problem is simpler, and the frozen Wav2Vec2.0 + HiFi-GAN handles the hard acoustic synthesis separately. **For cross-patient decoding: multiple patients' adaptors all target the same latent space, making their outputs directly comparable and poolable.**

### 2. Dual-pathway architecture to decouple acoustic and linguistic objectives
Rather than optimizing a single loss that trades off naturalness vs. intelligibility, train two independent lightweight adaptors — one targeting acoustic features (LSTM → Wav2Vec2.0 space), one targeting linguistic tokens (Transformer → TTS token space). Fuse at inference. This cleanly separates the two objectives and allows each adaptor to be specialized. The fusion via voice cloning is optional and can be replaced with simpler output selection or ensemble depending on use case.

### 3. Laplacian spectral loss for phonetic sharpening
In the acoustic adaptor loss, add a Laplacian operator applied to the mel-spectrogram: L_Lap = ||∇(Φ(x)) − ∇(Φ(x_a))||_1. This is a simple convolutional edge-detection loss on spectrograms, computationally cheap, and demonstrated to improve phonetic distinctiveness. Can be added to any mel-spectrogram regression training.

### 4. Frozen pre-trained generative backbone + lightweight adaptor training
Freeze all large pre-trained models (Wav2Vec2.0, HiFi-GAN, TTS). Only train small adaptors (~9–10M parameters) on neural data. This dramatically reduces overfitting risk with small datasets. The adaptor for 20 minutes of data trains in ~6–12 hours on a single GPU.

### 5. Speech-responsive electrode selection via temporal t-test
Select only electrodes with a statistically significant post-onset response (400–600 ms post-onset > −200 to 0 ms baseline, p < 0.01 Bonferroni-corrected, one-sided t-test). This is a simple, fast, neural-data-only filter that reduces input dimensionality and avoids overfitting on non-speech electrodes. Directly applicable to intra-op uECOG where electrode count may be large.

### 6. Seq2Seq Transformer with KL-divergence token loss for linguistic decoding
For the word/phoneme decoding branch, use a Seq2Seq Transformer with a KL-divergence loss on token distributions (not just cross-entropy on hard labels) plus a Huber loss on predicted sequence length. This is a softer training objective that may generalize better under data scarcity.

### 7. Voice cloning from 3-second reference for patient-specific output
CosyVoice 2.0's "3-second rapid cloning" mode allows patient-specific voice synthesis using only a 3-second clean speech reference. This is directly usable in intraoperative settings where minimal reference audio is available. Fine-tuning CosyVoice 2.0 on domain-specific sentences (as done here with TIMIT) improves articulation clarity.

### 8. Two-stage WER/PER evaluation pipeline
Use Whisper (base) for **word-level** transcription (WER), then a separately fine-tuned **Wav2Vec2.0 + logistic regression + CTC** phoneme decoder for **phoneme-level** recognition (PER). PER comes from a dedicated phoneme recognition system, not Whisper's output. Phoneme recognizer trained with AdamW, lr=1e-4, CTC loss.

### 9. TIMIT corpus for passive listening paradigm design
599 sentences, multiple speakers, broad phonetic coverage, precise phoneme-level timing annotations. Directly usable as a passive listening stimulus set for intraoperative neural recording. 20 minutes of TIMIT covers sufficient phonetic diversity for adaptor training.
