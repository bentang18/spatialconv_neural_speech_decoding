# Neural Decoding of Overt Speech from ECoG Using Vision Transformers and Contrastive Representation Learning

**Authors:** Mohamed Baha Ben Ticha, Xingchen Ran, Guillaume Saldanha, Gael Le Godais, Philemon Roussel, Marc Aubert, Amina Fontanell, Thomas Costecalde, Lucas Struber, Serpil Karakas, Shaomin Zhang, Philippe Kahane, Guillaume Charvet, Stephan Chabardes, Blaise Yvert

**Venue/Year:** Preprint / 2025 (appears to be a journal submission; earlier conference version at IEEE EMBC 2024, ref [33])

**DOI/URL:** Email contact: mohamed-baha.ben-ticha@inserm.fr; Affiliations at Univ. Grenoble Alpes, Inserm, U1216; Zhejiang University; CEA LETI Clinatec; CHU Grenoble Alpes

## TL;DR
An encoder-decoder pipeline using a Vision Transformer (ViT) encoder with CLIP-based contrastive learning directly regresses acoustic features from surface ECoG signals, achieving above-chance speech reconstruction on two participants including the first demonstration of speech decoding from a fully implantable epidural WIMAGINE system. Transfer learning across the two participants (one subdural, one epidural) further improved performance.

## Problem & Motivation
Most high-performing speech BCIs rely on intracortical recordings or classification-based decoding with downstream language models. Surface ECoG offers better long-term stability than intracortical implants but yields lower-resolution signals, making direct regression of continuous speech acoustics more challenging. Prior ECoG speech synthesis work used intermediate articulatory representations rather than direct acoustic regression. The authors aim to build an offline pipeline that directly regresses acoustic coefficients from surface ECoG, optimizing the neural decoder itself rather than relying on language model post-processing. They also seek to validate epidural ECoG (via the chronic, fully implantable WIMAGINE system) for speech decoding -- a modality never previously tested for this purpose.

## Method

### Overall Architecture
Encoder-decoder pipeline operating sentence-by-sentence (offline, not streaming). The encoder maps ECoG-derived neural features to a latent representation; a bidirectional LSTM decoder generates 29-dimensional acoustic feature vectors at each time step.

### Neural Feature Extraction (Preprocessing)
- Common average referencing to remove line noise artifacts
- Spectrograms via FFT with 200 ms Hamming window, 10 ms frame step
- Power spectral density averaged in 10 Hz bins from 0-200 Hz, yielding **20 spectral features** per electrode sampled at 100 Hz
- **1 LFP feature** per electrode: raw signal band-pass filtered between 0.5-5 Hz
- Total: **21 neural features per electrode** per time step

### Decoded Targets
- 25 Mel Cepstral coefficients (Mel)
- 2 aperiodicity coefficients (Ap)
- Fundamental frequency (F0)
- Voicing (UV, binary)
- Total: **29 acoustic features** per time step
- Reconstructed to audio via the WORLD vocoder [44]

### Encoder Architectures (Two Compared)

**CNN Encoder:**
- Input structured as [N_x, N_y, 21] preserving spatial electrode layout (9x8 for P5, 4x8 for CLIN)
- 3 layers of 2D convolutions with kernel sizes 3x3 (P5) and 2x2 (CLIN)
- 128, 64, 32 kernels per layer
- Batch normalization + Leaky ReLU (negative slope 0.2) after each layer
- CNN is participant-specific (different spatial configurations)

**ViT Encoder:**
- Input flattened to vectors of size N_f = N_x x N_y x 21 (1512 for P5, 672 for CLIN)
- MLP embedding: input dim -> 256 -> 176, each projection followed by tanh activation
- Sinusoidal positional encoding summed with embeddings
- Single multi-head attention block: 4 heads, head dimension 32
- MLP projector: 176 -> 64 (GeLU) -> 176
- Only the first projection layer is participant-specific; the rest is shared across participants, enabling transfer learning

### Decoder Architecture
- 3-layer bidirectional LSTM, hidden size 256
- MLP output head: 512 (bi-LSTM output) -> 1024 -> dropout (50%) -> 29
- Produces 29-dimensional acoustic vector at each time step

### Contrastive Learning (CLIP Loss)
- Additional training objective inspired by CLIP [46]
- For each batch of B ECoG sentences, the encoder output is projected via an MLP projector `p` into the acoustic feature space
- N+1 candidates sampled: N-1 negative audio samples (from different sentences in the batch), plus 2 positive samples (the target acoustic vector `a_m` and a noisy version `a_tilde_m`)
- Noisy positive: add Gaussian noise to target, then mean-normalize
- Distance matrix computed via MSE between projected ECoG vectors and candidates
- LogSoftmax applied, CLIP loss computed as negative sum over batch
- Total loss: **Loss = MSE Loss + CLIP Loss**
- MSE Loss trains both encoder and decoder; CLIP Loss trains encoder only
- MLP projector `p` is discarded at inference

### Data Augmentation
- Exploits multiple repetitions of same sentence/vowel sequence
- Each ECoG trial of sentence s mapped to a different audio repetition of that sentence
- Temporal alignment via Dynamic Time Warping (DTW)
- Multiplies effective training set by factor of 4

### Training Details
- Adam optimizer, learning rate 0.0004, no weight decay, no scheduler
- Early stopping on validation MCD with patience of 20
- 10-fold cross-validation: Train/Val/Test split at 72%/18%/10%
  - P5: (457, 114, 63) trials
  - CLIN: (960, 239, 133) trials

### Transfer Learning
- Possible only with ViT (shared architecture except first projection layer)
- Applied to ViT+CLIP+Augmentation setup
- Pre-train on one participant, fine-tune on the other (details inferred from shared weights)

## Data

### Participant P5
- Female epileptic patient, age 38
- **Implant:** 9x8 subdural ECoG grid (PMT) over left hemisphere, covering ventral sensorimotor cortex (vSMC), STG, Broca's area
- Part of Brainspeak clinical trial (NCT02783391)
- **Recording system:** Blackrock Neuroport, 30 kHz sampling rate
- **Corpus:** BY2014 corpus [42] -- French sentences and vowel sequences
  - 634 sentences total: 28 repetitions per vowel sequence, 5 repetitions per complete sentence
- **Dataset:** P5-D1, P5-D2, P5-D3 (3 recording blocks)

### Participant CLIN
- Male tetraplegic patient, age 35
- **Implant:** Epidural WIMAGINE system [36], 64-channel, chronic and fully implantable, wireless
  - Positioned over hand motor cortex and adjacent somatosensory cortex (left hemisphere)
  - NOT over traditional speech areas; motivated by dorsal laryngeal motor cortex (dLMC) proximity to hand motor cortex [41]
- Part of BCI&Tetraplegia clinical trial (NCT02550522)
- **Recording:** 32 most ventral electrodes of the left implant, 585.6 Hz via WIMAGINE, audio synchronized with CED micro1401 at 100 kHz
- **Corpus:** Same BY2014 corpus
  - 1332 sentences total: 63 repetitions per vowel sequence, 8 repetitions per complete sentence
- Recorded across two sessions: 2021 and 2022
- **Dataset:** CLIN-D1-B1 through CLIN-D1-B6, CLIN-D2-B1 through CLIN-D2-B3 (9 blocks)

### Acoustic Contamination
- Evaluated via Mean Diagonal Value (MDV) of cross-correlation matrix between audio spectrograms and neural signals (60-200 Hz)
- Compared against 10000 surrogates
- All blocks passed contamination test at Bonferroni-corrected thresholds (p < 0.05/9 for CLIN, p < 0.05/3 for P5)

## Key Results

All results reported as mean +/- SD across 10 cross-validation folds. Three metrics: Pearson Correlation Coefficient (PCC) between decoded and target acoustic features, Mel-Cepstral Distortion (MCD, lower is better), and F1 score for vowel classification.

### ViT vs. CNN (Table 2, Figure 4)
| Setup | P5 PCC | CLIN PCC | P5 MCD | CLIN MCD | P5 F1 | CLIN F1 |
|---|---|---|---|---|---|---|
| CNN | 0.426 +/- 0.109 | 0.532 +/- 0.084 | 4.362 +/- 0.682 | 3.973 +/- 0.571 | 0.12 | 0.15 |
| ViT | 0.479 +/- 0.114 | 0.546 +/- 0.095 | 4.119 +/- 0.526 | 3.869 +/- 0.642 | 0.16 | 0.20 |

- ViT outperformed CNN on all metrics for both participants (p < 0.01 or p < 0.001, Wilcoxon signed-rank test with Bonferroni correction)

### Adding CLIP (Table 2, Figure 5)
| Setup | P5 PCC | CLIN PCC | P5 MCD | CLIN MCD | P5 F1 | CLIN F1 |
|---|---|---|---|---|---|---|
| CNN+CLIP | 0.482 +/- 0.105 | 0.551 +/- 0.077 | 4.194 +/- 0.639 | 3.996 +/- 0.566 | 0.13 | 0.18 |
| ViT+CLIP | 0.518 +/- 0.100 | 0.557 +/- 0.089 | 3.929 +/- 0.650 | 3.886 +/- 0.636 | 0.14 | 0.20 |

- CLIP improved PCC for both models on both datasets

### Adding Data Augmentation (Table 2, Figure 6)
| Setup | P5 PCC | CLIN PCC | P5 MCD | CLIN MCD | P5 F1 | CLIN F1 |
|---|---|---|---|---|---|---|
| CNN+CLIP+Aug | 0.489 +/- 0.105 | 0.549 +/- 0.095 | 4.166 +/- 0.665 | 3.926 +/- 0.686 | 0.19 | 0.21 |
| ViT+CLIP+Aug | 0.529 +/- 0.105 | 0.563 +/- 0.100 | 3.926 +/- 0.686 | 3.836 +/- 0.686 | 0.29 | 0.26 |

### Best Setup: ViT+CLIP+Aug+Transfer Learning (Table 2, Figure 7)
| Setup | P5 PCC | CLIN PCC | P5 MCD | CLIN MCD | P5 F1 | CLIN F1 |
|---|---|---|---|---|---|---|
| ViT+CLIP+Aug+TL | 0.564 +/- 0.109 | 0.570 +/- 0.100 | 3.777 +/- 0.736 | 3.814 +/- 0.689 | 0.43 | 0.29 |

- Transfer learning yielded the largest single improvement for P5: PCC from 0.529 to 0.564, MCD from 3.926 to 3.777, F1 from 0.29 to 0.43
- Improvements on CLIN were more modest but consistent

### Vowel Decoding (Figure 8)
- Confusion matrices show above-chance vowel classification for both participants
- Performance significantly better than shuffled labels (p < 0.005, Mann-Whitney U test)
- Better vowel discrimination for P5 than CLIN

### Saliency Maps (Figure 10)
- **P5:** Strongest contributions from STG, vSMC; spectral features in low beta (10-20 Hz), gamma and high gamma (60-200 Hz), and slow LFP
- **CLIN:** Strongest contributions from most ventral motor electrodes (closest to dLMC); relatively uniform contribution across 10-200 Hz features, minimal LFP contribution

## Relevance to Cross-Patient uECOG Speech Decoding

This paper is directly relevant in several ways:

1. **Transfer learning across participants with different electrode types and brain coverage:** The ViT architecture enables weight sharing between participants with fundamentally different implant configurations (subdural 9x8 grid over speech areas vs. epidural 4x8 grid over motor cortex). The fact that transfer learning improved decoding for both participants -- despite different electrode modalities (subdural vs. epidural), different brain regions, and different dataset sizes -- is encouraging for cross-patient decoding approaches.

2. **ViT architecture for heterogeneous electrode layouts:** The ViT flattens spatial electrode arrangements into 1D vectors with a participant-specific projection layer, making it naturally adaptable to varying electrode counts and layouts. This is a practical design pattern for cross-patient models where electrode configurations differ.

3. **Contrastive learning for small datasets:** The CLIP-based contrastive loss provides a way to improve encoder representations when training data is limited, which is a common constraint in intraoperative uECOG settings.

4. **Data augmentation via neural variability:** The DTW-based augmentation strategy exploiting repeated trials could be adapted to intraoperative settings where repeated stimuli are presented.

5. **Epidural ECoG for speech:** The demonstration that even epidural recordings over non-speech motor areas can yield above-chance speech decoding suggests that speech-related information may be more widely distributed than assumed, relevant for uECOG arrays placed intraoperatively without precise speech cortex targeting.

6. **Direct acoustic regression without language models:** The pipeline reconstructs speech acoustics directly without any language model, providing a baseline for what neural signals alone can achieve -- important for understanding the true information content in cross-patient settings.

However, **key differences from uECOG** include: this study uses standard clinical ECoG grids (macro-scale, ~1 cm spacing) and an epidural system, not micro-ECoG; the recordings are chronic (not intraoperative); and the datasets are much larger (634-1332 sentences with multiple repetitions) than what is typically available intraoperatively.

## Limitations

- **Offline, sentence-level processing:** The pipeline processes entire sentences at once using a bidirectional LSTM, making it unsuitable for real-time streaming without modification.
- **Small participant count:** Only 2 participants, making it difficult to draw general conclusions about ViT vs. CNN or transfer learning benefits.
- **CLIN electrode placement:** The WIMAGINE implant for CLIN covers hand motor cortex and somatosensory cortex, not traditional speech areas. While above-chance decoding is demonstrated, performance is lower than P5 and the clinical relevance of this placement for speech BCIs is uncertain.
- **No language model integration:** Results reflect raw acoustic regression only. Integrating a language model would likely improve intelligibility substantially but was not tested.
- **Moderate absolute performance:** Even the best PCC values (~0.56-0.57) and F1 scores (0.29-0.43) are modest. The decoded audio quality, while above chance, is not yet at the level needed for practical communication.
- **No comparison to state-of-the-art intracortical systems:** The results are not benchmarked against intracortical speech decoding pipelines that achieve much higher performance.
- **French language only:** All stimuli are from the BY2014 French corpus; generalization to other languages or free speech is not tested.
- **Vowel-level evaluation only for F1:** The F1 metric evaluates only vowel classification, not consonant or phoneme-level accuracy.

## Key References

- [3] Silva et al. 2024 - "The speech neuroprosthesis" (review in Nat. Rev. Neurosci.)
- [11] Anumanchipalli et al. 2019 - Speech synthesis from neural decoding via intermediate articulatory representation (Nature)
- [24] Wairagkar et al. 2025 - Instantaneous voice-synthesis neuroprosthesis (Nature, intracortical)
- [33] Ben Ticha et al. 2024 - Earlier EMBC conference version of this work (ViT for ECoG speech)
- [34] Chen et al. 2024 - Subject-agnostic transformer for surface and depth electrode speech decoding
- [35] Defossez et al. 2023 - CLIP-based decoding of perceived speech from non-invasive recordings
- [36] Mestais et al. 2015 - WIMAGINE wireless 64-channel epidural ECoG system
- [41] Dichter et al. 2018 - Control of vocal pitch in human laryngeal motor cortex
- [44] Morise et al. 2016 - WORLD vocoder
- [46] Radford et al. - CLIP: Learning transferable visual models from natural language supervision
- [50] Komeiji et al. 2024 - Feasibility of decoding covert speech in ECoG with transformer
