# Qian et al. 2025 -- Real-time decoding of full-spectrum Chinese using brain-computer interface

## Citation

Qian, Y., Liu, C., Yu, P., Ran, X., Li, S., Yang, Q., Liu, Y., Xia, L., Wang, Y., Qi, J., Zhou, E., Lu, J., Li, Y., Tao, T. H., Zhou, Z., & Wu, J. (2025). Real-time decoding of full-spectrum Chinese using brain-computer interface. *Science Advances*, 11, eadz9968. https://doi.org/10.1126/sciadv.adz9968

## Setup

- **Patient count:** 1 (43yo female, right-handed, epilepsy presurgical monitoring, Huashan Hospital Shanghai)
- **Recording modality:** 256-ch flexible high-density ECoG (Neuroxess Co. Ltd.), 1.3mm contacts, 3mm pitch, 15kHz sampling. Four Intan RHD2164 chips.
- **Brain region:** vSMC, STG, middle temporal gyrus, small part of pars opercularis
- **Task:** Single-character reading (394 tonal syllables, 3 reps each, 30-60 reps/syllable over 11 days) + sentence reading + real-time sentence decoding (day 11, 5 fixed sentences x 6 trials)
- **Data amount:** ~9h over 11 days of intracranial monitoring. Mean inter-repetition interval 1.38s.

## Architecture

Dual-stream stacked biLSTM. Syllable stream (394-class) and tone stream (4-class) in parallel, same input.

Each stream: 4-layer stacked biLSTM organized as 2 sub-blocks.
- Block 1: 2-layer biLSTM (250 -> 500 hidden) + LayerNorm + Dropout
- Block 2: 2-layer biLSTM (500 -> 200 hidden) + LayerNorm + Dropout
- Temporal mean pooling across full input window -> FC to class logits

Tone stream uses focal loss for class imbalance. Both streams trained with Adam, ReduceLROnPlateau, mixup augmentation, 10-fold CV.

Compared architectures: CNN-LSTM ~62%, ViT ~62%, stacked LSTM 71.2% (P < 0.0001 ANOVA).

Sentence pipeline: onset detection -> syllable+tone parallel decode -> syllable-to-character dictionary -> 3-gram LM beam search.

Real-time: causal 3rd-order Butterworth, 50ms sliding window, audio-based onset detection (neural onset detector validated separately).

## SSL/Pretraining

None. Purely supervised, no SSL, no pretraining. Train on single-character reading task, fine-tune on sentence reading data. Authors explicitly propose foundation model as future work.

## Cross-Patient

None. Single patient only (N=1). Authors acknowledge as major limitation. Propose cross-patient foundation model mapping to MNI152/atlas parcellations as future direction.

## I/O Features

| Component | Detail |
|-----------|--------|
| **Input signal** | HGA 70-170Hz (real-time) / 70-150Hz (offline) |
| **Preprocessing** | 15kHz -> reject bad channels -> CAR -> 400Hz downsample -> 50Hz notch -> Gaussian bandpass (offline) or Butterworth (real-time) -> Hilbert envelope -> z-score per channel |
| **Input window** | 1000ms (-300 to +700ms relative to speech onset) |
| **Channels** | All 256, no selection |
| **Output (syllable)** | 394-class softmax (all Mandarin tonal syllables) |
| **Output (tone)** | 4-class softmax (4 Mandarin lexical tones) |
| **Loss** | CE (syllable) + focal CE (tone) |
| **Decoding** | Temporal mean pool -> classification (not CTC/AR) |

## Key Results

| Metric | Value | Condition |
|--------|-------|-----------|
| Offline syllable accuracy | 71.2% (99% CI [70.1, 72.2]) | 394-class, chance 0.25%, 10-fold CV |
| Offline tone accuracy | 69.1% (99% CI [66.0, 71.6]) | 4-class, chance 25% |
| Real-time CAR (neural only) | 61.5% (99% CI [50.0, 73.1]) | Sentence decoding, day 11 |
| Real-time CAR (+3-gram LM) | 73.1% (99% CI [61.5, 80.8]) | Sentence decoding, day 11 |
| Communication rate (+LM) | 49.7 CPM | Sentence decoding |
| Data scaling | 20.4% at 5 reps, ~40% at 10, ~50% at 15, 55.6% at 20, 71.2% at 30-60 | Steep 5-20, plateau 20+ |
| Vocab scaling | Flat: 394 syllables barely degrades from 50 | Syllable set size sweep |

## v12 Comparison

| Dimension | Qian 2025 | v12 |
|-----------|-----------|-----|
| **Patients** | 1 | 4 core + 7 extended (+ sEEG for SSL) |
| **Hardware** | 256-ch flexible ECoG, 3mm pitch | 128/256-ch uECoG, ~1mm pitch |
| **Signal** | HGA 70-170Hz | HGA 70-150Hz |
| **Spatial alignment** | N/A (single patient) | VE cross-attention + Brainnetome atlas |
| **Per-patient params** | N/A | 134 (diagonal norm + delta/omega) |
| **Architecture** | Dual-stream 4L biLSTM | Conv1d -> VE cross-attn -> factored self-attn -> AR decoder |
| **Output** | 394 syllables + 4 tones (classification) | 9 phonemes (AR sequence, 52 valid tokens) |
| **SSL** | None | Temporal span masking (BIT-style) |
| **Cross-patient** | None | Core design goal |
| **Data** | ~9h, 1 patient | 7.6h uECoG + 16.7h sEEG (29 patients) |

Key validations for v12: (1) HGA from dense surface ECoG sufficient for high-accuracy decoding even at 394 classes. (2) vSMC dominates articulatory encoding (fMRI validated). (3) Steep scaling 5-20 reps, plateau after -- relevant to our 46-178 trials/patient regime. (4) Foundation model direction explicitly proposed as future work -- v12 fills this gap. (5) N=1 limitation is exactly what v12 addresses.

## Regime Table

| Dimension | Value |
|-----------|-------|
| Patients | 1 |
| Channels | 256 |
| Contact size | 1.3mm |
| Pitch | 3mm |
| Signal | HGA 70-170Hz |
| Sampling | 15kHz -> 400Hz |
| Classes | 394 syllables + 4 tones |
| Data volume | ~9h, 11 days |
| Reps/class | 30-60 |
| Architecture | 4L stacked biLSTM |
| Training | Supervised, Adam, mixup, 10-fold CV |
| Eval metric | Accuracy (classification, not CTC/PER) |
| Best offline | 71.2% syllable, 69.1% tone |
| Best real-time | 73.1% CAR (+LM), 49.7 CPM |
| Cross-patient | None |
| SSL | None |
