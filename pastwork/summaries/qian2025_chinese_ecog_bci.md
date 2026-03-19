# Real-time decoding of full-spectrum Chinese using brain-computer interface

**Authors:** Youkun Qian, Changjiang Liu, Peixi Yu, Xingchen Ran, Shurui Li, Qinrong Yang, Yang Liu, Lei Xia, Yijie Wang, Jianxuan Qi, Erda Zhou, Junfeng Lu, Yuanning Li, Tiger H. Tao, Zhitao Zhou, Jinsong Wu

**Venue/Year:** Science Advances, Vol. 11, eadz9968, 5 November 2025

**DOI/URL:** 10.1126/sciadv.adz9968

---

## TL;DR

First real-time Mandarin Chinese speech BCI that decodes the full spectrum of 394 tonal syllables from 256-channel high-density ECoG, achieving 71.2% median offline syllable accuracy using neural signals alone. With a 3-gram language model, the system achieved real-time sentence decoding at 49.7 characters per minute (73.1% character accuracy), demonstrating the viability of syllable-based decoding for tonal monosyllabic languages.

---

## Problem & Motivation

Speech BCI research has been dominated by English phoneme-driven architectures, which exploit multiphonemic redundancy and statistical language models to compensate for decoding errors. Mandarin Chinese poses fundamentally different challenges: it is tonal (pitch variations convey lexical meaning), predominantly monosyllabic, and logographic. A single phoneme decoding error can produce a lexically valid but semantically wrong syllable, and the high homophone density (~13,000 characters mapping to ~3,500 commonly used characters from ~418 base syllables x 4 tones) makes disambiguation critical. Phoneme-level decoding, standard in English BCIs, is therefore poorly suited: Mandarin has only 1-4 phonemes per syllable, so a single phoneme error can completely change the decoded character. The authors argue that tonal syllables are a more appropriate intermediate decoding unit -- more robust than phonemes, semantically informative, and within a manageable set size for direct neural classification.

---

## Method

### Decoding Unit: Tonal Syllables

Rather than decoding phonemes (as in English BCIs), the system decodes full Mandarin tonal syllables directly. An initial set of 418 unique syllables from the Modern Chinese Dictionary was refined to 394 syllables based on participant familiarity and inclusion of two idiosyncratic pronunciations. Each syllable encompasses segmental (consonant + vowel) and tonal information.

### Dual-Stream Decoder Architecture

A dual-stream architecture processes syllable and tone information in parallel:

1. **Syllable stream:** A four-layer stacked bidirectional LSTM network. The first block transforms 250-dimensional input features to 500 dimensions; the second processes these into 200-dimensional features. Each block contains a two-layer bidirectional LSTM followed by Layer Normalization and Dropout. Per-timestep predictions are averaged across the entire input, then passed to a fully connected layer mapping to syllable probability distributions.

2. **Tone stream:** Same architectural design as the syllable decoder but with an independent output layer classifying among 4 Mandarin tone categories. A focal loss function was used during training to handle imbalanced tone proportions.

Both streams take the same input: 1000 ms of high-gamma activity (HGA) from all 256 channels, centered on speech onset (-300 to +700 ms relative to onset).

### Compared Architectures

Three architectures were evaluated: CNN-LSTM, Vision Transformer (ViT), and the Stacked LSTM. The four-layer Stacked LSTM significantly outperformed both alternatives (P < 0.0001, one-way ANOVA with Tukey's post hoc tests).

### Real-Time Pipeline

1. **Preprocessing:** Raw ECoG streamed in 10-ms steps; high-gamma (70-170 Hz) extracted in 50-ms sliding windows; z-scored using fixed normalization parameters from character-reading task data.
2. **Onset detection:** A dedicated model detects speech onset and aligns a 1000-ms neural segment centered on the onset for decoding. For real-time sentence decoding, onset was detected via audio signal (due to time constraints); a neural-based onset detector was also developed and validated separately.
3. **Syllable & tone decoding:** The dual-stream decoder outputs concurrent probability distributions over syllables and tones.
4. **Character mapping:** Syllable-tone combinations are matched to candidate Chinese characters via a predefined syllable-to-character dictionary.
5. **Language model:** A 3-gram language model trained on a Mandarin corpus (daily expressions and command-style phrases). A beam search algorithm integrates neural likelihoods with linguistic probabilities to select the most probable character sequence.

### Training Procedure

- Syllable and tone decoders initially trained on single-character reading task data (10 days).
- Fine-tuned using available sentence-reading task data from the same 10 days.
- All models trained with Adam optimizer, ReduceLROnPlateau learning rate scheduler, and mixup data augmentation.
- Evaluated with 10-fold cross-validation.

---

## Data

### Participant

Single participant: 43-year-old right-handed female undergoing presurgical evaluation for epilepsy at Huashan Hospital, Fudan University (Shanghai, China). IRB approval no. KY2024-842.

### Implant

- **Device:** Neuroxess Co. Ltd. integrated ECoG implant system
- **Electrode array:** 256-channel flexible high-density ECoG array
- **Electrode specs:** 1.3 mm diameter contacts, 3 mm center-to-center interelectrode distance
- **Signal processing:** Four Intan RHD2164 chips; sampling rate 15 kHz
- **Coverage:** Middle temporal gyrus, superior temporal gyrus, ventral sensorimotor cortex (vSMC), and small part of pars opercularis
- **Implantation duration:** 13 days total

### Tasks

1. **Single-character reading task:** 394 distinct tonal syllables, each represented by a common Chinese character. Participant read each character aloud 3 times. Recording schedule: 130 syllables read 60 times each (days 3-7), 264 syllables read 30 times each (days 8-11). Some syllables had additional repetitions beyond baseline counts. ~9 hours of neural data collected over 11 days of intracranial ECoG monitoring. Mean inter-repetition interval: 1.38 +/- 0.53 s; average reading speed: 30.45 +/- 4.60 CPM.

2. **Sentence-reading task:** Sentences of 2-11 characters, read word by word with deliberate pauses. Used for fine-tuning and real-time evaluation.

3. **Real-time sentence decoding task (day 11):** Fixed set of 5 sentences, each repeated once per trial, across 6 trials. Together the reading tasks covered all 394 distinct Mandarin syllables.

### Preoperative fMRI

Block-design language localizer task (4 motor conditions: jaw opening, tongue curling, lip protrusion, vocalizing "yi"/larynx) on a 3T United Imaging uMR790 scanner (voxel size = 2.5 mm^3). Used to localize oromotor cortex and validate electrode placement over articulatory regions.

### Neural Features

High-gamma activity (HGA), 70-150 Hz (offline) or 70-170 Hz (real-time), extracted via Hilbert transform of band-pass filtered signals (Gaussian filter offline; third-order Butterworth filter for real-time). z-scored per channel. All filters causal for real-time processing.

---

## Key Results

### Offline Syllable Decoding (394 syllables, 10-fold CV)

- **Stacked LSTM median syllable accuracy:** 71.2% (99% CI: [70.1, 72.2]); chance level 0.25%
- **CNN-LSTM:** ~62% (significantly lower, P < 0.0001)
- **ViT:** ~63% (significantly lower, P < 0.0001)
- **Tone decoding median accuracy:** 69.1% (99% CI: [66.0, 71.6]); chance level 25%

### Effect of Training Data Volume

- 5 repetitions per syllable: 20.4% (99% CI: [18.0, 22.8])
- 10 repetitions: ~40% (from figure)
- 15 repetitions: ~50% (from figure)
- 20 repetitions: 55.6% (99% CI: [53.8, 61.2])
- Full data (~30-60 repetitions): 71.2%

### Scalability Across Vocabulary Size

- Accuracy remained relatively stable as syllable set increased from 50 to 350 unique syllables, showing only marginal decline -- suggesting the model can handle a large vocabulary.

### Real-Time Sentence Decoding

- **Character accuracy rate (CAR), neural decoding alone:** 61.5% (99% CI: [50.0, 73.1])
- **CAR with 3-gram language model:** 73.1% (99% CI: [61.5, 80.8])
- **Communication rate, neural decoding alone:** 56.7 (99% CI: [56.5, 56.8]) CPM
- **Communication rate with LM:** 49.7 (99% CI: [49.5, 49.7]) CPM
- Note: Speech onset was detected via audio, not neural signal, for the real-time demo.

### Proof-of-Concept BCI Applications

- Robotic arm control: 78.3% CAR, 54.0% command accuracy (1-3 characters per command, full match required)
- Digital avatar: 76.9% CAR
- LLM interaction: 65.4% CAR

### Neural Correlates

- Syllable- and tone-discriminative electrodes were predominantly localized to the vSMC, consistent with prior literature.
- Distinct mean high-gamma signatures observed for different syllables in vSMC.
- Tone-discriminative electrodes also found in vSMC, despite electrode coverage not fully encompassing the laryngeal motor cortex (LMC).
- Task-related activity noted in middle temporal gyrus, possibly reflecting self-listening and orthographic-phonological mappings.
- Electrodes contributing to 90% of decoding performance overlapped substantially with fMRI-identified oromotor cortex (jaw, lips, tongue, larynx).

---

## Relevance to Cross-Patient uECOG Speech Decoding

1. **Tonal language decoding:** This is the first demonstration of full-spectrum tonal syllable decoding from ECoG, directly relevant to any future cross-patient work involving tonal languages (Mandarin, Cantonese, Vietnamese, Thai, etc.). The syllable-as-decoding-unit strategy may generalize better across patients than phoneme-level decoding for these languages.

2. **High-density flexible ECoG:** The 256-channel flexible array (1.3 mm contacts, 3 mm pitch) is denser than standard clinical ECoG grids but coarser than typical uECoG. The interelectrode distance (3 mm) is larger than uECoG arrays (~1 mm pitch), but the approach demonstrates that dense cortical coverage over vSMC is critical for high decoding performance, consistent with uECoG work.

3. **Single-patient limitation and cross-patient discussion:** The authors explicitly acknowledge the single-participant limitation and propose a cross-participant foundational model approach: mapping electrode locations onto MNI152 or atlas-based parcellations, incorporating anatomical information as model parameters, and using transfer learning / fine-tuning for new participants. This aligns directly with cross-patient uECOG decoder goals.

4. **Scalability findings:** The observation that decoding accuracy remains stable as vocabulary scales from 50 to 350 syllables is encouraging for cross-patient settings where the target vocabulary may vary.

5. **Clinical recording context:** Data was collected during epilepsy monitoring (similar to the intra-operative or clinical monitoring context of uECoG studies), providing a realistic assessment of signal quality constraints.

6. **Articulatory feature encoding:** The finding that vSMC electrodes encode articulatory manner and place of articulation for Chinese consonants reinforces the universality of articulatory-kinematic representations across languages, which is foundational for cross-patient decoders that might leverage shared articulatory representations.

---

## Limitations

- **Single participant only** -- generalizability is entirely unknown. The authors acknowledge this as the major limitation.
- **Audio-based onset detection** for real-time sentence decoding (not fully neural-based), though a neural onset detector was developed and validated separately.
- **Electrode coverage did not fully encompass the laryngeal motor cortex (LMC)**, which likely limited tone decoding accuracy (69.1% for 4 tones).
- **Overt speech only** -- no covert/imagined speech decoding attempted.
- **Clinical epilepsy monitoring context** with limited recording time (~11 days, ~9 hours of usable data), restricting the amount of training data per syllable.
- **No cross-session generalization analysis** reported (e.g., training on early days, testing on late days).
- **Language model trained on a constrained corpus** (predefined sentences); open-vocabulary performance is unclear.
- **Real-time sentence decoding used a limited sentence set** (fixed set of 5 sentences across 6 trials on a single day), not open-ended conversation.
- **No comparison with phoneme-level decoding** on the same data to directly validate the claimed superiority of syllable-level units for Mandarin.

---

## Key References

- Willett et al. (2023) -- High-performance speech neuroprosthesis (English, microelectrode arrays); ref 12
- Metzger et al. (2023) -- High-performance neuroprosthesis for speech decoding and avatar control; ref 13
- Card et al. (2024) -- Accurate and rapidly calibrating speech neuroprosthesis (English ECoG); ref 14
- Littlejohn et al. (2025) -- Brain-to-voice streaming neuroprosthesis; ref 15
- Zhang, Wang et al. (2024) -- Brain-to-text framework for decoding natural tonal sentences (Chinese, ECoG); ref 29
- Yu, Zhao et al. -- Decoding and synthesizing tonal language speech from brain activity (Chinese, ECoG); ref 30
- Feng et al. (2025) -- Acoustic inspired brain-to-sentence decoder for logosyllabic language (Chinese, SEEG/ECoG); ref 31
- Wu et al. (2025) -- Towards homogeneous lexical tone decoding from heterogeneous intracranial recordings; ref 32
- Chartier et al. (2018) -- Encoding of articulatory kinematic trajectories in human speech sensorimotor cortex; ref 7
- Bouchard et al. (2013) -- Functional organization of human sensorimotor cortex for speech articulation; ref 27
