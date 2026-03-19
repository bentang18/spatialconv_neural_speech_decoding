# Littlejohn et al. 2025 - Streaming Brain-to-Voice Neuroprosthesis

## Citation

Littlejohn, K. T., Cho, C. J., Liu, J. R., Silva, A. B., Yu, B., Anderson, V. R., Kurtz-Miott, C. M., Brosler, S., Kashyap, A. P., Hallinan, I. P., Shah, A., Tu-Chan, A., Ganguly, K., Moses, D. A., Chang, E. F., & Anumanchipalli, G. K. (2025). A streaming brain-to-voice neuroprosthesis to restore naturalistic communication. *Nature Neuroscience*, 28, 902--912. https://doi.org/10.1038/s41593-025-01905-6

## Setup (modality, patients, brain region, task, data amount)

- **Modality:** High-density surface ECoG (253-channel array, 3 mm center-to-center spacing, Blackrock Microsystems). Also tested generalization to single-unit MEA recordings and surface EMG of vocal tract.
- **Patients:** 1 clinical trial participant (47-year-old female) with severe paralysis and anarthria from brainstem stroke (right pontine stroke, basilar artery occlusion). Diagnosed with quadriplegia and anarthria. Speech intelligibility ~5% for words, 0% for phrases. Implanted September 2022. Part of the BRAVO clinical trial (NCT03698149).
- **Brain region:** Left hemisphere speech sensorimotor cortex -- predominantly precentral and postcentral gyri, plus a portion of STG.
- **Task:** Silent-speech (mimed/mouthed) sentence production. Participant silently attempted to articulate target sentences with short syllable-length pauses between words. No audible vocalization at any point during training or inference.
- **Data amount:**
  - 50-phrase-AAC set: 11,700 trials over 58 weeks (50 unique sentences, 119 unique words).
  - 1,024-word-General set: 23,378 trials (12,379 unique sentences, 1,024 unique words) over 65 weeks.
  - Each sentence seen on average 6.94 times during training.
- **Neural features:** High gamma activity (HGA; 70--150 Hz) and low-frequency signals (LFS; 0.3--17 Hz), common-average referenced, z-scored with a 30-s sliding window, sampled at 200 Hz.

## Method (preprocessing, decoder architecture, training)

### Preprocessing
- Extract HGA and LFS from 253-channel ECoG at 200 Hz sampling rate.
- Common average reference, then 30-s sliding window z-score normalization per channel.
- Neural features processed in 80-ms chunks (effectively downsampled to 12.5 Hz after the neural encoder CNN).

### Decoder Architecture: RNN-Transducer (RNN-T)
The core architecture is a **bimodal RNN-Transducer** that jointly decodes both speech (acoustic units) and text from neural activity in a streaming fashion. Three main components:

1. **Neural encoder (shared):** 1D convolutional network (two layers, 512 kernels, kernel size 7, stride 4) followed by an RNN (3 layers of unidirectional GRU, 512 hidden units, dropout 0.5). Downsamples signal by 16x to one prediction per 80 ms. Unidirectional (causal) for streaming.

2. **Language models (separate for each modality):**
   - Speech LM: 4-layer LSTM, 512 hidden units, dropout 0.3, layer normalization. Pretrained on 960 h of LibriSpeech audio (HuBERT acoustic-speech units). Frozen during neural training.
   - Text LM: Same architecture. Pretrained on LibriSpeech text (byte-pair encoded, 4,096 subword tokens). Frozen during neural training.

3. **Joiners (separate for each modality):** Combine encoder and LM outputs via addition, nonlinear activation (tanh), then linear projection to output distribution. Speech: 101 classes (100 HuBERT acoustic-speech units + 1 blank). Text: 4,097 classes (4,096 subword tokens + 1 blank).

### Decoding
- RNN-T beam search at each 80-ms step, keeping top K hypotheses per modality.
- Speech: most likely hypothesis selected at each step (audio chunks cannot be replayed). Acoustic-speech units from a buffer of 4 units are synthesized every 80 ms.
- Text: entire buffer converted to text and displayed on screen.

### Speech Synthesizer
- **HiFi-CAR:** custom generative adversarial network (acoustic-speech unit vocoder) that converts predicted HuBERT units into speech waveforms in 80-ms increments.
- Trained on LJSpeech dataset, then voice-converted using **YourTTS** conditioned on a short pre-injury voice clip of the participant to produce personalized audio.
- A duration predictor models unit durations to match the participant's speaking rate.

### Training Targets
- Acoustic-speech units extracted from reference waveforms using **HuBERT** (self-supervised speech representation model, K-means 100 clusters on 6th layer).
- Text targets: byte-pair encoded (SentencePiece tokenizer, 4,096 subword vocabulary).
- Reference audio generated via TTS (since participant cannot speak) -- no audible vocalization required at any point.
- Training uses the **RNN-T loss**, which jointly models emission probability and alignment between input neural data and target sequences without requiring explicit alignment.

### Speech Detection
- Separate model: 3-layer unidirectional LSTM (128, 96, 32 hidden units) predicting silence/speech/preparation labels from HGA + LFS features.
- Time-thresholded binary output (minimum 500-ms duration) for onset/offset detection.
- Used for latency calculations; not used during online streaming inference (the RNN-T model has implicit speech detection via blank token emission).

## Key Results (quantitative)

### Decoding Speed
- **47.5 WPM** median (99% CI 45.2, 49.3) for 1,024-word-General; **90.9 WPM** (99% CI 88.4, 95.4) for 50-phrase-AAC.
- Significantly faster than previous delayed approaches (28.3 WPM for prior work on same participant).

### Latency (streaming, from detected speech attempt onset to decoded output onset)
- 1,024-word-General: **1.12 s** speech synthesis, **1.01 s** text decoding (median).
- 50-phrase-AAC: **2.14 s** speech synthesis, **2.35 s** text decoding.
- Per-chunk inference latency: median **11.83 ms** per 80-ms chunk (well under real-time). 99.3% of inference steps completed within the 80-ms budget.

### Error Rates (speech synthesis, 1,024-word-General)
- **PER:** 45.3% (99% CI 40.7, 64.4)
- **WER:** 58.8% (99% CI 50.5, 76.0)
- **CER:** 44.7%

### Error Rates (text decoding, 1,024-word-General)
- **PER:** 23.9%
- **WER:** 31.9%
- **CER:** 22.8%

### Error Rates (speech synthesis, 50-phrase-AAC)
- **PER:** 10.8% (99% CI 4.21, 15.9)
- **WER:** 12.3% (99% CI 5.26, 23.1)
- **CER:** 11.2% (99% CI 4.67, 15.0)

### Error Rates (text decoding, 50-phrase-AAC)
- **PER:** 7.58%
- **WER:** 10.3%
- **CER:** 7.23%

### Unseen Word Generalization
- 46.0% median spectrogram classification accuracy on 24 unseen words (chance = 3.85%, P < 0.001). Decoded waveforms clustered by place of articulation.

### Long-Form Continuous Decoding (offline, 4 blocks of ~5.9 min each)
- Implicit speech detection: only 3 false positives out of 100 segments with no attempted speech. Robust over 16 minutes of rest data (zero false detections).
- Speech PER: 49.4%, WER: 65.0%, CER: 49.3%.
- Text PER: 34.7%, WER: 44.2%, CER: 34.1%.
- Significant improvements in onset latency and WPM vs. delayed approach.

### Cross-Modality Generalization (all streaming RNN-T)
- **ECoG** (same participant, prior dataset): PER 49.0%, WER 63.8%, CER 51.3%.
- **MEA** (different participant with paralysis, intracortical): PER 54.5%, WER 79.7%, CER 55.0%.
- **EMG** (healthy speaker, vocal tract surface electrodes): PER 44.8%, WER 73.0%, CER 44.3%.
- All significantly above chance; streaming RNN-T outperformed delayed CTC baseline on all modalities.

### Auditory Feedback Robustness
- Electrode contribution maps nearly identical with and without auditory feedback (Pearson r = 0.79, P < 0.0001).
- No significant difference in WER between feedback conditions (P = 0.95 for speech, P = 0.172 for text).
- Strongest contributions from electrodes along the central sulcus and precentral gyrus (articulatory motor cortex), not auditory areas.

## Cross-Patient Relevance

- **Architecture generalizes across recording modalities:** The same RNN-T framework worked on ECoG, MEA (intracortical single-unit), and EMG with only the feature extraction front-end changed. This is directly relevant to cross-patient decoding since different patients may have different electrode types/configurations.
- **No vocalization required at any stage:** Training targets are generated from TTS, not from the participant's own speech. This is critical for patients who cannot vocalize, and removes a major barrier to cross-patient transfer (no need for patient-specific acoustic targets).
- **HuBERT acoustic-speech units as a shared target space:** Using self-supervised speech representations (100 discrete units from HuBERT) as intermediate targets rather than raw spectrograms/waveforms creates a more compact, discrete target space that could potentially be shared across patients.
- **Pretrained and frozen language models:** The speech and text LMs are pretrained on large external corpora (LibriSpeech) and frozen. Only the neural encoder and joiners are trained on patient data. This separation means the LM component is already cross-patient by design.
- **Shared neural encoder across modalities:** A single neural encoder feeds both speech and text decoders, suggesting the encoder learns a modality-agnostic neural representation that could potentially transfer across patients.
- **Limitation:** Online demonstrations were conducted with only a single participant. The paper explicitly notes that generalizability to other participants was shown only offline and across modalities, not across patients with the same modality.

## Reusable Ideas

1. **RNN-T for streaming neural decoding:** The RNN-T framework enables true streaming (no need to buffer an entire utterance). The blank token mechanism provides implicit speech detection for free, eliminating the need for a separate VAD model during inference. This is a major architectural insight applicable to any real-time BCI.

2. **Bimodal joint training (speech + text):** Training a single encoder to predict both acoustic units and text simultaneously provides complementary supervision signals and enables synchronized multimodal output. The shared encoder learns richer representations than either modality alone.

3. **HuBERT units as intermediate targets:** Instead of decoding to raw audio or spectrograms, predict discrete HuBERT acoustic-speech units (K-means on 6th layer, K=100). These capture phonetic and articulatory information in a compact discrete space, sidestepping the need for aligned audio from the patient.

4. **TTS-generated training targets:** When patients cannot speak, generate reference audio via TTS to create HuBERT unit targets. This removes the requirement for patient-produced audio entirely.

5. **Pretrain LM on external speech corpus, freeze, then train encoder on neural data:** This two-stage approach means the LM provides strong language priors without being overfit to limited neural data. Applicable to any low-data BCI scenario.

6. **80-ms streaming chunks:** Processing neural data in 80-ms non-overlapping windows (after 16x downsampling by the CNN encoder) balances temporal resolution with computational feasibility. The 12.5 Hz effective rate is sufficient for speech decoding.

7. **Voice conversion for personalization:** Using YourTTS conditioned on a short pre-injury voice clip to personalize synthesized speech. Requires only a brief recording of the target voice.

8. **HiFi-CAR streaming vocoder:** A modified HiFi-GAN that operates on acoustic-speech units and generates audio in 80-ms increments, enabling true streaming synthesis.

9. **Ablation-based salience mapping for electrode contributions:** Occluding individual channels and measuring RNN-T loss change to quantify per-electrode importance. Useful for understanding which electrodes matter and for potential electrode selection/reduction in cross-patient work.

10. **Syllable-length pauses between words during silent speech:** Instructing the participant to insert brief pauses (~300--500 ms) between words during mimed speech to assist the decoder in identifying word boundaries without compromising naturalness or speed.

11. **Long-form decoding via autoregressive chunk processing with state resets:** For continuous operation beyond single trials, pass entire blocks of neural data autoregressively in 80-ms chunks, resetting hidden states when the model emits blanks for >4 consecutive seconds. This enables indefinite operation.

12. **Code availability:** https://github.com/cheoljun95/streaming.braindecoder
