# Jude et al. 2025 - Decoding Intended Speech in Locked-In Syndrome

## Citation

Jude, J. J., Haro, S., Levi-Aharoni, H., Hashimoto, H., Acosta, A. J., Card, N. S., Wairagkar, M., Brandman, D. M., Stavisky, S. D., Williams, Z. M., Cash, S. S., Simeral, J. D., Hochberg, L. R., & Rubin, D. B. (2025). Decoding intended speech with an intracortical brain-computer interface in a person with longstanding anarthria and locked-in syndrome. *bioRxiv*. https://doi.org/10.1101/2025.08.12.668516

## Setup (modality, patients, brain region, task, data amount)

- **Modality**: Intracortical microelectrode arrays (Blackrock Utah arrays, 6 x 64-electrode arrays = 384 electrodes total, 1.5 mm electrode length). Signals recorded at 30 kHz via Neuroplex-E system.
- **Patient**: Single participant T17 -- man with ALS, tetraplegia, anarthria, ventilator dependence, locked-in syndrome. Only remaining volitional motor control is over extraocular muscles. Has not spoken in over 2 years. Enrolled in BrainGate2 clinical trial (NCT00912041).
- **Brain regions**: Left hemisphere precentral gyrus at three sites:
  - Area 6v (ventral precentral gyrus) -- 2 arrays (128 electrodes)
  - Area 55b (middle precentral gyrus) -- 2 arrays (128 electrodes)
  - Area 6d (dorsal precentral gyrus) -- 2 arrays (128 electrodes)
- **Tasks**:
  1. Isolated cue task: instructed delay paradigm with 13 orofacial movements and 39 phonemes
  2. Sentence copy task: 50-word vocabulary sentences and unconstrained 125k-word vocabulary conversational sentences
  3. Three-phased reading/internal speech/attempted speech task with phonemes, words, phrases, and sentences (10 unique stimuli per language unit, 5 repetitions each)
- **Data amount**: Multiple sessions across 165+ trial days. Sentence decoding sessions on trial days 26, 33, 68, 75, 89, 165. Closed-loop sentence blocks included starting from trial day 33.

## Method (preprocessing, decoder architecture, training)

### Preprocessing
- Analog bandpass filter: 4th-order Butterworth, 250 Hz to 5 kHz
- Linear regression referencing (LRR) to remove common noise: each electrode's signal regressed against all others, computed at session start from an initial reference block
- Electrode-specific thresholds set at -3.5x standard deviation of filtered signal
- Two neural features extracted per electrode per 10ms bin:
  - **Threshold crossing rate** (ncTX): count of negative threshold crossings
  - **Spike band power** (SBP): sum of squared voltages in each 10ms bin
- Features z-scored (mean subtracted, divided by standard deviation) per electrode within each block
- Feature normalization during closed-loop blocks to account for nonstationarities

### Decoder architecture
- **Offline classification** (isolated cues): Gaussian Naive Bayes (GNB) classifier with leave-one-out cross-validation (scikit-learn). Time-averaged features over the go period.
- **Sentence decoding**: 5-layer GRU recurrent neural network (512 units per layer), implemented in TensorFlow 2
  - Phoneme probabilities inferred every 80ms time window
  - CTC (Connectionist Temporal Classification) loss function
  - Per-session non-linear input layer added to handle cross-session neural variability (same dimensionality as features: 512)
  - Regularization: dropout, Gaussian white noise, L2 weight norm
  - Data augmentation applied during training
- **Language model**: 5-gram phoneme-based WFST (weighted finite state transducer) built on Kaldi, trained with WeNet framework on OpenWebText2 corpus. Viterbi (beam) search over RNN phoneme probability lattices.
- **Text-to-speech**: Personalized neural TTS model (StyleTTS 2) fine-tuned on participant's pre-ALS voice recordings.

### Training
- Batches drawn from a single session at a time; corresponding session-specific input layer trained with the shared 5-layer GRU
- Trained on increasing cumulative sessions (cross-session ensemble)
- Closed-loop training data improved phoneme decoding accuracy substantially over open-loop data alone
- Real-time BRAND framework used for feature extraction, decoding, and task control (modular node-based Python, Redis messaging)

## Key Results (quantitative)

### Isolated cue classification (area 6v, 128 electrodes, GNB)
- Orofacial movements: **91.5%** accuracy (95% CI: 88.4-94.6%)
- Isolated phonemes (39 classes): **50.4%** (95% CI: 48.1-52.8%)
- Single words (50-word vocab): **51.1%** (95% CI: 48.0-54.2%)
- Orofacial decoding accuracy significantly higher than phoneme decoding (p < 5.57e-19)

### Isolated cue classification (area 55b, 128 electrodes, GNB)
- Orofacial movements: **45.4%** (95% CI: 40.0-50.8%)
- Phonemes: **17.4%** (95% CI: 15.6-19.2%)
- Words: **8.0%** -- but longer words (e.g., "comfortable", "computer", "success") decoded best

### Sentence decoding (RNN + LM, all 256 electrodes from 6v + 55b)
- **50-word vocabulary**: Word error rate (WER) fell from ~30% to **22%** by second session; raw phoneme error rate ~27%
- **125k-word unconstrained vocabulary**: Best block achieved **41% phoneme error rate** and **48% word error rate** (trial day 68, 4th block). Phoneme error rate consistently below 50% after initial sessions.
- Speaking rate: **45-60 words per minute** across sessions
- Offline (trained on 5 sessions): phoneme error rate **34%**, word error rate **39%**
- Closed-loop training data provided outsized improvement over open-loop data

### Three-phased task (reading / internal speech / attempted speech)
- Area 6v: best at decoding phonemes and short units; attempted speech phase accuracy up to ~64% for phrases, but sentence accuracy much lower
- Area 55b (dorsal array): best at decoding phrases and sentences; phrase reading accuracy **64%**, internal speech phrase accuracy **46%**
- Area 55b encodes higher-order linguistic units (phrases, sentences) better than phonemes
- 6v clusters neural activity by articulatory tongue position; 55b clusters by consonant/vowel distinction and higher-order language

### Key error patterns
- Consonant confusions between similarly articulated phonemes (P/B, CH/JH, F/V, D/T)
- Vowels more confused with other vowels; consonants with other consonants
- In sentence context: more vowel confusion (co-articulation effects) vs. isolated phoneme task showing more consonant confusion

## Cross-Patient Relevance

- **Locked-in / no overt speech**: This is the first demonstration of articulator-based speech decoding in a fully locked-in participant with no remaining speech ability. Prior high-performance speech BCIs (Willett 2023, Metzger 2023, Card 2024) worked with dysarthric participants who retained some articulator movement. This raises the question: for cross-patient models trained on dysarthric speakers, will representations transfer to anarthric patients?
- **Degraded articulatory representations**: Phoneme decoding accuracy in locked-in T17 is substantially lower than in dysarthric participants (~50% isolated phoneme accuracy vs. much higher in prior work). The neural signatures for fine articulatory control may degrade with prolonged disuse. Cross-patient decoders may need to be robust to this variability in representation quality.
- **Area 55b as a complementary signal source**: Middle precentral gyrus (area 55b) encodes phrases and sentences rather than phonemes. This is a different representational level than ventral precentral gyrus (area 6v), which is phonemic/articulatory. For uECoG cross-patient work, recording from both regions and fusing phoneme-level and phrase/sentence-level representations could improve robustness, especially for patients with degraded articulatory signals.
- **Cross-session nonstationarity solution**: The per-session non-linear input layer approach (shared GRU backbone + session-specific input projection) is directly analogous to per-patient input alignment layers for cross-patient decoding. This is a lightweight domain adaptation strategy.
- **Internal speech signal**: Area 55b showed sustained activity during internal speech and reading phases (not just attempted speech), with phrase decoding accuracy of 46% during internal speech. This suggests middle precentral gyrus might provide a modality-invariant speech representation that is less dependent on overt articulatory intent -- potentially more transferable across patients.

## Reusable Ideas

1. **Session-specific input layers for cross-domain alignment**: Adding a non-linear input layer per session (same dimensionality as features) while sharing the RNN backbone across sessions. Directly applicable as per-patient input layers in cross-patient decoders. Low parameter overhead, handles distribution shift at the input level.

2. **Multi-area ensemble decoding at different linguistic levels**: Combining phoneme-level decoding from articulatory cortex (6v) with phrase/sentence-level decoding from higher-order speech planning cortex (55b). For uECoG, this suggests value in electrode coverage spanning ventral-to-middle precentral gyrus and building hierarchical decoders that fuse articulatory and linguistic features.

3. **Closed-loop training for improved neural discriminability**: Closed-loop sentence blocks (where participant sees decoded output in real time) yielded substantially better training data than open-loop blocks. Consonant decoding improved markedly; the participant's neural patterns became more separable with feedback. Relevant for calibration protocols in any speech BCI.

4. **CTC loss for phoneme sequence decoding**: Using CTC loss with an RNN to decode variable-length phoneme sequences from fixed-rate neural features (80ms windows). Avoids explicit phoneme segmentation. Combined with a 5-gram WFST language model (Kaldi/WeNet on OpenWebText2) for beam search decoding.

5. **Linguistic (not just phonemic) decoding from middle precentral gyrus**: Area 55b electrodes encoded whole phrases and sentences better than individual phonemes, and showed semantic-level confusion patterns (confusing nonsense phrases with each other, not with real phrases). This suggests decoding linguistic units directly rather than only phonemes could bypass articulatory degradation -- particularly relevant for locked-in patients or cross-patient generalization where articulatory representations are unreliable.

6. **Mahalanobis distance + hierarchical clustering for characterizing neural representations**: Used to reveal that 6v organizes by articulatory place while 55b organizes by linguistic category. Useful diagnostic tool for understanding what a given electrode population actually encodes before choosing a decoder strategy.

7. **Feature set**: Threshold crossings + spike band power as dual features from intracortical arrays, z-scored per electrode per block, with LRR for noise removal. Simple but effective feature pipeline for intracortical recordings.
