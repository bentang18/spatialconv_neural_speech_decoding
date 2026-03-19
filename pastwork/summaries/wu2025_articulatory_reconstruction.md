# Across-Speaker Articulatory Reconstruction from Sensorimotor Cortex for Generalizable Brain-Computer Interfaces
**Authors:** Ruoling Wu, Julia Berezutskaya, Zachary V. Freudenburg, Nick F. Ramsey
**Venue/Year:** bioRxiv preprint, 2025
**DOI/URL:** https://doi.org/10.1101/2025.10.08.680888

## TL;DR
This study extracts generalizable articulatory features from an electromagnetic articulography (EMA) dataset of healthy Dutch speakers using tensor component analysis (TCA), then reconstructs those features from HD-ECoG recordings of three separate able-bodied participants. Mean Pearson's correlation coefficients (PCC) between reconstructed and original articulatory features were 0.80, 0.75, and 0.76 for the three participants, demonstrating the feasibility of cross-subject articulatory-neural transfer for speech BCIs.

## Problem & Motivation
Speech BCIs typically train decoders using articulatory or acoustic data from the same individual whose brain signals are being recorded. However, individuals with severe vocal tract paralysis (e.g., from ALS or brainstem stroke) cannot produce any speech movements or sound, leaving no articulatory or acoustic data available for training. The authors ask two questions: (1) Can generalizable articulatory features be extracted across healthy speakers despite individual variation in vocal tract anatomy? (2) Can these generalizable features be reconstructed from brain activity of a different group of individuals? A positive answer would enable a transfer-learning approach to speech BCI development for people who cannot produce any speech.

## Method

### Overall framework
The framework has two parallel pipelines (EMA and HD-ECoG), each with three steps: data preprocessing, tensor construction, and feature extraction via TCA. The extracted articulatory and neural features are then linked through a reconstruction model.

### EMA pipeline
- **Data source:** Publicly available EMA dataset (Wieling et al., 2016) collected with a 16-channel WAVE device (Northern Digital InC.) at 100 Hz sampling rate.
- **Participants used:** 8 Central Dutch dialect speakers (U1-U8) selected from 19 Central Dutch speakers after excluding participants with missing trials, interrupted trials, or exceptionally large mouth movements (beyond mean +/- 2.5*std).
- **Sensors:** 7 EMA sensors at jaw, lower lip, right lip corner, upper lip, tongue tip, tongue body, and tongue dorsal positions, each tracking movement in 3 directions (anterior-posterior, up-down, left-right). Left-right direction excluded from analysis (little extra information, confirmed by low variance scores for tongue).
- **Preprocessing:** Missing values imputed via linear interpolation; smoothed with 5th-order 20 Hz low-pass Butterworth filter; baseline removed by subtracting mean positions during a 10-second rest period.
- **Tensor construction:** EMA data segmented using a **0 to 1 second window AFTER speech onset** (not centered on onset). Resulting three-dimensional tensor per participant: 100 time points x 14 articulator-by-direction combinations x 194 word trials. Concatenated across 8 participants into a four-dimensional tensor (8 x 100 x 14 x 194).
- **TCA decomposition:** Nonnegative CANDECOMP/PARAFAC decomposition optimized by hierarchical alternating least squares (using Python library tensortools). Decomposed into 29 components yielding cumulative explained variance of 0.89 averaged across four runs. Components reduced to 10: first component excluded (~80% variance alone), then selected those above **mean explained variance across components 2-29** (the threshold). Further filtered to 6 components with commonality above the mean simulated commonality threshold of 4.57.
- **Commonality formula:** `com(a_r) = (Σ_i a_r^i)² / Σ_i (a_r^i)²` — squared L1/L2 ratio (Hurley & Rickard 2009). Range [1, N_participants]. This formula is directly reusable for measuring cross-patient commonality of learned neural features.
- **Articulatory features:** The word trial factor loadings from the 6 selected components serve as the generalizable articulatory features. **These are word-level scalar loadings, NOT continuous articulatory trajectories.** Each "feature" is a single number per word trial summarizing the contribution of that word to a spatiotemporal pattern.
- **EMA exclusion:** 11 excluded from 19 Central Dutch speakers (2 missing trials, 8 interrupted trials, 1 large mouth movements beyond mean ± 2.5×std), leaving 8.

### HD-ECoG pipeline
- **Preprocessing:** Visual inspection for noisy/flat channels (none removed for P1/P2; 14 removed from P3 due to overlapping grid). Line noise removed with notch filter (50 Hz and harmonics). Common average referencing applied. Downsampled to 512 Hz.
- **Feature extraction:** High frequency band (HFB; 60-170 Hz) extracted using **Morlet wavelet** decomposition in 1 Hz bins (different from our Hilbert envelope method at 70-150 Hz), averaged across frequency bins. Log-transformed and downsampled to 100 Hz. Baseline-corrected by subtracting mean HFB from a 10-second rest period before the first word trial.
- **Speech onset detection:** Azure speech-to-text service, manually corrected using Praat. Time window: **-0.25 to +0.75 sec** around speech onset (1 second total; 95th percentile of word duration distribution was 0.69 sec). Note: asymmetric with EMA window (0 to 1 sec); 0.25s pre-onset window validated in prior literature (Jiang 2016, Ramsey 2018) for capturing first phoneme info in vSMC.
- **Channel selection:** Channels with significant HFB increases during speech vs. silence (two-tailed paired t-test, p<0.001) and large effect size (Cohen's d > 0.8, corresponding to t-value of 12.44). Active channels: P1 = 25/128 (19.5%), P2 = 24/32 (75%), P3 = 21/82 (25.6%).
- **Tensor construction:** Three-dimensional tensor per participant: 100 time points x number of selected channels x 194 word trials. Not concatenated across participants (channel dimensions differ).
- **TCA decomposition:** Nonnegative CANDECOMP/PARAFAC, decomposed into 10, 14, and 17 components for P1, P2, and P3 respectively (corresponding to 80% explained variance). Components selected by PCC between word trial factor loadings across repetitions, using 95th percentile of shuffled distribution as threshold (0.33 for P1, 0.17 for P2, 0.31 for P3). This yielded 7, 6, and 6 neural features for P1, P2, and P3 respectively. Mean PCC of extracted components between repetitions: 0.59 for P1, 0.30 for P2, 0.49 for P3.

### Reconstruction model
- **Model:** Gradient boosting regression (chosen for robustness on small datasets, avoids overfitting).
- **Input:** Neural features (word trial factor loadings from HD-ECoG TCA).
- **Output:** Articulatory features (word trial factor loadings from EMA TCA).
- **Evaluation:** PCC between reconstructed and original articulatory features (194-length vectors of scalar loadings, not continuous trajectories). Cross-validated using leave-one-word-out scheme (both repetitions of left-out word excluded from training). Significance determined by permutation test (shuffling word labels 1000 times). Repetition 1 and repetition 2 evaluated separately.
- **Recording duration:** ~10-11 min per participant (194 trials × ~3-3.6 sec/trial), comparable to our intra-operative recording times.

## Data

### EMA dataset
- Publicly available dataset from Wieling et al. (2016)
- 40 speakers total (21 Low-Saxon Dutch, 19 Central Dutch); 8 Central Dutch speakers used
- Two tasks: picture naming (70 object names) and CVC sequence reading (27 sequences) = 97 Dutch words total, each repeated twice (194 trials)
- 7 EMA sensors tracking articulatory movements in anterior-posterior and up-down directions (14 articulator-by-direction combinations)

### HD-ECoG dataset
- 3 male participants (P1, P2, P3; ages 31, 26, and 40) at University Medical Centre Utrecht
- P1 and P2: clinical ECoG for epilepsy diagnostics (chronic implant); **P3: intra-operative (awake tumour resection surgery)** — most analogous to our setting, achieved PCC=0.76
- All HD-ECoG grids implanted over **ventral sensorimotor cortex (vSMC)** on the left hemisphere, straddling central sulcus (pre-central motor + post-central somatosensory)
- Electrode configurations:
  - P1: 128 electrodes, 3 mm inter-electrode distance (IED), 1 mm exposed diameter
  - P2: 32 electrodes, 4 mm IED, 1 mm exposed diameter
  - P3: 96 electrodes, 3 mm IED, 1 mm exposed diameter
- Sampling rates: 2000 Hz (P1, P3; Blackrock Microsystems) and 2048 Hz (P2; Micromed)
- Audio recorded at 30000 Hz
- Task: Read aloud the same 97 Dutch words (presented in text form), each repeated twice (194 trials)
- Trial duration: 2 sec (P1, P2) or 1.8 sec (P3); inter-trial interval: 1.2 sec (P1, P2) or 1.8 sec (P3)

## Key Results

### EMA articulatory features
- 6 generalizable articulatory features extracted with high commonality values: 7.92, 7.88, 7.39, 7.32, 6.37, and 4.72 for features 1-6 respectively (max possible = 8, indicating even contribution across all 8 participants)
- PCC between repetitions 1 and 2 for the word trial factor: 0.97, 0.93, 0.83, 0.95, 0.62, and 0.49 for features 1-6; significant (P<0.05) for the first four features

### HD-ECoG neural features
- Three types of brain responses identified in all participants: early (peak before speech onset), mid (peak after onset, before mean speech offset), and late (peak around/after mean speech offset)
- **Spatial organization:** In P1 and P3, channels with large weights are located **posteriorly** (post-central/somatosensory) for early and late responses; in P2 they are **anteriorly** located (pre-central/motor) for mid responses
- PCC for neural features corresponding to early responses was **lower** than for mid and late responses
- For P2, **two** time features were identified for mid response (unlike one each for early/late)

### Reconstruction performance
- **P1:** Mean PCC = 0.80 (p<0.05), averaged across word repetitions
- **P2:** Mean PCC = 0.75 (p<0.05)
- **P3:** Mean PCC = 0.76 (p<0.05)
- All three participants' reconstruction PCCs were significantly above chance (permutation test, alpha = 0.05)
- Correlation coefficients were only slightly above the significance threshold, suggesting articulatory features of different words may cluster closely in the low-dimensional space

## Relevance to Cross-Patient uECOG Speech Decoding
This paper is directly relevant to cross-patient decoder design because it demonstrates that articulatory representations can be transferred across individuals -- the articulatory features are extracted from one group (EMA speakers) and reconstructed from brain data of entirely separate individuals (HD-ECoG participants). Key implications:

1. **Transfer learning for paralyzed users:** The framework bypasses the need for the target user to produce any speech. Willett 2023 showed structural similarity between articulatory features from healthy participants and neural features from an individual with paralysis — articulatory invariance holds even across able-bodied vs. paralyzed.
2. **Generalizable intermediate representations:** The TCA-derived articulatory features are speaker-independent, suggesting a shared articulatory feature space could serve as a common target for cross-patient decoders.
3. **HD-ECoG over vSMC:** The neural recordings are from high-density grids over vSMC, similar in spirit to uECOG arrays. The HD-ECoG grids (3-4 mm IED, 1 mm exposed diameter) are lower density than uECOG (1.33-1.72 mm IED). **P3 is intra-operative** — most analogous to our setting.
4. **Articulatory features more robust than acoustics in vSMC with limited data:** Discussion cites Chartier 2018, Conant 2018, and Anumanchipalli 2019 as showing articulatory features are more robustly encoded in vSMC than acoustic features and "can be learned faster with limited neural data." This directly supports our articulatory auxiliary loss motivation.
5. **Reference-free alignment preferred:** Discussion contrasts TCA commonality with Bocquelet et al. 2016 (linear mapping to a reference speaker). TCA avoids bias from choosing a reference patient — suggests reference-free alignment may be preferable.
6. **Small data regime:** Gradient boosting regression was chosen specifically for robustness on small datasets (194 trials, ~10-11 min recording), comparable to our intra-operative setting.
7. **Limitation for intra-operative use:** Reconstruction is word-level factor loadings (not continuous articulatory trajectories), limiting comparison to continuous trajectory systems like Anumanchipalli 2019.

## Limitations
- Correlation coefficients were only slightly above the significance threshold (alpha = 0.05), indicating modest discriminability between words in the low-dimensional articulatory feature space
- Only 7 EMA sensors on the upper vocal tract; does not capture complete vocal tract movements (e.g., larynx, velum)
- Small vocabulary of 97 Dutch words; unlikely to capture the full range of articulatory gestures in natural conversation
- Only 2 repetitions per word; more repetitions could improve reliability of TCA features
- Only 3 HD-ECoG participants, all male, all able-bodied; generalization to individuals with paralysis is assumed but not tested
- Pipeline validated offline only; not adapted for real-time use
- EMA participants and HD-ECoG participants spoke potentially different variants of Dutch (Central Dutch dialect vs. standard Dutch)
- The articulatory features and neural features are both low-dimensional summaries (word-level factor loadings), not continuous time-series reconstructions of articulator trajectories

## Key References
- Anumanchipalli, G.K., Chartier, J. & Chang, E.F. (2019) Speech synthesis from neural decoding of spoken sentences. *Nature*, 568, 493-498.
- Chartier, J., Anumanchipalli, G.K., Johnson, K. & Chang, E.F. (2018) Encoding of articulatory kinematic trajectories in human speech sensorimotor cortex. *Neuron*, 98, 1042-1054.
- Metzger, S.L. et al. (2023) A high-performance speech neuroprosthesis for speech decoding and avatar control. *Nature*, 620, 1037-1046.
- Willett, F.R. et al. (2023) A high-performance speech neuroprosthesis. *Nature*, 620, 1031-1036.
- Wieling, M. et al. (2016) Investigating dialectal differences using articulography. *Journal of Phonetics*, 59, 122-143.
- Williams et al. (2018) [tensortools library for TCA]
- Salari, E. et al. (2018) Spatial-temporal dynamics of the sensorimotor cortex: sustained and transient activity. *IEEE TNSRE*, 26, 1084-1092.
- Silva, A.B. et al. (2024) The speech neuroprosthesis. *Nature Reviews Neuroscience*, 25, 473-492.
