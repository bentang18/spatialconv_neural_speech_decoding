# Duraivel et al. 2023 - High-Resolution Neural Recordings Improve the Accuracy of Speech Decoding

## Citation
Duraivel, S., Rahimpour, S., Chiang, C.-H., Trumpis, M., Wang, C., Barth, K., Harward, S.C., Lad, S.P., Friedman, A.H., Southwell, D.G., Sinha, S.R., Viventi, J., & Cogan, G.B. (2023). High-resolution neural recordings improve the accuracy of speech decoding. *Nature Communications*, 14, 6938. https://doi.org/10.1038/s41467-023-42555-1

## Setup
- **Array specs**: LCP-TF (liquid crystal polymer thin-film) uECoG arrays fabricated by Dyconex Micro Systems Technology (Bassersdorf, Switzerland). Two designs:
  - 128-channel: 8 x 16 grid, 1.33 mm center-to-center spacing, 200 um electrode diameter
  - 256-channel: 12 x 22 grid, 1.72 mm center-to-center spacing, 200 um electrode diameter
  - Density: 34-57 electrodes/cm^2 (vs. 1 electrode/cm^2 for macro-ECoG, 6 electrodes/cm^2 for high-density ECoG)
- **Patient count**: 4 uECoG subjects (S1-S4; 1 female, mean age 53); 11 control IEEG patients (7 female, mean age 30; 4 ECoG + 7 SEEG)
- **Brain region**: Speech motor cortex (SMC), covering pre-central (motor) and post-central (sensory) gyrus
- **Task**: Intraoperative non-word speech repetition task. Listen-and-repeat paradigm with 52 unique non-word tokens per block, 3 blocks (3 repetitions). CVC and VCV tokens using 9 phonemes (4 vowels: /a/, /ae/, /i/, /u/; 5 consonants: /b/, /p/, /v/, /g/, /k/).
- **Recording duration**: Up to 15 minutes of recording / 1.04 minutes of total spoken duration per subject. Overall experiment time 0.47 to 1.5 minutes of utterance duration (Supplementary Table 3).

## Problem Statement
Neural speech prostheses require accurate decoding of spoken features from brain signals. Existing recording technologies (macro-ECoG at 10 mm spacing, high-density ECoG at 4 mm spacing, SEEG at 3.5-5 mm spacing) have coarse spatial resolution that inadequately captures the rich spatio-temporal structure of human brain signals. Prior work showed articulatory neural properties are distinct at millimeter-scale resolution and that decoding improves with denser sampling, but no study had demonstrated micro-scale (sub-2 mm) surface recordings for speech decoding in humans.

## Method

### Neural preprocessing (signal processing pipeline)
1. **Decimation**: Raw recordings (20,000 samples/sec via Intan RHD system, analog filtered 0.1 Hz - 7500 Hz) decimated to 2,000 Hz with anti-aliasing filter.
2. **Re-referencing**: Common average referencing (CAR).
3. **Electrode exclusion**: Removed electrodes with impedance > 1 MOhm or recordings > 3x log-RMS. Removed trial epochs with excessive artifacts (> 10 RMS after CAR).
4. **Acoustic contamination check**: Compared spectral information between microphone audio channel and neural voltage time-series using cross-correlation in frequency domain (200 ms Hamming window). Confirmed no significant microphone contamination in the HG band of interest (except S1 at frequencies > 200 Hz).

### Features extracted
- **High gamma (HG) envelope**: Band-pass filtered into 8 center frequencies (logarithmically spaced) between 70-150 Hz. Envelope of each band extracted via Hilbert transform, averaged across bands, downsampled to 200 Hz with anti-aliasing filter, mean-normalized to baseline (500 ms pre-stimulus window).
- **HG-ESNR (evoked signal-to-noise ratio)**: Computed over **spectrograms** (multi-taper time-frequency analysis, 500 ms window, 50 ms step, 10 Hz frequency smoothing), not raw HG envelopes. Ratio of speech-period spectrogram (−500 to +500 ms around **utterance** onset) to average baseline PSD. Baseline normalization removes 1/f broadband activity. Used to identify significant electrodes (10,000-iteration one-sided permutation test, FDR-corrected p < 0.05).
- **Low frequency signals (LFS)**: Band-pass filtered 1-30 Hz using two-way least-squares FIR filter, downsampled to 200 Hz, mean-normalized. Tested as additional feature but did NOT improve performance.
- **Analysis windows**: 1-second windows of normalized HG centered on phoneme onset (-500 ms to 500 ms) for decoding; 200 ms windows (10 ms step) from -1000 ms to 1000 ms aligned to utterance for temporal analysis.

### Decoder architecture

**Linear decoder (SVD-LDA)**:
1. SVD to compress spatio-temporal HG matrix into low-dimensional features (retaining 80% of variance).
2. LDA (supervised linear discriminant analysis) for 9-way phoneme classification.
3. 20-fold nested cross-validation with grid search for optimal number of principal components (S1: 34, S2: 39, S3: 25, S4: 22).
4. Trained separately for each phoneme position (P1, P2, P3) within the non-word, and also collapsed across positions.

**Nonlinear decoder (seq2seq LSTM RNN)**:
1. **Temporal convolutional layer**: 1D convolution (kernel width = hop length = 50 ms = **10 samples at 200 Hz**, no nonlinearity) to reduce memory loading on encoder. Input: HG activations (1 sec window, 200 time points) from all significant electrodes.
2. **Encoder RNN**: Bidirectional LSTM (800 units), forward and backward memory states averaged for final state.
3. **Decoder RNN**: Unidirectional RNN (800 units) predicting one-hot phoneme at each position via dense layer + SoftMax. Uses teacher forcing during training.
4. Training: Adam optimizer, learning rate 1e-3, 800 epochs, L2 regularization on convolutional and recurrent layers, categorical cross-entropy loss. Hyperparameters tuned via keras-tuner with 200 random combinations (10-fold CV, 3 reps). 80% train / 20% test split.

### Comparison conditions (uECoG vs macro-ECoG vs SEEG)
- **Direct comparison**: 4-way vowel decoding (SVD-LDA) on anatomically matched SMC electrodes with significant HG-ESNR.
- **IEEG controls**: 11 epilepsy patients with standard clinical electrodes (ECoG: Ad-Tech 48/64 channels, 10 mm spacing, 2.3 mm diameter; SEEG: PMT/Ad-Tech depth electrodes, 3.5-5 mm spacing). Recorded with Natus Quantum LTM amplifier at 2,048 Hz.
- **Simulated resolution comparison**: Poisson disc subsampling of uECoG electrodes to simulate coarser spatial resolutions (from full array down to 10 mm equivalent spacing).
- **Spatial averaging**: Rectangular subgrids from 1x1 (200 um effective contact) to 8x8 (10 mm effective contact) to simulate macro-electrode contact sizes.

## Key Results

### Signal quality
- uECoG arrays had **57x higher electrode density** compared to macro-ECoG and **9x higher** than high-density ECoG.
- **48% higher signal-to-noise ratio** (1.7 dB increase in median HG-ESNR): uECoG median ESNR = 2.9 dB (S1: 4.2, S2: 3.5, S3: 1.9, S4: 2.1 dB) vs. standard IEEG median ESNR = 1.2 dB (p < 0.05, Mann-Whitney U).
- Inter-electrode HG correlation drops below r = 0.6 at < 2 mm spacing, confirming spatially discriminative signals at the micro-scale.
- Significant HG electrodes: S1: 111/128, S2: 111/128, S3: 63/128, S4: 149/256. 77.4% of significant electrodes were active before utterance onset (motoric activations preceding speech).

### Decoding accuracy - linear (SVD-LDA, 9-way phoneme)
- **Best single-position accuracy**: S1-P1: 57% (chance = 11.1%), with best consonant /g/ at 84.2% and best vowel /i/ at 75.3%. S2-P1: best consonant /g/ at 72.7% and best vowel **/u/ at 83.3%** (not /i/).
- **Mean P1 accuracy across subjects**: 49.9% (P2: 45%, P3: 38.8%, All positions combined: 36.6%).
- **Pooled across positions**: S1: 40.72%, S2: 42.96%, S3: 20.92%, S4: 41.67%.
- Decoding performance saturated at ~500 ms time window. **Caveat:** S3 and S4 showed **decreased** decoding beyond the optimal point — wider windows impaired performance at low SNR and reduced data, due to interference from adjacent phonemes.
- **Position-dependent decoding matters:** accuracy decreased when phonemes were pooled across positions (S1: 49.9% P1 → 40.72% all positions). Attributed to unique position-based neural information and interference from neighboring phonemes. Relevant for CTC decoder design.

### uECoG vs standard IEEG (4-way vowel classification)
- **uECoG: 71%** (n=4) vs. **ECoG: 36.7%** (n=4) vs. **SEEG: 36.5%** (n=7). Chance = 25%.
- uECoG outperformed standard clinical recordings by **36%** (p < 0.05, one-sided permutation test with 10,000 iterations). Nearly doubled performance.

### Resolution and coverage effects
- Electrode pitch < 1.5 mm reached 95% threshold of full-array decoding performance.
- Up to **29% improvement** in decoding compared to simulated 4-10 mm (standard ECoG) resolution.
- Decoding accuracy decreased with increased simulated contact size; best performance at smallest contact (200 um).
- Spacing requirement was inversely proportional to number of unique phonemes decoded.

### Nonlinear seq2seq decoder
- **Only S1 and S2 analyzed** — S3 (low SNR) and S4 (1 block) excluded because they "had the requisite SNR and amount of data."
- **S1**: 46.4% accuracy, 0.99 bits phoneme error vs. linear 34.4%, 1.68 bits (mean across positions).
- **S2**: 50.8% accuracy, 0.91 bits phoneme error vs. linear 43%, 1.4 bits.
- Seq2seq significantly outperformed linear model for both subjects. Comparison was against a **simultaneously trained SVD-LDA with matched 5-fold CV (20% held-out)**, not the standard 20-fold CV.
- Sequential decoding performance was dependent on high-resolution spatial sampling (reduced with subsampling; one-way ANOVA S1: F_3,99 = 40.47, p = 5.4e-17; S2: F_3,99 = 99.97, p = 2e-79).

### Error analysis
- Phoneme misclassifications were systematic: confusion errors decreased with phonological distance (Hamming distance on 17-bit Chomsky-Halle feature vectors).
- Phoneme error < 1 bit for S1 and S2 at P1-P3 (chance error = 2.4 bits).

## Hardware Details
- **Substrate**: Liquid crystal polymer thin-film (LCP-TF), custom non-commercial design.
- **Fabrication**: Dyconex Micro Systems Technology, Bassersdorf, Switzerland.
- **Electrode diameter**: 200 um exposed.
- **128-channel version**: 8 x 16 grid, 1.33 mm pitch, physical footprint ~11 mm x 21 mm. Narrow enough to tunnel through DBS burr-hole.
- **256-channel version**: 12 x 22 grid, 1.72 mm pitch, physical footprint ~38 mm (length). Placed directly on cortex during awake craniotomy, affixed to modified surgical arm.
- **Amplification**: Intan Technologies RHD chips via custom headstages, connected through micro-HDMI cables to Intan RHD recording controller. Digitized at 20,000 samples/sec, analog filtered 0.1 Hz - 7500 Hz.
- **Ground/reference**: DBS insertion canula (128-ch); surgical retractor grounding wire attached to scalp (256-ch).
- **In vivo impedance**: S1: 81.3 +/- 3.8 kOhm, S2: 12.9 +/- 1.3 kOhm, S3: 27.5 +/- 4.8 kOhm, S4: 19.7 +/- 4.2 kOhm. Electrodes > 1 MOhm discarded.
- **Sterilization**: Ethylene oxide gas prior to surgical implantation.
- **Registration**: Corners registered using Brainlab Neuronavigation (Munich, Germany) to subject-specific anatomical T1 structural image. Post-hoc localization via intraoperative CT or BrainLab markings.

## Task Details
- **Paradigm**: Listen-and-repeat non-word speech repetition, performed intraoperatively during awake DBS surgery (S1-S3) or awake craniotomy for tumor resection (S4).
- **Stimuli**: 52 unique non-word tokens per block. Equal CVC (26) and VCV (26) tokens. 9 phonemes: /a/, /ae/, /i/, /u/, /b/, /p/, /v/, /g/, /k/. Fixed set of 9 phonemes at each of 3 positions within the token.
- **Trial structure**:
  1. Visual cue: flashing white circle on laptop screen (detected by photodiode on Intan auxiliary channel).
  2. Auditory stimulus: battery-powered stereo speaker (Logitech) via USB DAC + amplifier (Fiio). Mean stimulus duration 0.5 +/- 0.1 seconds.
  3. Time to respond: 1.1 +/- 0.3 seconds from stimulus onset.
  4. Speech production response: mean utterance duration 0.45 +/- 0.1 seconds (range 0.3 - 0.7 seconds across subjects).
  5. Each trial lasted 3.5 - 4.5 seconds.
  6. 250 ms jitter between trials.
- **Blocks**: 3 consecutive blocks per session, 52 unique tokens per block shuffled. S4 completed only 1 block (~50 trials). When other subjects subsampled to 50 trials: S1: 36%, S2: 27%, S3: 15%, S4: 44% — S4's 256-channel array compensates via higher spatial coverage.
- **Behavioral performance**: > 95% correct repetition (S1: 96%, S2: 98%, S3: 98%, S4: 100%).
- **Audio recording**: Clip-on microphone (Movophoto) connected to pre-amplifier (Behringer), digitized at 44,100 Hz via Psychtoolbox in MATLAB. Also digitized on Intan system at 20,000 Hz (via photodiode channel).
- **Annotation**: Auditory stimulus, spoken non-word production response, and phoneme onsets manually identified using Audacity.
- **Task software**: Psychtoolbox scripts in MATLAB R2014a.
- **Total time**: ~15 minutes per subject.

## Reusable Ideas

### Preprocessing pipeline
1. Decimate to 2 kHz, CAR, impedance-based electrode exclusion (> 1 MOhm), artifact rejection (> 10 RMS post-CAR, > 3x log-RMS).
2. HG extraction: 8 log-spaced bands in 70-150 Hz, Hilbert envelope per band, average across bands, downsample to 200 Hz, normalize to 500 ms pre-stimulus baseline.
3. HG-ESNR metric (Eq. 2 in paper) for electrode selection: 10,000-iteration one-sided permutation test, FDR-corrected p < 0.05.
4. Acoustic contamination check: cross-correlate microphone and neural channels in frequency domain.

### Features and baselines
- HG envelope (70-150 Hz) is the primary feature; LFS (1-30 Hz) did not add value for this non-word task (but may help for naturalistic speech with prosodic content).
- Baseline performance to compare against: 9-way phoneme SVD-LDA at ~40-50% for P1 (chance 11.1%), 4-way vowel SVD-LDA at ~71% (chance 25%).
- Phonological distance (Hamming on Chomsky-Halle features) as error metric: quantifies whether errors are linguistically structured vs. random.
- Decoding window optimization: performance saturates at ~500 ms centered on phoneme onset.

### Analysis techniques worth replicating
- **SVD-tSNE** for cortical state-space visualization: SVD on spatio-temporal HG covariance matrix -> PCs explaining 80% variance -> t-SNE (Barnes-Hut, perplexity=30) for 2D visualization of phoneme/articulator clustering.
- **Silhouette analysis** with Poisson disc subsampling to quantify spatial resolution dependence of clustering quality.
- **Articulatory maps**: Per-electrode SVD-LDA for 4-way articulator classification (low vowel, high vowel, labial consonant, dorsal-tongue consonant), ROC-AUC per electrode, plotted as spatial heatmap showing somatotopic organization.
- **Sequential temporal decoding**: Train SVD-LDA on sliding 200 ms windows (10 ms step) to show temporal progression of phoneme encoding.
- **Subsampling analyses**: Poisson disc sampling to simulate different spatial resolutions; spatial averaging with rectangular subgrids to simulate different contact sizes.

### Code and data availability
- **Code**: MATLAB decoding analysis scripts on Zenodo: `coganlab/micro_ecog_phoneme_decoding: v1.0` (https://doi.org/10.5281/zenodo.8384194).
- **Data**: Available via DABI (Data Archive for The Brain Initiative): https://dabi.loni.usc.edu/dsi/7RKNEJSWZXFW (restricted access, IRB Pro0072892 and Pro00065476).
- **Software stack**: MATLAB R2021a, Python 3.7, TensorFlow 2.0, Keras 2.4.

### Additional details from re-read
- **S3 poor performance** attributed specifically to "tunneling of the electrode array through the burr hole" causing poor contact with SMC — a mechanical issue, not intrinsic neural limitation.
- **Coverage AND resolution independently contribute**: rectangular subgrid analysis (2×4 to 8×16) shows performance increasing with more electrodes even at fixed pitch. Maximizing both channel count and density matters.
- **77.4% of significant electrodes active before utterance onset**, starting up to 1 sec before — evidence of sensory-motor integration (auditory→motor program transformation).
- **Inter-electrode HG correlation at 10 mm spacing**: r < 0.2 (much lower than the < 0.6 threshold at < 2 mm).
- **Cortical localization varies**: S1-S3 used BioImage Suite (CT + T1 alignment); S4 used Brainlab landmarks on Freesurfer reconstruction.
