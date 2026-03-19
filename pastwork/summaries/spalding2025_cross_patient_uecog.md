# Spalding et al. 2025 - Shared Latent Representations for Cross-Patient Speech Decoding

## Citation
Spalding, Z., Duraivel, S., Rahimpour, S., Wang, C., Barth, K., Schmitz, C., Lad, S. P., Friedman, A. H., Southwell, D. G., Viventi, J., & Cogan, G. B. (2025). Shared latent representations of speech production for cross-patient speech decoding. *bioRxiv*. https://doi.org/10.1101/2025.08.21.671516

## Setup
- **Recording modality:** Micro-electrocorticography (uECoG) -- surface cortical recordings from custom-designed high-density arrays (Dyconex Micro Systems)
- **Patient count:** 8 awake patients (mean age 57.4, 7M/1F) at Duke University Medical Center
- **Brain region:** Left sensorimotor cortex (SMC) -- ventral/lateral regions involved in speech articulator control (all patients confirmed left SMC from Fig S1)
- **Patient clinical summary (Tables S1, S3):**

| Patient | Age | Sex | Diagnosis | Channels | Trials | Utt. (min) | Exp. (min) | Sig channels |
|---------|-----|-----|-----------|----------|--------|------------|------------|--------------|
| S1 | 61 | M | Parkinson's (DBS) | 128 | 144 | 1.08 | 9.00 | 111/128 (86.7%) |
| S2 | 64 | M | Parkinson's (DBS) | 128 | 148 | 1.11 | 9.25 | 111/128 (86.7%) |
| S3 | 26 | F | Tumor resection | 256 | 46 | 0.35 | 2.88 | 149/256 (58.2%) |
| S4 | 63 | M | Parkinson's (DBS) | 128 | 151 | 1.13 | 9.44 | 74/128 (57.8%) |
| S5 | 61 | M | Parkinson's (DBS) | 128 | 151 | 1.13 | 9.44 | 63/128 (49.2%) |
| S6 | 71 | M | Tumor resection | 256 | 137 | 1.03 | 8.56 | 144/256 (56.3%) |
| S7 | 53 | M | Tumor resection | 256 | 141 | 1.06 | 8.81 | 171/256 (66.8%) |
| S8 | 60 | M | Tumor resection | 256 | 178 | 1.34 | 11.13 | 201/256 (78.5%) |

- **Duraivel (2023) patient mapping:** Duraivel S1->Spalding S1, S2->S2, S3->S5, S4->S3
- **Task:** Speech repetition of non-word sequences (3 phonemes each, e.g. /abe/); 52 unique non-word tokens; patients listen then immediately repeat
- **Array specs:** Two designs:
  - 128 channels, 1.33 mm pitch, 11 mm x 21 mm (DBS implantation surgeries, S1, S2, S4, S5; placed through burr hole in subdural space)
  - 256 channels, 1.72 mm pitch, 38 mm x 21 mm (tumor resection surgeries, S3, S6, S7, S8; placed directly through craniotomy window)
  - Both: 200 um contact diameter micro-contacts
- **Recording duration:** ~8.23 minutes total utterance, ~68.51 minutes total experiment duration across all 8 patients; trials presented across 3 consecutive blocks (52 trials/block); intraoperative acute recordings (not chronic)
- **Audio recording:** Lavalier mic (Movophoto), synced at 20 ksps on Intan + 44.1 kHz laptop recording
- **Phoneme labels:** Montreal Forced Aligner + manual Audacity correction
- **MNI projection:** MNI-152 for all patients (BioImage Suite for DBS patients; Brainlab for tumor patients)
- **Phoneme inventory:** 4 vowels (/a/, /e/, /i/, /u/) + 5 consonants (/b/, /p/, /v/, /g/, /k/) = 9 phonemes; organized into 4 articulator categories (low, high, labial, dorsal)

## Problem Statement
Current speech BCIs require large volumes of patient-specific data collected over long periods (typically weeks) before the machine learning models can be trained to adequate accuracy. This limits patient usability and hinders adoption. The core barrier to combining data across patients is that electrode placement varies, neuroanatomy differs across individuals, and the sparse sampling of targeted anatomy creates patient-specific neural representations that cannot be naively pooled. This paper asks: can we align neural data from multiple patients into a shared latent space and thereby train cross-patient speech decoders that outperform patient-specific models?

## Method

### Neural preprocessing
1. **Decimation:** Raw signals decimated from 20 ksps to 2 ksps using a finite impulse response anti-aliasing filter (Kaiser window, MATLAB `resample`)
2. **Re-referencing:** Common-average reference (CAR) applied across channels
3. **Channel rejection:** Electrodes with impedance > 1 MOhm excluded from CAR
4. **Trial rejection:** Trials with large transient activity (> 50 uV between timepoints) discarded
5. **High-gamma (HG) extraction:** The key neural feature. Bandpass filtered using frequency-domain Gaussian filters centered at 8 logarithmically spaced center frequencies between 70-150 Hz (with logarithmically scaled bandwidths). Implemented via FFT -> multiply by Gaussian kernel -> inverse FFT. Envelope computed as absolute value of Hilbert transform at each band, then averaged across all 8 bands. Downsampled to 200 Hz with anti-aliasing filter.
6. **Normalization:** HG activity normalized by subtracting the average HG envelope of the baseline period (500 ms pre-stimulus). Analysis window: +/- 500 ms around patient response onset.
7. **Significant channel selection:** Channels with significant HG activity during speech (relative to pre-stimulus baseline, FDR-corrected non-parametric permutation test) are retained; non-significant channels excluded.
8. **All preprocessing done in MATLAB R2021a.**

### Alignment approach
The alignment is a two-step linear pipeline: **PCA then CCA**.

**Step 1 -- PCA (dimensionality reduction per patient):**
- Input: HG activity matrix per patient of dimensions (s, t, n) where s = trials, t = timepoints, n = significant channels
- PCA (scikit-learn) applied to reduce n channels to k principal components, where k is chosen to explain 90% of variance
- Each PC is a linear combination of activity across channels
- This yields patient-specific latent dynamics of dimensions (s, t, k)

**Step 2 -- CCA (cross-patient alignment):**
- Goal: find linear combinations of latent dynamics PCs from two patients that are maximally correlated
- **Pairwise alignment:** Each source patient aligned individually to the target patient (not jointly)
- **Condition averaging first:** Trials are averaged by phoneme sequence identity (up to 52 unique sequences), keeping only sequences present in both patients. This enforces that the same phoneme sequences have similar latent trajectories.
- Condition-averaged latent dynamics are folded with time to create matrices L1 (s_ca * t, k1) and L2 (s_ca * t, k2) where s_ca = number of shared conditions
- Mean subtraction applied to both L1 and L2
- QR decomposition: L1 = Q1 * R1, L2 = Q2 * R2 (Q is orthonormal basis)
- SVD of inner product: Q1^T * Q2 = U * S * V^T
- Projection matrices: M1 = R1^{-1} * U, M2 = R2^{-1} * V
- Aligned latent dynamics: L1^a = L1 * M1, L2^a = L2 * M2
- **Key design choice:** Rather than aligning to a shared abstract space, they align all source patients TO a specific target patient's space: L_{2->1}^a = L2 * M2 * M1^{-1}. This means you pick one target patient and pairwise-align all others to that patient's latent space. This avoids needing to transform the target patient's test data.
- d = min(k1, k2) canonical dimensions retained
- **Implemented in Python 3.10 with numpy**, following MATLAB's `canoncorr` function logic

**Critical detail:** The alignment is learned on the TRAINING set only (both PCA and CCA transforms), then applied to test data, to prevent information leakage.

### Decoder architecture
Two decoders evaluated:

**1. PCA-SVM (primary decoder):**
- PCA layer for additional feature reduction (80% variance explained, fixed)
- Bagged SVM classifier with radial basis function (RBF) kernel (scikit-learn, default parameters)
- Predicts phoneme identity (9 classes, chance = 0.11) at each of 3 positions in the sequence
- Evaluated with 20-fold cross-validation (95%/5% train/test split), balanced accuracy reported
- 50 iterations with different random train/test splits to generate a distribution
- **Note:** No stratification by phoneme class is documented. No random seeds are reported, making exact reproducibility uncertain

**2. Seq2Seq RNN (Table S4 details):**
- Conv1d: 100 filters, k=10, s=10 (50 ms stride at 200 Hz, downsamples 200->20 Hz), no activation, dropout=0.3
- Encoder: BiGRU, 500 units, 2 layers, dropout=0.3, forward + backward summed
- Decoder: GRU, 500 units, 1 layer, dropout=0.3, followed by fully connected layer producing phoneme output probabilities
- Sequentially decodes the 3-phoneme sequence
- Teacher forcing ratio: 0.5 (50% of the time uses ground truth as next input)
- Loss: cross-entropy between ground-truth and predicted phoneme sequence probabilities
- Optimizer: AdamW, lr=1e-4 linearly decayed to 1e-6 over 500 epochs, gradient clip=0.5, L2 regularization=1e-5
- Best model selected by validation accuracy (validation = random 10% of training fold)
- Implemented in Python 3.10, PyTorch 2.4.0, PyTorch-Lightning 2.3.3
- 20-fold cross-validation on target patient, accuracy reported across 50 iterations

### Training procedure
- **Leave-one-patient-out style:** One patient is designated the "target patient." All other 7 patients are "source patients." Source patient data is CCA-aligned to the target patient's space and concatenated with the target patient's training data.
- **Cross-validation:** 20-fold CV on the target patient's data. PCA and CCA transforms are learned on training folds only.
- **Test set:** Always held-out data from the target patient only (as would be the clinical use case).
- **This is repeated for all 8 patients as target** to get population-level results.
- **Three decoding contexts compared:**
  1. Patient-specific: train and test on single patient's data only
  2. Unaligned cross-patient: pool latent dynamics from all patients without CCA, train on pooled, test on target
  3. Aligned cross-patient: CCA-align source patients to target, pool, train, test on target

## Key Results

### Articulator decoding (4-class: low, high, labial, dorsal; chance = 0.25)
- Patient-specific SVM accuracy: mean 0.39
- Aligned cross-patient SVM accuracy: mean 0.43
- Unaligned cross-patient SVM accuracy: mean 0.23 (not significantly different from chance)
- No significant difference between patient-specific and aligned cross-patient (p = 0.79), showing alignment preserves articulatory information

### Phoneme decoding (9-class; chance = 0.11)
- **Patient-specific SVM:** mean 0.24
- **Unaligned cross-patient SVM:** mean 0.19 (significantly lower than patient-specific; p = 0.03)
- **Aligned cross-patient SVM:** mean 0.31 (significantly higher than patient-specific; p = 0.01)
- Significant main effect across contexts: F(2,14) = 12.59, p = 0.007 (repeated measures ANOVA)
- **Per-patient variation:** S4 and S7 are the weakest performers across all conditions. S3 benefits most from cross-patient data (0.53 aligned vs 0.29 patient-specific) due to having the fewest trials (46)
- **Note on unaligned baseline (0.19):** Unaligned pooling includes PC truncation to the minimum PC count across all patients, which weakens the unaligned condition beyond just the lack of alignment

### Best individual patient improvements (S3, which had least data)
- Patient-specific: 0.29
- Aligned cross-patient: 0.53 (massive boost due to S3 having ~1/3 the data of other patients)
- **Note:** Without S3's outlier 0.53, the mean aligned cross-patient SVM drops to ~0.27 — barely above patient-specific 0.24. The 0.31 headline includes this outlier

### Seq2Seq RNN decoding (9-class phoneme)
- Patient-specific RNN: mean 0.24
- Aligned cross-patient RNN: mean 0.27
- Significant improvement with cross-patient data (F(1,196) = 258.51, p = 1.20e-37)
- Cross-patient RNN significantly outperformed cross-patient SVM (p < 0.001, post-hoc Tukey HSD)
- RNN significantly outperformed SVM overall (F(1,196) = 830.89, p = 2.05e-72)

### Subsampling results
- Cross-patient decoding outperforms patient-specific at ALL subsampling ratios of target data (even at 20% of original trials; p = 0.008)
- With additional cross-patient data, patient-specific performance (mean 0.17) exceeded after ~210 trials (~1.6 min utterance, ~13.1 min experiment)
- Diminishing returns as more cross-patient data added (log-fit, r^2 = 0.98)

### Spatial resolution requirements (critical for modality choice)
- **Electrode density (pitch):** Cross-patient models only significantly outperform patient-specific for pitch < 3 mm (i.e., effective pitch in [1.5, 2] mm range). Standard clinical ECoG (~10 mm pitch) is insufficient.
- **Array coverage:** Cross-patient models only significantly better for grid sizes >= 6x12 (~8 mm x 17 mm). Small arrays insufficient.
- **Contact size:** Smaller contacts (micro-scale) trend toward better cross-patient correlation, but the effect is weaker than density/coverage. No significant cross-patient advantage at ANY contact size after FDR correction (p > 0.05 for all).

### Surrogate data control
- Aligned TME (Tensor Maximum Entropy) surrogate data yields accuracies not significantly different from chance (W = 8, p = 0.20), confirming alignment does not artificially create structure.
- **Caveat:** With n=8 and a single realization per patient, the Wilcoxon signed-rank test has very low statistical power. W=8 at n=8 gives p=0.20, not reaching significance at 0.05 — the test is underpowered rather than definitively validating the null. Our evaluation should use more surrogates per patient.

### Cross-patient latent dynamics correlation
- CCA alignment significantly improves Pearson correlation between all patients' latent dynamics and a reference patient (S1) (Wilcoxon signed-rank test, W = 0, n = 8, p = 0.02)
- Silhouette score for articulator clustering significantly higher after alignment than before (p < 0.001)

## Figures Summary

- **Figure 1:** Overview of the approach. (a) Conceptual diagram -- shared articulatory motor network despite different electrode placements. (b) MNI-152 projection of all 8 patients' array locations on average brain. (c) Task diagram -- listen then repeat non-word. (d) Example spectrogram showing HG (70-150 Hz) power increase around speech onset. (e) Array-level spectrograms for 128-ch and 256-ch arrays showing significant HG channels (black borders).

- **Figure 2:** Alignment pipeline and validation. (a) Spatiotemporal HG activation patterns by articulator category for S1 and S2 -- visually distinct across patients. (b) t-SNE of S1 latent dynamics colored by articulator -- clear clustering. (c) Silhouette score quantification significantly above chance. (d) Unaligned vs. CCA-aligned latent dynamics trajectories in 2D PC space -- alignment makes them visually similar. (e) Combined S1+S2 clustering strength improves with alignment. (f) Pearson correlation between all patients and S1 improves with alignment.

- **Figure 3:** Spatial articulator map preservation. (a-b) Spatial maps of electrode tuning (ROC-AUC) for articulators in S1 and S3. (c) Cross-patient projection preserves spatial maps -- significantly higher Pearson correlation with alignment across all patients. (d) Cross-patient articulator decoding via SVM: aligned reconstruction matches patient-specific accuracy; unaligned is at chance.

- **Figure 4:** Cross-patient phoneme decoding improvements. (a) Across all 8 patients, aligned cross-patient > patient-specific > unaligned (significant ANOVA). (b) Detailed results for S1, S2, S3 showing aligned consistently best. (c) Subsampling target data -- cross-patient wins at all ratios. (d) Scaling curve for additional cross-patient trials -- diminishing returns, log-fit. (e) RNN results confirm SVM findings; RNN outperforms SVM.

- **Figure 5:** Spatial resolution analysis. (a) Pitch subsampling -- cross-patient advantage only emerges below 3 mm pitch; correlation increases with density. (b) Grid subsampling -- need >= 6x12 coverage for significant cross-patient advantage. (c) Contact size (spatial averaging) -- smaller contacts trend better but effect less dramatic than density/coverage.

## Cross-Patient Relevance
This is the foundational paper demonstrating that cross-patient speech decoding is feasible in humans using uECoG. Key implications for extending the work:

1. **The shared latent dynamics hypothesis is validated for human speech** -- motor control of articulators produces population-level neural patterns that are similar enough across individuals to be aligned via simple linear methods.
2. **The PCA + CCA pipeline is intentionally simple and linear** -- the authors explicitly acknowledge this leaves room for nonlinear methods (LFADS, MARBLE) that could capture richer structure.
3. **Only 8 patients with acute intraoperative data** -- scaling to more patients (especially chronic implants) is the obvious next step.
4. **The RNN already outperforms the SVM significantly** -- there is clear headroom for more sophisticated decoders (transformers, etc.) on the aligned latent dynamics.
5. **The alignment requires condition-averaged data** -- phoneme labels must be known during alignment training, which is a practical constraint.

## Limitations
1. **Acute intraoperative recordings only** -- not chronic implant data. Recording conditions (anesthesia effects, limited time, patient stress) may differ from chronic BCI settings.
2. **Overt speech in able-bodied speakers** -- all patients had intact speech. Unknown whether shared latent dynamics persist in patients with motor speech impairments (ALS, brainstem stroke), though other BCI studies suggest articulatory representations are preserved.
3. **Small patient pool (n=8)** with limited data per patient (~8.23 min utterance total). The log-fit scaling curve suggests more patients would continue to help.
4. **Simple non-word repetition task** -- 3-phoneme sequences from 9 phonemes. Not continuous speech, not full English phoneme inventory, not sentences/words.
5. **PCA and CCA treat time points as independent observations** -- they discard temporal dynamics information, which is critical for speech. The authors acknowledge this explicitly.
6. **Linear alignment only** -- CCA finds linear transformations. Nonlinear relationships between patients' neural spaces are not captured.
7. **Condition averaging required for CCA** -- alignment needs labeled data (phoneme sequence identities must match across patients). Cannot do unsupervised alignment.
8. **Pairwise alignment to a single target patient** -- does not learn a universal shared space. Each new target patient requires re-computing all pairwise CCA transforms.
9. **Two different array designs** (128-ch and 256-ch) which vary in both coverage and density, making it hard to fully disentangle their contributions.
10. **Phoneme labels generated semi-automatically** (Montreal Forced Aligner + manual Audacity correction) -- label noise may affect results.

## Reusable Ideas
1. **PCA + CCA alignment pipeline:** Simple, interpretable, and effective. Can be implemented in ~50 lines of numpy. The specific workflow (condition-average -> fold time into features -> QR decomposition -> SVD of inner product -> project to target space) is well-documented and reproducible.
2. **Align to target patient space rather than abstract shared space:** Practical for BCI -- the target patient's test data does not need to be transformed.
3. **High-gamma extraction via frequency-domain Gaussian filters:** 8 log-spaced center frequencies in 70-150 Hz, Hilbert envelope, average across bands. More principled than simple bandpass.
4. **Condition averaging by stimulus identity for alignment:** Enforces that alignment is driven by speech content, not noise.
5. **Surrogate control via TME (Tensor Maximum Entropy):** Generates null data preserving first and second order statistics to verify alignment is not artificially boosting accuracy.
6. **Spatial subsampling analyses:** Poisson disk sampling for pitch, sub-grid selection for coverage, spatial averaging for contact size -- a systematic framework for evaluating recording requirements.
7. **Cross-patient projection of spatial maps:** Inverting PCA transforms to map from latent space back to electrode space across patients -- useful for validating that alignment preserves spatial tuning properties.
8. **Seq2Seq RNN architecture for phoneme sequence decoding:** Conv1d(k=10, s=10, 200->20Hz) -> BiGRU encoder (2x500, 2 layers) -> GRU decoder (500) with teacher forcing. A reasonable baseline architecture.
9. **Silhouette score on t-SNE projections** for quantifying clustering quality of latent dynamics by articulator/phoneme category.

## Code and Data
- **Code:** https://github.com/coganlab/cross_patient_speech_decoding
- **Data:** DABI https://dabi.loni.usc.edu/dsi/7RKNEJSWZXFW (restricted access, IRB Pro0072892)
- **Duraivel preprocessing MATLAB:** https://doi.org/10.5281/zenodo.8384194

## Open Questions
1. **Does this extend to continuous speech?** The current task is isolated 3-phoneme non-word repetition. Real BCIs need to decode running speech with coarticulation, prosody, and variable timing.
2. **Can nonlinear alignment methods (e.g., optimal transport, deep CCA, LFADS, contrastive learning) outperform linear CCA?** The authors flag LFADS and MARBLE as promising directions.
3. **What happens with more patients?** The log-fit scaling curve (r^2 = 0.98) suggests continued gains. At what n does performance plateau?
4. **Can alignment be learned with less labeled data, or unsupervised?** The current CCA requires condition-averaged labeled trials from the target patient. Can alignment be bootstrapped from unlabeled data or a smaller calibration set?
5. **Does the approach work for attempted/imagined speech?** Critical for the actual BCI use case in locked-in patients.
6. **Can transformer architectures replace the seq2seq RNN?** The temporal convolution + GRU architecture is dated; attention-based models may better capture long-range temporal dependencies.
7. **What is the optimal dimensionality for the shared latent space?** Currently determined by 90% variance threshold for PCA. Would a fixed lower-dimensional space be better?
8. **Can the alignment be made universal rather than pairwise?** A single shared space (like a foundation model embedding) would be more scalable than N pairwise alignments.
9. **How does chronic vs. acute recording stability affect alignment?** Neural representations may drift over time in chronic implants.
10. **Can this framework incorporate heterogeneous recording modalities?** E.g., align uECoG patients with standard ECoG or intracortical array patients, since the alignment operates on latent dynamics, not raw electrode signals.
11. **What is the role of the full English phoneme inventory?** Only 9 phonemes tested (4 vowels, 5 consonants). Would alignment hold for all ~40 English phonemes?
12. **Can the temporal information discarded by PCA/CCA be preserved?** Dynamical systems approaches that model temporal evolution could yield better alignment and decoding.
