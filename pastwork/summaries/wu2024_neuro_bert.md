# Di Wu et al. 2024 - Neuro-BERT

## Citation
Wu, D., Li, X., Bhatt, D., & Li, G. (2024). Neuro-BERT: Rethinking Masked Autoencoding for Self-Supervised Neurological Pretraining. *arXiv preprint*.

## Setup
- **Recording modalities**: Scalp EEG and EMG (NOT intracranial)
- **Datasets**:
  - **SleepEDF**: 500 subjects, scalp EEG, 100 Hz, sleep staging (5 classes). Two splits: SleepEDF-20 (20 subjects) and SleepEDF-78 (78 subjects)
  - **Epilepsy**: 500 subjects, single-channel scalp EEG, seizure detection (5 classes, binary in some evaluations)
  - **Ninapro DB2**: 10 subjects, 12-channel EMG, 2000 Hz, hand gesture recognition (17 classes + rest)
- **NOT intracranial**: All datasets are surface EEG or EMG. No sEEG, ECoG, or uECoG
- **Preprocessing**: Segment into fixed-length windows (30s for sleep, variable for others), z-score normalization

## Architecture
- **ViT-style encoder**: 4 Multi-head Self-Attention (MSA) blocks, d=128, FFN inner dim=512, ~0.798M total parameters. Lightweight by design
- **Patch tokenization**: 1D convolution maps raw signal segments into patch embeddings. Each patch covers a fixed temporal window
- **Learnable [M] mask token**: Masked positions are replaced with a shared learnable embedding vector (not zeroed or noised)
- **Positional encoding**: Standard learnable 1D positional embeddings for temporal position within the window
- **Decoder**: Linear projection (single layer) from encoder output back to signal space. Key finding: linear decoder suffices for SSL — deep decoders do not improve downstream performance
- **No multi-channel spatial model**: Channels are concatenated or processed independently depending on dataset

## SSL/Pretraining
- **Objective**: Fourier Inversion Prediction (FIP) — the core contribution
  - Masked patches are predicted in the **frequency domain**: model predicts DFT magnitude and DFT phase separately for each masked patch
  - Predicted magnitude + phase are inverse-DFT'd back to time domain
  - Final loss is **MSE between reconstructed time-domain signal and original** (NOT frequency-domain MSE)
  - This two-step process (predict freq → IDFT → MSE in time) acts as implicit regularization
- **Why FIP over direct MSE?**: Standard spatiotemporal MSE reconstruction degrades sharply at mask ratios >10%. FIP is robust across 10-60% mask ratios. The frequency-domain prediction provides a structured intermediate representation that regularizes reconstruction
- **FIP as plug-in**: FIP can replace the reconstruction head of any MAE method. When applied to existing MAE architectures, it consistently improves downstream performance by ~0.5-1pp
- **Masking**: Random patch masking. Tested mask ratios from 10% to 60%. FIP performs stably across this range; direct MSE degrades above 10%
- **Training details**: AdamW optimizer, cosine LR schedule, standard augmentations (not heavily specified)

## Cross-Patient Handling
- **None**: No cross-patient alignment mechanism. Single-dataset, mixed-subject training and evaluation
- **All subjects pooled**: During SSL pretraining, all subjects' data is mixed without subject identity
- **No per-patient layers, no coordinates, no atlas**
- **Evaluation**: Standard train/test splits within each dataset. Subject overlap between train and test is handled by dataset-specific protocols (SleepEDF uses subject-wise splits)

## I/O Features
- **Input**: Raw 1D signal segment (single or multi-channel), segmented into fixed-length windows
- **Patch embedding**: 1D conv tokenizes signal into temporal patches
- **Output (SSL)**: DFT magnitude + DFT phase per masked patch → IDFT → time-domain reconstruction → MSE loss
- **Output (downstream)**: Frozen or fine-tuned encoder → linear classifier for task-specific labels
- **No spatial features, no frequency-domain input features** (unlike Brant's PSD encoding)

## Key Results
- **Sleep staging (SleepEDF-78)**: 86.53% accuracy, outperforming TS-TCC (83.00%), TS2Vec (84.56%), and other contrastive/MAE methods
- **Gesture recognition (Ninapro)**: 94.28% accuracy (12-channel EMG)
- **Seizure detection (Epilepsy)**: 99.34% accuracy (binary)
- **FIP vs alternatives**: +2-3pp over contrastive methods (SimCLR, TS-TCC), +1-2pp over standard MaskedAE across all datasets
- **Semi-supervised (1% labels)**: +30pp over random initialization on SleepEDF. FIP provides massive gains in extremely low-label regimes
- **Mask ratio robustness**: FIP maintains stable performance from 10-60% mask ratio. Standard spatiotemporal MSE drops by 3-5pp from 10% to 40%. This is the core empirical finding
- **Linear decoder sufficiency**: Adding decoder depth (2-4 layers) does not improve downstream performance over a single linear layer. The encoder learns the representation; the decoder just needs to be a consistent training signal
- **FIP as plug-in**: Replacing MSE reconstruction with FIP in TS-TCC, TS2Vec, and PatchTST improves downstream accuracy by 0.5-1.2pp each

## v12 Comparison

**What Neuro-BERT contributes to v12 design:**
1. **Mask ratio fragility warning**: The finding that spatiotemporal MSE degrades above 10% mask ratio is directly relevant to v12's planned 50% temporal masking SSL. v12 plans BIT-style temporal span masking with MSE reconstruction — if the same fragility applies to HGA spectrograms, 50% could be too aggressive. Mitigation options: (a) use FIP-style frequency-domain prediction, (b) validate mask ratio sensitivity empirically on HGA, (c) use content-aware L1 loss (BrainBERT) which may be more robust
2. **FIP is not directly useful for HGA**: v12's input is already a frequency-domain feature (70-150 Hz high-gamma amplitude). Predicting DFT of HGA would be predicting the spectrum of a spectrum — circular and unlikely to help. FIP makes sense for raw voltage or broadband signals. For HGA, direct temporal MSE (possibly content-aware) is more natural
3. **Linear decoder suffices for SSL**: Confirms that the SSL reconstruction head can be a single linear layer. No need for a deep decoder during pretraining. This simplifies v12's SSL implementation
4. **Semi-supervised gains**: +30pp at 1% labels suggests SSL pretraining would be extremely valuable for v12's data-scarce regime (46-178 trials per patient). Even modest SSL gains compound when labeled data is this limited

**What Neuro-BERT lacks relative to v12:**
1. **Not intracranial**: All results are on scalp EEG/EMG. Transfer of findings to intracranial HGA is uncertain — different signal characteristics, SNR, and spatial resolution
2. **No spatial model**: Single-channel or concatenated multi-channel. Cannot model electrode interactions
3. **No cross-patient mechanism**: No per-patient layers, no coordinates, no atlas. Not designed for the variable-electrode cross-patient problem
4. **Tiny model**: 0.798M params. Appropriate for scalp EEG but the findings on decoder depth and mask ratio may not scale to v12's setting with richer intracranial signals

**Relationship to MIBRAIN**: Neuro-BERT is by the same first author (Di Wu) as MIBRAIN. MIBRAIN builds on Neuro-BERT's SSL recipe but adds atlas-grounded region mapping for cross-patient sEEG — the key missing ingredient. FIP was apparently dropped in MIBRAIN in favor of region-masked MAE with MSE, suggesting the frequency-domain prediction was less important than the spatial alignment mechanism.

**Key takeaway**: The mask ratio fragility finding is the most actionable result for v12. If v12's temporal masking SSL uses direct MSE reconstruction, start at low mask ratios (10-20%) and validate before scaling up to the planned 50%. Content-aware loss (BrainBERT) may provide an alternative path to high mask ratios without FIP.
