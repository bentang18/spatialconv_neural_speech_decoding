# Wang et al. 2023 - BrainBERT

## Citation
Wang, C., Subramaniam, V., Yaari, A.U., Kreiman, G., Hajiesmaili, M., Keshishian, M., & Mesgarani, N. (2023). BrainBERT: Self-Supervised Representation Learning for Intracranial Recordings. *ICLR 2023*.

## Setup
- **Recording modality**: sEEG (stereo-EEG), bipolar re-referenced
- **Subjects**: 10 subjects from the Brain Treebank dataset (naturalistic English speech listening + production)
- **Data**: ~43.7h total, 1688 electrodes raw (1249 after Laplacian bipolar re-referencing), 19 subjects for pretraining, 7 for downstream evaluation
- **Tasks**: Downstream probing on part-of-speech tagging, word frequency, word position, sentence length, word identity, pitch, volume, speech onset detection
- **Sampling**: 512 Hz after preprocessing

## Architecture
- **Processing unit**: Per-electrode. Each electrode is an independent sample — no multi-electrode model, no spatial interactions
- **Input representation**: Raw voltage waveform → STFT spectrogram (40 frequency bins x time) or superlet spectrogram. Each 2-second window produces a 2D time-frequency image
- **Encoder**: 6-layer Transformer (d=768, 12 attention heads, ~28M parameters). Standard BERT architecture applied to the spectrogram
- **Tokenization**: Time-frequency spectrogram treated as a sequence of tokens (each token = one time step across all 40 frequency bins)
- **No spatial model**: Electrodes processed completely independently. Spatial relationships between electrodes are never modeled

## SSL/Pretraining
- **Objective**: Masked spectrogram prediction. Mask time columns and frequency rows of the input spectrogram, predict original values at masked positions
- **Masking strategy**: p_mask=0.05 probability of starting a mask span, span length 1-5 (time or frequency), p_identity=0.10 (leave masked token unchanged), p_replace=0.10 (replace with random spectrogram value). Remaining 80% replaced with learnable [MASK] token
- **Loss**: Novel **content-aware L1 loss**. Key insight: 68% of z-scored spectrogram values are near zero — standard MSE/L1 is dominated by reconstructing silence. Content-aware loss upweights reconstruction error for high-activation bins (bins with |z| > threshold contribute more to loss). This forces the model to focus on neural events rather than background
- **Training**: LAMB optimizer, lr=1e-4, batch=256, 500K steps. All electrodes from all 19 pretraining subjects pooled into one dataset regardless of subject/brain region
- **Spectrogram comparison**: STFT slightly outperforms superlet transform for downstream tasks

## Cross-Patient Handling
- **No explicit mechanism**: Per-electrode processing completely sidesteps the cross-patient spatial alignment problem. Every electrode is treated as an independent data point
- **Implicit cross-patient transfer**: All electrodes from all subjects are pooled during SSL pretraining. The model learns a universal time-frequency representation that generalizes across subjects and brain regions
- **Leave-one-subject-out validation**: Hold-one-out pretraining (exclude one subject's electrodes) shows negligible degradation on that subject's downstream performance — cross-subject SSL transfer works
- **No per-patient layers**: Not needed since there is no spatial model to align

## I/O Features
- **Input**: 2-second raw voltage window per electrode → STFT spectrogram (40 freq bins x T time steps)
- **Output (SSL)**: Reconstructed spectrogram at masked positions
- **Output (downstream)**: Frozen pretrained encoder → linear probe on [CLS] token embedding for classification tasks
- **No electrode coordinates, no atlas, no spatial PE**

## Key Results
- **Overall**: 0.83 AUC averaged across downstream tasks (STFT fine-tuned) vs 0.63 baseline (spectrogram + logistic regression)
- **Pretraining benefit**: +0.23 AUC over random initialization. Pretrained BrainBERT consistently outperforms training from scratch
- **Data efficiency**: 5x — pretrained model with 150 labeled samples matches baseline performance with 1000 samples
- **Content-aware loss**: +0.01-0.05 AUC consistent improvement over standard L1/MSE across tasks
- **Cross-subject transfer**: Leave-one-out pretraining (N-1 subjects) performs comparably to all-subject pretraining — the representation generalizes
- **Task-specific findings**: Best on speech onset (AUC ~0.95), pitch (0.88), volume (0.85). Weaker on word identity (0.72) and word position (0.71)
- **Probing depth**: Middle layers (3-4) most informative for linguistic tasks; early/late layers better for acoustic tasks

## v12 Comparison

**What BrainBERT does that v12 should import:**
1. **Content-aware loss**: Directly applicable to v12's planned temporal masking SSL. HGA also has many near-zero time bins (no speech activity). Upweighting reconstruction of high-activation bins would focus SSL on learning speech dynamics rather than reconstructing silence. Simple to implement: scale MSE loss by |z| or indicator(|z| > threshold)
2. **SSL generalizes cross-subject on intracranial field potentials**: BrainBERT is sEEG (field potentials like HGA), not spikes. Proves the SSL-on-pooled-electrodes recipe works for the same signal type v12 will use. Cross-subject transfer works even without explicit alignment

**What BrainBERT lacks that v12 addresses:**
1. **No multi-electrode model**: Per-electrode processing cannot capture spatial patterns (e.g., traveling waves, population coding). v12's VE cross-attention explicitly models spatial relationships between electrodes via atlas-grounded distance bias
2. **No spatial alignment**: Works for single-electrode probing but cannot do multi-electrode decoding tasks (e.g., phoneme classification from a neural population). v12's entire architecture is built around mapping variable electrode arrays into a common spatial representation
3. **No per-patient layers**: Not needed for per-electrode processing, but essential for v12 where the shared backbone must disentangle patient-specific gain/offset from shared neural dynamics
4. **No coordinates**: Ignores electrode location entirely. v12 uses MNI coordinates + Fourier PE + learned rigid correction to encode spatial identity
5. **sEEG vs uECoG**: BrainBERT uses sEEG bipolar references (depth electrodes). v12 targets uECoG (surface grid) with denser, more regular spatial sampling

**Key architectural contrast**: BrainBERT avoids the hard cross-patient alignment problem by treating each electrode independently. This is clever for probing studies but fundamentally cannot scale to population-level decoding. v12 tackles the alignment problem head-on with atlas-grounded VE cross-attention. BrainBERT's content-aware loss is the most directly importable idea.

**Data regime**: 43.7h sEEG is comparable to v12's planned sEEG SSL corpus (~16.7h CoganLab sEEG). BrainBERT shows SSL benefits emerge at this scale for per-electrode tasks. v12's multi-electrode model may need more data to learn spatial structure, but the temporal representation learning should transfer.
