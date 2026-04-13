# Zhang et al. 2023 - Brant

## Citation
Zhang, D., Yang, Y., Wang, S., Zheng, Y., Du, N., Liu, X., Chen, T., Gao, B., & Li, G. (2023). Brant: Foundation Model for Intracranial Neural Signal. *NeurIPS 2023*.

## Setup
- **Recording modality**: sEEG (stereo-EEG), raw voltage waveforms, 2000 Hz sampling
- **Subjects**: 9 subjects, Zhejiang University Hospital, drug-resistant epilepsy patients
- **Data**: 2528 hours (1.01 TB), by far the largest intracranial pretraining corpus at time of publication
- **Electrode counts**: Variable per patient (not specified per-patient, but total across all subjects is large)
- **Tasks**: Neural signal forecasting, signal imputation, seizure detection (downstream). No speech decoding
- **Preprocessing**: Notch filter (50 Hz + harmonics), bandpass 0.5-300 Hz, resampled to 250 Hz for pretraining

## Architecture
- **Factored Transformer** with separate temporal and spatial stages (total ~505.69M params at largest scale; also 70M and 245M variants)
- **Temporal encoder** (12 layers, d=2048, 16 attention heads): Processes each channel independently. Input signal is divided into non-overlapping temporal patches (200 samples = 0.8s at 250 Hz). Each patch is linearly embedded into d=2048
- **Spatial encoder** (5 layers, d=2048, 16 attention heads): After temporal encoding, all channels at each time step are passed through spatial self-attention. Channels attend to each other — but with NO spatial identity
- **Frequency encoding**: Power spectral density (PSD) computed in 8 frequency bands (delta through high-gamma). Band powers are mapped to learnable frequency embeddings, weighted by PSD, and added to the temporal patch embeddings. This provides spectral context
- **Positional encoding**: Learnable temporal PE only. **No spatial PE, no electrode coordinates, no atlas, no channel identity**. Channels are an anonymous, order-invariant bag in the spatial encoder
- **Scaling**: Three model sizes tested: 70M, 245M, 505M. Performance scales with model size on most tasks

## SSL/Pretraining
- **Objective**: Masked Autoencoder (MAE). Random masking across both time and space dimensions (40% mask ratio). Predict raw waveform patches at masked positions
- **Loss**: MSE on reconstructed vs original waveform patches
- **Training**: Adam optimizer, 750K update steps, cyclic learning rate schedule 3e-6 → 1e-5, trained on 4x A100 GPUs for 2.8 days
- **Masking**: 40% of all time-space patches are randomly masked. Both temporal and spatial positions can be masked. No special span masking or structured masking strategy
- **No per-patient layers during pretraining**: All channels from all patients are pooled. Channel identity is not preserved

## Cross-Patient Handling
- **No explicit cross-patient mechanism**: No per-patient layers, no spatial coordinates, no atlas-based alignment, no electrode identity
- **Channels are anonymous**: The spatial encoder treats channels as a bag — channel 1 of patient A has no distinguished relationship to channel 1 of patient B. Order is arbitrary
- **Transfer approach**: Pretrained encoder is frozen or fine-tuned on downstream tasks. For new subjects (31 unseen subjects tested), the same architecture is applied — works for spatially-agnostic tasks but fails for tasks requiring spatial specificity
- **Cross-patient speech decoding**: FAILS. Confirmed by MIBRAIN (Wu et al. 2025), who tested Brant as a baseline for cross-patient consonant decoding and found it near-chance. The lack of spatial identity prevents the model from learning consistent brain-region-to-function mappings across patients

## I/O Features
- **Input**: Raw voltage waveform, shape (C, T) where C = channels, T = time samples at 250 Hz. Segmented into non-overlapping patches of 200 samples
- **Temporal patch embedding**: Linear projection of each 200-sample patch to d=2048
- **Frequency embedding**: PSD in 8 bands → weighted sum of learnable band embeddings, added to temporal embeddings
- **Output (SSL)**: Reconstructed waveform patches at masked positions
- **Output (downstream)**: Task-specific heads (linear classifiers for seizure detection, regression heads for forecasting/imputation)

## Key Results
- **Seizure detection**: 91.17% accuracy (binary), outperforming prior methods on the same dataset. Generalizes to 31 unseen subjects from different hospitals
- **Signal forecasting**: MSE 0.0261 (0.4s prediction horizon), outperforms N-BEATS, Informer, PatchTST, and other forecasting baselines
- **Signal imputation**: MSE 0.0218 (reconstructing missing channels), outperforms BRITS and SAITS
- **Scaling**: 505M > 245M > 70M on most tasks. Performance scales log-linearly with model size
- **Generalization to unseen subjects**: Works well for seizure detection and forecasting on 31 subjects not seen during pretraining — but these are spatially-agnostic tasks
- **Cross-patient speech**: Near-chance when tested by MIBRAIN as baseline for consonant classification across patients

## v12 Comparison

**What Brant proves (the anti-pattern):**
1. **Scale alone is insufficient without spatial identity**: 505M parameters and 2528 hours of data — orders of magnitude beyond what v12 will have — yet Brant fails at cross-patient speech decoding. The bottleneck is not data or model size but the absence of a spatial alignment mechanism. This is the strongest evidence for v12's core design decision: atlas-grounded VE cross-attention + MNI coordinates + per-patient layers are necessary, not optional
2. **Factored temporal→spatial attention is the right pattern**: Brant's two-stage design (temporal encoding per channel, then spatial attention across channels) matches v12's factored architecture and is independently validated by seegnificant (+0.06 R², 5.5x faster vs joint 2D). The failure is not in the attention pattern but in the spatial anonymity
3. **Anonymous spatial attention learns nothing transferable for speech**: Without electrode identity (coordinates, atlas labels, or even consistent ordering), the spatial encoder cannot learn "motor cortex channel 5 of patient A encodes the same articulatory features as motor cortex channel 12 of patient B." Each patient's spatial structure must be re-learned from scratch

**What v12 addresses that Brant lacks:**
1. **Atlas-grounded VE cross-attention**: v12 maps variable electrode arrays into 16 fixed Brainnetome atlas positions via distance-biased cross-attention. This provides the spatial identity Brant is missing
2. **MNI coordinates + Fourier PE**: Every electrode has a spatial identity based on its anatomical location. Brant has no equivalent
3. **Per-patient layers**: v12's diagonal normalization (128 params/patient) handles gain/offset variation. Brant has no per-patient mechanism
4. **Learned rigid correction (delta/omega)**: v12 corrects systematic MNI registration error (4-8mm). Brant does not use coordinates at all

**What Brant does well that v12 should note:**
1. **Frequency encoding via PSD-weighted band embeddings**: A lightweight way to inject spectral context. Not directly needed for v12 (input is already HGA, a single frequency band), but the principle of spectral-aware embeddings could be useful if v12 ever processes broadband signals
2. **Scale validation**: Proves that intracranial Transformers can scale to 500M+ params and 2500+ hours. v12's ~170K params and ~24h planned SSL corpus are at the opposite extreme — architectural efficiency must compensate for data scarcity
3. **MAE pretraining recipe**: 40% mask ratio, MSE loss, ~750K steps is a reasonable starting point. But note: Neuro-BERT (Wu 2024) shows spatiotemporal MSE degrades above 10% mask ratio — Brant's success at 40% may be due to its factored structure (temporal patches masked independently)

**Key takeaway**: Brant is the clearest existence proof that you cannot brute-force cross-patient neural decoding with scale alone. Spatial identity — whether through coordinates, atlas labels, or per-patient layers — is the decisive factor. v12's entire architecture is designed around this insight.
