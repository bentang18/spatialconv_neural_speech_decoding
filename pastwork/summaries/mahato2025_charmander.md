# Mahato et al. 2025 - Charmander: Scalable Self-Supervised iEEG Modeling

## Citation
Mahato, A., et al. (2025). A scalable self-supervised method for modeling human intracranial recordings during natural behavior. *NeurIPS 2025 Workshop*. AJILE12 + Brain Treebank datasets.

## Setup
- **Recording modalities**: Surface ECoG (64-106 electrodes) + depth sEEG (0-40 electrodes per patient)
- **Subjects**: AJILE12 dataset (12 ECoG patients, naturalistic behavior) + Brain Treebank (10 sEEG patients)
- **Tasks**: Activity classification (AJILE12), pitch/volume regression (Brain Treebank)
- **Scale**: 8M parameters (primary), also tested 33M and 142M variants

## Architecture
- **Input tokenization**: Temporal patching with P=5 samples per patch. Each patch projected via learnable linear layer to d=128
- **Token identity**: Per-channel per-participant learnable embedding + RoPE with log-spaced frequencies spanning 0.1ms to 20s
- **Latent bottleneck**: Perceiver-style architecture. H=32 "virtual channels" x M=8 "virtual timesteps" = 256 learnable latent tokens. Cross-attention maps from latent tokens to input tokens (latents attend to full input sequence)
- **Backbone**: 16 self-attention layers operating on the 256 latent tokens
- **Decoder**: Cross-attention from input token positions back to latent representations (for reconstruction)
- **Total**: 8M params (primary config)
- **Scaling**: Also tested 33M and 142M -- NO downstream benefit from scaling beyond 8M

## SSL/Pretraining
- **Objective**: Masked channel reconstruction. 50% of channels randomly masked. Reconstruct masked channels' raw voltage from unmasked channels via cross-attention decoder
- **Loss**: MSE on raw voltage reconstruction
- **Training**: LAMB optimizer, lr=3.125e-3, 300 epochs
- **Masking**: Random 50% channel masking (spatial, not temporal). Validates spatial masking for iEEG SSL (complementary to BIT's temporal masking)

## Cross-Patient Handling
- **Per-channel per-participant learnable embeddings**: No spatial coordinates used. Each channel for each participant gets a unique learned embedding vector. This is data-hungry -- requires learning the spatial identity of every electrode from scratch
- **New participant protocol**: Initialize new per-channel embeddings for the unseen participant. 50-epoch warm-up phase with encoder frozen (only embeddings train). Then unfreeze last 4 self-attention layers for fine-tuning
- **Scaling with patients**: MP8 (8 patients pretrained) > MP3 > MP1. More pretraining patients consistently improves downstream transfer
- **No atlas or coordinate information**: Spatial relationships learned entirely from data co-occurrence patterns

## I/O Features
- **Input**: Raw voltage time series, temporally patched (P=5 samples)
- **Output**: Activity classification labels (AJILE12) or continuous pitch/volume scalars (Brain Treebank)
- **Spatial**: Per-channel identity via learned embeddings (no coordinates)
- **Temporal**: RoPE with log-spaced frequencies (captures multi-scale temporal structure from milliseconds to tens of seconds)

## Key Results
| Task | Charmander (MP8) | Previous Best |
|---|---|---|
| AJILE12 activity F1 (novel finetuned) | **0.869** | 0.793 (Poyo+ MP3) |
| Brain Treebank pitch | **0.88** | SOTA |
| Brain Treebank volume | **0.93** | SOTA |

Key findings:
- **Model scaling plateau**: 8M -> 33M -> 142M provides NO downstream benefit. The data bottleneck dominates, not model capacity. Critical validation that v12's ~170K params is not undersized for the available data regime
- **More pretraining patients helps**: MP8 > MP3 > MP1 consistently. Cross-patient pretraining works for iEEG
- **50% channel masking is effective**: Spatial masking SSL objective works for intracranial recordings
- **Perceiver bottleneck works**: 256 latent tokens (32 x 8) successfully compress variable-length electrode arrays into a fixed representation

## v12 Comparison

**Perceiver bottleneck converges with v12's VE cross-attention.** Charmander's 32 "virtual channels" are architecturally analogous to v12's 16 virtual electrodes -- both use cross-attention to compress variable electrode counts into a fixed-size latent space. The key difference is what grounds the latent positions:

- **Charmander**: Learned latent tokens with no spatial prior. Must learn spatial structure entirely from data. Works when data is abundant (AJILE12 has extensive recordings per patient)
- **v12**: Atlas-grounded VE positions (16 Brainnetome ROIs) with distance-biased cross-attention. Spatial prior from Brainnetome population-average anatomy (N=40, somatotopic motor/sensory subdivisions). Critical for v12's data-scarce regime (~1 min/patient epoched, ~15 min continuous)

**Model scaling ceiling validates v12's parameter budget.** The finding that 8M -> 142M provides zero downstream benefit is strong evidence that model capacity is not the bottleneck for intracranial neural decoding. v12's ~170K params is well-calibrated for the available data (~456 min raw, ~11 min epoched across patients).

**Key differences:**
- Charmander uses learned per-channel per-participant embeddings (no coordinates). This is data-hungry and doesn't generalize to new electrode positions without retraining. v12's MNI coordinates + Fourier PE provide immediate spatial identity for any electrode placement
- Charmander's 50% spatial masking validates the concept for iEEG SSL, but v12 plans temporal span masking following BIT (speech is temporal; spatial masking teaches interpolation, temporal masking teaches dynamics)
- Charmander operates on raw voltage; v12 uses pre-extracted HGA
- Charmander's new-participant protocol (freeze encoder, warm-up embeddings, then unfreeze last 4 layers) is a reasonable transfer recipe. v12's approach (freeze backbone, train 134 per-patient params) is lighter and may be preferable for ~1 min/patient data

**What to import:**
- The log-spaced RoPE spanning 0.1ms-20s is a clever multi-scale temporal encoding. Could complement v12's Fourier PE on the temporal dimension
- The warm-up-then-unfreeze transfer protocol is a practical recipe if v12 needs more than just per-patient layer adaptation
- The 50% channel masking rate is a useful reference point for spatial augmentation during v12 SSL (electrode subset sampling at 50-100%)
