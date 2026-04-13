# Oganesian et al. 2025 - BarISTA: Brain Scale Informed Spatiotemporal Representation

## Citation
Oganesian, V., Rouhi, A., Pesaran, B., & Shanechi, M.M. (2025). BarISTA: Brain Scale Informed Spatiotemporal Representation of Human Intracranial Neural Activity. *NeurIPS 2025*. Code: github.com/ShanechiLab/BaRISTA.

## Setup
- **Recording modality**: sEEG, 2048 Hz
- **Subjects**: 10 sEEG patients from Brain Treebank, 26 sessions, 29.2 hours total pretraining data
- **Tasks**: Binary event detection (sentence onset, speech production) -- relatively easy compared to phoneme decoding
- **Compute**: <4 hours training on 4x RTX 6000 Ada

## Architecture
- **Input**: Channel-wise temporal patching -- 250ms windows = 512 samples at 2048 Hz per channel
- **Temporal encoder**: 5-layer dilated CNN producing d_out=64 per patch. Applied independently per channel. Outperforms linear projection by +6-9pp AUC
- **Projection**: Linear to d=64 shared embedding dimension
- **Spatial embedding**: Learnable lookup from discretized spatial categories. Three scales tested:
  - Channel-level: LPI integer coordinates discretized into 3 embedding tables (one per axis), summed
  - Parcel-level: Destrieux atlas, 1 learnable embedding per parcel. Channels in same parcel share embedding
  - Lobe-level: Desikan-Killiany atlas, 1 learnable embedding per lobe
- **Token sequence**: Interleaved flat sequence -- all channels x all timesteps concatenated (combined spatiotemporal, NOT factored)
- **Transformer**: 12-layer, d=64, 4 attention heads, RoPE applied to temporal dimension only. Combined space-time attention (every token attends to every other token regardless of spatial/temporal identity)
- **Predictor**: 5-layer MLP on output tokens
- **Scale**: ~1M parameters total. ZERO per-patient parameters

## SSL/Pretraining
- **Objective**: JEPA-style masked latent token reconstruction (predict masked token representations in latent space, NOT observation space)
- **Masking strategy**: Spatially-guided masking (~30%). Select random spatial categories (parcels/lobes/channels depending on encoding scale), then mask ALL tokens from those categories across ALL timesteps. Spatial-only masking -- no temporal masking component (authors acknowledge as limitation)
- **Target tokenizer**: Exponential moving average (momentum 0.996) of the encoder weights, following IJEPA/BYOL
- **Loss**: MSE in latent space (not reconstruction of raw signal)
- **SSL gain**: +8-25pp AUC over random initialization

## Cross-Patient Handling
- **No per-patient parameters**: All patients share the same model weights. Spatial identity encoded entirely through learnable embedding lookup from atlas parcels
- **Hold-out subject generalization**: Only ~2pp AUC degradation when evaluating on a held-out subject not seen during pretraining. This works because spatial embeddings generalize across patients (same parcel = same embedding)
- **KEY FINDING -- parcel-level >> channel-level**: Parcel-level encoding (Destrieux) beats channel-level (LPI coordinates) by +8-10pp AUC. Atlas-level spatial abstraction is more informative than fine-grained coordinate identity for cross-patient generalization. Lobe-level is comparable to parcel-level (not much gained from finer parcellation)

## I/O Features
- **Input**: Raw sEEG voltage at 2048 Hz, no explicit HGA extraction (temporal CNN handles feature extraction implicitly)
- **Output**: Binary classification (sentence onset / speech production detection)
- **Temporal resolution**: 250ms patches
- **Spatial resolution**: Varies by encoding scale (channel / parcel / lobe)

## Key Results
| Configuration | Sentence Onset AUC | Speech AUC |
|---|---|---|
| Parcel encode + channel mask (best) | 0.862 | 0.869 |
| Channel encode + channel mask | ~0.77 | ~0.78 |
| Parcel encode + parcel mask | 0.854 | 0.852 |
| Parcel encode + lobe mask | 0.845 | 0.850 |
| Combined attention | 0.836 | 0.847 |
| Factored attention | 0.828 | 0.825 |
| Random init (no SSL) | 0.611 | 0.621 |
| Linear projection (no dilated CNN) | ~0.80 | ~0.81 |

Key findings:
- Combined spatiotemporal attention > factored (contradicts seegnificant's finding that factored > combined)
- SSL provides substantial benefit (+8-25pp over random init)
- Dilated CNN temporal encoder > linear projection (+6-9pp)
- Spatial masking scale should match encoding scale (channel mask works well with parcel encoding)

## v12 Comparison

**Directly validates the atlas-level spatial abstraction approach.** BarISTA's core finding -- parcel-level >> channel-level encoding -- is the strongest empirical evidence that atlas-grounded spatial representations outperform coordinate-based ones for cross-patient iEEG models. This directly supports v12's VE cross-attention design over raw MNI coordinate approaches.

**Key parallels:**
- BarISTA's parcel-level embedding = hard atlas assignment. v12's distance-biased VE cross-attention to 16 Brainnetome positions = soft atlas assignment. v12 is strictly more general (continuous soft assignment vs discrete hard boundaries), which should handle electrodes near parcel borders better
- Both use atlas anatomy to define a shared spatial vocabulary across patients
- Both show that you don't need fine-grained per-electrode coordinates -- population-average anatomy suffices

**Key differences:**
- BarISTA uses ZERO per-patient parameters. Works for binary detection but v12's 134-param diagonal normalization (scale+bias) is cheap insurance for the harder 9-class phoneme decoding task
- BarISTA uses combined spatiotemporal attention; v12 uses factored (VE self-attn then temporal self-attn). The combined > factored finding here contradicts seegnificant but may be task-dependent (binary detection vs continuous regression). For v12's longer sequences and harder task, factored may still win on efficiency
- BarISTA masks spatially only (no temporal masking). v12's SSL plan uses temporal span masking following BIT. BarISTA authors acknowledge spatial-only as a limitation
- BarISTA's JEPA latent targets (MSE in latent space, EMA target encoder) are importable for v12's SSL stage. Avoids the issue of reconstructing noisy raw neural signals
- BarISTA operates on raw voltage; v12 operates on pre-extracted HGA (different preprocessing assumptions)

**Caution:** BarISTA's tasks (binary event detection) are dramatically easier than v12's 9-class phoneme decoding. The finding that zero per-patient parameters suffice should not be extrapolated to harder tasks. BIT and seegnificant both show per-patient layers are critical for more demanding decoding.
