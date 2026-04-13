# Antoniades et al. 2024 - Neuroformer: Multimodal and Multitask Generative Pretraining for Brain Data

## Citation
Antoniades, A., Wickens, J., Lenington, S., Bonhomme, S., Bhatt, P., & Bhatt, N. (2024). Neuroformer: Multimodal and Multitask Generative Pretraining for Brain Data. *ICLR 2024*.

## Setup
- **Recording modalities**: Two-photon calcium imaging (primary), also demonstrates on Neuropixels extracellular recordings
- **Species**: Mice, Allen Institute datasets (Visual Coding, Visual Behavior, Neuropixels)
- **Data scale**: Varies by experiment -- single sessions with hundreds of neurons. NOT a large-scale multi-session model
- **Tasks**: Spike prediction (next-interval neural activity), behavioral decoding (running speed, pupil diameter), stimulus classification (visual gratings, natural scenes)
- **Compute**: 40-100M parameters depending on configuration

## Architecture
- **Type**: GPT-style autoregressive Transformer with multimodal fusion
- **Neural tokenization**: Spike events discretized into (neuron_id, time_bin, spike_count) tuples. Sorted by time, then by neuron within each time bin. Vocabulary = neuron_ids
- **Temporal encoding**: Learned positional embeddings for time bins + interval embeddings for the prediction horizon
- **Multimodal fusion**: Cascading cross-attention (Perceiver-style) for integrating multiple input streams:
  - Neural activity (primary)
  - Visual stimuli (image features from pretrained vision encoder)
  - Behavioral variables (running speed, pupil)
  - Each modality has its own encoder; cross-attention layers fuse them hierarchically
- **Contrastive alignment**: CLIP-style contrastive loss aligns neural representations with visual stimulus representations. Neural embedding of a trial should be closer to its corresponding stimulus embedding than to other stimuli. Applied across neural-visual and neural-behavioral pairs
- **Backbone**: 8 Transformer layers, E=512, 8 attention heads
- **Total**: 40-100M parameters depending on the number of modalities and dataset size
- **No spatial encoding**: Neuron identity is purely via learned embedding lookup (no coordinates, no atlas, no spatial structure)

## SSL/Pretraining
- **Self-supervised objective**: Autoregressive next-interval spike prediction. Given neural activity in bins [0, t], predict activity in bin [t+1]. Applied per-neuron autoregressively (predict each neuron's spike count conditioned on all neurons in previous bins + already-predicted neurons in current bin)
- **Multimodal pretraining**: When stimulus/behavioral data is available, CLIP contrastive loss is added. The model learns aligned representations across modalities
- **Transfer finding**: **Pretrained on 1% of data > non-pretrained on 10% of data** for downstream decoding. This is the headline result -- pretraining provides 10x data efficiency
- **Fine-tuning**: Add task-specific linear heads on top of pretrained representations. Freeze backbone initially, optionally unfreeze

## Cross-Patient Handling
- **Single-dataset only**: All experiments are within a single recording session or a single animal. No cross-animal or cross-session transfer is demonstrated
- **No per-session/per-animal adaptation**: The model is trained and evaluated on the same session. Generalization across sessions is not addressed
- **Neuron identity via learned embeddings**: Each neuron gets a unique learned ID. These don't transfer across sessions (different neurons) -- would need relearning

## I/O Features
- **Input**: Discretized spike events (neuron_id, time_bin, count) + optional visual stimuli + optional behavioral variables
- **Output**: Next-interval spike counts (generative), behavioral predictions (discriminative), stimulus classification (discriminative)
- **Temporal**: Learned positional embeddings, autoregressive generation
- **Spatial**: None -- neurons are an unordered set with learned IDs

## Key Results
| Task | Neuroformer | Comparison |
|---|---|---|
| Spike prediction (bits/spike) | 0.68-0.82 | Better than LSTM baseline |
| Speed prediction (r) | 0.95-0.97 | Better than population vector decoding |
| Pupil prediction (r) | 0.85-0.90 | Better than LSTM |
| Stimulus classification | 92% | Better than SVM on PCA |
| 1% pretrained vs 10% no pretrain | Pretrained wins | 10x data efficiency |

Key findings:
- **CLIP contrastive alignment works for neural data**: Aligning neural representations with visual/behavioral representations via contrastive learning produces better downstream features than unimodal pretraining alone
- **Autoregressive spike prediction is effective SSL**: Next-interval prediction captures temporal dynamics well enough to transfer to behavioral decoding
- **10x data efficiency from pretraining**: The most striking result. Pretrained model trained on 1% of data outperforms non-pretrained model trained on 10%. Validates pretraining's value in data-scarce neural recording settings
- **Cascading cross-attention handles multimodal fusion**: Perceiver-style cross-attention successfully integrates neural + visual + behavioral streams without architecture-specific engineering

## v12 Comparison

**Demonstrates multimodal contrastive alignment and data-efficient pretraining for neural data.** The 10x data efficiency finding is encouraging for v12's data-scarce regime, though the experimental setup (single session, calcium imaging) is very different from v12's cross-patient uECoG.

**Key parallels:**
- Cascading cross-attention for multimodal fusion is architecturally similar to v12's VE cross-attention (mapping from input tokens to a shared representation space). Neuroformer fuses across modalities; v12 fuses across electrode configurations
- The autoregressive spike prediction SSL objective is analogous to v12's planned temporal span masking (both are temporal prediction tasks, though v12 uses masked reconstruction rather than autoregressive generation)
- CLIP-style contrastive alignment could be imported for v12's cross-patient VE consistency regularizer -- ensuring that overlapping patients (S26-S33, S22-S62) produce aligned VE representations

**Key differences:**
- **Single session only**: Neuroformer never demonstrates cross-session or cross-animal transfer. This is the fundamental gap. v12's core challenge (cross-patient alignment with different electrode placements) is entirely unaddressed
- **No spatial encoding**: Neuron identity is purely learned embeddings. This works for calcium imaging (consistent neuron populations within a session) but would fail for cross-patient iEEG where electrodes are in different locations. v12's MNI coordinates + atlas VEs are essential
- **Calcium imaging, not electrophysiology**: Different temporal dynamics (slow calcium transients vs fast HGA), different signal properties, different noise characteristics. Results don't directly translate
- **Autoregressive architecture**: GPT-style left-to-right generation. v12 uses autoregressive decoding for the output (phoneme sequences) but bidirectional encoding of neural input. Neuroformer is autoregressive throughout, which limits its ability to use future context for neural encoding
- **Scale**: 40-100M params for single-session data is massively overparameterized. v12's ~170K params is better calibrated for the available data

**What to import:**
- **CLIP-style contrastive loss for cross-patient alignment**: Apply contrastive alignment between VE representations of overlapping patients (S26-S33 have 1.3mm nearest-neighbor distance). If the same atlas VE is activated by both patients, their representations should align in latent space. This is directly implementable as the cross-patient VE consistency regularizer described in CLAUDE.md
- **10x data efficiency finding**: Motivates aggressive pretraining even with limited data. v12's 456 min raw continuous data, even if only partially useful, could provide substantial efficiency gains for the ~11 min epoched supervised data
- **Cascading cross-attention pattern**: The hierarchical cross-attention (neural -> fused -> behavioral) could inform v12's VE cross-attention -> self-attention -> decoder pipeline

**What doesn't transfer:**
- **Single-session scope**: No cross-session transfer, no per-session adaptation, no handling of electrode variability. The core v12 challenge is out of scope
- **Calcium imaging regime**: Temporal resolution, signal properties, and spatial organization are fundamentally different from uECoG HGA
- **Overparameterized for our regime**: 40-100M params for ~1 session of data. v12 cannot afford this with ~11 min epoched data across 11 patients
- **No constrained output space**: Neuroformer generates unconstrained spike predictions. v12's 52-token constrained vocabulary provides much stronger output structure for speech decoding
- **Behavioral predictions (r=0.95-0.97)**: Running speed from visual cortex is a much easier decoding target than phoneme identity from sensorimotor cortex. The high correlations don't indicate architectural superiority
