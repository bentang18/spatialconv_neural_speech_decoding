# Zhang et al. 2025 - NEDS: Multi-Task Masking for Neural Encoding and Decoding

## Citation
Zhang, Y., Azabou, M., Bhagat, B., Ye, J., Pandarinath, C., & Dyer, E.L. (2025). Multi-Task Masking Enables Simultaneous Neural Encoding and Decoding of Spiking Activity. *ICML 2025*.

## Setup
- **Recording modality**: Neuropixels extracellular electrophysiology (spike counts, 50ms bins)
- **Species**: 73 mice
- **Data scale**: 74 sessions from International Brain Laboratory (IBL), brain-wide recordings
- **Tasks**: Stimulus detection (visual contrast discrimination). Downstream: neural encoding (predict neural activity from behavior) + decoding (predict behavior from neural activity) simultaneously
- **Brain regions**: Distributed -- recordings span multiple cortical and subcortical regions per session

## Architecture
- **Base model**: Transformer encoder, 6 layers, d=256, 4 attention heads
- **Scale**: ~12M shared parameters + ~1.86M per-session parameters (stitchers)
- **Per-session stitchers**: Linear projection layers mapping each session's unique neuron set to shared embedding space + inverse projection for neural reconstruction. ~1.86M/session accounts for variable neuron counts (50-500+ per session)
- **Input tokenization**: Neuron tokens (50ms-binned spike counts) + behavior tokens (wheel velocity, stimulus contrast, reward, choice) interleaved in a single sequence
- **Neuron embeddings**: Per-neuron learned embeddings within each session. After pretraining, these embeddings become 83% predictive of brain region labels WITHOUT explicit region supervision -- the model discovers anatomical structure from neural dynamics alone
- **Behavioral embeddings**: Separate learned embeddings per behavioral variable type

## SSL/Pretraining
- **Multi-task masking**: Four masking schemes applied simultaneously during training:
  1. **Neural masking**: Mask neural tokens, predict from behavior + unmasked neurons (encoding)
  2. **Behavioral masking**: Mask behavior tokens, predict from neural activity (decoding)
  3. **Within-modal masking**: Mask some neural tokens, predict from other neural tokens (neural self-consistency)
  4. **Cross-modal masking**: Mask both some neural and some behavioral tokens simultaneously
- **Loss**: Poisson NLL for spike count reconstruction + MSE for continuous behavioral variables + CE for discrete behavioral variables (choice, stimulus identity)
- **Mask ratio**: 0.1 at scale (much lower than BERT's 0.15 or BIT's 0.5). Low mask ratio enables the model to learn from mostly-complete contexts
- **KEY: Within-modal masking is most critical for encoding** -- removing it causes the largest performance drop. Cross-modal masking is most critical for decoding
- **Training**: AdamW, 500 epochs, lr=1e-4

## Cross-Patient Handling
- **Per-session stitchers**: ~1.86M per session (linear projections). Expensive -- 10,000x more per-session params than v12's 134. Required because each session records different neurons in different brain regions
- **Neuron embeddings become region-predictive**: 83% accuracy at predicting brain region from learned neuron embeddings. The model learns spatial structure without coordinates or region labels. This is remarkable but requires abundant data per neuron
- **Session scaling**: Performance improves with more pretraining sessions up to ~40-50, then saturates. Consistent with Jiang 2025's finding that heterogeneous data scaling has limits
- **No spatial encoding**: No coordinates, no atlas. Spatial structure emerges entirely from temporal co-activation patterns

## I/O Features
- **Input**: 50ms-binned spike counts per neuron + behavioral variables (wheel velocity, stimulus contrast, reward, choice), interleaved
- **Output**: Simultaneously predicts masked neural activity (Poisson NLL) AND masked behavioral variables (MSE/CE). This is the key innovation -- encoding and decoding are unified in one model
- **Temporal**: 50ms bins, causal masking not enforced (bidirectional attention)
- **Spatial**: Per-neuron learned embeddings (no coordinates)

## Key Results
| Task | NEDS | Best Baseline |
|---|---|---|
| Neural encoding R^2 | 0.42 | 0.38 (separate encoder) |
| Behavioral decoding acc | 0.72 | 0.68 (separate decoder) |
| Neuron embedding → region | 83% | N/A (emergent) |
| Multi-session pretrain gain | +12% | Over single-session |

Key findings:
- **Multi-task masking enables joint encoding + decoding**: Single model does both better than separate specialized models. The shared representation captures bidirectional neural-behavioral relationships
- **Within-modal masking is critical for encoding**: Most important single masking scheme. Neural self-consistency (predict neurons from other neurons) forces the model to learn population dynamics
- **Neuron embeddings discover brain regions**: 83% region classification accuracy from learned per-neuron embeddings without any spatial supervision. Neural dynamics contain enough information to recover spatial structure
- **Low mask ratio at scale**: 0.1 masking ratio works better than higher ratios when scaling to many sessions. Lower masking = more context = better learning when data diversity is high
- **Poisson NLL for spike reconstruction**: More principled than MSE for count data. Not applicable to HGA (continuous, approximately Gaussian)

## v12 Comparison

**Multi-task masking is an interesting framework but largely orthogonal to v12's setup.** NEDS optimizes for simultaneous encoding and decoding of spike data; v12 does phoneme decoding from HGA only. The relevant insights are methodological, not architectural.

**Key implications for v12:**
1. **Within-modal masking validates temporal masking**: NEDS finds that neural self-consistency (predict neural activity from other neural activity) is the most important masking scheme for learning good neural representations. This is exactly what v12's temporal span masking does -- reconstruct masked time segments from observed ones. The convergence across modalities (spikes for NEDS, HGA for v12) strengthens the case for temporal masking SSL
2. **Low mask ratio at scale**: NEDS uses 0.1 masking at scale, much lower than BIT's 0.5 or MAE's 0.75. For v12's multi-patient pretraining with heterogeneous data, a lower mask ratio may be beneficial -- more context helps when data sources are diverse. Worth testing 0.1-0.3 alongside BIT's 0.5
3. **Emergent spatial structure from temporal dynamics**: Neuron embeddings becoming 83% region-predictive without spatial supervision is striking. For v12, this suggests that even without MNI coordinates, temporal self-attention could learn spatial relationships. But v12's atlas VEs provide this for free -- no need to discover what's already known

**Key differences:**
- **Modality**: Neuropixels spikes vs uECOG HGA. Different noise profiles, different reconstruction losses (Poisson vs MSE)
- **Behavioral variables as input**: NEDS includes behavioral variables (stimulus, choice, reward) as input tokens alongside neural data. v12 has no analogous behavioral input during SSL -- only neural data. During supervised fine-tuning, phoneme labels serve as targets, not inputs
- **Scale**: 74 sessions, brain-wide recordings vs 11 patients, focal speech cortex. NEDS has far more spatial diversity
- **Per-session params**: ~1.86M/session vs 134/patient. NEDS requires this because each session records completely different neurons. v12's diagonal normalization (scale+bias per electrode) is sufficient because electrodes measure the same quantity (HGA) with different gain/impedance

**What to import:**
- **Multi-task masking as ablation framework**: Test v12 with (a) temporal masking only (baseline), (b) temporal + electrode dropout (within-modal + spatial), (c) temporal + phoneme prediction (cross-modal). NEDS shows that combining complementary masking schemes helps
- **Low mask ratio option**: Test 0.1-0.2 masking alongside BIT's 0.5 for v12 SSL. Especially relevant with heterogeneous multi-patient data
- **Poisson NLL for spike count tasks only**: NOT applicable to v12's HGA (continuous, z-scored). Use MSE or content-aware MSE (BrainBERT)

**Common mistakes:**
- Do NOT use Poisson NLL for HGA reconstruction. HGA is continuous and approximately Gaussian after z-scoring. MSE is appropriate
- Do NOT adopt ~1.86M per-session stitchers. v12's 134-param diagonal normalization is physics-motivated (impedance/gain variation) and sufficient for the variation axis in uECOG
- Do NOT assume emergent spatial discovery is better than providing coordinates. NEDS needs 74 sessions and extensive data to learn 83% region accuracy. v12's atlas provides 100% spatial identity with zero data
- Do NOT conflate "multi-task" with "multi-objective." NEDS trains one model for two tasks (encoding + decoding). v12's SSL then supervised is sequential, not joint. The insights about masking scheme interactions still apply to each stage independently
