# Azabou et al. 2025 - POYO+: Multi-session Multi-task Neural Population Dynamics Modeling

## Citation
Azabou, M., Arora, V., Ganesh, V., Mao, X., Nachimuthu, S., Mendelson, M., Park, B., Ye, J., Pandarinath, C., Dyer, E.L., & Bhaskara, A. (2025). A Unified, Scalable Framework for Neural Population Decoding. *ICLR 2025*.

## Setup
- **Recording modality**: Two-photon calcium imaging (GCaMP6f), NOT electrophysiology
- **Species**: Mice (256 animals)
- **Data scale**: 1335 sessions from the Allen Institute Visual Behavior dataset, 6 visual cortical areas (VISp, VISl, VISrl, VISal, VISpm, VISam), 13 Cre lines (cell types including excitatory + inhibitory)
- **Tasks**: 12 simultaneous tasks -- stimulus classification (8 images), change detection, behavioral predictions (licking, running speed), temporal context
- **Compute**: Not explicitly stated; Perceiver architecture is relatively efficient

## Architecture
- **Input tokenization**: Per-neuron activity traces binned at 75ms. Each neuron's activity is a scalar time series. Token = (neuron_id, time_bin, activity_value)
- **Neuron identity**: Per-session per-neuron learnable embeddings. Additionally, neurons tagged with metadata: brain region (6 areas), cell type (13 Cre lines), imaging depth
- **Session identity**: Learnable per-session embedding vector, concatenated/added to all tokens from that session
- **Latent bottleneck**: Perceiver architecture with 128 learnable latent tokens, D=128. Cross-attention from latents to input neuron tokens, then self-attention among latents
- **Multi-modal decoder**: Separate read-out heads per task. Task embeddings index which decoder head to use. Cross-attention from task-specific queries to latent representations
- **Training**: Multi-task loss with task-specific weighting. All 12 tasks trained jointly
- **Transfer protocol**: Gradual unfreezing -- first train only per-session embeddings + decoder heads (backbone frozen), then progressively unfreeze backbone layers

## SSL/Pretraining
- **NOT self-supervised**: Fully supervised multi-task training. All 1335 sessions have behavioral + stimulus labels
- **Multi-task as implicit SSL**: Training on 12 diverse tasks simultaneously acts as a regularizer. Each task provides a different view of the same neural dynamics, preventing overfitting to any single objective
- **Transfer**: Pretrained model transferred to held-out sessions/animals via gradual unfreezing

## Cross-Patient Handling
- **Per-session embeddings**: Each of 1335 sessions gets a unique learnable embedding. This absorbs session-specific variability (different animals, different neuron populations, different imaging conditions)
- **No spatial coordinates**: Neuron positions within the imaging field are not used. Identity is purely via learned embeddings. This works because calcium imaging fields are small (~500um) with consistent functional organization at the area level
- **Cross-region transfer**: Model trained on ALL 6 visual regions simultaneously. Key finding: all-region model outperforms any single-region model (55.96% vs best single-region). Shared dynamics across cortical areas provide complementary information
- **Cross-cell-type**: Model handles both excitatory and inhibitory neurons. Key finding: inhibitory neurons carry broadly useful decoding information, not just local circuit modulation
- **Transfer to novel region**: Model pretrained on visual cortex transfers to hippocampus (out-of-distribution region) with minimal fine-tuning, suggesting learned representations capture general neural dynamics

## I/O Features
- **Input**: Calcium fluorescence traces (delta F/F0) at 75ms bins, per neuron
- **Output**: 12 simultaneous task outputs -- categorical (image identity, change detection), continuous (running speed), binary (licking)
- **Spatial**: Per-session learned embeddings + region/cell-type metadata tags (no coordinates)
- **Temporal**: 75ms bins, sequences spanning seconds

## Key Results
| Configuration | Stimulus Decoding Acc | Change Detection | Cross-Region |
|---|---|---|---|
| POYO+ (all regions, all tasks) | 55.96% | SOTA | All > any single |
| POYO+ (single region) | ~45-52% | -- | -- |
| Linear baseline | ~30% | -- | -- |
| POYO (Azabou 2023, predecessor) | ~48% | -- | -- |

Key findings:
- **All-region > single-region (55.96%)**: Cross-region pretraining consistently helps. Each cortical area provides complementary information about the shared visual processing pipeline
- **12 simultaneous tasks**: Multi-task decoding works without interference. Task-specific decoder heads + task embeddings successfully route computation
- **Inhibitory neurons are informative**: Removing inhibitory neurons hurts decoding. They carry broadly useful population-level information, not just local circuit computation
- **Transfer to hippocampus works**: Out-of-distribution region transfer suggests the model learns general neural population dynamics, not area-specific features
- **Gradual unfreezing is critical**: Full fine-tuning from scratch underperforms gradual unfreezing (embeddings first, then progressively deeper layers)

## v12 Comparison

**Validates the Perceiver bottleneck + per-session adaptation paradigm at large scale (1335 sessions, 256 animals).** POYO+ is the largest multi-session neural population model to date and demonstrates that cross-session, cross-animal transfer works with the right architecture.

**Key parallels:**
- POYO+'s 128 latent tokens (Perceiver) are architecturally analogous to v12's 16 virtual electrodes (VE cross-attention). Both compress variable-size neural populations into a fixed latent space via cross-attention. POYO+ uses more latents (128 vs 16) but for a very different data regime (hundreds of neurons per session vs 63-256 electrodes)
- Per-session embeddings (POYO+) ~ per-patient diagonal normalization (v12). Both absorb session/patient-specific variability while sharing the backbone
- Multi-task training (POYO+) could inform v12's SSL + supervised multi-objective training. The finding that diverse tasks regularize each other is relevant
- Gradual unfreezing transfer protocol is directly applicable to v12's 3-stage training (sEEG SSL -> uECoG SSL -> supervised fine-tune)

**Key differences:**
- **Calcium imaging vs electrophysiology**: Fundamentally different recording modality. Calcium imaging has slow temporal dynamics (~100ms), consistent within-session neuron identity, and small spatial fields. uECoG HGA is fast (~5ms), has variable electrode placement, and covers large cortical patches. The transfer challenges are qualitatively different
- **No spatial coordinates**: POYO+ relies entirely on learned per-session embeddings for neuron identity. This works for calcium imaging (small consistent fields) but is insufficient for uECoG (variable array placement requires explicit spatial encoding). v12's MNI coordinates + atlas VEs address this
- **256 animals vs 11 patients**: POYO+ has 20x more subjects. The per-session embedding approach works at scale but may not be practical for v12's limited patient pool
- **Binary/continuous tasks vs phoneme sequences**: POYO+'s tasks are simpler per-output (image classification, licking detection) than 9-class phoneme sequence decoding with temporal structure
- **No language model / constrained output**: POYO+ has independent per-task decoders. v12 has a constrained AR decoder (52 valid tokens) which provides much stronger output structure

**What to import:**
- **Gradual unfreezing protocol**: Train per-patient layers first (frozen backbone), then progressively unfreeze deeper backbone layers. This is more principled than v12's current binary freeze/unfreeze approach. Order: per-patient diagonal -> VE cross-attention -> temporal self-attention -> backbone
- **Multi-task regularization**: If v12 adds auxiliary objectives (e.g., temporal reconstruction SSL + phoneme classification + articulatory prediction), POYO+ validates that multi-task training works without interference when using task-specific decoder heads
- **Cross-region finding**: All visual regions > any single region. Analogous prediction for v12: all atlas VEs > any subset. Validates using all 16 Brainnetome VEs rather than patient-specific subsets

**What doesn't transfer:**
- **Calcium imaging regime**: Temporal resolution, signal properties, and spatial organization are fundamentally different from uECoG HGA. Cannot compare numbers or scaling laws
- **Learned-only spatial identity**: Per-session neuron embeddings without coordinates works for consistent imaging fields but fails for variable electrode arrays. v12 needs atlas-grounded spatial encoding
- **Data abundance**: 1335 sessions enables rich per-session embeddings. v12's 11 patients cannot support this approach -- hence the atlas prior
