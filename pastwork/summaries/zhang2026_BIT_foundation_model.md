# Zhang et al. 2026 - BIT: Cross-Species Neural Foundation Model

## Citation
Zhang, Y.\*, He, L.\*, Fan, C., Liu, T., Yu, H., Le, T., Li, J., Linderman, S., Duncker, L., Willett, F.R., Mesgarani, N., & Paninski, L. (2026). A Cross-Species Neural Foundation Model for End-to-End Speech Decoding. *Published as a conference paper at ICLR 2026*. arXiv:2511.21740v3.

## Setup
- **Recording modalities**: Utah microelectrode arrays; features are thresholded spike counts and spiking-band power (SBP), binned at 20 ms, z-scored across days
- **Species**: Human (2 participants: T12, T15) + non-human primates (monkeys, multiple datasets)
- **Datasets used**:
  - **Human (~98 hours)**: Willett et al. 2021 (handwriting, 17.4h), Willett et al. 2023 / Brain-to-Text '24 (15.7h, 128 electrodes, T12), Fan et al. 2023 (15.4h), Karpowicz et al. 2024 / FALCON (1.7h), Card et al. 2024 / Brain-to-Text '25 (39.2h, 256 electrodes, T15), Kunz et al. 2025 (8.4h, imagined speech)
  - **Monkey (~269 hours)**: Churchland 2012, O'Doherty 2017, Perich 2018, Chowdhury 2020, Even-Chen 2019, Ma 2023, Temmar 2025 -- all motor tasks (reaching, grasping, etc.)
  - **Total pretraining**: ~367 hours
- **Tasks**: Attempted speech decoding, imagined (inner) speech decoding

## Problem Statement
Most speech BCIs use cascaded frameworks: an RNN decodes phonemes, then an n-gram language model assembles sentences. This prevents joint end-to-end optimization -- lower phoneme error rates do not always translate to lower word error rates. Prior end-to-end approaches (Feng et al. 2024) used RNN encoders without pretraining, missing the benefits of transformers and large-scale self-supervised learning. BIT aims to build a single differentiable end-to-end system that translates neural activity directly into coherent sentences, powered by a cross-species, cross-task pretrained transformer encoder paired with an audio-LLM decoder.

## Method

### Architecture (transformer details, encoder/decoder)

**Neural Encoder (Transformer, ~7M params, ~13M with subject-specific layers):**
- Input: neural activity of shape (T, C) where T = time bins (20 ms), C = number of electrodes
- **Patch embedding**: groups every T_patch = 5 time bins into a patch, yielding tokens of shape (T/T_patch, C x T_patch). Patches pass through a module: LayerNorm -> Linear -> LayerNorm
- **Transformer blocks** (depth = 7): each block has self-attention (6 heads, head dim 512, embedding dim 384) followed by a feed-forward network. Uses RoPE (Rotary Position Embeddings) and bidirectional attention. Attention dropout = 0.4, general dropout = 0.2
- **Subject-specific layers**: separate linear read-in and read-out layers per subject for the patch embedding module; also per-day layers for handling non-stationarity

**LLM Decoder (audio-LLM, best model: Aero1-Audio 1.5B):**
- Neural encoder outputs are mapped to the LLM's text/audio embedding space via a shallow **MLP projector** (Linear -> ReLU -> Linear)
- A **modality aligner** (for contrastive learning) mean-pools neural and text embeddings, projects them via separate linear layers into a shared latent space
- A text prompt is inserted between neural and text embeddings: *"decode the above neural activity into an English sentence:"*
- **LoRA** (rank 8, scaling factor 32) applied to Q/K/V/O projections + feed-forward layers of the LLM, plus the multimodal projector for audio-based LLMs
- Neural activity can be treated as either a **neural modality** (projected directly to text embedding space) or an **audio modality** (mapped through the audio encoder's multimodal projector). Audio modality treatment slightly outperforms, but both work
- Text generation at inference: nucleus sampling (top-p = 0.9, temperature = 0.7, max 25 tokens)

### Pretraining strategy (self-supervised objective)
- **Masked modeling** inspired by MAE (He et al. 2022): neural activity is tokenized into temporal patches; a random subset is replaced with a learnable mask token. Masked patches form contiguous spans of variable length up to a predefined max (max mask time span = 15 patches). Overall mask ratio = 0.5 for T12, **0 for T15** (tuned per participant). **Note:** mask ratio 0 for T15 means the "SSL pretraining" for the better-performing participant involves NO masking at all — just reconstruction through the bottleneck. This undermines claims that masked modeling is essential; the benefit may come from learning to compress neural activity through the transformer bottleneck
- The model reconstructs original neural activity from the partially masked sequence
- **Loss**: Mean Squared Error (MSE) on the reconstructed vs. original neural data (both thresholded spikes and SBP are normalized, so MSE is appropriate)
- Pretraining uses **all 367 hours** of human and monkey Utah array data (labels discarded), including the downstream fine-tuning data
- Trained with AdamW optimizer on a single NVIDIA A100 GPU, under 2 days, 400 epochs

### Fine-tuning procedure
Three-stage pipeline:
1. **SSL pretraining** (masked modeling on 367h, MSE loss) -- produces general neural representations
2. **Phoneme-level fine-tuning** (CTC loss, **800 epochs for transformer, 600 for RNN**): the mask module is removed; a linear classifier on top of encoder outputs predicts 41 phoneme classes + blank token (42 total). Hyperparameter search: 30 random samples from defined ranges (batch 8-64, LR 5e-5 to 1e-3, weight decay 5e-5 to 0.1). This is critical -- without phoneme fine-tuning, the encoder does not encode enough phonetic information for the LLM to decode sentences
3. **Sentence-level fine-tuning** (cross-entropy + contrastive loss): the phoneme-aware encoder feeds representations into the audio-LLM decoder. Total loss = cross-entropy (next-token prediction) + symmetric InfoNCE contrastive loss (aligns mean-pooled neural and text embeddings in a shared space). LoRA is applied to the LLM; the encoder and projector are also updated. Trained for 150 epochs on a single A40/A100 GPU

### How does it handle different electrode configurations/counts?
- **Subject-specific patch embedding modules**: each subject/dataset gets its own linear read-in and read-out layers, which map from that subject's electrode count (C) into the shared transformer embedding dimension (384). This means different electrode counts (92, 96, 128, 192, 256) are all projected into the same latent space
- **Per-day layers**: handle non-stationarity across recording sessions (24 day layers for T12, 45 for T15)
- Dead channels: if fewer than 2 dead channels, they are interpolated with the mean; datasets with more than 2 dead channels are excluded from pretraining
- Only thresholded spikes are used for pretraining (SBP unavailable in some datasets); SBP is added during fine-tuning where available

## Key Results

### Cascaded framework (encoder + 5-gram LM from OpenWebText2 634M sentences + OPT 6.7B re-scoring; beam=100)
**Vocabulary: 125,000 words. Ensemble: 10 encoders for T12 (GPT-3.5 selection), 29 for T15 (GPT-4).**
| Benchmark | BIT Cascaded (single) | BIT Cascaded + Ensemble | Previous best |
|---|---|---|---|
| Brain-to-Text '24 (T12, 1200 sent holdout) | **6.35%** WER | **5.10%** WER | 5.68% (Feghhi et al. 2025 + Ensemble) |
| Brain-to-Text '25 (T15, 1450 sent holdout) | **4.06%** WER | **1.76%** WER | 3.09% (RNN + Ensemble) |

### End-to-end framework (encoder + audio-LLM)
| Benchmark | BIT End-to-End (single) | BIT E2E + Ensemble | Previous best E2E |
|---|---|---|---|
| Brain-to-Text '24 | 15.67% WER | **10.22%** WER | 24.69% (Feng et al. 2024) |
| Brain-to-Text '25 | 11.06% WER | **7.76%** WER | N/A |

### Pretraining ablations (validation PER, attempted speech)
| Encoder | T12 PER | T15 PER |
|---|---|---|
| RNN | 18.67% | 9.64% |
| BIT-TFS (transformer from scratch) | 17.26% | 8.87% |
| BIT-Human (pretrained on human only) | 15.95% | 7.61% |
| BIT-All (human + monkey) | **14.39%** | **7.12%** |

### Imagined speech (50-word vocabulary)
- BIT-All outperforms all baselines in both cascaded and end-to-end settings
- SSL pretraining provides even greater benefits for imagined speech (low-data regime: 500-712 sentences)
- Cross-subject SSL pretraining (BIT-All) outperforms same-subject supervised pretraining (BIT-Cross-Task-Only), showing the value of large-scale unlabeled data

### Cross-task generalization
- BIT aligns attempted and imagined speech embeddings through shared semantic structure (demonstrated via PCA, RSA analysis, and cross-attention visualization)
- RSA scores between neural encoder outputs and LLM text embeddings: BIT-All (0.22-0.26) >> RNN (0.09-0.03)

### LLM decoder findings
- Audio-LLMs (Aero1-Audio 1.5B) outperform text-only LLMs of comparable size
- Smaller LLMs (~1.5B) outperform larger ones (~7B) given limited labeled data
- Contrastive learning for cross-modal alignment further improves performance

### Scaling curve
- Adding human pretraining data yields larger performance gains than adding monkey data
- But monkey data still helps, especially for the cascaded framework

## Cross-Patient Relevance
**Could this approach work with ~20 intra-op uECOG patients, 15 min each?**

Challenges and considerations:

1. **Data quantity mismatch**: BIT's pretraining uses ~367 hours of Utah array data; 20 patients x 15 min = only 5 hours total. The SSL pretraining benefit scales with data (Figure 8), but even the human-only subset was ~98 hours. Five hours is very limited for SSL pretraining, though it could still help over training from scratch.

2. **Modality gap (Utah arrays vs. uECOG)**: BIT uses intracortical Utah arrays (thresholded spikes + SBP), not ECoG. uECOG has different spatial resolution, signal characteristics, and electrode counts. The subject-specific read-in layer design could theoretically accommodate different electrode counts, but the representation learned from spike data may not transfer well to ECoG field potentials. A new pretraining corpus of ECoG/uECOG data would be needed.

3. **Cross-patient SSL is the right idea**: The paper's key finding is that cross-subject, label-free SSL pretraining outperforms same-subject supervised pretraining (Appendix M). This is encouraging -- even though supervised cross-subject pretraining degraded performance (due to inter-subject functional variability), SSL pretraining still helped. This suggests SSL could pool across your 20 patients.

4. **Subject-specific layers are essential**: The per-subject patch embedding (read-in/read-out) layers handle electrode configuration differences. This design pattern would be critical for uECOG where electrode placement varies per patient.

5. **Adaptations needed**:
   - Replace thresholded spikes/SBP features with appropriate uECOG features (high-gamma power, broadband, etc.)
   - Possibly collect or incorporate existing public ECoG datasets for pretraining (e.g., from other speech ECoG studies)
   - The 15 min per patient is likely too short for the phoneme fine-tuning stage (CTC loss); may need to combine with aggressive data augmentation or few-shot approaches
   - Consider whether the end-to-end LLM decoding or cascaded approach is more appropriate given data scarcity (cascaded performs better with limited data in this paper)

## Limitations
- **End-to-end still lags cascaded**: WER gap remains (10.22% vs. 5.10% on '24 benchmark), though substantially narrowed
- **Inference speed**: End-to-end takes ~0.95s per sentence (vs. 0.24s cascaded), not suitable for real-time BCI
- **Bidirectional attention**: encoder uses bidirectional attention (sees future context), making it unsuitable for online/streaming decoding. Switching to causal attention is possible but hurts accuracy
- **Human data matters more than monkey data**: cross-species transfer provides diminishing returns; the model benefits most from task-relevant (speech) human data
- **Supervised cross-subject pretraining does not help** (Appendix M): incorporating labeled data from multiple subjects during phoneme/sentence fine-tuning degrades performance. **However: no quantitative numbers are given** — Appendix M provides only qualitative statements. The authors attribute failure to inter-subject *functional variability* (different neural-to-behavior mappings) rather than *sensor variability*. **Critical caveat:** this finding is from N=2 patients only (T12, T15) with very different setups — may not generalize to larger N with homogeneous arrays
- **Table 9 (not in original summary): SSL ≈ supervised at equal data for same-subject.** BIT-Cross-Task-Only (SL) 12.53% vs BIT-SameParticipant-SSL 12.67% WER on T12 imagined speech — no substantial difference. **The SSL advantage is specifically cross-subject, not inherent to the objective.** This means for our N≈20 setting with per-patient layers, Singh's supervised approach is equally well-motivated
- **BIT never freezes the backbone during fine-tuning** — all stages update the full encoder. This differs from Singh's Mode 3 (freeze backbone, train read-in + head). For our small-data regime, Singh's freeze approach likely prevents catastrophic forgetting better
- **Data requirements are substantial**: ~98 hours human + ~269 hours monkey for pretraining; the encoder alone needs ~13M parameters with subject-specific layers
- **Limited to Utah arrays**: all data is from intracortical Utah microelectrode arrays; applicability to ECoG, EEG, or other modalities is unvalidated
- **Compute**: pretraining takes <2 days on A100; phoneme fine-tuning and LLM fine-tuning each take 1-4 days on A40/A100 GPUs

## Reusable Ideas

1. **Subject-specific read-in/read-out layers with a shared transformer backbone**: This is the key architectural pattern for handling different electrode configurations across patients. Each patient gets their own linear projection into a common embedding space. Directly applicable to cross-patient uECOG.

2. **Temporal patch embedding**: Grouping multiple time bins (5 x 20ms = 100ms patches) into tokens aligns the transformer's token rate with speech production timescale (30-60 words/min) and shortens context length. Worth adopting for uECOG.

3. **Masked modeling pretraining (MAE-style) on neural data**: Self-supervised objective that works without labels, handles non-stationarity through data augmentation, and learns robust representations. The contiguous span masking strategy is specifically designed for temporal neural data.

4. **Three-stage training pipeline** (SSL pretrain -> CTC phoneme fine-tune -> LLM sentence fine-tune): The intermediate phoneme decoding stage is critical -- it injects phonetic structure into the encoder's representations before the LLM sees them. Without it, pretraining alone is insufficient for the LLM to decode sentences.

5. **Audio-LLMs outperform text-LLMs for neural-to-text**: Neural activity treated as an audio modality (through the audio encoder's multimodal projector) leverages speech knowledge acquired during LLM pretraining. Using Aero1-Audio 1.5B (audio-extended Qwen2.5-1.5B) is a practical choice.

6. **Contrastive alignment (InfoNCE) between neural and text embeddings**: Simple but effective -- mean-pool neural and text embeddings, project to shared space, pull matching pairs together. This is a lightweight addition (no extra model components) that improves cross-modal alignment.

7. **Smaller LLMs beat larger ones with limited labeled data**: For BCI applications where labeled data is scarce, 1-1.5B parameter LLMs are the sweet spot. Larger models (7B) overfit or require reasoning capabilities that are unnecessary for constrained sentence prediction.

8. **Ensemble via LLM merger**: Multiple neural encoders (different random seeds) produce candidate sentences; a fine-tuned LLM selects/merges the best output. This is a practical way to improve final decoding accuracy.

9. **Per-day normalization (z-scoring) and per-day read-in layers**: Addresses non-stationarity from probe drift, which is a universal problem in chronic neural recordings. For intra-op settings with single sessions per patient, this translates to per-patient normalization.
