# Ryoo et al. 2025 - POSSM: Generalizable Real-Time Neural Decoding with Hybrid State-Space Models

## Citation
Ryoo, A.H.\*, Krishna, N.H.\*, Mao, X.\*, Azabou, M., Dyer, E.L., Perich, M.G., & Lajoie, G. (2025). Generalizable, real-time neural decoding with hybrid state-space models. *39th Conference on Neural Information Processing Systems (NeurIPS 2025)*. arXiv:2506.05320v2.

---

## Setup
- **Recording modality**: Intracortical Utah-style microelectrode arrays; individual spike events tokenized at millisecond resolution (not binned). Speech experiment uses normalized multi-unit threshold crossings binned at 20 ms due to data availability
- **Species**: Non-human primates (NHP, macaques) and humans
- **Datasets used**:
  - **NHP pretraining (o-POSSM)**: 148 sessions across 4 datasets -- Perich et al. (M1/PMd, CO + RT tasks, 2 subjects, 93 sessions, 9040 units, 96.8M spikes), O'Doherty et al. (M1/S1, RT task, 2 subjects, 44 sessions, 14899 units, 87.8M spikes), Churchland et al. (M1/PMd, CO task, 2 subjects, 10 sessions, 1911 units, 739M spikes), NLB Maze (M1/PMd, Maze task, 1 subject, 1 session, 182 units). Total: ~670M spikes from 26,032 neural units
  - **NHP evaluation**: held-out sessions from Monkey C (CO task), Monkey T (CO and RT tasks), and a new Flint et al. dataset (M1, CO task, 1 subject, 5 sessions)
  - **Human handwriting**: Willett et al. 2021 -- 11 sessions, 1 participant, 2x96-channel microelectrode arrays in motor cortex, imagined character writing (26 letters + lines), classification task
  - **Human speech**: Willett et al. 2023 -- 24 sessions, 1 participant with speech deficits, 4x64-channel microelectrode arrays covering premotor cortex and Broca's area, attempted speech of sentences, phoneme error rate (PER) task
- **Tasks**: 2D hand velocity decoding (reaching: centre-out, random target, maze), imagined handwriting character classification, attempted speech phoneme sequence decoding

---

## Problem Statement
Neural decoders for BCIs must satisfy three requirements simultaneously: (1) high accuracy, (2) low-latency causal inference for online use, and (3) flexible generalization to new sessions, subjects, and tasks. Existing approaches fail to meet all three at once. RNNs are fast but cannot generalize to new neuron identities or sampling rates without full retraining, because they rely on fixed-size time-binned inputs. Transformer-based models like POYO offer flexible tokenization that enables generalization, but their quadratic attention complexity makes real-time inference expensive and often prohibitive for long-context tasks (e.g., sentence-length speech decoding). Hybrid attention-SSM models show promise in NLP but have not been explored for neural decoding. POSSM addresses this gap by pairing POYO-style spike tokenization with a recurrent SSM backbone, achieving transformer-level generalization with RNN-level inference speed.

---

## Method

### Architecture

POSSM is a two-component hybrid that processes neural spiking activity in causal, streaming 50 ms chunks:

**Spike tokenization (from POYO):**
- Each spike is represented as a D-dimensional token: `x = (UnitEmb(i), t_spike)`, where `UnitEmb(i)` is a learnable unit embedding for neuron identity `i`, and `t_spike` is encoded with Rotary Position Embedding (RoPE) to capture relative timing
- This produces a variable-length token sequence per time chunk; no fixed binning required
- Each neural unit can be assigned at any specificity level (single unit, multi-unit, or electrode channel), giving flexibility across datasets

**Input cross-attention tokenizer:**
- A PerceiverIO-style cross-attention module compresses the variable-length spike token sequence into a fixed-size latent vector `z(t)` of dimension M (M=1 in most experiments, yielding a single latent per 50 ms chunk)
- Query vectors `q` are learnable parameters; keys and values come from the spike tokens
- Scales attention by sqrt(D) following standard transformer notation

**Recurrent SSM backbone:**
- `z(t)` is fed into a recurrent backbone (GRU, S4D, or Mamba) that maintains a hidden state `h(t)` updated as: `h(t) = f_SSM(z(t), h(t-1))`
- The SSM integrates local chunk information with global temporal history via its hidden state
- This allows constant-time prediction per chunk (no reprocessing of past inputs), making inference O(1) per step
- Three backbone variants tested: S4D (diagonal structured SSM), GRU, and Mamba

**Output cross-attention and readout:**
- The k=3 most recent hidden states `{h(t-k+1:t)}` are selected
- An output cross-attention module queries these hidden states with P query vectors, one per behavioural timestamp to predict within the chunk
- Each query encodes its target timestamp (via RoPE) plus a learnable session embedding
- A linear readout maps the output cross-attention to the behavioural prediction
- This design allows predicting multiple outputs per chunk (dense supervision), handling non-aligned chunk/behaviour boundaries, and predicting future behaviour beyond the current chunk

### Cross-Species Pretraining Strategy

Two finetuning strategies for adapting a pretrained model to new sessions:

**Unit Identification (UI):**
- Freeze all model weights; initialize new unit embeddings and session embeddings for the new session
- Train only these new embeddings (< 1% of total parameters) on the new session's data
- Fast, parameter-efficient; works well when new sessions are from animals seen during pretraining

**Full Finetuning (FT):**
- Begin with UI for ~100 epochs (warm up embeddings)
- Then unfreeze entire model and train end-to-end for remaining ~400 epochs
- Consistently outperforms UI and single-session training from scratch, especially for new animals or new species
- Enables cross-species transfer: pretrain on NHP motor cortex data, then full finetune on human motor cortex data

**Pretraining procedure (o-POSSM):**
- Pretrain on all 4 NHP reaching datasets (148 sessions) simultaneously
- Loss: mean squared error on 2D hand velocity (continuous regression)
- Unit dropout augmentation: randomly drop all spikes from a random subset of units each batch, to encourage robustness to different electrode configurations
- Single-session models: trained on one RTX8000 GPU (<30 min); multi-dataset pretraining: 4x H100 GPUs (~36 hours)

**Speech adaptation (separate from NHP pipeline):**
- Since only normalized spike counts (not spike times) are available in the speech dataset, POYO-style tokenization is adapted: each multi-unit channel is treated as a neural unit, and normalized spike counts are embedded with value embeddings (instead of presence/absence tokens)
- Output cross-attention module is replaced with a 1D strided convolution layer to control output sequence length (phoneme sequence generation)
- Two-phase training: (1) train input cross-attention + unit/session embeddings on spike count reconstruction; (2) train full model with CTC loss on phoneme sequences

---

## Key Results

### NHP Reaching (R² metric, mean ± SD)

Evaluation includes three regimes: same animal other days, new animal (Monkey T), new dataset (Flint et al., Monkey H).

**From scratch (single-session) -- best models:**
| Model | C-CO 2016 | C-CO 2010 | T-CO | T-RT | H-CO |
|---|---|---|---|---|---|
| POSSM-Mamba-SS | **0.9550** | **0.7959** | 0.8747 | 0.7418 | **0.8350** |
| POSSM-GRU-SS | 0.9549 | 0.7930 | **0.8863** | **0.7687** | 0.8161 |
| POYO-SS | 0.9427 | 0.7381 | 0.8705 | 0.7156 | 0.4974 |

**Pretrained (o-POSSM, full finetuning) -- best models:**
| Model | C-CO 2016 | C-CO 2010 | T-CO | T-RT | H-CO |
|---|---|---|---|---|---|
| o-POSSM-S4D (FT) | 0.9609 | **0.8216** | **0.9068** | 0.7605 | **0.8941** |
| o-POSSM-GRU (FT) | **0.9614** | 0.8161 | 0.9024 | **0.7741** | 0.8743 |
| o-POSSM-Mamba (FT) | 0.9595 | 0.8195 | 0.8921 | 0.7464 | 0.8220 |
| POYO-1 (FT) | 0.9580 | 0.8300 | 0.8859 | 0.7718 | 0.8601 |

Key finding: o-POSSM-FT consistently matches or outperforms POYO-1-FT across all evaluation conditions, including the cross-dataset Flint et al. condition.

### Cross-Species Transfer (Human Handwriting)

| Model | Acc. (%) |
|---|---|
| PCA-KNN (SS) | 81.36 ± 7.53 |
| GRU | 93.57 ± 4.22 |
| POYO | 94.86 ± 3.53 |
| POSSM-GRU (from scratch) | **95.82 ± 3.41** |
| POYO-1 (FT on NHP) | 95.82 ± 3.12 |
| o-POSSM-S4D (FT on NHP) | 97.25 ± 2.88** |
| o-POSSM-Mamba (FT on NHP) | **97.73 ± 2.13**\*\* |
| o-POSSM-GRU (FT on NHP) | 97.37 ± 2.32* |

Finetuning o-POSSM (pretrained only on NHP motor data) on 9 human handwriting sessions provides a 2% gain over scratch POSSM-GRU and >5% over POSSM-S4D/Mamba. This demonstrates successful cross-species transfer of motor cortex dynamics.

### Human Speech Decoding (Phoneme Error Rate, uni-directional causal)

| Model | PER (%) |
|---|---|
| GRU (no aug.) | 39.16 |
| POSSM-GRU (no aug.) | **29.70** |
| GRU | 30.06 |
| S4D | 35.99 |
| Mamba | 32.19 |
| POSSM-GRU | **27.32** |
| GRU (mult. input) | 21.74 |
| POSSM-GRU (mult. input) | **19.80** |

Bi-directional results (not suitable for real-time but provided as upper bound):
| Model | PER (%) |
|---|---|
| BiGRU | 27.86 |
| POSSM-BiGRU | **25.80** |

POSSM-GRU reduces PER by ~2.7 points over GRU baseline (uni-directional). Maintaining competitive performance without noise augmentation (29.70 vs. 30.06) demonstrates architectural robustness.

### Inference Speed (NVIDIA RTX8000 GPU, per 50 ms chunk)

| Model | # Params | Inference time (ms) |
|---|---|---|
| MLP | ~0.1M | ~2 |
| GRU | ~0.5M | ~3 |
| POSSM-SS (GRU backbone) | ~0.47M | ~2.44 ms/chunk (CPU) |
| o-POSSM (GRU backbone) | ~8M | ~5.65 ms/chunk (CPU) |
| POYO | ~12M | ~30 ms/chunk |
| NDT-2 | ~12M | ~30 ms/chunk |

POSSM achieves up to 9x faster inference than POYO on GPU. Both single-session and pretrained POSSM variants operate well within the real-time BCI latency threshold of ≤10 ms.

### Few-Shot Sample Efficiency
- o-POSSM (pretrained) outperforms single-session POSSM trained from scratch across all sample sizes
- The UI (unit identification only) strategy gives an immediate performance boost even before any training steps, just from initializing new embeddings with a pretrained backbone
- Single-session models sometimes fail to converge to pretrained model performance even with extended training

---

## Cross-Patient Relevance
**What this means for cross-patient speech decoding from intra-op uECOG (~20 patients, ~15 min each):**

1. **Pretraining on NHP data can bootstrap human decoders**: The cross-species handwriting result is the strongest evidence here -- NHP motor cortex pretraining (no human data) yields significant accuracy gains on human motor cortex. If uECOG speech data is limited, pretraining on any large intracortical motor corpus (including NHP) could help, especially with the full finetuning strategy.

2. **Unit identification enables few-parameter adaptation**: The UI strategy (freeze backbone, train only unit + session embeddings) requires updating <1% of parameters. This is attractive for intra-op settings with very short recording windows (~15 min). A pretrained backbone could be adapted to a new patient's uECOG channel layout with minimal data by learning new per-channel embeddings while keeping the learned dynamics frozen.

3. **Spike tokenization handles variable electrode configurations naturally**: POSSM's per-unit embedding scheme means each patient's uECOG grid (which differs in size, layout, and coverage) is automatically handled without architecture changes. No fixed input dimensionality is assumed -- only a mapping from channel identity to a learned embedding.

4. **SSM backbone avoids transformer's quadratic cost for long contexts**: For sentence-level speech decoding (2-18 second trials), POSSM's recurrent backbone processes each 20-50 ms chunk in constant time and propagates context through the hidden state. This is directly applicable to streaming uECOG decoding where full trial reprocessing is infeasible.

5. **Cross-patient pooling via pretraining**: The multi-dataset NHP pretraining demonstrates that pooling data from multiple subjects/sessions/labs yields better generalization than any single session. The same logic applies to pooling across 20 intra-op uECOG patients -- even with 15 min each (5 hours total), pretraining the shared backbone and then finetuning per-patient embeddings may capture shared speech motor representations.

6. **Two-phase speech training pipeline is directly applicable**: The POSSM speech setup (phase 1: reconstruct spike counts with cross-attention encoder; phase 2: CTC phoneme decoding) can be applied to uECOG features (e.g., high-gamma power). The reconstruction pretraining phase uses no labels and could pool data across patients.

---

## Limitations

1. **No cross-patient speech pretraining demonstrated**: The cross-species transfer result is for handwriting (classification), not speech (sequence decoding). Whether NHP motor pretraining helps speech decoding is not tested. The speech experiments are single-participant only (Willett et al. 2023 dataset with 1 subject).

2. **Speech setup deviates significantly from NHP setup**: The speech experiments cannot use POYO-style spike tokenization (only binned spike counts are available), requiring a separate adapted architecture. This limits direct comparison and means the cross-species speech transfer advantage is unvalidated.

3. **No cross-patient transfer results for speech**: All speech decoding numbers come from a single patient trained on their own data. The cross-patient generalization capability (the core goal for intra-op uECOG) is not demonstrated.

4. **Intracortical spikes vs. uECOG field potentials**: POSSM is designed for intracortical spiking data (Utah arrays). uECOG records local field potentials at much lower spatial resolution; the unit embedding scheme and spike tokenization do not directly apply. Adaptation would require either treating electrode channels as "units" (as done for the speech dataset) or redesigning the tokenizer.

5. **Short intra-op recording duration**: The paper shows strong few-shot learning, but the minimum validated setting is multiple sessions per subject (11 sessions for handwriting). Adapting from a single 15-minute session is more extreme than anything tested here.

6. **Causal speech decoding still substantially behind non-causal**: Uni-directional POSSM-GRU achieves 27.32% PER vs. 25.80% for bi-directional -- the causal constraint imposed by real-time requirements costs ~1.5 PER points. For intra-op use, where real-time decoding may not be required (offline analysis), the bi-directional variant could be used.

7. **Speech baseline is Willett et al. 2023 data**: This dataset uses Utah arrays in the hand motor/premotor cortex + Broca's area, which is different from uECOG coverage. The recording modality, spatial resolution, and neural features differ from what would be available in an intra-op uECOG setting.

---

## Reusable Ideas

1. **Spike tokenization with per-unit embeddings (from POYO)**: Representing each neural channel as a learnable embedding allows the model to handle any number of channels, in any layout, without architecture changes. For uECOG, treat each electrode contact as a "unit" and learn a contact-specific embedding. This is the foundation for cross-patient pooling without requiring channel alignment across patients.

2. **Cross-attention tokenizer (PerceiverIO-style) to compress variable-length inputs**: Use cross-attention with learnable queries to map a variable-length token sequence (one per spike or per time-bin-per-channel) into a fixed-size latent vector. This is cheaper than full self-attention and decouples input dimensionality from model width.

3. **SSM recurrent backbone for constant-time streaming inference**: Replace the Transformer backbone with a GRU, S4D, or Mamba backbone that maintains a hidden state. Each new time chunk is processed in O(1) time regardless of history length. For intra-op sessions where streaming or low-latency output is desired, this is a practical engineering advantage.

4. **Unit identification (UI) for fast adaptation (<1% parameter update)**: After pretraining, initialize new unit and session embeddings for a new patient, freeze everything else, and train only those embeddings. This can be done with very little data and enables rapid deployment for each new intra-op patient.

5. **Full finetuning with gradual unfreezing**: Start with UI warm-up, then unfreeze the entire model for end-to-end adaptation. This retains pretrained structure while allowing the model to adapt to new patients. Outperforms both UI-only and training from scratch in all tested cross-subject/cross-species conditions.

6. **Multi-dataset pretraining with unit dropout augmentation**: Randomly dropping entire units (and all their spikes) from each batch forces the model to decode from partial electrode subsets. This implicitly simulates the channel variability seen across patients and recording sessions, improving robustness to missing or degraded channels -- highly relevant for intra-op uECOG where channels vary in quality.

7. **Session embedding for latent factors of recording context**: A learnable session embedding, injected into the output cross-attention module, captures session-specific factors (electrode impedances, recording depth, behavioral context) without explicit modeling. For uECOG, a patient embedding could capture inter-patient variability in functional organization.

8. **Two-phase speech training (reconstruction pretraining + CTC decoding)**: Phase 1 trains the encoder to reconstruct neural features without labels (self-supervised, can pool all patients). Phase 2 fine-tunes on phoneme sequences with CTC loss (supervised, per-patient or pooled if cross-patient labels are available). This pipeline is well-suited to the intra-op setting where labeled data is scarce.

9. **Output cross-attention with temporal RoPE queries**: Instead of predicting one output per input chunk, the output module queries the hidden state at arbitrary target timestamps encoded with RoPE. This allows predicting at finer temporal resolution than the chunk size, handling variable output rates, and predicting future states (with adjustable lag). Useful for aligning neural predictions with phoneme boundaries that may not align with fixed time bins.

10. **Leverage NHP motor data as a starting point for human cortical decoders**: The cross-species handwriting result shows that shared motor dynamics between macaques and humans are sufficient for NHP pretraining to benefit human decoder finetuning. Even without human pretraining data, NHP corpora (which are much larger and more diverse) can bootstrap a useful prior over motor cortex dynamics that transfers across species.
