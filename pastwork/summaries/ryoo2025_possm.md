# Ryoo et al. 2025 - POSSM: POYO + SSM for Real-Time Neural Decoding

## Citation
Ryoo, M., Azabou, M., Dyer, E.L., et al. (2025). Scaling Multi-Session Brain-Computer Interfaces with State Space Models. *NeurIPS 2025*.

## Setup
- **Recording modality**: Intracortical spike trains (Utah arrays, Neuropixels) + speech (BCI handwriting/speech datasets)
- **Species**: Monkeys (motor reaching) and humans (handwriting + speech)
- **Data scale**: 148 sessions pretrained (monkey motor). Cross-species transfer tested to human handwriting and human speech
- **Tasks**: Motor reaching/cursor control (monkey), handwriting decoding (human), speech decoding via CTC (human)
- **Compute**: Real-time inference <6ms per timestep on CPU (critical for BCI deployment)

## Architecture
- **Input tokenization**: POYO-style spike tokenization. Each spike is a token with (unit_id, timestamp). Per-unit learned embeddings + RoPE temporal encoding
- **Cross-attention bottleneck**: Perceiver-style -- L learned latent tokens cross-attend to the variable-length spike sequence. Compresses variable neuron counts into fixed-size latent representation (same as POYO)
- **Recurrent backbone (KEY INNOVATION)**: Replaces POYO's Transformer self-attention with a recurrent state space model. Three variants tested:
  1. **S4D**: Structured State Space for Sequences (diagonal parameterization). Best accuracy
  2. **GRU**: Standard gated recurrent unit. Competitive with S4D
  3. **Mamba**: Selective state space model. Slightly worse than S4D on some tasks
- **Scale**: ~4.6M (small) to ~86M (large) total parameters depending on configuration
- **Decoder**: Linear readout for kinematics (MSE) or CTC head for speech/handwriting
- **Two-phase training**:
  1. **Phase 1 (reconstruction)**: Self-supervised temporal masking -- predict held-out time segments of neural activity. Trains backbone representations
  2. **Phase 2 (task)**: Supervised fine-tuning with task-specific loss (MSE for kinematics, CTC for speech/handwriting). Optional: freeze backbone, train decoder only

## SSL/Pretraining
- **Phase 1**: Temporal masking reconstruction (predict future/masked neural activity from observed context). Self-supervised. 148 monkey motor sessions
- **Phase 2**: Supervised CTC or MSE on downstream task data
- **Cross-species transfer**: Pretrain on 148 monkey motor sessions, transfer to human handwriting or human speech. This is EXTREME domain transfer -- different species, different brain areas, different tasks
- **Transfer protocol**: Keep backbone weights, reinitialize per-unit embeddings for human sessions. Fine-tune entire model on human data

## Cross-Patient Handling
- **Per-unit learned embeddings**: Same as POYO. Each neuron gets a unique embedding. For new sessions/species, fresh embeddings are initialized and learned
- **Cross-attention handles variable neuron counts**: Different sessions can have 50-500+ neurons. The Perceiver bottleneck maps them all to the same latent dimension
- **Cross-species transfer works**: Monkey motor → human handwriting improves decoding by +2% over training from scratch. Monkey motor → human speech yields PER 19.80% (though the speech setup has extensive per-patient data)
- **Recurrent backbone enables real-time**: Unlike Transformer self-attention (O(T^2) or at minimum full-context), recurrent models process one timestep at a time with O(1) memory. Critical for online BCI

## I/O Features
- **Input**: Per-spike tokens (unit_id + timestamp) for spike data. Threshold crossings + spike-band power for speech/handwriting
- **Output**: Continuous kinematics (cursor velocity) via MSE, or character/phoneme sequences via CTC
- **Temporal**: RoPE on spike timestamps (continuous time, not binned)
- **Spatial**: Per-unit learned embeddings (no coordinates)

## Key Results
| Task | POSSM (S4D) | POYO (Transformer) | Notes |
|---|---|---|---|
| Monkey reaching R^2 | 0.94 | 0.96 | Slight accuracy tradeoff for speed |
| Human handwriting (cross-species) | +2% vs scratch | baseline | Monkey→human transfer |
| Human speech PER (cross-species) | 19.80% | N/A | Monkey motor → human speech CTC |
| Inference latency (CPU) | <6ms | ~50ms | 8x faster (recurrent vs Transformer) |

Key findings:
- **SSM/GRU ≈ Transformer accuracy with 8x inference speed**: The recurrent backbone trades <2% accuracy for 8x faster real-time inference. For BCI deployment, this tradeoff is overwhelmingly favorable
- **Cross-species transfer provides modest gains**: Monkey→human improves handwriting by +2%. Not transformative, but demonstrates that temporal dynamics have some cross-species universality
- **Cross-species speech (PER 19.80%)**: Monkey motor cortex pretraining transfers to human speech decoding. PER 19.80% is respectable but far from Nason 2024's 7-15% raw PER. The comparison is confounded by data regime differences
- **Two-phase training (SSL then supervised) works for recurrent models**: Temporal masking reconstruction provides useful initialization even for GRU/SSM backbones, not just Transformers
- **S4D slightly > GRU > Mamba for neural decoding**: S4D's structured linear recurrence best captures the smooth dynamics of motor cortex

## v12 Comparison

**POSSM validates three design choices relevant to v12: Perceiver cross-attention, temporal masking SSL, and the viability of non-Transformer backbones.** The cross-species transfer results are tangentially interesting but not directly applicable (spike vs HGA modality barrier).

**Key parallels:**
- Perceiver cross-attention (POYO/POSSM) maps variable sensor counts to fixed latent space -- architecturally identical to v12's VE cross-attention. POSSM uses learned latent positions; v12 uses atlas-grounded VE positions
- Two-phase training (SSL reconstruction then supervised fine-tuning) matches v12's planned 3-stage pipeline (sEEG SSL → uECoG SSL → supervised)
- Temporal masking for Phase 1 SSL directly parallels v12's temporal span masking plan

**Key differences:**
- **Modality**: Spikes vs HGA. Cannot pool. Cross-species spike transfer tells us nothing about cross-patient HGA transfer
- **Real-time constraint**: POSSM optimizes for <6ms inference. v12 has no real-time requirement (offline analysis). Transformer self-attention is fine for v12
- **Per-unit embeddings vs per-electrode diagonal**: POSSM learns unique embeddings per neuron (works because Utah arrays have fixed geometry). v12 uses MNI coordinates + diagonal normalization (handles variable array placement)
- **Cross-species vs cross-patient**: POSSM's monkey→human transfer is impressive but irrelevant to v12's human→human cross-patient problem. The variation axes are completely different

**What to import:**
- **Two-phase training validation**: SSL temporal masking reconstruction (Phase 1) → supervised task fine-tuning (Phase 2) works for recurrent backbones too. If v12 uses GRU components (BiGRU in the temporal attention path), this confirms the SSL→supervised pipeline generalizes beyond Transformers
- **S4D as backbone alternative**: If v12's temporal self-attention is a bottleneck, S4D offers a principled recurrent alternative. But v12 operates on 16 VEs x ~20 time bins -- the sequence is short enough that Transformer self-attention is efficient
- **PER 19.80% as cross-domain reference**: Even EXTREME domain transfer (monkey motor → human speech) achieves PER ~20%. v12's within-species, within-modality, within-task cross-patient transfer should substantially beat this

**Common mistakes:**
- Do NOT cite PER 19.80% as a competitive speech result. This is cross-species transfer with a massive domain gap. Nason 2024 achieves 7-15% PER with patient-specific training on the SAME modality
- Do NOT adopt SSM/GRU backbone for v12 because POSSM uses it. POSSM's motivation is real-time inference (<6ms). v12 has no latency constraint -- Transformer self-attention on 16 VEs is fast enough
- Do NOT assume cross-species transfer validates cross-patient transfer. The variation sources are fundamentally different (species biology vs electrode placement)
- Do NOT confuse POSSM's 148-session pretraining scale with data abundance. Each session is short motor reaching -- total data hours are modest compared to BIT's 367h or NDT3's 2000h
