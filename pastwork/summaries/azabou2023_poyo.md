# Azabou et al. 2023 - POYO: Population Dynamics from a Single Trial

## Citation
Azabou, M., Arora, V., Ganesh, V., Mao, X., Nachimuthu, S., Mendelson, M.J., Park, B., Adams, J.L., Dyer, E.L. (2023). A Unified, Scalable Framework for Neural Population Decoding. *NeurIPS 2023*.

## Setup
- **Recording modality**: Intracortical spike trains (Utah arrays, Neuropixels). Input is per-unit spike timestamps (not binned counts)
- **Species**: Monkeys (7 animals) and mice
- **Data scale**: 178 sessions, multiple tasks. ~35 sessions for the multi-session reaching pretraining
- **Tasks**: Reaching (cursor, cycling, maze), grasping, handwriting, speech attempted
- **Compute**: Not specified. Architecture is lightweight enough for single-GPU training

## Architecture
- **Input tokenization**: Each spike is a token: (unit_id, timestamp). No binning -- continuous time. This preserves sub-millisecond temporal resolution. Special `[CLS]` and `[SEP]` delimiter tokens between units
- **Unit embeddings**: Learned embedding per unit. NOT coordinate-based -- each unit gets a unique learned vector. For new sessions, new embeddings are initialized and trained. This is the primary per-session component
- **Temporal encoding**: Rotary Position Encoding (RoPE) applied to spike timestamps. Enables continuous-time attention without discretization
- **Value rotation**: Additional rotation applied to value vectors (not just queries/keys) based on temporal position. Improves temporal sensitivity beyond standard RoPE
- **Architecture**: Perceiver-style cross-attention. 512 learned latent tokens cross-attend to the variable-length spike sequence, then self-attend among themselves. This maps variable neuron counts to a fixed-size representation
- **Decoder**: Linear readout from the 512 latent tokens to behavioral variables
- **Scale**: ~7.4M parameters total. Unit embeddings are <1% of total params
- **Delimiter tokens**: `[SEP]` tokens inserted between different units' spike trains. When a unit is absent (dropped out or not recorded), its section is simply omitted -- the model naturally handles variable unit counts. No padding or masking needed

## SSL/Pretraining
- **NOT self-supervised**: Supervised multi-session pretraining on paired neural + behavioral data. Trains on many sessions simultaneously with per-session unit embeddings but shared backbone
- **Transfer protocol**: Gradual unfreezing. First train only unit embeddings (frozen backbone) on target session, then progressively unfreeze backbone layers from output to input. Prevents catastrophic forgetting of pretrained representations
- **No SSL objective**: Purely supervised. Contrasts with BIT/BarISTA/NEDS which use masking-based SSL

## Cross-Patient Handling
- **Per-unit learned embeddings**: Each neuron gets a unique embedding vector. For a new session, fresh embeddings are initialized and learned during the frozen-backbone warmup phase. This is the sole per-session adaptation mechanism
- **Cross-animal transfer**: R^2=0.94 on held-out animal (monkey reaching) after fine-tuning with gradual unfreezing. R^2=0.72 zero-shot (no fine-tuning). Strong evidence that motor cortex dynamics are shared across animals when properly aligned
- **No spatial encoding**: No electrode coordinates, no atlas, no MNI. Spatial identity encoded entirely through learned per-unit embeddings. Works because Utah arrays have consistent geometry within species, but fundamentally limited for cross-species or cross-array transfer
- **Unit identification**: The model can identify which unit is firing from the latent representation (<1% of params). This emergent capability suggests the unit embeddings capture meaningful spatial/functional identity

## I/O Features
- **Input**: Continuous spike timestamps per unit (NOT binned, NOT field potentials). Each spike is one token
- **Output**: Continuous behavioral kinematics (cursor velocity, hand position) via MSE regression
- **Temporal**: Continuous-time via RoPE -- no fixed temporal bins
- **Spatial**: Learned per-unit embeddings -- no coordinate system

## Key Results
| Configuration | Reaching R^2 | Notes |
|---|---|---|
| POYO multi-session (same animal) | 0.96 | 35-session pretrain |
| POYO cross-animal (fine-tuned) | 0.94 | Gradual unfreezing |
| POYO cross-animal (zero-shot) | 0.72 | No target data |
| NDT (single session) | 0.89 | Prior SOTA |
| LFADS (single session) | 0.87 | Established baseline |

Key findings:
- **Perceiver bottleneck scales to variable neuron counts**: 512 latent tokens successfully compress 50-500+ neurons into a fixed representation. Architecturally validated for neural decoding
- **Cross-animal transfer works**: R^2=0.94 with fine-tuning demonstrates that motor cortex dynamics are sufficiently conserved to transfer across individuals
- **Gradual unfreezing is critical**: Direct fine-tuning of all layers at once degrades pretrained representations. Progressive unfreezing preserves backbone quality while adapting to new data
- **Delimiter tokens for absent units**: Clean solution to variable electrode counts -- simply omit absent units' token sections. No padding needed

## v12 Comparison

**Perceiver cross-attention architecture directly validates v12's VE design.** POYO's 512 learned latents cross-attending to variable-length spike sequences is architecturally analogous to v12's 16 VEs cross-attending to variable electrode HGA. Both solve the same fundamental problem: mapping variable sensor counts to a fixed-size shared representation. The key difference is what grounds the latent positions:

- **POYO**: Learned latent tokens with no spatial prior. 512 latents is far more than needed (motor reaching is low-dimensional). Works because data is abundant and single-task
- **v12**: 16 atlas-grounded VE positions (Brainnetome) with distance-biased cross-attention. Spatial prior compensates for v12's scarce data (~1 min/patient epoched)

**Key parallels:**
- Per-unit embeddings (POYO) ~ per-electrode diagonal normalization (v12). Both provide per-sensor identity with minimal params. POYO's are learned from scratch; v12's are scale+bias (physics-motivated for impedance/gain variation)
- Gradual unfreezing transfer protocol is directly importable for v12's downstream adaptation if 134 per-patient params prove insufficient
- Cross-animal R^2=0.94 demonstrates that biological neural dynamics ARE shared -- the barrier is sensor variability, not neural variability. Validates v12's premise

**Key differences:**
- **Modality**: Spikes (action potentials) vs HGA (postsynaptic field potentials). Fundamentally different biophysics -- cannot pool. POYO's success on spikes does not directly predict HGA performance
- **Spatial encoding**: POYO uses NONE -- learned embeddings only. v12 uses MNI coordinates + Fourier PE + atlas VEs. POYO can afford this because Utah arrays have fixed geometry; v12 cannot because uECOG arrays are placed variably across patients
- **No SSL**: POYO is purely supervised. v12 plans temporal masking SSL. POYO's success without SSL suggests supervised pretraining is viable when behavioral labels exist (v12 has phoneme labels for all data)
- **Data regime**: POYO has ~35 sessions with hundreds of trials each. v12 has 11 patients with 46-178 trials each. POYO can learn per-unit embeddings from scratch; v12 needs the atlas prior

**What to import:**
- **Delimiter tokens for absent sensors**: Clean approach to variable electrode counts during batching. v12 could use this instead of padding+masking, though VE cross-attention already handles variable counts natively
- **Value rotation**: Applying positional rotation to values (not just Q/K) improved temporal sensitivity. Testable augmentation for v12's temporal self-attention
- **Gradual unfreezing protocol**: Freeze backbone, train per-patient params, then progressively unfreeze from output to input. Importable for v12's Stage 2 adaptation if needed beyond 134-param fine-tuning

**Common mistakes:**
- Do NOT assume POYO's success transfers to field potentials. Spikes and HGA have different biophysics and noise characteristics
- Do NOT use 512 latent tokens for v12. POYO is massively overparameterized for 16-dimensional motor reaching. v12's 16 VEs are calibrated to the number of speech-relevant cortical regions
- Do NOT skip spatial encoding because POYO doesn't use it. POYO works on fixed-geometry Utah arrays; v12 must handle variable uECOG placement across patients
