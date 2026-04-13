# Karpowicz et al. 2025 - NoMAD: Nonlinear Manifold Alignment of neural Dynamics

## Citation
Karpowicz, B.M., Sedler, A.R., Keshtkaran, M.R., Bodkin, K., Ma, X., Miller, L.E., & Pandarinath, C. (2025). Stabilizing brain-computer interfaces through alignment of latent dynamics. *Nature Communications*, 16, 2025.

## Setup
- **Recording modalities**: Utah microelectrode arrays (96ch), intracortical spike counts
- **Species**: Monkeys (M1 reaching, M2 reaching + grasping) and human (T5 handwriting BCI)
- **Data scale**: M1: 71 sessions over 2 years. M2: 109 sessions over 2 years. T5: 9 sessions (subset of Willett BrainGate data)
- **Tasks**: Cross-session decoder stabilization -- maintain decoder performance across days/months without behavioral recalibration
- **Compute**: LFADS training + lightweight alignment network training

## Architecture
- **Base decoder**: LFADS (Latent Factor Analysis via Dynamical Systems). Variational autoencoder for neural population dynamics. Trained on a reference session to extract low-dimensional latent dynamics
- **Alignment network**: Small feedforward network that maps new-session neural activity into the reference session's latent manifold
  - Architecture: Linear(96 → 64) → ReLU → Linear(64 → 64) → ReLU → Linear(64 → latent_dim)
  - ~36K-74K parameters depending on latent dimensionality (vs millions for the LFADS backbone)
  - Trained to minimize KL divergence between aligned new-session latent distribution and reference session's latent distribution
- **Pipeline**: New session neural data → alignment network → reference LFADS encoder space → frozen LFADS dynamics → frozen linear readout → kinematics
- **Frozen backbone**: LFADS encoder, dynamics, and readout are NEVER retrained. Only the alignment network is trained for each new session

## SSL/Pretraining
- **LFADS pretraining**: Trained as a generative model on the reference session. Learns latent dynamics of the neural population. This is the "backbone" that remains frozen
- **Alignment training**: Unsupervised -- uses KL divergence between aligned neural activity distribution and reference distribution. NO behavioral labels from the new session required
- **Real-time inference (RTI)**: Can also be combined with real-time behavioral feedback for semi-supervised alignment, which significantly improves human BCI performance

## Cross-Patient Handling
- **Within-subject ONLY**: NoMAD aligns sessions from the same subject. Different sessions have the same electrodes in the same brain locations, just with impedance/gain drift
- **NOT cross-subject**: Explicitly not designed for different electrode placements. The alignment network learns a session-specific mapping that assumes consistent spatial structure across sessions
- **Single reference advantage**: Training on a single good reference session outperforms sequential alignment (chaining alignments across days). This avoids error accumulation

## I/O Features
- **Input**: 96-channel spike counts at 20ms bins from Utah array
- **Output**: 2D cursor velocity (reaching) or character probabilities (handwriting)
- **Alignment**: Neural activity → lightweight network → LFADS latent space (unsupervised, no behavioral labels needed)

## Key Results
| Dataset | Metric | NoMAD | No Alignment |
|---|---|---|---|
| M1 reaching (2yr) | R^2 median | 0.91 | ~0.70 (degrades) |
| M1 stability half-life | days | 208 | ~30 |
| M2 reaching (2yr) | R^2 median | 0.87 | ~0.65 |
| T5 handwriting (RTI) | R^2 | 0.72-0.83 | ~0.50 |
| T5 handwriting (no RTI) | R^2 | ~0.55 | ~0.50 |

Key findings:
- **Stability half-life 208 days**: Decoder performance degrades with a half-life of 208 days with NoMAD alignment, vs ~30 days without. 7x improvement in stability
- **Unsupervised alignment works**: KL divergence alignment with zero behavioral labels maintains reasonable performance. Adding real-time inference (RTI) feedback substantially improves human BCI (R^2 0.55 -> 0.72-0.83)
- **Single reference session is best**: Training LFADS on ONE high-quality reference session, then aligning all future sessions to it, outperforms sequential alignment or multi-session training
- **Lightweight alignment network**: Only ~36K-74K params need to change per session, while the ~millions-param LFADS backbone stays frozen. Efficient adaptation
- **Nonlinear > linear alignment**: The feedforward alignment network outperforms linear Procrustes alignment, especially for longer temporal gaps between sessions

## v12 Comparison

**Establishes the frozen-backbone + lightweight-adapter paradigm for neural decoder stability.** NoMAD's architecture -- heavy pretrained backbone frozen, tiny per-session adaptation network -- is the same design philosophy as v12's shared backbone + 134-param per-patient diagonal normalization.

**Key parallels:**
- NoMAD's alignment network (~36-74K params, session-specific) maps new neural data into a reference latent space. v12's per-patient diagonal (128 scale + 128 bias = 128 params) + delta/omega (6 params) does the same conceptually -- maps new patient's neural data into a shared representational space
- Both freeze the backbone during adaptation. NoMAD never retrains LFADS; v12 freezes the shared backbone when fitting per-patient layers
- Both operate in a low-data adaptation regime. NoMAD needs minutes of unlabeled data; v12 has ~1 min of labeled data per patient
- Single reference analogy: NoMAD's best reference session ~ v12's atlas prior (Brainnetome). Both establish a canonical space to align toward

**Key differences:**
- **Within-subject vs cross-subject**: NoMAD handles temporal drift (same electrodes, different sessions). v12 handles spatial permutation (different electrodes, different patients). The latter is a fundamentally harder alignment problem -- NoMAD's alignment network can exploit consistent electrode-to-neuron mapping, which v12 cannot assume
- **Unsupervised vs supervised**: NoMAD aligns with KL divergence (no labels). v12's per-patient layers are trained supervised (phoneme labels). Unsupervised alignment could be valuable if v12 gets enough unlabeled continuous data from SSL
- **Spike input vs HGA**: Different biophysics. NoMAD's manifold structure arises from neural population dynamics of spikes; v12's from postsynaptic field potentials. Cannot directly compare
- **Continuous output vs discrete phonemes**: Kinematics vs 9-class classification -- different readout challenges

**What to import:**
- **KL-based manifold alignment**: If v12 acquires sufficient unlabeled continuous data (456 min available), unsupervised alignment of new patients to the pretrained latent space via KL divergence could enable zero-shot or minimal-label adaptation
- **Single reference principle**: Train backbone on the best data available (atlas-grounded, high-quality patients), then align everything else to that reference. Don't try sequential multi-patient training
- **RTI feedback loop**: For eventual online deployment, real-time inference feedback (user sees decoder output, adjusts behavior) dramatically improves alignment. Relevant for clinical translation

**What doesn't transfer:**
- **Same-subject assumption**: NoMAD's core assumption (same electrodes across sessions, consistent spatial structure) doesn't hold for v12's cross-patient setting. The alignment problem is qualitatively different
- **LFADS dynamics model**: LFADS models temporal dynamics of spike trains using a dynamical systems prior. This is well-suited for motor reaching but may not capture speech production dynamics (which are more discrete and sequential)
- **96-channel Utah arrays**: Fixed electrode count and layout. v12 must handle 63-256 variable channels with different spatial configurations
