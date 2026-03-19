# Karpowicz et al. (2025) — NoMAD: Stabilizing BCIs Through Alignment of Latent Dynamics

**Citation:** Karpowicz, B.M., Ali, Y.H., Wimalasena, L.N., Sedler, A.R., Keshtkaran, M.R., Bodkin, K., Ma, X., Rubin, D.B., Williams, Z.M., Cash, S.S., Hochberg, L.R., Miller, L.E. & Pandarinath, C. (2025). Stabilizing brain-computer interfaces through alignment of latent dynamics. *Nature Communications*, 16, 4662. https://doi.org/10.1038/s41467-025-59652-y

---

## Setup

- **Recording modality:** Intracortical microelectrode arrays (Utah arrays, 96-channel, 1.5 mm shanks, Blackrock Microsystems); threshold-crossing spike data
- **Subjects:** Two monkeys (Monkey J: isometric force task; Monkey C: reaching task) + one human clinical participant (T11, BrainGate2 trial, tetraplegia, two 96-channel arrays in left precentral gyrus / PCG)
- **Brain region:** Primary motor cortex (M1) for monkeys; left precentral gyrus (PCG, sensorimotor) for human
- **Tasks:**
  - Monkey isometric wrist force task: 2D center-out cursor control via wrist force, 8 targets, 20 sessions over 95 days
  - Monkey center-out reaching task: planar manipulandum, 8 targets, 12 sessions over 38 days
  - Human BCI closed-loop cursor control: imagined center-out movements to 4 targets, Simon memory-matching task paradigm
- **Data scale:** 96 channels per animal/human; spike data binned at 1 ms then rebinned to 20 ms for modeling

---

## Problem Statement

iBCIs suffer from progressive recording instabilities (electrode drift, cell death, foreign body response) that cause the recorded neural population to change over time, decoupling neural activity from the fixed decoder. Current clinical practice requires frequent supervised recalibration — the user must perform instructed movements while labeled data is collected, which is burdensome and impractical. Existing unsupervised approaches (e.g., Aligned FA, ADAN) treat each time step independently and ignore the temporal evolution of neural activity (i.e., latent dynamics). This paper addresses the gap: **how can dynamics-based latent variable models be used to perform fully unsupervised manifold alignment**, allowing a fixed Day 0 decoder to remain accurate on later-day neural data without any behavioral labels or supervised recalibration?

A secondary gap: prior approaches either fail at longer timescales, have high failure rates, or produce only marginal stability improvements — no prior method achieved stable decoding over months-long windows with zero supervision.

---

## Method

### Overall Architecture: Two-Stage Pipeline

**Stage 1 — Day 0 Supervised Training (LFADS + behavioral readout)**

NoMAD's core dynamics model is LFADS (Latent Factor Analysis via Dynamical Systems), a modified sequential variational autoencoder (VAE) adapted for spiking neural data. The architecture uses:

- **Encoder RNN:** Takes binned spike sequences X(t), produces a posterior distribution Q(Z|X(t)) over a latent code Z that initializes the dynamical system
- **Controller RNN:** Encodes observed data and infers time-varying inputs U(t) fed to the Generator, handling inputs the autonomous dynamics cannot model alone
- **Generator RNN (GRU-based):** The core recurrent network implementing the learned dynamics; takes initial state s[0] and inferred inputs U(t), evolves a latent state s(t) via s(t+1) = f(s(t), u(t))
- **Rates readout:** Linear matrix from Generator states to predicted firing rates; trained with Poisson negative log-likelihood loss against observed spikes
- **Behavioral readout (Wiener filter):** Linear matrix from Generator states to continuous behavioral variables (force, velocity); trained with MSE loss using a separate Adam optimizer on a separate learning rate schedule

Two key modifications from standard LFADS:
1. A **low-dimensional read-in matrix** at the model's input standardizes dimensionality across days regardless of how many channels are active (handles channel dropout/addition)
2. The **behavioral readout** during Day 0 training ensures that the learned manifold and Generator states are anchored to behaviorally relevant structure, making them more alignable

Training objective: maximize ELBO = Poisson NLL reconstruction + KL divergence from prior P(Z). All weights updated via backpropagation through time with Adam optimizer and gradient clipping. Spike jittering data augmentation (up to ±2 bins) prevents overfitting to individual spikes.

For human data, AutoLFADS is used for hyperparameter search (trains ~25 workers in parallel, mutates toward high-performing hyperparameter configurations).

**Stage 2 — Day K Unsupervised Alignment**

On a later day K, the LFADS model weights (Generator, Controller, Encoder) are **frozen**. A lightweight feedforward **Alignment Network** (2-layer Dense network with ReLU activations and identity initialization) is inserted before the frozen LFADS model. The alignment network learns a transformation of Day K spikes that makes them compatible with the Day 0 dynamics model.

Three additional components are also updated during alignment:
- The **low-D read-in** matrix
- The **linear readout from Generator states to inferred firing rates** for Day K
- The **linear readout from LFADS factors to inferred firing rates** for subsequent sessions

All other model weights (Generator RNN, Controller RNN, Encoder RNN, behavioral readout) remain frozen at Day 0 values.

### Alignment Training Objective

The alignment is trained simultaneously with two objectives:

1. **KL divergence minimization (distribution alignment):** The distributions of Generator hidden states are compared between Day 0 data and Day K data. Each distribution is approximated as a multivariate Gaussian. The Kullback-Leibler divergence between these two distributions is computed using the closed-form formula:

   D_KL(N_0 || N_K) = 1/2 * [tr(Sigma_K^{-1} Sigma_0) + (mu_K - mu_0)^T Sigma_K^{-1} (mu_K - mu_0) - m + ln(det Sigma_K / det Sigma_0)]

   where m = number of Generator RNN units (m = 100). This is the core alignment pressure: it forces the distribution of Day K Generator states to match the distribution of Day 0 Generator states, using first- and second-order statistical moments (mean and covariance) rather than higher-order statistics. The multivariate Gaussian approximation makes the problem tractable.

2. **Reconstruction loss (Poisson NLL):** The Day K spiking activity is reconstructed through the aligned pipeline, providing a self-supervised signal that ensures the alignment does not collapse to a degenerate solution.

Total loss = weighted sum of KL cost + reconstruction cost. Learning rate is annealed (multiplied by 0.95 when no validation improvement). Training stops when validation loss plateaus for a fixed number of consecutive epochs.

### Causal Inference

LFADS is normally acausal (future data informs current state). For real-time iBCI simulation, NoMAD uses a **sliding window** approach: the model receives N-1 bins of previously observed data plus the current Nth bin, making inference causal with minimal latency. This is evaluated as consistent with real-time use.

### Comparisons

- **Static decoder:** Wiener filter trained on Day 0 smoothed spikes, never updated
- **Aligned FA (factor analysis):** Linear dimensionality reduction baseline; fits a "baseline stabilizer" on Day 0, then aligns Day K data by identifying stable loading rows via iterative alignment + Procrustes rotation
- **ADAN:** Neural network autoencoder + GAN-style adversarial domain adaptation; a discriminator maximizes reconstruction difference between days while a generator minimizes it, achieving distribution alignment in a learned low-D space
- **NoMAD + RTI (Retrospective Target Inference):** Combination of manifold alignment (NoMAD) with retrospective behavioral recalibration for closed-loop sessions

---

## Key Results

### Isometric Force Task (Monkey J, 20 sessions, 95 days)

- **NoMAD cross-session median R² = 0.91** (p = 2.53e-64 vs. all other methods, one-sided Wilcoxon signed-rank test)
- Cross-session medians for baselines: Aligned FA ~0.59, ADAN ~0.65, Static ~0.14
- Day 0 within-day baselines (not cross-session): Aligned FA R² = 0.749 [0.722, 0.761], ADAN R² = 0.916 [0.908, 0.929], Static R² = 0.842 [0.834, 0.846]
- **NoMAD half-life = 208.7 days** vs. Aligned FA 45.1 days, ADAN 76.7 days
- NoMAD: **zero decoding failures** across 380 session pairs; Aligned FA (iso): 51 failures; Aligned FA (reaching): 53 failures; Static (reaching): 78 failures
- Day 0 baseline for NoMAD: median R² = 0.971 [0.965, 0.974] (highest among all methods, indicating the LFADS/Wiener pipeline itself is the best decoder)

### Center-Out Reaching Task (Monkey C, 12 sessions, 38 days)

- This task is harder (coordinated proximal muscle patterns; cursor velocity decoding)
- **NoMAD median R² = 0.78**, no decoding failures (vs. ADAN 0.29, 15 failures; Aligned FA ~0.17; Static 0.022)
- NoMAD half-life = 57.9 days vs. ADAN 7.12 days
- Aligned FA failed repeatedly (53 failures), Static decoder had 78 failures

### Closed-Loop Human BCI (Participant T11, BrainGate2)

- Two sessions tested
- Session 1: NoMAD R² = 0.73 (vs. Static 0.36); NoMAD + RTI = 0.73
- Session 2: NoMAD R² = 0.81 (vs. Static 0.53); NoMAD + RTI = 0.81
- NoMAD + RTI achieved **highest performance across all methods** (median R² = 0.72–0.83) with half-life = 11.7 h and 4.47 days across sessions
- NoMAD alone: half-life = 58.0 h and 21.05 h

### Ablations and Additional Comparisons

- LFADS "stitching" (unsupervised multi-session shared manifold) was attempted but failed without behavioral supervision, confirming that alignment pressure (KL cost) is necessary
- Targeted Neural Dynamical Modeling (TNDM) did not achieve comparable stability to NoMAD
- Aligning sequentially (session-to-session chains) worsened performance vs. always aligning to Day 0
- Normalization preprocessing does not unfairly bias NoMAD (validated in Supplementary Fig. 14)
- Causal inference introduces minimal performance degradation vs. acausal (Supplementary Fig. 3)

---

## Cross-Patient Relevance

**Context:** ~20 patients, ~20 min each, intra-operative uECOG over sensorimotor cortex, speech decoding task.

### What Directly Applies

**The KL divergence distribution matching objective** is the most directly transferable component. In the cross-patient setting, the "alignment" problem is structurally identical to the cross-session problem: a source patient provides labeled data to train a decoder on a latent manifold, and a target patient provides unlabeled data whose neural activity must be mapped to the source manifold. The KL cost between multivariate Gaussian approximations of Generator state distributions is computationally cheap, entirely unsupervised, and does not require behavioral labels from the target patient.

**The two-stage freeze-and-align architecture** transfers cleanly. Train LFADS + behavioral decoder on one (or several pooled) reference patient(s). For a new patient, freeze the dynamics model entirely and train only a small alignment network on the new patient's unlabeled intra-op data. This is well-suited to the ~20 min constraint: alignment can potentially converge in minutes of data if the neural manifold for speech is sufficiently similar across patients (which sensorimotor cortex speech data from uECOG suggests).

**The low-D read-in layer** is directly useful for uECOG. Different patients will have different electrode counts, coverage patterns, and spatial frequency profiles. The read-in matrix standardizes the input dimensionality across patients without any supervised signal, allowing the same LFADS Generator to receive input from heterogeneous arrays.

**Behavioral readout as training anchor.** For the reference patient(s) where labels exist, training with a behavioral readout (speech phoneme/word predictions via MSE or cross-entropy) as a secondary objective ensures the learned manifold is anchored to speech-relevant structure, making cross-patient alignment more likely to succeed.

**The dynamics assumption.** Speech production involves highly structured temporal dynamics in sensorimotor cortex (coarticulation, articulatory trajectories), making LFADS an arguably better fit for speech than for simple isometric force — the dynamical system governing speech motor commands may be more conserved across patients than the particular neurons doing the encoding, which is exactly the setting where latent dynamics alignment excels.

### Key Differences from NoMAD's Original Setting

- **Cross-session vs. cross-patient:** NoMAD addresses within-patient cross-session drift. Cross-patient alignment involves not just manifold rotation/translation but potentially different manifold geometry, dimensionality, and dynamics. The KL divergence between distributions may be much larger cross-patient than cross-session, requiring stronger alignment pressure or more data.
- **Session length:** NoMAD uses sessions of comparable length to the reference. With only ~20 min per intra-op patient, the alignment network may not have enough data to converge fully, especially if the behavioral distribution (phoneme/word coverage) is limited.
- **Task structure:** NoMAD benefits from rich, repeated behavioral sampling (8 reach targets, many trials). Intra-op speech sessions may have sparser phoneme/condition coverage, limiting the distributional diversity needed for robust KL-based alignment.
- **Single reference vs. population reference:** NoMAD aligns to a single Day 0 session. For cross-patient use, a population-level reference manifold (trained on multiple source patients) would be needed. Whether LFADS can learn a shared cross-patient dynamics model is an open question — unsupervised LFADS stitching failed in this paper without behavioral supervision, suggesting a cross-patient shared model may also require careful initialization.
- **uECOG vs. spikes:** NoMAD uses spike threshold crossings. uECOG provides high-gamma band power (continuous-valued). The Poisson NLL reconstruction loss in LFADS is designed for spikes; adaptation to continuous uECOG signals would require replacing this with a Gaussian likelihood or similar. LFADS has been successfully applied to ECoG (cited in the paper, Ref. 47), so this is feasible.

### Proposed Cross-Patient Adaptation

The most viable adaptation of NoMAD for cross-patient uECOG speech decoding would be:

1. Train a reference LFADS model (with behavioral readout) on one or more source patients with labeled speech data, using the modified architecture (low-D read-in, behavioral readout, Gaussian NLL for continuous uECOG)
2. For each new intra-op target patient, freeze the Generator and train only the alignment network + read-in using the KL + reconstruction loss on the unlabeled intra-op neural data
3. Apply the frozen Day 0 behavioral decoder to the aligned target patient latent states

---

## Limitations

- **Spike-based:** All quantitative results use threshold-crossing spike data from Utah arrays. The Poisson NLL loss is specific to spike counts. Direct application to uECOG high-gamma requires architectural modification (Gaussian or other continuous likelihood).
- **Within-patient cross-session only, tested:** NoMAD has not been validated for cross-patient transfer. The paper explicitly notes that multi-session LFADS "stitching" without behavioral supervision fails, raising questions about whether a shared cross-patient manifold can be learned unsupervised.
- **Stable manifold assumption:** The method assumes that the underlying neural manifold and dynamics are stable over the alignment period. For new-skill learning (e.g., a patient learning to use the BCI), manifold-level changes would violate this assumption. For intra-op naïve subjects with no BCI experience, the manifold may not be stable within the 20-min window in the way it is for trained monkeys.
- **Computationally intensive:** LFADS training and alignment are GPU-dependent and not suitable for implanted wireless devices. All processing must be done on external hardware. AutoLFADS for hyperparameter search is particularly resource-heavy.
- **Single reference session alignment:** NoMAD always aligns to Day 0 (a single reference session), avoiding error accumulation from sequential alignment. For cross-patient use, the reference "session" would need to be a population-level model — how to construct this robustly is unresolved.
- **Offline demonstration:** All quantitative results are from offline (retrospective) analysis. The authors acknowledge that offline decoding accuracy does not always predict online closed-loop performance improvements.
- **Behavior-rich reference required:** The Day 0 reference must contain rich behavioral diversity (many trials, many conditions) for the Wiener filter decoder and behavioral readout to be well-specified. Short intra-op sessions on source patients may not provide sufficient coverage.
- **Hyperparameter sensitivity:** Alignment performance is sensitive to learning rate, bin size, and segment overlap. The same hyperparameters worked for both monkey tasks, but human data required AutoLFADS search — suggesting cross-modal or cross-patient application may require additional tuning.
- **Does not generalize across behaviors:** The paper notes that different behaviors may occupy distinct manifolds (Ref. 57), and all tested datasets use single consistent behaviors. Aligning across different speech tasks or phoneme sets may require additional mechanisms.

---

## Reusable Ideas

### 1. KL Divergence Distribution Matching as Unsupervised Alignment Loss

The multivariate Gaussian KL divergence between Generator state distributions is the core alignment mechanism. It is:
- Closed-form (no sampling required)
- Computationally cheap (requires only computing mean and covariance of hidden states)
- Entirely unsupervised
- Differentiable end-to-end

This exact loss function can be borrowed verbatim for aligning cross-patient uECOG latent spaces, replacing or supplementing adversarial domain adaptation losses (like ADAN's GAN objective). The Gaussian approximation makes it stable to train compared to adversarial approaches.

### 2. Freeze-and-Align Architecture Pattern

The pattern of training a reference model fully supervised, then freezing the dynamics core and training only a small adaptation front-end on unlabeled target data, is a powerful and sample-efficient approach for cross-patient transfer. The alignment network (2-layer Dense, ReLU, identity init) is deliberately small to avoid overfitting to limited target data — a desirable property for 20-min sessions.

### 3. Low-Dimensional Read-In for Heterogeneous Arrays

The low-D read-in matrix handles electrode count variability across days/patients without supervision. For cross-patient uECOG where coverage and channel count vary substantially, this component is directly reusable and likely more important than in the within-patient setting.

### 4. Dual-Objective Training (Reconstruction + Distribution Alignment)

The combination of a self-supervised reconstruction loss (prevents degenerate alignment collapse) with a distribution-matching alignment loss (provides alignment pressure) is a general principle. This is analogous to a domain adaptation objective in transfer learning, but applied to dynamical systems. The dual objective can be ported to any latent variable model architecture, not just LFADS.

### 5. Behavioral Readout as Manifold Anchoring During Reference Training

Training the reference model with a behavioral prediction readout as a secondary objective (predicting force/velocity/phonemes from Generator states) ensures the learned manifold is behaviorally structured and not arbitrary. For speech decoding, including phoneme or articulatory feature prediction during source-patient LFADS training would make the resulting manifold more alignable to new patients performing the same speech task.

### 6. Always-Align-to-Reference (Not Sequential) Strategy

Aligning every new patient/session to a fixed reference (rather than chaining alignments session-to-session) prevents error accumulation. For a cross-patient system, this argues for maintaining a fixed "canonical" reference patient or population model and always aligning new patients to that reference.

### 7. Dynamics as Stabilizer Across Population

The paper's core insight — that latent dynamics (the rules governing how neural state evolves over time) are more stable and more cross-individual than the particular neurons doing the encoding — is the key conceptual contribution. For speech, sensorimotor dynamics during articulation are likely conserved across patients at the population level even if individual electrodes differ, making this assumption more defensible than for arbitrary motor tasks.

### 8. Causal Sliding Window Inference

The sliding window inference scheme (N-1 past bins + current bin) achieves near-acausal performance with real-time compatibility. This is directly applicable to online speech decoding where low latency is required.

### 9. Spike Jittering Data Augmentation

Randomly shifting spike times by ±2 bins during LFADS training reduces overfitting to precise spike timing artifacts without degrading dynamics learning. This is applicable to uECOG high-gamma data via analogous time-domain jittering.

### 10. Normalization per Channel Across Days/Patients

Per-channel z-scoring (zero mean, unit variance using statistics saved from reference session) is applied before the read-in, with statistics saved from the reference session and applied to the target session. For cross-patient normalization, per-channel statistics from the reference population could be used as a canonical normalization target.
