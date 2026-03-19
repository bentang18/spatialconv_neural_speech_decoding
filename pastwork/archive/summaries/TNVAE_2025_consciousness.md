# TN-VAE: Multi-modal, Multi-species, Multi-task Latent-space Model for Decoding Level of Consciousness

**Source:** Wang et al., NeurIPS 2025 (bioRxiv preprint doi: 10.64898/2025.12.03.692227, posted December 9, 2025)
**Affiliation:** Axoft, Inc. and Harvard University (Jia Liu)

---

## Setup

- **Recording modality:** Multi-modal: local field potentials (LFPs, specifically LFP wavelets at 50 log-spaced scales from 0.5–150 Hz), single-unit spike-sorted firing rates (per 200 µm depth bin along probe), and an estimated movement proxy (EMG estimated from high-frequency cross-channel correlation). Human data is LFP-only (unimodal).
- **Species:** Rat (primary training), pig (secondary fine-tuning), and human (intraoperative fine-tuning and zero-shot transfer target).
- **Patient count:**
  - Rat: 1 subject, 11 continuous recording sessions over 6 weeks (~13.5 hours total, 10 sessions train / 1 test)
  - Pig: 1 subject, 2 sessions across multiple weeks
  - Human: 5 subjects undergoing planned tumor resection surgeries (2 unconscious throughout, 1 fully conscious throughout, 2 transitioning from unconscious to conscious during recording); human recordings total ~100 min across subjects
- **Task:** Passive depth-of-anesthesia monitoring — no behavioral task required from the subject. For humans, an auditory local-global oddball paradigm was also presented to test multi-task generalization (consciousness-dependent auditory evoked responses).
- **Hardware:** Fleuron laminar neural probes (soft, high-density), 20 kHz sampling. Rat: somatosensory cortex + hippocampus. Pig: rostrum gyrus. Human: intraoperative cortical recordings during tumor resection.
- **Window size:** 2-second non-overlapping windows; state labeled at 30-second resolution from behavioral scoring.

---

## Problem Statement

Current standard of care for Disorders of Consciousness (DoC) — coma, unresponsive wakefulness syndrome, minimally conscious states — relies on bedside behavioral assessments (CRS-R, GCS) with misdiagnosis rates exceeding 40%. Existing neural decoding approaches:

1. Are supervised and require ground-truth behavioral or anesthetic labels that are imprecise or subjective.
2. Are unimodal (typically EEG-derived features), missing richer information from spiking activity and multiple signal types.
3. Generalize poorly across subjects, requiring per-patient calibration — a critical blocker for clinical BCI deployment where zero-shot transfer is required.
4. Operate at coarser temporal granularity than is physiologically achievable.

**Gap addressed:** A self-supervised, multi-modal foundation model that (a) constructs an interpretable continuous latent space of brain state without requiring behavioral labels at training time, (b) handles heterogeneous modality availability across species and subjects via lightweight adapter layers, and (c) achieves zero-shot transfer to new human intraoperative subjects without per-patient recalibration.

---

## Method

### Architecture: TN-VAE (Time-aware autoregressive Variational Autoencoder)

- Adapted from Wang et al. (2023), originally proposed for robust time-series representation learning.
- **Bottleneck:** Produces a 2-D latent space (interpretable by design — not reduced post-hoc).
- **Autoregressive objective:** Rather than reconstructing the current input, the model predicts the representation of the *next* time point. This promotes temporal smoothness in the latent trajectory.
- **Self-supervised training:** No anesthetic concentration or behavioral labels are used as training targets. The model is trained purely on the neural signal structure.
- **Input:** Per-channel multimodal feature vector at each 2-second window: LFP wavelet spectrogram + depth-binned spike firing rate + movement proxy.
- **Latent space is per-channel** (not a population-level representation), enabling transfer to subjects with different channel counts.
- **Training hyperparameters:** Adam optimizer, lr=1e-4, batch size=10,000, KL weight=1e-4, 200 epochs. 80/20 train/val split per initialization; final model selected by neighbor loss.
- **Baseline comparison:** Linear PCA — TN-VAE's nonlinear deep embedding outperforms PCA in separating the four anesthesia states.

### Cross-species / Cross-subject Transfer Mechanism: Stitching Layers

- Each species or recording modality subset has its own lightweight **species-specific stitching layer** that projects the species-specific input feature space into the shared base model's input space.
- The base model (TN-VAE trunk) is pretrained on trimodal rat data.
- **Transfer to humans (fine-tuning protocol):**
  1. A human-specific stitching layer is trained using only ~18 minutes of LFP data from a single human subject (Subject A) that contains both unconscious and awake states.
  2. The base model trunk is held fixed (or lightly updated); only the stitching layer adapts.
  3. The resulting model is then applied **zero-shot** to two entirely unseen human subjects (Subjects B and C) — no additional training on those subjects.
- **Transfer to pig:** Fine-tuned from rat-pretrained base using bimodal (LFP + movement) pig data from one session, tested on a second session.
- This architecture explicitly handles **incomplete modality sets**: a subject lacking spike data can still use the model via a stitching layer trained on available modalities only.

### Key Architectural Choices

| Choice | Rationale |
|---|---|
| 2-D latent space | Interpretability; axes align with known physiology |
| Autoregressive prediction (not reconstruction) | Encourages smooth temporal trajectories, better state separation |
| Self-supervised (no labels) | Avoids dependence on imprecise behavioral/anesthetic labels |
| Per-channel latent (not population) | Enables variable channel count across subjects |
| Stitching layers (not full fine-tuning) | Low-data adaptation; preserves pretrained representations |

---

## Key Results

### State Separation (Rat, held-out test session)
- The 2-D latent space **nonlinearly but smoothly separates all 4 anesthesia states** (awake, light, moderate, deep) at 2-second resolution.
- Latent trajectories return to the initial awake-state position after recovery, showing reversibility.
- **Linear decodability** of the 4 broad states from the latent space exceeds that of a PCA baseline.
- Broad state accuracy from linear SVM: approximately **0.65–0.70** (read from Fig. 4b bar chart; exact numbers not tabulated in text).
- Individual behavioral sub-components are also linearly decodable: exploration of area, spontaneous movement, toe pinch response, balancing — each with varying accuracy above chance.

### Latent Space Interpretability
- **Delta/alpha power** (low-frequency LFP) separates unconscious sub-states (moderate vs. deep).
- **Gamma power** and **single-unit firing rates** distinguish wake from unconscious states.
- Anesthesia state separation is consistent **across cortical depths** (S1 superficial/deep, CA1, CA3/DG hippocampus), with some depth-dependent variation.
- The latent axes are not arbitrarily rotated — they align with known physiological markers without supervision.

### Zero-shot Human Transfer
- Model pretrained on rat (trimodal) + fine-tuned on ~18 min of human LFP from Subject A successfully **separates awake vs. anesthetized states in two completely unseen human subjects (B and C)** with no per-subject training.
- This is a true zero-shot transfer: the stitching layer trained on Subject A generalizes directly.
- Labels for human subjects are based on anesthesiologist assessment (not exact to 2-second resolution), so the reported separation is qualitative/visual (Fig. 3a) rather than a precise accuracy number.

### Multi-task Generalization (Human, Auditory Stimuli)
- The same latent space captures **auditory local vs. global pattern deviation** responses (local-global oddball paradigm).
- Local response metric is consistent across subjects regardless of consciousness state.
- Global response metric (indicative of higher-order consciousness) separates conscious from unconscious subjects — captured without any stimulus-related supervision.

### Pig Transfer
- Fine-tuned on bimodal pig data (one session), tested on a second session: the latent space separates light vs. deep anesthesia states.

---

## Cross-Patient Relevance

**Context:** Intraoperative uECOG speech decoding from ~20 patients, ~20 minutes of recording each, highly variable electrode coverage and channel count.

### What Transfers Directly
- **The stitching layer paradigm is the most directly relevant idea.** With ~20 patients and ~20 minutes each, you cannot train a patient-specific model from scratch for every subject. The TN-VAE approach shows that a shared trunk pretrained on richer/longer data (here: rat) + a lightweight per-subject adapter trained on minimal data can achieve zero-shot transfer. The direct analog is: pretrain a speech decoding trunk on existing patients, then fit a per-patient stitching layer on a few minutes of held-in data (or a standardized calibration block), and deploy zero-shot on new surgical subjects.
- **Variable channel count / incomplete modality handling.** intra-op uECOG arrays vary substantially in electrode count and placement across patients. The per-channel (rather than population-level) latent representation, combined with stitching layers that absorb modality/channel variability, is directly applicable. A per-electrode feature encoder with shared trunk avoids the need to fix array geometry across patients.
- **Self-supervised pretraining on unlabeled neural data.** The TN-VAE is trained without behavioral labels. For intra-op speech, labeled speech tokens may be sparse or noisy; a self-supervised backbone pretrained on the LFP/ECoG structure itself could provide a better initialization than random weights.
- **Temporal smoothness as a training objective.** The autoregressive next-step prediction objective enforces smooth latent trajectories. Speech is also temporally smooth at the phoneme/syllable level; this inductive bias could help learn representations that track articulatory state rather than frame-level noise.
- **z-scoring per channel per session** to remove batch/session effects is a simple but important preprocessing step explicitly used here and directly applicable to intra-op ECoG where recording conditions vary session to session.

### Partial Relevance (Requires Adaptation)
- **Cross-species pretraining.** TN-VAE uses rat data as a proxy for human. For speech, there is no animal analog — but the principle of pretraining on *other patients* or *other tasks* (e.g., imagined movement, passive listening) applies. The stitching layer framework supports this.
- **2-D latent space.** Interpretable but likely too low-dimensional for speech (which has a much higher-dimensional manifold than awake/anesthetized). However, the broader principle — using a bottleneck VAE that encourages structured, smooth latent representations — is valuable.
- **LFP-only human input.** This paper's human model uses only LFP (no spikes, no movement) and still achieves transfer. For uECOG, high-gamma (70–150 Hz) power is the dominant speech signal; LFP wavelets in this range are the equivalent input. The preprocessing pipeline (complex Morlet wavelets, 0.5–150 Hz, 50 log-spaced scales) is directly applicable.

---

## Limitations

### Limitations Stated by the Authors
- **Human data is very limited:** Total human recording is ~100 minutes across 5 subjects, with class separation that "while successful, could be improved."
- **Single rat subject:** The rat backbone is from one animal, limiting diversity of the pretrained representation.
- **Sensitivity to artifacts and distributional shifts** across channels, sessions, and subjects — partially mitigated by z-scoring but acknowledged as an open problem.
- **Anesthesiologist labels are imprecise:** Human consciousness state labels are not time-locked to seconds; transitions (especially regaining consciousness) are not captured precisely.
- **Per-channel latent (not multi-channel population):** The model does not model inter-channel relationships, losing information about network-level dynamics. Multi-channel representations are listed as future work.

### Limitations for Speech Decoding Transfer (Not Stated in Paper)
- **Consciousness state decoding is a coarse 2-4 class problem.** Speech decoding requires distinguishing dozens to hundreds of phonemes or words — far higher output dimensionality. The 2-D latent space and the binary awake/anesthetized classification frame are not directly applicable to fine-grained speech decoding.
- **No temporal structure in the task.** Anesthesia state is quasi-stationary over seconds to minutes. Speech is structured at 50–200 ms timescales. The 2-second window is appropriate for consciousness monitoring but too coarse for phoneme-level speech decoding (typically 10–50 ms windows).
- **The "unseen subject" claim is limited.** Zero-shot here means no per-subject fine-tuning after the stitching layer is trained on Subject A. It is not truly zero-shot in the general sense — a stitching layer was still trained on one human subject. For speech, this is still a strong result (one patient as "calibration"), but it is not zero-data transfer.
- **No evaluation of decoding accuracy with hard metrics.** The paper uses qualitative latent space visualization and linear decodability; there are no word error rates or phoneme accuracy numbers comparable to speech BCI benchmarks.
- **Propofol anesthesia does not generalize to the speech-production brain state.** Intra-op speech mapping involves an awake patient actively producing speech — the neural dynamics are fundamentally different from the awake-vs.-anesthetized distinction studied here.

---

## Reusable Ideas

Ranked by relevance to intra-op uECOG speech decoding (~20 patients, ~20 min each):

### 1. Stitching Layers for Cross-patient Transfer (Highest Priority)
Train a shared speech decoding trunk on all available patients. For each new intraoperative patient, fit a lightweight per-patient stitching layer on a short calibration block (e.g., 2–5 minutes of repeated words/phonemes at the start of surgery). Deploy the trunk + new stitching layer for the remainder of the session without further training. This directly mirrors the zero-shot human transfer protocol and is the clearest methodological contribution to borrow.

**Implementation sketch:**
- Stitching layer: small MLP (1-2 layers) or linear projection from patient-specific electrode feature space to shared trunk input space.
- Trunk: pretrained VAE or transformer encoder on existing patient cohort.
- Fine-tuning data: calibration block only; trunk frozen or with very small learning rate.

### 2. Per-channel (Per-electrode) Feature Encoding with Shared Trunk
Rather than requiring a fixed-size electrode array, encode each electrode independently through a shared feature extractor (LFP wavelets → embedding), then aggregate. This handles variable array sizes across patients and allows the model to be applied to any subset of channels. The stitching layer then maps per-electrode embeddings into a patient-normalized space.

### 3. Self-supervised Autoregressive Pretraining Objective
Use an autoregressive next-step prediction objective (predict the representation of the next time window, not reconstruct the current one) during pretraining on unlabeled intra-op ECoG. This enforces temporal smoothness and can be applied to all available data including non-speech periods, maximizing use of the limited ~20-minute recording window.

### 4. LFP Wavelet Preprocessing Pipeline
Complex Morlet wavelets at 50 log-spaced scales from 0.5 to 150 Hz, downsampled to 500 Hz, with notch filters at 60/70/120/156 Hz for electrical artifact removal. This is a well-validated multi-scale spectral representation that captures both low-frequency oscillatory dynamics and high-gamma speech signals in a single unified feature vector.

### 5. Z-scoring Per Channel Per Session for Batch Effect Removal
A simple but critical step: z-score all features per channel per recording session before training and inference. This removes DC offsets, impedance differences, and session-level distributional shifts that would otherwise confound cross-patient generalization. The paper explicitly identifies sensitivity to distributional effects as a limitation and uses z-scoring as the primary mitigation.

### 6. Interpretable 2-D Latent Space for Brain State Monitoring
Even if the speech decoder operates in a higher-dimensional space, a separate 2-D latent "brain state monitor" trained alongside it could provide real-time visualization of the patient's neural state during surgery — useful for detecting state changes (drowsiness, distress, movement artifacts) that would invalidate speech decoding. The TN-VAE latent space could serve this secondary monitoring role.

### 7. Multi-task Training to Improve Generalization
Train the shared trunk simultaneously on multiple tasks across the patient cohort (e.g., speech production, motor imagery, auditory responses) to prevent the trunk from overfitting to any single task's idiosyncrasies. TN-VAE demonstrates that a single latent space can capture both consciousness state and auditory evoked response without task-specific heads beyond a linear readout.
