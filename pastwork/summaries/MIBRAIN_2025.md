# MIBRAIN: Towards Unified Neural Decoding with Brain Functional Network Modeling

**Citation:** Wu, Di et al. "Towards Unified Neural Decoding with Brain Functional Network Modeling." bioRxiv, 2025. https://doi.org/10.1101/2025.06.02.657011

---

## Setup

- **Recording modality:** Stereoelectroencephalography (sEEG), depth electrodes surgically implanted for epilepsy monitoring. Sampling at 2,000 Hz (participants 1-4, Neurofax EEG-1200) or 2,048 Hz (participants 5-11, Natus Quantum LTM). **Preprocessed to raw broadband signal at 512 Hz** (low-pass 200 Hz + notch 50/100 Hz, then downsampled). **Input is raw time-domain sEEG, NOT HGA or spectral features.** Per-participant electrode counts: 8-13 depth electrodes, 30-88 selected channels after quality filtering.
- **Channel selection:** PSD via Welch's method comparing mean power in **4-150 Hz** between articulation and resting conditions (paired t-tests, Benjamini-Hochberg FDR, corrected P < 0.05). Channels in visual cortex (occipital) explicitly excluded.
- **Patient count:** 11 participants (epilepsy patients, 5 female / 6 male, ages 14-35, all right-handed except one). Electrode placement determined solely by clinical requirements, yielding highly variable coverage across participants.
- **Brain regions covered:** 23 distinct regions (N^c=23) defined by **FreeSurfer anatomical parcellation** (NOT MNI coordinates). FreeSurfer reconstructs pial surface from T1 MRI and determines anatomical structure of each contact. MNI template used only for visualization (Fig. 1c). Regions span frontal (SFG, MFG, IFG, PreCen, PostCen, CG), temporal (STG, MTG, ITG, FuG), parietal (SPL, IPL, SMG, Angular), limbic (Hipp, Amyg, PHG, INS), and subcortical (Thalamus, Pcun, LG) areas. No single participant had coverage of all regions.
- **Task:** Mandarin Chinese phoneme articulation. Participants read 407 monosyllabic characters covering all possible initial + final phoneme combinations, either audibly or silently (mimed). 3-5 trials per character. Decoding target: **21 consonant initials + 2 simple finals ('i', 'u') = 23 classes** (compound vowels 'ao', 'ou' excluded due to dynamic articulation). **Only 5 of 11 participants performed silent articulation** (participants 7-11). Trial structure: 0.8-1.2 s preparation, 2 s articulation, 0.5-1.0 s rest. Audio labeled by 3 phonetics experts, aligned with PRAAT at 44.1 kHz.

---

## Problem Statement

Most neural decoding frameworks for implantable BCIs are highly individualized: they require extensive per-patient calibration and fail to generalize across subjects. This constraint arises because intracranial recordings exhibit significant heterogeneity in electrode placement and counts across individuals, making it difficult to pool data. Prior cross-subject methods (instance weighting, feature transformation, adversarial learning) assume homogeneous electrode configurations (e.g., standardized EEG grids) and therefore do not transfer to intracranial settings. More recent deep learning approaches that handle variable channel counts treat data dimensionality as the core problem but do not exploit the underlying functional brain network structure. The result is that current approaches require large amounts of data from each individual patient and cannot serve patients with limited recording capability or severe neurological impairment.

---

## Method

### Overall Architecture: Two-Stage Pipeline

**Stage 1 - Whole Functional Network Prototyping (self-supervised pre-training):**

1. A neurologist predefines a comprehensive set of N^c brain regions relevant to the task.
2. For each subject, per-region neural embedding encoders (two sequential 1D convolutional blocks, kernel sizes 7 and 5, strides 4 and 2, embedding dim d=32) map the variable-channel recordings of each covered region into fixed-dimensional region tokens. Temporal dimension is downsampled by factor 8 (T' = T/8).
3. For brain regions a subject lacks (due to no electrode implantation), learnable region prototype tokens P are substituted. These prototypes are shared across subjects and are the key cross-subject alignment mechanism.
4. The partial tokens (from real recordings) and prototype tokens (for missing regions) are concatenated and reordered to form a complete-region representation z_i^c in R^{N^c x T' x d}.
5. A temporal CNN encoder (4 **group-wise** 1D conv blocks — per-region independent processing, not cross-region — kernel size 3, stride 2, hidden dims 1d/2d/4d/4d, total temporal downsampling 16×, output token dim D=4d=128) further compresses. Combined with neuro-embedding (8×), **total temporal downsampling = 128×**. A 2s trial at 512 Hz (1024 samples) becomes ~8 temporal tokens.
6. **Self-supervised objective (masked autoencoding):** A random subset of region tokens is masked using a variable masking ratio r drawn from uniform distribution (bounds not specified in paper). The masked tokens are replaced by their corresponding prototype tokens. The model is trained to reconstruct the original **raw broadband sEEG** X_i from the masked-and-filled representation via **per-subject** lightweight reconstructors (discarded after pretraining). Loss is MSE. This forces prototype tokens to capture generalized neural features shared across subjects and regions. Trained with AdamW, lr=1e-5, beta1=0.9, beta2=0.999, weight decay=0.005, cosine annealing, 1000 epochs, batch size = **16 × N_subjects** (scales with cohort). Hardware: NVIDIA H800 GPUs.
- **Shared vs per-subject components:** Shared: temporal CNN encoder, region prototypes, region attention encoder. Per-subject: neuro-embedding encoders, MLP prediction heads, reconstructors (pretraining only).

**Stage 2 - Neural Decoding (supervised fine-tuning):**

1. The pre-trained neuro-embedding encoders, temporal CNN encoder, and prototype tokens are reused with their pre-trained weights.
2. A region attention encoder (E_RA, 2 MHSA blocks with FFN ratio 4) is appended. This encoder computes pairwise functional similarities between brain region tokens and dynamically groups functionally similar regions using Bipartite Soft Matching (BSM, adapted from ToMe). Merged region count N^m < N^c.
3. MLP prediction heads (one per subject in multi-subject mode) produce the final classification.
4. Trained with AdamW, lr=3e-4, beta1=0.9, beta2=0.999, weight decay=0.001, cosine annealing, 200 epochs. Evaluation: 7 samples per class × 23 classes = 161 test samples per subject.

### Cross-Patient Generalization Mechanism

For an unseen (m+1)-th subject, MIBRAIN uses a **channel similarity alignment + majority voting** strategy:
- For each brain region r shared between the new subject and existing subjects, compute pairwise cosine similarity between time-averaged channel responses to find the most similar channel mapping.
- Re-assemble the new subject's recordings to match the channel ordering of each existing subject's encoder.
- Run the new subject's data through each existing subject's neuro-embedding encoder to produce region tokens.
- Concatenate with prototype tokens for uncovered regions, pass through temporal encoder, region attention encoder, and prediction head.
- Final prediction is majority vote across all m existing subjects' encoder/head combinations.

---

## Key Results

### Offline Decoding Accuracy (23-class phoneme initial classification)

- MIBRAIN (multi-sub) significantly outperformed baselines (Brant, BrainBERT) and single-subject MIBRAIN for **all 11 participants** in audible articulation, all P < 0.001, except participant 4 (where both MIBRAIN variants failed to significantly outperform baselines).
- MIBRAIN (multi-sub) improvements over MIBRAIN (single-sub):
  - Audible: +8.08% (participant 10), +6.83% (participant 11)
  - Silent: +5.10% (participant 10), +4.97% (participant 11)
- Baselines (Brant, BrainBERT) barely exceeded chance level despite using large-scale pre-training, because they lack the multi-subject functional network integration mechanism.

### Real-Time Online Decoding (participants 10 and 11)

- Both MIBRAIN variants achieved above-chance accuracy **before any subject-specific calibration**.
- After 1-hour free-practice session, MIBRAIN (multi-sub) improvements over MIBRAIN (single-sub):
  - Participant 10 audible: +3.62%; silent: +7.97%
  - Participant 11 audible: +2.90%; silent: +4.35%

### Scaling with Number of Subjects

- Scaling experiment done **only for participants 10 and 11** during online (real-time) decoding, not all 11. Subjects added in **enrollment order** (not averaged over random orderings).
- Adding data from fewer than ~3 subjects initially **degrades** performance (inter-subject heterogeneity dominates).
- Consistent improvement begins with 4+ subjects; improvements become statistically significant (P < 0.05) once data from at least 6 additional subjects are included.
- Trend is monotonically upward beyond this threshold, suggesting further gains with more participants.
- For silent articulation (only 5 participants), the paper states scaling conclusions "remain preliminary."

### Cross-Subject Generalization (leave-one-subject-out)

- MIBRAIN trained on all subjects except the held-out subject achieved decoding accuracy **significantly above chance** on unseen subjects for both audible and silent tasks.
- Silent articulation generalized better than audible, hypothesized to reflect more stereotyped motor patterns without auditory feedback.

### Imputed Region Validation

- MIBRAIN's predicted representations for brain regions lacking direct electrode coverage correlated strongly with ground-truth representations from those same regions when electrodes were present (high diagonal in correlation matrix, Fig. 3a).
- Decoding using **only imputed representations** (no direct recordings) exceeded chance level significantly (P < 0.001) for most participants.
- Predicted Broca's area (IFG) and Wernicke's area (SMG, Angular) representations showed significant cross-correlation consistent with the arcuate fasciculus white matter tract, validating the anatomical plausibility of the imputed representations.

### Dynamic Brain Region Contributions

- SFG activates earliest (~500 ms before articulation onset), making it a better speech onset predictor than sensorimotor cortex.
- PreCen and PostCen show peak contributions around and after articulation onset.
- IFG (Broca's area) is active before and during onset (semantic/phonological processing).
- Post-onset: STG activates (auditory feedback).
- Notable contributions also from parahippocampal gyrus, cingulate gyrus, inferior temporal gyrus, hippocampus, amygdala, and thalamus.

---

## Cross-Patient Relevance

**What MIBRAIN demonstrates that is directly applicable to intra-op uECOG (~20 patients, ~20 min each):**

1. **Cross-patient pooling with heterogeneous coverage is tractable.** MIBRAIN explicitly handles the case where each patient has electrodes in different brain regions at different channel counts. The region-wise embedding approach (separate 1D conv encoder per brain region) maps variable-channel input to a fixed-dimensional token, which is exactly the situation with intra-op uECOG: grid placement varies by patient.

2. **Short-session patients are addressable.** MIBRAIN shows that a model pre-trained on other patients can generalize to an unseen subject above chance without any subject-specific fine-tuning. For patients with only ~20 min of recording, this zero-shot or few-shot regime is the relevant operating point.

3. **Prototype tokens as the cross-patient bridge.** The learnable region prototype tokens learn a shared neural representation space across patients. This is a concrete mechanism for pooling: instead of aligning raw electrode signals (infeasible with uECOG due to spatial resolution differences), you align at the level of brain-region representations.

4. **Scaling law confirmed at small N.** Performance improves monotonically with more subjects once N >= ~6. A cohort of 20 patients is above the threshold where MIBRAIN's multi-subject benefits clearly emerge.

5. **Silent articulation generalizes better cross-subject.** Intra-op tasks are typically constrained; if silent/mimed speech is used, MIBRAIN's result that silent articulation generalizes more consistently across subjects is encouraging.

6. **Majority voting for unseen subjects requires no retraining.** The inference-time channel alignment + majority vote mechanism allows a new patient's data to be decoded without any gradient update, which is operationally important for intra-op settings where training time is unavailable.

---

## Limitations

**What does not directly apply to intra-op uECOG:**

- **Recording modality mismatch.** MIBRAIN uses sEEG depth electrodes targeting specific deep and cortical structures across the whole brain. uECOG is a high-density surface grid covering a local cortical patch (typically sensorimotor or temporal cortex), not the distributed multi-region coverage sEEG provides. The whole-brain functional network modeling premise assumes spatially distributed and anatomically distinct region coverage per patient, which uECOG does not provide.

- **Recording duration.** sEEG patients are implanted for 1-2 weeks, producing hundreds of labeled trials per participant (407 characters x 3-5 trials each). Intra-op sessions are ~20 minutes, limiting labeled data volume substantially.

- **Task label availability.** MIBRAIN's supervised fine-tuning stage requires labeled phoneme trials. In intra-op settings, obtaining clean phoneme-aligned labels within the session is challenging.

- **Patient population.** MIBRAIN's cohort consists of epilepsy patients with intact language function. Intra-op patients may have tumor-adjacent cortex with altered neural representations or impaired speech.

- **Language.** MIBRAIN is validated specifically on Mandarin phoneme initials. Transfer to other languages or phoneme systems is unvalidated.

- **Participant-specific neuro-embedding encoders.** In the current architecture, each known subject has their own embedding encoder. For an unseen subject, MIBRAIN uses all existing encoders + majority vote, which scales linearly in compute with training cohort size. This may become unwieldy and has lower performance than the subject-specific path.

- **Initial performance degradation with small cohorts.** Adding 1-3 subjects actually hurts performance before the trend reverses. With a small intra-op cohort being built incrementally, early patients may see degraded cross-patient transfer.

- **No compound vowel decoding.** The current system decodes only consonant initials and two simple finals; compound vowels ('ao', 'ou') were explicitly excluded due to their dynamic articulation patterns. This limits the phoneme vocabulary.

- **No ablation of MAE pretraining vs training from scratch.** MIBRAIN never compares its two-stage approach against training the full architecture with supervised labels only (no MAE stage). The "single-sub" baseline uses the same architecture on one subject's data. Cannot determine how much MAE pretraining contributes vs. the architecture itself.

- **Input is raw broadband sEEG, not HGA.** The MAE reconstructs raw time-domain signals at 512 Hz. Our pipeline uses HGA envelopes at 200 Hz. The pretext task (reconstructing raw signal) does not directly transfer to our feature space.

- **Cross-task (audible→silent) decoding:** Audible-trained models decode silent speech better than the reverse — hypothesized because audible captures motor + auditory feedback (superset), while silent lacks auditory component.

- **Clinical potential for damaged speech cortex:** When vSMC is impaired (e.g., tumor patients), MIBRAIN could still decode via imputed representations from upstream regions (SFG, IFG). Relevant to our tumor patient cohort.

- **Mispronunciation impact:** Participants 3, 4, 8 showed lower performance due to regional dialectal pronunciation (e.g., 'n'/'l' confusion). Not a modality issue but relevant to any cross-patient phoneme decoding.

---

## Reusable Ideas

**Specific techniques worth borrowing for cross-patient uECOG speech decoding:**

1. **Brain-region-wise 1D convolutional embeddings.** Map each patient's variable-count electrode channels within an anatomical region to a fixed-dimensional token using per-region conv filters. This is the key mechanism for handling channel count heterogeneity without requiring electrode alignment. For uECOG: define region tokens by parcellating the grid into anatomical zones (e.g., using the MNI atlas) and apply separate lightweight conv encoders per zone.

2. **Learnable prototype tokens for missing regions.** For any brain region a patient lacks coverage for, substitute a learnable prototype vector initialized and updated during pre-training. This allows the model to operate on a standardized full-brain token sequence regardless of actual coverage. Directly applicable: for patients whose grid doesn't cover a particular cortical area, use a prototype token for that area.

3. **Masked autoencoding pretext task across patients.** Pre-train by randomly masking region tokens and reconstructing the original neural signals using information from other unmasked regions within the same patient and from the same region across other patients. This enforces cross-patient representational alignment without requiring a contrastive loss or explicit domain adaptation objective. The reconstruction loss is simple MSE on the original intracranial signal.

4. **Region attention encoder with token merging (Bipartite Soft Matching).** Instead of a fixed graph or hard-coded connectivity, use attention weights to dynamically group functionally co-active brain regions at each time step. The BSM operator (from ToMe) creates a sparse adjacency matrix and merges correlated region tokens, reducing dimensionality while preserving functional structure. This is interpretable: the grouping pattern over time reveals which regions co-activate during which phase of speech production.

5. **Channel similarity alignment for unseen subjects.** At inference time for a new patient, compute cosine similarity between time-averaged channel responses and find the best channel-to-channel mapping to reuse an existing patient's encoder. This requires zero gradient updates and takes only a forward pass. For uECOG: compute the channel similarity matrix between the new patient's grid and each training patient's grid, rearrange channels to best match, then re-use the existing encoder.

6. **Majority voting across training subjects' encoders.** When decoding an unseen patient, run their data through every training patient's encoder + prediction head combination and take majority vote. Simple and robust; no new parameters needed. Performance will be below the subject-specific model but is a strong zero-shot baseline.

7. **Two-stage training: self-supervised prototyping then supervised decoding.** Separating the cross-patient alignment objective (masked autoencoding) from the task-specific decoding (supervised classification) allows the shared representation space to be learned without task labels, which is valuable when labeled data is scarce. For intra-op: pre-train the prototype tokens on unlabeled or lightly labeled multi-patient data, then fine-tune the decoding head on the limited labeled intra-op trials.

8. **Gradient-based region contribution scores (Grad-CAM variant).** Compute the contribution score of each brain region at each timestep as the ReLU of the gradient of the predicted label with respect to the region token. This provides a time-resolved importance map across regions that reveals the temporal sequence of speech-related cortical activations (SFG earliest, then frontal motor, then sensorimotor, then STG). This is an analysis tool, not a training objective, but provides interpretability for which grid zones are most informative at each time point within a trial.
