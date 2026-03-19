# Levin et al. 2026 - Cross-Brain Transfer of Speech and Handwriting BCIs

## Citation

Levin, A. D., Avansino, D. T., Kamdar, F. B., Card, N. S., Wairagkar, M., Jacques, B. G., ... & Willett, F. R. (2026). Cross-brain transfer of high-performance intracortical speech and handwriting BCIs. *bioRxiv* preprint. https://doi.org/10.64898/2026.01.12.699110

## Setup
- **Recording modality:** Intracortical microelectrode arrays (NeuroPort, Blackrock Microsystems) -- 96-channel and 64-channel Utah arrays. Features = threshold crossing rates + spike band power, binned at 20 ms.
- **Patient count:** 5 BrainGate2 participants (T5, T12, T15, T16, T18). Speech: T12, T15, T16. Handwriting: T5 (source), T12 and T18 (targets).
- **Brain region:** Precentral gyrus -- speech arrays in ventral precentral gyrus (area 6v, orofacial motor cortex), area 55b, area 44 (inferior frontal gyrus / Broca's area), and area 4. Handwriting arrays in hand knob (area 6d).
- **Task:** Instructed delay paradigm -- attempted speech (vocalized or mouthed) and attempted handwriting of prompted sentences. Open-ended general English vocabulary (Switchboard corpus, OpenWebText2, Harvard Sentences for speech; Brown Corpus for handwriting).
- **Data amount:** 48.9 hours total across 5 participants, spanning many sessions over months to years. T12: ~5,700 speech sentences (16 sessions); T15: ~3,300 (16 sessions); T16: ~2,800 (15 sessions). This is 30-120× more data per patient than our setting (46-178 trials).

## Problem Statement

High-performance intracortical BCIs for speech and handwriting require substantial supervised training data (up to 10 days) to reach peak performance. This is a major clinical bottleneck. Can pre-training a decoder on data from other users ("cross-brain transfer") reduce the amount of target-user data needed, especially when the new user has limited data (< 200 sentences)? This is the first attempt at cross-brain transfer for intracortical recordings during complex motor behaviors (speech/handwriting) with a general, open-ended vocabulary.

## Method

### Neural preprocessing
1. Raw signals analog filtered (4th-order Butterworth, 0.3-7.5 kHz), digitized at 30 kHz.
2. Digital highpass (250 Hz) or bandpass (250 Hz-5 kHz) filtering applied non-causally per electrode.
3. Linear regression referencing (LRR) applied per 64-electrode array group to reduce noise artifacts.
4. Two features computed per electrode per time bin: **threshold crossing rates** (spike count above -4.5 or -3.5 SD threshold) and **spike band power** (sum of squared voltages).
5. Features binned into 20 ms time steps, z-scored (mean-subtracted, divided by SD), causally smoothed with a Gaussian kernel (sd=40 ms, delayed 160 ms).
6. Concatenated into a single feature vector: 512 x 1 for speech (= 2 features × 256 electrodes), 768 x 1 for handwriting (= 2 features × 384 electrodes).
7. **Zero-padding** used to match feature dimensions across participants (padded to max neural feature count: 512 for speech, 768 for handwriting). **Note:** 512 is feature count, not electrode count.

### Alignment/transfer approach (dataset-specific input layers)

The key architectural insight is the use of **unique, trainable, session-specific input layers** -- one per data collection session, per participant. Each input layer is an affine transformation followed by a softsign nonlinearity:

```
x_tilde_{i,t} = f(W_i * x_{i,t} + b_i)
```

where `W_i` is a p x p weight matrix, `b_i` is a bias vector, and `f(x) = x / (|x| + 1)` (softsign). Dropout is applied both before and after the softsign. **Per-speech-session: p=512, so W is 512×512 + 512 bias = ~262,656 parameters per session.** T12 alone has 16 speech sessions, each with its own input layer. This is architecturally very different from our planned 13.4K-param Linear(208,64).

This is **not** an explicit alignment method like CCA. Instead, each session gets its own learned linear+nonlinear transform that maps its raw neural features into a shared representation space that the downstream RNN can operate on. The input layer weights are optimized jointly with all shared RNN parameters via backpropagation and CTC loss.

**Critical finding:** Without these session-specific input layers (i.e., using a single shared input layer for all sessions/users), cross-brain transfer **failed** -- performance was often *worse* than training on target user data alone. The session-specific layers are what make the system work; they absorb differences in electrode placement, neural tuning, and day-to-day nonstationarities.

### Decoder architecture
- 5-layer stacked **gated recurrent unit (GRU) RNN** (same architecture as Willett et al. 2023).
- Outputs a phoneme probability vector (speech) or character probability vector (handwriting) every 80 ms (4-bin frequency with 20 ms bins).
- Trained with **connectionist temporal classification (CTC) loss** -- no ground truth timing labels needed (critical because participants cannot produce intelligible speech or physically write).
- Regularization: white noise added to input features at each time step; artificial constant offsets added to feature means to handle baseline firing rate drift.

### Training procedure
1. **Base decoder training:** Train on all source user data, sampling uniformly from each source day.
2. **Fine-tuning:** Fine-tune on target user's held-out session data. During fine-tuning, 30% of training samples come from source participants' sessions (replay buffer) and 70% from target user days. This source-data replay during fine-tuning is important.
3. Train/test split: 70% train / 30% test per session, shuffled across blocks.
4. Evaluation: Greedy decoding (most probable phoneme/character at each step, duplicates removed). Metric = edit distance (phoneme error rate for speech, character error rate for handwriting). Error rates: sum errors / total phonemes (not per-sentence average).
5. Progressively larger training set variants (10% to 100% in 10% increments) used to assess effect of target user data amount.
6. **Not in PDF:** Optimizer, learning rate, epochs, GRU hidden dimensions, and other hyperparameters are in an external spreadsheet (Table II), not in the paper text.
7. Source replay sampling: **uniform across source session days** (not weighted by session size), with 70% from target user days.
8. Electrode permutation control: a **single** random channel permutation per participant, applied to ALL of that participant's non-held-out sessions.

## Key Results

### Transfer vs. from-scratch
- Cross-brain transfer **improved decoding performance when target user training data was limited (< 200 sentences)** for all speech and handwriting target users except T18.
- T18 exception: decoder trained from scratch saturated at ~20 sentences with 2% character error rate, likely due to unusually high signal quality. Transfer was unnecessary.
- Improvement was consistent across T12, T15, T16 (speech) and T12 (handwriting).

### Session-specific input layers are essential
- With session-specific input layers: transfer works, clear improvement in low-data regime.
- With a single shared input layer: transfer **hurts** -- performance often worse than target-user-only training. This is the paper's most architecturally important finding.

### Effect of target user data amount
- Transfer benefit diminishes as target user data increases. At large data amounts (hundreds to thousands of sentences), from-scratch and transfer converge.
- The sweet spot is roughly < 200 sentences from the target user.

### Cross-brain transfer vs. electrode-permuted data (same user)
- Electrode permutation = random reordering of channels from the same user's other sessions, simulating "another brain with identical neural latent structure."
- For T16 speech and T12 handwriting: cross-brain transfer was **as effective as** permuted same-user data. This suggests the transfer is successfully leveraging shared structure.
- For T12 speech and T15 speech: electrode-permuted data from the same user was **more beneficial** than cross-brain transfer. This implies that for these users, the neural latent structure differs substantially across individuals -- the representations are not simply orthonormal rotations of each other.

### Quantitative examples (approximate, read from figures)
- T12 speech at ~84 sentences: transfer PER ~0.30, from-scratch PER ~0.40.
- T15 speech at ~80 sentences: transfer PER ~0.20, from-scratch PER ~0.30.
- T16 speech at ~50 sentences: transfer PER ~0.35, from-scratch PER ~0.45.
- T12 handwriting at ~28 sentences: transfer CER ~0.25, from-scratch CER ~0.35.

## Cross-Patient Relevance

### Comparison to CCA-based alignment (Spalding et al. 2025)
- **Spalding et al.** use CCA to explicitly align neural latent spaces across patients, requiring matched stimuli (same 52-word vocabulary spoken by all patients) to find shared low-dimensional representations. This is a geometric/statistical alignment in latent space.
- **Levin et al.** use learned session-specific input layers that implicitly map each session's neural activity into a shared space via backpropagation. No explicit alignment step, no requirement for matched stimuli or shared vocabulary. The alignment is learned end-to-end.
- Levin et al. explicitly note their approach "do[es] not require matching conditions for CCA-based alignment" and use phonemes as building blocks (39 phonemes) rather than decoding a fixed word set.
- **Key tension:** Levin et al. find that for some users (T12, T15 speech), cross-brain data is less useful than permuted same-user data, suggesting "underlying differences in neural encoding and dynamics across users may limit the benefit and applicability of linear re-alignment." This is a direct challenge to CCA-based approaches, which assume a shared latent structure exists.

### Intracortical vs. uECOG
- Intracortical arrays record single/multi-unit spiking activity from a small patch of cortex (~4x4 mm per array). Much higher spatial resolution but more variable across patients (electrode penetration depth, local neuron populations, exact placement).
- uECOG records local field potentials from the cortical surface over a broader area. Lower spatial resolution but potentially more consistent neural representations across patients (population-level signals, less sensitive to exact neuron sampling).
- The session-specific input layer approach may be less necessary for uECOG if surface field potentials are more stereotyped across patients, but it could still help with electrode layout differences.

### Applicability to intra-operative setting with limited data
- **Highly relevant.** The core finding -- transfer helps most when you have < 200 sentences -- directly addresses the intra-op constraint where you might get minutes, not hours, of data.
- However, Levin et al. still require *some* target user data for fine-tuning. Zero-shot transfer is not demonstrated (and would require the shared input layer, which failed).
- The session-specific input layer approach requires training a new input layer per session, which needs at least some labeled data from the target. For intra-op uECOG, you would need to decide whether you have enough data to fit this layer.
- The CCA approach (Spalding et al.) may be more practical for very limited data because CCA alignment can be computed on a small matched stimulus set without backpropagation through the full model.

## Limitations

1. **Only 5 participants** -- small N limits generalizability claims. They acknowledge a larger user library may be needed for consistent benefits.
2. **No zero-shot transfer** -- the shared input layer (which would enable zero-shot) failed badly. You always need some target user data.
3. **Electrode placement varies substantially** across participants (different cortical areas, different array counts/types). Zero-padding to match dimensions is crude.
4. **Transfer improvements are modest** -- the authors themselves describe gains as "modest." For some users, same-user permuted data outperforms cross-brain data, suggesting true neural differences across individuals.
5. **Offline evaluation only** -- all results are offline decoding performance, not real-time closed-loop BCI use.
6. **Heterogeneous participant population** -- ALS, spinal cord injury, pontine stroke with different motor capabilities (vocalized vs. mouthed speech, real vs. imagined handwriting). These differences in task execution strategy could confound transfer.
7. **No language model** -- evaluated with greedy CTC decoding only (no beam search with LM rescoring), which makes absolute error rates higher than would be seen in a deployed system.
8. **Preprint** -- not yet peer reviewed (posted January 14, 2026).

## Reusable Ideas

1. **Session-specific (or patient-specific) input layers** as a general strategy for combining neural data across sessions/patients. A simple affine transform + nonlinearity per dataset, jointly optimized with shared downstream layers. This is applicable to any neural decoding pipeline and avoids the need for explicit alignment.
2. **Electrode permutation as a control** for distinguishing "more data helps" from "cross-brain structure helps." If permuted same-user data is equally good, the benefit is from data augmentation, not shared neural representations.
3. **Source data replay buffer during fine-tuning** -- keeping 30% of training batches from source users during target fine-tuning to prevent catastrophic forgetting.
4. **Zero-padding** as a simple (if crude) way to handle different electrode counts across participants.
5. **CTC loss for speech/handwriting BCI** when ground truth timing is unavailable -- relevant for any setting where you have the transcript but not the alignment.
6. **The <200 sentence threshold** as a practical guideline: cross-brain transfer is most valuable below this amount of target data. This helps set expectations for how much intra-op data you would need before transfer stops being useful.
7. **Caution for CCA-based methods:** the finding that some users' latent structures are not orthonormal transforms of each other suggests linear alignment methods may have a ceiling. Nonlinear learned alignment (like the input layers here) or much larger user libraries may be needed.
