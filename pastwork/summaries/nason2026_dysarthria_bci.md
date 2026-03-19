# Nason-Tomaszewski, Deevi et al. 2026 - Restoring Brain-to-Text Communication in Pontine Stroke Using an Intracortical BCI

**Note:** The user's filename references "Levin et al. 2026 - Cross-Brain Transfer," but the actual PDF is Nason-Tomaszewski, Deevi et al. 2026 about restoring speech in a single pontine stroke participant using an intracortical BCI. This paper is **not** about cross-brain/cross-patient transfer. The summary below faithfully reflects the paper's content.

## Citation
Nason-Tomaszewski SR*, Deevi PI*, Rabbani Q, Jacques BG, Pritchard AL, Wimalasena LN, Richards BA, Karpowicz BM, Bechefsky PH, Card NS, Deo DR, Choi EY, Hochberg LR, Stavisky SD, Brandman DM, AuYong N, Pandarinath C. Restoring brain-to-text communication in a person with dysarthria from pontine stroke using an intracortical brain-computer interface. medRxiv preprint, 2026. doi: 10.64898/2026.02.19.26346583

## Setup
- **Recording modality:** Intracortical microelectrode arrays (NeuroPort, Blackrock Neurotech) -- 64-channel silicon Utah arrays, 1.5 mm electrode length
- **Patient count:** 1 (participant T16 in BrainGate2 clinical trial)
- **Brain region:** Left precentral gyrus, putative Brodmann's area 6v (ventral premotor / orofacial motor cortex). Four arrays implanted total (6v, border of PEF/55b, two in hand knob 6d), but only the single 6v array (64 channels) was used for speech decoding.
- **Task:** Mimed (mouthed, no vocalization) speech -- copy task (repeat Switchboard corpus sentences) and Q&A conversational task
- **Recording duration/amount of data:** Data collected between post-implant days 32-736 (approximately 2 years). 38 sessions with diagnostic task spanning 420 days. Closed-loop evaluation sessions spanned ~3 months to 2 years post-implant.

## Problem Statement
Whether intracortical BCIs (iBCIs), which have shown 0.8-24% WER for ALS patients, can also restore communication for people with dysarthria secondary to pontine stroke. Prior ECoG work in pontine stroke achieved only 25.5% WER with a 1,024-word vocabulary. The key question: can a small patch of cortex (single 64-channel array) in orofacial motor cortex yield sufficient neural information for accurate speech decoding despite cortical thinning and reduced functional connectivity associated with pontine stroke?

## Method

### Neural preprocessing
1. Raw voltage signals analog band-pass filtered (0.3 Hz to 7.5 kHz), digitized at 30 kHz
2. Digital high-pass filter at 250 Hz (acausal 4th order Butterworth)
3. Digital band-pass filter 250 Hz to 5 kHz (acausal 4th order Butterworth) for later sessions
4. Linear regression referencing (LRR) -- coefficients computed on a reference block, then applied prior to digital filtering
5. Two features extracted per channel in 20 ms bins:
   - **Threshold crossings (Spk):** threshold set at -4.5x RMS of each signal; count crossings per bin
   - **Spike-band power (SBP):** square filtered voltage, average in same 20 ms bins
6. Combined into a 512-dimensional feature vector (64 channels x 2 features x 4 -- though the text says 64 channels from 6v array with both features = 128 active features, zeroing non-6v channels)
7. Z-score normalization: offline = within-block mean/std; online = rolling stats from preceding 10-20 trials
8. Smoothing with 40 ms Gaussian kernel

### Alignment/transfer approach
**This paper does NOT perform cross-patient/cross-brain alignment.** It is a single-participant longitudinal study. The "transfer" is across sessions within the same patient over time:

- **Cross-session adaptation via finetuning:** A pretrained RNN is finetuned on new session data using a day-specific dense input layer with softsign activations to absorb neural drift
- When starting a new session, the previous session's online-finetuned model is loaded; the input layer parameters are copied to a new input layer for the current session
- Finetuning batches: 60% current session sentences + 40% prior session sentences
- **Two distinct paradigms:**
  - **Sessions 1–15:** Offline pretrain on all prior data, THEN online finetune ~50 sentences in-session (~30 min)
  - **After session 15:** Load previous session's online-finetuned model, copy input layer, online finetune only ~20 sentences until plateau (no offline pretrain)
- **Bug disclosure:** Sessions 39–68 input layers were initialized from session 38's params (stale) due to code bug, potentially affecting ~10 evaluation sessions
- Replay buffer includes **3–20 previous sessions** (not just the most recent)
- Validation PER computed every 50 batches; checkpoints saved at best-so-far (not last epoch)
- LDA analysis showed cross-session classification accuracy drops from ~92.5% (within-session) to ~75.5% immediately, then degrades further after ~40 days to ~39.1% at 200+ days (still above chance of 14.6% from feature scramble)
- **PER degradation onset:** On average, 25% relative increase in PER by 6.9 ± 1.8 days after training

### Decoder architecture
- **Phoneme decoder:** 5-layer GRU-based RNN (512 tanh units per layer) with day-specific dense input layers (softsign activation). Input is **512-dimensional** (4 arrays × 64 channels × 2 features, non-6v channels zeroed → 128 active of 512). Day-specific layer dimension is **unstated** in paper but likely 512→512 softsign (~262k params/day) following Willett 2023's same-dim design. **55b array explicitly excluded** due to "little phonemic tuning."
- Outputs: 41 classes (39 American English phonemes + silence + CTC blank)
- Phoneme logits output every 80 ms using most recent 280 ms (14 bins) of neural features
- **Language models:** Series of 5-gram LMs (50-word, 1,024-word, 125,000-word vocabularies) + a 6.7B-parameter open pretrained transformer LM for rescoring
- Final prediction: pruned 125k LM generates top 100 candidates, unpruned LM rescores them, transformer LM rescores separately, scores are weighted and combined

### Training procedure
1. **Pretraining:** RNN trained offline on all sentences from previous sessions (up to 10,000 batches of 64 sentences, early stopping on validation PER). ADAM optimizer, LR=0.01 with 0.9 decay.
2. **Online finetuning:** Load pretrained RNN + copy input layer weights to new day-specific input layer. Finetune with current session sentences (60%) + prior session sentences (40%). Static LR=0.004. Finetuning epochs: 32-200 batches of 64 sentences, stopping when avg loss drops below 0.5x of previous 10-batch average.
3. **Evaluation:** Model weights frozen, tested on held-out evaluation sentences.
4. Data augmentation during training: white noise and constant offsets added to neural features. Dropout and L2 regularization applied.

## Key Results

### Copy task (Switchboard corpus sentences)
- **125,000-word vocabulary:** Median **19.4%** WER (Results section; abstract says 19.6%), median 21.8% PER, at median 35.0 WPM
- **1,024-word vocabulary:** Median 10.0% WER offline (60.8% reduction vs. prior ECoG work's 25.5% WER)
- **50-word vocabulary:** Median 11.9% WER (53.5% reduction vs. prior ECoG work)
- PER: median 14.3% (50-word), 16.1% (1,024-word), 21.8% (125k-word)

### Q&A conversational task
- 102 sentences in response to 29 questions across 8 sessions
- 35.2% WER, 33.7% PER, 27.7 WPM
- Performance lower than copy task but demonstrates real conversational use

### Data efficiency
- 123 +/- 13.1 sentences from a single session reach 95% of best single-session PER (~21.4 minutes of miming)
- 13 +/- 3.6 sessions with 50 sentences each reach 95% of multi-session minimum PER (29.1%)
- Finetuning with just 35.5 +/- 4.3 new sentences (~6.19 minutes) achieves 25% PER reduction vs. no finetuning

### Stability analysis
- Within-session LDA: median 92.5% accuracy (8-class word classification)
- Cross-session LDA (no retraining): drops to ~75.5% immediately, stable for ~40 days, then gradual decline to ~39.1% at 200+ days
- 56 +/- 3.1 of 64 channels showed substantial modulation during speech

## Cross-Patient Relevance

### How this compares to CCA-based alignment
This paper does **not** do cross-patient transfer at all. It is entirely within-patient, longitudinal adaptation. The relevant comparison points for Ben's CCA-based cross-patient work:

1. **Their "alignment" is temporal, not spatial:** They handle drift across sessions within one patient using day-specific input layers and finetuning, rather than aligning neural spaces across different brains.
2. **No explicit latent space alignment:** Unlike CCA which finds a shared subspace between patients, this paper uses task-driven finetuning (supervised, phoneme-labeled data) to implicitly re-align representations each session.
3. **Their cross-session LDA analysis (Fig. 5e) is informative for your setting:** It shows that even within one patient, neural representations drift enough that a fixed decoder degrades to ~39% accuracy after 200 days -- motivating alignment methods.
4. **PCA + LDA stability analysis:** Their approach of using PCA to reduce dimensionality + LDA for cross-session generalization (with fixed reference session weights) is conceptually similar to CCA in that both seek a lower-dimensional space where representations are stable, but CCA explicitly optimizes for cross-subject correlation.

### What's different about their setting vs. intra-op uECOG
- **Intracortical microelectrodes vs. uECOG:** They record single-unit spikes and spike-band power from penetrating Utah arrays; uECOG captures surface field potentials. Very different signal characteristics and spatial resolution.
- **Chronic implant vs. intra-operative:** They have 736 days of longitudinal data from one patient. Intra-op recordings are acute, single-session, limited duration.
- **Single patient with extensive data vs. multiple patients with limited data:** Their challenge is temporal drift; your challenge is spatial alignment across brains with minimal per-patient data.
- **Mimed speech vs. overt speech:** T16 mouthed words silently. Intra-op patients likely produce overt speech, which has different neural correlates.
- **Motor cortex (area 6v) vs. potentially broader cortical coverage:** uECOG grids may cover larger areas including STG, STS, or other speech-relevant regions.

## Limitations
What does NOT apply to Ben's intra-op uECOG setting:

1. **Single patient, no cross-patient transfer:** The paper provides zero evidence about cross-brain generalization. All results are within-participant T16.
2. **Penetrating microelectrodes:** The signal modality (threshold crossings, spike-band power from Utah arrays) is fundamentally different from uECOG surface potentials. Feature extraction pipelines do not translate directly.
3. **Chronic implant with hundreds of sessions:** The luxury of extensive per-patient data for pretraining + finetuning is unavailable in acute intra-op settings where you may have minutes, not months.
4. **Mimed (silent) speech only:** T16 mouthed words without vocalization. Overt speech produces auditory feedback and engages different neural circuits. Decoder architectures tuned for mimed speech may not transfer.
5. **Pontine stroke population:** Cortical thinning and reduced functional connectivity specific to pontine stroke may not be relevant to the surgical populations in the Cogan lab.
6. **Language model dependency:** Their strong WER results depend heavily on large 5-gram LMs and a 6.7B transformer for rescoring. This is standard for iBCI work but may obscure the raw neural decoding quality.
7. **The title in the filename ("cross-brain transfer") does not match this paper's content.** This paper is not about cross-brain transfer.

## Reusable Ideas

1. **Day-specific input layers with softsign activation:** A lightweight way to absorb distribution shift across sessions/patients. Instead of retraining the whole network, only the first dense layer is re-initialized and finetuned. This could be adapted for cross-patient transfer: learn patient-specific input projections that map each patient's neural space into a shared representation, while keeping the deeper RNN layers frozen.

2. **Finetuning protocol (60/40 mix of new/old data):** When adapting to a new patient with limited data, mixing new patient data with data from other patients in finetuning batches could prevent catastrophic forgetting while enabling adaptation. **Clarification:** 35.5 sentences = 25% relative PER reduction = **>80% of peak performance**, NOT near-full. **123 sentences** = 95% of peak. The MEMORY.md claim that "~36 sentences reaches near-full performance" overstates the finding.

3. **PCA + LDA for stability analysis:** Their framework for quantifying cross-session representational drift (train LDA on one session, test on others, plot accuracy vs. temporal distance) could be directly applied to quantify cross-patient representational similarity in uECOG, substituting "patient" for "session."

4. **Exponential curve fitting for data requirements:** Their approach of fitting exponential decay curves (Eq. 1) to PER vs. number of training sentences/sessions to estimate data requirements is a clean way to characterize how much per-patient calibration data is needed.

5. **CTC loss for phoneme decoding:** CTC allows variable-length phoneme sequence prediction without requiring precise phoneme boundary annotations, which is valuable in any setting where alignment between neural data and speech segments is imprecise.

6. **Feature augmentation during training:** Adding white noise and constant offsets to neural features during training improved generalizability. This could help cross-patient decoders be more robust to inter-patient variability in signal characteristics.

7. **Channel count vs. performance scaling (Fig. 4e):** Their log-linear relationship between channel count and PER, consistent across ALS and pontine stroke patients, suggests this scaling law may generalize. Useful for estimating expected performance from uECOG electrode counts.
