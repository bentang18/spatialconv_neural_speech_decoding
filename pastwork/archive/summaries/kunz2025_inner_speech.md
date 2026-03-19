# Kunz et al. 2025 — Inner Speech in Motor Cortex and Implications for Speech Neuroprostheses

**Citation:** Kunz EM, Abramovich Krasa B, Kamdar F, et al. Cell 188, 4658–4673. August 21, 2025. https://doi.org/10.1016/j.cell.2025.06.015

---

## Setup

- **Recording modality:** Chronic intracortical microelectrode arrays (Utah arrays, 64-channel, 1.5 mm silicon, Blackrock Neurotech). Multi-unit threshold crossing rates and spike band power used as neural features (128-dimensional feature vectors per array per 500 ms window).
- **Patient count:** 4 BrainGate2 participants (T12, T15, T16, T17), T12, T15, and T17 with ALS (T17 severely anarthric and ventilator-dependent), T16 with tetraplegia from pontine stroke.
- **Brain regions:** Precentral gyrus (motor cortex), specifically the ventral (i6v, area 6v — orofacial motor) and mid (55b) precentral gyrus. Arrays also placed in area 6d (hand knob), PEF (premotor eye field), area 4 (primary motor), and inferior frontal gyrus (area 44). Only 6v and 55b arrays showed robust speech tuning across behaviors.
- **Tasks:**
  - Isolated verbal behavior instructed-delay task: 7 single-syllable words, 7 verbal behaviors (attempted vocalized speech, attempted mimed speech, motoric inner speech, 1st-person auditory inner speech, 3rd-person auditory inner speech, passive listening, silent reading).
  - Real-time self-paced inner speech sentence decoding (50-word and 125,000-word vocabularies).
  - Uninstructed inner speech tasks: 3-element arrow sequence recall, conjunctive counting, verbal/autobiographical thought prompts.
  - Motor-intent characterization: interleaved blocks of attempted and inner speech.
  - Privacy-prevention experiments: imagery-silenced training and keyword-gated decoding.

---

## Problem Statement

Two interrelated gaps addressed:

1. **Inner speech as a BCI modality:** Current speech BCIs require users to physically attempt (or mime) speech, which is fatiguing, produces distracting vocalizations, and creates physiological constraints (e.g., breathing control) for severely paralyzed users. Whether inner speech — which requires no motor output — is robustly represented in motor cortex and decodable in real time was unresolved.

2. **Mental privacy:** A concern raised by researchers and BCI users is whether attempted-speech BCIs could unintentionally decode private inner thoughts. Whether this is a real risk and whether it can be technically prevented had not been rigorously characterized.

---

## Method

### Neural Features
- Threshold crossing rates + spike band power, binned at 10 ms or 20 ms, smoothed with 60 ms Gaussian kernel, averaged into 500 ms windows (optimized per array/behavior via nested 10-fold CV).
- Features normalized per block (rolling z-score subtraction of mean firing rate drift) to address nonstationarity.

### Characterization (Word-Level, 7-Class)
- **Gaussian Naive Bayes (GNB)** classifier, nested 10-fold cross-validation, 500 ms window. Chance = 14.3% (1/7).
- Cross-validated Pearson correlation between 128-dimensional average word vectors used to assess shared neural geometry across behaviors.
- PCA to visualize 3D word-space geometry across behaviors; normalized Euclidean distances to quantify modulation magnitude relative to attempted vocalized speech.

### Real-Time Sentence Decoding
- **5-layer stacked gated recurrent unit (GRU) RNN** outputting phoneme probabilities every 80 ms (39 phonemes + silence token), trained with CTC loss, following the architecture from Willett et al. 2023 and Card et al. 2024.
- Separate input transformation layer per session (dataset-specific) allows mixing attempted speech data from prior sessions with inner speech training data from the current session.
- **n-gram language model** (Kaldi, OpenWebText2) with CMU Pronunciation Dictionary for 125,000-word decoding; Moses et al. 50-word vocabulary for closed-set evaluation. T15 additionally used a pre-trained OPT transformer LM for rescoring.
- Online retraining during evaluation blocks for T15 (continuous) and T16 (post-decode), not used for T12/T16 50-word sessions.
- Performance measured by word error rate (WER), bootstrap 95% CIs, chance estimated by shuffling decoded outputs against ground truth 10,000 times.

### Motor-Intent Dimension
- Defined as the normalized vector from the centroid of inner speech word representations to the centroid of attempted speech word representations in 128-dimensional neural feature space.
- Removed from neural data by projecting out: `x_res = x - (x · v_motor-intent) * v_motor-intent`.
- GNB classifiers re-trained on motor-intent-removed features to assess behavior separability independently of word content.

### Imagery-Silenced Training Strategy
- RNN trained on attempted speech trials (labeled with phonemes) plus inner speech trials (labeled as silence token "SIL").
- Contrasted against "imagery-naive" strategy (trains only on attempted speech trials).
- Evaluation: WER on attempted speech test sentences; false-decode rate on inner speech trials; correlation of phoneme logits between paired attempted and inner speech trials (using Dynamic Time Warping for temporal alignment).

### Keyword Gating
- Language model augmented with a specific keyword ("ChittyChittyBangBang") to act as an unlock trigger; decoder output suppressed until keyword detected.

---

## Key Results

### Word-Level Classification (7-Class, 500 ms window)
- **Best arrays (6v and 55b):** Robust decoding of attempted speech, inner speech, listening, and silent reading across participants.
  - T12-i6v: attempted vocalized 97.9%, motoric inner speech **72.6%** (95% CI [65.7%, 78.8%])
  - T16-i6v: listening 92.1% (95% CI [86.4%, 96.0%]), inner speech ~66–80% depending on type
  - T15-55b: 85.7% attempted vocalized, 77.9% attempted mimed; inner speech 17.6–38.6% (weaker)
- **T17** (anarthric): 6v arrays showed significant inner speech decoding despite no overt speech ability.
- Area 6d (hand knob), PEF, and primary motor cortex (area 4) did not decode speech reliably.

### Shared Neural Code
- Cross-validated pairwise neural correlations between the same words across all behaviors were high (majority > 0.5), confirming a shared neural encoding subspace for attempted, inner, listening, and silent reading.
- Inner speech neural distances normalized to attempted vocalized speech: T12 motoric inner speech ~52% of attempted, consistent with a scaled-down rather than orthogonal representation.
- A clear "motor-intent" dimension separates attempted from inner speech in PCA space; motor-intent modulation is comparable to or larger than word modulation in T12, T16, T17.

### Real-Time Sentence Decoding
- **50-word vocabulary:**
  - T12: WER 24% (95% CI [18.5%, 28.0%])
  - T15: WER 14% (95% CI [9.9%, 20.2%])
  - T16: WER 33% (95% CI [21.8%, 41.1%])
- **125,000-word vocabulary:**
  - T15: WER 26–32% (95% CI [15.5%, 32.3%] — 50-word equivalent sessions)
  - T16: WER 42–65% (95% CI [42.5%, 65.3%])
- Cross-behavior transfer (decoder trained on attempted speech, tested on inner speech): above chance for all participants, though worse than training directly on inner speech.

### Uninstructed Inner Speech
- 3-element arrow sequence recall task: sequence positions decoded above chance in T12-i6v (but not in hand-motor area 6d or s6v), with decoding increasing when participants explicitly instructed to use verbal memory strategy.
- Cross-task generalization: decoders trained on attempted speech direction words generalized to verbal memory task (T12-i6v, T16-i6v).
- Counting task (T15, T16): inner speech RNN decoded sequentially increasing numbers (T15 slope = 0.48, p = 1.69e-9; T16 slope = 0.33, p = 1.5684e-08), significantly different from null distribution from instructed inner speech sentence trials.
- Verbal thought prompts: significantly more words decoded during verbal thought cues (~22 words/trial for T15, ~5–9 for T16) than during "clear your mind" baseline.

### Motor-Intent Dimension and Privacy
- Removing motor-intent dimension: word accuracy largely unchanged (CIs overlap) but behavior classification accuracy drops significantly (non-overlapping 95% CIs for all participants) — confirming the motor-intent dimension is behavior-specific but not word-specific.
- **Imagery-silenced training:** false-decode rate during inner speech trials reduced to 1.0% (T12), 3.0% (T15), 0.33% (T16), while WER on attempted speech maintained.
- **Keyword gating (T12):** keyword detection accuracy 98.75% (95% CI [93.23%, 99.97%]); WER on 50-word vocabulary 43.45% (95% CI [35.7%, 49.4%]).

---

## Cross-Patient Relevance

Relevance to the target setting (intra-op uECOG, ~20 patients, ~20 min each, sensorimotor cortex):

1. **Shared neural code across behaviors is strong evidence for generalizability.** The finding that inner speech, attempted speech, listening, and silent reading all share a common neural subspace in motor cortex (particularly i6v and 55b) is directly relevant: if cross-patient alignment transfers the geometric structure of word representations, that structure will be preserved regardless of whether the patient produces overt or covert speech during recording. For short intra-op sessions, inner speech may be easier to elicit reliably without fatiguing patients.

2. **The 6v (orofacial motor cortex) and 55b (mid precentral) regions are the critical speech zones.** This strongly matches what is targeted in uECOG intra-op grids over sensorimotor cortex. Cross-patient decoding efforts should prioritize spatial alignment to these regions; other areas (hand knob 6d, PEF, primary motor area 4) are uninformative for speech.

3. **Inner speech is a scaled-down version of attempted speech, not an orthogonal signal.** This means a model pre-trained on attempted speech data from other patients should have non-trivial zero-shot or few-shot transfer to inner speech (confirmed empirically: cross-behavior decoders were above chance). For intra-op patients who cannot produce overt speech, inner speech might still activate the same subspace at reduced gain, making alignment and decoding harder but not impossible.

4. **The motor-intent dimension provides a concrete, computable handle for behavior type.** The centroid-difference vector between inner and attempted speech conditions is a simple, data-driven feature that could be useful for normalizing across patients whose overt motor output varies (e.g., dysarthric vs. anarthric patients). This is essentially a form of condition-level mean subtraction that removes behavior-type information while preserving phonemic content.

5. **T17 (anarthric, no observable speech output) still showed decodable inner speech in 6v.** This is the most directly relevant participant to intra-op locked-in or severely impaired patients. It demonstrates that motor cortex inner speech representations do not require intact corticospinal output to form.

6. **The RNN + CTC + n-gram LM architecture (from Willett/Card lineage) is the proven pipeline.** Using threshold crossings and spike band power as features (128-dim per 64-channel array, 500 ms window) is directly portable to uECOG if sufficient spatial resolution is available. The session-specific input transformation layer pattern is also relevant for handling nonstationarity across short intra-op recording sessions.

7. **Nonstationarity handling via block-mean subtraction.** The paper explicitly acknowledges firing rate drift across time as a confound and uses per-block mean subtraction. For very short intra-op sessions (~20 min), a diagnostic/rest block at the start for threshold calibration and LRR filter computation (as done here) is the appropriate approach.

---

## Limitations

1. **N=4 chronic implant participants (3 ALS, 1 pontine stroke).** Chronic implants (months to years post-implant) have very different signal characteristics from acute intra-op recordings. Spike amplitude, array impedance, spatial coverage, and session-to-session stability are all different. None of these findings have been validated in intra-op settings.

2. **Very long training data requirements.** The RNN decoders used up to 1 hour of cued inner speech data per session, supplemented with previously collected attempted speech data from multiple prior sessions. Intra-op settings provide ~20 minutes of recording with no prior sessions — this is a fundamental mismatch. The cross-behavior transfer result (attempted speech decoder tested on inner speech) is partially reassuring but WERs are notably higher.

3. **Individual participant inner speech strategy varied substantially.** T12 used motoric inner speech, T15 used a hybrid strategy, T16 used 1st-person auditory inner speech. Strategy inconsistency across patients and sessions adds variability that is hard to control for. Intra-op patients will not have time to identify their preferred strategy.

4. **No cross-patient decoding demonstrated.** All decoding results are within-patient. The shared neural geometry result is descriptive (via correlation matrices) but no model was trained on one patient and tested on another. The central challenge for the target setting — generalizing across 20 different patients — is not directly addressed.

5. **7-word / sentence-level vocabulary, not phoneme-level cross-patient.** The word-level GNB results use only 7 words; the RNN results rely on patient-specific input transformation layers and session-specific normalization. Neither architecture is designed for zero-shot cross-patient deployment.

6. **Intra-op uECOG spatial resolution is lower than Utah arrays.** Utah array multi-unit activity provides a much higher-dimensional and higher-SNR signal than macro-ECoG. The 128-dimensional feature vectors (2 features × 64 channels) reflect dense local spiking; uECOG feature vectors will be lower-dimensional and dominated by LFP/HGA rather than spiking, requiring different feature extraction.

7. **Free-form inner speech decoding remains poor.** Even with the best models, free-form verbal thought prompts produced largely gibberish output. Uninstructed inner speech during non-speech cognitive tasks was partially decodable but not interpretable at the sentence level. For patients who cannot produce cued speech, this gap is critical.

8. **Privacy experiments are offline evaluations.** The imagery-silenced strategy was evaluated offline; the keyword gating was only tested in real time for one participant (T12) with a 50-word vocabulary. Generalization to large-vocabulary real-time settings or cross-patient settings is untested.

---

## Reusable Ideas

1. **Cross-behavior neural correlation matrix as a diagnostic tool.** Before training any decoder, computing pairwise word-level neural correlations across behaviors (Figure 2A–B style analysis) can quickly assess whether a given recording site and patient show the expected shared phonemic structure across speech behaviors. This is low-cost and can guide array placement or patient selection for intra-op studies.

2. **Motor-intent dimension removal as a data normalization step.** Computing the centroid-difference vector between two behavioral conditions (e.g., overt vs. inner speech, or cued vs. resting) and projecting it out of neural features is a simple operation that removes behavior-type mean shifts while preserving word-level variance. For cross-patient alignment, analogous mean-shift removal (e.g., per-patient centering in shared feature space) could reduce inter-patient variability. The key insight is that this operation preserves word decoding accuracy (CIs overlap) while dramatically changing behavior classification accuracy.

3. **Imagery-silenced training strategy for cross-patient transfer.** Training an RNN on attempted speech data labeled with phonemes but treating an alternate behavioral condition (inner speech, or in the cross-patient case, held-out patients' data) as silence is a clean way to make a decoder robust to a signal it should not decode. For cross-patient models, this could be repurposed as a regularization strategy: if some patients' neural patterns look like "noise" relative to the canonical signal, label their data as silence during training to prevent the model from overfitting to their idiosyncratic patterns.

4. **Session-specific input transformation layer.** The RNN uses a separate linear input transformation layer per session/dataset, trained from scratch, while the recurrent layers are shared. This is effectively a lightweight session-specific adapter. For cross-patient settings, a patient-specific (or session-specific) affine input transform trained on a small amount of data could serve as an alignment layer on top of a shared recurrent backbone — analogous to the NoMAD/BIT approaches but with an explicit training procedure tied to the RNN rather than a separate alignment module.

5. **Combining attempted speech training data with inner speech evaluation.** Even in the absence of prior inner speech data, training on attempted speech and evaluating on inner speech was above chance for all participants. For intra-op settings where no prior data exists, using attempted speech from the first few minutes of recording (if the patient can produce any) to initialize a decoder that generalizes to inner speech is a viable strategy worth testing.

6. **Nested cross-validation for window optimization.** The nested 10-fold CV procedure for optimizing the neural decoding window start time (Figure S6) prevents overfitting the timing parameter to the test set. This is directly applicable to intra-op settings where the optimal delay from cue onset to peak neural modulation may vary by patient and cannot be assumed.

7. **GNB classifier as a fast, interpretable probe.** Before investing in RNN training, a GNB classifier over the 128-dim features provides a reliable upper-bound estimate of discriminability for a small word set and can diagnose which arrays and brain regions carry useful phonemic information. For intra-op sessions with limited time, this could serve as a rapid quality check during or immediately after recording.

8. **Counting/sequence task as an inner-speech elicitation probe.** The 3-element arrow sequence recall task and conjunctive counting task elicited decodable inner speech without explicit instruction. For intra-op patients who are unable or unwilling to produce overt speech but can follow simple visual instructions, these paradigms could provide additional neural data from which to estimate speech representations — useful for bootstrapping a cross-patient model on patients with severe motor impairment.
