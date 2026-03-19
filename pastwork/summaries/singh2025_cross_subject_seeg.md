# Singh et al. 2025 — Transfer learning via distributed brain recordings enables reliable speech decoding

**Citation:** Singh, A., Thomas, T., Li, J., Hickok, G., Pitkow, X., & Tandon, N. (2025). Transfer learning via distributed brain recordings enables reliable speech decoding. *Nature Communications*, 16, 8749. https://doi.org/10.1038/s41467-025-63825-0

---

## Setup

- **Modality:** Stereo-EEG (sEEG) depth electrodes; 14–21 probes per patient, 0.5 mm or 2 mm platinum-iridium contacts with 0.5–4.43 mm spacing; 3641 total electrodes across cohort (3641 blue / clean channels from original implants)
- **Patients:** 25 epilepsy monitoring patients (12 male, age 19–51, verbal IQ 95.6 ± 10.4); left-hemisphere language dominant; all performed the task correctly (>80% accuracy)
- **Brain regions:** Distributed coverage across perisylvian frontotemporal language network — subcortical gyrus (SCG), superior temporal gyrus (STG), posterior middle temporal gyrus, premotor cortex, inferior frontal gyrus (IFG); coverage varies by clinical electrode placement
- **Task:** Tongue twister paradigm — 64 sets of 4-word sequences (128 stimuli, 218 unique words), covering 36 of 44 English phonemes; overt articulation for 2 of 4 trials per set, memory recall for last 2; ~1 hour total recording per subject including both articulation and pre-articulatory periods
- **Signal:** Broadband gamma activity (BGA, 70–150 Hz), bandpass filtered, common-average referenced, smoothed with Savitzky-Golay FIR; expressed as % change from 500–100 ms baseline

---

## Problem Statement

Speech BCIs trained on single-subject intracranial data are highly individualized and do not generalize across people. Coverage varies by clinical need, meaning some subjects lack electrodes over critical speech hubs (e.g., sensorimotor cortex). Existing high-performance BCIs require weeks of dense, intact cortical coverage and lengthy calibration. There is no established framework for:
1. Building a shared phonemic representational manifold from distributed, heterogeneous sEEG coverage
2. Initializing decoders for subjects with sparse or missing coverage from a population-level model
3. Proving that cross-subject transfer actually improves over within-subject training in phoneme decoding from sEEG

---

## Method

### Architecture: Sequence-to-Sequence (Seq2Seq) with per-subject convolutional front-end

The core model is an encoder-decoder RNN:

1. **Subject-specific encoder front-end:** A 1D convolutional layer + maxpool over time, applied independently per subject. This layer handles variable electrode counts and learns subject-specific spatial filters over the electrode dimension. Its weights are kept trainable during transfer (not frozen), allowing adaptation to each subject's unique electrode configuration and anatomical trajectories.

2. **Shared recurrent encoder (LSTM):** A bidirectional LSTM that processes the convolved neural time series into a latent hidden state. This layer captures the shared articulatory dynamics — the core transferable representation. It is the component that is **frozen** during inference transfer.

3. **Linear readout decoder (per-subject):** Separate linear layers for each subject that map from the shared latent space into phoneme probabilities for a fixed-length CVC model or a variable-length output. For the variable-length model, a teacher-forcing decoder with blank/start/end tokens and a connectionist temporal classification (CTC)-style loss handles alignment between predicted and articulated sequences.

4. **Variable-length decoder specifics:** Operates on a dictionary of tongue twister phoneme sequences; uses teacher-forcing during training; accommodates merging, concatenation, and deletion of extra tokens; handles non-stationarity in speech rate.

### Transfer Learning Strategy: Three modes evaluated

**Mode 1 — Full Transfer:** All layers from training subject are frozen and applied to inference subject. Weights transferred wholesale.

**Mode 2 — Readout Transfer:** The recurrent LSTM encoder is frozen; only the linear readout layer is fine-tuned on the inference subject's data. Best performing single-subject-to-single-subject mode.

**Mode 3 — Recurrent Transfer (primary finding):** The LSTM encoder is frozen; the convolutional front-end and linear readout are both fine-tuned. Fine-tuning on the new subject is done for only 100 epochs (vs. 500 epochs for pre-training). This is the approach that achieves statistically significant improvement over within-subject performance (p < 0.001).

Key finding: Freezing the RNN (shared latent dynamics) and allowing the subject-specific Conv1D to adapt is the optimal configuration. This makes sense because the Conv1D adapts to the specific electrode geometry while the RNN captures phoneme sequence dynamics invariant across subjects.

### Multi-Subject Group Model (the main proposed architecture)

Rather than transferring from one subject to another, the group model trains a **shared recurrent layer across N subjects simultaneously**:

1. Each subject has its own Conv1D + maxpool front-end (subject-specific nonlinear dimensionality reduction).
2. All subjects' encoded representations are **concatenated** and batched together to train a **single shared LSTM** encoder.
3. After the shared LSTM generates hidden states, trials are split back to their respective subjects and decoded by **subject-specific linear readout layers**.
4. During inference on a held-out subject: the shared LSTM is **frozen**, and a new Conv1D + linear readout is fine-tuned for 100 epochs using that subject's data (K=5 cross-validation folds, withholding entire phrases). **Note:** no zero-shot evaluation exists — all held-out subjects receive 100 epochs of fine-tuning.

Training procedure: random batching across all subjects to build group latents; subject-specific linear decoders ensure subject-level behavioral alignment while latent space is shared.

### Alignment / Manifold Strategy

- No explicit domain adaptation loss (e.g., no adversarial alignment, no MMD, no contrastive loss). The alignment emerges from the shared LSTM being trained on pooled neural data with a shared phoneme prediction objective.
- Coverage correlation analysis: Cortex parcellated into 50 Destrieux regions; channel density per parcel used as a "barcode" for each subject; cosine similarity computed across subjects. Transfer improvement is significantly higher (p < 0.05) when training and inference subjects have correlated cortical coverage (Fig. 3E), indicating that anatomical similarity facilitates manifold sharing.
- Subjects selected for the group model were chosen from the top quartile of decoding performance AND dense sensorimotor cortex (SCG) coverage — the 5 best subjects used to seed the group model.

### Training Details

- Optimizer/framework: TensorFlow/Keras; bidirectional LSTM, 2 layers, 64 units each (base model)
- Hyperparameters (layers, units) determined by cross-validation per subject
- Pre-training on single best subject: 500 epochs; fine-tuning on new subject: 100 epochs
- Group model: K=5 cross-validation; entire tongue twister phrases held out as test set to prevent data leakage
- Chance performance established by shuffling both neural data and phoneme sequences while preserving sEEG channel structure
- **Not reported in paper:** optimizer type, learning rate, batch size, weight decay, early stopping criteria. Our planned differential LR (read-in 3e-3, backbone 1e-3) is our own design choice, not based on Singh
- **Conv1D front-end details absent:** kernel size, stride, output channels, and activation function are NOT specified. The paper describes it as "1D convolutional layer + maxpool" without hyperparameters
- **No dropout or regularization mentioned** in the paper. Note: Singh had ~1 hr/patient vs our ~1 min/patient — we need substantially more regularization
- **Phoneme class imbalance:** 5 instances (TH) to 280 instances (B) — a 56:1 ratio across the corpus
- **Group model subjects were curated:** 5 subjects selected from top quartile of decoding performance AND dense sensorimotor cortex (SCG) coverage — not random LOPO

---

## Key Results

### Single-Subject Decoding
- Linear model baseline: 24% accuracy (76% PER), articulation period; 20% accuracy (80% PER), pre-articulatory
- Seq2Seq fixed-length (CVC): median 27% PER (articulation), 34% PER (pre-articulatory) — significantly better than chance and linear model (p < 0.01)
- Seq2Seq variable-length: median 44% PER (articulation), 56% PER (pre-articulatory) — significantly above chance (10% accuracy, SD 8%)
- Best single subject: PER as low as 13% (articulation), 24% (pre-articulatory) for fixed-length; 26% and 43% for variable-length from pre-articulatory

### Data Scaling Effects
- Number of trials: R² = 0.64 correlation between trials and decoding accuracy; ~180 min of articulation needed to reach <10% PER (90% accuracy)
- Number of channels: significant factor (p < 0.05) even after controlling for trials; improvement observed with as few as 100 channels; optimal separation seen at 500–1500 channels

### Transfer Learning (Single Subject to Single Subject)
- Within-subject PER: 0.49 (0.47–0.51 IQR)
- Full transfer: 0.46 (0.44–0.48) — not significantly different from within-subject
- Readout transfer: 0.45 (0.43–0.47) — p < 0.05 vs. within-subject
- Recurrent transfer: 0.43 (0.41–0.45) — p < 0.001 vs. within-subject
- Transfer improves significantly for inference subjects when training subject has more trials (Fig. 3B) and shared coverage correlation (Fig. 3E, p < 0.05)

### Group Model (Multi-Subject Transfer)
- Single-subject model PER for 5 held-out subjects: ~0.52–0.54 (varies per subject)
- Group model (n=5): significantly lower PER for all 5 subjects (p < 0.001 to p < 0.01)
- Improvement from within-subject to group (n=5): ~57% to 49% PER (p < 0.001) — ~8 percentage point gain
- Scaling with group size: n=1 within-subject ~0.57; n=2 group ~0.50 (p < 0.05); n=3 ~0.48 (p < 0.05); n=4 ~0.47 (p < 0.05); marginal gain plateaus after 3–4 subjects
- Transferred to a subject with only frontotemporal coverage (no sensorimotor cortex): group model maintained performance; single-subject model degraded substantially

### Regional Electrode Occlusion (REO) Analysis
- Single-subject models: removing vSMC, pSTG, or STS causes significant PER degradation (p < 0.01) at pre-articulatory windows; SCG, pSTG, and STS each significantly contribute
- Group model: removal of sensorimotor cortex electrodes degrades top performers but temporal lobe occlusion has minimal effect — group model is resilient to electrode occlusion relative to single-subject models
- This robustness is the key clinical motivation: group model compensates for missing speech hubs

---

## Cross-Patient Relevance

**Applicability to intra-op uECOG (~20 patients, ~20 min, sensorimotor cortex, phoneme + word/nonword tasks):**

1. **Same core challenge:** Your setting also has heterogeneous coverage (uECOG grids placed differently per patient), short recording windows, and no ability to recalibrate — exactly the regime this paper targets.

2. **Group model architecture is directly relevant:** The Conv1D per-subject + shared LSTM + per-subject linear readout structure is exactly the right approach when electrode counts vary across patients and you cannot assume spatial alignment. The Conv1D front-end naturally handles variable channel counts.

3. **Frozen LSTM as initialization:** Their key finding — freeze the recurrent layer, fine-tune only the spatial filter (Conv1D) and readout — directly maps to your scenario. With ~20 min of intra-op data, you can freeze the group LSTM and fine-tune the patient-specific layers on the limited available data.

4. **Pre-articulatory decoding (−350 to −50 ms window):** Their model decodes phonemes 350–50 ms before articulation onset, which is highly relevant for imagined/covert decoding or for decoding from pre-movement sensorimotor activity in your word/nonword tasks.

5. **Sensorimotor cortex primacy confirmed:** REO analysis confirms vSMC/SCG as the dominant hub for pre-articulatory phoneme decoding. Your uECOG is placed precisely there — this is encouraging for achieving useful signal from your recordings.

6. **Phoneme-level decoding target:** Their use of PER as the primary metric and phoneme sequences as targets aligns well with your phoneme task. Their variable-length decoder handles natural speech without forced alignment.

7. **Coverage correlation predicts transfer success:** Patients with anatomically similar electrode distributions transfer better. If you cluster your uECOG patients by gyral placement / coverage pattern (Destrieux parcellation), you can predict which training subjects will best initialize the decoder for a new intra-op patient.

8. **Very short fine-tuning (100 epochs):** Fine-tuning a new subject takes far fewer gradient steps than pre-training. This matters for intra-op use where online adaptation time is limited.

---

## Limitations

**What does not directly apply to the uECOG setting:**

1. **sEEG vs. uECOG signal properties:** sEEG depth electrodes sample distributed subcortical and deep cortical sites with high spatial specificity but sparse coverage. uECOG on sensorimotor cortex provides dense surface coverage of a focal region. The broadband gamma signal characteristics, noise floor, and electrode impedance are different. Their Conv1D front-end architecture must be re-validated for uECOG spatial structure.

2. **~1 hour of training data per subject:** Their pre-training subject contributed ~180 min to reach <10% PER; even their "low-data" subjects needed sufficient trials for meaningful transfer. Your ~20 min intra-op window is substantially shorter — it is unclear whether the group model initialization is sufficient to compensate for this gap, or whether you will be operating in the regime where performance is still improving steeply (left side of their Fig. 2C curve).

3. **Overt articulation during sEEG recording:** All their data involves overt speech — clear acoustic feedback and confirmed articulatory events. Intra-op data may involve imagined speech, covert production, or constrained vocalizations; they explicitly note inner speech is impoverished in phonemic representation.

4. **Healthy subjects performing tongue twisters:** Their 25 patients were neurologically intact for speech production (epilepsy, not aphasia). Patients undergoing intra-op mapping may have variable language function and tumor-related reorganization. The group manifold trained on healthy-range neural dynamics may not bridge this gap.

5. **No word/nonword discrimination task:** Their decoder outputs phoneme sequences from a closed vocabulary (tongue twister dictionary). They do not address binary word/nonword discrimination, which is one of your primary task conditions. Their architecture is designed for phoneme sequence decoding, not classification.

6. **Fixed electrode placement for seizure localization:** sEEG trajectories are planned days in advance with full neuronavigation. Intra-op uECOG placement is more variable and time-constrained, meaning the anatomical "barcode" approach (cosine similarity of Destrieux parcellation densities) may be noisier for selecting the best training subject.

7. **Stopping criterion for group size:** They found diminishing returns after 3–4 subjects; with only ~20 intra-op patients, you have enough to build a small group model, but the plateau behavior and the sensitivity to which subjects are included (top quartile of decoders + SCG coverage) means careful subject selection is critical.

8. **Tongue twister vocabulary is closed and structured:** The target phoneme sequences come from a predefined 218-word dictionary. This restricts the decoding problem and may inflate apparent performance compared to open-vocabulary or binary classification settings.

---

## Reusable Ideas

1. **Conv1D + shared LSTM + per-subject linear readout:** Adopt this architecture directly. The Conv1D spatial filter allows variable electrode counts; the shared LSTM captures cross-patient articulatory dynamics; separate linear readouts preserve patient-specific tuning. Straightforward to implement in TensorFlow/Keras or PyTorch.

2. **Freeze the recurrent layer, fine-tune the spatial filter:** When transferring to a new intra-op patient, freeze the LSTM encoder (which embeds the group-level phoneme dynamics) and only fine-tune the Conv1D and linear readout. This minimizes required data from the new patient and avoids catastrophic forgetting of the group manifold.

3. **Coverage correlation as subject selection criterion:** Before an intra-op case, compute the Destrieux-parcellation channel density vector for the planned electrode grid and identify which training patients have the highest cosine similarity. Use those patients preferentially in the group model or as initialization sources — this is a practical, anatomy-driven way to improve transfer without any neural data from the new patient.

4. **Pre-articulatory decoding window (−350 to −50 ms):** Train and evaluate decoders on neural activity preceding movement onset, not just during articulation. For intra-op settings where overt speech may be limited, pre-articulatory BGA in sensorimotor cortex provides useful signal and may generalize better across patients (less acoustic contamination, more motor planning signal).

5. **Regional electrode occlusion (REO) as interpretability tool:** Apply REO analysis to your uECOG data to quantify which subregions of sensorimotor cortex (e.g., hand vs. mouth area, pre- vs. postcentral gyrus) contribute most to phoneme vs. word/nonword decoding. This is both scientifically informative and practically useful for guiding intra-op grid placement in future cases.

6. **Teacher-forcing variable-length decoder with CTC-style loss:** Their variable-length Seq2Seq with blank/start/end tokens and connectionist temporal classification handles variable speech rates and phoneme sequence lengths naturally. This is more robust than fixed-window classifiers for naturalistic speech and worth implementing for the phoneme task.

7. **Stopping criterion for group size (cross-validation on held-out phrases):** Their K=5 fold scheme withholds entire phrases (not individual trials) to prevent leakage of phoneme context. Adopt this strategy to avoid inflated cross-validation estimates when stimuli share phoneme structure.

8. **BGA preprocessing pipeline:** Their pipeline — bandpass 70–150 Hz, zero-phase Butterworth notch for line noise, frequency-domain Hilbert transform, analytic amplitude, Savitzky-Golay smoothing — is a well-validated, reproducible BGA extraction method directly applicable to uECOG. The SB-MEMA framework for group-level activation maps is also useful for characterizing which electrodes are task-responsive before decoding.

9. **Macro F1 / PER instead of accuracy:** They explicitly avoid class-accuracy due to unequal phoneme distributions; they use macro F1 and phoneme error rate (edit distance). For your word/nonword task, balanced accuracy or macro F1 is similarly more appropriate than raw accuracy given potential trial imbalances.
