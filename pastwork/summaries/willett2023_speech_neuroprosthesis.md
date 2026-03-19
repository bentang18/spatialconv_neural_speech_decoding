# Willett et al. 2023 - A High-Performance Speech Neuroprosthesis

## Citation

Willett, F. R., Kunz, E. M., Fan, C., Avansino, D. T., Wilson, G. H., Choi, E. Y., Kamdar, F., Glasser, M. F., Hochberg, L. R., Druckmann, S., Shenoy, K. V., & Henderson, J. M. (2023). A high-performance speech neuroprosthesis. *Nature*, 620, 1031-1036. https://doi.org/10.1038/s41586-023-06377-x

## Setup

- **Modality**: Intracortical microelectrode arrays (Utah arrays) recording spiking activity (threshold crossings and spike band power). Four 64-electrode arrays (8x8 grids, 3.2 x 3.2 mm each).
- **Patient**: Single participant (T12), a 67-year-old woman with bulbar-onset ALS who retains limited orofacial movement and an ability to vocalize but cannot produce intelligible speech. Part of BrainGate2 clinical trial.
- **Brain region**: Two arrays in area 6v (ventral premotor cortex) and two in area 44 (part of Broca's area). Array locations chosen via Human Connectome Project multimodal cortical parcellation. Area 44 contributed little to decoding; all final analyses used area 6v only.
- **Task**: Attempted speech -- participant attempted to speak sentences displayed on a monitor. Included both vocalized (with sound) and silent (mouthing) conditions. Also single-word, single-phoneme, and orofacial movement tasks for neural tuning analysis.
- **Data amount**: 10,850 total sentences over 8 evaluation days. Training data: 260-480 sentences per day (~41 +/- 3.7 min of data per day). Data collection and RNN training lasted ~140 min per day on average. Training was cumulative across days (up to ~15 days of prior data used).

## Method

### Preprocessing
- Neural activity band-pass filtered (250-5000 Hz) with 4th-order acausal Butterworth filter.
- Spiking events detected using -4.5 RMS threshold crossing. 128 electrodes from area 6v (2 of 4 arrays); area 44 excluded from all decoding.
- Two features per electrode: threshold crossing rate and spike band power. 256 total features (128 electrodes × 2).
- Features temporally binned at **20 ms** and smoothed with Gaussian kernel (sd=40 ms, delayed 160 ms for causal smoothing).
- Rolling z-score normalization: for first 10 sentences of new block, blends prior block's estimate with current (weighted average); after 10 sentences, uses mean of most recent min(20, N) sentences.
- Day-specific input layers: **affine + softsign** transformation `x̃ = softsign(Wx + b)` where W is 256×256 and b is 256×1 per day (**65,792 params/day**). This is NOT dimensionality-reducing — it preserves input dimension. Softsign constrains output to [-1, 1], acting as implicit normalization. Dropout applied both before and after softsign. Trained end-to-end with CTC loss (not frozen/separate).
- **Data augmentation**: White noise (Gaussian SD=1.0) added per time step + constant offset noise (Gaussian SD=0.2) added per minibatch (same across all time steps, unique per feature). Simulates baseline drift.

### Decoder Architecture
- 5-layer **unidirectional** GRU RNN, **512 hidden units per layer**, implemented in TensorFlow 2.
- Input: **14 consecutive 20 ms bins** (280 ms window) stacked into 256×14 = **3,584-dim** input vector at each time step, strided by **4 bins (80 ms)**. The RNN outputs one probability vector per 80 ms step.
- Output: probability distribution over **41 classes** (39 CMU phonemes + 1 silence token + 1 CTC blank) at each 80 ms step.
- Trained with CTC loss. **Adam optimizer** (beta1=0.9, beta2=0.999, epsilon=0.1), LR linearly decayed from 0.02 to 0.0, batch size 64 sentences, 10,000 minibatches, dropout 0.4, L2 weight regularization.
- **Two-stage per-day training**: Stage 1 trains on open-loop blocks (no active decoder) + all prior days. Stage 2 collects additional blocks WITH real-time feedback (Stage 1 decoder active), then retrains combining all data. Final evaluation uses Stage 2 RNN on held-out sentences.
- 27 total sessions spanning post-implant days 22-148. 8 evaluation sessions (5 vocal, 3 silent). Training was cumulative: each evaluation day used ALL prior days' data.

### Language Model
- Two vocabulary sizes tested:
  - **50-word vocabulary**: small language model suitable for simple sentences.
  - **125,000-word vocabulary**: 3-gram LM for online (pruned, threshold 1e-9), 5-gram LM for offline (threshold 4e-11). Built using SRILM with Good-Turing discounting on **OpenWebText2** corpus (634M sentences), NOT Switchboard. Switchboard provided the speaking sentences, not the LM training data.
- Decoded via **Weighted Finite-State Transducer (WFST)**, composed as T∘L∘G (token, lexicon, grammar), built with Kaldi.
- Offline analysis showed an improved language model could reduce WER from 23.8% to 17.4%, and proximal test sentences further to 11.8%.

### Training
- Sentences from the Switchboard corpus of spoken English.
- Each evaluation day: fresh RNN trained on all prior days' data combined with current day's data.
- Performance was reasonable even with zero same-day training data (~30% WER), suggesting unsupervised adaptation methods could work.

## Key Results

| Metric | 50-word vocab | 125,000-word vocab |
|--------|-------------|-------------------|
| **WER (vocal, online)** | 9.1% | 23.8% |
| **WER (silent, online)** | 11.2% | 24.7% |
| **WER (offline, improved LM)** | -- | 17.4% |
| **WER (offline, improved LM + proximal test)** | -- | 11.8% |
| **Phoneme error rate (vocal)** | 21.4% | 19.7% |
| **Phoneme error rate (silent)** | 22.1% | 20.9% |
| **Speaking rate** | 62 words/min | 62 words/min |

- **2.7x fewer errors** than previous state-of-the-art speech BCI (Moses et al. 2021, 50-word vocab).
- **3.4x faster** than previous record (18 words/min for handwriting BCI).
- 62 words/min approaches natural conversation speed (~160 words/min).
- First successful large-vocabulary (125k words) speech decoding from neural data.
- Doubling electrode count approximately halves WER (log-linear relationship, factor of 0.57).
- Silent and vocalized speech decoded at comparable accuracy, important for locked-in patients.
- Vocabulary size beyond ~1,000 words showed diminishing returns on accuracy improvement.

### Neural Coding Findings
- Area 6v (ventral premotor cortex) contains rich, intermixed representation of all speech articulators (jaw, lips, tongue, larynx) at the single-electrode level within 3.2 x 3.2 mm.
- Area 44 (Broca's area) contained almost no decodable information about orofacial movements, phonemes, or words (classification accuracy <12%).
- Articulatory representation of phonemes is preserved years after paralysis: neural similarity matrices for consonants mirror place-of-articulation structure from EMA data, and vowel representations mirror the two-formant (high/low, front/back) structure.
- Ventral 6v array contributed more to phoneme/word decoding; dorsal 6v array contributed more orofacial movement information. Both arrays useful -- combining them reduced WER from 32% to 21%.

## Cross-Patient Relevance

For cross-patient uECOG speech decoding in the Cogan lab:

- **Articulatory representations generalize**: The neural code for speech in area 6v mirrors known articulatory structure (place of articulation for consonants, formant structure for vowels). This structure, being grounded in the physics of articulation, should be consistent across patients and could serve as an alignment target for cross-patient transfer.
- **Phoneme-level decoding is the right intermediate target**: The RNN decodes phonemes first, then a language model maps to words. This two-stage approach means the neural decoder only needs to learn ~39 phoneme classes, not thousands of words, which is much more feasible for cross-patient generalization.
- **Small cortical area is sufficient**: All speech articulators are intermixed in a 3.2 x 3.2 mm patch of area 6v. This means uECOG grids over ventral premotor cortex should capture a similarly rich representation, even if electrode density is lower than Utah arrays.
- **Area 44 is not useful for decoding**: Broca's area carried negligible articulatory information. Target ventral premotor cortex (area 6v) for electrode placement.
- **Day-specific input layers and rolling z-scoring**: Cross-day neural drift is a major challenge. Their solution (day-specific input layers feeding into a shared RNN) is directly analogous to the cross-patient problem -- patient-specific input layers could map diverse neural signals into a shared latent space.
- **Silent speech works almost as well as vocalized**: Relevant for patients with more severe paralysis. Also suggests motor cortex representations are about intended articulation, not auditory/sensory feedback.
- **Single-patient limitation acknowledged**: The authors explicitly note that generalizability to additional participants with more profound orofacial weakness is an open question, and variability in brain anatomy is a concern. This is exactly the gap cross-patient work addresses.

## Reusable Ideas

1. **Day-specific input layers + shared decoder**: Train patient-specific (or session-specific) linear input layers that project neural features into a common space, then share the rest of the RNN across patients/sessions. This is their key strategy for handling non-stationarity and directly maps to cross-patient alignment.

2. **Rolling z-score for feature adaptation**: Simple but effective -- compute running mean/std of neural features and z-score to remove slow drifts. Should be applied per-electrode in any cross-patient pipeline.

3. **Phoneme-level CTC decoding**: Using CTC loss to train an RNN that outputs phoneme probabilities at each time step, without requiring exact phoneme-to-time alignment. This is attractive for uECOG where temporal alignment is uncertain.

4. **Language model as a separate, modular stage**: Decoupling the neural decoder (phoneme probabilities) from the language model (phoneme-to-word mapping) means the language model is patient-independent and only the neural decoder needs adaptation.

5. **Log-linear scaling of electrodes to accuracy**: Doubling electrodes halves WER. This provides a principled argument for denser uECOG grids and suggests that pooling electrodes across patients (in a shared latent space) could yield similar benefits.

6. **Articulatory structure as a cross-patient anchor**: Since neural representations mirror articulatory structure (which is universal), one could align cross-patient neural spaces by matching their articulatory geometry (e.g., Procrustes alignment on phoneme representations ordered by place/manner of articulation).

7. **Spike band power + threshold crossings outperform either alone**: For uECOG, the analogous lesson is to use multiple signal features (e.g., high-gamma power, broadband, phase) rather than a single feature type.

8. **Code and data publicly available**: Code at https://github.com/fwillett/speechBCI. Data on Dryad at https://doi.org/10.5061/dryad.x69p8czpq. Useful for benchmarking or pretraining.
