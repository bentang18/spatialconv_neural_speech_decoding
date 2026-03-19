# Improving Speech Decoding from ECoG with Self-Supervised Pretraining

**Citation:** Yuan & Makin (2024). arXiv:2405.18639. Preprint under review.

---

## Setup

- **Modality:** Sub-chronic ECoG (intracranial grid, 128 or 256 electrodes, 4-mm spacing)
- **Patients:** 4 participants (a, b, c, d) with epilepsy undergoing monitoring
- **Brain region:** Cortical areas near the Sylvian fissure, including speech production and perception regions
- **Task:** Overt read-aloud speech; sentences from MOCHA-TIMIT (~460 sentences, ~1800 unique words) and picture descriptions (~30 sentences, ~125 words)
- **Data volume per patient:** ~1 hour of OD data (participants a, c, d); ~30 min (participant b); supervised blocks: 30-50 sentences each
- **Signal processing:** 60 Hz notch, 70-200 Hz bandpass (high-gamma, note: 200 Hz cutoff vs original Makin et al.'s 150 Hz), analytic amplitude (envelope), CAR (vs original bipolar referencing). Output at **~3 kHz** — the ~100 Hz is the wav2vec encoder output rate, not the preprocessing output. For control condition (raw ECoG → ECOG2TXT), signals explicitly **downsampled to 100 Hz** to match wav2vec output rate; this 100 Hz downsampling **hurt participant a's performance** (a confound for their poor results)

---

## Problem Statement

ECoG-based speech decoders require 15-20 hours of labeled data to reach acceptable performance (~25% WER). Collecting this from non-speaking patients takes ~2 weeks. Meanwhile, patients produce large amounts of unlabeled speech outside experimental blocks that goes entirely unused. This paper asks whether that unlabeled "out-of-distribution" (OOD) speech data can be leveraged via self-supervised learning to reduce the labeled-data burden and improve decoding.

---

## Method

**Architecture:** WAV2VEC **1.0** (CPC-style, Schneider et al. 2019) adapted for ECoG. **NOT wav2vec 2.0** — there is no quantizer module or masked prediction. This is contrastive predictive coding (CPC).
- Input layer expanded to N_electrodes channels (128-245 after artifact rejection, varies by participant)
- Encoder: 3 **causal** convolutional layers (kernels 8, 4, 2; strides 5, 3, 2) downsampling **~3 kHz preprocessed ECoG** to ~100 Hz latent vectors z
- Context network: 3 additional causal conv layers (kernels 2, 3, 4; stride 1) producing context vectors c with ~70 ms receptive field
- 512 output channels throughout; ELU activations (vs. ReLU in original)
- All convolutions are **causal** (not bidirectional)
- Model scaled down ~3 orders of magnitude relative to Librispeech-scale WAV2VEC

**SSL objective:** Noise-contrastive / InfoNCE loss (variant of WAV2VEC loss). Context vector c_t must predict future latent z_{t+s} (s = 8 steps ahead by default, selected via hyperparameter tuning on participant c). Maximizes a lower bound on mutual information between c_t and z_{t+s}.

**SSL training data:** All OOD blocks (unlabeled; sentences not used in supervised training/testing). Split into 200,000-sample (~1 min) segments, 90/10 train/validation split. Trained with AdaM (lr=1e-4) until validation loss plateaus for 50 epochs (~2-3 hours on RTX A6000).

**Downstream decoder (ECOG2TXT):** Encoder-decoder with **GRUs** (LSTMs→GRUs from original Makin et al.) + **cross-attention** (replaces final hidden state coupling). Encoder predicts phoneme transcriptions (substituting MFCCs for privacy); decoder predicts next word autoregressively from a closed vocabulary (~1800 words, but effectively memorizes sentence chunks from 30-50 sentence set). **No CTC is used.** Trained on WAV2VEC context vectors (or raw preprocessed ECoG as control) for 400 epochs. 30 separate models trained per experiment; each result point is the **median over 30 instances**. Reimplemented in **PyTorch** (original was TensorFlow).

**Transfer learning variant:** WAV2VEC pretrained on participant b's OOD data, initial conv layer replaced with randomly initialized layer sized for participant a, frozen except input layer for 100 epochs on a's OOD data, then fully unfrozen until convergence.

---

## Key Results

**Single-subject SSL:**
- WAV2VEC context vectors outperform raw ECoG for nearly all participants and training set sizes
- No experiment shows context vectors performing worse than raw ECoG
- Best improvements (participant b, 3 labeled blocks): ~60% WER reduction (59.90%, p<0.005)
- Participant b at 2 blocks: median WER drops ~45%, bringing performance to the outer bound of usable speech transcription
- Participant c: significantly better at all training sizes (7-35% improvement)
- Participant d: significant improvement at smaller training sizes (up to 20.82%), advantage disappears at saturation (~2% WER floor)
- Participant a: no significant improvement from within-subject SSL; required cross-patient transfer

**Cross-patient transfer:**
- WAV2VEC pretrained on b, fine-tuned on a's OOD data: WER improves by 6.26% (p<0.01) at 1-block training (absolute WER: 0.81→0.76), and 14.6% (p<0.005) at 2-block training (0.68→0.58)
- **All pairwise** transfer combinations were tested; **only b→a produced significant improvement** (~1/12 success rate). Cross-patient SSL transfer is not reliably effective

**SSL vs labeled OOD (Figure 3, missing from prior summary):**
- Participant b: SSL context vectors **match** the performance of supervised training on OOD data with labels
- Participant d: SSL context vectors are **better** than supervised training with OOD labels
- Participant a: neither SSL nor OOD labels help
- Statistical test: Wilcoxon signed-rank with Holm-Bonferroni correction

**WAV2VEC loss vs. WER:** Lower WAV2VEC pretraining loss does not reliably predict lower downstream WER. Participant a has the best WAV2VEC loss (0.451) but worst downstream improvement.

**Preprocessing is necessary:** Removing CAR degrades decoding moderately to substantially. Removing the bandpass filter is fatal (low-frequency bands dominate amplitude, easy to predict but speech-irrelevant). Using filtered voltages instead of envelopes also fails. SSL must be layered on top of standard signal processing, not substituted for it.

---

## Cross-Patient Relevance

Directly relevant to the intra-op uECOG setting (~20 patients, ~20 min each, sensorimotor cortex):

- **Validates SSL for data-scarce ECoG:** The paper's core finding is that SSL from unlabeled ECoG substantially helps when labeled data is limited. The intra-op setting is exactly this regime: ~20 min per patient maps to very few labeled blocks (comparable to participants a and b here, where gains were largest).
- **Cross-patient pretraining works:** Even with a simple layer-swap approach, pretraining on one patient's OOD data and fine-tuning on another's yielded significant gains. With 20 patients instead of 4, a pooled SSL pretraining corpus could be considerably larger and more effective.
- **Gain is largest at low labeled-data regimes:** The WER improvements are most pronounced with 1-5 labeled blocks. Intra-op sessions will typically yield this range of usable labeled data, placing the use case squarely in the high-benefit zone.
- **The OOD data concept applies:** In intra-op settings, ECoG recorded outside the structured task (during spontaneous speech, preparation periods, inter-trial intervals) could serve as unlabeled SSL training data, analogous to the OOD blocks here.

---

## Limitations

- **Overt speech only:** All ECoG was recorded during overt read-aloud speech. Applicability to attempted/imagined speech (relevant for severely paralyzed patients) is untested.
- **Sub-chronic implants, not intra-op:** Grid electrodes implanted for epilepsy monitoring over days to weeks; different from acute intra-op uECOG (microelectrode arrays, sensorimotor cortex, ~20 min sessions). Signal characteristics, electrode count, and spatial resolution differ substantially.
- **Closed vocabulary:** Task used 30-50 repeated sentences from a fixed corpus. ECOG2TXT sometimes memorizes sentence chunks; results may overstate generalization to open vocabulary decoding.
- **SSL gains plateau with abundant labeled data:** Benefits disappear or reverse when training set is large (participants c and d at high block counts). In an intra-op setting this is not the binding constraint, but it means SSL is not a substitute for more data if obtainable.
- **Patient-specific input layers required:** Each patient needs their own initial convolutional layer due to varying electrode counts and placement. No truly shared encoder across patients is demonstrated.
- **4 participants only:** Transfer learning results are based on a single pair (b to a). All other pairwise combinations failed. Generalization of the cross-patient SSL approach is not established.
- **OOD data still came from controlled sessions:** The "unlabeled" blocks were **active read-aloud speech** without transcription labels, not silent/rest periods or truly unconstrained naturalistic speech. SSL from genuinely unconstrained neural activity or inter-trial silence remains unvalidated.
- **Minimum successful SSL corpus is ~30 min** (participant b). Our ~1 min/patient utterance is **30x smaller** than even the smallest successful case. Even pooling all ~20 patients gives ~20 min — still below this threshold. No data ablation was performed to find the true minimum.
- **All participants were epilepsy patients who could speak normally.** Transferability to anarthric/non-speaking patients is unvalidated.
- **Code:** https://github.com/b4yuan/ecog2vec (MIT license)

---

## Reusable Ideas

1. **WAV2VEC-style pretraining on high-gamma envelopes:** Adapt WAV2VEC directly to multichannel ECoG envelopes. Keep standard preprocessing (bandpass, envelope, CAR) upstream; WAV2VEC operates on those envelopes, not raw voltages. Use causal convolutions throughout to respect temporal causality.

2. **Pooled cross-patient SSL corpus:** Pretrain a single WAV2VEC model on unlabeled ECoG from all available patients (pooled OOD or inter-trial data). With ~20 intra-op patients, this aggregates roughly 20x more unlabeled neural data than any single patient, directly addressing the scale problem the paper identifies.

3. **Layer-swap transfer for variable electrode counts:** Replace only the patient-specific input projection layer when transferring WAV2VEC across patients; freeze all other layers initially, then unfreeze for fine-tuning. This is a low-cost adaptation strategy for patients with different uECOG array sizes/placements.

4. **Use inter-trial / task-silent ECoG as unlabeled SSL data:** In intra-op sessions, ECoG recorded between task trials (rest periods, instruction periods, electrode testing) can serve as the unlabeled corpus for SSL pretraining, analogous to OOD blocks here.

5. **Downsample to ~100 Hz for the latent space:** The architectural choice to bring latent representation rate down to ~100 Hz (matching phoneme production rate of 20-30 phones/sec) is well-motivated and may be appropriate for intra-op uECOG processing pipelines as well.

6. **s hyperparameter tuning:** The number of future steps s for the contrastive prediction objective materially affects both WAV2VEC loss and downstream WER. Tune this on a held-out patient or use cross-validation before committing to a value; the paper found s=8 optimal but this may differ with microelectrode signals.

7. **Context vectors as frozen feature extractor:** After SSL pretraining, the WAV2VEC context network can be used as a fixed feature extractor; only the lightweight downstream decoder (GRU encoder-decoder) needs to be trained per patient on labeled data. This decoupling is attractive for intra-op where per-patient compute time is limited.
