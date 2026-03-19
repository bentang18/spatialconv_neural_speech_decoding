# NeurIPT: Foundation Model for Neural Interfaces (NeurIPS 2025)

**Authors:** Zitao Fang, Chenxuan Li, Hongting Zhou, Shuyang Yu, Guodong Du, Ashwaq Qasem, Yang Lu, Jing Li, Junsong Zhang, Sim Kuan Goh

**Venue:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025)

**ArXiv:** 2510.16548v1

---

## Setup

- **Recording modality:** Scalp EEG (non-invasive), standard 10-20 electrode placement system, 20 bipolar channels in "double banana" montage for pretraining (e.g., FP1-F7, F7-T3, ..., P4-O2)
- **Pretraining data:** >2,000 hours of unlabeled public EEG data; the eight downstream datasets were explicitly excluded from pretraining
- **Downstream datasets (8 total):**

| Dataset | Task | Rate | Channels | Duration | Samples | Classes |
|---|---|---|---|---|---|---|
| MentalArithmetic | Mental Stress Detection | 500 Hz | 20 | 5 s | 1,707 | 2 |
| Mumtaz2016 | Mental Disorder Diagnosis | 256 Hz | 20 | 5 s | 6,963 | 2 |
| PhysioNetP300 | P300 | 2048 Hz | 20 | 2 s | 21,179 | 2 |
| Sleep-EDFx | Sleep Staging | 100 Hz | 2 | 30 s | 457,652 | 5 |
| SEED-V | Emotion Recognition | 1000 Hz | 20 | 1 s | 115,001 | 5 |
| BCIC-IV-2A | Motor Imagery | 250 Hz | 16 | 4 s | 5,184 | 4 |
| TUAB | Abnormal Detection | 250 Hz | 20 | 10 s | 409,455 | 2 |
| TUEV | Event Type Classification | 250 Hz | 20 | 5 s | 112,491 | 6 |

- **Paradigm:** Self-supervised pretraining followed by fine-tuning on downstream classification tasks

---

## Problem Statement

Existing EEG foundation models fail to handle three structural properties of EEG that are critical for generalization across recordings:

1. **Electrode configurations are treated as interchangeable:** Prior models use learned 1D or 2D positional encodings that ignore the physical 3D arrangement of electrodes on the scalp, losing spatial relationships and making cross-montage transfer fragile.

2. **Random masking in pretraining is too easy:** BERT-style masking of contiguous segments allows the model to solve the task by local interpolation between unmasked neighbors, producing representations that fail on downstream classification. The model learns to interpolate rather than to understand global signal structure.

3. **Single feedforward networks cannot handle EEG's temporal heterogeneity:** EEG signals span slow-wave sleep oscillations to rapid epileptic spikes, with spectral profiles that vary substantially across datasets (Figure 1). A single FFN sub-network is insufficient to capture this range.

4. **Downstream fine-tuning ignores brain regional structure:** Global or mean pooling discards the lobe-specific functional specialization that discriminates BCI tasks (e.g., contralateral motor cortex for hand vs. foot imagery).

---

## Method

### Architecture Overview

NeurIPT is a Crossformer-based encoder-decoder with four main innovations stacked together: 3D Electrode Embedding (spatial), AAMP pretraining (temporal self-supervised), Progressive Mixture-of-Experts (PMoE, temporal), and Intra-Inter Lobe Pooling (IILP, spatial fine-tuning head).

### 1. 3D Electrode Embedding (3D PE)

Each electrode channel at physical coordinate $(x_d, y_d, z_d)$ (from the 10-20 system) receives a sinusoidal positional encoding that encodes each axis separately:

$$PE_d^{(s)} = \text{Concat}(PE_x(x_d),\ PE_y(y_d),\ PE_z(z_d))$$

where each axis uses the standard sinusoidal formulation $PE_\alpha(\text{pos})_{2j} = \sin(\text{pos} \cdot 10000^{-2j/d_\alpha})$, with $d_\alpha = d_\text{model}/3$.

Each time-point embedding then receives both a temporal positional encoding and this spatial electrode encoding:

$$\mathbf{s}_{t,d}^{(i)} = \mathbf{E}\mathbf{x}_{t,d}^{(i)} + PE^{(t)} + PE^{(s)}$$

**Key property for transfer:** Because electrode positions are described by their physical 3D coordinates rather than by a learned index, a new electrode montage can be embedded directly without retraining. There is no padding, convolution, or interpolation step needed for variable channel counts — the spatial context is injected at the embedding level.

**Ablation result (Table 11):** 3D PE outperforms trigonometric 1D, learnable 1D, and learnable 2D embeddings consistently. On SEED-V, 3D PE reaches 39.34% balanced accuracy vs. 35.03/35.52/37.40% for the alternatives. On MentalArithmetic, 3D PE reaches 75.69% vs. 74.65/71.53/71.52%.

### 2. Amplitude-Aware Masked Pretraining (AAMP)

Rather than randomly masking contiguous segments (BERT-style), AAMP selects the mask region centered around a randomly sampled amplitude percentile of each channel. Specifically, for channel $d$, a percentile $\xi_d^{(i)} \sim U(0,1)$ is sampled and the mask covers the $T \cdot \mathcal{P}$ time points centered on the corresponding amplitude rank in the sorted signal.

This forces the model to reconstruct high-amplitude events (sharp peaks, bursts, transients) rather than smooth low-amplitude regions, which are easier to interpolate. The pretraining loss is the $\ell_p$-norm reconstruction loss over all channels.

**Result (Table 8):** AAMP improves SVM downstream balanced accuracy from 0.6320 (random masking) to 0.6845 on the TUEV dataset (+5.25%). Optimal masking ratio is 50% (Table 9).

### 3. Progressive Mixture-of-Experts (PMoE)

The transformer encoder has $N$ blocks. At block $l$, the number of specialized expert FFNs is $E_l$, chosen according to a progressive schedule (e.g., $[0, 2, 4, 4, 4, 6]$ — deeper layers have more experts). A gating network routes each token to top-$k$ experts via TopKSoftmax. A shared expert $\text{FFN}_\text{shared}$ is always active and captures general patterns, while the routed experts capture specialized temporal dynamics.

$$\text{PMoE}^{(l)}(\hat{\mathbf{Z}}^l) = \sum_{e=1}^{E_l} g_e^l \odot Y_e^l + \text{FFN}_\text{shared}^{(l)}(\hat{\mathbf{Z}}^l)$$

An auxiliary load-balancing loss $\mathcal{L}_\text{aux}$ encourages uniform expert utilization.

**Ablation (Table 3):** The progressive schedule outperforms no-MoE, uniform experts (same count per layer), and shrinking experts across 5 of 6 datasets. The BCIC-IV-2A result (44.01 vs. 44.62 for shrinking) is the only near-tie.

### 4. Intra-Inter Lobe Pooling (IILP)

Used only during fine-tuning. Channels are partitioned into $n$ anatomical brain lobes (frontal, occipital, temporal-L, temporal-R, parietal based on 10-20 positions). For each encoder block $l$:

- **Intra-lobe pooling:** Average all channel embeddings within each lobe to get lobe-level embedding $V_k^l$.
- **Inter-lobe concatenation:** Concatenate all lobe embeddings to form $v^l = \text{concat}(V_1^l, \ldots, V_n^l)$.

Final representation concatenates $v^l$ across all $L$ encoder blocks: $v = \text{concat}(v^1, \ldots, v^L)$, then passed to an MLP classifier.

**Ablation (Table 5):** IILP (86.46% on MentalArithmetic, 98.03% on Mumtaz2016, 55.04% on BCIC-IV-2A) consistently outperforms mean pooling, hemisphere-based, coronal, and sagittal groupings.

---

## Key Results

All results are from fine-tuned NeurIPT vs. prior EEG foundation models (BIOT, LaBraM, CBraMod, EEGPT, BENDR, NeuroLM).

**Table 2 highlights (NeurIPT vs. best prior baseline, balanced accuracy):**

| Dataset | NeurIPT | Best Prior | Delta |
|---|---|---|---|
| MentalArithmetic | 86.46% | 72.56% (CBraMod) | +13.90 |
| Mumtaz2016 | 98.03% | 95.60% (CBraMod) | +2.43 |
| PhysioNetP300 | 67.31% | 65.02% (EEGPT) | +2.29 |
| Sleep-EDFx | 70.47% | 69.17% (EEGPT) | +1.30 |
| SEED-V | 41.04% | 40.91% (CBraMod) | +0.13 |
| BCIC-IV-2A | 55.04% | 51.38% (CBraMod) | +3.66 |
| TUAB | 82.93% | 82.89% (CBraMod) | +0.04 |
| TUEV | 67.61% | 66.71% (CBraMod) | +0.90 |

NeurIPT achieves state-of-the-art on 8/8 datasets on balanced accuracy. The two exceptions where it does not lead are Cohen's Kappa and AUROC on TUAB (where CBraMod holds a slight edge at 92.58 kappa vs. NeurIPT's 90.40).

**Low-resource performance (Table 12):** With only 10% of fine-tuning data, NeurIPT retains 80-93% of full-data performance. With 1% of data, it retains 64-81%, demonstrating strong few-shot capability from pretraining.

**3D PE ablation (Table 11):** Consistent ~2-4 percentage point improvement over 1D/2D alternatives across all four tested datasets.

**Individual component ablation (Table 6):** Each component adds value. The full model (3D PE + PMoE + IILP) with pretraining achieves the best results. IILP is especially important for TUEV (epilepsy/abnormality) and Mumtaz2016 (depression), which require regional brain signal differentiation. 3D PE is most critical for BCIC-IV-2A (motor imagery, low channel count, spatial sensitivity).

---

## Cross-Patient Relevance

**What the problem has in common with intra-op uECoG speech decoding (~20 patients, variable electrode placement, ~20 min each):**

The core challenge NeurIPT addresses — generalizing a model across subjects with different electrode configurations and limited labeled data per subject — maps directly onto the intra-op setting. Specifically:

- **Variable electrode placement:** Each uECoG patient has a unique grid placement determined by clinical neurosurgical constraints. NeurIPT's 3D coordinate embedding approach offers a principled way to represent electrode identity via physical location rather than channel index. If physical coordinates of the uECoG electrodes are known (as they typically are from intraoperative imaging or anatomical localization), this approach allows the model to generalize across patients without requiring matched channel counts or consistent montages.

- **Small per-patient dataset:** ~20 minutes of intraoperative recording is a severely data-limited regime. NeurIPT's low-resource results (64-81% balanced accuracy with 1% of data) show that a pretrained backbone substantially reduces the labeled data burden during fine-tuning.

- **Cross-patient pretraining:** NeurIPT is pretrained entirely without labeled data. An analogous approach for uECoG speech decoding could pretrain on unlabeled or heterogeneous labeled data from multiple patients and transfer to new patients via fine-tuning, with electrode coordinates providing the spatial anchor.

- **Amplitude-driven masking:** Speech-relevant uECoG signals are highly non-stationary with bursts of high-gamma activity during articulation. AAMP's focus on masking amplitude-salient regions (rather than random intervals) is well-suited to this signal structure, where the discriminative content is concentrated in high-amplitude epochs.

**What requires translation:**

- IILP's lobe-based pooling assumes scalp EEG electrode coverage with standard lobe assignments. For uECoG, which typically covers only a portion of cortex (often perisylvian/temporal/frontal for speech), the pooling hierarchy would need to be redesigned based on functional anatomy (e.g., motor cortex, auditory cortex, inferior frontal gyrus) rather than scalp lobe regions. The concept of hierarchical regional pooling still applies, but the grouping logic must be replaced.

- The 3D PE uses the international 10-20 system coordinates. For uECoG, electrode 3D coordinates from the patient's own MRI/CT would be used instead. The formulation is directly compatible — the sinusoidal encoding is agnostic to the coordinate source.

---

## Limitations

- **EEG, not ECoG:** NeurIPT is developed and validated exclusively on scalp EEG. ECoG and uECoG have fundamentally different signal characteristics: higher frequency content (high gamma, 70-200 Hz), much higher spatial resolution (~400 µm inter-electrode spacing for uECoG vs. ~6 cm for scalp EEG), lower noise from skull/scalp, and different electrode count ranges (typically 16-128 for intraoperative grids vs. 20-256 for scalp EEG). The pretraining corpus (>2,000 hours of scalp EEG) would not transfer directly to ECoG without domain mismatch.

- **Pretraining corpus is scalp EEG only:** No ECoG, sEEG, or other intracranial modalities are included. Direct application of the pretrained NeurIPT weights to ECoG would require domain adaptation or pretraining from scratch on ECoG data.

- **Lobe-based IILP does not apply to partial-coverage grids:** The fine-tuning head assumes whole-scalp electrode coverage across frontal, occipital, temporal, and parietal lobes. Intraoperative uECoG grids cover only a localized cortical patch; the inter-lobe pooling concept breaks down or reduces to a single lobe.

- **Standard electrode placement assumed for 3D PE:** The 10-20 system provides well-defined 3D coordinates. For uECoG, coordinates must be derived from individual patient anatomy and may carry registration error. The encoding scheme is compatible but requires patient-specific coordinate input.

- **Downstream tasks are all classification:** NeurIPT's fine-tuning paradigm assumes a discrete classification head. Speech decoding involves continuous articulatory/phonemic sequences with temporal ordering, which may require sequence-to-sequence decoders (e.g., CTC, RNN-T, autoregressive LM head) beyond the MLP classifier used here.

- **No explicit handling of cross-session or intraoperative signal drift:** Intraoperative recordings involve anesthetic effects and rapidly evolving neural states. NeurIPT does not address distribution shift within a recording session.

- **Evaluation is cross-dataset, not cross-patient within dataset:** Most downstream datasets are evaluated as a whole; the paper does not report leave-one-subject-out or cross-patient generalization results explicitly, making it harder to directly benchmark patient-level transfer performance.

---

## Reusable Ideas

### 1. 3D Electrode Coordinate Embedding (highest priority)

**Directly applicable.** The sinusoidal 3D positional encoding (Eq. 1-2) is the most immediately reusable idea. For uECoG:

- Obtain 3D coordinates of each electrode from intraoperative imaging or anatomical co-registration with the patient's MRI.
- Encode each electrode's $(x, y, z)$ position using the same sinusoidal scheme: encode each axis independently into $d_\text{model}/3$ dimensions, concatenate.
- Add this to the channel embedding at the input layer alongside temporal positional encoding.

This eliminates the need for consistent channel count across patients. A model trained on patient A's 64-channel uECoG grid can be fine-tuned or applied directly to patient B's differently-placed 48-channel grid, because each electrode is identified by where it is physically located, not by its index. This is the most actionable contribution for cross-patient speech decoding.

**Implementation note:** The paper uses sinusoidal basis with $10000^{-2j/d_\alpha}$ decay (same as Vaswani et al. 2017), applied per spatial axis. The coordinate values are raw anatomical positions (not normalized to unit sphere), though normalization may be beneficial for uECoG where coordinates span MNI space.

### 2. Amplitude-Aware Masked Pretraining (AAMP)

**Applicable with adaptation.** The core insight — mask regions of high signal amplitude to prevent the model from solving pretraining by local interpolation — is directly relevant for uECoG pretraining. High-gamma bursts during speech production are the amplitude-salient events that carry the most decodable information. Masking these forces the model to learn their structure globally rather than interpolating around them.

Implementation: sort time points by amplitude per channel, sample a percentile, mask a window centered there. The 50% masking ratio found optimal for scalp EEG may need tuning for uECoG given different noise floors and signal dynamics.

### 3. Progressive Mixture-of-Experts (PMoE)

**Applicable in principle; resource-dependent.** If training a larger uECoG model from scratch or fine-tuning with sufficient data across patients, PMoE allows the model to allocate specialized sub-networks for different temporal dynamics (e.g., one expert may specialize in low-frequency LFP, another in high-gamma bursts). The progressive schedule (more experts at deeper layers) is a simple hyperparameter choice that improves over uniform expert allocation.

For the small-data intraoperative setting, this is likely secondary — MoE benefits scale with data diversity and volume. Consider this only if pooling data across many patients.

### 4. Hierarchical Regional Pooling (IILP concept)

**Applicable with redesign.** The principle of grouping electrodes by functional brain region and pooling within-group before concatenating across groups is sound for uECoG. Replace scalp lobe assignments with uECoG-relevant functional parcels:

- Motor cortex (precentral gyrus) electrodes
- Premotor / supplementary motor area
- Auditory cortex (superior temporal gyrus / Heschl's gyrus)
- Inferior frontal gyrus (Broca's area)
- Somatosensory cortex (postcentral gyrus)

Grouping can be done by MNI atlas labels projected onto each patient's electrode coordinates. This retains the intra-group suppression of redundancy and inter-group discriminative concatenation while being anatomically grounded for speech.

### 5. Low-Resource Pretraining Strategy

**Directly relevant.** NeurIPT's low-resource results show that even with 1% of labeled fine-tuning data, pretrained representations yield 64-81% of full-data performance. For intra-op uECoG with ~20 min of data, pretraining on a large unlabeled corpus (e.g., other patients' unlabeled recordings, or the labeled data from other patients in an unsupervised fashion) before patient-specific fine-tuning is the right paradigm to adopt.

### 6. SwiGLU Activation Function

**Minor but easy to implement.** Table 7 shows SwiGLU is the most consistent activation across all datasets, outperforming ReLU and GELU in average rank. This is a low-cost substitution in any transformer-based uECoG model.
