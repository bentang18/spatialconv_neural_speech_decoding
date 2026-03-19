# Brain Foundation Models: A Survey on Advancements in Neural Signal Processing and Brain Discovery

**Citation:** Zhou, X., Liu, C., Chen, Z., Wang, K., Ding, Y., Jia, Z., & Wen, Q. (2025). Brain Foundation Models: A Survey on Advancements in Neural Signal Processing and Brain Discovery. arXiv:2503.00580v2 [cs.LG], July 19, 2025.

---

## Scope

**Modalities covered:**
- EEG (primary focus — scalp, clinical, BCI)
- fMRI (structural and functional)
- Intracranial / intracortical signals (ECoG, sEEG) addressed under "Brant" [14,24] and BrainBERT [20]
- MEG, fNIRS, EMG, EOG, ECG mentioned as secondary or multimodal context signals
- Structural MRI, DTI mentioned as future directions

**Task domains covered:**
- Brain decoding: motor imagery, emotion recognition, sleep staging, seizure/epilepsy detection, speech BCI, cognitive workload, SSVEP, P300
- Cross-scenario generalization (same task, different subjects/settings)
- Cross-task generalization (same model, different BCI paradigms)
- Brain simulation and discovery: digital twin brain construction, disease mechanism exploration, cognitive process simulation

**Models surveyed (Table II, 25 models):**
BrainLM, LaBraM, NeuroLM, Brant, BrainSegFounder, McTSK, AnatCL, BRAINBERT, NeuroVNN, Brain-JEPA, FoME, FM-BIM, BrainWave, TF-C, Neuro-GPT, BrainMAE, EEGFormer, CEReB rO, CBraMod, EEGPT-Yue, TGBD, MBrain, EEGPT, Brant-X, FM-APP

---

## Key Frameworks

### 1. Three-Category BFM Taxonomy (by training/deployment strategy)
- **Pretrained-only models:** No fine-tuning required; rely on in-context learning for zero-shot generalization (e.g., NeuroLM). Suited for cross-scenario tasks.
- **Pretrained + fine-tuned models:** SSL pretraining followed by supervised fine-tuning on labeled downstream tasks. Most common approach; handles cross-task BCI applications and disease diagnosis.
- **Pretrained + interpretability models:** Pretraining combined with perturbation analysis or attention-based probing to build digital twin brains for brain discovery.

### 2. Application Taxonomy (Figure 3)
- **(A) Brain Decoding**
  - (A.a) Cross Scenarios: same task type, different subjects/settings/languages (e.g., speech BCI with different vocabularies)
  - (A.b) Cross Tasks: entirely different paradigms from one model (e.g., emotion + sleep + fatigue with one BFM)
- **(B) Brain Simulation and Discovery**
  - Digital twin brain construction at microscopic (Hodgkin-Huxley), mesoscopic (Wilson-Cowan), and macroscopic (whole-brain network) levels

### 3. BFMs vs. Traditional Foundation Models (Table I)
Key distinguishing axes: high-noise non-stationary data, heterogeneous channel configurations, neuroscience-specific learning objectives (universal neural expression, spatial/temporal modeling), ethical constraints (neural data privacy, biological interpretability, clinical safety).

### 4. SSL Objective Taxonomy
Two primary categories formalized as a unified loss:
```
L_SSL = alpha * L_reconstruction(x_i, x_hat_i) + beta * L_contrastive(f(x_i), f(x_j))
```
- **Reconstruction-based:** masked prediction (temporal or spatial) and autoregression
- **Contrastive learning:** intra-brain modality (within or across brain recording types) and brain-heterogeneous modality (brain signal vs. text/image)

---

## Relevant Methods

### Self-Supervised Learning Strategies

**Masked prediction (MAE-style):**
- Temporal masking: mask time windows, reconstruct from surrounding context. Well-suited for EEG due to high temporal resolution and temporal continuity.
- Spatial masking: mask channels/brain regions, reconstruct from remaining channels.
- Spatiotemporal joint masking: Brain-JEPA [14] introduces gradient-position-based spatiotemporal joint masking.
- Time-frequency masking: BrainBERT [20] masks EEG spectrograms (STFT or Superlet transform) and reconstructs missing spectral features.
- Survey recommendation: masked prediction is more suitable for fMRI (high spatial correlation); autoregression is more suitable for EEG (strong temporal continuity).

**Autoregressive pretraining:**
- BrainLM [21]: Transformer-based autoregressive framework for fMRI; sequentially predicts future time steps from past neural activity.
- NeuroLM [19]: Multi-channel causal autoregressive mechanism; integrates information across EEG channels jointly; also uses EEG-text cross-modal contrastive alignment.
- EEGPT-Yue [39]: EEG generalist foundation model via autoregressive pretraining (1.1B parameters).

**Contrastive learning:**
- Intra-brain: MBrain [23] constructs contrastive samples from different time segments across channels; EEGPT [18] uses temporal jittering, frequency shifts, and channel permutation for augmentation.
- Cross-modal (brain-heterogeneous): Brant-X [24] introduces patch-level and sequence-level EEG alignment with ECG, EOG, EMG; Kong et al. [25] aligns fMRI with visual image features using CLIP-style contrastive loss; NeuroLM aligns EEG with text representations.

### Architectures

**Transformer backbone (dominant):**
Most BFMs use standard Transformer encoder or decoder with modifications for brain data. Parallelism enables multi-channel modeling.

**Positional encoding strategies:**
- Fixed sinusoidal (temporal dimension): more stable, avoids overfitting temporal patterns that are relatively consistent across individuals.
- Learnable (spatial/channel dimension): accommodates variable channel configurations and individual-specific topographies.
- Rotation-based (RoPE-style): emerging approach balancing fixed and learnable; suitable for long-range temporal dependencies.
- Brain-JEPA: brain gradient positioning for region encoding + sinusoidal temporal encoding.
- Brant [15]: learnable positional matrix W_pos in R^(L x D).
- LaBraM [16]: separate learnable temporal W_temp and spatial W_spat encoding matrices.

**Signal segmentation (patch-based tokenization):**
- Standard approach: partition raw signal X in R^(C x T) into fixed-window patches x_{c,k} in R^w per channel using sliding window with step s.
- CBraMod [17] uses channel-intersection (subset of electrodes common across datasets); EEGPT [18] uses channel-union (superset, with zero-padding for missing channels).

**Brant [15] — the primary intracranial model:**
- 505M parameter foundation model for intracranial neural signals (sEEG/ECoG).
- Pretraining on 1TB+ of intracranial recordings.
- Key feature: designed specifically for high-channel-count, non-stationary intracranial LFP/broadband signals.
- Uses learnable positional encoding; NeurIPS 2023.

**BrainBERT [20] — intracranial SSL model:**
- Self-supervised learning for intracranial (ECoG/sEEG) recordings.
- Time-frequency masking strategy using short-time Fourier or Superlet transforms.
- arXiv:2302.14367, 2023.

**Key large-scale models (size reference):**
- NeuroLM: 1.7B parameters (largest)
- EEGPT-Yue: 1.1B
- Brant-X: 1B
- BrainLM: 650M
- FoME: 745M
- Brant: 505M
- LaBraM: 369M

### Cross-Subject / Cross-Patient Approaches

The survey identifies cross-scenario generalization as a primary motivation for BFMs. Key mechanisms:

1. **Heterogeneous channel handling:** Channel-intersection (common electrodes only) vs. channel-union (all electrodes, missing ones zero-padded). The survey flags both as suboptimal — intersection loses information, union inflates computation. No fully satisfactory solution is identified.

2. **Large-scale diverse pretraining:** Models pretrained across thousands of subjects (e.g., LaBraM uses 2500 hours of EEG from diverse datasets) develop subject-invariant representations. Cross-subject generalization emerges implicitly.

3. **Subject-agnostic tokenization:** Treating each channel independently as a patch (per-channel segmentation in Eq. 1) allows the model to handle variable numbers of channels without architectural changes.

4. **Federated learning (mentioned as future direction):** For privacy-constrained multi-site data; not yet widely implemented in current BFMs.

5. **Adapter / LoRA fine-tuning (identified as gap):** Parameter-efficient fine-tuning methods validated in NLP have not yet been widely applied to BFMs — flagged as a high-priority future direction.

---

## Cross-Patient Relevance

Insights most relevant for cross-patient speech decoding from intra-op uECoG:

### Directly Applicable

**Brant [15] and BrainBERT [20] are the closest precedents.** Both target intracranial recordings (sEEG and ECoG respectively) with SSL pretraining. Brant [24] (Brant-X) extends this with cross-modal contrastive alignment. For uECoG specifically:
- The channel-heterogeneity problem (variable electrode counts and layouts across patients) is the central unsolved problem in the survey. The patch-based per-channel tokenization with learnable spatial encodings is the current best practice.
- Autoregressive or masked-prediction pretraining on larger intracranial corpora before patient-specific fine-tuning mirrors the recommended "pretrained + fine-tuned" paradigm.

**Speech BCI is explicitly called out** (Section III.A.1) as a key application for pretraining-only cross-scenario BFMs: "BFMs pretrained on diverse speech-related brain data captured from different subjects, speech tasks, and sensory modalities can decode speech intentions more effectively across multiple scenarios. Whether the data comes from speech imagery, silent speech, or actual speech, the pretrained model can handle this variability."

**Fixed temporal / learnable spatial positional encoding** is the recommended design for intra-op ECoG: temporal patterns are relatively stable across patients (supporting fixed sinusoidal), while spatial relationships (electrode placement, cortical mapping) vary per patient (supporting learnable spatial embeddings). This aligns with the survey's "Remark" on page 9.

**Fine-tuning with Adapter or LoRA modules** is identified as an underexplored but promising approach for adapting a pretrained BFM to new patients without full retraining. Particularly relevant for the intra-op setting where data is scarce and session-specific.

### Key Gaps the Survey Identifies (Relevant to Our Problem)

1. **No BFM specifically targets uECoG or intra-operative recordings.** All intracranial work (Brant, BrainBERT) uses chronic implant data (sEEG, ECoG from epilepsy monitoring). The intra-op uECoG context — temporary, high-density, speech-only — is unaddressed.

2. **Cross-patient transfer is addressed superficially.** The survey focuses on cross-scenario (within-modality) and cross-task (different paradigms) generalization, but does not systematically address cross-patient transfer learning with few-shot adaptation, which is the core challenge in surgical BCI.

3. **Subject variability and unstable cross-task distribution** are noted as the reason BFMs remain "more reliant on fine-tuning to achieve effective transfer" than NLP/CV models. This directly predicts that even a well-pretrained BFM will require patient-specific fine-tuning.

4. **Biological priors not yet integrated.** Loss functions incorporating known brain oscillations (alpha, beta, gamma bands) or cortical gradients are suggested as future work. For speech cortex (STG/IFG), incorporating spectrotemporal speech priors (high-gamma envelope, phoneme-locked responses) into the SSL objective could improve transfer.

5. **No standardized benchmark for cross-patient generalization.** The survey calls for community-adopted benchmarks; none currently exist for intracranial speech decoding specifically.

---

## Key References to Follow Up

Papers cited in the survey that are particularly relevant and may not be in our collection:

| Ref | Citation | Relevance |
|-----|----------|-----------|
| [15] | D. Zhang et al., "Brant: Foundation model for intracranial neural signal," NeurIPS vol. 36, pp. 26304-26321, 2023. | Primary intracranial BFM; sEEG/ECoG pretraining at scale |
| [24] | D. Zhang et al., "Brant: Foundation model for intracranial neural signal," NeurIPS vol. 36, 2024. (Brant-X extension) | Cross-modal contrastive alignment for intracranial signals |
| [20] | C. Wang et al., "BrainBERT: Self-supervised representation learning for intracranial recordings," arXiv:2302.14367, 2023. | SSL pretraining on ECoG/sEEG with time-frequency masking |
| [14] | Z. Dong et al., "Brain-JEPA: Brain dynamics foundation model with gradient positioning and spatiotemporal masking," NeurIPS vol. 37, 2025. | Spatiotemporal joint masking; positional encoding design |
| [16] | W. Jiang, L. Zhao, B. Lu, "LaBraM: Large brain model for learning generic representations with tremendous EEG data in BCI," ICLR 2024. | Separate learnable temporal and spatial positional encodings |
| [19] | W.-B. Jiang et al., "NeuroLM: A universal multi-task foundation model for bridging the gap between language and EEG signals," arXiv:2409.00101, 2024. | EEG-text cross-modal alignment; multi-task instruction tuning |
| [18] | G. Wang et al., "EEGPT: Pretrained transformer for universal and reliable representation of EEG signals," NeurIPS vol. 37, pp. 39249-39280, 2025. | Contrastive learning with augmentation (jitter, frequency shift, channel permutation) |
| [25] | X. Kong et al., "Toward generalizing visual brain decoding to unseen subjects," ICLR 2025. | CLIP-style cross-subject fMRI generalization; directly addresses cross-patient transfer |
| [21] | J. O. Caro et al., "BrainLM: A foundation model for brain activity recordings," ICLR 2024. | 650M autoregressive model for fMRI; cross-scenario generalization |
| [39] | T. Yue et al., "EEGPT: Unleashing the potential of EEG generalist foundation model by autoregressive pre-training," arXiv:2410.19779, 2024. | 1.1B autoregressive EEG generalist |
| [23] | D. Cai et al., "MBrain: A multi-channel self-supervised learning framework for brain signals," KDD 2023. | Contrastive learning from cross-channel temporal segments |
| [34] | X. Zhang et al., "Self-supervised contrastive pre-training for time series via time-frequency consistency," NeurIPS vol. 35, 2022. | TF-C: time-frequency contrastive SSL applicable to neural signals |

---

## Reusable Ideas

### Frameworks to Adopt

1. **Unified SSL loss formulation (Eq. 2):** The alpha/beta weighting between reconstruction and contrastive objectives provides a clean framework for ablating which SSL strategy works best for uECoG. Start with alpha=1, beta=0 (MAE-only), ablate to alpha=0, beta=1 (contrastive-only), then joint.

2. **Per-channel patch tokenization with variable-channel support:** Treating each electrode's signal independently as patches (Eq. 1: x_{c,k} in R^w) is the cleanest way to handle variable electrode counts across patients. Combine with learnable spatial embeddings conditioned on anatomical coordinates if available.

3. **Fixed sinusoidal temporal + learnable spatial positional encoding:** The survey's explicit recommendation for BFMs: fixed sine/cosine for time (robust, reduces overfitting), learnable for spatial/channel dimension (accommodates patient-specific topographies). Directly applicable to uECoG where electrode layout varies per craniotomy.

4. **Two-stage fine-tuning pipeline:** (1) SSL pretraining on all available intracranial speech data across patients; (2) per-patient fine-tuning with lightweight adapter modules (LoRA or bottleneck adapters). The survey flags this as the highest-priority underexplored direction in BFMs.

5. **Cross-modal contrastive alignment (Brant-X style):** If audio, articulator, or phoneme labels are available, contrastive alignment between uECoG representations and speech acoustic features (following the CLIP paradigm) can enforce semantically meaningful cross-patient representations. The InfoNCE loss (Eq. 5) with cosine similarity is the standard formulation.

6. **Spectrotemporal masking (BrainBERT style):** Rather than masking raw signal patches, masking in the time-frequency domain (STFT or Superlet spectrogram) and reconstructing missing spectral features may be more appropriate for high-gamma speech signals in ECoG. This respects the spectrotemporal structure of speech-related neural activity.

### Evaluation Approaches to Adopt

1. **Cross-scenario vs. cross-task distinction:** When reporting results, explicitly distinguish whether a model is evaluated (a) on a held-out subject performing the same task (cross-subject/cross-scenario), vs. (b) on a different task entirely (cross-task). These require different evaluation protocols and test different aspects of generalization.

2. **Pretraining-only vs. pretrained+fine-tuned comparison:** Report both zero-shot/few-shot performance from the pretrained model and fine-tuned performance. The gap between these quantifies how much patient-specific data is needed.

3. **Table II format (model capability matrix):** The survey's Table II format — listing models by whether they support pretraining, fine-tuning, cross-scenarios, cross-tasks, and multi-modal — is a useful taxonomy for organizing and comparing our own model design choices.

4. **Standardized metrics:** ACC and F1 for classification; MSE and Pearson r for regression/reconstruction tasks. Reporting both ACC and AUROC for clinical detection tasks (following Table III).

5. **Benchmark on public EEG datasets for comparison:** Even if our primary target is uECoG, reporting results on public EEG benchmarks (TUAB for abnormal detection, TUEV for event classification, SEED for emotion) following the Table III format enables comparison with the broader BFM literature.

---

## Limitations and Gaps Noted by the Survey

- No BFM has been developed specifically for intra-operative or acutely implanted electrode recordings.
- ECoG and sEEG data are used in Brant and BrainBERT but chronic epilepsy monitoring contexts only; no surgery/mapping context.
- Current BFMs are "more AI adaptations than neuro-tailored models" — biological constraints (functional connectivity, known oscillations) are rarely incorporated into training objectives.
- Parameter-efficient fine-tuning (Adapter, LoRA) has not been applied to BFMs despite its demonstrated value in NLP/CV — high-priority gap.
- Federated learning for multi-site privacy-preserving pretraining is mentioned as a future direction but not yet implemented in any surveyed model.
- Fixed time-window segmentation does not adapt to the variable timing of neural events (e.g., variable phoneme durations in speech decoding).
