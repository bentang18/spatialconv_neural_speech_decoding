# Cross-Patient Speech Decoding from Intra-Operative uECOG: Research Synthesis

**Ben Tang | Greg Cogan Lab, Duke University | March 2026**
**Collaborator: Zac Spalding**

Synthesizes 19 papers (16 archived) on cross-patient speech decoding from intra-operative micro-ECoG. Goal: build decoders generalizing across ~20 patients (128/256-ch uECOG, ~20 min each, left sensorimotor cortex). See `implementation_plan.md` for the concrete design derived from this synthesis.

---

## 1. Landscape

### 1.1 Paper Table

| Paper | Alignment | Modality | Data | Target | Transfer | Key Result |
|---|---|---|---|---|---|---|
| Spalding 2025 | PCA + CCA | uECOG | ~8 min/pt | 9 phonemes | Cross-patient | 0.31 bal. acc. (S4/S7 weakest) |
| Boccato 2026 | Affine (end-to-end) | Utah | Hours-days | Phonemes→sentences | Cross-patient+session | PER 16.1% (T12); backbone ~100M params |
| Levin 2026 | Affine + softsign | Utah | Hours-days | Phonemes→sentences | Cross-patient | Transfer helps <200 sent; 30% source replay |
| Singh 2025 | Conv1D + shared LSTM | sEEG | ~1 hr/pt | Phoneme sequences | Cross-patient | Group PER 0.49 vs single 0.57; 25 pts; LR/optimizer not reported |
| MIBRAIN 2025 | Region prototypes + MAE | sEEG | Days | 23 Mandarin consonants | Cross-patient (LOO) | Above-chance zero-shot; scales N≥6 |
| BIT 2026 | Read-in + transformer + audio-LLM | Utah | 367 hrs pretrain | Phonemes→sentences | Cross-subject SSL | WER 5.10%; SSL = supervised same-subject (Table 9); SSL advantage is cross-subject only; N=2 |
| Wu 2025 | TCA articulatory features | HD-ECoG | ~hours | Articulatory reconstruction | Cross-speaker | PCC 0.80/0.75/0.76 |
| wav2vec ECoG 2024 | Contrastive SSL on ECoG | ECoG | ~1 hr/pt | Words/sentences | Cross-patient (1 pair) | 60% WER reduction; 14.6% cross-pt gain |
| Dual Pathway 2025 | SSL + per-pt adaptors | ECoG | ~20 min/pt | Speech reconstruction | Within-patient | WER 18.9%, R²=0.824 |
| Chen 2024 | None (per-patient) | ECoG sub | 400 trials/pt | 18 speech params | None | PCC 0.806; 3D ResNet >> LSTM; N=48 |
| Chen 2025 (SwinTW) | MNI coordinate PE | ECoG sub + sEEG | 400 trials/pt | 18 speech params | Cross-patient | PCC 0.837 multi-pt = 0.831 individual; 0.765 LOO; N=52 |
| Mentzelopoulos 2024 (seegnificant) | RBF PE on MNI + spatial self-attn | sEEG | 3600+ trials | Reaction time | Cross-subject | R²=0.39 multi-subj vs 0.30 single; PE ΔR²=-0.02 (NS); per-subj heads ΔR²=-0.18; N=21 |
| Duraivel 2023 | None (baselines) | uECOG | ~15 min | 9 phonemes | None | 71% vowel, 50% phoneme |
| Duraivel 2025 | None (SVD-LDA) | SEEG/ECoG/uECOG | 52+3 pts | 9 phonemes (CVC/VCV pseudo-words) | None | 93% syllable, 38.3% phoneme; planning→execution dynamics |
| Qian 2025 | None (single patient) | HD-ECoG 256-ch (3mm) | ~9 hrs / 11 days | 394 Mandarin syllables | None | 71.2% syllable acc; 49.7 CPM real-time |
| Willett 2023 | Day-specific input layers | Utah | Days | Phonemes→sentences | Cross-session | WER 9.1% (50-word) |
| Metzger 2023 | None (single patient) | ECoG 253-ch | Weeks | Phonemes+synth+avatar | None | WER 25.5% (1024-word); 78 WPM |
| Littlejohn 2025 | None (single patient) | ECoG 253-ch | Weeks | Speech synthesis+text | None | PER 10.8%; 47.5 WPM streaming |
| Nason 2026 | Day-specific dense | Utah 64-ch | 2 years | Phonemes→sentences | Cross-session | WER 19.6% (125k-word) |

### 1.2 Key Observations

1. **Converging architecture:** Per-patient input layers → shared backbone (GRU) → CTC loss. Used by Willett, Metzger, Levin, Boccato, Nason, Singh, BIT. Metzger 2023 and Willett 2023 (both Nature, same issue) independently converged on Conv1D→BiGRU→CTC for phoneme-level speech decoding. Littlejohn 2025 evolves this to RNN-T (autoregressive, streaming-capable) while keeping the conv→GRU backbone. Qian 2025 uses stacked BiLSTM but with frame-averaged classification, not CTC — a different pattern.

2. **No cross-patient transfer tested on uECOG** beyond Spalding's CCA+SVM baseline. Every other cross-patient result uses sEEG or Utah arrays.

3. **Cross-patient transfer is harder than cross-session.** Levin shows permuted same-user data can outperform cross-brain data for 2/3 speech patients. BIT finds supervised cross-subject pretraining degrades performance (Appendix M, qualitative only — no numbers given, N=2 patients). **However, BIT Table 9 shows SSL ≈ supervised for same-subject at equal data size** — the SSL advantage is specifically cross-subject, not inherent. **Critical nuance:** BIT's failure is with only N=2 very different patients (T12: ALS 128ch, T15: ALS 256ch), and BIT never freezes the backbone during fine-tuning. Singh's supervised *joint training* with per-patient layers at N=25 DOES work (group PER 0.49 vs single 0.57). Our N≈20 setting with homogeneous uECOG arrays and per-patient layers is closer to Singh than BIT. Levin's 30% source replay during fine-tuning is directly adoptable for our Stage 2.

4. **Articulatory features generalize across speakers.** Wu 2025 shows TCA-extracted articulatory features have even contributions across 8 speakers and reconstruct from HD-ECoG with PCC 0.75-0.80. This supports articulatory-level alignment.

5. **Dense ECoG can decode large vocabularies (with sufficient data).** Qian 2025 achieves 71.2% on 394 Mandarin syllables from 256-ch HD-ECoG (3mm spacing — denser than clinical 10mm but coarser than uECOG's <2mm), with real-time sentence decoding. However, this required **~9 hours of neural data over 11 days** from a single epilepsy-monitoring patient, with 30–60 repetitions per syllable. Data scaling curve: 5 reps → 20.4%, 20 reps → 55.6%, full (30–60 reps) → 71.2%. Our intra-op setting provides ~1 min/patient — orders of magnitude less, though with far fewer classes (9 phonemes vs 394 syllables).

6. **Training protocol details are poorly reported across the field.** Singh doesn't report optimizer, LR, or batch size. Levin puts hyperparameters in an external spreadsheet. Boccato's backbone (~100M params, d=2048) is far larger than what our data supports. Our implementation choices (differential LR, specific regularization) are our own design decisions, not precedent-based.

7. **Spatial processing of ECoG grids outperforms non-spatial approaches, but architecture choice is data-regime-dependent.** Chen 2024: 3D ResNet (PCC 0.806) > 3D Swin Transformer (0.792) > LSTM (0.745) on N=48 patients with 8×8 grids. Chen 2025: SwinTW (0.825) > ResNet (0.804) on same data — but the advantage came from **coordinate-based tokenization** (enabling multi-patient training), not from being a transformer per se (Chen 2024's grid-based Swin *lost* to ResNet). Qian 2025: ViT (~0.63) significantly underperformed stacked LSTM (0.712) on 256-ch HD-ECoG (p<0.0001). CNNs' inductive biases (translation equivariance, local receptive fields) act as free regularization at small data scales — ViTs need ImageNet-scale data to overcome this (Dosovitskiy 2020). For our ~200 trials/patient on regular grids, 2D conv is better regularized than attention (shared spatial kernels vs O(N²) attention). Chen 2024's 18-parameter speech intermediate representation (pitch, formants, voicing, loudness) is compact and cross-patient-invariant by construction.

8. **Coordinate-based tokenization enables cross-patient ECoG speech decoding without per-patient decoder layers.** Chen 2025 (SwinTW) replaces grid-based positional encoding with MNI cortical coordinates (x,y,z) + ROI index. Each electrode becomes a token; positional bias B_{i,j} = MLP(positions, deltas) + ROI embedding dot product tells attention how electrodes relate anatomically. The model learns brain location→function (shared across patients) rather than electrode index→function (patient-specific). Multi-subject: PCC 0.837 vs 0.831 individual (NS, 15 male patients). Leave-group-out on unseen patients: PCC 0.765. sEEG-only: PCC 0.798 (N=9). **Caveats:** (a) "No per-patient layers" applies to the neural decoder only — Speech Encoder/Synthesizer are still per-patient (acoustic targets are speaker-dependent; our phoneme targets wouldn't need this); (b) generalization gap 0.837→0.765 is real, per-patient results not reported, some patients decoded poorly; (c) spatial max pooling compresses all electrode info to 1 vector per timestep — lossy for 128–256 ch uECOG with distinct spatial info at <2mm; (d) at uECOG spacing (1.3–1.7mm), coordinate differences between neighbors (~1–2mm MNI) are often smaller than MNI registration error (~5mm), so coordinates encode *global array position* on cortex but don't resolve within-array spatial relationships as well as 2D conv; (e) position→function mapping learned from N=52 patients — unclear if our N≈20 suffices.

---

## 2. Alignment Methods

Every approach chooses along three axes:

| Axis | Options | Literature consensus |
|---|---|---|
| **What's aligned?** | Input features / latent dynamics / output predictions | Input features (Boccato, Levin, Singh, BIT) |
| **How learned?** | Statistical (CCA) / supervised end-to-end / SSL / by construction (PE) | Supervised end-to-end with per-patient layers |
| **What's shared?** | Input: per-patient. Backbone: shared. Decoder: usually shared | Per-patient read-in + shared backbone + shared CTC head |

### Alignment Levels

Cross-patient alignment can operate at three representational levels, each with different invariance properties:

| Level | Where | Examples | Cross-patient invariance | Data cost |
|---|---|---|---|---|
| **Input features** | Before backbone | Per-patient read-in (Boccato, Levin, Singh), PCA+CCA (Spalding), coordinate PE (Chen 2025) | Learned — transforms patient-specific neural space to shared input space | Per-patient params or MNI coords |
| **Latent dynamics** | Inside backbone | SSL pretraining (BIT, wav2vec ECoG, MIBRAIN), joint supervised training | Implicit — backbone learns invariant representations through multi-patient exposure | Unlabeled neural data (hours) |
| **Output targets** | Loss function | Phonemes, articulatory features, speech parameters, HuBERT units, Wav2Vec2.0 embeddings | By construction — target space is defined to be cross-patient | Varies (0 to paired audio) |

Most methods operate at the **input** level (per-patient layers). The **output target** level is underexplored but potentially highest-leverage for our setting:

- **Phoneme labels** (standard): Cross-patient invariant by definition. But phonemes are abstract linguistic categories — motor cortex encodes articulatory gestures, not phoneme identity directly. Two phonemes with similar articulation (e.g., /b/ and /p/) have similar neural representations despite being distinct labels.
- **Articulatory features** (Chomsky-Halle): More fundamental than phonemes. Describe the physical motor plan (place, manner, voicing, height, backness) — what motor cortex directly controls. Cross-patient invariant by biology (same vocal tract, same muscles, same somatotopic organization in vSMC). Errors already track articulatory distance (Duraivel 2023). Available from linguistic databases at zero neural data cost. Lower-dimensional (5 features vs 9 classes). Wu 2025: articulatory features "more robustly encoded in vSMC than acoustic features and can be learned faster with limited neural data." Used as **auxiliary loss** alongside CTC, not replacement.
- **Speech parameters** (Chen 2024): 18 acoustic parameters (pitch, formants, voicing, loudness). Physically grounded but **speaker-dependent** — formant frequencies differ by vocal tract size (gender, anatomy). Cross-patient in meaning, not value.
- **HuBERT/Wav2Vec2.0 embeddings**: Capture phoneme-level structure from 960 hrs of speech. Cross-patient in embedding space but require paired audio and are validated only for auditory cortex (Dual Pathway), not motor cortex.

**Why articulatory is the strongest auxiliary target for us:** Our electrodes sit over left sensorimotor cortex, which organizes somatotopically by place of articulation (lips→tongue→larynx, Bouchard 2013, Metzger 2023). An auxiliary loss predicting articulatory features pushes the backbone toward representations that respect this biological organization — providing a cross-patient-invariant training signal at zero data cost.

### Method Comparison

| Method | Params/pt | Zero-shot? | Cross-pt tested? | Speech? | Verdict for us |
|---|---|---|---|---|---|
| **PCA+CCA** (Spalding) | ~0 | No | Yes | Yes | Baseline (0.31). Linear-only, pairwise, discards dynamics |
| **Learned linear read-in** (Boccato, Levin) | 8k–21k | No | Yes | Yes | **Phase 2 default.** Task-driven, handles variable channels |
| **Conv1D per-patient** (Singh) | ~98k+ | No | Yes | Yes | Heavy. Linear read-in achieves same purpose |
| **Conv2d on grid** (engineering) | 80–664 | No | No | No | **Phase 3.** Conv2d factorization matches rigid-array physics (uniform within-array, variable across arrays). Learned spatial deblurring + orientation adaptation. 15–25mm placement offsets kill diagonal scaling. Depth/channels/pool empirical (E13) |
| **Coordinate PE** (**Chen 2025 SwinTW, seegnificant**) | 0 | Yes | **Yes (Chen 2025, seegnificant)** | **Yes** | Phase 3. Chen 2025: PCC 0.765 LOO (N=52). **Caveat: seegnificant PE ΔR²=-0.02, p=0.73 (NOT significant)** — spatial self-attention does the heavy lifting, not PE |
| **Region prototypes** (MIBRAIN) | Per-region | Yes | Yes | No | Not applicable — our patients all cover same cortical patch |
| **SSL pretraining** (wav2vec, BIT) | Backbone | No | Partial | Yes | Phase 3. ~3 hrs borderline; masked > contrastive |
| **Articulatory TCA** (Wu) | Regression | No | Cross-speaker | Indirect | Phase 3 as auxiliary loss. Free alignment signal |

### Method Details

**PCA + CCA** (Spalding 2025): Reduce to PCA subspace, find CCA projections maximizing cross-patient correlation on condition-averaged phonemes. Simple (~50 lines numpy), works with 8 min/pt, but linear-only, discards trial dynamics, requires labels, pairwise (not universal).

**Affine transforms / learned input layers** (Boccato, Levin, Singh, Willett, Nason): x̄ = Wx + b per patient, trained end-to-end. Operates **before the backbone**, transforming signal values. The backbone receives normalized, patient-independent input. Handles: per-channel gain/impedance (diagonal of W), noise floor (bias b), variable channel count (zero-pad + learned weights), patient-specific cortical organization. Does NOT encode spatial relationships between electrodes — treats channel dimension as unordered vector. Boccato shows transforms are predominantly diagonal (per-channel gain, not spatial remapping) — **but this is Utah-specific**: Utah arrays have fixed stereotactic placement and orientation, so cross-patient variation really is dominated by gain. uECOG has variable surgical placement/orientation → cross-patient variation includes spatial remapping (which electrodes overlie which cortical regions), not just gain. For uECOG, the full affine matrix may need significant off-diagonal structure, or a spatial read-in (Conv2d) may be more appropriate. Levin shows transfer fails without per-session layers; some patients resist linear realignment. ~13k params for our setting (208→64); requires calibration data per patient — not zero-shot.

**Coordinate-based PE** (**Chen 2025 SwinTW**): Operates **inside the backbone**, modifying attention patterns rather than signal values. Encodes electrode 3D position via MNI coordinates + ROI index. Positional bias B_{i,j} = 2-layer MLP on 12 features (positions + deltas) + ROI embedding dot product, added to **scaled cosine similarity** attention (learnable per-head temperature). The backbone sees raw electrode signals but knows where each sits in the brain — learns brain location→function, not electrode index→function. Handles: which brain region each electrode samples, spatial relationships between electrodes (distance-dependent attention), variable channel count (variable-length token set). Does NOT handle: per-channel gain/impedance, noise floor, patient-specific deviations from MNI-average anatomy. Zero per-patient params, zero-shot capable. **Caveats:** all-electrode attention O(N²) per temporal window; spatial max pooling compresses electrode dim to 1; LOO is actually leave-group-out (5-fold). PCC 0.765 unseen patients, 0.798 sEEG-only. **Authors propose hybrid per-patient + shared layers as future work — validates our Stage 2 design.**

**Why these are complementary (Phase 3.1G):** Read-in and coordinate PE handle orthogonal sources of cross-patient variation. Boccato's "predominantly diagonal" finding means read-in primarily does signal normalization (gain, impedance) — exactly what coordinate PE cannot do. Coordinate PE provides spatial context (where on cortex) — exactly what a linear read-in cannot provide. At standard ECoG (1cm spacing), spatial variation dominates → coordinates alone suffice (PCC 0.837). At uECOG (<2mm spacing), signal variation becomes relatively more important — adjacent electrodes are ~1–2mm apart in MNI space (below registration error ~5mm), but still differ in amplitude/noise characteristics. **Critical for our cohort:** coordinate PE assumes MNI-average functional anatomy, but our patients include 4 tumor cases (cortex may be physically displaced/reorganized) and 4 Parkinson's cases (altered motor cortex organization). Coordinate PE can actively *mislead* the model for tumor patients — pointing to the wrong functional region. Per-patient read-in doesn't assume anything about anatomy; it learns the actual mapping from data. The hybrid degrades gracefully: if coordinates are accurate, read-in is small (gain normalization); if coordinates are wrong, read-in can override the misleading spatial prior. Untested in the literature.

**SSL pretraining** (wav2vec ECoG, BIT): Pretrain on unlabeled neural data (contrastive or masked). wav2vec ECoG (**wav2vec 1.0/CPC**, not 2.0 — no quantizer, causal convolutions) showed 60% WER reduction, largest gains in low-data regime. **Only 1 of ~12 cross-patient transfer combinations worked** (b→a). Minimum successful SSL corpus: ~30 min (participant b); our ~1 min/pt is 30x smaller. BIT: SSL advantage is cross-subject only (Table 9 shows SSL ≈ supervised for same-subject). ~20 pts × ~10 min utterance = ~3 hrs pooled — borderline for SSL. Masked reconstruction (MAE-style) preferred over contrastive for small corpora — no negatives needed, MIBRAIN validated on **raw broadband sEEG at 512 Hz** (NOT HGA — their MAE reconstructs raw signal, not spectral features). **Note:** BIT's mask ratio of 0 for T15 suggests masking may not even be the key ingredient.

**Articulatory TCA** (Wu 2025): Extract speaker-invariant articulatory features via tensor component analysis on EMA data, reconstruct from HD-ECoG (vSMC) via gradient boosting. PCC 0.75-0.80. **However:** reconstruction is word-level scalar loadings, NOT continuous articulatory trajectories — each PCC value compares 194-length vectors of scalars, not time-varying traces. Commonality formula `com(a_r) = (Σ a_r^i)² / Σ (a_r^i)²` is reusable for measuring cross-patient feature sharing. P3 was **intra-operative** (tumor resection, PCC=0.76) — most analogous to our setting. Discussion cites literature showing articulatory features are "more robustly encoded in vSMC than acoustic features and can be learned faster with limited neural data." Provides biologically grounded cross-speaker targets. Not yet used as an alignment objective for cross-patient decoding.

### Key Constraints

1. **Per-patient input layers are non-negotiable.** Levin: shared input layer makes transfer WORSE. Every successful cross-patient result uses per-patient input layers.
2. **Supervised joint training works IF per-patient layers filter variation first.** Singh: group PER 0.49 vs single 0.57 (25 patients). BIT's Table 9 confirms SSL ≈ supervised for same-subject — the SSL advantage is specifically in the cross-subject regime. BIT's cross-subject supervised failure is at N=2 with heterogeneous setups; Singh's supervised success is at N=25 with per-patient layers.
3. **Boccato's "mostly diagonal" does NOT apply to uECOG.** Utah arrays have fixed stereotactic placement and orientation — variation is mostly per-channel gain. uECOG has variable surgical placement, variable orientation (electrode (1,1) is not anatomically consistent), and two different grid sizes (8×16, 12×22). Cross-patient variation includes spatial remapping, not just gain. Verify via SVD of learned weights.
4. **Freeze backbone, train read-in + readout is the optimal transfer mode.** Singh's Mode 3 ("Recurrent Transfer").
5. **Per-subject heads are the most critical component in multi-subject decoding.** Seegnificant ablation: removing per-subject heads ΔR²=-0.18 (devastating). Spatial self-attention is #2 (ΔR²=-0.10). PE barely helps (ΔR²=-0.02, p=0.73 NOT significant). This calibrates expectations: coordinate PE is NOT the primary mechanism for handling electrode variability — attention itself is. Per-patient capacity (2,081 params/subject in seegnificant vs our 70) may be a critical gap.
6. **Factored attention (temporal then spatial) outperforms joint 2D attention.** Seegnificant: +0.06 R² and 5.5× faster. Validates v12's factored design (spatial cross-attention → temporal self-attention).

### External Knowledge Paths

No existing dataset transfers directly (spike-based models use Poisson ≠ Gaussian HGA).

| Path | Data needed | Confidence | Status |
|---|---|---|---|
| **Articulatory feature vectors** (Chomsky-Halle) | Zero external data | Medium | Phase 3 ablation |
| **Speech FM projection** (Wav2Vec2.0) | Paired audio (confirmed) | Medium-Low | Untested on motor cortex; Dual Pathway validated auditory cortex only |
| **Internal SSL** (MAE on ~3 hrs HGA) | None | Low-Medium | 3 hrs borderline; masked > contrastive |
| **Architecture transfer** (MRoPE, hierarchical CTC) | None | High | Free — no data, just design patterns |
| **Pretrained ECoG models** (Brant, BrainBERT) | None | Low | MIBRAIN: barely exceed chance |

---

## 3. Gap Analysis

### 3.1 Cross-Task Alignment
No paper aligns neural representations across different speech tasks in a cross-patient setting. The Cogan lab has ~10 phoneme + ~10 word/nonword patients — combining them doubles the pool but requires task-invariant articulatory representations. Kunz 2025 shows shared code across attempted/inner speech within-patient. Wu 2025 shows articulatory features generalize across speakers. But cross-task + cross-patient is untested.

**New insight from Duraivel 2025:** The pseudo-word repetition task (CVC/VCV, 52 tokens, same 9 phonemes) uses the same hardware and overlapping patients as Spalding 2025. At least 3 Spalding patients (S8, S5, S1/S2) also performed this task with 156–208 trials each. Cross-task pooling could nearly double effective training data for overlapping patients. Both tasks produce 3-phoneme CTC targets — Spalding non-words (e.g., [a,b,e]) and Duraivel pseudo-words (CVC/VCV) — enabling seamless cross-task pooling. However, pseudo-words engage planning networks not activated by isolated phonemes — the backbone must learn task-invariant articulatory representations. The spatial gradient from planning (anterior) to execution (posterior) observed on uECOG arrays confirms that grid position carries functional information, motivating the 2D conv input layer ablation.

### 3.2 Alignment Method Bake-Off
No study compares multiple alignment strategies on the same dataset. Each paper benchmarks only its own method. The field lacks evidence on which method suits which data regime — especially acute intra-op uECOG.

### 3.3 Foundation Model / External Pretraining for Intra-Op uECOG
No foundation model targets uECOG or intra-op recordings. All intracranial FMs use chronic epilepsy or spike data. The ~3 hr uECOG corpus is small. **No existing large dataset transfers directly** — spike-based corpora use Poisson statistics incompatible with Gaussian HGA envelopes. **Indirect paths exist:** (1) Speech FM projection (Wav2Vec2.0) as alignment target — imports 960 hrs speech knowledge, paired audio confirmed, untested on motor cortex (Dual Pathway validated auditory cortex only); (2) Articulatory feature knowledge from linguistic databases — zero external neural data, cross-patient invariant by construction; (3) Architecture transfer (hierarchical CTC, coordinate PE) — modality-agnostic design patterns.

### 3.4 uECOG Spatial Architectures
uECOG arrays have regular grid geometry (8×16 or 12×22, <2mm pitch). No decoder exploits this. Duraivel 2023 showed HG correlation drops below r=0.6 at <2mm, confirming adjacent electrodes carry distinct info. 2D convolutions or graph attention on the grid are unexplored. Chen 2024 validates the general approach: 3D ResNet on standard ECoG grids (1cm spacing) outperformed LSTM (PCC 0.806 vs 0.745, N=48). Chen 2025 (SwinTW) goes further: coordinate-based tokenization (MNI + ROI) enables multi-patient training that matches individual models (PCC 0.837 vs 0.831) and achieves PCC 0.765 on unseen patients — the first cross-patient ECoG speech result without per-patient decoder layers. **However:** SwinTW uses 1cm-spaced ECoG (64 electrodes), not uECOG (128–256 at <2mm). At uECOG resolution, adjacent electrodes carry distinct information — coordinate PE may be less critical when spatial relationships are already dense and regular. The **untested hybrid** (coordinate PE + per-patient read-in) is the strongest Phase 3 candidate. Additionally, 2D conv has fewer effective degrees of freedom than full affine despite more raw params (spatial weight sharing reduces effective DoF), making it actually *better regularized* for our data regime (~117k frames/patient).

### 3.5 LoRA / Adapters for Neural Signals
BIT uses LoRA for the LLM decoder but not the neural encoder. Levin 2026 explicitly cites LoRA as a promising future direction for neural encoders. No paper applies LoRA/adapters to the neural feature encoder for cross-patient adaptation.

### 3.6 Speech FM Projection for Motor Cortex
Dual Pathway validated projecting neural → **Wav2Vec2.0** space for *auditory cortex (STG) perception* only — **NOT HuBERT** (paper never mentions HuBERT). The near-linear mapping claim is grounded entirely in auditory pathway literature (Millet 2022, Li 2023, Chen 2024). Motor cortex → acoustic features is indirect (motor commands → articulatory dynamics → acoustics). However: (1) uECOG grids over sensorimotor cortex likely capture both motor commands AND somatosensory/auditory feedback; (2) the Wav2Vec2.0 embedding space encodes phoneme-level structure related to articulatory categories; (3) even low-R² alignment provides correct "direction" in representation space. **Paired audio confirmed to exist** for all 8 Spalding patients. Never tested on motor cortex — a genuine open question.

### 3.7 Other Gaps
- No standardized benchmark for cross-patient intracranial speech decoding
- Articulatory feature space as explicit alignment target never implemented cross-patient (Wu 2025 is closest but doesn't do cross-patient decoding)
- SSL on uECOG high-gamma envelopes specifically untested
- All cross-patient results are offline; no streaming cross-patient decoder
- Combining methods (e.g., CCA init + affine fine-tuning) unexplored

---

## 4. Research Directions (Ranked)

### Recommended Sequence

**Step 1 — Baseline reproduction.** Reproduce Spalding PCA+CCA+SVM → confirm 0.31 bal. acc. on 8 patients.

**Step 2 — Core improvement (highest expected gain).** Per-patient nn.Linear(D_padded→D_shared) read-in → LayerNorm → Conv1d temporal downsampling (200→40 Hz) → shared BiGRU → CTC, two-stage LOPO (train backbone on sources, adapt read-in + CTC head on target). Kaiming init for read-in (not near-identity — channel ordering is arbitrary). BiGRU 2×64 or 2×128 depending on verified data budget. Add feature dropout + smooth time masking. This tests three independent improvements over Spalding simultaneously: end-to-end alignment, temporal dynamics modeling, sequence decoding. Articulatory auxiliary loss deferred to Step 3 as ablation.

**Step 3 — Novel contribution: 2D conv on grid.** Replace affine with 2D conv exploiting uECOG grid topology (8×16, 12×22). Head-to-head comparison against affine. First paper to exploit uECOG spatial structure.

**Step 4 — Coordinate PE.** Add MRoPE from BrainLab 3D coordinates. Test: coordinate PE alone, affine alone, affine + coordinate PE. Requires BrainLab coordinates for all patients.

**Step 5 — Speech FM alignment (paired audio confirmed).** Add MSE loss predicting Wav2Vec2.0 embeddings from backbone hidden states. Imports 960 hrs of speech knowledge as cross-patient alignment signal. Untested on motor cortex (Dual Pathway validated auditory cortex only) — genuine open question. Note: Dual Pathway uses Wav2Vec2.0, not HuBERT.

**Step 6 — SSL pretraining (if headroom remains).** Masked reconstruction (MAE-style, not contrastive — no negatives needed, MIBRAIN validated on raw broadband sEEG, not HGA) on ~3 hrs pooled HGA. Compare: random init vs SSL init for shared backbone. Thin per-patient data (~10 min each) is the bottleneck — use diagonal affine (256 params) as per-patient layer.

### Confidence Assessment

| Step | Confidence | Rationale |
|---|---|---|
| Step 2 (read-in+GRU+CTC) | **High** | Singh proved joint training works (25 pts, BiLSTM 2×64), Boccato proved affine suffices, two-stage LOPO validated |
| Step 3 (ablations: input bake-off, articulatory loss) | **Medium** | Each ablation changes one component; bake-off is the main publishable finding |
| Step 4 (coordinate PE) | **Medium-High** | Chen 2025 SwinTW validates on ECoG speech (PCC 0.765 LOO, N=52). Hybrid with per-patient read-in untested — novel |
| Step 5 (speech FM) | **Medium-Low** | Most promising external path but motor cortex→acoustic is indirect; never tested |
| Step 6 (SSL) | **Low-Medium** | 3 hrs total borderline; wav2vec ECoG had 3-6x more per patient |

### What Won't Work
- **Direct transfer from spike models** (BIT, NDT3, POSSM): Poisson spikes ≠ Gaussian HGA — fundamental modality mismatch
- **Supervised cross-subject pretraining without per-patient layers**: BIT's Appendix M (qualitative, no numbers, N=2). With per-patient layers (Singh), supervised works
- **Dual Pathway exactly as published**: Validated for auditory cortex (STG) perception only, using Wav2Vec2.0 (not HuBERT). No evidence near-linear mapping extends to motor cortex

---

## 5. Technical Building Blocks

Architecture, training protocol, and evaluation details are in `implementation_plan.md`. This section covers literature reference material.

### Decoder Architectures (Literature)

| Architecture | Citation | Best For |
|---|---|---|
| 5-layer unidirectional GRU (512 units) + CTC | Willett 2023 | Proven speech baseline; 256→256 affine+softsign per-day layer (65.8k params/day); kernel=14 stride=4; LM from OpenWebText2 |
| Hierarchical GRU + feedback | Boccato 2026 | Cross-patient speech (best published; ~100M params, d=2048) |
| Conv1D + BiGRU Seq2Seq | Spalding 2025 | uECOG phoneme baseline |
| 3D ResNet (causal) | Chen 2024 | Grid-structured ECoG (PCC 0.806, N=48); channels 16→32→64→128→256, InstanceNorm, LeakyReLU(0.2); LREQAdam lr=0.002 beta1=0.0; electrodes reshaped to sqrt(N)×sqrt(N) grid; gradient reversal speaker classifier exists in codebase (unused) |
| RNN-T | Littlejohn 2025 | Streaming |
| Transformer + read-in | BIT 2026 | Foundation model approach (N=2, 367 hrs) |

### Feature Extraction & Normalization

Cogan lab HGA pipeline: decimate 2kHz → CAR → 70-150Hz bandpass (8 Gaussian) → Hilbert → 200Hz → baseline normalize → HG-ESNR channel selection. Per-channel z-score within session. Per-trial baseline subtraction for structured tasks.

### Data Augmentation

See `pipeline_decisions.md` §9 for full analysis. Phase 2 stack: per-channel amplitude scaling (per-trial σ=0.15), Gaussian noise (2% RMS), feature dropout (p~U[0,0.3] in D_shared), smooth time masking. Key constraints: amplitude scaling is per-trial (impedance), not per-frame (noise). HGA is non-negative → sign flip invalid.
