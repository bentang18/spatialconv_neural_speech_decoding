# Pipeline Design Decisions

Every design choice for the cross-patient uECOG phoneme decoding pipeline. Each question captures a real tension — not just "A vs B" but why the choice is hard.

---

## 1. Feature Extraction

- **Output rate**: 200 Hz gives ~200 timepoints per 1-sec window. Phoneme transitions happen at ~10-20 Hz. Is 200 Hz wasting model capacity on redundant timepoints, or does the fine temporal grain help the backbone learn dynamics? (Spalding: 200 Hz, wav2vec ECoG: 100 Hz, Dual Pathway: 50 Hz — all work, lower rates reduce compute)

- **Channel selection**: The HG-ESNR permutation test discards non-significant channels, reducing input dimensionality but also removing channels that might contribute in combination. With a learned input layer, should we let the model discover which channels matter? (Risk: with 20 min data, learning to ignore 100+ noisy channels may overfit)

- **Normalization responsibility**: Pre-stimulus baseline subtraction is standard, but a learned input layer (affine W·x+b) can absorb per-channel offsets via the bias term. Does double-normalizing (baseline subtraction + learned bias) help or hurt? Should we drop one?

- **Analysis window**: ±500 ms around phoneme onset means the window must be aligned to detected speech onset. For CTC decoding (which handles alignment internally), should we use the full trial instead and let CTC find the relevant segment?

## 2. Per-Patient Input Layer

- **Is explicit alignment necessary?** Spalding showed CCA alignment is essential (unaligned pooling hurts). But CCA is linear and offline. Does end-to-end training with a shared backbone implicitly learn alignment, making a dedicated alignment layer redundant? (Singh's group model has no explicit alignment loss — shared LSTM training alone aligns representations)

- **Linear vs nonlinear**: Boccato found affine transforms are predominantly diagonal (gain adjustment). If cross-patient variation is mostly per-channel scaling, a full matrix is overkill and a diagonal + bias suffices. But if spatial remapping matters (different cortical folds = different channel-to-region mappings), the full matrix is necessary. **Note:** With confirmed D_padded=208 and D_shared=64, the full linear read-in is Linear(208,64) = ~13.4k params per patient — already modest. The diagonal+bias variant would need separate handling since input/output dims differ.

- **Grid-aware input**: uECOG has regular 2D grid topology (8×16 or 12×22) unlike sEEG or Utah arrays. A 2D conv could learn local spatial receptive fields (e.g., detecting articulatory activation patterns across adjacent electrodes). No paper has tried this on uECOG. Is the spatial structure informative enough to justify the added complexity, given Duraivel showed distinct information at <2mm spacing?

- **Handling two array geometries**: The 128-ch (8×16, 1.33mm) and 256-ch (12×22, 1.72mm) arrays differ in both channel count and pitch. Should these share an input layer (with zero-padding/masking), have separate input layers (halving effective training data per layer), or be projected to a common spatial grid (losing the native resolution)?

- **Warm-starting from CCA**: Spalding's CCA projections are known to align well. Should the per-patient input layers be initialized from CCA solutions (giving the model a good starting point) or random (letting end-to-end training find its own solution)? Risk: CCA initialization might trap the model in a linear local minimum.

### Resolved

See `implementation_plan.md` §2 for chosen design (nn.Linear D_padded→D_shared, Kaiming init) and §3.1 for the input layer bake-off. Key reasoning preserved here: (1) diagonal+bias preserves input dim but backbone needs fixed width → projection required regardless; (2) near-identity init is meaningless for non-square matrices; (3) Boccato's diagonal finding is for Utah arrays (fixed placement) — uECOG has variable placement, verify via SVD; (4) 2D conv has more raw params than diagonal but fewer effective DoF via weight sharing.

## 3. Spatial Encoding

- **Grid position vs MNI coordinates**: Grid-relative positions (row, col) capture local topology but not anatomical location. BrainLab MNI coordinates capture where each electrode sits on cortex but require registration. Spalding showed cross-patient CCA works — meaning the relevant variation is functional, not purely anatomical. Which coordinate frame captures what matters for alignment?

- **Does spatial encoding replace or complement per-patient layers?** Coordinate PE lets a transformer attend to electrode positions directly, potentially handling cross-patient spatial variation without per-patient parameters. **Chen 2025 (SwinTW) shows it CAN replace per-patient layers on standard ECoG:** a single 15-patient model matched 15 individuals (PCC 0.837 vs 0.831) and achieved PCC 0.765 on unseen patients with NO per-patient layers. But: SwinTW uses 1cm-spaced ECoG (64 electrodes), not uECOG (128–256 at <2mm). At uECOG resolution, coordinate PE may be less informative (electrodes are densely packed, all within a ~2cm² patch). **The untested hybrid — coordinate PE + per-patient read-in — is the strongest candidate for our Phase 3.1 bake-off.**

## 4. Backbone Architecture

- **GRU vs Transformer given data budget**: GRUs have strong inductive bias for sequential data (good when data-scarce), while transformers are more flexible but data-hungry. BIT needed 367 hrs for transformer pretraining; NDT3 needed 2000 hrs. With ~3 hrs total across patients, is a transformer feasible even with SSL? Or are we firmly in GRU territory?

- **What temporal scale matters?** Phoneme durations are ~50-100 ms. Coarticulation effects span ~200 ms. The full 3-phoneme token is ~450 ms. A 2-layer GRU with 512 units captures all of these. Is there any speech structure at longer timescales that a transformer's global attention would help with, given the simple non-word task?

- **Bidirectional vs causal**: Bidirectional sees the full trial (better accuracy) but can't stream. The non-word repetition task is inherently offline (short isolated tokens), so bidirectional is fine now. But if the pipeline extends to continuous speech (word/nonword task), causal may be needed. Design for offline first and add causal later, or build causal from the start?

### Resolved: Backbone Sizing

See `implementation_plan.md` §2 for chosen design (BiGRU 2×64 default, Conv1d k=5 s=5 temporal downsampling). Key reasoning: Singh decoded 36 phonemes from 25 patients with BiLSTM 2×64 — our 9-phoneme task is simpler. CTC on 200 frames for 3 phonemes has diffuse gradients without temporal downsampling (all successful CTC papers downsample). Mamba/SSM (Phase 3 ablation) offers ~6× fewer params but is unvalidated for intracortical speech.

## 5. Pretraining Strategy

- **Is ~3 hours pooled enough for SSL?** wav2vec ECoG worked with ~1 hr/patient, but only tested within-patient SSL. BIT's cross-subject SSL used 367 hrs. Each patient contributes only ~8-10 min of utterance data. Pooled cross-patient SSL on ~3 hrs from ~20 patients is in uncharted territory. The per-patient input layers each see only ~10 min — is that enough to learn even a simple channel projection?

- **What should SSL predict?** Contrastive (wav2vec): predict future neural states from context. Masked reconstruction (BIT/MAE): reconstruct masked timepoints/channels. Contrastive forces temporal abstraction; masking forces spatial-temporal interpolation. For uECOG with regular grid and 200 Hz temporal resolution, which inductive bias is more useful?

- **SSL on raw signals or HGA features?** wav2vec ECoG showed removing bandpass filtering is fatal for SSL — low-frequency power dominates and is easy to predict but speech-irrelevant. But SSL on already-extracted HGA means the SSL model can't discover features the pipeline missed. Is the standard HGA pipeline truly optimal, or might SSL on broadband data find something better?

- **Project into speech FM space (Dual Pathway) or learn neural structure (wav2vec ECoG)?** Dual Pathway leverages 960 hrs of speech knowledge but requires paired audio and is validated only for **auditory cortex (STG) perception** using **Wav2Vec2.0** (NOT HuBERT). The near-linear mapping claim is grounded entirely in auditory pathway literature — **no evidence it extends to motor cortex**. wav2vec ECoG (**wav2vec 1.0/CPC**, not 2.0 — no quantizer) works for production but has less external knowledge; only 1 of 12 cross-patient combinations worked; minimum successful SSL corpus was ~30 min (our ~1 min/pt is 30x smaller). **Update (confirmed from papers):** Paired audio EXISTS for all 8 Spalding patients — lavalier mic synced at 20ksps on Intan + 44.1kHz laptop, phoneme labels via Montreal Forced Aligner. Speech FM projection is feasible to test empirically, but motor cortex is an unvalidated domain for this approach.

- **Can any existing large dataset pretrain our backbone?** No existing pretrained model transfers directly. Every spike-based corpus (BIT 367 hrs, NDT3 2000 hrs, POSSM) uses Poisson statistics incompatible with Gaussian HGA. MIBRAIN showed Brant/BrainBERT barely exceed chance cross-patient. Scalp EEG has fundamentally different spatial resolution. **What does transfer:** architectural patterns (MRoPE, wav2vec framework, cross-attention aggregation, hierarchical CTC) and linguistic knowledge (articulatory feature vectors as alignment targets). The most data-efficient "external pretraining" is importing structured knowledge about speech production, not neural signal weights.

## 6. Decoder Design

- **Classification vs CTC**: The phoneme task has exactly 3 positions with known timing. Per-position classification is natural and simple. CTC handles variable-length output and doesn't need position labels, but it's designed for longer sequences where alignment is ambiguous — with only 3 symbols, is CTC solving a problem that doesn't exist? (Counter: CTC generalizes to words/sentences later; classification doesn't)

- **What to decode**: Phoneme identity (9 classes) is the established target. But articulatory features (place, manner, voicing) are more continuous and more conserved across patients (Wu 2025). Decoding articulatory features first, then mapping to phonemes, adds a bottleneck but might make cross-patient transfer easier. Is the articulatory intermediate representation worth the indirection?

- **Hierarchical CTC (Boccato)**: Auxiliary CTC losses at intermediate GRU layers provide richer gradient signal and partially address CTC's conditional independence assumption. But with only 3-phoneme sequences, conditional independence may not matter — there's minimal structure to exploit between symbols. Does hierarchical CTC help for very short sequences?

## 7. Loss Design

- **Should the loss explicitly encourage cross-patient alignment?** End-to-end CTC training implicitly aligns representations (same phonemes must produce same outputs). But adding a contrastive loss (InfoNCE on same-phoneme pairs across patients) explicitly rewards alignment. Is implicit alignment sufficient, or does the model need direct pressure to overcome inter-patient variability? **Key distinction:** BIT's Table 9 shows SSL ≈ supervised for same-subject — the SSL advantage is specifically cross-subject. BIT's cross-subject supervised failure (Appendix M) is qualitative only (no numbers), at N=2 with heterogeneous setups, and without backbone freezing. Singh showed supervised *joint training* with per-patient input layers works at N=25 (group PER 0.49 vs single 0.57). So implicit alignment via joint training with per-patient layers IS viable — the per-patient layers are the critical component, not an explicit alignment loss or SSL objective.

- **Articulatory auxiliary loss as free alignment signal**: Predicting phonological feature vectors (place, manner, voicing, height, backness) from backbone hidden states provides a cross-patient-invariant learning signal at zero external data cost. Duraivel showed confusion errors track Chomsky-Halle distance — the neural data naturally organizes along articulatory dimensions. This biases the backbone toward articulatory representations that are cross-patient invariant by construction (all humans share the same articulators). Implementation: auxiliary linear head → MSE on phonological feature vectors derived from phoneme labels.

- **Phonological distance weighting**: Duraivel showed errors track Chomsky-Halle feature distance — confusing /b/ with /p/ (differ by 1 feature) is less bad than confusing /b/ with /i/ (differ by many). Should the loss penalize phonologically distant errors more heavily (cost-sensitive CE), or let the model discover this structure on its own?

- **Auxiliary articulatory regression**: Simultaneously predicting articulatory features (4-way articulator class, or continuous features from Wu 2025) alongside phonemes could regularize the backbone toward cross-patient-invariant representations. But it adds a second output head and potentially conflicting gradients. Is the regularization benefit worth the complexity?

## 8. Training Protocol

- **Leave-one-patient-out (LOPO) vs leave-K-out**: LOPO maximizes training data but gives N=20 test points (one per patient). Leave-2-out or leave-3-out gives more test combinations but less training data. With only 20 patients, which gives more reliable estimates of generalization?

- **How to handle "hard" patients**: Some patients will have poor single-patient decoding — Spalding's S3 has only 46 trials/9.2k frames (confirmed from Table S3), and S5 has only 63/128 sig channels. Spalding found S3 benefited MOST from cross-patient data (0.53 aligned vs 0.29 patient-specific). Should source patient weighting prioritize similar-coverage patients (Singh found coverage correlation predicts transfer) or treat all sources equally?

### Resolved: Fine-tuning scope

See `implementation_plan.md` §2 Stage 2 (Singh Mode 3: freeze backbone, train read-in + CTC head, 100 epochs). **Source replay (Levin 2026): 30% of training batches from source patients, 70% from target, uniform across source days.** Even with backbone frozen, source replay helps the read-in layer maintain compatibility with the frozen backbone's expected input distribution.

### Training protocol gaps (from re-reading)

**Key finding:** Neither Singh nor Levin report optimizer, learning rate, or batch size in their papers. Our planned differential LR (read-in 3e-3, backbone 1e-3) is entirely our own design choice — it is reasonable but has no precedent to cite. Singh's Conv1D front-end hyperparameters (kernel, stride, filters) are also absent. Boccato does report full training details (Adam 5e-3 cosine→1e-4, batch 64, 120k steps) but their backbone is ~100M params — inapplicable to our ~130K. **Willett 2023** does report full details: Adam (beta1=0.9, beta2=0.999, eps=0.1), LR linearly decayed 0.02→0, batch 64, 10000 minibatches, dropout 0.4. Day-specific layer is 256→256 **affine + softsign** (65.8k params/day, NOT dimensionality-reducing). Data augmentation: white noise SD=1.0 + constant offset SD=0.2 per minibatch. Note: Willett's regime (hours/day, single patient) is very different from ours (~10 min/patient, N≈20).

## 9. Data Augmentation

### Resolved Questions

- **Channel/feature dropout rate**: Two distinct operations: (1) **Feature dropout** (after read-in, in D_shared space): spatial dropout — zero entire feature channels (same mask for all B×T positions), p ~ U[0, 0.3]. Regularizes backbone. (2) **Channel dropout** (before read-in, in channel space): zero random input channels. **Phase 2: p ~ U[0, 0.1]** (conservative — Linear read-in tolerates sparsity but Conv2d grid holes are more disruptive). **Phase 3: p ~ U[0, 0.2]** (Conv2d already handles sparse grids from non-sig channels, so tolerates more dropout). The SPINT design (r ~ U[0, 1]) is very aggressive — we cap at 0.3 for feature dropout.

- **Cross-patient augmentation (chicken-and-egg resolved)**: Yes, it's chicken-and-egg early in training. **Resolution**: warmup schedule — train per-patient affine layers for N epochs (alignment first), then enable cross-patient Mixup in the aligned space. Even partially aligned representations benefit from interpolation. Implementation: after warmup, for same-class trials from patients A and B, create z̃ = λ·affine_A(x_A) + (1−λ)·affine_B(x_B) with λ ~ Beta(0.2, 0.2).

- **Temporal augmentation**: Time warp has minimal value for phoneme repetition (timing is stimulus-locked, minimal speech rate variation). Smooth time masking is the right temporal augmentation — zero a random 20–80 ms window with cosine taper. This forces the GRU to rely on temporal context, which is compatible with CTC's temporal marginalization. Small temporal jitter (±10–30 ms) simulates motor response timing variability. Full time reversal is invalid (destroys phoneme temporal sequence).

### Augmentation Constraints from HGA Envelope Properties

HGA envelopes (70–150 Hz Hilbert amplitude, non-negative, baseline-normalized, 200 Hz) constrain which augmentations are valid:
- **Non-negative** → sign flip is meaningless (unlike raw EEG where reference polarity is arbitrary)
- **Baseline-normalized** → DC shift is redundant (re-introduces what normalization removed)
- **Short trials (~300–500 ms)** → aggressive cropping destroys phoneme dynamics; max crop = 30% of trial
- **Per-channel gain is dominant inter-patient variation** (Boccato) → per-channel amplitude scaling is the single most relevant augmentation

### Augmentation Stack (Ordered by Priority)

**Phase 2 (Tier 1):** See `implementation_plan.md` for exact placement in architecture.
1. Per-channel amplitude scaling (before read-in, **per-trial** σ=0.15) — simulates impedance variation
2. Gaussian noise (before read-in, 2% RMS per-frame) — universal regularizer
3. Channel dropout (before read-in, p~U[0,0.1]) — simulates missing electrodes (conservative in Phase 2)
4. Feature dropout / spatial dropout (after LayerNorm, p~U[0,0.3] in D_shared space, same mask for all B×T positions) — regularizes backbone by dropping entire feature channels
5. Smooth time masking (after temporal conv, 2–4 frames at 40 Hz, cosine taper; min 2 because mask_len=1 has no effect with cosine ramp) — forces context reliance

**Phase 3 (Tier 2):** Channel dropout increase (p~U[0,0.2] after Conv2d validated), trial averaging (Dirichlet-weighted), cross-patient Mixup (after warmup), FT phase perturbation (Rommel 2022), grid spatial recombination (BAR-style).

**Phase 3 (Tier 3):** Manifold Mixup in GRU hidden states, virtual electrode interpolation, automated search (CADDA).

**Not viable:** GAN/VAE (data-hungry), time reversal (destroys CTC sequence), sign flip (HGA is non-negative), CutMix across classes (chimeric phonemes).

### Architecture-Specific Placement

Augmentations belong at specific pipeline stages:
- **Before read-in** (simulates what the layer must learn): amplitude scaling (per-trial), Gaussian noise (per-frame)
- **After read-in + LayerNorm** (regularizes backbone input): feature dropout (in D_shared space), smooth time mask, FT phase perturbation, cross-patient Mixup
- **Before read-in** (Phase 3 only): electrode dropout (in channel space, p_max=0.1) — simulates missing channels
- **Data loader** (pre-processing): trial averaging, temporal jitter, cropping
- **GRU hidden states** (optional): manifold Mixup, DACL-style latent noise

### Synthetic Data & Test-Time Augmentation

**Synthetic data:** No generative approach works in our regime. GANs/VAEs data-hungry (~65 trials/phoneme/patient). Physics-informed HGA needs precise somatotopic maps we don't have. Only viable path: Speech FM → synthetic neural, but requires three levels of indirection (audio → Wav2Vec2.0 → projection → synthetic HGA). Defer until Speech FM alignment validated.

**Test-time augmentation (Phase 3):** K=5 augmented copies (amplitude σ=0.05, noise 1% RMS), average CTC posteriors. 1–3% free gain. Not yet reported for ECoG.

## 10. Cross-Task Extension

**Update (from Duraivel 2025):** The "other task" is now identified as **pseudo-word repetition** (CVC/VCV, 52 tokens, same 9 phonemes). Duraivel 2025 describes it in full detail. At least 3 patients (Spalding S8, S5, S1/S2) performed both tasks. Same hardware, same preprocessing pipeline, same IRB (Pro00072892).

- **Are articulatory representations task-invariant?** Duraivel 2025 shows pseudo-word repetition engages distinct prefrontal planning networks NOT activated by simple phoneme repetition. However, the articulatory execution networks (PrCG, PoCG, IPC) overlap — these are where our uECOG arrays sit. The backbone should learn articulatory execution dynamics (shared across tasks) rather than planning dynamics (task-specific, primarily prefrontal). **If the shared BiGRU learns execution-phase representations, cross-task pooling should work.** The articulatory auxiliary loss (Phase 3.2) explicitly biases toward this.

- **Cross-task data characteristics**: Pseudo-word trials produce 3-phoneme sequences (P1→P2→P3 with transitions). CTC naturally handles this (same as Spalding's seq2seq decoding P1, P2, P3). Duraivel 2025 confirms motor cortex encodes phonotactic transitions between phonemes, validating CTC's continuous output model. Key difference: pseudo-word task has richer temporal structure (planning→execution→monitoring) that may improve backbone temporal dynamics learning.

- **Array placement compatibility**: Confirmed compatible. Duraivel 2025 uECOG patients used the same arrays over left sensorimotor cortex. The anterior-posterior spatial gradient (planning→execution) is visible within the array, meaning arrays cover both planning and execution zones.

- **Label space**: Both tasks use the **same 9 phonemes** — no label space mismatch. CVC and VCV pseudo-words are built from the same phoneme inventory. Both tasks produce 3-phoneme CTC targets (Spalding non-words like [a,b,e] and Duraivel pseudo-words like [b,a,g]).

- **Data augmentation via cross-task pooling**: For overlapping patients, this nearly doubles trial count (~150 Spalding + ~200 Duraivel 2025 = ~350 per patient). For non-overlapping patients, cross-task patients provide additional source patients for Stage 1 LOPO training.

- **52 epilepsy patients NOT directly poolable**: Duraivel 2025's 52 epilepsy patients use SEEG/macro-ECoG (different modality, spatial resolution, coverage). Cross-modality pooling would require separate handling — a Phase 3+ direction if ever pursued.

### Open questions (for Zac)
1. Which Spalding patients also performed the pseudo-word task? (We infer S8, S5, S1/S2 from sig channel matching — confirm)
2. Are the pseudo-word HGA features pre-extracted, or do we need MATLAB preprocessing?
3. Are there additional uECOG patients who performed pseudo-word repetition but are NOT in Spalding's 8?

## 11. Evaluation

- **What metric isolates neural decoder quality?** Balanced accuracy (Spalding) measures classification; PER measures sequence decoding; WER adds language model effects. For comparing alignment methods, PER without LM is cleanest. But WER is what matters clinically. Report PER as primary for science, WER as secondary for clinical framing?

- **Is TME surrogate control necessary for every experiment?** Spalding used TME surrogates (preserving first/second order statistics) to prove CCA alignment doesn't artificially create structure. Should every new alignment method get its own TME control, or does Spalding's demonstration suffice as field-level validation?

- **Scaling behavior as the main finding**: The most publishable result may not be "our method gets X% accuracy" but rather "accuracy scales as f(N_patients, N_trials)" — showing how performance improves with more patients and data. This requires systematic subsampling experiments. Should scaling analysis be a primary deliverable rather than a side analysis?

## 12. Pipeline-Level

- **End-to-end vs staged training**: Spalding's PCA→CCA→SVM is fully staged (each step optimized independently). End-to-end training lets the input layer, backbone, and decoder co-adapt. But end-to-end on 20 min of data per patient risks overfitting the entire pipeline to noise. Is staged training safer with limited data, even though it's suboptimal in the limit?

- **Comparison fairness**: If the new pipeline uses a GRU backbone + CTC and Spalding uses SVM, a performance gain could come from the decoder alone, not the alignment. To isolate the alignment contribution, should we test: (a) Spalding alignment + new decoder, (b) new alignment + Spalding decoder, (c) new alignment + new decoder? This triples the experiment count but gives clean attribution.

- **What's the minimum publishable result?** Beating Spalding's 0.31 balanced accuracy on the same 8 patients is necessary but maybe not sufficient. What makes this a paper — the method comparison, the scaling analysis, the cross-task result, or the architectural novelty (2D conv on uECOG grid)?
