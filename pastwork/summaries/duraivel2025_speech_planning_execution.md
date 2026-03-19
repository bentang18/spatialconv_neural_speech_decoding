# Duraivel 2025 — Distinct neural processes link speech planning and execution

**Citation:** Duraivel, S., Rahimpour, S., Barth, K., Chiang, C.-H., Wang, C., Harward, S.C., Lad, S.P., Sexton, D.P., Friedman, A.H., Sinha, S.R., Hickok, G., Southwell, D.G., Viventi, J., & Cogan, G.B. bioRxiv (2025). [doi:10.1101/2024.10.07.617122](https://doi.org/10.1101/2024.10.07.617122)

**Status:** Preprint (bioRxiv, v2 posted 2025-09-23)

---

## One-Sentence Summary

Using intracranial recordings from 52 epilepsy patients and 3 intra-op uECOG patients performing pseudo-word repetition, the study reveals distinct neural processes for speech planning (prefrontal) and execution (sensorimotor), with a two-tiered hierarchical organization where syllables are coded before phonemes.

## Task

**Delayed pseudo-word repetition.** Patients hear a constructed pseudo-word, wait for a visual "Speak" cue (**1.5s delay, 150ms jitter** per Methods; Results says "1.2–1.6s"; Fig 1d shows "1.4 ± 0.2s"), then repeat it aloud.

- **Trial timing:** "Listen" cue → auditory stimulus (~0.5s) → 1.5s delay (150ms jitter) → "Speak" cue → 3s response window (200ms jitter). Each trial 5.1–6.1s total. 208 trials < 40 min.
- **Stimuli:** 52 unique pseudo-word tokens
- **Structure:** Monosyllabic CVC (26 tokens) + disyllabic VCV (26 tokens)
- **Phoneme inventory:** 9 total — 4 vowels (/a/, /æ/, /i/, /u/) + 5 consonants (/b/, /p/, /v/, /g/, /k/)
  - **Same 9 phonemes as Spalding 2025 and Duraivel 2023**
- **Biphone selection:** Forward biphoneme transition probability > 0 (from English corpus); triphone probability = 0 (ensuring pseudo-words, not real words)
- **Blocks:** 4 blocks × 52 tokens = **208 trials** per patient (epilepsy); 3–4 blocks = **156–208 trials** (intra-op). **S1 and S2 confirmed 4 blocks (208 each)**; S3 not specified
- **Response filtering:** Only correctly articulated trials with response time > 50ms from go cue used
- **Intra-op modification:** No delay or speak cue — patients repeat **immediately** after hearing the pseudo-word. This makes the intra-op task more similar to Spalding's phoneme repetition task
- **CVCVC:** A **control analysis** to rule out initial phoneme confound with syllable class, not a separate task condition
- **Task software:** Psychtoolbox scripts in MATLAB R2014a

## Participants

### Epilepsy monitoring unit (N=52)
- Mean age 33.1 years, 23 female
- Macro-ECoG: 2 patients (Ad-Tech 48/64 electrodes, 10mm spacing, 2.3mm contacts)
- SEEG: 50 patients (PMT/Ad-Tech/Dixi, 3.5–5mm spacing, 0.8–0.86mm contacts)
- Recording: Natus Quantum LTM, analog 0.01–1000 Hz, digitized 2048 sps

### Intra-operative uECOG (N=3)

| Patient | Array | Spacing | Sig ch | Diagnosis | Surgery |
|---------|-------|---------|--------|-----------|---------|
| S1 | 12×22, 256ch | 1.72mm | 201/256 | Tumor | Awake craniotomy |
| S2 | 8×16, 128ch | 1.33mm | 63/128 | DBS (movement) | DBS placement |
| S3 | 8×16, 128ch | 1.33mm | 111/128 | DBS (movement) | DBS placement |

- Recording: Intan RHX v3.0, analog 0.1–7500 Hz, digitized 20,000 sps
- Mean age 64.3 years, 0 female

### Patient mapping to Spalding 2025

| Duraivel 2025 | Sig ch | → Spalding 2025 | → Duraivel 2023 | Evidence |
|---------------|--------|-----------------|-----------------|----------|
| S1 (256ch, tumor) | 201/256 | **S8** | — | Array type + sig ch count match |
| S2 (128ch, DBS) | 63/128 | **S5** | **S3** | Array type + sig ch count match |
| S3 (128ch, DBS) | 111/128 | **S1 or S2** | **S1** | Paper explicitly says "see also patient S1 in our previous work^63" (ref 63 = Duraivel 2023) |

**Same patients performed both tasks** — pseudo-word repetition (this paper) and phoneme decoding (Spalding). This enables cross-task data pooling.

**Note:** Methods section has DBS/tumor assignment inconsistency with Results section. Methods says "S1 and S3 = DBS, S2 = tumor" but Results + figures consistently show S1 = 256ch awake craniotomy (tumor), S2 = 128ch DBS. Results/figures are authoritative.

**uECOG electrode contact diameter:** 200 μm.

## Preprocessing

**Same core pipeline as Duraivel 2023/Spalding 2025, with additions:**
1. Line-noise removal: multi-taper band-stop at 60, 120, 180, 240 Hz
2. Remove electrodes: outside cortex, muscle artifacts, recording power > 3 × log-RMS
3. CAR on remaining electrodes (9679 total → 1573 excluded → **8106 post-exclusion/pre-CAR**)
4. HGA extraction: 8 log-spaced Gaussian bands 70–150 Hz → Hilbert envelope → average → downsample 200 Hz
5. Z-score relative to baseline (−500 ms to 0 s before **trial start** = before auditory stimulus, NOT before response onset — different from Spalding's response-aligned baseline)
6. Response epoch: −1500 ms to +1500 ms around utterance onset (wider than Spalding's ±500ms to capture planning period)
7. Speech-significant electrodes: HG power response (−500 to 500ms) vs baseline, FDR-corrected permutation test (p < 0.05)
8. **ROI assignment:** BioImage Suite + FreeSurfer reconstruction, **Brainnetome Atlas** cortical parcellations. SEEG near white matter: 10mm radius sphere for nearest ROI.

**Result:** 3534/8106 electrodes showed significant speech HG activations (1701 left + 1665 right; 2895 grey matter contacts).

## Key Findings

### 1. Three distinct speech networks (temporal progression)
- **Planning network** (IFG: 92, rMFG: 137, cMFG: 186) — activates first, before utterance onset
- **Articulation network** (PrCG & PoCG: 351, IPC: 268) — activates during execution
- **Monitoring network** (Insula: 172, aSTG: 91, pSTG: 90, STS: 88, PAC: 43) — activates last
- Validated by data-driven NNMF decomposition (5 temporal components matched the 3 networks)

### 2. Syllable codes precede phoneme codes (hierarchical organization)
- Syllable decoding: **93%** (2-way CVC vs VCV, chance 50%)
- Phoneme decoding: **38.3%** (9-way, chance 11.11%)
- Syllable decoding onset: −1.3s (before phoneme decoding onset), p = 3.14e-8
- **Temporal compression:** Planning: onset 0.25s (p=0.039), peak 0.32s (p=3.11e-8); Articulation: onset 0.13s (p=3.5e-5), peak 0.11s (p=2.7e-8); Monitoring: onset 0.017s (p=0.082, **NOT significant**), peak -0.025s (p=0.9, **NOT significant**, negative = phonemes slightly precede syllables)

### 3. Sequential phonological coding only during execution
- Articulatory ROIs show ordered decoding of P1→P2→P3 (9-way per position)
- Planning regions do NOT show sequential ordering — phonemes are retrieved but not sequenced
- Monitoring regions also show sequential coding (auditory feedback tracking)

### 4. Spatial gradient on uECOG arrays
- Syllable coding timing progresses **anterior→posterior** on the array
- Anterior (prefrontal) sites: earlier coding → planning
- Posterior (sensorimotor) sites: later coding → execution
- Spatial linear model (F-test) of syllable coding timing vs anterior-posterior position (mm):
  - S1: onset R²=0.05, F(2,92)=6.02, p=0.016; peak R²=0.09, F(2,98)=10.9, p=1.3e-4 (text says 1.3e-3, figure caption 1.3e-4)
  - S2: onset R²=0.08, F(2,28)=6.02, p=0.07 (**NOT significant**); peak R²=0.13, F(2,28)=5.68, p=0.024
- **R² values are low (0.05–0.13)** — gradient explains only 5–13% of variance
- **18.9% of significant uECOG electrodes peaked before utterance onset** (S1: 12/201, S2: 38/63) — planning activity
- S1: 100/201 sig electrodes showed syllable contrasts; S2: 37/63

### 5. Phonotactic transitions in motor cortex
- Motor cortex tracks continuous phoneme transitions, not just discrete phonemes
- Forward transition probabilities (P(V|C) and P(C|V)) are decodable (3-way)
- Transitions are temporally embedded between positional phoneme codes: P1→Pfwd1→P2→Pfwd2→P3

## Behavioral Results

| Structure | Response Time (s) | Response Duration (s) |
|-----------|------------------|-----------------------|
| CVC | 0.56 ± 0.35 | 0.58 ± 0.17 |
| VCV | 0.60 ± 0.32 | 0.68 ± 0.14 |

VCV (2-syllable) had longer durations (p = 2.1e-26) and longer response times (p = 2.2e-4) than CVC (1-syllable), confirming syllable-level planning.

## Audio Recording

- Condenser microphone: Marantz Professional MPM-1000
- Pre-amplifier: Behringer, digitized at **44,100 Hz** on recording laptop
- Stimulus delivery: laptop with Psychtoolbox → powered stereo-speaker (AmazonBasics A100) via USB DAC + Fiio amplifier
- Trial synchronization: photodiode (Thorlabs) on stimulus laptop screen → BNC-to-mono → Natus Recording System
- Speech onset labeling: Audacity (manual mel-spectrogram) or Montreal Forced Aligner (automated)

## Decoding Method

- **SVD-LDA:** SVD to reduce to 80% variance components → LDA classifier → 10-fold CV, **20 decoding iterations** averaged
- **Multivariate:** Pooled across 52 epilepsy patients (2895 grey matter electrodes, 69 mean per subject) — **NOT per-patient decoding**. No per-patient uECOG decoding reported. Mix-Up augmentation for trial count mismatch
- **Normalized accuracy:** Acc_norm = (Acc_Test - Acc_Chance) / (100 - Acc_Chance)
- **Temporal decoding:** 200ms window, 10ms step, evaluated at each time segment
- **Chance computed by:** 250 shuffled label permutations

## Code & Data

- Code: [github.com/coganlab/IEEG_Pipelines](https://github.com/coganlab/IEEG_Pipelines) (MATLAB) — described as "will be updated upon publication" (preprint — may not yet be public)
- Data: DABI (restricted access, "will be archived upon publication")
- IRB: Pro00065476 (network dynamics), **Pro00072892** (uECOG — same as Spalding 2025)

## Relevance to Our Project

### Cross-task pooling opportunity
1. **Same 9 phonemes, same hardware, overlapping patients** — the pseudo-word repetition data from Duraivel 2025 can augment Spalding's phoneme data
2. **For shared patients (S8, S5, S1/S2):** Each patient has ~150 phoneme trials (Spalding) + ~200 pseudo-word trials (Duraivel 2025) = ~350 total trials
3. **CTC naturally handles both:** Single phoneme → length-1 target; pseudo-word → length-3 target (P1, P2, P3)
4. **Different task structures exercise different neural circuits** — planning vs pure execution — enriching backbone representations
5. **Cross-task pooling doubles effective training data** for overlapping patients without additional electrode counts

### Spatial gradient validates 2D conv motivation
- Anterior-to-posterior syllable coding gradient on uECOG arrays demonstrates that **spatial position on the grid encodes functionally distinct information**
- 2D conv (Phase 3.1D) would naturally capture this spatial gradient through receptive fields
- Planning→execution transition visible within a single uECOG array confirms the grid is not spatially homogeneous

### Phonotactic transitions support CTC
- Motor cortex encodes both discrete phonemes AND continuous transitions between them
- This is exactly what CTC models: the output probability trajectory transitions between phoneme labels
- Validates CTC over frame-wise classification for speech sequences

### Hierarchical syllable→phoneme coding
- Syllables coded 250–350ms before phonemes in planning regions
- Supports hierarchical CTC (Boccato's approach, Phase 3.3 ablation): first decode syllable structure, then phonemes
- Articulatory auxiliary loss (Phase 3.2) is further justified — phoneme codes are organized within syllabic frames

### 52 epilepsy patients are NOT directly poolable
- SEEG/macro-ECoG ≠ uECOG: different spatial resolution, coverage, electrode characteristics
- But they validate the task design across N=52 diverse brains
- Potential for cross-modality pretraining (Phase 3 if ever pursued)

## Reusable Ideas

1. **Cross-task data pooling** — Same phoneme inventory + overlapping patients = free data augmentation
2. **Spatial gradient analysis** — Apply the anterior-posterior timing analysis to learned read-in weights
3. **Syllable auxiliary loss** — Predict CVC vs VCV from GRU hidden states as additional regularizer
4. **Phonotactic transition encoding** — CTC naturally captures this; validate in our model's output behavior
5. **NNMF decomposition** — Apply to learned representations to verify functional specialization
6. **Temporal decoding analysis** — Sliding-window classification to measure when phoneme information appears
7. **Mix-Up augmentation across patients** — Validated here for trial count balancing across patients
