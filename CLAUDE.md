# Cross-Patient Speech Decoding from Intra-Op uECOG

## Project

Ben Tang, Greg Cogan Lab, Duke. Collaborating with Zac Spalding.
Extending Spalding 2025 (PCA+CCA alignment, SVM/Seq2Seq, 8 patients, 9 phonemes, 0.31 bal. acc.).

**Data**: ~10 non-word repetition + ~10 word/nonword patients (~20 total with good data), 128/256-ch uECOG arrays. Spalding's 8 patients: non-word repetition task (52 CVC/VCV tokens, 3 phonemes each, e.g. /abe/; 9 phonemes total). ~8.23 min utterance total, ~68.51 min experiment, 46–178 trials/pt, 63–201 sig channels. Left sensorimotor cortex, intra-operative (acute). All DBS patients are Parkinson's. CTC targets are 3-phoneme non-word sequences (e.g., [a,b,e]), NOT repeated single phonemes. Total pooled SSL corpus: ~3 hrs.

**Cross-task opportunity (Duraivel 2025):** Pseudo-word repetition task (CVC/VCV, 52 tokens, **same 9 phonemes**) performed by overlapping uECOG patients (S8, S5, S1/S2 in Spalding). 156–208 trials/pt. Could nearly double training data for shared patients. Same preprocessing pipeline, same IRB. 52 epilepsy patients (SEEG/macro-ECoG) also performed this task — different modality, not directly poolable.

**Codebase**: Zac's repo at `github.com/coganlab/cross_patient_speech_decoding` (Python-only, MIT). Not yet cloned. Preprocessing done upstream in MATLAB (`coganlab/IEEG_Pipelines`). HGA features are **pre-extracted** as `.mat` files: `{subj}_HG_p{phoneme}_sigChannel_goodTrials.mat` containing `hgTrace` (trials × ch_x × ch_y × time), `hgMap` (trials × time × channels), `phonSeqLabels` (trials × 3). Training data compiled into pickle `pt_decoding_data_S62.pkl`. Patient IDs in code (S14, S23…) differ from paper IDs (S1–S8).

## Key Files

- `docs/implementation_plan.md` — Two-stage implementation plan: Phase 2 = field standard, Phase 3 = two architectural innovations (v9)
- `docs/pipeline_decisions.md` — ~40 design decisions with tensions/tradeoffs
- `docs/research_synthesis.md` — 18-paper synthesis: landscape, gaps, ranked directions (16 archived)
- `docs/future_directions.md` — Cross-task pooling, Tier 2 ideas (separated from implementation plan)
- `docs/reading_list.md` — 10 essential papers in reading order
- `pastwork/paper_index.md` — Paper lookup by contribution category (10 categories)
- `pastwork/summaries/` — 18 active summaries (16 in `pastwork/archive/summaries/`)
- `pastwork/verification_results.md` — All 19 errors across 8 summaries corrected

## Code Structure

```
src/speech_decoding/
├── data/      (phoneme_map, grid, bids_dataset, augmentation, collate)
├── models/    (spatial_conv, linear_readin, backbone, articulatory_head, flat_head, assembler)
├── training/  (trainer, ctc_utils)
└── evaluation/ (metrics)
```

## Data

### Preprocessed HGA (Zac's .fif files — use directly)
Both datasets provide `productionZscore_highgamma.fif` (CAR → HGA 70-150Hz → z-score, 200Hz, all channels).
Per-phoneme epochs locked to response onset, window [-1.0, 1.5s].

**PS (primary)**: `BIDS_1.0_Phoneme_Sequence_uECoG/.../derivatives/epoch(phonemeLevel)(CAR)/sub-{id}/epoch(band)(power)/sub-{id}_task-PhonemeSequence_desc-productionZscore_highgamma.fif`
- 12 usable patients (S18 excluded — no preprocessed): S14, S16, S22, S23, S26, S32, S33, S36, S39, S57, S58, S62
- 8 are Spalding's published set: S14, S22, S23, S26, S33, S39, S58, S62
- 3 phonemes/trial (CVC/VCV), 9 phonemes, 52 tokens
- Mixed label notation: ARPAbet (AA1, EH1, IY0, B...) + lowercase (a, ae, i...) — normalize in phoneme_map.py

**Lexical (secondary)**: `BIDS_1.0_Lexical_µECoG/.../derivatives/epoch(phonemeLevel)(CAR)/sub-{id}/epoch(band)(power)/sub-{id}_task-lexical_desc-productionZscore_highgamma.fif`
- 13 usable patients: S41, S45, S47, S51, S53, S55, S56, S63, S67, S73, S74, S75, S76
- 5 phonemes/trial, 32 phonemes (full ARPABET), filter to 9 PS phonemes for cross-task

### Grid Layouts
- 128ch → 8×16 = 128 (no dead positions)
- 256ch → 12×22 = 264 positions, 256 channels → 8 dead positions (zero in Conv2d)
- Electrode TSV coords are normalized grid positions (0–1), NOT brain MNI coordinates

### Loading Pattern
```python
epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
data = epochs.get_data()        # (n_epochs, n_channels, n_times)
events = epochs.events[:, 2]    # event IDs
event_id = epochs.event_id      # {'label': int_id}
# Per-phoneme: 148 trials × 3 = 444 epochs interleaved [p1_t1, p2_t1, p3_t1, ...]
# Use phonemeIdx=1 (every 3rd) for CTC — window captures full utterance
```

### Zac's Reusable Constants (from code/decoding/decode_bids_crossPatientTask_phonemes.py)
- `PS_PTS = ['S14','S22','S23','S26','S33','S39','S58','S62']`
- `PS2ARPA = {'a':'AA', 'ae':'EH', 'i':'IY', 'u':'UH', 'b':'B', 'p':'P', 'v':'V', 'g':'G', 'k':'K'}`
- `TARGET_EVENT_ID` — 28-phoneme ARPABET→int mapping

## Established Findings (from literature review)

### Architecture consensus + our innovations (v9, 2026-03-17)
Field consensus: per-patient input layer → shared backbone (GRU) → CTC loss. Used by Willett, Metzger, Levin, Boccato, Nason, Singh, BIT. Spalding's PCA+CCA+SVM is outdated.

**Our architecture diverges from field consensus in two ways (architectural only — training is field-standard):**
1. **Per-patient spatial conv** (Conv2d, default 1-layer ~80 params, configurable to 2-layer ~664 params) replaces Linear read-in (~13k params). Conv2d's factorization (weight-shared within image, different across images) matches rigid-array physics (uniform intra-array, variable inter-array). Learns spatial deblurring (Laplacian/gradient) and orientation-adapted filters. Array placements span 15–25mm with variable orientation → spatial conv is per-patient, not shared. **Layer count, channels, pool resolution are empirical — quick-validate on 1 LOPO fold (E13)**
2. **Articulatory decomposition CTC head** replaces flat Linear(2H,9) — 6 parallel articulatory feature heads composed via fixed linguistic matrix. Blank logit initialized with +2.0 bias (phoneme logits are sums of 3-4 features → larger magnitude at init; blank must start competitive for CTC stability)

**Downgraded to exploratory (in `future_directions.md`):** Reptile, SupCon, and DRO. Per-patient layers already provide cross-patient transfer (Singh-style). DRO upweights patients hard for practical reasons (few trials, poor signal), not neural coding differences — noisy with N=7.

### Parameter budget (verified 2026-03-14, updated v9 2026-03-17)
- **Data confirmed from Spalding Tables S1/S3:** Trials 46–178/patient (S3=46 outlier), sig channels 63–201, frames 9.2k–35.6k/pt → **small regime locked**
- Per-patient input: Conv2d(1,8,k=3,pad=1), default 1-layer = **80 params** (167× fewer than Linear), configurable to 2-layer = 664 params. AdaptiveAvgPool2d(2,4) handles different grid sizes. Coarse pooling preferred — 15–25mm placement offsets cause finer pool cells to map to different cortex across patients. **Derivable from physics:** per-patient, Conv2d, coarse pool. **Empirical:** num_layers (1 vs 2), C (4 vs 8), pool dims — swept in E13. **Grid shape note:** 12×22=264 positions but arrays labeled "256ch" — 8 positions unaccounted for (dead corners or reference electrodes); Conv2d needs exact grid dims with dead positions zeroed
- Shared backbone: BiGRU 2×64, ~147k shared params (Conv1d 20.5k + GRU 124k + articulatory head 2k). Temporal downsampling configurable: default k=5 s=5 (40Hz), validated alternative k=10 s=10 (20Hz, Spalding's rate)
- CTC head: 6 articulatory feature heads → fixed composition matrix → 9 phoneme logits + blank. 2,064 params (vs 1,290 flat). Blank bias initialized +2.0 (compensates scale mismatch with composed logits)
- Two-stage LOPO: Stage 1 = standard multi-patient SGD with **held-out 20% source validation** for early stopping; Stage 2 freezes Conv1d+BiGRU, adapts spatial conv+LayerNorm+articulatory head (~2,272 trainable params at default config). **Stage 2 uses stratified 5-fold CV** on target patient (StratifiedKFold on phoneme labels)
- Stage 1 uses best-checkpoint selection (no SWA — no BCI paper uses SWA, adds unprecedented complexity). AdamW + CosineAnnealingLR (single cycle) + early stopping on held-out validation loss
- Systematic ablation: 4 experiments (E0 Spalding + E1 field standard + E2 full model + E3–E4 single-removal)
- Effective augmentation multiplier ~2–3× (not 5×)
- **Compute: ~88 GPU-hours** for 4 experiments × 8 LOPO folds × 3 seeds (each fold: ~30 min Stage 1 + 5×5 min Stage 2 CV)

### Alignment
- Boccato: learned affine transforms are predominantly diagonal (per-channel gain adjustment) — **but this is Utah-specific** (fixed stereotactic placement, same orientation). uECOG arrays span 15–25mm in MNI space with variable rotation — electrode channel numbers have no consistent anatomical meaning across patients. Diagonal per-channel scaling is inapplicable. Cross-patient variation includes spatial remapping, not just gain
- **Electrode placement context (from MNI projection):** Arrays cluster over left perisylvian/sensorimotor cortex but are substantially offset (15–25mm) and rotated. Dense central overlap zone where most arrays share coverage; peripheral coverage is patient-specific. Some arrays barely overlap with the most displaced others. This means: (a) per-channel gain is insufficient for alignment, (b) Conv2d's weight sharing (same filter everywhere on one patient's grid) correctly matches rigid-array uniformity, (c) per-patient filters adapt to each array's unique orientation, (d) coarser spatial pooling is preferred for cross-patient robustness
- Singh: freezing shared LSTM + fine-tuning per-patient Conv1D + readout is optimal transfer mode
- BIT Table 9: SSL ≈ supervised for same-subject at equal data size. **The SSL advantage is specifically cross-subject, not inherent.** BIT's cross-subject supervised failure (Appendix M) is qualitative only (no numbers), at N=2, with heterogeneous setups, and without backbone freezing
- **Critical nuance**: Singh's supervised *joint training* with per-patient layers at N=25 DOES work (group PER 0.49 vs single 0.57). Our N≈20 with homogeneous uECOG is closer to Singh than BIT. Per-patient layers are the critical component
- Levin 2026: 30% source replay during Stage 2 fine-tuning prevents catastrophic forgetting. Directly adoptable
- Articulatory auxiliary loss (predicting phonological features) provides cross-patient-invariant signal at zero data cost — Duraivel showed errors track articulatory distance
- No cross-patient method tested on uECOG beyond Spalding's CCA baseline. Chen 2025 SwinTW is closest (standard ECoG, 1cm spacing) but not uECOG

### Data regime
- **Confirmed:** ~8.23 min total utterance across 8 patients (~1 min/pt), ~68.51 min total experiment. ~3 hrs pooled across ~20 patients
- **Paired audio confirmed:** Lavalier mic synced at 20 ksps (Intan) + 44.1 kHz (laptop), phoneme labels via MFA + Audacity. Speech FM projection is feasible
- **MNI coordinates confirmed:** MNI-152 for all patients (BioImage Suite for DBS; Brainlab for tumor). Phase 3.1F/G ablations feasible
- wav2vec ECoG: SSL works with ~1 hr/patient. We have ~1 min/patient utterance — much thinner
- Nason: ~36 sentences (~6 min) reaches **>80% of peak** performance (NOT near-full; 123 sentences = 95% of peak). Day-specific input layer likely 512→512 softsign (~262k params/day)

### External pretraining
- **No existing dataset transfers directly**: spike-based corpora (BIT, NDT3, POSSM) use Poisson statistics incompatible with Gaussian HGA
- **Speech FM alignment — diagnostic-first design**: Motor cortex → speech FM mapping is untested (Dual Pathway validated STG only; Chang lab uses broad SMC+STG but never isolates motor contribution; Stavisky lab bypasses speech FMs for motor cortex). Our design: post-hoc linear probe on frozen E2 backbone → PCA-reduced HuBERT features (segment-level, MFA-aligned). If R² significant, add segment-level MSE aux loss as E13. Paired audio confirmed for all 8 patients. See `future_directions.md` § "Speech foundation model alignment" for full analysis
- **Articulatory knowledge is the highest bang-for-buck external source**: phonological feature vectors from linguistic databases, zero neural data needed, cross-patient invariant by construction
- **Architecture patterns transfer without weights**: MRoPE, wav2vec framework, cross-attention, hierarchical CTC
- **Internal SSL on ~3 hrs is borderline**: masked reconstruction preferred over contrastive (no negatives, MIBRAIN validated on raw broadband sEEG — NOT HGA). wav2vec ECoG is wav2vec **1.0** (CPC, not 2.0 — no quantizer). Minimum successful SSL corpus is ~30 min (participant b); our ~1 min/pt is 30x smaller. Per-patient input layer bottleneck: only ~10 min/patient vs wav2vec ECoG's 30-60 min/patient

### What doesn't apply to us
- Spike-based foundation models (BIT, NDT3): different signal modality (Poisson spikes vs Gaussian HGA envelopes). Transfer untested
- Large-scale SSL: BIT needed 367 hrs, NDT3 needed 2000 hrs. We have ~3 hrs total
- Supervised cross-subject pretraining without per-patient layers: BIT Appendix M (qualitative, N=2). With per-patient layers (Singh N=25), supervised works
- **Training protocol gaps**: Singh/Levin don't report optimizer, LR, or batch size. Our differential LR is our own design choice

### Key gaps we can fill
1. No alignment method bake-off on same dataset (every paper only tests its own method)
2. No cross-task alignment (phoneme + pseudo-word patients — Duraivel 2025 provides the pseudo-word task data with same 9 phonemes and overlapping patients)
3. No 2D conv exploiting uECOG grid structure (8x16, 12x22 regular grids) — our 2-layer Conv2d is engineering (matches rigid-array physics), not a radical innovation
4. No SSL on uECOG HGA specifically
5. No LoRA/adapters for neural signal encoders
6. Motor cortex → speech FM correspondence untested (Dual Pathway validated auditory cortex only; Chang lab's broad coverage never isolates motor contribution; Stavisky bypasses speech FMs) — planned as post-hoc diagnostic on E2
7. Articulatory feature vectors as explicit cross-patient alignment target never implemented

## Preprocessing Pipeline (do not change)

Decimate 2kHz → CAR → impedance exclusion (log10(impedance) > 6) → 70-150Hz Gaussian filterbank (8 bands, 1/7 octave spacing) → Hilbert envelope → **sum** across bands (not average) → 200Hz → z-score baseline normalize → significant channel selection (stored in `sigChannel.mat`, from permutation cluster test). Implemented in `coganlab/IEEG_Pipelines` (Python: `ieeg.timefreq.gamma.extract()`; MATLAB: `extractHiGamma()`). Validated in Duraivel 2023, Duraivel 2025, Spalding 2025.

## Conventions

- Ben prefers exhaustive understanding before implementation — explain every design choice with precedent and tradeoffs
- Batch work in smaller chunks to prevent context rot
- Keep markdown files lean and information-dense, no filler
- Report per-patient results, not just population means
- Use TME surrogate control to validate alignment isn't artificial
