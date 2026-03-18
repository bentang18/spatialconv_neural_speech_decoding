# Cross-Patient Speech Decoding from Intra-Op uECOG

## Project

Ben Tang, Greg Cogan Lab, Duke. Collaborating with Zac Spalding.
Extending Spalding 2025 (PCA+CCA alignment, SVM/Seq2Seq, 8 patients, 9 phonemes, 0.31 bal. acc.).

**Data**: ~10 non-word repetition + ~10 word/nonword patients (~20 total with good data), 128/256-ch uECOG arrays. Spalding's 8 patients: non-word repetition task (52 CVC/VCV tokens, 3 phonemes each, e.g. /abe/; 9 phonemes total). ~8.23 min utterance total, ~68.51 min experiment, 46–178 trials/pt, 63–201 sig channels. Left sensorimotor cortex, intra-operative (acute). All DBS patients are Parkinson's. CTC targets are 3-phoneme non-word sequences (e.g., [a,b,e]), NOT repeated single phonemes. Total pooled SSL corpus: ~3 hrs.

**Cross-task opportunity (Duraivel 2025):** Pseudo-word repetition task (CVC/VCV, 52 tokens, **same 9 phonemes**) performed by overlapping uECOG patients (S8, S5, S1/S2 in Spalding). 156–208 trials/pt. Could nearly double training data for shared patients. Same preprocessing pipeline, same IRB. 52 epilepsy patients (SEEG/macro-ECoG) also performed this task — different modality, not directly poolable.

**Codebase**: Zac's repo at `github.com/coganlab/cross_patient_speech_decoding` (Python-only, MIT). Not yet cloned. Preprocessing done upstream in MATLAB (`coganlab/IEEG_Pipelines`). HGA features are **pre-extracted** as `.mat` files: `{subj}_HG_p{phoneme}_sigChannel_goodTrials.mat` containing `hgTrace` (trials × ch_x × ch_y × time), `hgMap` (trials × time × channels), `phonSeqLabels` (trials × 3). Training data compiled into pickle `pt_decoding_data_S62.pkl`. Patient IDs in code (S14, S23…) differ from paper IDs (S1–S8).

## Key Files

### Documentation
- `docs/implementation_plan.md` — Two-stage implementation plan: Phase 2 = field standard, Phase 3 = two architectural innovations (v9)
- `docs/pipeline_decisions.md` — ~40 design decisions with tensions/tradeoffs
- `docs/research_synthesis.md` — 18-paper synthesis: landscape, gaps, ranked directions (16 archived)
- `docs/future_directions.md` — Cross-task pooling, Tier 2 ideas (separated from implementation plan)
- `docs/reading_list.md` — 10 essential papers in reading order
- `pastwork/paper_index.md` — Paper lookup by contribution category (10 categories)
- `pastwork/summaries/` — 18 active summaries (16 in `pastwork/archive/summaries/`)
- `pastwork/verification_results.md` — All 19 errors across 8 summaries corrected

### Configs
- `configs/default.yaml` — E2 full model (spatial conv + articulatory CTC)
- `configs/field_standard.yaml` — E1 field standard (linear + flat CTC)
- `configs/paths.yaml` — Machine-specific BIDS paths (gitignored)

## Code Structure

```
src/speech_decoding/
├── data/
│   ├── phoneme_map.py      # 9 PS phonemes, PS2ARPA normalization, CTC encoding, articulatory matrix (9×15)
│   ├── grid.py             # Electrode TSV → grid shape + channel-to-grid mapping (handles 8×16, 12×22, 8×32, 8×34)
│   ├── bids_dataset.py     # Load .fif epochs → position-1 extraction → grid reshape → BIDSDataset
│   ├── augmentation.py     # Pre-read-in: time shift, amplitude scale, channel dropout, Gaussian noise
│   └── collate.py          # Group samples by patient_id for multi-grid batching
├── models/
│   ├── spatial_conv.py     # Per-patient Conv2d read-in: (B,H,W,T)→(B,64,T), 80 params default
│   ├── linear_readin.py    # Per-patient Linear read-in: (B,D_padded,T)→(B,64,T), ~13k params
│   ├── backbone.py         # LayerNorm → Conv1d(k=5,s=5) → BiGRU(64,64,2L) with feat dropout + time mask
│   ├── articulatory_head.py # 6 feature heads → fixed A matrix → 9 phonemes + blank, 2,064 params
│   ├── flat_head.py        # Linear(128,10) → log_softmax, 1,290 params
│   └── assembler.py        # YAML config → (backbone, head, {patient_id: read_in})
├── training/
│   ├── ctc_utils.py        # CTC loss (T,B,C transpose), greedy decode, PER, blank ratio
│   └── trainer.py          # Per-patient training: stratified K-fold CV, AdamW + CosineAnnealingLR, early stopping
└── evaluation/
    └── metrics.py          # PER, per-position balanced accuracy, CTC length accuracy
```

### Scripts
- `scripts/train_per_patient.py` — CLI for per-patient training on all PS patients × seeds

### Tests (110 total: 100 fast + 10 slow)
```
tests/
├── test_phoneme_map.py     # 23 tests: label normalization, CTC encoding, articulatory matrix
├── test_grid.py            # 10 tests: grid inference from TSV, dead positions, reshape (2 slow)
├── test_bids_dataset.py    # 9 tests: dataset interface, real .fif loading (6 slow)
├── test_augmentation.py    # 14 tests: all augmentation ops
├── test_collate.py         # 4 tests: multi-patient grouping
├── test_models.py          # 24 tests: all model components + assembler
├── test_ctc_utils.py       # 13 tests: CTC loss, decode, PER, blank ratio
├── test_trainer.py         # 6 tests: metrics + per-patient trainer on synthetic data
└── test_integration.py     # 7 tests: end-to-end forward/backward/overfit + real S14 (2 slow)
```

Run: `pytest tests/ -v -m "not slow"` (fast, no data needed) or `pytest tests/ -v` (all, needs BIDS data)

## Data

### Preprocessed HGA (Zac's .fif files — use directly)
Both datasets provide `productionZscore_highgamma.fif` (CAR → HGA 70-150Hz → z-score, 200Hz, all channels).
Per-phoneme epochs locked to response onset, window [-1.0, 1.5s].

**PS (primary)**: `BIDS_1.0_Phoneme_Sequence_uECoG/.../derivatives/epoch(phonemeLevel)(CAR)/sub-{id}/epoch(band)(power)/sub-{id}_task-PhonemeSequence_desc-productionZscore_highgamma.fif`
- 12 usable patients (S18 excluded — no preprocessed): S14, S16, S22, S23, S26, S32, S33, S36, S39, S57, S58, S62
- 8 are Spalding's published set: S14, S22, S23, S26, S33, S39, S58, S62
- 3 phonemes/trial (CVC/VCV), 9 phonemes, 52 tokens
- PS labels use lowercase notation (`event_id`: `{'a':1, 'ae':2, 'b':3, 'g':4, 'i':5, 'k':6, 'p':7, 'u':8, 'v':9}`) — `phoneme_map.normalize_label()` handles conversion

**Lexical (secondary)**: `BIDS_1.0_Lexical_µECoG/.../derivatives/epoch(phonemeLevel)(CAR)/sub-{id}/epoch(band)(power)/sub-{id}_task-lexical_desc-productionZscore_highgamma.fif`
- 13 usable patients: S41, S45, S47, S51, S53, S55, S56, S63, S67, S73, S74, S75, S76
- 5 phonemes/trial, 28 phonemes (ARPABET without stress), e.g. `{'AA':1, 'AE':2, ..., 'Z':28}`
- **Cross-task filtering yields 0 trials** — no 5-phoneme English word uses only PS phonemes. Cross-task pooling requires per-position approach (future work)

### Grid Layouts (confirmed from electrode TSV inspection)
Grid shape is inferred from electrode coordinate TSVs, NOT channel count — multiple layouts exist for 256ch:

| Channels | Grid | Dead positions | Patients |
|----------|------|----------------|----------|
| 128 | 8×16 = 128 | 0–1 (S14 ch 105 has n/a coords) | S14, S16, S22, S23, S26, S36, S45, S47, S51, S53 |
| 256 | 12×22 = 264 | 8 (corners of top/bottom row) | S32, S33, S39, S58, S62, S41, S56, S67, S75 |
| 256 | 8×32 = 256 | 0–1 | S55, S63, S73, S74, S76 |
| 256 | 8×34 = 272 | 16 | S57 |

- Electrode TSV coords are normalized grid positions (0–1, z=0), NOT brain MNI coordinates
- TSV files have BOM (`\ufeff`) — `grid.py` opens with `encoding="utf-8-sig"`
- Dead positions zeroed in Conv2d input; `grid.load_grid_mapping()` returns `GridInfo` with `dead_mask`

### Loading Pattern
```python
# Via our loader (recommended):
from speech_decoding.data.bids_dataset import load_patient_data
ds = load_patient_data("S14", bids_root, task="PhonemeSequence", n_phons=3, tmin=-0.5, tmax=1.0)
# ds[i] → (grid_data[H,W,T], ctc_label[list[int]], patient_id)

# Raw MNE (reference):
epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
data = epochs.get_data()        # (n_epochs, n_channels, n_times)
# Per-phoneme: 153 trials × 3 = 459 epochs (S14) interleaved [p1_t1, p2_t1, p3_t1, ...]
# Position-1 extraction: data[0::3] gives trial-level epochs containing full utterance
```

### Zac's Codebase (reference only — our code replaces the decoding pipeline)
Located at `BIDS_1.0_Lexical_µECoG/.../BIDS/code/decoding/`. Key files:
- `decode_bids_crossPatientTask_phonemes.py` — source of `PS_PTS`, `PS2ARPA`, `TARGET_EVENT_ID` constants
- `PhonemeDatasetBIDS.py` — reference for .fif loading pattern (adapted into `bids_dataset.py`)
- `cross_pt_decoders.py`, `AlignCCA.py` — PCA+CCA pipeline we replace with Conv2d+CTC

## Implementation Status

### Complete (Sprints 0–4)
- **Sprint 0**: Project setup (uv, Python 3.11 venv, git, pyproject.toml, directory reorganization)
- **Sprint 1**: Data foundation — phoneme mapping, grid inference, BIDS loading, augmentation, collation
- **Sprint 2**: Model components — both E1 (Linear+Flat) and E2 (SpatialConv+Articulatory) architectures
- **Sprint 3**: Per-patient training — CTC utils, stratified CV trainer, evaluation metrics
- **Sprint 4**: Integration tests — synthetic overfit, real S14 forward pass + quick train

### Next (not yet implemented)
- **Sprint 5**: Cross-patient LOPO — multi-patient Stage 1 (gradient accumulation, held-out validation, differential LR), Stage 2 target adaptation (freeze backbone, source replay), LOPO orchestrator with Wilcoxon statistics
- **Sprint 6**: Cross-task pooling — Lexical patients as additional Stage 1 sources (requires per-position phoneme filtering, not per-trial)

## Established Findings (from literature review)

### Architecture consensus + our innovations (v9, 2026-03-17)
Field consensus: per-patient input layer → shared backbone (GRU) → CTC loss. Used by Willett, Metzger, Levin, Boccato, Nason, Singh, BIT. Spalding's PCA+CCA+SVM is outdated.

**Our architecture diverges from field consensus in two ways (architectural only — training is field-standard):**
1. **Per-patient spatial conv** (Conv2d, default 1-layer ~80 params, configurable to 2-layer ~664 params) replaces Linear read-in (~13k params). Conv2d's factorization (weight-shared within image, different across images) matches rigid-array physics (uniform intra-array, variable inter-array). Learns spatial deblurring (Laplacian/gradient) and orientation-adapted filters. Array placements span 15–25mm with variable orientation → spatial conv is per-patient, not shared. **Layer count, channels, pool resolution are empirical — quick-validate on 1 LOPO fold (E13)**
2. **Articulatory decomposition CTC head** replaces flat Linear(2H,9) — 6 parallel articulatory feature heads composed via fixed linguistic matrix. Blank logit initialized with +2.0 bias (phoneme logits are sums of 3-4 features → larger magnitude at init; blank must start competitive for CTC stability)

**Downgraded to exploratory (in `future_directions.md`):** Reptile, SupCon, and DRO. Per-patient layers already provide cross-patient transfer (Singh-style). DRO upweights patients hard for practical reasons (few trials, poor signal), not neural coding differences — noisy with N=7.

### Parameter budget (verified 2026-03-14, updated v9 2026-03-17, grid shapes confirmed 2026-03-17)
- **Data confirmed from Spalding Tables S1/S3:** Trials 46–178/patient (S3=46 outlier), sig channels 63–201, frames 9.2k–35.6k/pt → **small regime locked**
- Per-patient input: Conv2d(1,8,k=3,pad=1), default 1-layer = **80 params** (167× fewer than Linear), configurable to 2-layer = 664 params. AdaptiveAvgPool2d(2,4) handles different grid sizes. Coarse pooling preferred — 15–25mm placement offsets cause finer pool cells to map to different cortex across patients. **Derivable from physics:** per-patient, Conv2d, coarse pool. **Empirical:** num_layers (1 vs 2), C (4 vs 8), pool dims — swept in E13. **Grid shapes confirmed from electrode TSVs:** 12×22 (8 dead corners), 8×32 (no dead), 8×34 (S57, 16 dead). Shape must be inferred from TSV, not channel count. S14 ch 105 has n/a coords (bad electrode). All handled by `grid.load_grid_mapping()`
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
