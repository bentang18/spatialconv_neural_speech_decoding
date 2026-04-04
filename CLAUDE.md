# Cross-Patient Speech Decoding from Intra-Op uECOG

## Project

Ben Tang, Greg Cogan Lab, Duke. Collaborating with Zac Spalding.
Extending Spalding 2025 (PCA+CCA alignment, SVM/Seq2Seq, 8 patients, 9 phonemes, 0.31 bal. acc.).

**Data**: ~10 non-word repetition + ~10 word/nonword patients (~20 total with good data), 128/256-ch uECOG arrays. Spalding's 8 patients: non-word repetition task (52 CVC/VCV tokens, 3 phonemes each, e.g. /abe/; 9 phonemes total). ~8.23 min utterance total, ~68.51 min experiment, 46–178 trials/pt, 63–201 sig channels. Left sensorimotor cortex, intra-operative (acute). All DBS patients are Parkinson's. CTC targets are 3-phoneme non-word sequences (e.g., [a,b,e]), NOT repeated single phonemes. Total pooled SSL corpus: ~3 hrs.

**Cross-task opportunity (Duraivel 2025):** Pseudo-word repetition task (CVC/VCV, 52 tokens, **same 9 phonemes**) performed by overlapping uECOG patients (S8, S5, S1/S2 in Spalding). 156–208 trials/pt. Could nearly double training data for shared patients. Same preprocessing pipeline, same IRB. 52 epilepsy patients (SEEG/macro-ECoG) also performed this task — different modality, not directly poolable.

**Codebase**: Zac's repo at `github.com/coganlab/cross_patient_speech_decoding` (Python-only, MIT). Not yet cloned. Preprocessing done upstream in MATLAB (`coganlab/IEEG_Pipelines`). HGA features are **pre-extracted** as `.mat` files: `{subj}_HG_p{phoneme}_sigChannel_goodTrials.mat` containing `hgTrace` (trials × ch_x × ch_y × time), `hgMap` (trials × time × channels), `phonSeqLabels` (trials × 3). Training data compiled into pickle `pt_decoding_data_S62.pkl`. Patient IDs in code (S14, S23…) differ from paper IDs (S1–S8).

## Key Files

### Documentation
- `docs/neural_field_perceiver_v12.tex` — **Active design document**: Neural Field Perceiver architecture (MNI-based spatial tokenization + cross-attention + reconstruction loss)
- `docs/current_direction.md` — Active guardrail: current priorities, what's archived, practical rules
- `docs/dcc_setup.md` — Complete Duke DCC cluster documentation (SSH, conda, data, SBATCH, troubleshooting)
- `docs/experiment_log.md` — Full experiment history (55 LOPO + per-patient + SSL)
- `docs/training_log.md` — Per-patient tuning history (S14)
- `docs/research_synthesis.md` — 18-paper synthesis: landscape, gaps, ranked directions
- `docs/reading_list.md` — 10 essential papers in reading order
- `pastwork/paper_index.md` — Paper lookup by contribution category (10 categories)
- `pastwork/summaries/` — 18 active summaries (16 in `pastwork/archive/summaries/`)

### Archived Documentation
- `docs/archive/neural_field_perceiver_versions/` — v2-v11, beta (design doc evolution)
- `docs/archive/nca_jepa/` — SSL pretraining design specs, plans (Phase 0 complete, paused)
- `docs/archive/lopo_autoresearch/` — LOPO brainstorm and autoresearch plans
- `docs/archive/historical_supervised/` — Old implementation plans, pipeline decisions, future directions

### Configs
- `configs/per_patient_ce_s10_pool48.yaml` — Current recommended per-patient config (CE, stride=10, pool(4,8))
- `configs/lopo_ce.yaml` — LOPO cross-patient config
- `configs/paths.yaml` — Machine-specific BIDS paths (gitignored)
- `configs/archive/` — Historical baselines, sweeps, negative pivots, pretrain configs

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
│   ├── trainer.py          # Per-patient training: stratified K-fold CV, AdamW + cosine LR, early stopping
│   ├── lopo_trainer.py     # Stage 1: multi-patient SGD with gradient accumulation, step-based training
│   ├── adaptor.py          # Stage 2: target adaptation with frozen backbone, source replay, 5-fold CV
│   └── lopo.py             # LOPO orchestrator: loops targets × seeds, collects metrics, Wilcoxon
├── evaluation/
│   ├── metrics.py          # PER, per-position balanced accuracy, CTC length accuracy
│   ├── grouped_cv.py       # Grouped-by-token CV splitter (NCA-JEPA Phase 0)
│   └── content_collapse.py # Entropy, unigram-KL, stereotypy diagnostics (NCA-JEPA Phase 0)
└── pretraining/            # NCA-JEPA pretraining (ARCHIVED — on disk but not actively used)
    ├── pretrain_model.py   # UnifiedPretrainModel: spatial_mode ∈ {collapse, preserve, attend}
    ├── masking.py          # Span masking + learnable [MASK] token
    ├── decoder.py          # Linear reconstruction decoder
    ├── stage1_trainer.py   # Synthetic pretraining loop
    ├── stage2_trainer.py   # Neural adaptation loop
    ├── stage3_evaluator.py # Freeze backbone → CE fine-tune → PER + collapse diagnostics
    ├── synthetic_pipeline.py # Synthetic data generation
    └── generators/         # Smooth AR + switching LDS generators
```

### Scripts
- `scripts/train_per_patient.py` — CLI for per-patient training on all PS patients × seeds
- `scripts/train_lopo.py` — CLI for LOPO cross-patient training
- `scripts/sweep_tmin_perpos.py` — 6-condition temporal windowing sweep (tmin × head_type × per-phoneme)
- `scripts/sweep_tmin_dcc.sh` — SBATCH wrapper for sweep on DCC
- `scripts/analyze_real_data_stats.py` — Data statistics analysis
- `scripts/visualize_ecog.py` — ECoG visualization
- `scripts/archive/` — SSL evaluation, diagnostics, autoresearch experiments (83 experiments)

### Tests
```
tests/
├── test_phoneme_map.py      # 23 tests: label normalization, CTC encoding, articulatory matrix
├── test_grid.py             # 10 tests: grid inference from TSV, dead positions, reshape (2 slow)
├── test_bids_dataset.py     # 9 tests: dataset interface, real .fif loading (6 slow)
├── test_augmentation.py     # 14 tests: all augmentation ops
├── test_collate.py          # 4 tests: multi-patient grouping
├── test_models.py           # 24 tests: all model components + assembler
├── test_ctc_utils.py        # 13 tests: CTC loss, decode, PER, blank ratio
├── test_trainer.py          # 6 tests: metrics + per-patient trainer on synthetic data
├── test_integration.py      # 7 tests: end-to-end forward/backward/overfit + real S14 (2 slow)
├── test_lopo_trainer.py     # 7 tests: Stage 1 multi-patient training (synthetic)
├── test_adaptor.py          # 6 tests: Stage 2 target adaptation (synthetic)
├── test_lopo.py             # 4 tests: LOPO orchestrator + Wilcoxon (synthetic)
├── test_grouped_cv.py       # Grouped-by-token CV splitter
├── test_content_collapse.py # Content-collapse diagnostics
└── (NCA-JEPA tests)         # test_masking, test_pretrain_model, test_stage*_trainer, etc.
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

### Complete (Sprints 0–5)
- **Sprint 0**: Project setup (uv, Python 3.11 venv, git, pyproject.toml, directory reorganization)
- **Sprint 1**: Data foundation — phoneme mapping, grid inference, BIDS loading, augmentation, collation
- **Sprint 2**: Model components — both E1 (Linear+Flat) and E2 (SpatialConv+Articulatory) architectures
- **Sprint 3**: Per-patient training — CTC utils, stratified CV trainer, evaluation metrics
- **Sprint 4**: Integration tests — synthetic overfit, real S14 forward pass + quick train
- **Sprint 5**: Cross-patient LOPO — Stage 1 multi-patient trainer (`lopo_trainer.py`), Stage 2 target adaptation with source replay (`adaptor.py`), LOPO orchestrator with Wilcoxon (`lopo.py`), `augment_from_config` helper, `train_lopo.py` CLI

### Per-Patient Results (S14, see `docs/training_log.md`)

**CTC tuning (converged):**
- **Best CTC**: H=32, stride=5, blank_bias=0.0, original augmentation → **PER=0.778** (mean 3 seeds)
- Blank bias is the dominant CTC lever: 2.0→0.0 took PER from 1.000 to 0.778. +2.0 causes blank collapse per-patient.
- Flat head ≈ articulatory head per-patient: both ~0.77. Articulatory value is cross-patient transfer only.
- Augmentation/LR/weight-decay/dropout sweeps: all within noise. CTC per-patient is data-limited.

**CE loss ablation (new standard for per-patient):**
- **CE per-position beats CTC**: PER 0.700 vs 0.778 (10% relative), bal_acc 0.266 vs 0.15 (77% relative)
- CE provides stronger per-position gradients, no blank collapse, no blank bias tuning needed
- CTC's alignment-freedom is wasted capacity with small data — phoneme positions are already known from epoch structure
- **Use `training.loss_type: ce` for per-patient.** Keep CTC for LOPO until tested.

**Architecture bottleneck identified:**
- **GRU capacity is NOT the bottleneck**: H=128 CE showed worse overfitting (train/val gap 55× vs 4× for H=32)
- **Input layer compression IS the bottleneck**: Pool(2,4) → 64 dims destroys articulatory spatial resolution (4mm cells can't resolve 3-5mm somatotopy)
- **Pool(4,8) → d_shared=256** retains 2mm spatial resolution, matching somatotopic scale
- **Stride=10 (20Hz) matches field standard**: 40Hz was unnecessary (0.70 vs 0.70 PER). Spalding/Duraivel/Willett all use ≤20Hz.

**Per-patient results under different CV types** (all S14):
- **Grouped-by-token CV** (fair, no token leakage): PER **0.737** (full recipe: CE+focal+mixup+weighted k-NN+TTA 16+articulatory head)
- **Stratified CV** (leaky, tokens repeat across folds): PER 0.700 (CE only) / 0.662 (full recipe)
- **Gap**: 0.075 PER (10pp) from token leakage. Always use grouped-by-token CV for fair evaluation.
- **Grouped-by-token CE-only baseline**: PER 0.825. Recipe improvements (k-NN, TTA, articulatory head) contribute ~8.8pp.

**Recommended per-patient config**: CE loss, stride=10, pool(4,8), C=8, d_shared=256, H=32, articulatory bottleneck head, weighted k-NN eval, TTA 16.

**MPS compatibility** (local Mac only): `AdaptiveAvgPool2d` falls back to CPU for non-divisible grid sizes. CTC loss also falls back to CPU. Use `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Compute: Duke DCC Cluster (primary)

**Use DCC for all training runs.** Local MPS is for editing code only. See `docs/dcc_setup.md` for complete documentation.

- **SSH**: `ssh ht203@dcc-login.oit.duke.edu` (no MFA currently)
- **GPU**: 8× RTX 5000 Ada Generation (32 GB VRAM) on `coganlab-gpu` partition
- **Python**: `/work/ht203/miniconda3/envs/speech/bin/python` (do NOT use `conda activate` — broken base conda)
- **Repo**: `/work/ht203/repo/speech` (branch `autoresearch/run1`)
- **Data**: `/work/ht203/data/BIDS` (symlinked from coganlab)
- **Logs**: `/work/ht203/logs/`
- **Submit**: `sbatch scripts/sweep_tmin_dcc.sh` | Monitor: `squeue -u ht203`
- **CAUTION**: `/work/ht203` auto-purges after 75 days of no access. Copy results to `/hpc/group/coganlab/ht203/`.

### LOPO Pilot Results (complete)
**Config**: `lopo_pilot.yaml` — 8 Spalding patients, 1 seed (42), H=64, CTC, blank_bias=1.0

| Patient | PER |
|---------|-----|
| S14 | 0.828 |
| S22 | 0.851 |
| S23 | 0.849 |
| S26 | 0.846 |
| S33 | 0.853 |
| S39 | 0.856 |
| S58 | 0.854 |
| S62 | 0.832 |
| **Population** | **0.846 ± 0.010** |

- LOPO pilot (0.846) was near-chance. Subsequent recipe improvements (CE, focal, mixup, k-NN, TTA) reduced to 0.764, then architecture ablations to 0.750.
- Stage 1 consistently early-stops ~step 800 of 2000 (val diverges after step 500)
- S33 (52 trials) survived with inner stratification fix
- Very low variance across patients (0.828-0.856) — model is near-chance uniformly

### LOPO Autoresearch Results (2026-03-31)
**Setup**: 9 source patients (~1315 trials) → S1 multi-patient train → S2 adapt to S14, 5-fold grouped-by-token CV.
**Recipe**: CE + focal γ=2 + label smoothing 0.1 + mixup α=0.2 + per-position heads + dropout 0.3 + articulatory bottleneck head + weighted k-NN (k=10) + TTA 16 + multi-patient k-NN (source weight 0.5).

**Progression** (55 experiments across 5 waves, all S14 grouped-by-token CV):
- Pilot (CTC, no recipe): 0.846
- LOPO baseline (CE recipe): 0.764
- Multi-patient k-NN (exp13): 0.762
- Architecture ablations (exp41-76, 36 experiments): best single-seed 0.749 (MaxPool), 0.749 (stride=5); multi-seed means ~0.760-0.764
- **Wave 4 best: 0.750** (exp94: multi-scale temporal stride 3+5+10 + MaxPool read-in)

**Per-patient baselines for comparison** (same S14, same grouped-by-token CV):
- Per-patient (full recipe, exp17): **0.737** — LOPO is 1.3pp worse with best architecture (0.750)
- Per-patient (simple recipe, exp88): 0.800 — LOPO with any recipe beats this
- Per-patient (stratified CV, historical): 0.700 — NOT comparable (different CV type)

**Wave 4 findings** (14 experiments exp77-90 + 5 combinations exp91-95):
- Multi-scale temporal (stride 3+5+10): PER 0.757, 3-seed mean 0.760 ± 0.005
- Self-training with pseudo-labels: PER 0.758 (doesn't stack with multi-scale)
- **ALL SSL auxiliary losses failed**: VICReg (0.778), Transformer+masked SSL (0.776)
- **ALL domain adaptation failed or ≈ baseline**: DANN (0.778), CORAL (0.762), CCA on features (0.771), patient weighting (0.782)
- Joint training with S14 in S1 (Singh-style) hurt: 0.793
- Knowledge distillation from per-patient teachers: 0.773
- S2 contributes ~1.6pp (no-S2: 0.778 vs baseline: 0.762)

**Convergence analysis**: 55 LOPO experiments across every paradigm (supervised, SSL, domain adaptation, self-training, meta-learning) converge to PER 0.750-0.780. Root causes: (1) Fixed CV fold structure — fold 5 consistently ~0.85, fold 4 ~0.70; fold difficulty dominates experiment differences. (2) ~30 val samples per fold → ±0.05 PER noise per fold → mean PER differences <2pp are within measurement noise. (3) S1 backbone converges to same features regardless of training method. (4) S2 adaptation bottlenecked by 120 target samples.

**Breaking through requires**: More data (cross-task pooling, more patients), population-level evaluation (all patients not just S14), or fundamentally different approach (Spalding-style CCA + our features untested with full eval recipe).

See `docs/experiment_log.md` for full results.

### NCA-JEPA Pretraining (ARCHIVED — Phase 0 complete, paused)
SSL objectives failed at this data scale (~1 min utterance/patient). Code on disk in `src/speech_decoding/pretraining/`, docs in `docs/archive/nca_jepa/`.
- ALL tested SSL methods near-chance under grouped CV: BYOL, JEPA, masked span, LeWM, VICReg
- Not worth pursuing without fundamentally more data

### Next Direction: Neural Field Perceiver (v12)
**Design doc**: `docs/neural_field_perceiver_v12.tex`

Cross-patient architecture using MNI electrode coordinates instead of per-patient Conv2d:
- **Fourier PE** on MNI coords → spatial tokenization
- **Cross-attention** to fixed virtual electrodes (Brainnetome atlas) → shared spatial representation
- **Reconstruction loss** → spatial supervision (predict HGA from latent + position)
- **Learned temporal attention queries** for per-phoneme readout (existing `CEPositionHead`)

Implementation spec not yet written. See `docs/current_direction.md` for full context.

## Established Findings (from literature review)

### Architecture consensus + our innovations (v9, 2026-03-17)
Field consensus: per-patient input layer → shared backbone (GRU) → CTC loss. Used by Willett, Metzger, Levin, Boccato, Nason, Singh, BIT. Spalding's PCA+CCA+SVM is outdated.

**Our architecture diverges from field consensus in two ways (architectural only — training is field-standard):**
1. **Per-patient spatial conv** (Conv2d, default 1-layer ~80 params, configurable to 2-layer ~664 params) replaces Linear read-in (~13k params). Conv2d's factorization (weight-shared within image, different across images) matches rigid-array physics (uniform intra-array, variable inter-array). Learns spatial deblurring (Laplacian/gradient) and orientation-adapted filters. Array placements span 15–25mm with variable orientation → spatial conv is per-patient, not shared. **Layer count, channels, pool resolution are empirical — quick-validate on 1 LOPO fold (E13)**
2. **Articulatory decomposition CTC head** replaces flat Linear(2H,9) — 6 parallel articulatory feature heads composed via fixed linguistic matrix. Blank logit bias is configurable (`model.blank_bias`): +2.0 for LOPO (sufficient gradient signal from 7 sources), 0.0 for per-patient (small data can't overcome high blank bias → blank collapse). Per-patient ablation shows **flat head ≈ articulatory head** (both PER ~0.77) — articulatory value is cross-patient transfer only

**Downgraded to exploratory:** Reptile, SupCon, and DRO. Per-patient layers already provide cross-patient transfer (Singh-style). DRO upweights patients hard for practical reasons (few trials, poor signal), not neural coding differences — noisy with N=7.

### Parameter budget (updated 2026-03-18 from first-principles analysis)
- **Data confirmed from Spalding Tables S1/S3:** Trials 46–178/patient (S3=46 outlier), sig channels 63–201, frames 9.2k–35.6k/pt → **small regime locked**
- Per-patient input: Conv2d(1,8,k=3,pad=1), 1-layer = **80 params**. **Pool(4,8) preferred over pool(2,4)** — 2mm spatial resolution resolves 3-5mm somatotopic organization; pool(2,4) at 4mm cannot. d_shared=256 (up from 64). For LOPO cross-patient, coarser pooling may still be preferable (array offsets 15-25mm). **Grid shapes from TSVs:** 12×22 (8 dead corners), 8×32 (no dead), 8×34 (S57, 16 dead). Handled by `grid.load_grid_mapping()`
- Shared backbone: BiGRU 2×32 with d_shared=256 input. Conv1d(256,32,k=10,s=10) = **82K** temporal projection + GRU ~37K + head ~1.7K = **~121K shared params**. Temporal stride k=10 s=10 (**20Hz**, field standard — validated equivalent to 40Hz). H=32 sufficient — H=128 test showed more overfitting, not better generalization
- **Loss**: Per-position CE everywhere (PER 0.737 grouped-CV per-patient, 0.750 LOPO). CTC abandoned — CE provides stronger per-position gradients at this data scale. Articulatory bottleneck head (15-dim cosine sim) outperforms flat CE head for LOPO.
- **Cross-lab comparison**: Our ~121K backbone is 50-100× smaller than Spalding seq2seq (~6.3M), Duraivel (~4.5M), Willett (~12.3M). Even Singh (closest regime) uses BiLSTM-64 (~100-500K). The bottleneck is input compression, not backbone capacity
- Two-stage LOPO: Stage 1 = standard multi-patient SGD with **held-out 20% source validation** for early stopping; Stage 2 freezes Conv1d+BiGRU, adapts spatial conv+LayerNorm+articulatory head (~2,272 trainable params at default config). **Stage 2 uses stratified 5-fold CV** on target patient (StratifiedKFold on phoneme labels)
- Stage 1 uses best-checkpoint selection (no SWA — no BCI paper uses SWA, adds unprecedented complexity). AdamW + CosineAnnealingLR (single cycle) + early stopping on held-out validation loss
- Systematic ablation: 4 experiments (E0 Spalding + E1 field standard + E2 full model + E3–E4 single-removal)
- Effective augmentation multiplier ~2–3× (not 5×)
- **Compute**: LOPO pilot (8 patients × 1 seed) takes ~80 min on MPS (Stage 1 early-stops ~step 800 of 2000, Stage 2 folds ~3-5s each). Full run (8 patients × 3 seeds) estimated ~4 hours. Original estimate of ~88 GPU-hours was for 4 experiments × 8 folds × 3 seeds

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
- **Speech FM alignment — diagnostic-first design**: Motor cortex → speech FM mapping is untested (Dual Pathway validated STG only; Chang lab uses broad SMC+STG but never isolates motor contribution; Stavisky lab bypasses speech FMs for motor cortex). Our design: post-hoc linear probe on frozen E2 backbone → PCA-reduced HuBERT features (segment-level, MFA-aligned). If R² significant, add segment-level MSE aux loss as E13. Paired audio confirmed for all 8 patients. See `docs/archive/historical_supervised/future_directions.md` § "Speech foundation model alignment" for full analysis
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
