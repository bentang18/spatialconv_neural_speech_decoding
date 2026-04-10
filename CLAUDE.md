# Cross-Patient Speech Decoding from Intra-Op uECOG

## Project

Ben Tang, Greg Cogan Lab, Duke. Collaborating with Zac Spalding.
Extending Spalding 2025 (PCA+CCA, SVM/Seq2Seq, 8 patients, 9 phonemes, 0.31 bal. acc.).

**Task**: Non-word repetition (52 CVC/VCV tokens, 3 phonemes each, e.g. /abe/; 9 phonemes). Intra-operative, left sensorimotor cortex, 128/256-ch uECOG arrays. ~1 min utterance/patient. Stimulus-to-response delay: 1.1 ± 0.3s (Duraivel 2023); stimulus duration ~500ms; utterance ~450ms. Auditory stimulus ends ~600ms before response onset (t=0).

**Patients**: 11 unique PS patients (S18 excluded — no preprocessed; S36 excluded — duplicate of S32): S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62. 8 are Spalding's published set. 46–178 trials/pt, 63–201 sig channels. Core v12 set: S14, S26, S33, S62. Excluded: S32 (no HG response), S57 (52/256 sig, hybrid strip). Extended: S16, S22, S23, S39, S58.

## Current Direction: Neural Field Perceiver (v12) — Intracranial Foundation Model

**Design doc**: `docs/neural_field_perceiver_v12.tex`

**Architecture** (atlas-grounded multi-view reconstruction):
```
electrode HGA + MNI coord
→ Conv1d(1→d, k=10, s=10)           shared temporal binning
→ s^(i) ⊙ x + b^(i)                per-patient diagonal (128 params/pt)
→ + Fourier PE(MNI + Δ/ω)           directional spatial identity
→ VE cross-attention (L=16 atlas, distance bias)  maps N_i → 16 common space
→ [VE self-attn (16×16) → Temporal self-attn] × 2
→ Lightweight AR: pos attn → [z̄_k; emb(y_{k-1})] → Linear(2d→9) × 3 + beam search (52 tokens)
```
~175K total params (B=2), 182 per-patient (128 diagonal + 6 geometric + 48 VE offsets). Per-patient VE positions (atlas centroid + leashed offset ≤15mm) adapt to individual somatotopy; distance bias determines WHERE to look (per-patient), Q/K/V determines WHAT to extract (shared). Electrode self-attention is an ablation (A_elec_attn), NOT default — detailed spatial layout varies too much (R²=0.05-0.13) for shared electrode-level processing to transfer. Dual spatial encoding: distance bias (proximity) + Fourier PE (direction). Lightweight AR decoder (4.1K). 16 Brainnetome atlas positions (40 HCP subjects; somatotopic subdivisions > HCP-MMP for speech). Full-trial input tmin=-0.2, tmax=1.0s. Works for sEEG, uECoG, and macro-ECoG at any electrode density.

**Key design choices** (2026-04-07):
- **Dual spatial encoding**: Distance bias (proximity, isotropic) + Fourier PE on electrode tokens (direction, anisotropic). Distance bias can't distinguish equidistant electrodes in different directions; Fourier PE can. A3 tests removing PE.
- **Lightweight AR decoder**: Position-specific temporal attention + previous phoneme embedding → Linear(2d→9). 4.1K params vs 67K full transformer decoder. AR conditioning for ~2K extra params. A_no_ar tests removing AR.
- **2-stage pipeline (primary, contingent on external data)**: External chronic ECoG SSL (Flinker 48pts + Chang ~15-25pts, est. 50-100h) → per-patient supervised FT on CoganLab. SSL is task-agnostic (different speech tasks handled by masking objective, not labels). **Fallback (CoganLab-only)**: sEEG SSL → uECoG SSL → supervised (~24h).
- **VEs over self-attention**: N=10 can't learn population-average anatomy. Atlas (Brainnetome, N=40; somatotopic subdivisions > HCP-MMP for speech) provides spatial prior. A_self_attn tests electrode self-attention. E2.1 (learned latents, no atlas) isolates atlas from bottleneck.
- **Diagonal normalization over full affine**: impedance/gain = scale+offset, not rotation. 128/pt vs 4160/pt. Init from channel statistics.
- **Δ/ω hard-clamped** (15mm translation, 0.15 rad rotation): covers worst-case registration error.
- **Per-patient VE positions (default, was A_deform)**: Atlas centroid + learned offset ≤15mm (+48 params/pt, total 182/pt). Motor VE gap (A4tl↔A4hf: 32mm) exceeds full somatotopic gradient (~15-30mm). Individual somatotopy varies 5-15mm. A_no_deform reverts to fixed atlas. A_dist_only removes Q/K/V from cross-attention (distance-only pooling) — tests whether functional selectivity matters beyond proximity.
- **Electrode self-attention is an ablation (A_elec_attn), NOT default**: Detailed spatial layout varies too much across patients (spatial gradient R²=0.05-0.13, Duraivel 2025) for shared electrode-level spatial processing to transfer. The transferable signal is temporal dynamics at VE resolution, not spatial patterns at electrode resolution. A_elec_attn_no_pe tests distance-bias-only (generic smoothing, no PE).
- **Per-patient/shared boundary**: Per-patient = WHERE things are (VE positions, gain/offset, registration). Shared = WHAT things mean (VE queries, temporal dynamics, decoder).
- **Coordinate pipeline**: ACPC → talairach.xfm → MNI → learned Δ/ω correction → standardize → Fourier PE(F=3). **HARD BLOCKER: talairach.xfm is linear (12-param affine), ~5-15mm nonlinear residual. If T1 MRIs exist, ANTs SyN nonlinear warp could halve error. Do before Phase 0.**
- Core patients: S14, S26, S33, S62. Exclude S32, S57. Artifact channels dropped entirely.

**Paper direction**: First intracranial field potential (HGA) foundation model. Multi-view framing: each patient is a camera view of a shared neural manifold; VEs are the template mesh. BIT did this for spikes; nobody for field potentials. HGA ≠ spikes (postsynaptic inputs vs action potential outputs — different biophysics, can't pool).

**SSL pretraining** (JEPA with temporal masking, V-JEPA 2/2.1 recipe):
- **Default: single JEPA objective.** L1 prediction of masked VE-time latent representations via EMA target encoder (τ=0.996→1.0). Content-aware weighting (upweight high-activation bins, BrainBERT). Temporal span masking (max 6 patches / 300ms). Mask ratio r~U(0,1) per sample (RPNT). One loss, one auxiliary hyperparameter (EMA momentum). Rationale: V-JEPA 2/2.1 (Meta SOTA) uses a single JEPA loss — intensifying one objective beats diversifying across many. Prior 3-headed design (temporal discrimination + swap detection + JEPA regularizer) over-engineered: swap detection architecturally misaligned (electrode-level gradient doesn't reach VE layers), temporal discrimination is a pretext task superseded by masked prediction, hyperparameter coupling untuneable at 24-100h scale.
- **Ablations:** A_mse (L2 MSE replacing JEPA L1), A_dense (V-JEPA 2.1 dense prediction: supervise visible tokens with distance-weighted loss), A_deep (hierarchical supervision across encoder levels), A_disc (add temporal discrimination auxiliary), A_spatial_mask (add VE spatial masking).
- Per-patient layers during SSL: diagonal normalization + Δ/ω only. Shared backbone. Modality-agnostic.
- **Primary (2-stage)**: External chronic ECoG SSL (Flinker/Chang, 50-100h) → per-patient supervised FT. **Fallback (3-stage)**: sEEG SSL → uECoG SSL → supervised (~24h). 
- **If audio available from external data:** add neural-audio contrastive (CLIP-style) as auxiliary. Future direction, not default.
- Heavier augmentation for SSL: noise 0.10-0.20, constant offset SD=0.05, Gaussian smooth w=2.0

**Default regularization**: L_class (focal CE + label smoothing + mixup) + E_Δω (L2 on geometric params, λ=10^-3) + E_smooth (temporal smoothness). Plus AdamW weight decay (10^-4), dropout (0.2), DropPath (0.1), channel dropout (0.2). Per-patient loss weighting by signal quality. Optional alignment losses (contrastive, KL) are Phase 7 refinements, not default.

**Other design elements**:
- **Per-electrode uncertainty weighting**: σ_j predicted per electrode, down-weighted in cross-attention. Soft reliability mask.
- **Spatial augmentation**: Coordinate jitter (2-5mm), rigid perturbation, electrode subset sampling (70-100%), VE atlas jitter (3-8mm).
- **Neural-audio contrastive (CLIP)**: Future direction, not default. JEPA alone is the core SSL story.

**Scalable data sources** (all measure HGA/field potentials, all poolable):
- **Flinker/Chen lab (NYU)**: est. 50+h (48 patients, chronic ECoG speech) — request with data sharing agreement
- **Chang lab (UCSF)**: est. 20+h (~15-25 patients, chronic high-density ECoG) — request from corresponding author
- Bouchard/Chang (Figshare): small (3 patients, CV syllables) — **public today**
- CoganLab uECoG: ~7.6h (29 patients) — needs HGA extraction (fallback)
- CoganLab sEEG speech: ~16.7h (25 patients) — on Box (fallback, motor coverage ~15-30%)
- Epilepsy sEEG (DABI, OpenNeuro, hospital data): potentially 100s-1000s h
- NOT poolable: Utah arrays (spikes, different biophysics)

**v12 is the sole active direction.** Conv2d pipeline, JEPA, LeWM, LOPO autoresearch — all discontinued.

## Best Per-Patient Results (baseline for v12 to beat)

**Per-phoneme MFA + flat head + full recipe = PER 0.734 ± 0.007** (S14, grouped-by-token CV, 3-seed).

Optimal config from 5 DCC sweeps (2026-04-04):
```
Input: Per-phoneme MFA epochs (tmin=-0.15, tmax=0.5) — 3× labels but 85% temporal overlap (use trial-aware batching)
Spatial: Conv2d(1→8, k=3, pad=1) + AdaptiveAvgPool2d(4,8) → d=256
Temporal: Conv1d(256→32, stride=10) + BiGRU(32, 32, 2L, bidirectional)
Head: Flat Linear(64→9) — NOT articulatory (bottleneck hurts single-phoneme classification)
Readout: Global mean pool → single phoneme prediction (no learned attention needed)
Training: Focal CE (γ=2) + label smoothing (0.1) + mixup (α=0.2)
Eval: Weighted k-NN (k=10) + TTA (n=16)
```

Key sweep findings (see `docs/experiment_log.md` findings 86-101):
- Per-phoneme beats learned attention by 6pp in fair head-to-head (0.734 vs 0.797)
- Per-phoneme wins 8/11 patients, population mean +4.0pp over full-trial
- Flat head > articulatory for single-phoneme (0.734 vs 0.772)
- Padding not critical: tmin=-0.10 to -0.15 optimal, tmax=0.5

Previous baselines: LOPO best 0.750, per-patient full-trial 0.737, LOPO pilot 0.846.

## Key Files

### Data (local, gitignored)
- `data/mni_coords/<subj>_RAS.txt` — ACPC electrode coordinates (11/11)
- `data/channel_maps/<subj>_channelMap.mat` — Amplifier → physical grid mapping (11/11)
- `data/channel_maps/<subj>_sigChannel.mat` — Significant channel masks (9/11, missing S32/S57)
- `data/transforms/<subj>_talairach.xfm` — ACPC → Talairach/MNI transform (11/11)

### Active
- `docs/neural_field_perceiver_v12.tex` — Active design document
- `docs/current_direction.md` — Current priorities and what's archived
- `docs/dcc_setup.md` — Complete DCC documentation
- `docs/experiment_log.md` — Full experiment history (101 findings)
- `docs/research_synthesis.md` — 19-paper literature synthesis (seegnificant added)
- `docs/reading_list.md` — 11 essential papers (seegnificant added as #10)

### Configs
- `configs/per_patient_ce_s10_pool48.yaml` — Per-patient config (CE, stride=10, pool(4,8))
- `configs/lopo_ce.yaml` — LOPO cross-patient config
- `configs/paths.yaml` — Machine-specific BIDS paths (gitignored)

### Scripts
- `scripts/train_per_patient.py` — Per-patient training CLI
- `scripts/train_lopo.py` — LOPO cross-patient training CLI
- `scripts/sweep_head_to_head.py` — Fair comparison: learned attn vs per-phoneme (full recipe)
- `scripts/sweep_multipatient.py` — Per-phoneme vs full-trial across all 11 patients
- `scripts/sweep_full_recipe.py` — Full recipe (mixup + k-NN + TTA) sweep
- `scripts/sweep_tmin_perpos.py` — Temporal windowing sweep
- `scripts/sweep_padding_grid.py` — tmin/tmax fine-tuning

### Archived
- `docs/archive/` — Old NFP versions, NCA-JEPA specs, LOPO plans, historical design docs
- `scripts/archive/` — SSL eval, diagnostics, 83 autoresearch experiments
- `configs/archive/` — Historical baselines, sweeps, pretrain configs
- `src/speech_decoding/pretraining/` — NCA-JEPA code (on disk, not actively used)

## Code Structure

```
src/speech_decoding/
├── data/
│   ├── phoneme_map.py      # 9 PS phonemes, PS2ARPA, articulatory matrix (9×15)
│   ├── grid.py             # Electrode TSV → grid shape + channel-to-grid mapping
│   ├── bids_dataset.py     # load_patient_data() + load_per_position_data()
│   ├── augmentation.py     # Time shift, amplitude scale, channel dropout, noise
│   ├── coordinates.py      # ACPC electrode coords: RAS loading, chanMap bridge, hemisphere mirroring
│   ├── sig_channels.py     # Significant channel detection + artifact channel exclusion
│   ├── atlas.py            # Brainnetome atlas ROIs: 16 core + 8 extended virtual electrode positions for v12
│   └── collate.py          # Group samples by patient_id for multi-grid batching
├── models/
│   ├── spatial_conv.py     # Per-patient Conv2d read-in: (B,H,W,T)→(B,256,T)
│   ├── backbone.py         # LayerNorm → Conv1d(s=10) → BiGRU(32,32,2L)
│   ├── articulatory_head.py # 15-dim bottleneck → fixed A → 9 phonemes
│   ├── flat_head.py        # Linear(128,10) → log_softmax
│   └── assembler.py        # YAML config → model components
├── training/
│   ├── ctc_utils.py        # CTC loss, greedy decode, PER
│   ├── trainer.py          # Per-patient CV training
│   ├── lopo_trainer.py     # Stage 1: multi-patient SGD
│   ├── adaptor.py          # Stage 2: target adaptation
│   └── lopo.py             # LOPO orchestrator
└── evaluation/
    ├── metrics.py          # PER, balanced accuracy
    ├── grouped_cv.py       # Grouped-by-token CV splitter
    └── content_collapse.py # Collapse diagnostics
```

Run: `pytest tests/ -v -m "not slow"` (fast) or `pytest tests/ -v` (all, needs BIDS data)

## Data

### Loading
```python
# Per-phoneme MFA epochs (recommended — 3× more samples, 85% temporal overlap):
from speech_decoding.data.bids_dataset import load_per_position_data
ds = load_per_position_data("S14", bids_root, task="PhonemeSequence", n_phons=3,
                            tmin=-0.15, tmax=0.5, exclude_artifacts=True)
# ds[i] → (grid_data[H,W,T], label[list[int]], patient_id)  — 459 samples for S14

# Full-trial epochs (all 3 phonemes in one window):
from speech_decoding.data.bids_dataset import load_patient_data
ds = load_patient_data("S14", bids_root, task="PhonemeSequence", n_phons=3,
                       tmin=0.0, tmax=1.0, exclude_artifacts=True)
# ds[i] → (grid_data[H,W,T], ctc_label[list[int]], patient_id)  — 153 trials for S14
```

### Grid Layouts
| Channels | Grid | Dead positions | Patients |
|----------|------|----------------|----------|
| 128 | 8×16 | 0–1 | S14, S16, S22, S23, S26 |
| 256 | 12×22 | 8 (corners) | S32, S33, S39, S58, S62 |
| 256 | 8×32 | 0–1 | (Lexical patients only) |
| 256 | 8×34 | 16 | S57 |

Grid inferred from electrode TSVs, NOT channel count. TSVs have BOM (`\ufeff`). Dead positions zeroed in Conv2d input.

### Electrode Coordinates (ACPC, verified 2026-04-05)

Coordinates are in **ACPC space** (per-patient, AC-PC aligned), NOT MNI-152. Source: `Box/ECoG_Recon/<subj>/elec_recon/<subj>_elec_locations_RAS_brainshifted.txt`. Format: `prefix electrode_num x y z hemisphere type`.

**Coordinate mapping chain** — implemented in `src/speech_decoding/data/coordinates.py`:
- 128-ch: `fif ch N → chanMap[r,c]==N → phys_elec = r*16+c+1 → RAS(x,y,z)`. Mean error 8.55mm without chanMap, ~1.4mm with chanMap (verified).
- 256-ch: `fif ch N → RAS electrode N` directly (~85% overlap; fif-only = dead positions). S57/S58 are 0-indexed (need +1 offset). chanMapAll is NaN-filled — not needed.
- Use `build_electrode_coordinates()` — handles both paths automatically.

**DCC TSV vs RAS files**: DCC electrode TSVs have normalized 0-1 grid coordinates (synthetic, for per-patient Conv2d). RAS files have real ACPC coordinates (for v12). TSV grid = vertically-flipped chanMap (cosmetic, irrelevant for Conv2d).

**Coordinate acquisition**: Electrode positions obtained by measuring 4 array corners → interpolating grid → projecting onto smoothed cortical surface. Error is primarily systematic (rigid-body ~4-8mm), not random per-electrode noise. Relative within-array positions near-perfect (~0.1mm). Learned Δ/ω correction is the correct model for this systematic error.

**Hemisphere**: S22 and S58 are **right hemisphere** (positive x). All others left. `ElectrodeCoordinates.mirror_to_left()` negates x for cross-patient alignment.

### Significant Channels

.fif files contain ALL channels (not filtered). sigChannel.mat files identify task-responsive channels via permutation cluster test (upstream). Available for 9/11 patients (missing S32, S57).

| Patient | Sig ch | Total | % sig |
|---------|--------|-------|-------|
| S14 | 111 | 128 | 87% |
| S16 | 65 | 128 | 51% |
| S22 | 74 | 128 | 58% |
| S23 | 63 | 128 | 49% |
| S26 | 111 | 128 | 87% |
| S32 | ? | 256 | ? |
| S33 | 149 | 256 | 58% |
| S39 | 144 | 256 | 56% |
| S57 | ? | 256 | ? |
| S58 | 171 | 256 | 67% |
| S62 | 201 | 256 | 78% |

Sig channel filtering does NOT improve per-patient decoding for S14 (85% sig). Conv2d learns to suppress non-sig channels. For v12, include ALL non-artifact channels (model learns to weight via cross-attention).

### Artifact Channels (electronic, not brain signal)

Some channels exhibit extreme activations (>10 std in >5% of trials) — electronic artifacts from mic feedback / amplifier saturation, confirmed by Zac. These should be **excluded entirely** (not clipped — capped artifacts are still confounded signal). Detected by `detect_artifact_channels()` in `sig_channels.py`.

| Patient | Chronic artifact ch | Max value (std) |
|---------|-------------------|-----------------|
| S14 | 0 | 43 |
| S26 | 4 | 15 |
| S39 | **20** | **627** |
| S57 | **15** | 83 |
| S58 | **37** | 149 |

S39/S57/S58 are the worst. S14/S16/S23/S32 are clean (0 chronic).

### Inter-Patient Electrode Overlap (quantified 2026-04-06)

Arrays are placed by surgeon, not experimentally standardized. Mean centroid distance: **36mm** (range 8.5–75.6mm). Most pairs have no electrode-level overlap (<5mm).

Two spatial clusters:
- **Posterior-dorsal**: S14, S23, S39, S58, S62 (centroids ~MNI -60, 0, 40)
- **Anterior-ventral**: S26, S32, S33, S57 (centroids ~MNI -58, 20, 5)
- **Outliers**: S16 (y=50), S22 (right hemisphere)

Best overlapping pairs: S26↔S33 (1.3mm NN), S22↔S62 (2.7mm), S32↔S33 (4.4mm).

This is THE core challenge for cross-patient models — no shared electrode space. v12's VE cross-attention maps each patient's electrodes into a common functional space (16 atlas positions) regardless of array placement.

### Virtual Electrodes (Brainnetome atlas, `atlas.py`) — DEFAULT SPATIAL MECHANISM

16 core ROIs — top 16 by patient reachability from systematic check of all 123 LH Brainnetome ROIs + speech-relevant candidates (2026-04-06). Top 15 have ≥4 patients; #16 (A2) has 3.
- Motor (3): A6cvl (ventral PMC, 9pts), A4tl (tongue M1, 7pts), A4hf (face M1, 6pts)
- Sensory (3): A1/2/3tonIa (tongue S1, 8pts), A1/2/3ulhf (face S1, 5pts), A2 (proprioceptive, 3pts)
- Broca's (6): A44d (8pts), A45c (8pts), A44v (7pts), A45i (7pts), A45r (5pts), A44op (4pts)
- Auditory (2): STGpp (planum polare, 4pts), STGa (anterior STG, 4pts)
- Insula (1): INSa (articulatory planning, 4pts)
- Executive (1): MFG (dorsolateral PFC, 6pts)

8 extended ROIs for analysis (parietal, SMA — mostly unreachable). Patient-adaptive selection via `select_active_virtual_electrodes()`. Each patient reaches ~6-12 of 16 core VEs at 25mm threshold. VE cross-attention with distance bias maps variable electrode counts into this common functional space.

### Raw Continuous Recordings (for SSL)

456 min across 29 patients (13 PS + 17 Lexical, zero patient overlap). Raw 2kHz EDF files in BIDS: `sub-{id}/ieeg/sub-{id}_task-{task}_acq-01_run-01_ieeg.edf`. Need HGA extraction (CAR → 70-150Hz filterbank → Hilbert → 200Hz) to match existing productionZscore features. PS: ~199 min, Lexical: ~257 min. S14 longest at 31 min.

### .fif Path
`{bids_root}/derivatives/epoch(phonemeLevel)(CAR)/sub-{id}/epoch(band)(power)/sub-{id}_task-PhonemeSequence_desc-productionZscore_highgamma.fif`

PS labels: `{'a':1, 'ae':2, 'b':3, 'g':4, 'i':5, 'k':6, 'p':7, 'u':8, 'v':9}` — `phoneme_map.normalize_label()` handles conversion.

## Compute: Duke DCC Cluster

**Use DCC for all training.** See `docs/dcc_setup.md` for complete documentation.

- **SSH**: `ssh ht203@dcc-login.oit.duke.edu`
- **GPU**: 8× RTX 5000 Ada (32 GB) on `coganlab-gpu`
- **Python**: `/work/ht203/miniconda3/envs/speech/bin/python` (PyTorch 2.10.0+cu126; do NOT `conda activate`)
- **Repo**: `/work/ht203/repo/speech`
- **Data**: `/work/ht203/data/BIDS` — all 11 PS patients (.fif + electrode TSV)
- **Coordinates**: `/work/ht203/data/mni_coords/` — ACPC RAS brainshifted (11/11 patients)
- **Channel maps**: `/work/ht203/data/channel_maps/` — chanMap + sigChannel .mat files
- **Transforms**: `/work/ht203/data/transforms/` — talairach.xfm (11/11 patients)
- **Submit**: `sbatch scripts/<script>_dcc.sh` | Monitor: `squeue -u ht203`
- **CAUTION**: `/work/ht203` auto-purges after 75 days. Copy results to `/hpc/group/coganlab/ht203/`.

## Completed Exploration (summary — details in experiment_log.md)

- **LOPO** (55 experiments): Converged to PER 0.750-0.780 on S14. Measurement ceiling from fixed CV folds.
- **SSL / NCA-JEPA**: All methods near-chance on ~11 min epoched data. CoganLab-only SSL limited (~24h, intra-op data quality). Primary SSL plan: external chronic ECoG from Flinker (48 pts) and/or Chang (~15-25 pts), est. 50-100h of diverse speech. Contingent on data access (PI-level request). Fallback: CoganLab sEEG + uECoG (~24h).
- **Per-patient tuning**: CTC→CE (+7.8pp), pool(2,4)→pool(4,8), stride=10, H=32 sufficient.
- **Per-phoneme MFA sweep** (2026-04-04): Per-phoneme flat (0.734) beats learned attention (0.797) and full-trial (0.807). Generalizes 8/11 patients.

## Established Findings (from literature)

- **Field consensus**: per-patient input → shared backbone (GRU) → CTC/CE. Used by Willett, Metzger, Singh, Boccato, Levin, BIT.
- **Alignment**: uECOG arrays span 15-25mm in MNI with variable rotation. VE cross-attention with atlas positions + distance bias maps variable electrodes into a common functional space, accommodating anatomical variation via ~25mm soft receptive fields. Self-attention + MNI coordinates (seegnificant) is an ablation (A_self_attn).
- **Transfer**: Singh — freeze shared backbone, fine-tune per-patient layers. Levin — 30% source replay prevents forgetting.
- **SSL**: Advantage is cross-subject only (BIT Table 9: SL ≈ SSL same-subject). Supervised cross-subject pretraining FAILS without per-patient layers (BIT Appendix M, NDT3). Temporal masking > spatial for speech (BIT SOTA). Minimum corpus ~30 min (wav2vec ECoG); we have 456 min raw continuous (29 patients, needs HGA extraction).
- **Per-patient layers**: Decisive factor for cross-patient transfer. NDT3 (no per-patient, 2000h) fails cross-subject. BIT (per-patient read-in/out, 367h) succeeds. Singh (per-patient Conv1D, supervised) also succeeds. seegnificant: per-subject heads ΔR²=-0.18 (most important component).
- **Coordinate PE**: seegnificant: PE barely helps (ΔR²=-0.02, p=0.73). Spatial self-attention does the heavy lifting. Fourier PE = RBF PE (both R²=0.39). Treat PE as uncertain until A2 vs A3 tested.
- **Factored attention**: Temporal then spatial outperforms joint 2D (seegnificant: +0.06 R², 5.5× faster). Validates v12's factored design.
- **Data regime**: Small epoched (46-178 trials/pt), ~24h SSL corpus (~1000 min sEEG + 456 min uECoG). ~175K total params (B=2). Foundation model direction: scale data, not architecture.
- **HGA ≠ spikes**: HGA (70-150Hz) reflects postsynaptic activation (inputs to measured area). Spikes (Utah arrays) are action potentials (outputs). Different biophysics — cannot pool across modalities. Surface recordings (uECoG, ECoG, sEEG HGA) ARE poolable.
- **Nason 2024 (NEJM)**: Best speech BCI uses CTC encoder (5-layer GRU 512) + 5-gram LM + OPT rescoring. NOT autoregressive. Raw PER 7-15%, WER 2.5% with LMs. Day-specific 512→512+softsign layer.
- **Transformer > GRU**: Confirmed by B2T benchmarks (Feghhi 2025 NeurIPS: time-masked Transformer, Feghhi 2026: LightBeam). WER 5.01% (Transformer) vs 9.37% (GRU) on B2T '24.
- **MIBRAIN (Wu 2025)**: Independent convergence on atlas-grounded region mapping for cross-patient sEEG (11 pts, 23 Mandarin consonants). Hard FreeSurfer parcellation + learnable prototypes + region masking SSL. Multi-sub beats single-sub by 5-8%. NO coordinates used. Scaling: 1-3 added subjects HURT, need ≥6 for gains. v12 generalizes MIBRAIN: soft distance-biased attention subsumes hard region assignment, coordinates provide finer resolution, 134 per-patient params vs ~2-5K. Closest prior to v12.
- **Di Wu group evolution** (MIBRAIN → Neuro-MoBRE → H2DiLR): All use discrete FreeSurfer regions, zero coordinates, same 11-pt cohort. Neuro-MoBRE adds MoE routing (21 region experts, TopK=2) + freq-domain SSL target. H2DiLR uses VQ codebook disentanglement (1.55M/pt, impractical for us). Transformer-16-512 collapses in their data regime — validates our ~175K model.
- **Population Transformer (Chau 2025, ICLR)**: Self-attention + 3D sinusoidal PE on sEEG coordinates, ZERO per-patient params, ~20M total. Binary detection tasks (0.93 AUC speech). PE removal is most damaging ablation. Discriminative SSL >> reconstructive. BUT: no temporal sequence modeling, can't do phoneme decoding, tasks are much easier.
- **Brant family (2023-2025)**: Brant (505M, 2528h) → BrainWave (~100M, 40,907h, 16K subjects) → BrantX (1B, cross-modality). Scaled massively but NEVER added spatial identity or per-patient layers. Brant fails cross-patient speech (MIBRAIN baseline). BrainWave adds channel attention but channels are still anonymous (no coordinates). BrantX is cross-modality (EEG→EOG/ECG), orthogonal to our problem. The entire family proves: scale alone insufficient — atlas-grounded spatial mechanism + per-patient layers are necessary.
- **BrainBERT (Wang 2023, ICLR)**: Per-electrode SSL on sEEG spectrograms, ~28M, 43.7h. Content-aware L1 loss prevents collapse on sparse signals (importable). +0.23 AUC from pretraining, 5× data efficiency. Cross-subject transfer works. Fails as MIBRAIN baseline (no spatial model).
- **Evanson 2025**: Supervised contrastive pretraining on 83-108h sEEG. Log-linear scaling, NO plateau at 100h. Supervised > SSL for same-subject. Cross-day drift massive (r=0.95). Single-patient only.
- **Content-aware loss** (BrainBERT): 68% of z-scored neural data near zero. Upweight reconstruction of high-activation bins during SSL. Directly importable for v12's temporal masking MSE.
- **BarISTA (Oganesian 2025, NeurIPS)**: Parcel-level spatial encoding >> channel-level by +8-10pp AUC on iEEG. Strongest external validation of v12's VE approach. JEPA-style latent reconstruction with spatial masking. Combined attention > factored (+1-2pp, contradicts seegnificant). ~1M params, zero per-patient. Uses Destrieux atlas parcels.
- **Charmander (Mahato 2025, NeurIPS WS)**: Perceiver bottleneck (32 latents) for multi-patient iEEG with 50% channel masking. Model scaling 8M→142M: no downstream benefit. Converges with v12's VE cross-attention architecture.
- **NDT3 (Ye 2025)**: 350M-param intracortical motor FM. Cross-subject transfer fundamentally limited by sensor variability — channel shuffle cripples performance. Explicitly identifies per-patient layers as needed. THE key negative result motivating v12's per-patient design.
- **RPNT (Fang 2026)**: Uniform random mask ratio U(0,1) outperforms all fixed ratios — free improvement for SSL. MRoPE for metadata encoding. Cross-site contrastive loss helps (+3.5pp).
- **Factored vs combined attention**: seegnificant (factored +0.06 R²) vs BarISTA (combined +1-2pp). Resolution: may depend on spatial dimensionality. v12 operates on 16 VEs after cross-attention, so factored cost is minimal and combined is feasible. Test both.
- **Perceiver cross-attention bottleneck is consensus**: 5+ independent groups converge — POYO (512 latents, NeurIPS 2023), POYO+ (128, ICLR 2025), Charmander (32), Brain-OF ARNESS (128), v12 (16 VEs), POSSM (1/chunk). Model scaling provides no downstream benefit (Charmander 8M≈142M). Architecture > capacity.
- **Data heterogeneity limits scaling for spatial tasks but NOT temporal**: Jiang 2025: region heterogeneity kills co-smoothing scaling. Forward-prediction (temporal) scales consistently. 5 ranked sessions > 40 random (8× efficiency). Rankings are target-specific. Validates temporal masking SSL over spatial masking. Don't blindly pool all patients — rank and select.
- **Preserved latent dynamics across animals**: Safaie 2023 (Nature): PCA+CCA reveals shared motor cortex dynamics across 3 monkeys + 4 mice. Cross-animal LSTM R²≈0.86. Linear alignment sufficient. ~60 neurons minimum. Biological foundation for cross-patient approach.
- **Multi-session pretrain + few-shot FT dominates**: FALCON benchmark (NeurIPS 2024): NDT2 Multi > unsupervised alignment > zero-shot. CORP test-time adaptation with LM pseudo-labels wins communication (WER 0.11). Deep networks catastrophically unstable without per-session layers (RNN: -0.60 R² zero-shot). 1-2 min calibration sufficient.
- **Cross-species transfer works for speech**: POSSM (NeurIPS 2025): monkey→human speech PER 19.80% with multi-input (spike counts + spike-band power). Two-phase training (reconstruction→CTC). SSM backbone real-time (<6ms CPU).
- **Functional embeddings can outperform coordinates**: FunctionalMap (Javadzadeh 2025): contrastive-learned 32-dim functional embeddings beat MNI coordinates for SEEG reconstruction (p<0.001). Complementary to v12 (augment Fourier PE). But requires region labels, tested only on deep brain nuclei.
- **Within-modality masking most critical for encoding**: NEDS (ICML 2025): removing within-modality masking drops encoding 50%. Multi-task masking (4 schemes) enables simultaneous encoding + decoding. Neuron embeddings become 83% brain-region-predictive without labels. 12M right for 74 sessions; our 175K for 4-11 patients is correct regime.
- **Distribution matching (KL) enables manifold alignment**: NoMAD (Nature Comms 2025): frozen backbone + KL divergence alignment. R²=0.91, 208-day half-life. Single reference > sequential. Importable as auxiliary loss on VE representations. Authors note applicability to ECoG.
- **POYO family (Dyer group)**: Spike tokenization + Perceiver + per-unit embeddings + gradual unfreezing. No spatial encoding (works for chronic arrays, not variable placement). POYO: 7 monkeys, supervised. POYO+: 1335 sessions, multi-task. POSSM: + SSM, cross-species. All validate v12's Perceiver architecture but lack spatial mechanism.

## Preprocessing Pipeline (do not change)

Decimate 2kHz → CAR → impedance exclusion (log10>6) → 70-150Hz Gaussian filterbank (8 bands) → Hilbert envelope → sum → 200Hz → z-score → significant channel selection. Implemented in `coganlab/IEEG_Pipelines`.

## Conventions

- Explain every design choice with precedent and tradeoffs
- Batch work in smaller chunks to prevent context rot
- Keep markdown lean and information-dense
- Report per-patient results, not just population means
- Always use grouped-by-token CV (never stratified — token leakage inflates by ~10pp)
- All training on DCC, never local
