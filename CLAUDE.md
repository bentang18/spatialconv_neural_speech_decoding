# Cross-Patient Speech Decoding from Intra-Op uECOG

## Project

Ben Tang, Greg Cogan Lab, Duke. Collaborating with Zac Spalding.
Extending Spalding 2025 (PCA+CCA, SVM/Seq2Seq, 8 patients, 9 phonemes, 0.31 bal. acc.).

**Task**: Non-word repetition (52 CVC/VCV tokens, 3 phonemes each, e.g. /abe/; 9 phonemes). Intra-operative, left sensorimotor cortex, 128/256-ch uECOG arrays. ~1 min utterance/patient.

**Patients**: 11 unique PS patients (S18 excluded — no preprocessed; S36 excluded — duplicate of S32): S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62. 8 are Spalding's published set. 46–178 trials/pt, 63–201 sig channels.

## Current Direction: Neural Field Perceiver (v12)

**Design doc**: `docs/neural_field_perceiver_v12.tex`

Transformer-based cross-patient architecture (NOT GRU) using ACPC electrode coordinates:
- **Per-electrode tokenization**: shared Conv1d(1→d, k=10, s=10) → **per-patient Linear(d→d)** (identity init, ~4.2K/pt) → Fourier PE on MNI coords
- **Spatial cross-attention** to 10 fixed virtual electrodes (Brainnetome atlas, ventral speech motor ROIs) with learnable distance bias
- **Temporal self-attention** per virtual electrode across ~20 time bins
- **Reconstruction loss** → spatial supervision (predict HGA from latent + position)
- **Per-phoneme MFA epochs** as input (tmin=-0.15, tmax=0.5) with mean-pool temporal readout
- **Patient-adaptive VE selection** — only attend to VEs within 25mm of each patient's array

Key v12 design decisions (from data analysis, 2026-04-06):
- 10 virtual electrodes (not 16) — 6 dorsal/distal ROIs unreachable by all patients
- Artifact channels dropped entirely (no token), not zeroed
- Trial-aware batching for per-phoneme (85% temporal overlap between consecutive positions)
- Start with 8 good-MFA patients; add S16/S22/S39 later with full-trial fallback
- Atlas: Brainnetome (only atlas separating tongue/lip/laryngeal M1). Field uses DK/Destrieux but too coarse.

**SSL pretraining** (from BIT/NDT3 analysis, 2026-04-06):
- **Temporal span masking** (NOT spatial) — BIT-style masked temporal patch reconstruction (MSE). Speech is temporal; spatial masking teaches interpolation, temporal masking teaches dynamics.
- Per-patient input projection during SSL (same as supervised: shared Conv1d → per-patient Linear(d→d)), shared backbone. Per-patient layers are the decisive factor for cross-patient transfer (NDT3 without them fails; BIT/Singh with them succeed).
- Heavier augmentation for SSL: noise 0.10-0.20 (vs 0.02 supervised), constant offset SD=0.05, Gaussian smooth w=2.0
- **Blocked on HGA extraction** from 456 min raw continuous EDF (29 patients)
- Ablations: A18 (SSL vs scratch), A19 (SSL + spatial recon), A20 (spatial vs temporal masking)
- Minimum SSL threshold: ~30 min (wav2vec ECoG). Our 7.6h = 15× threshold.

**v12 is the sole active direction.** Conv2d pipeline, JEPA, LeWM, LOPO autoresearch — all discontinued.

See `docs/current_direction.md` for full context.

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
│   ├── atlas.py            # Brainnetome atlas ROIs: 10 core + 6 extended virtual electrode positions for v12
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

This is THE core challenge for cross-patient models — no shared electrode space. v12's coordinate-based cross-attention to virtual electrodes is specifically designed for this.

### Virtual Electrodes (Brainnetome atlas, `atlas.py`)

10 core ROIs (ventral speech motor strip reachable by our arrays):
- Motor: A4hf (face M1), A4tl (tongue M1), A6cvl (ventral PMC)
- Sensory: A1/2/3ulhf (face S1), A1/2/3tonIa (tongue S1), A2
- Broca's: A44d, A44v, A45c
- Bridge: A6cdl

6 extended ROIs available for ablation (>30mm from all patients). Patient-adaptive selection via `select_active_virtual_electrodes()`. Each patient reaches ~4-6 of 10 core VEs at 25mm threshold.

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
- **SSL / NCA-JEPA**: All methods near-chance on ~11 min epoched data. BUT: 456 min raw continuous EDF exists across 29 patients (15× the ~30 min SSL threshold). Needs HGA extraction.
- **Per-patient tuning**: CTC→CE (+7.8pp), pool(2,4)→pool(4,8), stride=10, H=32 sufficient.
- **Per-phoneme MFA sweep** (2026-04-04): Per-phoneme flat (0.734) beats learned attention (0.797) and full-trial (0.807). Generalizes 8/11 patients.

## Established Findings (from literature)

- **Field consensus**: per-patient input → shared backbone (GRU) → CTC/CE. Used by Willett, Metzger, Singh, Boccato, Levin, BIT.
- **Alignment**: uECOG arrays span 15-25mm in MNI with variable rotation. Per-channel scaling is inapplicable (Boccato's diagonal finding is Utah-specific). Per-patient Conv2d matches rigid-array physics.
- **Transfer**: Singh — freeze shared backbone, fine-tune per-patient layers. Levin — 30% source replay prevents forgetting.
- **SSL**: Advantage is cross-subject only (BIT Table 9: SL ≈ SSL same-subject). Supervised cross-subject pretraining FAILS without per-patient layers (BIT Appendix M, NDT3). Temporal masking > spatial for speech (BIT SOTA). Minimum corpus ~30 min (wav2vec ECoG); we have 456 min raw continuous (29 patients, needs HGA extraction).
- **Per-patient layers**: Decisive factor for cross-patient transfer. NDT3 (no per-patient, 2000h) fails cross-subject. BIT (per-patient read-in/out, 367h) succeeds. Singh (per-patient Conv1D, supervised) also succeeds. seegnificant: per-subject heads ΔR²=-0.18 (most important component).
- **Coordinate PE**: seegnificant: PE barely helps (ΔR²=-0.02, p=0.73). Spatial self-attention does the heavy lifting. Fourier PE = RBF PE (both R²=0.39). Treat PE as uncertain until A2 vs A3 tested.
- **Factored attention**: Temporal then spatial outperforms joint 2D (seegnificant: +0.06 R², 5.5× faster). Validates v12's factored design.
- **Data regime**: Small epoched (46-178 trials/pt), but 7.6h raw continuous available. ~121K shared params, 50-100× smaller than field.

## Preprocessing Pipeline (do not change)

Decimate 2kHz → CAR → impedance exclusion (log10>6) → 70-150Hz Gaussian filterbank (8 bands) → Hilbert envelope → sum → 200Hz → z-score → significant channel selection. Implemented in `coganlab/IEEG_Pipelines`.

## Conventions

- Explain every design choice with precedent and tradeoffs
- Batch work in smaller chunks to prevent context rot
- Keep markdown lean and information-dense
- Report per-patient results, not just population means
- Always use grouped-by-token CV (never stratified — token leakage inflates by ~10pp)
- All training on DCC, never local
