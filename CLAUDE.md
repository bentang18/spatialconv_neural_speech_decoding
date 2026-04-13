# Cross-Patient Speech Decoding from Intra-Op uECOG

## Project

Ben Tang, Greg Cogan Lab, Duke. Collaborating with Zac Spalding.
Extending Spalding 2025 (PCA+CCA, SVM/Seq2Seq, 8 patients, 9 phonemes, 0.31 bal. acc.).

**Task**: Non-word repetition (52 CVC/VCV tokens, 3 phonemes each, e.g. /abe/; 9 phonemes). Intra-operative, left sensorimotor cortex, 128/256-ch uECOG arrays. ~1 min utterance/patient. Stimulus-to-response delay: 1.1 ± 0.3s (Duraivel 2023); stimulus duration ~500ms; utterance ~450ms. Auditory stimulus ends ~600ms before response onset (t=0).

**Patients**: 11 unique PS patients (S18 excluded — no preprocessed; S36 excluded — duplicate of S32): S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62. 8 are Spalding's published set. 46–178 trials/pt, 63–201 sig channels. Core set: S14, S26, S33, S62. Excluded: S32 (no HG response), S57 (52/256 sig, hybrid strip). Extended: S16, S22, S23, S39, S58.

## Current Direction: Neural Field Perceiver (v14) — Intracranial Foundation Model

**Design doc**: `docs/neural_field_perceiver_v14.tex`

**Two-problem decomposition** (atlas-guided spatial calibration + shared dynamics):
- **Problem 1 — Calibration** (per-patient, physics-constrained): Map raw electrodes → atlas-grounded regional tokens via Brainnetome volumetric parcellation. Atlas does ~90% of calibration; supervised gradient refines ~10%. Analogous to template fitting with known sensor placement.
- **Problem 2 — Dynamics** (shared, unconstrained ML): Map atlas-grounded regional tokens → phoneme sequence via a small relational-temporal transformer + AR decoder. Same representation for every patient.

**Target architecture**:
```
── Phase 2+ / Full v14 target ──
Calibration: corrected electrode coordinates -> Brainnetome PM membership
             optional learned per-patient calibration only after Phase 1

── Shared Processing ──
(1) Shared temporal tokenizer                              -> (N_i × d × T)
(2) Canonical parcel-frame local point encoding
    + within-parcel Perceiver summarizer                   -> (N_tok × d × T)
(3) [Inter-region graph attention
     -> Temporal self-attn] × B                            -> (N_tok × d × T)
(4) 3 AR-conditioned decode queries attend over N_tok·T    -> phoneme sequence
```
Shared interface: `(N_tok × d × T)` atlas/subparcel token tensor, with default `N_tok=21` from 16 base parcels and selective 2-token splits in elongated speech-relevant parcels. Mean+gradient pooling is the main linear ablation. No Fourier PE. No global electrode->region cross-attention.

**Current Phase-1 implementation contract** (2026-04-13):
- **Use the real Brainnetome PM volume, not the old proxy**: Active membership source is `/Users/bentang/Documents/Code/speech/data/atlas/BNA_PM_4D.nii.gz`. Keep `~/nilearn_data/bnatlas.nii.gz` only for ROI indexing and sanity checks. No silent fallback to the old smoothed-MPM pseudo-probability proxy.
- **Fixed-atlas `v14-core`, not learned calibration yet**: Phase 1 freezes the spatial interface to the best verified ACPC→MNI path. No learned `Δ/ω`, `δ_l`, or `τ_l` yet. The coordinate pipeline is still a blocker and must be re-verified with Zac before implementation starts.
- **No extra gain / impedance correction in Phase 1**: Inputs already come from existing `productionZscore_highgamma` features. Do not add another gain/offset layer or channel-stat normalization in the first pass.
- **Within-parcel Perceiver summarizer is the default spatial mechanism**: Canonical parcel-frame coordinates + shared point encoder + fixed latent queries summarize each parcel locally into 1-2 tokens. Mean+gradient pooling is the main linear ablation.
- **Temporal front-end is shared and still an explicit blocker**: The next architectural decision is the exact temporal layer and its output contract into the parcel summarizer. Do not harden downstream code until that interface is frozen.
- **Unsupported parcels are masked, not hallucinated**: Phase 1 uses a fixed `N_tok` layout with `token_mask` and `token_support`. Unsupported parcels are computationally absent; zero-filled inactive slots are only a storage convenience. The exact support statistic and unsupported-vs-weak threshold are still blockers.
- **Inter-region attention is token-space, not sensor-space**: After parcel summarization, the backbone operates over atlas/subparcel tokens. Brainnetome SC/FC bias initialization is still intended, but the exact token-level expansion rule remains a blocker.
- **Phase 1 is supervised-only on `uECoG`**: First executable milestone is supervised `v14-core` on the existing intra-op `uECoG` data, verifying token construction, masking, and end-to-end correctness before SSL, `sEEG`, or external datasets.
- **Next step after `v14-core` is full-corpus `uECoG` SSL**: If the supervised path is correct, Phase 1.5 is SSL on the full continuous `uECoG` corpus, not only response-locked epochs.
- **Do not start implementation before blocker review**: The active blocker list in `docs/implementation_start.md` and `docs/implementation_tasks.md` must be explicitly discussed and frozen before coding begins.

**Paper direction**: Atlas-grounded common-space decoding for intracranial field potentials. The scientific bet is that electrodes are patient-specific observations, while parcel/subparcel tokens are the shared representation. Phase 1 is only the fixed-atlas supervised correctness pass; broader calibration claims come later.

**Near-term sequencing**:
- Phase 1: supervised `v14-core` on response-locked `uECoG`
- Phase 1.5: SSL on the full continuous `uECoG` corpus
- Phase 2: learned per-patient calibration
- Only after that: `sEEG`, external datasets, and broader scaling

Detailed SSL planning, historical literature findings, and broader data-scaling notes now live under:
- `docs/archive/literature_findings.md`
- `docs/archive/research_synthesis.md`
- `docs/archive/reading_list.md`

**v14 is the sole active direction.** v12 (cross-attention + distance bias + Fourier PE), Conv2d pipeline, JEPA, LeWM, LOPO autoresearch — all discontinued.

## Best Per-Patient Results (baseline for v14 to beat)

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

Key sweep findings (see `docs/archive/experiment_log.md` findings 86-101):
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
- `docs/neural_field_perceiver_v14.tex` — Active design document (v14: parcellation + two-problem decomposition)
- `docs/current_direction.md` — Current priorities and what's archived
- `docs/dcc_setup.md` — Complete DCC documentation
- `docs/implementation_start.md` — First-pass `uECoG`-only implementation scope
- `docs/implementation_tasks.md` — Active blockers and implementation tasks

### Archived but useful
- `docs/archive/experiment_log.md` — Full experiment history (101 findings)
- `docs/archive/research_synthesis.md` — 19-paper literature synthesis (seegnificant added)
- `docs/archive/reading_list.md` — Historical paper reading order and notes
- `docs/archive/literature_findings.md` — Historical literature-driven findings and data-scaling context

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
│   ├── atlas.py            # Brainnetome atlas ROIs: 16 core + 8 extended positions + parcellation lookup (v14)
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

### Electrode Coordinates (ACPC bookkeeping mostly checked; ACPC→MNI still blocked)

Coordinates are in **ACPC space** (per-patient, AC-PC aligned), NOT MNI-152. Source: `Box/ECoG_Recon/<subj>/elec_recon/<subj>_elec_locations_RAS_brainshifted.txt`. Format: `prefix electrode_num x y z hemisphere type`.

**What is currently trusted**:
- 128-ch ACPC bookkeeping: `fif ch N → chanMap[r,c]==N → phys_elec = r*16+c+1 → RAS(x,y,z)`. This fixed a large indexing error relative to ignoring `chanMap`.
- 256-ch ACPC bookkeeping: `fif ch N → RAS electrode N` directly for the standard 256-ch arrays, with `+1` handling for the 0-indexed `S57/S58` convention.
- `build_electrode_coordinates()` encodes the current ACPC-side mapping logic.

**What is NOT yet trusted**:
- the downstream **ACPC → MNI** transform path
- any quantitative cross-patient overlap analysis derived from that path
- any statement that the current coordinates are already in their final atlas-ready form

Treat the ACPC-side bookkeeping and the ACPC→MNI transform as separate issues. The former is mostly checked; the latter is still an active blocker and must be re-verified against Zac's MATLAB path before Phase 1 implementation.

**DCC TSV vs RAS files**: DCC electrode TSVs have normalized 0-1 grid coordinates (synthetic, for older per-patient Conv2d baselines). RAS files have real ACPC coordinates and are the relevant source for `v14`. TSV grid = vertically-flipped chanMap (cosmetic, irrelevant for Conv2d).

**Coordinate acquisition**: Electrode positions were obtained by measuring 4 array corners → interpolating the grid → projecting onto a smoothed cortical surface. This suggests relative within-array positions should be much better than absolute cross-patient placement, but any stronger claim about systematic rigid-body error or the exact correction model should be treated as provisional until the ACPC→MNI pipeline is verified.

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

Sig channel filtering did NOT improve per-patient decoding for S14 (85% sig) in the older Conv2d baseline, but the Phase 1 `v14-core` channel-inclusion policy is still unresolved. Treat `all non-artifact channels` versus `sig-only` as an explicit blocker to freeze before implementation.

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

### Inter-Patient Spatial Mismatch

Arrays are placed by surgeon, not experimentally standardized. The important operational fact is:

- there is **no shared channel-index space across patients**, and only partial anatomical overlap in where arrays happen to land

An older inter-patient MNI overlap analysis was run before the current ACPC→MNI pipeline was re-flagged as incorrect, so those quantitative overlap numbers should no longer be treated as trustworthy. For `v14`, the active solution is not electrode-overlap matching; it is mapping each patient's electrodes into a shared Brainnetome parcel/subparcel space after the coordinate pipeline is re-verified.

### Brainnetome Core Parcels (`atlas.py`)

16 core ROIs — top 16 by patient reachability from systematic check of all 123 LH Brainnetome ROIs + speech-relevant candidates (2026-04-06). Top 15 have ≥4 patients; #16 (A2) has 3.
- Motor (3): A6cvl (ventral PMC, 9pts), A4tl (tongue M1, 7pts), A4hf (face M1, 6pts)
- Sensory (3): A1/2/3tonIa (tongue S1, 8pts), A1/2/3ulhf (face S1, 5pts), A2 (proprioceptive, 3pts)
- Broca's (6): A44d (8pts), A45c (8pts), A44v (7pts), A45i (7pts), A45r (5pts), A44op (4pts)
- Auditory (2): STGpp (planum polare, 4pts), STGa (anterior STG, 4pts)
- Insula (1): INSa (articulatory planning, 4pts)
- Executive (1): MFG (dorsolateral PFC, 6pts)

8 extended ROIs for analysis (parietal, SMA — mostly unreachable). Older centroid-based helper functions still exist in `atlas.py` for ROI indexing and historical analysis, but the active `v14` path is volumetric atlas membership into parcel/subparcel tokens, not centroid-based routing or distance-thresholded VE selection.

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

## Completed Exploration (summary — details in `docs/archive/experiment_log.md`)

Historical literature context and older paper-positioning notes now live under `docs/archive/`. The active implementation contract is defined by:

- `docs/neural_field_perceiver_v14.tex`
- `docs/current_direction.md`
- `docs/implementation_start.md`
- `docs/implementation_tasks.md`

- **LOPO** (55 experiments): Converged to PER 0.750-0.780 on S14. Measurement ceiling from fixed CV folds.
- **SSL / NCA-JEPA**: All methods near-chance on ~11 min epoched data. CoganLab-only SSL limited (~24h, intra-op data quality). Primary SSL plan: external chronic ECoG from Flinker (48 pts) and/or Chang (~15-25 pts), est. 50-100h of diverse speech. Contingent on data access (PI-level request). Fallback: CoganLab sEEG + uECoG (~24h).
- **Per-patient tuning**: CTC→CE (+7.8pp), pool(2,4)→pool(4,8), stride=10, H=32 sufficient.
- **Per-phoneme MFA sweep** (2026-04-04): Per-phoneme flat (0.734) beats learned attention (0.797) and full-trial (0.807). Generalizes 8/11 patients.

## Archived Literature Context

Broader literature findings and data-scaling context now live in:

- `docs/archive/literature_findings.md`
- `docs/archive/research_synthesis.md`
- `docs/archive/reading_list.md`

Those are useful for paper framing and later scaling decisions, but they are no longer part of the active implementation contract in this file.

## Preprocessing Pipeline (do not change)

Decimate 2kHz → CAR → impedance exclusion (log10>6) → 70-150Hz Gaussian filterbank (8 bands) → Hilbert envelope → sum → 200Hz → z-score → significant channel selection. Implemented in `coganlab/IEEG_Pipelines`.

## Conventions

- Explain every design choice with precedent and tradeoffs
- Batch work in smaller chunks to prevent context rot
- Keep markdown lean and information-dense
- Report per-patient results, not just population means
- Always use grouped-by-token CV (never stratified — token leakage inflates by ~10pp)
- All training on DCC, never local
