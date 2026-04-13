# Cross-Patient Speech Decoding from Intra-Op uECOG

## Project

Ben Tang, Greg Cogan Lab, Duke. Collaborating with Zac Spalding.
Extending Spalding 2025 (PCA+CCA, SVM/Seq2Seq, 8 patients, 9 phonemes, 0.31 bal. acc.).

**Task**: Non-word repetition (52 CVC/VCV tokens, 3 phonemes each, e.g. /abe/; 9 phonemes). Intra-operative, left sensorimotor cortex, 128/256-ch uECOG arrays. ~1 min utterance/patient. Stimulus-to-response delay: 1.1 ± 0.3s (Duraivel 2023); stimulus duration ~500ms; utterance ~450ms. Auditory stimulus ends ~600ms before response onset (t=0).

**Patients**: 11 unique PS patients (S18 excluded — no preprocessed; S36 excluded — duplicate of S32): S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62. 8 are Spalding's published set. 46–178 trials/pt, 63–201 sig channels. Core set: S14, S26, S33, S62. Excluded: S32 (no HG response), S57 (52/256 sig, hybrid strip). Extended: S16, S22, S23, S39, S58.

## Current Direction: Neural Field Perceiver (v14) — Intracranial Foundation Model

**Design doc**: `docs/neural_field_perceiver_v14.tex`

**Two-problem decomposition** (atlas calibration + shared dynamics):
- **Problem 1 — Calibration** (per-patient, physics-constrained): raw electrodes → atlas-grounded regional tokens via Brainnetome volumetric parcellation. Atlas does ~90% of calibration; supervised gradient refines ~10%.
- **Problem 2 — Dynamics** (shared, unconstrained ML): regional tokens → phoneme sequence via a small relational-temporal transformer + AR decoder. Same representation for every patient.

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
- **Real Brainnetome PM volume only.** Source: `data/atlas/BNA_PM_4D.nii.gz`. `~/nilearn_data/bnatlas.nii.gz` is for ROI indexing and sanity checks only. No fallback to the old smoothed-MPM proxy.
- **Fixed-atlas `v14-core`, no learned calibration yet.** Phase 1 freezes the spatial interface to the verified ACPC→MNI path. No learned `Δ/ω`, `δ_l`, or `τ_l`. The coordinate pipeline is still a blocker — re-verify with Zac before coding.
- **No extra gain / impedance correction.** Inputs come from existing `productionZscore_highgamma` features. Don't add another gain/offset layer or channel-stat normalization.
- **Within-parcel Perceiver summarizer is the default.** Canonical parcel-frame coordinates + shared point encoder + fixed latent queries summarize each parcel into 1–2 tokens. Mean+gradient pooling is the main linear ablation.
- **Shared temporal front-end is still a blocker.** The next architectural decision is the exact temporal layer and its output contract into the parcel summarizer. Don't harden downstream code until that is frozen.
- **Unsupported parcels are masked, not hallucinated.** Fixed `N_tok` layout with `token_mask` and `token_support`. Zero-filled inactive slots are a storage convenience only. Exact support statistic and unsupported-vs-weak threshold are still blockers.
- **Inter-region attention is token-space, not sensor-space.** The backbone operates over atlas/subparcel tokens. SC/FC bias initialization is intended; the exact token-level expansion rule is still a blocker.
- **Phase 1 is supervised-only on `uECoG`.** First milestone: supervised `v14-core` on the existing intra-op `uECoG` data. Verify token construction, masking, and end-to-end correctness before SSL, `sEEG`, or external datasets.
- **Next step after `v14-core` is full-corpus `uECoG` SSL** — not response-locked-only SSL.
- **No implementation before blocker review.** The list in `docs/implementation_start.md` and `docs/implementation_tasks.md` must be discussed and frozen first.

**Paper direction**: Atlas-grounded common-space decoding for intracranial field potentials. The bet: electrodes are patient-specific observations; parcel/subparcel tokens are the shared representation. Phase 1 is the fixed-atlas supervised correctness pass only — broader calibration claims come later.

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

## Working Principle: Discuss Before Code (2026-04-13)

**v14 is slow, methodical, and precise. Everything before v14 was playing around.**

Every piece of the pipeline — raw voltages through phoneme decode — follows this rule:

1. **Discuss the logic first.** What it does, what it assumes, what it consumes and produces, why it is right for our data, what would make it wrong. Precedent and trade-offs are required, not optional.
2. **Agree on the contract.** Inputs, outputs, shapes, units, and the exact boundary with the pieces on either side. Ambiguity is a blocker.
3. **Only then write code.** A minimal, faithful encoding of the agreed logic. No silent extras.
4. **No pre-committed numeric defaults.** Window sizes, `d_model`, stride, hidden widths, thresholds, split counts — all justified before they enter the code, not after.
5. **Rewrite from scratch when needed.** No file or design is sacred. If it turns out handwavy, rewrite — don't patch.
6. **No legacy fallback.** Pre-v14 code lives under `src/speech_decoding/archive/legacy/`, blocked by `tests/v14/test_no_legacy_imports.py`. If a legacy helper looks useful, re-derive it fresh inside `src/speech_decoding/v14/`.
7. **Freeze blockers before coding.** See `docs/implementation_tasks.md`. Do not implement a component while its blockers are open.

This applies to every logic step: channel indices, channel-to-electrode bookkeeping, coordinate-frame verification, Brainnetome PM lookup, parcel support, parcel-frame construction, temporal front-end, local summarizer, inter-region attention, AR decoder, loss, eval split, metrics. No step is too obvious to skip.

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
- `configs/paths.yaml` — Machine-specific BIDS paths (gitignored).

No v14 training config exists yet. It will be written once the v14 training loop contract is agreed. Do not copy a legacy YAML and rename it.

### Scripts

No active v14 training/sweep scripts. All pre-v14 CLIs (`train_per_patient.py`, `train_lopo.py`, `sweep_*.py`, plotting/diagnostic utilities, `autoresearch/`, `autoresearch_lopo/`) are quarantined under `scripts/archive/legacy/`. New v14 scripts will be written fresh once the training loop contract is agreed.

### Archived
- `docs/archive/` — Old NFP versions, NCA-JEPA specs, LOPO plans, historical design docs.
- `src/speech_decoding/archive/legacy/` — All pre-v14 Python code (data loaders, Conv2d models, LOPO trainers, NCA-JEPA pretraining, v12 metrics/diagnostics). See `src/speech_decoding/archive/legacy/README.md`.
- `tests/archive/legacy/` — All pre-v14 tests. Auto-excluded from pytest collection.
- `scripts/archive/legacy/` — All pre-v14 training/sweep/plotting/autoresearch scripts.
- `configs/archive/` — Pre-v14 training YAMLs.

## Code Structure (post-quarantine, 2026-04-13)

Everything pre-v14 is quarantined under `src/speech_decoding/archive/legacy/` and fails `import` from active code. `tests/v14/test_no_legacy_imports.py` enforces this — no file under `src/speech_decoding/v14/` may import any legacy path.

```
src/speech_decoding/
├── v14/                    # Phase-1 implementation home (WIP, discussion-first)
│   ├── __init__.py         # re-exports config dataclasses + default_token_count
│   ├── config.py           # AtlasConfig / PatientCalibrationConfig / TemporalTokenizerConfig / LocalSummarizerConfig / BackboneConfig / DecoderConfig / V14Config
│   ├── token_spec.py       # DEFAULT_BASE_PARCELS (16) + DEFAULT_SPLIT_COUNTS (→21 tokens) — PROVISIONAL, still a blocker
│   ├── calibration.py      # atlas-resource path helpers (stub; no logic yet)
│   ├── tokenizer.py        # (stub) shared temporal front-end
│   ├── local_summarizer.py # (stub) within-parcel Perceiver summarizer
│   ├── backbone.py         # (stub) inter-region + temporal attention
│   ├── decoder.py          # (stub) 3-query AR decoder
│   └── model.py            # (stub) top-level assembly
├── data/
│   ├── __init__.py
│   └── phoneme_map.py      # label space (9 PS phonemes, PS→ARPA, normalize_label) — CTC/articulatory helpers inside this file are pre-v14 contract and are to be discussed before reuse
├── evaluation/
│   ├── __init__.py
│   └── grouped_cv.py       # grouped-by-token CV splitter (contract-neutral, kept)
└── archive/
    └── legacy/             # QUARANTINE — see archive/legacy/README.md; do not import
        ├── data/           # bids_dataset, grid, augmentation, collate, atlas, coordinates, sig_channels, audio_features
        ├── models/         # spatial_conv, backbone, flat_head, articulatory_head, linear_readin, ce_position_head, assembler
        ├── training/       # trainer, lopo_trainer, adaptor, lopo, ctc_utils, mfa_guided_trainer, phonological_aux*
        ├── pretraining/    # NCA-JEPA + BYOL / DINO / VICReg / LeWM / synthetic generators / stage1-3
        └── evaluation/     # metrics (PER helper + v12 per-position), content_collapse
```

Run: `.venv/bin/python -m pytest tests/ -q`. The suite is intentionally tiny — only `test_phoneme_map.py`, `test_grouped_cv.py`, and everything under `tests/v14/`. Any new test lives under `tests/v14/` once its underlying v14 component is discussed and written.

## Data

### Loading

**The old grid-based loader is quarantined.** `load_patient_data` / `load_per_position_data` returned `grid_data[H, W, T]` — wrong for v14's token-space pipeline. They also zero artifact channels in place, which biases parcel-support stats. Do not import.

The v14 loader is not written. Its contract is under discussion. Expected sample shape (pending agreement):

```
signal[N_ch, T]         # float32, z-scored HGA on non-artifact channels
mni_coords[N_ch, 3]     # verified ACPC→MNI positions (blocked on coord pipeline)
token_mask[N_tok]       # which atlas/subparcel tokens are supported for this patient
token_support[N_tok]    # support statistic (exact formula is a blocker)
label                   # phoneme sequence — decoder contract is a blocker
patient_id              # str
```

Nothing is locked. Every field is a separate discussion before code.

### Array Layouts (ground truth: `data/recording_details/uecog_recording_details.xlsx`)

Per-patient wiring comes from the Box `uECoG Recording Details` spreadsheet (`Duke Subjects` sheet, `Electrode (Mapping)` column), plus the per-patient files in `data/channel_maps/`. The old "infer the grid from the TSV" heuristic is retired.

| Array | Map | Shape | Populated | Pitch | Patients |
|---|---|---|---|---|---|
| 128 Strip | Map 4 | 8×16 | 128 / 128 | 1.33 mm | S14, S16, S22 (RH), S23, S26 |
| 256 Grid | Map 3 | 46×24 | 256 / 1104 (I/cross) | 1.72 mm | S32, S33, S39, S58, S62 |
| 256 Hybrid Strip | Map 8 | TBD (macro + micro) | TBD | TBD | S57 |

Notes:
- **Phase-1 core patients** (S14, S26, S33, S62) use Map 3 or Map 4. S32 and S57 are excluded from Phase 1, so Map 8 and the S57 micro wiring are deferred with the patient.
- **S39 is 256 Grid Map 3.** The stray `S39_channelMap.mat` (8×16, byte-identical to the 128 Strip template) must not be loaded. The authoritative S39 map is `S39_channelMapAll.mat`.
- **S58** uses Map 3. Local `S58_channelMap.mat` is a compact (12, 24) crop of the central rows of the full (46, 24) Map 3, with values re-encoded as zero-indexed amp channels 0..255. The verifier must prove the row-slice alignment (blocker #12).
- **S62 `Duke Subjects` row is stale** — it says `Electrode Used? No`, but the recording is real and the `Speech task` sheet gives the right numbers.
- Map 4 is 0-indexed in the spreadsheet; the local `*_channelMap.mat` is 1-indexed. Same layout, +1 offset.
- All local `*_channelMapAll.mat` files (S32, S33, S39, S62) are byte-identical. Map 3 is a single generic layout, not per-patient.

The 1-to-1 amp-channel → physical-electrode → RAS bridge is a discussion item, not a written function (blocker #12).

### Electrode Coordinates (ACPC bookkeeping mostly checked; ACPC→MNI still blocked)

Coordinates are in **ACPC space** (per-patient, AC-PC aligned), NOT MNI-152. Source: `Box/ECoG_Recon/<subj>/elec_recon/<subj>_elec_locations_RAS_brainshifted.txt`. Format: `prefix electrode_num x y z hemisphere type`.

**What is currently trusted**:
- 128-ch ACPC bookkeeping: `fif ch N → chanMap[r,c]==N → phys_elec = r*16+c+1 → RAS(x,y,z)`. This fixed a large indexing error relative to ignoring `chanMap`.
- 256-ch ACPC bookkeeping: `fif ch N → RAS electrode N` directly for the standard 256-ch arrays, with `+1` handling for the 0-indexed `S57/S58` convention.
- `build_electrode_coordinates()` encodes the current ACPC-side mapping logic.

**Not trusted**:
- the ACPC → MNI transform path
- any cross-patient overlap analysis derived from it
- any claim that the current coordinates are atlas-ready

ACPC bookkeeping and ACPC→MNI are separate issues. The first is mostly checked. The second is an active blocker — re-verify with Zac's MATLAB path before Phase 1.

**DCC TSV vs RAS files**: DCC electrode TSVs have normalized 0–1 grid coordinates (synthetic, for older Conv2d baselines). RAS files have real ACPC coordinates and are the relevant source for v14. TSV grid = vertically-flipped chanMap (cosmetic).

**Coordinate acquisition**: Positions came from measuring 4 array corners → interpolating the grid → projecting onto a smoothed cortical surface. Relative within-array positions should beat absolute cross-patient placement, but any stronger claim about rigid-body error is provisional until ACPC→MNI is verified.

**Hemisphere**: S22 and S58 are right-hemisphere (positive x). All others left. The old `mirror_to_left()` helper flipped x to fake left-hemisphere electrodes — cosmetic for the Conv2d grid, **wrong for volumetric membership** (Brainnetome has distinct L/R parcels). It's quarantined. v14 must route right-hemisphere patients to the right parcels directly. Exact rule is a discussion item.

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

Sig-channel filtering did not improve S14 (85% sig) in the Conv2d baseline. The v14 channel-inclusion policy is still unresolved — treat `all non-artifact` vs `sig-only` as a blocker.

### Artifact Channels (electronic, not brain signal)

Some channels show extreme activations (>10 std in >5% of trials) — electronic artifacts from mic feedback / amp saturation, confirmed by Zac. **Exclude entirely** (clipping leaves confounded signal). The legacy `detect_artifact_channels()` zeroed in place to preserve `(H,W)` — wrong for v14, since zeroed rows inflate parcel-support denominators. v14 must drop channels from the signal tensor and the coordinate tensor together. Discussion item.

| Patient | Chronic artifact ch | Max value (std) |
|---------|-------------------|-----------------|
| S14 | 0 | 43 |
| S26 | 4 | 15 |
| S39 | **20** | **627** |
| S57 | **15** | 83 |
| S58 | **37** | 149 |

S39/S57/S58 are the worst. S14/S16/S23/S32 are clean (0 chronic).

### Inter-Patient Spatial Mismatch

Arrays are placed by surgeon, not standardized. Key fact: **no shared channel-index space across patients**, only partial anatomical overlap in where arrays land.

An older MNI overlap analysis predates the ACPC→MNI re-flag — those numbers are no longer trustworthy. v14 solves this not by electrode matching but by mapping each patient into shared Brainnetome parcel/subparcel space (after coordinates are verified).

### Brainnetome Core Parcels (provisional 16-parcel candidate list)

The provisional 16-parcel base set, chosen as the top 16 LH ROIs by patient reachability from a systematic check of all 123 LH Brainnetome ROIs + speech-relevant candidates (2026-04-06). Top 15 have ≥4 patients; #16 (A2) has 3.

- Motor (3): A6cvl (ventral PMC, 9pts), A4tl (tongue M1, 7pts), A4hf (face M1, 6pts)
- Sensory (3): A1/2/3tonIa (tongue S1, 8pts), A1/2/3ulhf (face S1, 5pts), A2 (proprioceptive, 3pts)
- Broca's (6): A44d (8pts), A45c (8pts), A44v (7pts), A45i (7pts), A45r (5pts), A44op (4pts)
- Auditory (2): STGpp (planum polare, 4pts), STGa (anterior STG, 4pts)
- Insula (1): INSa (articulatory planning, 4pts)
- Executive (1): MFG (dorsolateral PFC, 6pts)

Lives as `DEFAULT_BASE_PARCELS` in `src/speech_decoding/v14/token_spec.py`. **Provisional** — pre-dates Phase 1, must be re-discussed before it locks. Same for the 2-token splits in `DEFAULT_SPLIT_COUNTS` (`A6cvl`, `A4hf`, `A1/2/3ulhf`, `A2`, `A1/2/3tonIa`) that give `N_tok = 21` — an open blocker in `docs/implementation_tasks.md`.

Older centroid-VE logic (reachability thresholds, distance-to-ROI routing, 25/15 mm thresholds) is quarantined under `archive/legacy/data/atlas.py`. v14 uses volumetric Brainnetome PM membership, not centroid routing.

### Raw Continuous Recordings (for SSL)

456 min across 29 patients (13 PS + 17 Lexical, zero patient overlap). Raw 2kHz EDF files in BIDS: `sub-{id}/ieeg/sub-{id}_task-{task}_acq-01_run-01_ieeg.edf`. Need HGA extraction (CAR → 70-150Hz filterbank → Hilbert → 200Hz) to match existing productionZscore features. PS: ~199 min, Lexical: ~257 min. S14 longest at 31 min.

### .fif Path
`{bids_root}/derivatives/epoch(phonemeLevel)(CAR)/sub-{id}/epoch(band)(power)/sub-{id}_task-PhonemeSequence_desc-productionZscore_highgamma.fif`

PS labels: `{'a':1, 'ae':2, 'b':3, 'g':4, 'i':5, 'k':6, 'p':7, 'u':8, 'v':9}` — `phoneme_map.normalize_label()` handles PS → ARPABET conversion.

> **Phoneme / label-space rigor audit (2026-04-13) — nothing below is locked.** A first-pass audit of the label space and the upstream assumptions the pre-v14 loader silently inherited surfaced several discussion items, now tracked as blockers **#17–#25** in `docs/implementation_tasks.md`. Most notable: the inherited `PS2ARPA` mapping `ae → EH` and `u → UH` is very likely phonetically wrong (standard ARPABET would be `ae → AE` for /æ/ and `u → UW` for /u/), and the event_id mapping quoted above has never been positively asserted at load time. Do not treat any phoneme-level metric as trustworthy and do not write a v14 data loader until the audit blockers are resolved.

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

- **Write simply.** Ordinary words. Short sentences. No throat-clearing, no redundant qualifiers, no ceremonial preambles. Cut is the main edit. Applies to all docs, commit messages, PR descriptions, memory files, and chat responses. Paul Graham's "Write Simply" is the reference. Code comments stay minimal (default: none) per the existing rule.
- **Discuss logic before writing code.** See Working Principle above. Every function, shape, threshold, and default — discussed, agreed, understood, *then* written.
- **No legacy reuse.** If a pre-v14 helper looks useful, rewrite it fresh in `src/speech_decoding/v14/`. Never `git mv` anything out of `archive/legacy/`.
- **All training on DCC, never local.**
