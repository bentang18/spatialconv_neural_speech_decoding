# Cross-Patient Speech Decoding from Intra-Op uECOG

## Project

Ben Tang, Greg Cogan Lab, Duke. Collaborating with Zac Spalding.
Extending Spalding 2025 (PCA+CCA, SVM/Seq2Seq, 8 patients, 9 phonemes, 0.31 bal. acc.).

**Task**: Non-word repetition (52 CVC/VCV tokens, 3 phonemes each, e.g. /abe/; 9 phonemes). Intra-operative, left sensorimotor cortex, 128/256-ch uECOG arrays. ~1 min utterance/patient. Stimulus-to-response delay: 1.1 ± 0.3s (Duraivel 2023); stimulus duration ~500ms; utterance ~450ms. Auditory stimulus ends ~600ms before response onset (t=0).

**Patients**: 11 unique PS patients (S18 excluded — no preprocessed; S36 excluded — duplicate of S32): S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62. 8 are Spalding's published set. 46–178 trials/pt, 63–201 sig channels. Core set (Phase 1, all LH): S14, S26, S33, S62. Extended (Phase 1, LH only, per `implementation_tasks.md` #30): S16, S23, S39. Deferred to Phase 2 with the sEEG join (RH, per #30): S22, S58. Excluded from Phase 1 entirely: S32 (no HG response), S57 (52/256 sig, hybrid strip).

## Current Direction: Neural Field Perceiver (v14) — Intracranial Foundation Model

**Design doc**: `docs/neural_field_perceiver_v14.tex`. **Per-patient tables and data reference**: `docs/data_reference.md`.

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
Shared interface: `(N_tok × d × T)` atlas/subparcel token tensor. `N_tok = 15` LH Brainnetome parcels, frozen 2026-04-14 under the argmax-centric rule `argmax_wins >= 10` on the frozen spatial pipeline (raw MNI + 8 mm PM dilation + σ=1.5 mm Gaussian). Uniform `k_parcel = 1`. Canonical list in `src/speech_decoding/v14/token_spec.py`; rationale in `docs/implementation_tasks.md` #4. Mean+gradient pooling is the main linear ablation. No Fourier PE. No global electrode->region cross-attention.

**Current Phase-1 implementation contract** (2026-04-13):
- **Real Brainnetome PM volume only.** Source: `data/atlas/BNA_PM_4D.nii.gz`. `~/nilearn_data/bnatlas.nii.gz` is for ROI indexing and sanity checks only. No fallback to the old smoothed-MPM proxy.
- **Fixed-atlas `v14-core`, no learned calibration yet.** Phase 1 freezes the spatial interface to the verified ACPC→MNI path. No learned `Δ/ω`, `δ_l`, or `τ_l`. The coordinate pipeline is still a blocker — re-verify with Zac before coding.
- **No extra gain / impedance correction.** Inputs come from existing `productionZscore_highgamma` features. Don't add another gain/offset layer or channel-stat normalization.
- **Within-parcel Perceiver summarizer is the default.** Canonical parcel-frame coordinates + shared point encoder + fixed latent queries summarize each parcel into 1–2 tokens. Mean+gradient pooling is the main linear ablation.
- **Shared temporal front-end is still a blocker.** The next architectural decision is the exact temporal layer and its output contract into the parcel summarizer. Don't harden downstream code until that is frozen.
- **Unsupported parcels are masked, not hallucinated.** Fixed `N_tok` layout with `token_mask` and `token_support`. Zero-filled inactive slots are a storage convenience only. Support statistic is DECIDED (`implementation_tasks.md` #5, PM-weighted sum). Exact unsupported-vs-weak threshold (#3) is still a blocker.
- **Inter-region attention is token-space, not sensor-space.** The backbone operates over atlas/subparcel tokens. SC/FC bias initialization is intended; the exact token-level expansion rule is still a blocker.
- **Phase 1 is supervised-only on `uECoG`.** First milestone: supervised `v14-core` on the existing intra-op `uECoG` data. Verify token construction, masking, and end-to-end correctness before SSL, `sEEG`, or external datasets.
- **Next step after `v14-core` is full-corpus `uECoG` SSL** — not response-locked-only SSL.
- **No implementation before blocker review.** The list in `docs/implementation_tasks.md` must be discussed and frozen first.

**Paper direction**: Atlas-grounded common-space decoding for intracranial field potentials. The bet: electrodes are patient-specific observations; parcel/subparcel tokens are the shared representation. Phase 1 is the fixed-atlas supervised correctness pass only — broader calibration claims come later.

**Near-term sequencing**:
- Phase 1: supervised `v14-core` on response-locked `uECoG`
- Phase 1.5: SSL on the full continuous `uECoG` corpus
- Phase 2: learned per-patient calibration
- Only after that: `sEEG`, external datasets, and broader scaling

**v14 is the sole active direction.** v12 (cross-attention + distance bias + Fourier PE), Conv2d pipeline, JEPA, LeWM, LOPO autoresearch — all discontinued. SSL planning, historical literature, and data-scaling notes live under `docs/archive/`.

## Working Principle: Discuss Before Code (2026-04-13)

**v14 is slow, methodical, and precise. Everything before v14 was playing around.**

Every piece of the pipeline — raw voltages through phoneme decode — follows this rule:

1. **Discuss the logic first.** What it does, what it assumes, what it consumes and produces, why it is right for our data, what would make it wrong. Precedent and trade-offs are required, not optional.
2. **Agree on the contract.** Inputs, outputs, shapes, units, and the exact boundary with the pieces on either side. Ambiguity is a blocker.
3. **Only then write code.** A minimal, faithful encoding of the agreed logic. No silent extras.
4. **No pre-committed numeric defaults.** Window sizes, `d_model`, stride, hidden widths, thresholds, split counts — all justified before they enter the code, not after.
5. **Rewrite from scratch when needed.** No file or design is sacred. If it turns out handwavy, rewrite — don't patch.
6. **No legacy reuse, ever.** Pre-v14 code lives under `src/speech_decoding/archive/legacy/` and fails `import` from active code — `tests/v14/test_no_legacy_imports.py` enforces this. If a legacy helper looks useful, re-derive it fresh inside `src/speech_decoding/v14/`. Never `git mv` anything out of `archive/legacy/`. This is the only place the no-legacy rule is stated; treat it as canonical.
7. **Freeze blockers before coding.** See `docs/implementation_tasks.md`. Do not implement a component while its blockers are open.
8. **Prefer standard, scalable contracts when they do not compromise Phase 1 correctness.** If two choices are equally valid for Phase 1, choose the one that keeps a cleaner path to cross-task use, external datasets, and broader scaling. Do not add speculative infrastructure early, but do avoid Phase-1-only conventions when a standard reusable contract works just as well.

This applies to every logic step: channel indices, channel-to-electrode bookkeeping, coordinate-frame verification, Brainnetome PM lookup, parcel support, parcel-frame construction, temporal front-end, local summarizer, inter-region attention, AR decoder, loss, eval split, metrics. No step is too obvious to skip.

## Engineering Discipline

These four rules are active on every task. They extend the Working Principle above — the Principle says *discuss before code*; these say *how* to think and act when you do.

### 1. Think before coding
Don't assume. Don't hide confusion. Surface tradeoffs.
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity first
Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Self-check: would a senior engineer say this is overcomplicated? If yes, simplify.

### 3. Surgical changes
Touch only what you must. Clean up only your own mess.
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that *your* changes made unused. Don't remove pre-existing dead code unless asked.

Test: every changed line should trace directly to the user's request.

### 4. Goal-driven execution
Define success criteria. Loop until verified.
- "Add validation" → "write tests for invalid inputs, then make them pass"
- "Fix the bug" → "write a test that reproduces it, then make it pass"
- "Refactor X" → "ensure tests pass before and after"

For multi-step tasks, state a brief plan with a verification check per step. Strong success criteria let you loop independently; weak ones ("make it work") require constant clarification.

## Environment

- **Python ≥3.11, managed via `uv`.** `pyproject.toml` + `uv.lock` are authoritative.
- **Bootstrap**: `uv sync` creates `.venv/`. Run everything as `.venv/bin/python -m ...` — do not `source activate` and do not rely on shell state.
- **Tests**: `.venv/bin/python -m pytest tests/ -q`. The suite is intentionally tiny — only `test_phoneme_map.py`, `test_grouped_cv.py`, and everything under `tests/v14/`. New tests live under `tests/v14/` once the underlying component is discussed and written.
- **Key deps**: `torch ≥2.0`, `mne`, `mne-bids`, `transformers`. Full list in `pyproject.toml`.
- **Machine-specific BIDS paths** live in `configs/paths.yaml` (gitignored).
- **All training on DCC, never local.** See `docs/dcc_setup.md`.

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

### Data (repo)
- `data/ps_tokens.csv` — committed canonical 52-token PS manifest (`token_id`, PS notation, ARPABET, IPA, structure)

### Data (local, gitignored)
- `data/mni_coords/<subj>_RAS.txt` — ACPC electrode coordinates (11/11)
- `data/channel_maps/<subj>_channelMap.mat` — Amplifier → physical grid mapping (11/11)
- `data/channel_maps/<subj>_sigChannel.mat` — Significant channel masks (9/11, missing S32/S57)
- `data/transforms/<subj>_talairach.xfm` — ACPC → Talairach/MNI transform (11/11)
- `data/atlas/BNA_PM_4D.nii.gz` — real Brainnetome PM volume (the only v14 atlas source)

### Box mount (macOS, personal laptop)
- Mount root: `/Users/bentang/Library/CloudStorage/Box-Box/`
- FreeSurfer reconstructions: `/Users/bentang/Library/CloudStorage/Box-Box/ECoG_Recon/<pt>/` with subfolders `surf/`, `elec_recon/`, `mri/`, `label/`.
- Average subject for MNI152 projection: `.../ECoG_Recon/cvs_avg35_inMNI152/`.
- **Patient folders are `S<num>`**, matching the internal patient IDs: `S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62`. All 11 PS patients are present. Each has `surf/{lh,rh}.{pial,pial-outer-smoothed,sphere.reg,sphere-outer.reg,sphere-outer-mni.reg}` and `elec_recon/<pt>.{LEPTO,LEPTOVOX,POSTIMPLANT,electrodeNames}` plus `<pt>_elec_locations_RAS{,_brainshifted}.txt`.
- **Do not confuse `S<num>` with `D<num>`**: `ECoG_Recon/` also contains unrelated `D<num>` folders from a different Duke cohort (`D14` is 120 electrodes, not the same person as `S14`). Always use `S<num>`.
- **Avoid the `_old`, `_no_tkr`, `_kumar`, `_diag`, `_med` sibling folders** (e.g. `S14_old`, `S14_no_tkr`). These are alternative reconstructions kept for history and are not the current trusted version. Read from the plain `S<num>/` folder.

### Active docs
- `docs/neural_field_perceiver_v14.tex` — v14 design document
- `docs/current_direction.md` — current priorities and what's archived
- `docs/data_reference.md` — per-patient tables: array layouts, sig/artifact channel counts, Brainnetome parcel list, raw-corpus sizes
- `docs/implementation_tasks.md` — active blockers and decisions (single source of truth for Phase 1 gating)
- `docs/dcc_setup.md` — DCC cluster setup

### Archived but useful
- `docs/archive/experiment_log.md` — 101-finding experiment history
- `docs/archive/research_synthesis.md` — 19-paper literature synthesis
- `docs/archive/reading_list.md`, `docs/archive/literature_findings.md` — historical literature and data-scaling context

### Configs & scripts
- `configs/paths.yaml` — machine-specific BIDS paths (gitignored).
- No v14 training config and no v14 training/sweep scripts exist yet. They will be written fresh once the training loop contract is agreed. Pre-v14 YAMLs live in `configs/archive/`; pre-v14 CLIs live in `scripts/archive/legacy/`.

## Code Structure (post-quarantine, 2026-04-13)

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
├── data/phoneme_map.py     # 9 PS phonemes, PS→ARPA, normalize_label. CTC/articulatory helpers are pre-v14 contract — discuss before reuse.
├── evaluation/grouped_cv.py # grouped-by-token CV splitter (kept, contract-neutral)
└── archive/legacy/         # QUARANTINE — see Working Principle #6
```

## Data

### Loader contract (frozen per `#13`)

Baseline `v14-core` sample:

```
signal[N_ch, T]         # float32, z-scored HGA on non-artifact channels; T from #29 trial window
coords[N_ch, 3]         # MNI electrode coordinates (#1, migrating to fsaverage per #36)
token_mask[N_tok]       # Tier-1 parcel support mask per #3 (N_tok = 15, #4)
token_support[N_tok]    # PM-weighted support statistic per #5
label                   # 3-slot phoneme sequence, alphabetical ARPABET indices per #16
patient_id              # str
```

Channel inclusion is all non-artifact channels (`#11`); sig-channel masks are ablation-only. Trial epoching follows `#29` (one epoch per trial, `tmin=-0.5s`, `tmax=1.0s`). Per-patient sig-channel counts, artifact-channel counts, and array layouts live in `docs/data_reference.md`. `#34` (phoneme-loading audit) is the one remaining gate before the loader is written.

### Electrode coordinates

ACPC source: `Box/ECoG_Recon/<subj>/elec_recon/<subj>_elec_locations_RAS_brainshifted.txt`. Format: `prefix electrode_num x y z hemisphere type`.

**ACPC → MNI pipeline (`#1`):** Python port of Zac's `sub2AvgBrainClinical.m` in `src/speech_decoding/v14/coordinates.py`, projecting patient `pial-outer-smoothed` → patient `sphere-outer-mni.reg` → `cvs_avg35` `sphere-outer.reg` → `cvs_avg35` `pial-outer-smoothed`. Output cached at `data/mni_coords/<pt>_MNI152.csv`, keyed by physical electrode name. Verified against the S14 oracle (max 1.39 mm, median 0.68 mm). **Migrating to fsaverage under `#36` (decided, executing);** the cvs_avg35 cache is the reference / migration oracle.

**Amp → physical → coordinate bridge (`#12`):** 128-strip patients (`S14 S16 S22 S23 S26`) use Map 4 from local `*_channelMap.mat` with `phys_idx = r*16 + c + 1`. 256-grid patients (`S33 S39 S62`) use Map 3 from `*_channelMapAll.mat`. `S58` resolves its 12×24 zero-indexed crop onto full Map 3. The lookup key at the coordinate cache is the concatenated electrode name from `<pt>.electrodeNames`, not a row index. `S39_channelMap.mat` is non-authoritative; never load it.

**Hemisphere (`#30`):** Phase 1 is left-hemisphere only. S22 and S58 (right-hemisphere) are deferred to Phase 2 alongside the sEEG join. The old `mirror_to_left()` helper is discarded; Brainnetome has distinct L/R parcels.

**DCC TSV vs RAS**: DCC electrode TSVs have normalized 0–1 grid coordinates (synthetic, for older Conv2d baselines). RAS files are the relevant source for v14.

### .fif path and labels

Phase 1 input (per `#29`, one epoch per trial, `tmin=-0.5s`, `tmax=1.0s`):

`{bids_root}/derivatives/epoch(CAR)/sub-{id}/epoch(band)(power)/sub-{id}_task-PhonemeSequence_desc-productionZscore_highgamma.fif`

Per-trial token (e.g. `bak`, `ugae`) comes from the `value` column of `events.tsv`; the canonical 52-token inventory is `data/ps_tokens.csv`. Per-phoneme targets are derived by decomposing each trial's token into its 3-phoneme ARPABET sequence via `phoneme_map.normalize_label()` (PS → ARPABET per `#17`, alphabetical index per `#16`).

The sister phoneme-level `.fif` at `epoch(phonemeLevel)(CAR)/...` is not loaded for training. It exists as an audit-only cross-check under `#34`; its event_id mapping `{'a':1, 'ae':2, 'b':3, 'g':4, 'i':5, 'k':6, 'p':7, 'u':8, 'v':9}` is asserted under `#18`.

> **Phoneme / label-space rigor audit.** The first-pass audit surfaced blockers `#17–#25` plus the operational audit `#34`. `#17–#25` are now closed in `docs/implementation_tasks.md`; `#34` is the one remaining open blocker and gates the v14 loader. Do not write the v14 loader until `#34` closes.

## Compute: Duke DCC cluster

**Use DCC for all training.** Full docs: `docs/dcc_setup.md`.

- **SSH**: `ssh ht203@dcc-login.oit.duke.edu`
- **GPU**: 8× RTX 5000 Ada (32 GB) on `coganlab-gpu`
- **Python**: `/work/ht203/miniconda3/envs/speech/bin/python` (PyTorch 2.10.0+cu126; do NOT `conda activate`)
- **Repo**: `/work/ht203/repo/speech`
- **Data**: `/work/ht203/data/BIDS` (all 11 PS patients), `/work/ht203/data/mni_coords/`, `/work/ht203/data/channel_maps/`, `/work/ht203/data/transforms/`
- **Submit / monitor**: `sbatch scripts/<script>_dcc.sh` | `squeue -u ht203`
- **CAUTION**: `/work/ht203` auto-purges after 75 days. Copy results to `/hpc/group/coganlab/ht203/`.

## Preprocessing Pipeline (do not change)

Decimate 2kHz → CAR → impedance exclusion (log10>6) → 70-150Hz Gaussian filterbank (8 bands) → Hilbert envelope → sum → 200Hz → z-score → significant channel selection. Implemented in `coganlab/IEEG_Pipelines`.

## Completed Exploration (summary)

Full experiment history in `docs/archive/experiment_log.md`.

- **LOPO** (55 experiments): Converged to PER 0.750–0.780 on S14. Measurement ceiling from fixed CV folds.
- **SSL / NCA-JEPA**: All methods near-chance on ~11 min epoched data. CoganLab-only SSL limited (~24h, intra-op data quality). Primary plan: external chronic ECoG (Flinker 48 pts, Chang ~15-25 pts), 50–100h of diverse speech, contingent on PI-level data access. Fallback: CoganLab sEEG + uECoG (~24h).
- **Per-patient tuning**: CTC→CE (+7.8pp), pool(2,4)→pool(4,8), stride=10, H=32 sufficient.
- **Per-phoneme MFA sweep** (2026-04-04): Per-phoneme flat (0.734) beats learned attention (0.797) and full-trial (0.807). Generalizes 8/11 patients.

## Conventions

- **Write simply.** Ordinary words. Short sentences. No throat-clearing, no redundant qualifiers, no ceremonial preambles. Cut is the main edit. Applies to all docs, commit messages, PR descriptions, memory files, and chat responses. Paul Graham's "Write Simply" is the reference. Code comments stay minimal (default: none).
- **Discuss logic before writing code.** See Working Principle above.
- **All training on DCC, never local.**
