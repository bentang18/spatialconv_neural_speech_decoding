# Cross-Patient Speech Decoding from Intra-Op uECOG

## Project

Ben Tang, Greg Cogan Lab, Duke. Collaborating with Zac Spalding.
Extending Spalding 2025 (PCA+CCA, SVM/Seq2Seq, 8 patients, 9 phonemes, 0.31 bal. acc.).

**Task**: Non-word repetition (52 CVC/VCV tokens, 3 phonemes each, e.g. /abe/; 9 phonemes). Intra-operative, left sensorimotor cortex, 128/256-ch uECOG arrays. ~1 min utterance/patient. Stimulus-to-response delay: 1.1 ± 0.3s (Duraivel 2023); stimulus ~500ms; utterance ~450ms. Auditory stimulus ends ~600ms before response onset (t=0).

**Patients**: 11 unique PS (S18 no preprocessed; S36 duplicate of S32): S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62. 46–178 trials/pt, 63–201 sig channels. Per-patient tables: `docs/references/data_reference.md`. Stage-specific patient scope in `docs/strategy/stage_<N>.md`.

## Current focus: Neuroprobe cross-subject hillclimb

**PS/uECoG stage program is paused 2026-04-24.** Active work is the Neuroprobe cross-subject leaderboard as external validation of the v14 atlas-anchored thesis. Return to the stage program after Neuroprobe lands (submit or abort).

**Live plan + rationale**: `docs/neuroprobe/plan.md`. Targets: **≥ 0.56 to submit, ≥ 0.58 stretch, < 0.539 abort.** Benchmark reference: `docs/references/neuroprobe_benchmark.md`.

**Active v14 default (Neuroprobe path, 2026-04-25)**: `per_cell + BNA-connectivity-init-attn-bias + flat-per-parcel-pool @ d ∈ {32, 64, 128} swept, depth=3` + RoPE temporal-only + BNA `P_emb` shared across subjects + no per-subject linear layer. **Stage-2 SSL pre-committed (single-loss, Evanson-aligned)**: D-SigLIP brain ↔ frozen Whisper-large mid-encoder + trained projection heads. Full design + deltas vs PS-pause default: `memory/project_v14_reeval_kinglab_eegfm_2026_04_25.md`.

The "Stage" language inside `docs/neuroprobe/plan.md` refers to *hillclimb stages*, not the PS program's stages. When this doc says "Stage 1", it means PS stage unless inside a Neuroprobe context.

## v14 program (paused — resumes after Neuroprobe hillclimb)

Atlas-grounded parcel tokens as the shared representation across patients and sensors. Two-problem decomposition:
1. **Calibration** (per-patient, physics-constrained): raw electrodes → atlas-grounded tokens via Brainnetome surface parcellation on fsaverage. Fixed atlas through PS-Stage 2; learned calibration deferred to PS-Stage 3+.
2. **Dynamics** (shared, unconstrained ML): tokens → phoneme sequence via a small relational-temporal transformer + AR decoder. Same representation for every patient.

**Triad doc layer** (for PS stage work — resumes after Neuroprobe):
- **Objectives** (program hypothesis + stage roadmap + advance gates): `docs/objectives.md`
- **Strategy** (per-stage architecture, frozen contract, scoreboard, rejected paths): `docs/strategy.md` → `docs/strategy/stage_<N>.md`
- **Tactics** (in-flight jobs, blockers, next actions): `docs/tactics.md`
- **Results log**: `docs/experiments/v14_ablation_log.csv`
- **Per-patient tables + corpus + Brainnetome parcel list**: `docs/references/data_reference.md`

**PS stages** (per `docs/objectives.md` — paused):

| Stage | Scope | Strategy |
|---|---|---|
| Stage 1 | Single-sensor supervised correctness pass on uECoG | `stage_1.md` (closed 2026-04-20) |
| Stage 2 | In-sensor scaling: PS + lex uECoG + continuous-corpus SSL | `stage_2.md` (paused 2026-04-24) |
| Stage 3 | Cross-sensor join (+ Cogan sEEG D-cohort) | *TBD* |
| Stage 4 | External-lab validation | *TBD* |

Frozen default at PS pause (2026-04-20, historical): `per_cell + partialconv + pe2d + hierarchical_atlas @ d=32, depth=3, pool=(4,8)`. Full pipeline, contract: `docs/strategy/stage_1.md`. `docs/tactics.md` reflects pre-pause Stage-2 priorities; revisit on resume.

**Note on resume**: PS-resume keeps the original `partialconv + pe2d + hierarchical_atlas` defaults — the 2-D Utah grid is real on PS uECoG, so those primitives are appropriate. The Neuroprobe-path changes (BNA-connectivity bias, flat per-parcel pool, no partialconv) are sEEG-specific. Two paths now diverge by modality, not by program version.

**v14 is the sole active architectural direction.** v12 (cross-attention + distance bias + Fourier PE), Conv2d pipeline, JEPA, LeWM, LOPO autoresearch — all discontinued. Historical notes: `docs/archive/`.

## Working Principle: Discuss Before Code

**v14 is slow, methodical, and precise. Everything before v14 was playing around.** Discuss every pipeline piece — assumptions, I/O, why right for this data, what would make it wrong, precedent and trade-offs — before any code:

1. **Agree on the contract.** Inputs, outputs, shapes, units, boundaries. Ambiguity is a blocker.
2. **No pre-committed numeric defaults.** Window sizes, `d_model`, widths, thresholds — all justified before landing.
3. **Rewrite from scratch when needed.** No file is sacred. If handwavy, rewrite — don't patch.
4. **No legacy reuse, ever.** Pre-v14 lives under `src/speech_decoding/archive/legacy/` and fails `import` from active code; `tests/v14/test_no_legacy_imports.py` enforces this. Re-derive fresh in `src/speech_decoding/v14/`. Never `git mv` out of `archive/legacy/`. Canonical rule; stated only here.
5. **Freeze blockers before coding.** See `docs/tactics.md`.
6. **Prefer standard, scalable contracts** when they do not compromise current-stage correctness. Avoid stage-scoped conventions when a reusable contract works equally well.

Applies to every logic step: channel indices, coordinate frames, Brainnetome PM lookup, per-electrode support, parcel embedding, temporal front-end, backbone, decoder, loss, eval, metrics. No step is too obvious to skip.

## Engineering Discipline

### 1. Think before coding
- State assumptions. If uncertain, ask.
- Present multiple interpretations — don't pick silently.
- Push back when a simpler approach exists.
- If unclear, stop. Name what's confusing.

### 2. Simplicity first
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" that wasn't requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite.

### 3. Surgical changes
- Don't "improve" adjacent code.
- Don't refactor what isn't broken.
- Match existing style.
- Unrelated dead code: mention, don't delete.
- Remove only what *your* changes made unused.

### 4. Goal-driven execution
Define success criteria. Loop until verified.
- "Add validation" → "write tests for invalid inputs, then make them pass"
- "Fix the bug" → "write a test that reproduces it, then make it pass"
- "Refactor X" → "tests pass before and after"

For multi-step tasks, state a brief plan with a verification check per step.

### 5. Verify before claiming done
Never claim tests pass, code works, or a bug is fixed without running the verification command in the current turn. "Should work," "probably fixed," "seems to pass" are lies, not confidence. No "Great!"/"Perfect!"/"Done!" before evidence.

### 6. Debug root cause, not symptoms
Read the full error, reproduce, check recent changes, log at every boundary before proposing a fix. "Just one quick fix" masks the real issue.

### 7. No performative agreement on pushback
When the user questions a design, do not capitulate with "You're absolutely right!" and start rewriting. Restate the technical point, verify against the codebase, and push back with reasoning if the feedback is wrong for this context. Technical correctness over social comfort.

### 8. Empirical iteration over armchair design
Discussion surfaces the known unknowns and the contract, but once done, *run the experiment*. Ablation deltas, LOPO numbers, loss curves beat another reasoning pass. When brainstorms stack elaborate architectures on shaky citations, stop reasoning and measure.

## Environment

- **Python ≥3.11, `uv`-managed.** `pyproject.toml` + `uv.lock` authoritative.
- **Bootstrap**: `uv sync` creates `.venv/`. Run everything as `.venv/bin/python -m ...` — do not `source activate`.
- **Tests**: `.venv/bin/python -m pytest tests/ -q`. Suite is intentionally tiny (`test_phoneme_map.py`, `test_grouped_cv.py`, everything under `tests/v14/`).
- **Machine-specific BIDS paths**: `configs/paths.yaml` (gitignored).
- **All training on DCC, never local.** See `docs/references/dcc_setup.md`.

## Baseline to beat

**PER 0.734 ± 0.007 on S14** (grouped-by-token CV, 3-seed, per-phoneme MFA + flat head). Population mean 0.825 / 11 patients. Config details + sweep findings: `docs/archive/experiment_log.md` findings 86–101.

## Key Files

### Data (repo)
- `data/ps_tokens.csv` — canonical 52-token PS manifest (`token_id`, PS notation, ARPABET, IPA, structure).

### Data (local, gitignored)
- `data/mni_coords/<subj>_RAS.txt` — ACPC electrode coordinates (11/11).
- `data/channel_maps/<subj>_channelMap.mat` — amp → physical grid mapping.
- `data/channel_maps/<subj>_sigChannel.mat` — sig-channel masks (9/11; missing S32/S57).
- `data/transforms/<subj>_talairach.xfm` — ACPC → Talairach/MNI.
- `data/atlas/BNA_PM_4D.nii.gz` — Brainnetome PM volume (only v14 atlas source).
- `data/atlas/fsaverage_bake_v2c/` — baked fsaverage atlas (projfrac-avg + mri_surf2surf, no smoothing).
- `data/atlas/support_cache_v2c_snap/<pt>_support_tier1.csv` — per-electrode Tier-1 BNA support.
- `data/fsaverage_coords/<pt>_fsaverage_pial.csv` — strict snap-to-pial coordinates.

### Box mount (macOS laptop)
- Root: `/Users/bentang/Library/CloudStorage/Box-Box/`
- FreeSurfer recons: `ECoG_Recon/<pt>/` with `surf/`, `elec_recon/`, `mri/`, `label/`. PS patients use `S<num>`.
- ACPC electrode source: `ECoG_Recon/<subj>/elec_recon/<subj>_elec_locations_RAS_brainshifted.txt`.
- **Do not confuse `S<num>` with `D<num>`** — `D<num>` is a different Duke sEEG cohort (Stage 3 scope).
- **Avoid sibling recon folders** (`_old`, `_no_tkr`, `_kumar`, `_diag`, `_med`) — alternative reconstructions; always read from plain `S<num>/`.
- `cvs_avg35_inMNI152/` — parity-oracle template only; not active.

### Docs organization (STRICT — enforce, don't dilute)

`/docs` follows the Sun Tzu triad: **objectives → strategy → tactics**. Exactly **three working docs**. Do not add a fourth.

**Working docs:**
- `docs/objectives.md` — program hypothesis, stage roadmap, evaluation philosophy, advance gates. Stage-stable.
- `docs/strategy.md` (index) + `docs/strategy/stage_<N>.md` (per-stage) — default architecture, frozen contract, patient scope, live scoreboard, rejected paths, discipline — scoped to that stage only. Stage-N is written only when Stage-(N−1) has concluded enough to define the entry point. Do not pre-write downstream stages.
- `docs/tactics.md` — concrete task list, in-flight jobs, blockers. Refreshed when jobs land.

**Rule:** do not create additional planning, tracker, or status docs under `/docs` for the PS stage program. Extend the relevant triad doc instead. Doc surplus breaks the organization and creates stale duplicates (paid this cost multiple times — see archived `v14_next_steps_2026_04_19_morning.md`, `decision_gate_2026_04_19_snapshot.md`). When tempted, ask: "which triad layer is this? Why doesn't it live there?" Parallel initiatives outside the PS program (e.g. `docs/neuroprobe/`) get their own subdirectory with a single active plan doc.

**Reference docs** (static):
- `docs/references/data_reference.md` — per-patient tables, corpus sizes, Brainnetome parcel list, field-landscape audit.
- `docs/references/dcc_setup.md` — DCC setup, rsync recipes, submission workflow.
- `docs/references/neuroprobe_benchmark.md` — Neuroprobe benchmark reference.
- `docs/experiments/v14_ablation_log.csv` — authoritative raw results (via `scripts/v14_core/update_ablation_log.py`).
- `docs/qc/`, `docs/figures/`, `docs/README.md`.

**Archived** (historical only): `docs/archive/{sessions,plans,experiments,design_docs,experiment_log.md,research_synthesis.md,...}`. Previously-live doc forms (`current_direction.md`, `implementation_tasks.md`, `v14-core.md`) live here; the triad above supersedes them.

### Configs & scripts

- `configs/paths.yaml` — machine-specific BIDS paths (gitignored). DCC: `ps_bids_root=/work/ht203/data/BIDS`, `support_cache_dir=/work/ht203/data/atlas/support_cache_v2c_snap`, `channel_maps_dir=/work/ht203/data/channel_maps`.
- `scripts/v14_core/` — active CLI (`train_v14_core.py`), aggregator (`update_ablation_log.py`), hand-written sbatch wrappers.
- `scripts/ablation/` — **default DCC tooling.** Seven CLIs, all with `--help`:
  - `submit.py` — render sbatch from CLI spec, rsync, sbatch, record to `.ablation_submissions.jsonl`.
  - `status.py` — per-job `done/run/pend/fail` from `squeue` + `sacct`; failed tasks decoded to (fold, seed).
  - `logs.py` — peek `.out`/`.err`, default tail 30 lines.
  - `collect.py` — rsync `*.result.json` and append to the ablation CSV.
  - `query.py` — slice CSV by patient / d / depth / pool / status / job.
  - `dcc_sync_check.py` — verify `local HEAD == DCC HEAD == origin`; run before submit.
  - `peek.py` — `cat` one or more `result.json` over ssh without rsyncing.

  Use these instead of cloning a hand-written wrapper for routine ablations. Keep hand-written `scripts/v14_core/v14_*_dcc.sh` for non-standard array math (multi-cell cross-products, LOPO pretrain→finetune chains). Shared plumbing in `_common.py`. Full recipe: `docs/references/dcc_setup.md`.

- Pre-v14 YAMLs live in `configs/archive/`; pre-v14 CLIs in `scripts/archive/legacy/`.

## Code Structure

Active module layout under `src/speech_decoding/v14/`:

- **Loader + model (active)**: `phoneme_dataset.py`, `phoneme_model.py`, `phoneme_decoder.py`, `phoneme_run_fold.py`, `phoneme_run_fold_pooled.py`, `backbone.py`, `train.py`, `eval.py`, `cv.py`, `config.py`, `pool.py`, `augmentation.py`.
- **Atlas + coordinates**: `fsaverage_projection.py` (strict snap-to-pial, active), `fsaverage_atlas.py` (atlas loader + support/argmax helpers), `support_cache.py`, `channel_map.py` (amp→physical bridge), `token_spec.py` (Tier-1 15 parcels). Parity oracles: `coordinates.py`, `cvsavg_projection.py`.
- **Stage-2+ stubs** (kept for interface continuity): `calibration.py`, `tokenizer.py`, `model.py`, `decoder.py`, `grid_mixer.py`, `parcel_embedding.py`, `parcel_frames.py`, `local_summarizer.py`.
- **Other**: `aggregate.py` (fold aggregation), `electrode_pool.py` (kernel-ablation exploration), `dataset.py` + `run_fold.py` (trial-level slot-CE; kept for Stage-2+ per-electrode experiments), `audit/` (closed phoneme-loading audit).

Dataset contract, loader I/O shapes, `.fif` paths, coordinate pipeline: `docs/strategy/stage_1.md §Frozen Stage-1 contract`.

Quarantine: `src/speech_decoding/archive/legacy/` (no imports from active code, enforced by test).

## Compute: Duke DCC cluster

Full docs: `docs/references/dcc_setup.md`.

- **SSH**: `ssh ht203@dcc-login.oit.duke.edu`
- **GPU**: 8× RTX 5000 Ada (32 GB) on `coganlab-gpu`
- **Python**: `/work/ht203/miniconda3/envs/speech/bin/python` (PyTorch 2.10.0+cu126; do NOT `conda activate`)
- **Repo**: `/work/ht203/repo/speech`
- **Data**: `/work/ht203/data/BIDS` (all 11 PS), `/work/ht203/data/{mni_coords,channel_maps,transforms,atlas}/`
- **Submit / monitor**: `scripts/ablation/{submit,status,logs}.py` | `sbatch` + `squeue -u ht203`
- **CAUTION**: `/work/ht203` auto-purges after 75 days. Copy results to `/hpc/group/coganlab/ht203/`.

## Preprocessing Pipeline (do not change)

Decimate 2kHz → CAR → impedance exclusion (log10 > 6) → 70–150 Hz Gaussian filterbank (8 bands) → Hilbert envelope → sum → 200 Hz → z-score → sig-channel selection. In `coganlab/IEEG_Pipelines`. **Z-score is per-channel mean/std pooled across all pre-auditory baseline trials + samples** (500 ms window immediately before auditory-stim onset; verified 2026-04-18 via reconstruction test on 7 patients, corr = 1.0000). NOT per-trial and NOT pre-production. **Recording-level median/MAD ≡ this recipe up to per-channel affine (ρ=1.0000 across tested patients)** — SSL (Stage 2) can swap recipes without bit-exact equivalence constraints. Details: `docs/references/data_reference.md`.

## Conventions

- **Write simply.** Ordinary words. Short sentences. No throat-clearing, no redundant qualifiers, no ceremonial preambles. Cut is the main edit. Applies to docs, commits, PRs, memory files, chat. Paul Graham's "Write Simply" is the reference. Code comments stay minimal (default: none).
- **Discuss logic before writing code.** See Working Principle.
- **All training on DCC, never local.**
- **Every architectural change reports both pooled joint AND LOPO warm-start.** See `docs/objectives.md §Evaluation philosophy`. LOPO is the foundation-model test; load-bearing for Stage-2 SSL, Stage-3 cross-sensor transfer, Stage-4 external-corpus transfer. Single-protocol evidence does not justify defaulting an arch change.
- **Always aggregate results into `docs/experiments/v14_ablation_log.csv`.** Every finished DCC run lands in the CSV — `/work/ht203` auto-purges after 75 days and the CSV is the only long-term record. Use `scripts/ablation/collect.py <job_ids>` for submissions in `.ablation_submissions.jsonl`; for hand-written sbatches (LOPO stages), rsync results and run `.venv/bin/python scripts/v14_core/update_ablation_log.py --results-root /tmp/v14_full_mirror --csv docs/experiments/v14_ablation_log.csv`. If an ablation introduces a new hyperparameter dimension (readout, masking, PE, aug preset), extend `_variant_suffix()` in `update_ablation_log.py` *before* the first result lands, or the aggregator silently collides new runs and clobbers the baseline. For LOPO cells the standard aggregator can't express, append by hand with a distinctive `experiment_id` + `patient` suffix. Never defer — a forgotten aggregation is a forgotten experiment.
