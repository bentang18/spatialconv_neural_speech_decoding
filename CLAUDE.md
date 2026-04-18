# Cross-Patient Speech Decoding from Intra-Op uECOG

## Project

Ben Tang, Greg Cogan Lab, Duke. Collaborating with Zac Spalding.
Extending Spalding 2025 (PCA+CCA, SVM/Seq2Seq, 8 patients, 9 phonemes, 0.31 bal. acc.).

**Task**: Non-word repetition (52 CVC/VCV tokens, 3 phonemes each, e.g. /abe/; 9 phonemes). Intra-operative, left sensorimotor cortex, 128/256-ch uECOG arrays. ~1 min utterance/patient. Stimulus-to-response delay: 1.1 ± 0.3s (Duraivel 2023); stimulus duration ~500ms; utterance ~450ms. Auditory stimulus ends ~600ms before response onset (t=0).

**Patients**: 11 unique PS patients (S18 excluded — no preprocessed; S36 excluded — duplicate of S32): S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62. 8 are Spalding's published set. 46–178 trials/pt, 63–201 sig channels. Core set (Phase 1, all LH): S14, S26, S33, S62. Extended (Phase 1, LH only, per `implementation_tasks.md` #30): S16, S23, S39. Deferred to Phase 2 with the sEEG join (RH, per #30): S22, S58. Excluded from Phase 1 entirely: S32 (no HG response), S57 (52/256 sig, hybrid strip).

## Current Direction: Neural Field Perceiver (v14) — Intracranial Foundation Model

**Design doc**: `docs/neural_field_perceiver_v14.tex`. **Per-patient tables and data reference**: `docs/data_reference.md`. **Live status**: `docs/current_direction.md`. **Open work**: `docs/implementation_tasks.md`. **Results log**: `docs/experiments/v14_ablation_log.csv`.

**Two-problem decomposition** (atlas calibration + shared dynamics):
- **Problem 1 — Calibration** (per-patient, physics-constrained): raw electrodes → atlas-grounded regional tokens via Brainnetome surface parcellation on fsaverage (`#36`). Atlas does ~90% of calibration; supervised gradient refines ~10% (Phase 2+).
- **Problem 2 — Dynamics** (shared, unconstrained ML): regional tokens → phoneme sequence via a small relational-temporal transformer + AR decoder. Same representation for every patient.

**Phase 1 architecture as implemented** (per-phoneme path, plan `docs/plans/v14-core-current.md`):
```
signal (B, N_e, 130) at 200 Hz, phoneme-centered window [-0.15, 0.5)s
→ grid-scatter (B, 1, H_p, W_p, 130)
→ Conv2d(1→8, k=3, pad=1) per time-step + GELU
→ masked-mean pool to (4, 8) = 32 cells        ← grid-level spatial compression
→ per-cell Conv1d(8→32, k=30, stride=10)       ← shared temporal front-end (baseline-exact)
→ + pooled_support @ P_emb[15, d=32]           ← cross-patient atlas anchor
→ flatten to (B, 352 tokens, d=32)
→ Backbone: 3 × [combined attention + FFN], 2 heads × 16, FFN 128, RoPE on temporal axis only, dropout 0.1
→ D1 decoder: mean-pool memory + prev_phoneme embedding + Linear(d, 9)
→ per-phoneme logits; train with flat CE + teacher forcing; eval exhaustive 9^3 AR per `#9`
```
47k parameters at the baseline. Samples are **phoneme-level** from `epoch(phonemeLevel)(CAR)`; 3 phonemes/trial, grouped-by-token CV.

The Phase-2+ target is the `B-1` per-electrode-token path without the `(4, 8)` pool — `signal (B, N_e, d, T)` flowing into the combined attention backbone unchanged. Shared interface: per-electrode tokens. Cross-patient sharing via the soft parcel embedding, not per-parcel pooling. `N_tok = 15` LH Brainnetome parcels (rule `argmax_wins ≥ 10`) is the **embedding-lookup table size**, not a token count; canonical list in `src/speech_decoding/v14/token_spec.py`. No within-parcel Perceiver summarizer, no `parcel_frames.npz`, no intra-parcel positional encoding (registration noise dominates anatomical variability — sub-parcel cross-patient alignment is below the SNR floor). No Fourier PE. No global electrode-to-region cross-attention. SC/FC additive logit bias deferred to Phase F ablation A4.

**Current Phase-1 contract** (all 36 blockers closed 2026-04-16 late; per-phoneme path implemented + shipped on DCC):
- **Spatial base is fsaverage (`#36` closed).** Patient side is `src/speech_decoding/v14/fsaverage_projection.py` — strict snap-to-pial via stock `sphere.reg`, one vertex per electrode, caches at `data/fsaverage_coords/<pt>_fsaverage_pial.csv`. Atlas side is `data/atlas/fsaverage_bake_v2c/` — `mri_vol2surf --projfrac-avg 0 1 0.1` from `ICBM152_fs` then `mri_surf2surf` to fsaverage, no smoothing. Physical PSF is baked into the atlas; no query-time Gaussian. The cvs_avg35 / `BNA_PM_dilated_8mm` path is a parity oracle only; not Phase-1 active.
- **Atlas source.** Ground truth for the atlas-side bake is `data/atlas/BNA_PM_4D.nii.gz`. `~/nilearn_data/bnatlas.nii.gz` is for ROI indexing and sanity checks only.
- **Fixed-atlas, no learned calibration.** No learned `Δ/ω`, `δ_l`, or `τ_l`.
- **No extra gain / impedance correction.** Inputs come from existing `productionZscore_highgamma` features. No additional normalization.
- **Soft parcel embedding (`#26` deprecated, `#3` relaxed).** Stage 5 is `pooled_support @ P_emb` with `P_emb: (15, d)` Xavier-init learnable lookup; raw support (not normalized). No within-parcel Perceiver summarizer. Electrodes with `max over Tier-1 == 0` are still emitted; their embedding contribution is zero. Only hard exclusion is `#11` (artifact channels).
- **Temporal front-end.** Per-cell `Conv1d(8→32, k=30, stride=10)` matches Ben's 0.734 baseline (150 ms kernel, 50 ms hop, 200 Hz → 20 Hz token rate). The original `#2`/`#6` per-electrode Conv1d at `kernel=28` is the Phase-2+ target for the full per-electrode-token path.
- **No per-parcel tokens, no token_mask, no token_support in the loader.** Per-electrode `electrode_active_mask[N_e]` and `support[N_e, 15]` replace them.
- **Backbone is combined spatiotemporal attention (`#27` revised).** `B = 3` blocks, combined attention over `(cell × time)` tokens, FFN, pre-norm, residual, dropout 0.1. `num_heads = 2`, `head_dim = 16` at `d=32`. FFN width `4d = 128`. RoPE on temporal axis only. `token_active_mask` from the pool-cell active mask broadcast across T; applied on both key and query axes. **No SC/FC bias in baseline** (deferred to Phase F ablation A4 per `#8`).
- **Width budget.** Current baseline `d_model = 32` (chosen for the 47k parameter target matching Ben's baseline). First width ablation `d_model = 64`; running now.
- **Phase 1 is supervised-only on `uECoG`.** Flat per-phoneme CE; teacher forcing in train; exhaustive `9^3 = 729` decode at eval per `#9`.
- **Next step after Phase 1** is full-corpus `uECoG` SSL (Phase 1.5) — not response-locked-only SSL.

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

This applies to every logic step: channel indices, channel-to-electrode bookkeeping, coordinate-frame verification, Brainnetome PM lookup, per-electrode support, grid scatter, soft parcel embedding, temporal front-end, combined attention backbone, AR decoder, loss, eval split, metrics. No step is too obvious to skip.

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

### 5. Verify before claiming done
Never claim tests pass, code works, or a bug is fixed without running the verification command in the current turn. "Should work," "probably fixed," "seems to pass" are lies, not confidence. If you haven't seen the exit code / test output / UI since your last change, run it before reporting. No "Great!" / "Perfect!" / "Done!" before evidence.

### 6. Debug root cause, not symptoms
On any bug or test failure: read the full error, reproduce it, check recent changes, and for multi-component systems log data at every boundary before proposing any fix. "Just one quick fix" masks the real issue and multiplies bugs.

### 7. No performative agreement on pushback
When the user questions a design or pushes back, do not capitulate with "You're absolutely right!" and start rewriting. Restate the technical point in your own words, verify against the codebase, and push back with reasoning if the feedback is wrong for this context. Technical correctness over social comfort.

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
- Target template for Phase 1 (per `#36`): FreeSurfer `fsaverage` under the local FreeSurfer `SUBJECTS_DIR` (typically `/Applications/freesurfer/<version>/subjects/fsaverage/`). Patient projection uses stock `lh.sphere.reg`.
- Migration-oracle average subject: `.../ECoG_Recon/cvs_avg35_inMNI152/`. Used for the cvs_avg35 path in `src/speech_decoding/v14/coordinates.py` as the parity oracle; not the Phase-1 active representation.
- **Patient folders are `S<num>`**, matching the internal patient IDs: `S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62`. All 11 PS patients are present. Each has `surf/{lh,rh}.{pial,pial-outer-smoothed,sphere.reg,sphere-outer.reg,sphere-outer-mni.reg}` and `elec_recon/<pt>.{LEPTO,LEPTOVOX,POSTIMPLANT,electrodeNames}` plus `<pt>_elec_locations_RAS{,_brainshifted}.txt`.
- **Do not confuse `S<num>` with `D<num>`**: `ECoG_Recon/` also contains unrelated `D<num>` folders from a different Duke cohort (`D14` is 120 electrodes, not the same person as `S14`). Always use `S<num>`.
- **Avoid the `_old`, `_no_tkr`, `_kumar`, `_diag`, `_med` sibling folders** (e.g. `S14_old`, `S14_no_tkr`). These are alternative reconstructions kept for history and are not the current trusted version. Read from the plain `S<num>/` folder.

### Active docs
- `docs/current_direction.md` — where we are now (post-P8, ablations underway, P9 pending)
- `docs/implementation_tasks.md` — live status summary + active work items
- `docs/plans/v14-core-current.md` — implementation plan with verification checks (P1–P9)
- `docs/data_reference.md` — per-patient tables: array layouts, sig/artifact channel counts, Brainnetome parcel list, raw-corpus sizes
- `docs/experiments/v14_ablation_log.csv` — live experiment results log (updated by `scripts/v14_core/update_ablation_log.py`)
- `docs/dcc_setup.md` — DCC cluster setup
- `docs/neural_field_perceiver_v14.tex` — v14 design document (historical, pre-B-1-amendment; being rewritten)
- `docs/questions.md` — unresolved architectural questions (thinking doc)
- `docs/qc/phoneme_level_fif_audit_cohort.md` — #34 cohort audit results
- `docs/qc/coord_bridge_verification.md` — #12 bridge verification
- `docs/qc/support_cache_v2c_snap_qc_report.md` — current support-cache QC

### Archived but useful
- `docs/archive/sessions/` — 2026-04-16/17 session logs (cras fix, kernel ablation, #36 handoff, B-1 amendment, slack draft)
- `docs/archive/plans/` — superseded plan drafts (`v14-core.md`, per-phoneme draft, open-notes)
- `docs/archive/qc_old_caches/` — pre-v2c support-cache QC reports
- `docs/archive/implementation_tasks_archived.md` — pre-closure blocker log (Apr 14)
- `docs/archive/implementation_tasks_2026-04-16_post-closure.md` — all-closed blocker log (2026-04-16 late)
- `docs/archive/experiment_log.md` — 101-finding experiment history
- `docs/archive/research_synthesis.md` — 19-paper literature synthesis
- `docs/archive/reading_list.md`, `docs/archive/literature_findings.md` — historical literature and data-scaling context

### Configs & scripts
- `configs/paths.yaml` — machine-specific BIDS paths (gitignored). On DCC: `ps_bids_root=/work/ht203/data/BIDS`, `support_cache_dir=/work/ht203/data/atlas/support_cache_v2c_snap`, `channel_maps_dir=/work/ht203/data/channel_maps`.
- `scripts/v14_core/` — active v14 CLI + sbatch wrappers (see Code Structure above).
- Pre-v14 YAMLs live in `configs/archive/`; pre-v14 CLIs live in `scripts/archive/legacy/`.

## Code Structure (Phase 1 implementation live, 2026-04-17)

```
src/speech_decoding/
├── v14/                    # Phase-1 implementation home
│   ├── __init__.py         # re-exports config dataclasses
│   ├── config.py           # V14Config + per-phoneme configs (PoolConfig, PerCellTemporalConfig, D1DecoderConfig, PerPhonemeConfig). Also holds the B-1 target configs (GridMixerConfig, SoftParcelEmbeddingConfig, BackboneConfig) — frozen per #15 + B-1 amendment.
│   ├── token_spec.py       # DEFAULT_BASE_PARCELS (15) — embedding-lookup keys for soft parcel embedding — frozen per #4
│   ├── pool.py             # masked-mean pool primitive (grid → cells) with divisibility assertion
│   ├── phoneme_dataset.py  # ACTIVE loader — V14PhonemeDataset (phoneme-level .fif, grouped-by-token CV)
│   ├── phoneme_model.py    # ACTIVE model — NeuralFieldPerceiverPerPhoneme
│   ├── phoneme_decoder.py  # D1 minimum AR decoder (mean-pool + prev_emb + Linear)
│   ├── phoneme_run_fold.py # per-phoneme fold runner (P1–P6)
│   ├── backbone.py         # combined spatiotemporal attention (revised #27)
│   ├── train.py            # train_one_fold + per-phoneme loss wrapper (tail-flush grad-accum fix)
│   ├── eval.py             # evaluate_per_phoneme + exhaustive 9³ AR decode
│   ├── cv.py               # make_outer_folds + make_val_split
│   ├── dataset.py          # (kept) V14TrialDataset — trial-level slot-CE path (deprecated B-1 B-1-full)
│   ├── run_fold.py         # (kept) slot-CE fold runner — deprecated B-1-full path
│   ├── fsaverage_projection.py # strict fsaverage snap-to-pial — ACTIVE (#36)
│   ├── fsaverage_atlas.py  # baked fsaverage atlas loader, support/argmax/argmax_wins helpers
│   ├── support_cache.py    # per-electrode Tier-1 cache I/O
│   ├── channel_map.py      # #12 amp-to-physical bridge per patient
│   ├── coordinates.py      # cvs_avg35 projection — PARITY ORACLE only
│   ├── cvsavg_projection.py # outer-envelope → cvs_avg35 pial snap — parity oracle alternative
│   ├── electrode_pool.py   # exploration tooling (kernel-ablation, not on canonical path)
│   ├── parcel_frames.py    # archived per B-1 amendment (kept for parity reference)
│   ├── local_summarizer.py # deprecated (was within-parcel Perceiver summarizer #26)
│   ├── grid_mixer.py       # B-1 target Stage 2 (not on per-phoneme path)
│   ├── parcel_embedding.py # B-1 target Stage 3 (not on per-phoneme path — emb lives in phoneme_model)
│   ├── aggregate.py        # fold result aggregation helper
│   ├── calibration.py      # atlas-resource path helpers (Phase 2+)
│   ├── decoder.py          # (stub) 3-query AR decoder for B-1-full path — not on per-phoneme path
│   ├── model.py            # (stub) B-1-full top-level assembly
│   ├── tokenizer.py        # (stub) B-1-full per-electrode temporal Conv1d
│   └── audit/              # #34 phoneme-loading audit (closed 2026-04-16)
├── data/phoneme_map.py     # 9 PS phonemes, PS→ARPA (alphabetical), normalize_label
├── evaluation/grouped_cv.py # grouped-by-token CV splitter (kept, contract-neutral)
└── archive/legacy/         # QUARANTINE — see Working Principle #6

scripts/v14_core/
├── train_v14_core.py                      # single-job CLI (--mode per-phoneme | slot)
├── update_ablation_log.py                 # aggregate result JSONs into the experiment CSV
├── v14_per_phoneme_smoke_dcc.sh           # P7 smoke (1 job)
├── v14_per_phoneme_s14_dcc.sh             # P8 S14 full (30 jobs)
├── v14_per_phoneme_s14_ablation_dcc.sh    # capacity ablation (45 jobs: d×depth)
├── v14_per_phoneme_s14_spatial_ablation_dcc.sh # spatial ablation (60 jobs: k, pool)
├── v14_per_phoneme_cohort_dcc.sh          # P9 cohort (180 jobs)
└── v14_core_*.sh                          # deprecated B-1-full slot-CE sbatch wrappers

docs/experiments/v14_ablation_log.csv       # per-cell aggregated results, updated by update_ablation_log.py
```

## Data

### Loader contracts

**Active per-phoneme loader** (`src/speech_decoding/v14/phoneme_dataset.py`) — reads phoneme-level `.fif` from `derivatives/epoch(phonemeLevel)(CAR)/...`, crops to `[-0.15, 0.5)` s (130 samples at 200 Hz), emits one sample per phoneme (3 per trial):

```
signal[N_e, 130]                   # float32, z-scored HGA, 200 Hz, 0.65 s (phoneme-centered)
patient_id                         # str
label                              # long — alphabetical ARPABET index (#16)
prev_phoneme                       # long — previous phoneme index in trial; -1 for slot 0 (BOS)
trial_id                           # long — to group-back phonemes into trials for exhaustive 9^3 AR
phoneme_pos                        # long ∈ {0, 1, 2}
electrode_grid_layout[N_e, 2]      # int (row, col) on patient device grid
electrode_grid_shape               # tuple (H_p, W_p) per-patient bounding rect
electrode_active_mask[N_e]         # bool: non-artifact AND not pad
support[N_e, 15]                   # float32, raw BNA probability over Tier-1 (#5)
```

**B-1-full trial-level loader** (`dataset.py`) — trial-level `.fif` from `derivatives/epoch(CAR)/...`, `[-0.5, 1.0)` s, 300 samples, slot-CE over 3 phoneme positions. Kept but deprecated for the per-phoneme path; will be revisited for Phase-2+ per-electrode-token experiments.

Channel inclusion is all non-artifact channels (`#11`); sig-channel masks are ablation-only. `#34` closed 2026-04-16. `support` comes from the per-electrode Tier-1 cache `data/atlas/support_cache_v2c_snap/<pt>_support_tier1.csv` (built once per patient; see `docs/plans/v14-core-current.md`). Per-patient sig/artifact channel counts and array layouts live in `docs/data_reference.md`.

### Electrode coordinates

ACPC source: `Box/ECoG_Recon/<subj>/elec_recon/<subj>_elec_locations_RAS_brainshifted.txt`. Format: `prefix electrode_num x y z hemisphere type`.

**Active spatial pipeline (`#36`, closed 2026-04-16 late):** Patient side is `src/speech_decoding/v14/fsaverage_projection.py` — strict snap-to-pial: patient `lh.pial` → patient `lh.sphere.reg` → fsaverage `lh.sphere.reg` → fsaverage `lh.pial`, one vertex per electrode. Output caches at `data/fsaverage_coords/<pt>_fsaverage_pial.csv`. Atlas side is `data/atlas/fsaverage_bake_v2c/` — full 246-frame bake via `mri_vol2surf --projfrac-avg 0 1 0.1` from `ICBM152_fs`, then `mri_surf2surf` to fsaverage, no additional smoothing.

**Parity oracle (`#1`):** Python port of Zac's `sub2AvgBrainClinical.m` in `src/speech_decoding/v14/coordinates.py` v2 (cras-corrected 2026-04-16), projecting patient `pial-outer-smoothed` → patient `sphere-outer-mni.reg` → `cvs_avg35` `sphere-outer.reg` → `cvs_avg35` `pial-outer-smoothed`, then adding avg cras to reach true MNI. Output cached at `data/mni_coords/<pt>_MNI152.csv`. Retained as parity oracle only via `scripts/compare_fsaverage_spatial_parity.py`; not the Phase-1 active representation. Pre-cras-fix caches are archived at `data/mni_coords/archive_pre_cras_fix/`.

**Amp → physical → coordinate bridge (`#12`):** 128-strip patients (`S14 S16 S22 S23 S26`) use Map 4 from local `*_channelMap.mat` with `phys_idx = r*16 + c + 1`. 256-grid patients (`S33 S39 S62`) use Map 3 from `*_channelMapAll.mat`. `S58` resolves its 12×24 zero-indexed crop onto full Map 3. The lookup key at the coordinate cache is the concatenated electrode name from `<pt>.electrodeNames`, not a row index. `S39_channelMap.mat` is non-authoritative; never load it.

**Hemisphere (`#30`):** Phase 1 is left-hemisphere only. S22 and S58 (right-hemisphere) are deferred to Phase 2 alongside the sEEG join. The old `mirror_to_left()` helper is discarded; Brainnetome has distinct L/R parcels.

**DCC TSV vs RAS**: DCC electrode TSVs have normalized 0–1 grid coordinates (synthetic, for older Conv2d baselines). RAS files are the relevant source for v14.

### .fif paths and labels

**Active per-phoneme path**: `{bids_root}/derivatives/epoch(phonemeLevel)(CAR)/sub-{id}/epoch(band)(power)/sub-{id}_task-PhonemeSequence_desc-productionZscore_highgamma.fif`

Event-id mapping asserted under `#18`: `{'a':1, 'ae':2, 'b':3, 'g':4, 'i':5, 'k':6, 'p':7, 'u':8, 'v':9}`. Labels derived directly from event codes; alphabetical ARPA index per `#16` (`AA AE B G IY K P UW V` → `0..8`). `.fif` is authoritative per `#34` closure (events TSV is a soft cross-check only).

**Trial-level path** (B-1-full, deprecated for per-phoneme): `{bids_root}/derivatives/epoch(CAR)/...`; token from the `value` column of `events.tsv`; canonical 52-token inventory at `data/ps_tokens.csv`.

> **Phoneme / label-space rigor audit.** The first-pass audit surfaced blockers `#17–#25` plus the operational audit `#34`. All closed.

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
- **Every architectural change reports both pooled joint AND LOPO warm-start** (see "Canonical experimental protocol" in `docs/current_direction.md`). LOPO warm-start is the foundation-model test — load-bearing for Phase 1.5 SSL, Phase 2+ cross-sensor transfer, and external-corpus transfer. Single-protocol evidence does not justify defaulting an arch change.
