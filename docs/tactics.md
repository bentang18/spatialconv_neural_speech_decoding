# Tactics — Concrete Task List

Tactics layer of the triad (objectives → strategy → tactics). Operational: what's running, what to do when it lands, what's blocked. Updated 2026-04-21 (Stage-2 kickoff after Box audit).

- Objectives: `objectives.md`
- Current-stage strategy: `strategy/stage_2.md`
- DCC tooling: `references/dcc_setup.md`

**Current stage:** Stage 2 opened 2026-04-21. Frozen architecture = Stage-1 T3.1 default (`per_cell + partialconv + pe2d + hierarchical_atlas` @ d=32, depth=3, pool=(4,8)). Stage-2 scope narrowed to **uECoG only** — Cogan sEEG D-cohort deferred to Stage 3 per 2026-04-21 alignment. Scoreboard is the **cohort-growth curve** (pooled joint + LOPO warm-start at 7-LH → 10-LH → 17-LH target).

---

## In flight (DCC queue)

- **Job 45899893** — 7-LH pooled under T3.1 default (`v14_pooled_7lh_t31_dcc.sh`, 15 tasks, submitted 2026-04-21). Fills Stage-2 scoreboard row cohort=7 (baseline). Save-ckpt on; copy to `/hpc/group/coganlab/ht203/stage2_ckpt_t31_7lh/` on completion.

Poll with `scripts/ablation/status.py`. Tail logs with `scripts/ablation/logs.py <job_id>`. Peek mid-flight JSONs with `scripts/ablation/peek.py <job_id>`.

---

## Stage-2 immediate priorities

Ordered by blockage: things that gate cohort expansion come first.

### Data unblock (ask / chase Zac)

- [ ] **Localization pipeline for 10 priority lex patients.** Box audit 2026-04-21: only S76/S78/S81 have recons on my mount; the spreadsheet's "Localization" column is empty across all lex rows. Zac said recons exist "back to S73" but they're not under `ECoG_Recon/` or `ECoG_Recon_Full/`. Priority pipeline order (best-HG first): **S73 (210/256), S75 (227/256), S56 (186/256), S67 (159/256), S74 (156/256), S41 (146/256), S53 (114/256), S47 (108/128), S45 (101/128), S55 (100/256)**.
- [ ] **Re-check spreadsheet / Box for localization tracking doc.** The "Localization" column in `uECoG_Upload/uECoG Recording Details.xlsx` is empty — the doc Zac referenced may live elsewhere. Scan `CoganLab/preprocessing_documentation/` and `CoganLab/uECoG_Meetings/`.
- [ ] **S52 + S71 usability (Zac checking).** S52 = MFA events in derivatives but no raw in BIDS; S71 = two incompatible events.tsvs, no merged `.fif`. Drop both from Stage 2 if unresolvable by next meeting.
- [ ] **S41 is not in Zac's pipeline back-reach.** Zac: pipeline reaches "back to S73" — S41 is from 2022-12-12 and won't be picked up. Ask explicitly whether S41 recon can be run as a one-off (best-HG patient after S73/S75).

### Architecture close-out (Stage-1 → Stage-2 handoff)

- [ ] **Re-run 7-LH pooled under T3.1 default** (`hierarchical_atlas + partialconv + pe2d`). This becomes the Stage-2 baseline row (cohort=7, reference for cohort-growth deltas). 15-task pooled sbatch; copy ckpts to `/hpc/group/coganlab/ht203/stage2_ckpt_t31_7lh/`.
- [ ] **7-LH LOPO under T3.1** after pooled lands. Fills the 7-LH row's LOPO column.

### Infrastructure (self-contained; unblocked)

- [ ] **28-ARPABET joint label map.** Extend `src/speech_decoding/data/phoneme_map.py`: add lex 28-class index, keep PS 9-class as a subset with a stable mapping. Add per-task mask for exhaustive AR decode (9³ for PS trials, 28³ for lex trials). Add class-frequency probe — PS-acc / lex-acc ratio in eval JSONs.
- [ ] **Lex phoneme-level loader.** Parallel to `phoneme_dataset.py` — reads `derivatives/epoch(phonemeLevel)(CAR)/...` from the lex BIDS root (`configs/paths.yaml:lex_bids_root`). 15/16 lex patients have phoneme-level `.fif` (S71 missing — needs merge from `events.tsv` split-role + MFA rerun).
- [ ] **Mixed-cohort sampler.** Same-patient-per-batch invariant holds; cohort set extends to PS ∪ lex. `V14PhonemeDataset` wraps both per-task dataset objects transparently — patient_id disambiguates.
- [ ] **Continuous-sample loader.** Raw `.fif` for SSL pretrain, not phoneme-epoched. Recipe A z-score (per-channel mean/std over all pre-auditory baselines). Target: `src/speech_decoding/v14/continuous_dataset.py`.
- [ ] **Per-lex-patient channel bridge.** As each lex recon lands, build `data/channel_maps/<pt>_channelMap.mat` lookup + `data/fsaverage_coords/<pt>_fsaverage_pial.csv` + `data/atlas/support_cache_v2c_snap/<pt>_support_tier1.csv`. Automate with a single `scripts/v14_core/prepare_new_patient.py`.

### SSL objective (Stage-2 mid-wave, triggered at ≥10-LH)

- [ ] **Choose SSL objective** (DIVER-1 multi-domain reconstruction preferred, see `strategy/stage_2.md §SSL ablation`). Candidate stack ordered by expected transfer strength.
- [ ] **Mask generator + reconstruction head**, checkpoint interop so SSL ckpt loads cleanly into the per-phoneme loader.
- [ ] **Calibration module stub.** `src/speech_decoding/v14/calibration.py` signature-only (no-op default). Preserves interface for Phase 2 without baking assumptions.

---

## Backlog (deferred to later stages)

- **Architectural re-tests at Stage-2 scale** — `P_emb` LOPO keep (T3.5 follow-up), per-electrode d=64 (T2.2 follow-up), plain hierarchical-alone LOPO. Queued until 17-LH pooled + LOPO lands.
- **T3.6 decomposed path** (old tasks #10/#11/#12) — dual-stream cross-attn + 611-token backbone. Subsumed into the Stage-2 atlas-mechanism ablation, where cross-attn on per-electrode tokens is strictly cleaner than per_cell dual-stream. Likely retired as originally specified.
- **Cogan sEEG D-cohort scoping** — Stage 3 kickoff. Full self-audit 2026-04-21 (`reports/seeg_cohort_scoping_2026_04_21/`): **85 unique D-patients** across 4 speech-task BIDS roots, **all with complete FreeSurfer recons on Box** (`ECoG_Recon/D<N>`, no zero-pad; BIDS `sub-D0023` → recon `D23`). RAS format identical to S-patients — `fsaverage_projection.py` transfers unchanged. PS events.tsv uses identical 52 CVC/VCV tokens — **same 9-phoneme label map**, no remap. BNA lookups pre-baked per D-recon at `D<N>_elec_location_radius_{1,2,3,5,7,10}mm_aparc.BN_atlas+aseg.mgz.csv` (patient-space probabilistic; convention-compatible with our Tier-1 naming). Projectable cohort × Tier-1 (3mm argmax ≥10 electrodes): PS 17 LH / 9 RH / 25 either; LexDelay 13 LH / 11 RH / 24 either; LexNoDelay 7 LH / 6 RH; SentenceRep 7 LH / 2 RH. PS ∩ LexDelay = 27. Derivatives are trial-level multi-band (`epoch(CAR)/sub-D*/epoch(band)(power)/`) — no phoneme-level fif; Stage 3 would MFA-epoch from production. Self-answerable prep complete 2026-04-24 (all 11 items landed — see `memory/project_seeg_stage3_prep_inflight_2026_04_24.md`): A1 sig-channel (`reports/seeg_sig_channels_2026_04_24/`), A2 events+muscle (`reports/seeg_events_audit_2026_04_24/`), A3 continuous corpus (`reports/seeg_corpus_audit_2026_04_24/` — **180.59 h / 87 unique D-patients, 26.6× uECoG**), A4 z-score recipe (`reports/seeg_zscore_recipe_2026_04_24/`), A5 DCC sync diff (`reports/dcc_sync_plan_2026_04_24/`), A6 sensor geometry (`reports/seeg_sensor_geometry_2026_04_24/`), B1 support caches (`data/atlas/support_cache_v2c_snap_dcohort/`, 122 patients), B2 coord caches (`data/dcohort_coords/`, 128 patients), B3 manifest (`data/dcohort_manifest.csv`, 122 ready + 6 partial), C1 RH-expansion stub (`docs/strategy/stage_3_rh_expansion.md`). Corpus section in `docs/references/data_reference.md`.
  - **Nanlin asks** (two; everything else is our call):
    1. **Laplacian / bipolar reference variant?** Box has stats directories for CAR / WM / M1 / STG / HIPP / LING — all global or anatomical-region references. BT cross-subject baseline uses a Laplacian re-reference (mean of adjacent same-stem depth contacts) and that's the winning recipe at 0.539. Ask whether the Cogan pipeline has a Laplacian/bipolar option we missed, or which of the six is closest to local-bipolar style. Default to CAR if no clear match.
    2. **MFA / TextGrid / production-WAV location for D-cohort.** `SCRIPTS_USAGE.md` at the BIDS root references `D_Data/Phoneme_Sequencing/` but that path isn't visible at `/datacommons/coganlab/D_Data/` on DCC. If they don't exist, we stay continuous-corpus / SSL-only on the D-cohort side — no Tier-2 blocker.
  - **Our calls** (don't bother him):
    - **DCC sync direction.** Box → `/work/ht203/data/` per `reports/dcc_sync_plan_2026_04_24/rsync_commands.sh`.
    - **Authoritative usability tiering.** A1 sig-fraction proxy is good enough for SSL pretrain (false positives nearly free). Defer his authoritative call until/unless we fine-tune on D-cohort.
    - **Z-score recipe exact form.** A4 confirmed `production_highgamma.fif` is directly z-scored and consistent within-patient. uECoG audit showed mean/std ≡ recording-level median/MAD up to per-channel affine (ρ=1.0000); recipe identity is academic for model input.
    - **Patient-space atlas convention.** Radius (3 mm), model-input form (full weighted Tier-1 support), normalization (raw [0, 100]), and Tier-1 selection rule (argmax_wins ≥ N pooled across cohort) are all our calls. Decide empirically by ablation if needed.
- **Phase-2 learned per-patient calibration** — enabled by `calibration.py` stub.
- **RH patient re-inclusion (S22, S58)** — Stage 3 with sEEG join.

---

## Reference

- DCC helpers: `scripts/ablation/{submit,status,logs,collect,query,dcc_sync_check,peek}.py`. Each has `--help`.
- DCC setup + rsync recipe: `references/dcc_setup.md`.
- Raw ablation log: `experiments/v14_ablation_log.csv` (authoritative results).
- Submissions ledger: `.ablation_submissions.jsonl` (gitignored; decodes task ids → fold/seed).
