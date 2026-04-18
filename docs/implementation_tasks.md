# v14 Implementation Status

Updated 2026-04-17. Source of truth for blocker state and active work items.

Full historical blocker decision logs are preserved in `docs/archive/`:
- `implementation_tasks_archived.md` — pre-closure version (Apr 14, `#1`–`#36` open/in-flight).
- `implementation_tasks_2026-04-16_post-closure.md` — all-closed version (2026-04-16 late, per-blocker amendments).

This file is the live status summary only.

## Status

- **Phase 1 blockers**: 36/36 closed. Frozen.
- **v14-core per-phoneme implementation (plan P1–P6)**: complete. Commit `875ccb8`. 269 v14 tests green.
- **DCC runs**:
  - P7 smoke (45707136) — passed, pipeline green on GPU.
  - P8 full S14 (45707149) — complete. `d=32 depth=3` mean PER 0.783 (target 0.78 hit). Best single run 0.644.
  - Capacity ablation (45707426, 45 jobs) — running, trending toward null.
  - Spatial ablation (45707427, 60 jobs) — queued.
- **P9 cohort extension**: pending.

## Active work

- [ ] **P9** — extend per-phoneme run to 6 remaining Phase-1 LH patients (`S26, S33, S62, S16, S23, S39`). 180 jobs. Launch via `scripts/v14_core/v14_per_phoneme_cohort_dcc.sh`. Gate: capacity ablation shows no signal-flipping finding that'd change the default.
- [ ] **Ablation consolidation** — aggregate capacity + spatial results once complete; update `docs/experiments/v14_ablation_log.csv` via `scripts/v14_core/update_ablation_log.py`.
- [ ] **Optional follow-up ablations** (discussed, not launched): temporal `Conv1d(k, stride)`, attention head count, no-RoPE.

## Frozen Phase 1 contract (authority for live state; not rationale)

Full rationale per blocker lives in the archived log. Load-bearing pointers only:

- **Spatial base**: fsaverage, strict snap-to-pial (`#36`). Patient side: `src/speech_decoding/v14/fsaverage_projection.py`. Atlas side: `data/atlas/fsaverage_bake_v2c/` via `mri_vol2surf --projfrac-avg 0 1 0.1` + `mri_surf2surf`, no smoothing. Support read from `data/atlas/support_cache_v2c_snap/<pt>_support_tier1.csv`.
- **Tier-1 parcel set**: 15 LH Brainnetome parcels (`argmax_wins ≥ 10`). Canonical list in `src/speech_decoding/v14/token_spec.py`. Used as **embedding-lookup keys** for `P_emb: (15, d)`, not token slots.
- **Loader contract** (`#13` B-1 amended): trial-level `.fif` is authoritative. Emitted fields: `signal[N_e, T], electrode_grid_layout[N_e, 2], electrode_grid_shape, electrode_active_mask[N_e], support[N_e, 15], label, patient_id, prev_tokens, phoneme_pos, trial_id`.
- **Per-phoneme window** (current training path): `tmin=-0.15, tmax≈0.5`, 130 samples at 200 Hz (0.65 s). Baseline-MFA-aligned; `#29`'s trial-level `[-0.5, 1.0)` window is the deprecated slot-CE path.
- **Hemisphere** (`#30`): LH only in Phase 1. S22 + S58 deferred.
- **Label alphabet** (`#16`/`#17`): 9 ARPA phonemes, alphabetical `AA AE B G IY K P UW V`. PS→ARPA maps `ae→AE`, `u→UW`.
- **Eval metric** (`#33`): slot-averaged PER + per-phoneme PER, per-patient + population mean, 3-seed `mean ± std`.
- **Exhaustive decode** (`#9`): 9³ = 729 at eval; teacher-forced in train.
- **CV** (`#31`): grouped-by-token CV, same-patient-per-batch invariant.
- **Artifact channels** (`#11`): hard-exclude only. Validity otherwise soft via `support` routing (`#3` relaxed).
- **Normalization** (`#32`): no beyond upstream `productionZscore_highgamma`.
- **Channel-map bridge** (`#12`): Map 4 for 128-strip, Map 3 for 256-grid, S58 crop resolves onto Map 3. `S39_channelMap.mat` is non-authoritative.

## Deferred (Phase 2+)

- Phase 1.5 SSL on full continuous `uECoG` corpus
- Learned per-patient calibration (`Δ/ω`, `δ_l`, `τ_l`)
- `sEEG`
- External chronic ECoG (Flinker, Chang)
- RH patients (S22, S58)
- SC/FC additive logit bias (Phase F ablation A4)
- Broad ablation sweeps beyond the current capacity + spatial set

## Deferred calibration path (for external datasets)

When an external dataset's native MNI projection is plausible but not trusted enough to use raw:

- Freeze shared neural network weights first.
- Fit a per-patient rigid-body transform (rotation + translation only, no free affine) on electrode coordinates.
- Treat it as a lightweight calibration adapter, not a change to the shared model.
- Evaluate as an explicit external-dataset adaptation step, not part of the base Phase 1 contract.

## Working Rule

- Discuss logic first.
- Freeze the contract.
- Then write code.
- When two choices are equally good for Phase 1, prefer the one that scales better to cross-task use and external datasets.
- Do not reuse legacy code directly.
