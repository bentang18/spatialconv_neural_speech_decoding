# Current Direction

Updated: 2026-04-13 — v14 Phase 1 is the sole active direction. The v14 design doc, the per-blocker decision log, and the data reference are the authoritative sources; this doc is the narrative that ties them together.

Canonical references (do not restate them here):

- **Architecture contract and design intent**: `docs/neural_field_perceiver_v14.tex`
- **Working principle (discuss before code)**: `CLAUDE.md` → "Working Principle"
- **Open blockers, decisions, and critical path**: `docs/implementation_tasks.md`
- **Per-patient tables and data reference**: `docs/data_reference.md`

## Active priority: Neural Field Perceiver v14, Phase 1

A supervised correctness pass of `v14-core` on the existing intra-op `uECoG` data, using a **fixed** Brainnetome volumetric mapping. No learned per-patient calibration, no SSL, no `sEEG`, no external datasets. The scientific bet is that electrodes are patient-specific observations and atlas/subparcel tokens are the shared representation; Phase 1 just proves the fixed-atlas path is correct before any later claim about calibration or scaling.

Phase 1 is not the final architecture. It is the minimum implementation that has to be right before anything else gets to matter.

## Top blocker

**ACPC → MNI transform pipeline is not trusted.** Zac flagged the current handling as wrong. Re-verify with Zac's MATLAB transform path before any spatial model code lands. See `implementation_tasks.md` **#1**.

The full critical path — spatial, temporal, label-space, training, and audit branches — lives in the Status header at the top of `implementation_tasks.md`. It is the single source of truth for which blocker unblocks which, so it is not duplicated here.

## Phase 1 decisions already in place

Each of these was discussed, agreed, and recorded in `implementation_tasks.md`. Rationale lives there, not here, so updates flow one way. Pointers only:

- **#5** parcel support statistic — PM-weighted sum of non-artifact contacts
- **#7** how `token_support` enters the model — concat-to-token before inter-region attention
- **#10** parcel-frame construction — PM-weighted centroid origin + fsaverage-based cortical-normal axes, cached to `parcel_frames.npz`
- **#29** input window / epoching — response-onset-locked full-trial epoch, `tmin = -0.5 s`, `tmax = 1.0 s`, one sample = one trial = three phonemes
- **#30** Phase 1 right-hemisphere exclusion — S22 and S58 deferred to Phase 2 alongside the sEEG join
- **#32** input normalization at the v14 boundary — no normalization beyond upstream `productionZscore_highgamma`
- **#33** PER metric — slot-averaged PER, per-patient plus population mean, 3-seed `mean ± std`

If any of these change, update `implementation_tasks.md` first and let this doc stay silent.

## Phoneme / label-space rigor audit

A first-pass audit surfaced nine discussion items, now blockers **#17–#25** plus the operational audit **#34**. The most load-bearing item is **#17** — the inherited `PS2ARPA` mapping (`ae → EH`, `u → UH`) looks phonetically wrong; the correct mapping is probably `ae → AE` (/æ/) and `u → UW` (/u/). Until these resolve, **no phoneme-level metric is trustworthy and no v14 data loader is written.**

## First implementation scope

### In scope
- Intra-op `uECoG` only.
- Full-trial 3-phoneme decoding on the `#29` response-onset-locked window.
- Existing `productionZscore_highgamma` features. No re-preprocessing.
- Fixed Brainnetome volumetric PM membership from verified ACPC→MNI coordinates.
- Supervised training only.
- `v14-core` interface: fixed spatial mapping, shared temporal layer, within-parcel Perceiver summarizer, relational-temporal backbone, AR cross-attention decoder.
- Core patients first: **S14, S26, S33, S62** (all LH). Expand to the LH extended set `S16, S23, S39` only after the core path is correct.

### Out of scope
- Learned per-patient calibration (`Δ/ω`, `δ_l`, `τ_l`, additional gain/offset). Phase 2.
- `sEEG`, external chronic ECoG, SSL / JEPA pretraining, audio contrastive losses. Phase 1.5+.
- Functional pooling variants, broad ablation sweeps.
- Right-hemisphere patients (S22, S58). Per `#30`.
- Excluded from Phase 1 entirely: S32 (no HG response), S57 (hybrid strip, 52/256 sig).

## First goal: implementation correctness, not final performance

Phase 1 is correct if:
- data reach the right parcels,
- the token interface is assembled correctly,
- missing and weakly supported parcels are handled sensibly,
- a tiny subset overfits cleanly end-to-end,
- the channel-to-electrode mapping is driven by the authoritative channel-map files, not the older TSV grid heuristic.

Final performance against the `0.734 ± 0.007` S14 baseline is the *next* conversation, after correctness lands.

## Implementation order

Do not begin any of these until the blockers they depend on are discussed and frozen per `implementation_tasks.md`.

1. **Data and atlas interface**
   - Load `uECoG` trials via a fresh v14 loader (not the quarantined `load_patient_data`).
   - Define the explicit active-channel set rather than zeroing artifact channels in place.
   - Build corrected electrode coordinates on top of the verified ACPC→MNI path (`#1`).
   - Replace the TSV-derived grid heuristic with the authoritative channel-map bridge (`#12`).
   - Compute volumetric Brainnetome PM membership from `BNA_PM_4D.nii.gz`.
   - Compute parcel support per `#5` and apply the unsupported-vs-weak rule from `#3`.

2. **Shared temporal layer**
   - Per-electrode temporal patches.
   - First-pass shared temporal encoder per `#2`.
   - Output contract per `#6`: `(N_elec × d × T_tok)` with frozen token rate and receptive field.

3. **Within-parcel summarizer**
   - Canonical parcel-frame coordinates from `parcel_frames.npz` (built per `#10`).
   - Local point encoder plus shared latent queries per the `#26` contract.
   - Emit the fixed `N_tok` token tensor (count pending re-derivation per `#4`), `token_mask`, and `token_support`.

4. **Relational-temporal backbone**
   - Inter-region attention with Brainnetome SC/FC bias init per `#8` and the `#27` contract.
   - Temporal attention per `#27`.
   - Hard masking for unsupported parcels, support-aware handling for weakly observed ones.

5. **Decoder**
   - Three queries per `#28`.
   - Additive AR conditioning.
   - Greedy decoding at eval; beam search only if `#28` agrees it's needed.

6. **Minimal training loop integration**
   - Grouped-by-token CV compatible, per `grouped_cv.py`.
   - Patient batching per `#31`.
   - Supervised loss only per `#9`.
   - Tiny-subset overfit test before any real run.

## Correctness checks before real training

These matter more than early leaderboard numbers.

- **Coordinate sanity**: left/right routing correct; array geometry consistent with `docs/data_reference.md` layout notes; authoritative channel-map bridge verified end-to-end against the mapping files, not TSV heuristics.
- **Atlas sanity**: expected speech parcels receive support on the core patients; support maps look plausible; no spatial mismatch between ACPC and MNI renders of the same array.
- **Token sanity**: default token count matches `N_tok` from the re-derivation (`#4`); split parcels emit the right number of sub-tokens; unsupported parcels are masked rather than hallucinated; weakly supported parcels remain active but carry low support.
- **Model sanity**: forward pass works on heterogeneous channel counts (128/256); gradients reach temporal layer, summarizer, backbone, and decoder; tiny subset overfits; model behaves sensibly when only one parcel/token is active; zero-filled inactive tokens cannot leak through as fake observations.

## Recommended first training order

Do not start with every optional degree of freedom active.

1. Run with fixed atlas membership only, fixed `Δ/ω` = 0, no gain/impedance normalization, supervised loss only, hard mask on unsupported parcels.
2. Verify end-to-end gradients and tiny-subset overfit.
3. Lock the shared temporal layer shape and token rate via `#2` / `#6`.
4. Only then compare against simpler local-pooling ablations (mean+gradient, uniform `k=1`).
5. Learned calibration enters only after the fixed-atlas baseline is stable.

This keeps implementation bugs separate from coordinate bugs, and stops overfitting caused by too much early flexibility.

## Success criteria for Phase 1

Phase 1 is successful if all of the following are true:

- `uECoG` core-patient data flow is stable and reproducible.
- Atlas/subparcel token construction is verified against the checks above.
- Model passes shape, mask, and support sanity tests under `tests/v14/`.
- A tiny subset overfits cleanly.
- The implementation is clean enough to support the next round of ablations without rewriting the interface.

Only after that does the project expand to: learned per-patient calibration, SSL on the full continuous `uECoG` corpus, extended `uECoG` coverage, `sEEG`, and external datasets — in that order.

## Immediate next steps

1. ~~v14 design doc rewrite~~ — done 2026-04-10.
2. ~~Presentation doc rewrite~~ — done 2026-04-10.
3. **Fix and verify ACPC → MNI** (`#1`, top blocker). Re-verify against Zac's MATLAB transform path before any serious spatial code.
4. **Lock the channel-map bridge** (`#12`). Write the 1-to-1 verifier and run it on every Phase 1 patient.
5. **Re-derive the parcel set** (`#4`) under the Phase 1 contract once `#1` and `#12` are green. Output: `reports/phase1_parcel_coverage_2026_04_13.md`.
6. **Execute the phoneme audit** (`#34`). Can run in parallel with the spatial chain. Output: `reports/phoneme_audit_2026_04_13.md`.
7. **Freeze the temporal front-end** (`#2`, `#6`). This is the next architectural decision after coordinates.
8. **Build `parcel_frames.npz`** per `#10` once `#1` is verified. Runs the full verification checklist before the cache is committed.
9. **`v14-core` implementation** under `src/speech_decoding/v14/` — only after the spatial, temporal, label, and training blockers are frozen.
10. **Phase 1.5: full-corpus uECoG SSL** once supervised `v14-core` is stable. Not response-locked-only SSL.
11. **External chronic ECoG acquisition** (Flinker, Chang) — high leverage, not on the critical path for the first local implementation.
12. **HGA extraction pipeline** for the 456-min raw EDF corpus — needed before SSL, not before supervised Phase 1.

## Patient scope (Phase 1)

- **Core**: `S14`, `S26`, `S33`, `S62` (all LH).
- **Extended (LH only, per #30)**: `S16`, `S23`, `S39`.
- **Deferred to Phase 2 with the sEEG join**: `S22`, `S58` (RH).
- **Excluded from Phase 1 entirely**: `S32` (no HG response), `S57` (hybrid strip, 52/256 sig, Map 8 wiring unresolved).

## Per-patient baseline (v14 must beat this)

**PER 0.734 ± 0.007** on S14, grouped-by-token CV, 3-seed, per-phoneme MFA flat head + full recipe. Population mean **0.825** across 11 patients. Both numbers are the historical Conv2d baseline; the comparison is apples-to-apples only after v14 runs under the same grouped-by-token CV and slot-averaged PER contract (`#33`).

## Practical rules

- Active design spec = `neural_field_perceiver_v14.tex`. Every other doc is downstream.
- Within-parcel Perceiver summarizer is the default spatial mechanism. Mean+gradient is the main linear ablation. Uniform `k=1` is the second ablation. Cross-attention variants are named and deferred.
- Always `exclude_artifacts=True`. Always grouped-by-token CV.
- All training on DCC. Never local. See `docs/dcc_setup.md`.
- Supervised training contract is blocked until `#9` is frozen — do not start writing the training loop against a moving target.
- If a doc references v12 / centroid VE / distance bias / Fourier PE / "multi-view reconstruction" framing / Q/K/V for electrode→VE mapping / spherical node decay / v12 A3 ablations — it is stale. Most of these now live under `docs/archive/`.
