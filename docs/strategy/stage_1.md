# Strategy — Stage 1 (Phase 1)

Strategy layer of the triad, scoped to **Stage 1: single-sensor supervised correctness pass on uECoG**.

- Objectives: `../objectives.md`
- Tactics: `../tactics.md`
- Stage index: `../strategy.md`

## Stage 1 recap (from objectives)

**Data:** 4-7 core LH PS uECoG, ~1 min/patient supervised. ~7 min total.

**Hypotheses:**
- **H1.1 (primary):** parcel-token pipeline decodes 9-phoneme uECoG at or above the 0.734 S14 baseline (0.825 population mean across 11 patients).
- **H1.2:** cross-patient pooled-joint training matches or beats solo per-patient training.

---

## Default architecture

Stage-1 default (confirmed by the 2026-04-19 wave — see scoreboard):

```
per_cell + partialconv + pe2d_frozen @ d=32, depth=3, pool=(4, 8)
```

Hierarchical readout was in the provisional default but **LOPO-dropped**: T3.4 composed LOPO mean 0.787 vs Default LOPO mean 0.810 (−0.023 pp). Pooled win did not transfer. `pe2d_frozen` ties learned `pe2d` at LOPO within noise (0.807 vs 0.810) and saves 176 learnable params, so it promotes.

### Pipeline

```
INPUT
 signal (B, N_e, 130)                 # 200 Hz, phoneme-centered [-0.15, 0.5) s
 electrode_active_mask (B, N_e)       # bool: non-artifact AND not pad
 support (B, N_e, 15)                 # Tier-1 BNA probability per electrode
 prev_phoneme (B,)

STAGE 1  Grid-scatter + Conv2d front-end
 grid-scatter → (B, 1, H_p, W_p, 130)
 Conv2d(1 → 8, k=3, pad=1) per time-step + GELU
 + Liu-2018 partial-conv mask renormalization (scale = nominal_sum / actual_sum)

STAGE 2  Masked-mean grid pool
 pool to (4, 8) = 32 cells
 + pe2d_frozen: row_emb[4, 16] ⊕ col_emb[8, 16] broadcast to cells (random-init, frozen)

STAGE 3  Per-cell temporal Conv1d (shared weights)
 Conv1d(8 → 32, k=30, s=10)
 150 ms kernel, 50 ms hop → 13 time-steps @ 20 Hz
 output: (B, 32 cells, 13 time, d=32)

STAGE 4  Atlas anchor (additive soft parcel embedding)
 + pooled_support @ P_emb[15, 32] broadcast across time

STAGE 5  Combined-attention backbone
 flatten to (B, 32 × 13 = 416 tokens, d=32)
 3 × [combined attention + FFN], 2 heads × 16, FFN 128, RoPE on time axis only
 pre-norm, residual, dropout 0.1
 token_active_mask (from pool-cell active mask broadcast across time) on Q and K

STAGE 6  D1 readout
 mean-pool memory across tokens
 concat prev_phoneme_emb → Linear → (B, 9)

LOSS  flat per-phoneme CE + teacher forcing; exhaustive 9³ AR at eval
```

~47 k parameters.

### Why this default

- **per_cell with (4, 8) pool** — rigid-grid pool is a local optimum at 4 patients / 1 min: pooled numbers match or edge the per-electrode arms within seed-noise. Grid-conditional; does not generalize to sEEG, but we are not at sEEG yet.
- **partialconv** — Liu-2018 mask renormalization upgrades conv correctness under missing / masked inputs. Composes cleanly; small pooled effect alone, beneficial in composition.
- **pe2d_frozen** — random-init row + column embeddings broadcast to pooled cells. LOPO-measured cross-patient regularizer (Batch 2: −0.013 uniform across 4 patients). Frozen version matches learned at LOPO within noise (0.807 vs 0.810), 176 fewer learnable params, no transfer cost.
- **D1 readout (mean-pool + prev_phoneme + Linear)** — the minimum AR head. Hierarchical readout was tested in composition (T3.4) and LOPO-dropped (−0.023 pp vs D1). Pooled win did not transfer.
- **d=32, depth=3** — matches Ben's 0.734 baseline param budget (~47 k). Growing capacity without growing the corpus does not move LOPO (Charmander scaling observation); revisit at Stage 2 scale.

This default is revisable at stage pass-through. Any Stage-1 win that requires rigid-grid assumptions (per_cell, pe2d) is re-tested on per-electrode tokens at Stage 2 and may be displaced.

---

## Frozen Stage-1 contract

Source of truth for stage-level decisions. Changes require stage pass-through.

| Item | Contract |
|---|---|
| **Spatial base** | fsaverage strict snap-to-pial. Patient side: `src/speech_decoding/v14/fsaverage_projection.py`. Atlas side: `data/atlas/fsaverage_bake_v2c/` (projfrac-avg + mri_surf2surf, no smoothing). Physical PSF is baked into the atlas; no query-time Gaussian. |
| **Support cache** | `data/atlas/support_cache_v2c_snap/<pt>_support_tier1.csv`. |
| **Tier-1 parcel set** | 15 LH Brainnetome parcels from `DEFAULT_BASE_PARCELS` in `src/speech_decoding/v14/token_spec.py`. Selection rule: `argmax_wins ≥ 10` on Phase-1 LH cohort. Embedding-lookup keys, not token slots. |
| **Loader** | Phoneme-level `.fif` (`derivatives/epoch(phonemeLevel)(CAR)/...`). Per-phoneme window `[-0.15, 0.5)` s = 130 samples @ 200 Hz. |
| **Hemisphere** | LH only. |
| **Label alphabet** | 9 ARPA phonemes alphabetical: `AA AE B G IY K P UW V`. |
| **Eval** | slot-averaged PER + per-phoneme PER, exhaustive 9³ AR decode at eval. |
| **CV** | grouped-by-token, same-patient-per-batch. |
| **Artifact channels** | hard-exclude only (no sig-channel filtering). |
| **Normalization** | upstream `productionZscore_highgamma` only. No additional normalization at load. |
| **Channel bridge** | Map 4 (128-strip), Map 3 (256-grid), S58 crop onto Map 3. `S39_channelMap.mat` non-authoritative; always use `S39_channelMapAll.mat`. |

---

## Patient scope (Stage 1)

| Role | Patients | Notes |
|---|---|---|
| **Core** | S14, S26, S33, S62 | All LH. Default pooled-joint training target. |
| **Extended** | S16, S23, S39 | All LH. Used in 7-LH pooled runs (Q1b/Q1d). |
| **Deferred to Stage 3** | S22, S58 | RH. Joins with sEEG. |
| **Excluded** | S32 (no HG response), S57 (hybrid strip, 52/256 sig, Map 8 unresolved) | Out of program entirely for Stage 1. |

Per-patient tables (sig channels, artifact counts, array layouts, corpus sizes) in `../references/data_reference.md`.

---

## Current scoreboard

All pooled-joint numbers: S14 + S26 + S33 + S62, 5 folds × 3 seeds, d=32, depth=3, pool=(4,8) unless noted. Update after each `scripts/ablation/collect.py <job_id>`. Raw source of truth: `../experiments/v14_ablation_log.csv`.

### Baselines (reference points)

| arm | pooled PER | LOPO mean (4 held-out) | notes |
|---|---|---|---|
| canonical per_cell | 0.794 ± 0.059 | 0.790 ± 0.056 (S14) | the thing to beat |
| flat Conv1d | 0.791 ± 0.034 | 0.797 / 0.758 / 0.777 / 0.793 | rigid-grid local optimum; ablation only |
| pe2d alone | 0.823 ± 0.059 | −0.013 uniform across 4 pts | LOPO-measured cross-patient regularizer |
| hierarchical alone | 0.761 ± 0.040 | LOPO-dropped via T3.4 composed | pooled win, LOPO loss |
| partial_conv alone | 0.795 ± 0.058 | trended −0.006 Batch-1 (within noise) | no-op alone; useful composed |
| per_cell + noemb (canonical alone) | 0.805 ± 0.027 | — | atlas-embedding pooled-inert; LOPO arm tested inside default (see T3.5 below) |
| default (partialconv + pe2d_frozen) | 0.816 ± 0.062 (4-core) / **0.833 ± 0.060 (7-LH)** | **0.810 ± 0.028** (S14 0.791 / S26 0.795 / S33 0.855 / S62 0.800) | LOPO confirmed via job 45768642; 7-LH pooled confirmed via job 45793090 |

### T3.3 — pe2d mechanism

| variant | job | pooled PER | LOPO mean (4 held-out) | gate |
|---|---|---|---|---|
| pe2d learned (baseline alone) | — | 0.823 ± 0.059 | −0.013 uniform | reference |
| pe2d learned (+ partialconv, default) | 45768642 | 0.816 ± 0.062 | 0.810 ± 0.028 | reference |
| pe2d_frozen (+ partialconv) | 45769655 | **0.814 ± 0.067** | **0.807 ± 0.015** (S14 0.790 / S26 0.804 / S33 0.829 / S62 0.804) | **PASSED**: ties learned within noise (−0.003 pp); 176 fewer params |
| row_only (+ partialconv) | 45768452 | 0.805 ± 0.049 | not run | pooled gate passed; LOPO not prioritized |
| col_only (+ partialconv) | 45768453 | 0.810 ± 0.057 | not run | pooled gate passed; LOPO not prioritized |

**Mechanism verdict:** learned / frozen / row / col cluster inside ±0.011 pooled. pe2d_frozen matches learned pe2d at LOPO within noise and saves 176 learnable params. **Promoted to default.**

### T3.4 — pe2d + hierarchical composition

| variant | job | pooled PER | LOPO mean (4 held-out) | gate |
|---|---|---|---|---|
| pe2d + hierarchical + partialconv | 45768287 / 45769580 | 0.770 ± 0.036 | **0.787 ± 0.002** (S14 0.790 / S26 0.788 / S33 0.786 / S62 0.785) | **LOPO FAILED**: −0.023 pp vs Default LOPO 0.810 |

Pooled win (0.770) did not transfer. Per-patient LOPO is uniformly at/below Default LOPO: hierarchical's slot-specific queries overfit in-distribution but fail on the held-out patient. **Hierarchical is LOPO-dropped from the Stage-1 default.**

### T3.1 — atlas-anchored hierarchical readout

| variant | job | pooled PER | LOPO | gate |
|---|---|---|---|---|
| hierarchical_atlas + partialconv + pe2d | 45769582 | **0.765 ± 0.042** | not run | pooled ties T3.4 (0.770); moot since T3.4 LOPO-failed |

Anatomy-indexed query trades the 32-cell free `q_cell` for `pooled_support @ parcel_embedding`. Pooled PER ties T3.4 within noise (0.765 vs 0.770). LOPO not run — T3.4 parent LOPO-failed the same readout primitive, so atlas-anchoring the query is not a rescue path at Stage 1. **Retired for Stage 1.**

### T1.2 — aug decomposition (per-op)

| variant | job | pooled PER | gate |
|---|---|---|---|
| legacy (composite) | — | 0.811 ± 0.068 | reference (pooled-only; not a clean win) |
| shift_only | 45768470 | 0.797 ± 0.069 | **PASSED** (< 0.800). LOPO candidate. |
| amp_only | 45768472 | **0.787 ± 0.062** | **PASSED** (< 0.800). LOPO candidate. |
| dropout_only | 45768473 | **0.800 ± 0.059** | on the line (= 0.800); not advancing |
| noise_only | 45768474 | **0.796 ± 0.059** | **PASSED** (< 0.800). LOPO candidate. |

**Decomposition verdict:** shift / amp / noise all clear the pooled gate; dropout ties it. Inside ±0.015 seed-noise. No single op stands out by an actionable margin. LOPO for these is **not prioritized** — ~47 k-param network is aug-inert at current scale, and the Stage-1 default does not currently include augmentation. Revisit at Stage 2 (larger corpus + larger model).

### T2.2 — scalable-arch capacity probe

Tests whether the per-electrode arm closes on per_cell as capacity grows — informs Stage-2 architecture pivot.

| variant | job | pooled PER | gate |
|---|---|---|---|
| per_electrode + fourier_mni, d=32, depth=3 | — | 0.826 ± 0.046 | reference |
| per_electrode + fourier_mni, d=64, depth=3 (h=4) | 45768288 | 0.786 ± 0.034 | **PASSED**: narrows pooled deficit vs per_cell (0.795) to 0.009 pp |
| per_electrode + fourier_mni, d=32, depth=4 | 45768489 | **0.841 ± 0.062** | **FAILED**: depth alone (no width) does not help per_electrode |

**Capacity probe verdict:** width (d=64) closes the per_cell gap; depth alone (d=32, depth=4) does not. Per-electrode capacity is width-bound at this corpus size. At Stage-2 scale, start per-electrode at d=64.

### T3.5 — noemb LOPO (atlas-mechanism isolation)

Ran at Stage-1 scale to isolate the atlas-mechanism question before Stage 2 introduces SSL + scale confounds. Identical to T3.3_frozen LOPO but with `--no-parcel-embedding` on both pretrain and finetune.

| variant | job | pooled PER (pretrain, mean over 4 held-out groups) | LOPO mean (4 held-out) | gate |
|---|---|---|---|---|
| default | 45768642 | 0.794 ± 0.032 (ref) | 0.810 ± 0.028 (S14 0.791 / S26 0.795 / S33 0.855 / S62 0.800) | reference |
| default − P_emb | 45793091 | 0.821 ± 0.036 | **0.823 ± 0.055** (S14 0.807 / S26 0.825 / S33 0.841 / S62 0.817) | atlas-mechanism LOPO-inert at 4-core |

Per-patient deltas vs default LOPO: **S14 +0.016, S26 +0.030, S33 −0.014, S62 +0.017 (aggregate +0.013)**. All inside the ±0.020 LOPO seed-noise band. **Finding: the soft parcel embedding does not earn its LOPO keep at Stage-1 scale.** Not actively hurting — but not mechanism-claim-supporting either.

**Decision:** keep `P_emb` in the architecture (inertness at 4-core does not preclude load-bearing at larger scale; zero-parameter path would require refactor); re-test under Stage-2 SSL + scale. Flag for discussion with Zac: current cross-patient story is carried by `per_cell + partialconv + pe2d_frozen` backbone, not by the atlas anchor at this scale.

### H1.2 — full Phase-1 LH scope validation

| run | job | pooled PER (mean-over-patients) | vs 4-core default | gate |
|---|---|---|---|---|
| 7-LH pooled default (S14/S16/S23/S26/S33/S39/S62) + save-ckpt | 45793090 | **0.833 ± 0.060** | +0.017 pp | **PASSED**: within noise; scaling to extended Phase-1 LH cohort does not degrade the default |

Per-patient (from pooled run): S14 0.826 / S16 0.835 / S23 0.851 / S26 0.830 / S33 0.821 / S39 0.840 / S62 0.828 — tight per-patient spread (range 0.030) suggests the default generalizes uniformly. **H1.2 confirmed.** Checkpoint saved for Stage-2 warm-start / SSL init (copy from `/work/ht203/results/v14_pooled/7lh_default_partialconv_pe2dfrozen_d32_depth3/` to `/hpc/group/coganlab/ht203/stage1_ckpt/` before `/work` 75-day purge).

### Gate thresholds

- **Pooled advance:** PER < 0.800 (≤ 5 pp from S14 baseline 0.734; matches current best pooled).
- **LOPO promote:** mean across 4 held-out ≤ pe2d-alone LOPO (−0.013 uniform reference).
- **Seed-noise band:** ±0.007 S14 / ±0.015 pooled / ±0.020 LOPO. Differences inside this band are not actionable.

### Updating this section

After each `scripts/ablation/collect.py <job_id>`:
1. Find new rows in `../experiments/v14_ablation_log.csv`.
2. Fill pooled PER, apply gate verdict in the matching table.
3. If gate passed and a LOPO wrapper exists: submit via `sbatch scripts/v14_core/v14_lopo_<variant>_dcc.sh`; record in `.ablation_submissions.jsonl`.
4. When LOPO lands: fill LOPO column, write a 1-line verdict.

### Wave closed 2026-04-19

8 jobs landed. Stage-1 default updated to `per_cell + partialconv + pe2d_frozen` (D1 readout). Hierarchical LOPO-dropped; atlas-anchored hierarchical retired. T1.2 aug decomposition inconclusive at current scale (all variants inside seed-noise, LOPO not prioritized). T2.2 width > depth for per-electrode capacity.

### Close-out wave 2026-04-20

2 jobs landed (45793090, 45793091). Stage 1 genuinely closed:

- **H1.2 confirmed at full Phase-1 LH scope.** 7-LH pooled default = 0.833 ± 0.060 (per-patient range 0.826–0.851), +0.017 pp vs 4-core within noise. Stage-1 default holds at the originally-defined patient scope. Checkpoint saved for Stage-2 warm-start.
- **Atlas-mechanism isolated.** noemb LOPO = 0.823 ± 0.055 vs default LOPO 0.810 ± 0.028 — noemb matches or slightly exceeds default on 3/4 patients, all inside ±0.020 LOPO seed-noise. **Soft parcel embedding is LOPO-inert at 4-core, ~1 min/patient.** Kept in architecture; mechanism-claim deferred to Stage-2 scale.

**Architectural ablation fully paused** per §Discipline. Stage-1 is closed. Pivot to Stage-2 prerequisite work (continuous-sample loader, SSL objective, calibration stub) while waiting on data unblock (13 missing lexical FreeSurfer recons). Stage-2 atlas-mechanism question (P_emb at larger scale / with SSL / vs cross-attn) is the decisive re-test for noemb.

---

## Rejected paths (Stage 1)

- **flat Conv1d as default** — rigid-grid local optimum (0.791 pooled ≈ per_cell). Retained as ablation only.
- **hierarchical readout (any composition)** — LOPO-dropped 2026-04-19. Pooled win (0.770, T3.4) did not transfer: LOPO mean 0.787 vs Default LOPO 0.810 (−0.023 pp). Atlas-anchored variant (T3.1 hieratlas, 0.765 pooled) tied the base form and was retired without LOPO on the same reasoning. Slot-specific queries overfit the 4-patient in-distribution set. Revisit at Stage-2 scale only with a new motivation, not as a reopened question.
- **noemb (no soft parcel embedding) as a Stage-1 rejection** — *not* rejected. Pooled-inert at current scale (0.805 vs canonical 0.794, inside ±0.015). LOPO tested 2026-04-20 (T3.5 / job 45793091): noemb LOPO 0.823 ± 0.055 vs default LOPO 0.810 ± 0.028 — matches within seed-noise on 3/4 patients. **Soft parcel embedding is LOPO-inert at 4-core, ~1 min/patient.** Kept in architecture; load-bearing claim deferred to Stage-2 scale re-test under SSL / 16+ patients.
- **T3.6 original specification** (per_cell + cross-attn + dual-stream + 611-token backbone) — deferred to Stage 2 where cross-attn lives naturally on per_electrode tokens without dual-stream.
- **FiLM / AdaIN / patient-stat normalization** — zero in-modality precedent (2026-04-19 iEEG-FM audit).
- **Register tokens** — parcel embedding already serves the shared-aggregator role at current parameter budget.
- **Single-stream parcel-only (MIBRAIN path)** — destroys within-parcel information.
- **Per-patient diagonal + softsign** — citation-to-form gap too wide without a data-scarcity-specific argument. Revisit only if Stage-2 residual-patient-variance evidence warrants.

---

## Discipline

Three rules override naive "keep adding wins":

1. **Pooled and LOPO can disagree.** pe2d lost pooled and won LOPO. Hierarchical won pooled, LOPO pending. Single-protocol evidence is insufficient to default an architectural change.
2. **Stop architectural ablation at the noise floor.** The pe2d mechanism probe already showed this: learned / frozen / row / col cluster inside ±0.015. More variants don't change the verdict.
3. **Stage-1 wins that require rigid-grid assumptions don't carry forward.** Any Stage-1 default that wins via per_cell or pe2d is re-tested on per-electrode tokens at Stage 2. Do not promote rigid-grid defaults as if they survive a sensor change.

When the current wave ends, architectural ablation pauses. Stage-2 data unblock is the trigger for the next architectural round; atlas-mechanism questions tied to scale (additive vs cross-attn, does P_emb earn its LOPO keep) become decidable there, not here.
