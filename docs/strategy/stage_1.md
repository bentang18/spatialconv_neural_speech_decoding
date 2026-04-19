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

Current Stage-1 default (provisional; gated on three in-flight LOPOs — see scoreboard):

```
per_cell + partialconv + pe2d_frozen + hierarchical readout @ d=32, depth=3, pool=(4, 8)
```

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

STAGE 6  Hierarchical readout
 atlas-anchored query over cell axis + learned query over time axis, fused
 concat prev_phoneme_emb → Linear → (B, 9)

LOSS  flat per-phoneme CE + teacher forcing; exhaustive 9³ AR at eval
```

~47 k parameters.

### Why this default

- **per_cell with (4, 8) pool** — rigid-grid pool is a local optimum at 4 patients / 1 min: pooled numbers match or edge the per-electrode arms within seed-noise. Grid-conditional; does not generalize to sEEG, but we are not at sEEG yet.
- **partialconv** — Liu-2018 mask renormalization upgrades conv correctness under missing / masked inputs. Composes cleanly; small pooled effect alone, beneficial in composition.
- **pe2d_frozen** — random-init row + column embeddings broadcast to pooled cells. LOPO-measured cross-patient regularizer (Batch 2: −0.013 uniform across 4 patients). Frozen version disambiguates mechanism from parameter-tuning; saves 176 learnable params vs learned pe2d.
- **hierarchical readout** — slot-specific recovery via learnable queries over cell + time axes. Best single architectural add at pooled probe; T3.4 composed pooled = 0.770 ± 0.036.
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

| arm | pooled PER | LOPO mean | notes |
|---|---|---|---|
| canonical per_cell | 0.794 ± 0.059 | 0.790 ± 0.056 (S14) | the thing to beat |
| flat Conv1d | 0.791 ± 0.034 | 0.797 / 0.758 / 0.777 / 0.793 | rigid-grid local optimum; ablation only |
| pe2d alone | 0.823 ± 0.059 | −0.013 uniform across 4 pts | LOPO-measured cross-patient regularizer |
| hierarchical alone | 0.761 ± 0.040 | pending (via T3.4 composed) | best single architectural add at pooled |
| partial_conv alone | 0.795 ± 0.058 | trended −0.006 Batch-1 (within noise) | no-op alone; useful composed |
| per_cell + noemb | 0.805 ± 0.027 | **not tested** | atlas-embedding pooled-inert; LOPO gap is the open question |
| default (partialconv + pe2d) | 0.816 ± 0.062 | **IN FLIGHT (45768642)** | no negative interaction vs pe2d alone |

### T3.3 — pe2d mechanism

| variant | job | pooled PER | LOPO | gate |
|---|---|---|---|---|
| pe2d learned (baseline alone) | — | 0.823 ± 0.059 | −0.013 uniform | reference |
| pe2d_frozen (+ partialconv) | 45768450 | **0.814 ± 0.067** | **IN FLIGHT (45769655)** | **PASSED**: +0.002 vs learned+partialconv; learned params not load-bearing |
| row_only (+ partialconv) | 45768452 | **0.805 ± 0.049** | — | **PASSED**: col axis not load-bearing |
| col_only (+ partialconv) | 45768453 | **0.810 ± 0.057** | — | **PASSED**: row axis not load-bearing |

**Mechanism verdict:** variants span 0.805–0.816, inside ±0.015 seed-noise. 192 learned PE params are inert at pooled. Simplification candidate: default to `frozen` or `row_only` (saves 176 params). Deferred until pe2d_frozen LOPO lands.

### T3.4 — pe2d + hierarchical composition

| variant | job | pooled PER | LOPO | gate |
|---|---|---|---|---|
| pe2d + hierarchical + partialconv | 45768287 | **0.770 ± 0.036** | **IN FLIGHT (45769580)** | **POOLED PASSED** (< 0.77 aspirational, < 0.795 required) |

Beats hierarchical alone (0.761) within noise AND beats pe2d alone (0.823) by +0.053 pp. Lowest pooled PER outside the rigid-grid flat arm. LOPO is the decisive gate.

### T3.1 — atlas-anchored hierarchical readout

| variant | job | pooled PER | LOPO | gate |
|---|---|---|---|---|
| hierarchical_atlas + partialconv + pe2d | 45769582 | **IN FLIGHT** | — | if pooled ≤ T3.4 (0.770) within noise → anatomy-indexed query viable |

Replaces free-learned `q_cell` in hierarchical with `cell_query = pooled_support @ parcel_embedding` (atlas-anchored). Tests whether anatomical priors improve transfer vs cell-indexed free query.

### T1.2 — aug decomposition (per-op)

| variant | job | pooled PER | gate |
|---|---|---|---|
| legacy (composite) | — | 0.811 ± 0.068 | reference (pooled-only; not a clean win) |
| shift_only | 45768470 | **0.797 ± 0.069** | **PASSED** (< 0.800). LOPO candidate. |
| amp_only | 45768472 | running 7/15 | if pooled < 0.800 → advance to LOPO |
| dropout_only | 45768473 | pending | if pooled < 0.800 → advance to LOPO |
| noise_only | 45768474 | pending | if pooled < 0.800 → advance to LOPO |

### T2.2 — scalable-arch capacity probe

Tests whether the per-electrode arm closes on per_cell as capacity grows — informs Stage-2 architecture pivot.

| variant | job | pooled PER | gate |
|---|---|---|---|
| per_electrode + fourier_mni, d=32, depth=3 | — | 0.826 ± 0.046 | reference |
| per_electrode + fourier_mni, d=64, depth=3 (h=4) | 45768288 | **0.786 ± 0.034** | **PASSED**: narrows pooled deficit vs per_cell (0.795) to 0.009 pp |
| per_electrode + fourier_mni, d=32, depth=4 | 45768489 | pending | if pooled < 0.810 → depth helps per_electrode |

### Default LOPO (decisive)

| arm | job | pooled | LOPO | verdict |
|---|---|---|---|---|
| per_cell + partialconv + pe2d | 45768642 | 0.816 ± 0.062 | pending | if ≤ pe2d-alone LOPO → default confirmed |

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

### When the wave ends

LOPOs above land → record verdicts in-place → update §Default architecture with new Stage-1 default. Then architectural ablation **pauses** until Stage 2 data unblocks.

---

## Rejected paths (Stage 1)

- **flat Conv1d as default** — rigid-grid local optimum (0.791 pooled ≈ per_cell) but beaten 2 pp by pe2d + hierarchical composition. Retained as ablation only.
- **hierarchical alone (no pe2d, no partialconv)** — pooled win, LOPO tie, patient-specific readout overfit signals. Not a default.
- **noemb (no soft parcel embedding) as a Stage-1 rejection** — *not* rejected. Pooled-inert at current scale (0.805 vs canonical 0.794, inside ±0.015) but LOPO never tested. This is the critical gap — deferred to Stage 2 atlas-mechanism ablation.
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
