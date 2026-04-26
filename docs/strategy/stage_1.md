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

Stage-1 default (updated 2026-04-20 after scoreboard re-audit corrected a direction-flipped LOPO verdict, and T3.1 LOPO landed tying T3.4):

```
per_cell + partialconv + pe2d + hierarchical_atlas @ d=32, depth=3, pool=(4, 8)
```

Previous default was `per_cell + partialconv + pe2d_frozen` with mean-pool (D1) readout. The T3.4 composition (hierarchical + learned pe2d + partialconv) wins pooled 0.770 / LOPO 0.787 vs previous 0.814 / 0.807. T3.1 (atlas-anchored `hierarchical_atlas` — same architecture as T3.4 but the cell query comes from `pooled_support @ P_emb` instead of a free learned `q_cell`) ties T3.4 pooled (0.765 vs 0.770) and LOPO (0.788 vs 0.787). **T3.1 promoted as default on the tiebreak:** atlas-grounded query carries the program-hypothesis cross-patient story (calibration via Brainnetome anchoring) that a free `q_cell` does not, at matched accuracy and parameter budget. T3.4 is kept as the second-choice baseline.

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
 + pe2d: row_emb[4, 16] ⊕ col_emb[8, 16] broadcast to cells (learnable)

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

STAGE 6  Hierarchical_atlas readout
 cell-vec = mean over time for each of 32 cells → (B, 32, d)
 pooled_support[32, 15] = masked-mean pool of support[N_e, 15] onto 32 cells
 cell_query = pooled_support @ P_emb[15, d] → (B, 32, d)   ← atlas-anchored (no free q_cell)
 scores  = (cell_vec · cell_query).sum(-1) * (1/√d) → (B, 32)
 readout = softmax(scores) · cell_vec → (B, d)
 concat prev_phoneme_emb → Linear → (B, 9)

LOSS  flat per-phoneme CE + teacher forcing; exhaustive 9³ AR at eval
```

~47 k parameters — same budget as T3.4, the 1 k free `q_cell[32, d]` is replaced with the pre-existing P_emb projection.

### Why this default

- **per_cell with (4, 8) pool** — rigid-grid pool is a local optimum at 4 patients / 1 min: pooled numbers match or edge the per-electrode arms within seed-noise. Grid-conditional; does not generalize to sEEG, but we are not at sEEG yet.
- **partialconv** — Liu-2018 mask renormalization upgrades conv correctness under missing / masked inputs. Composes cleanly; small pooled effect alone, beneficial in composition.
- **pe2d (learned)** — row + column embeddings broadcast to pooled cells. LOPO-measured cross-patient regularizer (pe2d-alone: −0.013 uniform across 4 patients). Learned version is used in T3.4 (the composition being elevated). Frozen variant ties at LOPO within noise and is the zero-parameter fallback; kept as ablation.
- **hierarchical_atlas readout** — cell query is computed as `pooled_support @ P_emb[15, d]` and softmax-weights the 32 cell vectors. Same information pathway as T3.4's free `q_cell[32, d]` but the query is anatomically grounded — it is literally a linear projection of the cell's Brainnetome support profile through the learned parcel embedding. T3.1 ties T3.4 on both pooled (0.765 vs 0.770) and LOPO (0.788 vs 0.787); chosen as default because the atlas-grounded query carries the cross-patient calibration story that the program hypothesis commits to.
- **d=32, depth=3** — matches Ben's 0.734 baseline param budget (~47 k). Growing capacity without growing the corpus does not move LOPO; revisit at Stage 2 scale.

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
| hierarchical alone (no partialconv, no pe2d) | 0.761 ± 0.040 | **not LOPO-tested** | best pooled 4-core in the log; edges T3.4 (0.770) within noise. Gap: if T3.1 LOPO ties T3.4, worth LOPO-testing hierarchical-alone to check whether partialconv+pe2d are load-bearing in the T3.4 composition. |
| partial_conv alone | 0.795 ± 0.058 | trended −0.006 Batch-1 (within noise) | no-op alone; useful composed |
| per_cell + noemb (canonical alone) | 0.805 ± 0.027 | — | atlas-embedding pooled-inert; LOPO arm tested inside old default (see T3.5 below) |
| old default (partialconv + pe2d learned, D1 readout) | 0.816 ± 0.062 (4-core) / 0.833 ± 0.060 (7-LH) | 0.810 ± 0.028 (S14 0.791 / S26 0.795 / S33 0.855 / S62 0.800) | LOPO via job 45768642; 7-LH pooled via job 45793090 |
| old default (partialconv + pe2d_frozen, D1 readout) | 0.814 ± 0.067 (4-core) | 0.807 ± 0.015 (S14 0.790 / S26 0.804 / S33 0.829 / S62 0.804) | LOPO via job 45769655 — dominated by T3.4 on both protocols |
| T3.4 (partialconv + pe2d + hierarchical) — second choice | 0.770 ± 0.036 (4-core) | 0.787 ± 0.002 (S14 0.790 / S26 0.788 / S33 0.786 / S62 0.785) | free `q_cell` version; see T3.4 below. Elevated 2026-04-20 interim. |
| **new default — T3.1 (partialconv + pe2d + hierarchical_atlas)** | **0.765 ± 0.042** (4-core) | **0.788 ± 0.014** (S14 0.806 / S26 0.773 / S33 0.789 / S62 0.783) | atlas-anchored query (`pooled_support @ P_emb`). Ties T3.4 on both protocols; wins the default tiebreak via program-hypothesis alignment. Job 45798311 (2026-04-20). |

### T3.3 — pe2d mechanism

| variant | job | pooled PER | LOPO mean (4 held-out) | gate |
|---|---|---|---|---|
| pe2d learned (baseline alone) | — | 0.823 ± 0.059 | −0.013 uniform | reference |
| pe2d learned (+ partialconv, default) | 45768642 | 0.816 ± 0.062 | 0.810 ± 0.028 | reference |
| pe2d_frozen (+ partialconv) | 45769655 | **0.814 ± 0.067** | **0.807 ± 0.015** (S14 0.790 / S26 0.804 / S33 0.829 / S62 0.804) | **PASSED**: ties learned within noise (−0.003 pp); 176 fewer params |
| row_only (+ partialconv) | 45768452 | 0.805 ± 0.049 | not run | pooled gate passed; LOPO not prioritized |
| col_only (+ partialconv) | 45768453 | 0.810 ± 0.057 | not run | pooled gate passed; LOPO not prioritized |

**Mechanism verdict:** learned / frozen / row / col cluster inside ±0.011 pooled. pe2d_frozen matches learned pe2d at LOPO within noise and saves 176 learnable params. **Promoted to default.**

### T3.4 — pe2d + hierarchical composition (second choice after T3.1)

| variant | job | pooled PER | LOPO mean (4 held-out) | gate |
|---|---|---|---|---|
| pe2d + hierarchical + partialconv | 45768287 / 45769580 | 0.770 ± 0.036 | **0.787 ± 0.002** (S14 0.790 / S26 0.788 / S33 0.786 / S62 0.785) | **PASSED** — tied by T3.1, superseded on anatomy-grounding tiebreak |

Both protocols win vs the previous default. Per-patient LOPO deltas vs `per_cell + partialconv + pe2d_frozen` default (0.807 LOPO mean):
- S14: 0.790 − 0.790 = 0.000 (tie)
- S26: 0.788 − 0.804 = −0.016 (T3.4 better, edge of noise)
- S33: 0.786 − 0.829 = **−0.043** (T3.4 dominant; S33 recovery from the previous default's outlier)
- S62: 0.785 − 0.804 = −0.019 (T3.4 better, edge of noise)

Aggregate LOPO delta −0.020 pp; T3.4 dominates on both protocols and all 4 LOPO patients. The previous verdict "LOPO FAILED −0.023 pp" was direction-flipped (PER is wrong/total, so a negative delta is an improvement). Corrected 2026-04-20. **Hierarchical readout promoted to Stage-1 default.**

### T3.1 — atlas-anchored hierarchical readout (new Stage-1 default)

| variant | job | pooled PER | LOPO mean (4 held-out) | gate |
|---|---|---|---|---|
| hierarchical_atlas + partialconv + pe2d | 45769582 (pooled) / 45798311 (LOPO) | **0.765 ± 0.042** | **0.788 ± 0.014** (S14 0.806 / S26 0.773 / S33 0.789 / S62 0.783) | **PASSED — promoted to default on tiebreak vs T3.4** |

Anatomy-indexed query: `cell_query = pooled_support @ P_emb[15, d]` replaces T3.4's free `q_cell[32, d]`. Pooled and LOPO both tie T3.4 within noise:

| protocol | T3.1 | T3.4 | Δ (T3.1 − T3.4) |
|---|---|---|---|
| pooled 4-core | 0.765 | 0.770 | −0.005 |
| LOPO S14 | 0.806 | 0.790 | +0.016 (T3.1 worse, edge of ±0.020 noise) |
| LOPO S26 | 0.773 | 0.788 | −0.015 (T3.1 better, edge of noise) |
| LOPO S33 | 0.789 | 0.786 | +0.003 (tie) |
| LOPO S62 | 0.783 | 0.785 | −0.002 (tie) |
| LOPO mean | 0.788 | 0.787 | +0.001 (tie) |

Per-patient spread differs: T3.4 is uniform (0.785–0.790 range 0.005); T3.1 is wider (0.773–0.806 range 0.033). T3.1 trades S14 saturation for S26 lift — consistent with the data-starved LOPO rule that capacity should shift toward weaker patients. Aggregate tie on both protocols; tiebreak resolves in T3.1's favor because `pooled_support @ P_emb` grounds the readout in Brainnetome anatomy, matching the program hypothesis (problem 1: calibration; problem 2: shared dynamics) — a free `q_cell` does not.

Also a param saving: T3.1's query is zero new parameters (reuses P_emb), vs T3.4's 1024-param `q_cell`. Total ~47 k (T3.1) vs ~48 k (T3.4).

**Caveat for Stage 2:** S14 regression (+0.016 pp) is at the noise-band edge. At Stage-2 scale (SSL + more patients), watch whether S14 recovers as P_emb becomes better-characterized — if S14 stays degraded, the atlas projection is noise-limited at 4 patients × 1 min and a larger embedding or cross-attention (T3.6-mid) is the next step.

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
| per_electrode + fourier_mni, d=64, depth=3 (h=4) | 45768288 | **0.786 ± 0.034** | **PASSED**: at d=64, per_electrode+fourier *leads* per_cell+partialconv pooled (0.795) by 0.009 pp |
| per_electrode + fourier_mni, d=32, depth=4 | 45768489 | 0.841 ± 0.062 | **FAILED**: depth alone (no width) does not help per_electrode |

**Capacity probe verdict:** width (d=64) closes the per_cell gap and edges past it; depth alone (d=32, depth=4) regresses. Per-electrode capacity is width-bound at this corpus size. At Stage-2 scale, start per-electrode at d=64. (Previous phrasing "narrows deficit to 0.009 pp" was direction-flipped; at d=64 per_electrode leads per_cell, it does not trail.)

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

8 jobs landed. Initial reading promoted `per_cell + partialconv + pe2d_frozen` (D1 readout) as default and retired the hierarchical family. **Scoreboard re-audit 2026-04-20 corrected a direction-flipped LOPO verdict** on T3.4 (PER is `wrong/total`; lower is better, so the −0.023 pp T3.4 delta was a win, not a loss). Corrected outcomes in current scoreboard. T1.2 aug decomposition inconclusive at current scale (all variants inside seed-noise, LOPO not prioritized). T2.2 width > depth for per-electrode capacity; at d=64 per_electrode+fourier edges past per_cell.

### Close-out wave 2026-04-20

3 jobs landed (45793090, 45793091, 45798311) + scoreboard re-audit:

- **H1.2 confirmed at full Phase-1 LH scope.** 7-LH pooled (under the *old* default) = 0.833 ± 0.060 (per-patient range 0.826–0.851), +0.017 pp vs 4-core within noise. Scaling to extended Phase-1 LH cohort does not degrade the old default. Checkpoint saved for Stage-2 warm-start. *Re-running 7-LH under the T3.1 default is a pending Stage-1 close-out task.*
- **Atlas-mechanism isolated.** noemb LOPO = 0.823 ± 0.055 vs old default LOPO 0.807 ± 0.015 — aggregate +0.016 pp (noemb worse); all inside ±0.020 LOPO seed-noise. **Soft parcel embedding is LOPO-inert at 4-core, ~1 min/patient** — not a dominant win, not a loss. Kept in architecture; mechanism-claim deferred to Stage-2 scale.
- **Scoreboard re-audit 2026-04-20 (load-bearing).** T3.4 `hierarchical + partialconv + pe2d` LOPO 0.787 vs old default LOPO 0.807 was mis-read as "−0.023 pp LOPO FAILED" — direction was flipped. T3.4 wins both pooled (0.770 vs 0.814) and LOPO on all 4 patients, with a dominant S33 recovery (0.855 → 0.786, −0.069 pp vs learned-pe2d default). T3.4 elevated to interim default; T3.1 LOPO requeued (had been skipped on the flipped rationale).
- **T3.1 LOPO landed (job 45798311).** T3.1 `hierarchical_atlas + partialconv + pe2d` LOPO mean 0.788 ± 0.014 (S14 0.806 / S26 0.773 / S33 0.789 / S62 0.783) — ties T3.4 at 0.787 within noise. **T3.1 promoted to Stage-1 default on the tiebreak:** atlas-grounded query (`pooled_support @ P_emb`) carries the program-hypothesis cross-patient story at matched accuracy and a ~1 k parameter saving. T3.4 retained as second-choice baseline.

**Stage 1 closed 2026-04-20.** Default: `per_cell + partialconv + pe2d + hierarchical_atlas` @ d=32, depth=3, pool=(4,8). Architectural ablation paused per §Discipline. Pivot to Stage-2 prerequisite work (continuous-sample loader, SSL objective, calibration stub) + data unblock on 13 missing lexical FreeSurfer recons. Open follow-ups at Stage-1 scope (not blocking Stage-2 kickoff): re-run 7-LH under the T3.1 default; LOPO-test plain `hierarchical alone` to check whether partialconv+pe2d are load-bearing in the composition.

---

## Rejected paths (Stage 1)

- **flat Conv1d as default** — rigid-grid local optimum (0.791 pooled ≈ per_cell). Retained as ablation only.
- **~~hierarchical readout (any composition)~~ — removed from rejected paths 2026-04-20.** Previously listed as LOPO-dropped 2026-04-19 based on a direction-flipped reading (PER is `wrong/total`; the T3.4 LOPO 0.787 vs old default 0.810 was a **win** of 0.023 pp, not a loss). Corrected: T3.4 wins both pooled and LOPO across all 4 patients with a dominant S33 recovery; elevated to Stage-1 default. T3.1 (atlas-anchored) LOPO is now in-flight.
- **noemb (no soft parcel embedding) as a Stage-1 rejection** — *not* rejected. Pooled-inert at current scale (0.805 vs canonical 0.794, inside ±0.015). LOPO tested 2026-04-20 (T3.5 / job 45793091): noemb LOPO 0.823 ± 0.055 vs default LOPO 0.810 ± 0.028 — matches within seed-noise on 3/4 patients. **Soft parcel embedding is LOPO-inert at 4-core, ~1 min/patient.** Kept in architecture; load-bearing claim deferred to Stage-2 scale re-test under SSL / 16+ patients.
- **T3.6 original specification** (per_cell + cross-attn + dual-stream + 611-token backbone) — deferred to Stage 2 where cross-attn lives naturally on per_electrode tokens without dual-stream.
- **FiLM / AdaIN / patient-stat normalization** — zero in-modality precedent (2026-04-19 iEEG-FM audit).
- **Register tokens** — parcel embedding already serves the shared-aggregator role at current parameter budget.
- **Single-stream parcel-only (MIBRAIN path)** — destroys within-parcel information.
- **Per-patient diagonal + softsign** — citation-to-form gap too wide without a data-scarcity-specific argument. Revisit only if Stage-2 residual-patient-variance evidence warrants.

---

## Discipline

Three rules override naive "keep adding wins":

1. **Pooled and LOPO can disagree.** pe2d lost pooled and won LOPO. Hierarchical won both (corrected 2026-04-20 after a direction-flipped LOPO reading). Single-protocol evidence is insufficient to default an architectural change — and direction of the metric must be verified before citing deltas.
2. **Stop architectural ablation at the noise floor.** The pe2d mechanism probe already showed this: learned / frozen / row / col cluster inside ±0.015. More variants don't change the verdict.
3. **Stage-1 wins that require rigid-grid assumptions don't carry forward.** Any Stage-1 default that wins via per_cell or pe2d is re-tested on per-electrode tokens at Stage 2. Do not promote rigid-grid defaults as if they survive a sensor change.

When the current wave ends, architectural ablation pauses. Stage-2 data unblock is the trigger for the next architectural round; atlas-mechanism questions tied to scale (additive vs cross-attn, does P_emb earn its LOPO keep) become decidable there, not here.
