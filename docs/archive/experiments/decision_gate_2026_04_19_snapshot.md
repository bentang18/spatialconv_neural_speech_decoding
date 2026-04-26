# Decision gate tracker — 2026-04-19 wave

Live scoreboard for the next-steps wave. Updated as results land. **Authoritative source is `docs/experiments/v14_ablation_log.csv`**; this doc extracts per-arm summaries with decision verdicts attached.

All pooled-joint numbers are S14+S26+S33+S62, 5 folds × 3 seeds, d=32, depth=3, pool=(4,8) unless noted.

## Phase-1 default vs Phase-1.5 target — DO NOT LOSE

The Phase-1 default and the paper's architectural direction are **different objects**. Don't conflate them.

| scope | default | rationale |
|---|---|---|
| **Phase-1 (now, 4 pts × 1 min)** | `per_cell + partialconv + pe2d_frozen + hierarchical` | current-scale pooled winner; gated on in-flight LOPOs. Uses Conv2d + (4,8) grid pool — **rigid-grid, sEEG-incompatible.** Expedient, not the paper. |
| **Phase-1.5+ target (27 pts, SSL-pretrained)** | `per_electrode + fourier_mni + partialconv + hierarchical`, d=64 | no Conv2d, no grid scatter, no pool. Works on sEEG by design. At d=64 it's already within 0.009 pp of per_cell pooled (0.786 vs 0.794 — inside seed-noise). The composition with `hier + partialconv` is the untested Phase-1.5 experiment and will be the first architectural ablation once data scales. |

**Why this matters.** The canonical research goal (see `feedback_do_not_overfit_to_current_scale_2026_04_18.md`) is cross-patient / cross-sensor scalability. `per_cell` wins Phase-1 pooled by 0.009 pp but breaks on sEEG. Picking it as *the* default because it wins at current scale is exactly the overfitting-to-current-scale pattern. The Phase-1 default is an expedient while data is tiny; the paper's architecture is per-electrode.

**Action when data scales.** First ablation in Phase 1.5 = `per_electrode + fourier_mni + partialconv + hier` at d=64 + SSL pretrain, pooled + LOPO. If it matches or beats the Phase-1 default under the new scale, Phase-1's default is retired and never migrates to the paper.

## Baselines (reference)

| arm | pooled-joint PER | LOPO mean | notes |
|---|---|---|---|
| canonical per_cell | 0.794 ± 0.059 | 0.790 ± 0.056 (S14) | the thing to beat |
| flat Conv1d | 0.791 ± 0.034 | 0.797 / 0.758 / 0.777 / 0.793 | rigid-grid local optimum, retained as ablation |
| pe2d alone | 0.823 ± 0.059 | −0.013 uniform across 4 pts | **LOPO win was decisive, pooled was worse** |
| hierarchical alone | 0.761 ± 0.040 | — | **pooled win**, LOPO not yet measured |
| partial_conv alone | 0.795 ± 0.058 | — | pooled within noise |
| **partial_conv + pe2d (new default co-test)** | **0.816 ± 0.062** | **IN FLIGHT (45768642)** | no negative interaction vs pe2d alone |

Key pattern: pe2d and hierarchical are probably both wins but on different protocols. Don't default on a single-protocol read.

## Wave 2 — in flight

### T3.3 pe2d mechanism (decide whether default drops learnable layer)

| variant | job | pooled PER | LOPO mean | gate |
|---|---|---|---|---|
| pe2d learned (baseline) | — | 0.823 ± 0.059 | −0.013 uniform | reference |
| pe2d_frozen (+ partialconv) | 45768450 | **0.814 ± 0.067** | **IN FLIGHT (45769655)** | **PASSED**: +0.002 vs learned+partialconv (0.816). LOPO fired. |
| row_only (+ partialconv) | 45768452 | **0.805 ± 0.049** | — | **PASSED**: col axis not load-bearing; single best pe2d variant within noise. |
| col_only (+ partialconv) | 45768453 | **0.810 ± 0.057** | — | **PASSED**: row axis not load-bearing. |

**Mechanism verdict**: learned / frozen / row_only / col_only span 0.805–0.816, within ±0.011 (inside seed-noise). 192 PE params are not load-bearing. Either 1D axis alone is sufficient. Simplification opportunity: default to `frozen` or `row_only` (−176 params). Defer default change until LOPO row lands.

LOPO wrappers ready: `v14_lopo_pe2d_{frozen,row,col}_dcc.sh`. Submit after pooled clears.

### T3.4 pe2d + hierarchical composition

| variant | job | pooled PER | LOPO mean | gate |
|---|---|---|---|---|
| pe2d + hierarchical + partialconv | 45768287 | **0.770 ± 0.036** | **IN FLIGHT (45769580)** | **POOLED PASSED** (< 0.77 aspirational, < 0.795 required). LOPO fired 2026-04-19. |

Pooled beats hierarchical alone (0.761) within noise AND beats pe2d alone (0.823) by +0.053 pp. Composition is not just additive — it's the lowest pooled PER we've seen outside the rigid-grid `flat` arm.

### T1.2 aug decomposition

| variant | job | pooled PER | gate |
|---|---|---|---|
| legacy (composite) | — | 0.811 ± 0.068 | reference (pooled-only; not a clean win) |
| shift_only | 45768470 | **0.797 ± 0.069** | **PASSED** (< 0.800). Single-op aug recovers most of composite. LOPO candidate. |
| amp_only | 45768472 | running 7/15 | if pooled < 0.800 → advance to LOPO |
| dropout_only | 45768473 | pending | if pooled < 0.800 → advance to LOPO |
| noise_only | 45768474 | pending | if pooled < 0.800 → advance to LOPO |

### T2.2 scalable capacity probe

| variant | job | pooled PER | gate |
|---|---|---|---|
| per_electrode + fourier_mni, d=32, depth=3 | — | 0.826 ± 0.046 | reference |
| per_electrode + fourier_mni, d=64, depth=3 (h=4) | 45768288 | **0.786 ± 0.034** | **PASSED**: −0.040 pp vs d=32 ref, narrows pooled deficit vs per_cell (0.795) to 0.009 |
| per_electrode + fourier_mni, d=32, depth=4 | 45768489 | pending | if pooled < 0.810 → depth helps per_electrode, consider promotion |

Capacity (d=64) nearly closes the per_electrode pooled gap to per_cell. If depth=4 lands similarly, the scalable bucket is within seed noise of canonical at current scale — migration-relevant. Still below per_cell+partial_conv+pe2d (0.816) and well below pe2d+hierarchical+partialconv (0.770); current-scale optimum stays `per_cell` for now.

### Default LOPO (decisive)

| arm | job | pooled | LOPO per held-out | verdict |
|---|---|---|---|---|
| per_cell + partial_conv + pe2d (default) | 45768642 | 0.816 ± 0.062 | pending | if ≤ pe2d-alone LOPO → default confirmed |

## T3.6 parcel-token architecture (parked, decomposed path)

Full T3.6 (cross-attn parcel pool + dual-stream + parcel readout) stacks three orthogonal changes: cross-attn pool (vs additive `support @ P_emb` bias), dual-stream tokens (47 vs 32), parcel-only readout (vs global mean). Win/lose hard to attribute; 2.2× backbone compute for an unvalidated bet. Decomposed into three attributable steps, each gated on the previous.

Blocked on T3.4 LOPO + T3.1 pooled landing — those decide whether the atlas-anchoring premise underneath T3.6 is worth more machinery.

| step | change | params Δ | gate |
|---|---|---|---|
| **T3.6-lite** | readout only: support-weighted parcel-mean over cell tokens | 0 | if pooled ≥ T3.4 within noise AND LOPO ≥ T3.4 LOPO → advance to T3.6-mid |
| **T3.6-mid** | + Stage 6 cross-attn parcel pool (log(support) bias, ε=1e-4); backbone still over cells | ~+4k | if pooled + LOPO beat T3.6-lite → advance to T3.6-full |
| **T3.6-full** | + dual-stream: cells + parcels concat to 611 tokens through backbone | ~+0 (tokens, not params) | only run if T3.6-lite and T3.6-mid both pass |

Rationale: T3.1 already tests "atlas-derived query vs free-learned query" with ~3 lines of change, inside the current hierarchical readout. If T3.1 fails, the atlas-anchoring premise of T3.6 weakens; if it passes, T3.6-lite is the smallest next step in the same direction.

Deferred out of T3.6 entirely (no direct precedent in the 4 reference papers): FiLM on cells, POYO+-style query-cross-attention readout, additional Fourier PE / register tokens.

## Gate thresholds

- **Pooled advance**: PER < 0.800 (≤5pp from S14 baseline 0.734, matches current best pooled).
- **LOPO promote**: mean across 4 held-out ≤ corresponding pe2d-alone LOPO (−0.013 uniform reference).
- **Seed-noise band**: ±0.007 S14 / ±0.015 pooled / ±0.020 LOPO. Differences inside this band are not actionable.
- **Cross-sensor sanity**: per_electrode + fourier_mni arms only promote if they match or beat per_cell + pe2d at same (d, depth). Grid-only defaults are local optima — see feedback_do_not_overfit_to_current_scale_2026_04_18.md.

### T3.1 atlas-hierarchical readout

| variant | job | pooled PER | LOPO mean | gate |
|---|---|---|---|---|
| hierarchical_atlas + partialconv + pe2d | 45769582 | **IN FLIGHT** | — | if pooled ≤ T3.4 (0.770) within noise → anatomy-indexed query viable; LOPO next. |

Replaces the free-learned `q_cell` in hierarchical with `cell_query = pooled_support @ parcel_embedding` (atlas-anchored). Same parameter budget as mean_pool + d params from q_temporal. Motivation: test whether anatomical priors improve transfer vs cell-indexed free query.

## How to update this doc

After each `collect.py <job_id>`:
1. Find the new row(s) in `docs/experiments/v14_ablation_log.csv`.
2. Fill the matching `pooled PER` cell and apply gate verdict.
3. If gate passed and LOPO wrapper exists: submit via `sbatch scripts/v14_core/v14_lopo_<variant>_dcc.sh` after rsync; record LOPO job in `.ablation_submissions.jsonl`.
4. When LOPO lands: fill `LOPO mean` column and write a 1-line verdict in the row.
