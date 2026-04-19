# Decision gate tracker — 2026-04-19 wave

Live scoreboard for the next-steps wave. Updated as results land. **Authoritative source is `docs/experiments/v14_ablation_log.csv`**; this doc extracts per-arm summaries with decision verdicts attached.

All pooled-joint numbers are S14+S26+S33+S62, 5 folds × 3 seeds, d=32, depth=3, pool=(4,8) unless noted.

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
| pe2d_frozen | 45768450 | pending | — | if within ±0.005 of learned → default = frozen |
| row_only | 45768452 | pending | — | if matches learned → col axis not load-bearing |
| col_only | 45768453 | pending | — | if matches learned → row axis not load-bearing |

LOPO wrappers ready: `v14_lopo_pe2d_{frozen,row,col}_dcc.sh`. Submit after pooled clears.

### T3.4 pe2d + hierarchical composition

| variant | job | pooled PER | LOPO mean | gate |
|---|---|---|---|---|
| pe2d + hierarchical | 45768287 | pending | — | if pooled < 0.795 AND LOPO ≤ default-LOPO → promote hier to default readout |

LOPO wrapper ready: `v14_lopo_pe2d_hier_dcc.sh`. Submit after pooled clears.

### T1.2 aug decomposition

| variant | job | pooled PER | gate |
|---|---|---|---|
| legacy (composite) | — | 0.811 ± 0.068 | reference (pooled-only; not a clean win) |
| shift_only | 45768470 | pending | if pooled < 0.800 → advance to LOPO |
| amp_only | 45768472 | pending | if pooled < 0.800 → advance to LOPO |
| dropout_only | 45768473 | pending | if pooled < 0.800 → advance to LOPO |
| noise_only | 45768474 | pending | if pooled < 0.800 → advance to LOPO |

### T2.2 scalable capacity probe

| variant | job | pooled PER | gate |
|---|---|---|---|
| per_electrode + fourier_mni, d=32, depth=3 | — | 0.826 ± 0.046 | reference |
| per_electrode + fourier_mni, d=64, depth=3 | 45768288 | pending | if pooled < 0.810 → capacity helps per_electrode, consider promotion |
| per_electrode + fourier_mni, d=32, depth=4 | 45768489 | pending | if pooled < 0.810 → depth helps per_electrode, consider promotion |

### Default LOPO (decisive)

| arm | job | pooled | LOPO per held-out | verdict |
|---|---|---|---|---|
| per_cell + partial_conv + pe2d (default) | 45768642 | 0.816 ± 0.062 | pending | if ≤ pe2d-alone LOPO → default confirmed |

## Gate thresholds

- **Pooled advance**: PER < 0.800 (≤5pp from S14 baseline 0.734, matches current best pooled).
- **LOPO promote**: mean across 4 held-out ≤ corresponding pe2d-alone LOPO (−0.013 uniform reference).
- **Seed-noise band**: ±0.007 S14 / ±0.015 pooled / ±0.020 LOPO. Differences inside this band are not actionable.
- **Cross-sensor sanity**: per_electrode + fourier_mni arms only promote if they match or beat per_cell + pe2d at same (d, depth). Grid-only defaults are local optima — see feedback_do_not_overfit_to_current_scale_2026_04_18.md.

## T3.1 atlas-hierarchical readout (not yet scheduled)

Plan calls for it regardless of T3.4 direction. Code surgery pending; see task #8. Ships after this wave or once T3.4 result directs scope.

## How to update this doc

After each `collect.py <job_id>`:
1. Find the new row(s) in `docs/experiments/v14_ablation_log.csv`.
2. Fill the matching `pooled PER` cell and apply gate verdict.
3. If gate passed and LOPO wrapper exists: submit via `sbatch scripts/v14_core/v14_lopo_<variant>_dcc.sh` after rsync; record LOPO job in `.ablation_submissions.jsonl`.
4. When LOPO lands: fill `LOPO mean` column and write a 1-line verdict in the row.
