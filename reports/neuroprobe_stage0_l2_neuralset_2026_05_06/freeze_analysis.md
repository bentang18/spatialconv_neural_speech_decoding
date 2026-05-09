# L.2 Freeze Analysis (Tier A)

Tier-A grid: 3 references × 3 views = 9 cells. 12 sessions × 15 tasks per cell.

Baseline (upstream Lap+spec): **L.2.R4xI2** = 0.6132 mean AUROC.

## Cell ranking (mean AUROC, bootstrap 95% CI)

| rank | cell | recipe | n | mean | CI_lo | CI_hi | Δ vs baseline |
|---|---|---|---|---|---|---|---|
| 1 | L.2.R3xI2 | bipolar × stft_abs | 180 | 0.6157 | 0.5970 | 0.6360 | +0.0025 |
| 2 | L.2.R4xI2 ★ | shaftLap × stft_abs [D.0 baseline] | 180 | 0.6132 | 0.5949 | 0.6334 | +0.0000 |
| 3 | L.2.R0xI2 | raw × stft_abs | 180 | 0.5923 | 0.5761 | 0.6108 | -0.0209 |
| 4 | L.2.R3xI3 | bipolar × HG envelope | 180 | 0.5893 | 0.5730 | 0.6067 | -0.0239 |
| 5 | L.2.R4xI3 | shaftLap × HG envelope (privileged) | 180 | 0.5868 | 0.5701 | 0.6037 | -0.0264 |
| 6 | L.2.R0xI3 | raw × HG envelope | 180 | 0.5743 | 0.5604 | 0.5901 | -0.0389 |
| 7 | L.2.R0xI0 | raw × voltage | 180 | 0.5534 | 0.5432 | 0.5635 | -0.0598 |
| 8 | L.2.R3xI0 | bipolar × voltage | 180 | 0.5516 | 0.5416 | 0.5626 | -0.0616 |
| 9 | L.2.R4xI0 | shaftLap × voltage | 180 | 0.5498 | 0.5405 | 0.5603 | -0.0634 |

## Reference marginal (averaged over views)

| ref | recipe | n | mean | CI_lo | CI_hi |
|---|---|---|---|---|---|
| R3 | bipolar (adjacent within-shaft) | 540 | 0.5855 | 0.5762 | 0.5955 |
| R4 | shaft Laplacian | 540 | 0.5833 | 0.5740 | 0.5928 |
| R0 | raw | 540 | 0.5733 | 0.5655 | 0.5816 |

## View marginal (averaged over references)

| view | recipe | n | mean | CI_lo | CI_hi |
|---|---|---|---|---|---|
| I2 | STFT magnitude | 540 | 0.6071 | 0.5967 | 0.6181 |
| I3 | HG envelope (70-150 Hz) | 540 | 0.5835 | 0.5740 | 0.5931 |
| I0 | raw voltage | 540 | 0.5516 | 0.5458 | 0.5578 |

## Factor interaction (max |residual| = 0.0092)

Residual = observed cell mean − (grand + ref_main + view_main). Large residuals = the ref × view choice cannot be reduced to two independent main effects. Top 3:

| ref | view | observed | additive_pred | residual |
|---|---|---|---|---|
| R0 | I0 | 0.5534 | 0.5442 | +0.0092 |
| R0 | I2 | 0.5923 | 0.5997 | -0.0074 |
| R3 | I0 | 0.5516 | 0.5564 | -0.0048 |

## Paired Wilcoxon (top significance)

| cell A | cell B | n pairs | Δ (A − B) | p |
|---|---|---|---|---|
| L.2.R3xI2 | L.2.R4xI0 | 180 | +0.0658 | 2.28e-25 |
| L.2.R0xI0 | L.2.R3xI2 | 180 | -0.0623 | 3.08e-25 |
| L.2.R3xI0 | L.2.R3xI2 | 180 | -0.0641 | 3.37e-25 |
| L.2.R0xI3 | L.2.R3xI2 | 180 | -0.0414 | 1.03e-24 |
| L.2.R3xI2 | L.2.R4xI3 | 180 | +0.0288 | 1.42e-24 |
| L.2.R4xI0 | L.2.R4xI2 | 180 | -0.0634 | 1.67e-24 |
| L.2.R3xI0 | L.2.R4xI2 | 180 | -0.0616 | 4.95e-24 |
| L.2.R0xI0 | L.2.R4xI2 | 180 | -0.0598 | 5.81e-24 |
| L.2.R3xI2 | L.2.R3xI3 | 180 | +0.0264 | 2.69e-23 |
| L.2.R4xI2 | L.2.R4xI3 | 180 | +0.0264 | 9.16e-23 |
| L.2.R0xI3 | L.2.R4xI2 | 180 | -0.0389 | 9.83e-23 |
| L.2.R3xI3 | L.2.R4xI2 | 180 | -0.0239 | 1.22e-20 |
| L.2.R0xI2 | L.2.R3xI2 | 180 | -0.0234 | 1.30e-16 |
| L.2.R0xI2 | L.2.R0xI3 | 180 | +0.0180 | 6.88e-15 |
| L.2.R0xI2 | L.2.R4xI2 | 180 | -0.0209 | 1.03e-14 |

## Freeze decision tree

1. **If a cell's CI strictly dominates the baseline CI AND paired Wilcoxon
   vs baseline gives p < 0.001 with Δ ≥ 0.005**: freeze that cell.
2. **Else, if the top cell ties baseline within CI overlap**: freeze
   baseline (`L.2.R4xI2`, upstream parity). The view/reference choice is
   not load-bearing at linear-readout scope.
3. **If view marginal swamps reference marginal** (|Δ_view| > 2× |Δ_ref|):
   the headline is 'view matters; reference does not.' This shifts the
   Stage-1 v14 priority from reference design to spectral feature design.
4. **If interaction residuals are large** (max |res| ≥ 0.01): a single
   main-effect story is misleading; report the 9-cell heatmap rather
   than just the marginals.
5. **Always**: confirm winner survives D.0a CrossSubject binary (Tier C)
   before final freeze. Cross-subject distribution shift may swap.


## Files

- `cell_ci_forest.png` — bootstrap-CI ranking (this analyzer)
- `factor_marginals.png` — ref/view marginal effects (this analyzer)
- `cell_task_heatmap.png`, `cell_aggregate_bar.png` — collector
- `paired_tests.csv` — collector (also embedded above)
- `aggregate_diagnostics.csv` — collector (per-row source)
- `freeze_analysis.{md,json}` — this analyzer
