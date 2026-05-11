# L.3 filtering-sweep analysis — F0 vs F1/F2/F3

Cell axis: L.2 winner R4xI2 (shaft_laplacian × stft_abs) + N1 (train_set_fixed).
F0 baseline source: L.2 winner R4xI2 (reports/neuroprobe_stage0_l2_neuralset_2026_05_06/L.2.R4xI2).

Noise band: ±0.005 (matches L.4 anchor-robustness convention).

## Aggregate

| cell | recipe | n | mean | sd | Δ vs F0 | sd(Δ) | better/worse/tie |
|---|---|---|---|---|---|---|---|
| F0 | no filter (parity to L.2 winner) | 12 | 0.6132 | 0.0254 | — | — | — |
| F1 | 60 + 120 + 180 Hz notch | 7 | 0.6027 | 0.0215 | +0.0016 | 0.0058 | 2/3/6 |
| F2 | F1 + 0.5 Hz HPF | 7 | 0.6043 | 0.0217 | +0.0032 | 0.0059 | 4/1/5 |
| F3 | F1 + 1.0 Hz HPF | 7 | 0.6042 | 0.0215 | +0.0031 | 0.0058 | 4/1/5 |

## Per-session AUROC

| session | F0 | F1 | F2 | F3 | Δ F1-F0 | Δ F2-F0 | Δ F3-F0 |
|---|---|---|---|---|---|---|---|
| (1,1) | 0.6124 | — | — | — | — | — | — |
| (1,2) | 0.6206 | 0.6181 | 0.6181 | 0.6178 | -0.0026 | -0.0026 | -0.0028 |
| (10,0) | 0.5734 | 0.5880 | 0.5883 | 0.5878 | +0.0145 | +0.0148 | +0.0144 |
| (10,1) | 0.6163 | — | — | — | — | — | — |
| (2,0) | 0.6455 | — | — | — | — | — | — |
| (2,4) | 0.6276 | — | — | — | — | — | — |
| (3,0) | 0.6391 | 0.6386 | 0.6425 | 0.6421 | -0.0004 | +0.0035 | +0.0030 |
| (3,1) | 0.6494 | — | — | — | — | — | — |
| (4,0) | 0.6145 | 0.6145 | 0.6145 | 0.6145 | +0.0000 | +0.0000 | +0.0000 |
| (4,1) | 0.5960 | 0.5957 | 0.5963 | 0.5965 | -0.0003 | +0.0003 | +0.0005 |
| (7,0) | 0.5813 | 0.5814 | 0.5877 | 0.5880 | +0.0001 | +0.0064 | +0.0066 |
| (7,1) | 0.5827 | 0.5827 | 0.5827 | 0.5827 | +0.0000 | +0.0000 | +0.0000 |

## Verdicts

- **F1** (60 + 120 + 180 Hz notch): no-op (Δ within ±0.005 noise band) — keep L.2 winner unchanged
- **F2** (F1 + 0.5 Hz HPF): no-op (Δ within ±0.005 noise band) — keep L.2 winner unchanged
- **F3** (F1 + 1.0 Hz HPF): no-op (Δ within ±0.005 noise band) — keep L.2 winner unchanged

## Decision tree

1. **All Δ within ±0.005**: filtering is no-op at linear-readout scope — freeze L.2 winner unchanged, log finding in MEMORY.md.
2. **Notch (F1) helps but HPFs (F2/F3) tie or hurt**: fold notch into the L.2 winner; HPF transient + spectral leakage not worth it.
3. **HPF helps (F2 or F3)**: the input view (stft_abs covers ≥0 Hz) carries DC drift signal — investigate whether v14's cross-attn is absorbing nuisance or whether band-limit the front-end.
4. **All hurt**: upstream-equivalent wiring is already at the cleaning ceiling for this protocol — focus moves to L.4 anchor + L.5 nuisance.
