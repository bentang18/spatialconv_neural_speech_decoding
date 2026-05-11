# Tier-C CrossSubject analysis — C.0 baseline vs C.1 + C.2 winners

Test of whether the within-session L.1 + L.2 winners survive the CrossSubject distribution shift. Subject 2 (upstream DS_DM_TRAIN_SUBJECT_ID) is excluded from the eval set.

Noise band: ±0.005 (matches L.3 / L.4 convention).

## Aggregate

| cell | recipe | n | mean | sd | Δ vs C.0 | sd(Δ) | better/worse/tie |
|---|---|---|---|---|---|---|---|
| C.0_baseline | upstream N1 × R4xI2 (reference) | 10 | 0.5310 | 0.0235 | — | — | — |
| C.1_l1_winner | L.1 winner × R4xI2 (norm transfer) | 10 | 0.5310 | 0.0235 | +0.0000 | 0.0000 | 0/0/10 |
| C.2_l1_l2_winner | L.1 + L.2 winners (joint generalization) | 10 | 0.5392 | 0.0277 | +0.0082 | 0.0135 | 6/4/7 |

## Per-session AUROC

| session | C.0_baseline | C.1_l1_winner | C.2_l1_l2_winner | Δ C.1_l1_winner-C.0 | Δ C.2_l1_l2_winner-C.0 |
|---|---|---|---|---|---|
| (1,1) | 0.5582 | 0.5582 | 0.5577 | +0.0000 | -0.0005 |
| (1,2) | 0.5664 | 0.5664 | 0.5734 | +0.0000 | +0.0070 |
| (10,0) | 0.5236 | 0.5236 | 0.5218 | +0.0000 | -0.0018 |
| (10,1) | 0.5075 | 0.5075 | 0.5123 | +0.0000 | +0.0048 |
| (3,0) | 0.5242 | 0.5242 | 0.5571 | +0.0000 | +0.0329 |
| (3,1) | 0.5392 | 0.5392 | 0.5730 | +0.0000 | +0.0338 |
| (4,0) | 0.5531 | 0.5531 | 0.5565 | +0.0000 | +0.0033 |
| (4,1) | 0.5340 | 0.5340 | 0.5336 | +0.0000 | -0.0004 |
| (7,0) | 0.4991 | 0.4991 | 0.5025 | +0.0000 | +0.0034 |
| (7,1) | 0.5041 | 0.5041 | 0.5040 | +0.0000 | -0.0002 |

## Verdicts

- **C.1_l1_winner** (L.1 winner × R4xI2 (norm transfer)): no transfer (Δ within ±0.005) — within-session winner doesn't carry
- **C.2_l1_l2_winner** (L.1 + L.2 winners (joint generalization)): generalizes (Δ > +0.005) — v14 inherits the gain

## Decision tree

1. **Both C.1 and C.2 generalize** (Δ > +0.005): freeze the joint winner; v14 inherits the L.1 norm + L.2 ref/view choice.
2. **C.1 generalizes but C.2 doesn't**: the L.2 ref/view choice was within-session-overfit; freeze L.1 winner only and revisit L.2.
3. **Neither generalizes (both Δ ≤ +0.005)**: within-session L-sweep winners do not transfer — v14 must rely on architecture (cross-attn, atlas anchoring) to make up the gap. Report this prominently.
4. **C.1 or C.2 hurts (Δ < −0.005)**: the within-session winner is actively bad cross-subject. Reject for v14 and document the distribution-shift signature.

