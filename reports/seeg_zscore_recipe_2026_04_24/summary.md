# A4 — sEEG z-score recipe distributional check (2026-04-24)

Tested 3 patients: D0040, D0073, D0096.

## Verdict

- Baseline looks raw (high physical-unit magnitude): **3/3**
- Production looks z-scored (per-channel std ~1): **3/3**

## Structural finding

sEEG `desc-production_highgamma.fif` is already z-scored. Unlike uECoG — where
`production_highgamma.fif` is a NaN-stub and reconstruction uses
`productionMeanSub` + baseline — sEEG has no `productionMeanSub` file on Box.
**Stage-3 loader consumes `desc-production_highgamma.fif` directly.**

## Limitation (Nanlin-clarification)

Without raw-production data, we cannot reverse-engineer whether the specific
recipe is A (per-channel mean/std over all baselines) or B (median/MAD) or
per-trial-normalized. uECoG audit 2026-04-18 showed A ≡ recording-level
median/MAD up to per-channel affine (ρ=1.0000), so the recipe question is
academic for model-input purposes as long as the convention is *consistent
within-patient*. This script confirms that consistency holds on D40/D73/D96.

## Per-patient stats

| patient | base grand|max| | base per_ch_std | prod per_ch_std | perc per_ch_std |
|---|---:|---:|---:|---:|
| D0040 | 1.31e-02 | 7.61e-06 | 0.551 | 0.743 |
| D0073 | 3.85e-05 | 1.31e-06 | 0.838 | 0.868 |
| D0096 | 2.00e-04 | 1.57e-06 | 0.948 | 1.006 |
