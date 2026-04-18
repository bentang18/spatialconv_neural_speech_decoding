# Tier-1 support cache QC report (Euclidean pool)

Per-patient summary of `docs/support_cache_euclidean/<pt>_support_tier1.csv`.

Source: 3D Euclidean pool over fsaverage pial vertices, sigma=1.3 mm, radius=5.0 mm, crown_bias=off, atlas=fsaverage_bake_v2d/smoothed, sliced to 15 `DEFAULT_BASE_PARCELS` columns (raw [0, 100] BNA probability).


## Tier-1 column order

- `support_A4hf` <- `A4hf`, BNA index 53
- `support_A1_2_3ulhf` <- `A1/2/3ulhf`, BNA index 155
- `support_A1_2_3tonIa` <- `A1/2/3tonIa`, BNA index 157
- `support_A4tl` <- `A4tl`, BNA index 61
- `support_IFJ` <- `IFJ`, BNA index 17
- `support_A44d` <- `A44d`, BNA index 29
- `support_A45c` <- `A45c`, BNA index 33
- `support_A6cvl` <- `A6cvl`, BNA index 63
- `support_A9_46v` <- `A9/46v`, BNA index 21
- `support_A40rd` <- `A40rd`, BNA index 139
- `support_IFS` <- `IFS`, BNA index 31
- `support_A2` <- `A2`, BNA index 159
- `support_A40rv` <- `A40rv`, BNA index 145
- `support_A44v` <- `A44v`, BNA index 39
- `support_A22r` <- `A22r`, BNA index 79

## Per-patient summary

| patient | n_elec | argmax_in_tier1 | low_tier1_mass (<1) | max [0,1) | max [1,10) | max [10,25) | max [25,50) | max [50,75) | max [75,100] |
|---|---|---|---|---|---|---|---|---|---|
| S14 | 128 | 128 | 0 | 0 | 0 | 14 | 78 | 36 | 0 |
| S16 | 128 | 128 | 0 | 0 | 0 | 15 | 98 | 15 | 0 |
| S23 | 128 | 128 | 0 | 0 | 0 | 20 | 76 | 18 | 14 |
| S26 | 128 | 128 | 0 | 0 | 0 | 0 | 68 | 27 | 33 |
| S33 | 256 | 256 | 0 | 0 | 0 | 2 | 172 | 66 | 16 |
| S39 | 256 | 256 | 0 | 0 | 0 | 21 | 145 | 90 | 0 |
| S62 | 256 | 256 | 0 | 0 | 0 | 0 | 123 | 110 | 23 |
