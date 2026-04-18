# Tier-1 support cache QC report

Per-patient summary of `data/atlas/support_cache/<pt>_support_tier1.csv`.

Source: fsaverage snap-to-pial + `fsaverage_bake_fast2/smoothed`, sliced to 15 `DEFAULT_BASE_PARCELS` columns (raw [0, 100] BNA probability).


## Tier-1 column order

- `support_A4hf` ← `A4hf`, BNA index 53
- `support_A1_2_3ulhf` ← `A1/2/3ulhf` (sanitized to `support_A1_2_3ulhf`), BNA index 155
- `support_A1_2_3tonIa` ← `A1/2/3tonIa` (sanitized to `support_A1_2_3tonIa`), BNA index 157
- `support_A4tl` ← `A4tl`, BNA index 61
- `support_IFJ` ← `IFJ`, BNA index 17
- `support_A44d` ← `A44d`, BNA index 29
- `support_A45c` ← `A45c`, BNA index 33
- `support_A6cvl` ← `A6cvl`, BNA index 63
- `support_A9_46v` ← `A9/46v` (sanitized to `support_A9_46v`), BNA index 21
- `support_A40rd` ← `A40rd`, BNA index 139
- `support_IFS` ← `IFS`, BNA index 31
- `support_A2` ← `A2`, BNA index 159
- `support_A40rv` ← `A40rv`, BNA index 145
- `support_A44v` ← `A44v`, BNA index 39
- `support_A22r` ← `A22r`, BNA index 79

## Per-patient summary

| patient | n_elec | argmax_in_tier1 | low_tier1_mass (<1) | max [0,1) | max [1,10) | max [10,25) | max [25,50) | max [50,75) | max [75,100] |
|---|---|---|---|---|---|---|---|---|---|
| S14 | 128 | 128 | 0 | 0 | 0 | 29 | 60 | 32 | 7 |
| S16 | 128 | 128 | 0 | 0 | 0 | 16 | 95 | 17 | 0 |
| S23 | 128 | 128 | 0 | 0 | 0 | 17 | 56 | 40 | 15 |
| S26 | 128 | 128 | 0 | 0 | 0 | 0 | 61 | 34 | 33 |
| S33 | 256 | 256 | 0 | 0 | 0 | 4 | 159 | 71 | 22 |
| S39 | 256 | 256 | 0 | 0 | 0 | 22 | 128 | 101 | 5 |
| S62 | 256 | 256 | 0 | 0 | 0 | 3 | 100 | 120 | 33 |
