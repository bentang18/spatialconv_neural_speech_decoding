# A1 — sEEG D-patient sig-channel inventory (2026-04-24)

Reads pre-computed cluster-corrected permutation-test masks from 
`stats/` (Cogan Lab upstream pipeline).
`n_sig_<phase>` = count of channels with any significant timepoint in that phase.

## Coverage

- PS D-patients in cohort audit: 50
- PS D-patients with complete stats: 43
- Missing stats (flagged 'unknown' usability): 7 — D0019, D0022, D0023, D0025, D0031, D0035, D0076

## Usability-proxy tiering (response-phase)

Rule: frac_sig_resp ≥ 20% → `ok`; 5–20% → `low`; < 5% → `likely_bad`.
First-pass proxy only — Nanlin's authoritative tiering overrides.

- ok: **40**
- low: 1
- likely_bad: 2

## Top 10 PS D-patients by production-phase sig channels (`n_sig_resp`)

| bids_id | n_ch_stats | lh_tier1 | rh_tier1 | n_sig_aud | n_sig_resp | frac_resp |
|---|---:|---:|---:|---:|---:|---:|
| D0041 | 201 | 10 | 13 | 69 | 149 | 0.7413 |
| D0084 | 150 | 0 | 16 | 27 | 137 | 0.9133 |
| D0061 | 204 | 11 | 6 | 87 | 129 | 0.6324 |
| D0057 | 141 | 0 | 10 | 66 | 123 | 0.8723 |
| D0103 | 186 | 7 | 0 | 94 | 114 | 0.6129 |
| D0060 | 150 | 9 | 8 | 95 | 113 | 0.7533 |
| D0088 | 143 | 7 | 0 | 71 | 111 | 0.7762 |
| D0054 | 146 | 0 | 19 | 16 | 105 | 0.7192 |
| D0052 | 128 | 8 | 0 | 35 | 100 | 0.7812 |
| D0049 | 161 | 11 | 0 | 43 | 97 | 0.6025 |

## Top 10 PS D-patients by auditory-phase sig channels (`n_sig_aud`)

| bids_id | n_ch_stats | lh_tier1 | rh_tier1 | n_sig_aud | n_sig_resp | frac_aud |
|---|---:|---:|---:|---:|---:|---:|
| D0086 | 224 | 1 | 3 | 102 | 74 | 0.4554 |
| D0060 | 150 | 9 | 8 | 95 | 113 | 0.6333 |
| D0103 | 186 | 7 | 0 | 94 | 114 | 0.5054 |
| D0061 | 204 | 11 | 6 | 87 | 129 | 0.4265 |
| D0093 | 100 | 22 | 0 | 74 | 97 | 0.74 |
| D0068 | 115 | 15 | 0 | 73 | 78 | 0.6348 |
| D0088 | 143 | 7 | 0 | 71 | 111 | 0.4965 |
| D0096 | 184 | 25 | 0 | 70 | 88 | 0.3804 |
| D0041 | 201 | 10 | 13 | 69 | 149 | 0.3433 |
| D0075 | 127 | 0 | 28 | 67 | 97 | 0.5276 |
