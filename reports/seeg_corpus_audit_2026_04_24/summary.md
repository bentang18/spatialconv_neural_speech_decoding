# A3 — sEEG D-cohort continuous corpus audit (2026-04-24)

JSON-sidecar-first scan of all 4 speech-task BIDS roots on Box.
Reads `sub-D*/ieeg/*_ieeg.edf` (non-recursive → skips `practice/`).
Extracted duration from BIDS `*_ieeg.json` for 674 runs; fell back to EDF header for 0; 0 failures.

## Per-task totals

| task | n_patients | hours |
|---|---:|---:|
| PS | 50 | 40.73 |
| LexDelay | 52 | 65.87 |
| LexNoDelay | 26 | 31.68 |
| SentenceRep | 34 | 42.31 |

**Grand total**: 180.59 h across **87** unique D-patients (union over tasks).

For comparison (uECoG — see `docs/references/data_reference.md`): 2.83 h PS + 3.96 h lexical = 6.79 h / 27 unique S-patients.

## Top 10 D-patients by total sEEG hours

| bids_id | total_hours | tasks |
|---|---:|---|
| sub-D0022 | 6.09 | PS,SentenceRep |
| sub-D0024 | 5.75 | LexDelay,LexNoDelay,PS,SentenceRep |
| sub-D0071 | 5.05 | LexDelay,LexNoDelay,PS,SentenceRep |
| sub-D0029 | 4.96 | LexDelay,LexNoDelay,PS,SentenceRep |
| sub-D0057 | 4.62 | LexDelay,LexNoDelay,PS,SentenceRep |
| sub-D0023 | 4.56 | LexDelay,PS,SentenceRep |
| sub-D0079 | 4.05 | LexDelay,PS |
| sub-D0059 | 4.0 | LexDelay,PS,SentenceRep |
| sub-D0053 | 3.98 | LexDelay,LexNoDelay,PS,SentenceRep |
| sub-D0028 | 3.91 | LexDelay,LexNoDelay,PS,SentenceRep |
