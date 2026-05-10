# V0.x — CrossSession stimulus-overlap audit (upper-bound)

Generated: 2026-05-10 18:44
Sessions: 12 BT Lite (subject, trial) pairs.
Mode: `upper-bound` (unique-word overlap across full per-trial words_df; upper bound)

## Per-task summary (median across subjects)

| task | median | min | max | n_pairs |
|---|---|---|---|---|
| ALL_WORDS_UPPER_BOUND | 0.414 | 0.323 | 0.450 | 12 |

## Flagged (overlap > 0.50)

No (subject, task) pairs exceed 50% overlap.

## How to read

`overlap_fraction = |unique(test_words) ∩ unique(train_words)| / |unique(test_words)|`. If 1.0, every test word also appears in train and the linear classifier can match stimulus identity rather than brain response. CrossSession protocol shuffles train/test by session, not by stimulus; movie repeats can re-expose words across sessions.

**v14 contract**: tasks with median overlap > 0.50 must be flagged in stage_0.md V6 as stimulus-recognition-confounded. L.2/L.3 winners must report numbers separately for flagged vs unflagged tasks.
