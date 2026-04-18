# Coordinate-bridge verification (blocker #12)

PASS: every retained `.fif` channel resolves to exactly one physical name, uniquely present in both the fsaverage coord cache and the Tier-1 support cache, with `(r, c)` inside the declared grid shape.

SKIP: `.fif` not available locally; re-runs on DCC in Phase E2.


## Summary

| patient | verdict | n_fif | n_bads | n_kept | grid_shape | notes |
|---|---|---|---|---|---|---|
| S14 | PASS | 128 | 0 | 128 | 8×16 | — |
| S26 | SKIP | 0 | 0 | 0 | 8×16 | .fif missing locally: /Users/bentang/Documents/Code/speech/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS/derivatives/epoch(CAR)/sub-S26/epoch(band)(power)/sub-S26_task-PhonemeSequence_desc-productionZscore_highgamma.fif |
| S33 | SKIP | 0 | 0 | 0 | 12×22 | .fif missing locally: /Users/bentang/Documents/Code/speech/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS/derivatives/epoch(CAR)/sub-S33/epoch(band)(power)/sub-S33_task-PhonemeSequence_desc-productionZscore_highgamma.fif |
| S62 | SKIP | 0 | 0 | 0 | 12×22 | .fif missing locally: /Users/bentang/Documents/Code/speech/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS/derivatives/epoch(CAR)/sub-S62/epoch(band)(power)/sub-S62_task-PhonemeSequence_desc-productionZscore_highgamma.fif |
