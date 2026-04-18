# Slack draft for Zac — morning 2026-04-17

Hey Zac — thanks for regenerating the trial-level fifs! Got them off Box.
Two questions on the new files (using S62 as ref since the others are still
uploading):

**1. Z-score + decimation?** S62's new `production_highGamma.fif` is at 2 kHz
raw (chan 0: mean≈1.07, std≈0.31), vs S14's old `productionZscore_highgamma.fif`
at 200 Hz z-scored (mean≈0, std≈1). My loader contract expects the fully
preprocessed version. Happy to run the z-score + decimate downstream myself if
that's easier — just want to confirm you meant to skip that step, or if it's
coming in a follow-up.

**2. Filename convention.** New files are `sub-S62_task-phoneme_desc-production_highGamma.fif`
vs the old `sub-S14_task-PhonemeSequence_desc-productionZscore_highgamma.fif`
(`task-phoneme` vs `task-PhonemeSequence`, `production` vs `productionZscore`,
case change on `highGamma`). Is the new naming intentional going forward, or
just a quirk of the regen? Want to know whether to update the loader or wait
for a re-drop.

No rush — S14 is running end-to-end on DCC now on the v2c atlas, so I'm ready
to scale the moment the other 6 patients land (S16/S22/S23/S26/S33/S39/S62).
