# FE filterbank sweep — findings (2026-06-03)

**Question.** Does v14's Multi-STFT log-octave (constant-Q) filterbank front end beat
iMINDBench's raw-STFT-bin front end for a frozen-feature **linear** probe, and does the
within-vs-cross AUROC delta support the broadband-power prior (filterbank should help
cross-subject *more*)?

**Verdict. No.** The constant-Q filterbank does not help, and on cross-subject it
**significantly hurts** — concentrated in exactly the speech/onset tasks where the
leaderboard margin lives. The broadband-power-helps-cross hypothesis is rejected.

## Design

- Control = **cell 0** = `raw_multi_stft` (iMINDBench-style raw rfft bins, 75 bins/ch).
- Cells 1–7 = log-octave constant-Q `multi_stft` filterbanks sweeping {f0, high-cap,
  octave_step, half_bw}. Cell G = within-only grid anchor.
- Probe = frozen-feature `LogisticRegression` (L2, lbfgs, C=1.0, class-weight balanced),
  `roc_auc_score` per (task, fold).
- Norm = `per_session_robust_mad_cf` (per-(C,F) median/MAD, L.1.N9). Ref = shaft Laplacian / CAR.
- Cross-subject = train sub2/trial4 → test held-out subject; **intersection mean-pool onto
  shared DK parcels** (`combine_regions`, upstream "pairwise"). Within = per-electrode, 2-fold CV.
- 15 Neuroprobe tasks; RANK4 = {speech, onset, volume, pitch}.
- 188 jobs, `common` CPU partition, 0 failures. Artifacts: `_per_task.csv` (4440 rows),
  `_summary.csv` (17 cell×eval rows).

## Paired significance (paired t-test on test_roc_auc, paired by subj×trial×task×fold)

**CROSS — every constant-Q cell is significantly below raw.**

| cell | RANK4 Δ | p | ALL-15 Δ | p |
|---|---|---|---|---|
| 1 (const-Q default) | −0.0168 | 1.4e-4 *** | −0.0063 | 7.3e-4 *** |
| 2 (f0=4) | −0.0135 | 1.0e-3 ** | −0.0060 | 1.1e-3 ** |
| 3 (cap=128) | −0.0189 | 1.3e-4 *** | −0.0045 | 1.8e-2 * |
| 4 | −0.0188 | 1.6e-5 *** | −0.0084 | 1.2e-5 *** |
| 5 | −0.0146 | 6.0e-5 *** | −0.0073 | 7.6e-6 *** |
| 6 | −0.0141 | 1.7e-3 ** | −0.0065 | 6.4e-4 *** |
| 7 | −0.0140 | 1.6e-3 ** | −0.0055 | 4.1e-3 ** |

**WITHIN — a wash.** Only cell 2 shows a within gain (RANK4 +0.0148, p=2.5e-4) and it does
**not** transfer cross (cell 2 cross RANK4 −0.0135). Cells 1/3/4/5 within ≈ 0 (n.s.);
cells 6/7/G slightly negative.

**CROSS per-task, cell 1 vs cell 0 — damage is concentrated in speech-coupled tasks.**

| task | raw | const-Q | Δ | p |
|---|---|---|---|---|
| onset | 0.708 | 0.676 | −0.032 | 0.001 |
| speech | 0.700 | 0.673 | −0.028 | 0.010 |
| word_index | 0.609 | 0.596 | −0.013 | 0.149 |
| volume | 0.606 | 0.596 | −0.010 | 0.106 |

Every other task sits near chance (~0.50) where raw vs const-Q is a coin flip.

## Why (directional)

1. **Linear-subspace argument (structural).** For a *linear* probe, the 75 raw rfft bins
   already span every decision the 15 octave bins can express — a fixed filterbank is a
   linear projection, so it can only *lose* rank, never add. The only way it helps a linear
   probe is as a variance-reduction regularizer. The sweep shows that route does not pay off
   cross-subject; it costs discriminative high-gamma structure instead. **This biases the
   sweep toward raw** — a *deep* encoder can exploit a filterbank prior a linear probe can't
   see, so this result bounds the linear-probe regime, not v14's pretrained encoder.
2. **High-gamma smearing.** Constant-Q triangular bins widen with frequency, averaging over
   the 70–150 Hz broadband-power band that carries onset/speech. Raw bins keep that structure
   resolved. Hence the loss lands exactly on onset/speech.

## Implication for v14

- **Does not unseat the Multi-STFT front end for the pretrained encoder.** The bias above is
  load-bearing: a linear probe cannot reward a filterbank prior, so a within-margin "wash" +
  a cross-subject deficit is the *expected* null for this probe even if the filterbank helps
  a deep model. This sweep rules out "filterbank is a free linear win" — it does **not** rule
  out filterbank-for-pretraining.
- **It does sharpen the front-end default question.** If the encoder front end is ever
  reduced toward linear behavior (e.g. a shallow patch embed), prefer raw bins / a wider,
  finer high-gamma resolution over aggressive constant-Q compression.
- Onset/speech are where the entire leaderboard margin lives (submission gate). Any front-end
  choice that smears high-gamma is penalized there first — keep high-gamma resolution explicit.

## Artifacts
- `fe_filterbank_sweep_2026_06_03_per_task.csv` — per (cell, eval, subject, task, fold).
- `fe_filterbank_sweep_2026_06_03_summary.csv` — per (cell, eval): rank4/all15 mean + Δ-vs-ctrl.
- Paired test: `$CLAUDE_JOB_DIR/tmp/paired_test.py` (reproducible from per_task.csv + scipy).
