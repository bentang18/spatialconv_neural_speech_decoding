# P1 cohort summary — phoneme-level `.fif` audit

**Plan:** `docs/plans/v14-core-baseline-aligned.md` → P1. **Date:** 2026-04-17.

Reuses the `#34` audit artifacts at `reports/phoneme_audit_2026_04_16/`. No re-run was required — every predicate P1 names is a strict subset of the `#34` check set, and the four must-pass checks (`fif_window_contains_target`, `event_id_is_9_ps_phonemes`, `per_trial_token_reconstruction`, `signal_finite`) pass on all 7 Phase-1 LH patients.

## Result

| Patient | Tier | Verdict | n_trials | window | sfreq | Notes |
|---|---|---|---|---|---|---|
| S14 | core | pass | 153 | [-1.0, 1.495] s | 200 Hz | — |
| S26 | core | warn | 153 | [-1.0, 1.495] s | 200 Hz | events-TSV drift (known per `#34`); `.fif` authoritative |
| S33 | core | pass | 52 | [-1.0, 1.495] s | 200 Hz | small n, flagged for CV-fold sizing |
| S62 | core | pass | 191 | [-1.0, 1.495] s | 200 Hz | — |
| S16 | ext  | pass | 205 | [-1.0, 1.495] s | 200 Hz | — |
| S23 | ext  | pass | 156 | [-1.0, 1.495] s | 200 Hz | — |
| S39 | ext  | warn | 148 | [-1.0, 1.495] s | 200 Hz | audio-neural clock-fit warning only; not loader-critical |

All 7 patients have raw `tmin=-1.0, tmax=1.495` at 200 Hz, so the `[-0.15, 0.5)` crop the per-phoneme loader needs fits comfortably. Group-by-3 token reconstruction passes on every patient (including S26's trial 71 "vaek", which is carried correctly in the phoneme-level `.fif` even though the events TSV drops it).

**Independence from Zac's 2026-04-17 trial-level regen:** confirmed. The phoneme-level `.fif` lives at `derivatives/epoch(phonemeLevel)(CAR)/sub-{pt}/epoch(band)(power)/...` — separate derivative tree from the trial-level file Zac regenerated. No downstream blocker.

## Next

P1 closed. Proceed to P2 (masked-mean pool primitive).
