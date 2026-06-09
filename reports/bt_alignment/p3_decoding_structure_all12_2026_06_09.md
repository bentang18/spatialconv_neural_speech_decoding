# P3 decoding structure — all 12 distill sessions

Date: 2026-06-09. Probe: `$CLAUDE_JOB_DIR/tmp/probe_goal_p3.py` (job 48006143).
Source: `/work/ht203/viz_p3_parity/p3_goal.json`.

Answers the governing question with the bar set to **all 12 P3 distill sessions**:
optimal/highest-decoding parcels, optimal lag, per-electrode-residual power over
the parcel mean, and frequency-band localization.

## Method

- **Coverage**: all 12 `V14_P3_DISTILL_SESSIONS` matched, 0 missing. 6 subjects
  {1,2,3,4,6,9}; 150-clip/session cap; 1,800 unique verbal clips. Signal scales
  with clip count: subj 2 (370 clips) cleanest, subj 9 (94) underpowered.
- **Target**: Whisper all-layer-mean AC component (within-clip temporal residual),
  the predictable low-SNR part.
- **Metric**: Pearson r (robust to scale-overfit), economy-SVD primal ridge,
  4-fold, per-target GCV. Genuineness gate = phase-randomization null (z ≥ 3.0).
- **Window** 2 frames (250 ms) at 8 Hz latent (125 ms/frame); **lags** −500…+500 ms.
- **Bands** = the F=50 raw |STFT| split into theta(4–8), alpha/beta(8–30),
  low-γ(30–70), high-γ(70–150), VHF(150–192).

## Q1 — which parcels genuinely decode

**L superior-temporal gyrus is the dominant, most reproducible decoder** —
genuine in 4/6 subjects, mean r-above-null 0.11. Consensus auditory-speech network
(genuine in ≥2 subjects):

| parcel | n subj | mean r>null |
|---|---|---|
| ctx-lh-superiortemporal | 4 | 0.112 |
| ctx-lh-insula | 3 | 0.098 |
| ctx-rh-superiortemporal | 2 | 0.129 |
| ctx-rh-insula | 2 | 0.097 |
| ctx-lh-bankssts (STS) | 2 | 0.078 |
| ctx-lh-supramarginal | 2 | 0.069 |
| ctx-lh-middletemporal | 2 | 0.051 |

Single-subject high-r hits are also auditory/sensorimotor: lh-transversetemporal
(Heschl's) 0.178, rh-postcentral 0.137, rh-transversetemporal 0.102. Subject 9
(94 clips) had no genuine parcels — underpowered, not contradictory.

The localization is anatomically exactly right (bilateral STG, Heschl's, STS,
insula, supramarginal). **The P3 target is real, correctly localized neural
signal — not an artifact or alignment error.**

## Q2 — optimal lag

**Consensus peak +125 ms** (neural leads the audio target). Lag curve rises
smoothly −500→+125 ms then declines (r: 0.041 → 0.090 → 0.077). Per-subject peaks
125–250 ms for the 5 powered subjects (subj 9 the −500 ms outlier). Matches the
prior 4-session +125–375 ms read. Physiological.

## Q3 — does per-electrode residual add over the parcel mean, and by how much

**Yes, a small but real increment, concentrated in auditory cortex.**

- Per parcel-test (24 subject×top-parcel): mean Δr = **+0.0044**, median +0.0063,
  significant in **11/24**.
- Pooling all electrodes per subject: mean Δr = **+0.0097** (5/6 positive,
  +0.010…+0.022; one −0.009).
- Biggest gains in STG/insula: lh-STG up to **+0.046**, rh-STG +0.018,
  insula +0.010…+0.031.

So the residual is **not pure nuisance** — there is reproducible sub-parcel
structure, but the magnitude is modest (typically a few % relative to the
parcel-mean r ~0.11; up to ~40% in the best lh-STG case). This refines the
earlier 4-session R²≈0 "nuisance" read: with more data and the robust r-metric,
the residual carries a small genuine decoding increment localized to STG/insula.

## Frequency bands — not band-localized

Mean r-above-null is **broadband across ~8–150 Hz**, high-γ only marginally
leading:

| band | mean r>null |
|---|---|
| high-γ 70–150 | 0.069 |
| alpha/beta 8–30 | 0.067 |
| low-γ 30–70 | 0.060 |
| theta 4–8 | 0.029 |
| VHF 150–192 | 0.029 |

Per-subject best-band votes split evenly across low-γ / alpha-beta / VHF — no
single dominant band. **Whisper-decodability is not a high-gamma-only
phenomenon.**

## Bottom line

The P3 distillation target is decodable from neural data, correctly localized to
auditory cortex, at a physiological +125 ms lag, broadband in frequency. P3-not-
training is **not** a signal, localization, or alignment problem. The signal is
real but low-SNR (peak r ~0.09–0.13), consistent with the regression-objective-
in-low-SNR diagnosis — motivating the contrastive + clip-level pivot.
