# `#34` phoneme-loading + trial-timing audit — rollup

Patients found: S14.

## Verdict table

| Patient | Verdict | Tier | Silent-suspect | Audio offset (s) | MAD (ms) |
|---------|---------|------|----------------|------------------|----------|
| S14 | **pass** | core | 0/153 | 1.281 | 115.3 |
| S26 | _(not run)_ | core | — | — | — |
| S33 | _(not run)_ | core | — | — | — |
| S62 | _(not run)_ | core | — | — | — |
| S16 | _(not run)_ | extended | — | — | — |
| S23 | _(not run)_ | extended | — | — | — |
| S39 | _(not run)_ | extended | — | — | — |

## Per-check pass/fail matrix

| Check | S14 |
|---|---|
| `fif_path_exists` | ✓ |
| `fif_window_contains_target` | ✓ |
| `event_id_is_52_ps_tokens` | ✓ |
| `per_epoch_token_decomposition` | ✓ |
| `fif_labels_match_authoritative` | ✓ |
| `no_leaked_tokens` | ✓ |
| `signal_finite` | ✓ |
| `events_stale_is_divergent` | ✓ |
| `fif_samples_match_authoritative` | ✓ |
| `epoch_t0_equals_response_onset` | ✓ |
| `audio_neural_clock_fit_exists` | ✓ |
| `audio_neural_clock_consistent` | ✓ |
| `silent_trial_fraction_low` | ✓ |

## Closure rule

**`#34` still OPEN**

- S14: pass
- S26: NOT RUN
- S33: NOT RUN
- S62: NOT RUN

## Artifacts per patient

- `{patient}.json` — full audit result
- `{patient}_exclusion_candidates.csv` — silent-suspect trials to drop at loader time
- `plots/{patient}_audio_alignment.png` — audio RMS envelope + per-trial offset distribution
