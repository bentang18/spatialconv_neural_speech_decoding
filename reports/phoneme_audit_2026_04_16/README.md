# `#34` phoneme-loading + trial-timing audit — rollup

Patients found: S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62.

## Verdict table

| Patient | Verdict | Tier | Silent-suspect | Audio offset (s) | MAD (ms) |
|---------|---------|------|----------------|------------------|----------|
| S14 | **pass** | core | 0/153 | 1.281 | 115.3 |
| S26 | **warn** | core | 0/153 | 1.278 | 110.5 |
| S33 | **pass** | core | 0/52 | 1.265 | 107.1 |
| S62 | **pass** | core | 0/191 | 1.283 | 117.4 |
| S16 | **pass** | extended | 0/205 | 1.280 | 115.2 |
| S23 | **pass** | extended | 0/156 | 1.299 | 115.2 |
| S39 | **warn** | extended | — | — | — |

## Per-check pass/fail matrix

| Check | S14 | S16 | S22 | S23 | S26 | S32 | S33 | S39 | S57 | S58 | S62 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `fif_path_exists` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `fif_window_contains_target` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `event_id_is_9_ps_phonemes` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `per_trial_token_reconstruction` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `signal_finite` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `fif_labels_match_events_tsv` | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `events_tsv_tokens_in_ps` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `fif_samples_match_events_tsv` | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `fif_t0_matches_events_tsv_onset` | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `events_stale_is_divergent` | ✓ | — | — | — | — | — | — | — | — | — | — |
| `audio_neural_clock_fit_exists` | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ |
| `audio_neural_clock_consistent` | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | ✓ |
| `silent_trial_fraction_low` | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | ✓ |

## Closure rule

**`#34` CLOSED**

- S14: pass
- S26: warn
- S33: pass
- S62: pass

## Artifacts per patient

- `{patient}.json` — full audit result
- `{patient}_exclusion_candidates.csv` — silent-suspect trials to drop at loader time
- `plots/{patient}_audio_alignment.png` — audio RMS envelope + per-trial offset distribution
