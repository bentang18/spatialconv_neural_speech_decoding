# `#34` phoneme-loading + trial-timing audit — rollup

Patients found: S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62.

## Verdict table

| Patient | Verdict | Tier | Silent-suspect | Audio offset (s) | MAD (ms) |
|---------|---------|------|----------------|------------------|----------|
| S14 | **pass** | core | 0/153 | 1.281 | 115.3 |
| S26 | **fail** | core | 0/153 | 1.278 | 110.5 |
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
| `fif_labels_match_authoritative` | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `no_leaked_tokens` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `signal_finite` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `events_stale_is_divergent` | ✓ | — | — | — | — | — | — | — | — | — | — |
| `fif_samples_match_authoritative` | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `epoch_t0_equals_response_onset` | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `audio_neural_clock_fit_exists` | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ |
| `audio_neural_clock_consistent` | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | ✓ |
| `silent_trial_fraction_low` | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | ✓ |

## Closure rule

**`#34` still OPEN**

- S14: pass
- S26: fail
- S33: pass
- S62: pass

## Artifacts per patient

- `{patient}.json` — full audit result
- `{patient}_exclusion_candidates.csv` — silent-suspect trials to drop at loader time
- `plots/{patient}_audio_alignment.png` — audio RMS envelope + per-trial offset distribution
