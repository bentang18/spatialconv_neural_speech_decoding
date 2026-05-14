# BT shaft/depth geometry contract

**Date**: 2026-05-13
**Audit script**: `scripts/neuroprobe/audit_shaft_geometry.py`
**BT root**: `/work/ht203/data/braintreebank`
**Cohort**: full BT minus S5 (DK-first-pass freeze 2026-05-13). Subjects = [1, 2, 3, 4, 6, 7, 8, 9, 10].

This contract freezes:
1. the canonical shaft parser,
2. the contact-order orientation policy,
3. which shaft/depth features are admissible v14 inputs.

## Headline

- **Contacts audited**: 1549 (17 trigger, 115 corrupted, 0 parse-failed).
- **Shafts parsed**: 128.
- **Multi-label shafts**: 119 of 128 (93.0%); these contain 1466 contacts (95.7% of all parsed contacts). One shaft commonly traverses multiple DK regions — confirms that DK alone discards within-shaft position.
- **Cross-hemisphere shafts**: 0. DK-first-pass audit expects 0/140; deviation = parser bug.
- **Linear shafts** (PCA PC1 var-ratio ≥ 0.97): 99.2% of orientation-audited shafts.
- **Suffix-monotonic shafts** (|r(suffix, PC1)| ≥ 0.98): 98.4%.

## Frozen decisions

### 1. Canonical shaft parser
`parse_shaft(channel_name)` (regex `^(.*?)(\d+)$`) defined at
`src/speech_decoding/extractors/reference.py:31`. Splits trailing integer
suffix as `contact_index` and preceding stem as `shaft`. Labels without a
numeric suffix → `(name, None)`; the audit reports them as
`parse_status="no_numeric_suffix"` and excludes them from shaft groupings.
Trigger channels (`DC*`, `TRIG*`) parse normally but are flagged
`is_trigger=True` and excluded from all reference / depth computations.

### 2. Contact-order orientation
Per-shaft principal axis (SVD of mean-centered (L, P, I) native coordinates)
gives PC1; `r(contact_index, PC1)` measures whether the integer suffix is a
monotonic projection of physical position. Sign of `r` tells the suffix→depth
direction *up to sign*.

Per-shaft `deeper_end` is computed by the per-subject-centroid heuristic
(contacts farther from the cohort-subject centroid are taken to be closer to
the skull entry point). Cohort breakdown: lower_suffix=96 (75.0%); higher_suffix=18 (14.1%); unknown=14 (10.9%).

### 3. Signed-depth-feature policy
**FORBIDDEN by default** — sign convention is not cohort-uniform (75.0% of shafts agree on dominant orientation `lower_suffix`; threshold for admission is ≥95%). Within-subject consistency ranges over the table below. Without a per-shaft sign verifier (would need brain-entry-point coords from Chris MNI), any signed-depth feature silently flips on the minority shafts and leaks patient identity. v14 must use orientation-invariant within-shaft features only:

- ordinal `contact_index` ✗ disallowed as a feature (subject-specific magnitude AND sign-ambiguous).
- normalized `contact_index / max_index` ✗ disallowed (sign-ambiguous).
- centered normalized position `2 * (i - 0.5*(max+min)) / range` ✗ disallowed (sign-ambiguous).
- same-shaft adjacency mask ✓ allowed.
- relative offset `|i - j|` on same shaft ✓ allowed.
- local-reference provenance metadata ✓ allowed (shaft-CAR / Laplacian indices).

### 4. Reference scheme compatibility
- **shaftCAR (R2, v14 default)**: orientation-INDEPENDENT. Uses `parse_shaft`
  stem grouping only. Cleared regardless of orientation outcome.
- **shaftLaplacian (R4, upstream Linear-Lap+spec parity)**: orientation-
  INDEPENDENT in BrainBERT's symmetric form (`S′ᵢ = Sᵢ − ½(Sᵢ₋₁ + Sᵢ₊₁)`).
  Boundary contacts fall back to one-sided difference (`S′ᵢ = Sᵢ − Sᵢ₋₁` or
  `Sᵢ − Sᵢ₊₁`) — the sign asymmetry is absorbed in spectral magnitude.
  Cleared.
- **Bipolar (R3)**: orientation-INDEPENDENT for inputs but creates virtual
  midpoint channels with interpolated coords. Already rejected for v14 on
  architectural grounds (parcel `support[i,p]` ill-defined at midpoints).
- **Signed depth embeddings**: gated on §3 above.

### 5. Rejected encodings
- **Learned `shaft_id` embedding**: rejected. `shaft` strings are patient-
  specific (`OFa12` on sub_1 ≠ `OFa12` on sub_2). A learned embedding would
  pattern-match subject identity. Permitted only as grouping key for local
  reference construction, never as a feature.
- **Raw `contact_index` as a numeric feature**: rejected. Sign-ambiguous and
  subject-specific magnitude.
- **`hemisphere` as a one-hot input feature**: rejected as a v14 default; the
  Graphormer-bias / parcel-id-tagged latents carry hemisphere via DK label
  scope (`lh_*` / `rh_*` parcels). May be revisited as a P2 ablation cell.

## Per-subject orientation summary

|   subject_id |   n_shafts_audited | majority_deeper_end   |   majority_consistency |
|-------------:|-------------------:|:----------------------|-----------------------:|
|            1 |                 13 | lower_suffix          |                  0.615 |
|            2 |                 16 | lower_suffix          |                  1     |
|            3 |                 12 | lower_suffix          |                  0.667 |
|            4 |                 15 | lower_suffix          |                  0.933 |
|            6 |                 12 | lower_suffix          |                  0.5   |
|            7 |                 18 | lower_suffix          |                  0.889 |
|            8 |                 13 | lower_suffix          |                  0.692 |
|            9 |                 12 | lower_suffix          |                  0.583 |
|           10 |                 17 | lower_suffix          |                  0.706 |

## Anomalies

- Parse anomalies: `parse_anomalies.csv` (0 rows).
- Subjects: [1, 2, 3, 4, 6, 7, 8, 9, 10].
- Triggers excluded (informational, count above).

## Files

- `shaft_contact_inventory.csv` — one row per (subject, raw label).
- `shaft_label_transition_summary.csv` — one row per (subject, shaft).
- `contact_order_orientation_audit.csv` — one row per (subject, shaft) with ≥3 contacts.
- `parse_anomalies.csv` — labels parse_shaft cannot resolve (incl. triggers and missing-anatomy rows).

## Unfreeze triggers

- Chris MNI lands → re-run orientation audit against true brain-surface skull
  intersection; signed-depth policy may relax.
- BNA subcortical scope decision (parked blocker, 2026-05-12) → if v14 extends
  beyond DK Tier-1, re-validate non-cortical shafts (hippocampus depth probes
  with single-shaft trajectories).
- Subject pool changes → re-run; per-subject orientation consistency is
  load-bearing.
