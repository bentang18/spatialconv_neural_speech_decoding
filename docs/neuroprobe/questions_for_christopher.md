# Questions for Christopher Wang

Running list. Pruned and reordered before each call. Last refreshed 2026-05-05.

Next contact: ping 2026-05-08, request a chat.

## A — fsaverage / surface mapping (original ask)

- What's the timeline on a per-electrode `fsaverage` table (subject, electrode, hemi, vertex index, surface RAS)?
- Is there an internal pre-projection step that produces fsaverage-coordinate intermediates before the public 2-D plotting overlay? Even unreleased internal artifacts would unblock us.
- Would you accept a community-contributed mapping if we derived one from public anatomy + plotting coords + (where available) per-subject FreeSurfer-style intermediates? What would make such a contribution land vs. not?
- Is there any provenance attached to `elec_coords_full.csv` we can use? E.g., the projection script, the snap-distance threshold, the hemisphere assignment rule.
- The braintreebank.dev quickstart notebook — does any cell in it touch a latent fsaverage step we missed in the public files? (Will check the notebook before the call regardless.)

## B — New linear-decoding NeurIPS submission

Goal: pin down **every single step from raw voltage to model input** in the new linear baseline. Every choice below maps to one of our L-sweep axes — his answers tell us whether our L sweeps are racing him to the same answer or asking different questions. The strategic angle is the last bullet (question 6); the rest is forensic.

### B.1 — Headline + scope
- Headline numbers: scope (which tasks, which split, multiclass vs. binary)?
- Does it move the published 0.611 multiclass cross-session bar? If so, by how much? What's the new floor we should target?
- Will a preprint or rebuttal-style code release accompany the submission? Timeline?

### B.2 — Pre-model pipeline (every step before data touches the classifier)
Walk us through the full chain on the new baseline, end to end:

1. **Raw voltage source** — which sessions, which subjects, Lite or Full, any data not in public Lite (Full sessions, additional features, external corpora)?
2. **Bad-channel / bad-segment exclusion** — what gets dropped and on what criterion (impedance, variance, line-noise, manual)? Done before or after referencing?
3. **White-matter / out-of-brain contact handling** — excluded from channel set entirely? Excluded only from features but included in reference math? Included throughout?
4. **Reference** — raw monopolar, global CAR, bipolar, shaft-CAR, shaft-Laplacian, or something else? Did the new baseline change reference vs. `c7b955b`? If reference uses an average, are WM contacts in or out of that average?
5. **Filtering** — notch (which frequencies, filter type, order)? High-pass (cutoff, order)? Anti-alias / low-pass? Any of these new vs. `c7b955b`?
6. **Resampling / decimation** — sample rate at the model input?
7. **Window anchor + length** — onset-locked at `[0, 1]s`, or moved off that? Any pre-window context?
8. **Feature extraction (input view)** — raw voltage, Laplacian, spectral / log-power bands (which bands, what FFT/STFT params), HG/HFA envelope, Hilbert, derivatives, stacked? Did the new baseline stack views (e.g. raw + spec) or swap one for another?
9. **Normalization scope** — per-window z-score, per-session, per-subject train-fixed, robust (median/MAD), none? Where in the pipeline does it sit (pre-feature / post-feature)?
10. **Per-class capping / class balancing** — same Lite cap as `c7b955b`, or changed?
11. **Electrode selection** — Lite-120 cap, random subset, anatomy-driven, full set?
12. **Classifier head** — still plain logistic regression, or moved to ridge / SVM / kernel method / something fancier? Regularization sweep? "Linear" is doing a lot of work in the framing — confirm it's still linear-in-features.

### B.3 — Strategic
- Anything you'd say is **"the genuinely hard thing for a foundation model to beat"** out of this work? (The unguarded answer here is what tells us where v14 actually has to win.)

## C — Cross-subject split semantics (decision 5) — TIER 1, ASK FIRST

This section drives v14's submit framing. Pairwise headline → small reproducible paper. Pooled headline → we own D.14 forever and likely have to re-run PopT-pooled ourselves.

### Mechanics
- In pinned commit `c7b955b0`, what does `include_all_train_subjects=True` actually return — pairwise 1-to-1 (one source subject per fold averaged) or pooled N-to-1 (all source subjects in one training set)?
- Is the headline cross-subject leaderboard number under pairwise, pooled, or both?
- Plans to ship a pooled-N-to-1 / LOSO / LOSESSION regime upstream? If so, timeline.
- Recommended way to define a pooled split locally that stays apples-to-apples with the upstream protocol (which subjects/sessions in source pool, what cap on held-out)?

### Strategy / meta — load-bearing for our paper
- **What number would you want to see in our paper to call cross-subject decoding solved?** (Single absolute number? Scaling curve? Both?)
- For the new NeurIPS linear-decoding submission (section B), is the headline pairwise, pooled, or both? Same question for the published 0.611 multiclass figure — pairwise, pooled, or per-subject averaged then aggregated?
- Would a scaling-curve presentation (v14 with K=1 source subject ≈ PopT pairwise; v14 with K=N pooled adds Δ ≥ 0.05) satisfy the "beat finetuned PopT by ≥0.05" reviewer ask, given PopT has no published pooled-N-to-1 number on Neuroprobe?
- If we re-run PopT pooled ourselves to get a head-to-head, would you be willing to vet our re-run protocol so the comparison isn't contested at review? (Pre-empting the inevitable "your PopT-pooled number is a strawman" challenge.)
- Is there a Neuroprobe-leaderboard convention being baked for v2 of the benchmark (pairwise vs pooled) that you'd want our paper to lead with so it's directly comparable later?

## D — Normalization recipe (decision 4) — TIER 3, PRE-RESOLVE FROM CODE

Resolve from `c7b955b0`'s `linear_baseline.py` before May 8. Ask only if Chris brings it up. L.1 sweep ablates this regardless of his answer.

- Confirm: linear baseline uses train-set/session-level fixed normalization; BrainBERT/PopT used per-window z-scoring inside the model — yes?
- Any newer evidence on which scope is the "right" default for cross-subject benchmarking? (Asking because we plan to ablate this on linear ourselves.)
- Would you consider standardizing normalization scope across the leaderboard so model-vs-model comparisons aren't confounded by recipe?

## D2 — Reference convention (Stage-0 ablation matrix) — TIER 3, PRE-RESOLVE FROM CODE

Resolve from `c7b955b0`'s `linear_baseline.py` before May 8. Ask only if Chris brings it up. L.2 sweep tests R0-R4 directly.

- Which reference does the upstream linear baseline (`run_linear_baseline.py` in the pinned `c7b955b0`) actually compute — raw monopolar, global CAR, bipolar, or shaft-Laplacian? The Lap+spec naming suggests Laplacian, but want to confirm what "Lap" resolves to in the implementation.
- Is global CAR ever appropriate for sparse asymmetric sEEG coverage, or do you treat within-shaft local / bipolar / shaft-Laplacian as the only physics-matched options?
- White-matter contacts and reference math: are WM contacts excluded from the channel set *before* CAR / shaft-CAR is computed, or included in the average and then dropped from the feature set?
- Recommended canonical handling of mixed gray/white shafts when computing a within-shaft local reference?

## E — Lite sampling — TIER 3, PRE-RESOLVE FROM CODE

V0 already verified the seeded-shuffle-then-cap behavior from `c7b955b0`'s code. Ask only if Chris brings it up.

- Confirm V0-derived behavior: each label's candidate indices are sampled with seeded `rng.choice(..., replace=False)` before the per-class cap, then sorted back into temporal order — yes?
- Recommended seed and any cross-version reproducibility caveats if we need to reproduce a specific Lite split exactly?

## F — Shaft / depth / contact metadata

- Is the canonical `(subject, shaft_id, contact_index)` parse from electrode label documented anywhere?
- Is contact-index orientation (deep → superficial vs. reverse) consistent across subjects? Per-subject convention or global rule?
- White-matter contacts: how are they flagged in the public release? Are upstream baselines including them?
- Any subject-specific notes on shaft adjacency or atypical contact spacing we should bake into the local-reference / shaft-Laplacian construction?

## G — Lite-vs-Full and electrode-set robustness

- Any internal experiments on Lite-vs-Full or random/anatomy/full electrode-set robustness beyond what's in the rebuttal Appendix?
- Sessions or electrodes you'd quietly exclude even though they pass the Lite mask? (V1 didn't find anything major, but you'd know.)

## H — Misc / future-facing

- Any plans for a multi-corpus iEEG release that would compose with BT (e.g. AJILE12-style)? We're targeting ~500–1000h Tier-0 SSL fuel and BT alone won't get us there.
- Anything about your roadmap for BT v2 / additional sessions we should know before committing to a v14 architecture freeze on the current Lite contract?

## Resolved (kept for reference)

- **2026-05-04 email reply** — confirmed no fsaverage-style table is publicly available yet; closest public artifact is the plotting overlay we already identified.
- **2026-05-05 D resolved from `c7b955b0` `examples/eval_population.py:248-250`** — linear baseline normalization is `sklearn.preprocessing.StandardScaler` fit on training-set features only, then applied to test. Mean/std per-feature, train-set fixed, post-flatten (after STFT/Laplacian features). Confirms our N1 cell IS the upstream baseline. `eval_utils.py:117-119` explicitly comments out a per-batch z-score with the rationale: "skipping batch norm here because in the regression pipeline, StandardScaler is used anyway, and we would like to avoid batch effects in case input items are processed one by one." So the train-set-fixed choice is intentional, not accidental.
- **2026-05-05 D2 resolved from `c7b955b0` `examples/eval_utils.py:190-273`** — "Laplacian" in the upstream baseline is **shaft-aware nearest-neighbor reference**: for each electrode `(stem, num)`, subtract the mean of present neighbors at `(stem, num±1)`. Stem parser strips trailing digits (so `'O1aIb4' → ('O1aIb', 4)`); attribution to BrainBERT (Wang et al. 2023). Critical detail: `remove_non_laplacian=False` is hardcoded in the upstream wrapper — non-Laplacian electrodes (singleton shafts, edge contacts) **pass through monopolar without rereferencing**. WM contacts are NOT excluded from the channel set before reference. Global CAR is not exposed as an upstream preprocess option at all. Implications for L.2: our R4 (shaft Laplacian) cell must match `remove_non_laplacian=False` to stay apples-to-apples with D.0; R0 (raw) is a strict subset of D.0's Laplacian for the singleton-shaft electrodes.
- **2026-05-05 E resolved from `c7b955b0` `neuroprobe/datasets.py:260`** — Lite sampling is `np.sort(self.rng.choice(label_indices, size=n_samples_each, replace=False))` where `self.rng = np.random.RandomState(NEUROPROBE_GLOBAL_RANDOM_SEED)` (default seed 42) and `n_samples_each = min(min_class_count, NEUROPROBE_LITE_MAX_SAMPLES // n_classes)`. Confirms V0's audit exactly: per-class candidate indices → seeded shuffle without replacement → cap → sort back to temporal order for `__getitem__` access. Reproducibility is deterministic given (seed, lite, eval_name, subject_id, trial_id).
- **2026-05-05 cross-subject baseline architecture finding (NEW, surfaced while resolving D/D2)** — `eval_population.py:220-224` calls `combine_regions()` from `eval_utils.py:863-903` for *every* CrossSubject linear evaluation. This **mean-pools electrode features within each Desikan-Killiany region**, then takes the *intersection* of DK regions between train and test subjects. So the upstream cross-subject linear baseline is **not** electrode-level — it's already DK-region-anchored with hard mean pooling. Reframes D.public from "DK hard-label control" to "the upstream cross-subject baseline architecture itself." v14's "shared anatomical coordinate frame" claim isn't novel at coarse DK granularity — what's novel is **finer (BNA Tier-1) parcels + soft `support[i,p]` Bayesian bias instead of hard DK pooling**. Mark this in stage_0.md and the Stage-1 strategy doc; carries paper-framing implications.
