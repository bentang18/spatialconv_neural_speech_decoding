# Questions for Christopher Wang

Running list. Pruned and reordered before each call. Last refreshed 2026-05-10.

Next contact: ping 2026-05-08, request a chat.

## A — fsaverage / surface mapping (original ask)

- What's the timeline on a per-electrode `fsaverage` table (subject, electrode, hemi, vertex index, surface RAS)?
- Is there an internal pre-projection step that produces fsaverage-coordinate intermediates before the public 2-D plotting overlay? Even unreleased internal artifacts would unblock us.
- Would you accept a community-contributed mapping if we derived one from public anatomy + plotting coords + (where available) per-subject FreeSurfer-style intermediates? What would make such a contribution land vs. not?
- Is there any provenance attached to `elec_coords_full.csv` we can use? E.g., the projection script, the snap-distance threshold, the hemisphere assignment rule.
- The braintreebank.dev quickstart notebook — does any cell in it touch a latent fsaverage step we missed in the public files? (Will check the notebook before the call regardless.)
- When the per-electrode `fsaverage` table ships, will it land as a `NeuralFetch` artifact (per-electrode metadata extractor on the `Wang2024Treebank` Study) or as a one-off CSV? If `NeuralFetch`, our v14 `BNAParcelMetadataExtractor` builds on top cleanly; if not, we build it ourselves from your coords as input.

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

## H — Stimulus overlap across CrossSession trials — TIER 2

V0.x DCC per-task audit landed 2026-05-10. **15/15 evaluated tasks leak**, 0 clean, 0 in the watch band. Per-task mean overlap fraction (test rows whose word also appears in train) ranges 33–48% across CrossSession; worst per-cell case sub7 onset 79.2% max, sub7 speech 75.9% max. `frame_brightness`, `pitch`, `face_num` average ≥0.479. Even `onset` (label stimulus-independent in principle) shows 33% mean — issue is brain-pattern memorization, not just label-bound content. Numbers in `reports/neuroprobe_stage0_v0x_stimulus_overlap_2026_05_10/`.

Earlier laptop upper bound (V0.x outer): ALL unique words in the test trial appear in the train trial **40% mean, 45% max** across all 12 (subject, trial-pair) cells. Same-movie repeats are not randomized, so word identity is shared between trials. Per-task DCC mode tightens after Neuroprobe's label-balancing + `rng.choice` cap and the picture is roughly the same.

For tasks whose label is a function of the word's acoustic / lexical / surprisal properties (pitch, volume / RMS, speech vs non-speech, GPT-2 surprisal, sentence-final, etc.), the linear classifier can in principle learn the (subject, word) → label association during train and re-recognize the same brain pattern at test — partial stimulus memorization rather than brain-decoding generalization. Tasks whose label is stimulus-independent (word onset y/n) are still affected by raw-pattern memorization at the 33%-overlap level.

- Are you aware of the train/test word overlap in the CrossSession protocol, and is it factored into the published headline numbers anywhere?
- Does the upstream pipeline (`eval_population.py` / `datasets.py` / `eval_utils.py`) do any word-level dedup at any stage, or are train and test allowed to share exemplars freely?
- For the new NeurIPS linear-decoding submission (section B): is there an "exclude-overlap" or "leakage-corrected" sister metric reported alongside the standard CrossSession AUROC? If so, what's the protocol (drop test rows whose word also appears in train; drop full word groups; something else)?
- Recommended canonical way for us to report a leakage-corrected number for label-content tasks so the comparison to PopT / BrainBERT stays apples-to-apples? Per-task post-hoc filter on the same eval rows, or re-run on a filtered Lite split?
- Plans for Neuroprobe v2 to mark or pre-filter overlapping exemplars in the eval set? (Would influence whether we ship our own filter for v14 vs. wait.)
- Practical take: do you read the CrossSession AUROC for label-content tasks as a clean brain-decoding number, or do you treat CrossSubject as the load-bearing eval for those? (We're planning to lean on CrossSubject in the v14 headline; want to know if you'd advocate or push back on that framing.)

## H2 — Within-subject session-id linear separability — TIER 2

L.5.P2 nuisance probe landed 2026-05-10 (`reports/neuroprobe_stage0_l5_nuisance_probes_2026_05_10_p2only/L.5.P2/metrics.json`). For each of the 6 BT Lite subjects, trained a per-subject LogReg to decode session-id (binary, since each Lite subject has 2 trials) from the L.2 winner features (shaft_laplacian × stft_abs, train_set_fixed normalization, 1s window). Result: **all 6 subjects ≥ 0.999 AUROC** (macro 0.9998, balanced accuracy 0.997). Within-subject, the L.2 winner view carries enough session-level structure for a linear probe to identify which session a window came from with near-perfect accuracy.

Caveat we want to discuss: feature dim is ~77K with ~15K train rows on a binary task — high-dim binary is biased toward near-perfect linear separability under any session-statistic shift (electrode drift, gain change, noise floor). So the kill-threshold of AUROC > 0.95 is loose for this regime, and the finding may not mean "the view is unusable" so much as "session statistics are non-trivially different and a CrossSession linear classifier could exploit them as a side channel for any task whose label is also session-correlated."

- Have you ever probed within-subject session-id separability from your linear-baseline features? If so, what numbers did you get and how did you read them?
- Is there any session-drift correction in the upstream pipeline we missed (per-session re-norm, channel-wise gain alignment, drift removal between trials)? `c7b955b0` `eval_population.py:248-250` fits StandardScaler on training features only — but train spans multiple sessions, so per-session mean shift would survive into the fit. We're not seeing a separate per-session re-baseline in your code.
- For tasks whose label distribution differs across the 2 sessions of a CrossSession pair (e.g. one trial features more high-pitch words than the other), a linear classifier could in principle short-circuit through "which session is this?" → "what label distribution to predict?" rather than learning a brain → label mapping. Have you checked whether any of the 15 CrossSession headline numbers are session-confounded in this way?
- Does the new NeurIPS linear-decoding submission (section B) include any session-drift mitigation we should adopt for v14?
- Practical: should we be reporting a "session-balanced" CrossSession AUROC alongside the standard one, where train and test are constructed to have matched per-session label priors? Or does that defeat the purpose of CrossSession?
- Stage-0 freeze impact: we're holding R4×I2 as the L.2 winner. The session-id leak doesn't disqualify the view (every reasonable preprocessing recipe will leak this on a binary high-dim probe), but it does mean v14's CrossSession wins must be checked against a session-shuffled control. Reasonable read?

## I — Misc / future-facing

- Any plans for a multi-corpus iEEG release that would compose with BT (e.g. AJILE12-style)? We're targeting ~500–1000h Tier-0 SSL fuel and BT alone won't get us there.
- Anything about your roadmap for BT v2 / additional sessions we should know before committing to a v14 architecture freeze on the current Lite contract?
- **L.7 audio-FM upper bound (Conwell veRSA control)**: do you happen to have cached Whisper-large-v3 features (any layer, but ideally L8) per word for the BT movie corpus? Public BT distribution has classical features.csv (mel/RMS/pitch) but no waveforms, so we can't run Whisper ourselves without sourcing the source movies separately. Cached features would short-circuit the audio-source problem entirely. (If not, we'll source the movies — wanted to check before doing so.)

## J — NeuralBench / NeuralFetch upstreaming — TIER 2

Context: NeuralBench-EEG v1.0 (Banville/King 2026, FAIR) explicitly invites iEEG tasks; v1 ships zero iEEG. Our Stage-0 substrate is already NeuralFetch / NeuralSet / NeuralTrain. Two upstream contributions are on the table — both touch your data and overlap with your NeurIPS submission, so we want to coordinate with you before opening any PR.

- **`Wang2024Treebank` Study → NeuralFetch upstream**: would you object to us upstreaming a `Wang2024Treebank` Study class (NeuralFetch's per-corpus metadata + raw-loader contract) so that any FAIR/Meta-built model can pull BT through the standard interface? It's your dataset; we'd want you on the PR (as co-author or reviewer, your call) and would not land it without your sign-off. Alternative: you have your own canonical Study you'd rather upstream — happy to defer.
- **Neuroprobe task definitions → NeuralBench iEEG slice**: the Neuroprobe tasks (15 binary + multi-class on word features) are the most usable iEEG-eval surface today. We were planning to propose them as the first iEEG slice of NeuralBench. Same coordination ask — would you support, push back, or want to lead the proposal? Worried we'd step on whatever your NeurIPS submission frames as the canonical BT eval.
- **Linear-baseline pinning**: if your NeurIPS result lands a new canonical multiclass number above the 0.611 published, we'd want both the v14 submit gate AND any NeuralBench iEEG-task YAMLs to pin to the new linear baseline. What's the cleanest way to coordinate that — track a pinned commit, ship the new baseline as a NeuralFetch eval recipe, something else?
- **NeuralBench awareness**: are you tracking NeuralBench as competing with Neuroprobe, complementing it, or orthogonal? Asking because if your view is "Neuroprobe is the iEEG benchmark; NeuralBench is for non-invasive," then our upstreaming proposal should be framed differently.
- **v14 thesis disclosure (briefly, before any infra PR)**: atlas-anchored shared coordinate frame (BNA Tier-1 soft support, not DK hard pooling), multi-FM cross-modal SSL with Whisper-L8 + DINOv3 teachers, zero per-subject parameters. You should know what we're doing on your data before agreeing to be on infra PRs that benefit from it.

## Resolved (kept for reference)

- **2026-05-04 email reply** — confirmed no fsaverage-style table is publicly available yet; closest public artifact is the plotting overlay we already identified.
- **2026-05-05 D resolved from `c7b955b0` `examples/eval_population.py:248-250`** — linear baseline normalization is `sklearn.preprocessing.StandardScaler` fit on training-set features only, then applied to test. Mean/std per-feature, train-set fixed, post-flatten (after STFT/Laplacian features). Confirms our N1 cell IS the upstream baseline. `eval_utils.py:117-119` explicitly comments out a per-batch z-score with the rationale: "skipping batch norm here because in the regression pipeline, StandardScaler is used anyway, and we would like to avoid batch effects in case input items are processed one by one." So the train-set-fixed choice is intentional, not accidental.
- **2026-05-05 D2 resolved from `c7b955b0` `examples/eval_utils.py:190-273`** — "Laplacian" in the upstream baseline is **shaft-aware nearest-neighbor reference**: for each electrode `(stem, num)`, subtract the mean of present neighbors at `(stem, num±1)`. Stem parser strips trailing digits (so `'O1aIb4' → ('O1aIb', 4)`); attribution to BrainBERT (Wang et al. 2023). Critical detail: `remove_non_laplacian=False` is hardcoded in the upstream wrapper — non-Laplacian electrodes (singleton shafts, edge contacts) **pass through monopolar without rereferencing**. WM contacts are NOT excluded from the channel set before reference. Global CAR is not exposed as an upstream preprocess option at all. Implications for L.2: our R4 (shaft Laplacian) cell must match `remove_non_laplacian=False` to stay apples-to-apples with D.0; R0 (raw) is a strict subset of D.0's Laplacian for the singleton-shaft electrodes.
- **2026-05-05 E resolved from `c7b955b0` `neuroprobe/datasets.py:260`** — Lite sampling is `np.sort(self.rng.choice(label_indices, size=n_samples_each, replace=False))` where `self.rng = np.random.RandomState(NEUROPROBE_GLOBAL_RANDOM_SEED)` (default seed 42) and `n_samples_each = min(min_class_count, NEUROPROBE_LITE_MAX_SAMPLES // n_classes)`. Confirms V0's audit exactly: per-class candidate indices → seeded shuffle without replacement → cap → sort back to temporal order for `__getitem__` access. Reproducibility is deterministic given (seed, lite, eval_name, subject_id, trial_id).
- **2026-05-05 cross-subject baseline architecture finding (NEW, surfaced while resolving D/D2)** — `eval_population.py:220-224` calls `combine_regions()` from `eval_utils.py:863-903` for *every* CrossSubject linear evaluation. This **mean-pools electrode features within each Desikan-Killiany region**, then takes the *intersection* of DK regions between train and test subjects. So the upstream cross-subject linear baseline is **not** electrode-level — it's already DK-region-anchored with hard mean pooling. Reframes D.public from "DK hard-label control" to "the upstream cross-subject baseline architecture itself." v14's "shared anatomical coordinate frame" claim isn't novel at coarse DK granularity — what's novel is **finer (BNA Tier-1) parcels + soft `support[i,p]` Bayesian bias instead of hard DK pooling**. Mark this in stage_0.md and the Stage-1 strategy doc; carries paper-framing implications.
