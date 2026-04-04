# NCA-JEPA Pretraining Design — Changelog

Version history for `2026-03-26-nca-jepa-pretraining-design.md`.

## v14 (2026-03-26, post-Zac-presentation polish)

**Three design additions:**
1. **Switching linear state-space generator added to the first round** — promoted into the generator ladder as the piecewise-linear middle ground between smooth AR and wave/PDE dynamics, and added to Method A's default first-round structured synthetic pool.
2. **Synthetic JEPA / latent target prediction promoted to first follow-up** — no longer a vague future ablation. If raw reconstruction looks shortcut-prone under the locked triggers, JEPA is the immediate next branch before widening the generator matrix.
3. **Synthetic nuisance model recalibrated** — replaced fixed `std=0.2` Gaussian noise with sequence-level IID and correlated noise ranges calibrated to raw residual HGA statistics, explicitly decoupled from clamped visualization renders.

**Three consistency fixes:**
4. **Tier 1 ladder ablations updated** — synthetic family list now includes Switching LDS and hotspot trajectory alongside AR, Wave, Gray-Scott, FHN, NCA, and Mixed.
5. **Implementation handoff updated** — first-round build order now names switching LDS explicitly and adds the JEPA follow-up gate.
6. **Reference list expanded** — added switching state-space, VideoMAE, and I-JEPA literature to support the new generator and objective framing.

82 resolved decisions, 8 open questions.

---

## v13 (2026-03-26, reviewer round 10)

**Two blocking issues fixed:**
1. **Stage 2 reframed as response-locked trial adaptation** — every SSL sample is a response-locked speech trial, giving the model the same phase-calendar shortcut that broke the acoustic-regression pivot (500ms-shifted controls were competitive). Fixed: (a) honest framing as weakly-supervised trial-epoch adaptation, (b) full-epoch off-center random crops from [-1.0, 1.5s] that remove entire phases from training samples, (c) time-reversal augmentation (10%) that destroys causal phase ordering.
2. **Architecture deconfounding completed** — conv-token GRU fully specified (shared BiGRU per spatial position, d=64, ~40K params, no cross-position temporal interaction), resolving former open question #7. ViT spot-check expanded to include Method B alongside Method A with both adapter variants, disambiguating architecture from pretraining.

**Four high-severity issues fixed:**
3. **Spatial pooling: top-k promoted to first-round co-default** — mean-pool erases focal articulatory activation on <2mm grids. Top-k (parameter-free) runs alongside mean-pool in first round.
4. **Masking ratio raised to 40-60%** — 10-17% masking on smooth 20Hz signal was trivially interpolatable. Default now 3-6 non-overlapping spans totaling 40-60% of frames. Mask-ratio sweep (30/50/70%) added as priority ablation.
5. **Transfer-proxy guardrails** — reduced probe frequency to 5K steps, added earliest-within-1SE checkpoint rule, locked dev split, added source-only sensitivity analysis.
6. **Method K: matched-statistics destroyed-dynamics control** — phase-randomized + time-shuffled variants preserve per-frame statistics while destroying temporal structure. Required for Claim 3. First round.

**Three medium issues fixed:**
7. **JEPA escalation rule** — hard switch criteria: promote latent prediction if event-frame pretext gains don't survive execution-only decoding, or patient-ID rises without phoneme-ID.
8. **Full-window is headline primary** — co-primary was rhetorically permissive. Full-window (validated at 0.700 PER) is headline endpoint; motor-focused is preregistered mechanistic secondary.
9. **Content-collapse diagnostics** — output entropy, unigram-KL, and stereotypy index catch failure modes that loss curves miss (concentrated phoneme distributions, stereotyped sequences).

**Structural:** Document streamlined — version history and resolved decisions moved to companion files.

79 resolved decisions, 8 open questions.

---

## v12 (2026-03-26, reviewer round 9)

**Three blocking issues fixed**: (1) **Stage 2 labeling honesty** — "no labels used" was misleading; phase-rebalanced loss requires response-onset timestamps (weak supervision). Split Stage 2 into pure-SSL default and phase-aware variant, with explicit labeling of supervision level; (2) **Architecture comparison deconfounded** — promoted conv-token GRU from Tier 1 into first-round execution (new Step 3b) as the bridge architecture that isolates spatial-token preservation from temporal-model-class change; (3) **Step 4b ViT spot-check uses conv stem** — Linear(1,128) too weak to be interpretable; paired linear-vs-conv-stem micro-ablation replaces single weak-adapter run. **Four design issues addressed**: (4) **Phase-rebalanced loss normalized** — raw MSE replaced by excess error over phase-template baseline (`sqrt(MSE - template_MSE)`); prevents raw amplitude/variance from dominating; (5) **Generator ladder coarse calibration** — source-only calibration pass added so poor hyperparameter instantiation can't be mistaken for bad inductive bias; (6) **Method J (B-extended) automatic** — promoted from future round; runs automatically if A > B confirmed by bootstrap CI; (7) **Position-aware probe added** — three-bin fixed temporal pooling alongside mean-pooled CE catches temporal discriminability that global mean-pool washes out. **Three precision issues addressed**: (8) **20Hz framing acknowledges prior evidence** — CE ablation showed 20Hz ≈ 40Hz; 40Hz alive only for micro-event/pretraining variants; (9) **Source sampling decoupled from dev patient** — similarity weighting uses leave-one-out within source patients, not dev patient; (10) **Fixed splits + surrogate null** — token-group splits locked once and reused across all methods; patient-level aggregation defined; label-permutation null added for headline result. 67 resolved decisions, 10 open questions.

## v11 (2026-03-26, reviewer round 8)

**Three blocking bugs fixed**: (1) **Phase-rebalancing formula corrected** — was `1/sqrt(MSE)` (inverted, amplified easy phases), now `sqrt(MSE)/sum(sqrt(MSE))` (correctly upweights hard phases); (2) **Double temporal downsampling removed** — preprocessing already handles 200→20Hz, so Family A's Conv1d changed from k=10,s=10 (catastrophic: 30→3 frames) to k=1,s=1 (pointwise projection only), reducing Family A from ~219K to ~71K params; (3) **"Multi-step prediction" wording fixed** in Stage 2 training and Methods A/B/C to say "masked span prediction" (matching the v10 objective change). **Seven design issues addressed**: (4) Family A bottleneck-pretext mismatch acknowledged; (5) ViT patient adapter weakness noted; (6) Pilot gate false-negative prevention (Step 4b); (7) Active anti-leakage in Stage 2 (random temporal crop); (8) Motor-focused and full-window decoding are co-primary; (9) CV split feasibility constraint; (10) Matching statistics: evoked + residual. 58 resolved decisions, 10 open questions.

## v10 (2026-03-26, reviewer round 7)

(1) **Masked temporal spans as default pretext** — resolves causal inconsistency where BiGRU leaked future context during causal prediction; both families now bidirectional, causal multi-step demoted to ablation; (2) Single fixed dev patient for transfer-proxy; (3) Random-scaffold control (Method F) promoted to first round; (4) Phase-rebalanced prediction loss as Stage 2 default; (5) Family A underspecification fixed; (6) 1x1 patches as pilot default; (7) Micro-event ablation requires 40Hz; (8) Matching statistics tightened; (9) Pilot-first execution; (10) Title narrowed. 51 resolved decisions, 9 open questions.

## v9 (2026-03-26, reviewer round 6)

(1) Motor-focused readout crop acknowledged as new hypothesis; (2) Stage 2 full-trial objective justified; (3) Synthetic sub-event templates changed from phoneme-specific to generic random; (4) Dead-position changed to dead-fraction; (5) Response-locked window wording corrected; (6) Micro-event structure within execution added as ablation; (7) Transfer-proxy power question added; (8) 46 resolved decisions, 9 open questions.

## v8 (2026-03-26, reviewer round 5)

(1) Trial anatomy restructured from 3-phase to 5-phase, motor-focused crop as default; (2) Generators changed from free-running to externally-driven; (3) Paper claim sharpened to motor-speech decoding; (4) Trial-calendar leakage elevated to primary risk with 3 stronger baselines; (5) Motor-focused vs full-trial vs execution-only evaluations; (6) Cue-to-response delay distribution added; (7) Phase timing variability replaces 25ms onset jitter; (8) 42 resolved decisions.

## v7 (2026-03-26, reviewer round 4)

(1) Stage 3 CE default locked to validated config (global mean-pool + 27-way head); (2) Synthetic trial anatomy restructured to baseline→speech→decay; (3) All temporal hyperparameters in milliseconds; (4) Generator ladder relabeled "comparative ladder"; (5) Local mm coordinates split from cross-patient MNI encoding; (6) Coordinate-augmentation consistency rule; (7) Conv-GRU framed as conservative control; conv-token GRU promoted to Tier 1; (8) Causal mask removed at Stage 3; (9) Transfer-proxy patients excluded from headline claims; (10) Event-frame diagnostics added; (11) Parameter counts corrected; (12) Grouped-by-token CV comparability note; (13) First-round matrix reduced.

## v6 (2026-03-26, initial design)

(1) Two architecture families (conv-GRU + ViT); (2) CE primary Stage 3 readout; (3) Grouped-by-token CV; (4) Generator ladder with controlled inductive biases; (5) Physical mm coordinates; (6) Transfer-proxy model selection gate; (7) Patch size 1x1 + conv stem promoted; (8) Step-matched protocol; (9) Capacity justification; (10) Anti-alias LPF; (11) Three-stage regime switching; (12) Real dead-electrode templates; (13) Shared decoder in Stage 2; (14) Negative transfer mitigation ablations; (15) Spatial pooling ablation; (16) Corrected parameter counts.
