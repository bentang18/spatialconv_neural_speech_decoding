# Neuroprobe Cross-Subject Hillclimb — Plan (first pass)

*Draft 2026-04-24. Parallel side-quest to v14 Phase-1. Starts post-finals.*

Background and rulebook: Zahorodnii et al. 2509.21671 (paper) + `insight-neuro/neuroprobe` repo. Active project memory: `memory/project_neuroprobe_cross_subject_hillclimb_2026_04_22.md`. Benchmark reference memory: `memory/reference_neuroprobe_zahorodnii_2025_09.md`.

## Thesis

At Neuroprobe's scale (3500 labels, 1 training subject for cross-subject) current iEEG foundation models learn subject-specific structure that hurts transfer — BrainBERT-trained 0.522 < BrainBERT-untrained 0.527 cross-subject is direct evidence. Linear (Laplacian+spectrogram) wins at 0.539 because it effectively region-averages via Desikan-Killiany common regions (`combine_regions()` in their baseline pipeline). A v14 model with explicit BNA parcel embeddings attached to each electrode (via support), pretrained on the permitted BrainTreebank sessions + external sEEG, should capture what the linear baseline captures and add back intra-parcel temporal structure.

## Shared frame (frozen 2026-04-24)

These commitments are the scaffolding. Change one and you're writing a different plan.

1. **What's shared cross-subject is at the parcel level, not the electrode level.** BNA parcel embeddings (`P_emb`) attached to each electrode via its soft support vector is the core mechanism — the principled upgrade of `combine_regions()` DK region-averaging.
2. **Cross-subject split only.** No Within-Session or Cross-Session submission.
3. **Pretraining is load-bearing.** The thesis doesn't live or die on cold-start; it lives on whether atlas-anchored pretraining produces a representation that transfers where raw-voltage SSL does not.
4. **BrainTreebank's hardcoded `NEUROPROBE_LITE_ELECTRODES` (~120/subject) is fixed.** No electrode selection, no sig-channel filtering.
5. **Benchmark protocol is fixed.** 1-second post-onset window, 15 binary AUROC tasks, S2/trial-4 training only, percentile-thresholded labels.

## Experiments (frozen for now, will expand)

These are the forks we cannot resolve by debate — only by running. Each needs at least one ablation on cross-subject mean AUROC. The list will grow once we have Stage-1 numbers.

1. **Latent bottleneck.** Perceiver-style cross-attention from electrode tokens into a small set of latents (learned queries, or anatomy-seeded e.g. one latent per BNA parcel), then decode. Tests whether bottlenecking stabilizes cross-subject transfer vs full per-electrode attention.
2. **Fixed vs learned `P_emb`.** Fixed (identity-init frozen, or anatomy-init frozen) isolates the "anatomy carries transfer" hypothesis. Learned lets the model discover functional parcel similarity. Orthogonal-learned ≈ fixed-identity up to rotation, so the empirical question is whether learned `P_emb` develops non-orthogonal structure.
3. **Per-subject embeddings.** One learned vector per subject added to every electrode token. Tests whether cross-subject noise floor needs explicit per-subject correction, or parcel-space pooling is sufficient.
4. **Spatial encoding beyond `P_emb`** — *resolved 2026-04-25 (revised same day)*. Pre-committed as the v14-default replacement for `pe2d`: **BNA-connectivity-init attention bias** (functional connectivity primary, anatomic + behavioral as ablation alternates), within-parcel free / across-parcel biased. Atlas-consistent — same BNA that drives `P_emb` drives the cross-parcel attention prior. Cortical-geodesic distance was an earlier proposal (same-day first pass) but BNA's pre-built functional/anatomic connectivity matrices are the principled prior, since adjacent parcels can be functionally distinct. Variants (a) `P_emb`-only and (b) MNI-Fourier kept as ablations.
5. **SSL objective** — *resolved 2026-04-26, single-loss*. **D-SigLIP brain ↔ frozen Whisper-large mid-encoder**, trained projection heads `R^d → R^k` and `R^1280 → R^k`. No auxiliary terms; matches Évanson Minutes-to-Days iEEG-on-modality precedent (CLIP-only) and the King-lab convergent stack (d'Ascoli D-SigLIP at scale; Brain2Qwerty CLIP). 3-s symmetric SSL windows decoupled from 1-s eval head-fit. Per-subject + per-session affine normalization on parcel features (cheap replacement for King-lab's per-subject linear layer). See `stage_2.md` *Pre-committed Stage-2 SSL design (single-loss, Evanson-aligned)*. Cross-subject parcel-id contrastive demoted to optional auxiliary (degenerate without stimulus alignment; redundant with brain↔FM when aligned). MSE λ=0.25 aux dropped (Évanson contrastive-only result holds on iEEG). Masked-parcel-recon and DeeperBrain-NSP kept as fallback ablations only if scaling curve under-delivers (slope < 0.04 per log-h).
6. **Frozen probe vs fine-tune.** For the submission number: linear probe on frozen backbone per task, or end-to-end fine-tune per task. Frozen is the honesty standard and avoids overfitting at 3500 samples; fine-tune may raise the number. Tangles with #5 — pretraining objective determines probe ceiling.
7. **Is the probablistic BNA actually worth over argmax?** Resolved by design into the Stage-0 D-matrix (`docs/neuroprobe/stage_0.md` Block D): D.1a vs D.1b and D.3a vs D.3b directly compare argmax-hard vs probabilistic-soft support under each prep. The "soft vs hard support" pairwise readout (twice, once per prep) gives the empirical answer before Stage 1 commits.
8. **How should we structure supervised learning on the rest of Braintreebank legal dataset?** Partition: Stage 2 keeps cheap, label-free regularization only (`docs/neuroprobe/stage_2.md`: 2a SSL-only + 2b SSL backbone with multi-patient head prior on the head side). Stage 3 (`docs/neuroprobe/stage_3.md`) does the supervised pretraining via continuous FM-embedding regression (DINOv3 frame features, Whisper / Wav2Vec audio features, GPT-2 hidden states) — strictly richer than the originally-planned discrete linguistic-label set (phoneme, syllable, word2vec, GPT-2 hidden, speaker-onset), since each discrete label is a low-rank summary of its corresponding FM hidden state. Open empirical questions: FM layer choice per modality, joint vs sequential schedule, modality contribution.
9. **Shaft-aware attention.** Depth contacts on the same shaft are physically adjacent (~3.3 mm intra-shaft) but can cross BNA parcel boundaries (GM/WM/GM transitions). v14's combined attention has no physical-adjacency inductive bias — it relies entirely on `P_emb` similarity. Add a shaft-aware mask using BT electrode name stems to bias attention toward same-shaft contacts. BT/D-cohort-specific; no analog for uECoG dense grids. Test in Stage 1 if cold-start sits in 0.527–0.539 cluster.
10. **Dilated-CNN temporal front-end on raw voltage vs hand-engineered HG envelope.** BarISTA on BT (Oganesian NeurIPS 2025) shows 5-layer dilated CNN on raw 2048 Hz beats linear projection by +6–9 pp AUC. v14 currently uses hand-engineered HG envelope at 200 Hz. Trade-off: HG is an informed prior; learned CNN may pick up sub-70 Hz linguistic-relevant bands and theta-gamma coupling that HG discards. Test as alternate input pipeline in Stage 1 (replace Block-E loader's filterbank+Hilbert+resample with raw 250 ms patches → 5-layer dilated-CNN at d=64; everything downstream of the per-electrode-token contract unchanged).
11. **Within-parcel / intra-parcel spatial expressivity.** Current default (BNA-connectivity-init bias + `P_emb`) carries *across*-parcel structure but treats electrodes within the same parcel identically — two contacts in the same parcel get the same bias term despite differing 3-D positions. Real signal lost: same-shaft sEEG contacts at ~3.3 mm spacing have strong volume-conduction correlation; within-parcel cortical patch geometry varies. Three candidate fixes, ablated against the BNA-conn-only default: (a) **within-parcel 3-D Euclidean pairwise bias** — `exp(-‖p_i − p_j‖² / σ²)` on MNI positions, applied only when both electrodes share a parcel (no cross-sulcus problem — within-parcel stays within a sulcus); (b) **shaft-aware additive bias** (Experiment #9, restated and elevated) — same-shaft pairs identified via BT electrode name stems; sEEG-only, no uECoG analog; (c) **3-D MNI Fourier PE on per-electrode tokens** — **SwinTW-style** (Chen 2025): demonstrates cross-subject without per-subject params using shared 3-D coords. v14 already commits to no-per-subject-params via atlas; SwinTW shows raw 3-D coords work too; here we'd add Fourier PE *on top of* `P_emb`'s parcel-id to inject within-parcel resolution. Run as Stage-1 ablation alongside the scaling curve. Promote winner into Stage-2 default if any clears +0.005 cross-subject AUROC vs default. **Note**: Chen 2025 (SwinTW) is what Évanson Minutes-to-Days actually cites as a "subject-embedding alternative" — slightly imprecise on Évanson's part, since SwinTW has *no* per-subject params; it's a "shared 3-D-coord-as-shared-coordinate" approach, not a subject-embedding one. v14's atlas anchor is the next step up (3-D coords → BNA parcels), justified by iEEG SNR resolving parcels (EEG can't).

**Tactical note (not experiment-only):** preprocessing front-end — HG time-domain at 200 Hz (our PS pipeline) vs Laplacian+spectrogram (baseline's trick). Staying HG-only makes a cleaner thesis; adding spectral concedes the baseline's ground but may raise the floor. Defer to Stage-1 empirics.

**2026-05-15 architecture lock** (`memory/project_v14_first_pass_simplification_2026_05_15.md`): Four architectural deltas locked after SOTA EEG-FM literature synthesis (NeuralBench REVE / Liu CBraMod / ZUNA / Charmander / MTDP). D1 drop DETR → single-query PMA readout (Set Transformer / CaiT); D2 add shaft-T5-bias via input-side self-attn block (PerceiverAR-family) on signal-only tokens, parcel-id PE injected fresh after; D3 run vanilla-raster + parcel-id-PE as P1 ablation V1; D4 add (not concat) for parcel-id PE. L1 distillation deferred to v15. Charmander identified as closest published Perceiver-on-iEEG precedent; the only architectural difference is anatomy-anchored vs abstract latents → v14 contribution-vs-Charmander headline = "DK parcel-anchoring enables zero-shot where Charmander's per-electrode c_j requires 50-epoch warmup." Pediatric MNI registration error (5-15 mm at cortical surfaces) is reframed as a *positive defensive reason* for discrete native-space DK routing over continuous-coord PE, not a workaround — paper claim: "the architectural commitment is anatomy-as-categorical-routing, not anatomy-as-coordinate."

## Target

**Cross-subject mean AUROC ≥ 0.56 to submit. ≥ 0.58 is the stretch target.** Sub-0.539 = we don't submit.

Only the Cross-Subject split. No Within-Session or Cross-Session submission — they dilute the story. We do not try to beat DIVER-1 on Within-Session (0.678 with 59k h pretraining is not our game).

## Current leaderboard (cross-subject, our target)

```
Linear (Lap+spec)         0.539   #1
Linear (spec)             0.528
BrainBERT untrained       0.527
PopulationTransformer     0.526
BrainBERT trained         0.522
Linear raw voltage        0.510
Chance                    0.500
DIVER-1                   —       (did not submit cross-subject)
BarISTA                   —       (not evaluated)
```

## Stages

### Stage 0 — Reproduce Linear (Lap+spec) = 0.539 + pipeline rigor (~1 week)

Detailed task plan: **`docs/neuroprobe/stage_0.md`**. Two interlocked goals — reproduce the #1 Linear baseline on DCC AND validate every pipeline primitive we'll reuse in Stages 1–3 (mesh identity, coord lookup, BNA bake rigor, support cache, loader contract). Six blocks (A–F) with per-block gates; final gate is per-task AUROC within ±0.005 of the leaderboard JSON across all 150 session-tasks. Block A (rigor) runs locally from already-downloaded metadata at `/tmp/bt_metadata/`; DCC work begins in Block B.

### Stage 1 — v14 cold-start, S2/trial-4 only (1 week)

Skeleton: **`docs/neuroprobe/stage_1.md`**. v14 per-electrode-token backbone (B-1 path, not per-cell pooled — electrode counts vary cross-subject), BNA `P_emb` from Stage-0 support cache, RoPE temporal-only, attention with **BNA-connectivity-init bias (within-parcel free, across-parcel biased; replaces `pe2d`)**, **flat per-parcel pool readout (replaces `hierarchical_atlas`; hierarchy retained as +tier ablation)**, **no `partialconv`** (2-D-grid assumption, no sEEG analog), `d=32`. No pretraining. Preprocess via Stage-0 Block-E loader (CAR + HG + exact 200 Hz + median/MAD z-score). Per-task binary CE, val early-stopping on Neuroprobe val half. 2 seeds. Augmentation (channel-dropout, time-warp ±5%, mixup) is legal under the protocol and underused on the leaderboard.

**Gate**: mean cross-subject AUROC **strictly > 0.539**. The BrainBERT/PopT/random-TF cluster sits at ~0.527; Linear Lap+spec at 0.539 is the ceiling for "alignment without learning." v14's anatomical anchoring is *architectural* (parcel embeddings part of cold model). To support the alignment-is-the-bottleneck thesis, cold-start must break the 0.539 ceiling without any pretraining. Decision tree: > 0.539 → anatomical priors do alignment cold (proceed to chase 0.56 in Stage 2); 0.527–0.539 → architecture nonlinear-feature lift but anchoring insufficient alone (Stage 2 becomes thesis-deciding); < 0.527 → below the cluster ceiling, structural bug, debug before continuing.

### Stage 2 — Atlas-anchored SSL pretraining + S2/trial-4 fine-tune (2–3 weeks)

Skeleton: **`docs/neuroprobe/stage_2.md`**. Two sub-stages, both sharing the legal Tier-1 corpus, both label-free at the backbone:

- **2a — SSL only.** Backbone trained with the pre-committed single-loss SSL (D-SigLIP brain ↔ frozen Whisper-large mid-encoder + trained projection heads `R^d → R^k` and `R^1280 → R^k`). Frozen backbone, linear L2 head fit on S2/trial-4. Contrastive-SSL ceiling on the atlas-anchored backbone.
- **2b — SSL backbone + multi-patient head prior (cheap regularization).** Same SSL backbone. Per (legal session × eval task) fit a small linear head; pool head weights across legal sessions per task → multi-patient prior. Use as init / L2 target for the S2/trial-4 head fine-tune. Cost: hours of inference + linear regression. Decouples "head saw multi-patient" from "backbone saw multi-patient" — strictly conservative, the backbone never sees task labels.

The originally-planned discrete-linguistic-label supervised sub-stage (was 2b) was cut on 2026-04-25; supervised pretraining moved to Stage 3 as continuous FM-embedding regression (strictly richer pretext).

**SSL design constraints (load-bearing, pre-committed 2026-04-26 single-loss revision).** The thesis predicts pretrained-v14 > cold-start-v14 cross-subject — the inverse of BrainBERT-trained < BrainBERT-untrained. This holds only if SSL doesn't smuggle subject-specific structure back in. Four rules: (i) **single contrastive loss = D-SigLIP brain ↔ frozen Whisper-large mid-encoder** (matches Évanson Minutes-to-Days iEEG-on-modality precedent + King-lab convergent stack; D-SigLIP > plain CLIP because cross-subject same-content batching corrupts CLIP's softmax negative pool — multiple BT subjects watched the same movies, identical audio targets in batch, plain CLIP would push their brain reps apart wrongly); (ii) **per-subject + per-session affine normalization** of parcel features (Évanson day-drift mitigation; cheap replacement for King-lab's per-subject linear layer at finer granularity); (iii) **3-s symmetric SSL windows decoupled from 1-s eval window**; (iv) avoid pretexts coupled to per-subject structure (no "predict next electrode in this montage"). Cross-subject parcel-id contrastive demoted to optional auxiliary (degenerate without stim-aligned pairs; redundant with brain↔FM when aligned). MSE λ=0.25 aux dropped (Évanson holds with contrastive-only on iEEG). Full design contract: `stage_2.md` *Pre-committed Stage-2 SSL design (single-loss, Evanson-aligned)*.

**Stage-2 deliverables**: Banville-style 4-point scaling curve (~8h / 16h / 33h / 66h-with-D-cohort) at the pre-committed pretext + `d_model` ∈ {32, 64, 128} sweep on Full Tier-1 + **mandatory side-by-side against atlas-anchored masked-parcel-recon SSL** at Tier-1 33 h (attribution: did lift come from atlas anchoring, brain↔FM, or only their combination?).

**Submission decision rule.** Submit best AUROC of {2a, 2b}. If neither clears 0.56, defer submission to Stage 3.

**Gate**: mean cross-subject AUROC ≥ 0.56. At ≥ 0.56 submit and proceed to Stage 3 for stretch; at < 0.539 don't submit and post-mortem instead.

**The bet — explicit acknowledgment.** Cross-subject brain↔FM SSL on iEEG has *never been measured* by anyone — Évanson Minutes-to-Days runs per-subject; King-lab's cross-subject results are EEG/MEG. The masked-recon iEEG SSL family (BrainBERT, PopT) has been measured cross-subject and **fails** (≤ 0.527; BrainBERT-trained 0.522 < BrainBERT-untrained 0.527 — the trained-worse-than-untrained inversion). DIVER-1 (5.3 k h iEEG, masked-recon family) declined to submit cross-subject despite #1 within-session — strong implicit signal. Stage 2 is therefore betting that combining (atlas anchoring) + (King-lab brain↔FM SSL) crosses the gap masked-recon couldn't. Without a side-by-side comparison, a Stage-2 positive result is ambiguous — did lift come from atlas anchoring, from brain↔FM, or only from their combination?

**Mandatory side-by-side at Tier-1 33 h: atlas-anchored + masked-parcel-recon.** Same backbone as the default (per-electrode + BNA-conn-bias + flat per-parcel pool + RoPE), same per-subject + per-session normalization, same 3-s SSL windows — just swap the SSL pretext. Masked-parcel-recon = mask 30 % of (electrode × time) cells, predict pooled parcel representation. Closest atlas-anchored analog of BrainBERT. The contrastive-vs-recon attribution is load-bearing: if masked-parcel-recon also clears 0.56, atlas anchoring is doing the cross-subject work and the SSL family is interchangeable; if only D-SigLIP clears, brain↔FM is the load-bearing piece; if neither clears, the architecture is the bottleneck and Stage 3 won't save it.

### Stage 3 — FM-embedding regression pretraining (~2 weeks)

Skeleton: **`docs/neuroprobe/stage_3.md`**. Continuous FM-embedding regression as the supervised pretext: predict frozen DINOv3 (vision), Whisper / Wav2Vec2 (audio), and GPT-2 (language) hidden states from neural input at every timestep. **Regression-target framing** (FM as target, not input). Joint with Stage-2 SSL on top of the Stage-2a backbone. Replaces the discrete-linguistic-label supervised pretraining originally planned in Stage 2b — continuous FM hidden states are a strict superset of every discrete linguistic label.

**Stimulus prereq (Block-S).** BT ships audio + MFA transcripts; movie video frames may need to be sourced externally (films are publicly identified per BT's metadata). Block-S confirms availability + builds an FM-target cache before Stage 3 launches.

**Gate**: cross-subject mean AUROC ≥ 0.58 (the existing stretch target). If Stage 2 already cleared 0.58, Stage 3 must beat Stage 2 by ≥ +0.01 to justify added complexity.

### Stage 4 — Submit (2 days)

Fork `insight-neuro/neuroprobe`, write 15 JSONs to `leaderboard/NFP_Ben_Tang_<DD>_<MM>_2026/Cross-Subject/`, `metadata.json` + `ATTESTATION.txt` + `PUBLICATION.bib` → our preprint, PR. CI is format-only (`tests/test_submission_format.py`); no reproducibility check. May happen twice — once after Stage 2 if it clears 0.56 (early submission to get on the board), once after Stage 3 if it beats Stage 2 (stretch-target update).

## Pretraining corpus (frozen 2026-04-24)

Same-modality-first sequencing. Isolate the atlas-anchored SSL claim before layering cross-cohort or cross-sensor transfer. Each tier is gated on the previous tier clearing Linear (0.539).

- **Tier 1 — BrainTreebank-only, ~33 h sEEG.** First-pass submission corpus.
  - **Allowed set (~25 h).** SUBMIT.md whitelist: 14 full non-eval sessions + 5 partials (btbank7_{100,101,102}, btbank10_{100,101}). Nuance: btbank1_0, btbank3_2, btbank4_2 contain electrodes of test subjects — legal but disclose.
  - **Unseen-subjects-only (~8 h).** S5/S6/S8/S9 + S7/S10 partials, zero test-electrode overlap. Serves double duty: contributes to submission cocktail *and* stands alone as the paper ablation showing the lift doesn't depend on within-test-subject exposure.
- **Tier 2 — + D-cohort sEEG (~25–50 h), gated on Tier 1 > 0.539.** 85 Duke pts, Box-available, BNA parity audited 2026-04-21 (74% Tier-1 argmax at 5 mm gate). Same modality (sEEG), different cohort. Tests whether cohort scale-up lifts further.
- **Tier 3 — + external public sEEG / ECoG, gated on Tier 2 lift.** Selection criterion split by stage:
  - **Stage-3-eligible (paired audio + transcripts, ideally video):** **Naturalistic film iEEG ds003688 (51 pts, audiovisual film — premium Stage-3 fit, full 4-target richness like BT)**, Zada et al. Podcast ECoG ds005574 (9 ECoG patients, 1 330 electrodes, comprehension-only, public — Hasson lab companion to Goldstein 2025), NeuroListen (5 sEEG, 10+ h, natural-speech listening), sEEG passive listening ds004703 (10 pts, license caveat).
  - **Stage-2-only SSL fuel (no aligned stimulus):** AJILE12 (12 pts, ~1 280 h ECoG), SWEC (**50 unique pts, ~6 672 h** — HF lists 68/9 328 but 18 are duplicate re-exports; band-pass 0.5–150 Hz so no content >150 Hz, zero electrode anatomy — audited 2026-05-19, `memory/reference_swec_ieeg_dataset_audit_2026_05_19.md`), UPenn RAM (251 pts, 1 000+ h), Omni-iEEG (302 pts, ~178 h). Useful at scale for SSL pretraining only.
  - **Cross-language production corpora (Stage 2 + partial Stage 3 acoustic):** Du-In (Fan et al. 2024, Mandarin sEEG), Kunii/Tsuboyama Japanese sEEG, Utrecht/Martin Dutch sEEG.
  - Cross-language + cross-cohort + cross-sensor (ECoG joins sEEG). Only if Tier 2 clears. Sensor-modality crossover (ECoG vs sEEG) is the load-bearing question for whether external corpora compose legally with the BT-only Tier 1 thesis.
- **Watch-list (paper-only, would transform Stage 3 if released).** Evanson "Minutes to Days" (Meta FAIR + APHP, arXiv 2512.15830 — 3 sEEG × {100, 108, 84} h ambient pretrain + {74, 43, 250} min audiobook task, the closest open chronic field-potential + audio dataset on the horizon), Evanson "Emergence of Language" (arXiv 2512.05718 — 46 pts sEEG+ECoG audiobook, 7 400 electrodes), Goldstein / Zada / Flinker NYU 24/7 ECoG (Nat Hum Behav 2025, ~100 h / 4 pts chronic conversations — currently private), Neuro2Semantic (Mesgarani / Flinker, perceived speech + transcripts). Tracked at `docs/references/data_acquisition_matrix.md`.

**Out of scope for this program.** uECoG (PS + lexical) is not in the Neuroprobe pretrain corpus — cross-sensor (surface ↔ depth) transfer is a separate claim. PS/lex uECoG stays in v14 Phase-1/1.5 for the main paper.

**Residual mismatch to accept even within Tier 1.** BrainTreebank pretrain is continuous movie-watching; Neuroprobe eval is 1 s word-locked windows. Same cohort + same sensor, but task-distribution still differs. The eval-window-shape constraint on Experiment #5 is the main mitigation.

## Infrastructure to build

1. `src/speech_decoding/neuroprobe/loader.py` — Neuroprobe → v14 adapter. 2048 Hz → 200 Hz HG z-score. Per-electrode tokens + support + active mask.
2. `src/speech_decoding/neuroprobe/coords.py` — BrainTreebank `coordinates_type="cortical"` → BNA. Resolves open-question #1 below.
3. `scripts/neuroprobe/stage0_linear_<cell>.sh` — per-cell Stage-0 SLURM arrays for the 13-cell linear ablation matrix (D.0 through D.10). See `stage_0.md` Block D.
4. `scripts/neuroprobe/stage1_v14_scratch.sh` — Stage-1 cold-start array.
5. `scripts/neuroprobe/stage2_ssl_pretrain.sh` — pretrain driver on the locked cocktail.
6. `scripts/neuroprobe/stage2_finetune.sh` — 15-task binary fine-tune array.
7. `src/speech_decoding/neuroprobe/fm_targets.py` — DINOv3 / Whisper / GPT-2 inference + per-session FM-target cache writer. Stage-3 prereq.
8. `scripts/neuroprobe/cache_fm_targets.py` — driver for (5)'s caching pass over the legal corpus.
9. `scripts/neuroprobe/stage3_fm_regression_pretrain.sh` — Stage-3 joint SSL + FM-regression pretrain driver.
10. `scripts/neuroprobe/submit_format.py` — writes the 15 JSONs + runs `tests/test_submission_format.py` locally before PR.

## Open questions to resolve in Stage 0

1. **BrainTreebank `coordinates_type="cortical"`** — what space? If fsaverage, direct BNA via our bake. If not, need BrainTreebank FreeSurfer recons.
2. **BNA Tier-1 parcel list for Neuroprobe** — LH-only Phase-1 list is wrong for BT's bilateral whole-brain coverage. Resolved by design at Stage-0 A0: derive a fresh BT-Tier-1 list (every parcel with ≥1 argmax-win on the 1145 Lite-with-coord cohort) as a **new** constant `BT_TIER1_PARCELS` in `src/speech_decoding/neuroprobe/atlas_tier1_bt.py`. Phase-1 `DEFAULT_BASE_PARCELS` is **not** mutated — that contract is frozen.
3. **Per-electrode tokens on variable electrode counts** — the B-1 per-electrode path was slated for Phase 2 in our Phase-1 plan. Stage 1 forces it forward. Scope the engineering lift before committing.
4. **Preprocessing recipe parity** — our 70–150 Hz Gaussian filterbank is for 2 kHz input; BrainTreebank at 2048 Hz should be a drop-in but needs verification.

## Explicitly not doing

- **Within-Session or Cross-Session submission.** Only Cross-Subject. DIVER-1 is the within-session boss and is out of scope.
- **DK-atlas alignment** (the baseline's `combine_regions`). BNA is our thesis.
- **Electrode selection.** Neuroprobe hardcodes ~120 electrodes per subject via `NEUROPROBE_LITE_ELECTRODES` — we use exactly those.
- **Any pretraining touching the 12 off-limits sessions** (`btbank1_{1,2}, btbank2_{0,4}, btbank3_{0,1}, btbank4_{0,1}, btbank7_{0,1}, btbank10_{0,1}`).

## Sequencing

Stage 0 starts post-finals; Stage 2 gated on Phase-1 v14 landing so DCC compute doesn't collide. Conditional sequencing: first submission from Stage 2 if it clears 0.56 (mid-June target), Stage 3 chases stretch as a follow-up submission (early–mid July). If Stage 2 falls short (< 0.56 but ≥ 0.539), Stage 3 becomes load-bearing for any submission. Full program: ~6 weeks walltime if everything goes well.
