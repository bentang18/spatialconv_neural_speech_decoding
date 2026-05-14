# Neuroprobe Stage 1 — v14 cold-start (skeleton)

*Drafted 2026-04-25. Provisional. Detail follows Stage-0 close-out.*

*2026-04-26 evening — success criterion + decision tree + open-question thresholds re-pegged to multi-class cross-session (0.611 floor; BrainBERT 0.557 / PopT 0.562 cluster). Cross-subject binary tracked as v1 secondary indicator. Canonical: `memory/project_neuroprobe_rejected_real_bar_2026_04_26.md`.*

*2026-04-27 evening — Three Stage-2 default-level changes are gated on Stage-1 cold-start verification: (i) **Schedule-Free AdamW vs cosine** at the same compute, two seeds — adopt as Stage-2 default if Schedule-Free wins by ≥0.005 OR ties (anytime-stopping advantage); (ii) **bf16 mixed precision** is now Stage-1 default (no verification gate — field-convergent across 5 papers); (iii) **Intermediate-layer readout sweep** (PE-Core "best embeddings hidden in intermediate layers" hypothesis) — Tier-1 ablation cell at Stage-1 cold-start, sweep task-attention pool over layer L ∈ {final, mid, mid-1}. If mid > final by ≥0.005, lifts cold-start AUC AND becomes Stage-2 readout default AND becomes paper headline. Synthesis: `memory/reference_sota_pretraining_synthesis_18papers_2026_04_27.md`.*

*2026-04-27 late night — final exhaustive sweep (28-paper supplement: `memory/reference_sota_supplement_28papers_2026_04_27.md`) adds two Stage-1-relevant changes: (iv) **Cross-modal target = Whisper-large-v3 layer 8 (~25% depth), NOT layer 16** — BrainWhisperer (ICLR 2026, arXiv 2603.13321) + WhisperBCI (ICLR 2026 reject — public OpenReview Sept 2025, openreview 064e8544) both use Whisper-tiny L3-of-6 (proportional 25%) for intracortical phoneme decoding. Évanson "mid" came from speaker-ID layer-probing literature, not phonetic. Stage-1 cold-start gates Stage-2 REPA target choice via a sweep across {Whisper-L8, Whisper-L16, Whisper-L30} + {w2v-BERT-2 mid (mandatory if vs TRIBE-v2), XEUS L9, PhoneticXEUS L8}. Pre-rank via DiME unsupervised layer-selection (Skean LeCun arXiv 2502.02013) on held-out brain corpus. (v) **FlashAttention-3 kernel + 4 ViT register tokens added alongside parcel latents** (free engineering pins, no verification gate; registers required for clean retroactive PopT-style attention analysis). Also: paper-pitch reframe — WhisperBCI compresses "first iEEG-Whisper alignment" hook → reframe to "first cross-subject-zero-param + Whisper-intermediate-layer."*

*2026-04-30 — artifact/invariance revision after Bijan Pesaran seminar discussion. Raw 2048 Hz voltage is no longer treated as artifact-neutral or the guaranteed Stage-1 default. Stage 1 must run an input-view matrix that includes raw, local/Laplacian raw, local/Laplacian STFT/log-power, local/Laplacian HG/HFA, and CAR+HG. Reference is a first-class transform: it changes the measurement operator and must carry transformed channel/support metadata. High gamma/HFA and local spectral power are the biologically privileged local-population views; raw remains an auxiliary/ablation view unless Stage-0/Stage-1 evidence shows it transfers without increasing artifact decodability.*

*2026-04-30 protocol revision — multiclass is the default task formulation. Pooled multi-source CrossSubject multiclass is the scientific generalization default. S2/trial-4 CrossSubject is demoted to leaderboard parity. The 120-electrode Lite cap is also parity-only; use full/anatomy/random electrode-set robustness where data access permits.*

Strategy anchor: `docs/neuroprobe/plan.md`. Predecessor: `docs/neuroprobe/stage_0.md`. Architectural spec: `memory/project_v14_parcel_token_readout_2026_04_26.md` + `stage_2.md` *(a) `pe2d` → Perceiver IO* (canonical).

## Objective

Train v14 from scratch with no SSL pretraining. Stage 1 is the architecture-as-cold-prior test under two lanes:

1. **Submission lane**: CrossSession multiclass, the official Neuroprobe submit-gate eval mode.
2. **Scientific generalization lane**: pooled multi-source CrossSubject multiclass, training on all allowed source subjects/sessions and testing held-out subjects/sessions.

S2/trial-4-only CrossSubject is reproduced only as a leaderboard-parity cell. Binary tasks are compatibility/threshold-sweep probes, not the default.

## Success criterion (sharp)

**Submit gate**: Multi-class cross-session mean AUROC strictly > 0.611 — strictly greater than Linear Lap+spec on the rebuttal-codebase floor.

Rationale: BrainBERT/PopT cluster sits at ~0.557–0.562 multi-class cross-session; Linear Lap+spec at 0.611 is the ceiling for "alignment without learning" in submit-gate mode. v14's anatomical anchoring is *architectural* — parcel embeddings are part of the cold model. To support alignment-is-the-bottleneck thesis, cold-start v14 must break 0.611 without pretraining.

**Generalization gate**: pooled multi-source CrossSubject multiclass must improve over matched linear/foundation baselines and must not raise subject/session/reference/coverage nuisance decodability. If cross-session improves but pooled cross-subject worsens, the architecture is learning benchmark-specific structure, not patient-invariant cortical content.

**Parity indicator**: S2/trial-4 CrossSubject binary > 0.539 is healthy directionally, but it is not a Stage-1 gate.

## Result decision tree

| Cold-start AUROC (multi-class cross-session + pooled CrossSubject multiclass) | Read | Stage-2 implication |
|---|---|---|
| **Cross-session > 0.611 and pooled CrossSubject improves** | Anatomical priors do alignment cold and transfer beyond the narrow source subject. | Stage 2 chases ≥0.65 submit gate and stronger pooled cross-subject lift. |
| **Cross-session > 0.611 but pooled CrossSubject flat/regresses** | Architecture learned within-subject/session structure or Lite/electrode artifacts. | Do not promote; fix nuisance/invariance before Stage 2 default. |
| **0.557–0.611 cross-session with healthy pooled CrossSubject** | Architecture provides nonlinear-feature lift but anchoring alone insufficient to clear linear floor; SSL load-bearing. | Stage 2 thesis-deciding. |
| **< 0.557 cross-session or nuisance probes high** | Below BrainBERT/PopT cluster or contaminated representation. Structural bug. | Debug — likely loader contract, support cache, `P_emb` wiring, input view, normalization, or eval-mode flag mis-set. |

## Empirical anchors from BarISTA (Oganesian NeurIPS 2025) on BrainTreebank

BarISTA is the closest published architecture to v14, evaluated on BT itself (10 sEEG, 26 sessions, 29.2h SSL pretraining at 2048 Hz raw — same cohort/modality/scale we'd run). Reference: `memory/reference_barista_oganesian_2025_12.md`. Architectural findings transfer where directly comparable; absolute AUC numbers do not (their ~0.86 is binary sentence-onset / speech-production with their own split, not Neuroprobe cross-subject leaderboard split).

| Choice | BarISTA finding on BT | Our position |
|---|---|---|
| Spatial encoding | Parcel-level (Destrieux 148) **>>** channel-level (LPI coords) by +8–10 pp AUC | Validates `P_emb` thesis on this exact dataset. v14's soft Brainnetome ≈ their hard Destrieux, strictly more general. |
| Parcellation granularity | Parcel ≈ lobe (DK ~70) — finer doesn't help in their setting | Informs Stage-0 D.3b vs D.4 expectation. |
| Attention factorization | Combined spatiotemporal **>** factored by +0.8–2.2 pp | Empirically validates v14's combined-attention default *specifically on BT*. |
| Temporal front-end | 5-layer dilated CNN on raw voltage **>** linear projection by +6–9 pp on **binary detection** | **NOT adopted as v14 default.** BaRISTA's win is binary-only and BaRISTA stands alone in FM lineage on dilated stacks. v14 default = single-layer 1D conv (modal FM choice). BaRISTA dilated → mandatory tokenizer ablation cell A. |
| Per-patient params | Zero per-patient + parcel encoding → only ~2 pp held-out-subject degradation | Sets held-out-subject benchmark v14 should match or beat. |
| Width × depth | d=64, 12 layers, ~1M params | Phase-1 default d=32 may under-fit BT. Surfaced as width sweep below. |

**Caveat — atlas family is different.** Their parcel ≈ lobe was Destrieux (gyral) vs DK (lobar). Brainnetome (cytoarchitectonic / functional) is a different family they didn't test. So "BNA-246 vs BT-Tier-1 ≈ parity" remains a Stage-0 D.4 empirical question.

**Caveat — task difficulty mismatch.** BarISTA's binary tasks are dramatically easier than 15-task Neuroprobe panel. "Zero per-patient params is enough" generalizes only if harder linguistic tasks don't require per-patient calibration BarISTA never tested.

## Frozen design commitments (revisit on Stage-0 close)

Architecture spec is canonical in `stage_2.md` *(a) `pe2d` → Perceiver IO*. Stage 1 inherits identically; commitments below are Stage-1-specific.

- **Tokenization**: per-electrode (B-1 path). Per-cell pooling unsupported with variable electrode counts cross-subject.
- **Temporal tokenizer**: **single-layer 1D conv with kernel=patch_size, stride=patch_size** (= "linear patch projection" in ViT terminology). Modal FM-lineage choice. Patch size pinned in first sweep (Charmander 10 ms ≈ 20 samples; BrainBERT-STFT analog 50 ms ≈ 100 samples). **Mandatory tokenizer ablation cells**: A = BaRISTA 5-layer dilated 1D CNN; B = HG envelope @ 200 Hz + linear patch (Cogan home-turf, informed by Stage-0 D.11); C = wavelet (Daubechies-4) + linear patch (MVPFormer); D (optional) = single strided 1D conv kernel 4 stride 4 (Metzger).
- **Architecture**: Perceiver IO with parcel-id-tagged latents + Option Z log-support QK bias + DETR task-attention readout. Pure-voltage electrode tokens. Always include all ~50 parcels regardless of per-subject coverage. Spec: `stage_2.md` *(a) `pe2d` → Perceiver IO* + `memory/project_v14_parcel_token_readout_2026_04_26.md`.
- **Anatomy-enforcement sweep (mandatory)**: 4-cell sweep over how strictly the cross-attention bias enforces anatomy. Brackets the design space for the load-bearing cross-subject scaffold claim.

| Cell | Bias form | Anatomy enforcement |
|---|---|---|
| Hard-mask (new) | `support[i,p] > τ` → 0, else → −∞ | Inviolable (electrode physically cannot route to non-overlapping parcel) |
| Z = `log(support + ε)` softmax (default) | `+ log(support[i,p] + ε)` added to QK, softmax | Soft prior, overrideable by confident QK content |
| L = learned linear init at `softmax(log support)` | Drop softmax; learned linear weights initialized at the softmax(log support) values | Soft init, fully learnable (from source-localization audit) |
| No-constraint baseline | Vanilla cross-attn, no support bias | None — falsifies the entire framing if it ties or beats Z |

Decision rule: Hard-mask wins or ties Z → ship Hard-mask (cross-subject claim becomes ironclad). Hard-mask loses by ≥0.005 → soft prior is doing useful work absorbing fsaverage projection noise; stay with Z and pin ε empirically. No-constraint baseline ≥ Z → architectural commitment falsified, rethink.
- **Readout**: task-attention pool over parcel latents (DETR-style; learnable query per task). Mean pool retained as **diagnostic ablation cell only**.
- **Argmax-hard vs probabilistic support**: inherit from Stage-0 D.3a vs D.3b winner.
- **Temporal**: RoPE temporal-only (multiplicative on Q/K inside attention, NOT additive at input).
- **Attention**: cross-attention electrodes → parcel latents (with `log(support)` bias) + latent self-attention; vanilla, no init bias on latent↔latent. PopT showed attention learns connectivity from data; BNA-conn-init bias rejected as baked prior.
- **Attention stability pins (DeepSeek-V4 read-through, 2026-05-01)**: apply head-wise RMSNorm to attention queries and keys/KV immediately before dot products in electrode→parcel cross-attention and parcel-latent self-attention. This keeps QK magnitude from overpowering the `log(support + ε)` bias and reduces bf16 norm-spike risk. Add a learned per-head attention sink/null option as a Tier-1 ablation (`sink-on` vs `sink-off`): promote only if it improves AUROC or lowers nuisance decodability without hurting held-out sparse-subject behavior. Rationale: weakly covered parcels should be allowed to attend to "no reliable electrode evidence" rather than being forced to softmax over bad evidence.
- **Width and depth**: `d ∈ {64, 128}` swept (default d=64; d=128 stretch). Depth ∈ {4, 6, 8, 12}, default 6. Prior `d=32, depth=3` PS-defaults underparameterized for BT Stage-2 SSL ~40-100h regime.
- **Decoder**: small linear per-task head; multiclass CE by default; binary CE only for parity/threshold-sweep cells. Val early-stopping on Neuroprobe val split.
- **Readout layer**: task-attention pool over **final** parcel-latent layer (default). **Tier-1 ablation cell**: sweep L ∈ {final, mid, mid-1} per PE-Core "best embeddings hidden in intermediate layers" hypothesis. If mid > final by ≥0.005, switch Stage-2 readout default.
- **Optimizer**: AdamW with cosine schedule (Stage-1 default for verification baseline). **Schedule-Free AdamW verification ablation**: same architecture / data / compute, β=0.9, LR ~2× cosine-tuned, two seeds. Adopt as Stage-2 default if Schedule-Free wins by ≥0.005 OR ties on multi-class cross-session AUROC (anytime-stopping advantage). **Muon hybrid optimizer cell**: Muon for matrix parameters only; keep AdamW for parcel embeddings, M-slot embeddings, task queries, RMSNorm/LayerNorm weights, scalar gates, and heads. Promote only if it improves AUROC or wall-clock/step efficiency without raising nuisance probes.
- **Mixed precision**: **bf16 default** (replaces fp16; field-convergent across DINOv3 / SigLIP 2 / PE-Core / OpenCLIP-scaling / Muon).
- **Seeds**: 2 (expand to 5 if 2-seed gap > 0.005).
- **Preprocessing / input views**: no single trusted default until Stage-0 V1-V6 + D.11/D.12/D.13 close. Stage 1 must run the input-view matrix below. Raw 2048 Hz voltage is an auxiliary/ablation view, not artifact-neutral. Local/Laplacian spectral or HG/HFA views are biologically privileged because they better match local population firing and reduce common-mode/reference artifacts. CAR+HG remains the Cogan-convention control. Reference transforms must export their operation, virtual-channel coordinates, transformed support, and bad-channel provenance; a rereferenced virtual channel must not be silently treated as the original physical contact.

| View | Transform | Why it is included |
|---|---|---|
| Raw | native 2048 Hz voltage | preserves all information; matches existing Neuroprobe models; highest artifact burden |
| Local/Lap raw | within-shaft/grid local rereference, no spectral transform | isolates reference/common-mode benefit without spectral prior |
| Local/Lap STFT/log-power | local rereference + spectral power | closest to winning Neuroprobe linear baseline mechanism |
| Local/Lap HG/HFA | local rereference + 70–150 Hz envelope/log-power | biologically privileged local population-firing view |
| CAR+HG | whole-array CAR + HG | Cogan-convention control; diagnostic for legacy comparability |

Decision rule: prefer the simplest view that improves or preserves task AUROC while lowering subject/session/reference/coverage decodability. Raw can remain default only if it wins or ties the local spectral/HG views and does not increase artifact probes.

### Normalization Matrix

Stage 1 must treat normalization as part of the model, not a harmless preprocessing detail. The Neuroprobe rebuttal reports that BrainBERT/PopT-style within-window z-scoring can remove between-window amplitude information and materially reduce decoding.

Run the minimal normalization matrix on the leading input views:

| Scope | Definition | Role |
|---|---|---|
| Train-set/global | fit mean/std on train split only; apply to val/test | default candidate; preserves relative window amplitude inside the dataset |
| Session/recording | fit per recording/session with split-safe statistics where possible | tests device/session gain correction without collapsing each clip |
| Window-local | z-score each 1 s or 3 s model input independently | anti-control; expected to erase useful amplitude/power |
| Robust train/session | median/MAD version of train or session scaling | optional if outliers dominate V1-V6 QC |

Decision rule: window-local normalization cannot become default unless it wins task AUROC and does not reduce amplitude-sensitive tasks or increase nuisance decodability. If train-set/global and session/recording tie, prefer the one with lower subject/session decodability.

### Temporal Scale And Anchors

The benchmark evaluates 1 s windows, but several current FMs were pretrained on fixed 5 s clips. Stage 1 should not silently inherit either timescale.

Run:

- 1 s eval-shaped crops as the primary cell;
- 3 s context crops as a Stage-2 compatibility check;
- 5 s crops as an anti-control for BrainBERT/PopT-style fixed-window mismatch.

Add a small anchor-jitter robustness cell around Neuroprobe's `[0, 1]` s word-onset window. Treat starts between about `-0.375` and `+0.125` s as an in-band robustness region, based on the rebuttal. Shifted-window artifact controls should use starts outside that band; success far outside the band is leakage risk, not cortical decoding.
- **Augmentation**: channel-dropout, time-warp ±5%, mixup. All legal under protocol and underused on leaderboard.

## Open questions (defer to results)

- **Spatial encoding beyond `P_emb` (Experiment #4)**: Fourier features on MNI coords, learned distance-bias attention (MV-BrainFM per-head Gaussian).
- **Fixed vs learned `P_emb` (Experiment #2)**: identity-init frozen, anatomy-init frozen, or learned. Test only if cold-start ≥ 0.611.
- **Per-subject embeddings (Experiment #3)**: now testable under pooled multi-source CrossSubject, but still conflicts with the zero-per-subject claim. Use only as an anti-control unless zero-param v14 is clearly below the nuisance floor.
- **Shaft-aware attention (Experiment #9)**: depth contacts on same shaft are physically adjacent (~3.3 mm) but can land in different BNA parcels. Add shaft-aware mask using BT name stems if cold-start sits in 0.557–0.611. BT/D-cohort-specific.
- **Width × depth sweep**: BarISTA empirical sweet spot d=64, 12 layers, ~1M. v14 Stage-1 default raised from earlier d=32/depth=3 to d=64/depth=6. Sweep d ∈ {64, 128} × depth ∈ {4, 6, 8, 12} if cold-start sits below 0.611.
- **Re-referencing sweep (mandatory)**: reference is part of the measurement operator, not cosmetic preprocessing. Local/Laplacian references are the physics-matched candidates: they strip probe-shared instrumentation noise and distant/common-mode fields while preserving focal local population activity better than whole-array CAR. Compatible with source-separation framing: they clean the noise/reference term in `v[i,t] = Σ_p L[i,p]·s[p,t] + noise`, but they also transform `L`, so transformed support/virtual-channel metadata is required.

| Cell | Operation | Assumption |
|---|---|---|
| None (raw auxiliary) | Whatever Neuroprobe `__getitem__` ships (notch + bandpass) | Common-mode / ground noise unaddressed; artifact-risk baseline |
| Per-probe (new) | Per timestep, subtract mean across contacts on same shaft/grid: `v_clean[i,t] = v[i,t] − mean_{j∈probe(i)} v[j,t]` | Probe-shared noise is hardware-level fact (no neural-source bake) |
| CAR (whole-brain mean) | Per timestep, subtract mean across all contacts patient-wide | Neural signal uncorrelated brain-wide (false for coherent oscillations) |
| Laplacian (optional) | Subtract mean of nearest-neighbor contacts | 2D-grid topology / valid neighborhood; awkward for sEEG depth shafts |

Decision rule: per-probe wins by ≥0.005 → adopt as default (cleanest physics, no baked neural prior). Per-probe wins by ≥0.02 → architectural finding; consider learnable per-probe re-reference as an input-layer module in Stage-2. Per-probe ties or loses → noise is being absorbed downstream by LayerNorm in the conv tokenizer + cross-attn; current "no re-reference" stays.

**Caveat to test**: if a probe sits entirely within one functional region (e.g., 16-contact shaft fully inside STG), the per-probe mean partially captures that region's neural source, and subtracting it removes wanted signal. Mitigation if hit: skip "active" channels from the mean, or use robust median instead of mean. For the cell, simplest is full per-probe mean — measure if it visibly hurts the obvious patients, then iterate.

## Mandatory Artifact Probes

Every Stage-1 run exports the task score and an artifact battery from the same frozen backbone/readout layer:

- subject/session ID decoding from parcel/global latents;
- coverage-mask decoding from parcel/global latents;
- reference/input-view decoding when multiple views are trained;
- pre-stimulus or shifted-window label decoding where the split permits it;
- reaction-time or word-position proxy decoding for timing leakage;
- line/common-mode summary decoding from low-level tokens.

An input view or architectural cell only advances if it improves task performance without increasing artifact decodability. A high task AUROC with higher subject/coverage/reference decodability is not evidence for a patient-invariant cortical model.

## Evaluation Lanes

Stage 1 reports every serious architecture/input-view cell in four lanes:

| Lane | Task form | Source set | Role |
|---|---|---|---|
| CrossSession multiclass | official multiclass | same subject, other session | submit-gate score |
| Pooled CrossSubject multiclass | official multiclass | all allowed source subjects/sessions except held-out target | scientific generalization default |
| S2-only CrossSubject | official v1 binary and multiclass if available | Subject 2 Trial 4 only | leaderboard parity / historical comparison |
| Threshold/regression robustness | binary threshold sweeps plus regression/R2 where labels are continuous | same split as the winning lane | checks task-form fragility |

If upstream `include_all_train_subjects=True` becomes available, first verify whether it is pairwise source-subject folds or true pooled N-to-1 training. Pairwise all-source is useful robustness; pooled N-to-1 is the default we need for architecture selection.

The 120-electrode Lite cap is used for leaderboard parity. Scientific robustness should include random-120, anatomy-120, and full/uncapped electrodes where public data permits. If only Lite is available, all claims must say Lite-selected electrodes.

## Frozen Stage-1 split contract

Operationalizes the four lanes above. Source of truth for what counts as a *matched protocol* comparison; any cell that deviates must say so explicitly. Promoting a cell to a freeze (L.* winner, default architecture commitment) requires passing every anti-control row.

| Item | Contract |
|---|---|
| **Cohort** | 12 BT Lite (subject, trial) sessions: (1,1) (1,2) (2,0) (2,4) (3,0) (3,1) (4,0) (4,1) (7,0) (7,1) (10,0) (10,1). 6 subjects × 2 trials each. |
| **Upstream pin** | `azaho/neuroprobe@c7b955b0a31464f4a5eec3f3bd78ff29841d61ac`. DCC: `/work/ht203/repo/neuroprobe_upstream/`. |
| **Window** | Anchor `[0, 1]` s post word-onset. Anchor jitter `[-0.375, +0.125]` s is in-band per *Temporal Scale and Anchors*; outside that band is leakage control. |
| **Tasks** | 15 official multiclass: `onset speech volume delta_volume pitch word_index word_gap gpt2_surprisal word_head_pos word_part_speech word_length global_flow local_flow frame_brightness face_num`. |
| **Electrodes** | Lite cap (≤120/subject). Any random/anatomy/full robustness cell must be tagged `Lite=False` and reported separately — not folded into Lite-cap means. |
| **Normalization** | `train_set_fixed` (L.1 winner, frozen 2026-05-08). Per-channel mean/std fit on TRAIN split only; applied to val/test. Never refit cross-split. |
| **Reference × view** | `shaft_laplacian × stft_abs` (R4xI2, L.2 winner, frozen 2026-05-09). Other ref/view cells must be tagged. |
| **Seeds — linear** | Single-seed sufficient. Verified byte-identical across seeds 42/43/44 (`reports/neuroprobe_stage0_l2_seed_robustness_2026_05_10/`); sklearn lbfgs is deterministic and the upstream `rng.choice` label cap does not fire at 15 rows × 15 tasks. |
| **Seeds — non-linear** | 2 seeds default. Expand to 5 if 2-seed gap > 0.005. |
| **Eval-stat split-safety** | All scalers/standardizers/normalizers fit on TRAIN only and applied to TEST. No exceptions. |

### Lane 1 — CrossSession multiclass (submit gate)

| Item | Contract |
|---|---|
| **Definition** | Per (subject, trial) target, train on the *other* trials of the same subject; test on the target. Same-subject only — never cross-subject in this lane. |
| **Folds** | One per (subject, trial) test pair = 12 folds (= cohort size). |
| **Source code** | Upstream `BrainTreebankSubjectTrialBenchmarkDataset` with `split_type=CrossSession`. |
| **Aggregation** | Per-task AUROC computed inside the upstream dataset; cell-level mean = mean over 15 tasks; sweep-level mean = mean over 12 sessions. Report `mean ± sd` across the 12 sessions. |
| **K-fold robustness** | 5-fold within-subject when ≥ 3 trials available, reported as a robustness column NOT as the submit number. |
| **Chronological robustness** | Within-subject oldest-trial → newest, reported as a robustness column. |

### Lane 2 — Pooled multi-source CrossSubject multiclass (scientific generalization)

| Item | Contract |
|---|---|
| **Definition** | Leave-one-subject-out (LOSO). Train pool = all sessions of all subjects EXCEPT the held-out target subject. Test = all sessions of the held-out subject. |
| **Folds** | 6 folds: hold out S1, S2, S3, S4, S7, S10 in turn. |
| **Source pools** | Stage-1 (no SSL): pool = remaining 5 BT subjects. Stage-2+: pool extends to legal external corpora (Cogan PS/lex/sEEG, ds003688, Podcast, NeuroListen) per `plan.md §Shared frame #8`. Every cell must list which sources are in the pool. |
| **Subject overlap** | Strict zero-overlap. A subject in pool MUST NOT appear in held-out for the same fold. Stage-2 pretrain: subjects used for SSL pretraining MUST NOT appear in any Stage-1 held-out fold — pre-bake an `exclude-and-retrain` sister run per `memory/feedback_pre_bake_data_leakage_defense.md`. |
| **Aggregation** | Per-fold AUROC = mean over 15 tasks averaged over the held-out subject's sessions. LOSO `mean ± sd` across 6 folds. |
| **Pairwise vs pooled** | Upstream `include_all_train_subjects=True` is *pairwise all-source* (each train subject contributes its own fold), NOT pooled N-to-1. Pooled N-to-1 is implemented locally; we do not inherit the upstream flag for this lane. |

### Lane 3 — S2/trial-4 CrossSubject (leaderboard parity)

| Item | Contract |
|---|---|
| **Definition** | Single fixed split: train = S2 trial 4 only; test = held-out subject. Reproduces leaderboard cell exactly. |
| **Source code** | Upstream `split_type=CrossSubject` with v1 binary tasks for parity rows; multiclass version when available upstream. |
| **Role** | Parity-only. Numbers reported but NOT used for architecture selection or freeze decisions. |
| **Caveat** | S2/trial-4-only is a narrow protocol; results from this lane do not generalize to Lane 2 verdicts. |

### Lane 4 — Threshold / regression robustness

| Item | Contract |
|---|---|
| **Definition** | For each Lane-1 winner cell: rerun with binary-threshold sweeps on continuous-label tasks + R² regression on continuous labels. Same train/test split as the winning lane. |
| **Role** | Task-form fragility check. A cell that wins multiclass but loses binary across multiple thresholds is task-form-overfit and not promoted. |

### Anti-control splits (mandatory for any freeze)

| Anti-control | Definition | Pass criterion |
|---|---|---|
| **Shifted-window** | Anchor `[-1.5, -0.5]` s pre-onset (well outside in-band region). | Must drop by ≥ 0.05 vs onset window. If not, signal is contaminated. |
| **Within-session shuffle** | Shuffle trial labels randomly inside each session, retain split structure. | Must regress to chance (~0.5 binary, ~0.067 = 1/15 multiclass). |
| **Subject ID nuisance** | Decode subject identity from the same backbone features. | Subject decodability ≤ 0.6 OR Δ vs baseline ≤ 0.05. Higher = patient-fingerprint contamination. L.5-P1 will produce per-cell numbers. |
| **Stimulus overlap** | V0.x per-task median upper-bound overlap. | ≤ 0.50; tasks above must be reported separately as *stimulus-recognition-confounded*. Upper-bound (full-words_df) audit done 2026-05-10 (`reports/neuroprobe_stage0_v0x_stimulus_overlap_2026_05_10/`); per-task audit on labels actually used by `BrainTreebankSubjectTrialBenchmarkDataset` in flight under `reports/neuroprobe_stage0_v0x_stimulus_overlap_per_task_2026_05_10/`. |

### Reproducibility pins

- Upstream Neuroprobe: `c7b955b0a31464f4a5eec3f3bd78ff29841d61ac`.
- L.1 normalization winner: N1 = `train_set_fixed`. Frozen 2026-05-08. See `reports/neuroprobe_stage0_l1_normalization_2026_05_05/freeze_analysis.md`.
- L.2 reference × view winner: R4xI2 = `shaft_laplacian × stft_abs`. Frozen 2026-05-09. See `reports/neuroprobe_stage0_l2_neuralset_*/freeze_analysis.md`.
- L.4 anchor robustness: confirmed within ±0.005 noise band 2026-05-10 (`reports/neuroprobe_stage0_l4_anchor_2026_05_09/anchor_robustness.md`). Full window sweep W2–W5 in flight 2026-05-13 (`reports/neuroprobe_stage0_l4_window_sweep_w2_w5_2026_05_13/`).
- L.2 seed-robustness: linear baseline seed-invariant 2026-05-10 (`reports/neuroprobe_stage0_l2_seed_robustness_2026_05_10/seed_robustness.md`).
- L.3 filtering: frozen as no-op 2026-05-13 (12/12 sessions). F1 (60/120/180 Hz notch) Δ +0.0007, F2 (F1 + 0.5 Hz HPF) Δ +0.0016, F3 (F1 + 1.0 Hz HPF) Δ +0.0015 — all within ±0.005 noise band vs L.2 winner. See `reports/neuroprobe_stage0_l3_filtering_2026_05_10/filtering_analysis.md`.
- L.4 norm × view interaction: greedy hill-climb safe 2026-05-10. Max |interaction residual| = 0.0008 across 8 cells (refs={shaft_laplacian, bipolar} × views={stft_abs, hg_envelope} × norms={train_set_fixed, train_set_scale_only}). See `reports/neuroprobe_stage0_l4_norm_view_interaction_2026_05_09/interaction_analysis.md`.
- Tier-C CrossSubject parity: C.0 baseline 0.5310 ± 0.0235; C.2 (L.1+L.2 winners joint) +0.0082 vs baseline 2026-05-10 (`reports/neuroprobe_stage0_tier_c_cross_subject_2026_05_09/tier_c_analysis.md`). C.3 (bipolar × stft) and C.4 (shaft_lap × HG envelope) in flight 2026-05-13 (`reports/neuroprobe_stage0_tier_c_alts_2026_05_13/`).

---

## What Stage 1 explicitly does NOT do

- No pretraining (Stage 2 territory).
- No multi-task joint head (per-task linear is the protocol).
- **No within-session evaluation.** Cross-session multiclass is the submit criterion; pooled CrossSubject multiclass is the scientific generalization criterion.
- No DK-atlas anchor.
- No electrode selection beyond BT Lite cap in leaderboard-parity cells.
- No labels other than the 15 official eval tasks.
