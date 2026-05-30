# Data Efficiency — v14 Neuroprobe

*Sole living doc for the data-scaling / data-efficiency STRATEGY and literature map. Lever specs (L0–L4) are promoted into named cells in `docs/neuroprobe/ablations.md` before any dispatch, mirroring its no-sweep-without-a-cell discipline; this doc then cites those cell IDs. We iterate here; do not fork. Started 2026-05-28. Direction memo: [[project_data_efficiency_primary_lever_2026_05_28]]. Plan it serves: `docs/neuroprobe/plan.md`. Ablation menu it feeds: `docs/neuroprobe/ablations.md`.*

## Why this doc exists

Architecture (B19→B32) and the SSL loss surface (B31) are locked. The data axis got almost none of that rigor. That is where the unspent leverage is. v14 sits in the **data-constrained / compute-rich** regime — ≤30M param cap, ~9 BT subjects, minutes/subject, DCC compute cheap relative to data. The data-constrained-scaling field (anchor: Kim et al. *Pre-training under Infinite Compute*, arXiv 2509.14786, ICLR 2026 Oral) maps onto this regime and is under-explored for iEEG-SSL-for-transfer.

## Core insights from the anchor paper (Kim et al. 2509.14786)

Faithful summary of what the paper actually claims. It optimizes **i.i.d. held-out LM loss** on ~200M DCLM tokens — the "transfer not loss" rule in the next section is *our* adaptation, not the paper's claim. (Asymptotes 3.43/3.34/3.17 and ensemble HPs from the paper's scaling-law fits; verified against arXiv HTML 2026-05-28.)

1. **Naive scaling overfits.** More epochs or more parameters on fixed data eventually *raises* loss (U-turn). Standard weight decay (0.1) is inadequate in this regime. This is the problem the rest of the recipe solves.
2. **Regularized parameter scaling.** Tune weight decay far higher (paper found ~**30× standard**, ≈3.0) and loss becomes *monotone* in N, fitting a power law L̂ = 0.05/N^1.02 + 3.43 (asymptote **3.43**). → **2.29× more data-efficient** than the standard recipe. ("Monotone in N" is an i.i.d.-loss property; under the gate objective the analogous claim is unproven and may itself U-turn — see the transfer rule. The asymptote triple is LM-only and is NOT a template for any iEEG fit.)
3. **Ensemble scaling (its own axis).** Train K *independent* models, average logits. K→∞ reaches asymptote **3.34** — beats single-model parameter scaling (3.43) on its own. Ensembles want *different* hyperparameters than single models (~**2× epochs, 0.5× weight decay**).
4. **Joint scaling composes.** Parameter + ensemble scaling together (N,K→∞) → asymptote **3.17**; the two axes stack. → **5.17× less data** than baseline for matched loss.
5. **Distillation collapses the gain into a fixed budget.** An 8-model ensemble distilled into a student **8× smaller** retains ~**83%** of the ensemble's improvement; self-distillation also helps. This is what lets you bank ensemble/parameter gains under a fixed param cap.
6. **Data scaling laws.** The efficiency multiplier is roughly *constant across data scales* (exponents ~0.23–0.24), so the recipe's benefit persists as data grows.
7. **It shows up downstream too.** Best ensemble +>9% / distilled +~7% on PIQA/SciQ/ARC-Easy; continued-pretraining on math hits 17.5× data efficiency. (LM numbers — directional only for us.)

Mental model carried into the levers below: **same architecture, more GPU** — regularize → ensemble (K independent) → distill into the capped artifact. None of it adds loss terms or input plumbing.

## The one rule that governs everything here

**Optimize for cross-subject / cross-session TRANSFER (the submission gate AUROC), not i.i.d. SSL recon loss.**

The entire data-constrained-scaling literature optimizes i.i.d. held-out loss. v14's gates are OOD transfer. The two diverge — not speculative:
- arXiv 2503.19206 (Catastrophic Overtraining, CMU): more pretraining tokens can *hurt* post-fine-tune downstream even as pretraining loss improves (OLMo-1B @3T >2% worse than @2.3T after instruction-tune). Its mechanism is rising fine-tune/parameter sensitivity; under v14's **frozen-probe** headline eval (plan.md Path A) the operative loss↔gate divergence is more plausibly subject/session overfit. Keep it as directional support for selecting on transfer, name the v14-specific mechanism — the paper itself says sensitivity is not limited to fine-tuning, so don't overstate that it's absent under a frozen probe.
- arXiv 2602.11137 (Han et al., Harvard, *Weight Decay Improves Language Model Plasticity*, Feb 2026): higher weight decay → *worse* pretraining loss but *better* fine-tune transfer.

Consequence: every knob below is scored on the gate, never on recon loss. A data-efficiency win that lowers recon loss but moves the gate backward is a loss.

Submission gate (from MEMORY.md §Status): ≥ **0.667** CSession multi-class AND ≥ **0.628** CSubject binary AUROC, ≥ 4 tasks, ≤ 30M params.

## Transfer-selection probe (frozen) — load-bearing for every lever

L0 and "score on the gate" both need one cheap, trustworthy transfer signal. Define it once, reuse everywhere:
- **Probe** = the existing Phase-4 Path-A1 readout (plan.md §Phase 4 / ablations.md `R-no-time-pool` default): frozen PMA-k=1 → flatten(T·d) → per-task Linear, so selection matches the submission-scored object.
- **Score** = both gate prongs on held-out (D.14 pooled CrossSubject multiclass + CrossSession multiclass); select on **CrossSubject** per Shared-frame commitment 1.
- **Cadence** = piggyback the 10k-step MON-* held-out pass for the cheap curve; run the full D.14 gate probe only on the top-2 checkpoints. Probe interval to-be-justified by a micro-cost run, not pre-committed.

**Scoring contract**: a lever win = ΔAUROC on this probe (CrossSubject headline + CrossSession submit lane), declared load-bearing only at **Δ ≥ 0.02** (ablations.md effect-size threshold), using the frozen stat method in `ablations.md §Statistical method` (bootstrap CIs, paired Wilcoxon, BH correction within sweep, ≥3 seeds). This is the stop/go rule for L0–L4.

## Sequencing (hard dependency)

1. **One clean baseline first** — three sub-gates, none assumed:
   - (a) the BTWordEvents fix merged + its parity test green on DCC ([[project_btwordevents_split_class_imbalance_bug_2026_05_15]]), an explicit gate.
   - (b) the baseline must cover the scoring surface — at minimum one **multiclass CrossSession** AND one **CrossSubject** number on ≥1 task. BTWordEvents alone is binary (speech/nonverbal) and does NOT establish the multiclass gate prong.
   - (c) name which baseline each lever forks from: the Stage-1 cold-start number unblocks pipeline confidence, but L0/L1/L2 fork from a clean **Stage-2 SSL checkpoint**. A reader must not start L1 from the SSL-less cold-start.
2. Cheap data-efficiency sweeps (L0 / L0.5 / L1 / L1.5 / L2).
3. The expensive bet (L3 ensemble→distill).

Scaling-law ambition does **not** jump this queue.

## Lever menu (ranked: cheap + gate-moving first)

All levers are **idle pending the clean baseline (§Sequencing)**; per-lever Status reappears only when one advances to ready/running/done.

### L0 — Checkpoint-select on transfer, not loss
- **Cost**: ~free (change the selection criterion only)
- **Backing**: 2503.19206, 2602.11137
- **Action**: pick the SSL checkpoint by the transfer-selection probe above, not by min recon loss. Expect the best-transfer checkpoint to have non-minimal recon loss.

### L0.5 — Cross-subject input alignment (label-free)
- **Cost**: very low
- **Backing**: He & Wu, arXiv 1808.05464 (*Transfer Learning for BCI: A Euclidean Space Data Alignment Approach*, IEEE TBME 2020) — Euclidean Alignment whitens per-subject covariance, unsupervised, needs no target labels.
- **Action**: per-subject covariance-whitening / reference-mean alignment before SSL and at held-out-subject eval (fits the no-test-labels constraint); paired cell vs the current Nv14 per-channel robust-z (plan.md), which is a per-channel affine, not cross-subject distribution matching.
- **Open**: EA assumes channel-aligned montages, which iEEG lacks — align in the parcel-routed latent space, not raw channels. (EA-revisit 2502.09203 unconfirmed — left out until checked.)

### L1 — Regularization × epochs sweep, scored on the gate
- **Cost**: cheap (Lite scale)
- **Backing**: anchor (WD ~30× standard for LM — do NOT import the constant), 2510.04071 (Gao et al., *What Makes Diffusion Language Models Super Data Learners?*, Oct 2025 — stochastic regularization: masking, dropout, weight decay drives multi-epoch data-efficiency), 2602.11137
- **Action**: 1-pass coordinate-descent, ranges to-be-justified: (1) init at the locked B01 optimizer config; (2) order WD → epochs → LR; (3) per-axis ~4–5-point log-spaced bracket whose range a ~1-GPU-h Lite micro-probe sets by locating the loss/AUROC U-turn — do NOT seed near 30×/≈3.0; (4) one round at Lite, re-sweep WD only if the epoch move > 2×; (5) stop at ΔAUROC < 0.02 CrossSession; (6) defer the ensemble-specific (WD, epochs) re-tune to L3 — L1's single-model optimum is NOT L3's; (7) keep mask fixed (B03/B30); mask×WD is P2-if-L1-wins. Also sweep stochastic-depth/DropPath (1603.09382) + attention/FFN dropout in the same descent.
- **Key caveat**: the gate-optimal WD may differ in sign and magnitude from the recon-loss-optimal WD (2602.11137). Sweep WD by gate AUROC; report the recon-loss optimum only as a control. The 30× was fit to i.i.d. LM loss and carries no prior on gate-optimal WD for low-SNR iEEG.

### L1.5 — Coreset / data-pruning (scarce-regime variant)
- **Cost**: low–medium
- **Backing**: Sorscher et al., arXiv 2206.14486 (*Beyond neural scaling laws: beating power law scaling via data pruning*, NeurIPS 2022 Outstanding) — pruning can break power-law scaling toward exponential, with a label-free self-supervised difficulty metric.
- **Action**: rank BT clips by the paper's self-supervised difficulty metric (no labels at SSL time); in the SCARCE regime KEEP easy/typical clips and down-weight outliers (the paper's keep-hard prescription *reverses* below a size threshold; ~9 BT subjects is squarely scarce). Score on the gate.
- **Open**: whether the easy-keep prescription survives the non-i.i.d. transfer objective — tie to "the one rule".

### L2 — Mixture-repetition + learned mixture weights
- **Cost**: cheap
- **Backing**: 2605.12715 (Sedova et al., Apple, May 2026) — scarce target reusable **15–20×** in a generic mixture vs ~4× alone (text-LM cross-entropy result; transfer caveat below). 2305.10429 (Xie et al., *DoReMi*, NeurIPS 2023) — proxy-model group-DRO reweighting, +6.5pp / 2.6× fewer steps, downstream-agnostic.
- **Action**: BT is the ~7.3%-share scarce target inside the joint cohort under the **B02-locked Phase-1 sampler** (α=0.5 hierarchical over valid-bin-electrode-hours; SWEC ~50% / AJILE12 ~27.7% / D ~15% / BT ~7.3%; plan.md §Pretraining corpus / v14_blockers.md §B02). L2 is the **gate-scored re-run** of the existing `R-alpha-*` + `R-include-bt-floor-*` cells (ablations.md §4b/§5) — the contribution is scoring on the transfer probe, NOT a parallel sweep. Minimal grid (ranges to-be-justified): α ∈ {0.5 B02, 0.3, 0.7} × BT-floor ∈ {none, one mid value}, Lite, scored on the L0 probe. Pre-sweep, state via the HB02 clip math whether current BT reuse is already in/below 15–20×. Optionally run a Lite DoReMi proxy (group-DRO over the 4 corpora) to propose weights and sanity-check vs B02 — its worst-case objective is closer to the transfer goal than perplexity-minimizing mixtures.
- **Key caveat**: BT is BOTH the scarce target AND the eval subject pool, so repetition trades data-efficiency against **subject-overfit** on the gate — the exact failure the project documents (BrainBERT-trained 0.522 < untrained 0.527 cross-subject, plan.md). 2605.12715 itself warns too-much-target → eventual overfitting. Sweep BT repetition **downward as well as upward**, monitoring MON-MASK-004 (subject-ID nuisance probe) at each level. Treat 15–20× as a text-LM ceiling that is prior-adversarial here — not a starting point.

### L3 — Ensemble → distill under the 30M cap
- **Cost**: expensive (post-baseline; the expensive bet — not "the gate-closer" until a pilot proves ensembling helps on iEEG at all)
- **Backing**: anchor (ensemble→student 8× smaller retains ~83% — i.i.d. LM loss; ensemble→OOD-transfer retention is unmeasured and capacity-dependent, the anchor's student was far larger than ≤30M), 2502.08606 (Distillation Scaling Laws — distill wins in the "teacher reused / many students" regime), 2404.03263 (small distilled model can match pretrain-then-finetune), 2407.04600 (self-distillation rounds)
- **Action**: the param cap is on the *artifact*, not training compute. Train K seeds → ensemble → distill into one ≤30M student.
  - (a) **Pilot first**: K∈{2,3}, measure gate-AUROC retention before committing K full runs.
  - (b) **Distill representations, not logits**: logit-averaging (insight 5) doesn't transfer to continuous EMA-teacher feature targets — target = ensemble-mean of frozen-encoder PMA-k=1 readout features under a feature-regression loss in the existing L1/Smooth-L1 family. (Optional: distill the ensemble *distribution*, not just the mean — Park et al. ICML 2025 flow-matching ensemble-distill — since diversity is what helps OOD.)
  - (c) **Diversify members**: under v14's frozen arch (B19→B32) + loss (B31) + tiny fixed cohort, members are correlated and the ensemble ceiling shrinks vs the anchor's free-hyperparameter large-pool setting. Deliberately diversify the unfrozen axes (seed, the anchor's 2× epochs / 0.5× WD per member, per-member data-subsampling).
- **Budget** (see Open-Q3): K = floor(L3-envelope / ~600 H100-h per run, HB02 median). The ~5,000 H100-h base budget is razor-thin ([[project_v14_hb02_compute_estimate_2026_05_23]]), so a K=4 independent-seed ensemble adds ~2,700 H100-h with no identified headroom. First L3 variant = the ~1× path the anchor also validates: **snapshot-ensemble** K reduced-step checkpoints from one long run + self-distillation, NOT K× independent seeds. Route any full-K independent ensemble to AWS H100 spot — the 8× Ada 5000 allocation is PCIe-only and collapses past ~4 GPUs ([[project_v14_compute_dcc_cogan_gpu_allocation_2026_05_23]]).

### L4 — Multi-teacher / frozen-teacher (P1, arch-adjacent — defer)
- **Cost**: arch-adjacent (do NOT reopen the loss surface right after B31)
- **Backing**: 2603.04478 (Li et al., Oxford — **scalp EEG** multi-teacher distill, TUEG pretrain, 10-20 montage; matches SSL with **25%** data — scalp→iEEG modality gap makes this DIRECTIONAL-only), 2509.24317 (Li et al., Apple, *Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers* (SALT) — student robust to teacher quality → favor student compute)
- **Action**: v14 **already realizes part of this lever via Whisper-L8 distillation**, so L4's incremental claim is ADDITIONAL teachers beyond Whisper, not a fresh 25% multiplier. Any L4 teacher change reopens the B31 2-term loss + B26/B27 EMA τ=0.999 full-input teacher contract, and must run head-to-head against the existing B31 P0 sisters (`R-add-m3-loss` / `R-add-utterance-loss`) rather than landing as a new default. Hold until L0–L3 land.

### Candidate levers (captured, not yet scoped)
- **Test-time self-supervised adaptation (TTT)** — Sun et al., arXiv 1909.13231 (ICML 2020). v14's masked-prediction SSL *is* TTT's auxiliary task; run a few SSL steps on the held-out subject's unlabeled data before probing. Conflicts with plan.md Shared-frame #3 (zero per-subject deploy params) unless framed as **transient** (adaptation discarded after the subject) — compatibility is the open question.
- **Active-learning-lite acquisition** — when new sEEG is acquired (Phase-2 fallback, plan.md), prioritize subjects whose anatomy populates currently-low-coverage / orphan DK parcels (the 80-slot reserve), using the §11 per-parcel coverage diagnostic (ablations.md) as the selection signal. Principle, not a sweep.

### Explicitly not doing
- **PEFT / per-subject LoRA adapters** — excluded by the zero-per-subject-params deployment commitment (plan.md Shared-frame #3). The only sanctioned per-subject capacity is the S1-C linear readout cell (ablations.md §2), shipped only if it beats S1-A by ≥0.02.

## Augmentation as a data-efficiency lever (scoped)

Rommel et al., arXiv 2206.14483 (*Data augmentation for learning predictive models on EEG: a systematic comparison*, J. Neural Eng. 2022) — up to **45%** accuracy gains in low-data EEG regimes, **no single best augmentation** (task-specific). Reconciliation with the compute-lavish "no input aug" stance: ref-aug (B32, `R-ref-aug-3-cell`) IS the one sanctioned neural-time-series augmentation; broader time/frequency/sensor augmentation stays deferred, justified by Rommel's task-specificity finding. The live falsifier is the existing `R-ref-aug-3-cell` (P1, BT-Lite, 4 kill criteria, REF-01).

## Open strategic questions

1. **Can a 9-subject cohort fit a usable scaling law?** Likely 2–3 noisy points, not a textbook power law. The *recipe* transfers; formal law-fitting may not. Methodology if attempted: (a) fit the data axis in **valid-bin-electrode-hours** (already computed for the B02 sampler, plan.md) not subject-count, for denser points; (b) bootstrap CIs on any fitted exponent, treat as directional; (c) the realistic in-domain bar is the iEEG/EEG scaling literature (Banville 2025; Hong 2024 ECoG — IDs to verify before citing, see below), NOT the LM anchor exponents.
2. **Transfer-selection probe** — defined above; confirm the probe interval via a micro-cost run.
3. **L3 compute budget** — see L3 Budget; decision surfaced below.

## Novelty positioning (honest)

Distillation-for-EEG is **not** greenfield — Oxford multi-teacher EEG distill (2603.04478, 2026), Neuroprobe itself (2509.21671). Do not claim distillation-invention. The open, distinctive slice to claim: the **regularize→ensemble→distill recipe, selected on cross-subject transfer, applied to iEEG-SSL** — with whatever data-efficiency the 9-subject cohort can actually support (a few points / recipe-transfer, **NOT a fitted multi-decade scaling law**; see Open-Q1). Reserve "scaling-law" language for any axis where ≥4–5 points are actually fit. (REVE 2510.21585 is a masked-autoencoding big-data EEG FM, not a distillation paper — it belongs in the opposite-regime contrast, not the not-greenfield list.)

## Verified literature map

All arXiv IDs checked against the abstract page (2026-05-28 research pass + 2026-05-28 audit). **Per the IRONCLAD rule, before any L2/L3/L4 number (15–20×, 25%, 83%, 5.17×, 2.29×) migrates into paper prose, re-verify externally.** Two leads `2605.01640` (Lovelace et al., Cornell) and `2310.04415` (D'Angelo et al., EPFL) DO resolve to their claimed titles but no number is drawn from them — kept flagged as conservative. PDF-confirm SCOTT/MIM-JEPA `2502.18056` top-1 figures before citing. In-domain scaling precedents Banville `2501.15322` / Hong (eLife) and EA-revisit `2502.09203` are UNVERIFIED — do not cite until checked.

**T1 — Data-constrained / repeated-data scaling laws**
- 2305.16264 — Muennighoff et al., *Scaling Data-Constrained LMs* (NeurIPS 2023). Up to ~4 epochs ≈ fresh data (abstract); repetition value decays toward ~0 by ~16 epochs (paper body). Foundational; the anchor extends it. *Caveat: epoch-equivalence / half-life are over UNIQUE DISCRETE TOKENS; for continuous autocorrelated iEEG a repeat re-presents the identical noise realization and effective-sample count ≪ patch count — no validated translation. Treat epoch as a gate-swept hyperparameter.*
- 2511.13421 — Yan et al., *Larger Datasets Can Be Repeated More* (Nov 2025). Linear-regression theory: repetition tolerance grows with dataset size. Transfers loosely.
- **2605.12715** — Sedova et al. (Apple), *Scaling Laws for Mixture Pretraining Under Data Constraints* (May 2026). Scarce target reusable 15–20× in mixture. → **L2.**

**T2 — Compute-for-data, overtraining, regularization-as-efficiency**
- 2403.08540 — Gadre et al., *LMs Scale Reliably with Over-training* (Mar 2024). Loss→error law; extrapolates over-trained regime.
- **2503.19206** — Springer et al. (CMU), *Overtrained LMs Are Harder to Fine-Tune* (Mar 2025). Catastrophic overtraining. → **L0/L1 motivation.**
- **2510.04071** — Gao et al., *What Makes Diffusion Language Models Super Data Learners?* (Oct 2025). Stochastic regularization (masking dominant; dropout/WD similar) drives multi-epoch data-efficiency. → **L1.**
- 2604.01411 — Roberts et al. (Wisconsin), *Test-Time Scaling Makes Overtraining Compute-Optimal* (Apr 2026). Low priority.

**T3 — Ensembling / ensemble-distill / self-distill / data-pruning**
- 2407.04600 — Pareek et al. (UW), *Gains from Repeated Self-Distillation* (Jul 2024). Multi-step self-distill gain grows with input dim. → L3.
- **2404.03263** — Farhat & Chen (UIUC), *Distillation as Alternative to Pre-Training Small Models* (Apr 2024). → **L3.**
- ICML 2025 (PMLR v267:48170) — Park et al. (KAIST), *Ensemble Distribution Distillation via Flow Matching*. Distill diversity, not just mean. → L3.
- **2206.14486** — Sorscher et al., *Beyond neural scaling laws: beating power law scaling via data pruning* (NeurIPS 2022 Outstanding). Label-free SS pruning metric; keep-easy in small-data regime. → **L1.5.**

**T4 — Distillation scaling laws / under data constraints**
- **2502.08606** — Busbridge et al. (Apple), *Distillation Scaling Laws* (Feb 2025). Distill wins when teacher reused / many students. → **L3.**
- **2603.04478** — Li et al. (Oxford), *Standing on the Shoulders of Giants: Rethinking EEG Foundation Model Pretraining via Multi-Teacher Distillation* (2026). **Scalp EEG**; matches SSL with 25% data. → **L4 (directional).**

**T5 — Data-efficient SSL / scientific & biosignal FMs**
- **2511.08544** — Balestriero & LeCun, *LeJEPA* (Nov 2025). Heuristics-free JEPA (no EMA/stop-grad), in-domain-small beats giant-transfer. Collapse-prevention alternative + thesis support.
- 2502.18056 — Vélez-García et al., *Escaping The Big Data Paradigm in Self-Supervised Representation Learning* (SCOTT / MIM-JEPA) (Feb 2025). Conv inductive bias for small-data SSL. (Confirm top-1 in PDF.)
- **2509.24317** — Li et al. (Apple), *Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers* (SALT) (Sep 2025). Student robust to teacher quality → favor student compute. → **L4.**
- 2510.21585 — El Ouahidi et al., *REVE* (Oct 2025). Big-data **scalp EEG** FM (MAE), setup-agnostic 4D PE. Opposite-regime contrast (NOT a distillation paper).
- 2509.21671 — Zahorodnii et al. (MIT), *Neuroprobe*. Our eval target (not a method).
- 2505.22964 — Zhang et al. (Microsoft), *Scaling Laws for EHR FMs* (May 2025). Loosely relevant precedent.

**T6 — Synthetic / self-generated data**
- 2603.18534 — Kim et al. (Stanford), *Scaling Synthetic Megadocs* (Mar 2026). Anchor successor: rephrasing 1.48×, megadocs 1.80× @32 gen/doc. Text-specific; megadoc-aggregation may not map to raw iEEG.
- 2503.19551 — Qin et al. (Microsoft/HKUST), *Scaling Laws of Synthetic Data* (Mar 2025). Rectified scaling law for synthetic. LLM-only.
- 2502.04235 — Hao et al., *Reformulation for Pretraining Data Augmentation* (MGA) (Feb 2025). Text reformulation augmentation. Low priority.

**T7 — Regularization recipes revisited**
- **2602.11137** — Han et al. (Harvard), *Weight Decay Improves Language Model Plasticity* (Feb 2026). Worse pretraining loss → better fine-tune with higher WD. → **L0/L1, transfer axis.**
- 1603.09382 — Huang et al., *Deep Networks with Stochastic Depth* (ECCV 2016). Stochastic depth / DropPath; implicit ensemble. → L1, cheap L3 pre-probe.
- (overlaps T2: 2510.04071.)

**Cross-subject transfer (input-side)**
- 1808.05464 — He & Wu, *Transfer Learning for BCI: A Euclidean Space Data Alignment Approach* (IEEE TBME 2020). Unsupervised per-subject alignment. → **L0.5.**
- 1909.13231 — Sun et al., *Test-Time Training with Self-Supervision...* (ICML 2020). → Candidate lever.

## Top 5 for a small data-scarce SSL FM
1. 2509.14786 (the recipe blueprint) · 2. 2605.12715 (mixture repetition, our exact setup) · 3. 2603.04478 (same family + distill lever, 25% data — directional, scalp) · 4. 2602.11137 (regularize-then-select-on-transfer) · 5. 2511.08544 (in-domain-small beats transfer; collapse-prevention alt). Runners-up: 2502.08606, 2509.24317, 2206.14486.

## Cross-doc notes
- Loss surface here = B31 2-term (`L_pre_frame@M2 + L_post_frame@M4`); this doc adds no terms. Note `ablations.md §4` (4-term) and `plan.md` (5 losses) still show the pre-B31 surface — a separate amendment, not this doc's error.

## Changelog
- 2026-05-28 — doc created; literature map + lever menu L0–L4 + sequencing. Awaiting BTWordEvents baseline.
- 2026-05-28 — added "Core insights from the anchor paper" section (faithful 7-point summary); flagged transfer-not-loss as our adaptation.
- 2026-05-28 — multi-agent audit (32 refinements, 5 P0). Applied: fixed L2 α=0.3→B02 α=0.5 sampler error; citation/title/venue fixes (Muennighoff NeurIPS 2023, 2510.04071 real title, SALT/SCOTT/MGA/Oxford exact titles, 2602.11137 full title, anchor ICLR 2026 Oral); transfer-validity caveats on WD/repetition/83%-retention/monotone-N/epoch-equivalence; added transfer-selection probe + scoring contract; runnable L1 protocol; L3 representation-distill + pilot + diversify + budget reconciliation; L4 scalp tag + Whisper double-count; novelty drop "scaling laws" + REVE-mislabel fix; new levers L0.5 (Euclidean Alignment 1808.05464), L1.5 (data-pruning 2206.14486), DoReMi in L2 (2305.10429), augmentation note (2206.14483), candidate TTT (1909.13231) / DropPath (1603.09382) / active-learning. All new arXiv IDs externally verified 2026-05-28. Surfaced to Ben: ablations.md cell-promotion, plan.md back-reference, L3 funding-gate, ablations/plan B31 staleness, Banville/Hong/EA-revisit verification.
