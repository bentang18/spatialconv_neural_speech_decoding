
# Neuroprobe Ablations — Running Menu

Single source of truth for every **mission-critical** ablation across Stage-0 (linear preprocessing hill-climb), Stage-1 (architectural defenses), Stage-2 (SSL loss + sampler + mask + EMA), Stage-3 (Whisper-L8 distillation), and Stage-4 (Neuroprobe readout).

**Source-of-truth rule.** This doc is the menu. Per-cell results, freeze write-ups, and dispatch artifacts live in their canonical homes: Stage-0 cells → `docs/neuroprobe/stage_0.md`; Stage-1/2/3 cells → relevant memos + `docs/neuroprobe/v14_blockers.md` for the contract locks themselves. Cite, don't duplicate.

**Status legend.** `frozen` = decision landed. `in-flight` = dispatched on DCC. `planned` = spec'd, awaiting prereq. `blocked` = waiting on external. `P0/P1/P2/P3` = priority within Stage-1+.

**Effect-size threshold.** "Load-bearing" = ΔAUROC ≥ 0.02 multiclass CrossSession (one CI half-width of Neuroprobe linear baseline). Below threshold → freeze upstream parity.

**Pruning principle (2026-05-26).** Cells that predate a current lock — v4 invisible front-end (5/24 B20), B19/B22/B03 5-term loss + mask discipline, single-teacher Whisper-L8 (5/22 iMINDBench pivot), DK-first atlas (5/13), MNI Fourier PE dropped (5/19), joint-from-step-1 within each SSL phase, 3-phase staged P1→P2→P3 — are excised. What remains is reviewer-defensible thesis falsifiers, P0 sisters of the current locks, conditional safety nets, and free diagnostics.

---

## 1. Stage-0 freezes (closed)

All Stage-0 cells whose decisions have landed are summarized below as one-line freezes. Detail lives in `docs/neuroprobe/stage_0.md` + report dirs under `reports/neuroprobe_stage0_*/`.

| Block | Decision | Source |
|---|---|---|
| **V** data-contract QC | closed 2026-05-01 | `reports/neuroprobe_stage0_v_data_qc_2026_05_01_v3/` |
| **L.1** normalization | FROZEN at N1 (train-set fixed StandardScaler). N0 (BrainBERT/PopT per-window z) −0.054. N8 (per-channel) −0.058 — v14 token-norm must pool across channels within a parcel. | `memory/project_l1_normalization_freeze_2026_05_08.md` |
| **L.2** reference × view | FROZEN at R4×I2 (shaftLap × stft_abs). View Δ=0.055 swamps reference Δ=0.012 by 4.5×; spectral mandatory. Phase ≈ chance; multi-band/wavelet add dim without signal. | `memory/project_l2_reference_view_freeze_2026_05_09.md` |
| **L.3** filtering + bad-channel | FROZEN as no-op within ±0.005 of F0; notch + HPF added back per-corpus at v14 preproc level for robustness. | `memory/project_l3_filtering_freeze_2026_05_11.md` |
| **L.4** window anchor | FROZEN; max \|residual\| = 0.0008 across the norm × view × ref interaction grid — greedy hill-climb safe. | `reports/neuroprobe_stage0_l4_window_anchor_*/` |
| **Shaft/depth contract** | FROZEN: 1549 contacts, 0 cross-hemi, 99.2% linear. Signed depth FORBIDDEN. | `memory/project_shaft_depth_geometry_freeze_2026_05_13.md` |

### L.5 — Diagnostic probes (kill-criteria gates, still active)

Live kill-criteria gates run on every sweep winner. Probes that became v14 architectural cells (parcel-anchoring, identity leakage) are kept; PS-era diagnostic probes (per-band leak, multi-band tokenizer candidates) dropped.

| Cell | Probe | Threshold | Status |
|---|---|---|---|
| **L.5.P1** | subject-id from features | KILL if held-out AUROC > 0.95 | dispatched 2026-05-10 |
| **L.5.P2** | session-id from features | KILL if held-out AUROC > 0.95 | dispatched 2026-05-10 |
| **L.5.P9** | acoustic/FM-leakage retrieval: brain → (env + f0 + Whisper-L8) | **v14-load-bearing**: contrastive must beat retrieval@10 by ≥ 5 pts + ≥ 0.05 R² on L8 | blocked (shares Whisper cache with L.7) |
| **L.5.P13** | post-aggregation identity (DK-mean-pooled features) | **v14-load-bearing**: tests parcel-anchoring premise; soft flag at AUROC > 0.85, soft positive at < 0.70 | planned |

### L.7 — Audio-FM upper bound + Phase-3 distillation-target ceiling

Dual-role cells: no-brain upper bound for v14 + distillation-target validity check before Phase-3 lock. Blocked on Ben self-sourcing 21 BT films (9 Lite priority).

| Cell | Audio FM / layer | Eval split | Role |
|---|---|---|---|
| **L.7.A0** | Whisper-L8 mean-pool → LogReg | within-session | upper bound (no-brain) |
| **L.7.B0** | Whisper-L8 mean-pool → LogReg | **CrossSession multiclass** (submit-lane) | submit-gate ceiling |
| **L.7.C0** | Whisper-L8 mean-pool → LogReg | **pooled CrossSubject multiclass** | headline-axis ceiling |
| **L.7.S-layer** | k ∈ {L4, L6, L8, L10, L12, L16, L20} mean-pool → LogReg | all 3 splits | picks audio layer by task discriminability (mirror of `plan.md` brain-fit ridge) |

v14 must clear A0/B0/C0 by ≥ 0.05 to claim brain adds signal beyond Whisper. **Layer pick is dual-validated, empirical, no Goldstein default** ([[feedback_no_default_ecog_to_seeg_transfer_2026_05_24]]) — S-layer and the `plan.md` brain-fit ridge sweep widened range {L4–L20}, not Goldstein's {L6–L10}. Convergent ⇒ lock; divergent ⇒ resolve at top-2 by a small Phase-4 probe. Shares cache with L.5.P9 and Phase-3 preflight — extract once, all three consume.

Blocker: `memory/project_l7_audio_fm_blocked_audio_source_2026_05_10.md`. Extension: `memory/project_whisper_ceiling_prerun_test_2026_05_24.md`.

### A — Atlas/surface-mapping gates (P0 blockers)

A0–A4 verify internal correctness of the surface-route fsaverage-bake pipeline (passed on Pipeline C native-vol aparc+aseg 0.5mm per `memory/project_v14_dk_first_pass_2026_05_13.md`). A5/A6 are P0 cross-route consistency gates before Stage-1 dispatch.

| Cell | Gate | Status |
|---|---|---|
| **A5 (P0)** | MNI ↔ BNA parity — surface route, three sub-gates ((a) lobe parity ≥ 95%, (b) DK ↔ BNA Tier-1 crosswalk ≥ 80%, (c) per-subject drift floor 70%). Runnable without Chris MNI; needs `data/atlas/dk_bna_tier1_crosswalk.csv`. | P0 blocker |
| **A6 (P0, post-Chris-MNI)** | Adds volumetric BNA(MNI) ↔ BT DK lobe/gyrus parity + surface ↔ volumetric cross-route agreement. Gates §6b continuous-MNI-Fourier-PE — but MNI Fourier PE itself was dropped 5/19, so A6 reduces to A5 robustness check post-Chris. | blocked on Chris |

### D — Atlas/pooling cells (surviving cells only)

DK-first pivot 5/13 closed most of the original D-block. Cells that survive as anchors:

| Cell | Eval | Prep | Atlas/pooling | Role |
|---|---|---|---|---|
| **D.public** | CrossSession multiclass | Lap + STFT | DK hard-mean (upstream `combine_regions()`) | upstream cross-subject baseline ARCHITECTURE — v14 must beat |
| **D.1a** | CrossSession multiclass | Lap + STFT | BNA Tier-1 hard mean | BNA hard vs DK baseline |
| **D.1b** | CrossSession multiclass | Lap + STFT | BNA Tier-1 soft support | **v14 novelty**: soft vs hard support |
| **D.14** | pooled multi-source CrossSubject multiclass | (winner prep) | (winner atlas) | scientific generalization default |
| **D.16** | electrode-set robustness: Lite-120 / random-120 / anatomy-120 / Full uncapped | (winner prep) | (winner atlas) | Lite-120 = leaderboard parity only |

---

## 2. Stage-1 entry — v14 thesis falsifiers

The cells reviewer-defense is built on. Each is a falsifier for a specific v14 commitment. Pre-committed before Stage-1 dispatch; pooled multi-source CrossSubject + S2/trial-4 parity + per-task breakdown for each.

| Cell | Tests | Status / decision rule |
|---|---|---|
| **AC2 frozen-features linear probe** | Whisper-L8 → linear → labels, no brain features | identical pipeline to L.7.A0 — L.7 IS the AC2 baseline; v14 must beat by ≥ 0.05 |
| **AC3 anatomy-blind random Perceiver** | Same Perceiver IO arch + budget, but `parcel_latents` random-init (no `P_emb[p]` BNA prior) AND no `log(support[i,p])` cross-attn bias | **P0**: tests whether anatomy-as-routing does work |
| **AC4 P_emb drift** | Unfreeze `P_emb[p]` (BNA-init, learnable); keep `support[i,p]` fixed and `log(support)` active | triangulates with AC3 (full v14 vs routing-only vs neither) |
| **AC6 shaft/depth feature matrix** | 3-cell: `hard_public_bt_label` only / shaft+depth only / hard label + shaft+depth | tests whether within-shaft depth adds cross-subject signal beyond anatomy alone; paired with subject-ID nuisance probe |
| **S1-A zero per-subject (default)** | Pure v14, no per-subject params anywhere | **P0** baseline arm of zero-per-subject defense (free — IS the v14 default) |
| **S1-C per-subject linear readout at deployment (sEEGnificant-style)** | Small affine post-PMA, one head per subject, kept at inference | **P0**: tests sEEGnificant -ΔR²=0.18 in-modality bar; if S1-C >> S1-A + 0.02 ship with the readout and reframe as "anatomy-aware routing + minimal per-subject calibration" |
| **X3 learned-correction routing bias** | `log(support+ε) + γ·Δ`, Δ zero-init learnable per-(electrode, parcel), γ small/learned | **P1**: sharper form of "anatomy enforcement: L" cell — at init = frozen default, training can deform; promotes on ≥ 0.005 CrossSubject lift WITHOUT raising subject-ID nuisance |
| **BiL-MNI** | MNI continuous PE on electrode tokens vs parcel-id-only | P0 bitter-lesson sister, gated on Chris MNI |
| **BiL-Freq-Plain** | Drop per-patch freq embed (10 vec) | P0 bitter-lesson sister (= F7) |

Source memos: [[project_v14_unique_contribution_2026_04_26]], [[project_v14_bitter_lesson_sister_bil_2026_05_19]], [[project_v14_spike_vs_field_potential_per_subject_defense_2026_05_16]], [[project_v14_competitor_review_2026_05_20]].

**Composability**: S1-A/B/C test the zero-per-subject default itself; AC3-AC5/V1/X3 test the architectural commitments; BiL-* test individual priors via bitter-lesson removal. Each isolates one v14 commitment cleanly.

---

## 3. Stage-1 — v4 front-end + arch sister cells (post-B20 lock)

v4 invisible front-end lock (5/24): Conv2d (3, 2) patches, hop = 256 (8 Hz), JOINT token-block attention, per-patch freq embed 10 vec, d=256, N=6, L=6, M=4. Surviving F-cells are the ones that test load-bearing pieces of v4 itself — not the pre-v4 factorized stack.

| Cell | Variant | Tests | Priority |
|---|---|---|---|
| **F0** | Linear F→d per time bin (no Conv2d patches) | Does the Conv2d (3, 2) patch earn its keep over a single linear patch per time bin? Most important comparison; if F0 ties, revert to single linear. | **P0** |
| **F7** | A1 patches, **drop per-patch freq embed (10 vec)** | Does the freq positional embed do work? Must-run alongside default to prove ❶b earns its 10×d params. | **P0** (= BiL-Freq-Plain) |
| **F12** | Hierarchical per-electrode PMA pool over patches before ❺ (M_per_elec ≈ 4–8 queries → ~1k tokens entering ❺) | Tests whether per-(freq, time) detail at ❺ is load-bearing or summarizable; ~49× compute savings at ❺ if pooling works | P1 |
| **F15** | Token depth N sweep {2, 4, 6, 8} | Default N=6 settled by Bundle-4 amendment but never empirically swept; find optimum | **P0** |
| **F16** | Token-vs-latent depth trade at fixed budget: (N=2, L=10), (N=4, L=8), (N=6, L=6), (N=8, L=4) | Where to spend depth budget for best perf-per-param? | P1 |
| **F18** | Add band-id embedding alongside per-patch freq embed (6-vocab delta/theta/alpha/beta/low-γ/high-γ) | Tests whether explicit band-level prior helps over per-patch PE alone | P1 |
| **F29** | CLS token prepended to 320 parcel-tagged latents; participates in ❻ latent SA × L=6 bidirectionally; read CLS state → linear → logits (replaces ❼ PMA) | BERT/ViT/AST/AudioMAE standard readout | P1 |
| **F30** | GAP over 320 latents → linear → logits (replaces ❼ PMA) | ViT-22B / DINOv2 standard readout; tests whether explicit readout aggregation is needed at all | P2 |

**For Stage-2 SSL `L_post_utterance`**: the readout MUST MATCH between SSL summary vector and classification summary. If F29 or F30 promotes, update Phase-3/4 readout accordingly.

Source: [[project_v14_v4_invisible_frontend_lock_2026_05_24]].

---

## 4. Stage-2 SSL — Loss design (B19 + B22 + B03 + B26 + B27 + B28 lock)

P1/P2 share a single **4-term default objective** (V-JEPA-2-canonical masked-only at M2; no context loss; B28 demoted DKoleo from default to sister-only):

```
L = L_pre_frame_masked@M2
  + L_mid_slot@LN_mid(M3)
  + L_post_frame@LN_frame(M4)
  + 1.0·L_post_utterance@LN_utt(M4)-PMA

[reactive: + 0.1  · L_Gram        if M4 trigger fires per B21]                              ← B21 (carryover)
[reactive: + 0.05 · L_DKoleo@M3   if M3 trigger fires per B22 Arm 2 (mechanism routed through chosen B28 DKoleo variant)]
```

All prediction terms use **pure L1** (B26 ✅ 2026-05-27 PM — supersedes B25's Smooth-L1; matches V-JEPA 2 §2.1 Eq 1). **No L_pre_frame_context term** (B27 ✅ 2026-05-27 PM-late — partial revert of B25 + B26 after V-JEPA 2.1 Tables 1+2 PDF re-read showed context loss costs ~10pp on clip-level SSv2: V-JEPA 2 baseline 72.8 → V-JEPA 2.1 best-case (λ=0.5+weighted+warmup) 62.5; only Deep Self-Supervision recovers, and v14 can't replicate uniform 4-level DSS because M2/M3/M4 live in different token spaces). **No λ_ctx warmup schedule** (no context loss to weight). **No default `0.1 · L_DKoleo@M4` term** (B28 ✅ 2026-05-27 PM-late — demoted to three sister cells gated by MON-SLOT-REDUNDANCY: the per-clip 320-slot DKoleo unit diverges from DINOv2/v3's per-batch CLS unit (different geometric claim, no precedent for the v14 unit), and B21's identity-anchored init + B22 M3 supervision + dedicated LN per head + reactive Gram already carry the collapse-prevention load). Loss-side gates `L_mid_slot / L_post_frame / L_post_utterance` PMA scoped to `parcels_supervised[subject]` per B03f; `L_pre_frame_masked` per-electrode-patch on hidden positions only (V-JEPA-2-canonical); reactive Gram + monitor F1 + MON-SLOT-REDUNDANCY (B28 NEW) + any armed B22 reactive `L_DKoleo@M3` operate over all 320 slots. EMA teacher fixed τ=0.999 throughout (B26 — drops V-JEPA 1 ramp per V-JEPA 2 §2.4 explicit). **Teacher sees FULL unmasked input at every depth** (B26 contract — no patch mask, no shaft mask on teacher; only per-corpus valid-bin mask). All-layer-averaged with per-layer instance-norm (EAT §3.1). Joint from step 1 within each phase. Phase 1 has cross-attn ❺ + latent stack ❻ ON; **anatomy bias linear warmup** (B28 Item 3 ✅ — λ_anat = 0 for first 75% of P1, then linear 0 → 1 over last 25% of P1 ∪ first 25% of P2, then 1 for remainder of P2/P3/P4; was discrete P1→P2 toggle in B19). Phase 2 has shaft-block electrode mask (B03 paradigm-B drop+predictor WARM-START across P1→P2). **B28 also reduces the encoder cross-attn count from 2 @ {0, 3} to 1 @ layer 0** (Perceiver-IO standard pattern; ~14.235M trainable post-B28, was ~15.024M).

| Cell | Tests | Priority |
|---|---|---|
| **BiL-Loss-Default** | The **4-term default** (4 prediction terms; DKoleo demoted to sister-only by B28) with B26 pure-L1 + fixed-EMA + B27 no-context-loss + B28 1-cross-attn + B28 anatomy-bias linear warmup — THE HEADLINE CELL | **P0** |
| **BiL-Loss-NoUtt** | λ_utt = 0 — falsifies the EAT-UFO utterance-lift transfer claim at v14 scale; if wins/ties, PMA loses P1/P2 gradient (20× sample-efficiency penalty at P3) | **P0** |
| **BiL-Loss-NoPre** | λ_pre = 0 (drops L_pre_frame_masked) — falsifies M2-level masked-prediction supervision at v14 scale; if wins, drop the 2-block predictor + L_pre_frame entirely | **P0** |
| **BiL-Loss-NoMid** | Drop L_mid_slot@LN_mid (revert B22) — tests whether M3 dense supervision earns its keep | **P0** |
| **BiL-Loss-LambdaUttSweep** | λ_utt ∈ {0.1, 0.5, 1.0, 2.0, 5.0} on single BT-Lite subject (~1 GPU-h total) — brackets EAT λ=1 default | **P0** |
| **R-l2-loss** | Pure L2 on all 4 prediction terms (data2vec 2.0 §3.1 form) — falsifies B26's L1 choice; tests whether iEEG outlier-robustness intuition is wrong (heavy-tail hypothesis untested at v14 scale) | **P0** (B26 retained through B27) |
| **R-mse-loss** | MSE (≡ L2) on all 4 prediction terms — original B25 falsifier; redundant with R-l2-loss, can be merged downstream | P0 (B25 retained, deprecated) |
| **R-smoothl1-beta-{0.5, 1.0, 2.0}** | Smooth-L1 with β sweep (β=1.0 = prior B25 default) — falsifies B26's pure-L1 choice; tests Huber-style outlier downweighting at v14 scale | P1 (B25 retained, repurposed) |
| **R-context-loss-vjepa21-recipe** | Reinstate full V-JEPA 2.1 §2.3.1 recipe at M2: pure-L1 visible-patch supervision with `λ_i = 0.5/√d_min(i, M)` weighting (Chebyshev d_min on per-electrode F-patch × T-patch grid) + linear λ_ctx warmup 0 → 0.5 over first 25% of P1. **Single falsification cell for the entire V-JEPA-2-vs-2.1 question on iEEG.** If it beats default on Neuroprobe gates, adopt B25/B26 recipe in a future amendment. | P1 (B27 NEW — sole replacement for the retired R-no-context-loss + R-no-warmup + R-ctx-lambda + R-p2-m4-context-loss roster) |
| **R-dkoleo-batch-cls-unit** | Per-batch DKoleo on the utterance-pooled CLS-analog vector (B, d) — DINOv2 Algorithm 1 + DINOv3 §3.3 faithful unit (per-batch CLS over the first global crop). Coeff 0.1. Optional batch sub-sample to 16-32 vectors per step to match DINOv2's 16-CLS bank-size regime. | **P1** (B28 NEW — DINOv2/v3-faithful KoLeo unit) |
| **R-dkoleo-intra-clip-slots** | Per-clip DKoleo on M4 slot means across all 320 latent slots (`M4.mean(dim=t_p) → L2_normalize → DKoleo` over 320, coeff 0.1) — the B21 prior default that B28 demoted, retained as the v14-original falsifier. | **P1** (B28 NEW — B21 default kept as falsifier) |
| **R-vicreg-slot-variance** | VICReg variance hinge per slot dimension over the per-clip 320 slots, no covariance term: `L_var = mean_d(max(0, 1 − std_{p ∈ 320}(slots[:, p, d])))` (Bardes/Ponce/LeCun 2022, arXiv:2105.04906). Coefficient TBD via 1-GPU-h BT-Lite preflight over {1, 5, 10, 25}. Anti-collapse without per-pair repulsion. | **P1** (B28 NEW — alternative anti-collapse family) |
| **R-perceiver-original-2-cross-attns** | Restore 2 cross-attns at latent-stack positions {0, 3} (original Perceiver iterative re-injection pattern, prior v4 default). Tests whether the second cross-attn at layer 3 is load-bearing on top of v14's L=6 latent stack. | **P0** (B28 NEW — settles the Perceiver-IO-vs-original-Perceiver question on iEEG) |
| **R-anatomy-bias-step** | Discrete P1→P2 anatomy-bias toggle (B19 prior default — λ_anat = 0 throughout P1, λ_anat = 1 from P2 step 0). Falsifies B28's linear-warmup design. | **P0** (B28 NEW — discrete-toggle baseline) |
| **R-anatomy-bias-on-from-p1** | λ_anat = 1 from P1 step 0 (full anatomy bias throughout, including SWEC P1 corpus). Stress-tests whether the P1 "anatomy-blind for SWEC" justification chain inherited from B19 is over-cautious; if a P1 run with bias-ON-throughout converges, the warmup design is also over-cautious. | **P0** (B28 NEW — bias-on-throughout stress test) |
| **R-perceiver-3-cross-attns-{0,2,4}** | 3 cross-attns at evenly-spaced positions (already a §3 F-cells candidate; re-anchored to B28's new 1-cross-attn default). | P1 (B28 re-anchored) |
| **R-ema-tau-{0.99, 0.9995, 0.9999}** | Fixed-τ sweep around the 0.999 B26 default — V-JEPA 2 doesn't disclose exact value, so sweep brackets the modal-value choice | P1 (B26 retained through B27) |
| **R-ema-ramp-v-jepa1** | Restore V-JEPA 1 linear ramp `0.99 → 0.9999 over 400k` — falsifies V-JEPA 2 §2.4's "fixed EMA is enough" simplification | P1 (B26 retained through B27) |
| **BiL-Loss-AddGram-P3** | DINOv3 Gram anchoring at P4 readout — only if P4 frozen-linear-probe shows latents drifted from semantic separability | P1 (conditional) |
| **R-p1-stage-a-only** | Phase 1 with cross-attn OFF + latent stack OFF + L_post_frame and L_post_utterance disabled (pre-B19 default) | **P0** load-bearing P1 sister — if it wins, B19 reverses |
| **R-p2-no-patch-mask** | P2 with shaft-mask only, no per-electrode patch mask | P1 |

**KoLeo demotion (B28 ✅, 2026-05-27 PM-late).** KoLeo @ 0.1 is NO LONGER in the default (B28 demotes from B21's default to sister-only — see the three `R-dkoleo-*` rows above; MON-SLOT-REDUNDANCY monitor gates dispatch). The conditional `BiL-Loss-AddKoLeo` is therefore subsumed (B28 makes KoLeo a sister, not a default). The prior `BiL-Loss-L1` cell is subsumed by the B26 default flip — L1 IS the default now; the Smooth-L1 / L2 / MSE alternatives are sister falsifiers. **B27 retirements** (subsumed/collapsed into the V-JEPA-2-canonical default or into the single `R-context-loss-vjepa21-recipe` sister): `R-no-context-loss` (default behavior now), `R-no-warmup` (no warmup to falsify), `R-ctx-lambda-{0.0, 0.25, 0.5, 1.0}` (collapsed to the single sister), `R-p2-m4-context-loss` (was gated on the P1 context loss, no longer exists).

**MON-SLOT-REDUNDANCY monitor (B28 NEW, always-on, every 10k steps).** Held-out probe batch (~256 clips). Logs per-clip 320-slot off-diag cosine (`mean / max / pct95`) over `slots_M4.mean(dim=t_p)` AND per-batch CLS-analog off-diag cosine (`mean / max / pct95`) over `L_post_utterance_clip_vec (B, d)`. Pre-registered escalation: `per_clip_cos.pct95 > 0.7 sustained 50k` → escalate `R-dkoleo-intra-clip-slots`; `batch_cos.pct95 > 0.7 sustained 50k` → escalate `R-dkoleo-batch-cls-unit`; both OR `per_clip_cos.diag-zeroed.mean > 0.5 sustained 50k` → escalate `R-vicreg-slot-variance`. B21's reactive Gram anchor at M4 (Component E, weight 0.1) remains the parallel collapse rescue at the geometry level (Frobenius anchor to a snapshot of the student backbone), separate from the diversity-regularizer family.

Sources: [[project_v14_loss_design_lock_2026_05_24]], [[project_v14_b22_collapse_prevention_dense_features_2026_05_25]], [[project_v14_collapse_prevention_lock_2026_05_25]], [[project_v14_b03_mask_lock_2026_05_25]], [[project_v14_loss_design_amendment_2026_05_27]] (B25, SUPERSEDED), [[project_v14_loss_design_amendment_b26_2026_05_27]] (B26, PARTIALLY SUPERSEDED — pure L1 + fixed EMA + full-input teacher retained; context loss + warmup reverted), [[project_v14_loss_design_amendment_b27_2026_05_27]] (B27, kept for context loss + EMA + teacher contract), [[project_v14_loss_design_amendment_b28_2026_05_27]] (B28, CANONICAL — DKoleo demote + 1 cross-attn + anatomy-bias warmup + citation cleanup).

---

## 4b. Stage-1/2 — B29 joint-default + Items 11/12/13/14/15 sister cells (post-B29 lock)

B29 collapses P1+P2 into a single joint SSL phase as DEFAULT and adds 5 mechanism-level changes (Items 11/12/13/14/15) + 2 corpus-level changes (AJILE12 back, α=0.3 sampling). Sister cells below test each load-bearing change. Anatomy bias schedule moves from B28 step warmup to B29 per-clip gate. The B28 step-warmup roster (`R-anatomy-bias-step`, `R-anatomy-bias-on-from-p1`) gets reinterpreted under B29 — sisters preserved.

| Cell | Tests | Priority |
|---|---|---|
| **R-keep-phase-split** | Restores 3-phase staged structure at matched 440k step budget. Head-to-head vs joint default. Falsifies the entire B29 phase collapse. | **P0** (B29 Item 1) |
| **R-subtype-embed-input-only** | M3AE-faithful: add `subtype_embed` at A1 patch embed but **NOT** in cross-attn K/V (Geng 2022 §3.1 mechanism without the v14-added K/V reuse). +512 params. Tests whether sensor conditioning helps at all without distorting routing. **NEW from 2026-05-28 Agent 2 audit; PROMOTED P1 → P0 5/28 PM Ben call when default flipped OFF.** | **P0** (B29 Item 11) |
| **R-subtype-embed-on-with-kv-reuse** | Full prior default: subtype_embed at A1 + reused in cross-attn K/V via same broadcast. +512 params. Tests whether K/V reuse adds value beyond input-only conditioning. **NEW 5/28 PM Ben call when prior default flipped OFF.** Provides counter-evidence on the K/V reuse contract. | **P0** (B29 Item 11) |
| **R-subtype-embed-3way** | Replace binary `{sEEG-depth, ECoG}` with 3-way `{sEEG-depth, ECoG-grid, ECoG-strip}` matching DIVER-1 §2.1 vocabulary. +256 params over binary. Cheapest falsifier on "grid/strip is just geometry" call. Only meaningful if input-only or K/V-reuse sister wins → tests vocab depth at that point. **NEW from 2026-05-28 Agent 2 audit.** | P2 if-budget (B29 Item 11) |
| **R-no-ref-embed** | Strips the 3-entry `{shaftCAR, bipolar, Laplacian}` `ref_embed` (768p). Tests whether explicit ref-operator signal is load-bearing vs implicit invariance learning from the per-clip ref draw alone. | P1 (B29 Item 11) |
| **R-parcels-supervised-gating** | Re-gates L_mid_slot + L_post_frame to `parcels_supervised[subject]` per-subject set + reinstates effective-batch reweighting. Falsifies B29's "uniform all 80 slots for all clips" Item 12 default. | **P0** (B29 Item 12, Ben call) |
| **R-swec-pseudo-parcel-per-shaft** | SWEC alternate fallback: assigns each SWEC shaft to a pseudo-parcel index (different mechanism than Item 12's "uniform all 80"); tests per-shaft pseudo-routing as a fallback if both Item 12 and the gated alternative fail. | P1 (B29, from Agent D) |
| **R-m4-slots** | Restores M=4 (320 slots = 80 DK parcels × 4) with LearnableSubSlotEmbed reinstated and index map `parcel_of_latent[i] = i // 4`. Tests whether per-parcel capacity earns its 4× cross-attn / 16× SA compute cost. | **P0** (B29 Item 13) |
| **R-d-bump-384** | d=256 → d=384 single-config override. +124% FLOPs +125% memory. ViT-Small / LaBraM-Large precedent. **Promotion gate**: must beat d=256 default by ≥0.02 AUROC at Cell-0 BT-Lite scale to become v15 default; otherwise documented as honest-negative. PROMOTED P1 → P0 must-run after MoE-FFN audit ruled out MoE as the capacity-recovery mechanism. | **P0** (B29 Item 15) |
| **R-d-bump-512** | d=256 → d=512 (+298% FLOPs). Breaks HB02 envelope; falsifier-only completeness check on the d-bump direction. | P2 (B29 Item 15) |
| **R-moe-ffn-soft-4** | Soft MoE-FFN per Puigcerver 2024 (arXiv:2308.00951) — 4 experts soft-mixed (no top-k), replaces dense FFN in latter 3 of 6 SA blocks. Explicitly framed as **negative-result candidate for paper rigor**: tests whether the EMA-teacher × MoE × masked-prediction integration has any signal at v14 scale. Runs only if all P0/P1 sisters complete and calendar permits. Soft MoE chosen over Sparse MoE for EMA-teacher compatibility per [[project_v14_moe_ffn_audit_2026_05_28]] Agent C. **Sparse MoE NOT in scope — deferred v15 future-work** per 4-agent unanimous audit. | P2 if-budget (B29 Item 14) |
| **R-drop-ajile12** | Drops AJILE12 from the cross-corpus pool (reverts to SWEC + D-cohort + BT). Falsifies the AJILE12 reinclusion call. | P1 (B29 Item 6) |
| **R-alpha-{0.1, 0.5, 0.7}** | α-temperature sweep around the α=0.3 default for the per-corpus weighted sampler. | P1 (B29 Item 5) |
| **R-include-bt-floor-{1, 5, 10}-percent** | Minimum BT gradient share floor (BT auto-share 12% at α=0.3; sister tests whether a hard floor protects BT downstream gate). | P1 (B29) |
| **R-with-anatomy-step-warmup** | Restores B28 linear-warmup schedule (λ_anat = 0 throughout first 75% of P1, ramp 0 → 1 over last 25% of P1 ∪ first 25% of P2). Falsifies B29's per-clip-gate Item 3 default. Subsumes the prior B28 `R-anatomy-bias-step` / `R-anatomy-bias-on-from-p1` cells under the per-clip-gate interpretation. | P1 (B29 Item 3) |

**Monitors added under B29**: MON-SENSOR-TYPE-CANARY (per-batch sensor-type linear-probe F1 from M2/M3 every 10k steps, target band `[0.7, 0.95]`), MON-REF-TYPE-CANARY (per-batch ref-type linear-probe F1, same band). MON-HEAD-BALANCE-005 (B29 Agent C; Item 12) demoted from kill criterion to free health canary — gradient ratio `‖∇θ_LN_frame‖ / ‖∇θ_LN_utt‖` `[0.3, 3.0]` interpreted as "investigate, don't gate" under Item 12's symmetric gradient share.

**Dispatch protocol**: Cell-0 sister-first (BT Lite ~5–10 H100-h, ~11–22 H100-h at d=384 for the R-d-bump-384 promotion gate) → full joint rollout on pass. 4 kill criteria + 2 new monitors gate Cell-0. Per [[project_v14_b29_joint_default_2026_05_27]] + [[project_v14_moe_ffn_audit_2026_05_28]].

Sources: [[project_v14_b29_joint_default_2026_05_27]] (CANONICAL — Items 1–15), [[project_v14_moe_ffn_audit_2026_05_28]] (Item 14 4-agent audit findings + R-d-bump-384 P0 promotion).

---

## 5. Phase-1 / Phase-2 sampler cells (B02 lock)

**Default = α=0.5 hierarchical over EXACT-precomputed valid-bin-electrode-hours.** 2-group split (SWEC vs broadband {AJILE12+D-cohort+BT}), macro 50/50, within-broadband α=0.5 over vb-eh. Shares: SWEC 50.0% / AJILE12 27.7% / D-cohort 15.0% / BT 7.3%. `torchdata.stateful_dataloader.StatefulDataLoader` + canonical fixed locality sharding. **P2 default = uniform-per-subject across 94 sEEG subjects (D 85 + BT 9), D 90.4% / BT 9.6%.**

| Cell | Tests | Priority |
|---|---|---|
| **R-sampler-alpha03** | α=0.3 over vb-eh: SWEC 42 / AJILE 27 / D 18 / BT 12 | **P1** load-bearing sister; if wins, default flips |
| **R-sampler-sqrth** | α=0.5 over hours (flat not hierarchical): SWEC 59 / AJILE12 26 / D-cohort 10 / BT 5 | P1 |
| **R-sampler-pure-h** | α=1.0 (pure pooled-h): SWEC 82 / BT 0.5 | P1 (de-facto-α=1 iEEG FMs like REVE/LaBraM/DIVER) |
| **R-sampler-uniform** | α=0.0 (uniform per corpus): each 25% | P1 (stress-tests BT over-representation + memorization) |
| **R-sampler-broadband-uniform** | SWEC 30 / AJILE 30 / D 25 / BT 15 | P1 (floors high-bin coverage for ripple/MUA differentiator) |
| **R-sampler-seeg-only** | AJILE12 dropped; SWEC + D-cohort + BT only, within α=0.5 hierarchical | P1 (per-electrode-features-modality-invariant defense) |
| **R-sampler-40-60** | macro 40/60 SWEC / broadband | P1 (3rd-round audit: 50/50 has no SOTA anchor) |
| **R-sampler-60-40** | macro 60/40 SWEC / broadband | P1 (BT-tokenizer-saturation mirror) |
| **R-p2-pooled-h** | Phase-2 pooled-by-hours: D-cohort 80 / BT 20 (old M06 default) | P1 |
| **R-p2-uniform-session** | Phase-2 uniform per (subject, session) | P1 |
| **R-p2a-bias-off-pretrain** | Sub-staged P2: ~5–10k step SWEC bias-off warmup before P2b sEEG-only bias-on | P2 |

Source: [[project_v14_cross_subject_pretraining_data_strategy_2026_05_22]], [[project_d_cohort_phase2_cohort_audit_2026_05_23]], `docs/neuroprobe/v14_blockers.md §B02 / M06 / M06-aux`.

---

## 6. Phase-1 / Phase-2 mask + EMA-K cells (Bundle-4 + B03 lock)

**P1 mask** = paradigm-B drop+predictor WARM-START across P1→P2; per-electrode patch mask ≤ 50%; shaft DROP via key_padding_mask in P2. **EMA layer averaging** = K=6 all-layers symmetric (P1 N=6 token blocks / P2 L=6 latent-stack) with per-layer instance-norm. **P2 mask** = shaft-block **K=1 default** `K = 1 if N_shafts ≥ 2 else 0` block α only ~12.5% effective on BT (clarified shaft NOT parcel; **K-cap revised twice 2026-05-27 from K=3**: AM K=3→K≤2 fraction-based; PM K≤2→K=1 — biosignal-FM precedent ~40–60% combined; K=3 reached 70% combined (V-JEPA territory); K ≤ 2 landed at 62.5% combined (biosignal-FM upper edge) with "fraction theatre" issue (cap fires at N≥5 → degenerated to fixed K=2 for BT cohort); K=1 lands at ~56% combined (band-centered), monotonic, Brain-JEPA Table 6 "tube" variant IS K=1 single-shaft × full-time; preserved as falsification sisters `R-shaft-K2` P0 + `R-shaft-K3-mixed-3block` P1).

| Cell | Tests | Priority |
|---|---|---|
| **R-p1-frame-1d-keep-3-mask-80** | Frame-1D inverse-block, keep-3 ≈200ms, 80% mask (pre-Bundle-4 default) | **P1** load-bearing sister against paradigm-B default |
| **R-p1-mask-rate-{50,55,60,65,70}** | Bracket around 60-65 midpoint | P2 |
| **R-K-last-only** | EMA target K=1 (final layer only) vs K=6 default | **P1** |
| **R-no-target-norm** | Drop per-layer instance-norm before averaging | P2 |
| **EMA-K-P2-sweep** | K ∈ {1, 3, 4, 5, 6} on single BT-Lite subject (~1 GPU-h total) | **P0** |
| **R-shaft-K2** | Fixed K=2 with blocks α+β; falsifies K=1 default and same-day K ≤ 2 fraction intermediate | **P0** (settles 2026-05-27 PM K-cap revision vs biosignal-FM upper-edge target) |
| **R-shaft-K3-mixed-3block** | Restores prior K=3 EX03 default (all 3 mixed-extent blocks α/β/γ); tests full literature-transfer of Brain-JEPA fMRI default in case both K=1 and K=2 underfit | P1 (demoted P0→P1 PM 2026-05-27) |
| **R-stratified-shaft-mask** | Stratifies shaft selection to avoid uniform-random orphan concentration; MON-MASK-002 `ratio < 0.7` escalation target | **P0** (mean-collapse mitigation) |
| **R-p2-random-electrode-mask** | P2 original Bernoulli 30% per-electrode (pre-Bundle-4) | P1 |
| **R-p2-shaft-tube-full-time** | P2 full-time × shaft-mask only (Brain-JEPA "tube"; semantically close to K=1 default) | P2 |
| **R-p2-parcel-mask** | P2 multi-block mask along PARCEL axis instead of SHAFT | **P1** (settles shaft/parcel ambiguity in original commit) |
| **R-p2-mask-rate-{8,12,18}** | J1 sweep around new ~12.5% midpoint under K=1 default (was {15,25,35} around K≤2 ~25% — superseded; before that {25,30,50} around K=3 ~40%) | P2 |
| **N-sweep {2, 4, 6}** | Token-block count sweep (also lives in §3 F15) | P0 |
| **EX09-teacher-mask-off** | Disable post-teacher zero-out at invalid bins — runtime-assert validation, NOT a candidate | diagnostic |

Source: `docs/neuroprobe/v14_blockers.md §B08, B11, EX03, EX09`.

---

## 7. Phase-1 — robust-z scope sister (B13)

B09 latent-SA bias falsifiers (BNA-Dist / T5-Learnable / Graphormer-1Step) **dropped for workshop scope** — bitter-lesson argument in paper text ("anatomy enters once at ❺; latent SA learns the rest") pre-empts the reviewer ambush without 3 empirical falsifiers.

### B13 — robust-z scope sister

| Cell | Tests | Priority |
|---|---|---|
| **R-norm-cohort-pooled** | Per-(freq-bin) median/MAD pooled over training cohort, applied to held-out CSubject test (no per-session calibration on held-out subject). Tests whether "leakage paranoia" beats per-session calibration. | **P1** |

Source: `docs/neuroprobe/v14_blockers.md §B13`.

---

## 8. Phase-3 — Whisper-L8 distillation (B05 + B06 lock)

**Default** = teacher rate 8 Hz, teacher-side triangular pool 50→8 Hz factor 6.25 FWHM 250 ms in dataloader → (40, 1280); 2-layer MLP Whisper-side `Linear(1280, 256)→GeLU→Linear(256, 256)` LLaVA-1.5 shape ~393k params → (40, 256); v14 student identity (40, 256) at 8 Hz native after PMA-k=1 + no time pool; Smooth-L1((40, 256), (40, 256)). Phase-4 readout: NO time pool — flatten T·d → linear (iMINDBench-parity).

| Cell | Tests | Priority |
|---|---|---|
| **R-rate-5Hz** | Phase-3 distillation rate: 5 Hz (slower averaging, halfway to word rate; lower-end bracket) | **P1** rate falsification |
| **R-rate-20Hz** | 20 Hz (upper end of preflight grid; Goldstein 2025 found ECoG alignment degraded above 20 Hz) | **P1** rate falsification |
| **R-event-locked** | MFA-aligned variable-width buckets at syllable / word / phoneme onsets, vs uniform rate-r\* triangular pool | P2 |
| **R-adapter-linear** | Single-linear adapter instead of 2-layer MLP | P1 (fallback) |
| **R-pool-then-probe** | Phase-4: DIVER-1-style mean-over-T → 256-dim → linear | P1 (reported, not headline) |
| **R-no-time-pool** (DEFAULT) | Phase-4: PMA k=1 → (T, d) → flatten T·d → linear | — |
| **R-flatten-with-parcels** | Phase-4: skip parcel-collapse; flatten (320 × T × d) ≈ 1.2M-dim → linear | P2 (capacity stress test) |
| **AC-Dual-Stream temporal + spectral** | Parallel temporal voltage stream + Multi-STFT spectral, fused via Hadamard at latent level (ASPEN) | **P2** (conditional on v14 underperforming time-derivative tasks) |
| **F-CQT** | Constant-Q transform front-end (Q ≈ 6 ⅓-octave) vs Multi-STFT default | P1 |
| **F-single-STFT** | Best single-STFT vs Multi-STFT default | P1 |
| **F-log-amplitude** | `log(|STFT|)` vs raw `|STFT|` (5/25 default swap; sister falsifier) | P1 |

**Phase-3 preflight (operational gate)**: dual ~1 GPU-h preflights before Phase-3 lock — (a) brain-fit ridge sweep `k ∈ {L4, L6, L8, L10, L12, L16, L20}` × `rate ∈ {5, 10, 20}` Hz × short lag sweep, picks (k, r, lag) by cross-validated speech-cortex r²; (b) task-fit ceiling = same grid, mean-pool → LogReg on gate tasks (= L.7.B0/C0/S-layer/S-combine). Convergent ⇒ lock k\* / r\*; divergent ⇒ resolve at top-2 by Phase-4 probe. **No Goldstein default; both layer and rate are empirical on sEEG.**

Sources: [[project_v14_imindbench_multistft_pivot_2026_05_22]], [[project_v14_stft_abs_default_2026_05_25]], [[project_whisper_ceiling_prerun_test_2026_05_24]].

---

## 9. Component transfer probe (post-P1 interpretability)

Tests whether NCA §5.3.1's "attention-as-transferable-substrate" finding holds at v14 ~15M scale (~25× smaller than NCA's 400M-1.6B test range). Endpoint = Phase-4 frozen linear-probe AUROC on CrossSubject + CrossSession. Cost ~60 GPU-h Lite total (≤ 1% of P1 budget). Runs parallel-to-P3, must NOT gate workshop critical path.

| Cell | Tests | Priority |
|---|---|---|
| **R-noreinit** | Full P1 checkpoint, no re-init (control / upper bound) | **P0** |
| **R-reinit-sa** | Re-init ❻ (latent SA L=6); keep P1-trained ❶❷❺ + parcel latents + LNs | **P0** — hypothesized largest drop if NCA transfers |
| **R-scratch-p2** | No P1; P2 from random init (lower bound — what SWEC contributes overall) | **P0** |

**Decision rule** (3-cell minimal):
- R-noreinit >> R-scratch-p2 → SWEC pretrain transfers at all.
- R-reinit-sa drop close to R-scratch-p2 → NCA attention-transfer holds at v14 scale; latent SA carries the SWEC signal; bias-OFF P1 design defended.
- R-reinit-sa drop small → NCA finding does NOT transfer to 15M iEEG-FM; SWEC signal lives elsewhere (tokenizer / parcel latents / xattn) — followup probe deferred to journal version.

Workshop scope: 3 cells suffice to answer "does P1 transfer; is it in attention." Component-localization detail (xattn / tokenizer / latents-only) deferred to journal followup.

Source: Lee/Han/Kumar/Agrawal "Training Language Models via Neural Cellular Automata" arXiv 2603.10055v12 §5.3.1.

---

## 10. Phase-4 / eval cells

| Cell | Tests | Priority |
|---|---|---|
| **X1 few-shot prototype eval, zero training** | Frozen backbone, one class prototype = mean readout over K ∈ {3, 8} support trials per class, classify by nearest prototype. Lead with CrossSubject. | **P1** removes the probe's own capacity from measurement; isolates backbone representation. Cite DINOv2 low-shot + BrainWave prototype protocol. |
| **R-whiten-latent** (L0.5) | Per-subject latent covariance whitening on the frozen PMA readout (contract below), vs the no-whitening default probe (§8 R-no-time-pool). Tests whether residual subject-identity second-order structure survives the anatomy routing and hurts transfer. Lead **CrossSubject binary AUROC** (the gate's weak prong); report CrossSession. **Planned — gated on the BTWordEvents-fix clean baseline.** | **P1** — cheap, test-time, aimed straight at the CSubject gap; promotes on ΔAUROC ≥ 0.02 |
| **R-whiten-global** | Single shared cohort-pooled W instead of per-subject. Control: for a linear probe a global W folds into the probe weights → expected null; isolates that the PER-SUBJECT axis is what does the work (per-subject W_s is not absorbable into a fixed linear probe). | P2 — non-absorbability control |

**L0.5 latent-whitening contract (3b — frozen encoder, test-time, per-subject).** Object = the d-dim PMA readout feature vector at each (clip, time-bin); the per-subject cloud is over all (clip, time-bin) pairs, so Σ_s is d×d (tractable; whitens the feature axes pooling over time within subject), applied before the §8 flatten T·d → linear. Per subject s, label-free from its UNLABELED clips: μ_s = mean(z), Σ_s = cov(z), shrink Σ̂_s = (1−λ)·Σ_s + λ·I (Ledoit-Wolf or small λ grid — minutes/subject makes the raw Σ_s near-singular, so Σ^(−1/2) amplifies noise directions without it), W_s = Σ̂_s^(−1/2) via eigendecomposition (U Λ^(−1/2) Uᵀ). Apply z̃ = W_s·(z − μ_s). Train the linear probe on whitened features of the training subjects; the held-out subject whitens with ITS OWN μ_s, Σ_s estimated from its unlabeled recording (no labels) → fits the no-test-labels gate. **Invariant**: stats are always per-subject, estimated from that subject's own data, train and test alike (EA protocol; He & Wu, arXiv:1808.05464, IEEE TBME 2020). Frozen Phase-4 transform, parallel-to-eval, does NOT gate critical path. **The hypothesis under test is redundancy**: anatomy-routed parcel tokens + zero per-subject params already place subjects in a common frame, so the honest expected outcome is a small or null Δ — a clean null is itself a paper-usable statement about the routing. **3a baked-in variant** (deferred P2, only if R-whiten-latent earns ΔAUROC ≥ 0.02): domain-specific whitening-BN on the latent inside the forward (SPDDSMBN, arXiv:2206.01323 / Decorrelated-BN, arXiv:1804.08450) — heavier (reopens training, per-subject batching, interacts with the B31 objective). Canonical lever context + literature map: `docs/neuroprobe/data_efficiency.md` L0.5.

Source: [[reference_few_shot_eval_protocols_2026_05_26]], [[project_v14_competitor_review_2026_05_20]], [[project_data_efficiency_primary_lever_2026_05_28]].

---

## 11. Free diagnostics

| Cell | Tests | Priority |
|---|---|---|
| **Anatomically-realistic mask sampling** | Sample JEPA mask from empirical clinical-coverage distribution vs uniform random parcel mask | P2 — train-inference mask-distribution match |
| **Per-parcel coverage diagnostic (post-pretraining)** | Per-parcel inference accuracy stratified by training-set coverage. If v14 predicts well only on high-coverage parcels → pattern completion within coverage manifold; if low-coverage parcels also predict well → true cross-subject transitivity via parcel-B bridging | **P1 free diagnostic** — single empirical answer to A↔B↔C transitivity question |
| **Brain-JEPA-style functional gradient PE post-hoc** | After P1, compute diffusion-map functional gradient from learned attention map (analog to Brain-JEPA Brain Gradient Positioning), freeze, inject as latent-side PE | **P1 free** — tests whether v14 latent SA recovers the functional gradient on its own |
| **Band attribution analysis** | Per-band IG + ablation + clustering on learned per-patch freq embed (10 vec) — the 5th novel paper claim | **P1** post-hoc interpretability |

Sources: [[project_v14_post_eeg_dino_synthesis_2026_05_10]], [[project_v14_band_attribution_analysis_2026_05_19]], [[project_v14_competitor_review_2026_05_20]].

---

## 12. Stage-3 — Foundation-model swap + cold-start (fallback only)

DINOv3 + multi-FM extension entries from the pre-5/22 era are dropped (single-teacher Whisper-L8 locked). What survives:

| Cell | Status |
|---|---|
| **MTDP-style stimulus-agnostic cold-start (fallback)** | DEMOTED to "Stage-3 fallback if Phase-1 JEPA saturates"; the empirical chain "MTDP > latent-JEPA" is NOT in the literature. If v14 needs an unpaired-data scaling lever, prefer Level C JEPA at scale / multi-target V-JEPA 2.1 / cross-session JEPA before reaching for foreign-FM distillation. |
| **Continuous-corpus alignment gate** | Required engineering check when continuous paired data lands (ds003688, Podcast, NeuroListen — clock-drift audit, precision event-time alignment) |

---

## 13. Out of Neuroprobe scope (pointers)

- **PS-program Stage-1/2/3 ablations** — PS-paused 2026-04-24. Canonical: `docs/strategy/stage_<N>.md`.

## 14. Archived / discontinued

| Cell | Why dropped |
|---|---|
| **§4 4-cell Stage-2 loss triangulation (Default / No-JEPA / No-DSigLIP / No-KoLeo)** | Pre-B19 era; current 5-term loss is the active triangulation (see §4) |
| **§5 Schedule cells (sequential 2a→2b default, late-add intrinsic, curriculum-warmup)** | Joint-from-step-1 locked within each SSL phase; 3-phase staged P1→P2→P3 locked across phases |
| **§5 Stage-2 establishment cells (SALT static-teacher, VideoPrism cross-modal-first, REPA-E, multi-teacher MSE, WhisperBCI)** | EMA-JEPA is default (7/7 brain-FM precedent); single-teacher Whisper-L8 locked; WhisperBCI principle covered by S1-B |
| **§5 SSL recipe cells (dense-ctx, freq curriculum, NSA, multi-window, specialist consolidation, etc.)** | Superseded by B19/B22/B03 5-term loss + mask discipline |
| **§5 Capacity sweep d ∈ {32, 64, 128}** | d=256 locked; 13M/25M/40M sister sweep uses μP/μTransfer (see [[project_v14_scaling_law_param_sizing_2026_05_20]]) |
| **§6 Gram anchoring** | Duplicate of §4 BiL-Loss-AddGram-P3 conditional |
| **§6 L_DSigLIP layer-sweep** | Superseded by AC-Multi-Layer-Whisper concat (§8) + Phase-3 preflight |
| **§6 J1 mask-rate sweep** | Folded into §6 R-p1-mask-rate-* |
| **§6b Continuous MNI Fourier PE head-to-head** | MNI Fourier PE dropped 5/19 |
| **§6b JEPA target level B vs C vs D** | Level C is default; B22 supervises at M3 (mid-slot) too — settled |
| **§6b Brain-JEPA Cross-ROI / Cross-Time / Double-Cross mask** | Covered by §6 R-p2-parcel-mask + R-p2-shaft-tube-full-time |
| **§6b SIGReg vs EMA + StopGrad** | EMA locked (7/7 brain-FM precedent) |
| **§6d V2-V13** | Predates v4 invisible front-end lock (Conv2d (3, 2), JOINT token-block, no separate ❸a/❸c/❸g blocks) |
| **§6g F1, F3, F4, F6, F8, F9, F10, F11, F13, F14, F17, F19, F21-F28** | Pre-v4: tested factorized stack / per-freq specialization / ❸g geometry block, none of which exist in v4 |
| **§7 D-SigLIP multi-FM extension, DINOv3 vision FM addition, PHI-S, MTDP gated multi-teacher, BrainGFM atlas-mix, BioX-Bridge** | DINOv3 dropped 5/22; single-teacher Whisper-L8 locked; multi-FM extension paused to v15 |
| **V1 vanilla raster + parcel-id-PE vs Perceiver IO** | Workshop scope: PopT direct comparison already covers "v14 beats vanilla raster" reviewer-defense; V1's same-budget-same-recipe rigor is main-conference territory |
| **S1-B per-subject linear on Phase-3 distillation path** | Workshop scope: S1-A (free) + S1-C (sEEGnificant -ΔR²=0.18 in-modality bar) bracket the zero-per-subject question; S1-B's "where to put the per-subject param" sub-question deferred to journal |
| **BiL-NoSpatial** | Redundant with AC3 in v4 (no ❸a/❸b/❸g blocks remain post v4 invisible front-end); AC3's anatomy-blind random Perceiver IS the spatial-prior falsifier |
| **BiL-LatentSA-Bias-{BNA-Dist, T5-Learnable, Graphormer-1Step}** | Workshop scope: bitter-lesson argument in paper text pre-empts the reviewer ambush; 3 empirical falsifiers were main-conference rigor |
| **R-reinit-xattn / R-reinit-tok / R-reinit-all-but-latents** | Workshop scope: 3-cell minimal (R-noreinit + R-reinit-sa + R-scratch-p2) answers headline question; component-localization detail deferred to journal followup |
| **AC1 FM-swap (Whisper → HuBERT-L9 / WavLM-L9 / EnCodec / w2v-BERT-2-mid)** | Heavy: 4 audio-FM inferences + Phase-3 distillation per cell. Whisper-L8 already anchored by 4-paper triangulation (Goldstein-2025 Nature + Antonello/Shimizu + Vaidya-2022 + Hong-2024); Conwell "diet > arch" ambush survivable via that citation block. Empirical AC1 = main-conference rigor. |
| **AC5 post-hoc SRM baseline** | Workshop "v14 beats Linear-Lap+spec + CNN-Lap+spec + PopT + DIVER-1 ceiling" baseline-set is enough; SRM (separate per-subject `W_i` shared-space pipeline) deferred to journal. |
| **R-rate-{10Hz, 16Hz}** | Workshop scope: preflight picks r\*; 2-rate sister bracket (5Hz + 20Hz) suffices to falsify lower + upper end. Full 4-rate grid = main-conference rigor. |
| **X2 per-task PMA query sets** | Neuro-MoBRE task-disentangled principle; workshop scope = shared-query + per-task linear head suffices. |
| **AC-Multi-Layer-Whisper concat (Evanson 2025 deep-layer)** | Conditional on v14 underperforming lexical-feature tasks; trigger deferred until post-Phase-3-lock. Whisper-L8 single-layer is the locked default. |
| **L.7.S-combine (multi-layer pool headroom)** | S-layer alone picks k\* fine; multi-layer pool headroom check = journal followup. |
| **X4 SSL pretext direction (causal vs bidirectional)** | Workshop: cite v14's bidirectional masked-latent default with data2vec / V-JEPA precedent; empirical falsifier (different predictor head) deferred. |
| **§2 L.6 deferred Tier-2 (ES, NR, WL, CB, FA)** | Deferred post-Stage-0; never load-bearing |
| **§2 D.2 / D.5 / D.8 / D.10 / D.11 / D.12 / D.13** | PS-era atlas/pooling sweep; superseded by DK-first pivot 5/13 + the surviving D.public/D.1a/D.1b/D.14/D.16 anchors |
| **§3 Stage-1 cold-start architecture cells (temporal tokenizer A/B/C/D, anatomy enforcement hard-mask/Z/L/no-constraint, Perceiver H/X/Y/unstructured, latent M ∈ {2,4,8}, width × depth sweep, hierarchy, support source argmax-vs-prob)** | Predates v14 arch lock; Conv2d patches / log(support+ε) / M=4 / d=256 / N=L=6 / flat per-parcel / BNA-soft P1-gated-on-Chris all locked |
| **§3 Stage-1 optimization/view/control cells** | B01 optimizer locked; bf16 + FA-3 engineering pins; L.1/L.2/L.3/L.4 frozen; shaftCAR locked |
| **Stage-2b as DEFAULT schedule** | Dropped 2026-04-26 for head-side leak risk; superseded by joint-from-step-1 default |
| **Curriculum warmup (recon-first 50-80%)** | No precedent; miscited to SigLIP-2 which is the opposite pattern |
| **v14 kernel ablation (pre-reset 2026-04-17)** | Superseded by Stage-0 L-sweeps + Stage-1 AC roster |

---

## Statistical method (binding for all cells)

Every freeze decision must cite Stage-0's stat appendix: bootstrap N=2000 percentile CIs, paired Wilcoxon + rank-biserial, BH within sweep, ≥ 3 seeds (42/43/44) on chosen + nearest competitor, train/test pair-overlap assert, upstream-commit + Whisper-commit + uv.lock SHA pinned. Source: `docs/neuroprobe/stage_0.md §"Statistical Methods"`.

**Trend-level reporting**: alongside bootstrap-CI primary stats, report Cohen's d + 95% CI + win-rate matrix + Bayesian baycomp posterior `(P(left), P(rope), P(right))` with ROPE ρ=0.01. Hierarchical evidence (strong / moderate / weak / minimal) based on `|d|` × relative improvement × cross-seed consistency. Complements, does not replace, classical tests — reviewer-defensible reporting for n=3-seed regimes. Source: MultiDiffNet (Zhang/Shapovalenko 2025) §D, [[reference_multidiffnet_zhang_2025]].

## Source memos

- Stage-0 freezes: [[project_l1_normalization_freeze_2026_05_08]], [[project_l2_reference_view_freeze_2026_05_09]], [[project_l3_filtering_freeze_2026_05_11]], [[project_shaft_depth_geometry_freeze_2026_05_13]]
- L.7 blocker: [[project_l7_audio_fm_blocked_audio_source_2026_05_10]]
- Phase-3 preflight: [[project_whisper_ceiling_prerun_test_2026_05_24]]
- Architecture: [[project_v14_arch_revision_2026_05_19_v3]] → [[project_v14_arch_post_v3_amendment_2026_05_19]] → [[project_v14_imindbench_multistft_pivot_2026_05_22]] → [[project_v14_stft_abs_default_2026_05_25]] → [[project_v14_v4_invisible_frontend_lock_2026_05_24]] → [[project_v14_collapse_prevention_lock_2026_05_25]] → [[project_v14_b22_collapse_prevention_dense_features_2026_05_25]] → [[project_v14_b03_mask_lock_2026_05_25]]
- SSL recipe: [[project_v14_three_phase_staged_recipe_2026_05_18]], [[project_v14_loss_design_lock_2026_05_24]]
- Sampler: [[project_v14_cross_subject_pretraining_data_strategy_2026_05_22]], [[project_d_cohort_phase2_cohort_audit_2026_05_23]]
- DK-first: [[project_v14_dk_first_pass_2026_05_13]]
- Zero-per-subject defense: [[project_v14_spike_vs_field_potential_per_subject_defense_2026_05_16]]
- Competitor review: [[project_v14_competitor_review_2026_05_20]]
- Scaling-law sizing: [[project_v14_scaling_law_param_sizing_2026_05_20]]
- Component transfer source: Lee/Han/Kumar/Agrawal arXiv 2603.10055v12 §5.3.1
- Statistical method: [[reference_multidiffnet_zhang_2025]]
