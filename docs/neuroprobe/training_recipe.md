# v14 Canonical Training Recipe

**Status: 2026-05-28 lock state** (v4 invisible front-end lock + B21 collapse prevention + B22 dense-features amendment + B03 mask-discipline lock + B25 loss-form & context-loss amendment + B26 V-JEPA-2.1-fidelity amendment + B27 context-loss revert + B28 defaults trim + citation cleanup + **B29 joint-default + Items 11/12/13/14/15** + **MoE-FFN audit** + **B31 V-JEPA-2-canonical 2-term SSL + PMA-trained-only-at-P3 lock**). Flat compiled current-state doc — the implementation contract. Reads top-to-bottom for engineers building NeuralTrain `Experiment` / `Data` / dispatch.

**B31 LOCK (2026-05-28 PM, see [[project_v14_b31_vjepa2_canonical_loss_2026_05_28]]) — supersedes the loss section below.** Joint SSL phase runs **2 terms only**: `L_total = L_pre_frame_masked @ M2 + L_post_frame @ M4`. `L_mid_slot @ M3` and `L_post_utterance @ M4-PMA` are DROPPED from default (preserved as `R-add-m3-loss` / `R-add-utterance-loss` / `R-add-both` P0 falsifier sisters). PMA query is **NOT trained in the joint SSL phase** (zero gradient). PMA receives gradient only during P3 cross-modal distillation against Whisper-L8, then is frozen at P4. The 4-term loss equation + `L_mid_slot` / `L_post_utterance` bullets + "all four mask-prediction losses" rationale below are PROVENANCE for the path B31 simplified away — read them for context, but the live contract is the B31 memo. Code drift: see `docs/neuroprobe/v14_blockers.md` B31 row.

**AMENDED 2026-05-27 PM-late & 2026-05-28** by [[project_v14_b29_joint_default_2026_05_27]] (Items 1–15) + [[project_v14_moe_ffn_audit_2026_05_28]] (Item 14):
- **Phase structure**: P1+P2 → single joint SSL phase as DEFAULT (`R-keep-phase-split` P0 falsifier). P3 unchanged.
- **Corpus**: SWEC + AJILE12 + D-cohort + BT, α=0.3 temperature. AJILE12 reincluded.
- **Embeds default-on**: `subtype_embed {sEEG-depth, ECoG}` 512p + `ref_embed {shaftCAR, bipolar, Laplacian}` 768p additive at A1 + reused in cross-attn K/V.
- **Item 12**: DROP `parcels_supervised` gating; L_mid_slot + L_post_frame fire on all 80 slots for all clips. Effective-batch reweighting retired. MON-HEAD-BALANCE-005 demoted to health canary.
- **Item 13 (M=1 default, 80 slots)**: `z_init[p] = LearnableParcelEmbed[p] + ε`; SubSlotEmbed dropped. The arch section below referencing "K = 80 DK parcels × M = 4 slots = 320 latent slots" / "320-slot tensor" / "LearnableSubSlotEmbed" / index map `i // M` is **2026-05-25 provenance**; under B29 Item 13 the index map collapses (latent[i] = parcel[i]; no `i // M`). LN_mid / DKoleo-monitor / Gram all rescale 320 → 80. `R-m4-slots` P0 falsifier. 16× SA + 4× cross-attn compute reduction. MON-SLOT-REDUNDANCY thresholds need rescaling for 80 slots (open task — pending B28 amendment).
- **Item 14**: Dense FFN preserved across all 6 SA blocks. MoE-FFN deferred v15. `R-moe-ffn-soft-4` P2 if-budget only (Soft MoE per Puigcerver 2024 arXiv:2308.00951). 4-agent unanimous audit verdict on EMA + masked-prediction + MoE incompatibility ([[project_v14_moe_ffn_audit_2026_05_28]]). Story angle "anatomy in routing, function in experts" DROPPED (empirically unsupported per OpenMoE / Mixtral routing / ST-MoE).
- **Item 15**: d=256 stays default. `R-d-bump-384` PROMOTED P1 → P0 must-run sister with ≥0.02 AUROC promotion gate at Cell-0 BT-Lite scale. `R-d-bump-512` P2 if-budget (breaks HB02 envelope at +298% FLOPs).
- **HB02 re-cost** still OPEN — Item 13 saves ~25–35% step FLOPs (16× SA reduction is material); B29 corpus mix changes per-step electrode count (SWEC ~64–128, AJILE12 mixed, BT ~150–200). The arch section's "~14.235M params" + "L = 6 latent-stack blocks" count carries through unchanged.

**Contract.** Memos under `memory/project_v14_*` are changelog and provenance; **this doc is the current state.** When a blocker resolves (`docs/neuroprobe/v14_blockers.md` ✅), update both: amend the canonical memo for provenance and edit the corresponding section here for implementation. Do not cascade through 4 stacked amendments at build time; read this.

**Currently locked**: 244 enumerated gaps in `v14_blockers.md`; 19 BIG closed [ARG04/IE09/IE14 SSL clip-length, B07/ARG03 PMA timeline, B01 optimizer/LR/schedule, B02 cross-corpus sampler, **B19 P1/P2 loss-design lock (2026-05-24)**, **B20 v4 invisible front-end lock (2026-05-24)**, **B21 collapse-prevention package (2026-05-25)**, **B22 dense-features amendment (2026-05-25)**, **B03 / B03b / B03c / B03d / B03f mask-discipline lock (2026-05-25 PM)**, **B25 Smooth-L1 default + V-JEPA 2.1 Context Self-Supervision at L_pre_frame (2026-05-27 AM, SUPERSEDED PM by B26)**, **B26 V-JEPA-2.1-fidelity amendment (2026-05-27 PM) — pure L1 across all SSL prediction terms + λ_ctx 0→0.5 warmup over first 25% of P1 + teacher full-input contract + EMA τ=0.999 fixed (drop V-JEPA 1 ramp); CONTEXT LOSS + WARMUP PARTS REVERTED PM-LATE by B27**, **B27 context-loss revert (2026-05-27 PM-late) — drops L_pre_frame_context + λ_ctx warmup after PDF re-read of V-JEPA 2.1 Tables 1+2 found context loss is a dense-prediction tool that costs ~10pp on clip-level SSv2 (72.8 → 62.5 best-case); v14's downstream is clip-level only (PMA → mean → linear) so cost ≠ offsetting benefit + can't replicate uniform 4-level DSS (M2/M3/M4 in different token spaces); keeps B26's pure L1 + fixed EMA + full-input teacher; adds R-context-loss-vjepa21-recipe P1 falsification sister**, **B28 defaults trim + citation cleanup (2026-05-27 PM-late) — (1) DKoleo demoted from default 5-term to 4-term + 3 sister cells (R-dkoleo-batch-cls-unit / R-dkoleo-intra-clip-slots / R-vicreg-slot-variance) + MON-SLOT-REDUNDANCY monitor, after recognizing DINOv2/v3 KoLeo coefficient 0.1 matches numerically but per-batch CLS unit differs from v14's per-clip 320-slot unit (different geometric claim, no precedent for the v14 unit, and B21's identity-anchored init + B22 M3 supervision + dedicated LN per head + reactive Gram already carry the collapse-prevention load); (2) cross-attn count 2 @ {0, 3} → 1 @ {0} per Perceiver-IO standard pattern (the v3 default of 2 cross-attns followed the original Perceiver pattern, NOT Perceiver IO; Perceiver IO appendix explicitly drops repeated cross-attends as "not worth the 303B FLOPs"), saves ~5–8% full-forward / ~12% cross-attn-subset compute, −~0.79M params → ~14.235M total, sister `R-perceiver-original-2-cross-attns` P0; (3) P1→P2 anatomy bias linear warmup λ_anat = 0 → 1 over last 25% of P1 ∪ first 25% of P2 (Option B), replaces the B19 discrete bias-toggle that risked QK miscalibration at the P1→P2 boundary (anatomy bias overlay can shift attention logits by ~5 nats instantaneously), sisters `R-anatomy-bias-step` + `R-anatomy-bias-on-from-p1` P0; (4) B22 citation reframe: V-JEPA 2.1 §2.3.2 Deep SS and DINOv3 §4 Gram anchoring become motivation only (the V-JEPA 2.1 win is contingent on the context loss that B27 reverted; DINOv3 §4 is Frobenius gram anchoring, not mask-pred at depth); mechanism cites V-JEPA-2 predictor (Assran 2025 §2.1); iBOT (Zhou 2022) is conceptual ancestor only, NOT mechanism (CE-on-soft-tokenizer ≠ v14 L1-on-features; corrected 2026-05-27 evening after full-paper re-read); (5) Perceiver IO citation cleanup: v14 = Perceiver-IO encode-process-decode + JFT-variant no-weight-sharing + 1 encoder cross-attn; PMA k=1 IS the Perceiver-IO output-query decoder (k=1 query), but the per-token frame-head Linear and downstream Neuroprobe linear probe are NOT Perceiver-IO decoders (added 2026-05-27 evening); (6) Related-work: HiP (Carreira 2022; future-direction anchor for anatomical hierarchy at scale, NOT adopted per Ben's explicit "throw away the change to perciever HiP") + Perceiver AR (Hawthorne 2022; completeness only); (7) Graphormer rectangular-adaptation framing for the `log(support+ε)` cross-attn bias (added 2026-05-27 evening after Graphormer paper re-read) — additive-QK-bias mechanism cites Graphormer §3.2 spatial encoding, with three explicit adaptations: rectangular electrode×parcel attention (not symmetric node×node), analytical bias (not learned scalar lookup, preserving zero per-subject params per PopT-lineage thesis), and linear-warmup schedule (V-JEPA 2.1 §2.3.1 precedent, not Graphormer — Graphormer has no warmup precedent); R-learned-bias-scaling P1 sister (per-parcel α_p ~80 params) + R-eps-{1e-1, 1e-3} P2 sister. Math correctness of `log(support+ε)` (ε=1e-2: soft-mask unsupported pairs at log(0.01)≈-4.6, neutral at log(1.01)≈0; rectangular shape mathematically valid for softmax over parcel axis) verified during the re-read; no math change needed. HB02 trigger fields untouched; estimate moves favorably; no re-cost**] + 1 MEDIUM closed [M06 Phase-2 sampler]. Top-30 critical path: `v14_blockers_closing_report.md`.

---

## 1. Architecture

**Params.** d = 256, heads = 8 (32 dim/head), ~14.235M total (≈ +23k from B21 identity-anchored init + dedicated LN_frame/LN_utt per loss head, 5/25; + ~1k from B22 LN_mid for M3 supervision, 5/25; **−~0.79M from B28 cross-attn count 2 → 1, 5/27 PM-late**). **N = 6 token blocks** (amended 2026-05-23 from N=4 — sweep placeholder lift; symmetric with L=6, median of comparable iEEG FMs BrainBERT/PopT), L = 6 latent-stack blocks, K = 80 DK parcels × M = 4 slots = 320 latent slots (**identity-anchored init**, B21 lock 2026-05-25). **Single encoder cross-attn @ layer 0 of the latent stack (B28 lock 2026-05-27 PM-late — was 2 @ {0, 3} per v3 default; revised to Perceiver-IO standard encode-process-decode pattern + JFT-variant no-weight-sharing; sister `R-perceiver-original-2-cross-attns` P0 falsifies)**; **M3 (post the single cross-attn @ 0, pre self-attn-0) is a supervised checkpoint via LN_mid (B22 lock 2026-05-25; B28 citation reframe — mechanism cites iBOT or V-JEPA-2 predictor, DINOv3 §4 / V-JEPA 2.1 §2.3.2 are motivation only)**. Each parcel's 4 slots share the same `log(support[e, p] + ε)` cross-attn bias via index map `parcel_of_latent[i] = i // M` carried as a registered buffer; **bias is scaled by `λ_anat(step)` linear-warmup 0 → 1 over last 25% of P1 ∪ first 25% of P2 (B28 Item 3 ✅; was discrete P1→P2 toggle in B19)**. **N and L are sweep placeholders, not principled defaults** — N-sweep {2, 4, 6} and L-sweep {default + neighbors} pre-baked in §8.

**Latent init (B21 ✅ 2026-05-25; see [[project_v14_collapse_prevention_lock_2026_05_25]]).** The 320-slot tensor is NOT a single trunc-normal-init free parameter (the prior v3 default). It is reconstructed at every forward pass from two embedding tables:

```
LearnableParcelEmbed   : (80, d=256)    trunc-normal std=0.02     [21,504 params]
LearnableSubSlotEmbed  : (4,  d=256)    trunc-normal std=0.02     [1,024 params]
ε                      : N(0, 0.02²) per (p, s, d)  — broken-symmetry noise at construction only

z_init[p·M + s] = LearnableParcelEmbed[p] + LearnableSubSlotEmbed[s] + ε
```

`(p, s)` are recovered from the slot index via the existing `parcel_of_latent[i] = i // M` and `sub_slot_of_latent[i] = i % M` index maps. Gradients flow back to both embeddings through the standard cross-attn + latent-stack path; no special handling. Identity-anchored init is the symmetry-breaker that lets cross-attn ❺ pool meaningfully in Phase 1 when the anatomy bias is OFF (NuCLR Arora 2025 + DINOv3 § 4 + DINOv2 KoLeo precedent — POYO+'s unanchored learnable per-unit embedding scored F1=0.3521, worst baseline). The construction preserves the activation-magnitude invariant at first forward (sum of two N(0, 0.02²) matches the prior single-tensor std=0.02 to within a small constant).

**Init policy (uniform).** All `nn.Linear`, MLP layers, learnable embeddings (freq embedding, PMA query, LearnableParcelEmbed, LearnableSubSlotEmbed): truncated-normal init std=0.02 (BERT / ViT / Perceiver-IO / DINOv3 convention). LayerNorm γ=1 β=0 (PyTorch default; do not override). No mixed init across the model. **The 320 latent slots are NOT a single free tensor under B21** — they're reconstructed every forward from LearnableParcelEmbed + LearnableSubSlotEmbed + ε; see "Latent init" above.

**Norm.** LayerNorm throughout (V-JEPA 2.1 convention at this scale; RMSNorm is LLaMA-family preference, not vision/biosignal-FM). `F-RMSNorm` P2 sister swaps `nn.LayerNorm → nn.RMSNorm` (PyTorch ≥2.4 native) with eps 1e-5 → 1e-6.

**RoPE.** Per-block at each MHA layer's Q/K independently (Llama / V-JEPA / Mistral standard), per-head rotation (not shared). Precompute table at `max_seq_len=128` (covers SSL T=73 + safety margin); slice to actual T at forward; pre-P1 unit test asserts `slice_from_128 ≡ build_at_T` for T ∈ {15, 73}.

**Registers / dropout / stochastic-depth.** None. Parcel-latents act as structured queries (no `[CLS]` / register / sink tokens needed); `F-registers` P2 sister added only if attention-pattern visualization shows sink artifacts. Dropout = 0.0 and stochastic-depth = 0.0 throughout SSL (DINOv3 / V-JEPA / data2vec convention at <100M); do not instantiate `nn.Dropout(0)` modules (still adds graph nodes under `static_graph=True`). Phase-4 probe-side dropout 0.1 is a sister cell only.

**Factorization order.** Token blocks JOINT (t·f) self-attention (B20 v4 lock 2026-05-24 — was factorized t-then-f in v3); latent stack time-then-parcel (TimeSformer divided). `N2-factorization-order` P1 sister swaps the latent-stack order; `F-joint-vs-factorized` P0 sister tests token-block joint-vs-factorized (red-team note: factorization saving disappears at v4's 200-token-per-electrode count; AST/Audio-MAE/EAT/SSLAM modal-recipe parity uses joint).

**Forward path.**

```
Preproc → (C electrodes, F=30 bins, T=clip_len × 8) tensor

❶ A1 patch-embed   Conv2d(1, d=256, kernel=(3, 2), stride=(3, 2)) non-overlap (B20 v4 lock)
                   + per-patch freq embedding (10 learnable d-vectors, additive)
                   NO time PE at input (RoPE in ❷)
                   → (C, F_p=10, T_p=T/2, d)

❷ Token block × N=6   per electrode, JOINT (t_p·f_p) self-attention (B20 v4 lock):
                      pre-norm, GeLU, MLP(4×), RoPE Q/K per-head
                      hard −∞ cross-electrode mask
                      → (C, F_p=10, T_p, d)    [conceptually preserved; SA flat over 10·T_p tokens]

❺ Cross-attn @ layer 0   (B28 lock 2026-05-27 PM-late — single encoder cross-attn, Perceiver-IO standard pattern; was ❺a + ❺b @ {0, 3} in v3; sister `R-perceiver-original-2-cross-attns` P0 restores the 2-cross-attn pattern)
                          pools (electrode, f_p) → parcels, strict 1:1 per time-patch step:
                          K/V = (C × 10) F-patch tokens at time-patch step t_p
                          key_padding_mask = pad_mask_C_MAX | shaft_mask    (B03 ✅; shaft_mask=False outside P2 student)
                          bias = λ_anat(step) · log(support[e, p] + ε), ε = 1e-2 (DK one-hot)
                          λ_anat(step) = 0 for step < 0.75·N_P1
                                       = linear ramp 0 → 1 over (0.75·N_P1, N_P1 + 0.25·N_P2)
                                       = 1 for step ≥ N_P1 + 0.25·N_P2          (B28 Item 3 ✅; was discrete P1→P2 toggle)
                          LayerNorm on latent Q-stream entering this cross-attn (collapse insurance)
                          → M3: (320, T_p, d)    ← LATENTS CARRY TIME-PATCH AXIS; B22 SUPERVISION POINT

❺′ M3 head split (B22 ✅ 2026-05-25; SSL phases only — bypassed at P4):
    LN_mid   ← LayerNorm(d=256)             feeds Loss 2′ (L_mid_slot)  [~512 params]
    teacher has matching LN_mid_T as EMA copy
    student / teacher M3 routed through their respective LN_mid for the L_mid_slot L1
    (NB: the latent stack continues to receive RAW M3 — LN_mid is on the loss path, not the forward path)

❻ Latent self-attn blocks 0..L−1 (L = 6, B28 — was split 3+3 around ❺b @ position 3 in v3; now contiguous L=6 downstream of the single cross-attn @ layer 0):
                                  factorized (time-patch × parcel) per block:
                                  time-patch SA (RoPE) → parcel SA → MLP(4×)
                                  parcel-SA key_padding_mask = ~slot_mask(parcels_supervised[subject])    (B03b ✅; all phases)
                                  → M4: (320, T_p, d)    ← LOSS HEAD DIVERGENCE POINT

❻′ M4 head split (B21 ✅ 2026-05-25; SSL phases only — bypassed at P4):
   LN_frame ← LayerNorm(d=256)              feeds Loss 3 (L_post_frame)  [~512 params]
   LN_utt   ← LayerNorm(d=256)              feeds Loss 4 (L_post_utterance) via PMA  [~512 params]
   teacher has matching LN_frame_T / LN_utt_T as EMA copies (mirror; same dedicated split)

❼ Readout (phase-asymmetric, see Phase 3 / Phase 4):
   parcel-collapse (shared): PMA k=1 learned seed → (T_p, d) per clip
                             SSL: PMA reads LN_utt(M4); P4: PMA reads raw M4 (LN_utt bypassed)
   Phase-3 SSL → triangular pool / linear-interp to preflight-chosen rate ∈ {5, 8, 10, 20} Hz × buckets → (n_buckets, d)
                 default candidate: 10 Hz / 50 buckets (Goldstein syllable rate); requires upsample from 8 Hz frame rate
   Phase-4 downstream → NO time pool; flatten (T_p, d) → T_p·d → per-task Linear
```

**Why factorized at the latent stack but joint at the token blocks (B20 v4 lock).** Latent stack t_p×parcel — structural separation, axial-attention precedent (Ho 2019 / DPTNet / TF-GridNet). Token blocks at v4's small per-electrode token count (10·T_p ≈ 200 at P1, 40 at P4) drop the factorization saving — the 2-layer + 2-MLP overhead of factorized exceeds joint's O((10·T_p)²·d) attention cost. AST/Audio-MAE/EAT/SSLAM all use joint at this scope; only PaSST uses factorized PE. Cross-frequency-coupling counter-argument moot: input is filterbank magnitude (5/25 swap dropped the log step; F-log-amplitude sister keeps the option), phase discarded at preprocessing, so phase-amplitude CFC is gone before the model sees anything either way.

**Why latents keep a time-patch axis.** Phase-3 cross-modal target is a 50-token syllable-rate sequence (Whisper-L8 mean-pool-by-5). Time-collapsed latents would force a static representation to match 50 temporally-distinct teacher buckets — degenerate. Time-collapse also kills cross-parcel temporal dynamics (token blocks are per-electrode masked). v4 reduces T to T_p (T_p = T/2 under Conv2d (3, 2)), so the latent T-axis is T_p — still preserves the temporal-patch sequence the Phase-3 triangular pool / linear interp consumes.

**Why identity-anchored latents (B21 ✅ 2026-05-25).** Two redundant identity signals are now present at every cross-attn ❺ forward: (a) the additive `log(support[e, p] + ε)` bias on QK logits (zeroed in P1, ON in P2/P3/P4), and (b) the LearnableParcelEmbed[p] component of the latent slot itself (always ON). When P1 zeroes the bias for the SWEC anatomy-blind corpus, the latent-side parcel signal still differentiates 320 slots — so QK content-only routing has a non-degenerate target to attend to. Pure trunc-normal free init (the prior v3 default) failed the precedent check: POYO+'s identical regime scored F1=0.3521 (worst baseline) on IBL brain-region prediction (NuCLR Arora 2025); DINOv3 § 4 documents the same collapse pattern at scale and prescribes identity-anchored slots + Gram anchoring; DINOv2 KoLeo confirms slot-level repulsion is the right operator. Sub-slots within a parcel share LearnableParcelEmbed[p] but differ by LearnableSubSlotEmbed[s] — within-parcel symmetry breaks at init, not after thousands of steps of hope.

**Mask-discipline matrix (B03 ✅ 2026-05-25; see [[project_v14_b03_mask_lock_2026_05_25]]).** Four mask layers across all four phases:

| Mask | Active in | Mechanism | Realized as |
|---|---|---|---|
| C_MAX pad | all phases, all sides | structural | -1e4 (B12) sentinel at `key_padding_mask[c >= n_real[subject]]` at every cross-attn / latent-SA |
| Per-electrode patch | P1+P2 student only | B03c paradigm B = **drop visible-only + 2-block predictor (~0.2M, discarded after P1)** | encoder runs on packed `visible_patches[C, F_p · T_p · (1-r), d]`; predictor fills masked positions before L_pre_frame |
| Shaft-block electrode | P2 student only | B03 = **drop into key_padding_mask** | `key_padding_mask = pad_mask_C_MAX \| shaft_mask` at cross-attn ❺a/❺b |
| parcels_supervised[subject] | all phases (gate for slot-level supervision AND latent-SA key_padding_mask) | B03b + B03f = **per-subject** (not per-clip), invariant to within-clip masks; SWEC fallback = all 320 | latent-SA `key_padding_mask=~slot_mask(parcels_supervised[subject])`; PMA softmax over parcels masked to parcels_supervised; L_mid_slot/L_post_frame/L_post_utterance/L_DKoleo scoped to parcels_supervised |

P2 EMA teacher sees **full electrodes** (no shaft mask, no patch mask) — B03d. The teacher-student asymmetry is what produces the JEPA prediction task for shaft-orphaned parcels. Teacher's key_padding mask is C_MAX pad only.

**Why drop, not [MASK] embed.** Symmetric with shaft-mask = drop (V-JEPA/Brain-JEPA/MAE/I-JEPA family) and saves ~12% P1 step time (encoder skips ~50% of patches). Sister `BiL-B03-PatchMask-Paradigm-A` falsifies if the saving costs downstream accuracy.

**Why parcels_supervised[subject], not valid_parcels[clip].** Per-clip `valid_parcels` recomputed post-shaft-mask in P2 would EXCLUDE shaft-orphaned parcels — but those orphaned parcels ARE the JEPA prediction targets (teacher has signal there, student must infer from context). Per-subject `parcels_supervised` is the JEPA-correct supervision set. Sister `R-clip-level-valid-parcels` falsifies.

**PMA k=1 query training timeline (B07/ARG03 ✅; amended 2026-05-24 B19).** Trained in Phase 1 (gradient via `L_post_utterance` — the M4 PMA-pool over parcels → mean-over-T → (d,) utterance vector) AND Phase 2 (same `L_post_utterance` term + electrode-mask asymmetry on M4) AND Phase 3 (gradient via Smooth-L1 vs Whisper-L8). Frozen at Phase 4. Implementation: `freeze=False` default through P1/P2/P3; loop calls `model.pma.query.requires_grad_(False)` before Phase-4 probe instantiation. Same shared PMA-k=1 query across all phases — never re-initialized between phases.

**M3 supervision (B22 ✅ 2026-05-25; see [[project_v14_b22_collapse_prevention_dense_features_2026_05_25]]).** The post-cross-attn-1 / pre-self-attn-0 representation (M3, `(320, T_p, d)`) is supervised in P1+P2 by a fourth loss term `L_mid_slot = MSE(loss_mid_head(LN_mid(M3_student)), LN_mid_T(M3_teacher_avg))` over `(parcel ∈ valid_parcels[clip], t_p)`, divided by `|valid_parcels|·T_p·d` (same per-clip normalization as `L_post_frame`). LN_mid (~512 params) and `loss_mid_head` (1-layer linear `(d,d)`) are NEW B22 params; teacher carries an EMA-tracked LN_mid_T mirror. Weight 1.0. **No predictor on the student side at M3 — cross-attn-1 alone IS the predictor for L_mid_slot** (the deeper stack is the predictor for L_post_frame@M4). M3 is downstream-isolated from the latent stack: the latent stack receives RAW M3 through its first self-attn block; LN_mid is on the LOSS path only, NOT on the forward path. V-JEPA 2.1 §2.3.2 Deep SS (Table 12: 42.0 → 43.9 ADE20k mIoU, last-layer probe ≈ best-of-4 under deep SS) is the precedent — same JEPA mask-pred objective at an intermediate encoder checkpoint pushes meaningful structure into every supervised location, not just the loss-head one.

**Phase-1 cross-attn-ON, anatomy-bias linear-warmup (B19 + B28 Item 3 ✅ 2026-05-27 PM-late).** The full ❶❷❺❻ stack is forward-pass-active in Phase 1; the `log(support[e,p] + ε)` bias term in ❺ is gated by a scheduled scalar `λ_anat(step)` that is **0 for the first 75% of P1 steps** (QK content-only routing — what SWEC needs anyway, since SWEC has no electrode anatomy), then **linearly ramps 0 → 1 over the last 25% of P1 ∪ the first 25% of P2** (B28 Option B), then stays at 1 for the remainder of P2/P3/P4. This replaces the prior B19 discrete toggle (`bias_enabled=False` in P1, `bias_enabled=True` in P2 from step 0), which risked a ~5-nat QK shift at the P1→P2 boundary that would override the P1-learned routing pattern before the slots could integrate the anatomical scaffold. The linear warmup lets slots first specialize unconstrained (P1 first 75%) → gradually overlay the anatomy scaffold onto the existing routing pattern (P1 last 25% + P2 first 25%) → run with full bias for the rest of P2. Same warmup precedent class as V-JEPA 2.1 §2.3.1 (B26's λ_ctx 0 → 0.5 schedule, reverted by B27 along with the context loss itself; the warmup pattern is reused here for a different scheduled parameter). Sisters `R-anatomy-bias-step` (P0; B19 discrete toggle baseline) and `R-anatomy-bias-on-from-p1` (P0; λ_anat = 1 from P1 step 0, stress-tests whether the bias-OFF-in-P1 SWEC justification chain is over-cautious) bracket the warmup design.

**Geometry / electrode depth.** OUT of first pass. `F-depth-bias` sister (learned `B[m, depth]` table) settles empirically.

---

## 2. Preprocessing

**Recipe** (BT 2048 Hz, Nyquist 1024 Hz; uniform across corpora):

```
HPF 0.5 Hz
→ comb @ mains_hz                     scipy.signal.iircomb(w0=mains_hz, Q=30, ftype='notch', fs=fs)
                                      # **Per-corpus** (lift to dispatch field; see MASK-01 in v14_implementation_fix_list.md):
                                      #   60 Hz for BT / D-cohort / AJILE12 (US sites)
                                      #   50 Hz for SWEC (CH site)
                                      # Currently `dispatch_v14.py:123` hardcodes notch_filter=60.0 — must lift before SWEC pretrain.
→ MNE-LOF channel-quality flag
→ Ref draw (REF-AUG, 5/27 PM lock; sister-first dispatch — see §"Ref-aug dispatch" below):
    per-clip uniform-random over {shaftCAR, bipolar, Laplacian}, NOT raw
    one ref per clip (NOT multi-view); same-view V-JEPA contract preserved
    SWEC degenerates to global-CAR-only (no shaft annotation)
    bipolar drops 1 channel per shaft (n → n-1); C_MAX pad mask recomputed per-clip
→ drop bad channels
→ slice
→ Multi-STFT (iMINDBench Appendix E + v14 high-band extension):
    STFT_low  Nperseg=1024  (~2–40 Hz)
    STFT_mid  Nperseg=512   (~20–148 Hz)
    STFT_hi   Nperseg=256   (~80–813 Hz, extended past iMINDBench's 248 Hz cap)
    common hop = 256 samples @ 2048 Hz → 8 Hz frame rate (B20 v4 lock; was hop=128 → claimed 14.7 Hz, actual 16 Hz at hop=128)
→ Triangular ⅓-octave filterbank, 30 log-SPACED bins, mel-style edges, uniform spacing
    f_k = 2^(k/3) Hz, k = 0..29
    k0–k14 sourced from STFT_low, k15–k21 from STFT_mid, k22–k29 from STFT_hi
                                       (5/25 swap: post-filterbank value-axis log step dropped.
                                        Raw filterbank magnitude is now the default;
                                        F-log-amplitude sister recovers `log(energy+ε)` via apply_log=True.)
→ Nv14 robust-z per (electrode, freq-bin, session):
    (x − median) / (1.4826 · MAD), invalid bins skipped during stats
→ (C, F=30, T=clip_len × 8)
```

**Frame rate is pinned at 8 Hz** (the Multi-STFT hop @ 256 samples; B20 v4 lock). Phase 1/2/3 = 5s clips, T=40 (→ T_p=20 after Conv2d (3,2) at A1). Phase 4 = 1s clips, T=8 (→ T_p=4).

**Ref-aug dispatch (5/27 PM lock; see `[[project_v14_ref_aug_input_distribution_lock_2026_05_27]]`).** Sister-first protocol gates the full P1 all-corpora rollout. Per-clip ref draw uniform over `{shaftCAR, bipolar, Laplacian}`; raw skipped per per-corpus definitional ambiguity (each corpus ships with different baseline ref). Optional `ref_embed: (3, d=256)` learnable additive at A1 patch-embed output (default ON; sister `R-no-ref-cond` drops it). Effect lands as 3 distinct *distributions* per band — common-mode dominates δ/θ (~50–60% relative difference across refs), shrinks to ~1–4% in HG/v-HG (the band v14's parcel-routed cross-attn reads); ref-aug therefore exercises patches 0–3 most, barely touches patches 7–9.

**5-step dispatch protocol**:
1. **Smoke (~1h, free, Nano).** No NaN/inf; all 3 refs hit at ~1/3; loss curve direction matches `R-no-ref-aug` to ±20% at step 100.
2. **Sister falsification (5–10 H100-h, Lite BT, gate on full P1).** Paired runs ref-aug ON vs ref-aug OFF at 5k steps, same seed. **Kill criteria** (any one fires → defer to post-paper): (a) HG-patch (6–9) `L_pre_frame_masked` regresses > 5%, (b) MON-MASK-002 ratio leaves `[0.7, 1.5]` sustained 1k steps, (c) Monitor F1 deviates > 0.1, (d) MON-MASK-004 subject-ID F1 ↑ > 0.05 over sister.
3. **Full P1 rollout (~600 H100-h, all corpora) IF sister passes.** SWEC + AJILE12 + D-cohort + BT. SWEC degenerates to global-CAR-only — flag as known SWEC caveat.
4. **Mid-P1 probe (~50 H100-h, step 50k).** Frozen linear probe on BT held-out validation slice; kill if HG-band within-session AUROC regresses > 0.005 vs `R-no-ref-aug` parallel run.
5. **Full Neuroprobe eval (~0 extra).** Ref-aug = part of headline only if it wins ≥ 0.005 on CSubject AUROC.

**Storage if sister passes**: 3× Multi-STFT cache on `/work/ht203/cache/multi_stft/`. SWEC ~15–24 TB, AJILE12 ~3 TB, D ~0.5 TB, BT ~0.15 TB → **~18–30 TB total**, structurally invisible against 233 TB free per `[[project_v14_hb01_multistft_cache_location_2026_05_23]]`. Regenerate ~75 CPU-hr on 32 workers (3× single-ref cost).

**HB02 trigger-fields check**: {d, N, L, steps, batch, electrode count, F bins, STFT windows} all UNCHANGED. Per-step compute delta is dataloader-side (CPU-worker parallelized) = 0%. No HB02 re-cost.

**Sister roster** (cells under `ablations.md §6`): `R-no-ref-aug` (P0 must-run; paired falsifier; fixed shaftCAR — the current default), `R-no-ref-cond` (P1; ref-aug ON, embedding OFF), `R-ref-aug-2cell-shaftCAR-Laplacian` (P1; drop bipolar), `R-ref-aug-broadband-only` (P1 contingency; skip ref-aug on SWEC since it degenerates), `R-ref-aug-4cell-add-raw` (P2; only if sister passes by ≥0.01 on CSubject).

**Post-A1 patch grid (B20 v4)**: F_p=10 freq patches × T_p time patches per electrode. Per-corpus valid-bin mask propagates to a **valid F-patch mask**: an F-patch (3 bins wide) is INVALID for a corpus if any of its 3 bins is invalid. Resulting valid F-patches: SWEC k0–k21 → patches 0–6 (7 valid; patch 7 partial = invalid, patches 8–9 invalid); AJILE12 k0–k20 → patches 0–6 (same); BT / D-cohort → all 10 patches. Alternative (zero-fill partial patches before Conv2d) carried by sister `F-patch-partial-valid`.

**Per-corpus valid-bin mask** (closes corpus-conditioned bin set; recipe memo §"Sample-rate handling"):

| Step | Behavior on invalid bins | Reason |
|---|---|---|
| Robust-z stats | Skip invalid bins (don't zero-fill before stats) | median/MAD on post-band-pass noise floor meaningless |
| Input fill (post-z) | 0 (= z-score median = neutral) | Fixed-shape tensor |
| Freq-SA key mask | −∞ added to attention logits at invalid keys before softmax | Otherwise zero-tokens contribute weight |
| Freq-SA query at invalid bin | Output zeroed (not propagated) | Don't carry garbage |
| L_recon target mask | Exclude invalid bins from Smooth-L1/MSE sum; divide by valid count | **Critical** — otherwise model learns "k22–k29 = 0" trivially from SWEC |
| EMA teacher | Same mask pattern; teacher's invalid-bin outputs never targets | Keeps teacher noise out of student gradient |
| Categorical freq embedding | Indices 0–29 always; no gradient at invalid-bin indices for that corpus | Embedding for k22–k29 trained only on broadband corpora — correct |

**Valid bins per corpus.**
- **SWEC**: k0–k21 (band-pass 0.5–120 Hz @ 1024 Hz sample, content ceiling well below Nyquist).
- **AJILE12**: k0–k20 (band-pass 0.5–200 Hz @ 500 Hz sample, Peterson 2022 / Charmander). Bins k21–k29 carry zero signal — same valid-bin-mask machinery as SWEC.
- **BT, D-cohort**: all 30 bins (broadband, 2048 Hz sample, no anti-alias cap below 813 Hz).

**Per-corpus exact totals**, computed once at sampler-build from the events manifest (NOT corpus-medians):
```
vb_eh[corpus] = Σ_session (session_hours × session_n_electrodes × |valid_bins[corpus]|)
```
Used by the B02 §3 sampler. See §3 "Exact totals, NOT corpus-medians" for the LLM-pretrain precedent (Llama 3 §3.1.2, OLMo 2 Table 4, Megatron `BlendedDataset`).

**Fill ordering invariant.** log → robust-z (skip invalid per mask) → fill = 0 (post-z). Parity test enforced in suite.

**SWEC anatomy assert (amended 2026-05-24 B19).** SWEC has no electrode anatomy. Loader-level invariant: `if corpus == 'SWEC': assert sample.has_anatomy is False and current_phase == 'P1'`. SWEC samples DO reach cross-attn ❺ in P1 now — but P1's `bias_enabled` is globally `False` (anatomy bias zeroed across the entire P1 batch), so the SWEC samples' absence of anatomy is no longer a routing problem (no per-sample anatomy lookup is needed when the bias term is zeroed wholesale). The prior "route only through ❶❷ and never reach ❺" half of this invariant is RETIRED. The SWEC valid_parcels mask for L_mid_slot and L_post_frame falls back to the union of all parcels (no per-sample anatomy → use all 320 latent slots, weighted equally; equivalent to disabling the anatomy-filtered divisor of L_mid_slot / L_post_frame for SWEC samples). Pattern still matches IE08 sparse-mask invariant for the broadband corpora.

**Per-session normalization cache.** Robust-z stats precomputed offline per `(subject, session)` to a parquet under `EXCA_CACHE_FOLDER`, looked up at `__getitem__`. Online running-stat would drift across the 8.2k h corpus. For Phase-4 CSubject eval on held-out subjects (no train-side per-subject stats), use train-cohort pooled stats per (electrode-group, freq-bin); coupled to B13 lock.

---

## 3. Phase 1 — SSL pretrain (full ❶❷❺❻ stack, anatomy bias OFF)

**Goal.** Per-electrode spectro-temporal features AND content-routed cross-attn pooling AND clip-level utterance pooling — all trained jointly on the 8.2k h unpaired corpus with anatomy bias zeroed. Replaces the prior "Stage A only" framing where ❺❻ were frozen / un-instantiated in P1.

**Trainable params** (~14.235M — full v14 stack post B19 + B21 + B22 + **B28** 2026-05-27 PM-late amendments; was "~6.6M Stage-A-only" before B19, "~15M" after B19, "~15.023M" after B21 identity-anchored init + dedicated LN_frame/LN_utt, "~15.024M" after B22 LN_mid, **"~14.235M" after B28 cross-attn count 2 → 1**):
- A1 patch-embed (~0.01M)
- Freq categorical embedding (~0.01M)
- 6 token blocks ❷ at d=256 ≈ 6.3M (1.05M × N=6)
- **1 cross-attn ❺ ≈ 0.79M** (B28 Item 2 — was 2 @ {0, 3}; the `log(support+ε)` bias is a buffer, scaled by the scheduled `λ_anat(step)` scalar; λ_anat = 0 throughout P1 first 75% via the warmup)
- LearnableParcelEmbed (80, d=256) ≈ 0.022M  (B21 identity-anchored init; replaces the prior single-tensor 320 free latents)
- LearnableSubSlotEmbed (4, d=256) ≈ 0.001M   (B21 identity-anchored init)
- 6 latent-stack blocks ❻ ≈ 6.3M (contiguous L=6 downstream of the single cross-attn @ layer 0 per the forward path in §1; was split 3+3 around ❺b @ position 3 in v3)
- LN_mid at M3 (post the single cross-attn ❺) ≈ 0.001M   (B22 dedicated LN for L_mid_slot)
- loss_mid_head (1-layer Linear(d, d)) ≈ 0.066M  (B22; for L_mid_slot only; discarded if isolating Loss 2′ at eval)
- LN_frame + LN_utt at M4 divergence ≈ 0.001M  (B21 dedicated LN per loss head)
- PMA k=1 query ≈ 0.8M (**B31 ✅, 2026-05-28 PM**: NOT trained in joint SSL phase — no `L_post_utterance` in default. PMA receives gradient only at P3 via Whisper-L8 distillation, then is frozen at P4. The B19 line "trained by L_post_utterance; same query reused at P2/P3/P4" is the pre-B31 historical contract (superseded by B31); closes the long-standing B19-vs-code drift. Code state (verified 2026-05-29): `V14ParcelCollapsePMA.__init__` default is `freeze=False` (`v14_encoder.py:1265`); the P4 freeze is configured by `pma_freeze=True` (`v14_encoder.py:1461`) and applied at the P4 construction site (`v14_encoder.py:1522`), not by a wrong `__init__` default.)
- 2-block predictor (transformer, hidden=128, heads=4, ~0.2M) — for L_pre_frame only; discarded after Phase 1
- *(Audit-note 2026-05-27 PM-late: B21 added ~0.023M params over B19's ~15M; B22 adds ~0.001M LN_mid (+ ~0.066M loss_mid_head, discardable at eval); B28 subtracts ~0.79M cross-attn (2 → 1); net ~14.235M, well under the 30M cap.)*

**Frozen.** Nothing (full stack trains). The `λ_anat(step)` scalar on ❺'s bias buffer is the only thing scheduled (warmup 0 → 1 over last 25% of P1 ∪ first 25% of P2 per B28 Item 3).

**Loss (B19 ✅, 2026-05-24 + B21 amendment ✅, 2026-05-25 + B22 amendment ✅, 2026-05-25 + B25 amendment ✅, 2026-05-27 AM + B26 V-JEPA-2.1-fidelity amendment ✅, 2026-05-27 PM + B27 context-loss revert ✅, 2026-05-27 PM-late + B28 DKoleo-demote ✅, 2026-05-27 PM-late — see [[project_v14_loss_design_lock_2026_05_24]], [[project_v14_collapse_prevention_lock_2026_05_25]], [[project_v14_b22_collapse_prevention_dense_features_2026_05_25]], [[project_v14_loss_design_amendment_b26_2026_05_27]], [[project_v14_loss_design_amendment_b27_2026_05_27]], [[project_v14_loss_design_amendment_b28_2026_05_27]]).** Four unified bootstrap mask-prediction losses (V-JEPA-2-canonical masked-only at M2; B22 checkpoint at M3; frame + utterance at M4) computed in a single student forward, summed with **fixed coefficients (1, 1, 1, 1)** joint from step 1. **B28 (2026-05-27 PM-late) demotes the prior `0.1 · L_DKoleo@M4` regularizer from the default to three sister cells + MON-SLOT-REDUNDANCY** after recognizing the per-clip 320-slot DKoleo unit diverges from DINOv2/v3's per-batch CLS unit (different geometric claim; no precedent for the v14 unit; B21's identity-anchored init + B22 M3 supervision + dedicated LN per head + reactive Gram already carry the collapse-prevention load). **B27 (2026-05-27 PM-late) drops B25's L_pre_frame_context companion + B26's λ_ctx warmup schedule** after PDF re-read of V-JEPA 2.1 Tables 1+2 + §2.3.1 found the context loss is a dense-prediction tool (their SSv2 clip-level: V-JEPA 2 baseline 72.8 → V-JEPA 2.1 best-case with λ=0.5+weighted+warmup 62.5, only Deep Self-Supervision recovers to 72.1); v14's Phase-4 readout is clip-level only (PMA → mean → linear) and v14 can't replicate uniform 4-level DSS because M2/M3/M4 live in different token spaces, so context loss would be cost without offsetting benefit:

```
**B31 default (2026-05-28 PM — see [[project_v14_b31_vjepa2_canonical_loss_2026_05_28]]):**

L_total = L_pre_frame_masked @ M2  +  L_post_frame @ M4
                  (pure L1)                  (pure L1)

PMA: NOT instantiated on the SSL graph. Joint SSL phase has zero
gradient on PMA query, LN_utt, and any utterance-head modules.
PMA gets gradient at P3 (Whisper distillation), then is frozen at P4.

[B31-deferred sister R-add-m3-loss        : + 1.0 · L_mid_slot       @ LN_mid(M3)]
[B31-deferred sister R-add-utterance-loss : + 1.0 · L_post_utterance @ LN_utt(PMA(M4))]
[B31-deferred sister R-add-both           : both of the above]

[reactive: + 0.1  · L_Gram        if M4 trigger fires]                       ← B21 (carryover; only L_post_frame's monitor arms it now)
[B22 reactive DKoleo@M3 arm is moot — no L_mid_slot in the default path]

--- PROVENANCE BELOW (B19/B22 4-term, superseded by B31; kept for sister-cell reference) ---

L_total = L_pre_frame_masked  +  L_mid_slot  +  L_post_frame
        +  1.0 · L_post_utterance

[reactive: + 0.1  · L_Gram        if M4 trigger fires as defined below]                                  ← B21 (carryover, unchanged)
[reactive: + 0.05 · L_DKoleo@M3   if M3 cos>0.7 sustained 50k OR parcel-ID F1@M3 < 0.4 sustained 50k]   ← B22 arm 3 (still gated; mechanism routes through the chosen B28 DKoleo variant)
```

**B28 DKoleo sisters (P1; settle the unit empirically; MON-SLOT-REDUNDANCY gates dispatch).** Three replacement cells for the demoted default — see [[project_v14_loss_design_amendment_b28_2026_05_27]] §"Item 1" for full math:

- `R-dkoleo-batch-cls-unit` — per-batch DKoleo on the utterance-pooled CLS-analog `(B, d)` vector (DINOv2 Algorithm 1 + DINOv3 §3.3-faithful unit), coeff 0.1.
- `R-dkoleo-intra-clip-slots` — per-clip DKoleo on M4 slot means over all 320 slots (B21's prior default, retained as the v14-original falsifier), coeff 0.1.
- `R-vicreg-slot-variance` — VICReg variance hinge per slot dimension over the per-clip 320 slots, no covariance term (Bardes/Ponce/LeCun 2022, arXiv:2105.04906), coeff TBD via 1-GPU-h BT-Lite preflight over {1, 5, 10, 25}.

**MON-SLOT-REDUNDANCY monitor (always-on, every 10k steps).** Logs per-clip 320-slot off-diag cosine (`mean / max / pct95`) AND per-batch CLS-analog off-diag cosine on a held-out 256-clip probe batch. Escalation rules: `per_clip_cos.pct95 > 0.7 sustained 50k` → `R-dkoleo-intra-clip-slots`; `batch_cos.pct95 > 0.7 sustained 50k` → `R-dkoleo-batch-cls-unit`; both OR `per_clip_cos.diag-zeroed.mean > 0.5 sustained 50k` → `R-vicreg-slot-variance`. The B21 reactive-Gram anchor at M4 (Component E, weight 0.1) remains the parallel collapse rescue at the geometry level (Frobenius anchor to a snapshot of the student backbone), separate from the diversity-regularizer family.

**B27 sister cell** `R-context-loss-vjepa21-recipe` (P1, single cell) — full V-JEPA 2.1 §2.3.1 recipe at M2: pure-L1 visible-patch supervision with `λ_i = 0.5 / √d_min(i, M)` weighting (Chebyshev d_min on per-electrode F-patch × T-patch grid) + linear warmup 0 → 0.5 over first 25% of P1. Single falsification cell — if it beats the V-JEPA-2-style default on Neuroprobe gates, adopt B25/B26's recipe in a future amendment. The B26 sisters `R-no-context-loss`, `R-no-warmup`, and `R-ctx-lambda-{0.0, 0.25, 0.5, 1.0}` are retired by B27 (subsumed by default; or collapsed to this single cell).

**Loss form: pure L1 (‖·‖_1) across all four SSL prediction terms (B26 ✅, 2026-05-27 PM — supersedes B25's Smooth-L1; retained by B27 PM-late).** L_pre_frame_masked, L_mid_slot, L_post_frame, and L_post_utterance all use `nn.L1Loss(reduction='none')` then masked mean. Direct match to V-JEPA 2 §2.1 Eq 1 (`‖P_ϕ(Δy, E_θ(x)) − sg(E_θ̄(y))‖_1`) — masked-only supervision at M2 is V-JEPA-2-canonical. B25's prior Smooth-L1 β=1.0 default was retracted (B26) after PDF re-read found all three cited precedents (V-JEPA 2 §2.4, V-JEPA 2.1 §2.3.1, data2vec 2.0 §3.4) actually use either L1 or L2, NOT Smooth-L1 — citation chain was empirically false. Phase-3 Whisper-L8 distillation **stays Smooth-L1 β=1.0** (cross-modal regression at a different abstraction level; see §5). Sister `R-smoothl1-beta-{0.5, 1.0, 2.0}` (P1, repurposed from B25 default) brackets the prior B25 form; sister `R-l2-loss` (P0, NEW B26) tests pure L2 (data2vec 2.0 §3.1 form). Sister `R-mse-loss` (P0, retained from B25; redundant with R-l2-loss, can be merged downstream).

**Teacher contract for ALL prediction terms (B26 ✅, 2026-05-27 PM — locks the B25 ambiguity).** The EMA teacher encodes the FULL unmasked input (full electrodes + full patches; no shaft mask, no patch mask; only the per-corpus valid-bin mask). Every loss compares **student-predictor-output-at-position vs teacher-encoder-output-at-same-position on the full pass**. The asymmetry (student sees masked input; teacher sees full input) IS the JEPA supervision source. Wherever the loss-term language below says "teacher's X" it means "teacher's X on the full-input pass at the same position the student is being supervised at."

- **`L_pre_frame_masked`** — gradient on M2 (token-block) output `(C, F, T, d)`. Per-electrode per-(t, f) **L1** over the input-patch mask intersected with the per-corpus valid-bin mask. **Paradigm B (B03c ✅ 2026-05-25)**: encoder runs only on visible patches; a 2-block transformer predictor (hidden=128, heads=4, ~0.2M, discarded after P1) takes `LearnedMaskQuery[d] + PE` queries at masked grid cells, cross-attends to `M2_visible` as K/V, outputs `M2_full[masked_positions]`. L1 is then computed at `M2_full[masked_positions]` against teacher's M2 from the full-input pass (B26 contract). Teacher target is **all-N=6 layer-averaged with per-layer instance-norm** (EAT §3.1 + data2vec 2.0 recipe; B11 ✅ K=6 lock).
- **`L_pre_frame_context` — DROPPED by B27 (2026-05-27 PM-late).** Was B25's V-JEPA 2.1 §2.3.1 Eq 2 Context Self-Supervision companion; was B26's λ_ctx-warmup-scheduled visible-patch supervision. Removed from default after V-JEPA 2.1 Table 1+2 PDF re-read found the context loss is a dense-prediction tool that costs ~10pp on clip-level SSv2 (their numbers: V-JEPA 2 baseline 72.8 → V-JEPA 2.1 best-case 62.5; even Deep Self-Supervision only recovers to 72.1). v14's Phase-4 readout is clip-level only (PMA → mean → linear) and v14's M2/M3/M4 live in different token spaces (per-electrode F-patch × T-patch / 320 parcel slots / 320 parcel slots), so v14 cannot replicate the uniform 4-level DSS rescue. M2 supervision is now V-JEPA-2-canonical masked-only (the `L_pre_frame_masked` term above). Sister `R-context-loss-vjepa21-recipe` (P1, single cell) retains the full V-JEPA 2.1 §2.3.1 recipe as a falsification cell: if it beats the V-JEPA-2-style default on Neuroprobe gates, adopt B25/B26's recipe in a future amendment. Retired sisters (subsumed/collapsed by B27): `R-no-context-loss`, `R-no-warmup`, `R-ctx-lambda-{0.0, 0.25, 0.5, 1.0}`.
- **`L_mid_slot`** (**B31 ✅, 2026-05-28 PM — DROPPED FROM DEFAULT**; preserved verbatim below as `R-add-m3-loss` P0 falsifier sister spec. Rationale: no precedent in the Perceiver-family for loss at post-cross-attn / pre-process state, V-JEPA 2.1 DSS is motivation-only per B28, iBOT mechanism-mismatched. Geeling minimalism. B22 ✅, 2026-05-25; B03f amendment ✅ 2026-05-25; B26 L1 ✅ 2026-05-27 PM; **B28 citation reframe ✅, 2026-05-27 PM-late**) — gradient on `LN_mid(M3)` (post the single cross-attn ❺ @ layer 0 / pre self-attn-0, the parcel-routing checkpoint; B22). Per-(parcel, t) **L1** summed over `(p ∈ parcels_supervised[subject], t ∈ T, d)` and divided by `len(parcels_supervised[subject]) × T × d` (same per-clip normalization as L_post_frame). Predictor: 1-layer Linear(d, d) `loss_mid_head` — cross-attn ❺ alone IS the predictor for the deeper signal. Target: EMA teacher's `LN_mid_T(M3_T)` from the full-input pass (B26 contract), layer-averaged with per-layer instance-norm (the M3 representation comes from a single point in the teacher graph, so "layer averaging" here is the identity — the per-layer instance-norm still applies to keep teacher / student normalization symmetric). Weight 1.0. **B28 mechanism citation = V-JEPA-2 predictor (Assran et al. 2025, arXiv:2506.09985, §2.1)** — visible-position → masked-position predictor, L1 regression on continuous EMA-teacher features at the supervised position. Same mechanism class as `L_mid_slot`. **iBOT (Zhou et al. 2022, arXiv:2111.07832) is a conceptual ancestor only, NOT the mechanism citation** — iBOT's MIM head is cross-entropy over a shared 3-layer-MLP softmax tokenizer (`L_MIM = −Σ m_i · P_θ'_patch(u_i)^T log P_θ_patch(û_i)`, Eq. 3), which is categorically different from v14's L1-on-continuous-features (B27 lock). iBOT supplies the *concept* of auxiliary patch-level masked-prediction at intermediate semantic depth alongside a [CLS]/utterance objective; the mechanism is V-JEPA-2-predictor (or data2vec 2.0, Baevski 2022 §3.1). Pre-empt reviewer confusion: v14 does NOT use iBOT's online tokenizer, multi-crop (2 global + 10 local), or shared MIM-head — see [[project_v14_loss_design_amendment_b28_2026_05_27]] Item 4 ⚠️ box for the full mismatch breakdown. **Motivation** cites DINOv3 §4 / DINOv2 KoLeo / B21 collapse-prevention lineage (anti-collapse via dense supervision at intermediate depth); V-JEPA 2.1 §2.3.2 Deep Self-Supervision is no longer cited as the mechanism precedent (their Table 12 win 42.0 → 43.9 ADE20k mIoU was contingent on the §2.3.1 context loss that B27 reverted; their §2.3.2 applies L_predict + L_context at each of 4 encoder depths in uniform fashion, whereas v14's M3 lives in a different token space (320 parcel slots) than M2's per-electrode F-patch × T-patch grid, so the §2.3.2 implementation doesn't port — cite §2.3.2 *philosophy* only). **No context-loss companion at M3** (V-JEPA 2.1 Eq 2 requires a mask partition at the same depth as the loss; v14's mask lives only at the input depth in P1).
- **`L_post_frame`** (B03f amendment ✅ 2026-05-25; B26 L1 ✅ 2026-05-27 PM) — gradient on `LN_frame(M4)` (latent-stack output passed through Loss-3 dedicated LayerNorm; B21). Per-(parcel, t) **L1** summed over `(p ∈ parcels_supervised[subject], t ∈ T, d ∈ D_model)` and divided by `len(parcels_supervised[subject]) × T × d` (NOT static 320 × T × d — otherwise per-clip gradient scales with anatomy density; NOT `valid_parcels[clip]` post-mask — JEPA-inverted, see B03f). No predictor (cross-attn + latent stack ARE the predictor). Target: EMA teacher's `LN_frame_T(M4_T)` from the full-input pass (B26 contract), all-L=6 layer-averaged with per-layer instance-norm. In P1, teacher receives the same valid-bin mask as student (EX09), and the same per-electrode patch mask is NOT applied to teacher (B26 contract — teacher sees full patches). No context-loss companion at M4 in any phase (B27 PM-late revert — see `L_pre_frame_context` entry above; the prior B25 `R-p2-m4-context-loss` reactive sister is retired).
- **`L_post_utterance`** (**B31 ✅, 2026-05-28 PM — DROPPED FROM DEFAULT**; preserved verbatim below as `R-add-utterance-loss` P0 falsifier sister spec. Rationale: EAT's +1.3% mAP on AS-2M (λ=1.0 optimal) is the only precedent and is on a different downstream profile (audio classification, ~10s clips, weak labels) than v14's Neuroprobe gates (~2s clips, single label). Geeling minimalism. PMA query is moved off the SSL graph entirely under B31. B03f amendment ✅ 2026-05-25; B26 L1 ✅ 2026-05-27 PM) — gradient on a single (d,) vector per clip: `PMA_k=1(LN_utt(M4), query=q, axis=parcel) → (T, d)` then `mean(., dim=T) → (d,)`. PMA softmax over parcels is masked to `parcels_supervised[subject]` (additive `-1e4` at invalid slots per B12 sentinel). **L1** vs teacher's matching (d,) vector from the full-input pass (B26 contract). **No predictor** (EAT §3.1 utterance head: direct regression). **PMA query is the same shared seed reused at P2/P3/P4** — trained here, frozen at P4. No context-loss companion (single clip-level scalar, no patch domain). **NOTE**: V-JEPA 2/2.1 have no clip-level pooled loss; L_post_utterance is v14-original, cite EAT (Chen 2024) precedent, NOT V-JEPA.
- **`L_DKoleo`** (B21 ✅, 2026-05-25; B03 scope correction REVERTED PM 2026-05-25; **B28 DEMOTED to sister-only ✅, 2026-05-27 PM-late**) — REMOVED from the default 4-term objective. Mechanism preserved as one of three B28 sister cells. B21's prior body (full-bank scope over 320 latent slots, coefficient 0.1, slot-diversity regularizer on student M4) IS the `R-dkoleo-intra-clip-slots` sister; the DINOv2/v3-faithful per-batch CLS-analog version is `R-dkoleo-batch-cls-unit`; VICReg variance-hinge is `R-vicreg-slot-variance`. B28 demoted the default after recognizing the per-clip 320-slot unit diverges from DINOv2's per-batch CLS-bank unit (DINOv2 Algorithm 1 applies KoLeo on the 16 CLS tokens of the first global crop per batch; v14's default applied to all 320 slots per clip), making B21's appeal to "DINOv2 +8.3pp Oxford-M" a precedent for the operator (KoLeo) but NOT for the unit (per-clip slots). B21's identity-anchored init + B22 M3 supervision + dedicated LN per head + reactive Gram already carry the collapse-prevention load. MON-SLOT-REDUNDANCY now makes the no-default decision observable (every 10k steps, per-clip 320-slot AND per-batch CLS-analog cosine probes) and gates sister escalation by pre-registered thresholds. The cross-subject `LearnableParcelEmbed[p]` reachability concern that motivated the PM revert is now handled by identity-anchored init alone (the parcel embeddings receive gradient even when their slot is unsupervised at a given clip, via the cross-attn forward path).
- **`L_DKoleo@M3` (B22, REACTIVE; B03 scope correction REVERTED PM 2026-05-25)** — same DKoleo operator on M3 slot means over all 320 latent slots (`M3.mean(dim=t_p) → L2_normalize → DKoleo`), weight 0.05 (half of M4's 0.1, since M4 carries the load-bearing downstream PMA target). OFF by default; armed at M3-side diagnostic trigger (see Diagnostic monitor extension below). Discarded at P1→P2 boundary (re-armed from P2 step 0).

**~~Why all four mask-prediction losses~~ — SUPERSEDED BY B31 (2026-05-28 PM, see [[project_v14_b31_vjepa2_canonical_loss_2026_05_28]])**. B31 collapses the default to 2 mask-prediction terms (V-JEPA-2-canonical: `L_pre_frame @ M2 + L_post_frame @ M4`) on Geeling-minimalism grounds. The rationale below is retained for the `R-add-utterance-loss` / `R-add-m3-loss` / `R-add-both` P0 falsifier sisters that put the dropped terms back on the default and quantify the cost of B31 simplification.

**~~Why all four mask-prediction losses (provenance)~~.** EAT (IJCAI 2024, arXiv:2401.03497) Fig. 4 λ_UFO sweep over {0, 0.01, 0.1, 1, 5, 10} on AS-2M: λ=1 optimal at 40.2% mAP, +1.3% over frame-only — at the scale of v14's submission-gate cushion. **B22's `L_mid_slot` motivation is collapse-prevention via parcel-routing checkpoint supervision** (DINOv3 §4 motivation / DINOv2 KoLeo / B21 lineage); B22's **mechanism precedent is V-JEPA-2 predictor (Assran 2025 §2.1)** — L1 regression on EMA-teacher continuous features at supervised positions (B28 citation cleanup; was V-JEPA 2.1 §2.3.2 in pre-B28 docs). **iBOT (Zhou 2022) is a conceptual ancestor only, NOT the mechanism** — iBOT's MIM head is cross-entropy over a shared softmax tokenizer, mechanically different from v14's L1-on-continuous-features (corrected 2026-05-27 evening after iBOT paper re-read; see [[project_v14_loss_design_amendment_b28_2026_05_27]] Item 4 ⚠️). V-JEPA 2.1 §2.3.2's DSS specifically rescues the clip-level loss that V-JEPA 2.1's context loss imposes (their Table 1: context loss alone drops SSv2 72.8 → 62.5; DSS recovers to 72.1); v14 without the context loss (B27 revert) has nothing to rescue at depth, but B22's checkpoint supervision still serves the orthogonal purpose of pushing meaningful structure into the parcel-routing transition (M3 = post-cross-attn ❺ @ layer 0, the moment per-electrode patches collapse into the 320 parcel slot bank). All four losses sit in the same bootstrap latent-prediction family — three depths (M2 / M3 / M4) + one clip pool — all masked-only supervision (V-JEPA-2-canonical), all pure L1 (V-JEPA 2 §2.1 Eq 1).

**SWEC fallback for parcels_supervised.** SWEC has no electrode anatomy → `parcels_supervised[swec_subj]` falls back to the union of all 320 latent slots. L_mid_slot and L_post_frame then supervise all 320 × T positions, L_post_utterance's PMA softmax is uniform over 320 parcels. Reactive Gram + MON-SLOT-REDUNDANCY + any armed B22 reactive `L_DKoleo@M3` operate on all 320 slots (B21 default scope; B28 DKoleo sisters when armed inherit the same all-320 scope per B21 PM revert reasoning). L_pre_frame is unchanged (no parcel dependence). See §2 SWEC anatomy assert + B03f canonical memo §"SWEC special case".

**Framework note.** v14 IS JEPA: predictor on `L_pre_frame` here in Phase 1; "cross-attn + latent stack IS the predictor" for `L_post_frame`; no predictor on `L_post_utterance` (direct regression, EAT-precedent). Layer-averaging is target-objective enrichment ON TOP OF JEPA architecture, not a JEPA alternative. EAT (IJCAI 2024) and SSLAM (ICLR 2025 SOTA) precedent the hybrid; pure-K=1 precedents (I-JEPA / Brain-JEPA / V-JEPA-original / Laya / DINOv3) sit in regimes that don't transfer (image-patch redundancy / fMRI BOLD smoothness / no predictor / Hubel-Wiesel alignment).

**~~UFO scale~~ — SUPERSEDED BY B31 (2026-05-28 PM)**. The B19 frame+utterance "UFO" framing is dropped. Joint SSL = frame-only at two depths (M2, M4) per V-JEPA 2 §2.1 Eq 1; utterance signal arrives via the P3 Whisper-L8 distillation target on `PMA(M4)`. EAT precedent retained only on `R-add-utterance-loss` sister.

**Masking (B08 ✅, 2026-05-23 — 4-agent SOTA audit; tightened 2026-05-24 by B19; mask grid revised 2026-05-24 by B20 v4; substitution paradigm locked 2026-05-25 by B03c).** **Option-B negotiated median**: 2D inverse-block masking on the **F-patch × T-patch grid (10 × T_p)** **per electrode, independently** (B20 v4 — was bin × frame grid in v3). **Keep-block ≈ 3 F-patches × 3 T-patches** (~3 octaves × ~375 ms), **mask rate ≤ 50% per electrode** (HARD cap, unchanged from B19), **M sized to hit ≤ 50% keep** (scaled down from v3's M ≈ 8–12 due to fewer total patches: 200 cells in P1 vs 2190 in v3). **Substitution paradigm = B03c paradigm B = DROP + predictor**: student encoder operates only on the visible (1−r) fraction of patches per electrode; the 2-block predictor (hidden=128, heads=4, ~0.2M, discarded after P1) fills in M2 at masked positions before L_pre_frame. NOT a learnable `[MASK]` embed in the encoder (paradigm A, sister `BiL-B03-PatchMask-Paradigm-A`). ~25% reduction on per-electrode-SA compute × 50% mask rate ≈ ~12% P1 step-time saving net of predictor cost. **No whole-shaft masking. No whole-electrode masking.** Both reserved for Phase 2's separate electrode-mask layer. Rationale (unchanged): prior 80% / frame-1D / keep-3 ≈200ms commit failed audit on 3 axes — frame-1D weakly supported (V-JEPA tubes redundancy axis, VideoMAE worst), 80% is biosignal-FM outlier (Brant 50%, Brant-2 40%, LaBraM 50%, DIVER-1 50%, REVE 55%, Laya 60% — conscious SNR downward divergence), keep-3 below 800ms EAT optimal and Laya 500ms–1s. Sister `R-p1-frame-1d-keep-3-mask-80` retained for empirical attribution; mask-rate bracket `R-p1-mask-rate-{30,40,50}` (added 2026-05-24, narrowed to ≤ 50% range).

**Valid-bin symmetric teacher mask (EX09 ✅, 2026-05-23).** EMA teacher receives the SAME per-corpus valid-bin mask as the student (SWEC k0–k21, AJILE12 k0–k20, D-cohort/BT k0–k29): byte-identical input fill (invalid bins → 0) and key-padding mask (invalid bins → −∞). Defensive zero-out post-teacher-forward: `teacher_features[..., invalid_bins_for_this_sample] = 0.0`. Loss supervision mask is intersection: `supervised_positions = ssl_mask & valid_bin_mask`. Runtime assert in debug builds: `assert (teacher_features[..., ~valid_bin_mask] == 0).all()`.

**Data.** SWEC 6,672 h + AJILE12 1,280 h + D-cohort 180 h + BT 43.5 h ≈ **8,176 h** total (PS+lex 7h dropped). AJILE12 is 500 Hz / 0.5–200 Hz bandpass (Peterson 2022; ECoG-dominant 89.7%), so it trains valid-bin k0–k20 only — same per-corpus mask machinery as SWEC.

**Sampler (B02 ✅, 2026-05-23 — 3rd re-lock after 4×4-agent SOTA audit).** **α = 0.5 hierarchical (XLS-R + DINOv3 pattern) over valid-bin-electrode-hours.** Two-group split (SWEC vs broadband {AJILE12+D-cohort+BT}); macro shares hard-set 50/50 (SWEC corpus-cap); within-broadband α=0.5 over vb-eh. Per-corpus gradient shares: **SWEC 50.0% / AJILE12 ~27.7% / D-cohort ~15.0% / BT ~7.3%** (recomputed at sampler-build from exact per-session totals; the listed broadband shares are an audit estimate that the build-time computation will refine — see "Exact totals" below). Per-row weight = `(group_macro_share / sum_within_group(vb_eh^0.5)) × vb_eh[source]^0.5 / count_rows[source]`, fed to a `WeightedRandomSampler(replacement=True)` wrapped by `torchdata.stateful_dataloader.StatefulDataLoader`. **Unit-of-temperature self-consistency**: sampler base unit (vb-eh) matches loss unit (row-mean-then-batch-mean over per-electrode-token rows) — the stated marginals equal effective gradient shares.

**Exact totals, NOT corpus-medians.** At sampler-build, compute `vb_eh[corpus] = Σ_session (session_hours × session_n_electrodes × |valid_bins[corpus]|)` directly from the events manifest. Do **not** use a corpus-median × n_sessions approximation. LLM-pretrain SOTA (Llama 3 §3.1.2 p.7, OLMo 2 Table 4, Megatron-LM `BlendedDataset`) uses exact token totals; corpus-medians introduce bias proportional to within-corpus electrode-count variance, which is high for BT (64–250 electrodes/subject).

**Why hierarchical at α=0.5, not flat α=0.3.** Two-round audit:
1. 1st re-lock had been α=0.3 over hours (mT5-ship default). 2nd audit caught that no iEEG-FM uses α=0.3; the right speech analogue is XLS-R hierarchical (Babu 2022) — 2-tier sampling within language families.
2. v14's structure mirrors XLS-R: anatomy-blind (SWEC, AJILE12) vs anatomy-bearing (D-cohort, BT). Hierarchical 50/50 hard-caps SWEC's pure-data-scale dominance without collapsing macro budget into 6 672 h, and lets broadband corpora share α=0.5 over the unit that actually counts (vb-eh).
3. AJILE12 bandwidth bug (500 Hz / 0.5–200 Hz) forces vb-eh as base unit anyway — flat-hours over-credits AJILE12 by k29/k20 ≈ 1.45×.

**Loss reduction (locked with sampler).** **Row-mean-then-batch-mean** (MAE / V-JEPA standard) with fp32 accumulator. `reduction='none'` per (electrode, time, bin) token → mean per row → mean over rows in batch. Required for stated per-corpus marginals to equal effective gradient shares; under token-mean reduction, large-electrode-count corpora silently get ~1.8× their stated weight. fp32 accumulator avoids bf16 row-sum loss-of-precision at C×T×F ≈ 130×73×30 ≈ 285k tokens per row.

**DDP + mid-epoch resume via TorchData `StatefulDataLoader`.** Wrap NeuralSet's `SegmentDataset` in a `WeightedRandomSampler(replacement=True)` with per-rank `num_samples = N_epoch / world_size`, then hand the (dataset, sampler) pair to `torchdata.stateful_dataloader.StatefulDataLoader` (TorchTitan arXiv:2410.06511 §3.3; TorchData `meta-pytorch/data` `stateful_dataloader/README.md`). StatefulDataLoader provides native `state_dict / load_state_dict` for mid-epoch resume, per-rank deterministic worker RNG, and `persistent_workers=True` reseeding. Per-rank seeding still required for the WRS itself: `generator = torch.Generator().manual_seed(base_seed + rank + epoch)`. **This replaces the prior custom `StatefulSampler` wrapper + `persistent_workers=False` + manual generator pass-through** — chosen because (a) NeuralSet/NeuralTrain expose a map-style `Dataset` and leave the sampler layer to the user (so this is gap-filling, not substrate-fighting), (b) 2025–2026 SOTA stack (TorchTitan, NeMo Lhotse Shar, fairseq2 streaming, MosaicML Streaming) has moved off WRS-only solutions, (c) removes ~150 lines of custom wrapper code we'd otherwise own.

**Multi-STFT cache storage location (HB01 closed 2026-05-23 PM).** Phase-1 Multi-STFT precompute (~5–8 TB fp16 across SWEC + AJILE12 + D + BT, dominated 91% by SWEC) lives at **`/work/ht203/cache/multi_stft/`**, NOT `/hpc/group/coganlab/` (the lab persistent share is 1 TB total with ~400 GB free — placing 5–8 TB there would block other lab members). DCC `/work/` is 950 TB cluster-wide with 233 TB free; 5–8 TB is structurally invisible. The 75-day `/work/` purge is acceptable because (a) regenerate is embarrassingly parallel CPU (`torch.stft` ~10× realtime/core; full corpus regenerate = ~25 CPU-hours on 32 workers via `MapInfra`), (b) one regenerate per training campaign is dwarfed by Phase-1 GPU-hours, (c) we add a `cache_regenerate.py` MapInfra job spec'd to rerun against any subject manifest. **Scope-amended CLAUDE.md rule**: persistent caches (Exca metadata, checkpoints, Whisper-L8 teacher cache, eval artifacts) stay in `/hpc/`; bulk feature precomputes that are cheap to regenerate go to `/work/` with documented regenerate procedure. Memo: `[[project_v14_hb01_multistft_cache_location_2026_05_23]]`. Closes HB01.

**v14 extractor cache storage location (task #120 closed 2026-05-29).** Per-extractor outputs (notch-filtered Log-STFT view, DK support lookup, valid mask, per-clip ref-idx, subject-subtype embed lookup, λ-anatomy lookup, optional shaft-mask) live at **`/work/ht203/cache/v14_extractors/<extractor_name>/`** — one subdirectory per extractor class so cache keys are independent and any extractor can be evicted / regenerated in isolation. The wiring is `_apply_extractor_cache(extractor, name, root)` in `src/speech_decoding/experiments/dispatch_v14.py`, which sets `extractor.infra.folder = root / name` when the env var `EXCA_EXTRACTOR_CACHE_FOLDER` is set (`scripts/dcc/dispatch` injects it). The two-tier split exists because exca's `TaskInfra.folder` (set via `EXCA_CACHE_FOLDER` → `/hpc/group/coganlab/ht203/cache_neuroai/`) keys the Experiment slot by full pydantic config, so any model-side knob (precision, batch_size, depth, d_model, …) invalidates the whole slot — including the multi-minute extractor prep that has nothing to do with that knob. Pinning each `MapInfra.folder` onto a path keyed only by the extractor's own config decouples lifetimes: the Experiment cache rotates per sister cell, while the extractor cache amortizes across cells. **Volume**: ~5–10 GB per subject for the Log-STFT view (notch + STFT @ 8 Hz hop, dense fp16), <100 MB combined for the five lookup-only extractors, ~1 GB per subject for the joint-phase shaft-mask precompute when sister gates it on. Full BT-Lite + AJILE12-Lite + D-Lite Stage-0 footprint ≈ 50–100 GB; full Phase-1 SWEC corpus could push toward 500 GB–1 TB if Log-STFT is run there instead of in `multi_stft/` (it isn't today). All structurally invisible against 233 TB free. **Regenerate**: each cache key is `(extractor pydantic config × per-event uid)`; clearing the directory and rerunning any dispatch repopulates it via the standard `prepare_extractors` → `_get_data` decorated by `@infra.apply` path in `neuralset/extractors/`, embarrassingly parallel on `n_jobs=-1` LokyBackend (~13–16 min cold per subject pair on a single dispatch's CPU prep step — verified on job 47295714). 75-day purge is fine: any single training campaign rebuilds the cache once on first dispatch, then reads it across the dispatch loop's sister sweep until the campaign closes. Wiring tests at `src/speech_decoding/experiments/test_v14_dispatch_wired.py::test_b15_extractor_cache_*`. CLAUDE.md storage-tiering rule compliance: `/work/` for bulk + cheap regenerate ✅; per-tier `MapInfra` regenerate documented ✅.

**Warm-read per-sample profile + anatomy memoization (task #126 closed 2026-05-29).** With the extractor cache warm, one `Dataset.__getitem__` is **~4.3 ms/sample** (Lite, subject 2 trial 4; profiled on job 47309998) ≈ 232 samples/s single-threaded — in the expected range for the ~14 M-param encoder. The split: `LogStftView` 3.43 ms (the per-sample `CAR + torch.stft` recompute — the `MapInfra` cache stores only the raw waveform, a 0.011 ms memmap read, NOT the STFT view, so the transform reruns each `__getitem__`) and ≤ 0.06 ms each for the six lookup extractors. The DK-support + valid-mask `BaseStatic` extractors carry no `MapInfra` field, so `_apply_extractor_cache` skips them; before #126 they re-read + re-parsed `depth-wm.csv` via `load_public_bt_anatomy` on *every* sample (support 5.0 ms + valid-mask 4.0 ms = 9 ms = ~70 % of a warm `__getitem__`). #126 wraps those per-subject lookups in `@functools.lru_cache(maxsize=64)` (`_cached_hard_support` in `extractors/dk_support.py`, `_cached_n_real` in `extractors/valid_mask.py`, both keyed on `bt_root, subject_id, unknown_label_policy, parcel_labels`), dropping them to 0.04–0.06 ms (≈ 90×) with all FileNotFoundError / KeyError / ValueError semantics preserved (`lru_cache` does not cache exceptions). The original 10h baseline ran at ~12 samples/s because it predated this cache AND ran cold (empty `/work/ht203/cache/v14_extractors`); warm + memoized + `num_workers=4` (dispatch default, task #124) is ~19× faster. Remaining lever if ever GPU-starved: `MapInfra`-cache the `LogStftView` *output* (not just the raw waveform) to retire the 3.4 ms STFT — unnecessary today since 3.4 ms across 4 workers ≈ 0.85 ms effective ≫ model step time.

**Cold-warm memory requirement.** The first dispatch that warms the cache runs `MneRaw`'s `notch_filter` with `LokyBackend n_jobs=-1`, which loads concurrent full-session copies of each BT movie recording (e.g. `(135, 27_476_024)` float32 ≈ **14.8 GB** per session). A 64 GB allocation OOM-kills the warm; budget **≥ 128 GB** (240 GB comfortable) for the cold cache-warming dispatch. Steady-state reads after warm need only normal RAM. exca reclaims OOM-killed in-flight items and resumes cleanly on rerun.

**Page-cache thrash + I/O bandwidth mitigation (canonical fixed sharding + node-local NVMe stage-in).** SWEC's ~5 TB Multi-STFT cache (was 12 TB cite, refined to ~5 TB fp16 post-audit) exceeds DCC node page cache (~256 GB). Four-layer mitigation:
1. **Numpy memmap, no compression.** Multi-STFT cache is dense fp16 in `(n_clips, C, T, F, W)` layout per subject, one `.npy` per subject + JSON sidecar. Zero-copy mmap, no decode overhead. Compression (LZ4/Zstd) tested and rejected — STFT entropy ~80% of raw bits, ~1.3× ratio doesn't justify decode cost on local NVMe.
2. **Canonical fixed locality-aware sharding** — at job-start, partition SWEC subjects across ranks deterministically (rank `r` of world size `W` owns subjects `{s : hash(s) % W == r}`); rank `r`'s WRS draws only from its local SWEC subset, while broadband corpora (AJILE12/D-cohort/BT) are small enough to mmap globally. Within-shard shuffle happens via the WRS draw per epoch; **the subject→rank assignment stays static across epochs** so the kernel page cache amortizes across epochs. (Pattern: MosaicML StreamingDataset canonical-nodes; Megatron-LM `BlendedDataset` deterministic shard sharding.) **This replaces an earlier "redraw 10 SWEC subjects per rank each epoch" plan** — flagged by 3rd-audit LLM lens as inverting the locality argument (redrawing defeats cache amortization). Tracked under DP02 + HB01.
3. **Node-local NVMe stage-in** (new 2026-05-23 PM — the biggest I/O lever). At Slurm job start, sbatch prologue copies rank `r`'s SWEC shard from `/work/` to node-local `/scratch/$SLURM_JOB_ID/` (Ada-5000 nodes ~1 TB NVMe; A100 ~2 TB). Per-rank shard for W=8 ranks: 50 SWEC pts / 8 ≈ 6 subjects/rank × ~100 GB/subject ≈ ~600 GB — fits 1 TB NVMe with headroom for D/BT/AJILE12 broadcasts. Local NVMe read = ~5 GB/s vs `/work/` NFS ≈ ~500 MB/s. **10× I/O speedup, gates whether file-read is 80% of step time or 10%.** Stage-in cost ~20 min one-time at job start, amortized over the full training run. Sister `R-no-nvme-stagein` (run from `/work/` directly) quantifies the speedup empirically.
4. **DataLoader tuning**: `num_workers=8`, `prefetch_factor=4`, `persistent_workers=True`, `pin_memory=True` per TorchTitan §3.3 defaults. (This `num_workers=8` is the full-Phase-1 multi-node SWEC target. The single-GPU **Lite Stage-0 dispatch default is `num_workers=4`** — `DEFAULT_NUM_WORKERS` in `dispatch_v14.py`, CLI-overridable via `--num-workers`; `prefetch_factor`/`persistent_workers`/`pin_memory` defaults match this list, set on `experiments/data.py::Data`. 4 saturates the single Ada-5000 at the ~4.3 ms warm `__getitem__` profiled above; bump to 8 when scaling to the SWEC corpus.)

**B01-coupling status: ✅ satisfied.** XLS-R validated regime requires effective batch ≥ 256; B01 v3 (post-v3 amendment 2026-05-23) locks P1 effective batch = 1024.

**Sister roster** (see §8): `R-sampler-alpha03` (1st-audit prior default, demoted to load-bearing sister), `R-sampler-sqrth`, `R-sampler-pure-h`, `R-sampler-uniform`, `R-sampler-broadband-uniform`, `R-sampler-seeg-only` (drop AJILE12 entirely, modality-invariance probe).

**EMA teacher (B26 ✅, 2026-05-27 PM — drop ramp, fixed τ=0.999).** Single teacher network used for all four frame/utt/mid losses (L_pre_frame reads teacher's M2, **L_mid_slot reads teacher's `LN_mid_T(M3)` — B22**, L_post_frame reads teacher's `LN_frame_T(M4)`, L_post_utterance reads teacher's `LN_utt_T(M4)`-via-PMA). **Fixed τ=0.999 throughout P1 — NO ramp** (B26 ✅; V-JEPA 2 §2.4 explicit: "we simplified the recipe from Bardes et al. (2024) by maintaining fixed teacher EMA and weight decay coefficients instead of using ramp-up schedule"). The prior `linear 0.99 → 0.9999 over 400k steps` ramp was a V-JEPA 1 (Bardes 2024) pattern; V-JEPA 2 dropped it. τ=0.999 chosen as the modal value across V-JEPA / DINO / DINOv2 / DINOv3 / data2vec at sub-100M params + sub-1M-step budgets (V-JEPA 2 doesn't disclose its exact τ value in main text). **Teacher always sees FULL unmasked input** (B26 contract — no shaft mask, no patch mask on teacher; only per-corpus valid-bin mask). Stop-grad on the teacher path; teacher PMA reads student's PMA query with `query.detach()`. Teacher has its own EMA-tracked LN_mid_T (B22), LN_frame_T, and LN_utt_T (B21) mirroring the student's three dedicated LNs. Sister `R-ema-tau-{0.99, 0.9995, 0.9999}` (P1) brackets the fixed-τ default; sister `R-ema-ramp-v-jepa1` (P1) restores the V-JEPA 1 linear ramp `0.99 → 0.9999 over 400k` as a falsifier for the "fixed EMA is enough" simplification.

**Diagnostic monitor (B21 ✅, 2026-05-25 + B22 M3 extension ✅, 2026-05-25 — always-on).** Cheap collapse instrumentation every 10k steps on a held-out probe batch (~256 clips). Logged to `wandb` / Exca metadata, not gradient-bearing. **Mirrored at M3 and M4** (B22): each slot-side metric runs on both checkpoints:
1. **Pairwise slot cosine similarity** over `(320, d)` slot means — separately at **M3** (B22 extension) and **M4** (B21 default). Log `mean(off-diag)`, `max(off-diag)`, distribution percentiles. Healthy: `mean(off-diag) ≲ 0.3`; trigger (per checkpoint): `max(off-diag) > 0.7` sustained 50k steps.
2. **Cross-attn attention entropy per slot** averaged across batch — single cross-attn ❺ @ layer 0 (post-B28; was cross-attn-1 + cross-attn-2 in v3, the second @ position 3 was dropped by B28 Item 2). Healthy: per-slot entropy < log(C × F_p) − 1 (slots discriminate).
3. **PMA-k=1 attention entropy across parcels** per clip over 320 parcels (masked to `parcels_supervised[subject]` per B03f). Healthy: clip-level entropy < log(|parcels_supervised[subject]|) − 0.5; collapse signal: entropy ≈ log(|parcels_supervised[subject]|) (uniform attention).
4. **Linear-probe parcel-ID F1** (B03 scope correction REVERTED PM 2026-05-25): sklearn LogisticRegression on the FULL slot bank `(320, d)` → `parcel_of_latent[i] = i // M` labels (80 classes), 5-fold CV. Operates on all 320 slots — probing only the supervised subset cannot diagnose whether the unsupervised `LearnableParcelEmbed` entries are diverse vs collapsed at the cohort level (the question the monitor exists to answer). Run separately at **M3** (B22 extension) and **M4** (B21 default). Healthy: F1 ≥ 0.7 over the supervised subset (restricted on the report side, NOT the fit side) and F1 ≥ 0.5 over the full 320 (probes cohort-level embedding diversity for parcels rarely seen in any subject's supervised set); trigger (per checkpoint): F1 < 0.4 over supervised subset sustained 50k steps.

**B22-specific reactive arm (DKoleo@M3).** Triggered by either of the M3-side metrics above (`cos_sim_M3 > 0.7 sustained 50k` OR `parcel-ID F1 at M3 < 0.4 sustained 50k`). When armed: augmented loss term `+ 0.05 · L_DKoleo@M3` (weight 0.05 = half of M4's 0.1 since M4 carries the load-bearing PMA target). Discarded at P1→P2 boundary; re-armed from P2 step 0 (mirror of B21's reactive-Gram re-arm rule).

**Reactive Gram anchoring (B21 ✅, 2026-05-25 — OFF by default; B03 scope correction).** Augmented loss term activated at trigger condition:

```
trigger ⇔ (slot cos sim max > 0.7 sustained 50k over full 320 slots)
       OR (L_post_frame plateau Δ < 0.5% over 50k AND L_post_utterance decreasing)
       OR (parcel-ID F1 over supervised slots < 0.4 sustained 50k)

at trigger time T_trigger:
  M4_gram_teacher ← snapshot of student backbone at T_trigger
                    (separate from main EMA teacher — DINOv3 § 4.2)
  refresh every 10k steps

X_S, X_G ← L2_normalize(M4 reshaped to (P=320·T_p, d))   (B03 scope correction REVERTED PM 2026-05-25 — Gram operates on full 320 slots, mirroring DKoleo revert)
L_Gram   = ||X_S @ X_S.T − X_G @ X_G.T||_F²
λ_gram   = 0.1

L_total_post_trigger = L_pre_frame + L_mid_slot + L_post_frame + 1.0·L_post_utterance + 0.1·L_DKoleo + 0.1·L_Gram
```

Gram teacher is a snapshot of the STUDENT backbone at trigger time (not the EMA teacher) — DINOv3 § 4.2 anchors to a frozen student snapshot, not the EMA path. This avoids reinforcing the collapse the EMA was already participating in. Discarded at P1→P2 boundary; re-evaluate trigger conditions at P2 step 0. Compute cost when active: ~5–10% extra per step at full 320 scope (O((P·T_p)² · d)). The earlier AM "supervised-slot-restricted" Gram scope was reverted PM same-day with the same logic as DKoleo: the DINOv3 § 4.2 Gram precedent is over the full latent grid, and restricting to ~60-100 supervised slots would reintroduce the cross-subject `LearnableParcelEmbed` reachability problem.

**Fallback (B21 last-resort, NOT default).** If Gram anchoring + diagnostic still shows collapse after 100k additional steps: drop SWEC, run P1 on D-cohort 180h + BT 43.5h = 224h with anatomy bias ON. Loses 36× pretrain scale but guarantees non-collapsed cross-attn. Sister `BiL-CollapsePrev-DropSWEC-Fallback` runs this as a default-from-start cell to bound the worst-case collapse cost.

**Optimizer.** See §7 (cross-phase). Peak LR 5e-4 @ effective batch 1024, 20k linear warmup, cosine → 0, 400k steps (sisters 300k).

**Output.** Full-stack checkpoint (❶❷❺❻ + LN_mid + LN_frame + LN_utt + PMA + LearnableParcelEmbed + LearnableSubSlotEmbed + loss_mid_head). Discard teacher, teacher LNs (LN_mid_T / LN_frame_T / LN_utt_T), P1 predictor, and Gram teacher (if active). The cross-attn bias buffer is saved with `bias_enabled=False` value — P2 toggles it to `True` after load.

---

## 4. Phase 2 — SSL pretrain (full stack, anatomy bias ON, + electrode-mask asymmetry)

**Goal.** Cross-electrode organization under anatomy-informed routing + electrode dropout. **Layered on top of Phase-1's already-warmed-up full stack** — Phase-1 trained ❶❷❺❻ + LN_mid + LN_frame + LN_utt + PMA + LearnableParcelEmbed + LearnableSubSlotEmbed + loss_mid_head jointly with anatomy bias λ_anat = 0 for its first 75% and λ_anat linearly ramping 0 → ~0.5 over its last 25% (B28 Item 3 warmup); Phase 2 continues the warmup λ_anat ~0.5 → 1 over its first 25%, then runs with full anatomy bias, narrows the corpus to sEEG-only (224 h, D 85 + BT 9), adds shaft-block electrode masking on top of the per-electrode patch mask, and continues with the same 4-term default objective (B19 3 losses + B22 L_mid_slot; B28 demotes L_DKoleo to sister-only).

**Bias warmup completes over first 25% of P2 (B28 Item 3 ✅, 2026-05-27 PM-late; replaces B19 discrete toggle).** All 94 P2 subjects (D 85 + BT 9) are anatomy-bearing sEEG → every ❺ forward pass uses the warmup-scheduled `λ_anat(step) · log(support[e,p] + ε)` bias. The scheduled scalar λ_anat hits 1.0 at step `0.25 · N_P2` and stays there for the remainder of P2/P3/P4. The prior "Bias is on throughout Phase 2" framing (B19 default) was a discrete toggle at the P1→P2 boundary that risked a ~5-nat QK shift overriding the P1-learned routing pattern; B28's linear warmup over the P1→P2 boundary integrates the anatomical scaffold gradually. The anatomy-blind sub-staging proposal (Phase 2a SWEC bias-off → Phase 2b bias-on) was rejected 2026-05-23 (M06-aux) over the same QK miscalibration risk + topology contamination from AJILE12 surface ECoG. **B19 (2026-05-24) made Phase 2a obsolete by a different mechanism**: Phase 1 already did the bias-off pretrain on the full 8.2k h corpus, so the anatomy-blind warmup is done before P2 even starts; B28 adds the missing 25%+25% linear-ramp piece across the P1→P2 boundary. Preserved as `R-p2a-bias-off-pretrain` P2 sister (B19); B28 sisters `R-anatomy-bias-step` (P0; B19 discrete-toggle baseline) and `R-anatomy-bias-on-from-p1` (P0; λ_anat = 1 from P1 step 0) bracket the warmup design.

**Trainable params** (~14.235M — full stack, same as Phase 1 post B19 + B21 + B22 + B28):
- A1 patch-embed (~0.01M)
- Freq categorical embedding (~0.01M)
- 6 token blocks ❷ at d=256 ≈ 6.3M (continues training from P1 checkpoint, NOT frozen)
- **1 ❺ cross-attn ≈ 0.79M** (B28 Item 2 ✅; was 2 @ {0, 3} in pre-B28 v3; bias scalar `λ_anat` continues warmup from P1 tail and reaches 1.0 by step 0.25·N_P2; continues from P1)
- LearnableParcelEmbed + LearnableSubSlotEmbed ≈ 0.023M (B21; continues from P1)
- 6 latent-stack blocks ❻ ≈ 6.3M (contiguous L=6 downstream of the single cross-attn @ layer 0 per §1 forward path; was split 3+3 around ❺b @ position 3 in v3)
- LN_mid at M3 + loss_mid_head ≈ 0.067M (B22; continues from P1)
- LN_frame + LN_utt at M4 divergence ≈ 0.001M (B21; continues from P1)
- ❼ PMA k=1 query ≈ 0.8M (continues from P1)
- ❺'s `log(support+ε)` bias is registered as buffer (not parameter); `λ_anat(step)` schedule is a registered tensor / scalar lookup, also not a parameter

**Frozen.** Nothing (full stack continues training). The B19 amendment removes the "Stage A frozen during P2" lock — Phase 2 is now full-stack fine-tuning of the P1 checkpoint with shaft-block masking added. The prior "Stage A frozen via `for p in stage_a.parameters(): p.requires_grad_(False)`" instruction is RETIRED. (See §7 for slow-LR option if Phase-2 fine-tuning shows Stage-A drift on sister cell `R-p2-stage-a-slow-lr` — not the default.)

**Loss.** Same 4-term default structure as Phase 1 (4 SSL prediction terms; B28 demoted L_DKoleo to sister-only; B22 L_mid_slot continuation), all using **pure L1** per B26 (kept through B27 + B28), all parcel-level terms scoped to `parcels_supervised[subject]` per B03f (per-subject DK coverage; invariant to within-clip shaft mask — shaft-orphaned parcels remain in the supervision set BY CONSTRUCTION, since they are the JEPA prediction targets). **No L_pre_frame_context term** (B27 PM-late revert; see §3 for the V-JEPA 2.1 Tables 1+2 evidence). **No λ_ctx schedule** (no context loss to weight). **No `R-p2-m4-context-loss` reactive sister** (retired by B27 — was gated on the P1 context loss). **No default `0.1 · L_DKoleo@M4` term** (B28 PM-late demotion; routes through MON-SLOT-REDUNDANCY to one of three B28 sister cells when escalation thresholds fire).

```
L_total = L_pre_frame_masked  +  L_mid_slot  +  L_post_frame
        +  1.0 · L_post_utterance

[reactive: + 0.1  · L_Gram                 if M4 trigger fires from P2 step 0]                  ← B21 (carryover)
[reactive: + 0.05 · L_DKoleo@M3            if M3 trigger fires from P2 step 0]                  ← B22 arm 3 (mechanism routes through chosen B28 DKoleo variant)
```

- **`L_pre_frame_masked`** (B26 L1 ✅) — same as P1 (paradigm B drop + predictor, pure L1), computed over the per-electrode patch mask. **Predictor is warm-started from the P1 checkpoint at P2 step 0 (PM revert 2026-05-25)**: re-init coincides with the bias-OFF→ON flip, and no JEPA-family precedent (V-JEPA, data2vec-2.0, I-JEPA) discards predictors between SSL phases. Re-init at the most fragile transition (cross-attn ❺ + latent ❻ bias flip) risks gradient noise propagation into the trunk; warm-start preserves the JEPA-family contract. Sister `R-p2-predictor-reinit` retains the re-init alternative as a falsification cell (P1 sister; see §8 B03 mask-discipline section). Teacher receives FULL patches AND FULL electrodes (B26 contract — no patch mask, no shaft mask on teacher; only per-corpus valid-bin mask).
- **`L_pre_frame_context` — DROPPED by B27 in P2 (mirrors §3 P1 revert).** See §3 entry for full rationale. P2 M2 supervision is V-JEPA-2-canonical masked-only (the `L_pre_frame_masked` term).
- **`L_post_frame_context` — DROPPED by B27 in P2.** Was B25's reactive sister `R-p2-m4-context-loss` extending V-JEPA 2.1 Eq 2 to M4 via parcel-adjacency `d_min`. Removed from the roster because (a) it was gated on the P1 context loss as a "tie the gates" extension, which no longer exists; (b) its independent value would require the same dense-vs-clip-level trade-off that B27 rejected at M2; (c) MON-MASK-002's bounded-ratio monitor at M4 retains its detector role for orphan-side collapse, but the mitigation, if it fires, would be `R-stratified-shaft-mask` or `R-shaft-K2` (mask-discipline changes), not a deeper context-loss extension.
- **`L_mid_slot`** (B22 continuation; B03f scope; B26 L1 ✅) — same **pure L1** on `LN_mid(M3)` over `(p ∈ parcels_supervised[subject], t ∈ T_p, d)`, divided by `len(parcels_supervised[subject]) × T_p × d`. In P2 the supervision is JEPA-correct on the asymmetric mask: student's cross-attn ❺a never sees shaft-blocked electrodes (B03 drop), so for shaft-orphaned parcels the student's M3 latent is pure prediction-from-context (`LearnableParcelEmbed[p] + LearnableSubSlotEmbed[s] + ε` seed + 1 cross-attn-from-visible-electrodes pass). Teacher (which sees full electrodes per B26 contract) has real signal at those shaft-orphaned parcels — that asymmetry IS the JEPA prediction signal. Same target machinery (LN_mid_T(M3_T) with per-layer instance-norm, EX09 symmetric valid-bin teacher mask). LN_mid + loss_mid_head continue from the P1 checkpoint.
- **`L_post_frame`** (B03f scope; B26 L1 ✅) — same **pure L1** on `LN_frame(M4)` over `(p ∈ parcels_supervised[subject], t ∈ T_p, d)`, divided by `len(parcels_supervised[subject]) × T_p × d`. Same JEPA-correct shaft-orphan handling as L_mid_slot — shaft-orphaned parcels are prediction targets, not exclusions. Same target machinery (all-L=6 layer-averaged with per-layer instance-norm, EX09 symmetric valid-bin teacher mask, B26 full-input teacher contract). LN_frame continues from the P1 checkpoint.
- **`L_post_utterance`** (B03f scope; B26 L1 ✅) — same PMA-k=1 over parcels (with `LN_utt(M4)` input) → mean-over-T → (d,). PMA softmax masked to `parcels_supervised[subject]`. **Pure L1** vs teacher's matching (d,) vector from the full-input pass (B26 contract). Same shared query that's been training since P1 step 1. LN_utt continues from the P1 checkpoint.
- **`L_DKoleo`** (B21 continuation; B03 scope correction REVERTED PM 2026-05-25) — same all-320-slot operator on student M4 (`M4.mean(dim=t_p) → L2_normalize → DKoleo`), weight 0.1, student-only, no teacher path. Reverted to all 320 slots after Tier-1 red-team flagged the cross-subject geometry contract break (see §3 L_DKoleo entry for full reasoning). Identity-anchored init + LearnableParcelEmbed continue training from P1.
- **λ_pre_masked : λ_mid : λ_post_frame : λ_post_utt : λ_DKoleo = 1.0 : 1.0 : 1.0 : 1.0 : 0.1** (B19 + B21 + B22 + B26 + B27 lock). No λ_pre_ctx after B27 revert. Sister `R-context-loss-vjepa21-recipe` (P1) reinstates the V-JEPA 2.1 §2.3.1 recipe (visible-patch supervision at M2 with `λ_i = 0.5 / √d_min` + linear warmup 0→0.5 over first 25% of P1) as a single falsification cell.
- **EMA teacher** (B26 ✅, 2026-05-27 PM — drop ramp, fixed τ=0.999): fresh teacher copy from the P1 checkpoint at P2 step 0, including LN_mid_T (B22), LN_frame_T, and LN_utt_T (B21) EMA copies. **Fixed τ=0.999 throughout P2 — NO ramp** (B26 ✅; V-JEPA 2 §2.4 explicit). The prior `linear 0.999 → 0.9999 over 40k steps` was retired by B26. Teacher always sees FULL unmasked input (B26 contract). Sisters `R-ema-tau-{0.99, 0.9995, 0.9999}` and `R-ema-ramp-v-jepa1` apply identically to P2.
- **Diagnostic monitor** (B21 + B22 M3 extension) continues every 10k steps with both M3- and M4-side slot/parcel-ID metrics — trigger conditions re-evaluated from P2 step 0 (P1 trigger state does NOT carry over; the bias-on change at P2 substantially shifts attention statistics).
- **B03 MON-MASK series (P2 specifics, PM 2026-05-25 — load-bearing for the B03 cascade):**
  - **MON-MASK-001 (per-batch)** — log fraction of P2-student electrodes dropped via shaft mask + fraction of student patches dropped via patch mask. Watch for drift from expected ~12.5% (shaft, post-2026-05-27 PM K=1 default; 1/N_shafts per clip, BT median 8 shafts) / ≤50% (patch).
  - **MON-MASK-002 (every 10k steps; BOUNDED-RATIO REVISION PM 2026-05-25; escalation update 2026-05-27 PM)** — for a held-out P2 probe batch, log `MSE(student_M4@orphan_parcels, teacher_M4@orphan_parcels)` separately from `MSE(student_M4@visible_parcels, teacher_M4@visible_parcels)`. **Pre-registered bound**: `orphan_MSE / visible_MSE ∈ [0.7, 1.5]` over a rolling 10k-step window. Violation in EITHER direction triggers escalation. **Under K=1 default (2026-05-27 PM)**: `ratio < 0.7` → mean-parcel-collapse → escalate to `R-stratified-shaft-mask` (P0; avoids uniform-random shaft selection making some parcels orphan in nearly every clip; the canonical mean-collapse mitigation, since reducing K further would mean K=0 which retires the shaft-mask test entirely); `ratio > 1.5` → student genuinely failing the orphan prediction task → escalate UP to `R-shaft-K2` (P0; increase orphan-supervision density on the theory that sparser orphans under K=1 are under-providing gradient), with `R-stratified-shaft-mask` as a secondary investigation if shaft-selection uniformity is the root cause. Direction-only `> 1.5` (original AM version) admits the mean-collapse failure; bounded ratio rules out both. (Prior K ≤ 2 era named `R-shaft-K1` as the step-down; that variant IS the default now, so the escalation targets reverse direction.)
  - **MON-MASK-003 (every 10k steps)** — log `|parcels_supervised[batch_subjects]|` distribution. Subject-level outliers (< 5 supervised parcels) signal anatomy registration bug; exclude before P2 dispatch.
  - **MON-MASK-004 (every P2/P3 checkpoint; subject-ID leakage canary, PM 2026-05-25)** — freeze the encoder, extract M4 mean-over-T per clip on a held-out probe batch (~256 clips × 9 BT subjects), fit logistic regression on `(d=256,) → subject_id ∈ {0..8}` with 5-fold CV. **Pre-registered threshold**: subject-ID F1 > 0.50 (above 1/9 = 11% chance) by P2 step ~30k signals subject-shape leakage via `parcels_supervised[subject]`'s deterministic per-subject footprint. Hit → escalate to `R-per-clip-pre-mask` (P0 sister) for the next dispatch. Rationale: B03f's gate is per-subject-deterministic, so M4 attention can encode subject ID; at P4 the CSubject linear probe may read subject directly from M4 instead of from latent acoustic content — corrupting the CSubject AUROC gate metric (≥ 0.628 vs CNN_Lap 0.578 + 0.05).
- **Reactive Gram anchoring** (B21) — re-armed at P2 step 0, OFF by default; activates only if P2 hits its own M4 trigger thresholds.
- **Reactive DKoleo@M3** (B22 Arm 3) — re-armed at P2 step 0, OFF by default; activates only if P2 hits its own M3 trigger thresholds.

Pre-baked sisters: `R-K-last-only` (K=1 final layer), `R-no-target-norm`. Micro-ablation cell (~1 GPU-h, single BT-Lite subject): K ∈ {1, 3, 4, 5, 6} sweep — K=6 is the default, K=1/3/4/5 bracket below.

**Masking (EX03 ✅ 2026-05-23 + B19 layering clarification 2026-05-24 + B20 v4 patch-grid 2026-05-24 + B03 substitution lock 2026-05-25).** Two mask layers stack in Phase 2 on the **student forward only**; the EMA teacher forward sees FULL electrodes and FULL patches (B03d asymmetry):
1. **Per-electrode patch mask** (same as P1; B03c paradigm B): 2D inverse-block on **F-patch × T-patch grid (10 × T_p)** per electrode, mask rate ≤ 50% per electrode. Student encoder operates on visible patches only; 2-block predictor (~0.2M, **warm-started from the P1 checkpoint at P2 step 0** per PM revert 2026-05-25) fills M2 at masked positions before L_pre_frame. Feeds `L_pre_frame` directly. No whole-electrode at this layer.
2. **Shaft-block electrode mask** (P2-only; B03 = drop): mask unit = SHAFT (physical sEEG depth electrode, 8–16 contacts on one needle). **K = 1 default with safety floor**: `K = 1 if N_shafts ≥ 2 else 0` (revised 2026-05-27 PM from the same-day K ≤ 2 fraction-based intermediate, which had revised the prior K=3 EX03 default). Block α only — shaft-extent 0.45–0.60 × time-extent 0.15–0.30 (blocks β/γ from the prior specs dropped). **Effective shaft-mask rate ≈ 1/N_shafts per clip** (~12.5% on BT 8-shaft median; was ~25% under K ≤ 2, ~40% under K=3), time-block span ≈ 1–2 s at T_p=20 (B20 v4 — was T=73 in v3). Combined with the 50% patch mask, the student sees ~43.75% of (electrode × patch) cells (was ~37.5% under K ≤ 2, ~30% under K=3); **this sits centered in the biosignal-FM band** (~44% visible = 56% combined mask vs Brant 40%, Brant-2 40%, LaBraM 50%, DIVER-1 50%, REVE 55%, Laya 60%). **Realized by ORing the shaft mask into the C_MAX key_padding_mask consumed by cross-attn ❺a / ❺b** (`key_padding_mask = pad_mask_C_MAX | shaft_mask`); cross-attn K/V never sees shaft-masked electrodes. NO learnable `[MASK]` token broadcast (paradigm A rejected at B03 lock 2026-05-25 — paradigm B drop is symmetric with B03c patch-mask drop and V-JEPA/Brain-JEPA precedent). Cross-attn cost drops ~12.5% on the student forward (was ~25% under K ≤ 2, ~40% under K=3). Feeds `L_mid_slot`, `L_post_frame`, and `L_post_utterance` via the cross-attn / latent-stack reconstruction objective.

   **Rationale for K = 1 default (revised 2026-05-27 PM)**: the same-day K ≤ 2 revision (which had revised the prior K=3 EX03 default) still landed at the biosignal-FM upper edge (62.5% combined, 37.5% visible) and inherited the "K=2 in fraction theatre" criticism — for BT's 7–10 shafts, `min(2, ⌈0.25·N⌉)` fires at the cap, so the formula collapsed to fixed K=2 with a K=1 fallback for ≤ 4-shaft outliers (the 0.25 fraction did no work for the real cohort). K=1 simplifies to a monotonic threshold (`K = 1 if N_shafts ≥ 2 else 0`), lands the combined mask **centered** in the biosignal-FM band (~56% combined, ~44% visible) rather than at the upper edge, and closes the catastrophic N_shafts=1 hole the K ≤ 2 formula had (`min(2, ⌈0.25·1⌉) = 1` → 100% drop). Brain-JEPA's own Table 6 "tube" lesion variant IS single-shaft × full-time, so K=1 is not v14-novel — it's a documented Brain-JEPA configuration elevated to default. Per-clip shaft-orphan supervision is sparser (~0.6–1.25 supervised parcels orphan per clip on BT vs ~1.25–2.5 at K=2), but cumulative across 400k P1 + 40k P2 steps the prediction signal is plenty, and sparser orphans reduce mean-parcel-collapse risk (MON-MASK-002 `ratio < 0.7` failure mode). K=2 demoted to P0 falsifier `R-shaft-K2`; K=3 stays as P1 falsifier `R-shaft-K3-mixed-3block`.
3. **`parcels_supervised[subject]`** (B03f; per-subject, NOT per-clip): the supervision gate for L_mid_slot / L_post_frame / L_post_utterance PMA / L_DKoleo / L_DKoleo@M3 is `parcels_supervised[subject]` — the per-subject set of DK parcels with ≥ 1 electrode for the subject's anatomy, computed ONCE at extractor time, invariant to within-clip shaft mask. Shaft-orphaned parcels REMAIN in `parcels_supervised[subject]` and ARE supervised — that is the JEPA prediction task. The prior B19 "valid_parcels[clip] recomputed post-mask" semantic was JEPA-inverted (it excluded the prediction targets). Cross-ref [[project_v14_b03_mask_lock_2026_05_25]] §B03f for full reasoning.

Reason for shaft as the mask unit: contacts on the same shaft are highly signal-correlated (spatial-proximity ~mm scale); masking by parcel is impractical because BNA parcels often have only 1–2 electrodes per subject — too sparse to be a meaningful drop unit. Brain-JEPA's "ROI" maps semantically closer to v14's "parcel," but their fMRI ROIs have dense voxel coverage; v14 doesn't. Observation context = 84–100% × 84–100% of remaining tokens (Brain-JEPA Table 6). Figure 6 verified 6× efficiency on HCP-Aging Age (50ep spatiotemporal ≈ 300ep vanilla), robust across HCP-Aging Sex (100ep ≈ 300ep) and ADNI NC/MCI (200ep ≈ 300ep). Pre-baked sisters: `R-shaft-K2` (NEW 2026-05-27 PM, P0 — fixed K=2 with blocks α+β; falsifier for the K=1 default and the same-day K ≤ 2 fraction intermediate that was held ~5h; settles the call empirically against the biosignal-FM upper-edge target), `R-shaft-K3-mixed-3block` (P1 falsifier; 2026-05-27 AM addition demoted P0→P1 PM same-day — restores the prior K=3 EX03 default with all three mixed-extent blocks α/β/γ; tests the full literature-transfer of Brain-JEPA's fMRI default in case both K=1 and K=2 underfit the orphan task), `R-stratified-shaft-mask` (P0 — stratifies shaft selection to avoid uniform-random making some parcels orphan in nearly every clip; mean-collapse mitigation), `R-p2-random-electrode-mask` (original Bernoulli 30% baseline; demoted from default), `R-p2-shaft-tube-full-time` (full-time × shaft-mask only — Brain-JEPA "tube" lesion variant; semantically close to K=1 default), `R-p2-parcel-mask` (parcel-axis instead of shaft-axis — tests whether anatomical-region grouping beats physical-shaft grouping for masks; was the implicit "parcel/shaft" ambiguity in the original commit), `R-p2-mask-rate-{8,12,18}` (J1 sweep around new ~12.5% midpoint under K=1 default; previous {15,25,35} (K ≤ 2 era) and {25,30,50} (K=3 era) both superseded), `R-p2-no-patch-mask` (B19-added — P2 with shaft-mask only, no per-electrode patch mask; tests whether the patch-mask layer is load-bearing on top of the shaft mask).

**Valid-bin symmetric teacher mask (EX09 ✅, 2026-05-23).** Same as Phase 1: teacher and student receive byte-identical per-corpus valid-bin masks (input fill 0 + key-padding −∞). Post-forward defensive zero-out: `teacher_features[..., invalid_bins] = 0.0`. Loss supervision uses `ssl_mask & valid_bin_mask` intersection. Runtime assert in debug.

**Mask token mechanism (B03 ✅ 2026-05-25).** Shaft-mask substitution = **DROP** (B03 lock; via key_padding_mask, see Masking §2 above). Patch-mask substitution = **DROP + 2-block predictor** (B03c paradigm B; see Masking §1 above + canonical memo). Paradigm A (learnable `[MASK]` token) retained as sister `BiL-B03-PatchMask-Paradigm-A` (P0 must-run) + `R-p2-mask-learnable` (P1).

**Data (locked 2026-05-22, sEEG-only diet).** D-cohort 180 h + BT 43.5 h ≈ **224 h**. AJILE12 dropped per Peterson 2022 Table 2 (89.7% surface ECoG / 10.3% depth — topology mismatch vs BT sEEG eval). SWEC has no electrode anatomy → cannot feed cross-attn ❺. Fallback if underfit: chronic Cogan-lab sEEG (future acquisition).

**Sampler (M06 ✅; cohort expanded 2026-05-23 PM after D-cohort audit).** **Uniform-per-subject** across **94 sEEG subjects (D 85 + BT 9)**. Per-row weight = `1 / (N_subjects_global × clips_per_subject[subject_id])`. Shares: **D-cohort 90.4% / BT 9.6%** (85:9 subject ratio). Phase 2's unit of learning is one subject's anatomy/coverage pattern; hour-weighting would make a 30h D-cohort subject contribute 6× more routing-pattern signal than a 5h BT subject — backwards for cross-subject generalization. **D-cohort expanded from "10" to "85" (97.7% of 87 unique D-pts pass FS recon + RAS coords + raw `.fif` gate; D107/D139 fail anatomy)** — recipe's prior "10 D-subjects" was a stale PoC-era cite. Hours unchanged at 224h (D 180h union of 4 tasks across 85 patients + BT 43.5h); the audit corrected subject count, not hours. **BT gradient share drops from 47.4% → 9.6% under uniform-per-subject; accepted by Ben on 2026-05-23 as the correct cross-subject setup** (Phase 2's unit of learning is anatomy-config diversity, and 85 D-anatomy-configs provide that). Per-subject audit memo: `memory/project_d_cohort_phase2_cohort_audit_2026_05_23.md`.

**Predictor.** 2-block transformer for `L_pre_frame` (warm-started from P1 checkpoint at P2 step 0 per PM revert 2026-05-25 — see §4 Loss). 1-layer Linear(d, d) `loss_mid_head` for `L_mid_slot` (B22; continues from P1 checkpoint). None for `L_post_frame` or `L_post_utterance` (cross-attn + latent stack IS the predictor for L_post_frame; direct regression for L_post_utterance per EAT §3.1).

**Optimizer.** See §7. Peak LR 3e-4 @ effective batch 512, 5k linear warmup, cosine → 0, 40k steps. (EMA teacher schedule already specified in the §4 Loss block above.) **No cool-down at end-of-P2** — an end-of-P2 cool-down (clip 1s → 2s, linear LR re-decay) was scoped at B22 AM lock 2026-05-25 and reverted PM same day on coherence review: P3 retrains at 5s for ~18k unfreeze steps and washes out any P2-tail adaptation before P4 sees it, AND the V-JEPA-2.1 §2.3.5 precedent is direction-mismatched (V-JEPA cools UP to higher-res eval; v14 would cool DOWN to lower-T_p eval, with Whisper-L8 supervision density dropping 5× at the P3-tail alternative). The only coherent place a v14 cool-down can land is the P3 tail (preserved as P1 contingency sister `R-p3-tail-cooldown-1s` in §8). See B22 canonical memo Decision provenance item 9.

**Output.** Full-stack checkpoint (❶❷❺❻ + LN_mid + LN_frame + LN_utt + PMA + loss_mid_head + LearnableParcelEmbed + LearnableSubSlotEmbed). Discard teacher (incl. LN_mid_T / LN_frame_T / LN_utt_T) and P2 predictor.

---

## 5. Phase 3 — Cross-modal distillation (single-teacher Whisper, layer k\* preflight-picked)

> **No-Goldstein-default callout** ([[feedback_no_default_ecog_to_seeg_transfer_2026_05_24]]). Every `L8` / `10 Hz` / `50 buckets` mention in this section is a **search-range anchor**, not a locked default. The preflight below picks (k\*, r\*) empirically on sEEG; if it doesn't land on (L8, 10 Hz) the downstream §5 specs (target shape, mean-pool factor, triangular-pool spec, frozen lists) substitute accordingly. ECoG↔sEEG modality gap (surface dense vs depth sparse, mixed white/grey, different per-band SNR, different conduction lag) is too large to inherit literature picks as defaults — the literature anchors a sweep range, no more.

**Goal.** Align v14's latents to syllable-rate acoustic-phonetic representation. Stage A + B frozen during Stage 3b-warmup; Stage A slow-LR unfrozen in Stage 3b-unfreeze.

**Teacher.** Whisper-large-v3 layer k\* (empirical, see preflight). Search-range anchor L8 from Antonello / Shimizu / Mesgarani — acoustic-phonetic, not semantic; NOT Goldstein, which uses Whisper-*medium* L4 + decoder L3 on ECoG. Frozen throughout.

**Preflight (~1 GPU-hour, before lock; layer range widened past Goldstein's neighborhood per the no-default callout above; rate sweep also widened to {5, 8, 10, 20} Hz given v4's 8 Hz native frame rate).** Ridge sweep over (Whisper-layer k ∈ {L4, L6, L7, L8, L9, L10, L12, L16, L20}, rate ∈ {5, 8, 10, 20} Hz, short lag sweep) on one BT-Lite subject with high STG/IFG coverage. Per-electrode ridge regression Whisper-Lk → BT HG-envelope; α ∈ {1e-3, 1e-1, 1, 10}; r² in speech cortex (STG/IFG/motor) on 80/20 train/val split, report val. **No literature-anchor tie-break** — within Δ ≤ 0.005 take the lower-layer / lower-rate candidate (cheaper at distillation time), not the literature pick. Cross-checked by the task-fit ceiling preflight (`ablations.md §L.7.{B0, C0, S-layer, S-combine}`); convergent picks ⇒ lock, divergent ⇒ resolve by a small Phase-4 probe at the top-2 candidates.

### Target side (Whisper)

Per 5s clip:
- Encoder input 80-mel × 100 Hz × 30 s. Front conv stride-2 → transformer at 50 Hz, d_W = 1280.
- L8 for 5s: (250, 1280).
- Teacher-side triangular pool 50 → 8 Hz, factor 6.25, triangle base 250 ms (half-base 6.25 = stride; true half-max FWHM 125 ms) → (40, 1280) (**B06 PM lock 2026-05-25**: matches v14 student native 8 Hz; identity passthrough on student side; sum-to-1 per bucket; zero-pad edges).
- **2-layer MLP adapter** Whisper-side `Linear(1280, 256) → GeLU → Linear(256, 256)` (LLaVA-1.5 shape ~393k params) → (40, 256) target.

**No instance-norm on Whisper target** (deleted 2026-05-23). Whisper-L8 is pre-norm transformer output (already LN'd); data2vec-2.0 instance-norm guards EMA-self-teacher collapse, a risk absent in frozen-external-teacher distillation. Double-norm destroys per-token magnitude carrying acoustic-phonetic content.

### v14 side

Per 5s clip (B20 v4 frame rate revision + B06 PM lock 2026-05-25):
- (320 parcels, T_p=40, d=256) → PMA k=1 over parcels → (T_p=40, 256) at 8 Hz. PMA softmax masked to `parcels_supervised[subject]` (B03f; per-subject, NOT per-clip). Latent stack ❻ self-attn key_padding_mask = `~slot_mask(parcels_supervised[subject])` (B03b; carried from P1/P2 checkpoint).
- **Identity passthrough on student side** at 8 Hz native: (T_p=40, 256). No time pool; matches teacher (40, 256) after teacher-side triangular pool 50 → 8 Hz factor 6.25 triangle base 250 ms (half-base 6.25 = stride; true half-max FWHM 125 ms).
- **Default**: r* = **8 Hz** / 40 buckets (B06 PM lock 2026-05-25; matches v14 student native; first-principles syllable+theta 4–8 Hz; Goldstein's 10 Hz now a sister-cell falsifier `R-rate-10Hz`). Sisters: `R-rate-{5, 10, 16}Hz` retain as rate falsification.

**Triangular pool spec (B05 + B06 PM lock 2026-05-25).** Teacher-side pool only: Whisper-L8 50 Hz → 8 Hz, factor 6.25; bucket j centered at `j × 125 ms`, triangle base 250 ms = 1 bucket stride (half-base 6.25 frames = stride); true half-max FWHM 125 ms; each bucket aggregates ~12-13 frames; linearly-decaying weights, sum-to-1 per bucket, zero-pad edges. Student-side is identity passthrough at 8 Hz native (no pool). Note: `weight_matrix.sum(dim=-1).allclose(torch.ones(40))` is a per-ROW sum-to-1 normalization check, NOT a width or partition-of-unity check — both the live kernel and the 2×-wider variant pass it.

### Loss

**Smooth-L1 raw** on un-normalized (50, 256) v14 vs (50, 256) Whisper-projected, β=1.0 (PyTorch / data2vec 2.0 default; `nn.SmoothL1Loss(beta=1.0, reduction='none')` then masked mean). 50 alignable supervision signals × 144k clips = 7.2M total. AB10 sister sweeps β ∈ {0.5, 1.0, 2.0} + MSE + cosine.

### Stage 3b-warmup (adapter-only, ~10% of P3 budget = 2k of ~20k steps)

- **Frozen**: Whisper-L8 + Stage A + Stage B (incl. PMA query). All `requires_grad_(False)` for the warmup window.
- **Trains**: linear projector to d_v14 (~0.6-0.8M params). The triangular pool is parameterless. **B31 ✅ 2026-05-28 PM**: PMA query receives gradient **for the first time** when Stage B unfreezes in 3b-unfreeze — the joint SSL phase has no `L_post_utterance` term so PMA stayed at its initial values through P1+P2. (Pre-B31 spec said "PMA gradient resumes" because B19 trained PMA via `L_post_utterance` at P1+P2; B31 dropped that term.)
- **Optimizer**: AdamW (β1=0.9, β2=0.95, wd=0.05). Peak 3e-4 @ effective batch 256, 500 linear warmup, constant, **2k steps**.

### Stage 3b-unfreeze (~95% of P3 budget)

- **Frozen**: Whisper-L8 only.
- **Slow-LR (LR/10)**: Stage A per-electrode tokenizer (preserve 10k h pretrain).
- **Normal LR**: Stage B latent stack + cross-attn + PMA + adapter.

- **Optimizer**: AdamW (β1=0.9, β2=0.95, wd=0.05). Adapter peak 4.2e-4 (= 3e-4 × √(512/256)), Stage B at /3 = 1.4e-4, Stage A at /10 = 4.2e-5. 1k Stage-A-only linear ramp, cosine → 0, **18k steps cap + early-stop** on val Smooth-L1 (5% held-out-by-session, patience 5 ep, eval every ~1k steps). Effective batch 512.

Total Phase 3 ≈ 20k steps ≈ 33 ep over 144k paired clips (MTDP-band).

**Adapter EMA.** None (teacher is external + frozen).

**Contingency (NOT default).** L2-SP regularizer on Stage A vs P2 checkpoint, λ=1e-4, only if 3b-unfreeze shows Stage-A drift.

**Data scope.** BT 9 subjects (~40 h) + D-cohort (~180 h) = ~220 h paired iEEG + audio. PS+lex dropped.

---

## 6. Phase 4 — Downstream eval

**Clip length.** 1 s (T = 8, T_p = 4 after Conv2d (3,2); B20 v4 — was T=15 in v3 at 14.7 Hz) — matches Neuroprobe / iMINDBench eval. SSL trained on 5 s; RoPE generalizes downward without recompile.

**Readout.**
1. Parcel collapse: PMA k=1 frozen seed query (trained in P1+P2+P3) → (T_p=4, d=256). PMA softmax masked to `parcels_supervised[subject]` (B03f; per-subject, NOT per-clip; identical mask as P1/P2/P3). Latent stack ❻ self-attn key_padding_mask = `~slot_mask(parcels_supervised[subject])` (B03b; carried from P3 checkpoint; frozen at P4).
2. **No time pool.** Flatten (T_p, d) → T_p·d = **1024-dim** feature vector (B20 v4 — was 3840-dim in v3 at T=15).
3. Per-task `Linear(1024 → n_classes)`.

iMINDBench parity: matched probe-side protocol (flatten, don't pool), matched probe capacity (1024 dim ≪ iMINDBench's ~50k flattened — comfortably fair comparison). Sister `R-p4-clip-2s` recovers probe capacity to 2048 dim (T_p=8) if the 1024→3840 capacity drop costs accuracy on probe-capacity-bound tasks.

**Three transfer paths (ordered by paper-claim strength).**

- **Path A (HEADLINE)**: Frozen Stage A + Stage B + adapter + flattened (T·d) linear-probe per task. Competes directly with DIVER-1 frozen-linear-probe 0.678 and iMINDBench Multi-STFT-Logistic 0.663 (both within-session).
- **Path B (sensitivity column, NOT a gate prong)**: Frozen everything + 2-layer MLP probe per task. Hidden = d = 256, GeLU, no dropout, same optimizer/schedule as Path A. Reported to show probe-capacity ceiling; Path A remains THE headline.
- **Path C**: Light task fine-tune — Charmander recipe (50-epoch decoder warmup → partial 4-layer unfreeze).

**Linear-probe optimizer (Path A + B).** AdamW (lr=1e-3, wd=0.0), 100 ep cap, early-stop on val balanced-accuracy patience=10, batch 256. Val and test splits distinct: val used only for early-stop; test number reported unconditionally (DIVER-1 / iMINDBench / CLIP Radford 2021 Table 9 linear-probe protocol). Backbone wrapped in `torch.no_grad()` to eliminate the gradient path through 13M frozen params.

**Probe scheduling.** Sequential per-task: one NeuralTrain `Experiment` instance per task, dispatched via Exca grid; no shared state across tasks. Joint multi-task probe is a sister only if a downstream task scale forces it (no current trigger).

**Headline pairing.** Report BOTH Path A1 (Stage A frozen, post-P3) and Path A2 (Stage A unfrozen via 3b-unfreeze) per Cambrian Table 16 lever. Δ tells whether Phase 3 improves Stage A beyond pure SSL.

**Submission gate.** ≥ **0.667** CrossSession multi-class AUROC (Linear-Lap+spec 0.617 + 0.05) AND ≥ **0.628** CrossSubject binary AUROC (CNN-Lap+spec 0.578 + 0.05), ≥ 4 tasks, ≤ 30M params. iMINDBench's 0.663 / DIVER-1's 0.678 are within-session — NOT gate ceilings.

---

## 7. Optimizer / precision / distributed (cross-phase)

**Optimizer (all phases).** AdamW. β1 = 0.9, **β2 = 0.95**, weight_decay = 0.05, eps = 1e-8. grad_clip = 1.0. Two param groups: no-WD on biases, LN/RMSNorm γβ, the 320 free latents, and the freq embedding.

**Precision.** bf16 mixed-precision; fp32 master weights + fp32 AdamW state + fp32 EMA buffers. **Front-end runs fp32 explicitly** (autocast disabled): Multi-STFT → filterbank → log → robust-z. log-ε = 1e-6.

**Distributed.** DDP only (no FSDP / ZeRO at 13M). `gradient_as_bucket_view=True`, `static_graph=True`. Activation checkpointing ON for token blocks ❷ + latent stack ❻.

**LR-batch scaling.** √-rule: `LR_eff(B) = LR_ref × √(B/B_ref)`, valid to ~64k effective batch (Cherti 2024).

### Phase table

| Phase | Trainable | Peak LR @ B_ref | Warmup | Schedule | Steps | Effective batch |
|---|---|---|---|---|---|---|
| 1 | full ❶❷❺❻ + LN_mid + LN_frame + LN_utt + PMA + 2-block predictor + loss_mid_head (~15.024M; B19 + B21 + B22 — cross-attn ON, bias OFF) | 5e-4 @ 1024 | 20k linear | cosine → 0 | 400k (sisters 300k) | 1024 |
| 2 | full stack (~15.024M, continues from P1 checkpoint; bias ON; +shaft mask layer **K = 1 default** `K = 1 if N_shafts ≥ 2 else 0` per 2026-05-27 PM K-cap revision of B03/EX03 — supersedes same-day K ≤ 2 fraction intermediate held ~5h; 2-block predictor **warm-started from P1 checkpoint** at P2 step 0 per B03c PM revert 2026-05-25) | 3e-4 @ 512 | 5k linear | cosine → 0 | 40k | 512 |
| 3a (warmup) | adapter (~0.8M) | 3e-4 @ 256 | 500 linear | constant | 2k | 256 |
| 3b (unfreeze) | adapter @ 4.2e-4; B @ 1.4e-4 (/3); A @ 4.2e-5 (/10) | derived | 1k Stage-A-only ramp | cosine → 0 | 18k cap + early-stop | 512 |

**EMA momentum spans full phase** (NOT warmup window): P1 linear 0.99 → 0.9999 over full 400k; P2 linear 0.999 → 0.9999 over full 40k.

**B_crit monitoring hook.** Log per-step grad norm + per-corpus gradient contributions during P1 first 20k steps. Empirical batch-size validation.

**Confidence at lock.** ~88% global / >95% on high-impact items (AdamW family, precision, DDP, cosine→0, no double-norm on Whisper-L8). Remaining ~12% in {effective-batch precise value, P1/P2 step counts, EMA endpoint, predictor shape} — all sister-cell-falsifiable or M0-empirically-resolvable.

---

## 8. Sister cells (P0 + P1, current roster)

### Encoder / arch (P0 default sweep set)

- `F-joint-vs-factorized` (B20 v4 — renamed/reframed from v3's `F-joint-tF`) — token-block JOINT default vs factorized t-then-f sister. Default is now joint; sister tests whether factorization saves at v4's token count.
- `F-single-STFT` (B20 v4 — **promoted from P1 to P0 must-run**) — single Nperseg=1024 STFT vs Multi-STFT default. Gates Multi-STFT defense; iMINDBench's main win is spectrogram-vs-waveform (+0.128 AUC), multi-vs-single only ~+0.005–0.010.
- `F-patch-(2,2)` (B20 v4) — F_p=2 patch alternative, under-mixed band-wise but cleaner band purity.
- `F-patch-(4,2)` (B20 v4) — F_p=4 patch alternative, more compute saving, over-mixed band-wise.
- `F-linear-STFT-38` — linear STFT 38-bin vs log ⅓-octave 30-bin.
- `F-150Hz-cap` — cap log-bins at 150 Hz vs extend to 813 Hz.
- `single-cross-attn` — 1 cross-attn vs 2 @ {0, 3}.
- `3-cross-attn @ {0,2,4}` — 3 cross-attns vs 2.
- `F10b` — AST 2×4 patches stem.
- `BiL-NoSpatial`, `BiL-MNI`, `BiL-Freq-Plain` — Bitter-Lesson prior load-bearing checks.
- N-sweep {2, 4, 6}, L-sweep {default + neighbors}, d-sweep {128, 192, 256}.

### Encoder / arch (P1)

- `F-CQT`, `F-Morlet`, `B-edges`, `F-band-prior-spacing`, `F-½-octave`. (F-single-STFT promoted to P0; see above.)
- `F-CQT` — principled log-spaced multi-resolution (constant-Q across log-freq); zero iEEG-FM precedent + degenerate sub-2 Hz at 5s clip (Q=12 needs 12s window for 1 Hz). Stays P1; promote only if Multi-STFT defense (F-single-STFT) weakens.
- `F-quarter-octave-high-band` (B20 v4) — non-uniform binning: ⅓-oct below 64 Hz, ¼-oct above.
- `F-patch-partial-valid` (B20 v4) — zero-fill invalid bins within partial F-patches before Conv2d, vs default of marking partial patches invalid.
- `factored-freq-embed`, `F-no-freq-embed`, `F-per-bin-freq-embed-retained` (B20 v4 — keep per-bin AND per-patch).
- `F-freq-prepool` (fallback if cross-attn band attention degenerates).
- `F-depth-bias` — learned B[m, depth] bias table.
- `F-shaft-id` — electrode-shaft id as input feature.
- `N2 factorization-order` — time-first vs parcel-first **latent stack** (token blocks are joint per B20 v4, no longer a factorization-order knob).

### Loss design (P0 — must-run before Stage-1 lock; B19 2026-05-24 + B21 2026-05-25 + B22 2026-05-25)

- `BiL-Loss-Default` (RETIRED 2026-05-25 in favor of `BiL-B22-Default` below — B22 added `L_mid_slot` making the headline 5-term, NOT 4-term). The label is reserved to avoid roster confusion; `BiL-B22-Default` IS the canonical headline cell from 5/25 onward.
- `BiL-Loss-NoUtt` — drop L_post_utterance (set λ_utt = 0). Tests whether the utterance term is load-bearing at v14 scale. Falsifies the EAT utterance-lift transfer claim.
- `BiL-Loss-NoPre` — drop L_pre_frame (set λ_pre = 0). Tests whether the M2-output gradient is load-bearing once M4-output supervision exists. Falsifies the V-JEPA-2-canonical M2 masked-prediction loss transfer at v14 scale.
- `BiL-Loss-LambdaSweep` — λ_utt sweep over {0.1, 0.5, 1.0, 2.0, 5.0} on a single BT-Lite subject (~1 GPU-h). Brackets EAT's λ=1 default.
- `BiL-IdentityInit-Default` (B21) — identity-anchored init + DKoleo + dedicated-LN per loss head package as defined. THIS IS the headline collapse-prevention cell (dispatched with `BiL-Loss-Default`).
- `BiL-CollapsePrev-NoIdentityInit` (B21) — drop identity-anchored init (return to trunc-normal init for the 320 latents). Tests whether identity-anchored init is load-bearing. Falsifies the NuCLR / DINOv3 / Slot-Attention precedent at v14 scale.

### Loss design / collapse prevention (P1)

- `BiL-CollapsePrev-NoDKoleo` (B21) — drop DKoleo (λ_DKoleo = 0). Tests whether DKoleo is load-bearing on top of identity-anchored init.
- `BiL-CollapsePrev-SharedLN` (B21) — drop dedicated LN per loss head (single LN shared by both heads, prior v3 behavior). Falsifies DINOv3 § 3.3 transfer.
- `BiL-CollapsePrev-IdentityInitOnly` (B21) — identity-anchored init only (no DKoleo, no dedicated LN). Minimum viable identity scaffold.
- `BiL-CollapsePrev-AllOnFromStep0` (B21) — force Gram anchoring on from step 0 with weight 0.1 (DINOv3 reports this is fine, just inefficient). Tests whether reactive vs always-on matters at v14 scale.
- `BiL-CollapsePrev-DropSWEC-Fallback` (B21) — fallback cell as default-from-start (P1 on 224h D+BT only with bias ON). Bounds the worst-case collapse cost: if this beats default P1, default P1 is broken.
- `BiL-CollapsePrev-Gram-snapshot-{5k,10k,20k}` (B21 — dispatch-only-if-Gram-triggers) — snapshot frequency for Gram teacher in reactive regime.
- `BiL-Loss-AddKoLeo-PMA` — add DINOv3 KoLeo uniformity regularizer on the PMA output (`+ 0.1·L_KoLeo` on `clip_summary`). Only if Loss 4's PMA collapses despite slot-side DKoleo (diagnostic D measures PMA entropy separately).
- `BiL-Loss-AddGram-P4-readout` — add DINOv3 Gram anchoring at the P4 readout (separate from B21's reactive Gram anchoring on M4 during SSL). Only if frozen-linear-probe at P4 shows latents have drifted from semantic separability after P3.

### Loss design / dense features (B22 2026-05-25)

P0 must-run (before Stage-1 lock; dispatched alongside `BiL-IdentityInit-Default`):

- `BiL-B22-Default` — the 5-term loss (L_pre_frame + L_mid_slot + L_post_frame + 1.0·L_post_utterance + 0.1·L_DKoleo) package as defined; reactive DKoleo@M3 armable. THIS IS the B22 headline cell.
- `BiL-B22-NoMidSupervision` — drop Arm 1 (return to B21's 4-term objective; no L_mid_slot). Tests whether M3 supervision is load-bearing on top of B21's identity-init + DKoleo@M4. Falsifies B22's parcel-routing-checkpoint supervision claim (DINOv3 §4 / DINOv2 KoLeo / B21 collapse-prevention lineage; **B27 reframed** away from V-JEPA 2.1 §2.3.2 DSS — DSS was contingent on the context loss B27 reverted).

P1 (load-bearing sisters):

- `BiL-B22-MidLambdaSweep` — λ_mid sweep over {0.25, 0.5, 1.0, 2.0} on a single BT-Lite subject (~1 GPU-h). Brackets the B22 default λ_mid=1.0.
- `BiL-B22-MidPredictor` — adds a 2-block predictor on the M3-side (mirroring L_pre_frame's predictor) instead of direct linear loss_mid_head. Tests whether a deeper predictor at M3 transfers V-JEPA 2.1's deep-SS gain better than the 1-layer head.
- `BiL-B22-DKoleo-M3-AllOnFromStep0` (dispatch-only-if-Arm-2-triggers) — force DKoleo@M3 on from step 0 with weight 0.05. Tests reactive vs always-on for the M3-side regularizer (mirrors B21's `BiL-CollapsePrev-AllOnFromStep0` for the M4 side).
- `R-p3-tail-cooldown-1s` (B22 PM contingency; dispatch only if Path-A P4 baseline shows large 5s SSL → 1s eval drop) — final ~3% of P3 unfreeze (~500 of 18k steps) cools clip 5s → 1s with Whisper-L8 target re-pooled to (10, 1280) at the same 10 Hz rate. Only coherent place a v14 cool-down can land: the P2-tail cool-down was rejected because P3 retraining at 5s would wash it out before P4 (Decision provenance item 9 in canonical memo). Discard if cool-down portion's Smooth-L1 exceeds pre-cool-down by >5% (Whisper supervision density 5× drop = real cost).
- `BiL-Loss-L1` — swap MSE → Smooth-L1 β=1.0 on all three losses. Only if MSE shows outlier-driven gradient instability. data2vec 2.0 §3.4 precedent.

### Phase 1 SSL recipe (P1)

- `S-constant-p1` — DINOv3 constant LR.
- `S-wsd-p1` — OmniMouse WSD with cooldown @ 200k.
- `R-K-last-only` — target from last token block only vs K=4 average (P1) / vs top-K=4 of 6 (P2).
- `R-no-target-norm` — drop per-layer instance-norm on teacher latents.
- `R-p1-mask-rate-{30,40,50}` — per-electrode patch mask-rate bracket at ≤ 50% cap (B19 narrows the prior 60–65% spec).
- `R-p1-stage-a-only` — B19-added — Phase 1 with cross-attn OFF + latent stack OFF + L_post_frame and L_post_utterance disabled (the prior pre-B19 default). Load-bearing P1 sister: if it wins, B19 reverses and we go back to per-electrode-only Phase 1.
- `R-p1-frame-1d-keep-3-mask-80` (added 2026-05-23, B08 closure) — original frame-1D / keep-3 ≈200ms / 80% mask commit (audio-MAE-inheritance). Load-bearing P1 sister; if it wins, the Option-B default flips back.
- `R-predictor-mlp` — replace 2-block transformer predictor with MLP.
- `R-clip-length` — 3 s / 5 s / 8 s SSL clip length sweep.
- `R-UFO-window` — Phase-2 utterance window 2 s / 5 s / 8 s.
- `R-sampler-alpha03` — α=0.3 flat over vb-eh (1st-audit prior default, demoted 2026-05-23 2nd-round; load-bearing — if it wins, the default flips back).
- `R-sampler-sqrth` (α=0.5 flat over hours, ignores vb-eh correction).
- `R-sampler-pure-h` (α=1.0 over hours).
- `R-sampler-uniform` (α=0.0 per corpus, 25% each).
- `R-sampler-broadband-uniform` (SWEC capped + broadband boosted).
- `R-sampler-seeg-only` — drop AJILE12 from P1 entirely; α=0.5 hierarchical over remaining 3 corpora (SWEC vs D+BT). Modality-invariance probe — tests whether 27.7% AJILE12 ECoG share distorts Stage-A statistics enough to hurt Phase-2 keys.
- `R-sampler-40-60`, `R-sampler-60-40` (added 2026-05-23, 3rd audit) — macro-split sweep around the 50/50 default. Only theoretical-anchor-free component in B02; no 2026 SOTA pretrain (audio/video/vision/LLM) commits to a specific macro-cap percentage. These two sisters bracket the default and empirically settle it.

### Phase 2 (P1 / P2)

- `R-p2-pooled-h` — old hour-pooled sampler (D 80% / BT 20%).
- `R-p2-uniform-session` — session-level vs subject-level granularity.
- `R-p2a-bias-off-pretrain` — brief SWEC bias-off warmup before P2b (recipe-default rejected; preserved as sister).
- `R-shaft-K2` (NEW 2026-05-27 PM — P0 falsifier for the K=1 default) — fixed K=2 with blocks α+β. Falsifies the K=1 default and the same-day K ≤ 2 fraction-based intermediate that was held ~5h between the AM K=3→K≤2 cascade and the PM K≤2→K=1 cascade. Settles the call empirically against the biosignal-FM upper-edge target (62.5% combined / 37.5% visible).
- `R-shaft-K3-mixed-3block` (2026-05-27 — demoted P0→P1 PM same-day) — restores the prior K=3 EX03 default with all three mixed-extent blocks α/β/γ. Tests the full literature-transfer of Brain-JEPA's fMRI default in case both K=1 and K=2 underfit the orphan task.
- `R-stratified-shaft-mask` (P0) — mean-collapse mitigation: stratifies shaft selection to avoid uniform-random making some parcels orphan in nearly every clip. MON-MASK-002 `ratio < 0.7` escalation target under K=1 default.
- `R-p2-random-electrode-mask` (added 2026-05-23, EX03 closure) — original 30% Bernoulli per-electrode mask; load-bearing sister against new shaft-block multi-block default.
- `R-p2-shaft-tube-full-time` (added 2026-05-23, EX03 closure) — full-time × shaft-mask only (Brain-JEPA "tube" lesion variant; semantically close to K=1 default).
- `R-p2-mask-rate-{8,12,18}` (J1 sweep, revised 2026-05-27 PM) — bracket the new ~12.5% effective midpoint under K=1 default. Previous {15,25,35} (K ≤ 2 era) and {25,30,50} (K=3 era) both superseded.

### B03 mask-discipline sisters (2026-05-25 — load-bearing)

P0 must-run (before Stage-1 lock):

- `BiL-B03-PatchMask-Paradigm-A` (B03c lock — load-bearing P0) — paradigm A alternative for per-electrode patch mask: learnable `[MASK]` token broadcast across F_p × T_p of masked positions, NO predictor. Tests whether paradigm B (drop + 2-block predictor, the new default) is load-bearing on top of biosignal-FM precedent (LaBraM, Brant-2, REVE — all paradigm A). Cited as `BiL-` because it falsifies a Bitter-Lesson v14-prior (predictor head); if A wins, drop the predictor + return to BERT-style fill.
- `R-p2-teacher-shaft-mask-symmetric` (PROMOTED FROM P1 → P0, PM 2026-05-25 — B03d sister) — symmetric teacher: EMA teacher also receives shaft mask (P2-only); breaks the B03d asymmetry. Promoted to P0 because Tier-1 red-team flagged a mean-parcel-collapse degenerate solution under the asymmetric teacher: student must predict the teacher's orphan-parcel output from `LearnableParcelEmbed[p] + LearnableSubSlotEmbed[s] + neighbor-parcel SA` — optimal degenerate solution = output the EMA-teacher's running mean per parcel, which satisfies MSE while learning nothing cross-electrode. MON-MASK-002 bounded-ratio monitor (in §3 / §4 diagnostic block + B03 canonical memo) is the pre-registered detector; this sister is the canonical falsifier.
- `R-per-clip-pre-mask` (NEW P0 SISTER, PM 2026-05-25 — B03f sister) — third per-clip alternative for the supervision gate: compute `parcels_covered[clip ∩ session]` BEFORE shaft mask (per-clip but pre-mask, so shaft-orphaned parcels REMAIN in the supervised set even though it's per-clip-scoped). Mediates between B19's per-clip `valid_parcels[clip]` (JEPA-inverted post-mask) and B03f's per-subject `parcels_supervised[subject]` (JEPA-correct but per-subject-deterministic and therefore subject-ID leakable). Tier-1 red-team motivated: B03f's per-subject set is deterministic per subject, so the M4 attention pattern encodes subject ID; at P4 the linear probe (CSubject task) may read subject from M4 instead of latent acoustic content, corrupting CSubject AUROC. MON-MASK-004 subject-ID linear-probe canary (in §3 / §4 diagnostic block + B03 canonical memo) is the pre-registered detector; this sister is the canonical mitigation.

P1 sisters:

- `R-p2-mask-zeros` (B03 sister) — paradigm C: zero-fill shaft-masked electrode embeddings (no `[MASK]` token, no drop, kept in cross-attn K/V as zero vectors). Tests whether the cross-attn cost saving (B03 drop ~40% on student forward) is what matters, vs the substitution semantics.
- `R-p2-mask-learnable` (B03 sister) — paradigm A applied at shaft-mask layer specifically: learnable `[MASK]` token broadcast across F_p × T_p of every contact on a shaft-masked electrode. Symmetric with `BiL-B03-PatchMask-Paradigm-A` but at the shaft layer; tests whether B03 = drop is load-bearing at the *shaft* (P2-only) layer separately from the patch (P1+P2) layer.
- `R-p2-predictor-reinit` (NEW P1 SISTER, PM 2026-05-25 — B03c sister) — re-init the per-electrode patch-mask predictor at P2 step 0 (NOT warm-started from P1). Falsifies the warm-start default that replaced the original re-init plan after Tier-1 red-team review (re-init coincides with bias-OFF→ON flip, no JEPA-family precedent for inter-phase predictor discard, gradient noise risk through ❺ and ❻ during the most fragile transition).
- `R-clip-level-valid-parcels` (B03f sister) — revert to B19's `valid_parcels[clip]` per-clip semantic (computed from post-mask electrode set, JEPA-inverted). Tests whether the per-subject (B03f) vs per-clip (B19) supervision scope is load-bearing on top of B03 drop. Load-bearing: if this wins, B03f reverses.

### Phase 3 (P1 / P2)

- `R-frame` — K=250 frame-aligned (v1 lock; tests pooled vs frame).
- `R-MSE` — Smooth-L1 → MSE.
- `R-cosine-vs-smoothL1` — cosine vs Smooth-L1 head-to-head.
- `R-rate-5Hz`, `R-rate-20Hz` — half / 2× Goldstein rate.
- `R-event-locked` (P2) — MFA-aligned variable-width buckets.
- `R-no-warmup` — skip Stage 3b-warmup.
- `R-warmup-long` — warmup at ~20% of P3 budget vs the ~10% default (and a short ~3-5% falsifier) so the adapter-freeze-duration axis is bracketed by distinct points.
- `R-frozen-throughout` — skip 3b-unfreeze (BLIP-2 pattern, pure adapter).
- `R-PMA-K` — K=250 PMA over time vs K=50 triangular pool.
- `R-PMA-frozen-random` — PMA never trained (vs trained P2+P3).

### Phase 4 (P1 / P2)

- `R-pool-then-probe` — DIVER-1-style mean-over-T_p → 256-dim → linear.
- `R-no-time-pool` — explicit name for the headline (flatten T_p·d → linear).
- `R-flatten-with-parcels` — skip parcel collapse, flatten (320 × T_p × d).
- `R-p4-clip-2s` (B20 v4) — recover P4 probe capacity from 1024 → 2048 dim by using 2s clips (T_p=8). Tests whether v4's probe-capacity drop (3840 → 1024 vs v3) costs accuracy on probe-capacity-bound tasks.
- `X2` — per-task PMA queries (k = n_tasks) vs shared k=1.

### Contingency (NOT default)

- L2-SP regularizer on Stage A vs P2 checkpoint, λ=1e-4. Only if 3b-unfreeze shows Stage-A drift.

---

## 9. Provenance

Canonical memos (consult for full rationale; this doc is the implementation contract):

- `memory/project_v14_b03_mask_lock_2026_05_25.md` — **canonical B03 mask-discipline lock (2026-05-25 PM)**, wins on conflict above B19 + B21 + B22 for: shaft-mask substitution (DROP via key_padding_mask, NOT learnable `[MASK]` token), patch-mask substitution (paradigm B = DROP + 2-block predictor; **warm-started P1→P2 per PM revert**, NOT re-init), teacher-side asymmetry (EMA teacher sees FULL electrodes AND FULL patches in P2; B03d), supervision-set semantics (`parcels_supervised[subject]` per-subject, NOT `valid_parcels[clip]` per-clip; B03f). Loss-side supervision gates restricted to `parcels_supervised[subject]`: L_mid_slot / L_post_frame / L_post_utterance PMA. Slot-bank regularizer scope (PM-reverted from earlier AM amendment): DKoleo / DKoleo@M3 / reactive Gram / parcel-ID F1 monitor operate on all 320 slots per B21 default (cross-subject `LearnableParcelEmbed` reachability + DINOv2/DINOv3 full-bank precedent override the supervised-only restriction). Includes TST-MASK-001/002/003/004/005/006 unit-test specs + MON-MASK-001/002/003/004 monitoring hooks (MON-MASK-002 bounded-ratio orphan/visible MSE ∈ [0.7, 1.5]; MON-MASK-004 subject-ID linear-probe leakage canary). Compute impact: cross-attn cost drops ~40% on P2 student forward (shaft-mask drop); patch-mask drop saves ~12% of per-electrode SA P1 compute. Param cost: +~0.2M for predictor (P1 trained, warm-started into P2, discarded at P2→P3 boundary). No HB02 re-cost.
- `memory/project_v14_b22_collapse_prevention_dense_features_2026_05_25.md` — **canonical B22 dense-features amendment (2026-05-25; cool-down arm reverted same-day PM)**, wins on conflict above B19 + B21 for: P1+P2 loss term count (4 → 5, adding `L_mid_slot@LN_mid(M3)` at weight 1.0), forward-path M3 head split (new dedicated LN_mid + loss_mid_head), reactive DKoleo@M3 trigger conditions (mirrors B21 reactive-Gram pattern, off default, weight 0.05). Phase-2 schedule UNCHANGED from B19/B21 baseline (originally-scoped end-of-P2 cool-down dropped on coherence review; see canonical memo Decision provenance item 9).
- `memory/project_v14_collapse_prevention_lock_2026_05_25.md` — **canonical collapse-prevention package (B21 2026-05-25)**, wins on conflict above B19 for: latent init policy (identity-anchored, not free trunc-normal), M4 head-side normalization (dedicated LN per loss head), M4 slot-level regularizer (DKoleo), P1+P2 monitoring contract (diagnostic monitor every 10k steps), reactive Gram anchoring trigger conditions, drop-SWEC fallback.
- `memory/project_v14_v4_invisible_frontend_lock_2026_05_24.md` — **canonical v4 invisible front-end lock (B20 2026-05-24)**, wins on conflict above the v3 amendment for: A1 patch embed (Conv2d (3, 2)), token-block attention (joint not factorized), STFT hop (256 / 8 Hz), per-patch freq embedding (10 vectors), Phase-3 frame-rate revision, paper claims (5 → 3).
- `memory/project_v14_loss_design_lock_2026_05_24.md` — **canonical loss-design (B19 2026-05-24)**, wins on conflict for the loss section (except where B21 amends: head-side LN, DKoleo term).
- `memory/project_v14_arch_post_v3_amendment_2026_05_19.md` — **canonical architecture (v4 state)**.
- `memory/project_v14_imindbench_multistft_pivot_2026_05_22.md` — **wins on conflict** above the v3 amendment for: Multi-STFT front-end, Whisper-L8 single-teacher, phase-asymmetric readout, valid-bin mask spec, Phase-3 preflight, UFO scale assignment.
- `memory/project_v14_three_phase_staged_recipe_2026_05_18.md` — 3-phase recipe with two amendment blocks (5/23 SSL clip-length + 5/23 B01 v3 optimizer).
- `memory/project_v14_cross_subject_pretraining_data_strategy_2026_05_22.md` — corpus diet (P1 mix, P2 sEEG-only, AJILE12 audit).
- `memory/project_v14_arch_revision_2026_05_19_v3.md` — parent v3 lock (superseded by post-v3 amendment).

Closed blocker entries (`docs/neuroprobe/v14_blockers.md`):

BIG:
- **B03 / B03b / B03c / B03d / B03f** (2026-05-25 PM, with same-day PM Tier-1 red-team revisions) — mask-discipline lock. Resolves the long-open Cluster-J B03 (mask-token mechanism) + 4 new bundled sub-blockers from the same user request. **B03**: shaft-mask substitution = DROP (ORed into C_MAX key_padding_mask consumed by cross-attn ❺a/❺b); learnable `[MASK]` token rejected, paradigm B drop chosen for symmetry with patch-mask + V-JEPA / Brain-JEPA precedent. Cross-attn cost drops ~40% on P2 student forward. **B03b**: latent self-attn ❻ key_padding_mask = `~slot_mask(parcels_supervised[subject])` in ALL phases (already implemented in code; documented as contract). **B03c**: per-electrode patch mask uses paradigm B (drop + 2-block predictor, ~0.2M params, hidden=128, heads=4) in P1 AND P2; **predictor warm-started from P1 checkpoint at P2 step 0 per PM revert** (no JEPA-family precedent for inter-phase predictor discard; re-init coincides with bias-OFF→ON flip, gradient noise risk through ❺ and ❻); predictor discarded at P2→P3 boundary; sister `R-p2-predictor-reinit` retains the re-init alternative as falsifier. **B03d**: EMA teacher receives FULL electrodes AND FULL patches in P2 (no shaft mask, no patch mask on teacher) — asymmetry IS the JEPA prediction signal. Sister `R-p2-teacher-shaft-mask-symmetric` PROMOTED to P0 (was P1) after Tier-1 red-team flagged mean-parcel-collapse degenerate solution under asymmetric teacher; MON-MASK-002 bounded-ratio orphan/visible MSE ∈ [0.7, 1.5] (was direction-only `> 1.5`) is the pre-registered detector. **B03f**: `parcels_supervised[subject]` (per-subject DK-coverage set, computed ONCE at extractor time, invariant to within-clip masks) replaces B19's `valid_parcels[clip]` (per-clip, recomputed post-mask) for slot-level supervision gates (L_mid_slot / L_post_frame / L_post_utterance PMA). B19 semantic was JEPA-inverted (excluded the prediction targets). Tier-1 red-team flagged subject-shape leakage risk (per-subject-deterministic gate encodes subject ID in M4 attention pattern → corrupts CSubject AUROC gate metric); MON-MASK-004 subject-ID linear-probe canary added as pre-registered detector + `R-per-clip-pre-mask` (third option: `parcels_covered[clip ∩ session]` pre-mask) added as P0 sister mitigation. **Slot-bank regularizer scope (PM-reverted from earlier AM amendment)**: DKoleo / DKoleo@M3 / reactive Gram / parcel-ID F1 monitor operate on all 320 slots per B21 default, NOT restricted to `parcels_supervised[subject]`. Reasons for revert: DINOv2 KoLeo +8.3pp Oxford-M / DINOv3 Gram precedents are over full feature banks; cross-subject training would never reach `LearnableParcelEmbed[p]` for parcels never in any subject's supervised set → identity-init geometry contract (B21 §A) broken at cohort level. Compute impact: cross-attn ~40% saving on P2 student + ~12% saving on per-electrode SA P1; +~0.2M predictor params (discarded after P2 / P3 respectively). No change to {d, N, L, steps, batch, electrode count, F bins, STFT windows} → **no HB02 re-cost**. Cascades into §1 (forward path key_padding_mask spec + mask-discipline matrix subsection), §3 (Phase-1 5-term loss scope correction + Masking paradigm B note + diagnostic monitor parcel-ID F1 all-320 revert + reactive Gram anchoring all-320 operand + SWEC fallback rename + L_DKoleo / L_DKoleo@M3 all-320 revert + L_mid_slot inclusion in `L_total_post_trigger`), §4 (Phase-2 5-term loss scope correction + Masking section + paradigm B description + B03d teacher full-electrode note + shaft-mask realization via key_padding_mask + B03f `parcels_supervised[subject]` as supervision gate + Mask token mechanism B03 ✅ resolution + Predictor warm-start clarification), §5 (P3 PMA softmax + latent SA gating), §6 (P4 PMA softmax + latent SA gating), §7 (phase-table P2 predictor warm-start note), §8 (B03 mask-discipline P0 must-run including `R-p2-teacher-shaft-mask-symmetric` + `R-per-clip-pre-mask` + P1 sister section including `R-p2-predictor-reinit`). Canonical: `memory/project_v14_b03_mask_lock_2026_05_25.md`. Closes the load-bearing B03 entry previously listed under "Still open and load-bearing for pre-P1 dispatch" Cluster J.
- **B22-dense-features** (2026-05-25 AM lock; cool-down arm reverted same-day PM) — two-component amendment on B21 (dense-features signal at M3 + reactive M3-side slot-diversity regularizer). (1) **M3 third-level supervision**: new loss term `L_mid_slot = MSE(loss_mid_head(LN_mid(M3_student)), LN_mid_T(M3_teacher_avg))` over `(p ∈ parcels_supervised[subject], t)` (per B03f amendment 2026-05-25 PM; was `valid_parcels[clip]` at AM lock) divided by `|parcels_supervised[subject]|·T·d`, weight 1.0. M3 = latent stack post-cross-attn-1 / pre-self-attn-0 (parcel-routing checkpoint). Adds LN_mid (student, ~512 params) + LN_mid_T (teacher, EMA mirror) + loss_mid_head (1-layer Linear(d, d), ~66k params). loss_mid_head is on the loss path only — the latent stack receives RAW M3. Operates in P1 + P2. V-JEPA 2.1 §2.3.2 + Table 12 precedent (42 → 43.9 ADE20k mIoU, last-layer ≈ best-of-4 under deep SS). (2) **Reactive DKoleo@M3**: OFF default, weight 0.05, triggers on (M3 cos>0.7 sustained 50k) OR (M3 parcel-ID F1<0.4 sustained 50k). Mirrors B21 reactive-Gram pattern. Diagnostic monitor extended: pairwise cosine + parcel-ID F1 + cross-attn entropy now mirrored at M3 and M4 (where B21 only had M4). **Reverted**: an end-of-P2 cool-down (clip 1s → 2s, linear LR 3e-4 → 1e-6 decay, final ~8% of P2) was scoped as Arm 2 at AM lock and dropped on PM coherence review — P3 retrains at 5s clips for ~18k unfreeze steps, washing out any P2-tail adaptation before P4; the cited V-JEPA-2.1 §2.3.5 precedent is direction-mismatched (V-JEPA cools UP to higher-res eval; v14 would cool DOWN to lower-T_p eval, with Whisper supervision density dropping 5× if applied at P3 tail instead). The only coherent place a v14 cool-down can land — the P3 tail — is preserved as P1 contingency sister `R-p3-tail-cooldown-1s` (dispatch only if Path-A P4 baseline shows large 5s SSL → 1s eval drop). Compute impact: ~+5% P1 step-time over B21; P2 wall-clock unchanged from B19 baseline; no change to {d, N, L, steps, batch, electrode count, F bins, STFT windows} → **no HB02 re-cost** (slack-log only). Param impact: ~+1k (LN_mid) + ~66k (loss_mid_head, discardable at eval). Cascades into §1 (params header + forward path M3 head split + M3 supervision paragraph), §3 (Phase-1 5-term loss + EMA teacher LN_mid_T mirror + diagnostic monitor M3 extension + reactive DKoleo@M3 arm + output checkpoint contents), §4 (Phase-2 5-term loss + trainable params + output checkpoint contents — **no schedule change**), §7 (phase table P1/P2 trainable column), §8 (B22 P0 + P1 sister cells), §9 (this entry + canonical memo). Canonical: `memory/project_v14_b22_collapse_prevention_dense_features_2026_05_25.md`.
- **B21-collapse-prevention** (2026-05-25) — six-component collapse-prevention package against B19's Phase-1 bias-off regime (320 free latents + MSE-to-EMA + zeroed anatomy bias = zero-precedent collapse risk). (A) Identity-anchored latent init `LearnableParcelEmbed[p] + LearnableSubSlotEmbed[s] + ε` replaces single-tensor trunc-normal for the 320 slots (~22k extra params). (B) DKoleo on M4 slot means, weight 0.1, P1+P2. (C) Dedicated LayerNorm per loss head at M4 divergence (LN_frame / LN_utt, ~1k extra params, all phases except P4). (D) Diagnostic monitor every 10k steps (slot pairwise cosine + cross-attn entropy + PMA entropy + parcel-ID linear-probe F1). (E) Reactive Gram anchoring (`||X_S·X_S^T − X_G·X_G^T||_F²`, weight 0.1) off by default; triggers on cos>0.7 sustained 50k, OR L_post_frame plateau + L_post_utterance decreasing, OR parcel-ID F1 < 0.4 sustained 50k. Gram teacher = student snapshot at trigger time, refreshed every 10k. (F) Drop-SWEC fallback (last-resort, 224h D+BT only with bias ON). Precedents: DINOv3 § 4 (Loss of patch-level consistency + Gram anchoring + dedicated LN), DINOv2 KoLeo (+8.3pp Oxford-M), NuCLR (POYO+ unanchored embedding F1=0.3521 worst). Compute impact <0.5% expected, <2% worst-case; no change to {d, N, L, steps, batch, electrode count, F bins, STFT windows} → **no HB02 re-cost**. Cascades into §1 (latent init replacement + init policy + forward path M4 head split + "Why identity-anchored latents" paragraph), §3 (Phase-1 4-term loss + EMA teacher LN mirror + diagnostic monitor + reactive Gram anchoring + fallback + output checkpoint contents), §4 (Phase-2 4-term loss + LN continuation + diagnostic re-arm + Gram re-arm + trainable params), §8 (P0 collapse-prevention sisters + P1 conditional sisters). Canonical: `memory/project_v14_collapse_prevention_lock_2026_05_25.md`.
- **B20-v4-frontend** (2026-05-24) — v4 invisible front-end lock: Conv2d (3, 2) patches @ A1, hop=256/8 Hz, joint token-block attention (was factorized t-then-f), per-patch freq embed (10 vectors, was 30 per-bin), single 50% mask (multi-mask dropped from roster entirely), F-single-STFT promoted P1→P0 must-run, F-CQT/F-Morlet stay P1 sisters. Frame-rate bug fix (v3 claimed 14.7 Hz at hop=128, actual was 16 Hz; v4 uses hop=256 → 8 Hz). Paper claims collapse 5→3 (front-end as engineering plumbing, not novelty); Whisper-L8 citation chain fix (Goldstein → Antonello/Shimizu; Goldstein uses *medium* L4 not large-v3 L8). Cascades into §1 (forward path + A1 + ❷ joint + factorization paragraphs), §2 (preprocessing recipe hop / frame rate + valid F-patch table), §3 (loss mask grid on 10 × T_p), §4 (P2 mask layering + shaft-block time-block span at T_p=20), §5 (Phase-3 v14 side T_p=20 @ 8 Hz + preflight rate widened to {5, 8, 10, 20} + triangular pool spec revision flag + Whisper citation), §6 (P4 T_p=4, flatten=1024, R-p4-clip-2s sister), §8 (sister roster: P0 promotions + F-patch + R-p4-clip-2s). Canonical: `memory/project_v14_v4_invisible_frontend_lock_2026_05_24.md`. HB02 re-cost triggered per [[feedback_recipe_amendments_need_compute_recost_2026_05_23]] (hop/patch/attention/freq-embed changes all hit the trigger).
- **B19** (2026-05-24) — Phase-1/Phase-2 loss-design lock: 3 unified bootstrap mask-prediction losses (L_pre_frame @ M2 + L_post_frame @ M4 + 1.0·L_post_utterance @ M4-PMA), Phase-1 cross-attn-ON with anatomy bias OFF (full ❶❷❺❻ stack), valid_parcels[clip] mask, per-electrode patch mask ≤ 50%, PMA-k=1 over parcels trained by Loss 3 + reused at P3/P4, +17% P1 step-time cost re-cost per feedback rule. Cascades into §1 (Phase-1 cross-attn-ON note + PMA training timeline amendment), §2 (SWEC anatomy assert amendment), §3 (loss + mask + EMA teacher + output), §4 (loss + mask layering + trainable + predictor + output), §7 (phase-table trainable param counts), §8 (BiL-Loss-* sister cells). Canonical: `memory/project_v14_loss_design_lock_2026_05_24.md`.
- **B01** (2026-05-23, 4+4 agent walkthrough) — optimizer / LR / schedule v3 lock (§7).
- **B02** (2026-05-23, 3rd re-lock after 4×4-agent SOTA audit) — cross-corpus α=0.5 hierarchical (XLS-R/MMS speech + DINOv3 vision dual-precedent) sampler over **exact-precomputed** valid-bin-electrode-hours + row-mean-then-batch-mean reduction + **TorchData `StatefulDataLoader`** (replaces custom `StatefulSampler`+WRS) + **canonical fixed locality sharding** (replaces redraw-per-epoch) + AJILE12 k0–k20 valid-bin mask bug-fix (§2, §3). Confidence: 50/50 macro split (the one component with no theoretical anchor) settled empirically via new `R-sampler-40-60` + `R-sampler-60-40` sisters.
- **B07 / ARG03** (2026-05-23) — PMA k=1 query training timeline: P2 + P3 trained, P4 frozen (§1).
- **B08** (2026-05-23, 4-agent SOTA audit — audio-FM / vision-MAE / biosignal-FM / EMA-K) — Phase-1 mask Option-B negotiated median: 2D inverse-block on (t, f) grid, keep-block 5×6 (~340ms × ⅓oct), mask 60–65%, M≈8–12; original frame-1D / keep-3 / 80% retained as load-bearing sister `R-p1-frame-1d-keep-3-mask-80` (§3).
- **B11** (2026-05-23, 5-agent unanimous JEPA-vs-data2vec crowd vote; amended PM on Ben review) — EMA teacher layer averaging: **Phase-1 K=6 (all 6 token-block layers post N=4→6 amendment), Phase-2 K=6 (all 6 latent-stack layers)**, both with per-layer instance-norm; framework clarification: v14 IS JEPA architecture with data2vec-recipe target enrichment (EAT/SSLAM-precedented hybrid). Amended-PM correction: prior "Phase-2 top-K=4 drop bottom-2" was an unilateral Claude pick — no evidence the bottom-2 latent-stack layers should be dropped; they sit right after cross-attn @ position 0 carrying freshest parcel-routed signal (§3, §4).
- **ARG04 / IE09 / IE14** (2026-05-23) — SSL clip-length 5 s, eval 1 s (§3 / §6).

MEDIUM:
- **M06** (2026-05-23) — Phase-2 sampler uniform-per-subject (§4).

**2026-05-23 26-item lock-and-delete batch (4-agent merge audit: SOTA / code-engineering / v14-consistency / red-team).** Following items locked into recipe (this doc) and deleted from `v14_blockers.md` to keep the blockers doc focused on load-bearing open questions. Per-item one-liners landed in §1 / §2 / §4 / §5 / §6 of this doc; defaults trace to PyTorch/SOTA convention or are already in §7's B01-v3 lock.

LOCK (10 — body addition + delete from blockers): M04 (Smooth-L1 β=1.0; §5), M09 (PMA query init trunc-normal std=0.02; §1), M12 (no registers / no [CLS] / no dropout; §1), M16 (sequential per-task probes; §6), M21 (SWEC anatomy assert; §2), M22 (LayerNorm; F-RMSNorm sister; §1), M26 (factorized order time-then-parcel; §1), S05 (LN γ=1 β=0; §1), S07 (RoPE per-block; §1), S08 (RoPE per-head; §1).

REFINE (8 — body addition with adjustment + delete from blockers): B10 (Path A/B optimizer, distinct val/test splits; §6), M01 (UFO frame stale per recipe §3; deleted as already-fixed), M11 (zero dropout in SSL; P4 probe-side 0.1 dropout sister; §1), M13 (P3b-warmup folds into §7 phase table — constant LR not cosine), M14 (Stage A frozen via requires_grad_(False)+eval() before first forward; §4), S04 (uniform trunc-normal std=0.02 across all Linear/MLP; §1), S09 (Path B 2-layer MLP, hidden=d, GeLU, no dropout, same B10 optim; §6), IE02 (RoPE max_seq_len=128 + slice + unit test; §1).

REDUNDANT (8 — already in recipe via B01 lock or §7; delete from blockers, no addition needed): M03 (EMA momentum span full phase, §7 line 300), M15 (P1=400k/P2=40k step counts, §7 table), M24 (d=256 default, §1), M25 (N=4/L=6/M=4 defaults, §1), S01 (bf16, §7), S02 (DDP only at 13M, §7), S03 (grad_clip=1.0, §7), S11 (freq embed dim=d=256, §1).

EX (external-precedent appendix):
- **EX03** (2026-05-23, Brain-JEPA Table 6 + Figure 6 paper-verified; clarified PM: shaft NOT parcel; **K-cap revised twice on 2026-05-27**: AM K=3 → K ≤ 2 fraction-based; PM K ≤ 2 → K=1) — Phase-2 mask: **SHAFT-block** multi-block, mask unit is the physical sEEG depth electrode (8–16 contacts on one needle), not BNA parcel — contacts on the same shaft are signal-correlated by adjacency; parcels too sparse (1–2 electrodes per subject). **K = 1 default with safety floor**: `K = 1 if N_shafts ≥ 2 else 0`, ~12.5% effective rate on BT (1/N_shafts per clip), ~1–2 s time-blocks, block α only (shaft-extent 0.45–0.60 × time-extent 0.15–0.30). **Revised 2026-05-27 PM from the same-day K ≤ 2 intermediate (held ~5h) and the original K=3 default**: EX03 was a literature-precedent transfer from Brain-JEPA's fMRI HCP-Aging configuration, never empirically validated on v14 biosignal. (i) K=3 stacked with patch 50% reached 70% combined mask, drifting out of the biosignal-FM ~40–60% band (Brant 40–50%, LaBraM 50%, DIVER-1 50%, REVE 55%, Laya 60%) into V-JEPA / MAE vision territory on a noisier modality. (ii) The same-day intermediate K ≤ 2 fraction-based default landed at 62.5% combined (37.5% visible), still on the biosignal-FM upper edge, and inherited the "fraction theatre" criticism (the `min(2, ⌈0.25·N⌉)` cap fires at N ≥ 5, so for the real cohort the formula collapsed to "K=2 with K=1 floor for ≤ 4-shaft outliers"). (iii) K=1 lands at ~56% combined (~44% visible) — centered in biosignal-FM band, monotonic in N_shafts, closes the N=1 catastrophic-drop hole the K ≤ 2 formula had, and Brain-JEPA's Table 6 "tube" lesion variant IS K=1 single-shaft × full-time so it's a documented Brain-JEPA configuration not a v14 invention. Per-clip shaft-orphan supervision is sparser (~0.6–1.25 supervised parcels orphan per clip on BT vs ~1.25–2.5 at K=2), but cumulative across 440k SSL steps the signal is plenty and sparser orphans reduce mean-parcel-collapse risk. Original commit's "parcel/shaft" ambiguity resolved to shaft + `R-p2-parcel-mask` sister to settle empirically. Original Bernoulli 30% per-electrode demoted to sister `R-p2-random-electrode-mask`; K=2 promoted to P0 falsifier `R-shaft-K2`; K=3 retained as P1 falsifier `R-shaft-K3-mixed-3block`; 6× efficiency claim ratified across HCP-Aging Age/Sex + ADNI NC/MCI (§4) — claim is on Brain-JEPA's modality, not transferred to v14.
- **EX09** (2026-05-23) — Symmetric valid-bin teacher mask: EMA teacher receives byte-identical per-corpus valid-bin mask as student (input fill 0 + key-padding −∞); post-forward defensive zero-out at invalid bins; loss supervision uses `ssl_mask & valid_bin_mask` intersection; runtime assert in debug builds (§3, §4).

Still open and load-bearing for pre-P1 dispatch (`v14_blockers_closing_report.md` top-30):

- **Cluster A — code scaffolding**: NT01 (`Experiment` class shape), NT02 (`Data` class for 4-phase corpus), DP03 (`SWECStudy` + `DCohortStudy` + `AJILE12Study` NeuralFetch classes).
- **Cluster B — data-loader contract**: DP01 (variable-T RoPE + variable-C collate); DP02 (sampler spec ✅ locked by B02; only implementation open).
- **Cluster C — Phase-1 training contract**: ✅ closed 2026-05-23 (B08, B11, EX03, EX09 — Bundle 4 walkthrough).
- **Cluster D — compute fit**: ✅ HB01 closed 2026-05-23 PM (Multi-STFT cache → `/work/ht203/cache/multi_stft/`, ~5–8 TB fp16, regenerate via `MapInfra` job every ≤75 days, node-local NVMe stage-in). HB02 (Ada-5000 32GB GPU-hour estimate) still open.
- **Cluster E — pre-dispatch verification**: TST01 (pytest marker for BIG/MEDIUM), TST05 (Phase-1 loss NaN detector), TST10 (DCC dispatch pre-flight).
- **Cluster F — schedule reality**: TIME01 (wall-clock conversion via M0), TIME04 (critical-path bottleneck), TIME07 (Jul 4 fork-gate undefined), TIME11 (writing cadence unanchored).
- **Cluster H — leakage**: BP20 (subject-overlap audit P1 corpora vs Neuroprobe eval).
- **Cluster I — interp suite**: VIS13 (statistical power + multiple-comparison correction).
- **Cluster J — pre-Phase-2 BIG**: ✅ B03 closed 2026-05-25 PM (see B03/b/c/d/f entry above), B04 (λ partial settle via B01 v3, teacher-sharing open), B09 (latent-stack parcel-SA bias), ✅ CQ12 closed 2026-05-23 PM (C_MAX=384, covers D=366/AJILE≈200/BT=256/SWEC=128; runtime `ValueError` in `dk_support.py`, `view.py`, `valid_mask.py` if any subject exceeds), RT10 (Phase-boundary checkpoint `strict=False` silent-skip risk), TST03 (P1↔P2 strict-mode compatibility test), IM11 (high-bin under-training cascade — severity upgraded 2026-05-23 after AJILE12 k0–k20 fix: bins k22–k29 trained only by D-cohort + BT ≈ 22.3% of P1 share).
- **Pre-Phase-3**: B05 (triangular-pool exact spec), B06 (preflight protocol details).

### Update protocol

When a blocker resolves:
1. Mark ✅ + Decision line + date in `docs/neuroprobe/v14_blockers.md`.
2. Cascade to the canonical memo for provenance.
3. **Edit the corresponding section here.** Don't add an amendment block — edit the body. This doc stays flat.
4. Add an entry to the §9 "Closed blocker entries" list above.

Engineering reads §1–§8. §9 is the provenance trail.
