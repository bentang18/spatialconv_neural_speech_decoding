# Neuroprobe Cross-Subject Hillclimb — Plan

*Live plan doc for the Neuroprobe cross-subject side-quest. Rewritten 2026-05-20 from the stale 2026-04-24 first-pass draft (which described single-loss D-SigLIP SSL, a 0.56 target, BNA parcel embeddings, and a Stage 0–4 program — all superseded). In-flight jobs and blockers: `MEMORY.md §Status (live)`. Naming: "Stage N" in this doc = hillclimb stage; elsewhere in the repo "Stage N" = PS-program stage.*

**AMENDED 2026-06-01 (B36 — TOP OF STACK, wins on conflict)** by `memory/project_v14_b36_perparcel_pool_structured_jepa_2026_06_01.md`. Encoder + SSL + phase redesign:
- **Throw away the Perceiver.** One-hot DK assignment collapses soft global routing to K independent **hard per-parcel attention pools** (block-diagonal mask — query k reads only parcel-k electrodes). This DELETES the soft `log(support+ε)` cross-attn bias, the `λ_anat` gate, the B28 anatomy-bias warmup, and the M3 routing tap. Then **inter-parcel self-attn** over the K=80 parcel tokens (the thesis core). Mechanically = the existing cross-attn with the bias hardened to block-diagonal.
- **SSL = paradigm-B parcel×time structured JEPA** (replaces the B31 inert 2-term self-distill). Visible-only encoder + a **separate narrow transformer predictor (3 blocks @ d=128, 4 heads, ~0.6M, discarded at deploy)** predicts masked (parcel×time) M4 latents from an **EMA teacher on full input**; **L1 on the post-`encoder_ln` M4** (canonical V-JEPA target normalization — drops the separate `LN_frame` loss head). Mask = **30% of covered (parcel×time) blocks with the masked parcels' electrodes DROPPED UPSTREAM** (leakage-free; the front-end never sees masked cells). `latent_valid` → **three-way** (visible=`covered&~masked` / target=`covered&masked` / teacher=`covered`).
- **Phases RE-STAGED** (reverses B29 joint default): **P1** front-end only @ M2, all corpora (SWEC can only help the front-end, undiluted) → **P2** pool + inter-parcel encoder + predictor @ M4, anatomy corpora only, front-end @ **LR/10**, no M2-aux → **P3** Whisper anchor, SEPARATE phase (B33 unchanged) → **P4** frozen-PMA→mean→linear (B35 unchanged).
- **Predictor = paradigm B** because closest task analog (Brain-JEPA region×time) uses it + frozen-probe cleanliness (MAE: a dedicated predictor buys +8pt linear-probe; BaRISTA's MLP head is licensed by *fine-tuning*, which we don't do) + collapse-safety (Tian 2021). Depth = **P0 sweep {2,3,4}**, RankMe-monitored from step 0.
- **Mandatory ablations**: `R-no-bottleneck` (BaRISTA-shaped, no compression — if the bottleneck can't beat it, v14 is the FM-anchor paper) + `R-no-whisper` (separates anatomy-bottleneck transfer from FM-anchor transfer). Sisters `R-paradigm-a-mlp` / `R-joint-ssl` / `R-whole-region-mask` / `R-cross-time-predictor` / `R-mask-ratio-{15,45}` / `R-m4-slots` / `R-l2-loss`.
- **Leaves intact**: B33 (P3 project-up), B35 (P4), DK atlas, Whisper all-layer-mean teacher, B27 (pure-L1 + EMA τ=0.99925 fixed + teacher full-input), B32 (no input-aug). The Architecture / SSL-recipe / corpus prose below is **pre-B36 provenance** wherever it describes Perceiver latents, soft `log(support+ε)` routing, the 5-loss / 2-loss surface, or the joint P1+P2 phase — the B36 banner supersedes those.

**AMENDED 2026-05-27 PM-late & 2026-05-28** by [[project_v14_b29_joint_default_2026_05_27]] (Items 1–15) and [[project_v14_moe_ffn_audit_2026_05_28]] (Item 14):
- **Phase structure**: P1+P2 collapse to single joint SSL phase as DEFAULT (`R-keep-phase-split` P0 falsifier). P3 Whisper-L8 distillation on BT unchanged.
- **Corpus**: SWEC + AJILE12 + D-cohort + BT under α=0.3 temperature sampling. AJILE12 reincluded (`R-drop-ajile12` P1 falsifies).
- **Conditioning embeds**: `subtype_embed {sEEG-depth, ECoG}` 512p + `ref_embed {shaftCAR, bipolar, Laplacian}` 768p, both additive at input + reused in cross-attn K/V, default ON.
- **Loss gating (Item 12)**: DROP `parcels_supervised` gating — L_mid_slot + L_post_frame fire on all 80 slots for all clips regardless of supervision.
- **Slot count (Item 13)**: **M=1 default**, 80 slots = 80 DK parcels × 1 (was M=4 = 320 slots). SubSlotEmbed dropped. `R-m4-slots` P0 falsifier. **The §Architecture line below referencing M=4 / 320 latents reflects the 5/25 lock and is now provenance**; the implementation contract is M=1 / 80 slots per B29 Item 13.
- **MoE-FFN (Item 14)**: dense FFN preserved across all 6 SA blocks; sparse MoE deferred v15 future-work. `R-moe-ffn-soft-4` P2 if-budget only (Soft MoE per Puigcerver 2024). 4-agent unanimous audit ([[project_v14_moe_ffn_audit_2026_05_28]]).
- **d_model (Item 15)**: d=256 stays default. `R-d-bump-384` PROMOTED P1 → P0 must-run sister with **≥0.02 AUROC promotion gate** at Cell-0 BT-Lite scale (becomes v15 default if it wins).
- **~14.235M params unchanged** (cross-attn count was already collapsed 2 → 1 by B28).

The text below preserves the 2026-05-25 lock chain wording for provenance; B29 + MoE-FFN audit memos are the implementation contract.

Background and rulebook: Zahorodnii et al. 2509.21671v2 + `insight-neuro/neuroprobe` (pinned `c7b955b`). Benchmark technical reference: `docs/references/neuroprobe_benchmark.md`. Running ablation menu: `docs/neuroprobe/ablations.md`. Project memory: `memory/project_neuroprobe_cross_subject_hillclimb_2026_04_22.md`.

## Thesis

**(B36 2026-06-01)** v14 = **a fixed anatomy-indexed bottleneck + inter-parcel dynamics + a cross-modal speech-model anchor for cross-patient transfer.** Concretely: a **hard per-parcel attention pool** over BrainTreebank Desikan-Killiany one-hot assignment (each electrode → exactly one DK parcel; query k reads only parcel-k electrodes) bottlenecks heterogeneous electrode layouts into K=80 parcel tokens; **inter-parcel self-attn** models cross-region dynamics; **parcel×time structured JEPA** (paradigm B, separate predictor, EMA teacher) pretrains it; and a **Whisper all-layer-mean cross-modal anchor** (P3 distillation, zero per-subject params) supplies the cross-patient invariance. This beats PopT cross-subject by ≥0.05 AUROC at ≤30M params on a 9-subject BrainTreebank cohort. The bottleneck must beat a BaRISTA-shaped no-bottleneck control (`R-no-bottleneck`); the Whisper anchor is the clean novelty (BaRISTA has no FM analog).

*(Pre-B36 framing — "anatomy-tagged Perceiver IO + soft `log(support+ε)` parcel-routing cross-attention" — is retired: the Perceiver was thrown away. One-hot DK assignment made the soft global routing degenerate to per-parcel pooling, so we hardened it and dropped the bias machinery. This shrank the "Perceiver moat" deliberately — the moat is now the bottleneck + inter-parcel dynamics + FM anchor, grounded in reality rather than theoretical elegance.)*

The cross-subject leaderboard is precisely v14's claim. Every submitted model treats electrodes as patient-specific indices; v14 treats them as anatomy-pooled parcel tokens. Direct evidence the axis is real: BrainBERT-trained (0.522) < BrainBERT-untrained (0.527) cross-subject — current iEEG SSL learns subject-specific structure that hurts transfer. The cross-session linear floor (0.651) sits far above the cross-subject linear floor (0.539): most of the gap is the subject-shift axis, which is where atlas anchoring should land.

## Shared frame (frozen)

These commitments are the scaffolding. Change one and you are writing a different plan.

1. **Cross-subject is the headline; cross-session multiclass is the submit lane.** No within-session submission — DIVER-1's 0.678 with large-corpus pretraining is not our game.
2. **Anatomy is shared at the parcel level.** DK-first: **(B36) hard per-parcel attention pool** over BT-DK one-hot assignment (each electrode → its single DK parcel; query reads only that parcel's electrodes). Replaces the pre-B36 soft `log(support+ε)` cross-attn bias. BNA-soft support is a P1 sister, gated on Christopher Wang's fsaverage mapping. MNI Fourier PE dropped 2026-05-19.
3. **Zero learnable per-subject parameters in the deployment forward path.** Parcel-routing is the only subject-conditioning mechanism at inference. Defended empirically by the §6e 3-arm S1-A/B/C ablation, not asserted. See `memory/project_v14_spike_vs_field_potential_per_subject_defense_2026_05_16.md`.
4. **Pretraining is load-bearing.** The thesis lives on whether staged atlas-anchored SSL transfers where raw-voltage SSL does not — not on cold-start.
5. **Leaderboard-parity cells are not architecture-selection defaults.** S2/trial-4 CrossSubject and the 120-electrode Lite cap are leaderboard-parity cells. Pooled multi-source CrossSubject multiclass is the scientific generalization default.
6. **No legacy reuse.** Old loaders, old training loops, old sbatch tooling stay in git history. Active path is the NeuroAI substrate (`Study → Events DataFrame → Transforms/Chain → Segmenter → Dataset → NeuralTrain Experiment → Exca`).

## Architecture (v14, ~15.1M params)

Canonical: `memory/project_v14_arch_post_v3_amendment_2026_05_19.md` (the v4 state — wins on conflict) + the 2026-05-22 amendment in `memory/project_v14_imindbench_multistft_pivot_2026_05_22.md` + 2026-05-25 B19/B21/B22/B03 lock chain. Factorized latent stack; `d=256`, heads=8, N=6, L=6, M=4, ~15.1M params (post-B19/B21/B22 SSL stack + identity-anchored init + dense-feature supervision; within ≤30M cap). No open architectural blockers — B1 (SWEC sampling rate) closed by the 2026-05-19 SWEC audit.

```
Preproc (BT 2048 Hz): HPF 0.5 Hz → comb @ mains_hz (per-corpus) → MNE-LOF flag → **ref draw** {shaftCAR, bipolar, Laplacian} per-clip (5/27 PM ref-aug; SWEC degenerates to global-CAR) → slice
  → Multi-STFT (iMINDBench Appendix E + v14 high-band extension):
       STFT_low  Nperseg=1024 (~2–40 Hz),  STFT_mid Nperseg=512 (~20–148 Hz),
       STFT_hi   Nperseg=256  (~80–813 Hz, extended past iMINDBench's 248 Hz cap)
       common hop = 256 samples @ 2048 Hz → 8 Hz frame rate (B20 v4 lock 2026-05-24)
  → triangular ⅓-octave filterbank (30 log-SPACED bins, mel-style edges)
       k0–k14 from STFT_low, k15–k21 from STFT_mid, k22–k29 from STFT_hi
  → Nv14 robust-z per (electrode, freq-bin, session)
       (5/25 swap: post-filterbank `log(energy+ε)` step dropped; raw filterbank magnitude is the default.
        F-log-amplitude sister re-enables `log` via `apply_log=True`.)
       invalid bins (per-corpus valid-bin mask) skipped → input-filled 0
  → (C, F=30, T = clip_length × 8)
❶  A1 Conv2d (3,2) patches + per-patch freq embed (10 vec) + ref_embed (3, d) additive (5/27 PM ref-aug)
❷  Token block × N=6   per electrode, JOINT t·f attention (B20 v4 lock 2026-05-24; supersedes N=4 factorized): RoPE on time only · pre-norm · GeLU · MLP 4×
❺  Cross-attn   pools (electrode, freq) → parcels, strict 1:1 per time-step
      320 free Perceiver-IO latents (K=80 DK parcels × M=4 slots), bias = log(support+ε)
      2 cross-attn layers @ stack positions {0, 3}        → latents (320, T, d)
❻  Latent stack × L=6   factorized (time × parcel): time SA (RoPE) · parcel SA · MLP
❼  Readout (phase-asymmetric — see ablations.md §6 readout cells):
      parcel-collapse: PMA k=1 frozen seed → (T, d)
      Phase-3 SSL:    triangular-window pool over T → T_r* buckets @ rate r* vs Whisper layer k* (Goldstein search-range anchors (acoustic-phonetic L8) inform the grid only; **B06 PM lock 2026-05-25**: r* = 8 Hz, k* = L8 with `R-rate-{5,10,16}Hz` falsifiers)
      Phase-4 probe:  NO time pool — flatten (T, d) → T·d → per-task Linear (iMINDBench-parity)
```

**Latents keep a time axis** `(320, T, d)` — v3's time-collapsed latents broke Phase-3 SSL (the cross-modal target is a 50-token syllable-rate sequence) and killed cross-parcel temporal dynamics. Multi-STFT is the patch embedding (replaces single-STFT 2026-05-22 — resolves the post-v3 low-frequency-floor TODO honestly; preserves the 30-bin log-SPACED ⅓-octave filterbank on top, value-axis is raw magnitude post-5/25). Latent init is a free `nn.Parameter` (parcel identity carried by the cross-attn bias, not the latent vector).

## SSL recipe (3-phase staged)

Canonical: `memory/project_v14_three_phase_staged_recipe_2026_05_18.md` + `memory/project_v14_loss_design_lock_2026_05_24.md` (B19 — wins on conflict for the loss section). Replaces the v3-lock joint-from-step-1 recipe. Rationale: BLIP-2/LLaVA precedent (big SSL backbones + small adapter on small paired data), DIVER-1 evidence (frozen-encoder linear-probe beats fine-tune at iEEG scale), and ~220 h paired iEEG-audio being orders of magnitude below CLIP-from-scratch's regime.

P1 and P2 share a single objective: **five unified bootstrap mask-prediction losses** (per B19/B21/B22 locks 2026-05-24..25) computed in one student forward — `L_pre_frame` @ M2 (per-electrode per-patch) + `L_mid_slot` @ LN_mid(M3) (post cross-attn-1, parcels_supervised[subject] gate, B22 dense feature supervision) + `L_post_frame` @ LN_frame(M4) + `1.0·L_post_utterance` @ LN_utt(M4) PMA-k=1 over parcels (clip-level (d,) vector, no predictor — EAT-UFO precedent) + `0.1·L_DKoleo` @ M4 (slot-level uniformity over all 320 slots, B21 collapse-prevention). All loss-side gates use `parcels_supervised[subject]` per-subject set (B03f); the 320-slot regularizers (L_DKoleo) operate unrestricted. EMA teacher all-layer-averaged with per-layer instance-norm (EAT §3.1). Three dedicated LayerNorms — LN_mid (post cross-attn-1), LN_frame, LN_utt — at M3 and M4 divergence. MSE loss. Joint from step 1, no schedule. The PMA-k=1 query trained here is the same shared seed reused at P3 (cross-modal) and P4 (probe, frozen). Phases differ in {corpus, anatomy bias, mask discipline, predictor lifecycle}, not in loss structure.

- **Phase 1 — pretrain full ❶❷❺❻ stack with anatomy bias OFF (~15M trainable).** 8.2k h unpaired iEEG. Cross-attn `log(support[e,p]+ε)` term zeroed (QK content-only routing); SWEC's lack of anatomy is no longer a routing problem since bias is globally zeroed in P1. Mask = per-electrode independent inverse-block patch mask (keep-block ≈ 5×6, ≤ 50% per electrode HARD cap — every electrode retains ≥ 50% keep so Loss 2 has cross-attn input from every electrode). No whole-shaft, no whole-electrode in P1. PMA softmax over parcels falls back to uniform-over-all-320 for SWEC samples (no per-clip anatomy filter); broadband corpora use `valid_parcels[clip]`.
- **Phase 2 — pretrain full stack with anatomy bias ON + shaft-block electrode-mask layer (~15M trainable, continues from P1 checkpoint).** 224 h sEEG-only (D 85 + BT 9). Bias-on throughout (B19 retires the prior "Stage A frozen during P2" lock — the bias-off pretrain already happened in P1, so no anatomy-blind warmup is needed). Mask = per-electrode patch mask (same as P1) + shaft-block mask on top (Brain-JEPA EX03 spec, ~40% effective rate, K=3 mixed-extent blocks). `valid_parcels[clip]` is recomputed from the UNMASKED electrode set — shaft-masked electrodes' parcels are removed, so Loss 2 + Loss 3 supervise reconstruction of missing parcels' latents from context only. Predictor for Loss 1 re-initialized fresh at P2 step 0.
- **Phase 3 — Whisper cross-modal distillation (single-teacher).** Stage A + B frozen-then-unfrozen; ~220 h paired iEEG-audio. Target: Whisper-large-v3 layer **k\*** (empirical, see preflight) mean-pool to rate **r\*** Hz → (T_r*, 1280) → linear-project → (T_r*, 256) per clip. v14 side: PMA k=1 frozen seed → (T, d) → parameterless triangular-window pool to T_r* buckets → Smooth-L1. Sub-staged: 3b-warmup adapter-only (~3–5% budget) → 3b-unfreeze slow-LR on Stage A (~95%). DINOv3 dropped 2026-05-22 (no visual-cortex equivalent in clinical sEEG) — Stage 3a gating MLP deleted. **No Goldstein default** ([[feedback_no_default_ecog_to_seeg_transfer_2026_05_24]]): Goldstein 2025's (L8, 10 Hz) was ECoG-derived and informs the search range only — the ECoG↔sEEG modality gap (surface dense vs depth sparse, mixed white/grey, different per-band SNR, different conduction lag) makes inheriting it unsafe. **Phase-3 preflight (dual, ~1 GPU-h each)**: (a) **brain-fit ridge** — sweep Whisper layers k ∈ {L4, L6, L8, L10, L12, L16, L20} × rate ∈ {5, 10, 20} Hz × lag sweep, on one BT-Lite subject; per-electrode ridge predicting v14 preproc output; cross-validated R²/r averaged across electrodes picks (k, r, lag). (b) **task-fit ceiling** — same layer/rate grid, mean-pool → LogReg on the gate tasks across all three eval splits (within-session + CrossSession + CrossSubject), `ablations.md §L.7.{B0,C0,S-layer,S-combine}`. Validates that the distillation target itself clears existing iEEG-model SOTA (so distillation has headroom) and picks the layer by task discriminability. Convergent picks ⇒ lock k\* / r\*; divergent ⇒ resolve by a small Phase-4 probe at the top-2 candidates, not by either preflight alone. Both share the same Whisper-feature cache.
- **Phase 4 — downstream eval.** Headline readout protocol: PMA k=1 frozen → (T, d) → **flatten (T·d) → per-task Linear**. Matches iMINDBench's preprocessing-track logistic baseline (which flattens (C × F × T)). Path A frozen-everything is the headline; Path B 2-layer MLP; Path C light task fine-tune. Report Path A1 (Stage A frozen) and A2 (Stage A unfrozen) as a paired ablation surface. R-pool-then-probe (DIVER-1-style mean-over-T) is a sister, not the default.

EMA + StopGrad anti-collapse default. Recipe sister cells (R-joint, R-staged, R-no-phase-3, R-frame, R-no-warmup …) in `ablations.md §5`.

## Evaluation

### Splits

- **CrossSession multiclass — submit lane.** Official Neuroprobe submit score (v2 Supp Table 13).
- **Pooled multi-source CrossSubject multiclass — scientific generalization default.** Architecture selection happens here. Train all allowed source subjects/sessions, test held-out (`ablations.md` cell D.14). First-to-report — v2 has no CrossSubject multiclass table.
- **S2/trial-4 CrossSubject binary — leaderboard-parity only.** Numbers reported, not used for architecture selection.
- **Anti-controls (mandatory before any freeze):** shifted-window, within-session label shuffle, subject/session-ID nuisance probe, stimulus-overlap flag.

### Submission gate (dual-prong, SOTA-at-submission, anchored 2026-05-15)

| Prong | Gate | Anchor |
|---|---|---|
| CrossSession multiclass | **≥ 0.667** | Linear-Lap+spec 0.617 (v2 Supp T13) + 0.05 |
| CrossSubject binary AUROC | **≥ 0.628** | CNN-Lap+spec 0.578 (v2 leaderboard) + 0.05 |

Plus: ≥ 4 tasks pre-baked, ≤ 30M params, K-fold / chronological splits. Stretch ≥ 0.70. Criterion "beat SOTA-at-submission by ≥0.05" is self-updating if the leaderboard moves. **DIVER-1's 0.678 and iMINDBench's Multi-STFT Logistic 0.663 are both within-session only** — neither is a CrossSubject/CrossSession gate ceiling; the earlier "auto-bump to 0.728" claim is retracted. Beating finetuned PopT (CSession-MC 0.546) is subsumed by clearing 0.667. (iMINDBench reshapes the *preprocessing track* — Multi-STFT is v14's new front-end default; it does not change the gate.)

Few-shot prototype eval (zero training, BrainWave protocol — `ablations.md §6h` cell X1) runs on all pretrained checkpoints as a complement to the linear probe: it removes the probe's own fitted capacity from the measurement and isolates the backbone representation. Lead with CrossSubject.

## Stages

- **Stage 0 — preprocessing + protocol freeze. CLOSED 2026-05-14.** Reproduced the upstream linear baseline within SEM and froze every reusable pipeline primitive. Frozen contracts: L.1 normalization (N1 `train_set_fixed`), L.2 reference×view (R4×I2 `shaftLap × stft_abs` for the linear hillclimb), L.3 filtering (F0 no-op), L.4 anchor robustness, shaft/depth geometry, Tier-C CrossSubject parity. v14's own preproc recipe (`memory/project_v14_preproc_recipe_2026_05_12.md` + 5/25 amendment `project_v14_stft_abs_default_2026_05_25.md`): shaftCAR + Multi-STFT-abs + Nv14 robust-z, HPF/notch re-added. Detail: `docs/neuroprobe/stage_0.md` + `reports/neuroprobe_stage0_*`.
- **Stage 1 — v14 cold-start (architecture, no SSL). IN FLIGHT.** v14 per-electrode-token backbone, DK-routed cross-attn, no pretraining. `v14-q1-word-events` 7-commit stack merged into main (push gated). Nano smoke GREEN on DCC. Next gate: Lite cell rerun after the BTWordEvents class-imbalance fix (`memory/project_btwordevents_split_class_imbalance_bug_2026_05_15.md`). Architectural roster: `ablations.md §3, §6d, §6g, §6h`.
- **Stage 2 — 3-phase staged SSL pretraining + downstream eval.** Deferred until Stage 1 lands. The 3-phase recipe above; eval via Phase 4. Loss/schedule/corpus ablations: `ablations.md §4–6`.
- **Submit.** Fork `insight-neuro/neuroprobe`, write per-task JSONs for the CrossSubject and CrossSession splits + `metadata.json` + `ATTESTATION.txt` + `PUBLICATION.bib`, open the PR. CI is format-only.

## Pretraining corpus

Phase-asymmetric corpus per modality. Phase 1 (per-electrode SSL) is modality-invariant in spectral structure → full unpaired iEEG. Phase 2 (cross-electrode parcel routing) is sensitive to surface vs depth electrode topology → sEEG-only diet. Phase 3 uses only paired iEEG-audio. **Locked 2026-05-22.**

- **Phase 1 — unpaired iEEG (~8.2k h)** — SWEC 6,672 h + AJILE12 1,280 h + internal D-cohort 180 h + BT 43.5 h. (SWEC corrected from headline 9,328 h / 68 folders to 6,672 h / 50 unique subjects — 18 are duplicate re-exports; `memory/reference_swec_ieeg_dataset_audit_2026_05_19.md`.) SWEC is band-limited 0.5–120 Hz and has no electrode anatomy, so it trains only filterbank bins k0–k21 and only the per-electrode token stack — never the parcel-routed cross-attn. **Sampler (B02 ✅ 3rd re-lock 2026-05-23 after 4×4-agent SOTA audit): α=0.5 hierarchical over EXACT-precomputed valid-bin-electrode-hours; DUAL precedent = XLS-R §4.1 + MMS §4.2 speech-side + DINOv3 §3 vision-side (10/90 macro split structurally identical).** Two groups: SWEC (anatomy-blind/bandlimited) vs broadband (AJILE12 + D-cohort + BT). Macro uniform 50/50 (hard SWEC cap), within-broadband α=0.5 over vb-eh. Audit-estimated shares **SWEC 50.0% / AJILE12 ~27.7% / D-cohort ~15.0% / BT ~7.3%** — recomputed at sampler-build from exact per-session totals `Σ_session (session_hours × session_n_electrodes × |valid_bins[corpus]|)` per Llama 3 / OLMo 2 / Megatron `BlendedDataset` precedent; the corpus-median × n_sessions approximation in earlier locks is dropped. **AJILE12 valid-bin mask k0–k20** (AJILE12 is 500 Hz / 0.5–200 Hz bandpass — bug fix, surfaced by 2nd-round audit). **Loss reduction = row-mean-then-batch-mean** (MAE/V-JEPA) with fp32 accumulator. **DataLoader stack**: WRS wrapped by `torchdata.stateful_dataloader.StatefulDataLoader` (TorchTitan §3.3) — native state_dict / load_state_dict for mid-epoch DCC scavenger resume, per-rank deterministic worker RNG, `persistent_workers=True`. Replaces prior custom StatefulSampler + WRS plan; fills the gap NeuralSet's map-style `SegmentDataset` leaves open. **Page-cache mitigation**: canonical fixed locality sharding (subject→rank assignment static across epochs via `hash(subject) % W == r`); within-shard shuffle via WRS draw per epoch. Replaces earlier "redraw subjects per epoch" plan (3rd-audit LLM lens caught the locality inversion). **B01-coupling: ✅ satisfied** (P1 batch=1024 inside XLS-R's validated regime). Sisters: `R-sampler-alpha03`, `R-sampler-sqrth`, `R-sampler-pure-h`, `R-sampler-uniform`, `R-sampler-broadband-uniform`, `R-sampler-seeg-only`, **`R-sampler-40-60` + `R-sampler-60-40`** (new — empirically settles the macro split, the only B02 component with no theoretical anchor across 4-lens 2026 SOTA review). Full audit: `docs/neuroprobe/v14_blockers.md` §B02.
- **Phase 2 — sEEG-only unpaired (~224 h)** — D-cohort 180 h + BT 43.5 h. **AJILE12 dropped from Phase 2** per per-subject electrode audit (Peterson 2022 Table 2: 89.7% surface ECoG / 10.3% depth aggregate; 7 of 12 subjects pure surface) — dominantly an ECoG corpus, cross-electrode topology mismatches the BT sEEG eval target. SWEC has no anatomy so cannot feed Phase 2 in either modality. **Fallback if Phase 2 underfits**: source chronic sEEG recordings from Cogan-lab patients (believed to exist outside the lab; separate future acquisition). **Sampler (M06 ✅ 2026-05-23): uniform-per-subject** across 19 sEEG subjects, NOT pooled-by-hours. Per-row weight = `1 / (N_subjects_global × clips_per_subject[subject_id])`. Shares **D-cohort 52.6% / BT 47.4%** (10:9 subject ratio). Rationale: Phase-2 unit-of-learning is the per-subject routing config (anatomy + coverage pattern), so subject-uniform sampling is the cross-subject-generalization-aligned default; pooled-by-hours would let one heavy-hour D-cohort subject dominate the parcel-routing gradient. **Sub-staging Phase 2 into 2a (bias-off, full corpus) + 2b (bias-on, sEEG-only) was 4-agent audited 2026-05-23 and rejected as default** — QK miscalibration at the bias-on switch (~100× attention shift overnight), AJILE12 topology contamination survives bias-off (QK-distribution problem not bias problem), 15-blocker engineering cascade. Preserved as P2 sister `R-p2a-bias-off-pretrain` (Lite-cell scope, ~5–10k-step SWEC bias-off warmup before P2b). Full rationale: `docs/neuroprobe/v14_blockers.md` §M06 + §M06-aux.
- **Phase 3 — paired iEEG-audio (~220 h)** — BT 9 subjects (~40 h) + internal D-cohort (~180 h). Single-teacher Whisper-large-v3 only, layer k\* preflight-picked (no ECoG default — see §SSL recipe Phase 3).
- **Downstream cohort — 9 BT subjects {1,2,3,4,6,7,8,9,10}.** S5 dropped (frontal lesion). ~40–50 h finetuning.

uECoG (PS + lexical) stays out of the Neuroprobe pretrain corpus entirely — surface↔depth cross-sensor transfer is a separate claim for the main v14 paper.

## Ablations

Single source of truth is `docs/neuroprobe/ablations.md`. Stage-0 L/A/D blocks, Stage-1 architectural roster (AC1–6, V1–13, F0–F30, S1-A/B/C), Stage-2 loss/schedule/corpus cells, and §6h post-competitor-review cells (X1 few-shot prototype eval, X2 per-task PMA queries, X3 learned-correction routing bias, X4 SSL direction A/B). Do not duplicate cell specs here — cite that doc.

## Infrastructure

Active package layout under `src/speech_decoding/` is documented in `CLAUDE.md §Code Structure` (`atlas/`, `extractors/`, `studies/braintreebank/`, `experiments/`, `models/`). Training dispatch is NeuralTrain/Exca via `TaskInfra + run_grid`, not bespoke sbatch scripts. Reorg blueprint: `docs/neuroprobe/repo_reorg_plan.md`. Adapter spec: `docs/neuroprobe/neuralset_integration_plan.md`. DCC helpers: `scripts/dcc/{sync,dispatch,status,rerun-failed}`.

## Open questions

**Full living blocker list**: `docs/neuroprobe/v14_blockers.md` (244 enumerated gaps as of 2026-05-23 — 50 first-pass + 44 second-pass [IE/AB/EV/EX] + 42 third-pass [DP/PT/NT/IM/CR] + 52 fourth-pass [12 PF / 9 HB / 7 RT / 9 BP / 15 CQ] + 56 fifth-pass [11 TST testing-CI / 16 VIS interp-outputs / 10 ARG cross-memo-coherence / 11 TIME schedule-feasibility / 8 DOC methods-doc] after 19-agent fan-out; recursive convergence-driven audit, per-wave yield W2=8.8 W3=10.4 W4=11.2/lens — **increasing**, not converged below 5/lens floor; discovery phase closed 2026-05-23). **Closing report**: `docs/neuroprobe/v14_blockers_closing_report.md` — top-30 pre-Phase-1 critical-path blockers + 6-bundle walkthrough order. The list below carries the few "open" items above the blocker-doc level.

1. **BNA-soft vs DK-hard support** — DK one-hot is the routing default; BNA-soft is a P1 sister gated on Chris Wang's fsaverage mapping. `memory/project_v14_dk_first_pass_2026_05_13.md`.
2. **Per-corpus valid-bin mask correctness** — fixed 30-bin tensor + per-corpus mask is the spec; load-bearing detail is the L_recon target mask (invalid bins excluded from loss + EMA-teacher sees the same mask, otherwise model learns "those bins = 0" trivially on SWEC). See `memory/project_v14_imindbench_multistft_pivot_2026_05_22.md` §4. Front-end is iMINDBench Multi-STFT (3 windows, 8 Hz hop per B20 v4 lock 2026-05-24) — STFT-param question is closed.
3. **Phase-2 mask rate** — 0.30 default, J1 sister sweep settles it.
4. **Within-parcel ordinal depth** — out of the first pass; single `F-depth-bias` P1 sister decides by a BT-Lite number.
5. **Phase-3 (Whisper-layer, rate)** — empirical pick on sEEG, no ECoG default ([[feedback_no_default_ecog_to_seeg_transfer_2026_05_24]]). Goldstein's (L8, 10 Hz) is the search-range anchor, not the default. Dual ~1 GPU-h preflights on BT-Lite settle it before Phase-3 lock: brain-fit ridge (layer × rate × lag, layers widened to {L4, L6, L8, L10, L12, L16, L20}) + task-fit ceiling (same grid on the gate tasks across all 3 splits; `ablations.md §L.7.{B0,C0,S-layer,S-combine}`).
6. **Stimulus-overlap per-task audit** — upper bound clean (max 0.450 < 0.50 kill); per-task DCC refinement pending.
7. **Ref-aug sister-first protocol** (5/27 PM lock, `[[project_v14_ref_aug_input_distribution_lock_2026_05_27]]`) — BT-Lite paired run (ref-aug ON vs OFF, ~5–10 H100-h) gates full P1 all-corpora rollout. 4 kill criteria: HG-patch 6–9 loss > 5%, MON-MASK-002 out of [0.7, 1.5], Monitor F1 dev > 0.1, MON-MASK-004 subject-ID F1 ↑ > 0.05. Pass → full P1 all-corpora with 3× cache expansion (+~18–30 TB on `/work/`); fail → defer ref-aug to post-paper, keep fixed shaftCAR.
8. **(B36) Predictor paradigm + depth** — default paradigm B (visible-only encoder + separate narrow transformer predictor, 3 blocks @ 128); depth is a P0 sweep {2,3,4} with `R-paradigm-a-mlp` (BaRISTA 5-layer FC) as the in-domain simplicity sister, settled by the frozen-probe number + RankMe collapse monitor. `memory/project_v14_b36_perparcel_pool_structured_jepa_2026_06_01.md` §5.
9. **(B36) Mandatory bottleneck-vs-FM-anchor controls** — `R-no-bottleneck` (BaRISTA-shaped, no compression) and `R-no-whisper` (JEPA-only, cross-subject track) are first-class, not optional: they decide whether the headline is "anatomy bottleneck" or "FM anchor." Run before any architecture freeze. B36 §9.

## Explicitly not doing

- Within-session submission. Only CrossSubject + CrossSession.
- DK-atlas region-averaging as the model (the baseline's `combine_regions`) — v14's **learned per-parcel attention pool + inter-parcel self-attn (B36)** is the principled upgrade (a fixed mean-over-electrodes is the degenerate special case the pool generalizes).
- Electrode selection beyond the hardcoded `NEUROPROBE_LITE_ELECTRODES` for parity cells.
- Any pretraining touching the 12 off-limits eval sessions.
