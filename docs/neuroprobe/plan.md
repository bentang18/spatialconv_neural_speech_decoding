# Neuroprobe Cross-Subject Hillclimb — Plan

*Live plan doc for the Neuroprobe cross-subject side-quest. Rewritten 2026-05-20 from the stale 2026-04-24 first-pass draft (which described single-loss D-SigLIP SSL, a 0.56 target, BNA parcel embeddings, and a Stage 0–4 program — all superseded). In-flight jobs and blockers: `MEMORY.md §Status (live)`. Naming: "Stage N" in this doc = hillclimb stage; elsewhere in the repo "Stage N" = PS-program stage.*

Background and rulebook: Zahorodnii et al. 2509.21671v2 + `insight-neuro/neuroprobe` (pinned `c7b955b`). Benchmark technical reference: `docs/references/neuroprobe_benchmark.md`. Running ablation menu: `docs/neuroprobe/ablations.md`. Project memory: `memory/project_neuroprobe_cross_subject_hillclimb_2026_04_22.md`.

## Thesis

An iEEG foundation model with anatomy-tagged Perceiver IO and soft parcel-routing cross-attention — `log(support+ε)` additive bias over BrainTreebank Desikan-Killiany one-hot support — pretrained with a 3-phase staged SSL recipe (factorized-t×f Stage A + electrode-mask Stage B + multi-teacher P3 distillation against Whisper-L8 + DINOv3), beats PopT cross-subject by ≥0.05 AUROC at ≤30M params on a 9-subject BrainTreebank cohort.

The cross-subject leaderboard is precisely v14's claim. Every submitted model treats electrodes as patient-specific indices; v14 treats them as anatomy-routed parcel queries. Direct evidence the axis is real: BrainBERT-trained (0.522) < BrainBERT-untrained (0.527) cross-subject — current iEEG SSL learns subject-specific structure that hurts transfer. The cross-session linear floor (0.651) sits far above the cross-subject linear floor (0.539): most of the gap is the subject-shift axis, which is where atlas anchoring should land.

## Shared frame (frozen)

These commitments are the scaffolding. Change one and you are writing a different plan.

1. **Cross-subject is the headline; cross-session multiclass is the submit lane.** No within-session submission — DIVER-1's 0.678 with large-corpus pretraining is not our game.
2. **Anatomy is shared at the parcel level.** DK-first routing: `log(support+ε)` cross-attn bias over BT-DK one-hot support. BNA-soft support is a P1 sister, gated on Christopher Wang's fsaverage mapping. MNI Fourier PE dropped 2026-05-19.
3. **Zero learnable per-subject parameters in the deployment forward path.** Parcel-routing is the only subject-conditioning mechanism at inference. Defended empirically by the §6e 3-arm S1-A/B/C ablation, not asserted. See `memory/project_v14_spike_vs_field_potential_per_subject_defense_2026_05_16.md`.
4. **Pretraining is load-bearing.** The thesis lives on whether staged atlas-anchored SSL transfers where raw-voltage SSL does not — not on cold-start.
5. **Leaderboard-parity cells are not architecture-selection defaults.** S2/trial-4 CrossSubject and the 120-electrode Lite cap are leaderboard-parity cells. Pooled multi-source CrossSubject multiclass is the scientific generalization default.
6. **No legacy reuse.** Old loaders, old training loops, old sbatch tooling stay in git history. Active path is the NeuroAI substrate (`Study → Events DataFrame → Transforms/Chain → Segmenter → Dataset → NeuralTrain Experiment → Exca`).

## Architecture (v14, ~13M params)

Canonical: `memory/project_v14_arch_post_v3_amendment_2026_05_19.md` (the v4 state — wins on conflict) + `memory/project_v14_arch_revision_2026_05_19_v3.md`. Factorized throughout; `d=256`, heads=8, ~13M params (within the 5–15M target, far below the ≤30M cap). No open architectural blockers — B1 (SWEC sampling rate) closed by the 2026-05-19 SWEC audit.

```
Preproc (BT 2048 Hz): HPF 0.5 Hz → comb @ 60 Hz → MNE-LOF flag → shaftCAR → slice
  → STFT → triangular ⅓-octave filterbank (30 log bins, mel-style edges)
  → log-power → Nv14 robust-z per (electrode, freq-bin, session)   → (C, F=30, T)
❶  A1 linear embed Linear(1→d) per (t,f) cell  +  flat categorical freq embedding (30 vectors)
❷  Token block × N=4   per electrode, factorized t × f: temporal SA (RoPE) · freq SA · MLP
❺  Cross-attn   pools (electrode, freq) → parcels, strict 1:1 per time-step
      320 free Perceiver-IO latents (K=80 DK parcels × M=4 slots), bias = log(support+ε)
      2 cross-attn layers @ stack positions {0, 3}        → latents (320, T, d)
❻  Latent stack × L=6   factorized (time × parcel): time SA (RoPE) · parcel SA · MLP
❼  PMA readout   k=1 (or k=n_tasks) query downstream;  k=50 @ 10 Hz for Phase-3
```

**Latents keep a time axis** `(320, T, d)` — v3's time-collapsed latents broke Phase-3 SSL (the cross-modal target is a 50-token syllable-rate sequence) and killed cross-parcel temporal dynamics. STFT is the patch embedding. Latent init is a free `nn.Parameter` (parcel identity carried by the cross-attn bias, not the latent vector).

## SSL recipe (3-phase staged)

Canonical: `memory/project_v14_three_phase_staged_recipe_2026_05_18.md`. Replaces the v3-lock joint-from-step-1 recipe. Rationale: BLIP-2/LLaVA precedent (big SSL backbones + small adapter on small paired data), DIVER-1 evidence (frozen-encoder linear-probe beats fine-tune at iEEG scale), and ~200 h paired iEEG-audio being orders of magnitude below CLIP-from-scratch's regime.

- **Phase 1 — Stage A pretrain.** Trains the per-electrode token stack (❶❷) on unpaired iEEG. `L_recon` Level B: 80% inverse-block (t,f) mask, EMA-teacher latent target (data2vec-2.0), MSE + UFO.
- **Phase 2 — Stage B pretrain.** Stage A frozen; trains the cross-attn + latent stack (❺❻) on the same unpaired corpus. `L_recon` Level A: 30% random electrode-mask (J1 sister sweep {0.25–0.50}), EMA-teacher latent prediction. Teaches cross-attn to pool electrodes into parcel latents under electrode dropout — directly the cross-subject-transfer capability.
- **Phase 3 — MTDP cross-modal distillation.** Stage A + B frozen-then-unfrozen; ~200 h paired iEEG-audio(-video). Targets: Whisper-large-v3 L8 + DINOv3, fused at K=50 buckets @ 10 Hz (syllable rate, Goldstein-2025). Smooth-L1. Sub-staged: 3a gating MLP → 3b-warmup adapter-only (~3–5% budget) → 3b-unfreeze slow-LR on Stage A (~95%).
- **Phase 4 — downstream eval.** Path A (headline): frozen Stage A+B+adapter + per-task linear probe — competes directly with DIVER-1's frozen linear probe. Path B: 2-layer MLP probe. Path C: light task fine-tune. Report Path A1 (Stage A frozen) and A2 (Stage A unfrozen) as a paired ablation surface.

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

Plus: ≥ 4 tasks pre-baked, ≤ 30M params, K-fold / chronological splits. Stretch ≥ 0.70. Criterion "beat SOTA-at-submission by ≥0.05" is self-updating if the leaderboard moves. **DIVER-1's 0.678 is within-session only** — not a CrossSubject/CrossSession gate ceiling; the earlier "auto-bump to 0.728" claim is retracted. Beating finetuned PopT (CSession-MC 0.546) is subsumed by clearing 0.667.

Few-shot prototype eval (zero training, BrainWave protocol — `ablations.md §6h` cell X1) runs on all pretrained checkpoints as a complement to the linear probe: it removes the probe's own fitted capacity from the measurement and isolates the backbone representation. Lead with CrossSubject.

## Stages

- **Stage 0 — preprocessing + protocol freeze. CLOSED 2026-05-14.** Reproduced the upstream linear baseline within SEM and froze every reusable pipeline primitive. Frozen contracts: L.1 normalization (N1 `train_set_fixed`), L.2 reference×view (R4×I2 `shaftLap × stft_abs` for the linear hillclimb), L.3 filtering (F0 no-op), L.4 anchor robustness, shaft/depth geometry, Tier-C CrossSubject parity. v14's own preproc recipe (`memory/project_v14_preproc_recipe_2026_05_12.md`): shaftCAR + log-STFT + Nv14 robust-z, HPF/notch re-added. Detail: `docs/neuroprobe/stage_0.md` + `reports/neuroprobe_stage0_*`.
- **Stage 1 — v14 cold-start (architecture, no SSL). IN FLIGHT.** v14 per-electrode-token backbone, DK-routed cross-attn, no pretraining. `v14-q1-word-events` 7-commit stack merged into main (push gated). Nano smoke GREEN on DCC. Next gate: Lite cell rerun after the BTWordEvents class-imbalance fix (`memory/project_btwordevents_split_class_imbalance_bug_2026_05_15.md`). Architectural roster: `ablations.md §3, §6d, §6g, §6h`.
- **Stage 2 — 3-phase staged SSL pretraining + downstream eval.** Deferred until Stage 1 lands. The 3-phase recipe above; eval via Phase 4. Loss/schedule/corpus ablations: `ablations.md §4–6`.
- **Submit.** Fork `insight-neuro/neuroprobe`, write per-task JSONs for the CrossSubject and CrossSession splits + `metadata.json` + `ATTESTATION.txt` + `PUBLICATION.bib`, open the PR. CI is format-only.

## Pretraining corpus

Same-modality-first. Phase 1 + 2 (unpaired SSL) use the full corpus; Phase 3 (paired distillation) uses only the paired subset.

- **Unpaired iEEG (~8.2k h)** — SWEC 6,672 h + AJILE12 1,280 h + internal D-cohort 180 h + BT 43.5 h + PS/lex ~7 h. (SWEC corrected from the headline 9,328 h / 68 folders to 6,672 h / 50 unique subjects — 18 are duplicate re-exports; `memory/reference_swec_ieeg_dataset_audit_2026_05_19.md`.) SWEC is band-limited 0.5–150 Hz and has no electrode anatomy, so it trains only filterbank bins k0–k21 and only the per-electrode token stack — never the parcel-routed cross-attn.
- **Paired (~200 h)** — BT 9 subjects (~40 h, iEEG + audio + video, the only multi-teacher corpus) + internal D-cohort (~160 h audio-only).
- **Downstream cohort — 9 BT subjects {1,2,3,4,6,7,8,9,10}.** S5 dropped (frontal lesion). ~40–50 h finetuning.

uECoG (PS + lexical) stays out of the Neuroprobe pretrain corpus — surface↔depth cross-sensor transfer is a separate claim for the main v14 paper.

## Ablations

Single source of truth is `docs/neuroprobe/ablations.md`. Stage-0 L/A/D blocks, Stage-1 architectural roster (AC1–6, V1–13, F0–F30, S1-A/B/C), Stage-2 loss/schedule/corpus cells, and §6h post-competitor-review cells (X1 few-shot prototype eval, X2 per-task PMA queries, X3 learned-correction routing bias, X4 SSL direction A/B). Do not duplicate cell specs here — cite that doc.

## Infrastructure

Active package layout under `src/speech_decoding/` is documented in `CLAUDE.md §Code Structure` (`atlas/`, `extractors/`, `studies/braintreebank/`, `experiments/`, `models/`). Training dispatch is NeuralTrain/Exca via `TaskInfra + run_grid`, not bespoke sbatch scripts. Reorg blueprint: `docs/neuroprobe/repo_reorg_plan.md`. Adapter spec: `docs/neuroprobe/neuralset_integration_plan.md`. DCC helpers: `scripts/dcc/{sync,dispatch,status,rerun-failed}`.

## Open questions

1. **BNA-soft vs DK-hard support** — DK one-hot is the routing default; BNA-soft is a P1 sister gated on Chris Wang's fsaverage mapping. `memory/project_v14_dk_first_pass_2026_05_13.md`.
2. **Phase-1 sample-rate handling** — corpus-conditioned bin set (SWEC trains k0–k21 only) is the working answer; STFT-param finalization still open.
3. **Phase-2 mask rate** — 0.30 default, J1 sister sweep settles it.
4. **Within-parcel ordinal depth** — out of the first pass; single `F-depth-bias` P1 sister decides by a BT-Lite number.
5. **Stimulus-overlap per-task audit** — upper bound clean (max 0.450 < 0.50 kill); per-task DCC refinement pending.

## Explicitly not doing

- Within-session submission. Only CrossSubject + CrossSession.
- DK-atlas region-averaging as the model (the baseline's `combine_regions`) — v14's routed cross-attn is the principled upgrade.
- Electrode selection beyond the hardcoded `NEUROPROBE_LITE_ELECTRODES` for parity cells.
- Any pretraining touching the 12 off-limits eval sessions.
