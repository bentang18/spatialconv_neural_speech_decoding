# Neuroprobe Stage 1 — v14 cold-start (skeleton)

*Drafted 2026-04-25. Provisional. Detail follows Stage-0 close-out (D-cell results + A4 sign-off determine atlas/prep/support choices).*

Strategy anchor: `docs/neuroprobe/plan.md`. Predecessor: `docs/neuroprobe/stage_0.md` (must close before Stage 1 begins).

## Objective

Train v14 from scratch on S2/trial-4 only — the official cross-subject train set. **No pretraining.** This is the architecture-as-cold-prior test.

## Success criterion (sharp)

**Cross-subject mean AUROC strictly > 0.539** — strictly greater than Linear Lap+spec, not equal.

Rationale: the BrainBERT/PopT/random-TF cluster sits at ~0.527; Linear Lap+spec at 0.539 is the ceiling for "alignment without learning" on this benchmark. v14's anatomical anchoring is *architectural* — parcel embeddings are part of the cold model, not learned from pretraining. To support the alignment-is-the-bottleneck thesis, cold-start v14 must break the 0.539 ceiling without any pretraining.

## Result decision tree

| Cold-start AUROC | Read | Stage-2 implication |
|---|---|---|
| **> 0.539** | Anatomical priors do alignment cold; thesis already supported pre-pretraining. | Stage 2 chases 0.56–0.58 to clear submission threshold. |
| **0.527–0.539** | Architecture provides nonlinear-feature lift but anchoring alone insufficient; SSL load-bearing. | Stage 2 becomes thesis-deciding. |
| **< 0.527** | Below the random-TF cluster ceiling — architecture below the noise floor. Structural bug. | Debug before continuing — likely loader contract, support cache, or `P_emb` wiring. |

## Empirical anchors from BarISTA (Oganesian et al. NeurIPS 2025) on BrainTreebank

BarISTA is the closest published architecture to v14, evaluated on BrainTreebank itself (10 sEEG subjects, 26 sessions, 29.2 h SSL pretraining at 2048 Hz raw — i.e. the same cohort/modality/scale we'd be running). Reference memory: `memory/reference_barista_oganesian_2025_12.md`. Architectural findings transfer to our Stage-1 design *where directly comparable*; absolute AUC numbers do not (their ~0.86 is binary sentence-onset / speech-production with their own split, not the Neuroprobe cross-subject leaderboard split S2/trial-4-only train).

| Choice | BarISTA finding on BT | Our position |
|---|---|---|
| Spatial encoding | Parcel-level (Destrieux 148) **>>** channel-level (LPI coords) by +8–10 pp AUC | Validates our `P_emb` thesis on this exact dataset. v14's soft Brainnetome ≈ their hard Destrieux, strictly more general (continuous over parcel subset). |
| Parcellation granularity | Parcel ≈ lobe (DK ~70) — finer doesn't help in their setting | Informs Stage-0 D.3b vs D.4 expectation (full BNA 246 unlikely to beat BT-Tier-1 by much). Doesn't change Stage-1 architecture. |
| Attention factorization | Combined spatiotemporal **>** factored by +0.8–2.2 pp on BT | Empirically validates v14's combined-attention default *specifically on BT*. (Contradicts Seegnificant's factored-wins on different data — task / dataset dependent.) |
| Temporal front-end | 5-layer dilated CNN on raw voltage **>** linear projection by +6–9 pp | **Novel question for v14**: we currently extract hand-engineered HG envelope at 200 Hz before tokenization, not learned CNN on raw 2048 Hz. Surfaced as Experiment #10 below. |
| Per-patient params | Zero per-patient params + parcel encoding → only ~2 pp held-out-subject degradation | Sets the held-out-subject benchmark v14 should match or beat. v14 has zero per-patient params by design (parcel embeddings are shared); the prediction is the same ~2 pp gap. |
| Width × depth | d=64, 12 layers, ~1 M params | Phase-1 default is d=32 (smaller). May under-fit BT. Surfaced as a width sweep below. |

**Caveat — atlas family is different.** Their "parcel ≈ lobe" was Destrieux (gyral) vs DK (lobar). Brainnetome (cytoarchitectonic / functional) is a different family they didn't test. So "BNA-246 vs BT-Tier-1 ≈ parity" remains a Stage-0 D.4 empirical question, not a BarISTA-settled finding.

**Caveat — task difficulty mismatch.** BarISTA's binary sentence-onset / speech-production are dramatically easier than the 15-task Neuroprobe panel (especially the linguistic / GPT-2-surprisal / POS tasks). The "zero per-patient params is enough" finding generalizes only if the harder linguistic tasks don't require per-patient calibration that BarISTA never tested.

## Frozen design commitments (revisit on Stage-0 close)

- **Tokenization**: per-electrode (B-1 path). Per-cell pooling unsupported with variable electrode counts cross-subject.
- **Spatial prior**: BNA `P_emb` attached per electrode via support vector from Stage-0 Block-C cache. Tier-1 = BT-derived from Stage-0 A0. *Empirically anchored*: BarISTA's parcel-level >> channel-level finding on BT (+8–10 pp AUC) is direct on-cohort validation of this choice.
- **Argmax-hard vs probabilistic support**: inherit from Stage-0 D.3a vs D.3b winner.
- **Temporal**: RoPE temporal-only.
- **Attention**: combined (channel × time). *Empirically anchored on BT*: BarISTA combined > factored by +0.8–2.2 pp.
- **Width**: `d=32` (Phase-1 default).
- **Decoder**: small linear per-task head; binary CE; val early-stopping on Neuroprobe val half.
- **Seeds**: 2 (expand to 5 if 2-seed gap > 0.005).
- **Preprocessing**: from Stage-0 Block-E loader (CAR + 70–150 Hz Gaussian filterbank + Hilbert envelope sum + exact 200 Hz via `resample_poly(125, 1280)` + recording-level median/MAD z-score per channel).
- **Augmentation**: channel-dropout, time-warp ±5%, mixup. All legal under the protocol (the train *set* is constrained, not augmentation produced from it) and underused on the leaderboard.

## Open questions (defer to results)

- **Latent bottleneck (Experiment #1)**: Perceiver-style cross-attention from electrode tokens into a small set of latents (anatomy-seeded one-per-parcel, or learned). Test if cold-start clears 0.539 but stalls below 0.55.
- **Spatial encoding beyond `P_emb` (Experiment #4)**: Fourier features on MNI coordinates, learned distance-bias attention (MV-BrainFM per-head Gaussian). Same trigger.
- **Fixed vs learned `P_emb` (Experiment #2)**: identity-init frozen, anatomy-init frozen, or learned. Test only if cold-start ≥ 0.539.
- **Per-subject embeddings (Experiment #3)**: degenerate at Stage 1 with N=1 train subject (single embedding learned, no contrast). Reopens at Stage 2 with multi-patient pretrain.
- **Shaft-aware attention (Experiment #9, new)**: depth contacts on the same shaft are physically adjacent (~3.3 mm intra-shaft) but can land in different BNA parcels (crossing GM/WM/another GM). v14's combined attention has no inductive bias for physical adjacency — it relies entirely on `P_emb` similarity. Add a shaft-aware attention mask using BT electrode name stems (`T1b1, T1b2, T1b3` share stem `T1b`) to bias attention toward same-shaft contacts. Test if cold-start sits in 0.527–0.539 (architecture lifts off the floor but doesn't break the alignment ceiling). Free metadata, cheap to implement, BT/D-cohort-specific (not generalizable to uECoG dense grids).
- **Dilated-CNN temporal front-end (Experiment #10, new)**: BarISTA on BT shows learned 5-layer dilated CNN on raw 2048 Hz voltage beats linear projection by +6–9 pp. v14 currently uses hand-engineered HG envelope at 200 Hz instead. Trade-off: HG is an informed prior (specific band well-validated for speech motor + STG); learned CNN may pick up signal HG discards (sub-70 Hz linguistic-relevant bands, theta-gamma coupling helpful for the linguistic Neuroprobe tasks). Test as an alternate input pipeline if cold-start clears 0.539 but stalls. Specifically: replace Block-E loader's filterbank+Hilbert+resample with raw-voltage 250 ms patches (512 samples @ 2048 Hz) → 5-layer dilated-CNN at d=64; everything downstream of the per-electrode-token contract unchanged.
- **Width × depth sweep (BarISTA d=64, 12 L vs v14 default d=32)**: BarISTA's empirical sweet spot on BT is d=64, 12 layers, ~1 M params. v14 Phase-1 default is d=32, depth 3 (smaller). For Stage-1 cold-start on BT specifically, d=32 may under-fit. Width sweep d=32 / 48 / 64 if d=32 cold-start sits below 0.539. Depth sweep 3 / 6 / 12 layered separately so we attribute lift correctly.
- **CAR vs Laplacian re-reference**: distribution-of-input choice, not strictly architectural. Default for Stage 1: CAR (parity with Cogan/PS pipeline). Run Laplacian (matches BT's winning baseline at 0.539) as one architecture-frozen ablation cell. Decision rule: Laplacian wins by ≥0.005 → architecture is robust to re-ref, use whichever wins. Laplacian wins by ≥0.02 → v14 isn't doing local denoising itself, sharper architectural finding worth lifting into Stage-2 (consider learnable per-shaft Laplacian as input layer).

## What Stage 1 explicitly does NOT do

- No pretraining (Stage 2 territory).
- No multi-task joint head (per-task linear is the protocol).
- No within-session or cross-session evaluation.
- No DK-atlas anchor.
- No electrode selection beyond the BT Lite cap.
- No labels other than the 15 official eval tasks (proxy / auxiliary labels are Stage-2 territory).
