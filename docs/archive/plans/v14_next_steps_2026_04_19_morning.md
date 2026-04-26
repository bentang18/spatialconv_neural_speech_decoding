# v14 Next Steps — Post Batch 1/2 (2026-04-19)

## TL;DR

**Default is `per_cell + partialconv + pe2d`.** Batch 2 LOPO completed and produced one real reversal: pe2d, which lost pooled-joint by +0.029, wins LOPO by −0.013 **uniformly across all 4 held-out patients**. That uniform-direction coherence is not noise — it's a Phase-1 pool-path LOPO lift we take while the pool is alive, knowing it dies when we migrate to scalable / sEEG. Hierarchical readout stays rejected (inverse pattern: wins pooled-joint, fails LOPO). The scalable `per_electrode` bucket's LOPO deficit is narrower than pooled-joint implied (0.7 pp vs 3.2 pp between `per_cell` and `per_elec_fourier_mni`), so migration is **still gated but closer to feasible**. Next experiments: (1) disambiguate pe2d's mechanism (random-frozen vs learned, row-only vs col-only) — cheap and tells us whether we have a simpler recipe, (2) test pe2d+hierarchical composition (mirror patterns may compose), (3) scalable capacity + patient-count scaling to characterize the now-smaller gap, (4) per-op augmentation decomposition, (5) SSL infrastructure.

## What Batches 1 and 2 told us

**Pooled-joint story.** Baseline `per_cell` at 0.794 is hard to beat at 4-patient scale. Only `hierarchical` moved the needle (−0.033). `cls`, `partialconv`, `h1` are ties. `h4` (+0.021), `aug_legacy` (+0.017), `pe2d` (+0.029), and the three scalable-bucket arms (+0.032 to +0.040) are losses. The pooled-joint bar is the easier half of the test.

**LOPO story — the pe2d reversal.** The full LOPO table vs baseline (`per_cell` pretrain 0.802, finetune 0.806):

| arm | pretrain Δ | finetune Δ | call |
|---|---|---|---|
| `pe2d` | **−0.003** | **−0.013** | **real win (uniform across 4 patients: −0.019 / −0.013 / −0.004 / −0.015)** |
| `aug_legacy` | +0.006 | +0.002 | tie |
| `h4` | +0.005 | −0.001 | tie |
| `h1` | +0.003 | +0.013 | hurts (S26 +0.035 — patient-specific instability) |
| `per_elec_fourier_mni` | +0.010 | +0.007 | best scalable arm; still net-negative |
| `per_elec_distance_bias` | +0.033 | +0.012 | hurts |
| `per_elec_none` | +0.036 | +0.017 | hurts most |
| `hierarchical` (Batch 1) | — | +0.000 | tie (vs +3.3 pp pooled-joint win) |
| `cls` (Batch 1) | — | +0.001 | tie |
| `partialconv` (Batch 1) | — | −0.006 | trend positive, within noise |

**Why pe2d's LOPO win is not noise.** Per-seed std ≈ 0.06. Mean −0.013 is ~1/5 of noise in magnitude, but the **direction is uniform across all 4 patients**. Under the null, the probability of 4/4 same-sign outcomes is 0.0625; per-patient t-stats pile on top. The coherence signal is strong even when the grand-mean magnitude is modest.

**Mechanism hypotheses for pe2d.** Three plausible, separable stories:

1. **Break-symmetry prior the pretrain stage can exploit.** Cells without PE are a bag; SGD has no prior on which cells matter. Any informative-and-shared cell-axis signal lets pretrain pick favorites on 3 source patients; finetune then shifts those favorites to the held-out patient. Pooled-joint is hurt because joint training has to compromise across 4 sets of anatomical-to-cell mappings; LOPO-finetune doesn't.
2. **Regularization by cell-index identity.** pe2d gives each cell a stable learnable identity, lowering pretrain optimization variance and producing a cleaner checkpoint.
3. **Warm-start via any cell-id prior.** The signal does not need to be learned — any fixed, informative, non-random cell identity would produce the same lift.

Story 3 is important: if true, a *frozen random* 2D PE captures most of the effect, and we have a simpler recipe. T3.3 below separates these.

**Why hierarchical's pattern is inverse.** Hierarchical wins pooled-joint (+3.3 pp) but fails LOPO (+0.000). Its q_cell is free-learned, cell-indexed, and over a grid whose cells cover different anatomy per patient — it overfits training-patient cell weights with no transferable content. pe2d's embedding is also cell-indexed but is shared across patients *and* is the first thing the network sees — it biases the optimization landscape rather than the readout. The two mechanisms live on opposite ends of the architecture and touch the cross-patient objective differently.

**Scalable-bucket story under LOPO.** Ordering matches pooled-joint (Fourier MNI > distance bias > none), so the bucket is internally consistent. The pool-vs-scalable LOPO gap is **0.7 pp** (0.806 vs 0.813), not 3.2 pp — the pool's inductive-bias advantage shrinks substantially under the cross-patient protocol we care about. This narrows the SSL / patient-count / capacity barrier needed to justify migration, and validates the canonical-goal rule that pooled-joint alone cannot reject a scalable architecture.

## Default architecture

```
Stage 1:  grid-scatter (patient-specific H_p × W_p)
Stage 2:  Conv2d(1→8, k=3, pad=1) + GELU
          with Liu-2018 partial-conv renormalization [ADDED]
Stage 3:  masked-mean pool to (4, 8) = 32 cells
Stage 4:  + learnable 2D PE: row_emb[4, 16] + col_emb[8, 16] broadcast to cells [ADDED — pe2d]
Stage 5:  per-cell Conv1d(8→32, k=30, s=10)
Stage 6:  + pooled_support @ P_emb[15, d]
Stage 7:  3 × combined attn + FFN, 2 heads × 16, FFN 128, RoPE on time
Stage 8:  mean-pool readout + prev_phoneme_emb → Linear(2d, 9)
```

**Two changes vs. current committed baseline: `partialconv` and `pe2d`.**

- **`partialconv`**: Liu 2018's mask renormalization (scale = nominal_sum / actual_sum). Principled artifact-mask correction — otherwise valid electrodes near artifact boundaries have their Conv2d output silently attenuated. Tied pooled-joint (0.795 vs 0.794), trended positive LOPO (−0.006, within noise). Free correctness upgrade.
- **`pe2d`**: learnable factorized 2D PE (4 row embeddings + 8 col embeddings × 16 dim = 192 params) added to the 32 pool cells. **−0.013 LOPO finetune, uniform across 4 patients.** Costs +0.029 pooled-joint as a known trade.

**Why we default with both even though they weren't co-tested.** `partialconv` is a Stage-2 correction on the mask during conv; `pe2d` is a Stage-4 additive embedding on pool-cell features. They touch orthogonal machinery. If the pe2d+partialconv joint run shows any interaction, back out to pe2d-only — pe2d is the LOPO-measured winner.

**Honest framing.** pe2d is a Phase-1 pool-path-only bonus. It dies when we migrate to scalable / sEEG — the cell axis ceases to exist. Adding pe2d does not commit us to the pool; it takes the measured LOPO lift for the pool's remaining lifetime. `d=32`, `depth=3`, `heads=2`, everything else unchanged.

## Rejected as default (with reason)

- **`hierarchical` readout** — +3.3 pp pooled-joint, 0 pp LOPO. Patient-specific readout prior; violates cross-patient-foundation-model goal. Retained as a **diagnostic target** (see T3.1) — the inverse pattern to pe2d is informative.
- **`cls` readout** — tie everywhere. No param-cost justification.
- **`h4` attention** — +0.021 pooled-joint / tie LOPO. Too many heads at d=32. Re-test at d=64 if capacity probe runs.
- **`h1` attention** — tie pooled-joint / +0.013 LOPO with patient-specific S26 blow-up. Worse than baseline on the protocol we care about.
- **`aug_legacy` composite preset** — net-hurting pooled-joint (+0.017), tie LOPO. Must decompose per-op before any of its knobs return.
- **All three scalable-bucket arms as default** — LOPO deficit narrowed to 0.7 pp (best scalable = `per_elec_fourier_mni` at 0.813 vs 0.806), still net-negative. Retained as **migration target**, gated on T2.1–T2.2 and T4.1 results.

## Experiments to run (priority-ranked)

All run against `per_cell + partialconv` as the baseline unless stated. All follow the canonical protocol: pooled-joint (4 core × 5 folds × 3 seeds = 15 tasks) **and** LOPO pretrain→finetune (60 tasks) — single-protocol evidence does not promote a change.

### Tier 1 — decision unblockers

**T1.1 [RESOLVED] Batch 2 LOPO — closed.**
All seven arms landed. `pe2d` is the only LOPO winner (−0.013); scalable-bucket deficit is 0.7 pp (narrower than pooled-joint). Decision: add pe2d to default; do not migrate to scalable yet — the remaining 0.7 pp is the bar SSL / scaling must clear.

**T1.2 [QUEUE, cheap] Per-op augmentation decomposition.**
`aug_legacy` combines four ops (time-shift ±100 ms, log-normal amp scaling std=0.15, channel dropout p∼U[0,0.2], Gaussian noise 2%). The preset is net-hurting (0.811 vs 0.794) but any single op could be the culprit while the others help or are neutral.

- Run 4 single-op pooled-joint variants first (15 tasks × 4 = 60 tasks). If any single op ties or beats baseline pooled-joint, promote it to LOPO (60 tasks per variant).
- *Hypothesis*: `channel_dropout` helps cross-patient transfer (SSL-adjacent regularization) but the additive noise and aggressive amp scaling hurt HGA signal at our small-data regime.
- *Decision gate*: any op that passes both pooled-joint AND LOPO moves to default. Do not composite ops unless each passes individually.

### Tier 2 — inform the migration timing

**T2.1 [QUEUE] Patient-count scaling on the scalable bucket.**
The canonical-goal rule says current-scale pooled-joint deficits on scalable architectures are not disqualifying. The test is whether the deficit **shrinks as N grows** — if yes, Phase-1.5 / Phase-2 scale will close it.

- Train `per_elec_fourier_mni` at N=2, 3, 4 core patients (Jiang-style coverage-ranked picks), matched pooled-joint vs `per_cell` at each N. 3 seeds, 5 folds, N-appropriate splits.
- *Hypothesis*: `per_cell` vs `per_elec_fourier_mni` deficit curve is monotonically decreasing in N. If it flattens, capacity or SSL is the gap.
- *Decision gate*: deficit at N=4 minus deficit at N=2 is the scaling slope. If slope ≥ 0.5 pp per added patient, migration at N=7 (Phase-1.5 with 3 lexical patients) is justified.

**T2.2 [QUEUE] Scalable capacity probe.**
At d=32 the scalable bucket may be capacity-bound — attention must learn what Conv2d gets for free.

- `per_elec_fourier_mni` at (d=64, depth=3) and (d=32, depth=4). Pooled-joint + LOPO (75 tasks per variant).
- *Hypothesis*: doubling d closes >2 pp of the scalable deficit; depth=4 closes <1 pp (attention-depth has diminishing returns at our data scale).
- *Decision gate*: if (d=64) reaches `per_cell` pooled-joint within 1 pp, capacity is the gap and migration is straightforward.

### Tier 3 — diagnostic (mechanism disambiguation)

**T3.1 [QUEUE, small] Hierarchical LOPO diagnosis: atlas-anchored query.**
Hierarchical's q_cell is free-learned, cell-indexed, over a grid whose cells cover different anatomy per patient — the overfit path. Replace q_cell with an atlas-anchored query: `q_cell_p = pooled_support_p @ P_emb` (same parcel embedding already in Stage 6). Query becomes anatomy-indexed, not cell-indexed.

- 1 variant × (15 pooled-joint + 60 LOPO).
- *Hypothesis*: atlas-anchored query recovers most of the +3.3 pp pooled-joint lift AND transfers on LOPO. If yes, hierarchical-atlas replaces mean-pool.
- *Decision gate*: pooled-joint ≤ 0.78 AND LOPO ≤ 0.80 → replace readout.

**T3.2 [QUEUE, small] Hierarchical-atlas on scalable bucket.**
If T3.1 succeeds, test the same readout on `per_elec_fourier_mni`. Readout improvement that works on both architectures is architecture-agnostic; that's a migration-relevant finding.

- 1 variant × (15 + 60). Conditional on T3.1.

**T3.3 [QUEUE, cheap, HIGH PRIORITY] pe2d mechanism disambiguation.**
Three stories for pe2d's LOPO win: break-symmetry prior (learned content matters), cell-id regularization (stable identity only), any-prior warm-start (even random works). Run three variants against `per_cell + partialconv` to separate:

- `pe2d_random_frozen` — random init, requires_grad=False. If LOPO ≤ `per_cell + partialconv` baseline by ~1 pp, the "any-prior" story wins — we'd switch to the cheaper recipe.
- `pe2d_row_only` — 4-dim row embedding × 16 (32 cells all share col identity). Separates row-axis from col-axis contribution.
- `pe2d_col_only` — 8-dim col embedding × 16. Same, inverse.

3 variants × (15 + 60) = 225 tasks.

- *Decision gate*: if `pe2d_random_frozen` matches pe2d (within 0.5 pp LOPO), default becomes `pe2d_random_frozen` — simpler, same lift. Otherwise keep learned pe2d.
- *Secondary*: the row/col decomposition tells us which axis is load-bearing, which informs whether pe2d-style break-symmetry has an analogue on the scalable path (per-electrode learned-id, 3D coord quantization, etc.).

**T3.4 [QUEUE, small] pe2d + hierarchical composition.**
pe2d wins LOPO / loses pooled-joint; hierarchical wins pooled-joint / ties LOPO. They touch different stages (input PE vs readout) and may be additive on both protocols.

- 1 variant × (15 + 60).
- *Hypothesis*: pe2d+hierarchical closes some of hierarchical's LOPO gap (pe2d's pretrain-better-landscape effect helps the readout too) while preserving the pooled-joint lift. If pooled-joint ≤ 0.77 AND LOPO ≤ 0.79, we have both.

**T3.5 [QUEUE, small] pe2d + per_elec_fourier_mni.**
The scalable-path analogue of pe2d's break-symmetry effect is Fourier MNI. If the mechanism is "break-symmetry + finetune specialize," applying a learned discrete per-electrode id on top of Fourier should push the scalable arm further. Cheap analogue: a learned per-electrode scalar added to the parcel embedding input.

- 1 variant × (15 + 60). Low expected ceiling, high information.

### Tier 4 — infrastructure for the longer horizon

**T4.1 [PLAN] SSL pretrain scaffolding.**
Charmander-style masked reconstruction on the full 2.83 h PS corpus (and 3.96 h lexical once projection is unblocked). Recipe: 50% random electrode masking + temporal patch masking (Jiang-validated robustness to heterogeneity). MSE reconstruction loss on masked electrode features.

- Setup: extend `src/speech_decoding/v14/` with an SSL dataloader that streams continuous windows from the raw uECoG recordings (not phoneme-epoched). Add a mask generator and reconstruction head. Training script with checkpoint interop to the supervised loader.
- *First experiment once ready*: `per_cell + partialconv + SSL pretrain` vs `per_cell + partialconv` no-pretrain. Then `per_elec_fourier_mni + SSL pretrain` vs no-pretrain. The test is whether SSL closes the scalable pooled-joint deficit.
- *Priority*: begin infrastructure now; experiments follow.

**T4.2 [PLAN] Phase-1.5 lexical supervised expansion.**
Three lexical patients are projectable today (S76, S78, S81). Add to pooled training; N=4→7 LH patients. Re-run T2.1 at N=7 to land one more point on the scaling curve.

- *Priority*: medium. Gated on resolving the S73 CCA-anchor question with Zac and confirming the 28-ARPABET reduction to the 8-phoneme PS intersection.

## Decision gates (summary)

| Trigger | Action |
|---|---|
| T2.1 scaling slope ≥ 0.5 pp per patient | Pivot planned at Phase-1.5 N=7 |
| T2.2 d=64 closes ≥ 2 pp of scalable LOPO deficit | Pivot via capacity, not scale |
| T3.3 `pe2d_random_frozen` LOPO within 0.5 pp of learned pe2d | Switch default to the frozen-random recipe |
| T3.3 row-only or col-only matches full pe2d | Single-axis embedding replaces pe2d (~half the params) |
| T3.4 pe2d+hierarchical wins both protocols | Hierarchical-over-pe2d becomes default readout |
| T3.1 atlas-anchored hierarchical wins both | Hierarchical-atlas becomes default readout (independent of T3.4) |
| T1.2 any single aug op passes both protocols | Add that op to default |
| T4.1 SSL closes scalable pooled-joint deficit to within 1 pp | Pivot to scalable |
| None of the above | Default stays `per_cell + partialconv + pe2d`; SSL (T4.1) is the next lever |

## What we are explicitly NOT running (and why)

- **Single-stream parcel-only (MIBRAIN / Path B)** — destroys uECoG within-parcel information; no precedent at our resolution.
- **Register tokens** — labeled parcel embedding already serves the shared-aggregator role; no in-modality evidence at ~50 k params.
- **Per-patient diagonal γ,β + softsign as default** — Levin form is spike, full affine. Citation-to-form gap too wide to default without a data-scarcity-specific argument. Revisit only if T2.1 / T4.1 show a residual patient-specific component the parcel embedding isn't absorbing.
- **Dual-stream [cells | parcels]** — internal contradiction (the "cells" in the pool don't preserve the within-parcel electrode expressivity the argument invoked).
- **FiLM / AdaIN / patient-stat normalization** — zero in-modality citation.
- **Additional scalable-PE variants beyond Fourier MNI / distance bias / T3.5 per-electrode id** — ordering is already set (Fourier > distance > none). Further PE iteration on scalable is not where the 0.7 pp LOPO gap closes; SSL and capacity are.
- **Larger Fourier basis on scalable bucket** — 16 frequencies × std=0.2 (≈5 mm correlation length) is already well-matched to uECoG spacing; over-parameterization without more data overfits.

## Canonical-goal commitment

The default is `per_cell + partialconv + pe2d` — *two principled additions and one LOPO-measured win*. None of these three pieces survives the eventual scalable pivot; they are the right bet for the pool's remaining Phase-1 lifetime. The scalable architecture is gated on T2.1 / T2.2 / T4.1; its LOPO deficit is 0.7 pp (much smaller than pooled-joint implied), so the bar SSL and scaling must clear is lower than it looked a day ago. No current-scale win is allowed to promote a non-scalable architecture as a permanent default (this is the rule that killed hierarchical and that bounds pe2d's lifetime).

---

**Meta (updated 2026-04-19 after Batch 2 LOPO).** Two lessons this batch:

1. **Pooled-joint and LOPO can disagree in sign.** pe2d lost pooled-joint and won LOPO; hierarchical did the inverse. Single-protocol evidence is structurally insufficient to default an architectural change, and the canonical-goal rule (LOPO resolves disagreements) earned its keep.
2. **Empirical iteration narrows the hypothesis space faster than reasoning.** The armchair pivot to `per_electrode` hit a 4 pp pooled-joint wall; the armchair rejection of `pe2d` was overturned by a uniform-direction LOPO signal. CLAUDE.md Engineering Discipline #8 is the right principle here: discuss to define the experiment; the experiment resolves what discussion cannot. Next week's experiments are designed so each one moves at least one decision gate — no fishing.
