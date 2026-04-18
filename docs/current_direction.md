# Current Direction

Updated: 2026-04-18 (late) — Phase 1 ablations converged. `per_cell` + attention + atlas parcel embedding is canonical. `flat` is retained as ablation only. Q1d (7-LH pooled) landed; flat-arm LOPO pretrain→finetune complete (null, as expected — flat shows no pooled transfer); per_cell-arm LOPO pretrain→finetune submitted as `45724809` (the informative test; first submission `45724593` cancelled after 0/60 succeeded due to an uncommitted `cv.py` helper — see commit `41bc7cc`).

## Canonical research goal

**Cross-patient / cross-sensor scalable representation** of intracranial field potentials for speech decoding. Per-subject peak performance is **not the metric**. Transferable architecture is.

**Roadmap:** (1) Phase 1 supervised on uECoG (current), (2) ~2× uECoG patient count, (3) Phase 1.5 SSL on the full continuous uECoG corpus, (4) Phase 2 learned per-patient calibration, (5) Phase 2+ ~25 h sEEG + external chronic ECoG (Flinker, Chang). Steps (2)–(5) all require an architecture that scales across electrodes, grid layouts, and array geometries. **Do not promote a design to default if it wins per-subject but requires rigid uECoG geometry.** Flat Conv1d is a rigid-grid local optimum; `per_cell` + attention + atlas parcel embedding is the canonical path.

Canonical references (do not restate them here):

- **Implementation plan**: `docs/plans/v14-core-current.md`
- **Per-patient tables and data reference**: `docs/data_reference.md`
- **Experiment log (results)**: `docs/experiments/v14_ablation_log.csv`
- **DCC setup**: `docs/dcc_setup.md`
- **Design document (historical, pre-B1-amendment)**: `docs/neural_field_perceiver_v14.tex`
- **Working principle**: `CLAUDE.md`
- **Open work items**: `docs/implementation_tasks.md`

## Active priority

**Q1d** (landed): pooled run on the full Phase-1 LH cohort (`S14, S16, S23, S26, S33, S39, S62`) with the flat front-end. Job `45720982`, 15 jobs. At 4 patients, pooled flat (pop 0.791) ≈ per-subject flat mean (0.784) — no transfer signal. 7-LH pooled flat: 0.794 ± 0.025 (per-patient mean 0.797). Per-subject wins on variance but not on mean; Phase-1 data-scaling alone does not recover transfer.

**LOPO warm-start on `flat`** (job `45723956`, 60/60 done): pretrain pooled flat on 3 core patients, finetune on held-out. S14 0.797 ± 0.051 (Δ +0.007); S26 0.758 ± 0.058 (Δ +0.003); S33 0.777 ± 0.092 (Δ −0.002); S62 0.793 ± 0.042 (Δ −0.019). All four Δs within noise. **Caveat**: pooled joint training on flat showed no cross-patient lift per-subject (Q1a 4-core: pooled flat 0.791 ≈ per-subject flat mean 0.784), so running LOPO on flat was a null test by construction — confirms flat has no transfer handle, not that warm-start fails.

**LOPO warm-start on `per_cell`** (job `45724593`, 60 jobs, queued 2026-04-18): same pretrain→finetune contract but on the canonical arm. Pooled joint training on per_cell does pull S26 (−7.6pp) and S33 (−10.9pp) toward the pop mean at 4 patients, so this is the informative test of whether a warm-started backbone transfers without the held-out patient's tokens in the pretrain set.

## Where we are

**Phase 1 supervised correctness pass on `uECoG`**, fixed Brainnetome Tier-1 soft parcel embedding over per-electrode tokens. No learned per-patient calibration, no SSL, no `sEEG`, no external datasets. The scientific bet: electrodes are patient-specific; the parcel set is the shared cross-patient anchor.

**All Phase 1 blockers** (`#1`–`#36`) closed 2026-04-16 late.

**Per-phoneme pipeline** (plan P1–P6, committed `875ccb8`) complete. 285 v14 tests green. Grad-accum tail-flush bug fixed.

## Ablation summary (as of 2026-04-18)

**Solo vs pooled, per patient** (15 runs each, d=32 depth=3):

| patient | solo per_cell | solo flat | pooled-4 flat | pooled-7 flat | baseline |
|---|---|---|---|---|---|
| S14 | 0.788 | 0.790 | 0.773 | 0.792 | **0.734** |
| S26 | 0.879 | 0.755 | 0.770 | **0.751** | 0.707 |
| S33 | 0.873 | 0.779 | 0.792 | **0.762** | 0.749 |
| S62 | 0.824 | 0.812 | 0.794 | 0.792 | 0.761 |

**Pop means**:

| setup | pop | std |
|---|---|---|
| pooled Q1a 4-core per_cell | 0.794 | 0.059 |
| pooled Q1a 4-core flat | 0.791 | 0.034 |
| pooled Q1b 7-LH per_cell | 0.834 | 0.051 |
| pooled Q1d 7-LH flat | 0.794 | 0.025 |

**Reading**: at 4 patients pooled, per_cell (0.794) ≈ flat (0.791) — the capacity gap closes with data. Flat only wins on variance at fixed n. The long-term goal is cross-patient / cross-sensor, so `per_cell` stays canonical; its prior is right for variable arrays and its capacity scales with more data.

**Other axes**:
- Training-side (LS 0.1 + mixup 0.2 on flat): 0.787 ± 0.043 — wash at this scale.
- Pool geometry (adaptive_avg vs masked_mean): +0.4pp noise on S62. Not a driver.
- Parcel embedding: load-bearing (Q1a `--no-parcel-embedding` regressed +1.1pp pop, S33 lost 5.7pp). Kept.
- Per_cell at 7 patients regressed (0.794 → 0.834) — capacity needs to scale with cohort size. Expected to resolve as we go to 2× patients + SSL pretrain.

## Current architecture (per-phoneme path, canonical)

```
signal (B, N_e, 130) at 200 Hz, window [-0.15, 0.5)s around MFA phoneme onset
→ grid-scatter (B, 1, H_p, W_p, 130)
→ Conv2d(1→8, k=3, pad=1) per time-step + GELU
→ masked-mean pool to (4, 8) = 32 cells
→ per-cell Conv1d(8→32, k=30, s=10) applied independently per cell (shared weights)
→ + pooled_support @ P_emb[15, 32]  (atlas parcel anchor, broadcast across time)
→ flatten cell-major to (B, 352 tokens, d=32)
→ Backbone: 3 × [combined attention + FFN], 2 heads × 16, FFN 128, RoPE on temporal
→ D1 decoder (mean-pool memory + prev-phoneme emb + Linear) → (B, 9)
```

~47k parameters. Permutation-equivariant across cells; cross-patient alignment is the atlas parcel embedding; handles variable grid shapes and per-electrode tokens (required for sEEG / external arrays).

**`flat` path** (collapses cell dim via `Conv1d(256 → 32)`, ~285k params, position-specific weights, no parcel embedding) is retained as an **ablation only**. Pin with `--temporal-frontend flat`. Wins 1–3pp per-subject at current scale, but binds to rigid uECoG geometry. Not the canonical path.

## Baseline to beat

**PER 0.734 ± 0.007** on S14, grouped-by-token CV, 3-seed, per-phoneme MFA flat head + full recipe. Population mean **0.825** across 11 patients. Current v14 best-per-patient: S14 0.773, S26 0.755, S33 0.761, S62 0.794.

## Patient scope (Phase 1)

- **Core**: `S14, S26, S33, S62` (all LH).
- **Extended (LH)**: `S16, S23, S39`.
- **Deferred to Phase 2 with the sEEG join**: `S22, S58` (RH).
- **Excluded from Phase 1**: `S32` (no HG response), `S57` (hybrid strip, 52/256 sig, Map 8 wiring unresolved).

## Near-term sequencing

1. Q1d verdict (in flight): is data-scaling within Phase 1 recovering transfer?
2. Phase 1.5: SSL on the full continuous `uECoG` corpus (~24 h once HGA pipeline runs).
3. Phase 2: learned per-patient calibration (`Δ/ω`, `δ_l`, `τ_l`).
4. Phase 2+: `sEEG` join, external chronic ECoG (Flinker, Chang).

## Canonical experimental protocol

Every architectural or data-scaling change is evaluated against **two** protocols, always:

1. **Pooled joint** — one model trained on all patients simultaneously, each patient's held-out fold evaluated on that shared model. Tests whether weight-sharing during training helps each patient. Cheap: one training run per (fold, seed).
2. **LOPO warm-start** — pretrain pooled on `N−1` patients (held-out patient's data never seen), then finetune per-patient on the held-out patient's fold-train split, loading the pretrained checkpoint as initialization. Tests whether pretraining transfers to a *new* patient. This is the foundation-model test and is load-bearing for Phase 1.5 SSL → supervised finetune, Phase 2+ cross-sensor transfer, and external-corpus transfer.

The informative signal is the **gap between pooled joint and LOPO warm-start**. If warm-start matches or beats pooled joint, the backbone is learning transferable structure. If warm-start is no better than scratch per-patient, no transfer is happening — regardless of how good pooled joint looks.

Sbatch wrappers: every architectural config should have both `v14_pooled_<tag>_dcc.sh` and `v14_lopo_<tag>_dcc.sh`. Aggregator handles LOPO outputs via the existing variant-suffix mechanism.

## Practical rules

- Always `exclude_artifacts=True`. Always grouped-by-token CV. Always 3-seed runs.
- All training on DCC. Never local. See `docs/dcc_setup.md`.
- Supervised contract `#9` is frozen for Phase 1.
- **Every architectural change reports both pooled joint and LOPO warm-start** (see protocol above). Single-protocol evidence does not justify defaulting an arch change.
- If a doc references v12, factored spatial-then-temporal attention, within-parcel Perceiver summarizer, `parcel_frames.npz` as a runtime input, `N_tok = 15` atlas-pool tokens, or cvs_avg35 as the active spatial base — it is stale. Authoritative pipeline is the per-phoneme path above.
