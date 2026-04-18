# Current Direction

Updated: 2026-04-18 — Phase 1 ablations converged. Flat Conv1d front-end promoted to default. Q1d (7-LH pooled) is the last Phase-1 gate before Phase 1.5 SSL planning.

Canonical references (do not restate them here):

- **Implementation plan**: `docs/plans/v14-core-current.md`
- **Per-patient tables and data reference**: `docs/data_reference.md`
- **Experiment log (results)**: `docs/experiments/v14_ablation_log.csv`
- **DCC setup**: `docs/dcc_setup.md`
- **Design document (historical, pre-B1-amendment)**: `docs/neural_field_perceiver_v14.tex`
- **Working principle**: `CLAUDE.md`
- **Open work items**: `docs/implementation_tasks.md`

## Active priority

**Q1d**: pooled run on the full Phase-1 LH cohort (`S14, S16, S23, S26, S33, S39, S62`) with the flat front-end. Job `45720982`, 15 jobs. Open question: at 4 patients, pooled flat (pop 0.791) ≈ per-subject flat mean (0.784) — no transfer signal. If 7 patients still don't beat per-subject, the bottleneck is architectural, not data — that decision gates Phase 1.5 SSL planning.

## Where we are

**Phase 1 supervised correctness pass on `uECoG`**, fixed Brainnetome Tier-1 soft parcel embedding over per-electrode tokens. No learned per-patient calibration, no SSL, no `sEEG`, no external datasets. The scientific bet: electrodes are patient-specific; the parcel set is the shared cross-patient anchor.

**All Phase 1 blockers** (`#1`–`#36`) closed 2026-04-16 late.

**Per-phoneme pipeline** (plan P1–P6, committed `875ccb8`) complete. 285 v14 tests green. Grad-accum tail-flush bug fixed.

## Ablation summary (as of 2026-04-18)

**Per-subject best pop means** (15 runs each, 5 folds × 3 seeds):

| patient | per_cell canonical | flat (new default) | Δ |
|---|---|---|---|
| S14 | 0.790 | 0.790 | 0.000 |
| S26 | 0.864 | 0.755 | **−10.9pp** |
| S33 | 0.798 | 0.779 | −1.9pp |
| S62 | 0.874 | 0.812 | −6.2pp |

Flat recovers most of the per-subject gap vs the 2026-04-04 per-phoneme baseline.

**Pooled (Q1a, 4 core)**: pop PER 0.794 ± 0.059 (per_cell) → **0.791 ± 0.034 (flat)**. Mean unchanged; **variance halved**. Flat promoted to default (`src/speech_decoding/v14/config.py`).

**Training-side tricks (flat + LS 0.1 + mixup 0.2)**: pop 0.787 ± 0.043 — wash against flat alone. Label smoothing and mixup don't stack on top of the atlas embedding at this scale. Retained as a variant but not in the canonical recipe.

**Pool geometry (adaptive_avg vs masked_mean)**: S14+S62 flat+adaptive_avg recovered +0.4pp on S62 (noise). Falsified as a driver of the residual gap.

**Parcel embedding (Q1a `--no-parcel-embedding`)**: pop regressed +1.1pp, S33 lost 5.7pp. Load-bearing for pooled low-n; kept on.

## Current architecture (per-phoneme path, frozen)

```
signal (B, N_e, 130) at 200 Hz, window [-0.15, 0.5)s around MFA phoneme onset
→ grid-scatter (B, 1, H_p, W_p, 130)
→ Conv2d(1→8, k=3, pad=1) per time-step + GELU
→ masked-mean pool to (4, 8) = 32 cells
→ flat Conv1d(8·32=256 → 32, k=30, s=10) → (B, 11 tokens, d=32)    ← NEW DEFAULT
→ Backbone: 3 × [combined attention + FFN], 2 heads × 16, FFN 128, RoPE on temporal
→ D1 decoder (mean-pool memory + prev-phoneme emb + Linear) → (B, 9)
```

~285k parameters (flat front-end dominates). Parcel embedding is auto-disabled in flat mode — there is no cell dimension at the token stage for `pooled_support @ P_emb` to broadcast over.

**`per_cell` path** (shared Conv1d(8→32) per cell, ~45k params, with parcel embedding active) is retained as an ablation but no longer the default. Pin with `--temporal-frontend per_cell`.

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

## Practical rules

- Always `exclude_artifacts=True`. Always grouped-by-token CV. Always 3-seed runs.
- All training on DCC. Never local. See `docs/dcc_setup.md`.
- Supervised contract `#9` is frozen for Phase 1.
- If a doc references v12, factored spatial-then-temporal attention, within-parcel Perceiver summarizer, `parcel_frames.npz` as a runtime input, `N_tok = 15` atlas-pool tokens, or cvs_avg35 as the active spatial base — it is stale. Authoritative pipeline is the per-phoneme path above.
