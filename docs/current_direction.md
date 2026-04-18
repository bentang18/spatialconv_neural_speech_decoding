# Current Direction

Updated: 2026-04-17 — Phase 1 `v14-core` per-phoneme implementation is complete, P8 S14 run hit the 0.78 PER target (mean 0.783 at `d=32, depth=3`), and capacity + spatial ablations are running. P9 (cohort extension) is pending.

Canonical references (do not restate them here):

- **Implementation plan**: `docs/plans/v14-core-current.md`
- **Per-patient tables and data reference**: `docs/data_reference.md`
- **Experiment log (results)**: `docs/experiments/v14_ablation_log.csv`
- **DCC setup**: `docs/dcc_setup.md`
- **Design document (historical, pre-B1-amendment)**: `docs/neural_field_perceiver_v14.tex`
- **Working principle**: `CLAUDE.md`
- **Open work items**: `docs/implementation_tasks.md`

Archived on 2026-04-17: the architecture-amendment memo, the three session logs from `#34 / #36 / kernel-ablation`, the pre-amendment plan, and the per-phoneme plan drafts. See `docs/archive/sessions/` and `docs/archive/plans/`.

## Active priority

**P9**: extend the per-phoneme pipeline from S14 alone to the other 6 Phase-1 LH patients (`S26, S33, S62, S16, S23, S39`). Everything through P8 is frozen.

## Where we are

**Phase 1 supervised correctness pass on `uECoG`**, fixed Brainnetome Tier-1 soft parcel embedding over per-electrode tokens. No learned per-patient calibration, no SSL, no `sEEG`, no external datasets. The scientific bet: electrodes are patient-specific; the parcel set is the shared cross-patient anchor.

**All Phase 1 blockers** (`#1`–`#36`) closed 2026-04-16 late. `#34` (phoneme-loading audit: `.fif` is authoritative, events TSV is a cross-check) closed earlier the same day; `#36` (fsaverage spatial pivot) closed after the cras-bug fix.

**Per-phoneme implementation** (plan P1–P6, committed `875ccb8`) complete: `pool.py`, `phoneme_dataset.py`, `phoneme_model.py`, `phoneme_decoder.py`, `phoneme_run_fold.py`, per-phoneme `eval.py` + `train.py` wiring, 269 v14 tests green. Grad-accum tail-flush bug fixed.

**DCC runs so far:**
- **P7 smoke** (45707136) — pipeline green on GPU, 8 s end-to-end.
- **P8 full S14** (45707149) — 30 jobs, 5 folds × 3 seeds × 2 depths. `d=32, depth=3` mean **0.783**, `depth=1` mean 0.820. Best single run `fold2_seed1_depth3 = 0.644` (beats 0.734 baseline by 9 pp).
- **Ablation — capacity** (45707426) — `d=32 depth=5 / d=64 depth=3 / d=64 depth=5`, 45 jobs, running. Early read trending toward null.
- **Ablation — spatial** (45707427) — `k=1 / k=5` Conv2d and `(2,4) / (8,16)` pool, 60 jobs, queued.

## Current architecture (per-phoneme path, frozen)

Target architecture per `docs/plans/v14-core-current.md`:

```
signal (B, N_e, 130) at 200 Hz, window [-0.15, 0.5)s around MFA phoneme onset
→ grid-scatter (B, 1, H_p, W_p, 130)
→ Conv2d(1→8, k=3, pad=1) per time-step + GELU
→ masked-mean pool to (4, 8) = 32 cells
→ per-cell Conv1d(8→32, k=30, stride=10) → (B, 32 cells, 32 ch, 11 tokens)
→ + pooled_support @ P_emb[15, 32]  (broadcast across tokens)
→ flatten cell-major to (B, 352 tokens, d=32)
→ Backbone: 3 × [combined attention over (cell × time) + FFN]
    2 heads × 16, FFN 128, RoPE on temporal axis only, dropout 0.1
→ D1 decoder (mean-pool memory + prev-phoneme emb + Linear) → (B, 9)
```

47k parameters total. Matches Ben's 0.734 baseline temporal front-end (Conv1d k=30 s=10). Trains in ~20–40 min per fold on a single RTX 5000 Ada.

**What's shared cross-patient:** Conv2d weights, per-cell Conv1d weights, `P_emb[15, 32]`, backbone, decoder. **What's patient-specific:** grid layout (not learned), `support[N_e, 15]` (atlas lookup, not learned).

## Baseline to beat

**PER 0.734 ± 0.007** on S14, grouped-by-token CV, 3-seed, per-phoneme MFA flat head + full recipe. Population mean **0.825** across 11 patients. Apples-to-apples only when v14 runs the same grouped-by-token CV and slot-averaged PER contract (`#33`).

## Patient scope (Phase 1)

- **Core**: `S14, S26, S33, S62` (all LH).
- **Extended (LH)**: `S16, S23, S39`.
- **Deferred to Phase 2 with the sEEG join**: `S22, S58` (RH).
- **Excluded from Phase 1**: `S32` (no HG response), `S57` (hybrid strip, 52/256 sig, Map 8 wiring unresolved).

## Near-term sequencing

1. P9: extend per-phoneme run to 6 other Phase-1 LH patients (180 jobs).
2. Ablation consolidation: backfill `v14_ablation_log.csv`, decide which axes carry signal.
3. Phase 1.5: SSL on the full continuous `uECoG` corpus (~24 h once HGA pipeline runs).
4. Phase 2: learned per-patient calibration (`Δ/ω`, `δ_l`, `τ_l`).
5. Phase 2+: `sEEG` join, external chronic ECoG (Flinker, Chang).

## Practical rules

- Always `exclude_artifacts=True`. Always grouped-by-token CV. Always 3-seed runs.
- All training on DCC. Never local. See `docs/dcc_setup.md`.
- Supervised contract `#9` is frozen for Phase 1.
- If a doc references v12, factored spatial-then-temporal attention, within-parcel Perceiver summarizer, `parcel_frames.npz` as a runtime input, `N_tok = 15` atlas-pool tokens, or cvs_avg35 as the active spatial base — it is stale. Authoritative pipeline is the per-phoneme path above.
