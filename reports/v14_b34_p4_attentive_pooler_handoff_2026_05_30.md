# Handoff — B34: drop PMA from Phase 4 + per-task attentive readout

**Date**: 2026-05-30
**From**: architecture/docs (Ben + Claude)
**To**: implementation engineer
**Scope**: code-side completion only. The architecture/docs half (B34 memo, MEMORY.md, training_recipe.md, v14_blockers.md) is owned by Ben and was written alongside this handoff — do **not** re-touch those.

Canonical decision memo: `memory/project_v14_b34_p4_attentive_pooler_2026_05_30.md`.

---

## What the change is

**Before (B31 Phase-4 readout):** `V14ParcelCollapsePMA` (Set-Transformer k=1 query collapsing the parcel axis per time-patch, `(B,L,T,d)→(B,T,d)`, frozen at P4) → `V14Phase4FlatHead` (trainable). The PMA discarded the parcel axis before the probe.

**After (B34 Phase-4 readout):** PMA is **removed from Phase 4 entirely** (it stays P3-only, for Whisper distillation, plus the `R-add-utterance-loss` SSL sister). The P4 readout is a fresh **per-task trainable attentive query over the full `(B, L, T_p, d)` parcel×time field**, key-masked by `latent_valid`. A masked **mean-pool baseline** (`readout="meanpool"`) must always be reported alongside for subject-leakage control.

Design note: the attentive pooler is the **lean V-JEPA §4.3 form** (single learnable query, single cross-attn layer, small MLP + classifier) — NOT the V-JEPA-2 4-block predictor — to keep subject-identity leakage controllable on the CSubject prong. A large attentive-vs-meanpool CSubject gap is a **leakage red flag, not a win**.

---

## Implementation state (all in working tree, uncommitted)

Feature is wired end to end. Full suite at handoff: **887 tests, 1 failing, 5 skipped.** The 1 failure is a test-sensitivity issue, not a wiring bug (see below).

### `src/speech_decoding/models/v14_encoder.py`
- **`V14PerTaskAttentivePooler(d_model, n_heads, n_classes, *, mlp_ratio=4, dropout=0.0)`** — NEW default P4 readout. Single `nn.Parameter` query; ln_q/ln_kv/ln_post; q_proj/kv_proj(2·d)/out_proj (bias=False); MLP; `Linear(d, n_classes)`. Flattens `(B,L,T,d)→(B,L·T,d)`, masks invalid latents via `latent_valid` broadcast over time → `masked_fill(NEG_INF_MASK_VALUE)` → softmax. **792,074 params at d=256, nc=10.**
- **`V14MeanPoolLinearHead(d_model, n_classes)`** — NEW leakage-control baseline. Masked mean over valid `(L,T)` cells → `Linear`. **2,570 params.**
- **`V14Phase4FlatHead`** — kept as a reusable primitive; docstring demoted ("no longer the Phase-4 default").
- **`V14ParcelCollapsePMA`** — docstring updated to "Phase-3 ONLY".
- **`V14ParcelPerceiverWithHead`** — rewritten `__init__(encoder, readout, eps)`; `forward` computes `latent_valid` from support+valid_mask and calls `self.readout(latents, latent_valid=...)`. Confirmed it threads `shaft_mask`, `lambda_anat`, `subject_subtype`, `ref_idx` into the encoder.
- **Config `V14ParcelPerceiver`**: removed `pma_freeze`; added `readout: Literal["attentive","meanpool"] = "attentive"`. `build()` constructs the readout via if/elif and returns `V14ParcelPerceiverWithHead(...)`.

### `src/speech_decoding/models/__init__.py`
- Exports `V14MeanPoolLinearHead` + `V14PerTaskAttentivePooler`.

### `src/speech_decoding/experiments/dispatch_v14.py`
- `pma_freeze` param / cfg-dict / CLI / logging / call all replaced by `readout`.
- CLI: old `--unfreeze-pma` (store_false) → `--readout {attentive,meanpool}`, default `attentive`.

### Tests updated
- `models/test_v14_encoder.py` — PMA-freeze build tests replaced by `test_v14_config_build_default_readout_is_attentive` + `..._meanpool_readout_baseline`. (Param-budget test at the first-pass defaults only asserts `<20M`/`<30M`; the real default-shape total is now **12,647,946**, passes comfortably.)
- `models/test_v14_readout_mask.py` — rewritten: invalid-latent invariance, unmasked-attends-everywhere, meanpool-ignores-invalid, two wrapper `latent_valid` integration tests.
- `experiments/test_v14_dispatch_wired.py` — `pma_freeze=` → `readout=` assertion; meanpool smoke test replaces the unfreeze-pma smoke test.

---

## Loose end 1 — the one failing test (decision needed)

`models/test_v14_encoder.py::test_v14_head_wrapper_forwards_b29_conditioning_to_encoder` (a **pre-existing** B29 conditioning test, not new to B34).

It flips `lambda_anat` between `[1.0,1.0]` and `[1.0,0.0]` across a 2-clip batch and asserts the logit delta `> 1e-4`. Observed: **1.87e-5**.
- Clip 0 (λ unchanged): delta exactly `0.0` ✓
- Clip 1 (λ 1.0→0.0): delta `~1.9e-5`, nonzero but below threshold.

**Diagnosis (verified):** the wrapper *does* forward `lambda_anat` (encoder forward call passes it, `v14_encoder.py:1578`). The new attentive readout's extra LayerNorms + softmax **attenuate** the small bias-induced latent shift more than the old flat head did, and the test's near-uniform `graded_support` gives the anatomy gate weak contrast. A probe confirmed peakier support widens the gap (power=8 support → ~6e-5, still under 1e-4).

**Recommended fix:** make the test's `graded_support` peakier (sharper anatomy contrast) so the gate has real signal to scale — a ~2-line test change at `test_v14_encoder.py:664–667`. Do **not** just lower the `1e-4` threshold; that weakens an unrelated regression guard for `subject_subtype`/`ref_idx` asserted earlier in the same test.

---

## Loose end 2 — numbering collision: rename B33 → B34 in code

The code currently labels this change **"B33"**, but **B33 is already taken** (`project_v14_b33_project_up_phase3_2026_05_30.md` = the P3 project-up lock). This P4 change is **B34**. All "B33" occurrences in the touched files are the P4 change; **zero** project-up references live in these files, so a scoped find-replace is safe.

Two things to rename:
1. Bare token **`B33` → `B34`** in comments/docstrings.
2. Wikilink slug **`project_v14_b33_p4_attentive_pooler_2026_05_30` → `project_v14_b34_p4_attentive_pooler_2026_05_30`** (the memo uses the `b34` slug; code wikilinks are dangling until renamed).

Files containing the strings (verified at handoff):
- `src/speech_decoding/models/v14_encoder.py` (9 bare "B33" + 5 slug wikilinks)
- `src/speech_decoding/models/__init__.py` (1 bare "B33")
- `src/speech_decoding/experiments/dispatch_v14.py` (3 bare "B33")
- `src/speech_decoding/models/test_v14_encoder.py` (3 bare "B33")
- `src/speech_decoding/models/test_v14_readout_mask.py` (1 bare "B33" header + 1 slug wikilink)

Verify with: `grep -rn "B33\|project_v14_b33_p4" src/speech_decoding/` and confirm every hit is the P4-attentive change (NOT the project-up lock) before replacing.

---

## Param accounting (measured at d=256, n_classes=10, T_p=4)

| readout | trainable params | total params | note |
|---|---|---|---|
| OLD B31 P4: PMA (frozen) + flat head | **10,250** | 800,010 | PMA 789,760 frozen at P4; flat head 10,250 trainable |
| NEW B34 P4: attentive pooler (default) | **792,074** | 792,074 | per-task query over parcel×time |
| NEW B34 P4: meanpool baseline | **2,570** | 2,570 | leakage-control reference |
| Full model (F=38/T=17/K=80/d=256/nc=10) | **12,647,946** | — | encoder 11.86M + attentive 0.79M; under 30M cap |

The default trainable probe (792K) is far more expressive than the old frozen-PMA+flat-head (10K trainable) — deliberate, hence the mandatory meanpool baseline.

---

## Engineer checklist

1. [ ] Fix `test_v14_head_wrapper_forwards_b29_conditioning_to_encoder` (peakier `graded_support`).
2. [ ] Rename B33 → B34 (bare token + slug) across the 5 files above; verify no project-up hit was touched.
3. [ ] Clear `__pycache__` (`find src -name __pycache__ -type d -exec rm -rf {} +`) and run the full suite: `.venv/bin/python -m pytest -q src/` — expect **888 pass / 0 fail / 5 skip** (it was 887/1/5 with the failure).
4. [ ] Commit on a branch (NOT main, NOT the DCC clone). Do not push unless Ben asks.

## Guardrails (project policy)
- All training on DCC, never local; never commit on the DCC clone.
- Do not push to remote unless explicitly asked.
- Never skip hooks (`--no-verify`) or bypass signing.
- Surgical changes only — don't refactor adjacent code.
