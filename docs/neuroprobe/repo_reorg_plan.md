# Repo reorganization plan — NeuralSet-mirrored layout

**Date**: 2026-04-26 (rev 3 — pre-Stage 0).
**Goal**: restructure `src/speech_decoding/` so that NeuralSet integration is "drop one extractor file + one studies subfolder," not "weave through five subpackages." Mirror NeuralSet's organizing pattern wherever it applies; fill in the layers they don't ship using their stylistic conventions.
**Scope**: full reorganization. Nothing is sacred.
**Status**: **executing pre-Stage 0**, after smoke test passes (Phase 0 of `neuralset_integration_plan.md`). Stage 0 author-time = clean target shape; NeuralSet adapter PR adds to it. Decision: `project_pre_stage0_reorg_neuralset_adoption_2026_04_26.md`.

---

## NeuralSet's pattern (what we're mirroring)

| Pattern | NeuralSet | We adopt |
|---|---|---|
| Two leaf subpackages: `events/` + `extractors/` | ✓ | ✓ — same names, same role |
| Modality-flat split (audio/image/text/video/neuro/meta) | ✓ | ✓ — one file per concept |
| Tests colocated (`test_X.py` next to `X.py`) | ✓ | ✓ — `tests/v14/*` migrate inward |
| Single-file modality concept | ✓ | ✓ — collapse `fsaverage_projection.py` + `fsaverage_atlas.py` into `atlas/fsaverage.py` |

**Where we diverge** (justified): we add `models/` + `training/` (NeuralSet is substrate, not full ML stack); `studies/<cohort>/` is nested rather than flat (each cohort has 4 concept files); cohort-specific Ieeg subclasses live with their cohort (in `studies/<cohort>/study.py`), not in a shared `events/etypes.py` — keeps cohort code cohesive.

**Empty-package rule.** Don't create a subpackage until the first real file lands. `events/` and `ssl/` are documented intents (cross-cohort event subclasses; Stage-2 SSL heads — joint loss `L_recon` + `L_DSigLIP` + `L_KoLeo` per 4/27-night decision, NOT a separate Stage 2a/2b split) but stay un-created until Phase 3 / Stage 2 SSL respectively. Same rule for future cohort folders (`studies/cogan_seeg/`, `cogan_lex/`) — created when their first file is written.

---

## Target structure

```
src/speech_decoding/
├── __init__.py                       # public API exports
│
├── extractors/                       # MIRRORS neuralset.extractors
│   ├── __init__.py
│   └── parcel.py                     # V14ParcelMetadataExtractor (shared across cohorts; written in adoption PR)
│
├── studies/                          # Study subclasses + per-cohort data prep
│   ├── __init__.py
│   ├── braintreebank/                # populated by adoption PR (Phase 3)
│   │   ├── __init__.py
│   │   ├── study.py                  # BraintreebankIeeg + BraintreebankStudy
│   │   ├── loader.py                 # bt_load_raw() — raw 2048 Hz voltage, no re-ref  ← Stage 0 plan E1
│   │   ├── labels.py                 # 15-task label derivation  ← Stage 0 plan E3
│   │   └── manifest.py               # Tier-1 whitelist + BT-Tier-1 parcel list + Lite electrode lists
│   └── cogan_ps/
│       ├── __init__.py
│       ├── dataset.py                # ← FROM v14/phoneme_dataset.py (vanilla, non-NeuralSet path)
│       ├── channels.py               # ← FROM v14/channel_map.py (PS Utah amp→grid bridge)
│       ├── study.py                  # CoganIeeg + CoganStudy (built later, Stage 2 PS)
│       └── manifest.py               # 11-patient table
│
├── atlas/                            # atlas + parcel infrastructure
│   ├── __init__.py
│   ├── fsaverage.py                  # ← MERGE v14/fsaverage_projection.py + v14/fsaverage_atlas.py
│   ├── support.py                    # ← FROM v14/support_cache.py
│   └── tokens.py                     # ← FROM v14/token_spec.py (PS Tier-1 15 parcels)
│
├── models/                           # v14 architecture
│   ├── __init__.py
│   ├── backbone.py                   # ← FROM v14/backbone.py
│   ├── decoder.py                    # ← FROM v14/phoneme_decoder.py
│   ├── phoneme.py                    # ← FROM v14/phoneme_model.py (current PS-path model)
│   ├── pool.py                       # ← FROM v14/pool.py (PS-resume hierarchical_atlas)
│   └── perceiver_io.py               # NEW — Perceiver IO with parcel-id-tagged latents + Graphormer log(support) cross-attn QK bias + DETR-style task-attention readout (v14 Neuroprobe-path default; written by Stage-0/1, NOT this reorg)
│
├── training/                         # train loop, eval, CV, configs
│   ├── __init__.py
│   ├── train.py                      # ← FROM v14/train.py
│   ├── eval.py                       # ← FROM v14/eval.py
│   ├── cv.py                         # ← FROM v14/cv.py + evaluation/grouped_cv.py (collapsed)
│   ├── augmentation.py               # ← FROM v14/augmentation.py
│   ├── config.py                     # ← FROM v14/config.py
│   ├── run_fold.py                   # ← FROM v14/phoneme_run_fold.py
│   ├── run_fold_pooled.py            # ← FROM v14/phoneme_run_fold_pooled.py
│   ├── aggregate.py                  # ← FROM v14/aggregate.py
│   └── phoneme_map.py                # ← FROM data/phoneme_map.py
│
└── archive/                          # quarantined; no imports from active code
    ├── legacy/                       # existing pre-v14 quarantine (untouched)
    ├── regression_pivot/             # existing (untouched)
    ├── stage2_stubs/                 # ← v14/{calibration,tokenizer,model,decoder,grid_mixer,parcel_embedding,parcel_frames,local_summarizer}.py
    ├── parity_oracles/               # ← v14/{coordinates,cvsavg_projection}.py
    └── exploration/                  # ← v14/{electrode_pool,dataset,run_fold}.py + v14/audit/
```

**Created later (per the empty-package rule):**
- `events/` — when first cross-cohort event subclass is needed.
- `ssl/` — Stage 2 SSL kickoff (joint loss, NOT 2a/2b split per 4/27-night decision), when `d_siglip.py` + `projection.py` + `L_recon` JEPA-target generation (data2vec-2.0 contextualized targets, V-JEPA-2.1 dense predictive loss, EMA teacher; REPA projector if PE-Core/REPA cells adopted) are written.
- `studies/cogan_seeg/` — Stage 3 join.
- `studies/cogan_lex/` — Stage 2 PS extension.

Test files appear next to their modules when there's a real test to write — no pre-created empty files.

---

## File-by-file mapping

### Active code: `v14/` → new home

| Current path | New path | Notes |
|---|---|---|
| `v14/fsaverage_projection.py` + `v14/fsaverage_atlas.py` | `atlas/fsaverage.py` | merged |
| `v14/support_cache.py` | `atlas/support.py` | rename: drop `_cache` (it's both build + lookup) |
| `v14/token_spec.py` | `atlas/tokens.py` | PS Tier-1 list; BT-Tier-1 lives in `studies/braintreebank/manifest.py` |
| `v14/channel_map.py` | `studies/cogan_ps/channels.py` | PS Utah amp→grid bridge; cohort-specific (sEEG cohorts have no analog) |
| `v14/phoneme_dataset.py` | `studies/cogan_ps/dataset.py` | vanilla loader stays under cogan_ps |
| `v14/backbone.py` | `models/backbone.py` | |
| `v14/phoneme_decoder.py` | `models/decoder.py` | |
| `v14/phoneme_model.py` | `models/phoneme.py` | current PS-path active model; model code → models/, not training/ |
| `v14/pool.py` | `models/pool.py` | hierarchical_atlas pooling is model-side |
| `v14/train.py` | `training/train.py` | |
| `v14/eval.py` | `training/eval.py` | |
| `v14/cv.py` + `evaluation/grouped_cv.py` | `training/cv.py` | collapse two CV utility files |
| `v14/config.py` | `training/config.py` | |
| `v14/augmentation.py` | `training/augmentation.py` | |
| `v14/phoneme_run_fold.py` | `training/run_fold.py` | drop `phoneme_` prefix |
| `v14/phoneme_run_fold_pooled.py` | `training/run_fold_pooled.py` | |
| `v14/aggregate.py` | `training/aggregate.py` | |
| `data/phoneme_map.py` | `training/phoneme_map.py` | collapses `data/` package |

### Archive: `v14/` → `archive/`

| Current path | New path | Reason |
|---|---|---|
| `v14/{calibration,tokenizer,model,decoder,grid_mixer,parcel_embedding,parcel_frames,local_summarizer}.py` | `archive/stage2_stubs/` | Stage-2 stubs, never built out |
| `v14/{coordinates,cvsavg_projection}.py` | `archive/parity_oracles/` | parity oracles; not active path |
| `v14/{electrode_pool,dataset,run_fold}.py` | `archive/exploration/` | trial-level slot-CE + kernel-ablation exploration; never adopted |
| `v14/audit/` (8 files) | `archive/exploration/audit/` | closed audit |

NeuralSet-adoption deliverables (`extractors/parcel.py`, `studies/braintreebank/{study, loader, labels, manifest}.py`) are added by the **adoption PR** (Phase 3 of integration plan), not the reorg PR. Reorg PR moves existing files; adoption PR adds new ones into established locations.

---

## Tests: colocate per NeuralSet convention

`tests/v14/` has 34 test files. They split three ways:

**Migrate inward (~17 active-module tests):**

| Current path | New path |
|---|---|
| `tests/v14/test_phoneme_map.py` | `training/test_phoneme_map.py` |
| `tests/v14/test_grouped_cv.py` | `training/test_cv.py` |
| `tests/v14/test_aggregate.py` | `training/test_aggregate.py` |
| `tests/v14/test_augmentation.py` | `training/test_augmentation.py` |
| `tests/v14/test_config.py` | `training/test_config.py` |
| `tests/v14/test_eval.py` | `training/test_eval.py` |
| `tests/v14/test_phoneme_eval.py` | `training/test_phoneme_eval.py` |
| `tests/v14/test_phoneme_run_fold.py` | `training/test_run_fold.py` |
| `tests/v14/test_phoneme_run_fold_pooled.py` | `training/test_run_fold_pooled.py` |
| `tests/v14/test_phoneme_train_integration.py` | `training/test_phoneme_train_integration.py` |
| `tests/v14/test_train_step.py` | `training/test_train_step.py` |
| `tests/v14/test_run_one_fold.py` | `training/test_run_one_fold.py` |
| `tests/v14/test_backbone.py` | `models/test_backbone.py` |
| `tests/v14/test_phoneme_decoder.py` | `models/test_decoder.py` |
| `tests/v14/test_phoneme_model_shapes.py` | `models/test_phoneme.py` |
| `tests/v14/test_phoneme_dataset.py` | `studies/cogan_ps/test_dataset.py` |
| `tests/v14/test_support_cache.py` | `atlas/test_support.py` |
| `tests/v14/test_token_spec.py` | `atlas/test_tokens.py` |
| `tests/v14/test_fsaverage_spatial.py` | `atlas/test_fsaverage.py` |
| `tests/v14/test_ps_tokens_fixture.py` + `tests/v14/fixtures/` | `studies/cogan_ps/test_ps_tokens_fixture.py` (+ adjacent fixtures dir) |
| `tests/v14/test_no_legacy_imports.py` | `archive/test_no_legacy_imports.py` (the archive enforcer) |

**Delete (~13 quarantined-module tests — modules are archive-bound, no value in keeping their tests):**

`test_calibration.py` (no current file, already absent), `test_tokenizer.py`, `test_model.py`, `test_decoder.py` (≠ phoneme_decoder), `test_grid_mixer.py`, `test_parcel_embedding.py`, `test_parcel_frames.py`, `test_electrode_pool.py`, `test_dataset.py` (slot-CE exploration), `test_coordinates.py`, `test_cvsavg_spatial.py`, `test_collate.py`, `test_coord_bridge.py`, `test_masked_mean_pool.py`, `test_overfit_smoke.py`. (Quarantine is enforced one-way by `test_no_legacy_imports`; tests-of-archive add no signal.)

Top-level `tests/` becomes empty; remove. Configure `tool.pytest.ini_options.testpaths = ["src"]` in `pyproject.toml`.

---

## Import-graph changes

Every site that does `from speech_decoding.v14.X import Y` needs updating. Categories: internal cross-imports (auto-fixed when files move); `scripts/v14_core/*.py`; `scripts/ablation/submit.py` (builds sbatch commands referencing module paths); `scripts/*.py` (30+ flat scripts — audit + archive in same PR); `docs/CLAUDE.md` + `docs/strategy/stage_*.md` + `docs/references/*.md`; `pyproject.toml` testpaths.

**Rule**: no compatibility shims. No `v14/__init__.py` re-exports. Per `CLAUDE.md`: "Avoid backwards-compatibility hacks."

---

## PR strategy — 4 commits

1. **Skeleton**: create empty target subpackages with `__init__.py`. No moves.
2. **Archive sweep**: move stubs / parity oracles / exploration / audit into `archive/`. Delete the ~13 quarantined-module tests (list above). Update `test_no_legacy_imports.py` to enforce new archive paths.
3. **Bulk move + call-sites (merged)**: atlas + models + training + studies/cogan_ps. Update internal imports + `scripts/v14_core/*.py` + `scripts/v14_core/v14_*_dcc.sh` + `scripts/ablation/submit.py` + flat `scripts/*.py` (audit + archive unused) + `docs/CLAUDE.md` + `docs/strategy/stage_*.md` + `docs/references/*.md`. Smoke: `python -c "from speech_decoding import atlas, models, training, studies"` and `python -m scripts.v14_core.train_v14_core --help`.
4. **Tests + cleanup**: colocate the ~17 active-module tests under their new homes, update `pyproject.toml` `testpaths = ["src"]`, delete empty `v14/` + `data/` + `evaluation/` packages and `tests/v14/`. Run `pytest -q`.

Each commit passes `pytest -q`. (Splitting commits 3+4 from the original plan into Python-imports and CLI-call-sites would have left CLIs broken between them; merged here so end-to-end CLI invocation works after every commit too.)

---

## Verification gates (block PR merge if any fail)

1. `pytest -q` — all tests pass (existing 3 + colocated new ones).
2. `python -m scripts.v14_core.train_v14_core --help` — CLI imports cleanly.
3. `python -c "import speech_decoding; from speech_decoding import atlas, models, training, studies, extractors"`. (`events/` and `ssl/` are deferred per the empty-package rule.)
4. `grep -r "from speech_decoding.v14" src/ scripts/ tests/ docs/` returns nothing.
5. **DCC dry run**: `scripts/ablation/dcc_sync_check.py` passes; one trivial `submit.py` invocation generates valid sbatch.

---

## Coordination with NeuralSet adoption

Reorg + adoption are **sequenced as one continuous arc** (pre-Stage 0):

| Phase | Action | Outcome |
|---|---|---|
| 0 | Smoke test (1-2 days), self-contained under `scripts/scratch/` | API contract confirmed |
| 1 | Python 3.12 bump (1 day), sibling DCC env `speech_py312/` | Env ready, independent of Phase 0 |
| 2 | Reorg PR (~1 week, 4 commits) | New shape lands; adapter slots are empty |
| 3 | NeuralSet adoption PR (~3 days) | `studies/braintreebank/*` + `extractors/parcel.py` populated |
| 4 | Stage 0 begins | Block A starts on new shape |

Reorg-first means the adoption PR is a *small additive diff* against established locations, not a sprawling restructure tangled with new files. Both plans depend on smoke test passing; very high prior on passing per the source-level deep-dive (`reference_neuralset_extractor_pattern_2026_04_26.md`).

---

## Decisions baked into the plan

Per user directive "fully import NeuralSet's structure":

- **Studies layout**: per-cohort folder (`studies/braintreebank/study.py + loader.py + ...`). Each cohort has enough concept files to justify the nesting.
- **Cohort-specific Ieeg subclasses live with their cohort** (`studies/<cohort>/study.py`), not in a shared `events/etypes.py`. Keeps cohort code cohesive.
- **Empty-package rule**: don't create a subpackage until the first real file lands. Applies to `events/`, `ssl/`, future cohort folders.
- **PS-Utah `channels.py` lives in `studies/cogan_ps/`, not `atlas/`.** It's the amp→Utah-grid bridge; sEEG cohorts have no analog. Atlas package stays modality-neutral.
- **Test colocation**: adopt. Migrate the 3 existing test files inward.
- **`models/` naming**: use `models/` (PyTorch convention; NeuralSet has no analog).
- **Drop `v14/` namespace**: yes. The whole codebase is v14; extra namespace level adds no information.
- **`scripts/*.py` flat scripts**: audit + archive to `scripts/archive/` in commit 4 (same PR as call-site updates).
- **Smoke test first**: Phase 0 of integration plan (1-2 days) runs against current `v14/` shape to confirm API. Reorg PR opens after smoke passes.
