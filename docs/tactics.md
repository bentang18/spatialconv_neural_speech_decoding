# Tactics — Concrete Task List

Tactics layer of the triad (objectives → strategy → tactics). Operational: what's running, what to do when it lands, what's blocked. Refreshed 2026-04-26 evening for the **pre-Stage 0 reorg + NeuralSet adoption** sprint.

- Objectives: `objectives.md`
- Active plans: `neuroprobe/{neuralset_integration_plan.md, repo_reorg_plan.md}`
- Project decision rationale: `memory/project_pre_stage0_reorg_neuralset_adoption_2026_04_26.md`
- DCC tooling: `references/dcc_setup.md`

**Current sprint:** pre-Stage 0 — full repo reorganization to a NeuralSet-mirrored layout + full NeuralSet adoption. Lands BEFORE any Stage 0 code. ~2 weeks walltime. PS Stage-2 work paused 2026-04-24; queued for resume after Neuroprobe submission. Archived Stage-2 priorities at the bottom of this doc for resume reference.

---

## Active sprint — pre-Stage 0 reorg + NeuralSet adoption

Sequenced as one continuous arc; each phase bisectable.

### Phase 0 — Smoke test (DONE 2026-04-28)

Verify NeuralSet API contract end-to-end on `BraintreebankIeeg + IeegExtractor + Pulse + Segmenter` before reorg PR opens. **Self-contained under `scripts/scratch/neuralset_smoke_bt.py`** — no `speech_decoding.studies...` imports — survives the reorg, deleted in Phase 3.

- [x] **BT smoke script** at `scripts/scratch/neuralset_smoke_bt.py`. Synthetic `mne.io.RawArray` stub via overridden `Ieeg._read()`; `Segmenter(start=0.0, duration=1.0, trigger_query="type=='Word'")` over 4 Word triggers + 1 BraintreebankIeeg row standardized through `ns.events.utils.standardize_events`; `IeegExtractor(event_types="Ieeg", frequency=2048.0, channel_order="original")` + `Pulse(event_types="Word")`. Verified `batch.data["neural"].shape == (4, 128, 2048)` raw voltage at 2048 Hz.
- [x] **Day-2**: inline `V14ParcelMetadataExtractor(BaseStatic, event_types="Ieeg")` returning `(N_CHANNELS, 5)` per Ieeg event (parcel_id, support, fsaverage_xyz). Verified `batch.data["metadata"].shape == (4, 128, 5)`. Real cache I/O lands in Phase 3 production extractor.
- [x] **Run**: `/tmp/neuralset_scratch/.venv/bin/python scripts/scratch/neuralset_smoke_bt.py` — passes.
- **Findings to fold into Phase 3 production code**: (a) row's `type` column must be the **subclass name** (`"BraintreebankIeeg"`), not `"Ieeg"` — otherwise `Event.from_dict` reconstructs the parent and our `_read()` override is bypassed; (b) events DataFrame must go through `standardize_events()` before `Segmenter.apply()`; (c) `IeegExtractor.event_types="Ieeg"` matches `BraintreebankIeeg`-typed rows via `EventTypesHelper`'s subclass-aware filter — no need to add a literal for the subclass.

### Phase 1 — Python 3.12 bump (DONE 2026-04-28)

Required by `neuralset>=0.1.0`. Fully independent of reorg and smoke.

- [x] **Bump `pyproject.toml`** `requires-python = ">=3.12,<3.13"` (pinned to 3.12 — uv otherwise picked 3.14, drifting from NeuralSet's verified version).
- [x] **Regenerate `uv.lock` + `.venv/`** via `uv sync` (then `uv sync --extra dev` for pytest). Local venv now Python 3.12.12. Resolved versions: torch 2.10.0, mne 1.11.0, mne_bids 0.18.0, transformers 5.4.0, numpy 2.4.3, scipy 1.17.1, scikit-learn 1.8.0, h5py + pyyaml + tqdm refreshed.
- [x] **Existing tiny test suite passes** (local + DCC): `tests/test_phoneme_map.py` + `tests/test_grouped_cv.py` + `tests/v14/test_no_legacy_imports.py` = 65/65 on both. Full `tests/` collection fails on pre-existing pyproject.toml gaps (`nibabel`, `pandas` not declared) — fix in Phase 3.
- [x] **DCC env built** at `/work/ht203/repo/speech/.venv` (Python 3.12.13, uv-managed via `python-build-standalone`). Switched from the planned conda sibling env after DCC's miniconda base rotted (`frozendict` ImportError); uv is more reliable for this use case (single static binary, hermetic Python tree, matches local workflow). Existing `miniconda3/envs/speech/` Python 3.11 env left untouched as fallback through ~2026-05-05. Recipe lives in `docs/references/dcc_setup.md`.
- [x] **Path swap landed** — 58 active files (52 sbatch + `scripts/ablation/_common.py` + `scripts/ablation/submit.py` + 4 doc spots) plus `docs/references/dcc_setup.md`. Replaced `/work/ht203/miniconda3/envs/speech/bin/python` → `/work/ht203/repo/speech/.venv/bin/python` and `/work/ht203/miniconda3/envs/speech/lib` → `/work/ht203/repo/speech/.venv/lib/python3.12/site-packages/torch/lib`. `scripts/archive/legacy/*` (16 files) intentionally untouched (quarantined, no longer executed). Verified: `grep -r envs/speech scripts/v14_core/ scripts/ablation/` returns nothing.

### Phase 2 — Reorg PR (~1 week, 4 commits)

Full restructure to NeuralSet-mirrored layout. Plan: `neuroprobe/repo_reorg_plan.md`. Each commit passes `pytest -q`.

- [ ] **Commit 1 — Skeleton**: create empty target subpackages (`extractors/`, `studies/`, `atlas/`, `models/`, `training/`) with `__init__.py`. No moves. (`events/` + `ssl/` deferred per empty-package rule.)
- [ ] **Commit 2 — Archive sweep**: move stubs / parity oracles / exploration / audit into `archive/{stage2_stubs,parity_oracles,exploration}/`. **Delete** the ~17 quarantined-module tests under `tests/v14/` (calibration, tokenizer, model, decoder, grid_mixer, parcel_embedding, parcel_frames, electrode_pool, slot-CE dataset, coordinates, cvsavg_spatial, audit-suite tests, etc.). Update `test_no_legacy_imports.py` to enforce new archive paths.
- [ ] **Commit 3 — Bulk move + call-sites (merged from old commits 3+4)**: atlas + models + training + studies/cogan_ps. Update internal imports + `scripts/v14_core/*.py` + `scripts/v14_core/v14_*_dcc.sh` + `scripts/ablation/submit.py` + flat `scripts/*.py` (audit + archive unused) + `docs/CLAUDE.md` + `docs/strategy/stage_*.md` + `docs/references/*.md`. Smoke: `python -c "from speech_decoding import atlas, models, training, studies"` and `python -m scripts.v14_core.train_v14_core --help`.
- [ ] **Commit 4 — Tests + cleanup**: colocate the ~17 active-module tests under their new module homes (not 3 — the original count was wrong; full list = the active half of `tests/v14/`); update `pyproject.toml` `tool.pytest.ini_options.testpaths = ["src"]`; delete empty `v14/` + `data/` + `evaluation/` packages; delete now-empty `tests/v14/`. Run `pytest -q`.
- [ ] **Verification gates** all pass: `pytest -q`; `python -m scripts.v14_core.train_v14_core --help`; `python -c "import speech_decoding; from speech_decoding import atlas, models, training, studies, extractors"`; `grep -r "from speech_decoding.v14\|speech_decoding\.evaluation\|speech_decoding\.data" src/ scripts/ tests/ docs/` returns nothing in active code; DCC dry run via `scripts/ablation/dcc_sync_check.py` against the `.venv` env.

### Phase 3 — NeuralSet adoption PR (PAUSED 2026-04-28 evening)

Add NeuralSet adapter files. Plan: `neuroprobe/neuralset_integration_plan.md`. **Paused mid-step** pending full read of the four neuroai docs (NeuralSet / NeuralFetch / NeuralTrain / Exca) at `https://facebookresearch.github.io/neuroai/`. Discovery during Phase 3: NeuralFetch ships `Wang2024Treebank` (= BrainTreebank) pre-registered, which likely supersedes the custom `BraintreebankStudy` written in `eb0910d`. State + resume plan: `memory/project_phase3_pause_neuroai_docs_2026_04_28.md`.

- [x] **Add `neuralset>=0.1.0` (+`nibabel>=5.1`, `pandas>=2.2`) to `pyproject.toml`** + `uv sync` (commit `69a3631`).
- [x] **`src/speech_decoding/extractors/parcel.py`** — `V14ParcelMetadataExtractor(BaseStatic)` reading per-patient support cache + fsaverage coords (commit `eb0910d`).
- [~] **`src/speech_decoding/studies/braintreebank/{study,loader,test_loader}.py`** — written in `eb0910d`, **likely to be deleted on resume** if `Wang2024Treebank` proves equivalent. Architectural decision deferred to post-doc-read.
- [x] **`labels.py` + `manifest.py`** — empty stubs landed in `eb0910d`. Will stay regardless of (above).
- [ ] **Decision on resume**: keep custom `BraintreebankStudy` OR swap to `ns.Study(name="Wang2024Treebank", ...)`. Verify `Wang2024Treebank` returns raw 2048 Hz voltage, no re-reference (PopT-comparability).
- [ ] **Decide on neuralfetch / neuraltrain / exca direct deps** (currently only neuralset declared; exca lands transitively).
- [ ] **Revise `docs/neuroprobe/neuralset_integration_plan.md`** — its "BT is h5, not MNE-readable" friction-finding is stale post-NeuralFetch.
- [ ] **DCC env questions resolved** (3 open at integration plan §Open questions): exca cache location (`CACHE_FOLDER=/hpc/group/coganlab/ht203/exca_cache`) — **must answer before first `dataset.prepare()` invocation on DCC**, otherwise cache lands on auto-purging `/work/`; `MapInfra(cluster="slurm")` partition kwargs; coexistence with `scripts/ablation/`.

**Resume preflight**: verify DCC `uv sync` completed cleanly via `/work/ht203/repo/speech/.venv/bin/python -c "import neuralset, nibabel, pandas, pyarrow; print(neuralset.__version__)"` (sync was running at pause; `pyarrow` was mid-install with broken `__version__` attr).

### Phase 4 — Stage 0 begins

Cleared to start Stage 0 Block A on the new shape. Entry point: `neuroprobe/stage_0.md`.

---

## Reference

- DCC helpers: `scripts/ablation/{submit,status,logs,collect,query,dcc_sync_check,peek}.py`. Each has `--help`.
- DCC setup + rsync recipe: `references/dcc_setup.md`.
- Raw ablation log: `experiments/v14_ablation_log.csv` (authoritative results).
- Submissions ledger: `.ablation_submissions.jsonl` (gitignored; decodes task ids → fold/seed).

---

## Paused — PS Stage-2 (resume after Neuroprobe submission)

Frozen architecture at PS pause = `per_cell + partialconv + pe2d + hierarchical_atlas` @ d=32, depth=3, pool=(4,8). All file paths below assume **post-reorg layout** — substitute when resuming.

### Data unblock (ask / chase Zac)

- [ ] **Localization pipeline for 10 priority lex patients.** Box audit 2026-04-21: only S76/S78/S81 have recons on my mount; the spreadsheet's "Localization" column is empty across all lex rows. Zac said recons exist "back to S73" but they're not under `ECoG_Recon/` or `ECoG_Recon_Full/`. Priority order (best-HG first): **S73, S75, S56, S67, S74, S41, S53, S47, S45, S55**.
- [ ] **Re-check spreadsheet / Box for localization tracking doc.** Scan `CoganLab/preprocessing_documentation/` and `CoganLab/uECoG_Meetings/`.
- [ ] **S52 + S71 usability (Zac checking).** S52 = MFA events in derivatives but no raw in BIDS; S71 = two incompatible events.tsvs, no merged `.fif`. Drop both from Stage 2 if unresolvable.
- [ ] **S41 not in Zac's pipeline back-reach.** Pipeline reaches "back to S73" — S41 from 2022-12-12 won't be picked up. Ask explicitly whether S41 recon can be run as a one-off.

### Architecture close-out (Stage-1 → Stage-2 handoff)

- [ ] **Re-run 7-LH pooled under T3.1 default** (`hierarchical_atlas + partialconv + pe2d`) — Stage-2 baseline row. Copy ckpts to `/hpc/group/coganlab/ht203/stage2_ckpt_t31_7lh/`.
- [ ] **7-LH LOPO under T3.1** after pooled lands.

### Infrastructure (self-contained; unblocked)

- [ ] **28-ARPABET joint label map.** Extend `src/speech_decoding/training/phoneme_map.py`: add lex 28-class index, keep PS 9-class as a subset. Per-task mask for AR decode (9³ for PS, 28³ for lex). Class-frequency probe in eval JSONs.
- [ ] **Lex phoneme-level loader.** Parallel to `studies/cogan_ps/dataset.py` — reads `derivatives/epoch(phonemeLevel)(CAR)/...` from the lex BIDS root. 15/16 lex patients have phoneme-level `.fif` (S71 missing).
- [ ] **Mixed-cohort sampler.** Same-patient-per-batch invariant; cohort set extends to PS ∪ lex.
- [ ] **Continuous-sample loader.** Raw `.fif` for SSL pretrain — `src/speech_decoding/studies/cogan_ps/continuous_dataset.py` (or unified across cohorts via NeuralSet `Ieeg` event).
- [ ] **Per-lex-patient channel bridge.** As each lex recon lands, build `data/channel_maps/<pt>_channelMap.mat` lookup + `data/fsaverage_coords/<pt>_fsaverage_pial.csv` + `data/atlas/support_cache_v2c_snap/<pt>_support_tier1.csv`. Automate with `scripts/v14_core/prepare_new_patient.py`.

### SSL objective (Stage-2 mid-wave, triggered at ≥10-LH)

- [ ] **Choose SSL objective.** Post-NeuralSet adoption: joint `L = L_recon + 1{paired}·L_DSigLIP + 0.1·L_KoLeo` per `docs/neuroprobe/plan.md` Experiment 5 (resolved 4/27 night). `L_recon` = JEPA-family latent prediction (data2vec 2.0 + V-JEPA 2.1). `L_DSigLIP` = brain ↔ frozen **Whisper-large-v3 L8** (~25% depth; default revised 4/27 late-night per `docs/neuroprobe/plan.md` line 17; sweep {L8, L16, L30}) via NeuralSet `Segmenter(extractors={neural, metadata, stim_audio})`. See `neuroprobe/stage_2.md §SSL recipe ablation cells` for full ablation matrix.
- [ ] **Mask generator + reconstruction head**, ckpt interop so SSL ckpt loads cleanly into the per-phoneme loader.
- [ ] **Calibration module stub.** `src/speech_decoding/models/calibration.py` signature-only (no-op default).

### D-cohort / Stage 3 prep (open Nanlin questions)

- [ ] **Laplacian / bipolar reference variant for D-cohort?** Box has CAR / WM / M1 / STG / HIPP / LING. BT cross-subject baseline uses Laplacian re-reference. Default to CAR if no clear match.
- [ ] **MFA / TextGrid / production-WAV location for D-cohort.** `SCRIPTS_USAGE.md` references `D_Data/Phoneme_Sequencing/` not visible at `/datacommons/coganlab/D_Data/` on DCC. If absent, stay continuous-corpus / SSL-only on D side.

Stage-3 prep status: 11/11 tasks landed 2026-04-24 (per `memory/project_seeg_stage3_prep_inflight_2026_04_24.md`). 87 D-pts / 180.59 h is the **4-speech-task subset**; the full D-cohort is **113 D-pts / 384.7 h across 14 BIDS tasks** (134 have recons) — re-derived 2026-06-03, `memory/project_d_cohort_data_inventory_2026_06_03.md`. Ready to consume on resume.

### Backlog (deferred to later stages)

- Architectural re-tests at Stage-2 scale — `P_emb` LOPO, per-electrode d=64, plain hierarchical-alone LOPO. Queued until 17-LH pooled + LOPO lands.
- Phase-2 learned per-patient calibration — enabled by `models/calibration.py` stub.
- RH patient re-inclusion (S22, S58) — Stage 3 with sEEG join.
