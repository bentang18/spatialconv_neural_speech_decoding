# Cross-Patient Speech Decoding from Intra-Op uECOG

## Project

Ben Tang, Greg Cogan Lab, Duke. Collaborating with Zac Spalding. Extending Spalding 2025 (PCA+CCA, SVM/Seq2Seq, 8 patients, 9 phonemes, 0.31 bal. acc.).

**Task**: Non-word repetition (52 CVC/VCV tokens, 3 phonemes each, e.g. /abe/; 9 phonemes). Intra-operative, left sensorimotor cortex, 128/256-ch uECOG arrays. ~1 min utterance/patient. Stimulus-to-response delay 1.1 ± 0.3s (Duraivel 2023); stimulus ~500ms; utterance ~450ms; auditory stimulus ends ~600ms before response onset (t=0).

**Patients**: 11 unique PS (S18 no preprocessed; S36 duplicate of S32): S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62. 46–178 trials/pt, 63–201 sig channels. Per-patient tables: `docs/references/data_reference.md`. Stage-specific patient scope: `docs/strategy/stage_<N>.md`.

## Current focus: Neuroprobe cross-subject hillclimb

PS/uECoG stage program **paused 2026-04-24**. Active work is the Neuroprobe cross-subject leaderboard as external validation of the v14 atlas-anchored thesis. Return to PS after Neuroprobe lands (**submit or abort**).

- **Live plan**: `docs/neuroprobe/plan.md`. Targets: ≥ 0.65 multi-class cross-session + beat finetuned PopT by ≥ 0.05 + ≥ 4 tasks pre-baked + K-fold/chronological splits + ≤ 30M params. Stretch ≥ 0.70. Benchmark reference: `docs/references/neuroprobe_benchmark.md`.
- **S2/trial-4 CrossSubject + 120-electrode Lite cap are leaderboard-parity cells, not architecture-selection defaults.** Pooled multi-source CrossSubject multiclass is the scientific generalization default.
- **Live status**: `MEMORY.md §Status (live)` — frozen contracts (L.1, L.2), in-flight jobs, blockers.
- **NeuroAI substrate**: active path is `Study → Events DataFrame → Transforms/Chain → Segmenter → Dataset/DataLoader → NeuralTrain Experiment → Exca`. Custom code is guilty until proven necessary. Keep BNA/fsaverage/support logic, v14 parcel metadata, BrainTreebank labels/splits/leakage transforms, v14 architecture, and the ablation-log export adapter. **Do not** keep old loaders, old training loops, old sbatch tooling, or archived experiments in active import paths.
- **Raw-voltage gate**: `Wang2024Treebank` proof passed on DCC 2026-04-29 (`reports/neuroai_raw_voltage_proof_2026_04_29/`).
- **Naming gotcha**: "Stage 1" inside `docs/neuroprobe/plan.md` = *hillclimb* stage. Elsewhere "Stage N" = PS-program stage.
- **v14 is the sole active architectural direction.** v12, Conv2d pipeline, v12-era Brain-JEPA, LeWM, LOPO autoresearch — all discontinued. *JEPA-family note*: "Brain-JEPA discontinued" refers to the v12-era prototype only; Stage-2 SSL `L_recon` IS JEPA-family (data2vec 2.0 + V-JEPA 2.1 latent prediction).
- **Provenance** (1-line lineage): PopT (zero per-subject params, ICLR 2025 Oral) → BaRISTA (parcel-level encoding > channel-level, NeurIPS 2025 poster) → Evanson (cross-modal SSL paradigm; rejected, right paradigm wrong arch) → v14's novel piece = multi-FM extension on PopT's zero-per-subject architecture. Diagnosis: `memory/reference_evanson_lost_to_popt_diagnosis_2026_04_26.md`.

**Load on demand for v14 architectural / paper-framing work** (don't keep in always-on context):
- Arch spec: `memory/project_v14_parcel_token_readout_2026_04_26.md` (Perceiver IO + parcel-id-tagged latents + Graphormer support bias + DETR readout + Stage-2 SSL contract).
- Paper claim: `memory/project_v14_unique_contribution_2026_04_26.md` (3 commitments, retroactive-validation hooks).
- Paper-framing live: `memory/project_v14_paper_corrections_post_newpapers6_batch2_2026_05_09.md` (Goldstein-2025 scope; Whisper-L8 = acoustic-phonetic NOT semantic; defensive language playbook; 5 ablation cells; Podcast pretrain-with-sister-run policy).
- Stage-2 schedule: `memory/project_stage2_ssl_initial_diet_bt_only_joint_step1_2026_05_09.md`.

## PS program (paused — won't resume until Neuroprobe closes; ~summer of work away)

Don't read PS-resume context until Neuroprobe submits/aborts. Pointers when you do: `memory/project_phase1_default_vs_phase15_target_2026_04_19.md` (PS-resume defaults: per_cell + partialconv + pe2d + hierarchical_atlas @ d=32, depth=3, pool=(4,8); Utah grid is real on PS uECoG so partialconv stays — Neuroprobe-path's BNA-connectivity bias / flat pool / no partialconv are sEEG-specific, two paths diverge by modality not version), `docs/objectives.md` (4-stage roadmap; Stage 1 closed 2026-04-20, Stage 2 paused 2026-04-24, Stage 3+ TBD), `docs/strategy/stage_<N>.md`, `docs/tactics.md`. v14 calibration scope: fixed atlas through PS-Stage 2; learned calibration deferred to PS-Stage 3+.

## v14 Paper Authorship — IRONCLAD

Ben drafts the bulk of the v14 preprint/paper. Claude does **not** bulk-draft paper prose. After Ben has written the bulk, we iterate surgically piece-by-piece. In that surgical mode, every citation, every number, every empirical claim Claude touches must be rigorously double-checked against a real source. **NO EXCEPTIONS.**

Mechanical guard: `.claude/hooks/paper_guard.py` (PreToolUse) fires on every Claude Write/Edit/MultiEdit/NotebookEdit to `paper/neuroprobe-hillclimb/`. Three layers: hard-block on meta-LLM phrases, hard-block on bulk drafting (>25 net lines added), force-ask permission prompt on all surviving edits so Ben visually approves every diff regardless of auto-accept mode. No env-var bypass — for a legitimate large edit, Ben writes it directly via his editor (the hook only intercepts Claude's tool calls).

Full rule + verification checklist: `memory/feedback_arxiv_llm_content_responsibility_2026_05_16.md`. Floor: arXiv policy (Dietterich 2026-05-15) — un-checked LLM output → 1-year ban + peer-review requirement on subsequent submissions. Ceiling: the paper is Ben's; voice, framing, and claim structure come from Ben.

## Working Principle: Discuss Before Code

**v14 is slow, methodical, and precise.** Discuss every pipeline piece — assumptions, I/O, why right for this data, what would make it wrong, precedent and trade-offs — before any code:

1. **Agree on the contract.** Inputs, outputs, shapes, units, boundaries. Ambiguity is a blocker.
2. **No pre-committed numeric defaults.** Window sizes, `d_model`, widths, thresholds — all justified before landing.
3. **Rewrite from scratch when needed.** No file is sacred. If handwavy, rewrite — don't patch.
4. **No legacy reuse, ever.** Old code lives in git history and the external reset backup under `/Users/bentang/Documents/Code/backups/`. Re-derive fresh; don't copy back without a specific, reviewed reason.
5. **Freeze blockers before coding.** See `docs/tactics.md`.
6. **Prefer standard, scalable contracts** when they don't compromise current-stage correctness.

Applies to every logic step — channel indices, coordinate frames, Brainnetome PM lookup, per-electrode support, parcel embedding, temporal front-end, backbone, decoder, loss, eval, metrics. No step is too obvious to skip.

## Engineering Discipline

### 1. Think before coding
- State assumptions. If uncertain, ask.
- Present multiple interpretations — don't pick silently.
- Push back when a simpler approach exists.
- If unclear, stop. Name what's confusing.

### 2. Simplicity first
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" that wasn't requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite.

### 3. Surgical changes
- Don't "improve" adjacent code.
- Don't refactor what isn't broken.
- Match existing style.
- Unrelated dead code: mention, don't delete.
- Remove only what *your* changes made unused.

### 4. Goal-driven execution
Define success criteria. Loop until verified.
- "Add validation" → "write tests for invalid inputs, then make them pass"
- "Fix the bug" → "write a test that reproduces it, then make it pass"
- "Refactor X" → "tests pass before and after"

For multi-step tasks, state a brief plan with a verification check per step.

### 5. Verify before claiming done
Never claim tests pass, code works, or a bug is fixed without running the verification command in the current turn. "Should work," "probably fixed," "seems to pass" are lies, not confidence. No "Great!"/"Perfect!"/"Done!" before evidence.

### 6. Debug root cause, not symptoms
Read the full error, reproduce, check recent changes, log at every boundary before proposing a fix. "Just one quick fix" masks the real issue.

### 7. No performative agreement on pushback
When the user questions a design, do not capitulate with "You're absolutely right!" and start rewriting. Restate the technical point, verify against the codebase, and push back with reasoning if the feedback is wrong for this context. Technical correctness over social comfort.

### 8. Empirical iteration over armchair design
Discussion surfaces the known unknowns and the contract, but once done, *run the experiment*. Ablation deltas, LOPO numbers, loss curves beat another reasoning pass. When brainstorms stack elaborate architectures on shaky citations, stop reasoning and measure.

## Environment

- **Python ≥3.12, `uv`-managed.** `pyproject.toml` + `uv.lock` authoritative.
- **Bootstrap**: `uv sync` creates `.venv/`. Run as `.venv/bin/python -m ...` — do not `source activate`.
- **Tests**: `.venv/bin/python -m pytest -q`. Tests colocated under `src/`.
- **NeuroAI substrate**: `neuralset[all]`, `neuralfetch`, `neuraltrain`, `exca` are explicit dependencies. NeuralSet owns data orchestration; NeuralTrain/Exca own training runs, caching, grids, DCC dispatch. Canonical sweep dispatch is `TaskInfra + neuraltrain.utils.run_grid` (one Slurm array per sweep), not `MapInfra` — `MapInfra` is for extractor-layer per-subject precompute.
- **Machine-specific paths**: `configs/paths.yaml` (gitignored).
- **All training on DCC, never local.**
- **Exca cache folder**: must point at `/hpc/group/coganlab/ht203/cache_neuroai/` (persistent), never `/work/ht203/` (75-day purge). Set via `EXCA_CACHE_FOLDER` env var or per-`infra.folder=`.
- **Upstream Neuroprobe clone**: pinned at `c7b955b0a31464f4a5eec3f3bd78ff29841d61ac`. Local: `.cache/neuroprobe_upstream/`. DCC: `/work/ht203/repo/neuroprobe_upstream/`. Stage-0 wrappers default automatically. Bundled `braintreebank_features_time_alignment/*.csv` enables laptop-side audits without BT data. Full reference + bootstrap: `memory/reference_neuroprobe_upstream_api.md`.
- **`Wang2024Treebank` upgrade-path**: when upstream NeuralFetch ships its own `Wang2024Treebank`, our local registration will collide on `STUDY_PATHS` import — rename local class to private name (keep `name="Wang2024Treebank"` ClassVar) or add precedence handling.

## Baseline to beat

PER 0.734 ± 0.007 on S14 (grouped-by-token CV, 3-seed, per-phoneme MFA + flat head). Population mean 0.825 / 11 patients. Details: `docs/archive/experiment_log.md` findings 86–101.

## Key Files

### Data (repo)
- `data/ps_tokens.csv` — canonical 52-token PS manifest.

### Data (local, gitignored)
- `data/mni_coords/<subj>_RAS.txt` — ACPC electrode coordinates (11/11).
- `data/channel_maps/<subj>_{channelMap,sigChannel}.mat` — amp→grid + sig-channel masks (sig: 9/11; missing S32/S57).
- `data/transforms/<subj>_talairach.xfm` — ACPC → Talairach/MNI.
- `data/atlas/BNA_PM_4D.nii.gz` — Brainnetome PM volume (only v14 atlas source).
- `data/atlas/fsaverage_bake_v2c/` — baked fsaverage atlas (projfrac-avg, no smoothing).
- `data/atlas/support_cache_v2c_snap/<pt>_support_tier1.csv` — per-electrode Tier-1 BNA support.
- `data/fsaverage_coords/<pt>_fsaverage_pial.csv` — strict snap-to-pial coordinates.

### Box mount (macOS laptop)
- Root: `/Users/bentang/Library/CloudStorage/Box-Box/`
- FreeSurfer recons: `ECoG_Recon/<pt>/` (PS = `S<num>`). ACPC source: `<pt>/elec_recon/<subj>_elec_locations_RAS_brainshifted.txt`.
- `D<num>` = different Duke sEEG cohort (Stage 3 scope) — do not confuse with `S<num>`.
- Avoid sibling recon folders (`_old`, `_no_tkr`, `_kumar`, `_diag`, `_med`); always use plain `S<num>/`. `cvs_avg35_inMNI152/` = parity-oracle template only, not active.

### BrainTreebank raw data
DCC-only at `/work/ht203/data/braintreebank/`. `ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank` for any DCC-side script touching neural data.

### Docs organization (STRICT — enforce, don't dilute)

`/docs` follows the Sun Tzu triad: **objectives → strategy → tactics**. Exactly **three working docs**. Do not add a fourth.

- `docs/objectives.md` — program hypothesis, stage roadmap, evaluation philosophy, advance gates.
- `docs/strategy.md` (index) + `docs/strategy/stage_<N>.md` (per-stage) — default architecture, frozen contract, scoreboard, rejected paths. Stage-N is written only when Stage-(N−1) has concluded enough to define the entry point.
- `docs/tactics.md` — concrete task list, in-flight jobs, blockers.

**Rule**: do not create additional planning, tracker, or status docs under `/docs` for the PS stage program. Extend the relevant triad doc instead. Doc surplus breaks the organization and creates stale duplicates (paid this cost multiple times). The triad above supersedes deprecated forms — do not recreate `current_direction.md`, `implementation_tasks.md`, or `v14-core.md` (all archived). Parallel initiatives (e.g. `docs/neuroprobe/`) get their own subdirectory with a single active plan doc.

**Reference docs** (static): `docs/references/{data_reference,dcc_setup,neuroai_reference,neuroprobe_benchmark}.md`. **Experiment exports**: `docs/experiments/README.md` (pre-reset logs archived under `docs/archive/experiments/pre_neuroai_reset_2026_04_29/`). **QC + figures**: `docs/qc/`, `docs/figures/` (history under `docs/archive/`).

**Reports** (`reports/`): point-in-time audit/scoping artifacts. Archive to `reports/archive/` when findings have migrated to memory or strategy.

## Code Structure

Active module layout under `src/speech_decoding/` after the NeuroAI reset:

- **`extractors/`** — mirrors `neuralset.extractors`. `parcel.V14ParcelMetadataExtractor`, `reference.py` post-extractor reference transforms.
- **`studies/braintreebank/`** — local NeuralFetch-style `Wang2024Treebank(study.Study)`. Public API: `ns.Study(name="Wang2024Treebank", ...)`. Plus `manifest.py` (BT_NANO/LITE/FULL_SESSIONS), `labels.py`, `anatomy.py`.
- **`atlas/`** — atlas + parcel infrastructure. `fsaverage.py` (strict snap-to-pial + atlas loader + support helpers), `support.py`, `tokens.py`.
- **`experiments/`** — NeuralTrain/Exca scaffolding (`Data`, `Experiment`, `BrainModule`, `ExperimentLogger`, `collect_experiment_records`).
- **`models/`** — empty shell until v14 Perceiver IO lands.
- **Training** — do not recreate the old `training/` package. Add NeuralTrain pydantic `Data` and `Experiment` classes (already scaffolded under `experiments/`) when Stage 0 training begins.

**Empty-package rule**: don't create a subpackage until the first real file lands. Documented intents:
- `events/` — first cross-cohort event subclass or transform.
- `ssl/` — Stage 2 SSL kickoff (`d_siglip.py`, `projection.py`, JEPA-target generation, EMA teacher).
- `studies/cogan_seeg/`, `studies/cogan_lex/` — Stage-3 / Stage-2 PS-extension.

Tests colocated next to modules. Active package stays small: NeuroAI integration plus v14 science only. Reorg blueprint: `docs/neuroprobe/repo_reorg_plan.md`. Adapter spec: `docs/neuroprobe/neuralset_integration_plan.md`.

**Retired-script anti-pattern**: do **not** revive `scripts/v14_core/`, `scripts/ablation/`, or `scripts/archive/` — backed up externally, removed from active tree. Future training dispatch belongs in NeuralTrain/Exca. The active proof harness `scripts/neuroprobe/prove_wang2024treebank_raw_voltage.py`, the four `scripts/dcc/*` helpers, and `scripts/git/worktree-sweep` are the only sanctioned operational scripts.

## Compute: Duke DCC cluster

All training on DCC, never local. Repo: `/work/ht203/repo/speech`. Python: `.venv/bin/python` (uv 3.12, no conda). SSH: `ssh ht203@dcc-login.oit.duke.edu`.

**DCC helpers — prefer these over manual git/ssh/sbatch chains.** When the session involves syncing, dispatching, monitoring, or rerunning DCC jobs, USE the four helpers under `scripts/dcc/`:

- `scripts/dcc/sync` — push current branch + reset DCC clone to it. Replaces `git push && ssh dcc 'git fetch && git reset --hard origin/<branch>'`.
- `scripts/dcc/dispatch <submitter> [args...]` — sync, then SSH-run `.venv/bin/python <submitter> [args...]` on DCC. Use `--help` first as a "does this even import on DCC" preflight.
- `scripts/dcc/status [report-glob]` — `squeue -u ht203` + `status_l_sweeps.py` per report dir.
- `scripts/dcc/rerun-failed <report-dir> [--mem 64G]` — rescue OOM/traceback jobs.

**Load `docs/references/dcc_setup.md` on demand** when first touching any of these helpers, `submit_*` scripts, or DCC paths in a session — full cheatsheet (identity+paths, env vars `ROOT_DIR_BRAINTREEBANK`/`EXCA_CACHE_FOLDER`, sbatch template, `/work` 75-day purge rule, NeuroAI dispatch example). Skip when the session has nothing to do with the cluster.

**Sweep discipline — close every dispatch loop.** A sweep started and never analyzed is the default failure mode; these rules force the loop shut.
- **Never commit on the DCC clone.** It is a `git reset --hard` target — commits made there are silently destroyed on the next `scripts/dcc/sync`. `sync` now aborts if it detects them (push from DCC, or rerun `--force`). Always commit on the laptop → push → sync.
- **No sweep without a collector.** A `submit_*` script is not dispatched until its paired `collect_*`/`analyze_*` exists. A sweep is not *done* until results are pulled, analyzed, and recorded under `docs/experiments/`.
- **Every dispatch is logged.** `scripts/dcc/dispatch` appends a row to `docs/experiments/dispatch_log.csv`. Rows still `dispatched` are open loops — clear them (pull + analyze, set `analyzed`) before launching new sweeps.

## Preprocessing Pipeline (do not change)

Decimate 2kHz → CAR → impedance exclusion (log10 > 6) → 70–150 Hz Gaussian filterbank (8 bands) → Hilbert envelope → sum → 200 Hz → z-score → sig-channel selection. In `coganlab/IEEG_Pipelines`. **Z-score is per-channel mean/std pooled across all pre-auditory baseline trials + samples** (500 ms window immediately before auditory-stim onset; verified 2026-04-18 via reconstruction test on 7 patients, corr = 1.0000). NOT per-trial and NOT pre-production. **Recording-level median/MAD ≡ this recipe up to per-channel affine** (ρ=1.0000 across tested patients) — SSL (Stage 2) can swap recipes without bit-exact constraints. Details: `docs/references/data_reference.md`.

## Conventions

- **Write simply.** Ordinary words. Short sentences. No throat-clearing, no redundant qualifiers, no ceremonial preambles. Cut is the main edit. Applies to docs, commits, PRs, memory files, chat. Paul Graham's "Write Simply" is the reference. Code comments stay minimal (default: none).
- **Discuss logic before writing code.** See Working Principle.
- **All training on DCC, never local.**
- **Every architectural change reports both pooled joint AND LOPO warm-start.** `docs/objectives.md §Evaluation philosophy`. LOPO is the foundation-model test; load-bearing for Stage-2 SSL, Stage-3 cross-sensor transfer, Stage-4 external-corpus transfer. **Single-protocol evidence does not justify defaulting an arch change.**
- **Always export DCC results into `docs/experiments/`.** Every finished DCC run needs a durable record because `/work/ht203` auto-purges every 75 days. The old CSV + aggregator were retired; define the NeuralTrain/Exca export schema before the first Stage-0 result lands.
- **Tear down agent worktrees on merge.** When a `.claude/worktrees/` branch lands on main, `git worktree remove` + `git branch -d` in the same step. Run `scripts/git/worktree-sweep` periodically to clear merged + clean worktrees.
