# Cross-Patient Speech Decoding from Intra-Op uECOG

## Project

Ben Tang, Greg Cogan Lab, Duke. Collaborating with Zac Spalding.
Extending Spalding 2025 (PCA+CCA, SVM/Seq2Seq, 8 patients, 9 phonemes, 0.31 bal. acc.).

**Task**: Non-word repetition (52 CVC/VCV tokens, 3 phonemes each, e.g. /abe/; 9 phonemes). Intra-operative, left sensorimotor cortex, 128/256-ch uECOG arrays. ~1 min utterance/patient. Stimulus-to-response delay: 1.1 ± 0.3s (Duraivel 2023); stimulus ~500ms; utterance ~450ms. Auditory stimulus ends ~600ms before response onset (t=0).

**Patients**: 11 unique PS (S18 no preprocessed; S36 duplicate of S32): S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62. 46–178 trials/pt, 63–201 sig channels. Per-patient tables: `docs/references/data_reference.md`. Stage-specific patient scope in `docs/strategy/stage_<N>.md`.

## Current focus: NeuroAI-first clean reset

**EXECUTING NOW (pre-Stage 0)**: NeuroAI is the default repo architecture. The active path is `Study -> Events DataFrame -> Transforms/Chain -> Segmenter -> Dataset/DataLoader -> NeuralTrain Experiment -> Exca`. Custom code is guilty until proven necessary. Keep BNA/fsaverage/support logic, v14 parcel metadata, BrainTreebank labels/splits/leakage transforms, v14 architecture, and the ablation-log export adapter. Do not keep old loaders, old training loops, old sbatch tooling, or archived experiments in active import paths.

The raw-voltage gate for the local NeuralFetch-style `Wang2024Treebank` study passed on DCC on 2026-04-29. The proof compared NeuralSet/Segmenter/`IeegExtractor` against `neuroprobe.BrainTreebankSubject.get_all_electrode_data()` for subjects 1, 2, and 10 across early/middle/late/word-aligned 1 s windows. All rows matched exactly after float32 casting with `2048.0` Hz sampling and aligned channel labels. Report: `reports/neuroai_raw_voltage_proof_2026_04_29/`.

## Underlying focus: Neuroprobe cross-subject hillclimb (resumes after reorg)

**PS/uECoG stage program is paused 2026-04-24.** Active work is the Neuroprobe cross-subject leaderboard as external validation of the v14 atlas-anchored thesis. Return to the stage program after Neuroprobe lands (submit or abort).

**Live plan + rationale**: `docs/neuroprobe/plan.md`. Targets (recalibrated 2026-04-30 post-reviewer concerns): **≥ 0.65 multi-class cross-session to submit + pooled multi-source CrossSubject multiclass as the scientific generalization default + beat finetuned PopT by ≥ 0.05 + ≥ 4 tasks pre-baked + K-fold/chronological splits + ≤ 30M params**. S2/trial-4 CrossSubject and the 120-electrode Lite cap are leaderboard-parity cells, not architecture-selection defaults. Stretch ≥ 0.70. Benchmark reference: `docs/references/neuroprobe_benchmark.md`.

**Active v14 default (Neuroprobe path, revised 2026-04-30 — Perceiver IO with M-latents-per-parcel + Graphormer-style support bias + artifact-aware input views + patient-invariant SSL + task-attention readout)**: framed as a **source-separation problem**: voltage at electrode i is a noisy linear mixture of latent parcel-level activity (`v[i,t] = Σ_p L[i,p] · s[p,t] + noise`); architecture maps observed mixtures (electrodes) → latent sources (parcel latents). Multiclass is the default task formulation. Pooled multi-source CrossSubject multiclass is the scientific generalization default; S2/trial-4 CrossSubject is parity only. Raw 2048 Hz voltage is no longer artifact-neutral or guaranteed default. Stage 1 must compare raw, local/Laplacian raw, local/Laplacian STFT/log-power, local/Laplacian HG/HFA, and CAR+HG. Reference is a first-class transform because it changes the measurement operator; transformed support/virtual-channel metadata must travel with rereferenced data. Temporal tokenizer = **single-layer 1D conv with kernel=patch_size, stride=patch_size** by default, with BaRISTA dilated CNN, HG envelope, wavelet, and Metzger-style short-stride cells retained. **Architecture = Perceiver IO with parcel-id-tagged latents**: `parcel_latents[p, m] = P_emb[p] + M_emb[m]` for p ∈ {1, ..., ~50 BNA Tier-1 parcels with corpus coverage} × m ∈ {1, ..., M} sub-slots **(M ∈ {2, 4, 8} swept, default M=4)**; cross-attention from parcel latents to electrode tokens with **`log(support[i,p] + ε)` as Graphormer-style attention bias on QK score** (Option Z, NOT additive embedding on electrode tokens). **Always include all ~50 parcel latents**, but loss masks distinguish `covered`, `masked-covered`, and `uncovered`; uncovered parcels do not receive fake JEPA teacher targets. Stage-0 A4 parcel co-coverage graph bounds any cross-parcel completion claim. **Readout = DETR-style task-attention pool** over P×M parcel latents. No per-subject linear layer. **Stage-2 SSL**: `L = L_content_JEPA + L_view_invariance + 1{paired}·L_DSigLIP + 0.1·L_KoLeo (+ optional nuisance suppression if probes fail)`. `L_content_JEPA` is data2vec 2.0 + V-JEPA 2.1 latent prediction, but direct loss applies only to `masked-covered` parcels and never reconstructs raw voltage by default. `L_view_invariance` forces the high-level parcel/global content readout to agree across valid measurement views, while low-level electrode tokens may differ. `L_DSigLIP` = brain ↔ frozen **Whisper-large-v3 L8** on stim-aligned batches and is the primary patient-invariant **speech/acoustic-phonetic** anchor (per BrainWhisperer/WhisperBCI proportional-depth precedent; Whisper encoder mid-layers carry acoustic-phonetic content, NOT semantic content — per Goldstein 2025, semantic content in Whisper lives in *decoder* mid-late layers, encoder top maps to STG/SM, decoder L3 maps to IFG/AG/pMTG. "Semantic anchor" wording in pre-2026-05-09 docs is a misnomer). Genuine semantic anchors (Whisper-large-v3 decoder mid-layer, GPT-2 mid-layer, Llama-3.2-mid) remain Stage-3 multi-FM extension targets, not Stage-2 default. Mandatory probes: subject/session/reference/input-view/coverage decoding, line/common-mode decoding, timing proxies. Charmander-style raw masked-channel-recon remains a Track-2 intrinsic-SSL ablation and artifact stress test, NOT default. **Stage-3 SSL**: extend the same invariant contract to multi-FM (audio + vision via DINOv3 + optional language via GPT-2), with per-corpus co-coverage audits and nuisance probes; no anatomy-routed loss, let attention learn FM routing and validate via Goldstein-style cortical mapping.

**Paper-level architectural commitment (canonical source: `memory/project_v14_unique_contribution_2026_04_26.md`)**: three minimal commitments, no baked priors — (1) zero per-subject parameters [PopT-inherited], (2) shared anatomical *coordinate frame* (BNA parcel labels [BaRISTA-inherited]) PLUS *functional alignment* via parcel cross-attention with shared `P_emb[p]` across subjects + `log(support[i,p])` Graphormer bias [SRM-analog mechanism per Bhattacharjee/Nastase 2024 — anatomy labels alone = PCA-control analog; cross-attention is what does the alignment work], (3) multi-FM cross-modal SSL targets [Evanson-paradigm extended]. The neuroscience contribution: **cortical-FM functional specialization (Goldstein 2025) and BNA cortical-cortical connectivity emerge from data alone**, validated retroactively as the *prediction* that v14's parcel-latent self-attention map will recover the encoder-L4 → STG/SM, decoder-L3 → IFG/AG/pMTG gradient observed in Goldstein 2025. *A neuroscience finding via an ML architecture.*

**Honest scope of the two distinct Goldstein 2025 papers** (per `memory/reference_goldstein_2025_n4_specifics.md` and `memory/reference_goldstein_2025_nat_commun_temporal_hierarchy.md`): (1) **Goldstein 2025 Nature** = n=4 pts, electrode-level per-subject linear ridge, Whisper *medium* encoder L4 + decoder L3 cortical mapping — observational, NOT cross-subject, NOT parcel-level, NOT Whisper-large-v3 L8. (2) **Goldstein 2025 Nat Commun** = n=9 ECoG pts, GPT-2 XL (48 layers, absolute index 1-48) + Llama-2, layer-index ↔ peak-encoding-lag correlation r=0.85 IFG / r=0.92 aSTG / r=0.93 TP / mSTG flat under Pearson (r=−0.24, p=0.09) but weakly significant under 100k permutation (p<0.02) — primary retroactive-validation target for v14's parcel-latent self-attention depth ↔ time-after-onset gradient prediction. **Second independent retroactive hook from same paper**: inverted-U over depth (peak encoding correlation at intermediate layer 22 of 48); v14 should predict its own encoding peaks at intermediate v14 depth (≈ 3 of 6). Cross-subject statistic = LMM `lag ~ layer + ROI + (1+layer|electrode) + (1+layer|participant)` — v14 must replicate this model to claim cross-subject gradient. v14's Whisper-large-v3 L8 ≈ 25% depth choice triangulates from **3 sources**: BrainWhisperer/WhisperBCI proportional-depth precedent + Antonello 2023 NeurIPS ("Whisper upper-mid best for brain encoding") + Hong 2024 ("best layer shifts earlier with model scale"). NOT from Goldstein 2025-Nature L4 (Whisper-medium, different model, different layer-percentage). Bhattacharjee 2025 (Nat Comp Sci, 8-pt ECoG SRM, +37% encoding) is the closest *direct* v14 precedent — same Podcast data, same modality, same lab cluster; v14's delta is parcel-cross-attn-during-pretraining vs their post-hoc SRM. **Two head-to-head bars to beat**: (a) +37% encoding improvement (r=0.188→0.257); (b) **93% time-segment classification at 40-word segments (Bhattacharjee Suppl Fig 5)** vs PCA-control 37%, chance 1.6%. Multi-FM SSL precedent = Tang 2023 NeurIPS BridgeTower (5-subj fMRI story↔movie cross-modal transfer outside V1/AC) — v14 inherits "frozen multi-FM features as stim bank" only; **brain-side learned cross-modal SSL is v14's own piece** (Tang's method = L2 ridge on frozen BridgeTower, NOT learned SSL). Retroactive-validation menu: **primary** Goldstein 2025-NC layer-time + inverted-U; **secondary** Caucheteux & King 2022 visual→lexical→compositional (DEMOTED from co-primary — silent reading text, modality mismatch with v14 speech anchor); Kumar 2024 attention-head functional specialization (interpretability template). **Pre-register variance-partitioning analysis** (de Heer 2017 method) on parcel-latent self-attention vs Whisper-L8 features at matched lag — methodologically dominant skeptic-defensible test. Skeptic citation: Antonello & Huth 2024 NoL — frame v14 as "FM features happen to capture right phenomena", NOT predictive-coding evidence. **Defensive language playbook (cite in Discussion only, not abstract)**: AVOID "predictive", "predictive coding", "the brain predicts", "anticipatory", "compositional hierarchy emerges", "evidence for prediction", "prediction error"; ADOPT "feature discovery", "representational generality", "FM features happen to capture phenomena the brain encodes", "depth-gradient correspondence; we do not claim the brain trains on next-word prediction". **Untrained-net caveat (Caucheteux & King 2022)**: random-init networks produce significant brain scores (R=0.018-0.019, p<10⁻¹⁶); v14 anatomy-blind random-Perceiver ablation must beat random by matched-rank gap, not assume zero floor. **Five v14 ablation cells** from this audit (`memory/project_v14_paper_corrections_post_newpapers6_2026_05_09.md` + `memory/project_v14_paper_corrections_post_newpapers6_batch2_2026_05_09.md`): **(a) FM-swap** Whisper-L8 → HuBERT/WavLM/EnCodec (Conwell "diet > arch" test); **(b) frozen-features linear probe** Whisper-L8 → linear → Neuroprobe (must be beaten — Conwell veRSA caution; protocol modeled on Tang 2023 ridge+FIR); **(c) anatomy-blind random Perceiver** at matched (M·d), no `P_emb`, no `log(support)` (SRM PCA-control analog); **(d) P_emb drift** — unfreeze BNA-init `P_emb[p]`, keep `support[i,p]` fixed (Cogan functional-vs-anatomical-alignment); **(e) post-hoc SRM baseline** k=5 on raw HFB → linear → Neuroprobe (Bhattacharjee 2025 closest precedent; v14 must beat). (a)+(c)+(d) triangulate: anatomy as routing+content (full v14) vs routing-only (P_emb drift) vs neither (anatomy-blind). **Podcast (ds005574) policy (revised 2026-05-09 evening — flipped from eval-only to pretrain-with-sister-run)**: Podcast IS in v14 Stage-2 pretraining (4.5h, ~1% of Tier-0 corpus) AND v14 pre-bakes a sister `--exclude-podcast` retraining run. Both models evaluated on (a) Bhattacharjee 2025 head-to-head (184-electrode language-sensitive mask, 10-fold contiguous-segment temporal CV, GPT-2 XL embeddings → 50d PCA, ±2000 ms lag grid 25 ms steps; bars to beat: +37% encoding r=0.188→0.257 + 93% time-segment classification at 40-word segments) and (b) Goldstein 2025-NC retroactive layer-time gradient (LMM `lag ~ layer + ROI + (1+layer|electrode) + (1+layer|participant)`). Sister run is the leakage defense — if both models recover the layer-time gradient, the gradient is genuine cross-subject signal, not Podcast memorization. Rationale: Bhattacharjee's own protocol pretrains SRM on Podcast (10-fold CV), so head-to-head requires v14 also see the data; pre-bake leakage defense (CBraMod TUEV ⊂ TUEG playbook) keeps retroactive-validation citations defensible. License = CC-BY-NC-ND 4.0 (academic OK; flag for downstream weight-distribution review). Same stim used by entire Hasson-lab ECoG canon (Bhattacharjee/Hong/Goldstein-NC/Kumar/Zada/Goldstein-2022/2024). Spec: `memory/project_v14_p_emb_drift_ablation_2026_05_09.md` + Podcast pretrain-pair added to `memory/feedback_pre_bake_data_leakage_defense.md`.

**Provenance (PopT → BaRISTA → Evanson + multi-FM extension)**:
- **PopT** (Oral 8/8/8/8): zero per-subject params + retroactive interpretability template
- **BaRISTA** (poster 5/4/5/3): parcel-level encoding > channel-level for cognitive tasks
- **Evanson** (rejected 4/4/2/2 — but right paradigm on wrong architecture): cross-modal SSL paradigm
- **v14's novel piece**: multi-FM extension (audio + vision + language) on top of PopT's zero-per-subject architecture

PopT's intrinsic SSL beat Evanson's cross-modal SSL on Evanson's own data (0.06–0.18 gap, rebuttal record) NOT because intrinsic SSL is better, but because Evanson inherited d'Ascoli's per-subject-layer architecture which prevented cross-subject pretraining. v14 fixes this by combining PopT's architecture with Evanson's SSL paradigm. Diagnosis: `memory/reference_evanson_lost_to_popt_diagnosis_2026_04_26.md`.

Earlier framing notes (v14 re-eval 2026-04-25 specifics): `memory/project_v14_reeval_kinglab_eegfm_2026_04_25.md`.

The "Stage" language inside `docs/neuroprobe/plan.md` refers to *hillclimb stages*, not the PS program's stages. When this doc says "Stage 1", it means PS stage unless inside a Neuroprobe context.

## v14 program (paused — resumes after Neuroprobe hillclimb)

Atlas-grounded parcel tokens as the shared representation across patients and sensors. Two-problem decomposition:
1. **Calibration** (per-patient, physics-constrained): raw electrodes → atlas-grounded tokens via Brainnetome surface parcellation on fsaverage. Fixed atlas through PS-Stage 2; learned calibration deferred to PS-Stage 3+.
2. **Dynamics** (shared, unconstrained ML): tokens → phoneme sequence via a small relational-temporal transformer + AR decoder. Same representation for every patient.

**Triad doc layer** (for PS stage work — resumes after Neuroprobe):
- **Objectives** (program hypothesis + stage roadmap + advance gates): `docs/objectives.md`
- **Strategy** (per-stage architecture, frozen contract, scoreboard, rejected paths): `docs/strategy.md` → `docs/strategy/stage_<N>.md`
- **Tactics** (in-flight jobs, blockers, next actions): `docs/tactics.md`
- **Experiment exports**: `docs/experiments/README.md` (pre-reset PS/v14 logs are archived under `docs/archive/experiments/pre_neuroai_reset_2026_04_29/`)
- **Per-patient tables + corpus + Brainnetome parcel list**: `docs/references/data_reference.md`

**PS stages** (per `docs/objectives.md` — paused):

| Stage | Scope | Strategy |
|---|---|---|
| Stage 1 | Single-sensor supervised correctness pass on uECoG | `stage_1.md` (closed 2026-04-20) |
| Stage 2 | In-sensor scaling: PS + lex uECoG + continuous-corpus SSL | `stage_2.md` (paused 2026-04-24) |
| Stage 3 | Cross-sensor join (+ Cogan sEEG D-cohort) | *TBD* |
| Stage 4 | External-lab validation | *TBD* |

Frozen default at PS pause (2026-04-20, historical): `per_cell + partialconv + pe2d + hierarchical_atlas @ d=32, depth=3, pool=(4,8)`. Full pipeline, contract: `docs/strategy/stage_1.md`. `docs/tactics.md` reflects pre-pause Stage-2 priorities; revisit on resume.

**Note on resume**: PS-resume keeps the original `partialconv + pe2d + hierarchical_atlas` defaults — the 2-D Utah grid is real on PS uECoG, so those primitives are appropriate. The Neuroprobe-path changes (BNA-connectivity bias, flat per-parcel pool, no partialconv) are sEEG-specific. Two paths now diverge by modality, not by program version.

**v14 is the sole active architectural direction.** v12 (cross-attention + distance bias + Fourier PE), Conv2d pipeline, the v12-era Brain-JEPA prototype, LeWM, LOPO autoresearch — all discontinued. Historical notes: `docs/archive/`. *Note: "JEPA discontinued" here refers to the v12-era prototype, not the JEPA family — Stage-2 SSL `L_recon` IS JEPA-family (data2vec 2.0 + V-JEPA 2.1 latent prediction); see Stage-2 SSL paragraph above.*

## Working Principle: Discuss Before Code

**v14 is slow, methodical, and precise. Everything before v14 was playing around.** Discuss every pipeline piece — assumptions, I/O, why right for this data, what would make it wrong, precedent and trade-offs — before any code:

1. **Agree on the contract.** Inputs, outputs, shapes, units, boundaries. Ambiguity is a blocker.
2. **No pre-committed numeric defaults.** Window sizes, `d_model`, widths, thresholds — all justified before landing.
3. **Rewrite from scratch when needed.** No file is sacred. If handwavy, rewrite — don't patch.
4. **No legacy reuse, ever.** Old code lives outside the active tree in git history and the external reset backup under `/Users/bentang/Documents/Code/backups/`. Re-derive fresh in active subpackages. Do not copy code back from backup unless there is a specific, reviewed reason.
5. **Freeze blockers before coding.** See `docs/tactics.md`.
6. **Prefer standard, scalable contracts** when they do not compromise current-stage correctness. Avoid stage-scoped conventions when a reusable contract works equally well.

Applies to every logic step: channel indices, coordinate frames, Brainnetome PM lookup, per-electrode support, parcel embedding, temporal front-end, backbone, decoder, loss, eval, metrics. No step is too obvious to skip.

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

- **Python ≥3.12, `uv`-managed.** `pyproject.toml` + `uv.lock` authoritative. (Bumped from 3.11 in pre-Stage 0 Phase 1 — required by `neuralset>=0.1.0`.)
- **Bootstrap**: `uv sync` creates `.venv/`. Run everything as `.venv/bin/python -m ...` — do not `source activate`.
- **Tests**: `.venv/bin/python -m pytest -q`. Tests are colocated with modules under `src/`; `tool.pytest.ini_options.testpaths = ["src"]`. Suite is intentionally tiny.
- **NeuroAI substrate**: `neuralset[all]`, `neuralfetch`, `neuraltrain`, and `exca` are explicit dependencies. NeuralSet owns data orchestration; NeuralTrain/Exca own training runs, caching, grids, and DCC Slurm dispatch unless a required split/metric/artifact contract cannot be expressed without fighting them. Canonical grid-sweep dispatch is `TaskInfra + neuraltrain.utils.run_grid` (one Slurm array per sweep), not `MapInfra` — `MapInfra` is for extractor-layer per-subject precompute.
- **Machine-specific BIDS paths**: `configs/paths.yaml` (gitignored).
- **All training on DCC, never local.** See `docs/references/dcc_setup.md`.
- **Exca cache folder**: must point at `/hpc/group/coganlab/ht203/cache_neuroai/` (persistent) on DCC, never `/work/ht203/` (75-day purge). Set via `EXCA_CACHE_FOLDER` env var or per-`infra.folder=`. See `docs/references/dcc_setup.md`.
- **`Wang2024Treebank` upgrade-path**: our local `Wang2024Treebank` registers under the canonical `[project.entry-points."neuralset.studies"]` because `neuralfetch==0.1.0` doesn't ship the upstream version yet. When upstream eventually ships it, NeuralFetch's `STUDY_PATHS` collision check (`study.py:455`) will raise on import. Resolution: rename the local class to a private name keeping `name="Wang2024Treebank"` ClassVar, or add explicit precedence handling.
- **Upstream Neuroprobe clone**: pinned at `c7b955b0a31464f4a5eec3f3bd78ff29841d61ac` (rebuttal codebase). Local: `.cache/neuroprobe_upstream/` (gitignored). DCC: `/work/ht203/repo/neuroprobe_upstream/`. Stage-0 wrappers default to these paths automatically. The pip `neuroprobe==0.1.7` package (in `.venv/`) is *also* installed for `neuroprobe.config` imports, but the load-bearing `examples/eval_utils.py` is only in the clone. Bootstrap locally with `git clone https://github.com/insight-neuro/neuroprobe .cache/neuroprobe_upstream && (cd .cache/neuroprobe_upstream && git checkout c7b955b0)`. The bundled `neuroprobe/braintreebank_features_time_alignment/*.csv` (per-trial words_df / nonverbal_df) lets laptop-side audits — like `audit_stimulus_overlap_cross_session.py --mode upper-bound` — run without BT data.
- **BrainTreebank raw data**: DCC-only at `/work/ht203/data/braintreebank/` (no laptop copy). Set `ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank` for any DCC-side script that touches neural data.

## Baseline to beat

**PER 0.734 ± 0.007 on S14** (grouped-by-token CV, 3-seed, per-phoneme MFA + flat head). Population mean 0.825 / 11 patients. Config details + sweep findings: `docs/archive/experiment_log.md` findings 86–101.

## Key Files

### Data (repo)
- `data/ps_tokens.csv` — canonical 52-token PS manifest (`token_id`, PS notation, ARPABET, IPA, structure).

### Data (local, gitignored)
- `data/mni_coords/<subj>_RAS.txt` — ACPC electrode coordinates (11/11).
- `data/channel_maps/<subj>_channelMap.mat` — amp → physical grid mapping.
- `data/channel_maps/<subj>_sigChannel.mat` — sig-channel masks (9/11; missing S32/S57).
- `data/transforms/<subj>_talairach.xfm` — ACPC → Talairach/MNI.
- `data/atlas/BNA_PM_4D.nii.gz` — Brainnetome PM volume (only v14 atlas source).
- `data/atlas/fsaverage_bake_v2c/` — baked fsaverage atlas (projfrac-avg + mri_surf2surf, no smoothing).
- `data/atlas/support_cache_v2c_snap/<pt>_support_tier1.csv` — per-electrode Tier-1 BNA support.
- `data/fsaverage_coords/<pt>_fsaverage_pial.csv` — strict snap-to-pial coordinates.

### Box mount (macOS laptop)
- Root: `/Users/bentang/Library/CloudStorage/Box-Box/`
- FreeSurfer recons: `ECoG_Recon/<pt>/` with `surf/`, `elec_recon/`, `mri/`, `label/`. PS patients use `S<num>`.
- ACPC electrode source: `ECoG_Recon/<subj>/elec_recon/<subj>_elec_locations_RAS_brainshifted.txt`.
- **Do not confuse `S<num>` with `D<num>`** — `D<num>` is a different Duke sEEG cohort (Stage 3 scope).
- **Avoid sibling recon folders** (`_old`, `_no_tkr`, `_kumar`, `_diag`, `_med`) — alternative reconstructions; always read from plain `S<num>/`.
- `cvs_avg35_inMNI152/` — parity-oracle template only; not active.

### Docs organization (STRICT — enforce, don't dilute)

`/docs` follows the Sun Tzu triad: **objectives → strategy → tactics**. Exactly **three working docs**. Do not add a fourth.

**Working docs:**
- `docs/objectives.md` — program hypothesis, stage roadmap, evaluation philosophy, advance gates. Stage-stable.
- `docs/strategy.md` (index) + `docs/strategy/stage_<N>.md` (per-stage) — default architecture, frozen contract, patient scope, live scoreboard, rejected paths, discipline — scoped to that stage only. Stage-N is written only when Stage-(N−1) has concluded enough to define the entry point. Do not pre-write downstream stages.
- `docs/tactics.md` — concrete task list, in-flight jobs, blockers. Refreshed when jobs land.

**Rule:** do not create additional planning, tracker, or status docs under `/docs` for the PS stage program. Extend the relevant triad doc instead. Doc surplus breaks the organization and creates stale duplicates (paid this cost multiple times — see archived `v14_next_steps_2026_04_19_morning.md`, `decision_gate_2026_04_19_snapshot.md`). When tempted, ask: "which triad layer is this? Why doesn't it live there?" Parallel initiatives outside the PS program (e.g. `docs/neuroprobe/`) get their own subdirectory with a single active plan doc.

**Reference docs** (static):
- `docs/references/data_reference.md` — per-patient tables, corpus sizes, Brainnetome parcel list, field-landscape audit.
- `docs/references/dcc_setup.md` — DCC setup, rsync recipes, submission workflow.
- `docs/references/neuroai_reference.md` — NeuralSet/NeuralFetch/NeuralTrain/Exca setup, examples, and repo conventions.
- `docs/references/neuroprobe_benchmark.md` — Neuroprobe benchmark reference.
- `docs/experiments/README.md` — fresh NeuroAI/Exca experiment export area. Pre-reset logs live under `docs/archive/experiments/pre_neuroai_reset_2026_04_29/`.
- `docs/qc/` for current QC, `docs/figures/` for current generated figures, `docs/README.md`. Historical QC/figures live under `docs/archive/qc/` and `docs/figures/archive/`.

**Archived** (historical only): `docs/archive/{sessions,plans,experiments,design_docs,experiment_log.md,research_synthesis.md,...}`. Previously-live doc forms (`current_direction.md`, `implementation_tasks.md`, `v14-core.md`) live here; the triad above supersedes them.

**Reports** (`reports/`): point-in-time audit/scoping artifacts (READMEs + CSVs/JSONs). Keep active proof reports that support current decisions, including `reports/neuroai_raw_voltage_proof_2026_04_29/`. Archive a report folder to `reports/archive/` when findings have migrated to memory or strategy and no active code reads its outputs.

### Configs & scripts

- `configs/paths.yaml` — machine-specific paths (gitignored).
- `scripts/neuroprobe/prove_wang2024treebank_raw_voltage.py` — DCC proof harness for raw BrainTreebank voltage equivalence.
- Retired scripts were backed up externally and removed from the active tree. Do not use old `scripts/v14_core/`, `scripts/ablation/`, or `scripts/archive/`; future training dispatch belongs in NeuralTrain/Exca.

## Code Structure

Active module layout under `src/speech_decoding/` after the NeuroAI reset:

- **`extractors/`** — MIRRORS `neuralset.extractors`. Our `BaseExtractor` subclasses.
  - `parcel.py` — `V14ParcelMetadataExtractor` (per-electrode parcel id + support weight + fsaverage xyz; shared across cohorts).
- **`studies/braintreebank/`** — local NeuralFetch-style `Wang2024Treebank(study.Study)`, minimal raw-voltage bridge, labels, and manifest. The public API is `ns.Study(name="Wang2024Treebank", ...)`.
- **`atlas/`** — atlas + parcel infrastructure (NeuralSet has no opinion here).
  - `fsaverage.py` (strict snap-to-pial + atlas loader + support/argmax helpers; merge of legacy `fsaverage_projection.py` + `fsaverage_atlas.py`), `support.py` (build + lookup), `tokens.py` (PS Tier-1 15 parcels).
- **`models/`** — empty shell until the v14 Perceiver IO implementation lands.
- **Training** — do not recreate the old `training/` package. Add NeuralTrain pydantic `Data` and `Experiment` classes when Stage 0 training begins.

**Empty-package rule**: don't create a subpackage until the first real file lands. Documented intents:
- `events/` — created when first cross-cohort event subclass or transform is needed.
- `ssl/` — Stage 2 SSL kickoff (`d_siglip.py`, `projection.py`, plus `L_recon` JEPA-target generation: data2vec-2.0 contextualized targets, V-JEPA-2.1 dense predictive loss, EMA teacher; REPA projector if PE-Core/REPA cells adopted).
- `studies/cogan_seeg/`, `studies/cogan_lex/` — when Stage-3 / Stage-2 PS-extension lands.

Tests are colocated next to their modules (`atlas/test_atlas.py`, `studies/braintreebank/test_loader.py`, etc.) per NeuralSet convention.

The active package should stay small: NeuroAI integration plus v14 science only. Reorg blueprint + file-by-file mapping: `docs/neuroprobe/repo_reorg_plan.md`. NeuroAI adapter spec: `docs/neuroprobe/neuralset_integration_plan.md`.

## Compute: Duke DCC cluster

Full docs: `docs/references/dcc_setup.md`.

- **SSH**: `ssh ht203@dcc-login.oit.duke.edu`
- **GPU**: 8× RTX 5000 Ada (32 GB) on `coganlab-gpu`
- **Python**: `/work/ht203/miniconda3/envs/speech/bin/python` (PyTorch 2.10.0+cu126; do NOT `conda activate`)
- **Repo**: `/work/ht203/repo/speech`
- **Data**: `/work/ht203/data/BIDS` (all 11 PS), `/work/ht203/data/{mni_coords,channel_maps,transforms,atlas}/`
- **Submit / monitor**: NeuralTrain/Exca `TaskInfra`/`MapInfra` plus Slurm (`sbatch`, `squeue`, `sacct`) for direct diagnostics.
- **CAUTION**: `/work/ht203` auto-purges after 75 days. Copy results to `/hpc/group/coganlab/ht203/`.

## Preprocessing Pipeline (do not change)

Decimate 2kHz → CAR → impedance exclusion (log10 > 6) → 70–150 Hz Gaussian filterbank (8 bands) → Hilbert envelope → sum → 200 Hz → z-score → sig-channel selection. In `coganlab/IEEG_Pipelines`. **Z-score is per-channel mean/std pooled across all pre-auditory baseline trials + samples** (500 ms window immediately before auditory-stim onset; verified 2026-04-18 via reconstruction test on 7 patients, corr = 1.0000). NOT per-trial and NOT pre-production. **Recording-level median/MAD ≡ this recipe up to per-channel affine (ρ=1.0000 across tested patients)** — SSL (Stage 2) can swap recipes without bit-exact equivalence constraints. Details: `docs/references/data_reference.md`.

## Conventions

- **Write simply.** Ordinary words. Short sentences. No throat-clearing, no redundant qualifiers, no ceremonial preambles. Cut is the main edit. Applies to docs, commits, PRs, memory files, chat. Paul Graham's "Write Simply" is the reference. Code comments stay minimal (default: none).
- **Discuss logic before writing code.** See Working Principle.
- **All training on DCC, never local.**
- **Every architectural change reports both pooled joint AND LOPO warm-start.** See `docs/objectives.md §Evaluation philosophy`. LOPO is the foundation-model test; load-bearing for Stage-2 SSL, Stage-3 cross-sensor transfer, Stage-4 external-corpus transfer. Single-protocol evidence does not justify defaulting an arch change.
- **Always export DCC results into `docs/experiments/`.** Every finished DCC run needs a durable record because `/work/ht203` auto-purges after 75 days. The old CSV and aggregator were retired; define the NeuralTrain/Exca export schema before the first Stage 0 result lands.
