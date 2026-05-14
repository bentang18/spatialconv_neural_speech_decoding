# Neuroprobe Stage 0 — Current Execution Plan

*Last revised 2026-05-09.*

## Purpose

Stage 0 is the foundation pass before v14 model training. It proves the data, labels, evaluation protocol, and QC path are correct enough to build on.

Stage 0 does **not** implement or train v14.

Data scope is BrainTreebank Lite only: 12 eval sessions plus metadata. Full BrainTreebank pretraining sessions are Stage 2 work.

## Protocol Notes

Neuroprobe's ICLR 2026 rebuttal changes four Stage 0 assumptions:

- **Multiclass is official protocol, not an internal variant.** Upstream exposes `binary_tasks=False`, and the authors report CrossSession multiclass tables. D.0b therefore remains a required Stage 0 reference, with Linear Laplacian + spectrogram at `0.611 ± 0.003`.
- **Lite sampling is class-balanced and time-shuffled before capping.** In the current public code, each label's candidate indices are sampled with a seeded `rng.choice(..., replace=False)` before the per-class cap is applied, then sorted back into temporal order for access. Lite is still a selected subset, but it is not simply the first chronological 3500 samples.
- **S2-only CrossSubject is leaderboard parity only.** The project now depends on `azaho/neuroprobe@main` pinned at `c7b955b0a31464f4a5eec3f3bd78ff29841d61ac`. It exposes `include_all_train_subjects`, but the default remains S2/trial-4. Treat S2-only as a parity cell, not the scientific default.
- **Scientific cross-subject default is pooled multi-source multiclass.** GitHub-main `include_all_train_subjects=True` returns separate 1-to-1 source-subject folds, not pooled N-to-1 training. Stage 0 still needs a local pooled leave-one-subject/session-out split for architecture selection.

The rebuttal also matters for Stage 1 entry:

- Linear's advantage is partly preprocessing/normalization, not proof that linear models are intrinsically better. BrainBERT/PopT used within-window z-scoring; the linear baseline uses train-set normalization after Laplacian + spectrogram features. Stage 1 must treat normalization and input view as explicit ablation axes.
- The authors report window-anchor robustness for 1 s windows starting from roughly `-0.375` to `+0.125` s relative to word onset. Stage 0 keeps Neuroprobe's `[0, 1]` s benchmark window, but Stage 1 should include a small window-anchor robustness cell.
- Performance-driven Lite electrode selection is only modestly stronger than random or anatomy-driven selection in their rebuttal. Still, the 120-electrode Lite cap is biased toward decodable electrodes. Use it for leaderboard parity; do not let it define the scientific default if full/anatomy/random electrode-set robustness is available.

## Current State

Completed:

- **NeuroAI data path**: local `Wang2024Treebank` NeuralSet study exists and emits raw iEEG events.
- **Raw-voltage proof**: NeuralSet/Segmenter/IeegExtractor output matches Neuroprobe direct raw h5 reads at 2048 Hz. Report: `reports/neuroai_raw_voltage_proof_2026_04_29/`.
- **Block B data integrity**: all 12 Lite h5 sessions open, use 2048 Hz, have expected Lite electrodes, and have parseable transcript feature CSVs. Report: `reports/neuroprobe_stage0_b_bt_lite_integrity_with_transcripts_2026_04_30/`.
- **Transcript repair**: truncated `thor-ragnarok/features.csv` and `lotr-1/features.csv` in `/work/ht203/data/braintreebank/` were backed up with `.corrupt_20260430T015020Z` suffixes and replaced from complete copies in `/work/ht203/data/braintreebank_smoke/`.
- **E3 label parity**: our label-index derivation matches Neuroprobe exactly for all 15 tasks in binary and multiclass modes on `sub2/trial4`. Report: `reports/neuroprobe_stage0_e3_label_parity_2026_04_30/`.
- **D.0 preflights**: exact upstream linear baseline wrapper passed for `sub1/trial1/pitch` in both modes. Reports:
  - `reports/neuroprobe_stage0_d0a_preflight_pitch_2026_04_30/`
  - `reports/neuroprobe_stage0_d0b_preflight_pitch_rerun_2026_04_30/`
- **D.0 full replication**: exact upstream linear Laplacian + spectrogram baseline rerun closed. Report: `reports/neuroprobe_stage0_d0_upstream_baselines_2026_04_30/`.
  - D.0a CrossSubject binary: `0.539210` mean AUROC vs figure reference `0.539212`.
  - D.0b CrossSession multiclass: `0.613232` mean AUROC vs figure reference `0.611336`.
  - All collected D.0 job records succeeded.
  - D.0b task means do not pass the old fixed `±0.005` task-level gate; max task delta is `0.018576`. This is documented as multiclass reference drift because the upstream figure reference was generated from `older_neuroprobe_results_fromEngaging`, while the rerun uses pinned public upstream code and current public Lite data. The aggregate target is reproduced within the published SEM.
- **A-public hard-label coverage audit**: public BT `depth-wm.csv:DesikanKilliany` labels are complete for all filtered electrodes and can support a DK/Destrieux-style hard-label control. Report: `reports/neuroprobe_stage0_public_bt_hard_labels_2026_05_05/`.
  - Full filtered BT subjects: 10 subjects, 1,550 labeled electrodes, 64 unique labels, 0 missing labels.
  - Neuroprobe Lite electrode cap across all 10 subjects: 1,160 labeled electrodes, 60 unique labels, 0 missing labels.
  - No public label appears in all 10 subjects. Top shared full-filtered labels are `ctx-lh-superiortemporal` (94 electrodes, 7 subjects), `ctx-lh-middletemporal` (70, 7), `ctx-lh-insula` (59, 7), and `ctx-lh-precentral` (54, 6). Lite-cap top shared labels are `ctx-lh-superiortemporal` (93, 7), `ctx-lh-middletemporal` (63, 7), and `ctx-lh-insula` (38, 7).
  - Pairwise label-set overlap is modest: full-filtered mean Jaccard `0.217` with range `0.000-0.762`; Lite-cap mean Jaccard `0.206` with range `0.000-0.609`.
  - Strongest co-coverage edges are the LH temporal/insula triad: `ctx-lh-insula` + `ctx-lh-middletemporal`, `ctx-lh-insula` + `ctx-lh-superiortemporal`, and `ctx-lh-middletemporal` + `ctx-lh-superiortemporal`, each in 7 subjects.
  - Same-shaft public-label uniqueness check from `electrode_label_audit.csv`: Lite has 115 parsed shafts; 97 shafts (84.3%) span more than one public label, covering 1,060/1,160 contacts (91.4%). Full filtered has 140 parsed shafts; 123 shafts (87.9%) span more than one label, covering 1,451/1,550 contacts (93.6%).
  - Interpretation: this is a public hard-label support control only. It does not create BNA support, surface coordinates, or a valid A0-A4 BNA co-coverage graph.

Blocked:

- **A0-A4 BNA / C / D.1+ BNA / V7-V8 surface geometry** remain blocked until we get true per-electrode `fsaverage` surface mapping from Christopher Wang, or Ben explicitly approves a weaker Destrieux-first fallback.
- **BT shaft/depth geometry contract** is now a Stage-0 blocker before any v14 BT architecture run. sEEG shaft-relative depth is a real measurement feature: contacts on one shaft commonly traverse multiple anatomical labels. We must decide which shaft/depth features are transferable and which are patient-ID shortcuts before encoding them.

## Stage-0 Closeout (2026-05-13)

Stage 0 closes 2026-05-13 with explicit disposition for every close-criterion. Frozen contracts are load-bearing for Stage 1; deferred items have named unblock paths and named owners. The DK-first pivot 2026-05-13 reframes the BNA / fsaverage block: BNA route is deferred to Stage-3+ (still gated on Chris MNI), DK route is the Stage-0-close anatomy contract.

### Frozen contracts (Stage-1 inheritances)

| ID | Decision | Date | Source |
|---|---|---|---|
| L.1 | Normalization = N1 `train_set_fixed` | 2026-05-08 | L.1 freeze block above |
| L.2 | Reference × view = R4×I2 (`shaft_laplacian × stft_abs`) | 2026-05-09 (held 2026-05-10 under 24-cell exhaustive) | L.2 freeze block above |
| L.3 | Filtering = F0 (no-op; cleaning ceiling) | 2026-05-11 | L.3 freeze block above |
| Shaft/depth | Same-shaft adjacency + `|i−j|` offset + local-ref provenance only; signed depth FORBIDDEN by default | 2026-05-13 | `memory/project_shaft_depth_geometry_freeze_2026_05_13.md` |
| L.4 norm × view interaction | Greedy hill-climb safe (max |residual| 0.0008) | 2026-05-10 | `reports/neuroprobe_stage0_l4_norm_view_interaction_2026_05_09/interaction_analysis.md` |
| Tier-C C.0/C.1/C.2 | C.2 generalizes (+0.0082 Δ over C.0) at CrossSubject | 2026-05-10 | `reports/neuroprobe_stage0_tier_c_cross_subject_2026_05_09/tier_c_analysis.md` |
| L.5.P1/P2/P3/P4/P5/P6/P8 (on L.2 winner) | No kill criterion fired; Stage-1 view passes nuisance gates | 2026-05-08 → 2026-05-12 | reports under `reports/neuroprobe_stage0_l5_p*` |

### Stage-0-close decisions on previously blocked items

| ID | Disposition | Rationale | Owner / unblock path |
|---|---|---|---|
| Anatomy A0-A4 (BNA route) | DEFERRED to Stage-3+ | DK-first pivot 2026-05-13 (`memory/project_v14_dk_first_pass_2026_05_13.md`) makes BNA-soft-support a P1 sister cell, not v14 default. BNA work resumes when Chris MNI lands or Stage-3 PS-extension needs it. | Christopher Wang fsaverage mapping |
| Anatomy DK route (Stage-1-entry) | DEFERRED to Stage-1 entry | DK extractor + `support_cache_dk` + E5 smoke with parcel metadata. Code skeleton lives at `src/speech_decoding/extractors/parcel.py`; `studies/braintreebank/anatomy.py` provides the DK side. Not a Stage-0 blocker — DK metadata derives from BT public hard-label CSVs (already audited 2026-05-05, 0 missing labels). | Stage-1 entry implementer |
| C support cache | DEFERRED to Stage-1 entry (DK variant) | DK-hard-one-hot 80-vocab support cache replaces BNA-soft-support cache as Stage-0 close artifact. Build at Stage-1 entry alongside DK extractor. | Stage-1 entry implementer |
| E5 (NeuralSet smoke with neural + parcel metadata) | DEFERRED to Stage-1 entry | Gated on DK extractor producing parcel metadata. Run as the first Stage-1 dispatch sanity. | Stage-1 entry implementer |
| L.7 audio-FM upper bound | DEFERRED to Stage-1 entry | BT data on DCC has `features.csv` (mel/RMS/pitch) but no `.wav`. Two unblock paths: source movies externally OR ask upstream for cached Whisper-L8 features. **Stage-1 v14 must beat L.7.A0 + 0.05** to claim brain-relevance — gate is in front of Stage-2 SSL dispatch, not Stage-0 close. | Ben to source movies, or upstream Whisper cache |
| L.5.P9 (FM-leakage on Whisper-L8) | DEFERRED to Stage-1 entry | Same blocker as L.7 (needs Whisper-L8 features). Spec is locked; replays once features land. | Same as L.7 |
| BT shaft/depth geometry contract | FROZEN 2026-05-13 (parallel session) | See `memory/project_shaft_depth_geometry_freeze_2026_05_13.md`; Stage-0-close artifact. | — |
| MNI ↔ BNA parity gate | SIDESTEPPED by DK-first pivot | Stays as gating predicate for the BNA-soft-support P1 sister cell only. | Chris MNI |
| v14 subcortical scope blocker | RESOLVED by DK-first pivot | DK aseg labels handle hippocampus/amygdala/putamen natively. | — |
| Stage-1 split contract (multiclass + pooled-CS default + S2-CS parity + Lite-120 parity) | FROZEN 2026-05-09 → 2026-05-13 | Documented in this doc + Stage-1 plan. Pooled multi-source CS = scientific default; S2-only CS = leaderboard parity only; Lite-120 = leaderboard parity only. | — |
| L.0a Experiment inherits BaseExperiment | DONE | `src/speech_decoding/experiments/experiment.py:22` `class Experiment(BaseExperiment)`. | — |
| L.0b EXCA_CACHE_FOLDER on DCC | DONE | Documented in `docs/references/dcc_setup.md`; cache routing verified 2026-05-11 substrate smoke. | — |
| L.0c DeriveLabelIndices + Wang2024Treebank `ns.Chain` | DEFERRED to Stage-1 entry (per L.0c spec; not L-sweep blocker) | Stage-0 wrapper path is canonical. | Stage-1 entry implementer |
| L.0d ExperimentLogger sidecars | DONE | `scripts/neuroprobe/run_stage0_linear_baseline.py:127` wraps every run in `ExperimentLogger`. | — |
| `CARIeegExtractor` + `ShaftCARIeegExtractor` | DONE | `src/speech_decoding/extractors/reference.py` (verified 2026-05-13). | — |

### In-flight on DCC at close (analyzers will fold into freeze blocks above on landing)

| ID | Status | Disposition |
|---|---|---|
| L.4 W2-W5 (W4/W5 RUNNING 2026-05-13; W2/W3 PD on quota) | in-flight | Robustness curve, not single-winner. When all land, write L.4 freeze block + decide whether Stage-1 anchor-robustness ablation is mandatory or optional. |
| L.5.P11 label-permutation null | in-flight (PD on quota) | KILL gate; expected pass (held-out AUROC ≤ chance + 0.02). When lands, fold result into L.5 disposition row. |
| Tier-C C.3 (bipolar × stft) + C.4 (shaftLap × HG) | in-flight (PD on quota) | C.3 settles the bipolar tie at CrossSubject; C.4 sanity-checks HG vs spec at CrossSubject. When land, fold into Tier-C disposition row. |
| D.14 pooled multi-source CS (smoke 46908857) | smoke queued | When smoke confirms pool path runs, dispatch full 48-job sweep. D.14 is the SOTA-at-submission anchor for the v14 gate "beat SOTA by ≥0.05". |
| D.15 upstream `include_all_train_subjects` probe | blocked on D.14 | Runs once D.14 lands; sister cell verifying our pool layer matches upstream's per-source-fold protocol. |
| L.5 full sweep on L.4 winner | blocked on L.4 freeze | Per `stage_0.md:285` "L.5 probes run as kill-criteria gates after every sweep winner". L.4 robustness has no single winner, so this collapses to "re-run L.5 on the L.4 W winning by ±0.02 of W0" — only fires if L.4 finds a non-W0 winner. |
| V0.x per-task accurate stimulus-overlap audit | in-flight (DCC dispatched) | Bounded above by upper-bound result (max 0.450 < 0.50 kill). Refines per-task numbers; not gating. |
| L.6 sub-cells (WL/NR/ES/CB/FA) | dispatched, partial analyzer | Tier-2 robustness, non-blocking per L.6 spec. Fold into stage_0.md when all analyzers settle. |

### Explicit non-closures (will not happen this Stage-0)

- BNA support cache from public PNG plotting coordinates — explicit non-goal (line in §Explicit Non-Goals below).
- D.1+ BNA atlas/linear cells — explicit non-goal until valid surface geometry exists.
- v14 architecture implementation, neural network training, SSL pretraining, full BT pretraining download — explicit non-goals.

### Stage-1 entry handoff

The Stage-1 entry implementer inherits:

1. **Frozen preprocessing recipe**: N1 train-set z + R4 shaftLap + I2 stft_abs + F0 no-filter + L.0 wrapper.
2. **Frozen split contract**: pooled multi-source CS multiclass = scientific default; S2-CS + Lite-120 = leaderboard parity only.
3. **DK-first anatomy contract**: DK extractor + `support_cache_dk` + E5 smoke with parcel metadata; BNA route deferred to Stage-3+.
4. **L.7 + L.5.P9 unblock-then-run**: Whisper-L8 cache or .wav source needed before Stage-2 SSL dispatch.
5. **In-flight DCC tail folded back into stage_0.md**: L.4 W2-W5 freeze block, L.5.P11 disposition, Tier-C C.3/C.4 update, D.14 baseline.

## Remaining Work

### 1. Run V0-V6 Data QC

> **Status: COMPLETED 2026-05-01.** Report: `reports/neuroprobe_stage0_v_data_qc_2026_05_01_v3/`. Headlines: raw monopolar has high common-mode burden (median top eigen-fraction 0.400, max 0.711, median |corr| 0.412). Robust global CAR reduces to 0.214 / 0.169 — diagnostic, not default. Shaft-local median + shaft-bipolar reduce to 0.167-0.179 / 0.124 — these are the real Stage-1 references (consistent with L.2's R3/R4 winners). Cheap nuisance probes decode subject-ID at balanced acc 0.699 (chance 0.167) and session-ID at 0.518 (chance 0.083) — corroborates the L.5.P1/P2 kill-criterion necessity. The Stage-1 input-view roles below are inherited from this report. The cell-by-cell description that follows in V0-V6 is preserved as the **specification** the report was built against; refer to the dated report for the actual numbers.

V0-V6 is a data-contract QC report, not a model experiment. It should decide which signal views are valid Stage-1 candidates and which artifact probes must travel with every later result. It should use all 12 Lite sessions for tabular summaries, with readable plots for a fixed representative subset.

Output should land under a dated report folder:

```text
reports/neuroprobe_stage0_v_data_qc_YYYY_MM_DD/
```

Expected artifacts:

- `README.md` with interpretation and Stage-1 implications.
- `summary.json` with pass/fail flags and open decisions.
- `session_channel_metrics.csv` with one row per session/electrode.
- `session_view_metrics.csv` with one row per session/reference/input-view.
- `lite_full_audit.csv` for Lite-vs-Full selection summaries where Full data is available.
- `nuisance_probe_metrics.csv` for subject/session/reference/coverage/timing proxy summaries.
- Representative plots, not an unreadable dump of every channel.

#### V0 — Lite Vs Full Selection Audit

Lite is the leaderboard-parity set, but it is selected. Stage 0 must quantify what that selection changes before v14 treats Lite performance as scientific evidence.

Compare Lite against Full/uncapped BrainTreebank where public data permits:

- channel/electrode overlap by subject/session;
- label distributions by task/class;
- movie-position and time-bin histograms;
- Lite-channel vs non-Lite-channel raw amplitude, PSD, covariance, and bad-channel summaries;
- whether Lite capping/shuffling creates temporal artifacts;
- whether Lite is biased toward cleaner, more active, or more decodable electrodes.

Record the decision explicitly: Lite is for leaderboard parity; Full/uncapped data is for robustness whenever available.

**V0.x — Stimulus-overlap audit (CrossSession leakage)**: for every (task, label) pair in the 15-task suite, check whether the same word/exemplar appears in both train sessions and test sessions. CrossSession protocol shuffles train/test by session, not by stimulus — so a stimulus heard in sub1/trial1 (train) can re-appear in sub1/trial2 (test). If yes, the linear classifier can pattern-match stimulus identity rather than brain response, making the AUROC partially a stimulus-recognition score rather than a brain-decoding score. Single-shot audit; report fraction of test stimuli also present in train per (subject, task). Add to V0 deliverables. Rerun expected when any new task or session enters the suite.

> **Status (2026-05-10) — Upper-bound landed; per-task pending DCC**. `scripts/neuroprobe/audit_stimulus_overlap_cross_session.py --mode upper-bound` ran on the bundled `braintreebank_features_time_alignment/` CSVs (no BT data needed). Per (subject, test_trial), unique-word overlap across the full per-trial words_df: **median 0.414, min 0.323, max 0.450 across 12 BT Lite pairs**. **No (subject, task) pair exceeds the 50% kill threshold** at the upper bound. Per-task accurate mode (`--mode per-task`) needs DCC because it instantiates the upstream dataset (loads neural data caches) to apply each task's `label_indices` selection; that pass will refine the per-task numbers but is bounded above by the upper-bound result. Artifact: `reports/neuroprobe_stage0_v0x_stimulus_overlap_2026_05_10/`.

#### V1 — Raw Signal Health

Generate raw-voltage QC:

- raw voltage traces
- amplitude and bad-channel audit
- sample-rate, duration, channel count, and channel-label/order checks
- NaN/inf, flatline, clipping-like repeated-extrema, and robust outlier metrics

V1 should answer whether BT's Lite mask is sufficient or whether additional split-aware bad-channel exclusions are needed.

#### V2 — Reference Transform QC

Re-reference is a first-class transform, not hidden preprocessing. It changes the measurement operator:

```text
raw:           y  = Ls + c + noise
rereferenced: y' = R y = R L s + R c + R noise
```

Therefore every later v14 reference transform must carry provenance. For Stage 0 QC, compare signal behavior under:

- **R0 raw/monopolar** — diagnostic and raw-view ablation.
- **R1 robust global CAR** — diagnostic only unless later evidence justifies promotion.
- **R2 shaft-local robust reference / within-shaft CAR** where channel names support shaft grouping.
- **R3 bipolar or shaft Laplacian/local reference** where adjacent-contact order is reliable.

For each transform, export enough metadata to reconstruct the virtual channels:

- reference type;
- source physical channels;
- virtual channel labels;
- `R` matrix or sparse equivalent;
- bad-channel exclusions;
- virtual-channel coordinate/support status. Before exact geometry arrives, this field must be marked unresolved rather than inferred.

Global CAR is not the default. In sparse asymmetric clinical coverage it can inject artifacts into covariance structure. For BrainTreebank sEEG, within-shaft local/bipolar/Laplacian-style references are the physics-matched family; raw and CAR remain diagnostics.

#### V3 — Frequency-View QC

Compare the relevant input views after each valid reference:

- raw voltage;
- low-frequency LFP summaries;
- STFT/log-power;
- HG/HFA envelope;
- CAR+HG;
- local-reference HG/HFA.

The biologically privileged supervised speech view is local-reference HG/HFA because high-gamma field activity is the conventional proxy closest to local population firing/spiking. The richer intrinsic-SSL candidate is local-reference multi-band log-power/STFT. Raw 2048 Hz voltage remains an auxiliary/ablation view because it preserves information but carries the highest artifact and subject/device burden.

#### V4 — Artifact Source Battery

For every reference/input-view pair, quantify the nuisance sources most likely to fake cross-patient generalization:

- subject/session/device separability from cheap summary features;
- reference/common-mode structure;
- line noise and harmonics;
- channel-count and coverage-proxy separability;
- amplitude/gain/noise-floor fingerprints;
- slow drift;
- high-frequency EMG-like broadband bursts;
- pre-event label decodability;
- shifted-window label decodability;
- reaction-time, block-order, movie-time, or other timing proxies where available.

The report should flag any view that improves apparent signal quality while increasing subject/session/reference/coverage decodability. That pattern is not a Stage-1 default candidate without stronger task evidence.

#### V5 — Event-Locked Sanity

Check benchmark windows and nearby anchors:

- word/event-locked raw, HG, and spectrogram averages;
- trial-locked heatmaps;
- pre-stimulus controls;
- shifted-window controls;
- event-count and valid-window summaries by session;
- whether response/event structure is aligned or smeared.

Stage 0 keeps Neuroprobe's `[0, 1]` s benchmark window. Anchor robustness is a Stage-1 cell, but V5 should decide whether such a cell is mandatory.

#### V6 — Stage-1 Input-View Decision Table

Synthesize V0-V5 into a decision table. For each candidate view, record:

- task-relevant event signal: low/medium/high;
- line/common-mode burden;
- subject/session/reference decodability;
- Lite-vs-Full shift;
- preprocessing cost;
- virtual-channel/support bookkeeping burden;
- allowed Stage-1 role: default candidate, required ablation, or diagnostic only.

The expected roles before evidence are:

| View | Provisional role |
|---|---|
| raw monopolar | diagnostic + ablation |
| robust global CAR raw | diagnostic, not default |
| within-shaft local raw | serious Stage-1 cell |
| within-shaft log-STFT | serious Stage-1 cell |
| within-shaft HG/HFA | biologically privileged Stage-1 cell |
| CAR+HG | conventional control |

V0-V6 should answer:

- Are there dead/saturated/noisy channels beyond BT's mask?
- Does 60 Hz or harmonic noise dominate any sessions?
- Is noise low-rank enough for an SSP-style Stage 1 re-reference sweep?
- Are event-locked responses aligned well enough to trust downstream labels/windows?
- Are there session-specific quirks that should be excluded before training?
- Which signal view should be trusted for Stage 1: raw voltage, Laplacian/local raw, Laplacian/local STFT/log-power, Laplacian/local HG/HFA, or CAR+HG?
- Which normalization scope should be trusted: train-set/session-level normalization, recording-level normalization, or window-local normalization? This is load-bearing because upstream attributes part of the BrainBERT/PopT deficit to within-window z-scoring.
- Does Neuroprobe-Lite shuffle samples across time before class-balanced capping? Verify the upstream sampling code, record whether the first-N temporal-bias risk is real or closed, and do not rely on author rebuttal text alone.
- Which cross-subject split is actually implemented in the checked-out Neuroprobe code: S2-only, pairwise all-source, pooled leave-one-subject/session-out, or some mixture?
- How much do label distributions, movie moments, and task difficulty differ between Lite and Full/uncapped samples?
- How much patient/session/reference identity is visible in simple spectral and covariance summaries before any neural net sees the data?

### 2. Resolve BT Surface Mapping

Current public BT localization files are insufficient for rigorous BNA support construction:

- `depth-wm.csv` gives anatomical/localization labels and coordinates.
- `elec_coords_full.csv` is a 2-D plotting overlay for PNG brain visualizations.
- Public files do not expose `fsaverage` hemisphere, vertex index, or surface RAS coordinate.

Need one of:

- per-electrode `fsaverage` hemisphere + vertex index + surface RAS table;
- the exact script/transform used to create the pre-plotting fsaverage coordinates;
- explicit approval to use a weaker Destrieux-first fallback.

Until this resolves, do not build BT BNA support caches or run BNA/atlas linear cells.

Do not substitute the Cogan D-cohort volumetric convention here. BT does not release per-subject FreeSurfer recons, and the public plotting coordinates do not recover volumetric truth.

Public hard-label fallback status:

- `depth-wm.csv:DesikanKilliany` is valid for a DK/Destrieux-style one-hot support control: `support_kind="hard_public_bt_label"`.
- This support axis is the public BT label vocabulary, not BNA. It can test whether even coarse hard anatomy helps cross-subject transfer before Christopher shares surface coordinates.
- The coverage report is `reports/neuroprobe_stage0_public_bt_hard_labels_2026_05_05/`; use `label_vocabulary.csv`, `subject_label_coverage.csv`, `pairwise_label_overlap.csv`, and `label_co_coverage_edges.csv` when deciding hard-label ablations.
- The hard-label co-coverage graph is not a substitute for A4. It is a control showing the current public-label overlap structure: no label spans all 10 subjects, the best shared labels cover 7 subjects, and mean pairwise overlap is only about `0.21`.

### 3. Resolve BT Shaft/Depth Geometry Contract

> **Status: FROZEN 2026-05-13.** Audit script `scripts/neuroprobe/audit_shaft_geometry.py`, report `reports/neuroprobe_stage0_shaft_depth_geometry_2026_05_13/` (README + 4 CSVs). Headline: 1549 contacts across 9 subjects (S5 dropped per DK-first-pass), 128 shafts, 0 parse anomalies, 0 cross-hemisphere shafts, 99.2% linear, 98.4% suffix-monotonic; **signed depth FORBIDDEN by default** — sign convention is 75% cohort-uniform (lower_suffix), below the 95% admission threshold. v14 may use orientation-invariant within-shaft features only (adjacency mask, relative offset `|i − j|`, local-reference provenance). shaftCAR (R2, v14 default) + shaftLaplacian (R4, upstream parity) both cleared because they are orientation-invariant under symmetric form. Unfreeze triggers: Chris MNI ship + subject-pool change + BNA subcortical scope decision.

This is a blocker before any v14 BT architecture run. It can proceed before surface mapping arrives because it uses electrode labels and public anatomy, not MNI/fsaverage coordinates.

Why it matters:

- sEEG contacts are ordered samples along depth shafts, not unordered cortical surface sensors.
- One shaft commonly spans several public anatomy labels: in Lite, 84.3% of parsed shafts span more than one label, and 91.4% of contacts sit on multi-label shafts.
- A public hard label alone discards within-shaft position. A future BNA support vector alone will also not encode whether a contact is deep, superficial, adjacent to another contact, or part of a local-reference virtual channel.
- Raw `shaft_id` is patient-specific. It is valid for grouping and local reference construction, but a learned shaft-ID embedding can become a subject-identity shortcut.

Decisions to freeze before coding:

- **Shaft parser**: define the canonical parse from electrode label to `(subject, shaft_id, contact_index)`. The default candidate is final integer suffix as contact index and preceding stem as shaft id; exceptions and non-numeric contacts must be audited.
- **Contact order orientation**: determine whether contact indices are consistently deep-to-superficial or superficial-to-deep across BT subjects. If unknown, signed depth is not allowed as a default feature.
- **Depth feature**: choose one of: no depth; ordinal contact index; normalized shaft position `index / max_index`; centered normalized position; orientation-invariant features only. No numeric default is accepted until orientation is checked.
- **Transferable geometry**: allowed by default only if shared across subjects: same-shaft adjacency, relative contact offset, normalized position, and local-reference provenance. Raw shaft-ID embeddings are disallowed by default.
- **Local reference contract**: define how shaft grouping creates bipolar/Laplacian/local-reference virtual channels, and how virtual-channel metadata records source contacts, contact offsets, and unresolved coordinate/support status.
- **Interaction with anatomy**: public hard label or future BNA support supplies brain-region identity; shaft/depth supplies within-shaft measurement geometry. Do not conflate them.
- **Nuisance risk**: run subject/session decoding probes with and without shaft/depth features. Promote a depth feature only if it improves transfer or task evidence without making subject identity trivially separable.
- **Ablation matrix**: Stage 1 must include at least `hard_public_bt_label` only, shaft/depth only, and hard label + shaft/depth before making depth default.

Required outputs:

- `reports/neuroprobe_stage0_shaft_depth_geometry_YYYY_MM_DD/shaft_contact_inventory.csv`
- `reports/neuroprobe_stage0_shaft_depth_geometry_YYYY_MM_DD/shaft_label_transition_summary.csv`
- `reports/neuroprobe_stage0_shaft_depth_geometry_YYYY_MM_DD/contact_order_orientation_audit.csv`
- `reports/neuroprobe_stage0_shaft_depth_geometry_YYYY_MM_DD/README.md` with the frozen feature contract and rejected encodings.

### 4. Linear Ablation Matrix (Block L)

The linear baseline is a preprocessing oracle. Anything the architecture would otherwise have to learn — reference choice, input view, normalization recipe, filtering, bad-channel exclusion, window anchor — can be tested on the linear and baked into Stage 1. Whatever the linear cannot already exploit is what v14 actually has to learn.

L cells use the D.0 protocol harness extended with the relevant config flag:

- **L.1 (normalization sweep)** runs through `scripts/neuroprobe/run_stage0_linear_baseline.py` (own wrapper, mirrors upstream `eval_population.py` byte-for-byte except for an internalized `--normalization` step). Submitter: `scripts/neuroprobe/submit_l1_normalization_sweep.py`. Collector + per-sweep aggregate viz: `scripts/neuroprobe/collect_l1_normalization_sweep.py`. The wrapper writes per-cell `signal_qc.png` (feature-magnitude histogram pre vs post normalization) + `metrics.json` + `diagnostics.csv` + `experiment_record.json` sidecar; the collector emits `cell_task_heatmap.png` + `cell_aggregate_bar.png` + `n0_vs_n1_delta.csv`.
- **L.2/L.3/L.4** extend the same wrapper by adding NeuralFetch-driven preprocess swaps (reference + filter + window anchor). The N1 cell of L.1 is the byte-equivalent reproduction of the upstream D.0 baseline; this serves as the regression check on the wrapper's own pipeline before any new axis sweeps. L.2 is FROZEN at R4×I2 (2026-05-09); L.3 + L.4 dispatch on the L.1 + L.2 frozen winners.

Default eval is D.0b CrossSession multiclass on all 15 tasks, full upstream subject matrix. D.0a CrossSubject binary runs as a confirmation pass on each Sweep winner via the same wrapper with `--include-cross-subject`.

**Sequencing**: greedy hill-climb. L.1 → L.2 → L.3 → L.4. Each sweep freezes one Stage-1 default; the next sweep inherits it. Greedy was chosen over a small factorial for compute reasons; the interaction-risk assumption (norm × view × ref independent enough that fixing one before sweeping the next doesn't lose meaningful AUROC) is checked retroactively via an interaction sanity row recorded on the final L.4 winner. L.5 probes run as kill-criteria gates after every sweep winner.

**Visualization deliverable per cell** (non-negotiable, Block L is co-developed with Stage 0 visualizations): every cell folder ships with `signal_qc.png` showing what its recipe does to the signal on a representative subject/session, in addition to its `metrics.json`. The `signal_qc.png` axes vary by sweep:

- **L.1**: per-cell feature-magnitude histogram pre vs post normalization.
- **L.2**: per-cell representative voltage trace + spectrogram side-by-side.
- **L.3**: per-cell PSD before vs after filter, with 60 Hz family annotated.
- **L.4**: per-anchor event-locked heatmap on a representative task.
- **L.5**: per-probe AUROC bar with kill-threshold line drawn.

**Visualization deliverable per sweep**: each sweep produces an aggregate plot in its report folder:

- **L.1**: cell × task AUROC heatmap with N0-vs-N1 delta highlighted.
- **L.2**: reference × input-view AUROC matrix.
- **L.3**: cell-vs-floor delta bar chart.
- **L.4**: anchor-robustness curve (anchor offset on x, AUROC on y, ribbon = SEM across tasks).
- **L.5**: probe bar chart with kill thresholds annotated.

These are the pictures Ben reviews to make Stage-1 freeze decisions; the CSVs are evidence, the PNGs are how the decision actually gets made.

#### L.0 — Prerequisites (must clear before any L sweep dispatches)

Surfaced by the 2026-05-05 NeuroAI integration audit. None of the L sweeps run cleanly until these land.

- **L.0a** `Experiment` must inherit `neuraltrain.utils.BaseExperiment` so `neuraltrain.utils.run_grid` will type-accept it. Without this, the canonical Slurm grid-array dispatch is unreachable. One-line change in `src/speech_decoding/experiments/experiment.py`.
- **L.0b** Set the exca cache folder to `/hpc/group/coganlab/ht203/cache_neuroai/` (persistent) on DCC, not `/work/ht203/` (75-day purge). Document `EXCA_CACHE_FOLDER` env var convention in `docs/references/dcc_setup.md`. Without this every sweep silently recomputes after the next purge cycle.
- **L.0c** *(Stage-1-entry preparation, NOT an L-sweep blocker)* Write a `DeriveLabelIndices` `EventsTransform` and a `Wang2024Treebank` `ns.Chain` so the events DataFrame carries `code` + `split` before the `Segmenter` sees it. This unblocks running D.0 and L cells through our canonical `Experiment` class. L sweeps themselves are dispatched via the Stage-0 wrapper path (`scripts/neuroprobe/run_stage0_linear_baseline.py`, which mirrors `run_upstream_linear_baseline.py` byte-for-byte for the N1 cell and adds the `--normalization` / `--backend {upstream, neuralset}` / `--ref-kind` / `--view-kind` flags). Both wrappers write `ExperimentLogger` sidecars. Treat L.0c as a Stage-1-entry deliverable; it shouldn't gate L.1.
- **L.0d** Verify `scripts/neuroprobe/run_upstream_linear_baseline.py` wraps each run in `ExperimentLogger`. If not, retrofit so D.0 cells and all L cells write the canonical `experiment_record.json` sidecar that `collect_experiment_records.py` aggregates into `docs/experiments/runs.csv`.
- **L.0e** *(cleared 2026-05-06)* `CARIeegExtractor` ships in `src/speech_decoding/extractors/reference.py` (`car="global"` and `car="shaft"`). The `ShaftCARIeegExtractor` is the same class with `car="shaft"`. R1/R2 cells of L.2 are unblocked. `bipolar_ref` and Laplacian are handled in `scripts/neuroprobe/preprocess_views.py` directly (see L.2 below); NeuralSet kwargs alone don't cover shaft Laplacian.

#### L.1 — Normalization Scope (Sweep 1, runs first)

Tests the rebuttal claim that normalization recipe explains a meaningful chunk of the linear-vs-foundation-model gap. Fixed input view: Lap+spec (D.0 default). Fixed model: logistic regression. Vary only the per-channel normalization scope.

| Cell | Recipe | Provenance |
|---|---|---|
| L.1.N0 | per-window z-score | BrainBERT/PopT recipe per rebuttal |
| L.1.N1 | train-set fixed (mean/std over training windows) | current upstream linear baseline |
| L.1.N2 | per-session fixed | closest analog to "recording-level" |
| L.1.N3 | train-set fixed, scale-only (no demean) | isolates demean from scale |
| L.1.N4 | none / raw Lap+spec | sanity baseline |
| L.1.N5 | per-session robust (median/MAD) | Cogan-pipeline analog; ratifies our PS recipe transferring to sEEG |

Headline: gap between L.1.N0 and L.1.N1 averaged across tasks. Anything above ~0.02 multiclass AUROC means recipe is load-bearing for the rebuttal claim. Whatever's left is genuinely architecture. Secondary: N2 vs N5 — if mean/std and median/MAD give the same result on sEEG (as they did on PS uECoG, ρ=1.0), the choice is moot; if they diverge, sEEG has heavier outlier tails. Stage-1 v14 default normalization is frozen from this sweep.

> **2026-05-08 — L.1 FROZEN at N1 (`train_set_fixed`).** Full 9-cell ranking (Tier-A + Tier-B) on 12 sessions × 15 tasks = 180 (session, task) pairs/cell, bootstrap N=2000:
>
> | rank | cell | recipe | mean | CI |
> |---|---|---|---|---|
> | 1 | N2 | per_session_fixed (transductive) | 0.6175 | [0.598, 0.638] |
> | 2 | N3 | train_set_scale_only | 0.6137 | [0.595, 0.632] |
> | 3 | **N1** | **train_set_fixed (upstream)** | **0.6132** | **[0.595, 0.633]** |
> | 4 | N5 | per_session_robust_mad (transductive) | 0.6114 | overlap |
> | 5 | N6 | train_set_robust_mad (Tier-B) | 0.6072 | overlap |
> | 6 | N7 | per_session_robust_scale (transductive Tier-B) | 0.6068 | overlap |
> | 7 | N0 | per_window_z (BrainBERT/PopT) | 0.5597 | clear loss |
> | 8 | N8 | per_channel_train_set_z (Tier-B) | 0.5550 | clear loss |
> | 9 | N4 | none | 0.5536 | clear loss |
>
> **Decision tree application** (rule 2): top-3 cells (N2/N3/N1) all have overlapping CIs; N2 is transductive (refits scaler on test session's own features). Prefer inductive → eliminate N2. N3 vs N1 is statistically indistinguishable (Wilcoxon p=0.46). Default to upstream-parity baseline → **N1**.
>
> **Tier-B decomposition findings**:
> - N5 vs N6 (Δ=+0.0042, CIs overlap): per-session scope is decorative — the robust statistic does what little work there is. *No reason to adopt per-session normalization.*
> - N5 vs N7 (Δ=+0.0046, overlap): centering vs scale-only barely matters under robust statistics.
> - N8 vs N1 (Δ=−0.0582): per-channel z-scoring is decisively *worse* than pooled. *Reject per-channel as a Stage-1 default.*
>
> **Headline**: N0 − N1 = −0.0536 (~5.4 pp tax for the BrainBERT/PopT per-window-z recipe). Above the 0.02 load-bearing threshold; preprocessing recipe **is** load-bearing for the linear-vs-FM gap.
>
> **Architectural consequences for v14**:
> - Stage-1 preprocessing contract: `train_set_fixed` (StandardScaler fit on training fold). No per-session test-time adaptation required.
> - Per-channel scope rejected — token-level normalization in v14 should pool across channels within a parcel, not per-electrode.
> - The transductive ~0.004 lift is in v15-territory, not load-bearing for v14.
>
> Artifacts: `reports/neuroprobe_stage0_l1_normalization_2026_05_05/{freeze_analysis.md, freeze_analysis.json, cell_ci_forest.png, paired_tests.csv}`.

#### L.2 — Reference × Input-View (Sweep 2, after L.1)

> **2026-05-06 status.** First L.2 dispatch (`reports/neuroprobe_stage0_l2_reference_view_2026_05_05/`) was degenerate — 9 cells collapsed onto 4 upstream `preprocess_type` strings (`laplacian-stft_abs` / `stft_abs` / `laplacian` / `none`). Five `laplacian-stft_abs` cells produced byte-identical diagnostics; the headline ref × view comparisons (bipolar vs shaft-Lap; HG vs multi-band) were untested. Pivoted to `scripts/neuroprobe/preprocess_views.py` which factors `(reference, view)` into discrete primitives. Wrapper gains `--backend {upstream, neuralset}` flag; upstream backend stays default for byte-compat with L.1 N1 baseline. Tier-A grid (3 ref × 3 view = 9 distinct cells) re-dispatched 2026-05-06.

> **2026-05-09 — L.2 FROZEN at R4×I2 (`shaft_laplacian × stft_abs`, upstream parity).** Full Tier-A 9-cell ranking on 12 sessions × 15 tasks = 180 rows/cell, bootstrap N=2000:
>
> | rank | cell | recipe | mean | CI | Δ vs baseline |
> |---|---|---|---|---|---|
> | 1 | R3×I2 | bipolar × stft_abs | 0.6157 | [0.597, 0.636] | +0.0025 |
> | 2 | **R4×I2** | **shaftLap × stft_abs [D.0 baseline]** | **0.6132** | **[0.595, 0.633]** | **+0.0000** |
> | 3 | R0×I2 | raw × stft_abs | 0.5923 | [0.576, 0.611] | −0.0209 |
> | 4 | R3×I3 | bipolar × HG envelope | 0.5893 | [0.573, 0.607] | −0.0239 |
> | 5 | R4×I3 | shaftLap × HG envelope | 0.5868 | [0.570, 0.604] | −0.0264 |
> | 6 | R0×I3 | raw × HG envelope | 0.5743 | [0.560, 0.590] | −0.0389 |
> | 7 | R0×I0 | raw × voltage | 0.5534 | [0.543, 0.564] | −0.0598 |
> | 8 | R3×I0 | bipolar × voltage | 0.5516 | [0.542, 0.563] | −0.0616 |
> | 9 | R4×I0 | shaftLap × voltage | 0.5498 | [0.540, 0.560] | −0.0634 |
>
> **Decision tree application** (rule 2): top cell R3×I2 ties baseline R4×I2 (Δ = +0.0025, CI overlap, NOT load-bearing at 0.02 threshold). Default to upstream-parity → **R4×I2**. (Bipolar tie carried into Tier-C as the CrossSubject contrast.)
>
> **Headline** (rule 3 fires): view marginal Δ = 0.0555 [I2=0.6071, I0=0.5516] swamps reference marginal Δ = 0.0122 [R3=0.5855, R0=0.5733] by 4.5×. *View matters; reference does not.* For v14 architectural design, this shifts complexity budget from reference design (Laplacian/bipolar/CAR variants) to spectral feature design (STFT/HG/multi-band/wavelet/learned tokenizer).
>
> **Spectral floor**: all three I0 (raw voltage) cells cluster at 0.55, separated from the I2/I3 cluster (0.59–0.62) by a clear gap. Linear readout cannot extract speech-relevant structure from raw voltage at this scale; spectral pre-tokenization is load-bearing for the linear baseline (consistent with Neuroprobe's "Linear (Lap+spec) 0.611" finding).
>
> **Interaction residuals**: max |residual| = 0.0092 < 0.01 → main-effect story holds; no hidden ref × view non-additivity.
>
> **Architectural consequences for v14**:
> - Stage-1 input-view contract: `stft_abs` (or richer learnable spectral). Raw voltage tokens cannot be the only input view.
> - Reference choice not load-bearing for linear readout — defer R-design complexity until non-linear backbone is in place. Stage-1 default = `shaft_laplacian` (upstream parity).
> - Three I0 cells at 0.55 = reviewer-defensible floor: linear-on-raw is genuinely weak. Architecture's job is to do better than spectral-pretokenizer-on-linear, not better than linear-on-raw.
>
> **2026-05-10 — L.2 EXHAUSTIVE 24-cell ranking lands; freeze HOLDS at R4×I2.** Full Tier-A + Tier-B + I6 sweep on 12 sessions × 15 tasks, bootstrap N=2000:
>
> | rank | cell | recipe | mean | CI |
> |---|---|---|---|---|
> | 1 | R3×I2 | bipolar × stft_abs | 0.6157 | [0.598, 0.635] |
> | 2 | R4×I2L | shaftLap × log-STFT | 0.6150 | [0.597, 0.636] |
> | 3 | R2×I2 | shaftCAR × stft_abs | 0.6138 | [0.596, 0.633] |
> | 4 | **R4×I2** ★ | **shaftLap × stft_abs [D.0 baseline]** | **0.6132** | [0.594, 0.633] |
> | 5 | R1×I2 | globalCAR × stft_abs | 0.6111 | [0.593, 0.631] |
> | 6 | R5×I2 | median × stft_abs | 0.6016 | [0.584, 0.620] |
> | 7 | R4×I3W | shaftLap × wide-HG (70-250) | 0.5924 | [0.575, 0.610] |
> | 13 | R4×I4 | shaftLap × multi-band log-power | 0.5793 | [0.563, 0.597] |
> | 15 | R4×I5 | shaftLap × wavelet | 0.5766 | [0.561, 0.595] |
> | 23 | R4×I1 | shaftLap × low-LFP (<30 Hz) | 0.5243 | [0.517, 0.532] |
> | 24 | R4×I6 | shaftLap × theta-band phase | 0.5004 | [0.498, 0.503] |
>
> **Top-5 cells are all I2-family** (stft_abs / log-STFT) regardless of reference — confirms "view dominates reference" at greater scale. **R4×I2L (log-STFT) ties baseline at +0.0018 — well below 0.02 threshold; freeze holds.** **R4×I3W (wide HG 70-250 Hz) +0.0055 over standard R4×I3** — modest improvement, not architecture-changing. **R4×I4 multi-band and R4×I5 wavelet *underperform* stft_abs** — spectral richness ≠ better; STFT magnitude already captures what's needed for linear. **R4×I6 (theta phase) = chance (0.500)**: phase-only features carry zero linear signal; phase encoders dead unless paired with magnitude. **R4×I1 (<30 Hz LFP) = 0.524**: speech-relevant signal does not live below 30 Hz.
>
> Full ranking: `reports/neuroprobe_stage0_l2_exhaustive_2026_05_09/freeze_analysis.md`.
>
> **Tag-along sweeps dispatched 2026-05-09** (gated on L.1 + L.2 winners):
> - **Tier-C CrossSubject parity** (`reports/neuroprobe_stage0_tier_c_cross_subject_2026_05_09/`): C.1 = N1 + R4×I2 (baseline at CrossSubject); C.2 = N1 + R3×I2 (does the bipolar tie hold at distribution shift?). 11 BT-Lite sessions × 2 cells, sub2 excluded.
>   - **C.1 status (2026-05-10)**: 10/10 OOM at 24G, re-dispatched at 64G mem.
>   - **C.2 status (2026-05-10) — WRAPPER FIX LANDED, pending DCC re-run**: previously hit upstream `combine_regions()` IndexError because bipolar collapses pairs into virtual channels and the DK-region mask was sized from the original electrode count. Wrapper now derives virtual regions from post-reference labels (`run_stage0_linear_baseline.py` `_derive_virtual_regions` + per-subject `apply_reference` round-trip). Bipolar pair `chA-chB` inherits chA's DK region (within-shaft adjacent → same region in practice). Defensive: also handles all other refs uniformly via label lookup. Pending DCC re-dispatch to confirm bipolar CrossSubject parity.
> - **L.4 anchor sanity** (`reports/neuroprobe_stage0_l4_anchor_2026_05_09/`): A.0 = baseline [0, 1]s, A.1 = lead [-0.375, +0.625]s on N1 + R4×I2. **Status (2026-05-10)**: 15/24 done; 9 OOM/exit-1 jobs re-dispatched at 64G.
> - **L.4 norm × view interaction** (96 jobs, `reports/neuroprobe_stage0_l4_norm_view_interaction_2026_05_09/`): refs={shaft_laplacian, bipolar} × views={stft_abs, hg_envelope} × norms={train_set_fixed, train_set_scale_only}. 8 cells × 12 sessions. **VERDICT (2026-05-10): greedy hill-climb safe.** Max interaction |residual| = 0.0008 across all 8 cells — refs, views, and norm act as nearly independent factors. Stage-1 contract (L.1 N1 + L.2 R4×I2 + L.3 winner + L.4 robustness) is robust to interaction effects. Artifact: `interaction_analysis.md`.
> - **L.2 seed43/seed44** (24 jobs total): seed-variance reruns of L.2 winner R4×I2 at seed 43/44. **Status (2026-05-10)**: 14/24 done; 10 OOM jobs re-dispatched at 64G.
>
> Artifacts: `reports/neuroprobe_stage0_l2_exhaustive_2026_05_09/{freeze_analysis.md, freeze_analysis.json, cell_ci_forest.png, factor_marginals.png, paired_tests.csv, aggregate_summary_by_cell.csv}` (24-cell exhaustive); `reports/neuroprobe_stage0_l2_neuralset_2026_05_06/` (original 9-cell Tier-A frozen).

Two crossed axes. Reference transform changes the measurement operator (`y' = R y = R L s + R c + R noise`); input view changes which signal property is exposed.

Reference (rows):

| Cell | Recipe | preprocess_views.py kind |
|---|---|---|
| R0 | raw monopolar | `ref_kind="raw"` |
| R1 | robust global CAR | `ref_kind="global_car"` (Tier-B) |
| R2 | within-shaft local / shaft-CAR | `ref_kind="shaft_car"` (Tier-B) |
| R3 | bipolar (adjacent within-shaft pair) | `ref_kind="bipolar"` |
| R4 | shaft Laplacian (upstream byte-parity) | `ref_kind="shaft_laplacian"` |
| R5 | WM-rejected variants of R2/R3/R4 | gated on Chris's WM-flag answer |

Input view (columns):

| Cell | Recipe | preprocess_views.py kind |
|---|---|---|
| I0 | raw 2048 Hz voltage | `view_kind="raw_voltage"` |
| I1 | low-frequency LFP (<30 Hz) | `view_kind="low_lfp"` (Tier-B) |
| I2 | STFT magnitude (`stft_abs`) | `view_kind="stft_abs"` (reuses upstream `preprocess_stft`) |
| I3 | HG/HFA envelope (70–150 Hz) | `view_kind="hg_envelope"` (Butterworth band + Hilbert) |
| I4 | multi-band log-power (6 bands) | `view_kind="multi_band_log_power"` (Tier-B) |
| I5 | wavelet (Morlet, 6 scales) | `view_kind="wavelet"` (Tier-B) |
| I6 | theta-band (4-8 Hz) instantaneous-phase mean cos+sin | `view_kind="instantaneous_phase"` (Tier-B, added 2026-05-09 to close phase axis) |
| I2L | log STFT (`log(|STFT|² + ε)`) | `view_kind="log_stft"` (Tier-B) |
| I3W | wider HG envelope (70–250 Hz) | `view_kind="hg_envelope_wide"` (Tier-B) |

Note: I2 is the upstream `stft_abs` magnitude (not log-power) so the existing N1 baseline of 0.6132 reproduces byte-for-byte. I2L is the deferred Tier-B "stft log-power" variant.

Earlier drafts listed I6 = Lap+spec and I7 = CAR+HG as input views. They are not — they are (reference, view) pairs (R4, I2) and (R1, I3) respectively. Dropped to remove alias collisions. The D.0 default "Lap+spec" is now spelled R4×I2, and the conventional CAR+HG control is R1×I3.

Don't run all 5×6=30. Run the **Tier-A 9-cell grid first** (3 ref × 3 view, all distinct, all bytes-different). Tier-B (12 additional cells) only runs if Tier-A flags an open question.

**Tier-A (FROZEN 2026-05-09 — see freeze block above)**:

| Cell | Recipe | Role |
|---|---|---|
| R0×I0 | raw monopolar × raw voltage | floor / v14-aligned ceiling |
| R0×I2 | raw × STFT | spectral without re-reference |
| R0×I3 | raw × HG envelope | spectral via HG without re-reference |
| R3×I0 | bipolar × raw voltage | bipolar floor |
| R3×I2 | bipolar × STFT | bipolar spectral |
| R3×I3 | bipolar × HG | bipolar HG conventional |
| R4×I0 | shaft Laplacian × raw voltage | shaft-Lap floor |
| R4×I2 | shaft Laplacian × STFT | **D.0 default ("Lap+spec") — parity anchor** |
| R4×I3 | shaft Laplacian × HG | biologically privileged candidate |

**Tier-B (gated on Tier-A results)**:

| Cell | Recipe | Role |
|---|---|---|
| R1×I3 | global CAR × HG | conventional CAR+HG control |
| R2×I3 | within-shaft CAR × HG | physics-matched sEEG candidate |
| R2×I4 | within-shaft CAR × multi-band | richer view on shaft-CAR |
| R3×I4 | bipolar × multi-band | bipolar wide |
| R4×I4 | shaft Laplacian × multi-band | rich shaft-Lap view |
| R4×I5 | shaft Laplacian × wavelet | scale-localized shaft-Lap |
| R4×I6 | shaft Laplacian × theta phase | phase-axis hedge (added 2026-05-09; closes the phase axis missing from Tier-A magnitude views) |
| R4×I2L | shaft Laplacian × log STFT | log-power variant of upstream parity |
| R4×I3W | shaft Laplacian × wide HG (70–250) | high-band richer cousin |
| R*×I1 | any × low-LFP | spectral floor (sub-30 Hz) |

Tier-B fires if (a) Tier-A's R3 vs R4 winner is < 0.005 ahead and CAR (R1/R2) might break the tie, or (b) HG envelope wins and we want to confirm with the multi-band richer cousin, or (c) any Tier-A loser comes from a feature class (phase, wider HG, log-power) not covered by Tier-A.

Stage-1 v14 default reference + input view are frozen from Tier-A unless Tier-B is needed.

#### L.3 — Filtering + Bad-Channel (Sweep 3, after L.2)

Run on the L.2 winner reference+input. Tests whether more aggressive front-end cleaning helps the linear.

| Cell | Recipe | NeuralSet path |
|---|---|---|
| L.3.F0 | none | default |
| L.3.F1 | 60 Hz notch + harmonics | `notch_filter=(60, 120, 180)` |
| L.3.F2 | F1 + 0.5 Hz HPF | `notch_filter=... + filter=(0.5, None)` |
| L.3.F3 | F1 + 1 Hz HPF | `notch_filter=... + filter=(1, None)` |
| L.3.E0 | BT Lite mask only | status quo |
| L.3.E1 | E0 + flatline + amplitude-outlier + clipping | V1-derived per-channel exclusions |

> **2026-05-11 — L.3 FROZEN at F0 (no-op winner; filtering does no work at linear-readout scope).** Tier-A 4-cell sweep on 12 sessions (F0) / 7 sessions (F1/F2/F3) × 15 tasks, ±0.005 noise band (matches L.4 anchor convention):
>
> | rank | cell | recipe | n | mean | sd | Δ vs F0 | sd(Δ) | better/worse/tie |
> |---|---|---|---|---|---|---|---|---|
> | — | **F0** ★ | **no filter (parity to L.2 winner)** | 12 | **0.6132** | 0.0254 | — | — | — |
> | 1 | F2 | F1 + 0.5 Hz HPF | 7 | 0.6043 | 0.0217 | +0.0032 | 0.0059 | 4/1/5 |
> | 2 | F3 | F1 + 1.0 Hz HPF | 7 | 0.6042 | 0.0215 | +0.0031 | 0.0058 | 4/1/5 |
> | 3 | F1 | 60 + 120 + 180 Hz notch | 7 | 0.6027 | 0.0215 | +0.0016 | 0.0058 | 2/3/6 |
>
> **Decision tree application** (rule 1): all three cells' Δ vs F0 fall within ±0.005 noise band — filtering is no-op at this scope. **Freeze L.2 winner (R4×I2 N1) unchanged**; do not fold notch or HPF into the Stage-1 default.
>
> **Why filtering doesn't help here**: stft_abs (L.2 winner) integrates DC drift out of every spectrogram bin; shaft_laplacian (L.2 winner) kills slow common-mode at the reference layer. Both 60 Hz line noise and ≤1 Hz drift are already removed by the upstream contract — F1/F2/F3 are double-cleaning.
>
> **Architectural consequences for v14**:
> - Stage-1 filtering contract: `none` (parity to L.2 winner; rely on STFT + shaft-Lap for cleaning).
> - L.4 (window anchor) and L.5 (probes) inherit F0 — no front-end filter on top of L.2.
> - **Cleaning ceiling reached for the linear baseline.** Architecture work moves to L.4 anchor + L.5 nuisance (already in flight).
> - Notch/HPF are not load-bearing for v14's Stage-2 SSL pretokenizer either; the front-end can stay minimal.
>
> Artifacts: `reports/neuroprobe_stage0_l3_filtering_2026_05_10/{filtering_analysis.md, filtering_analysis.json, launch_manifest.csv}`.

#### L.4 — Window Anchor Robustness (Sweep 4, after L.3)

Run on the L.3 winner. Tests the rebuttal's claim of near-equivalent decoding for 1 s windows starting between roughly `-0.375` and `+0.125` s relative to word onset.

| Cell | Window | Note |
|---|---|---|
| L.4.W0 | [0, 1] s | D.0 default |
| L.4.W1 | [-0.375, 0.625] s | rebuttal lower bound |
| L.4.W2 | [-0.125, 0.875] s | midrange |
| L.4.W3 | [+0.125, 1.125] s | rebuttal upper bound |
| L.4.W4 | [0, 2] s | wider |
| L.4.W5 | [0, 0.5] s | narrower |

Output is a robustness curve, not a single winner. Stage-1 keeps Neuroprobe's [0, 1] s default; this sweep decides whether a Stage-1 anchor-robustness ablation is mandatory or optional.

#### L.5 — Diagnostic Probes (Gates, run after every Sweep winner)

Run on the chosen view from each sweep. **Used as kill criteria, not for selection**: a view that makes the task easier *and* makes nuisance variables more decodable is suspect.

| Cell | Probe | Kill criterion |
|---|---|---|
| L.5.P1 | subject-id from features | KILL: drop view if held-out AUROC > 0.95 |
| L.5.P2 | session-id from features | KILL: drop view if held-out AUROC > 0.95 |
| L.5.P3 | reference-id (R0 vs winner) | POSITIVE sanity: AUROC ≈ 1 expected; surprise if it isn't |
| L.5.P4 | pre-stim window [-1, 0] s on the same task | flag if above chance + 0.05 |
| L.5.P5 | shifted window [+5, +6] s on the same task | flag if above chance + 0.05 |
| L.5.P6 | channel-shuffled-per-subject | KILL: cross-subject task accuracy must drop substantially; if not, channel-order shortcut |
| L.5.P7 | movie-time / block-order from features | flag if above chance + 0.10 |
| L.5.P8 | 60 Hz residual power post-notch | flag if median residual > floor by > 6 dB; bad notch / wrong harmonic set |
| L.5.P9 | acoustic / FM-leakage: linear regression of stim envelope (Hilbert) + f0 (pYIN) + Whisper-large-v3 L8 pooled features from response-window features. Report (a) R² per target, (b) retrieval@10 of held-out brain features against held-out Whisper-L8 features in joint L2-projected space (regression-then-NN baseline). | v14-load-bearing. **Real kill is comparative, not absolute** — Goldstein 2025 reports brain↔Whisper R² peaks at 0.10-0.15 in speech cortex; absolute thresholds rarely trigger. Spec: the v14 contrastive (`L_DSigLIP`) must beat this regression-then-NN retrieval@10 by ≥ 5 points absolute and ≥ 0.05 R² on Whisper-L8. If it doesn't, contrastive is decorative — linear already does the alignment work. Soft flag at any R² > 0.10 on Whisper-L8 (means the SSL has less headroom). |
| L.5.P10 | per-band identity leakage: P1 (subject-id) re-run on each frequency band of `multi_band_log_power` view in isolation (delta / theta / alpha / beta / gamma / HG) | Diagnostic only (not kill): identifies which band carries the subject-ID leak. Informs v14's per-band weighting / suppression. Run only if `multi_band_log_power` becomes a v14 tokenizer candidate. |
| L.5.P11 | feature-permutation null: train logistic on shuffled task labels at the chosen view | KILL: held-out AUROC must be ≤ empirical-chance + 0.02 across 3 permutation seeds. Anything above flags label-into-features leakage in the pipeline. |
| L.5.P12 | split-membership null: train logistic on `is_train_or_test` from the same features. **Collapses to P2 under CrossSession (train/test = different sessions) and to P1 under CrossSubject; only adds new information under WithinSession (random-fold split).** | KILL (WithinSession only): held-out AUROC must be ≤ empirical-chance + 0.05. Distinct failure mode from P11 — P11 catches feature-via-label leakage, P12 catches split-via-feature leakage (e.g., test fold accidentally copied into train, time leaking through chronological-but-labeled-random splits). |
| L.5.P13 | post-aggregation identity: P1 (subject-id) re-run on **DK-region mean-pooled features** (the upstream `combine_regions()` aggregator, applied to the chosen view's features before the linear classifier). | v14-load-bearing. Tests v14's parcel-anchoring premise — if subject-ID is still decodable at AUROC > 0.95 after region-pooling, the anatomical bottleneck does not kill subject identity. Implication: v14's `L_view_invariance` SSL term carries 100% of the invariance load, not the architecture. Soft flag at AUROC > 0.85 (anatomy reduces but doesn't eliminate identity). Soft positive at AUROC < 0.70 (anatomy buys real invariance for free). Run on the L.2-winner view. |

**Notes on probe semantics**

- **Empirical chance floor**: every "above chance + X" threshold uses the empirical majority-class baseline per task, not the nominal 1/n_classes — Neuroprobe class balances are uneven across tasks.
- **P9 architecture coupling**: P9 produces a measurement that the eventual v14 Stage-2 SSL run must beat in retrieval@10. Spec it now (linear infra is still hot) and replay the same regression-then-NN baseline against the v14 contrastive embedding when Stage 2 lands.
- **P13 architecture coupling**: P13 uses DK-region mean pooling as a cheap proxy for v14's BNA-soft-support pooling. Once v14 lands, replay P13 on the actual v14 parcel-pooled features.
- **Deferred to Cogan-stage** (not run at Neuroprobe Stage 0): EMG / myogenic 80-200 Hz contamination (only matters for production-speech tasks, BT is passive listening); audio-bleed through off-task electrode (BT setup vetted upstream); heartbeat-phase decoding (matters for deep mesial temporal contacts, BT Lite is largely lateral-cortical); eye-movement residual (orbital-frontal contacts; rare in BT Lite). All of these become mandatory probes when v14 moves to Cogan PS uECoG / sEEG D-cohort.

#### L.7 — Audio-FM upper-bound baseline (Conwell veRSA control)

**One cell. v14-load-bearing for paper framing.** Tests whether Neuroprobe's 15 tasks are partially solvable from the *audio* alone via a frozen Foundation Model — establishing a "no-brain" upper-bound that v14 must beat by a margin to claim brain-relevance.

| Cell | Pipeline | Output |
|---|---|---|
| L.7.A0 | stim audio (1 s aligned to word onset) → frozen Whisper-large-v3 L8 → mean-pool over time → Logistic Regression → Neuroprobe label, per task, per session, CrossSession protocol | mean test AUROC per task; aggregate summary against L.2 winner R4×I2 (brain-only spectral linear, 0.6132). |
| L.7.A1 | same as A0 but L16 (~50% depth) | layer-depth control |
| L.7.A2 | same as A0 but HuBERT-large L6 (~25% depth, 24-layer encoder) | FM-identity control (not Whisper-specific?) |

**Read-out**: if L.7 AUROC ≥ L.2 winner, Neuroprobe is largely solvable from audio — v14's brain contribution must clear `L.7.A0 + 0.05` to claim brain-decodability is real and not just a sophisticated audio classifier. If L.7 AUROC ≪ L.2 winner, brain features carry information audio-FMs do not — strong v14 framing. Either outcome is publishable.

This is the Conwell 2024 veRSA caution operationalized for our specific FM choice (`memory/feedback_diet_over_architecture_priority_2026_05_09.md`). Distinct from L.5.P9 (P9 regresses brain → Whisper for the inverse direction; L.7 regresses audio → label with no brain involvement).

Implementation: Whisper-large-v3 L8 forward passes are cheap (one-time per session, cache to `/hpc/group/coganlab/ht203/cache_neuroai/whisper_l8_features/`). 12 sessions × 15 tasks × 3 FM-cells = 540 fits, all CPU LogReg, fits inside L.1 wrapper. **Cache shared with L.5.P9** — both L.7 and P9 hit the same Whisper-L8 features (P9 regresses brain → these features; L.7 regresses these features → label). Extract once, both consume.

#### L.6 — Deferred Tier-2 (post-Stage-0 close, only if budget permits)

Not Stage-0 close-criterion. Listed so future-us doesn't reinvent the cell IDs.

- **L.6.ES** electrode-set robustness — *redefined 2026-05-13 to random-dropout robustness only*: keep_n ∈ {60, 90} of 120 Lite electrodes × 3 seeds {42, 43, 44}, sorted by original index to preserve shaft ordering. The full anatomy-120 / Full uncapped / hard-label-restricted variants from the original sketch are deferred until BT shaft/depth geometry contract lands. **In flight 2026-05-13** (`reports/neuroprobe_stage0_l6_es_electrode_dropout_2026_05_13`, 72 jobs).
- **L.6.NR** *narrowband HG sub-band sweep — redefined 2026-05-13 from feature-level nuisance regression*: three narrowband HG envelopes (70-90 / 90-120 / 120-150 Hz) at L.2 winner ref (shaft_laplacian) and L.1 winner norm (train_set_fixed). Feeds the v14 D-SigLIP layer-match decision (whether HG should be banded or broadband for cross-modal alignment). The original feature-level nuisance regression cells (per-trial channel-mean / per-session top-k PCs / ComBat / CORAL) are deferred. **In flight 2026-05-13** (`reports/neuroprobe_stage0_l6_nr_hg_subbands_2026_05_13`, 36 jobs).
- **L.6.WL** *window-length sweep — redefined 2026-05-13 from window length × sub-windowing*: onset-anchored (`anchor_start_before=0.0`) sweep of `anchor_end_after` ∈ {0.25, 0.5, 0.75, 1.0, 1.5, 2.0} s. 1.0 s = L.4 D.0 baseline (control). The sub-windowing variants (1×1 s vs 4×250 ms concat vs 2×500 ms concat vs mean-pool vs 8×125 ms Perceiver-token-grid) are deferred to v14 — they intersect the readout architecture and are best ablated within the Perceiver IO design. **In flight 2026-05-13** (`reports/neuroprobe_stage0_l6_wl_window_length_2026_05_13`, 72 jobs).
- **L.6.CB** class balance: uniform vs inverse-frequency vs effective-number-of-samples vs per-task weighted.
- **L.6.FA** feature aggregation: mean vs median vs PCA-50-per-channel vs PCA-50-cross-channel.

Atlas-coordinate cells (Tier-4) live under Block D (D.public, D.1+) since they extend the upstream linear baseline rather than ablating it. Cross-reference, don't duplicate.

#### L sweep ownership

After each sweep, regenerate `docs/experiments/runs.csv` via `scripts/neuroprobe/collect_experiment_records.py` and update `docs/experiments/stage0_summary.csv` with the cell winner + delta vs. D.0 reference. The frozen Stage-1 default for that knob is recorded in `docs/strategy/stage_1.md`.

### 5. After Surface Mapping Arrives

Run the anatomy-dependent path:

- **A0**: derive BT Tier-1 BNA parcel list from Lite electrode surface positions. Gate: parcel ids in `1..246`; cardinality and LH/RH split recorded; cohort coverage `>=99%`.
- **A1**: verify fsaverage mesh identity and electrode snap distances. Gate: `163842` vertices per hemisphere; mean snap distance `<0.5 mm`; max `<2 mm`.
- **A2**: compare snapped Destrieux labels to BT region labels. Gate: `>=95%` exact match without hemisphere-clustered failures.
- **A3**: verify BNA fsaverage bake at Lite electrode vertices. Gate: overall argmax match `>=90%`; every BT Tier-1 parcel Dice `>=0.85`; no Tier-1 parcel match `<80%`.
- **A4**: compute the BT Lite parcel co-coverage graph. Nodes are BNA Tier-1 parcels; weighted edges count subject/session support for parcel pairs under the approved coverage threshold. Export node coverage, edge weights, connected components, bridge parcels, shortest-path distances, and per-session coverage matrices to `reports/neuroprobe_stage0_a4_parcel_cocoverage_YYYY_MM_DD/`. This is a blocker for any Stage-1/2 claim that v14 can learn cross-parcel completion: direct A-C structure is learnable only when parcels co-occur; indirect A-B-C structure is plausible only through connected overlap paths. The Stage-2 JEPA contract must distinguish covered, intentionally masked, and uncovered parcels before training.
- **C**: build BT support cache using the approved schema.
- **E5 extension**: rerun real NeuralSet smoke with both neural tensor and parcel metadata tensor.
- **D.1+**: run atlas/BNA linear ablations.
- **V7-V8**: plot surface geometry and parcel coverage.

## Block D After D.0

Only D.0 and public hard-label controls can run now. BNA cells still depend on valid surface geometry.

Planned cells after the surface-mapping blocker clears:

| Cell | Eval | Prep | Atlas/pooling | Purpose |
|---|---|---|---|---|
| D.public | CrossSession multiclass | Lap + STFT | public BT hard label mean / support bias | **DK hard-mean pooling — confirmed 2026-05-05 to be what `c7b955b0`'s upstream CrossSubject linear baseline already does** via `combine_regions()`; reframed from "DK control" to "upstream cross-subject baseline architecture itself" |
| D.1a | CrossSession multiclass | Lap + STFT | BNA Tier-1 hard mean | BNA hard vs DK baseline |
| D.1b | CrossSession multiclass | Lap + STFT | BNA Tier-1 soft support | soft vs hard support |
| D.2 | CrossSession multiclass | CAR + HG | DK mean | prep-only control |
| D.3a | CrossSession multiclass | CAR + HG | BNA hard mean | HG + BNA hard |
| D.3b | CrossSession multiclass | CAR + HG | BNA soft support | HG + BNA soft |
| D.5 | CrossSession multiclass | CAR + HG | old PS LH-only parcels | anti-control |
| D.8 | CrossSession multiclass | Lap + STFT | zero-fill missing parcels | tests always-include parcel commitment |
| D.10 | CrossSession multiclass | engineered composite | BNA soft support | Better-Linear candidate |
| D.11 | CrossSession multiclass | raw 2048 Hz | BNA soft support | v14-aligned raw linear ceiling |
| D.12 | CrossSession multiclass | Laplacian raw | BNA soft support | raw re-reference check |
| D.13 | CrossSession multiclass | Laplacian/local + HG/HFA | BNA soft support | biologically privileged local population-firing view |

D.11 is no longer a default-input coronation test. It is the raw-view ceiling and artifact-risk baseline. The Stage-1 input decision must compare D.11/D.12/D.13 against D.0b and V0-V6 QC. Raw voltage keeps a role as an auxiliary or ablation view because it preserves information and matches existing Neuroprobe models, but Laplacian/local spectral or HG/HFA views are the biologically privileged candidates after the Pesaran artifact review.

Conditional cells:

- D.4 if Tier-1 looks too narrow.
- D.6/D.7 if D.10 is strong enough to warrant attribution.
- D.9 only if WM rejection is approved as a free label/filter.
- D.14 pooled multi-source CrossSubject multiclass split. Train on all allowed source subjects/sessions and test held-out subjects/sessions. This is the scientific generalization default. Implement locally if upstream still lacks it.
- D.15 upstream all-source CrossSubject robustness if Christopher's newer `include_all_train_subjects=True` code lands. Record whether it is pairwise 1-to-1 or pooled N-to-1; do not assume from rebuttal text.
- D.16 electrode-set robustness: Lite-120 parity set, random-120, anatomy-120, and full/uncapped where public data permits. Use Lite-120 for leaderboard parity only.

Stage 1 should also carry a small window-anchor robustness cell around Neuroprobe's `[0, 1]` s window because the rebuttal reports near-equivalent decoding for 1 s windows starting between about `-0.375` and `+0.125` s relative to word onset.

## Statistical Methods (binding for all Stage-0 reports)

Single appendix so every L-sweep + V-QC + D-block + L.5 probe uses the same machinery. Reviewers in 2026 expect explicit stat-method specs; per-cell choices are not defensible.

- **CIs on AUROC / accuracy**: bootstrap N=2000, percentile method, sampling units = (session, task) pairs. For multi-task aggregates, resample (session, task) pairs jointly so cross-task correlation is preserved. Report `[lo, hi]` alongside means in every freeze decision.
- **Paired comparisons across cells**: paired Wilcoxon signed-rank on (session, task)-paired AUROCs (one signed delta per (session, task) pair, two cells). Report W, n, p, rank-biserial r as effect size.
- **Multi-cell screening**: when comparing K > 2 cells against a baseline, apply Benjamini-Hochberg correction at α = 0.05 across the K-1 paired tests. Report both raw and BH-adjusted p; interpretive decisions use BH-adjusted.
- **Effect-size threshold for "load-bearing"**: ΔAUROC ≥ 0.02 (multiclass CrossSession) for a cell to override the upstream / inherited default. Below 0.02, freeze upstream parity (decision rule 2 from L.1). The threshold is calibrated to the published 95% CI half-width of the Neuroprobe linear baseline (~0.018-0.020 across 12 Lite sessions) — anything under one CI half-width is below noise.
- **Seed variance**: each freeze decision quoted with ≥ 3 seeds (42, 43, 44) on the chosen cell + nearest competitor. Report seed-variance SEM separately from session-task SEM. Cells whose ΔAUROC is within 1× seed-variance SEM are not load-bearing regardless of CI.
- **Multiple-comparison correction across L-sweeps**: do not BH-correct across sweeps (L.1, L.2, L.3, L.4 are independent decisions on independent axes). Within a sweep, BH-correct across cells.
- **L.5 probes**: kill criteria use raw thresholds (no MC correction) because they are conservative pre-registered floors, not exploratory tests. P9 / P13 measurements report R² + retrieval@10 with bootstrap CI but no kill-threshold p-value.
- **Session-token leakage**: every paired comparison must verify train/test (session, task) pairs do not overlap. Add an assert to every freeze-analyzer script.
- **Reproducibility pinning**: every report README cites upstream Neuroprobe commit (`c7b955b0a31464f4a5eec3f3bd78ff29841d61ac`), Whisper-large-v3 HuggingFace commit (record on first L.7 / L.5.P9 run), `pyproject.toml` + `uv.lock` git SHA at run time. Version-drift between report regenerations is a defect.

## Stage-1 Entry Pre-Commitments

The architectural ablation roster v14 must run between Stage-1 dispatch and Stage-1 close. These are not Stage-0 work but are spec'd here so the handoff is unambiguous and Stage-0's frozen contracts (L.1 N1 + L.2 R4×I2 + L.3 winner + L.4 robustness) feed directly into them. Canonical source: `memory/project_v14_paper_corrections_post_newpapers6_2026_05_09.md` + `memory/project_v14_p_emb_drift_ablation_2026_05_09.md`.

| Cell | Tests | Why |
|---|---|---|
| **AC1 FM-swap** | v14 with frozen Whisper-large-v3 L8 → swap to HuBERT-large L9 / WavLM-large L9 / EnCodec / w2v-BERT-2-mid at fixed v14 architecture | Conwell "diet > arch" test for our specific FM choice. If FM identity dominates architecture, v14's contribution is FM-selection, not the architecture. |
| **AC2 frozen-features linear probe** | Whisper-L8 (or per-task winning FM from AC1) → linear → Neuroprobe labels, no brain features | Must be beaten by v14's brain-FM contrastive. If linear-on-FM-only matches v14, the brain contribution is decorative. **Identical pipeline to L.7.A0 — L.7 IS the AC2 baseline run early.** |
| **AC3 anatomy-blind random Perceiver** | Same Perceiver IO architecture, same M·d budget, but `parcel_latents[p, m]` initialized **random** (no `P_emb[p]` BNA prior) and **no `log(support[i,p])` Graphormer cross-attn bias** | Bhattacharjee 2024 SRM PCA-control analog: tests whether anatomy-as-routing actually does work vs random latents. If random-Perceiver matches v14, anatomy is decorative — biggest single architectural ablation. |
| **AC4 P_emb drift** | Unfreeze `P_emb[p]` (BNA-init, learnable) while keeping `support[i,p]` anatomy-fixed and `log(support)` cross-attn bias active | Tests Cogan's functional-vs-anatomical-alignment question. Triangulates with AC3: full v14 (both fixed) vs P_emb-drift (routing only) vs anatomy-blind (neither). Free interpretability story either way: either anatomy-as-content-prior holds, or attention finds a better functional prior than BNA. |

Each cell reports pooled multi-source CrossSubject multiclass + S2/trial-4 CrossSubject parity + per-task breakdown using Stage-0's stat-method appendix. AC2 (= L.7) lands earliest because it needs no architecture; AC3/AC4 land last.

## Close Criteria

Stage 0 closes when:

- D.0a and D.0b aggregate reference baselines are reproduced, with D.0b task-level reference drift documented.
- V0-V6 QC is reviewed and any data exclusions are decided.
- Lite-vs-Full selection effects are documented, including whether Full/uncapped data can support robustness analyses.
- Reference transforms are treated as first-class provenance-bearing transforms, including virtual-channel metadata and unresolved geometry/support status where applicable.
- Surface mapping is resolved or a fallback is approved.
- Public hard-label coverage is reviewed as a DK/Destrieux control and kept distinct from BNA/fsaverage support.
- BT shaft/depth geometry contract is frozen, including shaft parser, contact-order orientation, transferable depth features, local-reference provenance, and nuisance-probe gates.
- A0-A3 pass on valid geometry.
- A4 co-coverage graph is reviewed, including disconnected parcels and bridge parcels, and its implications for Stage-1 architecture claims and Stage-2 JEPA loss masks are written down.
- The Stage-1 input-view matrix is decided from V0-V6, L.2, and D.11/D.12/D.13, with explicit notes on reference physics, artifact burden, Lite-vs-Full shift, and whether raw voltage is default, auxiliary, or ablation-only.
- The Stage-1 normalization contract is decided from L.1, with train-set/session-level normalization compared against window-local normalization. The chosen recipe is recorded in `docs/strategy/stage_1.md`.
- The Stage-1 reference transform is decided from L.2, with explicit notes on R0/R1/R2/R3/R4 deltas and WM-rejection treatment.
- The Stage-1 filtering and bad-channel mask is decided from L.3.
- The Stage-1 anchor-robustness ablation is marked mandatory or optional from L.4.
- L.5 diagnostic probes have run on each Sweep winner, and no chosen view fails the L.5.P1/P2/P6/P11/P12 hard kill criteria. P9 (FM-leakage) and P13 (post-aggregation subject-ID) are v14-load-bearing measurements rather than hard kills — their results are recorded as baselines that v14's contrastive loss (P9) and parcel pooling (P13) must beat at Stage 2 dispatch. P10 runs only if `multi_band_log_power` becomes a v14 tokenizer candidate. P12 only adds new information under WithinSession splits.
- L.7 audio-FM upper-bound has run (L.7.A0 minimum; A1/A2 if A0 is competitive with L.2 winner) and the per-task table comparing L.7 to L.2 R4×I2 is recorded. Stage-1 v14 must clear `L.7.A0 + 0.05` to claim brain-decodability is real.
- V0.x stimulus-overlap audit has run and the train/test stimulus overlap fraction per (subject, task) is recorded. Tasks with > 50% overlap are flagged in V6 as stimulus-recognition-confounded and reported alongside L.2/L.3 numbers.
- All freeze decisions (L.1, L.2, L.3 winner, L.4 anchor robustness) cite the Statistical Methods appendix: bootstrap N=2000 percentile CIs, paired Wilcoxon + rank-biserial, BH within sweep, ≥ 3 seeds on chosen + nearest competitor, train/test pair-overlap assert, upstream-commit + Whisper-commit + uv.lock SHA pinned.
- L.0 prerequisites have all landed: `Experiment` inherits `BaseExperiment`, exca cache folder is set on DCC, `DeriveLabelIndices` `EventsTransform` is wired into a `ns.Chain`, the linear baseline wrapper writes `ExperimentLogger` sidecars, and `CARIeegExtractor` + `ShaftCARIeegExtractor` exist.
- Neuroprobe-Lite temporal sampling behavior is verified from code, not assumed from review-thread claims.
- The Stage-1 split contract is written down: multiclass default, pooled multi-source cross-subject generalization default, S2-only cross-subject parity cell, and Lite-120 electrode cap parity-only.
- C support cache is built and schema-tested.
- E5 passes with neural + parcel metadata tensors aligned.
- D.1+ required linear cells are collected and interpreted.
- `docs/experiments/runs.csv` and Stage 0 summary CSVs are regenerated from sidecar records.

## Explicit Non-Goals

- No v14 architecture implementation.
- No neural network training.
- No SSL pretraining.
- No full BrainTreebank pretraining download.
- No BNA support cache from public PNG plotting coordinates.
- No D.1+ atlas/BNA linear cells until valid surface geometry exists.
