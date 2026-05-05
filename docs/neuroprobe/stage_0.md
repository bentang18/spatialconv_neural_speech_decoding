# Neuroprobe Stage 0 — Current Execution Plan

*Last revised 2026-05-01.*

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

## Remaining Work

### 1. Run V0-V6 Data QC

This is the main unblocked work while waiting for Christopher.

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
- **L.2/L.3/L.4** will extend the same wrapper by adding NeuralFetch-driven preprocess swaps (reference + filter + window anchor). The N1 cell of L.1 is the byte-equivalent reproduction of the upstream D.0 baseline; this serves as the regression check on the wrapper's own pipeline before any new axis sweeps.

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
- **L.0c** *(Stage-1-entry preparation, NOT an L-sweep blocker)* Write a `DeriveLabelIndices` `EventsTransform` and a `Wang2024Treebank` `ns.Chain` so the events DataFrame carries `code` + `split` before the `Segmenter` sees it. This unblocks running D.0 and L cells through our canonical `Experiment` class. L sweeps themselves are dispatched via the upstream-wrapper path (`scripts/neuroprobe/run_upstream_linear_baseline.py` extended with the relevant config flag), which already works and writes `ExperimentLogger` sidecars. Treat L.0c as a Stage-1-entry deliverable; it shouldn't gate L.1.
- **L.0d** Verify `scripts/neuroprobe/run_upstream_linear_baseline.py` wraps each run in `ExperimentLogger`. If not, retrofit so D.0 cells and all L cells write the canonical `experiment_record.json` sidecar that `collect_experiment_records.py` aggregates into `docs/experiments/runs.csv`.
- **L.0e** Write `CARIeegExtractor` and `ShaftCARIeegExtractor` subclasses in `src/speech_decoding/extractors/reference.py`. NeuralSet's `IeegExtractor` exposes `bipolar_ref`, `notch_filter`, `filter`, `apply_hilbert`, `scaler`, `clamp` as constructor kwargs — those cover R0/R3/R4 and the F-/I-family cells via config swaps. CAR has no first-class kwarg; needs a thin subclass that subtracts the per-channel-mean post-load. Without this, L.2 cells R1 and R2 cannot run.

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

#### L.2 — Reference × Input-View (Sweep 2, after L.1)

Two crossed axes. Reference transform changes the measurement operator (`y' = R y = R L s + R c + R noise`); input view changes which signal property is exposed. Both are configured by swapping `IeegExtractor` configs (except R1/R2 which need L.0e).

Reference (rows):

| Cell | Recipe | NeuralSet path |
|---|---|---|
| R0 | raw monopolar | default `IeegExtractor` |
| R1 | robust global CAR | `CARIeegExtractor` (L.0e) |
| R2 | within-shaft local / shaft-CAR | `ShaftCARIeegExtractor` (L.0e) |
| R3 | bipolar (adjacent-pair) | `bipolar_ref=True` |
| R4 | shaft Laplacian | `bipolar_ref=...` shaft-aware |
| R5 | WM-rejected variants of R2/R3/R4 | gated on Chris's WM-flag answer |

Input view (columns):

| Cell | Recipe | NeuralSet path |
|---|---|---|
| I0 | raw 2048 Hz voltage | default `IeegExtractor` |
| I1 | low-frequency LFP (<30 Hz) | `filter=(None, 30)` |
| I2 | STFT log-power | downstream transform |
| I3 | HG/HFA envelope (70–150 Hz) | `filter=(70, 150) + apply_hilbert=True` |
| I4 | multi-band log-power (6 bands) | downstream transform |
| I5 | wavelet (Morlet, 6 scales) | downstream transform |

Earlier drafts listed I6 = Lap+spec and I7 = CAR+HG as input views. They are not — they are (reference, view) pairs (R4, I2) and (R1, I3) respectively. Dropped to remove alias collisions. The D.0 default "Lap+spec" is now spelled R4×I2, and the conventional CAR+HG control is R1×I3.

Don't run all 5×6=30. Pre-register a 12-cell hand-picked subset covering the physically meaningful combinations:

| Cell | Recipe | Role |
|---|---|---|
| R0×I0 | raw monopolar × raw 2048 Hz | floor / v14-aligned ceiling |
| R0×I2 | raw × STFT | spectral without re-reference |
| R1×I3 | global CAR × HG | conventional control (was I7) |
| R2×I3 | within-shaft CAR × HG | physics-matched sEEG candidate |
| R2×I4 | within-shaft CAR × multi-band | richer view on shaft-CAR |
| R3×I2 | bipolar × STFT | bipolar spectral |
| R3×I3 | bipolar × HG | bipolar HG conventional |
| R3×I4 | bipolar × multi-band | bipolar wide |
| R4×I2 | shaft Laplacian × STFT | **D.0 default ("Lap+spec")** |
| R4×I3 | shaft Laplacian × HG | biologically privileged |
| R4×I4 | shaft Laplacian × multi-band | rich shaft-Lap view |
| R4×I5 | shaft Laplacian × wavelet | scale-localized shaft-Lap |

Stage-1 v14 default reference + input view are frozen from this sweep.

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

#### L.6 — Deferred Tier-2 (post-Stage-0 close, only if budget permits)

Not Stage-0 close-criterion. Listed so future-us doesn't reinvent the cell IDs.

- **L.6.ES** electrode-set robustness: Lite-120 vs random-120 (×3 seeds) vs anatomy-120 (DK gray-matter top-120) vs Full uncapped vs hard-label-restricted.
- **L.6.NR** feature-level nuisance regression: regress out per-trial channel-mean / per-session top-k PCs / subject-mean / ComBat / CORAL.
- **L.6.WL** window length × sub-windowing: 1×1 s vs 4×250 ms concat vs 2×500 ms concat vs mean-pool sub-windows vs 8×125 ms (matches Perceiver token grid).
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
- L.5 diagnostic probes have run on each Sweep winner, and no chosen view fails the L.5.P1/P2/P6 kill criteria.
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
