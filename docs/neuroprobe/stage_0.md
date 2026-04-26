# Neuroprobe Stage 0 — Reproduce Linear baselines + pipeline rigor + linear ablation matrix

*Drafted 2026-04-24. Revised 2026-04-25 to reflect: (i) BT-derived Tier-1 selection (the LH-only PS Tier-1 is wrong for BT); (ii) hardened BNA bake gate (must-pass, no soft-fail); (iii) dropped A3.5 Gaussian-volumetric alternate readout (structurally unavailable for BT — pial-snap discards volumetric truth); (iv) expanded Block D from "reproduce 0.539" into an 8-cell linear ablation matrix that diagnoses every claim about the v14 prior at the linear-decoding regime, before any neural-net compute; (v) excluded missing electrodes by default rather than zero-row; (vi) added Cogan-lab convention questions to A4 for Stage-3 prep; (vii) extended Block D to 13 cells with the D.6–D.10 Better-Linear feature-engineering matrix (log-power, L1, zero-fill, WM-rejection, composite) — diagnoses whether v14's atlas-anchoring claim is already pre-empted by smarter feature engineering before any neural-net compute, and yields a paired-submission decomposition story.*

Strategy anchor: `docs/neuroprobe/plan.md`. Project memory: `memory/project_neuroprobe_cross_subject_hillclimb_2026_04_22.md`. Benchmark reference: `docs/references/neuroprobe_benchmark.md`.

## Stage goals

Two interlocked goals — neither alone closes the stage.

1. **Reproduce the #1 Linear (Laplacian+spectrogram) cross-subject baseline = 0.539 on DCC.** Protocol-correctness proof. If we can't match the leaderboard within ±0.005 per session-task, we don't understand the eval protocol and cannot trust Stage-1/2 comparisons.
2. **Validate every pipeline primitive we'll reuse in Stages 1–3** *and* **diagnose every claim about the v14 prior at the linear-decoding regime, before spending neural-net compute.** Coordinate lookup on fsaverage, BNA bake readout at BT electrode positions, support cache schema, loader contract — each gets a pass/fail gate. Plus a 13-cell linear ablation matrix (Block D) in two groups: D.0–D.5 isolate v14-prior axes (atlas, prep, soft-vs-hard, Tier-1 selection); D.6–D.10 test the orthogonal axis — feature-engineering ceiling on a "Better Linear" baseline (log-power, L1, zero-fill, WM-rejection, composite). Answers we read *before* Stage 1 instead of post-mortem after.

**Stage 0 closes when:** Blocks A + B + C + D pass their gates. Block E (loader scaffold) lands inside Stage 0 in parallel with D but its only gate is "loader runs end-to-end on sub_2/trial_4 without error" — Stage 1 should be able to start the day Block D passes.

## Known limitation — Cogan-lab volumetric convention is unavailable for BT

Cogan lab's canonical sEEG BNA-sampling convention is **patient-space volumetric**: for each contact, aggregate BNA mass over a 3 mm sphere centered on the native-T1 RAS coordinate, lookup against per-patient `aparc.BN_atlas+aseg.mgz` (volumetric BNA + aseg merge, queried in patient T1 space). Source format: `ECoG_Recon/<D>/elec_recon/<D>_elec_location_radius_{1,2,3,5,7,10}mm_aparc.BN_atlas+aseg.mgz.csv`. Per-electrode CSV with columns = BNA parcels + cortical/subcortical labels; values are probabilistic mass [0, 100] within the radius sphere.

Verified parity-cached for 122 D-patients via `scripts/bna_parity_dpatients.py`. The 2026-04-21 audit (`reports/bna_parity_dpatients_2026_04_21/`) found surface-snap inflates Tier-1 support **2.2×** vs the volumetric truth — surface snap aliases WM-resident depth contacts onto cortex they aren't actually sampling.

For BrainTreebank we **cannot** apply this convention:

- BT releases only **fsaverage-projected** coords (`coordinates_type="cortical"`) and **patient-LPI** coords (`coordinates_type="lpi"`). `coordinates_type="mni"` raises NotImplementedError.
- BT does **not** release per-patient FreeSurfer recons. So we have no per-patient `aparc.BN_atlas+aseg.mgz` to query volumetrically, and no patient sphere registration to do a custom projection ourselves.
- BT's pipeline (per their §A.5) projects GM and GM-WM-boundary contacts to the patient pial along the cortical normal *before* mapping to fsaverage. Non-GM (deep WM, subcortical) contacts are **omitted from the fsaverage CSV**, which is why 15 of the 1160 Lite electrodes have no `elec_coords_full.csv` entry (sub_9 T1aI1..8 = 8 contacts; sub_10 F10Fa1..4 + F10Fa14..16 = 7 contacts). Those 15 are correctly omitted by BT — they're the WM contacts whose projection would be physically meaningless. Not a bug, a feature.
- Consequence: **the 2.2× inflation pattern from the D-cohort audit applies to BT too, but baked in by BT's pipeline, not by ours.** A 3D Gaussian readout on top of BT's snapped coords would not recover the volumetric truth — the volumetric information was discarded the moment they snapped. Any volumetric-truth correction would require BT to release recons or raw T1s. (This is why the originally-proposed A3.5 Gaussian alternate readout was dropped from this plan.)

**Mitigation we can do.** A2 (Destrieux match) verifies the projection is *self-consistent* (≥95% agreement with BT's `Region` column = projection isn't broken). A3 (BNA bake rigor) verifies our bake at the projected positions is correct. We document the inflation as a known cross-cohort comparison limit in the eventual paper / submission write-up.

**Stage-3 implication.** When we resume the D-cohort program, Cogan-lab volumetric remains our default (122 D-patients already cached). The limit is BT-specific.

## Data footprint

Stage 0 only needs the Neuroprobe Lite eval subset (12 trials per `NEUROPROBE_LITE_SUBJECT_TRIALS`) plus metadata zips — exactly what BT's `braintreebank_download_extract.py --lite` produces. Tier-1 pretrain sessions (14 full + 5 partial whitelist) are **not** downloaded in Stage 0 — they enter in Stage 2. Estimated DCC footprint: ~50 GB unzipped at `/work/ht203/data/braintreebank/` (12 h5 files + ~530 MB metadata).

## Blocks

### Block A — Coordinate + BNA pipeline rigor (local, ~1 day local + async wait on A4)

Cheap. Runs against metadata zips already at `/tmp/bt_metadata/` + our existing `data/atlas/fsaverage_bake_v2c/`. No DCC required. A4 (Zac/Nanlin sign-off) comes after A0–A3 land — we present numbers, not hypotheses.

**Pre-conditions for A2/A3:** local FreeSurfer install with `fsaverage` subject for Destrieux (`$FREESURFER_HOME/subjects/fsaverage/label/{l,r}h.aparc.a2009s.annot`); BNA authors' shipped fsaverage annot at `data/atlas/BN_Atlas_freesurfer/fsaverage/label/{l,r}h.BN_Atlas.annot`. Default FS path `/Applications/freesurfer/8.2.0/...` per `scripts/verify_bna_fsaverage_bake.py`. Confirm both before A2.

- **A0 — BT-derived Tier-1 parcel list.** `src/speech_decoding/v14/token_spec.py::DEFAULT_BASE_PARCELS` is **LH-only, 15 parcels, frozen for Phase-1**, derived from the PS uECoG cohort (7 LH patients, motor strip + STG concentration). It is the wrong list for BT — BT's 1145 Lite-with-coord sEEG contacts span the whole brain bilaterally (frontal, MTG, hippocampus, insula, RH everything) and the 15 Neuroprobe tasks include visual / linguistic / audio whole-brain decoders. A naive LH→RH mirror just doubles the wrong list. **Fresh derivation:** snap each Lite-with-coord electrode to nearest fsaverage vertex, compute argmax BNA parcel id, count argmax-wins per parcel across the cohort. Tier-1 = **every parcel with ≥1 argmax-win** (no further threshold — we do not bias toward speech-motor, the eval is whole-brain). Land as `BT_TIER1_PARCELS` in **new** module `src/speech_decoding/neuroprobe/atlas_tier1_bt.py`. **Do not mutate `token_spec.py`** — Phase-1 contract is frozen. **Gate:** all parcel ids in 1–246; cardinality recorded; LH/RH distribution recorded; cohort coverage stat (% of 1145 electrodes whose argmax lands in the list) ≥99%.

- **A1 — Mesh identity.** Verify our bake uses standard 163,842-vertex/hemi fsaverage. Vertex count is recorded in `data/atlas/fsaverage_bake_v2c/bake_manifest.json` — confirm via `src/speech_decoding/v14/fsaverage_atlas.py::load_bake_manifest`. Then snap each of the 1145 Lite-with-coord electrodes; compute snap-distance histogram. **Gate:** vertex count = 163,842/hemi; mean snap distance <0.5 mm; max <2 mm. Failure = different mesh subdivision than BT, re-derive bake on correct mesh.

- **A2 — Destrieux cross-check (projection end-to-end verification).** For each Lite electrode: snap to nearest fsaverage vertex, look up Destrieux label from `aparc.a2009s.annot`, compare to BT's `Region` column in `elec_coords_full.csv`. **Gate:** ≥95% exact-label match across all 1145 electrodes; disagreements must not cluster by hemisphere. **This is the single sharpest integrity test** — it validates mesh alignment, axis convention, hemi sign, and vertex lookup simultaneously, *without* depending on our BNA bake. It is also the empirical verification that BT's coords actually live on fsaverage — if A2 passes, the BT-fsaverage assumption is verified end-to-end. Failure = XYZ axis order, hemi sign convention, or mesh version is wrong.

- **A3 — BNA bake rigor at Lite electrode positions (must-pass, no soft-fail).** Extend `scripts/verify_bna_fsaverage_bake.py` (which already runs the cohort-wide bake-vs-published-annot per-vertex argmax compare) to **restrict the comparison to the 1145 Lite-electrode nearest-vertex set**. Add a `--electrodes-csv` flag or a wrapper script `scripts/neuroprobe/verify_bna_at_lite.py` that reuses the existing comparison code. Tabulate disagreements per parcel.

  **Gates (hardened, all must pass):**
  - Overall argmax match ≥**90%** on Lite vertices (was ≥85% cohort-wide; tightening for the Lite-only restriction since we only need the bake correct where we read it).
  - Per-parcel Dice ≥**0.85** for **every parcel in the BT-derived Tier-1 list** (A0 output) — not just speech-motor; the whole BT-Tier-1.
  - Zero parcels in BT-Tier-1 with overall match <80% (no quiet sinks).

  **Failure mode:** re-bake using authors' GCS classifier on fsaverage directly, instead of the current `mri_vol2surf --projfrac-avg` column-walk. Stage 0 blocks until re-bake passes. **No soft-fail — every Stage 1+ claim about atlas-anchored cross-subject transfer rests on the bake being correct at every electrode position we ever read.**

- **A4 — Zac / Nanlin sign-off (async, 0–3 days).** Present A0–A3 results plus the 15 missing Lite electrodes (sub_9 T1aI ×8, sub_10 F10Fa ×7) to Zac and Nanlin.

  **Stage-0-blocking asks:**
  1. Is BT's native-to-pial nearest-neighbor projection trustworthy for sEEG depth contacts (within the GM-only restriction BT applies)?
  2. Is **excluding** the 15 missing electrodes from the loader (default — see C2) preferable to zero-support emission with `active_mask=0`? They are functionally equivalent for attention architectures with mask support; differ only if a Stage-1+ variant uses subject-fixed electrode index as a feature (we currently don't).
  3. Is the BNA bake disagreement rate from A3 acceptable for Stage-0 proceed, or do we re-bake?
  4. Is `localization/sub_*/depth-wm.csv` a per-electrode WM/GM column we can use as a free filter (gates Block-D D.9 cell), or does it require its own labeling pass?

  **Stage-3 prep asks (non-blocking, while we have her in conversation):**

  5. Cogan-lab canonical convention questions for the D-cohort program:
     a. Is **3 mm** her canonical sphere radius for the model-input convention, or a different radius (1/2/5/7/10 mm available in the cached CSVs)?
     b. **Argmax-only**, **top-k**, or the **full weighted support vector** as the model input?
     c. **Probability normalization** — raw [0, 100] (current `support_cache.py` schema), softmax, or unit-sum?
     d. **Tier-1 selection rule** for D-cohort — same `argmax_wins ≥ N` family, or different?

  Sign-off on (1)–(4) gates Blocks C–E. (5) lands in a separate Stage-3-prep memory note (see F2); not Stage-0-blocking.

### Block B — Data download to DCC (~0.5 day, parallel with A)

- **B1 — DCC scratch budget.** Confirm ≥200 GB free on `/work/ht203/`. Stage-2 footprint (full 27 trials) = ~270 GB; reserve.
- **B2 — Run BT's own `--lite` download script.** Use `braintreebank_download_extract.py --lite` from `insight-neuro/neuroprobe` (cloned at `/tmp/neuroprobe_explore/neuroprobe/` locally; rsync to DCC under `/work/ht203/repo/neuroprobe/` for execution). The `--lite` filter targets exactly `NEUROPROBE_LITE_SUBJECT_TRIALS = [(1,1),(1,2),(2,0),(2,4),(3,0),(3,1),(4,0),(4,1),(7,0),(7,1),(10,0),(10,1)]` plus all metadata zips, downloads to `braintreebank_zip/`, and extracts to `braintreebank/`. Total ~50 GB. **Do not roll our own curl loop** — upstream handles DC-trigger files, partial extracts, and root-vs-data path quirks correctly. Move/symlink the resulting `braintreebank/` into `/work/ht203/data/braintreebank/` so paths.yaml entries are stable.
- **B3 — Integrity check.** For each downloaded h5 (`sub_N_trialT.h5`), open with `h5py` and verify the per-subject electrode count matches `len(NEUROPROBE_LITE_ELECTRODES["btbankN"])`. (No upstream SHA available; `Content-Length` is the only server-side hint and `--lite` already retries truncated downloads.) Script: `scripts/neuroprobe/verify_bt_download.py`. **Gate:** all 12 trials open cleanly; per-subject electrode counts match.

### Block C — BT support cache build (~0.5 day, depends on A passing)

- **C1 — `scripts/neuroprobe/build_bt_support.py`.** Consume `elec_coords_full.csv` + our fsaverage bake → nearest-vertex per electrode → BNA support vector restricted to BT-Tier-1 (A0 output) → cache at `/work/ht203/data/bt_support/<sub>_support_tier1.csv`. **Reuse the v14 support-cache IO contract**: call `src/speech_decoding/v14/support_cache.py::write_support_cache` and `src/speech_decoding/v14/fsaverage_atlas.py::sample_baked_support` directly. The `TIER1_COLUMNS` / `TIER1_BNA_INDICES_1BASED` tuples must be **parameterized** on the BT-Tier-1 list — write a thin BT-specific wrapper module `src/speech_decoding/neuroprobe/support_cache_bt.py` that builds `BT_TIER1_COLUMNS` / `BT_TIER1_BNA_INDICES_1BASED` analogously to the v14 constants. **Do not invent a new CSV format.** Stage-1 code loads via the same `read_support_cache` primitive.

- **C2 — Missing-electrode contract: exclude (default).** The 15 Lite electrodes with no `elec_coords_full.csv` entry are **excluded from the loader's electrode list** by default. This is consistent with what BT's own `combine_regions()` does (drops DK-Unknown electrodes silently) and matches the volumetric-truth picture (those contacts are non-GM and have no meaningful BNA labels — see Known Limitation above). Per-subject electrode count drops accordingly: sub_9 from 93 → 85, sub_10 from 120 → 113; others unchanged. Loader emits a structured log entry per excluded electrode (subject, name, reason: "no fsaverage coord; non-GM contact omitted by BT").

  **Optional `--include-zero-support` flag** preserves the 15 with all-zero support rows + `active_mask=0`. Functionally equivalent for attention architectures (zero attention contribution, zero gradient) but useful if any future Stage-1+ architecture variant uses subject-fixed electrode index as a feature. We do not currently use such a variant.

- **C3 — Corrupted-contact mask.** Apply `corrupted_elec.json` at load time. Block A confirms zero intersection between the Lite set and the corrupted list, but emit the mask anyway so Stage-2 pretraining (which sees non-Lite electrodes) inherits the filter for free.

### Block D — Linear ablation matrix (~3 days, depends on A + B passing)

Restructured from the original "reproduce 0.539 only" into a **13-cell ablation matrix** in two groups: D.0–D.5 isolate v14-prior axes (atlas, prep, soft-vs-hard, Tier-1 selection); D.6–D.10 test the orthogonal axis ("how much of the v14 thesis evaporates if you just engineer the linear baseline harder?"). All at the linear-decoding regime, **before any neural-net compute**. The post-mortems we'd otherwise do at Stage-1 failure are pre-empted: every claim about the v14 prior gets a number we read before Stage 1 spends a single GPU-hour.

**All 13 cells are must-run.** Linear regression on ≤3500 samples is CPU-cheap (~800 core-hours total for all 13 cells, i.e. ~1950 jobs). D.6–D.9 are one-line hyperparameter switches on the D.0 pipeline (~37 core-hours each). D.10 is the composite "Better-Linear" submission candidate. The marginal cost of D.6–D.10 over D.0–D.5 is ~200 core-hours — trivial against the value of (a) catching whether v14's atlas-anchoring claim is already pre-empted by smarter feature engineering, and (b) holding a credible Better-Linear submission that decomposes which v14 component is load-bearing.

| Cell | Re-ref + spectral | Atlas | Pooling | Hypothesis tested |
|---|---|---|---|---|
| **D.0** | Lap + STFT (theirs) | DK `combine_regions()` (theirs) | mean | Reproduce 0.539. Protocol-correctness gate. |
| **D.1a** | Lap + STFT | BNA BT-Tier-1 (ours) | argmax-hard + mean | BNA-mean vs DK-mean (their prep held constant). |
| **D.1b** | Lap + STFT | BNA BT-Tier-1 | probabilistic-support weighted sum | Soft vs hard support (their prep held constant). |
| **D.2** | CAR + HG (ours) | DK `combine_regions()` | mean | Our prep vs theirs (their atlas held constant). |
| **D.3a** | CAR + HG | BNA BT-Tier-1 | argmax-hard + mean | Full-stack-ours, hard. |
| **D.3b** | CAR + HG | BNA BT-Tier-1 | probabilistic | Full-stack-ours, soft. The "is v14 even needed?" linear ceiling. |
| D.4 | CAR + HG | full BNA 246 (no Tier-1 filter) | probabilistic | Does BT-Tier-1 selection help, hurt, or wash? |
| D.5 | CAR + HG | Phase-1 PS LH-only 15 (`DEFAULT_BASE_PARCELS`) | probabilistic | Anti-control: confirm the PS Tier-1 list is wrong for whole-brain bilateral BT. |

#### Better-Linear feature-engineering extensions (D.6–D.10)

D.0–D.5 test the v14 prior at the linear regime. D.6–D.10 test the orthogonal axis: **how much of the v14 thesis evaporates if you just engineer the linear baseline harder?** Each of D.6–D.9 is a +1 change vs D.0; D.10 is the composite "Better Linear" — submission candidate if it lands ≥ 0.55. Strategic motivation: a paired-submission story (Better Linear flat-mean baseline + v14 learned within-parcel attention) decomposes which v14 component is load-bearing far more cleanly than v14-alone vs leaderboard-#1.

| Cell | Change vs D.0 | Hypothesis tested |
|---|---|---|
| **D.6** | + `log(power + ε)` before standardize | Distribution-matched scaling for linear classifier. Free expected 0.003–0.008. |
| **D.7** | + L1 (or elastic-net) instead of L2 | Right regularizer for p≈25k, n≈3500. Expected 0.005–0.015. |
| **D.8** | + zero-fill missing parcels (no DK intersect) | Stops discarding signal at the alignment step. **Tests one of v14's implicit choices** — v14 attaches every parcel embedding to every electrode; D.8 emulates "don't discard signal" in DK. Expected 0.003–0.010. |
| **D.9** | + WM-contact rejection (drop electrodes flagged WM in `depth-wm.csv`) | Free if labels reliable. Volume conduction in WM is structurally mis-aligned for cross-subject. Expected 0.002–0.008. |
| **D.10** | composite: D.6 + D.7 + D.8 + D.9 + BNA BT-Tier-1 + soft support (i.e. D.3b stacked with all four engineering improvements) | Better-Linear submission candidate. Linear ceiling under best-effort feature engineering. |

**Open question for A4 (Zac/Nanlin) — adding to the blocking list:** is `depth-wm.csv` a derived label column (free filter for D.9) or a separate file requiring its own labeling pass? Determines whether D.9 is a one-line filter or a heavier scoping change.

#### Pairwise readouts (consume **before** Stage 1)

- **D.0** sets the protocol gate. Within ±0.005 of the leaderboard JSON per session-task; per-task mean within ±0.003. Failure → debug binary-search BT's `eval_population.py` flags before any other cell is interpreted (other cells share the split + label code).
- **D.0 → D.3b** is the headline: full-stack-ours linear vs full-stack-theirs linear.
  - **D.3b ≥ 0.539** → the v14 prior alone (atlas + HG prep) beats DK + STFT without a neural net. Stage 1+ is locking it in, not discovering it.
  - **D.3b < 0.539** → our prior is individually worse than theirs at the linear regime; understand why before Stage-1 compute. Possible causes: BNA bake error (A3 caught it; if not it's a real architectural problem), Tier-1 too narrow (D.4 will tell), HG vs STFT genuinely bad for these tasks (D.2 will tell).
- **D.0 → D.1*** isolates **atlas effect** (DK `combine_regions` vs BNA BT-Tier-1) holding their prep constant.
- **D.0 → D.2** isolates **prep effect** (Lap+STFT vs CAR+HG) holding their atlas constant.
- **D.1a → D.1b** and **D.3a → D.3b**: **soft vs hard support**, twice (under each prep). If soft ≈ hard in both, hard simplifies the architecture and we drop probabilistic support; if soft > hard, probabilistic support is empirically buying us something and we keep it.
- **D.3b → D.4**: does Tier-1 filtering matter? D.4 > D.3b → drop the filter; D.4 ≈ D.3b → filter harmless; D.4 < D.3b → filter is doing useful denoising.
- **D.5**: anti-control. Expected to be the worst cell (LH-only on bilateral whole-brain tasks). If D.5 ≈ D.3b, our Tier-1 selection is irrelevant and the parcel set itself doesn't matter — that would be a surprising and important finding worth investigating.
- **D.0 → D.6**: log-transform effect. Free regularization for any later cell; if positive, fold into D.10.
- **D.0 → D.7**: L1 vs L2. Tests whether the under-determined-regime regularizer matters at all here. If D.7 < D.0, surprising — investigate whether `combine_regions` already handled the dimensionality issue implicitly.
- **D.0 → D.8**: zero-fill vs intersect. **Specifically tests one of v14's implicit choices.** If D.8 ≈ D.0, intersection isn't actually discarding meaningful signal in DK. If D.8 > D.0 by ≥ 0.005, signal-discard is a real problem worth our architectural attention.
- **D.0 → D.9**: WM-rejection. Independent benefit from filtering non-cortical contacts before any spatial averaging.
- **D.6–D.9 → D.10**: composition test. Sub-linear sum = redundancy among the engineering steps. Super-linear = surprising synergy worth investigating before Stage 1.
- **D.3b → D.10**: pure feature-engineering lift on top of full-stack-ours. Closes the "is Linear-BNA-soft-zero-fill enough" question.
- **D.10 ≥ 0.55**: submission candidate. Submit it as the Better-Linear baseline alongside (or before) v14. Sets the bar v14 must beat and gives us a paired-submission decomposition story.
- **D.10 ∈ [0.539, 0.55]**: clears the leaderboard #1 by a couple thousandths but doesn't clear v14's ≥ 0.56 submission threshold; submit only if we want a leaderboard win regardless of the v14 program.
- **D.10 < 0.539**: feature-engineering ceiling sits at the Linear Lap+spec line; v14's gain has to come from learning, not engineering. Don't submit Better Linear.

#### What we are explicitly **not** sweeping in D

- Time-window (1 s vs 0.5 s vs 2 s) — benchmark fixes 1 s; ablating it confounds with everything.
- Pooling strategy beyond mean (median, top-k, max) — `combine_regions` mean and our soft-support sum are the two principled choices; others are arbitrary.
- Frequency-range alternatives (broad gamma 30–150, theta-alpha) — testing these at the linear regime doesn't inform Stage 1+.
- Per-task vs joint multi-task linear — benchmark eval is per-task by definition.
- Within-session linear — we're submitting Cross-Subject only.

If D.0–D.10 results force any of these open, they reopen as Stage-1 ablations.

#### Implementation

- **D.D1 — DCC Python env.** Install `insight-neuro/neuroprobe` as a local package in `/work/ht203/miniconda3/envs/speech/`. Add `h5py`, `scikit-learn`, missing deps. Verify `from neuroprobe.braintreebank_subject import BrainTreebankSubject` works and `BrainTreebankSubject(2, 4).get_electrode_data(...)` returns sensible shapes.
- **D.D2 — Hand-written sbatch arrays.** 13 cells × 10 sessions × 15 tasks = **~1950 jobs** total. Per-cell wrapper: `scripts/neuroprobe/stage0_linear_<cell>.sh` (e.g. `stage0_linear_d0.sh`, `stage0_linear_d1a.sh`, …, `stage0_linear_d10.sh`). D.6–D.9 reuse D.0's pipeline with one swapped step (log-transform / L1 solver / zero-fill at the `combine_regions` step / WM filter at the loader). D.10 stacks all swaps + the D.3b atlas/prep choice. **Why hand-written and not `scripts/ablation/submit.py`** — Block D runs BT's reference pipeline (D.0, D.1) or hybrid scripts swapping atlas/prep modules in place (D.2–D.10); not v14 train. CLAUDE.md retains hand-written sbatches for "non-standard array math"; this is one. Per-job walltime <30 min; total ~800 CPU core-hours, cluster-cheap.
- **D.D3 — Run + collect.** Submit per cell. Aggregate via rsync: `rsync -av ht203@dcc-login:/work/ht203/results/stage0_linear_<cell>/ docs/neuroprobe/stage0_results/linear_<cell>_<date>/` (target dir gitignored — large, reproducible). `scripts/neuroprobe/diff_vs_leaderboard.py` consumes the local mirror, writes per-cell `diff_table.csv` and a cross-cell `pairwise_readouts.csv` that emits the comparisons listed above.
- **D.D4 — Reference JSONs committed.** Reference D.0 JSONs from upstream `leaderboard/Linear_Laplacian_rereferencing_spectrogram_.../Cross-Subject/population_*.json` are committed to `docs/neuroprobe/reference_jsons/linear_lapspec_cross_subject/` (small, few KB each — checked in so the source-of-truth doesn't sit in `/tmp/neuroprobe_explore/...` which is ephemeral). License: upstream is Apache-2.0; quoting reference numbers is fair use.
- **D.D5 — Block D closure gate.** Block D passes (and Stage 0 closes on the D-axis) iff:
  - **D.0 numerical gate:** all 150 session-tasks within ±0.005 of reference; per-task mean within ±0.003.
  - **D.1a–D.10 completeness gate:** all cells run, all numbers collected, `pairwise_readouts.csv` emitted. No numerical gate on these — they inform Stage 1+ but don't fail-block. (D.0 failure does fail-block all of them, since they share split + label code.)
  - **Strategy gates (not mechanical):**
    - If D.0 passes but D.3b is catastrophically below 0.5, we re-evaluate the v14 thesis with the user before launching Stage 1.
    - If D.10 ≥ 0.55, we decide with the user whether to submit Better Linear as a paired baseline (default: yes — strengthens the v14 paper as the explicit ablation control).
    - If D.10 ≥ v14 Stage-1 cold-start by ≥ 0.005, the v14 thesis sharpens to "*learned* within-parcel attention beats *flat* within-parcel mean," not "atlas-anchoring beats DK." Update Stage-1/2 framing accordingly.

### Block E — Our-pipeline BT loader scaffold (~1 day, parallel with D)

Lands inside Stage 0 in parallel with D. Stage-0 gate: "loader runs end-to-end on sub_2/trial_4 without error" (E1+E3). Stage 1 starts day-of-D-pass.

- **E1 — `src/speech_decoding/neuroprobe/loader.py`.** BrainTreebank h5 → our HG z-scored tokens at **exact 200 Hz**. Pipeline: read channel data via `BrainTreebankSubject.get_electrode_data()` → CAR across Lite electrodes per session → Gaussian filterbank 70–150 Hz (8 bands) → Hilbert envelope → sum → **anti-alias FIR + `scipy.signal.resample_poly(up=125, down=1280)` to exact 200 Hz** (2048 × 125 / 1280 = 200; do **not** stride-decimate, which lands at 204.8 Hz and breaks parity with the PS pipeline output rate) → recording-level median/MAD z-score per channel. Output per 1-s event-locked window: `(N_e, 200)` float32 + `(N_e,)` active mask + `(N_e, |BT-Tier-1|)` BNA support from Block C cache. `N_e` after C2 default exclusion: subject-specific (sub_9 = 85, sub_10 = 113, others = 120 / 119 / 109). API mirrors `src/speech_decoding/v14/phoneme_dataset.py` so Stage-1 can swap loaders.
- **E2 — Unit tests under `tests/neuroprobe/test_loader.py`.** Shape; **exact** 200 Hz sample rate; z-score stats (mean ≈ 0, MAD-scaled σ ≈ 1); CAR correctness on synthetic 2-channel data (common mode cancels); filterbank passband sanity (power in 70–150 Hz band >> out-of-band).
- **E3 — Label contract replication.** Re-implement BT's 15-task label derivation from `datasets.py::get_label()` as `src/speech_decoding/neuroprobe/labels.py`. **Gate:** 100% match against `BrainTreebankSubjectTrialBenchmarkDataset` output on a 100-window subset of sub_2/trial_4.
- **E4 — h5 read parity.** Feed sub_2/trial_4 through our loader's raw-h5-read step (no CAR, no filterbank) and compare against `BrainTreebankSubject.get_electrode_data()` on the same channels and time window. Numerical outputs should match exactly. Confirms our h5 read path doesn't drop channels or shuffle time. (We do **not** compare against `--preprocess.type none` — that path skips re-referencing entirely; the original "preprocessing parity smoke" framing was wrong. CAR correctness is covered by E2 synthetic test.)

### Block F — Close-out (~0.5 day)

- **F1 — Update `docs/neuroprobe/plan.md`.** Move resolved open questions from §"Open questions to resolve in Stage 0" into the body as decided facts: coord space (fsaverage pial, no recons needed); BT-derived Tier-1 (A0 output, parcels list + cardinality); per-electrode token scoping (defer to Stage 1); preprocessing parity (E4 + 200 Hz exact resample landed in E1); known limitation (Cogan-lab volumetric convention not applicable to BT). Append the linear ablation matrix results table — **D.0 through D.10 numbers** + pairwise readouts. Decide and record the Better-Linear submission outcome: D.10 ≥ 0.55 → submit Better Linear paired with eventual v14; D.10 ∈ [0.539, 0.55] → optional standalone leaderboard win; D.10 < 0.539 → don't submit Better Linear.
- **F2 — Memory updates.**
  - New: `project_neuroprobe_stage0_closeout_<YYYY>_<MM>_<DD>.md` capturing A0–A3 numbers, **all 13 D-cell AUROCs** + pairwise readouts, Zac/Nanlin sign-off outcome on (1)–(4), Better-Linear submission decision.
  - New: `project_cogan_lab_bna_convention_canonical_<YYYY>_<MM>_<DD>.md` capturing Nanlin's answers to A4 question (5) — Stage-3 prep, separate file because it concerns the D-cohort program, not Neuroprobe.
  - Update `MEMORY.md` index with one-line pointers to both.
- **F3 — Commit scaffolding + paths + CLAUDE.md.** All code under `scripts/neuroprobe/*.py`, `src/speech_decoding/neuroprobe/*.py`, `tests/neuroprobe/*.py`. Add DCC paths to `configs/paths.yaml` (gitignored locally): `bt_root`, `bt_metadata_dir`, `bt_support_cache_dir`, `bt_results_dir`. Update `CLAUDE.md` §"Code Structure" to register `src/speech_decoding/neuroprobe/` as a new module (key files: `atlas_tier1_bt.py`, `support_cache_bt.py`, `loader.py`, `labels.py`); update §"Key Files" with new scripts dir, support cache dir, reference JSONs dir.

## How our BNA fsaverage bake is built (informational reference)

Provided so future readers can see what A3's gate is testing.

1. **Source.** `data/atlas/BNA_PM_4D.nii.gz` — Brainnetome's probabilistic atlas in MNI152 space. 4D NIfTI, 246 frames (one per parcel), each voxel value ∈ [0, 100] = probability of belonging to that parcel.
2. **Volume → fsaverage surface.** `mri_vol2surf --projfrac-avg 0 1 0.1`. This is a **column walk**: at every fsaverage vertex, sample the MNI volume at 11 depths from white-matter (projfrac=0) to pial (projfrac=1) and average. **Depth integration is baked into the surface representation** — every vertex's BNA support reflects the column above/below it, not just the pial-tangent slab. **For sEEG GM contacts that are roughly column-aligned with their snap target, this means BT's pial-snap + our column-walk read = "BNA distribution sampled at this electrode's column"** — a defensible approximation. For non-GM contacts (the 15 missing) the column-walk would be meaningless, but BT correctly omits those upstream.
3. **Cross-subject normalization.** `mri_surf2surf` to ensure values land on the canonical fsaverage mesh.
4. **Smoothing.** None on `fsaverage_bake_v2c`. The PSF is whatever `--projfrac-avg` integrated. (A prior bake used 3.5 mm 2D geodesic; we retired it post-cras-fix because it didn't help.)
5. **Verification (2026-04-16).** vs the BNA authors' shipped `BN_Atlas_freesurfer/fsaverage/*.annot`: 85–87% interior vertex argmax agreement, 90%+ Dice on Tier-1 speech-motor parcels. A3 hardens this to ≥90% argmax / Dice ≥0.85 *on the BT Lite vertex set specifically* and *on every BT-Tier-1 parcel specifically*, with no quiet sinks.

## What BT's preprocessing does (informational reference)

For Block D.0 reproduction context. From `insight-neuro/neuroprobe`:

- **Coord pipeline.** Native individual T1 FreeSurfer recon per subject → CT-MR co-registration + manual contact labeling → for GM / GM-WM-boundary contacts: project to native pial along cortical normal → patient `sphere.reg → fsaverage sphere.reg` → fsaverage pial NN snap. Output: `elec_coords_full.csv` with columns (ID, Z, X, Y, Hemisphere, Subject, Electrode, Region) where `Region` is Destrieux. `depth-wm.csv` keeps the pre-snap (L, I, P) plus `ShiftDist` (snap distance) and `ConfType`.
- **Signal pipeline (leaderboard `--preprocess.type laplacian-stft_abs`).** 2048 Hz raw voltages from the h5 → 1-second window from word onset (2048 samples) → Laplacian re-referencing (subtract local-neighborhood mean per electrode based on physical adjacency on the shank) → STFT magnitude (their default windows / hops; spectral feature extraction) → flatten for the linear classifier.
- **Cross-subject alignment.** `examples/eval_utils.py::combine_regions()` mean-pools electrodes within each Desikan-Killiany region, then takes the intersection of DK regions present in both train and test subjects. DK-Unknown electrodes are silently dropped. **This is the move D.1*** **upgrades — replacing DK region-averaging with BNA BT-Tier-1 weighted support pooling.**

## Dependency graph

```
A0 (BT Tier-1, local) ─→ C1 (support cache schema)
A1/A2/A3 (rigor, local) ─→ A4 (Zac/Nanlin sign-off, async) ─┬─→ C (support cache, DCC)
                                                            │
                                                            └─→ D (13-cell linear matrix, DCC) ─→ F (close-out)
                                                                       ↑
B (download, DCC) ──────────────────────────────────────────────────────┘
                                                                       ↓
                                                                    E (loader scaffold, parallel with D)
```

A0–A3 and B run in parallel from day 0. A4 is async wait on Zac/Nanlin (0–3 days). C and D both depend on A (via A4) + B. E parallel with D. F last.

Expected Stage-0 walltime: **6–11 days**. Breakdown: A0–A3 ~1 d local; A4 0–3 d async; B ~0.5 d (parallel A); C ~0.5 d; **D ~4 d** (13 cells incl. debug per cell; D.6–D.10 add ~1 d over the original 8-cell budget); E ~1 d (parallel D); F ~0.5 d.

## What Stage 0 explicitly does NOT do

- **No v14 cold-start training.** That's Stage 1. Block E only produces the loader scaffold.
- **No SSL pretraining or pretraining-corpus download beyond Lite.** Tier-1 pretrain sessions enter in Stage 2.
- **No architecture / SSL-objective commitment.** Experiments #1–#6 in `docs/neuroprobe/plan.md` stay empirically open.
- **No HG vs STFT lock-in.** Block D measures both at the linear regime; for Stage 1+ we default to HG (v14 thesis) but re-evaluate if D.2 shows STFT advantage is structural, not just at the linear regime.
- **No Cogan-lab volumetric BNA on BT.** See Known Limitation. Stage-3 D-cohort still uses it.
- **No 3D Gaussian alternate readout.** (Originally proposed as A3.5; dropped 2026-04-25.) BT pre-projects to pial; volumetric truth was discarded at the snap step; a 3D Gaussian on snapped coords would not recover it.
- **No time-window / pooling-strategy / frequency-range / per-task-vs-joint sweeps beyond the D matrix.** If D-results force them open, they reopen as Stage-1 ablations.

## Fail-closed escape hatches

- **A0 fails** (cohort coverage <99%): debug snap pipeline first; if real, document the uncovered tail and proceed (informational, not catastrophic).
- **A1 fails** (mesh mismatch): re-derive bake on correct subdivision. Stage-1 blocked.
- **A2 fails** (<95% Destrieux match): debug axis / hemi / vertex mapping in our bake's I/O before touching BT data further. Likely 0.5 d fix, not a re-bake. Implies the BT-fsaverage assumption is wrong end-to-end.
- **A3 fails** (must-pass — overall <90%, any BT-Tier-1 parcel <80% match or Dice <0.85): re-bake using authors' GCS classifier on fsaverage directly. **Stage 0 blocks until re-bake passes.** No soft-fail.
- **D.0 fails** (Linear repro outside tolerance): binary-search the preprocessing + split pipeline against BT's reference. Most likely culprits: undocumented flag in `eval_population.py`, env-version drift in `scikit-learn` LR solver, wrong split generator, wrong time-window crop. See `docs/references/neuroprobe_benchmark.md` for the option inventory. Other D cells share split + label code, so D.0 failure invalidates them all until fixed.
- **A4 (1)–(4) reject** (depth-contact projection untrusted, BNA bake disagreement deemed unacceptable, or `depth-wm.csv` unusable as a free filter — the last drops D.9 from must-run to deferred): escalate to BT authors (Wang lab, czlwang@mit.edu per paper) on (1)–(3); D.9 reverts to a scoping decision rather than a Stage-0 fail. Stage-0 close-out defers on (1)–(3) until response. (5) — Cogan-lab convention questions — is non-blocking; whatever Nanlin says lands in the separate Stage-3-prep memory note (F2).
