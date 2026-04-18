# Session: atlas/electrode kernel refactor + ablation (2026-04-17)

## Context

Two-part setup for this session:

1. Zac flagged 2026-04-16 that uECoG grids sit on the pial surface above
   gyral crowns and do **not** conform to sulci. Our FWHM 3.5 mm geodesic
   smoothing in the atlas bake was therefore doing double duty: modelling
   both atlas registration uncertainty (correct, geodesic) and electrode
   LFP catchment (incorrect geometry — should be 3D-Euclidean, crown-biased).
2. `fsaverage_atlas.py:18` still pointed at `fsaverage_bake` even though the
   adopted bake is `fsaverage_bake_fast2`. One-liner but worth rolling into
   the refactor.

The plan, agreed before code, was to ship three changes together so the
atlas kernel and the electrode kernel become cleanly separated:

- **A.** Re-bake with `--projfrac 0.9` (pial-biased), replacing `projfrac-avg 0 1 0.1`.
- **B.** Thin atlas smoothing to FWHM 1.5 (only registration noise).
- **C.** New electrode-side 3D-Euclidean Gaussian pool with crown bias — the
  physically correct model of what a uECoG contact senses.

All three were to write to new artifact paths so the existing v14-core work
(snap-based `data/atlas/support_cache/` + `fsaverage_bake_fast2`) stayed
untouched. Any swap of defaults would be a separate commit, post v14-core merge.

## What was built

| Artifact | Purpose |
|----------|---------|
| `scripts/bake_bna_on_fsaverage.py` +`--projfrac-mode {avg,single} --projfrac X` flags | A+B: re-bake with a single pial-biased depth and/or thinner smoothing. Default behaviour unchanged. |
| `src/speech_decoding/v14/electrode_pool.py` + `tests/v14/test_electrode_pool.py` (8 tests) | C: 3D-Euclidean Gaussian pool over pial vertices with sigmoid crown bias. Hemisphere-agnostic. |
| `scripts/build_euclidean_support_cache.py` | Per-patient pool cache builder, schema-identical to `build_support_cache.py`, separate output path. |
| `scripts/compare_electrode_support.py` | Per-patient snap-vs-pool argmax-change, Pearson-r, per-parcel retention. |

All reused the existing `support_cache.py` / `fsaverage_atlas.py` APIs.
No model, loader, or `fsaverage_atlas.py` file was modified.
v14 test suite: 127/127 green (119 pre-existing + 8 new).

## Four bakes run, one winner

All bakes produced locally with `data/freesurfer_subjects/ICBM152_fs` as the
nonlinear surface subject. Verification compares argmax to the authors'
published fsaverage annot (`BN_Atlas_freesurfer/fsaverage/label/*.BN_Atlas.annot`),
restricted to cortical BNA ids 1..210.

### Interior agreement vs authors' annot (smoothed)

| Bake | projfrac | FWHM | LH interior | RH interior |
|------|----------|------|-------------|-------------|
| `fast2` (current) | avg 0..1 | 3.5 | 85.3% | 87.1% |
| `v2a` | **0.9** | 3.5 | 80.9% | 85.2% |
| `v2b` | avg 0..1 | **1.5** | **84.8%** | **86.4%** |
| `v2` | 0.9 | 1.5 | 80.2% | 84.1% |

### Contribution breakdown on LH interior

- projfrac avg → 0.9 alone: **−4.4 pp** (v2a vs fast2)
- FWHM 3.5 → 1.5 alone: **−0.5 pp** (v2b vs fast2)
- Combined: −5.1 pp, almost perfectly additive

### Tier-1 Dice vs fast2 (LH, smoothed)

- **v2b**: every Tier-1 parcel within ±0.013 of fast2. A4hf 0.917 vs 0.920, A1/2/3ulhf 0.899 vs 0.906, A6cvl 0.902 vs 0.901. **Indistinguishable.**
- **v2**: A4hf drops 0.920 → 0.824, A1/2/3ulhf 0.906 → 0.800, A44v 0.872 → 0.503. Real regression.
- **v2a**: same pattern as v2 — projfrac 0.9 carries the damage regardless of FWHM.

### Snap-vs-pool on matched v2b atlas

| patient | n_elec | n_changed | frac | r mean |
|---------|--------|-----------|------|--------|
| S14 | 128 | 2  | 0.016 | 0.998 |
| S16 | 128 | 3  | 0.023 | 0.995 |
| S23 | 128 | 10 | 0.078 | 0.990 |
| S26 | 128 | 7  | 0.055 | 0.995 |
| S33 | 256 | 3  | 0.012 | 0.997 |
| S39 | 256 | 4  | 0.016 | 0.998 |
| S62 | 256 | 0  | 0.000 | 0.999 |

Per-parcel retention ≥88% on every Tier-1 parcel except A2 (62% of 21
electrodes — noise-dominated).

## Further ablation after Ben's pushback

Two questions surfaced after the initial v2b recommendation:

1. **The BNA PM is already probabilistic** — why smooth it further on the
   surface at all?
2. **fsaverage folds are fictional** (statistical composite of HCP cohort) —
   geodesic surface smoothing imports a fold prior that isn't real per-patient.

Both are correct. My "FWHM 1.5 covers registration residual" argument is
real but second-order, and routing that residual through geodesic smoothing
on fsaverage is the wrong coordinate system. Two more bakes:

- **v2c**: no smoothing anywhere (`--smooth-fwhm 0`). Tests whether the PM
  alone is enough.
- **v2d**: 3D isotropic Gaussian FWHM 1.5 mm applied to the PM volume
  **before** projection, then `--smooth-fwhm 0`. Tests whether the residual
  is best modelled in volume space instead of on fsaverage.

### Four-bake table (LH smoothed interior, smoothed Tier-1 Dice)

| Bake | Kernel location | LH interior | Max Tier-1 Dice loss vs fast2 |
|------|-----------------|-------------|-------------------------------|
| `fast2` (old) | surface FWHM 3.5 geodesic | 85.3% | baseline |
| `v2b`         | surface FWHM 1.5 geodesic | 84.8% | −0.013 (A44v) |
| `v2d`         | **volume FWHM 1.5 isotropic** | 84.5% | −0.025 (A44v, A45c) |
| `v2c`         | none                       | 84.3% | −0.031 (A44v, A45c) |

Surface smoothing scores marginally higher against the authors' annot
because the authors' annot was itself built with a surface classifier — both
import fsaverage-fold structure and that agrees with itself. This is not
evidence that surface smoothing is more correct; if anything, it's evidence
of a shared bias.

### Crown bias in the pool

The first pool implementation included a sigmoid crown bias
`w *= sigmoid(-curv(v) / 0.1)`. Three reasons this isn't rigorous:

- `tau = 0.1 mm⁻¹` is a magic number with no principled derivation.
- It uses fsaverage's per-vertex curvature as a proxy for sulcal-wall
  identification, which (like the folds themselves) is a statistical
  composite, not the patient's curvature.
- It's redundant with the 3D Gaussian when sulcal walls are 3D-far from the
  electrode, and only helps in the narrow case where the wall is both 3D-close
  and geodesically close. The proper fix for that case is line-of-sight
  raycasting (1–2 days of work with `trimesh`), not a curvature proxy.

Dropping crown bias on v2d: argmax change 1–11% per patient, r ≥ 0.99, Tier-1
retention ≥88% on every parcel except A2 (19 electrodes, noise-dominated).

## Pool validation — my hypothesis was partially wrong

After landing on v2d + pool, Ben asked whether the pool has real intellectual
weight. I ran a curvature stratification (`scripts/validate_electrode_pool.py`)
to test my prior claim that the pool corrects electrodes snapped to
sulcal walls.

Pooled across 1280 LH electrodes from 7 patients:

| |curv| bin at snap vertex | n | argmax change rate | frac curv > 0 |
|---|---|---|---|
| [0.00, 0.02) (flat) | 50 | 8.0% | 52% |
| [0.02, 0.05) | 92 | 0.0% | 36% |
| [0.05, 0.10) | 196 | 6.1% | 17% |
| [0.10, 0.20) | 429 | 4.9% | 5% |
| [0.20, 1.00) (deep crown/fundus) | 513 | **0.8%** | 0% |

The highest-|curv| bin has the LOWEST change rate. Almost no electrodes snap
to sulcal walls (positive curvature) in the first place — **Zac's observation
is already baked into the snap itself**. After sphere.reg registration,
uECoG contacts land on fsaverage gyral crown vertices, not sulcal walls.

Where the pool DOES change argmax (~3% of electrodes at σ=1.3 mm), it's
concentrated in low-|curv| transitions between gyri — i.e., near parcel
boundaries on flat stretches of cortex. That is a real but small effect.

### Sigma sweep (pooled, 1280 LH electrodes, vs v2c snap)

| σ (mm) | argmax change rate | mean L1 delta |
|---|---|---|
| 0.3 | 0.6% | 1.3 |
| 0.5 | 2.3% | 3.4 |
| 1.0 | 2.8% | 7.0 |
| 1.3 | 3.2% | 9.1 |
| 2.0 | 4.3% | 14.6 |
| 3.0 | 7.0% | 22.8 |
| 5.0 | 10.9% | 39.4 |

Smooth and monotonic — pool is a stable dial, not a threshold. But there is
no "physics kicks in at σ=X" inflection; σ is a free parameter with
literature-ballpark values 0.5–1.0 mm for uECoG high-gamma (Dubey & Ray 2019,
Kellis et al. 2016). My 1.3 mm was on the generous end.

## Final decision

**Snap + no smoothing anywhere. Canonical pipeline:**

```
BNA PM (cohort-probabilistic, as-is)
    │
    ├── mri_vol2surf --projfrac-avg 0 1 0.1  →  mri_surf2surf
    │   (no pre-smoothing, no post-smoothing anywhere)
    ▼
fsaverage per-vertex PM (data/atlas/fsaverage_bake_v2c/)
    │
    ├── per electrode: snap to nearest fsaverage pial vertex
    ▼
support[N_e, 15]  (data/atlas/support_cache_v2c_snap/)
```

One probabilistic object (the PM's cohort frequencies), one discrete readout
(single-vertex snap). No free parameters. No fsaverage-fold bias. No magic
numbers. If Phase-1 decoding reveals a gap, we revisit with the artifacts
preserved under `data/atlas/kernel_exploration_2026_04_17/`.

Tier-1 Dice cost of `v2c` vs `fast2`: up to −0.031 on any parcel (A44v
0.841 vs 0.872; A4hf 0.909 vs 0.920). LH interior agreement 84.3% vs 85.3%.
Small cost for a pipeline with no conflated kernels and zero free parameters.

The physical argument (atlas kernel should only represent registration noise;
electrode catchment belongs on the electrode side as 3D-Euclidean) is
satisfied by v2b + pool. The projfrac-0.9 change went one step past the
argument and into territory where the atlas sampling no longer matches what
the authors' GCS classifier does on the ribbon — hence the 9–11 pp Tier-1
Dice drop on A4hf / A1/2/3ulhf that v2a pinned on projfrac alone.

Why this is a meaningful change rather than a cosmetic one:

- FWHM 3.5 was a single kernel carrying two physical meanings. That was the
  load-bearing problem Zac surfaced. FWHM 1.5 + pool separates them.
- Pool + v2b atlas barely reshuffles argmax vs snap + v2b (0-8% per patient,
  r ≥ 0.99), but the weight on each Tier-1 column now respects 3D Euclidean
  geometry — exactly what a uECoG contact physically senses.
- Tier-1 Dice agreement with authors' annot is preserved (within ±0.013).

## What didn't ship

- `projfrac 0.9` / single-depth ribbon sampling. Agreement with authors'
  annot drops too much; no countervailing downstream evidence to justify it.
- Normal-vector orientation weighting (option E in the pre-session brainstorm).
  Deferred — v2b + pool already lands the physics-correctness win without it.

## Deferred (post-compaction menu)

Still on the table for when decoding numbers motivate another pass:

1. True line-of-sight mask via raycasting on the pial mesh (trimesh BVH).
2. Vertex-normal dot-product weighting.
3. HCP-individual → fsaverage PM (heroic, Phase 2+).
4. GCS posterior from the authors' classifier binary (2-3 days).

## Artifacts on disk

Canonical (top-level):

- `data/atlas/fsaverage_bake_v2c/` — canonical atlas (projfrac-avg 0..1 0.1, `--smooth-fwhm 0`)
- `data/atlas/support_cache_v2c_snap/` — canonical per-electrode Tier-1 support cache (single-vertex snap)
- `docs/qc/support_cache_v2c_snap_qc_report.md`
- `data/atlas/fsaverage_bake_fast2/` and `data/atlas/support_cache/` — still top-level because
  they are the current loader-active pair. They become historical once the coordinated swap
  lands post v14-core merge.

Explorations preserved under `data/atlas/kernel_exploration_2026_04_17/`:

- `bakes/` — `fsaverage_bake_v2`, `v2a`, `v2b`, `v2d` with a per-bake verdict in the archive README.
- `caches/` — pool caches at v2b/v2d, snap caches at v2, v2b, v2d.
- `volume_presmoothed_pm/BNA_PM_4D_vol_fwhm1.5.nii.gz`

Scripts kept at `scripts/` as re-runnable exploration tooling (not on the canonical path):

- `scripts/volume_presmooth_bna_pm.py`
- `scripts/build_euclidean_support_cache.py`
- `scripts/compare_electrode_support.py`
- `scripts/validate_electrode_pool.py`

Code kept under `src/speech_decoding/v14/` as a library (unused by the canonical pipeline, stays for exploration):

- `src/speech_decoding/v14/electrode_pool.py` + `tests/v14/test_electrode_pool.py`

## Coordinated follow-up (outside this session)

After the v14-core branch lands on main, two lines change:

- `src/speech_decoding/v14/fsaverage_atlas.py:18` → `fsaverage_bake_v2c`
- v14 loader support-cache path → `data/atlas/support_cache_v2c_snap/`

Plus CLAUDE.md atlas-paragraph updates:

- Replace "Atlas side is `data/atlas/fsaverage_bake_fast2/` (`mri_vol2surf --projfrac-avg 0 1 0.1`
  from `ICBM152_fs`, then `mri_surf2surf` to fsaverage, then 2D geodesic surface smoothing FWHM 3.5 mm)"
  with "Atlas side is `data/atlas/fsaverage_bake_v2c/` (`mri_vol2surf --projfrac-avg 0 1 0.1`
  from `ICBM152_fs`, then `mri_surf2surf` to fsaverage, no surface smoothing — the PM's own
  cohort-frequency probabilities represent atlas uncertainty)."
- Replace FWHM 3.5 mm references with "no additional smoothing" where the old value was cited.
- No mention of an electrode-side pool. Snap-to-pial is the production readout. Pool and volume
  pre-smooth explorations are archived under `data/atlas/kernel_exploration_2026_04_17/`.
