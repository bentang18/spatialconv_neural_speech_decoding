# Current Direction

Updated: 2026-04-13 (repo cleaned for implementation; first pass narrowed to uECoG-only v14 correctness; working principle locked)

## Working Principle

**v14 is slow, methodical, and precise. Everything before v14 was playing around.** Every logic step — raw voltages through phoneme decode — is **discussed, agreed, and understood before any code is written**. No handwavy implementation. No pre-committed numeric defaults. No legacy reuse.

If a legacy helper looks useful, re-derive it from scratch under `src/speech_decoding/v14/`. The pre-v14 tree is quarantined under `src/speech_decoding/archive/legacy/` and blocked by `tests/v14/test_no_legacy_imports.py`. No file or design is sacred.

Each blocker here and in `docs/implementation_tasks.md` is a **discussion item**, not an engineering ticket. Resolution means the logic, the contract, the units, and the trade-offs are agreed. Then code. See `CLAUDE.md` "Working Principle: Discuss Before Code".

**Phoneme / label-space rigor audit (2026-04-13):** a first-pass audit of the PS label space surfaced 9 discussion items, now blockers **#17–#25**. The most load-bearing is **#17** — the inherited `PS2ARPA` mapping (`ae → EH`, `u → UH`) looks phonetically wrong; it should probably be `ae → AE` (/æ/) and `u → UW` (/u/). Until #17–#25 are resolved, no phoneme-level metric is trustworthy and no v14 data loader should be written.

## Active Priority: Intracranial Field Potential Foundation Model (v14)

**Design doc**: `docs/neural_field_perceiver_v14.tex`
**Implementation scope**: `docs/implementation_start.md`
**Implementation task list**: `docs/implementation_tasks.md`

**Two-problem decomposition** (atlas-guided calibration + shared dynamics):
- **Problem 1 — Spatial Calibration**: Map raw electrodes into atlas-grounded parcel/subparcel tokens via Brainnetome volumetric membership. Long-term this may include small learned per-patient corrections, but Phase 1 uses a fixed atlas mapping once the ACPC→MNI pipeline is verified.
- **Problem 2 — Shared Dynamics**: Map atlas-grounded parcel/subparcel tokens → phoneme sequence via a small relational-temporal model + AR decoder. Same `(N_tok × d × T)` token interface for every patient.

**Active Phase-1 architecture contract**:
```
── Fixed Spatial Interface ──
corrected electrode coordinates -> Brainnetome PM membership -> parcel support

── Shared Processing ──
(1) Shared temporal tokenizer                              -> (N_i × d × T)
(2) Canonical parcel-frame local point encoding
    + within-parcel Perceiver summarizer                   -> (N_tok × d × T)
(3) [Inter-region graph attention
     -> Temporal self-attn] × B                            -> (N_tok × d × T)
(4) 3 AR-conditioned decode queries attend over N_tok·T    -> phoneme sequence
```

Phase 1 assumptions:
- fixed atlas mapping first; no learned `Δ/ω`, `δ_l`, or `τ_l` yet
- no extra gain / impedance correction beyond the existing HGA preprocessing
- no Fourier PE
- no global electrode→region cross-attention
- fixed `N_tok` layout with `token_mask` and `token_support`
- supervised-only `uECoG` implementation before SSL or modality expansion

Long-term target: the same token interface, but with optional learned per-patient calibration once the fixed-atlas path is correct and stable.

**Paper direction**: Atlas-grounded common-space decoding for intracranial field potentials. The scientific claim is that electrodes are patient-specific observations, while atlas parcel/subparcel tokens are the shared representation. Phase 1 is only the fixed-atlas supervised correctness pass.

## First Implementation Scope (2026-04-13)

Before expanding to `sEEG`, external chronic ECoG, or SSL, the first executable target is:

- **implement and verify `v14` on the existing intra-op `uECoG` data only**

This means:

- core patients first: `S14`, `S26`, `S33`, `S62`
- architecture correctness before broad ablations
- end-to-end overfit and token-construction sanity before scaling

Deferred until the `uECoG` path is verified:

- `sEEG`
- external datasets
- functional pooling variants

The immediate next step after supervised `v14-core` correctness should be:

- **SSL / JEPA on the full continuous `uECoG` corpus**

Not:

- SSL only on response-locked epochs

Response-locked windows are the right unit for the first supervised decoding check. For SSL, the better default is the full continuous `uECoG` data because it preserves more temporal context and uses more of the limited intra-op recording time.

Code boundary for the new implementation:

- `src/speech_decoding/v14/`

## Top Blocker (2026-04-13)

- **ACPC → MNI transform pipeline is not trusted**

Zac flagged the current handling as wrong. Top blocker — every spatial step in v14 depends on it: Brainnetome membership, parcel support, parcel-frame coordinates, learned calibration. Do not treat coordinates as settled until the transform is re-verified.

- **Unsupported-vs-weak parcel support threshold is not yet locked**

The first implementation already assumes a fixed atlas-token interface with `token_mask` and `token_support`, but the rule separating:

- unsupported parcels that should be masked out entirely
- from weakly observed parcels that should stay active with low support

is still unresolved. This matters because it changes the effective active-token set per patient and the semantics of low-coverage training cases.

- **Default parcel split map is not yet fully locked**

The current design assumes a 21-token default built from selective 2-token parcel splits, but that split set should still be treated as an explicit implementation decision. It affects the shared interface, the local summarizer, and the expansion of Brainnetome connectivity from parcels to token-level graph bias. The current likely default is to split `A6cvl`, `A4hf`, `A1/2/3ulhf`, `A2`, and `A1/2/3tonIa`, with `A4tl` left as the first extension candidate.

- **Exact parcel support statistic is not yet locked**

The model now depends on parcel support explicitly, but the exact support formula still needs to be frozen. This matters because it drives `token_mask`, `token_support`, and low-coverage behavior.

- **Temporal-layer output contract is not yet locked**

The temporal layer choice is not the whole issue; the exact output contract into the local summarizer also needs to be frozen. Token rate, receptive field, and time-axis semantics all still need to be written down explicitly.

- **How `token_support` enters the model is not yet locked**

The design assumes `token_support` exists, but not yet whether it is only diagnostic or also enters the backbone/decoder as an explicit signal. That remains an implementation blocker.

- **Token-level connectivity expansion is not yet fully specified**

The design assumes Brainnetome SC/FC initializes graph attention, but the exact token-level expansion rule is still open: SC vs FC, normalization, sibling-token bias, and random/no-bias fallback.

- **Supervised training contract is not yet fully locked**

Phase 1 is supervised-only, but the exact training contract still needs to be frozen: supervised input window, target semantics, decoder training behavior, and how train-time and eval-time decoding relate.

- **Parcel-frame construction is not yet fully locked**

The local summarizer depends on canonical parcel coordinates, but the exact parcel-frame construction rule still needs to be written down explicitly: centroids, axes, scales, and whether these are computed offline and cached.

- **Channel inclusion policy is still not fully decided**

It remains unresolved whether Phase 1 should use all non-artifact channels, sig-only channels, or another filtering rule. That changes support statistics and active-token counts, so it should be treated as a true implementation blocker rather than a minor preprocessing detail.

- **The old grid/loader path is not trustworthy enough for `v14-core`**

The existing baseline data path was built for the older grid-based Conv2d model, not for the current atlas-first implementation. Two specific risks should be treated as blockers:

- `load_patient_data()` / `load_per_position_data()` currently zero artifact channels in place rather than defining an explicit active-channel interface.
- `src/speech_decoding/data/grid.py` currently reconstructs the physical grid by heuristically quantizing normalized TSV coordinates, even though there is an authoritative channel-to-physical-electrode mapping file available.

Do not blindly reuse the old loader/grid logic for `v14-core`.

- **No stale Brainnetome proxy fallback should be allowed**

Phase 1 should use the real PM volume at `/Users/bentang/Documents/Code/speech/data/atlas/BNA_PM_4D.nii.gz` as the active membership source. The MPM label map remains useful for ROI indexing and sanity checks only. The implementation should not silently fall back to the old smoothed-MPM pseudo-probability construction.

## Key Design Choices (Phase-1 active assumptions only)

- **Use the real Brainnetome PM volume** — Active membership source is `/Users/bentang/Documents/Code/speech/data/atlas/BNA_PM_4D.nii.gz`. Keep the MPM label map only for ROI indexing and sanity checks. Do not silently fall back to the old smoothed-MPM proxy.
- **Replace the old TSV-derived grid heuristic before `v14-core`** — The old grid-based baseline inferred physical array layout from normalized electrode TSV coordinates. That is not rigorous enough for the new implementation phase. Use the authoritative channel-mapping files instead of relying on heuristic grid reconstruction.
- **Within-parcel Perceiver summarizer is the default spatial mechanism** — Electrode→parcel mapping remains anatomy-guided aggregation, but the default parcel representation is now a small local point encoder plus fixed latent queries, not mean-only or mean+gradient pooling.
- **Mean+gradient stays as the main linear ablation** — It is still useful as a lower-capacity baseline, but not the default shared spatial interface.
- **Temporal front-end remains an explicit blocker** — The temporal layer and its output contract into the parcel summarizer are not frozen yet. Do not treat the current tokenizer sketch as settled.
- **Coverage and support are explicit** — Unsupported parcels are masked, not hallucinated. Weakly supported parcels remain active with low support. The exact support statistic, threshold rule, and use of `token_support` are still blockers.
- **Inter-region attention happens in token space** — After parcel summarization, the model operates on atlas/subparcel tokens, not on raw electrodes. Brainnetome SC/FC bias initialization is still intended, but the exact token-level expansion rule is not frozen yet.
- **No learned per-patient calibration in Phase 1** — `Δ/ω`, `δ_l`, `τ_l`, and any extra gain/impedance correction are deferred until after the fixed-atlas supervised path is correct.
- **Phase 1 is supervised-only on intra-op `uECoG`** — The first milestone is end-to-end correctness on the existing `uECoG` data. SSL, `sEEG`, and external datasets come only after that.
- **Core patients first** — `S14`, `S26`, `S33`, `S62`. Keep `S32` and `S57` excluded for the first pass. Always exclude artifact channels. Channel inclusion beyond that is still a blocker.

## Immediate Next Steps

1. ~~**v14 design doc rewrite**~~ — DONE (2026-04-10). Full rewrite: two-problem decomposition, parcellation pooling, cross-attention decoder, updated ablations.
2. ~~**Update presentation doc**~~ — DONE (2026-04-10). `physics_informed_architecture.tex` rewritten for v14 (4 pages).
3. **Fix and verify ACPC → MNI transform pipeline (TOP BLOCKER)** — current handling is not trusted; re-verify the coordinate chain with Zac before serious spatial-model implementation.
4. **Replace the old loader/grid assumptions** — before `v14-core`, freeze the active-channel semantics and replace the TSV-derived grid heuristic with the authoritative channel-mapping path.
5. **uECoG-only v14 implementation** — once coordinates and channel/electrode mapping are trustworthy, build the clean end-to-end path under `src/speech_decoding/v14/` and verify correctness on core patients before any modality expansion.
6. **Full-corpus uECoG SSL after v14-core** — once supervised `v14-core` is stable, pretrain on the full continuous `uECoG` corpus rather than only response-locked epochs.
7. **Nonlinear MNI normalization** — Only revisit this after the correct baseline ACPC→MNI path is confirmed. Do not optimize around the wrong transform.
8. **Verify and wire the real probabilistic maps cleanly** — the local PM file now lives at `/Users/bentang/Documents/Code/speech/data/atlas/BNA_PM_4D.nii.gz`; verify orientation/indexing and use it as the active spatial-membership source.
9. **Request external chronic ECoG data** — Greg → Flinker (48pts, NYU) + Chang (~15-25pts, UCSF). High leverage, but not on the critical path for the first local implementation.
10. **HGA extraction pipeline** — 456 min raw EDF, 29 patients. Needed before SSL / broader scaling.
11. **Linear/local ablations** — Mean+gradient pooling, selective splitting, and parcel-frame 2D conv compared against the default local summarizer after the core implementation is stable.
12. **Large-scale generic iEEG pretraining watchlist** — SWEC (`~10,000 h`, MVPFormer release) belongs in the acquisition roadmap as a staged generic pretraining source, but not in the first implementation phase.

## Data Readiness

**Electrode coordinates**:
- ACPC electrode files exist for 11/11 patients.
- ACPC-side channel/electrode bookkeeping is mostly checked.
- The ACPC→MNI transform path is still the top blocker, so do not treat cross-patient spatial alignment as ready yet.

**Brainnetome atlas**:
- real PM volume is present at `/Users/bentang/Documents/Code/speech/data/atlas/BNA_PM_4D.nii.gz`
- MPM label map remains useful for ROI indexing and sanity checks
- remaining work is implementation-path verification of PM orientation/indexing, not atlas download

**Artifact exclusion**:
- artifact-channel exclusion is active and should remain on for Phase 1

**Raw continuous data**:
- 456 min across 29 unique patients are available for later SSL
- HGA extraction is still needed before that Phase 1.5 step

**Patient selection**:
- core: `S14`, `S26`, `S33`, `S62`
- excluded for the first pass: `S32`, `S57`
- extended set comes only after the core path is correct

## Per-Patient Baseline (v14 must beat this)

**PER 0.734 ± 0.007** (S14, grouped-by-token CV, 3-seed). Per-phoneme MFA flat head + full recipe.
**Population: 0.825 mean** across 11 patients.

## Ablation Posture

Do not treat the full ablation program as active until `v14-core` is implemented correctly.

Immediate comparison targets after the fixed-atlas supervised path is stable:
- local Perceiver summarizer vs mean+gradient
- exact parcel split map
- temporal front-end choice
- token-level SC/FC bias init vs no-init

Broader ablation planning still lives in the design doc, but it is not on the critical path for the first implementation milestone.

## Practical Rules

- Active design = `neural_field_perceiver_v14.tex`.
- Shared within-parcel Perceiver-style summarization is the default spatial interface. Mean+gradient, selective splitting, and cross-attention are ablations.
- Core patients: S14, S26, S33, S62. Extended: S16, S22, S23, S39, S58. Excluded: S32, S57.
- Always `exclude_artifacts=True`. Always grouped-by-token CV.
- All training on DCC. See `docs/dcc_setup.md`.
- Treat the supervised training contract as blocked until the exact loss/decoder/eval setup is frozen.
- If a doc references: VE cross-attention as default spatial mechanism, distance bias, Fourier PE, Wendland kernel, "multi-view reconstruction" framing, "camera view" analogy, ~175K shared params, 182/pt, "dual spatial encoding", Q/K/V for electrode→VE mapping, spherical node decay, "distance determines WHERE, Q/K/V determines WHAT", v12 as active, A3/A_no_dist/A_dist_only ablations, or any of the v12 stale markers listed previously — it's stale.
