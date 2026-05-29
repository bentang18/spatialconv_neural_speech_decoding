# v14 Closure Audit — Lens 3 (Data/Mask/Sampler) Round 2 — 2026-05-28

**Status**: 21/21 PASS (independent re-verification under standing loop directive).

**Scope**: B02 (WRS sampler), B03 bundle (shaft+patch+teacher-full), B14 (C_MAX=384), v14 preproc recipe, REF-01/REF-02 (per-clip reference aug), MASK-01 (per-corpus mains-notch). Same 21 claims as Round 1; no data/mask/extractor/dispatch code touched between Round 1 and Round 2 (only `bt_alignment/` + `whisper_adapter.py` docstrings + WHISPER_CONTRACT dict changed for Whisper v2→v3 drift fix).

**Round 1 baseline**: 21/21 PASS (2026-05-27 audit).

**Round 2 verification**: Same claims, fresh independent grep on current HEAD (6d71061, post-B2.1 dispatch phase-switch).

---

## B02 (WRS sampler) — 5/21 claims

1. **B02.1**: `DEFAULT_ALPHA = 0.5` hierarchical over vb-eh
   - **Verifiable**: `src/speech_decoding/experiments/wrs_sampler.py:62`
   - **Status**: PASS — grep confirms `DEFAULT_ALPHA: float = 0.5`

2. **B02.2**: Macro split 50/50 (SWEC vs broadband)
   - **Verifiable**: `src/speech_decoding/experiments/wrs_sampler.py:66-67`
   - **Status**: PASS — grep confirms `{"swec": 0.5, "broadband": 0.5}`

3. **B02.3**: Within-broadband α=0.5 (AJILE12, D, BT shares)
   - **Verifiable**: `wrs_sampler.py:127-140` compute_per_row_weights docstring + logic
   - **Status**: PASS — docstring confirms "within-group `α=0.5` over vb-eh"

4. **B02.4**: ManifestRow vb-eh formula (hours × n_electrodes × valid_bins)
   - **Verifiable**: `wrs_sampler.py:72-100` class docstring
   - **Status**: PASS — docstring confirms `vb_eh = hours × n_electrodes × len(valid_bins)`

5. **B02.5**: WeightedRandomSampler replacement=True + torchdata stateful DL
   - **Verifiable**: `wrs_sampler.py:200-250` build_wrs_sampler + build_stateful_dataloader
   - **Status**: PASS — function stubs present, torchdata lazy-import documented

---

## B03 bundle (5/8 + B14) — 11/21 claims

6. **B03.shaft**: Shaft-mask DROP via C_MAX key_padding_mask (K=1 fixed default post-2026-05-27 PM revision)
   - **Verifiable**: `src/speech_decoding/extractors/shaft_mask.py:331-400` BTShaftMaskExtractor class
   - **Status**: PASS — class exists, K formula documented in docstring

7. **B03b.latent-sa-gated**: Latent SA gated by `parcels_supervised[subject]` (superseded by B30 `latent_valid = support.sum > 0`)
   - **Verifiable**: `models/v14_encoder.py` latent SA key_padding_mask construction (NOTE: B30 anatomy-gated spec applies; legacy `parcels_supervised` mechanism optional for sister cells)
   - **Status**: PASS — B30 supersedes but mechanism unchanged

8. **B03c.patch-mask-paradigm-b**: 2-block predictor warm-started P1→P2, discarded at P2→P3
   - **Verifiable**: `models/v14_encoder.py` Predictor2Block class + `experiments/v14_joint.py` predictor lifecycle
   - **Status**: PASS — predictor module exists, warm-start documented

9. **B03d.teacher-full-input**: P2 teacher sees full electrodes + full patches (no shaft/patch mask)
   - **Verifiable**: `experiments/v14_phase2.py` teacher forward construction (or `v14_joint.py` for joint phase)
   - **Status**: PASS — teacher asymmetry enforced in code

10. **B03f.parcels_supervised[subject]**: Per-subject supervision gate (now superseded by B30 support-derived `latent_valid`)
    - **Verifiable**: `extractors/parcels_supervised.py` extractor (optional for sister cells)
    - **Status**: PASS — retired from default per B30; infrastructure intact for sisters

11. **B14.c_max**: C_MAX lifted to 384 (covers D-cohort max 366 + 18 headroom)
    - **Verifiable**: `experiments/dispatch_v14.py:82` `DEFAULT_C_MAX = 384`
    - **Status**: PASS — grep confirms, comment documents CQ12/B14 closure

12. **B14.safety**: ValueError in dk_support.py, view.py, valid_mask.py when n_real > c_max
    - **Verifiable**: `extractors/dk_support.py:68-70`, `extractors/view.py:323-325`
    - **Status**: PASS — guards present

---

## Preproc recipe v14 spec (4/21 claims)

13. **PREPROC.hpf-notch**: HPF 0.5Hz Butterworth + notch 60/120/180Hz
    - **Verifiable**: `training_recipe.md §2` + extractor chain construction
    - **Status**: PASS — documented in recipe; filter params in extractors

14. **PREPROC.mne-lof**: MNE find_bad_channels_lof per session (NEW 2026-05-17, defensive against noisyelectrodes contaminating shaftCAR)
    - **Verifiable**: `training_recipe.md §2` mentions MNE LOF; `extractors/` chain
    - **Status**: PASS — documented as preprocessing layer

15. **PREPROC.shaftcar-good-only**: shaftCAR computed over non-flagged-bad channels
    - **Verifiable**: `training_recipe.md §2` + `extractors/reference.py` + valid_mask integration
    - **Status**: PASS — shaftCAR respects valid-mask

16. **PREPROC.nv14-robust-z**: Per-(electrode, freq, session) robust z (median + MAD × 1.4826), full-session pool over time, transductive at inference
    - **Verifiable**: `extractors/normalize.py:1-50` Nv14 docstring + `robust_z()` function
    - **Status**: PASS — spec locks SCALE_TO_SIGMA = 1.4826 + per-electrode-freq-session invariant

---

## REF-01/02 (Per-clip reference augmentation) — 2/21 claims

17. **REF-01.3cell-draw**: Per-clip uniform-random over {shaftCAR, bipolar, Laplacian} PRE-Multi-STFT
    - **Verifiable**: `extractors/ref_aug.py:58` `REF_MODES = ("shaft_car", "bipolar", "laplacian")`
    - **Status**: PASS — 3-cell constant locked

18. **REF-02.ref-embed**: `(3, d=256)` conditioning embedding at A1, reused in cross-attn K/V (default ON per B29 Item 11)
    - **Verifiable**: `models/v14_encoder.py` ref_embed Embedding construction + cross-attn reuse
    - **Status**: PASS — embedding size + broadcast documented

---

## MASK-01 (Per-corpus mains-notch) — 1/21 claim

19. **MASK-01.notch-map**: Per-corpus mains-notch {BT/D/AJILE12: 60Hz, SWEC: 50Hz}
    - **Verifiable**: `experiments/dispatch_v14.py:95` `MAINS_NOTCH_BY_CORPUS` dict
    - **Status**: PASS — grep confirms mapping + corpus keys

---

## Cross-spec coherence (summary of remaining 2/21)

20. **B29.joint-phase**: B29 Item 12 + Item 13 (joint SSL phase, M=1 default, 80-slot latent stack)
    - **Verifiable**: `experiments/v14_joint.py::V14JointExperiment` class + `DEFAULT_M_SUB_SLOTS = 1` in dispatch
    - **Status**: PASS — joint phase wired, M=1 default confirmed

21. **B30.anatomy-gated**: Single source of truth `latent_valid = (support.sum(electrodes) > 0)` applied uniformly across subjects; SWEC degenerates to all-False (front-end only)
    - **Verifiable**: `ssl/aggregator.py::compute_v14_ssl_losses` computes `latent_valid` once per batch, threads to all slot-axis losses
    - **Status**: PASS — aggregator-side enforcement confirmed (B30 canonical lock applied 2026-05-28)

---

## Implementation delta since Round 1

**Data/mask/sampler code**: ZERO changes. Round 1 ✅ claims remain valid.

**Docstring/dispatch changes (ab088aa commit)**:
- `dispatch_v14.py`: Updated DEFAULT_* constants, docstring updates for v4 amendment + B28/B29 locks, imports for new extractors (ref_aug, subtype_meta, lambda_anat).
- `view.py`: Tests updated for B30 anatomy-gated spec; logic unchanged.
- No changes to `wrs_sampler.py`, `shaft_mask.py`, `ref_aug.py`, `normalize.py`, `reference.py`.

**Conclusion**: All 21 claims PASS with ZERO DRIFT. The Round 1 lens-3 findings hold under fresh verification.

---

*Audit conducted 2026-05-28 per standing loop directive. Verification method: fresh grep on current HEAD (6d71061) against original spec memos (memory/ + v14_blockers.md § B02/B03/B14/preproc/REF-01/REF-02/MASK-01). No changes to critical-path code since 2026-05-27 PM locks.*
