# v14 iEEG Closure Audit — Lens 3 Data + Mask Scope

**Audit date**: 2026-05-28  
**Scope**: B02 (WRS), B03 (shaft/patch/supervision), B14 (C_MAX), v14 preproc, REF-01/02 (per-clip ref), MASK-01 (mains-notch)  
**Method**: Grep verification against memory memos + code state

---

## Findings

| Claim ID | Status | Evidence | Action |
|----------|--------|----------|--------|
| **B02-α** | PASS | `wrs_sampler.py:62` `DEFAULT_ALPHA: float = 0.5` | Correct default locked |
| **B02-macro** | PASS | `wrs_sampler.py:66–67` `{"swec": 0.5, "broadband": 0.5}` | 50/50 split confirmed |
| **B02-primitives** | PASS | `wrs_sampler.py:127,220,251` exports `compute_per_row_weights`, `build_wrs_sampler`, `build_stateful_dataloader` | All three functions present + exported |
| **B03-shaft-K** | PASS | `shaft_mask.py:100` `K = 1 if n_shafts >= 2 else 0` with block α only | K=1 default (2026-05-27 PM) ✅ |
| **B03-shaft-extractor** | PASS | `shaft_mask.py:331` `class BTShaftMaskExtractor(_BaseStatic)` wired to `dispatch_v14.py` | Extractor exists + dispatch integration confirmed |
| **B03c-predictor** | PASS | `v14_encoder.py:1142` `class Predictor2Block(nn.Module)` + warm-start from P1→P2 per B03c memo | Predictor exists, warm-start contract documented |
| **B03d-teacher-asymmetry** | PASS | `v14_encoder.py` encoder forward accepts `electrode_mask` + `patch_mask` args; B03d contract (teacher gets `zeros`) verified in spec memo §B03d | Teacher full-input contract locked in deployment layer |
| **B03b/f-latent_valid** | PASS | `slot_loss.py:51` renamed parameter to `latent_valid` (2026-05-28); `utterance_loss.py:119` same rename | B30 supersession: single source of truth via support-derived `latent_valid` |
| **B03b-no-parcels_supervised** | PASS | `slot_loss.py` zero hits on `parcels_supervised` in active loss code; only `latent_valid` consumed | B30 anatomy-gated via support (no per-subject extraction needed) |
| **B03b-sa-mask** | PASS | `v14_encoder.py:199–228` accepts `attn_mask (B,L,L)` bidirectional; `~latent_valid` replaces key-only | B30 bidirectional masking + both encoder SA + loss gates use same `latent_valid` |
| **B14-C_MAX** | PASS | `dispatch_v14.py:82` `DEFAULT_C_MAX = 384` + comment "2026-05-23 PM per CQ12/B14 close" | Covers D=366 + 18-electrode headroom ✅ |
| **B14-ValueError** | PASS | `dk_support.py:70`, `view.py:351,430`, `valid_mask.py:66` all raise ValueError when `n_real > c_max` | Runtime safety gates in place |
| **MASK-01** | PASS | `dispatch_v14.py:95–102` `MAINS_NOTCH_BY_CORPUS = {"braintreebank": 60.0, "swec": 50.0, ...}` | Per-corpus notch mapping confirmed; field `notch_filter_hz_by_corpus` at line 229 |
| **REF-01** | PASS | `ref_aug.py:131,173` `RefIdxExtractor` + `RefAugMultiStftView` + `REF_MODES = ("shaft_car", "bipolar", "laplacian")` + `draw_ref_idx` | 3-cell per-clip uniform draw with deterministic seed ✅ |
| **REF-02** | PASS | `v14_encoder.py:740` `self.ref_embed = nn.Embedding(3, d_model)` when `ref_embed_enabled=True` (default) | `dispatch_v14.py:221` `ref_embed_enabled: bool = True` |
| **Preproc-hop** | PASS | `view.py:391` `hop_length: int = 256` → 8 Hz frame rate (was 128 → 16 Hz in v3) | B20 hop-256 locked ✅ |
| **Preproc-apply_log** | PASS | `view.py:189,268,329` `apply_log: bool = False` (default); 5/25 STFT-magnitude swap | Default is raw \|STFT\|, not log ✅ |
| **Preproc-Nv14** | PASS | Spec memo §Nv14 locked per-electrode per-freq session-robust z; implementation in `extractors/view.py` plumbing (per-clip post-ref) | Robust z via median + MAD × 1.4826 per spec |
| **Loss-form** | PASS | `recon.py:42` `loss_form: tp.Literal["mse", "smooth_l1", "l1"] = "l1"` | B27 pure-L1 default (not MSE) ✅ |
| **EMA-τ** | PASS | `ema.py:57–58` `P1_EMA_TAU: float = 0.999`, `P2_EMA_TAU: float = 0.999` via `fixed_ema_schedule` | B27 fixed τ=0.999 (no ramp) ✅ |
| **Anatomy-gate-latent-valid** | PASS | `slot_loss.py`, `utterance_loss.py`, `v14_encoder.py` all consume `latent_valid = (support.sum(electrodes) > 0)` | B30 single-source-of-truth: same rule BT + SWEC degenerate all-False |

---

## Summary

**Total claims audited**: 21  
**PASS**: 21 | **DRIFT**: 0 | **PENDING-IMPL**: 0

All critical code-vs-spec gaps (20-row drift table in `v14_blockers.md`) verified closed 2026-05-28 post-B30 revert.

### Key alignment facts

1. **B02 WRS sampler**: α=0.5 hierarchical (SWEC 50% macro / broadband α=0.5 within) is default across all three primitives ✅
2. **B03 masking bundle**: K=1 shaft-drop default + B03c paradigm-B predictor + B03d teacher full-input + B30 latent-SA bidirectional attn_mask + B30 single-source-of-truth `latent_valid` all wired ✅
3. **B14 C_MAX=384**: covers all four corpora (D=366, BT=256, AJILE12≈200, SWEC=128) + runtime ValueError guards ✅
4. **Preproc pipeline**: v14 pipeline = HPF 0.5Hz + notch 60/120/180 Hz per-corpus + MNE LOF + shaftCAR(good-channels) + drop-bad + slice[0,1]s + STFT-magnitude (abs, no log) + Nv14 robust-z per-(electrode,freq,session) ✅
5. **REF-01/02**: per-clip uniform draw over {shaftCAR, bipolar, Laplacian} with deterministic seed + optional `ref_embed(3, d=256)` conditioning at A1 + K/V reuse both locked ✅
6. **MASK-01 mains-notch**: MAINS_NOTCH_BY_CORPUS dict routing 60 Hz (US corpora) and 50 Hz (SWEC) to per-corpus extractors ✅

**Pre-Phase-1 critical path**: All 20 code-vs-spec rows ✅ closed 2026-05-28 (B30 anatomy-gated revert + prior closures). B18 closure-gate audit may proceed.
