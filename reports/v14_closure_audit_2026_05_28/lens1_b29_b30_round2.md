# v14 Closure-Gate Audit Round 2 — Lens 1 (B29/B30 Lock Alignment)

**Report Date:** 2026-05-28  
**Scope:** B29 (joint-default amendment 5/27 PM-late) + B30 (anatomy-gated symmetric 5/28) spec claims vs code  
**Methodology:** Grep trigger words across `src/speech_decoding/` + Read precise file locations  
**Code Touchpoints Changed Since Round 1:** Only docstrings + `WHISPER_CONTRACT` dict in `whisper_adapter.py` and BT alignment module (per user statement — verified: no B29/B30 trigger words touched in actual logic)

---

## Round 2 Audit Results: 11/11 PASS

| # | Claim | Evidence | Status |
|---|-------|----------|--------|
| 1 | **Joint-default phase** (single SSL phase, not staged P1+P2) | `src/speech_decoding/ssl/aggregator.py:1` "B28/B29 joint default::" + `src/speech_decoding/experiments/v14_joint.py:*` "joint-by-default surface" | **PASS** |
| 2 | **M=1 lock** (80 slots = 80 parcels × 1, dropped M=4 sub-slots) | `src/speech_decoding/models/v14_encoder.py:508` `m_sub_slots: int = 1` + comment "Item 13 lock 2026-05-27 PM-late: default M=1 (was M=4)" | **PASS** |
| 3 | **Subtype_embed wired** (binary `{sEEG-depth, ECoG}`, additive + K/V reuse config) | `src/speech_decoding/models/v14_encoder.py:549-550` `subtype_embed_enabled: bool = False` (5/28 flip to OFF per DIVER-1) + `self.subtype_embed = nn.Embedding(subtype_vocab, d_model)` if enabled; dispatch wiring in `v14_dispatch_wired.py` confirms both input-only and K/V-reuse sisters testable | **PASS** |
| 4 | **Ref_embed wired** (3-entry `{shaftCAR, bipolar, Laplacian}`, additive + K/V reuse, default ON) | `src/speech_decoding/models/v14_encoder.py:551-552` `ref_embed_enabled: bool = True`, `ref_embed_reuse_kv: bool = True` + `self.ref_embed = nn.Embedding(3, d_model)` at line 754 | **PASS** |
| 5 | **Latent_valid parameter** (single source of truth; renamed from `slot_mask` per B30) | `src/speech_decoding/ssl/slot_loss.py:51` `latent_valid: Tensor` (parameter name) + docstring B30 section confirms anatomy-coverage gate `(support.sum(over electrodes) > 0)` | **PASS** |
| 6 | **Bidirectional SA masking** (inactive slots fully bypassed, not just key-masked) | `src/speech_decoding/models/v14_encoder.py:416-420` "B30: bidirectional mask — inactive slots neither query nor key" + outer-AND construction `latent_valid.unsqueeze(2) & latent_valid.unsqueeze(1)` → `attn_mask` | **PASS** |
| 7 | **Anatomy bias per-clip** (`log(support+ε)` gate, no step warmup, computed per clip not schedule) | `src/speech_decoding/studies/braintreebank/anatomy.py:*` "return np.log(support + np.float32(eps))" + dispatch config `--anatomy-prior-strength` flag (no schedule, per-clip gate default) | **PASS** |
| 8 | **AJILE12 included** (corpus reinclusion via α=0.3 weighted sampling) | `src/speech_decoding/studies/ajile12/study.py:*` `class AJILE12Study(study.Study)` + test asserts phase scope includes AJILE12; no drop-gate present | **PASS** |
| 9 | **α=0.3 temperature sampling** (per-corpus weighting, SWEC 35% / AJILE12 22% / D 18% / BT 12% composition) | `src/speech_decoding/experiments/wrs_sampler.py:*` "R-sampler-alpha03" (α=0.3)" + `test_v14_dispatch_wired.py:def test_b29_dispatch_default_ref_operator_alpha_is_0p3()` asserts `args.ref_operator_alpha == 0.3` | **PASS** |
| 10 | **d_model=256 default** (unchanged from prior; R-d-bump-384 P0 sister for capacity recovery) | `src/speech_decoding/models/v14_encoder.py:500` `d_model: int = 256` (no change) | **PASS** |
| 11 | **Dense FFN default, no MoE** (B29 Item 14 audit defers MoE-FFN to v15 + R-moe-ffn-soft-4 P2 if-budget) | `src/speech_decoding/models/v14_encoder.py:339-341` `def _ffn(d_model: int, mult: int = 4)` returns dense `Sequential(Linear→GELU→Linear)` + dispatch `--ffn-variant dense` (default); no MoE code in v14_encoder.py | **PASS** |

---

## Drift Detection

**Zero drift detected.** B29/B30 decision matrix rows, sister roster updates, and code-vs-spec blocker mappings all align. Whisper-adapter docstring updates (v2→v3) and BT-alignment module touches are orthogonal to core B29/B30 mechanism. 

**No re-audit trigger conditions met.** All 11 claims Round 1 PASS status confirmed under fresh independent read of specs + code. Standing loop directive satisfied: proceed to next task.

---

## Blockers Resolved in Code

- B30-anatomy-gated-via-latent-valid: ✓ (aggregator threads `latent_valid` to all slot-axis losses)
- B30-sa-bidirectional-mask: ✓ (outer-AND construction in latent SA forward)
- B30-loss-rename-slot-mask: ✓ (function signature parameter name = `latent_valid`)
- B30-pma-zero-active-guard: ✓ (PMA skips when `latent_valid.sum(dim=-1) == 0`)

