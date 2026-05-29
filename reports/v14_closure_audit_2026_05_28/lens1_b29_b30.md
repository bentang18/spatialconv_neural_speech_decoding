# v14 Spec-vs-Code Audit: Lens 1 (B29/B30 Joint-Default + Anatomy-Gated Symmetric)

**Scope**: B29 (joint-default amendment 5/27 PM-late) + B30 (anatomy-gated symmetric 5/28) lock alignment with `src/speech_decoding/`.

**Report Date**: 2026-05-28  
**Auditor**: Claude  
**Memos Read**: 
- `project_v14_b29_joint_default_2026_05_27.md` (B29)
- `project_v14_anatomy_gated_symmetric_2026_05_28.md` (B30)
- `project_v14_subtype_embed_precedent_audit_2026_05_28.md` (precedent audit)
- `project_v14_moe_ffn_audit_2026_05_28.md` (MoE audit)

---

## Per-Claim Verdicts

| # | Claim | Status | Evidence (file:line) | Notes |
|---|-------|--------|----------------------|-------|
| 1 | **M=1 default (B29 Item 13)**: `m_sub_slots: int = 1` in encoder + dispatch | **PASS** | `dispatch_v14.py:68` (DEFAULT_M_SUB_SLOTS = 1); `v14_encoder.py:508, 1434` (m_sub_slots: int = 1) | Both encoder config and dispatch use M=1 as the locked default. |
| 2 | **80 slots (B29 Item 13)**: K_total = 80 DK parcels × 1 sub-slot; `LearnableSubSlotEmbed` removed; parcel identity at M=1 | **PASS** | `dispatch_v14.py:69` (DEFAULT_K_PARCELS = 80); `v14_encoder.py:504-610` (comment confirms LearnableSubSlotEmbed dropped, identity init via LearnableParcelEmbed only) | Architecture reflects 80-slot pool with identity anchoring. SubSlotEmbed is gone. |
| 3 | **`latent_valid` single source of truth (B30 Item 12)**: is `latent_valid = (support.sum(over electrodes) > 0)` the single name in encoder + losses? Did `slot_mask` → `latent_valid` rename happen? | **PASS** | `slot_loss.py:51, 67, 99-105, 114, 121` (latent_valid parameter, B30 docstring); `utterance_loss.py:119-160` (latent_valid); `v14_encoder.py:1119, 1248-1272` (latent_valid in both blocks and PMA); no `slot_mask` parameter in active paths | Parameter renamed and single-sourced. B30 convention enforced. |
| 4 | **SA bidirectional masking (B30)**: is latent-SA mask applied symmetrically (not key-only)? Check attn_mask construction | **PASS** | `v14_encoder.py:199-260` (B30 lock docstring 206-210 confirms bidirectional via attn_mask; replaces pre-B30 key-only); line 234-248 shows attn_mask construction masking both rows/columns for invalid slots; no `key_padding_mask` argument to latent SA | attn_mask (B, L, L) bool applied symmetrically. Key-only mask retired. |
| 5 | **DROP parcels_supervised gating (B29 Item 12 → B30 supersedes)**: zero hits on `parcels_supervised` indexing in active loss sites | **PASS** | `dispatch_v14.py:665-669` shows `parcels_supervised` only in sister option (R-item-12-all-true, R-parcels-supervised-gating), NOT default path; slot_loss.py/utterance_loss.py use latent_valid only, no parcels_supervised active | Default path uses latent_valid exclusively. parcels_supervised confined to sister overrides. |
| 6 | **subtype_embed default OFF (5/28 audit)**: is `subtype_embed_enabled = False` the default? | **PASS** | `v14_encoder.py:549` (subtype_embed_enabled: bool = False); `dispatch_v14.py:218-220` (subtype_embed_enabled default False in build_v14_experiment); line 1462 (V14Config default False) | Matches 5/28 Agent 2 audit verdict. Default OFF per B29-supersession. |
| 7 | **ref_embed default ON (B29 Item 11)**: is `ref_embed_enabled = True` the default? Are 3-way ref-idx + nn.Embedding(3, d=256) wired? | **PASS** | `v14_encoder.py:551` (ref_embed_enabled: bool = True); line 740 (nn.Embedding(3, d_model)); line 741 trunc_normal init std=0.02; `dispatch_v14.py:221, 1464` (ref_embed_enabled: bool = True default) | 3-way embedding (shaftCAR/bipolar/Laplacian) wired at input and cross-attn K/V (lines 905, 1058, 1066). |
| 8 | **MoE-FFN rejected (5/28 audit)**: dense FFN preserved? Zero hits on Soft MoE / Sparse MoE in active paths? | **PASS** | `dispatch_v14.py:132-136` (FFN_VARIANTS with dense default, soft_moe_4 reserved P2); `v14_encoder.py:263-268` (_ffn returns dense Sequential, no MoE); line 279 raises NotImplementedError if soft_moe_4 requested; `dispatch_v14.py:275-279` guards soft_moe_4 | Dense FFN locked. soft_moe_4 (Puigcerver 2024) deferred P2-if-budget with explicit error guard. |
| 9 | **AJILE12 reinstated (B29 Item 8)**: is AJILE12 in the joint corpus list? Check dispatch + corpus manifests. | **PASS** | `dispatch_v14.py:165-170` (DEFAULT_CORPUS_MIX includes ajile12 at 22/87 share); line 175 (DEFAULT_INCLUDE_AJILE12: bool = True); corpus_mix dict keys: swec, ajile12, d_cohort, braintreebank | AJILE12 included in default corpus mix (B29 reversal of earlier drop). Share normalized to sum to 1.0. |
| 10 | **α=0.3 temperature sampling default (B29 Item 5)**: is DEFAULT_REF_OPERATOR_ALPHA = 0.3 or did B02's α=0.5 stay? Memo says 0.3; flag conflict. | **PASS** | `dispatch_v14.py:155-157` (DEFAULT_REF_OPERATOR_ALPHA: float = 0.3); line 691-692 command-line arg for ref_operator_alpha default 0.3 | B29 default α=0.3 locked. No B02 α=0.5 conflict in active code. |
| 11 | **Single joint phase (B29 Item 1)**: dispatch routes `--phase 1` to V14JointExperiment, phases 2/3 raise NotImplementedError? | **PASS** | `dispatch_v14.py:141-142` (PHASE_MODES with joint_b29 default); line 478-482 (phase_mode=="joint_b29" routes to V14JointExperiment); line 240 note "phase=1" → V14JointExperiment; line 834-835 raises NotImplementedError for phase 2/3 in split path | Default is joint phase via --phase 1. Phases 2/3 raise NotImplementedError in default path. |

---

## Summary

### PASS (11/11 claims verified)

All load-bearing B29 + B30 claims are **code-reflected** and **locked as defaults**:

1. **M=1 + 80 slots**: encoder/dispatch defaults match.
2. **latent_valid single source**: B30 rename complete, used uniformly in encoder + losses.
3. **Bidirectional SA masking**: attn_mask applied symmetrically; key-only retired.
4. **parcels_supervised dropped**: gating removed from default loss paths (confined to sister overrides).
5. **subtype_embed OFF**: matches 5/28 audit; default disabled.
6. **ref_embed ON**: 3-way embedding (3, d=256) wired at input + cross-attn K/V.
7. **MoE-FFN rejected**: dense FFN preserved; soft_moe_4 guarded with NotImplementedError.
8. **AJILE12 included**: corpus mix includes ajile12 at 22/87 share.
9. **α=0.3 locked**: temperature sampling default 0.3 (no B02 conflict).
10. **Joint phase default**: --phase 1 routes to V14JointExperiment; phases 2/3 not implemented.

### Drift List

**None**. All 11 claims are in-spec.

### Pending Implementation

None identified in active code paths. Sisters (R-item-12-all-true, R-subtype-embed-input-only, R-keep-phase-split, R-moe-ffn-soft-4) are properly gated with runtime guards or dispatch-time overrides.

### Open Conflicts

None. α=0.3 cleanly supersedes any prior B02 α=0.5 assumption.

---

## Code Path Verification Summary

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| M=1 default | `dispatch_v14.py` / `v14_encoder.py` | 68, 508, 1434 | ✓ LOCKED |
| 80 DK parcels | `dispatch_v14.py` / `v14_encoder.py` | 69, 504-610 | ✓ LOCKED |
| latent_valid uniformity | `slot_loss.py` / `utterance_loss.py` / `v14_encoder.py` | 51, 119, 1119-1272 | ✓ COMPLETE |
| Bidirectional SA attn_mask | `v14_encoder.py` | 199-260, 234-248 | ✓ IMPLEMENTED |
| parcels_supervised confined | `slot_loss.py` / `utterance_loss.py` / `dispatch_v14.py` | 665-669 | ✓ RETIRED |
| subtype_embed OFF | `v14_encoder.py` / `dispatch_v14.py` | 549, 1462, 218 | ✓ LOCKED |
| ref_embed ON (3-way) | `v14_encoder.py` / `dispatch_v14.py` | 551, 740-741, 1464 | ✓ LOCKED |
| Dense FFN (no MoE default) | `v14_encoder.py` / `dispatch_v14.py` | 263-268, 132-136, 275-279 | ✓ LOCKED |
| AJILE12 in corpus_mix | `dispatch_v14.py` | 165-170, 175 | ✓ LOCKED |
| α=0.3 temperature | `dispatch_v14.py` | 155-157 | ✓ LOCKED |
| --phase 1 → joint | `dispatch_v14.py` | 141-142, 478-482, 834-835 | ✓ LOCKED |

---

**Audit Status**: COMPLETE — All B29/B30 specs verified in source code. No drifts detected. Ready for cell-0 dispatch.
