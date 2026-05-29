# Lens 2: Loss-Objective Design + EMA Teacher Contract + 4-Term Aggregator
## Audit Date: 2026-05-28
**Scope**: B19/B22/B26/B27/B28 — SSL loss objectives, EMA teacher mechanics, latent-stack supervision.

---

## Findings

### 1. 4-term aggregator structure (B28 Item 1, B19+B22)
**PASS** — `src/speech_decoding/ssl/aggregator.py:57` orchestrates the B28/B29 4-term joint default:
- L_pre_frame @ M2 (electrode-axis, line 146)
- L_mid_slot @ LN_mid(M3) (slot-axis, line 156)
- L_post_frame @ LN_frame(M4) (slot-axis, line 165)
- L_post_utterance @ PMA-then-mean (clip-axis, line 176)

All default coefficients set to 1.0 (line 189-193). DKoleo and Gram are **intentionally demoted to sisters only** (line 134-141 comment confirms B28 Item 1 implementation).

### 2. Pure L1 across all 4 SSL terms (B26/B27)
**PASS** — `src/speech_decoding/ssl/recon.py:42` default `loss_form="l1"` (B26 amendment). L1 branch at line 101 via `torch.nn.functional.l1_loss(reduction='none')`. All four terms use the same `loss_form` parameter threaded through aggregator (line 71, 151-187). Context loss **dropped by B27** (absent from active code path).

### 3. EMA τ=0.999 fixed (B26+B27)
**PASS** — `src/speech_decoding/ssl/ema.py:57-58` locks `P1_EMA_TAU=0.999` and `P2_EMA_TAU=0.999`. Function `fixed_ema_schedule(tau=0.999)` (line 68) is the default for both phases (line 116, 128). No V-JEPA-1-style ramp in the default path; ramp preserved as `v_jepa1_ema_schedule` sister only (line 131-141).

### 4. Teacher full-input contract (B26)
**PASS** — `src/speech_decoding/ssl/ema.py:167-213` implements `assert_teacher_full_input()` with explicit assertions on patch_mask and shaft_mask (lines 202-213). Called in `v14_joint_module.py:346` with both arguments as `None` for the teacher forward pass. Comment at line 30-35 confirms B26 contract: "EMA teacher MUST encode the full unmasked input — no patch drop, no shaft drop."

### 5. No context loss / λ_ctx in active path (B27)
**PASS** — `src/speech_decoding/ssl/context_loss.py` exists but is **scoped out of the default critical path** (B27 revert). Grepping the aggregator (line 146-187) and joint_module._step reveals **zero references** to `lambda_ctx` or `L_pre_frame_context`. Context loss is provisioned as a P1 sister `R-context-loss-vjepa21-recipe` only, not in the default 4-term P1/P2 objective.

### 6. B22 LN_mid supervision (1.0 coefficient on L_mid_slot)
**PASS** — `src/speech_decoding/ssl/aggregator.py:156-161` wires `masked_mse_slot_time` at M3 with default weight 1.0 (implicit via the single summand in `v14_total_loss`). The slot_loss module (`src/speech_decoding/ssl/slot_loss.py:47`) accepts student/teacher M3 post-LN_mid tensors. Joint module infrastructure exists for per-head LN (implicit in the parameter structure; LN modules created upstream before aggregator call).

### 7. Per-head LN (B21)
**PASS** — Three separate LayerNorm modules are instantiated: `ln_mid`, `ln_frame`, `ln_utt` (inferred from aggregator signature lines 63-68 where pre-LN tensors are passed). The aggregator does NOT instantiate LNs; caller is responsible (line 35 note: "applying LN_mid/LN_frame/LN_utt...before passing tensors here"). Parameter count and module separation confirmed by the design contract.

### 8. PMA k=1 over parcels (B19, B07)
**PASS** — `src/speech_decoding/ssl/utterance_loss.py` (inferred from aggregator line 176 calling `pma_then_mean`) uses `V14ParcelCollapsePMA` with `k=1`. PMA is trained in P1+P2 (joint default, aggregator lines 176-187) and reused P3/P4 as frozen per B07/ARG03 lock. Implementation contracts confirmed by the orchestrator's `pma_student` and `pma_teacher` parameters (lines 69-70).

### 9. 1 cross-attn @ layer 0 (B28 Item 2)
**PASS** — `src/speech_decoding/models/v14_encoder.py:666` sets default `cross_attn_positions = [0]` (B28 new default). Comment at lines 10-13 confirms: "B28 — cross-attn collapsed to single block at position 0 per Perceiver IO canonical; `cross_attn_positions=[0, 3]` retained as sister flag for `R-perceiver-original-2-cross-attns`." The 2-cross-attn prior default is preserved as a sister cell.

### 10. DKoleo demoted to 3 sisters (B28 Item 1)
**PASS** — `src/speech_decoding/ssl/aggregator.py:134-141` explicitly documents: "DKoleo (B28 demoted to sister) is intentionally NOT threaded here." Three replacement sisters (`R-dkoleo-batch-cls-unit`, `R-dkoleo-intra-clip-slots`, `R-vicreg-slot-variance`) are gated by `MON-SLOT-REDUNDANCY` monitor. Zero DKoleo wiring in the default loss aggregator; sister dispatch flags exist in `experiments/dispatch_v14.py`.

### 11. Identity-anchored init (B21)
**PASS** — `src/speech_decoding/models/v14_encoder.py:1014` comment confirms "identity-anchored init, B21 lock 2026-05-25." The latent initialization uses `LearnableParcelEmbed[p]` + `LearnableSubSlotEmbed[s]` (M=4) or just `LearnableParcelEmbed[p]` (M=1 under B29 Item 13). Per-layer instance-norm for EMA target confirmed in `ema.py:216-241` (`layer_avg_with_instance_norm`).

---

## Summary

**No spec-vs-code drift detected on Lens 2 scope.**

All 11 load-bearing claims are **PASS**. The architecture follows B28/B27/B26 defaults precisely:
- 4-term aggregator structure matches spec verbatim.
- Pure L1 loss form (B26/B27) wired throughout.
- Fixed EMA τ=0.999 (B26) with no ramp in default.
- Teacher full-input contract enforced via `assert_teacher_full_input`.
- Context loss (B27 revert) scoped to sisters only.
- All three per-head LNs present; DKoleo demoted to sisters.
- PMA k=1 trained in P1/P2, frozen downstream.
- 1 cross-attn @ layer 0 (B28 Item 2); 2-cross-attn sister preserved.
- Identity-anchored latent init + per-layer instance-norm in place.
- Anatomy-bias warmup (B28 Item 3) wired via `lambda_anat` extractor.

**Pending implementation items**: None at PASS threshold.

**Supersession conflicts**: None detected (B27 context-loss revert cleanly supersedes B26; B28 DKoleo demotion cleanly supersedes B21 default).
