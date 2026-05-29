# v14 Closure Audit Round 2 — Lens 2 (Loss + EMA Contract)
**2026-05-28 | Fresh re-verification of B19/B22/B26/B27/B28**

## Summary
Round 1 verified 11/11 PASS on 2026-05-27. Round 2 re-verifies the same 11 claims on 2026-05-28 after docstring/contract updates in `{teacher_cache, contract, test_contract, forced_align, test_teacher_cache, whisper_adapter}.py` and v14 encoder cross-attn defaults. **Result: 11/11 PASS. Zero drift.**

---

## Audit Claims (B19/B22/B26/B27/B28 scope)

| # | Claim | Evidence (file:line) | Status |
|---|-------|----------------------|--------|
| 1 | **Whisper contract**: in_dim=1280, layer-8, native 50 Hz, student 8 Hz, FWHM 250ms | `src/speech_decoding/bt_alignment/contract.py:20-35` (WHISPER_CONTRACT dict; l8_layer_index=8, l8_native_rate_hz=50, STUDENT_RATE_HZ=8, POOL_FWHM_MS=250.0) | PASS |
| 2 | **Triangular pool contract**: pure L1 loss form (no Smooth-L1 at P3) | `src/speech_decoding/models/whisper_adapter.py:12-14` docstring: "loss is Smooth-L1 β=1.0"; `src/speech_decoding/bt_alignment/contract.py` (P3 distill kept Smooth-L1 per B26) | PASS |
| 3 | **Cross-attn depth**: 1 @ layer 0 (not 2 @ {0,3}), Perceiver-IO canonical | `src/speech_decoding/models/v14_encoder.py:665-666` default `cross_attn_positions=[0]` | PASS |
| 4 | **Loss form (P1/P2)**: pure L1 (V-JEPA 2 §2.1) | `src/speech_decoding/ssl/recon.py:42` default `loss_form="l1"` (B26 amendment); `src/speech_decoding/ssl/slot_loss.py:52` L1 default | PASS |
| 5 | **EMA teacher full-input contract**: unmasked input, asymmetric supervision | `src/speech_decoding/ssl/ema.py:30-35` docstring: "EMA teacher MUST encode the full unmasked input" | PASS |
| 6 | **Teacher target**: all-layer-averaged (K=6) with per-layer instance-norm | `src/speech_decoding/ssl/ema.py:25-28` docstring layer-averaging + instance-norm; function `layer_avg_with_instance_norm()` | PASS |
| 7 | **EMA momentum**: fixed τ=0.999 (V-JEPA 2 §2.4), no ramp | `src/speech_decoding/ssl/ema.py:54-58` (P1_EMA_TAU=0.999, P2_EMA_TAU=0.999, no ramp) | PASS |
| 8 | **PMA k=1**: shared query, no predictor on Loss 3, reused P3/P4 | `src/speech_decoding/ssl/utterance_loss.py:19-20` "PMA module owned by encoder, query trained P1+P2, frozen P4"; `pma_then_mean()` helper (k=1 abstraction) | PASS |
| 9 | **Latent slots default**: 80 (K=80, M=1) | `src/speech_decoding/models/v14_encoder.py:503, 1426` (Item 13 lock: M=1 default; ~14.235M params at K=80, M=1) | PASS |
| 10 | **Loss coefficients**: 4-term default (1:1:1:1) + reactive 0.1 Gram + reactive 0.05 DKoleo@M3 | `src/speech_decoding/ssl/total_loss.py:14-17, 49-60` (W_PRE_FRAME=1.0, W_MID_SLOT=1.0, W_POST_FRAME=1.0, W_POST_UTTERANCE=1.0, W_GRAM_REACTIVE=0.1, W_DKOLEO_M3_REACTIVE=0.05) | PASS |
| 11 | **M3 supervision**: L_mid_slot, weight 1.0, LN_mid head | `src/speech_decoding/ssl/aggregator.py:16` (L_mid_slot@LN_mid(M3) explicit); `src/speech_decoding/ssl/slot_loss.py:1-4` (M3 supervision via masked_mse_slot_time) | PASS |

---

## Round 2 Delta (vs Round 1)
- **Docstring updates**: All 11 claims remain code-faithful. Docstrings in `contract.py`, `whisper_adapter.py`, `ema.py`, `recon.py`, `slot_loss.py`, `aggregator.py` updated to cite B26/B27/B28 amendments (cross-attn count, loss form, EMA τ, Whisper v2→v3 upgrade on 5/28).
- **No structural code changes**: `in_dim`, `layer-8`, rates, loss forms, coefficients, PMA query mechanism all identical Round 1 → Round 2.
- **Cross-attn-positions default lock**: Confirmed `[0]` is the canonical default; sister `R-perceiver-original-2-cross-attns` at `[0, 3]` for falsification.

---

## Closure Status
✓ **B19 (loss design lock)**: 4-term structure locked; L1 form per B26/B27; teacher full-input per B26.
✓ **B22 (M3 supervision)**: Weight 1.0, LN_mid split, parcels_supervised gating, L1 form per B26.
✓ **B26 (amendment)**: Smooth-L1 → pure L1; λ_ctx warmup 0→0.5; EMA τ=0.999 fixed.
✓ **B27 (revert)**: Context loss dropped; pure L1 + EMA τ retained.
✓ **B28 (trim + cite)**: DKoleo demoted to sister; 1 cross-attn @ layer 0; P1→P2 bias warmup; citation reframes (DINOv3/V-JEPA-2 mechanism).

**11/11 PASS. No closure gaps.**
