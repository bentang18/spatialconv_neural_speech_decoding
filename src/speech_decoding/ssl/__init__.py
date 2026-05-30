"""v14 self-supervised pretraining components.

Recipe (canonical sources below) — B29 (2026-05-27) collapsed the old
Phase 1 + Phase 2 into a single joint SSL phase, and B31 (2026-05-28) reduced
the SSL loss surface to two terms:

    Joint SSL phase: V-JEPA-2-canonical masked prediction. Two pure-L1 terms,
                     ``L_pre_frame @ M2 + L_post_frame @ M4`` (B31). EMA +
                     StopGrad teacher encodes the full unmasked input.
    Phase 3        : Single-teacher Whisper-L8 distillation (DINOv3 dropped
                     5/22, Stage 3a deleted): 3b-warmup ~3–5% LR → 3b-unfreeze
                     slow-LR ~95%. PMA is trained here (frozen P4).
    Phase 4        : Downstream — frozen PMA + flat-head linear classifier.

The dropped B19 terms (``L_mid_slot @ M3``, ``L_post_utterance``) and the
demoted DKoleo regularizer survive as opt-in sisters, NOT in the default.

Files in this package are intentionally small and decoupled so individual
pieces can be swapped without cascading rewrites. Loss composition lives in
``total_loss.py`` (the ``v14_total_loss`` composer) wired by ``aggregator.py``
(``compute_v14_ssl_losses``, the live single-source-of-truth); the substrate
primitives (EMA teacher, recon loss, distillation loss, KoLeo) sit in their
own modules.

Order of operations in a single joint-SSL training step:

    student_out = student(x_masked)
    with torch.no_grad():
        teacher_out = teacher(x_full)              # EMA + StopGrad, full input
    breakdown = compute_v14_ssl_losses(...)        # B31 2-term default
    loss = breakdown.total
"""
