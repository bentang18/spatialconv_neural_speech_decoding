"""v14 self-supervised pretraining components.

Recipe (still in flux — see canonical sources below) is the
3-phase staged plan from
``memory/project_v14_three_phase_staged_recipe_2026_05_18.md`` amended by
``memory/project_v14_imindbench_multistft_pivot_2026_05_22.md``:

    Phase 1 (Stage A): EAT Level-B time-frequency mask + UFO frame target.
    Phase 2 (Stage B): Electrode-mask Level-A (mask_rate=0.30, J1 sweep) +
                       UFO utterance target.
    Phase 3        :   Single-teacher Whisper-L8 distillation (DINOv3 dropped
                       5/22, Stage 3a deleted): 3b-warmup ~3–5% LR → 3b-unfreeze
                       slow-LR ~95%.
    Phase 4        :   Downstream — frozen PMA + flat-head linear classifier.

Files in this package are intentionally small and decoupled so individual
pieces can be swapped without cascading rewrites. Loss composition lives in
``compose.py``; the substrate primitives (EMA teacher, recon loss,
distillation loss, KoLeo) sit in their own modules.

Order of operations in a single training step (Phase 1/2 example):

    student_out = student(x_masked)
    with torch.no_grad():
        teacher_out = teacher(x_target)            # EMA + StopGrad
    loss_recon  = recon_loss(student_out, teacher_out, valid_bin_mask)
    loss_koleo  = koleo_loss(student_readout)
    loss = lam_recon * loss_recon + lam_koleo * loss_koleo
"""
