"""v14 self-supervised pretraining components.

Recipe (canonical: ``project_v14_b36_perparcel_pool_structured_jepa_2026_06_01``
+ ``project_v14_b36_masking_ratified_2026_06_03``). B36 (2026-06-01) re-split
SSL into two staged phases of paradigm-B masked JEPA — a visible-only encoder
plus a SEPARATE predictor that reconstructs masked targets from the EMA +
StopGrad teacher's full-input features; pure-L1 on masked cells only:

    Phase 1 : front-end only (token blocks + ``frontend_ln``). Structured-1D
              spectro-temporal band M2 mask at held-out 0.50, all corpora.
    Phase 2 : per-parcel pool + inter-parcel encoder + M4 predictor (front-end
              continues at LR/10), anatomy-bearing corpora. Parcel×time tube
              mask at 0.20 + cross-time predictor; target = the teacher's
              post-``encoder_ln`` M4 (the canonical V-JEPA target — no LN_frame).
    Phase 3 : single-teacher Whisper distillation, all-layer-mean target
              (project-up; PMA is trained here, frozen at P4).
    Phase 4 : downstream — frozen encoder + frozen PMA → mean-over-time →
              per-task linear classifier (B35).

The LIVE objective is ``masked_jepa.p1_frontend_m2_loss`` /
``p2_parcel_m4_loss``, called from
``experiments.v14_joint_module.V14JointBrainModule._step``. The older
``total_loss.v14_total_loss`` composer wired by
``aggregator.compute_v14_ssl_losses`` is NOT on the default path — it survives
only as the ``b31_plus_*`` falsifier sisters (``R-add-m3-loss`` etc.), so it is
retained, not deleted. Substrate primitives (EMA teacher, recon loss,
distillation loss, KoLeo) sit in their own modules.

Files in this package are intentionally small and decoupled so individual
pieces can be swapped without cascading rewrites.
"""
